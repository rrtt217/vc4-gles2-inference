// Clean GLES2 inference engine for llama2.c, Q8M1 and I16M1 checkpoints.
// Targets OpenGL ES 2.0 (Raspberry Pi VC4) and desktop GLES2 implementations.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef GL_COLOR_ATTACHMENT1
#define GL_COLOR_ATTACHMENT1 0x8CE1
#endif

#include "common.h"
#include "dump_shaders.h"
#include "shaders.h"
#include "xformat.h"
#include "gpu.h"
#include "app.h"
#include "chat.h"
#include "worker.h"
#include "chat_format.h"
#include "prefix_build.h"
#include "cli.h"
#include "ppl.h"
#include "model.h"
#include "chattmpl.h"
#include "jinja/json.h"
#include <algorithm>
#include <filesystem>
#include <poll.h>
#include <unistd.h>
#include "session.h"
#include "tokenizer.h"
#include "sampling.h"
#include "tools.h"
#include "lineedit.h"

// The default system prompt is *not* compiled in: it comes from the container's
// metadata (key 6) and `--system` overrides it (see system_prompt below).  A
// literal here would be dead code that reads like the default and is not --
// `-Wunused-variable` found exactly that on 2026-11-14.


// Informational stderr lines (template, tokenizer, specials, rope, cache) are
// useful when something is wrong and noise when it is not; -q turns them off.
static bool g_quiet = false;

// TIMING has to survive every way a turn can end: Ctrl-C during prefill, Ctrl-C
// during decode, a context that filled up, or the normal stop.  The early
// returns that mark an interruption are exactly the runs someone wants numbers
// for, which is why the old inline print - at the end of the happy path - was
// silent in every case that mattered.  A destructor runs on all of them; `cut`
// names the reason when the turn did not finish on its own, so a short `gen=`
// is never ambiguous.



// Requests and control lines arrive on stdin, and a *cancel* has to be seen
// while a request is generating -- which getline() cannot do without swallowing
// the next request.  One buffer, filled from the raw fd, serves both: blocking
// lines for the request loop, and a non-consuming peek for the token loop.
int main(int argc, char** argv) {
    install_stop_handlers();
    gpu_set_run_t0(Clock::now());
    // No default: the engine only reads MMCF containers, so a bare run has
    // nothing sensible to load and "set MODEL=" is a better answer than a magic
    // number from a file that happens to sit at the old llama2.c path.
    Options opt;
    bool quiet = g_quiet;
    int cli_exit = 0;
    if (!parse_args(argc, argv, opt, quiet, cli_exit)) return cli_exit;
    g_quiet = quiet;

    if (opt.dump_mode) return dump_shaders(opt.dump_mode == 2);
    if (opt.top_k < 0) opt.top_k = 0;
    if (opt.top_k_candidates && opt.top_k <= 0) {
        if (!g_quiet) cerr << "--top-k-candidates needs --top-k N; using full logits\n";
        opt.top_k_candidates = false;
    }
    if (opt.top_k > 0 && !g_quiet)
        cerr << "sampling: top-k " << opt.top_k
             << (opt.top_k_candidates ? " (candidates preferred)" : " (full logits)") << "\n";
    if (opt.chat && !opt.gen_set) opt.gen = 128;

    if (opt.model_path.empty()) {
        fprintf(stderr, "no model: set MODEL=<file.mmcf> or pass --model PATH (see --help)\n");
        return 2;
    }
    {
        ifstream probe(opt.model_path, ios::binary);
        if (!probe) {
            fprintf(stderr, "cannot open model %s\n"
                            "  set MODEL=<file.mmcf> or pass --model PATH; the store lives in "
                            "models/<name>/<name>.<quant>.mmcf (see --help)\n", opt.model_path.c_str());
            return 2;
        }
    }
    App app(opt);
    if (!app.model.load(opt.model_path)) return 1;
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  model loaded\n", run_time_ms());
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  egl ready\n", run_time_ms());
    // Everything the engine does to the model between reading it and uploading
    // it.  A model swap has to repeat *all* of it: split_ffn decides the FFN
    // kernel width (chunked 1216 or whole 2432 on minimind3), apply_hidden_scale
    // applies metadata key 15, and apply_precision/quantize_q4_weights rewrite
    // Mats.  Running only some of them is not a crash but a different kernel
    // set and a different scaling, i.e. numbers that no longer match a fresh
    // process -- measured as logits cosine 0.9998 for the dense models and 0.96
    // for the hybrid, whose hidden scale is the larger half of the difference.
    app.system_prompt = opt.system_prompt;          // the CLI value; setup_from_model
    app.system_from_cli = !app.system_prompt.empty();  // may replace it with the container's
    app.prepare_model();
    // The checkpoint names its template (a kind byte plus a name), so --template
    // is only needed to override it.  The kind is authoritative; the name is the
    // fallback for a kind this build does not know.
    // Chat mode with no chat template falls back to the built-in [INST] wrapper,
    // which is right for a Llama-2 chat model and nonsense for a completion
    // model.  Say which one is happening instead of letting the output explain.
    if (opt.chat && app.model.template_kind == 0 && app.model.chat_template.empty() && !g_quiet)
        fprintf(stderr, "chat: this checkpoint has no chat template (kind 0, plain); "
                        "using the built-in [INST] wrapper.  For a completion model drop "
                        "--chat and use --prompt.\n");
    // Capabilities: a checkpoint that says it is not a chat model must not be
    // driven through the chat template.
    if (app.model.have_capabilities && opt.chat && !(app.model.capabilities & 1)) {
        cerr << "this checkpoint does not advertise chat support (capabilities 0x"
             << std::hex << app.model.capabilities << std::dec << "); drop --chat\n";
        return 2;
    }
    if (app.model.have_capabilities && !opt.chat && (app.model.capabilities & 1) && !opt.gen_set) {
        opt.chat = true;
        opt.gen = 128;
    }
    // Everything that comes from the container's own metadata: the template
    // kind, the tokenizer and the Jinja template.  Called at startup and again
    // by the REPL's /model, which is why it reads `model` instead of taking it
    // as an argument.  Returns false when the container is unusable.
    if (!app.setup_from_model()) return 2;
    // Tool definitions, normalised to one JSON array for the template context.
    const string tools_array = chat_tools_array(opt.tools_json);
    // BOS policy: the container states it when it can (key 10); otherwise the
    // historical rule is kept, which is derived from the template kind.
    const bool enc_bos = app.model.add_bos >= 0 ? (app.model.add_bos != 0)
                                            : (!opt.qwen_template && !opt.tbb_template);
    // CHAT_DUMP=1 prints the rendered prompt with its input marking, which is
    // what the byte-for-byte template comparisons are run against.
    const bool chat_dump = env_set("CHAT_DUMP");
    // Stop-token test.  The container states EOS and the extra stop id (key 8).
    // The CLI used to test the literal ids 1 and 2, which is MiniMind3's
    // coincidence; a model whose EOS is 7 never stopped.
    // BOS counts as a stop while generating: no model means to re-open its own
    // stream, and a story model that emits <s> mid-text has run out of
    // distribution - it will repeat one token until --gen otherwise.

    if (getenv("SHADER_TIME"))
        cerr << "rmsnorm eps: " << app.model.rms_eps << " (container"
             << (app.model.rms_eps == 1e-5f ? "" : "-stated") << ")\n";
    // Uploading the weights is a separate phase from reading the container, and
    // a model swap has to repeat both.  Kept as one lambda so the two callers
    // cannot drift: forgetting an upload leaves every Mat with tex == 0, and the
    // forward pass then answers with EOS on the first token (measured: the first
    // swap produced empty replies until this was shared).
    const auto up_t0 = Clock::now();
    app.upload_weights();
    if (getenv("SHADER_TIME") || getenv("UPLOAD_TIME"))
        fprintf(stderr, "UPLOAD %.2f s wall, %d textures, texture time %.0f ms\n",
                elapsed_ms(up_t0, Clock::now()) / 1000.0, gpu_upload_textures(), gpu_upload_texture_ms());
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  weights uploaded\n", run_time_ms());

    Session session(app.model, app.gpu);

    // A cache path may be a single file or a directory.  With a directory the
    // engine picks the entry whose *header* matches this model (container size
    // and mtime, the shape, the KV packing and the scale), so one setting serves
    // several models: after a model swap the cache that belongs to the new
    // container is loaded and the others are ignored.  The header is the
    // authority; the file name is not.
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  session ready\n", run_time_ms());

    if (!opt.ppl_path.empty()) return run_ppl(opt.ppl_path, app.model, app.gpu, app.tokenizer);

    if (!opt.build_cache_prompt.empty())
        return build_prefix_cache(app.model, app.tokenizer, app.chat_tmpl.get(), opt, tools_array, enc_bos,
                                  app.bos_str, app.eos_str, g_quiet, session);

    if (opt.worker) return run_worker(app, session, enc_bos, chat_dump);

    // The chat path: the turn loop, its tool turns, the marker mask and the
    // CLI prefix cache.  Built before the dispatch below so both paths share it.
    Chat chat(app, session, enc_bos, chat_dump, g_quiet);
    chat.load_cli_cache();

    if (app.opt.chat) return chat.repl();
    return chat.complete();
    return 0;
}
