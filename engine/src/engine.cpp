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
#include "switches.h"
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


// Informational stderr lines (template, tokenizer, specials, rope, cache) are
// useful when something is wrong and noise when it is not; -q turns them off.
static bool g_quiet = false;


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

    if (opt.dump_switches) { sw_dump(stdout); return 0; }
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
    // A version 5 container carries an encoder stack and a cross-attention; the
    // decode loop that drives them is the next step of this branch, and until it
    // exists the dense loop would read an empty rms_att and die in the upload.
    // Refuse here, where the message can say why, instead of there.
    // A v5 container has no chat path and no single-stream forward: it is a
    // source sentence in and a target sentence out, so `--prompt` is the source.
    const bool translate_mode = app.model.is_encdec();
    // Tool definitions, normalised to one JSON array for the template context.
    const string tools_array = chat_tools_array(opt.tools_json);
    // BOS policy: the container states it when it can (key 10); otherwise the
    // historical rule is kept, which is derived from the template kind.
    const bool enc_bos = app.model.add_bos >= 0 ? (app.model.add_bos != 0)
                                            : (!opt.qwen_template && !opt.tbb_template);
    // --tokenize: the container's own tokenizer over one string, then exit.
    // No BOS and with the space marker on, which is what the reference does for
    // a bare source text; scripts/check_unigram.py is the oracle it is diffed
    // against, and it needs the ids and nothing else on stdout.
    if (opt.tokenize) {
        const vector<int> ids = app.tokenizer.encode(opt.tokenize_text, /*bos=*/false,
                                                    /*dummy_prefix=*/true);
        for (size_t i = 0; i < ids.size(); ++i)
            printf("%d%s", ids[i], i + 1 == ids.size() ? "\n" : " ");
        if (ids.empty()) printf("\n");
        fprintf(stderr, "n=%zu\n", ids.size());
        return 0;
    }
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

    if (translate_mode) {
        // tokens -> encoder -> greedy decoder, with the source <eos> appended the
        // way the reference tokenizer does it (without it the decoder repeats one
        // token forever, docs: engine/tools/ref_encdec.py).
        vector<int> src = app.tokenizer.encode(opt.prompt, /*bos=*/false,
                                              /*dummy_prefix=*/true);
        src.push_back(app.model.id_eos);
        fprintf(stderr, "encdec: %zu source tokens, %d decoder layers (act %s res %s)\n",
                src.size(), app.model.layers,
                x_format_name(app.gpu.act_fmt()), x_format_name(app.gpu.res_fmt()));
        // The attention's geometry, because a scan bound is the first thing to
        // check when a pass costs 20x what the same pass costs on another model.
        if (env_flag("LAYER_PROFILE")) app.gpu.att_geometry_line();
        // ENCDEC_REPEAT=n repeats encode+greedy decode in this process and
        // reports the last one.  A translation is short enough (a second or
        // two) that a fresh process measures shader compilation and a
        // downclocked GPU: two runs of the same binary spread by 30%, which is
        // wider than every draw-fusion option this path has.  A repeat is safe
        // because the encoder rewrites the K/V cache the decoder used and the
        // decode is greedy.
        const int reps = std::max(1, env_int("ENCDEC_REPEAT", 1));
        vector<int> out;
        double enc_s = 0.0, dec_ms = 0.0;
        long d0 = 0, d_enc = 0, d_dec = 0;
        for (int rep = 0; rep < reps; ++rep) {
            const int max_gen = opt.gen_set ? opt.gen : 32;
            d0 = app.gpu.draws();
            const auto t0 = std::chrono::steady_clock::now();
            session.encdec_encode(src);
            enc_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            d_enc = app.gpu.draws();
            const auto t_enc = std::chrono::steady_clock::now();
            out.clear();
            int tok = app.model.decoder_start_id;
            for (int i = 0; i < max_gen; ++i) {
                    vector<float> lg = session.encdec_step(tok, i);
                    if (lg.empty()) { fprintf(stderr, "encdec: no logits at %d\n", i); break; }
                    int best = 0;
                    for (int v = 1; v < (int)lg.size(); ++v) if (lg[v] > lg[best]) best = v;
                    // IGNORE_EOS extends the run for timing: a translation stops
                    // after a dozen tokens, which is too short to average the
                    // board's decode and leaves a fresh process measuring a
                    // clock ramp.  The tokens after EOS are garbage, the cost
                    // of one is not.
                    if (best == app.model.id_eos && !env_flag("IGNORE_EOS")) break;
                    out.push_back(best);
                    tok = best;
            }
            dec_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_enc).count();
            d_dec = app.gpu.draws() - d_enc;
            // A per-repetition line is what the repeat exists for: this GPU's
            // clock moves by 4x inside one run (the draws are small and bursty),
            // so the last repetition is one sample, not the answer.  Take the
            // median of the warm ones.
            if (reps > 1)
                fprintf(stderr, "ENCDEC rep %d enc=%.3fs dec_avg=%.1fms/token gen=%zu "
                        "draws enc=%ld dec=%ld\n",
                        rep, enc_s, out.empty() ? 0.0 : dec_ms / (double)out.size(),
                        out.size(), d_enc - d0, d_dec);
        }
        printf("%s\n", app.tokenizer.decode(out).c_str());
        fprintf(stderr, "encdec: %zu target tokens:", out.size());
        for (int t : out) fprintf(stderr, " %d", t);
        fprintf(stderr, "\n");
        // The switch the chat path already reads.  Without this, `TIMING=1` on the
        // one path with *two* stages reports nothing (found asking for the Pi's
        // translation timings and getting only the wall clock).
        if (char* on = getenv("TIMING"); on && *on) {
                // Two stages, so two numbers: the encoder runs once over every
                // source position (its cost is the *source's*), the decoder once
                // a token.  The chat path's `TIMING prefill/decode` would report
                // one of them as if it were the other.
                fprintf(stderr, "TIMING enc=%.3fs dec_avg=%.1fms/token gen=%zu "
                        "draws enc=%ld dec=%ld\n",
                        enc_s, out.empty() ? 0.0 : dec_ms / (double)out.size(),
                        out.size(), d_enc - d0, d_dec);
        }
        return 0;
    }

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
    // A grammar that does not compile is a startup error: the alternative is a
    // run that looks constrained and quietly is not.
    if (!chat.seed_constraint()) return 2;

    if (app.opt.chat) return chat.repl();
    return chat.complete();
    return 0;
}
