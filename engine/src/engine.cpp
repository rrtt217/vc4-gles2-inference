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

// A prompt segment can contain newlines and control characters; printing it raw
// splits one logical line across several and scatters the PROMPT tags, which is
// what made CHAT_DUMP look broken.  Escape, and print a byte count so a
// whitespace difference is still visible.
static std::string dump_escape(const std::string& t){
    std::string o;
    for (unsigned char c : t) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

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
struct TurnTiming {
        Clock::time_point start = Clock::now();
        Clock::time_point decode_start {};
        int generated = 0;
        bool decoding = false;
        const char* cut = nullptr;

        void begin_decode() {
                decode_start = Clock::now();
                decoding = true;
        }
        ~TurnTiming() {
                const char* on = getenv("TIMING");
                if (!on || !*on) return;
                const auto now = Clock::now();
                const double prefill = elapsed_ms(start, decoding ? decode_start : now) / 1000.0;
                const double avg = (decoding && generated)
                                       ? elapsed_ms(decode_start, now) / generated
                                       : 0.0;
                cerr << "TIMING prefill=" << prefill << "s decode_avg=" << avg
                     << "ms/token gen=" << generated;
                if (cut) cerr << " cut=" << cut;
                cerr << "\n";
        }
};


// Requests and control lines arrive on stdin, and a *cancel* has to be seen
// while a request is generating -- which getline() cannot do without swallowing
// the next request.  One buffer, filled from the raw fd, serves both: blocking
// lines for the request loop, and a non-consuming peek for the token loop.
static long long proc_meminfo_kb(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    const size_t n = strlen(key);
    while (std::getline(f, line))
        if (line.compare(0, n, key) == 0) return atoll(line.c_str() + n + 1);
    return -1;
}

static std::string g_inbuf;

static bool stdin_fill(int timeout_ms) {
    struct pollfd fd = {0, POLLIN, 0};
    if (poll(&fd, 1, timeout_ms) <= 0) return false;
    char tmp[4096];
    const ssize_t n = read(0, tmp, sizeof(tmp));
    if (n <= 0) return false;
    g_inbuf.append(tmp, (size_t) n);
    return true;
}

// timeout_ms < 0 blocks until a line arrives or the stream closes; 0 only takes
// what is already there.
static bool stdin_line(std::string& line, int timeout_ms) {
    for (;;) {
        const size_t nl = g_inbuf.find('\n');
        if (nl != std::string::npos) {
            line = g_inbuf.substr(0, nl);
            g_inbuf.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        if (!stdin_fill(timeout_ms)) {
            if (timeout_ms >= 0) return false;
            if (g_inbuf.empty()) return false;      // EOF
            line.swap(g_inbuf);
            g_inbuf.clear();
            return true;
        }
    }
}

// Is the buffer's first line a cancel?  Consumes it only then, so a request that
// arrived early stays queued for the request loop.
static bool take_pending_cancel() {
    const size_t nl = g_inbuf.find('\n');
    if (nl == std::string::npos) return false;
    const std::string head = g_inbuf.substr(0, nl);
    if (head.find("\"op\"") == std::string::npos || head.find("cancel") == std::string::npos)
        return false;
    g_inbuf.erase(0, nl + 1);
    return true;
}

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
    auto is_bos_token = [&](int id) { return app.model.id_bos >= 0 && id == app.model.id_bos; };
    auto is_stop_token = [&](int id) {
        if (is_bos_token(id)) return true;
        if (app.model.have_specials)
            return id == app.model.id_eos || (app.model.id_stop >= 0 && id == app.model.id_stop);
        return id == 1 || id == 2;
    };

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

    if (opt.worker) {
        // Every worker event is a JSON object; common_json decides the spelling
        // (escaping, numbers) so no line here assembles JSON by hand.
        // dump_safe, not dump: a token's bytes can be invalid UTF-8 on their own
        // (the event stream is line-framed, so a throw here would kill the turn),
        // and bad bytes become U+FFFD instead.
        auto emit = [&](const common_json& j) { cout << j.dump_safe() << "\n" << flush; };
        string req_id;            // column 13, echoed in this request's events
        auto emit_req = [&](common_json j) {
            if (!req_id.empty()) j["id"] = req_id;
            emit(j);
        };
        auto emit_error = [&](const string& message) {
            emit_req(common_json::object({{"event", string("error")}, {"message", message}}));
        };
        // Process-level defaults (CLI flags / env vars), restored before each request.
        const SamplingSettings base_sampling = g_sampling;
        vector<int> cache_tokens;
        if (!opt.prefix_cache_path.empty() && !app.load_prefix_cache_any(opt.prefix_cache_path, cache_tokens, session))
            cerr << "PREFIX_CACHE ignored " << opt.prefix_cache_path << "\n";
        // The HTTP layer renders the prompt text itself, so it has to know which
        // template this checkpoint wants.  The container states it (kind + name)
        // and the resolution above already turned it into these three flags, so
        // report the effective one: the server must not guess from a file name.
        const char* effective_template = opt.tbb_template ? "tinybrainbot"
                                        : opt.minimind3_template ? "minimind3"
                                        : opt.qwen_template ? "chatml" : "plain";
        emit(common_json::object({
            {"event", string("ready")},
            {"prefix_cache_tokens", (long long) cache_tokens.size()},
            {"template", string(effective_template)},
            {"chat_template", app.chat_tmpl != nullptr},
            {"cma_free_kb", (long long) proc_meminfo_kb("CmaFree:")},
            {"programs", (long long) app.gpu.program_count()}}));
        string line;
        while (stdin_line(line, -1)) {
            if (g_stop) break;
            if (line.empty()) continue;
            // A control line: one JSON object, e.g. {"op":"load","model":"..."}.
            // A request line starts with base64 of the prompt, whose alphabet has
            // no '{', so the two cannot be confused and the tab-separated
            // request format (and its byte-compatible events) is untouched.
            if (line[0] == '{') {
                const common_json op = common_json::parse_no_throw(line);
                const string kind = (op.is_discarded() || !op.is_object() || !op.contains("op") ||
                                     !op.at("op").is_string()) ? string() : (string) op.at("op");
                if (kind == "cancel") {
                    const string want_id = (op.contains("id") && op.at("id").is_string())
                                               ? (string) op.at("id") : string();
                    emit(common_json::object({{"event", string("cancelled")},
                                              {"id", want_id},
                                              {"idle", true}}));
                    continue;
                }
                if (kind == "stats") {
                    // What only the engine can answer: the CMA left on the board
                    // (the number that decides whether another model fits), the
                    // compiled-program count (what a swap keeps) and the plan
                    // this process is running.
                    emit(common_json::object({{"event", string("stats")},
                                              {"cma_free_kb", (long long) proc_meminfo_kb("CmaFree:")},
                                              {"mem_available_kb", (long long) proc_meminfo_kb("MemAvailable:")},
                                              {"programs", (long long) app.gpu.program_count()},
                                              {"seq", app.model.seq},
                                              {"kv_pack", (long long) (env_int("KV_PACK", 0))},
                                              {"template", app.model.template_name.empty() ? string("(kind ") + std::to_string(app.model.template_kind) + ")" : app.model.template_name},
                                              {"uptime_ms", (long long) run_time_ms()}}));
                    continue;
                }
                if (kind != "load") { emit_error("unknown op \"" + kind + "\""); continue; }
                const string want = (op.contains("model") && op.at("model").is_string())
                                        ? (string) op.at("model") : string();
                if (want.empty()) { emit_error("load: no model path"); continue; }
                // Same sequence as the REPL's /model and as startup, in the same
                // order: release the scene (CMA), read, prepare (FFN split and
                // hidden scale -- the Mats), template/tokenizer, upload, session.
                const auto t_load = Clock::now();
                app.gpu.release_scene();
                app.model.reset();
                if (!app.model.load(want)) {
                    emit_error("load: cannot read " + want);
                    // No model is loaded any more, so every later request would
                    // fail differently.  A driver deserves one clear failure.
                    return 1;
                }
                app.prepare_model();
                if (!app.setup_from_model()) return 2;
                app.upload_weights();
                session.reset();
                // The rolling prefix cache belongs to the model that produced
                // it: keeping it made the next request skip prefill against a
                // prefix the new model never saw, which answered with repeated
                // syllables ("德德德德...").  Same reason the REPL clears
                // kv_tokens.
                cache_tokens.clear();
                // The cache belongs to a container: the one loaded for the
                // previous model went with session.reset(), so the new
                // container's is loaded here.  With a directory this is the
                // whole per-model story; with a single file the header check
                // refuses it and the swap runs uncached.
                if (!opt.prefix_cache_path.empty()) app.load_prefix_cache_any(opt.prefix_cache_path, cache_tokens, session);
                // The template and the chat support belong to the *container*,
                // so a client that caches them (the server decides from these
                // whether a chat completion can be served) has to be told.
                emit(common_json::object({{"event", string("loaded")},
                                          {"model", want},
                                          {"load_ms", (double) elapsed_ms(t_load, Clock::now())},
                                          {"template", app.model.template_name.empty() ? app.model.template_kind >= 2 ? string("minimind3") : string("chatml") : app.model.template_name},
                                          {"chat_template", app.chat_tmpl != nullptr},
                                          {"chat_support", app.model.have_capabilities ? (bool) (app.model.capabilities & 1) : true},
                                          {"seq", app.model.seq},
                                          {"cma_free_kb", (long long) proc_meminfo_kb("CmaFree:")},
                                          {"programs", (long long) app.gpu.program_count()}}));
                continue;
            }
            vector<string> f;
            size_t start = 0;
            while (start <= line.size()) {
                size_t tab = line.find('\t', start);
                f.push_back(line.substr(start, tab == string::npos ? string::npos : tab - start));
                if (tab == string::npos) break;
                start = tab + 1;
            }
            if (f.size() < 5) { emit_error("bad request line"); continue; }
            string prompt = base64_decode(f[0]);
            int max_new = max(0, atoi(f[1].c_str()));
            float wt = atof(f[2].c_str());
            float wp = atof(f[3].c_str());
            uint32_t seed = (uint32_t)strtoul(f[4].c_str(), nullptr, 10);
            if (wp <= 0.0f || wp > 1.0f) wp = 0.9f;
            // Optional per-request sampling fields (6..10).  Absent fields keep the
            // process defaults, so an old client that sends five columns still works.
            g_sampling = base_sampling;
            if (f.size() > 5 && !f[5].empty()) g_sampling.repetition_penalty = atof(f[5].c_str());
            if (f.size() > 6 && !f[6].empty()) g_sampling.frequency_penalty = atof(f[6].c_str());
            if (f.size() > 7 && !f[7].empty()) g_sampling.presence_penalty = atof(f[7].c_str());
            if (f.size() > 8 && !f[8].empty()) g_sampling.no_repeat_ngram = atoi(f[8].c_str());
            if (f.size() > 9 && !f[9].empty()) g_sampling.min_p = atof(f[9].c_str());
            // Column 13: opaque request id, echoed so a log line and an HTTP
            // request can be matched.  Column 14: 1 = report a logprob per
            // token in `done`.
            req_id = f.size() > 12 ? f[12] : string();
            const bool want_logprobs = f.size() > 13 && !f[13].empty() && atoi(f[13].c_str()) != 0;
            if (g_sampling.repetition_penalty < 0.5f) g_sampling.repetition_penalty = 0.5f;
            if (g_sampling.repetition_penalty > 2.0f) g_sampling.repetition_penalty = 2.0f;
            if (g_sampling.frequency_penalty < -2.0f) g_sampling.frequency_penalty = -2.0f;
            if (g_sampling.frequency_penalty > 2.0f) g_sampling.frequency_penalty = 2.0f;
            if (g_sampling.presence_penalty < -2.0f) g_sampling.presence_penalty = -2.0f;
            if (g_sampling.presence_penalty > 2.0f) g_sampling.presence_penalty = 2.0f;
            if (g_sampling.no_repeat_ngram < 0) g_sampling.no_repeat_ngram = 0;
            if (g_sampling.no_repeat_ngram > 8) g_sampling.no_repeat_ngram = 8;
            if (g_sampling.min_p < 0.0f) g_sampling.min_p = 0.0f;
            if (g_sampling.min_p > 1.0f) g_sampling.min_p = 1.0f;
            // Column 11 (new): base64 of {"messages":[..],"tools":[..]}.  The
            // server sends it only when this container carries a chat template,
            // because only then can the engine be the renderer.
            vector<PromptPart> req_parts;
            if (f.size() > 10 && !f[10].empty()) {
                vector<ChatMsg> req_msgs;
                string req_tools, perr;
                int req_thinking = -1;
                if (!chat_parse_request(base64_decode(f[10]), req_msgs, req_tools, perr, req_thinking)) {
                    emit_error("bad request: " + perr);
                    continue;
                }
                if (!app.chat_tmpl) {
                    emit_error("this checkpoint carries no chat template; send a prompt string");
                    continue;
                }
                req_parts = app.chat_tmpl->render(req_msgs, req_tools, true,
                                              req_thinking >= 0 ? req_thinking != 0 : opt.open_thinking,
                                              app.bos_str, app.eos_str);
                if (chat_dump)
                    for (auto& p : req_parts)
                        fprintf(stderr, "PROMPT[%s] %3zu B  %s\n", p.is_input ? "in" : "tmpl",
                            p.text.size(), dump_escape(p.text).c_str());
            }
            auto tokens = req_parts.empty() ? app.tokenizer.encode(opt.prompt, false, false)
                                            : app.tokenizer.encode_parts(req_parts, enc_bos, enc_bos);
            if (chat_dump) {
                fprintf(stderr, "PROMPT_IDS %zu:", tokens.size());
                for (int t : tokens) fprintf(stderr, " %d", t);
                fprintf(stderr, "\n");
            }
            if (tokens.empty()) {
                emit_error("empty prompt");
                continue;
            }
            if ((int)tokens.size() >= app.model.seq) {
                emit_error("context full");
                continue;
            }
            int common = 0;
            while (common < (int)cache_tokens.size() && common < (int)tokens.size() &&
                   cache_tokens[common] == tokens[common]) ++common;
            // Same rule as the CLI: a partial prefix reuse invalidates the
            // loaded conv history.
            if (common < (int)cache_tokens.size() && app.model.hybrid)
                session.zero_conv_state();
            int prefill_from = common;
            if (prefill_from >= (int)tokens.size()) prefill_from = (int)tokens.size() - 1;
            int prefill_tokens = (int)tokens.size() - prefill_from;
            emit_req(common_json::object({
                {"event", string("start")},
                {"prompt_tokens", (long long) tokens.size()},
                {"cached_tokens", common},
                {"prefill_tokens", prefill_tokens},
            }));
            int req_top_k = opt.top_k;
            if (f.size() > 11 && !f[11].empty()) req_top_k = atoi(f[11].c_str());
            if (req_top_k < 0) req_top_k = 0;
            bool greedy_req = wt <= 0.0f;
            // The candidate path reads back only the top-k logits, so it cannot
            // answer "what was the probability of this token" over the whole
            // vocabulary; a request that asks for it uses the full logits.
            bool cand_mode = opt.top_k_candidates && req_top_k > 0 && !greedy_req &&
                             !sampling_penalties_active() && !want_logprobs;
            vector<float> logits;
            CandidateResult cand;
            for (int i = prefill_from; i < (int)tokens.size(); ++i) {
                if (g_stop) break;
                bool last = i + 1 == (int)tokens.size();
                if (last && cand_mode) {
                    cand = session.step_candidates(tokens[i], i, req_top_k, wt, wp);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(tokens[i], i, last);
                }
            }
            if (g_stop) break;
            int pos = (int)tokens.size();
            max_new = min(max_new, app.model.seq - pos);
    // The request's own seed (column 5 of the request line).  It was parsed
    // and then never used -- `rng` was built from the CLI seed, so the seed the
    // server sends on every request did nothing; -Wall found it on 2026-11-14.
    mt19937 rng(seed);
            vector<common_json> lp_rows;   // one per generated token, when asked
            string pending, full, finish = "length";
            int generated = 0;
            int prev = tokens.back();
            vector<int> all_tokens = tokens;
            vector<int> recent;
            bool cancelled = false;
            for (int i = 0; i < max_new; ++i) {
                if (g_stop) { finish = "stop"; break; }
                // A cancel is a control line on the same stream.  A draw cannot
                // be interrupted, so a token is the granularity; it keeps the
                // request protocol intact instead of needing a thread.
                if (i % 4 == 0 && stdin_fill(0) && take_pending_cancel()) {
                    cancelled = true;
                    finish = "cancelled";
                    break;
                }
                apply_sampling_penalties(logits, recent);
                int next;
                float next_lp = 0.0f;
                if (want_logprobs) {
                    const SampledToken s = greedy_req
                        ? argmax_token_report(logits)
                        : sample_token_report(logits, wt, wp, req_top_k, rng);
                    next = s.id;
                    next_lp = s.logprob;
                } else {
                    next = greedy_req ? argmax_token(logits)
                                      : (cand_mode ? sample_candidate_result(cand, wt, wp, req_top_k, rng)
                                                   : sample_token(logits, wt, wp, req_top_k, rng));
                }
                const bool is_eos = app.model.have_specials
                        ? (next == app.model.id_eos || (app.model.id_stop >= 0 && next == app.model.id_stop))
                        : (next == 1 || next == 2);
                if (is_eos) {
                    session.step(next, pos++, false);
                    all_tokens.push_back(next);
                    finish = "stop";
                    break;
                }
                string piece = app.tokenizer.decode({next}, prev);
                if (want_logprobs)
                    lp_rows.push_back(common_json::object(
                        {{"id", next}, {"text", piece}, {"lp", next_lp}}));
                pending += piece;
                string text = utf8_take(pending);
                if (!text.empty()) {
                    full += text;
                    emit_req(common_json::object({{"event", string("token")}, {"text", text}}));
                }
                prev = next;
                ++generated;
                all_tokens.push_back(next);
                recent.push_back(next);
                if (cand_mode) {
                    cand = session.step_candidates(next, pos++, req_top_k, wt, wp);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(next, pos++, true);
                }
            }
            if (!pending.empty()) {
                string text = "\xEF\xBF\xBD";
                full += text;
                emit_req(common_json::object({{"event", string("token")}, {"text", text}}));
            }
            if (g_stop) finish = "stop";
            if (g_stop) cache_tokens.clear();
            else cache_tokens = all_tokens;
            if (cancelled)
                emit_req(common_json::object({{"event", string("cancelled")}, {"id", req_id}}));
            common_json done_ev = common_json::object({
                {"event", string("done")},
                {"text", full},
                {"finish_reason", finish},
                {"prompt_tokens", (long long) tokens.size()},
                {"completion_tokens", generated},
                {"cached_tokens", common},
                {"prefill_tokens", prefill_tokens},
                {"cma_free_kb", (long long) proc_meminfo_kb("CmaFree:")},
                {"mem_available_kb", (long long) proc_meminfo_kb("MemAvailable:")},
                {"programs", (long long) app.gpu.program_count()},
            });
            if (want_logprobs) {
                common_json rows = common_json::array();
                for (auto& r : lp_rows) rows.push_back(r);
                done_ev["logprobs"] = rows;
            }
            emit_req(done_ev);
        }
        return 0;
    }


    vector<int> cli_cache_tokens;
    if (!opt.worker && !opt.prefix_cache_path.empty()) {
        if (app.load_prefix_cache_any(opt.prefix_cache_path, cli_cache_tokens, session)) {
            cerr << "PREFIX_CACHE loaded " << cli_cache_tokens.size() << " tokens\n";
            // Probe: write the just-loaded rows straight back out, so a built
            // file and a loaded file can be compared byte for byte.
            if (getenv("PREFIX_CACHE_RESAVE")) {
                const char* rp = getenv("PREFIX_CACHE_RESAVE");
                if (session.save_prefix_cache(rp, cli_cache_tokens))
                    cerr << "PREFIX_CACHE resaved " << rp << "\n";
                else cerr << "PREFIX_CACHE resave failed\n";
            }
        } else
            cerr << "PREFIX_CACHE ignored " << opt.prefix_cache_path << "\n";
    }
    auto cache_prefix_len = [&](const vector<int>& tokens) {
        size_t n = 0;
        while (n < cli_cache_tokens.size() && n < tokens.size() &&
               cli_cache_tokens[n] == tokens[n]) ++n;
        // The cache carries KV rows, not the logits of the last cached token,
        // so a prompt that is fully covered must still run its last token at
        // its own position: generation would otherwise start from an empty
        // logits vector and argmax() would return token 0.  The worker path
        // clamps prefill_from the same way; this is the CLI equivalent.
        if (!tokens.empty() && n == tokens.size()) n = tokens.size() - 1;
        return (int)n;
    };

    // Optional: block the <|...|> marker tokens while generating.  Off by
    // default, because the engine cannot tell a marker from content - this
    // model's <think> is a special token too, and blocking it would remove the
    // scratchpad the model was trained to use.  Even for the marker case it is
    // partial: the echo it was written for started with a plain '<' and spelled
    // the rest out, which no token mask can stop.
    vector<char> marker_mask;
    if (env_flag("BLOCK_MARKERS")) {
        marker_mask.assign((size_t)app.model.vocab, 0);
        for (const auto& sp : app.tokenizer.specials()) {
            const int id = sp.second;
            if (id < 0 || id >= app.model.vocab) continue;
            if (id == app.model.id_eos || id == app.model.id_stop || id == app.model.id_bos) continue;
            if (sp.first.size() < 3 || sp.first.find('|') == string::npos) continue;
            marker_mask[(size_t)id] = 1;
        }
    }
    auto mask_markers = [&](vector<float>& logits) {
        for (size_t i = 0; i < logits.size() && i < marker_mask.size(); ++i)
            if (marker_mask[i]) logits[i] = -1e30f;
    };

    mt19937 rng(opt.seed);
    // Tokens currently in the KV cache; the REPL keeps this so a turn can
    // prefill only what changed instead of the whole conversation.
    vector<int> kv_tokens;
    auto run_turn = [&](const string& user, bool first, int pos, vector<ChatMsg>* hist = nullptr) {
        TurnTiming tt;
        string text;
        vector<PromptPart> parts;
        if (app.chat_tmpl) {
            const string& sys = app.system_prompt;
            vector<ChatMsg> msgs;
            if (hist) {
                // The REPL keeps the conversation; the system turn goes in once,
                // at the front, and the whole history is re-rendered so the
                // template sees the same shape it was trained on.
                if (hist->empty() && !sys.empty()) hist->push_back(ChatMsg{"system", sys, "", ""});
                hist->push_back(ChatMsg{"user", user, "", ""});
                msgs = *hist;
            } else {
                if (first && !sys.empty()) msgs.push_back(ChatMsg{"system", sys, "", ""});
                msgs.push_back(ChatMsg{"user", user, "", ""});
            }
            parts = app.chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/true,
                                      opt.open_thinking, app.bos_str, app.eos_str);
            if (opt.open_thinking) {
                bool has = false;
                for (auto& p : parts) if (p.text.find("<think>") != string::npos) has = true;
                if (!has) parts.push_back(PromptPart{"<think>", false});
            }
        } else if (opt.minimind3_template) {
            string sys = app.system_prompt;
            if (!opt.tools_json.empty()) {
                sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                       "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                       opt.tools_json +
                       "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                       "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
            }
            if (first && !sys.empty())
                text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
            text += "<|im_start|>user\n" + user + "<|im_end|>\n"
                    "<|im_start|>assistant\n";
            text += opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        } else if (opt.tbb_template) {
            // No system turn and no BOS: the markers carry the roles, and the
            // first user turn opens with a space (the model was trained so).
            // "<|user|> {q}<|assistant|> " with a trailing space: the model
            // was trained with a space after the assistant marker, and without
            // it the first token is <|end|> (measured on the q4 and i16 files,
            // where the other four wrappers I tried collapse to one token).
            text = string("<|user|> ") + user + "<|assistant|> ";
        } else if (opt.qwen_template) {
            if (first)
                text = "<|im_start|>system\n" + app.system_prompt +
                       "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            else
                text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            if (opt.open_thinking) text += "<think>\n";
        } else {
            if (first && !app.system_prompt.empty())
                text = "[INST] <<SYS>>\n" + app.system_prompt + "\n<</SYS>>\n\n" + user + " [/INST]";
            else
                text = "[INST] " + user + " [/INST]";
        }
        if (chat_dump) {
            if (parts.empty()) {
                // No template (kind 0): the prompt is one hand-built string.
                // Without this branch CHAT_DUMP printed nothing for such a
                // container, which is indistinguishable from "the switch does
                // not work".
                fprintf(stderr, "PROMPT[text] %3zu B  %s\n", text.size(),
                        dump_escape(text).c_str());
            } else {
                for (auto& p : parts)
                    fprintf(stderr, "PROMPT[%s] %3zu B  %s\n", p.is_input ? "in" : "tmpl",
                            p.text.size(), dump_escape(p.text).c_str());
            }
        }
        auto tokens = parts.empty() ? app.tokenizer.encode(text, enc_bos, enc_bos)
                                    : app.tokenizer.encode_parts(parts, enc_bos, enc_bos);
        int skip = (first && !cli_cache_tokens.empty()) ? cache_prefix_len(tokens) : 0;
        // A hybrid model that only reuses part of the cache must not keep the
        // loaded conv history: the history is a running state, and the tokens
        // that produced it are not the ones about to be prefilled.
        if (first && !cli_cache_tokens.empty() && skip < (int)cli_cache_tokens.size())
            session.zero_conv_state();
        if (hist) {
            size_t n = 0;
            while (n < kv_tokens.size() && n < tokens.size() && kv_tokens[n] == tokens[n]) ++n;
            // the KV holds rows, not the last token's logits, so the final
            // token of an otherwise-cached prompt still has to run
            if (!tokens.empty() && n == tokens.size()) n = tokens.size() - 1;
            skip = (int)n;
        }
        if (getenv("CHAT_CACHE"))
            fprintf(stderr, "CHAT_CACHE turn cached=%d prefill=%zu prompt=%zu\n",
                    skip, tokens.size() - (size_t)skip, tokens.size());
        for (int i = 0; i < skip; ++i) pos = i + 1;
        vector<float> logits;
        CandidateResult cand;
        bool cand_mode = opt.top_k_candidates && opt.top_k > 0 && !opt.greedy && !sampling_penalties_active();
        for (size_t i = skip; i < tokens.size(); ++i) {
            if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
            if (pos >= app.model.seq) { tt.cut = "context"; cout << "[context full]\n"; return; }
            bool last = (i + 1 == tokens.size());
            if (last && cand_mode) {
                cand = session.step_candidates(tokens[i], pos++, opt.top_k, opt.temp, opt.top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else if (!last && i + 2 < tokens.size() && pos + 1 < app.model.seq && session.pair_ok()) {
                // Two prompt tokens per layer pass; the last token stays on the
                // single-position path because its logits are the ones we need.
                session.step_pair(tokens[i], pos, tokens[i + 1], pos + 1, false);
                pos += 2;
                ++i;
            } else {
                logits = session.step(tokens[i], pos++, last);
            }
        }
        if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
        tt.begin_decode();
        int generated = 0;
        if (!cand_mode && getenv("LOGITS_OUT")) {
            ofstream f(getenv("LOGITS_OUT"), ios::binary);
            f.write((char*)logits.data(), logits.size() * 4);
        }
        cout << "Assistant: "; cout.flush();
        string reply;
        string pending;   // holds an incomplete UTF-8 sequence between tokens
        int prev = tokens.empty() ? 1 : tokens.back();
        vector<int> recent;
        for (int i = 0; i < opt.gen && pos < app.model.seq; ++i) {
            if (g_stop) { tt.cut = "decode"; app.gpu.sync(); return; }
            if (!marker_mask.empty()) mask_markers(logits);
            apply_sampling_penalties(logits, recent);
            int next = opt.greedy ? argmax_token(logits)
                              : (cand_mode ? sample_candidate_result(cand, opt.temp, opt.top_p, opt.top_k, rng)
                                           : sample_token(logits, opt.temp, opt.top_p, opt.top_k, rng));
            if (!getenv("IGNORE_EOS") && is_stop_token(next)) {
                session.step(next, pos++, false);
                break;
            }
            // A token may carry only the first bytes of a multi-byte character.
            // Emitting those bytes would put ill-formed UTF-8 into the reply,
            // and the reply goes back into the template on the next turn, where
            // the JSON marshal rejects it (SIGABRT).  Hold the tail until it is
            // complete, exactly as the worker does.
            pending += app.tokenizer.decode({next}, prev);
            string chunk = utf8_take(pending);
            if (!chunk.empty()) {
                cout << chunk; cout.flush();
                reply += chunk;
            }
            tokens.push_back(next);
            recent.push_back(next);
            ++generated;
            tt.generated = generated;
            prev = next;
            if (cand_mode) {
                cand = session.step_candidates(next, pos++, opt.top_k, opt.temp, opt.top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else {
                logits = session.step(next, pos++, true);
            }
        }
        if (!pending.empty()) {
            cout << "\xEF\xBF\xBD"; cout.flush();
            reply += "\xEF\xBF\xBD";
        }
        cout << "\n";
        // The loop also stops at the KV limit, which otherwise looks like a
        // model that simply had nothing more to say.
        if (pos >= app.model.seq) tt.cut = "context";
        if (!g_quiet && !g_stop && generated >= opt.gen && !is_stop_token(prev) && generated)
            fprintf(stderr, "[stopped at the --gen limit: %d tokens.  This model did "
                            "not emit its end token (eos %d, stop %d); raise --gen]\n",
                    generated, app.model.id_eos, app.model.id_stop);
        if (hist) {
            hist->push_back(ChatMsg{"assistant", reply, "", ""});
            kv_tokens = tokens;
        }
        if (getenv("TOKENS_OUT")) {
            ofstream f(getenv("TOKENS_OUT"), ios::binary);
            for (int t : tokens) f.write((char*)&t, 4);
        }
        if (g_stop) app.gpu.sync();
    };

    // The assistant turn is re-rendered from the template, so the raw
    // <tool_call> blocks must not also travel inside `content` or the template
    // would emit them twice.

    auto run_tool_turn = [&](const string& user, bool first, vector<ChatMsg>* hist = nullptr) {
        string sys = app.system_prompt.empty()
            ? string("你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。")
            : app.system_prompt;
        if (!opt.tools_json.empty()) {
            sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                   opt.tools_json +
                   "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        // Template path: the whole prompt is re-rendered from this list every
        // round, so the tool loop is model-agnostic.  The legacy path below
        // keeps building one string, which is why it only ever worked for the
        // ChatML family.  With a history the list *is* the conversation, so a
        // REPL turn after a tool round-trip sees the same exchange the HTTP
        // server would have sent; without it (one-shot --prompt) the list is
        // local and holds this turn only.
        vector<ChatMsg> local_msgs;
        vector<ChatMsg>& msgs = hist ? *hist : local_msgs;
        if (app.chat_tmpl) {
            if (hist) {
                if (msgs.empty() && !app.system_prompt.empty())
                    msgs.push_back(ChatMsg{"system", app.system_prompt, "", ""});
            } else if (!app.system_prompt.empty()) {
                // Always the system turn: dropping it after the first user line
                // was the old bug, and it also kills the shared prefix.
                msgs.push_back(ChatMsg{"system", app.system_prompt, "", ""});
            }
            msgs.push_back(ChatMsg{"user", user, "", ""});
        }
        string text;
        if (first || !opt.tools_json.empty())
            text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
        text += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        text += opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

        for (int round = 0; round < 3; ++round) {
            TurnTiming tt;   // one line per model turn, interrupted or not
            vector<PromptPart> parts;
            if (app.chat_tmpl)
                parts = app.chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/true,
                                          opt.open_thinking, app.bos_str, app.eos_str);
            // A template without an open_thinking switch (LFM2.5's) still gets
            // the block: the caller asked for it, and `<think>` is one token.
            if (app.chat_tmpl && opt.open_thinking) {
                bool has = false;
                for (auto& p : parts) if (p.text.find("<think>") != string::npos) has = true;
                if (!has) parts.push_back(PromptPart{"<think>", false});
            }
            if (chat_dump) {
                if (parts.empty()) {
                    // No template: the prompt is one string built by hand.
                    // Without this branch CHAT_DUMP printed nothing at all for
                    // a plain container, which reads exactly like the switch
                    // being broken.
                    fprintf(stderr, "PROMPT[text] %3zu B  %s\n", text.size(),
                            dump_escape(text).c_str());
                } else {
                    for (auto& p : parts)
                        fprintf(stderr, "PROMPT[%s] %3zu B  %s\n", p.is_input ? "in" : "tmpl",
                                p.text.size(), dump_escape(p.text).c_str());
                }
            }
            auto tokens = app.chat_tmpl ? app.tokenizer.encode_parts(parts, enc_bos, enc_bos)
                                    : app.tokenizer.encode(text, false, false);
            // CHAT_DUMP also prints the ids the model will see: a template that
            // renders to the same bytes can still tokenise differently, and the
            // ids are what the model actually reads.
            if (chat_dump) {
                fprintf(stderr, "PROMPT_IDS %zu:", tokens.size());
                for (int t : tokens) fprintf(stderr, " %d", t);
                fprintf(stderr, "\n");
            }
            if ((int)tokens.size() >= app.model.seq) { tt.cut = "context"; cout << "[context full]\n"; return; }
            vector<float> logits;
            CandidateResult cand;
            bool cand_mode = opt.top_k_candidates && opt.top_k > 0 && !opt.greedy && !sampling_penalties_active();
            int skip = 0;
            if (!kv_tokens.empty()) {
                size_t n = 0;
                while (n < kv_tokens.size() && n < tokens.size() && kv_tokens[n] == tokens[n]) ++n;
                // A fully covered prompt still runs its last token: the KV keeps
                // rows, not the logits that token would produce.
                if (!tokens.empty() && n == tokens.size()) n = tokens.size() - 1;
                skip = (int)n;
            } else if (first && round == 0 && !cli_cache_tokens.empty()) {
                skip = cache_prefix_len(tokens);
            }
            if (getenv("CHAT_CACHE"))
                fprintf(stderr, "CHAT_CACHE round=%d cached=%d prefill=%zu prompt=%zu\n",
                        round, skip, tokens.size() - (size_t)skip, tokens.size());
            for (size_t i = skip; i < tokens.size(); ++i) {
                if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
                bool last = (i + 1 == tokens.size());
                if (last && cand_mode) {
                    cand = session.step_candidates(tokens[i], (int)i, opt.top_k, opt.temp, opt.top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else if (!last && i + 2 < tokens.size() && (int)i + 1 < app.model.seq && session.pair_ok()) {
                    session.step_pair(tokens[i], (int)i, tokens[i + 1], (int)i + 1, false);
                    ++i;
                } else {
                    logits = session.step(tokens[i], (int)i, last);
                }
            }
            if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
            tt.begin_decode();
            cout << "Assistant: "; cout.flush();
            string out;
            string pending;
            int prev = tokens.empty() ? 1 : tokens.back();
            int pos = (int)tokens.size();
            vector<int> recent;
            for (int i = 0; i < opt.gen && pos < app.model.seq; ++i) {
                if (g_stop) { tt.cut = "decode"; app.gpu.sync(); return; }
                apply_sampling_penalties(logits, recent);
                int next = opt.greedy ? argmax_token(logits)
                                  : (cand_mode ? sample_candidate_result(cand, opt.temp, opt.top_p, opt.top_k, rng)
                                               : sample_token(logits, opt.temp, opt.top_p, opt.top_k, rng));
                if (is_stop_token(next)) {
                    if (!g_quiet && is_bos_token(next))
                        fprintf(stderr, "\n[stopped after %d tokens: the model emitted its "
                                        "BOS token mid-generation (%d)]\n", i, next);
                    session.step(next, pos++, false);
                    break;
                }
                pending += app.tokenizer.decode({next}, prev);
                string chunk = utf8_take(pending);
                if (!chunk.empty()) {
                    cout << chunk; cout.flush();
                    out += chunk;
                }
                tokens.push_back(next);
                recent.push_back(next);
                ++tt.generated;
                prev = next;
                if (cand_mode) {
                    cand = session.step_candidates(next, pos++, opt.top_k, opt.temp, opt.top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(next, pos++, true);
                }
            }
            if (!pending.empty()) {
                cout << "\xEF\xBF\xBD"; cout.flush();
                out += "\xEF\xBF\xBD";
            }
            cout << "\n";
            if (pos >= app.model.seq) tt.cut = "context";
            string reasoning;
            string body = take_thinking(out, reasoning);
            vector<ToolCall> calls = parse_tool_calls(body);
            if (app.chat_tmpl) {
                // The final answer goes into the history too.  Before this the
                // tool path returned without recording it, so the next REPL turn
                // lost the answer and had no shared prefix to reuse.
                msgs.push_back(ChatMsg{"assistant", strip_tool_calls(body),
                                       calls.empty() ? string() : tool_calls_json(calls),
                                       reasoning});
            }
            kv_tokens = tokens;
            if (getenv("TOKENS_OUT")) {
                ofstream f(getenv("TOKENS_OUT"), ios::binary);
                for (int t : tokens) f.write((char*)&t, 4);
            }
            if (calls.empty()) return;
            if (!app.chat_tmpl) {
                text += out;
                text += "<|im_end|>\n<|im_start|>user\n";
            }
            for (const auto& c : calls) {
                string result = run_tool(c.name, c.args);
                cout << "[tool] " << c.name << " " << c.args << "\n";
                cout << "[tool result] " << result << "\n";
                if (app.chat_tmpl) msgs.push_back(ChatMsg{"tool", result, "", ""});
                else text += "<tool_response>\n" + result + "\n</tool_response>\n";
            }
            if (!app.chat_tmpl) {
                text += "<|im_end|>\n<|im_start|>assistant\n";
                text += opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
            }
        }
    };

    if (opt.chat) {
        if (!opt.prompt_set) opt.prompt.clear();
        if (!opt.prompt.empty()) {
            if (opt.tools_json.empty()) run_turn(opt.prompt, true, 0);
            else run_tool_turn(opt.prompt, true);
        } else {
            bool first = true;
            int cancels = 0;
            vector<ChatMsg> history;
            while (!g_stop) {
                string line;
                if (!line_edit_read("User: ", line)) break;
                if (line.empty()) {
                    // Ctrl-C cancels the line; a second one in a row leaves,
                    // which is what people expect from a prompt.
                    if (++cancels >= 2) break;
                    fprintf(stderr, "(Ctrl-C again or /exit to leave)\n");
                    continue;
                }
                cancels = 0;
                if (line == "/exit" || line == "/quit") break;
                if (line == "/clear") {
                    first = true;
                    history.clear();
                    kv_tokens.clear();
                    cout << "[conversation cleared]\n";
                    continue;
                }
                if (line.rfind("/model ", 0) == 0) {
                    const string path = line.substr(7);
                    const auto t_swap = Clock::now();
                    app.gpu.release_scene();
                    app.model.reset();
                    if (!app.model.load(path)) { cerr << "no model loaded; leaving\n"; return 1; }
                    // The startup path refuses a container that does not
                    // advertise chat support while --chat is in effect; a swap
                    // must apply the same rule instead of quietly running it
                    // (stories15M's capabilities are 0, and it happily answered
                    // through the built-in [INST] wrapper before this guard).
                    if (app.model.have_capabilities && opt.chat && !(app.model.capabilities & 1)) {
                        cerr << "/model: this checkpoint does not advertise chat support "
                                "(capabilities 0x" << std::hex << app.model.capabilities << std::dec
                             << "); leaving\n";
                        return 2;
                    }
                    app.prepare_model();          // FFN split, hidden scale, seq clamp
                    if (!app.setup_from_model()) return 2;
                    app.upload_weights();
                    session.reset();
                    history.clear();
                    kv_tokens.clear();
                    first = true;
                    cout << "[model] " << (app.model.template_name.empty() ? string("(kind 0)") : app.model.template_name)
                         << "  chat_template " << (app.chat_tmpl ? "yes" : "no")
                         << "  " << (long) elapsed_ms(t_swap, Clock::now()) << " ms\n" << flush;
                    continue;
                }
                if (line == "/help") {
                    cout << "/clear  forget the conversation so far\n"
                            "/model PATH  load another container, keeping the GPU context\n"
                            "/exit   leave\n"
                            "/help   this list\n"
                            "anything else is sent to the model; arrows edit, Up/Down recall,\n"
                            "Ctrl-Left/Right (or Alt-b/f) move by word.\n";
                    continue;
                }
                if (opt.tools_json.empty()) run_turn(line, first, 0, app.chat_tmpl ? &history : nullptr);
                else run_tool_turn(line, first, app.chat_tmpl ? &history : nullptr);
                first = false;
                if (g_stop) {
                    cout << "\n[interrupted]\n";
                    g_stop = 0;
                }
            }
            if (g_stop) cout << "\n";
        }
        return 0;
    }

    auto tokens = app.tokenizer.encode(opt.prompt, true, true);
    // The single-shot path never reuses a loaded prefix, so a hybrid model must
    // not keep the conv history that came with the cache file.
    if (!cli_cache_tokens.empty()) session.zero_conv_state();
    cout << "tokens:";
    for (int t : tokens) cout << " " << t;
    cout << "\n";
    vector<float> logits;
    CandidateResult cand;
    bool cand_mode = opt.top_k_candidates && opt.top_k > 0 && !opt.greedy && !sampling_penalties_active();
    // PREFILL_REPEAT=n runs the same prefill n times in one process.  The first
    // pass pays every lazy shader link and every QPU program generation, which
    // on VC4 is seconds; a comparison between two modes that compile different
    // program sets otherwise measures the compiler, not the kernels (that is
    // how step 7c came out "6% slower" while its kernels were faster).  The
    // repeats are exact re-runs - every KV entry is rewritten with the same
    // bytes before it is read - so prefill_warm is the number the server sees.
    auto prefill_once = [&]() {
        vector<float> lg;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (g_stop) break;
            bool last = (i + 1 == tokens.size());
            if (last && cand_mode) {
                cand = session.step_candidates(tokens[i], (int)i, opt.top_k, opt.temp, opt.top_p);
                if (!cand.complete) { lg = std::move(cand.full); cand_mode = false; }
            } else if (!last && i + 2 < tokens.size() && (int)i + 1 < app.model.seq && session.pair_ok()) {
                session.step_pair(tokens[i], (int)i, tokens[i + 1], (int)i + 1, false);
                ++i;
            } else {
                lg = session.step(tokens[i], (int)i, last);
            }
        }
        return lg;
    };
    TurnTiming tt;
    auto start = Clock::now();
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  first step\n", run_time_ms());
    logits = prefill_once();
    if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); cout << "\n[interrupted]\n"; return 0; }
    double prefill = chrono::duration<double>(Clock::now() - start).count();
    cout << "prefill " << prefill << " s\n";
    if (getenv("PREFILL_REPEAT")) {
        int n = max(1, atoi(getenv("PREFILL_REPEAT")));
        for (int r = 1; r < n; ++r) {
            auto t = Clock::now();
            logits = prefill_once();
            double w = chrono::duration<double>(Clock::now() - t).count();
            cout << "prefill_warm " << w << " s\n";
        }
    }

    if (!cand_mode && getenv("LOGITS_OUT")) {
        ofstream f(getenv("LOGITS_OUT"), ios::binary);
        f.write((char*)logits.data(), logits.size() * 4);
    }
    tt.begin_decode();
    int generated = 0;
    double decode_ms = 0;
    vector<int> recent;
    bool ended_on_eos = false;
    const char* stop_reason = nullptr;
    int stop_id = -1;
    for (int i = 0; i < opt.gen; ++i) {
        if (g_stop) break;
        // The chat and tool loops stop at the container's seq; this one did
        // not, so a --gen larger than seq walked past the KV cache and the rope
        // table and printed whatever came back.  Measured on void.1 (seq 1024,
        // --gen 4096): the text turns to "walk inkinkinkinkin" at position
        // 1023 and stays garbage -- and the same file packed at seq 2048 turns
        // to " t t t t" at 2047 instead, which is how the boundary was found.
        if ((int)tokens.size() >= app.model.seq) {
            tt.cut = "context";
            if (!g_quiet)
                fprintf(stderr, "\n[context full] %d tokens, seq %d; raise --seq when "
                                "packing or lower --gen\n", generated, app.model.seq);
            break;
        }
        if (!marker_mask.empty()) mask_markers(logits);
        apply_sampling_penalties(logits, recent);
        int next = opt.greedy ? argmax_token(logits)
                          : (cand_mode ? sample_candidate_result(cand, opt.temp, opt.top_p, opt.top_k, rng)
                                       : sample_token(logits, opt.temp, opt.top_p, opt.top_k, rng));
        if (!getenv("IGNORE_EOS") && is_stop_token(next)) {
            ended_on_eos = true;
            stop_reason = is_bos_token(next) ? "the model emitted its BOS token mid-generation"
                                             : "the model emitted its end token";
            stop_id = next;
            break;
        }
        cout << app.tokenizer.decode({next}, tokens.empty() ? 1 : tokens.back()); cout.flush();
        tokens.push_back(next);
        recent.push_back(next);
        auto a = Clock::now();
        if (cand_mode) {
            cand = session.step_candidates(next, (int)tokens.size() - 1, opt.top_k, opt.temp, opt.top_p);
            if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
        } else {
            logits = session.step(next, (int)tokens.size() - 1, true);
        }
        decode_ms += elapsed_ms(a, Clock::now());
        ++generated;
        tt.generated = generated;
    }
    cout << "\n";
    if (getenv("TOKENS_OUT")) {
        ofstream f(getenv("TOKENS_OUT"), ios::binary);
        for (int t : tokens) f.write((char*)&t, 4);
    }
    if (g_stop) { tt.cut = "decode"; app.gpu.sync(); cout << "[interrupted]\n"; return 0; }
    // A completion model that never emits its end token looks like "it will not
    // stop"; say which limit ended it so that is diagnosable from the output.
    if (!g_quiet && ended_on_eos && stop_reason && is_bos_token(stop_id))
        fprintf(stderr, "\n[stopped after %d tokens: %s (%d)]\n", generated, stop_reason, stop_id);
    if (!g_quiet && !ended_on_eos && generated >= opt.gen)
        fprintf(stderr, "\n[stopped at the --gen limit: %d tokens.  This model did not "
                        "emit its end token (eos %d, stop %d); raise --gen or press "
                        "Ctrl-C]\n", generated, app.model.id_eos, app.model.id_stop);
    if (generated) cout << "decode avg " << decode_ms / generated << " ms/token\n";
    if (getenv("DRAWCOUNT") && generated)
        cout << "draws " << app.gpu.draws() << " total, "
             << (double)app.gpu.draws() / generated << " per token\n";
    return 0;
}
