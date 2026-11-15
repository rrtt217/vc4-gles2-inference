// The worker protocol: `engine_mm3 --worker`, one request at a time.
//
// Moved out of `main()` (2026-11-14, P2-1 of the 2026-11-13 audit).  It is the one
// part of main() whose per-turn state nothing else shares -- its own request loop,
// its own sampler, its own event stream -- which is why it could be lifted whole.
// The stdin reader came with it: a cancel has to be seen *while* a request is
// generating, so the buffer is the worker's, not the CLI's.  The protocol is
// documented in `server/README.md`; `scripts/swap_matrix.py` drives it.

#include "worker.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <poll.h>
#include <unistd.h>

#include "common.h"
#include "jinja/json.h"   // -isystem third_party
#include "sampling.h"
#include "tools.h"



// Requests and control lines arrive on stdin, and a *cancel* has to be seen
// while a request is generating -- which getline() cannot do without swallowing
// the next request.  One buffer, filled from the raw fd, serves both: blocking
// lines for the request loop, and a non-consuming peek for the token loop.
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

int run_worker(App& app, Session& session, bool enc_bos, bool chat_dump) {

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
    if (!app.opt.prefix_cache_path.empty() && !app.load_prefix_cache_any(app.opt.prefix_cache_path, cache_tokens, session))
        cerr << "PREFIX_CACHE ignored " << app.opt.prefix_cache_path << "\n";
    // The HTTP layer renders the prompt text itself, so it has to know which
    // template this checkpoint wants.  The container states it (kind + name)
    // and the resolution above already turned it into these three flags, so
    // report the effective one: the server must not guess from a file name.
    const char* effective_template = app.opt.tbb_template ? "tinybrainbot"
                                    : app.opt.minimind3_template ? "minimind3"
                                    : app.opt.qwen_template ? "chatml" : "plain";
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
            if (!app.opt.prefix_cache_path.empty()) app.load_prefix_cache_any(app.opt.prefix_cache_path, cache_tokens, session);
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
                                          req_thinking >= 0 ? req_thinking != 0 : app.opt.open_thinking,
                                          app.bos_str, app.eos_str);
            if (chat_dump)
                for (auto& p : req_parts)
                    fprintf(stderr, "PROMPT[%s] %3zu B  %s\n", p.is_input ? "in" : "tmpl",
                        p.text.size(), dump_escape(p.text).c_str());
        }
        auto tokens = req_parts.empty() ? app.tokenizer.encode(app.opt.prompt, false, false)
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
        int req_top_k = app.opt.top_k;
        if (f.size() > 11 && !f[11].empty()) req_top_k = atoi(f[11].c_str());
        if (req_top_k < 0) req_top_k = 0;
        bool greedy_req = wt <= 0.0f;
        // The candidate path reads back only the top-k logits, so it cannot
        // answer "what was the probability of this token" over the whole
        // vocabulary; a request that asks for it uses the full logits.
        bool cand_mode = app.opt.top_k_candidates && req_top_k > 0 && !greedy_req &&
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
