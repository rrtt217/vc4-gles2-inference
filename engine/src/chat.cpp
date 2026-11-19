// Chat: the chat path (see chat.h).

#include "chat.h"

#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "chat_format.h"
#include "common.h"
#include "lineedit.h"
#include "sampling.h"
#include "tools.h"

namespace {

// Strict numeric parsing for the slash commands: "1.0x" is a typo, not 1.0, and
// silently accepting it is how a knob looks set and is not.
bool parse_float(const string& s, float& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = strtod(s.c_str(), &end);
    if (errno != 0 || end == s.c_str() || *end != '\0') return false;
    out = (float)v;
    return true;
}

bool parse_int(const string& s, int& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long v = strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || v < INT_MIN || v > INT_MAX) return false;
    out = (int)v;
    return true;
}

// One formatter for a knob, so /params and the echo after a change agree, and
// 0.90 prints as 0.9 instead of a width the terminal has to truncate.
string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

string num(float v) {
    char b[32];
    snprintf(b, sizeof b, "%.2f", (double)v);
    string s(b);
    while (s.size() > 1 && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

}  // namespace

Chat::Chat(App& a, Session& s, bool eb, bool dump, bool q)
    : app(a), session(s), enc_bos(eb), chat_dump(dump), quiet(q), rng(app.opt.seed) {
    tools_array = chat_tools_array(app.opt.tools_json);
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
}


void Chat::load_cli_cache() {

    if (app.load_prefix_cache_any(app.opt.prefix_cache_path, cli_cache_tokens, session)) {
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
        cerr << "PREFIX_CACHE ignored " << app.opt.prefix_cache_path << "\n";}


bool Chat::is_bos_token(int id) {
    return app.model.id_bos >= 0 && id == app.model.id_bos;
}

bool Chat::is_stop_token(int id) {
    if (is_bos_token(id)) return true;
    if (app.model.have_specials)
        return id == app.model.id_eos || (app.model.id_stop >= 0 && id == app.model.id_stop);
    return id == 1 || id == 2;
}

int Chat::cache_prefix_len(const std::vector<int>& tokens) {
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
}

void Chat::mask_markers(std::vector<float>& logits) {
    for (size_t i = 0; i < logits.size() && i < marker_mask.size(); ++i)
        if (marker_mask[i]) logits[i] = -1e30f;
}

void Chat::run_turn(const std::string& user, bool first, int pos, std::vector<ChatMsg>* hist) {
    TurnTiming tt;
    tt.out = &last_turn;   // the REPL's status line, interrupted turns included
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
                                  app.opt.open_thinking, app.bos_str, app.eos_str);
        if (app.opt.open_thinking) {
            bool has = false;
            for (auto& p : parts) if (p.text.find("<think>") != string::npos) has = true;
            if (!has) parts.push_back(PromptPart{"<think>", false});
        }
    } else if (app.opt.minimind3_template) {
        string sys = app.system_prompt;
        if (!app.opt.tools_json.empty()) {
            sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                   app.opt.tools_json +
                   "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        if (first && !sys.empty())
            text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
        text += "<|im_start|>user\n" + user + "<|im_end|>\n"
                "<|im_start|>assistant\n";
        text += app.opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    } else if (app.opt.tbb_template) {
        // No system turn and no BOS: the markers carry the roles, and the
        // first user turn opens with a space (the model was trained so).
        // "<|user|> {q}<|assistant|> " with a trailing space: the model
        // was trained with a space after the assistant marker, and without
        // it the first token is <|end|> (measured on the q4 and i16 files,
        // where the other four wrappers I tried collapse to one token).
        text = string("<|user|> ") + user + "<|assistant|> ";
    } else if (app.opt.qwen_template) {
        if (first)
            text = "<|im_start|>system\n" + app.system_prompt +
                   "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        else
            text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        if (app.opt.open_thinking) text += "<think>\n";
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
    bool cand_mode = app.opt.top_k_candidates && app.opt.top_k > 0 && !app.opt.greedy && !sampling_penalties_active();
    // Greedy fast path: the argmax runs on the GPU and only the winning id
    // crosses the bus.  Everything that touches the logits vector on the CPU
    // (penalties, the marker mask, the LOGITS_OUT dump) disqualifies it; the
    // reduction itself is byte-identical to argmax_token (GPU_ARGMAX=0 A/Bs).
    const bool fast_greedy = app.opt.greedy && !sampling_penalties_active()
                          && marker_mask.empty() && !getenv("LOGITS_OUT")
                          && session.greedy_fast_ok();
    int next_fast = -1;
    for (size_t i = skip; i < tokens.size(); ++i) {
        if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
        if (pos >= app.model.seq) { tt.cut = "context"; cout << "[context full]\n"; return; }
        bool last = (i + 1 == tokens.size());
        if (last && fast_greedy) {
            next_fast = session.step_greedy(tokens[i], pos++);
        } else if (last && cand_mode) {
            cand = session.step_candidates(tokens[i], pos++, app.opt.top_k, app.opt.temp, app.opt.top_p);
            if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
        } else if (!last && i + 2 < tokens.size() && pos + 1 < app.model.seq && session.pair_ok()) {
            // Two prompt tokens per layer pass; the last token stays on the
            // single-position path because its logits are the ones we need.
            session.step_pair(tokens[i], pos, tokens[i + 1], pos + 1, false);
            pos += 2;
            ++i;
        } else if (!fast_greedy && !cand_mode && session.prefill_batch_ok()) {
            // RWKV-7 chunked prefill (P1, task-13): one layer pass per layer for
            // a whole batch of prompt tokens, with every projection drawn once
            // per weight for the batch.  The elementwise chain and the WKV
            // recurrence stay per token, in token order, so the numbers and the
            // state are the serial path's (the batch kernel is the same text
            // with the input row substituted).  The batch may include the last
            // prompt token -- then it is also the call that returns the logits.
            size_t n = (size_t)session.prefill_batch_size();
            n = std::min(n, tokens.size() - i);
            n = std::min(n, (size_t)(app.model.seq - pos));
            const bool ends = (i + n == tokens.size());
            vector<int> span(tokens.begin() + i, tokens.begin() + i + (long)n);
            vector<float> lg = session.prefill_batch(span, ends);
            if (ends && !lg.empty()) logits = std::move(lg);
            pos += (int)n;
            i += n - 1;
        } else {
            logits = session.step(tokens[i], pos++, last);
        }
    }
    if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
    tt.begin_decode();
    // DRAWCOUNT / SYNCCOUNT count the decode loop only: the prompt's draws are a
    // prompt-specific extra, and a per-token figure that includes them moves with
    // the prompt rather than with the kernel (docs/test-hooks.md).
    const long draw0 = app.gpu.draws(), sync0 = app.gpu.syncs();
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
    for (int i = 0; i < app.opt.gen && pos < app.model.seq; ++i) {
        if (g_stop) { tt.cut = "decode"; app.gpu.sync(); return; }
        int next;
        if (fast_greedy) {
            next = next_fast;   // produced by the previous step_greedy
        } else {
            if (!marker_mask.empty()) mask_markers(logits);
            apply_sampling_penalties(logits, recent);
            next = app.opt.greedy ? argmax_token(logits)
                          : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                       : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
        }
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
        if (fast_greedy) {
            next_fast = session.step_greedy(next, pos++);
        } else if (cand_mode) {
            cand = session.step_candidates(next, pos++, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
    if (!quiet && !g_stop && generated >= app.opt.gen && !is_stop_token(prev) && generated)
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
    last_turn.draws_decode = app.gpu.draws() - draw0;
    last_turn.syncs_decode = app.gpu.syncs() - sync0;
    if (g_stop) app.gpu.sync();
}

void Chat::run_tool_turn(const std::string& user, bool first, std::vector<ChatMsg>* hist) {
    string sys = app.system_prompt.empty()
        ? string("你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。")
        : app.system_prompt;
    if (!app.opt.tools_json.empty()) {
        sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
               app.opt.tools_json +
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
    if (first || !app.opt.tools_json.empty())
        text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
    text += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    text += app.opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

    for (int round = 0; round < 3; ++round) {
        TurnTiming tt;   // one line per model turn, interrupted or not
        tt.out = &last_turn;
        vector<PromptPart> parts;
        if (app.chat_tmpl)
            parts = app.chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/true,
                                      app.opt.open_thinking, app.bos_str, app.eos_str);
        // A template without an open_thinking switch (LFM2.5's) still gets
        // the block: the caller asked for it, and `<think>` is one token.
        if (app.chat_tmpl && app.opt.open_thinking) {
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
        bool cand_mode = app.opt.top_k_candidates && app.opt.top_k > 0 && !app.opt.greedy && !sampling_penalties_active();
        // Same greedy fast path as run_turn; this loop has no marker mask
        // and no LOGITS_OUT dump, so only the penalties can disqualify it.
        const bool fast_greedy = app.opt.greedy && !sampling_penalties_active()
                              && session.greedy_fast_ok();
        int next_fast = -1;
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
            if (last && fast_greedy) {
                next_fast = session.step_greedy(tokens[i], (int)i);
            } else if (last && cand_mode) {
                cand = session.step_candidates(tokens[i], (int)i, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
        for (int i = 0; i < app.opt.gen && pos < app.model.seq; ++i) {
            if (g_stop) { tt.cut = "decode"; app.gpu.sync(); return; }
            int next;
            if (fast_greedy) {
                next = next_fast;
            } else {
                apply_sampling_penalties(logits, recent);
                next = app.opt.greedy ? argmax_token(logits)
                              : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                           : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
            }
            if (is_stop_token(next)) {
                if (!quiet && is_bos_token(next))
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
            if (fast_greedy) {
                next_fast = session.step_greedy(next, pos++);
            } else if (cand_mode) {
                cand = session.step_candidates(next, pos++, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
            text += app.opt.open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        }
    }
}

// The sampler knobs a slash command can set, one row each: the name, the syntax
// hint for /help, the accepted range and which `CliOpt` field it writes.  The
// gate for the range lives here as data, so "/min-p 2" and "/topp 2" are refused
// the same way and neither can be forgotten.
struct ParamDef {
    const char* name;
    CliOpt opt;          // the same vocabulary the command line uses (cli.h)
    int kind;            // 0 = float in [lo, hi], 1 = int in [lo, hi], 2 = on|off
    float lo, hi;
    const char* hint;
};

static const ParamDef kParams[] = {
    {"/temp", CliOpt::Temp, 0, 0.0f, 5.0f, "F  [0 = greedy]"},
    {"/topp", CliOpt::Topp, 0, 0.0f, 1.0f, "F  (0, 1]"},
    {"/topk", CliOpt::TopK, 1, 0.0f, 1024.0f, "N  [0 = off]"},
    {"/gen", CliOpt::Gen, 1, 0.0f, 4096.0f, "N  tokens per reply"},
    {"/seed", CliOpt::Seed, 1, 0.0f, 4294967295.0f, "N"},
    {"/greedy", CliOpt::Greedy, 2, 0.0f, 0.0f, "on|off"},
    {"/penalty", CliOpt::RepeatPenalty, 0, 1.0f, 10.0f, "F  repetition [1 = off]"},
    {"/freq-penalty", CliOpt::FrequencyPenalty, 0, -2.0f, 2.0f, "F  [0 = off]"},
    {"/pres-penalty", CliOpt::PresencePenalty, 0, -2.0f, 2.0f, "F  [0 = off]"},
    {"/min-p", CliOpt::MinP, 0, 0.0f, 1.0f, "F  [0 = off]"},
    {"/ngram", CliOpt::NoRepeatNgram, 1, 0.0f, 16.0f, "N  block an n-gram [0 = off]"},
    {"/topk-candidates", CliOpt::TopKCandidates, 2, 0.0f, 0.0f, "on|off  approximate top-k"},
    {"/thinking", CliOpt::Thinking, 2, 0.0f, 0.0f, "on|off"},
};

static const ParamDef* find_param(const std::string& name) {
    for (const ParamDef& p : kParams)
        if (name == p.name) return &p;
    return nullptr;
}

// ---- the REPL's command surface, one table -------------------------------
// Same regime as the command line (cli.cpp): the table owns the names, the syntax
// and the help, `repl_command` walks it, and `/help` is rendered from it.  Before
// this, a command lived in three places -- the name list in apply_param, an `if`
// in the REPL loop, and a line of hand-written help -- and the three disagreed
// (the `/params` name was missing from the list once and typed at a model).
struct CmdDef {
    const char* name;
    const char* alias;     // "" for none
    const char* arg;       // syntax hint, "" for a bare command
    const char* help;
    bool (Chat::*fn)(const std::string& arg);
};

static const CmdDef kCmd[] = {
    {"/help", "", "", "this list", &Chat::cmd_help},
    {"/model", "", "PATH", "load another container, keeping the GPU context",
     &Chat::cmd_model},
    {"/clear", "", "", "forget the conversation so far", &Chat::cmd_clear},
    {"/stats", "", "", "the container, the context and the memory state",
     &Chat::cmd_stats},
    {"/params", "", "", "the sampling knobs (same as any knob with no value)",
     &Chat::cmd_params},
    {"/system", "", "TEXT", "replace the system prompt (forgets the conversation)",
     &Chat::cmd_system},
    {"/exit", "/quit", "", "leave", &Chat::cmd_exit},
};

// Levenshtein, for the "did you mean" hint only: two rows of the table, not a
// library.
static int edit_distance(const std::string& a, const std::string& b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = (int) j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = (int) i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            cur[j] = std::min(std::min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
        }
        prev.swap(cur);
    }
    return prev[b.size()];
}

static const CmdDef* find_cmd(const std::string& name) {
    for (const CmdDef& c : kCmd)
        if (name == c.name || (c.alias[0] && name == c.alias)) return &c;
    return nullptr;
}

bool Chat::cmd_help(const std::string&) {
    print_help();
    return true;
}

void Chat::print_help() const {
    cout << "commands\n";
    for (const CmdDef& c : kCmd) {
        string head = c.name;
        if (c.alias[0]) head += string(" (") + c.alias + ")";
        if (c.arg[0]) head += string(" ") + c.arg;
        cout << fmt("  %-22s %s\n", head.c_str(), c.help);
    }
    // One line per knob: the syntax *and* its range come from the table, so
    // /help cannot promise something apply_param refuses.
    cout << "sampling knobs (a bare name, or /params, prints the live state)\n";
    string row;
    for (size_t i = 0; i < sizeof(kParams) / sizeof(kParams[0]); ++i) {
        row += (row.empty() ? "  " : "   ") + string(kParams[i].name) + " " + kParams[i].hint;
        if (row.size() > 46 || i + 1 == sizeof(kParams) / sizeof(kParams[0])) {
            cout << row << "\n";
            row.clear();
        }
    }
    cout << "anything else is sent to the model; arrows edit, Ctrl-Left/Right\n"
            "(or Alt-b/f) move by word.  Enter sends, Ctrl-V Enter or Alt-Enter\n"
            "starts a new line, and pasted text keeps its line breaks, so a pasted\n"
            "block is one message.  Up/Down move inside such a block; above its\n"
            "first line they recall earlier inputs.\n";
}

bool Chat::cmd_clear(const std::string&) {
    cleared_ = true;   // repl() owns `first` and `history`; it resets them
    cout << "[conversation cleared]\n";
    return true;
}

bool Chat::cmd_stats(const std::string&) {
    print_stats();
    return true;
}

bool Chat::cmd_system(const std::string& text) {
    // The system prompt is part of the rendered prefix, so the KV cache no longer
    // matches: this is a /clear that also changes the prompt, and it says so.
    app.opt.system_prompt = text;
    cleared_ = true;
    cout << "[system prompt set; conversation cleared]\n";
    return true;
}

bool Chat::cmd_params(const std::string&) {
    print_params();
    return true;
}

bool Chat::cmd_exit(const std::string&) {
    quit_after_line = true;
    return true;
}

bool Chat::apply_param(const std::string& line) {
    // A multi-line input is a message, never a command: the first line of a pasted
    // block can easily start with "/temp".
    if (line.find('\n') != std::string::npos) return false;
    const size_t sp = line.find(' ');
    const string name = line.substr(0, sp == std::string::npos ? line.size() : sp);
    const string arg = sp == std::string::npos ? string() : line.substr(sp + 1);

    const ParamDef* p = find_param(name);
    if (!p && name != "/params") return false;
    // The bare form of any of these prints the state; "/params" exists as the one
    // name that says so, and it was the name left out of this list once already.
    if (!p || arg.empty()) { print_params(); return true; }

    auto bad = [&](const std::string& what) {
        cerr << what << "\n";
        print_params();
        return true;
    };
    char span[96];
    if (p->kind == 2)
        snprintf(span, sizeof(span), "on or off");
    else if (p->kind == 1)
        snprintf(span, sizeof(span), "an integer in %ld..%ld", (long) p->lo, (long) p->hi);
    else
        snprintf(span, sizeof(span), "a float in %g..%g", (double) p->lo, (double) p->hi);
    const string range = std::string(p->name) + " takes " + span;
    if (p->kind == 2) {                    // on|off
        if (arg != "on" && arg != "off") return bad(range);
        const bool on = arg == "on";
        switch (p->opt) {
            case CliOpt::Greedy: app.opt.greedy = on; break;
            case CliOpt::Thinking: app.opt.open_thinking = on; break;
            case CliOpt::TopKCandidates: app.opt.top_k_candidates = on; break;
            default: return bad(range);
        }
        print_params();
        return true;
    }
    if (p->kind == 1) {
        int v = 0;
        if (!parse_int(arg, v) || (float) v < p->lo || (float) v > p->hi) return bad(range);
        switch (p->opt) {
            case CliOpt::TopK:
                app.opt.top_k = v;
                if (v == 0) app.opt.top_k_candidates = false;   // nothing to keep
                break;
            case CliOpt::Gen: app.opt.gen = v; break;
            case CliOpt::Seed:
                app.opt.seed = (uint32_t) v;
                rng.seed(app.opt.seed);   // the next turn draws from the new stream
                break;
            case CliOpt::NoRepeatNgram: g_sampling.no_repeat_ngram = v; break;
            default: return bad(range);
        }
        print_params();
        return true;
    }
    float v = 0;
    if (!parse_float(arg, v) || v < p->lo || v > p->hi) return bad(range);
    switch (p->opt) {
        case CliOpt::Temp:
            // Temperature 0 *is* greedy everywhere in this engine (the sampler
            // defines it that way, see sampling.cpp), and saying it here also picks
            // the fast greedy path.  A non-zero temperature turns sampling back on.
            app.opt.temp = v;
            app.opt.greedy = (v == 0.0f);
            break;
        case CliOpt::Topp: app.opt.top_p = v; break;
        case CliOpt::MinP: g_sampling.min_p = v; break;
        case CliOpt::RepeatPenalty: g_sampling.repetition_penalty = v; break;
        case CliOpt::FrequencyPenalty: g_sampling.frequency_penalty = v; break;
        case CliOpt::PresencePenalty: g_sampling.presence_penalty = v; break;
        default: return bad(range);
    }
    print_params();
    return true;
}

void Chat::print_params() const {
    const Options& o = app.opt;
    cout << "sampling: " << (o.greedy ? "greedy" : "sampled")
         << "  temp " << num(o.temp) << "  top_p " << num(o.top_p)
         << "  top_k " << o.top_k
         << (o.top_k_candidates ? " (candidates only)" : "") << "  seed " << o.seed
         << "  gen " << o.gen
         << "  thinking " << (o.open_thinking ? "on" : "off") << "\n";
    cout << "penalties: repeat " << num(g_sampling.repetition_penalty)
         << "  frequency " << num(g_sampling.frequency_penalty)
         << "  presence " << num(g_sampling.presence_penalty)
         << "  min_p " << num(g_sampling.min_p)
         << "  no_repeat_ngram " << g_sampling.no_repeat_ngram << "\n";
    cout << flush;
}

bool Chat::repl_command(const std::string& line) {
    if (line.empty() || line[0] != '/') return false;
    const size_t sp = line.find(' ');
    const string name = line.substr(0, sp == std::string::npos ? line.size() : sp);
    const string arg = sp == std::string::npos ? string() : line.substr(sp + 1);
    // A command is one line: a pasted block that happens to start with "/model "
    // is a message, not a path with newlines in it.
    if (line.find('\n') != std::string::npos) return false;
    if (const CmdDef* c = find_cmd(name)) {
        if (c->arg[0] && arg.empty() && c->fn != &Chat::cmd_exit) {
            cerr << name << " takes " << c->arg << "\n";
            return true;
        }
        return (this->*c->fn)(arg);
    }
    if (apply_param(line)) return true;   // the sampler knobs, their own table
    // A one-word slash command that is two edits away from a real one is a typo,
    // and sending it to the model (what "anything else is a message" did) wastes a
    // turn and reads as if the model answered the user's mistyped command.
    if (name.size() > 1) {
        for (const CmdDef& c : kCmd)
            if (edit_distance(name, c.name) <= 2) {
                cerr << "unknown command " << name << " (did you mean " << c.name
                     << "?  /help lists them)\n";
                return true;
            }
        for (const ParamDef& p : kParams)
            if (edit_distance(name, p.name) <= 2) {
                cerr << "unknown command " << name << " (did you mean " << p.name
                     << "?  /help lists them)\n";
                return true;
            }
    }
    return false;
}

bool Chat::cmd_model(const std::string& path) {
    const auto t_swap = Clock::now();
    // There is deliberately no capacity pre-check here.  The free CMA the kernel
    // reports is not the budget a new scene actually gets -- it is often less than
    // what the driver can find -- and a desktop kernel has no CMA at all, so the
    // old `st_size + 48 MB` heuristic refused swaps that would have worked and
    // could not answer on a laptop.  A container that does not fit now fails in the
    // upload, which is where the truth is; there is nothing to guess.
    //
    // What *is* decidable before releasing anything is whether the container is
    // usable at all, because the capability bits live in the metadata.  The old
    // code released the scene first and then refused (stories15M advertises
    // capabilities 0, and its refusal *left* the process with no model); the probe
    // reads the header and the metadata and never the tensors, so a refusal costs
    // nothing and keeps the session.
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        cerr << "/model: cannot read " << path << "\n";
        return true;
    }
    Model probe;
    if (!probe.probe(path)) {
        cerr << "/model: not a container this build can read: " << path
             << "  (engine/tools/mmcf_dump.py explains a file)\n";
        return true;
    }
    if (app.opt.chat && probe.have_capabilities && !(probe.capabilities & 1)) {
        cerr << "/model: this checkpoint does not advertise chat support "
                "(capabilities 0x" << std::hex << probe.capabilities << std::dec
             << "); keeping the current model\n";
        return true;
    }
    app.gpu.release_scene();
    app.model.reset();
    if (!app.model.load(path)) { cerr << "no model loaded; leaving\n"; quit_after_line = true; return true; }
    app.opt.model_path = path;    // /stats reports the live container
    app.prepare_model();          // FFN split, hidden scale, seq clamp
    if (!app.setup_from_model()) { quit_after_line = true; return true; }
    app.upload_weights();
    session.reset();
    cleared_ = true;
    cout << "[model] " << (app.model.template_name.empty() ? string("(kind 0)") : app.model.template_name)
         << "  chat_template " << (app.chat_tmpl ? "yes" : "no")
         << "  " << (long) elapsed_ms(t_swap, Clock::now()) << " ms\n" << flush;
    return true;
}

void Chat::print_stats() const {
    const Model& m = app.model;
    cout << fmt("model    %s  %.1f MB\n", app.opt.model_path.c_str(),
                (double)m.source_size / (1024.0 * 1024.0));
    cout << fmt("shape    dim %d  hidden %d  layers %d  heads %d/%d  vocab %d  seq %d\n",
                m.dim, m.hidden, m.layers, m.heads, m.kv_heads, m.vocab, m.seq);
    string ffn = fmt("%d wide", m.ffn_width());
    if (m.is_moe()) ffn += fmt("  (MoE, %d experts, top-%d)", m.experts, m.top_k);
    if (m.hybrid) ffn += fmt("  (hybrid, %d conv layers)", m.n_conv_layers());
    cout << "ffn      " << ffn << "\n";
    cout << fmt("template %s (kind %d)  chat template %zu B  bos %d  eos %d  stop %d\n",
                m.template_name.empty() ? "(unnamed)" : m.template_name.c_str(),
                m.template_kind, m.chat_template.size(), m.id_bos, m.id_eos, m.id_stop);
    cout << fmt("rms eps  %g (%s)\n", (double)m.rms_eps,
                m.rms_eps_stated ? "from the container" : "compiled literal");
    // MemAvailable, not CmaFree: the kernel's free-CMA figure is not the budget a
    // GL scene gets (it is often smaller), and a desktop kernel has none at all, so
    // it answered a question nobody could act on.  The scene's own ledger is
    // GPU_MEM's job.
    const long long avail_kb = proc_meminfo_kb("MemAvailable");
    const long long total_kb = proc_meminfo_kb("MemTotal");
    if (avail_kb >= 0 && total_kb > 0)
        cout << fmt("memory   %lld of %lld MB available\n", avail_kb >> 10, total_kb >> 10);
    long milli = 0;
    {
        ifstream t("/sys/class/thermal/thermal_zone0/temp");
        t >> milli;
    }
    if (milli > 0) cout << fmt("soc      %.1f C\n", milli / 1000.0);
    const int used = (int)kv_tokens.size();
    cout << fmt("context  %d / %d tokens (%d%%)\n", used, m.seq,
                m.seq > 0 ? (int)(100.0 * used / m.seq) : 0);
    if (!cli_cache_tokens.empty())
        cout << fmt("cache    %zu tokens loaded from --prefix-cache\n", cli_cache_tokens.size());
    cout << flush;
}

// DRAWCOUNT / SYNCCOUNT: what a decoded token costs in launches instead of in
// milliseconds, which is the quantity the RWKV fusion work moves -- that path is
// launch/sync-bound, not MAC-bound (docs/rwkv7.md).  Decode only:
// the prompt's launches are a prompt-specific extra, so the turn records the
// delta around its decode loop (TurnReport::draws_decode).
static void print_launch_counts(const TurnReport& turn) {
    if (turn.generated <= 0) return;
    if (getenv("DRAWCOUNT"))
        cout << "draws " << turn.draws_decode << " decode, "
             << (double)turn.draws_decode / turn.generated << " per token\n";
    if (getenv("SYNCCOUNT"))
        cout << "syncs " << turn.syncs_decode << " decode, "
             << (double)turn.syncs_decode / turn.generated << " per token\n";
}

void Chat::print_status() const {
    // Ambient, not an answer: stderr, and only for a human at a terminal, so a
    // redirected stdout is exactly the conversation.
    if (quiet || !isatty(STDERR_FILENO)) return;
    fprintf(stderr, "[%d tok", last_turn.generated);
    if (last_turn.decode_ms_per_token > 0)
        fprintf(stderr, "  %.1f tok/s", 1000.0 / last_turn.decode_ms_per_token);
    if (last_turn.prefill_s > 0) fprintf(stderr, "  %.2f s prefill", last_turn.prefill_s);
    fprintf(stderr, "  ctx %zu/%d", kv_tokens.size(), app.model.seq);
    if (last_turn.cut) fprintf(stderr, "  cut %s", last_turn.cut);
    fprintf(stderr, "]\n");
}

int Chat::repl() {

    if (!app.opt.prompt_set) app.opt.prompt.clear();
    if (!app.opt.prompt.empty()) {
        if (app.opt.tools_json.empty()) run_turn(app.opt.prompt, true, 0);
        else run_tool_turn(app.opt.prompt, true);
        print_launch_counts(last_turn);
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
            if (repl_command(line)) {
                if (cleared_) {
                    first = true;
                    history.clear();
                    kv_tokens.clear();
                    cleared_ = false;
                }
                if (quit_after_line) break;
                continue;
            }
            if (app.opt.tools_json.empty()) run_turn(line, first, 0, app.chat_tmpl ? &history : nullptr);
            else run_tool_turn(line, first, app.chat_tmpl ? &history : nullptr);
            first = false;
            print_status();
            print_launch_counts(last_turn);
            if (g_stop) {
                cout << "\n[interrupted]\n";
                g_stop = 0;
            }
        }
        if (g_stop) cout << "\n";
    }
    return 0;}

int Chat::complete() {
auto tokens = app.tokenizer.encode(app.opt.prompt, true, true);
// The single-shot path never reuses a loaded prefix, so a hybrid model must
// not keep the conv history that came with the cache file.
if (!cli_cache_tokens.empty()) session.zero_conv_state();
cout << "tokens:";
for (int t : tokens) cout << " " << t;
cout << "\n";
vector<float> logits;
CandidateResult cand;
bool cand_mode = app.opt.top_k_candidates && app.opt.top_k > 0 && !app.opt.greedy && !sampling_penalties_active();
// Greedy fast path, same rules as run_turn.  The per-token decode timer
// below wraps step_greedy, so the argmax reduction is inside the measured
// per-token cost, exactly as the read-back it replaces was.
const bool fast_greedy = app.opt.greedy && !sampling_penalties_active()
                      && marker_mask.empty() && !getenv("LOGITS_OUT")
                      && session.greedy_fast_ok();
int next_fast = -1;
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
        if (last && fast_greedy) {
            next_fast = session.step_greedy(tokens[i], (int)i);
        } else if (last && cand_mode) {
            cand = session.step_candidates(tokens[i], (int)i, app.opt.top_k, app.opt.temp, app.opt.top_p);
            if (!cand.complete) { lg = std::move(cand.full); cand_mode = false; }
        } else if (!last && i + 2 < tokens.size() && (int)i + 1 < app.model.seq && session.pair_ok()) {
            session.step_pair(tokens[i], (int)i, tokens[i + 1], (int)i + 1, false);
            ++i;
        } else if (!fast_greedy && !cand_mode && session.prefill_batch_ok()) {
            // RWKV-7 chunked prefill; same rule as the chat loop above.
            size_t n = (size_t)session.prefill_batch_size();
            n = std::min(n, tokens.size() - i);
            n = std::min(n, (size_t)(app.model.seq - (int)i));
            const bool ends = (i + n == tokens.size());
            vector<int> span(tokens.begin() + i, tokens.begin() + i + (long)n);
            vector<float> lg2 = session.prefill_batch(span, ends);
            if (ends && !lg2.empty()) lg = std::move(lg2);
            i += n - 1;
        } else {
            lg = session.step(tokens[i], (int)i, last);
        }
    }
    return lg;
};
TurnTiming tt;
tt.out = &last_turn;
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
const long cdraw0 = app.gpu.draws(), csync0 = app.gpu.syncs();
int generated = 0;
double decode_ms = 0;
vector<int> recent;
bool ended_on_eos = false;
const char* stop_reason = nullptr;
int stop_id = -1;
for (int i = 0; i < app.opt.gen; ++i) {
    if (g_stop) break;
    // The chat and tool loops stop at the container's seq; this one did
    // not, so a --gen larger than seq walked past the KV cache and the rope
    // table and printed whatever came back.  Measured on void.1 (seq 1024,
    // --gen 4096): the text turns to "walk inkinkinkinkin" at position
    // 1023 and stays garbage -- and the same file packed at seq 2048 turns
    // to " t t t t" at 2047 instead, which is how the boundary was found.
    if ((int)tokens.size() >= app.model.seq) {
        tt.cut = "context";
        if (!quiet)
            fprintf(stderr, "\n[context full] %d tokens, seq %d; raise --seq when "
                            "packing or lower --gen\n", generated, app.model.seq);
        break;
    }
    int next;
    if (fast_greedy) {
        next = next_fast;
    } else {
        if (!marker_mask.empty()) mask_markers(logits);
        apply_sampling_penalties(logits, recent);
        next = app.opt.greedy ? argmax_token(logits)
                      : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                   : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
    }
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
    if (fast_greedy) {
        next_fast = session.step_greedy(next, (int)tokens.size() - 1);
    } else if (cand_mode) {
        cand = session.step_candidates(next, (int)tokens.size() - 1, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
last_turn.draws_decode = app.gpu.draws() - cdraw0;
last_turn.syncs_decode = app.gpu.syncs() - csync0;
if (g_stop) { tt.cut = "decode"; app.gpu.sync(); cout << "[interrupted]\n"; return 0; }
// A completion model that never emits its end token looks like "it will not
// stop"; say which limit ended it so that is diagnosable from the output.
if (!quiet && ended_on_eos && stop_reason && is_bos_token(stop_id))
    fprintf(stderr, "\n[stopped after %d tokens: %s (%d)]\n", generated, stop_reason, stop_id);
if (!quiet && !ended_on_eos && generated >= app.opt.gen)
    fprintf(stderr, "\n[stopped at the --gen limit: %d tokens.  This model did not "
                    "emit its end token (eos %d, stop %d); raise --gen or press "
                    "Ctrl-C]\n", generated, app.model.id_eos, app.model.id_stop);
if (generated) cout << "decode avg " << decode_ms / generated << " ms/token\n";
print_launch_counts(last_turn);
return 0;
}
