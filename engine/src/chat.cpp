// Chat: the chat path (see chat.h).

#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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
    for (size_t i = skip; i < tokens.size(); ++i) {
        if (g_stop) { tt.cut = "prefill"; app.gpu.sync(); return; }
        if (pos >= app.model.seq) { tt.cut = "context"; cout << "[context full]\n"; return; }
        bool last = (i + 1 == tokens.size());
        if (last && cand_mode) {
            cand = session.step_candidates(tokens[i], pos++, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
    for (int i = 0; i < app.opt.gen && pos < app.model.seq; ++i) {
        if (g_stop) { tt.cut = "decode"; app.gpu.sync(); return; }
        if (!marker_mask.empty()) mask_markers(logits);
        apply_sampling_penalties(logits, recent);
        int next = app.opt.greedy ? argmax_token(logits)
                          : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                       : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
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
            apply_sampling_penalties(logits, recent);
            int next = app.opt.greedy ? argmax_token(logits)
                              : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                           : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
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
            if (cand_mode) {
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

int Chat::repl() {

    if (!app.opt.prompt_set) app.opt.prompt.clear();
    if (!app.opt.prompt.empty()) {
        if (app.opt.tools_json.empty()) run_turn(app.opt.prompt, true, 0);
        else run_tool_turn(app.opt.prompt, true);
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
                if (app.model.have_capabilities && app.opt.chat && !(app.model.capabilities & 1)) {
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
            if (app.opt.tools_json.empty()) run_turn(line, first, 0, app.chat_tmpl ? &history : nullptr);
            else run_tool_turn(line, first, app.chat_tmpl ? &history : nullptr);
            first = false;
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
            cand = session.step_candidates(tokens[i], (int)i, app.opt.top_k, app.opt.temp, app.opt.top_p);
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
    if (!marker_mask.empty()) mask_markers(logits);
    apply_sampling_penalties(logits, recent);
    int next = app.opt.greedy ? argmax_token(logits)
                      : (cand_mode ? sample_candidate_result(cand, app.opt.temp, app.opt.top_p, app.opt.top_k, rng)
                                   : sample_token(logits, app.opt.temp, app.opt.top_p, app.opt.top_k, rng));
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
if (getenv("DRAWCOUNT") && generated) {
    cout << "draws " << app.gpu.draws() << " total, "
         << (double)app.gpu.draws() / generated << " per token\n";
}
return 0;
}
