#pragma once
// The chat path: a turn, its tool turns, the marker mask and the CLI prefix cache.
//
// These were `[&]` lambdas and a block of locals inside `main()` (2026-11-14, P2-1
// of the 2026-11-13 audit).  They share the engine state (`App`), the live
// `Session` and their own token bookkeeping, so they live in one object that
// main() builds once; `main` keeps only the dispatch.

#include <random>
#include <string>
#include <vector>

#include "app.h"
#include "session.h"

struct Chat {
    Chat(App& a, Session& s, bool enc_bos, bool dump, bool quiet);

    App& app;
    Session& session;
    bool enc_bos = false;
    bool chat_dump = false;
    bool quiet = false;
    std::mt19937 rng;   // seeded from --seed, one stream per run
    std::vector<char> marker_mask;   // BLOCK_MARKERS: ids to suppress
    std::vector<int> kv_tokens;      // tokens already in the session's KV
    std::vector<int> cli_cache_tokens;   // what --prefix-cache loaded
    std::string tools_array;         // the normalised tool list the CLI passed

    // Slash commands that change a sampling knob or report state.  A knob takes
    // effect on the next turn: the REPL reads stdin only between turns, so nothing
    // typed while the model is decoding can reach the sampler.
    bool apply_param(const std::string& line);   // true = the line was a command
    void print_params() const;
    void print_stats() const;
    // One line after each turn, on stderr and only at a terminal, so a redirected
    // stdout stays the conversation.
    void print_status() const;
    // Filled by TurnTiming, which is the only code that sees interrupted turns.
    TurnReport last_turn;

    void load_cli_cache();
    bool is_bos_token(int id);
    bool is_stop_token(int id);
    int cache_prefix_len(const std::vector<int>& tokens);
    void mask_markers(std::vector<float>& logits);
    void run_turn(const std::string& user, bool first, int pos,
                         std::vector<ChatMsg>* hist = nullptr);
    int repl();      // --chat without --prompt: the REPL, /model included
    int complete();  // --prompt without --chat: one prefill, one decode loop
    void run_tool_turn(const std::string& user, bool first,
                              std::vector<ChatMsg>* hist = nullptr);
};
