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

    void load_cli_cache();
    bool is_bos_token(int id);
    bool is_stop_token(int id);
    int cache_prefix_len(const std::vector<int>& tokens);
    void mask_markers(std::vector<float>& logits);
    void run_turn(const std::string& user, bool first, int pos,
                         std::vector<ChatMsg>* hist = nullptr);
    void run_tool_turn(const std::string& user, bool first,
                              std::vector<ChatMsg>* hist = nullptr);
};
