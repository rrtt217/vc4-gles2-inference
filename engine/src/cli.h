#pragma once
// The command line: every option the engine takes, and the one parser for them.
//
// Moved out of `main()` (2026-11-14, P2-1 of the 2026-11-13 audit).  `Options` is
// the whole CLI surface as data; `parse_args` fills it and returns false when the
// process should stop (--help, --version, a bad argument), with `exit_code` saying
// what to return.  `quiet` is separate because `g_quiet` is read all over the
// engine.

#include <cstdint>
#include <string>

struct Options {
    std::string model_path, tokenizer_path, prompt, system_prompt, tools_json;
    std::string build_cache_prompt, build_cache_out, prefix_cache_path, ppl_path;
    int gen = 8;
    int dump_mode = 0;   // --dump-shaders: 1 = digests, 2 = raw GLSL
    bool dump_switches = false;  // --dump-switches: the registry, then exit
    bool prompt_set = false, gen_set = false;
    bool chat = false, qwen_template = false, minimind3_template = false, greedy = false, worker = false;
    bool tbb_template = false;   // <|user|>/<|assistant|>/<|end|>, no system turn, no BOS
    bool open_thinking = false;
    uint32_t seed = 1234;
    float temp = 1.0f, top_p = 0.9f;
    int top_k = 0;
    bool top_k_candidates = false;
};

// false = stop the process (help/version/error); `exit_code` is what to return.
bool parse_args(int argc, char** argv, Options& opt, bool& quiet, int& exit_code);
