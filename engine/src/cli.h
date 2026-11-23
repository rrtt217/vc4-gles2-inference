#pragma once
// The command line: every option the engine takes, in one table.
//
// Moved out of `main()` (2026-11-14, P2-1 of the 2026-11-13 audit); the options
// became a table on 2026-11-17, for the same reason the environment switches did
// (`engine/src/switches.cpp`): the help text and the parser were two hand-kept
// copies of the same fact, nothing recorded what an option took, and a new flag
// cost three edits in two styles.  Now `kFlags` is the option surface, `--help`
// is rendered from it, and the parser walks it.
//
// `Options` is the state the flags fill.  `parse_args` returns false when the
// process should stop (--help, --version, a bad argument), with `exit_code`
// saying what to return.  `quiet` is separate because `g_quiet` is read all over
// the engine.

#include <cstdint>
#include <cstdio>
#include <string>

// One per flag.  The REPL reuses this vocabulary for `/temp` and friends, which
// is the point of naming them: a sampler knob is `CliOpt::Temp` wherever it is
// set, not a second list of names.
enum class CliOpt {
    Model, Chat, Prompt, System, Gen, Ppl, Worker,
    Greedy, Temp, Topp, TopK, TopKCandidates, Seed, MinP,
    RepeatPenalty, FrequencyPenalty, PresencePenalty, NoRepeatNgram,
    Thinking, Tools, ToolsFile,
    PrefixCache, BuildPrefixCache,
    DumpShaders, DumpSwitches, Tokenize,
    Quiet, Version, Help,
};

struct FlagDef {
    CliOpt id;
    const char* name;     // the long form, the one the help prints
    const char* alias;    // a short or legacy spelling, "" for none
    int nargs;            // 0 = a switch, n = n values, -1 = 0 or one `=VALUE`
    const char* argname;  // "PATH", "PROMPT OUT", ... ("" when nargs == 0)
    const char* section;  // the help's grouping
    const char* help;     // one line, no trailing period
};

struct Options {
    std::string model_path, tokenizer_path, prompt, system_prompt, tools_json;
    std::string build_cache_prompt, build_cache_out, prefix_cache_path, ppl_path;
    int gen = 8;
    int dump_mode = 0;   // --dump-shaders: 1 = digests, 2 = raw GLSL
    bool dump_switches = false;  // --dump-switches: the registry, then exit
    // --tokenize TEXT: print the token ids the container's tokenizer produces,
    // then exit.  The unigram path is a segmentation algorithm, so "the ids
    // look plausible" is not a check -- scripts/check_unigram.py compares this
    // output against a Viterbi oracle, and a flag is the only way to see it.
    bool tokenize = false;
    std::string tokenize_text;        // the text --tokenize was given
    bool prompt_set = false, gen_set = false;
    bool chat = false, qwen_template = false, minimind3_template = false, greedy = false, worker = false;
    bool tbb_template = false;   // <|user|>/<|assistant|>/<|end|>, no system turn, no BOS
    bool open_thinking = false;
    uint32_t seed = 1234;
    float temp = 1.0f, top_p = 0.9f;
    int top_k = 0;
    bool top_k_candidates = false;
};

int flag_count();
const FlagDef& flag_at(int i);
const FlagDef* flag_find(const char* name);   // long form or alias

// `--help`, rendered from the table: no second copy of the option surface.
void print_usage(FILE* out);

// false = stop the process (help/version/error); `exit_code` is what to return.
bool parse_args(int argc, char** argv, Options& opt, bool& quiet, int& exit_code);
