// The command line parser.  The option surface is `kFlags` (below), the help is
// rendered from it, and the parser walks it.  See cli.h for the contract; the
// option semantics are unchanged from when this lived in `main()` -- including
// the environment-only switches (REPETITION_PENALTY and friends) that predate
// the flags.

#include "cli.h"

#include "common.h"
#include "sampling.h"   // g_sampling: the penalty switches predate the flags

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static const char* kVersion = "1.0";

// One row per option: `nargs` says what it takes, and the parser and `--help`
// both read that, so a flag cannot be documented and unparsed (or the reverse).
// Section order in the help is the order the sections first appear here.
static const FlagDef kFlags[] = {
    {CliOpt::Model, "--model", "", 1, "PATH", "Input and output",
     "container to run (or set MODEL)"},
    {CliOpt::Chat, "--chat", "", 0, "", "Input and output",
     "chat mode; with no --prompt it is a REPL on stdin"},
    {CliOpt::Prompt, "--prompt", "", 1, "TEXT", "Input and output",
     "one-shot prompt ('-' reads stdin to EOF)"},
    {CliOpt::System, "--system", "", 1, "TEXT", "Input and output",
     "system prompt (default: the container's)"},
    {CliOpt::Gen, "--gen", "", 1, "N", "Input and output",
     "tokens to generate (default 8, 128 with --chat)"},
    {CliOpt::Ppl, "--ppl", "", 1, "FILE", "Input and output",
     "perplexity of a text file"},

    {CliOpt::Greedy, "--greedy", "", 0, "", "Sampling",
     "argmax instead of sampling"},
    {CliOpt::Temp, "--temp", "", 1, "F", "Sampling",
     "temperature (default 1.0; 0 = greedy)"},
    {CliOpt::Topp, "--topp", "", 1, "F", "Sampling",
     "nucleus cutoff (default 0.9)"},
    {CliOpt::TopK, "--top-k", "", 1, "N", "Sampling",
     "keep only the N most likely tokens (0 = off)"},
    {CliOpt::TopKCandidates, "--top-k-candidates", "", 0, "", "Sampling",
     "with --top-k, read back only the candidates (approximate; falls back "
     "when it cannot decide)"},
    {CliOpt::Seed, "--seed", "", 1, "N", "Sampling",
     "sampling seed (default 1234)"},
    {CliOpt::MinP, "--min-p", "", 1, "F", "Sampling",
     "min-p cut (the sampler's default is in sampling.cpp; MIN_P also works)"},
    {CliOpt::RepeatPenalty, "--repeat-penalty", "", 1, "F", "Sampling",
     "repetition penalty (REPETITION_PENALTY also works)"},
    {CliOpt::FrequencyPenalty, "--frequency-penalty", "", 1, "F", "Sampling",
     "frequency penalty (FREQUENCY_PENALTY also works)"},
    {CliOpt::PresencePenalty, "--presence-penalty", "", 1, "F", "Sampling",
     "presence penalty (PRESENCE_PENALTY also works)"},
    {CliOpt::NoRepeatNgram, "--no-repeat-ngram", "", 1, "N", "Sampling",
     "block an n-gram from repeating (NO_REPEAT_NGRAM also works)"},

    {CliOpt::Thinking, "--thinking", "--open-thinking", 0, "", "Chat and tools",
     "open a thinking block before the answer"},
    {CliOpt::Tools, "--tools", "", 1, "JSON", "Chat and tools",
     "tool definitions for the model"},
    {CliOpt::ToolsFile, "--tools-file", "", 1, "F", "Chat and tools",
     "read the tool definitions from a file"},
    {CliOpt::Worker, "--worker", "", 0, "", "Chat and tools",
     "JSON-lines worker (the HTTP server drives this)"},

    {CliOpt::PrefixCache, "--prefix-cache", "", 1, "F", "Prefix cache",
     "load a prefix cache (needs GPU attention)"},
    {CliOpt::BuildPrefixCache, "--build-prefix-cache", "", 2, "PROMPT OUT", "Prefix cache",
     "prefill a fixed prompt, write the cache and exit"},

    {CliOpt::DumpShaders, "--dump-shaders", "", -1, "[=raw]", "Diagnostics",
     "print every shader builder's GLSL digest (or the text with =raw)"},
    {CliOpt::DumpSwitches, "--dump-switches", "", 0, "", "Diagnostics",
     "print the environment-switch registry (no GPU, no model)"},

    {CliOpt::Quiet, "--quiet", "-q", 0, "", "Process",
     "no informational lines on stderr"},
    {CliOpt::Version, "--version", "-V", 0, "", "Process",
     "print the version"},
    {CliOpt::Help, "--help", "-h", 0, "", "Process",
     "this text ('help' without the dashes also works)"},
};

static const int kFlagCount = (int) (sizeof(kFlags) / sizeof(kFlags[0]));

int flag_count() { return kFlagCount; }

const FlagDef& flag_at(int i) {
    if (i < 0 || i >= kFlagCount) i = 0;
    return kFlags[i];
}

const FlagDef* flag_find(const char* name) {
    if (!name) return nullptr;
    for (int i = 0; i < kFlagCount; ++i) {
        if (!strcmp(kFlags[i].name, name)) return &kFlags[i];
        if (kFlags[i].alias[0] && !strcmp(kFlags[i].alias, name)) return &kFlags[i];
    }
    return nullptr;
}

static void wrap(FILE* out, const char* text, int width, int indent) {
    // The help is one line per flag in the table; the terminal is not.  Break on
    // spaces at `width`, indenting the continuation to the description's column.
    std::string line;
    int pos = 0;
    const char* p = text;
    while (*p) {
        const char* q = strchr(p, ' ');
        const size_t len = q ? (size_t) (q - p) : strlen(p);
        if (pos > 0 && indent + pos + 1 + (int) len > width) {
            fprintf(out, "%s\n%*s", line.c_str(), indent, "");
            pos = 0;
            line.clear();
        }
        if (pos) {
            line += ' ';
            ++pos;
        }
        line.append(p, len);
        pos += (int) len;
        p = q ? q + 1 : p + strlen(p);
    }
    fprintf(out, "%s\n", line.c_str());
}

void print_usage(FILE* out) {
    fprintf(out, "usage: engine_mm3 [options]            (MODEL=<file.mmcf> or --model)\n");
    const char* section = "";
    for (int i = 0; i < kFlagCount; ++i) {
        const FlagDef& f = kFlags[i];
        if (strcmp(f.section, section)) {
            fprintf(out, "\n%s\n", f.section);
            section = f.section;
        }
        // The spelling column: `-q, --quiet` when the alias is the short form,
        // `--thinking (--open-thinking)` when it is a second long spelling.
        std::string head;
        const bool alias_is_short = f.alias[0] == '-' && strlen(f.alias) == 2;
        if (alias_is_short) head = std::string(f.alias) + ", ";
        head += f.name;
        if (f.nargs > 0) head += std::string(" ") + f.argname;
        else if (f.nargs < 0) head += f.argname;
        if (f.alias[0] && !alias_is_short) head += std::string(" (") + f.alias + ")";
        const int col = (int) head.size() + 2 > 30 ? (int) head.size() + 3 : 32;
        fprintf(out, "  %s", head.c_str());
        for (int k = (int) head.size() + 2; k < col; ++k) fputc(' ', out);
        wrap(out, f.help, 78, col);
    }
    fprintf(out,
            "\nIn the REPL: /help, /model PATH, /system TEXT, /temp F, /top-k N,\n"
            "/greedy, /clear, /exit.  Sampling penalties and the rest of the\n"
            "environment switches are in docs/test-hooks.md.  The container carries\n"
            "the tokenizer, the template, rope and the special-token ids, so there\n"
            "are no flags for those.\n");
}

// What one option does with its values.  Kept as a switch on the table's id
// rather than a function per flag: the table owns the *surface* (name, arity,
// help) and this owns the *effect*, and the compiler makes the two agree because
// the switch has no default that could silently drop a new row.
static bool apply(CliOpt id, const char* const* v, int n, Options& opt, bool& quiet,
                  int& exit_code) {
    switch (id) {
        case CliOpt::Model: opt.model_path = v[0]; break;
        case CliOpt::Chat: opt.chat = true; break;
        case CliOpt::Prompt:
            opt.prompt = v[0]; opt.prompt_set = true;
            if (opt.prompt == "-") {                       // everything on stdin
                stringstream ss; ss << cin.rdbuf(); opt.prompt = ss.str();
                while (!opt.prompt.empty() &&
                       (opt.prompt.back() == '\n' || opt.prompt.back() == '\r'))
                    opt.prompt.pop_back();
            }
            break;
        case CliOpt::System: opt.system_prompt = v[0]; break;
        case CliOpt::Gen: opt.gen = atoi(v[0]); opt.gen_set = true; break;
        case CliOpt::Ppl: opt.ppl_path = v[0]; break;
        case CliOpt::Worker: opt.worker = true; break;
        case CliOpt::Greedy: opt.greedy = true; break;
        case CliOpt::Temp: opt.temp = atof(v[0]); break;
        case CliOpt::Topp: opt.top_p = atof(v[0]); break;
        case CliOpt::TopK: opt.top_k = atoi(v[0]); break;
        case CliOpt::TopKCandidates: opt.top_k_candidates = true; break;
        case CliOpt::Seed: opt.seed = (uint32_t) strtoul(v[0], nullptr, 10); break;
        case CliOpt::MinP: g_sampling.min_p = atof(v[0]); break;
        case CliOpt::RepeatPenalty: g_sampling.repetition_penalty = atof(v[0]); break;
        case CliOpt::FrequencyPenalty: g_sampling.frequency_penalty = atof(v[0]); break;
        case CliOpt::PresencePenalty: g_sampling.presence_penalty = atof(v[0]); break;
        case CliOpt::NoRepeatNgram: g_sampling.no_repeat_ngram = atoi(v[0]); break;
        case CliOpt::Thinking: opt.open_thinking = true; break;
        case CliOpt::Tools: opt.tools_json = v[0]; break;
        case CliOpt::ToolsFile: {
            ifstream tf(v[0]);
            if (!tf) { cerr << "cannot open tools file\n"; exit_code = 1; return false; }
            stringstream ss; ss << tf.rdbuf(); opt.tools_json = ss.str();
            break;
        }
        case CliOpt::PrefixCache: opt.prefix_cache_path = v[0]; break;
        case CliOpt::BuildPrefixCache:
            opt.build_cache_prompt = v[0]; opt.build_cache_out = v[1];
            break;
        case CliOpt::DumpShaders:
            opt.dump_mode = (n > 0 && !strcmp(v[0], "raw")) ? 2 : 1;
            if (n > 0 && strcmp(v[0], "raw")) {
                cerr << "--dump-shaders takes 'raw' or nothing\n";
                exit_code = 2; return false;
            }
            break;
        case CliOpt::DumpSwitches: opt.dump_switches = true; break;
        case CliOpt::Quiet: quiet = true; break;
        case CliOpt::Version:
            printf("engine_mm3 %s (MMCF v2)\n", kVersion);
            exit_code = 0; return false;
        case CliOpt::Help:
            print_usage(stdout);
            exit_code = 0; return false;
    }
    return true;
}

bool parse_args(int argc, char** argv, Options& opt, bool& quiet, int& exit_code) {

    // The environment owns these defaults: MODEL/TOKENIZER select the container and
    // the tokenizer blob (the tokenizer one is a legacy path, and an empty MODEL is
    // better than a magic number from a file that happens to sit there), and the
    // five penalty switches predate the flags.
    opt.model_path = getenv("MODEL") ? getenv("MODEL") : "";
    opt.tokenizer_path = getenv("TOKENIZER") ? getenv("TOKENIZER")
                                            : "../models/llama2c/tokenizer.bin";
    opt.prompt = "One day, Lily";
    if (getenv("REPETITION_PENALTY")) g_sampling.repetition_penalty = atof(getenv("REPETITION_PENALTY"));
    if (getenv("FREQUENCY_PENALTY")) g_sampling.frequency_penalty = atof(getenv("FREQUENCY_PENALTY"));
    if (getenv("PRESENCE_PENALTY")) g_sampling.presence_penalty = atof(getenv("PRESENCE_PENALTY"));
    if (getenv("NO_REPEAT_NGRAM")) g_sampling.no_repeat_ngram = atoi(getenv("NO_REPEAT_NGRAM"));
    if (getenv("MIN_P")) g_sampling.min_p = atof(getenv("MIN_P"));

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        // `help` without the dashes is the one bare word the parser accepts.
        const FlagDef* f = strcmp(arg, "help") ? flag_find(arg) : flag_find("--help");
        const char* attached = nullptr;
        if (!f && !strncmp(arg, "--", 2)) {
            // `--flag=value`: only a flag whose arity allows it.
            const char* eq = strchr(arg, '=');
            if (eq) {
                std::string name(arg, (size_t) (eq - arg));
                const FlagDef* g = flag_find(name.c_str());
                if (g && (g->nargs < 0 || g->nargs == 1)) {
                    f = g;
                    attached = eq + 1;
                }
            }
        }
        if (!f) {
            // Silently ignoring an argument is how "CHAT_DUMP=1" ends up on the
            // command line (where it is not an environment variable) and the
            // switch appears not to work at all.
            if (arg[0] == '-') {
                fprintf(stderr, "unknown option: %s  (see --help; environment "
                                "switches like CHAT_DUMP=1 go before the command)\n", arg);
                exit_code = 2; return false;
            }
            if (strchr(arg, '='))
                fprintf(stderr, "%s looks like an environment assignment; put it before "
                                "the command, not after it\n", arg);
            else
                fprintf(stderr, "unexpected argument: %s  (the model comes from MODEL or "
                                "--model; see --help)\n", arg);
            exit_code = 2; return false;
        }
        const char* vals[2] = {"", ""};
        int n = 0;
        if (attached) {
            vals[0] = attached;
            n = 1;
        } else if (f->nargs < 0) {
            n = 0;
        } else {
            if (f->nargs > 0 && i + f->nargs >= argc) {
                fprintf(stderr, "%s needs %d value%s (%s)\n", f->name, f->nargs,
                        f->nargs == 1 ? "" : "s", f->argname);
                exit_code = 2; return false;
            }
            for (int k = 0; k < f->nargs; ++k) vals[k] = argv[++i];
            n = f->nargs;
        }
        if (!apply(f->id, vals, n, opt, quiet, exit_code)) return false;
    }

    // A zero temperature means greedy.  The sampler defines it that way too
    // (`sampling.cpp`, `keep_distribution`), but saying it here also switches the
    // fast path on: before this, `--temp 0` divided by zero, every probability came
    // out NaN and the draw returned the *least* likely token -- silently.
    if (opt.temp <= 0.0f) opt.greedy = true;

    return true;
}
