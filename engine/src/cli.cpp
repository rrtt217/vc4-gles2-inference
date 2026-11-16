// The command line parser.  See cli.h for the contract; the option semantics are
// unchanged from when this lived in `main()` -- including the environment-only
// switches (REPETITION_PENALTY and friends) that predate the flags.

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

static const char* kUsage =
    "usage: engine_mm3 [options]                       (MODEL=<file.mmcf> or --model)\n"
    "\n"
    "  --model PATH          container to run (or set MODEL)\n"
    "  --chat                chat mode; with no --prompt it is a REPL on stdin\n"
    "  --prompt TEXT         one-shot prompt ('-' reads stdin to EOF)\n"
    "  --system TEXT         system prompt (default: the container's)\n"
    "  --gen N              tokens to generate (default 8, 128 with --chat)\n"
    "  --greedy             argmax instead of sampling\n"
    "  --temp F --topp F     sampling knobs (default 1.0 / 0.9; temp 0 = greedy)\n"
    "  --top-k N            keep only the N most likely tokens (0 = off)\n"
    "  --top-k-candidates   with --top-k, read back only the candidates instead\n"
    "                       of the full logits (approximate; falls back when it\n"
    "                       cannot decide).  Without this flag the CPU path is exact\n"
    "  --seed N              sampling seed (default 1234)\n"
    "  --min-p F --repeat-penalty F --frequency-penalty F --presence-penalty F\n"
    "  --no-repeat-ngram N   sampling penalties\n"
    "  --tools JSON | --tools-file F   tool definitions for the model\n"
    "  --thinking            open a thinking block before the answer\n"
    "  --prefix-cache F      load a prefix cache (needs GPU attention)\n"
    "  --build-prefix-cache PROMPTFILE OUT   write one and exit\n"
    "  --ppl FILE            perplexity of a text file\n"
    "  --dump-shaders[=raw]  print every shader builder's GLSL digest (or the\n"
    "                        text) and exit; needs no model and no GPU\n"
    "  --worker              JSON-lines worker (the HTTP server drives this)\n"
    "  -q, --quiet           no informational lines on stderr\n"
    "  -V, --version         print the version\n"
    "  -h, --help            this text\n"
    "\n"
    "In the REPL: /help, /clear (forget the conversation), /exit.\n"
    "Sampling penalties also come from the environment (MIN_P, ...); see\n"
    "docs/test-hooks.md.  The container carries the tokenizer, the template,\n"
    "rope and the special-token ids, so there are no flags for those.\n";


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
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") || !strcmp(argv[i], "help")) {
            printf("%s", kUsage); exit_code = 0; return false;
        }
        else if (!strcmp(argv[i], "--version") || !strcmp(argv[i], "-V")) {
            printf("engine_mm3 %s (MMCF v2)\n", kVersion); exit_code = 0; return false;
        }
        else if (!strcmp(argv[i], "-q") || !strcmp(argv[i], "--quiet")) quiet = true;
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) opt.model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            opt.prompt = argv[++i]; opt.prompt_set = true;
            if (opt.prompt == "-") {                       // everything on stdin
                stringstream ss; ss << cin.rdbuf(); opt.prompt = ss.str();
                while (!opt.prompt.empty() && (opt.prompt.back() == '\n' || opt.prompt.back() == '\r'))
                    opt.prompt.pop_back();
            }
        }
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) { opt.gen = atoi(argv[++i]); opt.gen_set = true; }
        else if (!strcmp(argv[i], "--chat")) opt.chat = true;
        else if (!strcmp(argv[i], "--ppl") && i + 1 < argc) opt.ppl_path = argv[++i];
        else if (!strcmp(argv[i], "--worker")) opt.worker = true;
        else if (!strcmp(argv[i], "--build-prefix-cache") && i + 2 < argc) {
            opt.build_cache_prompt = argv[++i]; opt.build_cache_out = argv[++i];
        }
        else if (!strcmp(argv[i], "--prefix-cache") && i + 1 < argc) opt.prefix_cache_path = argv[++i];
        else if (!strcmp(argv[i], "--greedy")) opt.greedy = true;
        else if (!strcmp(argv[i], "--thinking") || !strcmp(argv[i], "--open-thinking")) opt.open_thinking = true;
        else if (!strcmp(argv[i], "--tools") && i + 1 < argc) opt.tools_json = argv[++i];
        else if (!strcmp(argv[i], "--tools-file") && i + 1 < argc) {
            ifstream tf(argv[++i]);
            if (!tf) { cerr << "cannot open tools file\n"; exit_code = 1; return false; }
            stringstream ss; ss << tf.rdbuf(); opt.tools_json = ss.str();
        }
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) opt.system_prompt = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) opt.seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) opt.temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) opt.top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) opt.top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k-candidates")) opt.top_k_candidates = true;
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) g_sampling.repetition_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frequency-penalty") && i + 1 < argc) g_sampling.frequency_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--presence-penalty") && i + 1 < argc) g_sampling.presence_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-repeat-ngram") && i + 1 < argc) g_sampling.no_repeat_ngram = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) g_sampling.min_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump-shaders")) opt.dump_mode = 1;
        else if (!strcmp(argv[i], "--dump-shaders=raw")) opt.dump_mode = 2;
        else {
            // Silently ignoring an argument is how "CHAT_DUMP=1" ends up on the
            // command line (where it is not an environment variable) and the
            // switch appears not to work at all.
            if (argv[i][0] == '-') {
                fprintf(stderr, "unknown option: %s  (see --help; environment "
                                "switches like CHAT_DUMP=1 go before the command)\n", argv[i]);
                exit_code = 2; return false;
            }
            if (strchr(argv[i], '='))
                fprintf(stderr, "%s looks like an environment assignment; put it before "
                                "the command, not after it\n", argv[i]);
            else
                fprintf(stderr, "unexpected argument: %s  (the model comes from MODEL or "
                                "--model; see --help)\n", argv[i]);
            exit_code = 2; return false;
        }
    }

    // A zero temperature means greedy.  The sampler defines it that way too
    // (`sampling.cpp`, `keep_distribution`), but saying it here also switches the
    // fast path on: before this, `--temp 0` divided by zero, every probability came
    // out NaN and the draw returned the *least* likely token -- silently.
    if (opt.temp <= 0.0f) opt.greedy = true;

    return true;
}
