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
#include "shaders.h"
#include "xformat.h"
#include "gpu.h"
#include "model.h"
#include "chattmpl.h"
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

// One short English prompt for every chat model.  MiniMind3's old Chinese
// prompt was longer and made the default behave differently from the Qwen
// template; a single ASCII prompt keeps prefill small and is easy to override
// with --system.
static const char* kDefaultSystem = "You are a helpful assistant.";

static const char* kVersion = "1.0";

// Informational stderr lines (template, tokenizer, specials, rope, cache) are
// useful when something is wrong and noise when it is not; -q turns them off.
static bool g_quiet = false;

static const char* kUsage =
    "usage: engine_mm3 [options]                       (MODEL=<file.mmcf> or --model)\n"
    "\n"
    "  --model PATH          container to run (or set MODEL)\n"
    "  --chat                chat mode; with no --prompt it is a REPL on stdin\n"
    "  --prompt TEXT         one-shot prompt ('-' reads stdin to EOF)\n"
    "  --system TEXT         system prompt (default: the container's)\n"
    "  --gen N              tokens to generate (default 8, 128 with --chat)\n"
    "  --greedy             argmax instead of sampling\n"
    "  --temp F --topp F     sampling knobs (default 1.0 / 0.9)\n"
    "  --seed N              sampling seed (default 1234)\n"
    "  --min-p F --repeat-penalty F --frequency-penalty F --presence-penalty F\n"
    "  --no-repeat-ngram N   sampling penalties\n"
    "  --tools JSON | --tools-file F   tool definitions for the model\n"
    "  --thinking            open a thinking block before the answer\n"
    "  --prefix-cache F      load a prefix cache (needs GPU attention)\n"
    "  --build-prefix-cache PROMPTFILE OUT   write one and exit\n"
    "  --ppl FILE            perplexity of a text file\n"
    "  --worker              JSON-lines worker (the HTTP server drives this)\n"
    "  -q, --quiet           no informational lines on stderr\n"
    "  -V, --version         print the version\n"
    "  -h, --help            this text\n"
    "\n"
    "In the REPL: /help, /clear (forget the conversation), /exit.\n"
    "Sampling knobs also come from the environment (TOPK, MIN_P, ...); see\n"
    "docs/test-hooks.md.  The container carries the tokenizer, the template,\n"
    "rope and the special-token ids, so there are no flags for those.\n";

int main(int argc, char** argv) {
    install_stop_handlers();
    gpu_set_run_t0(Clock::now());
    // No default: the engine only reads MMCF containers, so a bare run has
    // nothing sensible to load and "set MODEL=" is a better answer than a magic
    // number from a file that happens to sit at the old llama2.c path.
    string model_path = getenv("MODEL") ? getenv("MODEL") : "";
    string tokenizer_path = getenv("TOKENIZER") ? getenv("TOKENIZER") : "../models/llama2c/tokenizer.bin";
    string prompt = "One day, Lily";
    string system_prompt;
    int gen = 8;
    bool prompt_set = false, gen_set = false;
    bool chat = false, qwen_template = false, minimind3_template = false, greedy = false, worker = false;
    bool tbb_template = false;   // <|user|>/<|assistant|>/<|end|>, no system turn, no BOS
    bool open_thinking = false;
    string tools_json;
    string build_cache_prompt, build_cache_out, prefix_cache_path, ppl_path;
    uint32_t seed = 1234;
    float temp = 1.0f, top_p = 0.9f;
    int topk = getenv("TOPK") ? atoi(getenv("TOPK")) : 0;
    if (getenv("REPETITION_PENALTY")) g_sampling.repetition_penalty = atof(getenv("REPETITION_PENALTY"));
    if (getenv("FREQUENCY_PENALTY")) g_sampling.frequency_penalty = atof(getenv("FREQUENCY_PENALTY"));
    if (getenv("PRESENCE_PENALTY")) g_sampling.presence_penalty = atof(getenv("PRESENCE_PENALTY"));
    if (getenv("NO_REPEAT_NGRAM")) g_sampling.no_repeat_ngram = atoi(getenv("NO_REPEAT_NGRAM"));
    if (getenv("MIN_P")) g_sampling.min_p = atof(getenv("MIN_P"));

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help") || !strcmp(argv[i], "help")) {
            printf("%s", kUsage); return 0;
        }
        else if (!strcmp(argv[i], "--version") || !strcmp(argv[i], "-V")) {
            printf("engine_mm3 %s (MMCF v2)\n", kVersion); return 0;
        }
        else if (!strcmp(argv[i], "-q") || !strcmp(argv[i], "--quiet")) g_quiet = true;
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            prompt = argv[++i]; prompt_set = true;
            if (prompt == "-") {                       // everything on stdin
                stringstream ss; ss << cin.rdbuf(); prompt = ss.str();
                while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r'))
                    prompt.pop_back();
            }
        }
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); gen_set = true; }
        else if (!strcmp(argv[i], "--chat")) chat = true;
        else if (!strcmp(argv[i], "--ppl") && i + 1 < argc) ppl_path = argv[++i];
        else if (!strcmp(argv[i], "--worker")) worker = true;
        else if (!strcmp(argv[i], "--build-prefix-cache") && i + 2 < argc) {
            build_cache_prompt = argv[++i]; build_cache_out = argv[++i];
        }
        else if (!strcmp(argv[i], "--prefix-cache") && i + 1 < argc) prefix_cache_path = argv[++i];
        else if (!strcmp(argv[i], "--greedy")) greedy = true;
        else if (!strcmp(argv[i], "--thinking") || !strcmp(argv[i], "--open-thinking")) open_thinking = true;
        else if (!strcmp(argv[i], "--tools") && i + 1 < argc) tools_json = argv[++i];
        else if (!strcmp(argv[i], "--tools-file") && i + 1 < argc) {
            ifstream tf(argv[++i]);
            if (!tf) { cerr << "cannot open tools file\n"; return 1; }
            stringstream ss; ss << tf.rdbuf(); tools_json = ss.str();
        }
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) system_prompt = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeat-penalty") && i + 1 < argc) g_sampling.repetition_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frequency-penalty") && i + 1 < argc) g_sampling.frequency_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--presence-penalty") && i + 1 < argc) g_sampling.presence_penalty = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-repeat-ngram") && i + 1 < argc) g_sampling.no_repeat_ngram = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) g_sampling.min_p = atof(argv[++i]);
        else {
            // Silently ignoring an argument is how "CHAT_DUMP=1" ends up on the
            // command line (where it is not an environment variable) and the
            // switch appears not to work at all.
            if (argv[i][0] == '-') {
                fprintf(stderr, "unknown option: %s  (see --help; environment "
                                "switches like CHAT_DUMP=1 go before the command)\n", argv[i]);
                return 2;
            }
            if (strchr(argv[i], '='))
                fprintf(stderr, "%s looks like an environment assignment; put it before "
                                "the command, not after it\n", argv[i]);
            else
                fprintf(stderr, "unexpected argument: %s  (the model comes from MODEL or "
                                "--model; see --help)\n", argv[i]);
            return 2;
        }
    }
    if (chat && !gen_set) gen = 128;

    if (model_path.empty()) {
        fprintf(stderr, "no model: set MODEL=<file.mmcf> or pass --model PATH (see --help)\n");
        return 2;
    }
    {
        ifstream probe(model_path, ios::binary);
        if (!probe) {
            fprintf(stderr, "cannot open model %s\n"
                            "  set MODEL=<file.mmcf> or pass --model PATH; the store lives in "
                            "models/<name>/<name>.<quant>.mmcf (see --help)\n", model_path.c_str());
            return 2;
        }
    }
    Model model;
    if (!model.load(model_path)) return 1;
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  model loaded\n", run_time_ms());
    GPU gpu;
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  egl ready\n", run_time_ms());
    if (getenv("Q8_PARTS")) model.apply_precision(getenv("Q8_LAYERS") ? getenv("Q8_LAYERS") : "", getenv("Q8_PARTS"));
    model.split_ffn();
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  ffn split\n", run_time_ms());
    if (getenv("Q4_WEIGHTS")) model.quantize_q4_weights(getenv("Q4_SCOPE") ? getenv("Q4_SCOPE") : "all");
    if (!tools_json.empty() && model.seq < 768) model.seq = 768;
    if ((worker || !build_cache_prompt.empty()) && model.seq < 1024 && !getenv("SEQ")) model.seq = 1024;
    const int target_limit = gpu.max_target_dim();
    if (model.seq > target_limit) {
        cerr << "SEQ " << model.seq << " exceeds GL target limit " << target_limit
             << "; clamping to " << target_limit << "\n";
        model.seq = target_limit;
    }
    if (model.seq < 16) model.seq = 16;
    if (system_prompt.empty() && !model.default_system.empty()) system_prompt = model.default_system;
    // The checkpoint names its template (a kind byte plus a name), so --template
    // is only needed to override it.  The kind is authoritative; the name is the
    // fallback for a kind this build does not know.
    if (!model.template_name.empty() || model.template_kind >= 0) {
        const int k = model.template_kind;
        bool want_qwen = false, want_mm3 = false, want_tbb = false;
        if (k == 3) { want_tbb = true; }
        else if (k == 2) { want_qwen = want_mm3 = true; }
        else if (k == 1) { want_qwen = true; }
        else if (k == 0) { want_qwen = want_mm3 = false; }
        else {   // unknown kind: fall back to the name
            want_tbb = model.template_name == "tinybrainbot";
            want_qwen = model.template_name == "qwen" || model.template_name == "minimind3";
            want_mm3 = model.template_name == "minimind3";
        }
        const bool was_qwen = qwen_template, was_mm3 = minimind3_template;
        qwen_template = qwen_template || want_qwen;
        minimind3_template = minimind3_template || want_mm3;
        tbb_template = tbb_template || want_tbb;
        if (tbb_template) qwen_template = false;   // its own markers, not chatml
        if ((!was_qwen && qwen_template) || (!was_mm3 && minimind3_template))
            if (!g_quiet) cerr << "template: " << (model.template_name.empty() ? "kind " + std::to_string(k) : model.template_name)
                 << " (from the checkpoint)\n";
    }
    // Metadata key 9: the model's own Jinja template.  When it is present it is
    // the prompt builder; template_kind stays as the fallback for containers
    // written before the key existed.  A template the vendored engine cannot
    // parse is refused here, not silently swapped for a built-in one.
    std::unique_ptr<ChatTemplate> chat_tmpl;
    if (!model.chat_template.empty()) {
        chat_tmpl.reset(new ChatTemplate(model.chat_template));
        if (!chat_tmpl->ok()) {
            cerr << "chat template rejected: " << chat_tmpl->error() << "\n";
            return 2;
        }
        if (!g_quiet) cerr << "chat template: " << model.chat_template.size() << " B from the checkpoint ("
             << model.chat_templates.size() << " variant(s))\n";
    }
    // Capabilities: a checkpoint that says it is not a chat model must not be
    // driven through the chat template.
    if (model.have_capabilities && chat && !(model.capabilities & 1)) {
        cerr << "this checkpoint does not advertise chat support (capabilities 0x"
             << std::hex << model.capabilities << std::dec << "); drop --chat\n";
        return 2;
    }
    if (model.have_capabilities && !chat && (model.capabilities & 1) && !gen_set) {
        chat = true;
        gen = 128;
    }
    Tokenizer tokenizer;
    if (model.tok_blob && !getenv("TOKENIZER")) {
        if (!tokenizer.load_memory(model.tok_blob, model.tok_blob_bytes, model.vocab)) return 1;
        if (!g_quiet) cerr << "tokenizer: " << model.tok_blob_bytes << " B embedded in the checkpoint\n";
    } else if (!tokenizer.load(tokenizer_path, model.vocab)) {
        return 1;
    }
    tokenizer.set_bos(model.have_specials ? model.id_bos : 1);
    if (model.have_specials)
        if (!g_quiet) cerr << "specials: bos " << model.id_bos << " eos " << model.id_eos
             << " stop " << model.id_stop << " (from the checkpoint)\n";
    tokenizer.set_qwen(qwen_template || minimind3_template);
    tokenizer.set_bytelevel(minimind3_template);
    // Tool definitions, normalised to one JSON array for the template context.
    const string tools_array = chat_tools_array(tools_json);
    // BOS policy: the container states it when it can (key 10); otherwise the
    // historical rule is kept, which is derived from the template kind.
    const bool enc_bos = model.add_bos >= 0 ? (model.add_bos != 0)
                                            : (!qwen_template && !tbb_template);
    // CHAT_DUMP=1 prints the rendered prompt with its input marking, which is
    // what the byte-for-byte template comparisons are run against.
    const bool chat_dump = getenv("CHAT_DUMP") != nullptr;
    // Stop-token test.  The container states EOS and the extra stop id (key 8).
    // The CLI used to test the literal ids 1 and 2, which is MiniMind3's
    // coincidence; a model whose EOS is 7 never stopped.
    auto is_stop_token = [&](int id) {
        if (model.have_specials)
            return id == model.id_eos || (model.id_stop >= 0 && id == model.id_stop);
        return id == 1 || id == 2;
    };

    gpu.set_rms_eps(model.rms_eps, model.rms_eps_stated);
    if (getenv("SHADER_TIME"))
        cerr << "rmsnorm eps: " << model.rms_eps << " (container"
             << (model.rms_eps == 1e-5f ? "" : "-stated") << ")\n";
    const auto up_t0 = Clock::now();
    for (auto& m : model.qkv) gpu.upload(m);
    for (auto& m : model.wo) gpu.upload(m);
    for (auto& m : model.w13) gpu.upload(m);
    for (auto& m : model.w1) gpu.upload(m);
    for (auto& m : model.w3) gpu.upload(m);
    for (auto& m : model.w2) gpu.upload(m);
    for (auto& m : model.w1a) gpu.upload(m);
    for (auto& m : model.w1b) gpu.upload(m);
    for (auto& m : model.w3a) gpu.upload(m);
    for (auto& m : model.w3b) gpu.upload(m);
    for (auto& m : model.w2a) gpu.upload(m);
    for (auto& m : model.w2b) gpu.upload(m);
    for (auto& m : model.logits_tiles) gpu.upload(m);
    model.release_map();
    if (getenv("SHADER_TIME") || getenv("UPLOAD_TIME"))
        fprintf(stderr, "UPLOAD %.2f s wall, %d textures, texture time %.0f ms\n",
                elapsed_ms(up_t0, Clock::now()) / 1000.0, gpu_upload_textures(), gpu_upload_texture_ms());
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  weights uploaded\n", run_time_ms());

    Session session(model, gpu);
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  session ready\n", run_time_ms());

    if (!ppl_path.empty()) {
        string text;
        if (!read_text_file(ppl_path, text)) { cerr << "cannot read " << ppl_path << "\n"; return 1; }
        auto tokens = tokenizer.encode(text, false, false);
        if ((int)tokens.size() < 4) { cerr << "not enough tokens\n"; return 1; }
        int n = min((int)tokens.size(), model.seq - 1);
        double nll = 0.0;
        int cnt = 0;
        for (int i = 0; i + 1 < n; ++i) {
            if (g_stop) break;
            vector<float> lg = session.step(tokens[i], i, true);
            if ((int)lg.size() <= tokens[i + 1]) break;
            double mx = -1e30;
            for (float v : lg) mx = max((double)v, mx);
            double se = 0.0;
            for (float v : lg) se += exp((double)v - mx);
            double logp = ((double)lg[tokens[i + 1]] - mx) - log(se);
            nll += -logp;
            ++cnt;
        }
        if (!cnt) { cerr << "no tokens scored\n"; return 1; }
        printf("PPL %.4f  nll=%.4f  tokens=%d\n", exp(nll / cnt), nll / cnt, cnt);
        return 0;
    }

    if (!build_cache_prompt.empty()) {
        string text;
        if (!read_text_file(build_cache_prompt, text)) { cerr << "cannot read cache prompt\n"; return 1; }
        auto tokens = tokenizer.encode(text, false, false);
        if (tokens.empty() || (int)tokens.size() > model.seq) {
            cerr << "cache prompt empty or too long (" << tokens.size() << ")\n"; return 1;
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (g_stop) return 1;
            session.step(tokens[i], (int)i, false);
        }
        if (!session.save_prefix_cache(build_cache_out, tokens)) {
            cerr << "cannot save prefix cache (GPU attention required)\n"; return 1;
        }
        cout << "CACHE_SAVED " << build_cache_out << " tokens=" << tokens.size() << "\n";
        return 0;
    }

    if (worker) {
        auto emit = [&](const string& s) { cout << s << "\n" << flush; };
        // Process-level defaults (CLI flags / env vars), restored before each request.
        const SamplingSettings base_sampling = g_sampling;
        vector<int> cache_tokens;
        if (!prefix_cache_path.empty()) {
            if (session.load_prefix_cache(prefix_cache_path, cache_tokens))
                cerr << "PREFIX_CACHE loaded " << cache_tokens.size() << " tokens\n";
            else
                cerr << "PREFIX_CACHE ignored " << prefix_cache_path << "\n";
        }
        // The HTTP layer renders the prompt text itself, so it has to know which
        // template this checkpoint wants.  The container states it (kind + name)
        // and the resolution above already turned it into these three flags, so
        // report the effective one: the server must not guess from a file name.
        const char* effective_template = tbb_template ? "tinybrainbot"
                                        : minimind3_template ? "minimind3"
                                        : qwen_template ? "chatml" : "plain";
        emit("{\"event\":\"ready\",\"prefix_cache_tokens\":" + to_string(cache_tokens.size()) +
             ",\"template\":\"" + json_escape(effective_template) + "\"" +
             ",\"chat_template\":" + (chat_tmpl ? "true" : "false") + "}");
        string line;
        while (getline(cin, line)) {
            if (g_stop) break;
            if (line.empty()) continue;
            vector<string> f;
            size_t start = 0;
            while (start <= line.size()) {
                size_t tab = line.find('\t', start);
                f.push_back(line.substr(start, tab == string::npos ? string::npos : tab - start));
                if (tab == string::npos) break;
                start = tab + 1;
            }
            if (f.size() < 5) { emit("{\"event\":\"error\",\"message\":\"bad request line\"}"); continue; }
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
                    emit("{\"event\":\"error\",\"message\":\"" + json_escape("bad request: " + perr) + "\"}");
                    continue;
                }
                if (!chat_tmpl) {
                    emit("{\"event\":\"error\",\"message\":\"this checkpoint carries no chat template; send a prompt string\"}");
                    continue;
                }
                req_parts = chat_tmpl->render(req_msgs, req_tools, true,
                                              req_thinking >= 0 ? req_thinking != 0 : open_thinking,
                                              "", "");
                if (chat_dump)
                    for (auto& p : req_parts)
                        fprintf(stderr, "PROMPT[%s] %3zu B  %s\n", p.is_input ? "in" : "tmpl",
                            p.text.size(), dump_escape(p.text).c_str());
            }
            auto tokens = req_parts.empty() ? tokenizer.encode(prompt, false, false)
                                            : tokenizer.encode_parts(req_parts, enc_bos, enc_bos);
            if (tokens.empty()) {
                emit("{\"event\":\"error\",\"message\":\"empty prompt\"}");
                continue;
            }
            if ((int)tokens.size() >= model.seq) {
                emit("{\"event\":\"error\",\"message\":\"context full\"}");
                continue;
            }
            int common = 0;
            while (common < (int)cache_tokens.size() && common < (int)tokens.size() &&
                   cache_tokens[common] == tokens[common]) ++common;
            int prefill_from = common;
            if (prefill_from >= (int)tokens.size()) prefill_from = (int)tokens.size() - 1;
            int prefill_tokens = (int)tokens.size() - prefill_from;
            emit("{\"event\":\"start\",\"prompt_tokens\":" + to_string(tokens.size()) +
                 ",\"cached_tokens\":" + to_string(common) +
                 ",\"prefill_tokens\":" + to_string(prefill_tokens) + "}");
            vector<float> logits;
            for (int i = prefill_from; i < (int)tokens.size(); ++i) {
                if (g_stop) break;
                logits = session.step(tokens[i], i, i + 1 == (int)tokens.size());
            }
            if (g_stop) break;
            int pos = (int)tokens.size();
            max_new = min(max_new, model.seq - pos);
            bool greedy_req = wt <= 0.0f;
            mt19937 rng(seed);
            string pending, full, finish = "length";
            int generated = 0;
            int prev = tokens.back();
            vector<int> all_tokens = tokens;
            vector<int> recent;
            for (int i = 0; i < max_new; ++i) {
                if (g_stop) { finish = "stop"; break; }
                apply_sampling_penalties(logits, recent);
                int next = greedy_req ? argmax_token(logits)
                                      : sample_token(logits, wt, wp, rng);
                const bool is_eos = model.have_specials
                        ? (next == model.id_eos || (model.id_stop >= 0 && next == model.id_stop))
                        : (next == 1 || next == 2);
                if (is_eos) {
                    session.step(next, pos++, false);
                    all_tokens.push_back(next);
                    finish = "stop";
                    break;
                }
                string piece = tokenizer.decode({next}, prev);
                pending += piece;
                string text = utf8_take(pending);
                if (!text.empty()) {
                    full += text;
                    emit("{\"event\":\"token\",\"text\":\"" + json_escape(text) + "\"}");
                }
                prev = next;
                ++generated;
                all_tokens.push_back(next);
                recent.push_back(next);
                logits = session.step(next, pos++, true);
            }
            if (!pending.empty()) {
                string text = "\xEF\xBF\xBD";
                full += text;
                emit("{\"event\":\"token\",\"text\":\"" + json_escape(text) + "\"}");
            }
            if (g_stop) finish = "stop";
            if (g_stop) cache_tokens.clear();
            else cache_tokens = all_tokens;
            emit("{\"event\":\"done\",\"text\":\"" + json_escape(full) +
                 "\",\"finish_reason\":\"" + finish +
                 "\",\"prompt_tokens\":" + to_string(tokens.size()) +
                 ",\"completion_tokens\":" + to_string(generated) +
                 ",\"cached_tokens\":" + to_string(common) +
                 ",\"prefill_tokens\":" + to_string(prefill_tokens) + "}");
        }
        return 0;
    }


    vector<int> cli_cache_tokens;
    if (!worker && !prefix_cache_path.empty()) {
        if (session.load_prefix_cache(prefix_cache_path, cli_cache_tokens)) {
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
            cerr << "PREFIX_CACHE ignored " << prefix_cache_path << "\n";
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

    mt19937 rng(seed);
    // Tokens currently in the KV cache; the REPL keeps this so a turn can
    // prefill only what changed instead of the whole conversation.
    vector<int> kv_tokens;
    auto run_turn = [&](const string& user, bool first, int pos, vector<ChatMsg>* hist = nullptr) {
        auto turn_start = Clock::now();
        string text;
        vector<PromptPart> parts;
        if (chat_tmpl) {
            string sys = system_prompt.empty() ? string(kDefaultSystem) : system_prompt;
            vector<ChatMsg> msgs;
            if (hist) {
                // The REPL keeps the conversation; the system turn goes in once,
                // at the front, and the whole history is re-rendered so the
                // template sees the same shape it was trained on.
                if (hist->empty()) hist->push_back(ChatMsg{"system", sys, "", ""});
                hist->push_back(ChatMsg{"user", user, "", ""});
                msgs = *hist;
            } else {
                if (first) msgs.push_back(ChatMsg{"system", sys, "", ""});
                msgs.push_back(ChatMsg{"user", user, "", ""});
            }
            parts = chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/true,
                                      open_thinking, "", "");
        } else if (minimind3_template) {
            string sys = system_prompt.empty() ? string(kDefaultSystem) : system_prompt;
            if (!tools_json.empty()) {
                sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                       "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                       tools_json +
                       "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                       "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
            }
            if (first)
                text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
            text += "<|im_start|>user\n" + user + "<|im_end|>\n"
                    "<|im_start|>assistant\n";
            text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        } else if (tbb_template) {
            // No system turn and no BOS: the markers carry the roles, and the
            // first user turn opens with a space (the model was trained so).
            // "<|user|> {q}<|assistant|> " with a trailing space: the model
            // was trained with a space after the assistant marker, and without
            // it the first token is <|end|> (measured on the q4 and i16 files,
            // where the other four wrappers I tried collapse to one token).
            text = string("<|user|> ") + user + "<|assistant|> ";
        } else if (qwen_template) {
            if (first)
                text = "<|im_start|>system\n" + (system_prompt.empty() ? string(kDefaultSystem) : system_prompt) +
                       "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            else
                text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            if (open_thinking) text += "<think>\n";
        } else {
            if (first && !system_prompt.empty())
                text = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user + " [/INST]";
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
        auto tokens = parts.empty() ? tokenizer.encode(text, enc_bos, enc_bos)
                                    : tokenizer.encode_parts(parts, enc_bos, enc_bos);
        int skip = (first && !cli_cache_tokens.empty()) ? cache_prefix_len(tokens) : 0;
        if (hist) {
            size_t n = 0;
            while (n < kv_tokens.size() && n < tokens.size() && kv_tokens[n] == tokens[n]) ++n;
            // the KV holds rows, not the last token's logits, so the final
            // token of an otherwise-cached prompt still has to run
            if (!tokens.empty() && n == tokens.size()) n = tokens.size() - 1;
            skip = (int)n;
        }
        for (int i = 0; i < skip; ++i) pos = i + 1;
        vector<float> logits;
        CandidateResult cand;
        bool cand_mode = topk != 0 && !greedy && !sampling_penalties_active();
        for (size_t i = skip; i < tokens.size(); ++i) {
            if (g_stop) { gpu.sync(); return; }
            if (pos >= model.seq) { cout << "[context full]\n"; return; }
            bool last = (i + 1 == tokens.size());
            if (last && cand_mode) {
                cand = session.step_candidates(tokens[i], pos++, topk, temp, top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else if (!last && i + 2 < tokens.size() && pos + 1 < model.seq && session.pair_ok()) {
                // Two prompt tokens per layer pass; the last token stays on the
                // single-position path because its logits are the ones we need.
                session.step_pair(tokens[i], pos, tokens[i + 1], pos + 1, false);
                pos += 2;
                ++i;
            } else {
                logits = session.step(tokens[i], pos++, last);
            }
        }
        if (g_stop) { gpu.sync(); return; }
        auto dec_start = Clock::now();
        int generated = 0;
        if (!cand_mode && getenv("LOGITS_OUT")) {
            ofstream f(getenv("LOGITS_OUT"), ios::binary);
            f.write((char*)logits.data(), logits.size() * 4);
        }
        cout << "Assistant: "; cout.flush();
        string reply;
        int prev = tokens.empty() ? 1 : tokens.back();
        vector<int> recent;
        for (int i = 0; i < gen && pos < model.seq; ++i) {
            if (g_stop) { gpu.sync(); return; }
            apply_sampling_penalties(logits, recent);
            int next = greedy ? argmax_token(logits)
                              : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                           : sample_token(logits, temp, top_p, rng));
            if (!getenv("IGNORE_EOS") && is_stop_token(next)) {
                session.step(next, pos++, false);
                break;
            }
            cout << tokenizer.decode({next}, prev); cout.flush();
            reply += tokenizer.decode({next}, prev);
            tokens.push_back(next);
            recent.push_back(next);
            ++generated;
            prev = next;
            if (cand_mode) {
                cand = session.step_candidates(next, pos++, topk, temp, top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else {
                logits = session.step(next, pos++, true);
            }
        }
        cout << "\n";
        if (hist) {
            hist->push_back(ChatMsg{"assistant", reply, "", ""});
            kv_tokens = tokens;
        }
        if (getenv("TIMING")) {
            double ms = elapsed_ms(dec_start, Clock::now());
            cerr << "TIMING prefill=" << elapsed_ms(turn_start, dec_start) / 1000.0
                 << "s decode_avg=" << (generated ? ms / generated : 0.0) << "ms/token gen=" << generated << "\n";
        }
        if (getenv("TOKENS_OUT")) {
            ofstream f(getenv("TOKENS_OUT"), ios::binary);
            for (int t : tokens) f.write((char*)&t, 4);
        }
        if (g_stop) gpu.sync();
    };

    // The assistant turn is re-rendered from the template, so the raw
    // <tool_call> blocks must not also travel inside `content` or the template
    // would emit them twice.
    auto strip_tool_calls = [](const string& text) {
        string out;
        size_t pos = 0;
        while (true) {
            size_t a = text.find("<tool_call>", pos);
            if (a == string::npos) { out += text.substr(pos); break; }
            out += text.substr(pos, a - pos);
            size_t b = text.find("</tool_call>", a);
            if (b == string::npos) break;
            pos = b + strlen("</tool_call>");
        }
        string trimmed;
        for (char c : out) if (!isspace((unsigned char)c)) { trimmed += c; }
        return trimmed.empty() ? string() : out;
    };
    auto tool_calls_json = [](const vector<ToolCall>& calls) {
        string j = "[";
        for (size_t i = 0; i < calls.size(); ++i) {
            if (i) j += ",";
            j += "{\"type\":\"function\",\"function\":{\"name\":\"" + json_escape(calls[i].name) +
                 "\",\"arguments\":\"" + json_escape(calls[i].args) + "\"}}";
        }
        return j + "]";
    };

    auto run_tool_turn = [&](const string& user, bool first) {
        string sys = system_prompt.empty()
            ? string("你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。")
            : system_prompt;
        if (!tools_json.empty()) {
            sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                   tools_json +
                   "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        // Template path: the whole prompt is re-rendered from this list every
        // round, so the tool loop is model-agnostic.  The legacy path below
        // keeps building one string, which is why it only ever worked for the
        // ChatML family.
        vector<ChatMsg> msgs;
        if (chat_tmpl) {
            if (first) msgs.push_back(ChatMsg{"system",
                system_prompt.empty() ? string(kDefaultSystem) : system_prompt, "", ""});
            msgs.push_back(ChatMsg{"user", user, "", ""});
        }
        string text;
        if (first || !tools_json.empty())
            text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
        text += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

        for (int round = 0; round < 3; ++round) {
            vector<PromptPart> parts;
            if (chat_tmpl)
                parts = chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/true,
                                          open_thinking, "", "");
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
            auto tokens = chat_tmpl ? tokenizer.encode_parts(parts, enc_bos, enc_bos)
                                    : tokenizer.encode(text, false, false);
            if ((int)tokens.size() >= model.seq) { cout << "[context full]\n"; return; }
            vector<float> logits;
            CandidateResult cand;
            bool cand_mode = topk != 0 && !greedy && !sampling_penalties_active();
            int skip = (first && round == 0 && !cli_cache_tokens.empty()) ? cache_prefix_len(tokens) : 0;
            for (size_t i = skip; i < tokens.size(); ++i) {
                if (g_stop) { gpu.sync(); return; }
                bool last = (i + 1 == tokens.size());
                if (last && cand_mode) {
                    cand = session.step_candidates(tokens[i], (int)i, topk, temp, top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else if (!last && i + 2 < tokens.size() && (int)i + 1 < model.seq && session.pair_ok()) {
                    session.step_pair(tokens[i], (int)i, tokens[i + 1], (int)i + 1, false);
                    ++i;
                } else {
                    logits = session.step(tokens[i], (int)i, last);
                }
            }
            if (g_stop) { gpu.sync(); return; }
            cout << "Assistant: "; cout.flush();
            string out;
            int prev = tokens.empty() ? 1 : tokens.back();
            int pos = (int)tokens.size();
            vector<int> recent;
            for (int i = 0; i < gen && pos < model.seq; ++i) {
                if (g_stop) { gpu.sync(); return; }
                apply_sampling_penalties(logits, recent);
                int next = greedy ? argmax_token(logits)
                                  : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                               : sample_token(logits, temp, top_p, rng));
                if (is_stop_token(next)) {
                    session.step(next, pos++, false);
                    break;
                }
                string piece = tokenizer.decode({next}, prev);
                cout << piece; cout.flush();
                out += piece;
                recent.push_back(next);
                prev = next;
                if (cand_mode) {
                    cand = session.step_candidates(next, pos++, topk, temp, top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(next, pos++, true);
                }
            }
            cout << "\n";
            vector<ToolCall> calls = parse_tool_calls(out);
            if (calls.empty()) return;
            if (chat_tmpl) {
                msgs.push_back(ChatMsg{"assistant", strip_tool_calls(out), tool_calls_json(calls), ""});
            } else {
                text += out;
                text += "<|im_end|>\n<|im_start|>user\n";
            }
            for (const auto& c : calls) {
                string result = run_tool(c.name, c.args);
                cout << "[tool] " << c.name << " " << c.args << "\n";
                cout << "[tool result] " << result << "\n";
                if (chat_tmpl) msgs.push_back(ChatMsg{"tool", result, "", ""});
                else text += "<tool_response>\n" + result + "\n</tool_response>\n";
            }
            if (!chat_tmpl) {
                text += "<|im_end|>\n<|im_start|>assistant\n";
                text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
            }
        }
    };

    if (chat) {
        if (!prompt_set) prompt.clear();
        if (!prompt.empty()) {
            if (tools_json.empty()) run_turn(prompt, true, 0);
            else run_tool_turn(prompt, true);
        } else {
            bool first = true;
            vector<ChatMsg> history;
            while (!g_stop) {
                string line;
                if (!line_edit_read("User: ", line)) break;
                if (line.empty()) continue;
                if (line == "/exit" || line == "/quit") break;
                if (line == "/clear") {
                    first = true;
                    history.clear();
                    kv_tokens.clear();
                    cout << "[conversation cleared]\n";
                    continue;
                }
                if (line == "/help") {
                    cout << "/clear  forget the conversation so far\n"
                            "/exit   leave\n"
                            "/help   this list\n"
                            "anything else is sent to the model; arrows edit, Up/Down recall.\n";
                    continue;
                }
                if (tools_json.empty()) run_turn(line, first, 0, chat_tmpl ? &history : nullptr);
                else run_tool_turn(line, first);
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

    auto tokens = tokenizer.encode(prompt, true, true);
    cout << "tokens:";
    for (int t : tokens) cout << " " << t;
    cout << "\n";
    vector<float> logits;
    CandidateResult cand;
    bool cand_mode = topk != 0 && !greedy && !sampling_penalties_active();
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
                cand = session.step_candidates(tokens[i], (int)i, topk, temp, top_p);
                if (!cand.complete) { lg = std::move(cand.full); cand_mode = false; }
            } else if (!last && i + 2 < tokens.size() && (int)i + 1 < model.seq && session.pair_ok()) {
                session.step_pair(tokens[i], (int)i, tokens[i + 1], (int)i + 1, false);
                ++i;
            } else {
                lg = session.step(tokens[i], (int)i, last);
            }
        }
        return lg;
    };
    auto start = Clock::now();
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  first step\n", run_time_ms());
    logits = prefill_once();
    if (g_stop) { gpu.sync(); cout << "\n[interrupted]\n"; return 0; }
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
    int generated = 0;
    double decode_ms = 0;
    vector<int> recent;
    for (int i = 0; i < gen; ++i) {
        if (g_stop) break;
        apply_sampling_penalties(logits, recent);
        int next = greedy ? argmax_token(logits)
                          : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                       : sample_token(logits, temp, top_p, rng));
        if (!getenv("IGNORE_EOS") && is_stop_token(next)) break;
        cout << tokenizer.decode({next}, tokens.empty() ? 1 : tokens.back()); cout.flush();
        tokens.push_back(next);
        recent.push_back(next);
        auto a = Clock::now();
        if (cand_mode) {
            cand = session.step_candidates(next, (int)tokens.size() - 1, topk, temp, top_p);
            if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
        } else {
            logits = session.step(next, (int)tokens.size() - 1, true);
        }
        decode_ms += elapsed_ms(a, Clock::now());
        ++generated;
    }
    cout << "\n";
    if (getenv("TOKENS_OUT")) {
        ofstream f(getenv("TOKENS_OUT"), ios::binary);
        for (int t : tokens) f.write((char*)&t, 4);
    }
    if (g_stop) { gpu.sync(); cout << "[interrupted]\n"; return 0; }
    if (generated) cout << "decode avg " << decode_ms / generated << " ms/token\n";
    if (getenv("DRAWCOUNT") && generated)
        cout << "draws " << gpu.draws() << " total, "
             << (double)gpu.draws() / generated << " per token\n";
    return 0;
}
