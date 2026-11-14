// `--build-prefix-cache PROMPTFILE OUT`: write one KV cache and exit.
//
// Moved out of `main()` (2026-11-14, P2-1 of the 2026-11-13 audit).  The header is
// rendered with the container's own template when the prompt argument is `-` or
// `auto` (the Python builder used to hard-code ChatML, which is wrong for every
// template that is not ChatML); a real file is tokenised as raw text instead.
// `docs/prefix-cache.md` owns the format and the measurements.

#include "prefix_build.h"

#include "chattmpl.h"
#include "cli.h"
#include "common.h"
#include "model.h"
#include "session.h"
#include "tokenizer.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int build_prefix_cache(Model& model, Tokenizer& tokenizer,
                       const ChatTemplate* chat_tmpl, const Options& opt,
                       const std::string& tools_array, bool enc_bos,
                       const std::string& bos_str, const std::string& eos_str,
                       bool quiet, Session& session) {

    vector<int> tokens;
    const bool auto_header = opt.build_cache_prompt == "-" || opt.build_cache_prompt == "auto";
    uint64_t system_fnv = 0, tools_fnv = 0;
    if (auto_header) {
        // Render the header the way a request would: the container's own
        // template, the system prompt and the tool list.  This is the whole
        // point of the mode -- the Python builder used to hard-code ChatML,
        // which is wrong for every other template.
        if (!chat_tmpl) {
            cerr << "no chat template in this container; pass a prompt file instead of '-'\n";
            return 1;
        }
        vector<ChatMsg> msgs;
        ChatMsg sys;
        sys.role = "system";
        sys.content = opt.system_prompt;
        msgs.push_back(sys);
        vector<PromptPart> parts;
        try {
            parts = chat_tmpl->render(msgs, tools_array, /*add_generation_prompt=*/false,
                                      opt.open_thinking, bos_str, eos_str);
        } catch (const std::exception& e) {
            cerr << "chat template failed: " << e.what() << "\n";
            return 1;
        }
        tokens = tokenizer.encode_parts(parts, enc_bos, enc_bos);
        system_fnv = fnv64(opt.system_prompt);
        // The *raw* tools text, so a verifier can hash what it passed
        // without reproducing the array normalisation.
        tools_fnv = fnv64(opt.tools_json);
        if (!quiet)
            cerr << "CACHE_HEADER template=" << (model.template_name.empty() ? "(kind)" : model.template_name)
                 << " system=" << opt.system_prompt.size() << " B tools=" << tools_array.size() << " B\n";
    } else {
        string text;
        if (!read_text_file(opt.build_cache_prompt, text)) { cerr << "cannot read cache prompt\n"; return 1; }
        tokens = tokenizer.encode(text, false, false);
    }
    if (tokens.empty() || (int)tokens.size() > model.seq) {
        cerr << "cache prompt empty or too long (" << tokens.size() << ")\n"; return 1;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (g_stop) return 1;
        session.step(tokens[i], (int)i, false);
    }
    if (!session.save_prefix_cache(opt.build_cache_out, tokens, system_fnv, tools_fnv)) {
        cerr << "cannot save prefix cache (GPU attention required)\n"; return 1;
    }
    cout << "CACHE_SAVED " << opt.build_cache_out << " tokens=" << tokens.size() << "\n";
    return 0;
}
