#pragma once
// The engine's live state: the container, the GL context, the tokenizer and the
// chat template, plus the steps that turn a loaded container into a runnable one.
//
// These were locals and `[&]` lambdas in `main()` (2026-11-14, P2-1 of the
// 2026-11-13 audit).  A model swap has to repeat *every* step between reading the
// file and uploading it, which is why they live together: `prepare_model()` is the
// one place that knows the order, and both startup and `/model` call it.

#include <memory>
#include <string>
#include <vector>

#include "chattmpl.h"
#include "cli.h"
#include "gpu.h"
#include "model.h"
#include "session.h"
#include "tokenizer.h"

struct App {
    // The CLI options by reference, not by value: `setup_from_model()` decides the
    // template flags from the container (and overwrites them), and `main()` reads
    // them back -- with a copy, the two sides would disagree, which is exactly the
    // bug the default-system-prompt check caught on 2026-11-14.
    explicit App(Options& o) : opt(o) {}
    Options& opt;
    bool quiet = false;
    Model model;
    GPU gpu;
    Tokenizer tokenizer;
    std::unique_ptr<ChatTemplate> chat_tmpl;
    std::string bos_str, eos_str;
    std::string system_prompt;
    bool system_from_cli = false;

    void prepare_model();
    bool setup_from_model();
    void upload_weights();
    // The session is built after the model and the context exist, so it is a
    // parameter rather than a member.
    bool load_prefix_cache_any(const std::string& path, std::vector<int>& toks,
                              Session& session);
};
