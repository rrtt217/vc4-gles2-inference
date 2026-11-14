// App: the engine's state and the load/prepare/upload steps that main() used to
// carry as `[&]` lambdas (2026-11-14, P2-1 of the 2026-11-13 audit).

#include "app.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

void App::prepare_model() {
    if (getenv("Q8_PARTS"))
        model.apply_precision(getenv("Q8_LAYERS") ? getenv("Q8_LAYERS") : "", getenv("Q8_PARTS"));
    model.split_ffn(gpu.max_target_dim());
    model.apply_q4_fmt();          // Q4_FMT=rgba4: gate matrices as 4-bit texels
    model.apply_hidden_scale();
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  ffn split\n", run_time_ms());
    if (getenv("Q4_WEIGHTS"))
        model.quantize_q4_weights(getenv("Q4_SCOPE") ? getenv("Q4_SCOPE") : "all");
    if (!opt.tools_json.empty() && model.seq < 768) model.seq = 768;
    if ((opt.worker || !opt.build_cache_prompt.empty()) && model.seq < 1024 && !getenv("SEQ")) model.seq = 1024;
    const int target_limit = gpu.max_target_dim();
    if (model.seq > target_limit) {
        cerr << "SEQ " << model.seq << " exceeds GL target limit " << target_limit
             << "; clamping to " << target_limit << "\n";
        model.seq = target_limit;
    }
    if (model.seq < 16) model.seq = 16;
    if (!system_from_cli && !model.default_system.empty()) system_prompt = model.default_system;
}

bool App::setup_from_model() {
    chat_tmpl.reset();
    opt.qwen_template = opt.minimind3_template = opt.tbb_template = false;
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
        const bool was_qwen = opt.qwen_template, was_mm3 = opt.minimind3_template;
        opt.qwen_template = want_qwen;
        opt.minimind3_template = want_mm3;
        opt.tbb_template = want_tbb;
        if (opt.tbb_template) opt.qwen_template = false;   // its own markers, not chatml
        if (((!was_qwen && opt.qwen_template) || (!was_mm3 && opt.minimind3_template)) && !quiet)
            cerr << "template: " << (model.template_name.empty() ? "kind " + std::to_string(k) : model.template_name)
                 << " (from the checkpoint)\n";
    }
    // Metadata key 9: the model's own Jinja template.  When it is present it
    // is the prompt builder; template_kind stays as the fallback for
    // containers written before the key existed.  A template the vendored
    // engine cannot parse is refused here, not silently swapped for a
    // built-in one.
    if (!model.chat_template.empty()) {
        chat_tmpl.reset(new ChatTemplate(model.chat_template));
        if (!chat_tmpl->ok()) {
            cerr << "chat template rejected: " << chat_tmpl->error() << "\n";
            return false;
        }
        if (!quiet) cerr << "chat template: " << model.chat_template.size()
             << " B from the checkpoint (" << model.chat_templates.size() << " variant(s))\n";
    }
    tokenizer.reset();
    if (model.tok_blob && !getenv("TOKENIZER")) {
        if (!tokenizer.load_memory(model.tok_blob, model.tok_blob_bytes, model.vocab)) return false;
        if (!quiet) cerr << "tokenizer: " << model.tok_blob_bytes << " B embedded in the checkpoint\n";
    } else if (!tokenizer.load(opt.tokenizer_path, model.vocab)) {
        return false;
    }
    tokenizer.set_bos(model.have_specials ? model.id_bos : 1);
    if (model.have_specials && !quiet)
        cerr << "specials: bos " << model.id_bos << " eos " << model.id_eos
             << " stop " << model.id_stop << " (from the checkpoint)\n";
    // The template's `{{ bos_token }}` / `{{ eos_token }}` need the *text* of
    // those ids, not just the id: LFM2.5's template starts with
    // `{{- bos_token -}}`, and handing it "" dropped the BOS from every chat
    // prompt (the tokenizer does not add one, because the template is
    // supposed to).  llama.cpp passes the strings; so must this.
    bos_str = tokenizer.id_text(model.id_bos);
    eos_str = tokenizer.id_text(model.id_eos);
    tokenizer.set_qwen(opt.qwen_template || opt.minimind3_template);
    tokenizer.set_bytelevel(opt.minimind3_template);
    return true;
}

void App::upload_weights() {
    // The rmsnorm epsilon is model state that lives in the GPU (and, per
    // AGENTS.md, changing it changes the emitted shader text).  It used to
    // sit one line above this block, so a model swap kept the previous
    // model's value: invisible when reloading the same file, wrong for any
    // other container.
    gpu.set_rms_eps(model.rms_eps, model.rms_eps_stated);
for (auto& m : model.qkv) gpu.upload(m);
for (auto& m : model.wo) gpu.upload(m);
for (auto& m : model.conv_in_b) if (m.rows) gpu.upload(m);
for (auto& m : model.conv_in_c) if (m.rows) gpu.upload(m);
for (auto& m : model.conv_in_x) if (m.rows) gpu.upload(m);
for (auto& m : model.conv_out) if (m.rows) gpu.upload(m);
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
for (auto& m : model.router) gpu.upload(m);
for (auto& layer : model.moe) {
    for (auto& e : layer) {
        // A MoeExpert holds the whole FFN or its two chunks, never both:
        // the unused Mats are empty and must not reach make_texture.
        Mat* mats[] = {&e.w1, &e.w3, &e.w2, &e.w1a, &e.w1b,
                       &e.w3a, &e.w3b, &e.w2a, &e.w2b};
        for (Mat* m : mats) if (m->rows > 0) gpu.upload(*m);
    }
}
for (auto& m : model.logits_tiles) gpu.upload(m);
model.release_map();
}

bool App::load_prefix_cache_any(const string& path, vector<int>& toks, Session& session) {
    namespace fs = std::filesystem;
    // One place that loads and says so: a tool that checks "did the engine
    // really load this cache" greps for that line, so it must not depend on
    // whether the path was a file or a directory.
    auto load_one = [&](const string& p) -> bool {
        toks.clear();
        if (!session.load_prefix_cache(p, toks)) return false;
        if (!quiet) cerr << "PREFIX_CACHE " << p << " " << toks.size() << " tokens\n";
        return true;
    };
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        vector<string> files;
        for (const auto& e : fs::directory_iterator(path, ec))
            if (e.path().extension() == ".kv") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        for (const string& f : files)
            if (load_one(f)) return true;
        return false;
    }
    return load_one(path);
}
