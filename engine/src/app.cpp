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
    // RWKV-7: qkv[l] is [W_r; W_k; W_v] in one matrix, and the layer loop
    // draws the three bands with three different inputs.  The split has to
    // happen while the weights are still borrowed from the container mapping
    // (release_map() drops it at the end of upload_weights()), so it runs here
    // with the other prepare-time steps and the swap repeats it.
    model.split_rwkv();
    model.apply_q4_fmt();          // Q4_FMT=rgba4: gate matrices as 4-bit texels
    model.apply_hidden_scale();
    if (getenv("SHADER_TIME")) fprintf(stderr, "STARTUP %7.1f ms  ffn split\n", run_time_ms());
    if (getenv("Q4_WEIGHTS"))
        model.quantize_q4_weights(getenv("Q4_SCOPE") ? getenv("Q4_SCOPE") : "all");
    if (!opt.tools_json.empty() && model.seq < 768) model.seq = 768;
    if ((opt.worker || !opt.build_cache_prompt.empty()) && model.seq < 1024 && !getenv("SEQ")) model.seq = 1024;
    // The GL target limit bounds the *attention* score/KV textures, which are
    // seq tall.  RWKV-7 has neither (its state is O(1): H*64*64 per layer plus two
    // shift vectors), so it allocates no seq-sized GL target and must not be
    // clamped here -- on VC4 that would wrongly cap its context at 2048.
    if (!model.is_rwkv()) {
        const int target_limit = gpu.max_target_dim();
        if (model.seq > target_limit) {
            cerr << "SEQ " << model.seq << " exceeds GL target limit " << target_limit
                 << "; clamping to " << target_limit << "\n";
            model.seq = target_limit;
        }
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
    // RWKV-7's World vocabulary is greedy longest match over raw bytes: no BPE
    // merges, no <0xNN> spelling (docs/rwkv7-plan.zh-CN.md, A1-A3).  The mode
    // belongs to the container, so it is set here where the tokenizer is built
    // -- and this function is the one the REPL's /model calls again, which is
    // what makes a swap out of (or into) an RWKV container set it correctly.
    tokenizer.set_rwkv(model.is_rwkv());
    return true;
}

void App::upload_weights() {
    // The rmsnorm epsilon is model state that lives in the GPU (and, per
    // AGENTS.md, changing it changes the emitted shader text).  It used to
    // sit one line above this block, so a model swap kept the previous
    // model's value: invisible when reloading the same file, wrong for any
    // other container.
    gpu.set_rms_eps(model.rms_eps, model.rms_eps_stated);
    // One list of everything that reaches a texture.  It exists so the progress
    // bar can know the total before the first texture does, and so a new tensor
    // group cannot be uploaded but forgotten in the count.  Empty Mats are the
    // precision a chunked or dense model does not use; the MoE branch has always
    // skipped them and now every branch does.
    vector<Mat*> mats;
    auto add_one = [&](Mat& m) { if (m.rows > 0) mats.push_back(&m); };
    auto add = [&](vector<Mat>& v) {
        for (auto& m : v) add_one(m);
    };
    // An RWKV-7 container's qkv is [W_r; W_k; W_v]: rwkv_r/k/v below are its
    // three row bands, so uploading qkv as well would put the same bytes in
    // textures twice.
    if (!model.is_rwkv()) add(model.qkv);
    add(model.rwkv_r);
    add(model.rwkv_k);
    add(model.rwkv_v);
    // The four low-rank pairs (decay, ICLR, value-residual, gate).  Empty Mats
    // are places the container does not use -- layer 0 has no value pair (it
    // records v_first instead) and an ungated model has no gate.
    add(model.rwkv_w1); add(model.rwkv_w2);
    add(model.rwkv_v1); add(model.rwkv_v2);
    add(model.rwkv_a1); add(model.rwkv_a2);
    add(model.rwkv_g1); add(model.rwkv_g2);
    add(model.wo);
    add(model.conv_in_b);
    add(model.conv_in_c);
    add(model.conv_in_x);
    add(model.conv_out);
    add(model.w13);
    add(model.w1);
    add(model.w3);
    add(model.w2);
    add(model.w1a);
    add(model.w1b);
    add(model.w3a);
    add(model.w3b);
    add(model.w2a);
    add(model.w2b);
    add(model.router);
    for (auto& layer : model.moe) {
        for (auto& e : layer) {
            // A MoeExpert holds the whole FFN or its two chunks, never both.
            add_one(e.w1); add_one(e.w3); add_one(e.w2);
            add_one(e.w1a); add_one(e.w3a); add_one(e.w2a);
            add_one(e.w1b); add_one(e.w3b); add_one(e.w2b);
        }
    }
    add(model.logits_tiles);
    uint64_t total = 0;
    for (const Mat* m : mats) total += mat_bytes(*m);
    // With MODEL_BORROW (the default) the weights are still in the checkpoint
    // mapping, so this loop is where the container is actually *read*: the pages
    // fault as their bytes reach a texture.  On the Pi that is 6.12 s of SD card
    // against 2.99 s of texture time, which is why the bar spans this phase.
    g_progress.begin("upload", total);
    uint64_t done = 0;
    for (Mat* m : mats) {
        gpu.upload(*m);
        done += mat_bytes(*m);
        g_progress.set(done);
    }
    g_progress.end();
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
