// The attention block: qkv draws, RoPE, scores, softmax, the KV writes and the
// probes that isolate each of them.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  The GLSL is not
// here -- `shaders_attn.cpp` owns that -- this is the dispatch, and `attn_dup`
// (ATTN_DUP) is the per-draw probe that named the layer's cost when the block was
// being written.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "switches.h"
#include "xformat.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Default-off probe (ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v or all):
// replay one attention pass per layer.  Every shader involved is deterministic
// and writes the same FBO from unchanged inputs, so the second draw reproduces
// the same texels and the output stays byte-identical.  The end-to-end delta is
// the true marginal cost of that pass, unlike the LAYER_PROFILE laps, whose glFinish
// also retires the draw issued before the lap started.
static bool attn_dup(const char* name){
    static const std::string env = getenv("ATTN_DUP") ? getenv("ATTN_DUP") : "";
    if (env.empty()) return false;
    const std::string want(name);
    size_t p = 0;
    while (p <= env.size()) {
        const size_t q = env.find(",", p);
        const std::string tok = env.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok == want || tok == "all") return true;
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return false;
}

void GPU::init_attention(int dim, int kv_dim, int heads, int kv_heads, int head_dim,
                        int seq, bool rope_half, bool qk_norm, int layers,
                        const vector<float>& qn, const vector<float>& kn,
                        const vector<float>& rcos, const vector<float>& rsin, bool qkv_pair,
                        bool hybrid, const vector<uint8_t>* layer_type){
        att_dim_ = dim; att_kv_dim_ = kv_dim; att_heads_ = heads;
        att_kv_heads_ = kv_heads; att_head_dim_ = head_dim; att_seq_ = seq;
        att_rope_half_ = rope_half; att_qk_norm_ = qk_norm;
        att_qkv_pair_ = qkv_pair && ((dim + 2 * kv_dim) % 2 == 0);
        // The KV axis, one value: i24 | q8 | q16 | bf16 | f32.  It replaces
        // KV_PACK (the width) and KV_FMT (the codec), which were the two halves
        // of this axis and whose interaction was undefined: bf16 could be
        // "packed" and "not packed" at once, depending on which one you set
        // last.  kRenames maps both spellings onto the values below.
        // A hybrid model defaults to the packed 16-bit KV: the 24-bit path
        // produces garbage on LFM2.5's shape (kv_dim 512, head_dim 64) while
        // the packed one is correct, and the packed layout is the canonical
        // Pi/Xe setting anyway.
        const char* kv = sw_enum("KV");
        if (!kv) kv = hybrid ? "q16" : "i24";
        att_kv_bits_ = !strcmp(kv, "q8") ? 8 : !strcmp(kv, "q16") ? 16 : 0;
        att_kv_packed_ = att_kv_bits_ != 0;
        att_score_scale_ = (float) env_double("ATT_SCORE_GRID", kAttnScoreScale);
        if (att_score_scale_ < 1.0f) att_score_scale_ = 1.0f;
        att_sum_scale_ = (float) env_double("ATT_SUM_GRID", kAttnSumScale);
        if (att_sum_scale_ < 1.0f) att_sum_scale_ = 1.0f;
        // The sum's domain, not its grid: see AttnConfig::sum_log for why the
        // linear grid stays the default even though log2 is the more accurate
        // one (the two MiniMind3 containers disagree in sign).
        if (const char* sd = getenv("ATT_SUM")) att_sum_log_ = !strcmp(sd, "log2");
        // The weight's exponential: VC4's exp and exp2 are the same fixed-point
        // approximation (4.6e-05 worst relative error against Xe's 2.2e-06), so
        // the portable polynomial exists for the board.  Default stays the
        // driver's -- the pins and the digests belong to it.
        if (const char* ax = getenv("ATT_EXP")) att_exp_poly_ = !strcmp(ax, "poly");
        // bf16 keeps two values a texel (14 in the codec, the scale only in the
        // grid ATT_DUMP prints on); f32 uses the one-value-a-texel layout (the
        // 24-bit path's) with no rounding to flip, which is the only encoding
        // that removes the last-bit disagreement between the two drivers.
        att_kv_fmt_ = !strcmp(kv, "bf16") ? 1 : !strcmp(kv, "f32") ? 2 : 0;
        // bf16 *is* the packed pair codec: two 16-bit halves in one texel, so it
        // goes with the q16 layout (width kv_dim).  Leaving it on the
        // one-value-a-texel layout wrote 24-bit fixed-point and read bf16 back
        // -- MiniMind3 q8 PPL 71840 with KV_FMT=bf16 alone, 0.0000 of that
        // being bf16's 0.1%: the two halves of the axis had to agree, which is
        // what folding KV_PACK and KV_FMT into one value is for.
        if (att_kv_fmt_ == 1) {
                att_kv_bits_ = 16;
                att_kv_packed_ = true;
        } else if (att_kv_fmt_ == 2) {
                att_kv_bits_ = 0;
                att_kv_packed_ = false;
        }
        att_kv_scale_ = (float) env_double("KV_GRID",
                        att_kv_bits_ == 16 ? 2048.0 : att_kv_bits_ == 8 ? 16.0 : 16384.0);
        if (att_kv_scale_ < 1.0f) att_kv_scale_ = 1.0f;
        att_kv_width_ = att_kv_bits_ == 16 ? kv_dim : att_kv_bits_ == 8 ? kv_dim / 2 : 2 * kv_dim;
        if (att_kv_fmt_ == 2) {
                // KVSCALE is only the grid the dump and the checker print on.
                att_kv_scale_ = 512.0f;
                att_kv_width_ = 2 * kv_dim;
        }
        if (att_kv_bits_ != 0 &&
            ((kv_dim % 2) != 0 || (att_kv_bits_ == 8 && ((kv_dim % 4) != 0 || (head_dim % 4) != 0)))) {
            cerr << "KV: width not divisible for the packed layout; falling back to i24\n";
            att_kv_bits_ = 0; att_kv_packed_ = false;
            att_kv_scale_ = 16384.0f; att_kv_width_ = 2 * kv_dim;
        }
        // 8-bit KV: an opt-in per-head, per-position grid for V (max/127),
        // because V head ranges differ by an order of magnitude while K is
        // qk-normed.  Off by default: it buys q4 6% of PPL but costs q8 6%
        // and one extra pass per layer (docs/performance.md).  K always keeps
        // the global KVSCALE grid.
        att_kv_vstep_ = att_kv_bits_ == 8 && env_flag("KV_VSTEP");
        q_norm_tex_.clear(); k_norm_tex_.clear();
        if (qk_norm) {
            for (int l = 0; l < layers; ++l) {
                vector<float> q(qn.begin() + (size_t)l * head_dim, qn.begin() + (size_t)(l + 1) * head_dim);
                vector<float> k(kn.begin() + (size_t)l * head_dim, kn.begin() + (size_t)(l + 1) * head_dim);
                q_norm_tex_.push_back(make_q16_texture(q));
                k_norm_tex_.push_back(make_q16_texture(k));
            }
        }
        const int half = head_dim / 2;
        const int qkv_rows = dim + 2 * kv_dim;
        qkv_target_ = make_target(att_qkv_pair_ ? (qkv_rows + 1) / 2 : qkv_rows, 1);
        att_scores_ = make_target(seq / 2, heads);
        att_stats_ = make_target(heads, 1);
        att_probs_ = make_target(seq, heads);
        att_out_ = make_target(x_texels(dim, act_fmt_), 1);
        // Only attention layers carry a KV cache.  A hybrid model has 8 conv
        // layers whose entries would never be read: at SEQ 2048 their share is
        // 8 x 512 x 2048 x 4 = 33.5 MB of CMA, which is what kept SEQ above
        // 1024 out of reach on the Pi.
        auto is_attn = [&](int l) {
                return !(layer_type && (int)layer_type->size() > l && (*layer_type)[l]);
        };
        kv_cache_.clear();
        kv_cache_.reserve(layers);
        for (int l = 0; l < layers; ++l)
                kv_cache_.push_back(is_attn(l) ? make_target(att_kv_width_, seq) : Target());
        if (att_kv_vstep_) {
                vs_targets_.clear();
                for (int l = 0; l < layers; ++l)
                        vs_targets_.push_back(is_attn(l) ? make_target(att_kv_heads_, seq) : Target());
        }
        const int rw = 2 * half;
        vector<uint8_t> data((size_t)seq * rw * 4);
        for (int p = 0; p < seq; ++p) {
            for (int i = 0; i < half; ++i) {
                int c = (int)lrintf(rcos[(size_t)p * half + i] * 4096.0f) + 8388608;
                int s = (int)lrintf(rsin[(size_t)p * half + i] * 4096.0f) + 8388608;
                for (int k = 0; k < 2; ++k) {
                    int u = k == 0 ? c : s;
                    size_t o = ((size_t)p * rw + 2 * i + k) * 4;
                    data[o + 0] = 0; data[o + 1] = (u >> 16) & 255;
                    data[o + 2] = (u >> 8) & 255; data[o + 3] = u & 255;
                }
            }
        }
        rope_tex_ = make_texture(rw, seq, data.data());
    }

GLuint GPU::attention_block(const Mat& qkv, GLuint x, int layer, int pos){
        if (qkv_target_.tex == 0 || kv_cache_.empty()) return 0;
        // Per-pass attribution: the block is five dependent draws and the whole
        // thing used to be one number in the profile.
        const bool prof = env_set("LAYER_PROFILE");
        auto tp = Clock::now();
        auto lap = [&](const char* name) {
            if (!prof) return;
            glFinish();
            auto now = Clock::now();
            cerr << "PROFATT " << name << " " << elapsed_ms(tp, now) << "\n";
            tp = now;
        };
        // Two qkv rows per fragment: one activation decode feeds two dot
        // products, exactly like the paired logits pass.  The kernel writes the
        // same per-row q16 integers the 24-bit writer produced (x_st2q applies
        // the identical round/clamp), so the rope/qk-norm/qrot readers see the
        // same values.
        auto draw_qkv = [&]{
            if (att_qkv_pair_ && qkv.type == WeightType::INT8) {
                const bool wr = qkv.has_row_scale();
                const int pairs = (qkv.rows + 1) / 2;
                GLuint p = x_prog("xqkvp_" + to_string(qkv.rows) + "_" + to_string(qkv.cols) +
                                  "_" + x_format_name(act_fmt_) + (wr ? "_wr" : ""),
                                  matvec_q8_pair_logits(qkv.rows, qkv.cols, act_fmt_, wr));
                bind_fbo(qkv_target_.fbo);
                glViewport(0, 0, pairs, 1);
                use_program(p);
                bind_tex(0, qkv.tex); set_sampler(p, "W", 0);
                bind_tex(1, x); set_sampler(p, "X", 1);
                glUniform1f(uloc(p, "ws"), qkv.scale);
                if (wr) { bind_tex(2, qkv.row_scale_tex); set_sampler(p, "WS", 2); }
                setup_attrib(p);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            } else {
                matvec_to(qkv, x, qkv_target_);
            }
        };
        draw_qkv();
        if (attn_dup("qkv")) draw_qkv();
        att_finish();
        lap("qkv");
        // QKV_DUMP: exact per-row q16 values of the matvec output, layout-agnostic
        // (QKV_DUMP=1 hashes every layer/position, 2 also prints the first two
        // positions).  This is how a 1-LSB scheduling difference is caught: the
        // row-pair family has produced one before.
        if (getenv("QKV_DUMP") && layer < 8) {
            const char* d = getenv("QKV_DUMP");
            if (atoi(d) >= 2 && pos < 2) {
                const vector<int> v = qkv_values(qkv.rows);
                cerr << "QKVV " << layer << " " << pos;
                for (int x : v) cerr << " " << x;
                cerr << "\n";
            } else {
                cerr << "QKVD " << layer << " " << pos << " " << qkv_hash(qkv.rows) << "\n";
            }
        }
        if (att_kv_vstep_) { kv_vscale(layer, pos); lap("vscale"); }
        kv_rope_write(layer, pos);
        if (attn_dup("rope")) kv_rope_write(layer, pos);
        lap("rope");
        att_scores(layer, pos);
        lap("scores");
        att_stats(pos);
        if (attn_dup("stats")) att_stats(pos);
        lap("stats");
        // q16-packed scores/stats are decoded directly in att_v; the old
        // standalone probability pass is no longer compatible and is retired.
        lap("probs");
        att_v(layer, pos);
        if (attn_dup("v")) att_v(layer, pos);
        if (getenv("ATT_DUMP") && pos == atoi(getenv("ATT_DUMP"))) dump_attention(layer, pos);
        lap("v");
        return att_out_.tex;
    }

void GPU::att_finish() {
    // VC4 needs an explicit finish between FBO -> texture passes.
    layer_finish("NF_ATT");
}

void GPU::kv_vscale(int layer, int pos){
        // 8-bit KV only: one fragment per KV head, one texel per position of
        // this layer.  The texel holds the step (head max / 127) and is read by
        // both the V write (to quantise) and att_v (to dequantise), so the two
        // always agree.  It is per layer: the head max is a property of this
        // layer qkv, and one shared plane would be overwritten by the next
        // layer before the previous layer attention is read back.
        GLuint p = x_vscale();
        bind_fbo(vs_targets_[layer].fbo);
        glViewport(0, pos, att_kv_heads_, 1);
        use_program(p);
        bind_tex(0, qkv_target_.tex);
        set_sampler(p, "QKV", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}

void GPU::att_scores(int layer, int pos){
        // The q16 score-pair layout requires the hoisted q path; ATTN_HOIST=0
        // was removed with the kernel it selected (see att_score).
        const bool sprof = env_set("LAYER_PROFILE");
        auto stp = Clock::now();
        auto slap = [&](const char* n) {
            if (!sprof) return;
            glFinish();
            auto nw = Clock::now();
            cerr << "PROFATT2 " << n << " " << elapsed_ms(stp, nw) << "\n";
            stp = nw;
        };
        // Position-only work first: qk-norm factor per head, then the rotated q.
        if (att_qk_norm_) {
            if (qscale_target_.tex == 0 || qscale_target_.w != att_heads_)
                qscale_target_ = make_target(att_heads_, 1);
            GLuint pq = x_qnorm();
            auto go = [&]{
                bind_fbo(qscale_target_.fbo);
                glViewport(0, 0, att_heads_, 1);
                use_program(pq);
                bind_tex(0, qkv_target_.tex);
                set_sampler(pq, "R", 0);
                setup_attrib(pq);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
                att_finish();
            };
            go();
            if (attn_dup("qknorm")) go();
            slap("qknorm");
        }
        if (qrot_target_.tex == 0 || qrot_target_.w != att_dim_ / 2)
            qrot_target_ = make_target(att_dim_ / 2, 1);
        GLuint pr = x_qrot();
        auto go_qrot = [&]{
            bind_fbo(qrot_target_.fbo);
            glViewport(0, 0, att_dim_ / 2, 1);
            use_program(pr);
            bind_tex(0, qkv_target_.tex);
            set_sampler(pr, "R", 0);
            bind_tex(1, rope_tex_);
            set_sampler(pr, "RT", 1);
            if (att_qk_norm_) {
                bind_tex(2, q_norm_tex_[layer]);
                set_sampler(pr, "QN", 2);
                bind_tex(3, qscale_target_.tex);
                set_sampler(pr, "QS", 3);
            }
            glUniform1f(uloc(pr, "pos"), (float)pos);
            setup_attrib(pr);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            att_finish();
        };
        go_qrot();
        if (attn_dup("qrot")) go_qrot();
        slap("qrot");

        GLuint p = x_score();
        auto go_score = [&]{
            bind_fbo(att_scores_.fbo);
            glViewport(0, 0, att_seq_ / 2, att_heads_);
            use_program(p);
            bind_tex(0, qrot_target_.tex);
            set_sampler(p, "Q", 0);
            bind_tex(1, kv_cache_[layer].tex);
            set_sampler(p, "KV", 1);
            glUniform1f(uloc(p, "pos"), (float)pos);
            glUniform1f(uloc(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            att_finish();
        };
        go_score();
        if (attn_dup("score")) go_score();
        slap("score");
    }

void GPU::att_stats(int pos){
        GLuint p = x_stats();
        bind_fbo(att_stats_.fbo);
        glViewport(0, 0, att_heads_, 1);
        use_program(p);
        bind_tex(0, att_scores_.tex);
        set_sampler(p, "S", 0);
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }

void GPU::att_probs(int pos){
        GLuint p = x_prob();
        bind_fbo(att_probs_.fbo);
        glViewport(0, 0, att_seq_ / 2, att_heads_);
        use_program(p);
        bind_tex(0, att_scores_.tex);
        set_sampler(p, "S", 0);
        bind_tex(1, att_stats_.tex);
        set_sampler(p, "ST", 1);
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }

void GPU::att_v(int layer, int pos){
        // q16 score/stats layout has one packed texel per head; fused decode is mandatory.
        static const bool fuse_prob = true;
        GLuint p = x_v();
        bind_fbo(att_out_.fbo);
        glViewport(0, 0, att_out_.w, 1);
        use_program(p);
        if (fuse_prob) {
            bind_tex(0, att_scores_.tex);
            set_sampler(p, "S", 0);
            bind_tex(2, att_stats_.tex);
            set_sampler(p, "ST", 2);
        } else {
            bind_tex(0, att_probs_.tex);
            set_sampler(p, "P", 0);
        }
        bind_tex(1, kv_cache_[layer].tex);
        set_sampler(p, "KV", 1);
        if (att_kv_vstep_) {
            bind_tex(3, vs_targets_[layer].tex);
            set_sampler(p, "VS", 3);
        }
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }


void GPU::kv_rope_write(int layer, int pos){
        GLuint p = x_kv_rope();
        bind_fbo(kv_cache_[layer].fbo);
        glViewport(0, pos, att_kv_width_, 1);
        use_program(p);
        bind_tex(0, qkv_target_.tex);
        set_sampler(p, "QKV", 0);
        bind_tex(1, rope_tex_);
        set_sampler(p, "RT", 1);
        if (att_qk_norm_) {
            bind_tex(2, k_norm_tex_[layer]);
            set_sampler(p, "KN", 2);
        }
        if (att_kv_vstep_) {
            bind_tex(3, vs_targets_[layer].tex);
            set_sampler(p, "VS", 3);
        }
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }
