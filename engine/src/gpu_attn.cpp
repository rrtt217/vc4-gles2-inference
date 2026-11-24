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


// Long context: bind the segment chain.  `nseg == 1` binds the caller's own
// texture under its own name, which is the shipped call.  With more, sampler
// `S` gets segment 0 and `S1..S{nseg-1}` the rest, in that order from
// `base_unit` -- the order the builder's uniform list declares them in, because
// set_sampler takes the name and the unit separately and a mismatch binds a
// segment to the wrong name silently.
void GPU::bind_seg(int base_unit, const char* base, const vector<GLuint>& tex, GLuint prog) {
        for (int k = 0; k < att_nseg_ && base_unit + k < 8; ++k) {
                bind_tex(base_unit + k, tex[(size_t)k]);
                const string name = k == 0 ? string(base) : string(base) + to_string(k);
                set_sampler(prog, name.c_str(), base_unit + k);
        }
}

void GPU::att_geometry_line() const {
        fprintf(stderr, "attn: dim %d seq %d heads %d head_dim %d nseg %d seg %d "
                        "kv_width %d kv_bits %d kv_fmt %d qkv_pair %d qkv_pair_ok %d rope_none %d causal %d\n",
                att_dim_, att_seq_, att_heads_, att_head_dim_, att_nseg_, att_seg_,
                kv_.texels, kv_.bits, (int)kv_.codec,
                (int)att_qkv_pair_, (int)att_qkv_pair_,
                (int)att_rope_none_, (int)att_causal_);
}

void GPU::init_attention(int dim, int kv_dim, int heads, int kv_heads, int head_dim,
                        int seq, bool rope_half, bool qk_norm, int layers,
                        const vector<float>& qn, const vector<float>& kn,
                        const vector<float>& rcos, const vector<float>& rsin, bool qkv_pair,
                        bool hybrid, const vector<uint8_t>* layer_type, bool rope_none){
        // v5: no rotation at all.  The causal/rope flags are AttnConfig, so the
        // builders emit NOROPE only for this caller.
        att_rope_none_ = rope_none;
        att_dim_ = dim; att_kv_dim_ = kv_dim; att_heads_ = heads;
        att_kv_heads_ = kv_heads; att_head_dim_ = head_dim; att_seq_ = seq;
        att_rope_half_ = rope_half; att_qk_norm_ = qk_norm;
        att_qkv_pair_ = qkv_pair && ((dim + 2 * kv_dim) % 2 == 0);
        // The KV axis, one value: i24 | q8 | q16 | bf16 | f32.  It replaces
        // KV_PACK (the width) and KV_FMT (the codec), whose interaction was
        // undefined.  The layout built here is the one place the codec, the
        // values a texel, the row width and the grid come from; the shaders
        // read it through AttnConfig and the cache flags through kv_.
        // A hybrid model defaults to the packed 16-bit KV: the canonical
        // Pi/Xe setting, and the one LFM2.5's shape was validated on.
        const char* kv = sw_enum("KV");
        if (!kv) kv = hybrid ? "q16" : "i24";
        KvLayout L;
        if (!strcmp(kv, "bf16")) { L.codec = KvCodec::Bf16; L.bits = 16; }
        else if (!strcmp(kv, "f32")) { L.codec = KvCodec::F32; L.bits = 0; }
        else if (!strcmp(kv, "q16")) { L.bits = 16; }
        else if (!strcmp(kv, "q8")) { L.bits = 8; }
        L.vpt = L.bits == 16 ? 2 : L.bits == 8 ? 4 : 1;
        L.texels = L.bits == 16 ? kv_dim : L.bits == 8 ? kv_dim / 2 : 2 * kv_dim;
        att_score_scale_ = (float) env_double("ATT_SCORE_GRID", kAttnScoreScale);
        if (att_score_scale_ < 1.0f) att_score_scale_ = 1.0f;
        att_sum_scale_ = (float) env_double("ATT_SUM_GRID", kAttnSumScale);
        if (att_sum_scale_ < 1.0f) att_sum_scale_ = 1.0f;
        // The sum's domain, not its grid: see AttnConfig::sum_log for why the
        // linear grid stays the default even though log2 is the more accurate
        // one (the two MiniMind3 containers disagree in sign).
        if (const char* sd = getenv("ATT_SUM")) att_sum_log_ = !strcmp(sd, "log2");
        L.scale = (float) env_double("KV_GRID",
                        L.bits == 16 ? 2048.0 : L.bits == 8 ? 16.0 : 16384.0);
        if (L.scale < 1.0f) L.scale = 1.0f;
        if (L.codec == KvCodec::F32) {
                // KVSCALE is only the grid the dump and the checker print on.
                L.scale = 512.0f;
                L.texels = 2 * kv_dim;
        }
        if (L.bits != 0 &&
            ((kv_dim % 2) != 0 || (L.bits == 8 && ((kv_dim % 4) != 0 || (head_dim % 4) != 0)))) {
            cerr << "KV: width not divisible for the packed layout; falling back to i24\n";
            L = KvLayout();
        }
        // 8-bit KV: an opt-in per-head, per-position grid for V (max/127),
        // because V head ranges differ by an order of magnitude while K is
        // qk-normed.  Off by default: it buys q4 6% of PPL but costs q8 6%
        // and one extra pass per layer (docs/performance.md).  K always keeps
        // the global KVSCALE grid.
        L.vstep = L.bits == 8 && env_flag("KV_VSTEP");
        kv_ = L;
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
        // Long context: how many seq-sized textures the driver's limit forces.
        // `seg` is the positions a texture holds, `nseg` how many textures the
        // cache is cut into; nseg == 1 is the shipped layout and every kernel
        // emits its text unchanged (gate/shader_hashes.sh proves it).
        const int lim = max_target_dim() > 0 ? max_target_dim() : 2048;
        // SEG_CAP forces a smaller segment than the driver needs, so the split
        // can be exercised -- and diffed against the single-texture path -- on a
        // machine whose limit is larger than the window.  It never enlarges.
        const int cap = (int) env_int("SEG_CAP", 0);
        const int seg_lim = (cap > 0 && cap < lim) ? cap : lim;
        // A slice must hold an even number of positions.  The score grid packs
        // two positions a texel, so an odd slice length puts its last pair half
        // in this slice and half in the next -- the fragment then scores a
        // position that belongs to the following slice, and the stats and value
        // scans see it in both.  Rounding the length down keeps every pair
        // inside one slice (and costs at most one position a slice).
        int seg = std::min(seq, seg_lim);
        if (seg < seq && (seg & 1)) --seg;
        if (seg < 2) seg = 2;
        att_seg_ = seg;
        att_nseg_ = (seq + att_seg_ - 1) / att_seg_;
        // A texture holds nseg positions per row; the score grid packs two
        // positions a texel, so its width is seg/2 and its capacity is 2*seg
        // positions a row.  Both are bounded by the driver's limit.
        const int score_w = (att_seg_ + 1) / 2;
        if (score_w > lim) {
                cerr << "SEQ " << seq << ": a " << att_nseg_ << "-segment score grid is "
                     << score_w << " texels wide, over the GL target limit " << lim << "\n";
                exit(1);
        }
        if (getenv("SEG_DUMP"))
                fprintf(stderr, "SEGGEO seq=%d lim=%d cap=%d seg=%d nseg=%d score_w=%d\n",
                        seq, lim, cap, att_seg_, att_nseg_, score_w);
        score_parts_.clear();
        for (int k = 0; k < att_nseg_; ++k) score_parts_.push_back(make_target(score_w, heads));
        att_scores_ = score_parts_[0];
        att_stats_ = make_target(heads, 1);
        // att_probs_ is the unfused probability texture, and the fused q16 path
        // has not written or read it since the probability pass was retired into
        // att_v.  Allocating it `seq` tall was the one target that would have
        // kept SEQ clamped to the driver limit after the split, so it is sized
        // to the fused layout's own width (one texel a head) and stays unused.
        att_probs_ = make_target(heads, 1);
        att_out_ = make_target(x_texels(dim, act_fmt_), 1);
        // Only attention layers carry a KV cache.  A hybrid model has 8 conv
        // layers whose entries would never be read: at SEQ 2048 their share is
        // 8 x 512 x 2048 x 4 = 33.5 MB of CMA, which is what kept SEQ above
        // 1024 out of reach on the Pi.
        auto is_attn = [&](int l) {
                return !(layer_type && (int)layer_type->size() > l && (*layer_type)[l]);
        };
        kv_cache_.clear();
        kv_parts_.clear();
        vs_targets_.clear();
        vs_parts_.clear();
        kv_cache_.reserve(layers);
        kv_parts_.reserve((size_t)layers * att_nseg_);
        vs_targets_.reserve(layers);
        vs_parts_.reserve((size_t)layers * att_nseg_);
        for (int l = 0; l < layers; ++l) {
                const bool a = is_attn(l);
                Target first;
                for (int k = 0; k < att_nseg_; ++k) {
                        Target t = a ? make_target(kv_.texels, att_seg_) : Target();
                        if (k == 0) first = t;
                        kv_parts_.push_back(t);
                }
                kv_cache_.push_back(first);
                if (kv_.vstep) {
                        Target vfirst;
                        for (int k = 0; k < att_nseg_; ++k) {
                                Target t = a ? make_target(att_kv_heads_, att_seg_) : Target();
                                if (k == 0) vfirst = t;
                                vs_parts_.push_back(t);
                        }
                        vs_targets_.push_back(vfirst);
                }
        }
        // The rope table is one texture a segment: a position's (cos, sin) pair
        // for each rope pair, at x = pair*2, y = the position inside its segment.
        rope_parts_.clear();
        const int rw = 2 * half;
        for (int k = 0; k < att_nseg_; ++k) {
            const int p0 = k * att_seg_;
            const int rows = std::min(att_seg_, seq - p0);
            vector<uint8_t> data((size_t)rows * rw * 4);
            for (int p = 0; p < rows; ++p) {
                for (int i = 0; i < half; ++i) {
                    int c = (int)lrintf(rcos[(size_t)(p0 + p) * half + i] * 4096.0f) + 8388608;
                    int s = (int)lrintf(rsin[(size_t)(p0 + p) * half + i] * 4096.0f) + 8388608;
                    for (int kk = 0; kk < 2; ++kk) {
                        int u = kk == 0 ? c : s;
                        size_t o = ((size_t)p * rw + 2 * i + kk) * 4;
                        data[o + 0] = 0; data[o + 1] = (u >> 16) & 255;
                        data[o + 2] = (u >> 8) & 255; data[o + 3] = u & 255;
                    }
                }
            }
            if (getenv("SEG_DUMP"))
                fprintf(stderr, "SEGROPE k=%d rows=%d rw=%d first=%d last=%d\n", k, rows, rw,
                        data.empty() ? -1 : (int)data[1], (int)data[(size_t)(rows - 1) * rw * 4 + 1]);
            rope_parts_.push_back(make_texture(rw, rows, data.data()));
        }
        rope_tex_ = rope_parts_[0];

    }

void GPU::att_qkv_draw(const Mat& qkv, GLuint x){
        if (att_qkv_pair_ && wf_can(qkv.type, WStage::QkvPair)) {
            const bool wr = qkv.has_row_scale();
            const int pairs = (qkv.rows + 1) / 2;
            // A t3 qkv holds both rows of the pair in one texel, so its
            // kernel fetches once where the q8 one fetches twice; both write
            // x_st2q, which is why the rope/qk-norm readers do not care.
            GLuint p = qkv.type == WeightType::T3
                           ? x_t3_pair(qkv, wr)
                           : x_prog("xqkvp_" + to_string(qkv.rows) + "_" +
                                    to_string(qkv.cols) + "_" + x_format_name(act_fmt_) +
                                    (wr ? "_wr" : ""),
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
        // v5: a block whose projection carries a bias (Marian biases q/k/v
        // separately).  The add writes into a second target and the two swap, so
        // kv_rope_write and att_qrot_stage keep reading `qkv_target_` and none of
        // their text moves -- the same "flag, not a new kernel" rule the causal
        // and rope switches follow.
        if (att_qkv_bias_) {
                bias_add_wide(qkv_target_.tex, att_qkv_bias_, qkv_biased_, 3 * att_dim_);
                std::swap(qkv_target_, qkv_biased_);
        }
}

GLuint GPU::attention_block(const Mat& qkv, GLuint x, int layer, int pos,
                            GLuint kv, bool stage_kv){
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
        // A replay's delta is a wall delta; DUP_PROFILE=1 answers whether the
        // replay cost QPU cycles or a finish.  Same columns as the PROF stage
        // line (see session.cpp).
        static const bool dup_prof = env_set("DUP_PROFILE");
        const bool dup_c = dup_prof && pc_available();
        auto dup_price = [&](const char* name, auto&& fn) {
            if (!dup_prof) { fn(); return; }
            unsigned long long idle = 0, frag = 0, exec = 0, quads = 0;
            const auto t0 = Clock::now();
            sync();
            if (dup_c) pc_stage_begin();
            fn();
            sync();
            const double ms = elapsed_ms(t0, Clock::now());
            const bool have = dup_c && pc_stage_end(idle, frag, exec, quads);
            if (have) {
                const double to_ms = 1.0 / (GPU::pc_qpus() * (double)GPU::pc_mhz() * 1000.0);
                cerr << "DUPA " << name << " wall=" << ms
                     << " gpu=" << ((double)frag * to_ms)
                     << " exec=" << ((double)exec * to_ms) << "\n";
            } else {
                cerr << "DUPA " << name << " wall=" << ms << "\n";
            }
        };
        // Two qkv rows per fragment: one activation decode feeds two dot
        // products, exactly like the paired logits pass.  The kernel writes the
        // same per-row q16 integers the 24-bit writer produced (x_st2q applies
        // the identical round/clamp), so the rope/qk-norm/qrot readers see the
        // same values.
        att_qkv_draw(qkv, x);
        if (attn_dup("qkv")) dup_price("qkv", [&]{ att_qkv_draw(qkv, x); });
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
        if (stage_kv && kv_.vstep) { kv_vscale(layer, pos); lap("vscale"); }
        if (stage_kv) kv_rope_write(layer, pos);
        if (stage_kv && attn_dup("rope"))
            dup_price("rope", [&]{ kv_rope_write(layer, pos); });
        lap("rope");
        att_scores(layer, pos, kv);
        lap("scores");
        att_stats(pos);
        if (attn_dup("stats")) dup_price("stats", [&]{ att_stats(pos); });
        lap("stats");
        // q16-packed scores/stats are decoded directly in att_v; the old
        // standalone probability pass is no longer compatible and is retired.
        lap("probs");
        att_v(layer, pos, kv);
        if (attn_dup("v")) dup_price("v", [&]{ att_v(layer, pos, kv); });
        if (getenv("ATT_DUMP") && pos == atoi(getenv("ATT_DUMP"))) dump_attention(layer, pos, kv);
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
        // This kernel reads one sampler, QKV, and nothing else: the rope table
        // was never one of its inputs, so asking for RT here only produced a
        // "uniform not found" line on the q8 KV step path.
        const int u_qkv = uloc(p, "QKV");
        if (u_qkv != 0) forget_state();
        bind_tex(u_qkv < 0 ? 0 : u_qkv, qkv_target_.tex);
        set_sampler(p, "QKV", u_qkv < 0 ? 0 : u_qkv);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}

void GPU::att_scores(int layer, int pos, GLuint kv){
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
        // The rope table is segmented with the KV cache, so the rotated-Q pass
        // reads the plane for *this* position's segment.  A one-segment session
        // is segment 0 and the shipped text, so this is the identical call.
        const int qseg = seg_of(pos);
        GLuint pr = x_qrot(qseg);
        auto go_qrot = [&]{
            bind_fbo(qrot_target_.fbo);
            glViewport(0, 0, att_dim_ / 2, 1);
            use_program(pr);
            // NOROPE: the compiler drops RT and pos, so the host does not ask
            // for them (a "uniform not found" here reads like a typo).
            const int u_r = uloc(pr, "R");
            const int u_rt = att_rope_none_ ? -1 : uloc(pr, "RT");
            if (u_r != 0 || u_rt != 1) forget_state();
            bind_tex(u_r < 0 ? 0 : u_r, qkv_target_.tex);
            set_sampler(pr, "R", u_r < 0 ? 0 : u_r);
            if (!att_rope_none_) {
                bind_tex(u_rt < 0 ? 1 : u_rt, rope_parts_[qseg]);
                set_sampler(pr, "RT", u_rt < 0 ? 1 : u_rt);
            }
            if (att_qk_norm_) {
                const int u_qn = uloc(pr, "QN"), u_qs = uloc(pr, "QS");
                bind_tex(u_qn < 0 ? 2 : u_qn, q_norm_tex_[layer]);
                set_sampler(pr, "QN", u_qn < 0 ? 2 : u_qn);
                bind_tex(u_qs < 0 ? 3 : u_qs, qscale_target_.tex);
                set_sampler(pr, "QS", u_qs < 0 ? 3 : u_qs);
            }
            if (!att_rope_none_) glUniform1f(uloc(pr, "pos"), (float)pos);
            setup_attrib(pr);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            att_finish();
        };
        go_qrot();
        if (attn_dup("qrot")) go_qrot();
        slap("qrot");

        auto go_score = [&]{
            // Long context: one program and one draw a segment.  Segment k's
            // grid is `seg/2` wide and its fragments are the global positions
            // [k*seg, (k+1)*seg); the shader's own SEGK/SEG put the causal test
            // and the fetch on the global index, so the union of the draws is
            // the same work the single draw did.
            for (int k = 0; k < att_nseg_; ++k) {
                GLuint p = x_score(k);
                // The last slice can be shorter than `seg` when the window does
                // not divide evenly.  Drawing its full width would score pairs
                // that hold no position, which then feed the stats scan and the
                // value pass as if they were real.
                const int rows = std::min(att_seg_, att_seq_ - k * att_seg_);
                bind_fbo(score_parts_[k].fbo);
                glViewport(0, 0, (rows + 1) / 2, att_heads_);
                use_program(p);
                bind_tex(0, qrot_target_.tex);
                set_sampler(p, "Q", 0);
                bind_tex(1, (kv && k == 0) ? kv : kv_part(layer, k).tex);
                set_sampler(p, "KV", 1);
                glUniform1f(uloc(p, "pos"), (float)pos);
                glUniform1f(uloc(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
                setup_attrib(p);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            }
            att_finish();
        };
        go_score();
        if (attn_dup("score")) go_score();
        slap("score");
    }

void GPU::att_stats(int pos){
        if (att_nseg_ == 1) {
                // The shipped form, unchanged: one draw into att_stats_ itself.
                GLuint p0 = x_stats();
                bind_fbo(att_stats_.fbo);
                glViewport(0, 0, att_heads_, 1);
                use_program(p0);
                bind_tex(0, att_scores_.tex);
                set_sampler(p0, "S", 0);
                glUniform1f(uloc(p0, "pos"), (float)pos);
                setup_attrib(p0);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
                att_finish();
                return;
        }
        // Two passes a slice.  The first writes (own max, sum-on-own-max) and
        // the fold takes the max of maxima; the second re-sums every slice
        // against that global max, so the fold's remaining job is a plain add --
        // one rounding on the 1/32 grid, which is what the single-draw kernel
        // does.  A per-slice rescale instead rounds twice and lands an LSB away,
        // and that LSB reaches att_v and every layer after it.
        if (stats_gmax_.tex == 0 || stats_gmax_.w != att_heads_)
                stats_gmax_ = make_target(att_heads_, 1);
        auto pass = [&](bool use_gmax) {
            for (int k = 0; k < att_nseg_; ++k) {
                if (k >= (int)stats_parts_.size()) stats_parts_.resize(att_nseg_);
                if (stats_parts_[k].tex == 0 || stats_parts_[k].w != att_heads_)
                    stats_parts_[k] = make_target(att_heads_, 1);
                GLuint p = x_stats(k);
                bind_fbo(stats_parts_[k].fbo);
                glViewport(0, 0, att_heads_, 1);
                use_program(p);
                bind_tex(0, score_parts_[k].tex);
                set_sampler(p, "S", 0);
                glUniform1f(uloc(p, "use_gmax"), use_gmax ? 1.0f : 0.0f);
                if (use_gmax) {
                        bind_tex(1, stats_gmax_.tex);
                        set_sampler(p, "GM", 1);
                }
                glUniform1f(uloc(p, "pos"), (float)pos);
                setup_attrib(p);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            }
            att_finish();
        };
        pass(false);
        att_stats_fold(false);        // the per-head max, into stats_gmax_
        pass(true);
        att_stats_fold(true);         // the plain add, into att_stats_
    }

// Long context: fold the per-segment (max, sum) pairs into one.  The max is a
// max of maxima; the sum is recomputed as sum_k exp(max_k - max_global) * sum_k,
// which is exact for the same reason the softmax is: the exponentials differ by
// a constant factor that the max has already normalised out.  Reading the
// segment sums back as texels and doing the arithmetic on the host keeps this a
// handful of bytes a layer rather than another shader -- and a shader would have
// to be a second reduction over N samplers anyway.
void GPU::att_stats_fold(bool final){
        const int h = att_heads_;
        vector<float> mx((size_t)att_nseg_ * h), sm((size_t)att_nseg_ * h);
        for (int k = 0; k < att_nseg_; ++k) {
                const vector<uint8_t> b = gpu_read_target_row(stats_parts_[k], 0);
                for (int i = 0; i < h; ++i) {
                        const size_t o = (size_t)i * 4;
                        auto dec = [&](int comp) {
                                const int v = (int)b[o + (size_t)comp * 2] * 256 +
                                              b[o + (size_t)comp * 2 + 1];
                                return (float)(v >= 32768 ? v - 65536 : v);
                        };
                        mx[(size_t)k * h + i] = dec(0) / att_score_scale_;
                        sm[(size_t)k * h + i] = dec(1) / att_sum_scale_;
                }
        }
        if (!final) {
                // Pass 1: the global max per head, on the I24_1 layout xformat.h
                // defines -- offset-binary int24 in .gba with a zero top byte,
                // big-endian, which is what x_q24raw decodes (g*65536+b*256+a
                // -2^23).  Writing it as r*2^24+... reads back as -2^23 and
                // makes every sum 0.
                vector<uint8_t> g((size_t)h * 4, 0);
                for (int i = 0; i < h; ++i) {
                        float m = -1e30f;
                        for (int k = 0; k < att_nseg_; ++k)
                                m = std::max(m, mx[(size_t)k * h + i]);
                        long q = lrintf(m * att_score_scale_);
                        q = std::max(-8388608L, std::min(8388607L, q)) + 8388608L;
                        g[(size_t)i * 4 + 0] = 0;
                        g[(size_t)i * 4 + 1] = (uint8_t)((q >> 16) & 0xFF);
                        g[(size_t)i * 4 + 2] = (uint8_t)((q >> 8) & 0xFF);
                        g[(size_t)i * 4 + 3] = (uint8_t)(q & 0xFF);
                }
                glBindTexture(GL_TEXTURE_2D, stats_gmax_.tex);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, g.data());
                forget_state();
                return;
        }
        vector<uint8_t> out((size_t)h * 4);
        for (int i = 0; i < h; ++i) {
                float g = -1e30f;
                for (int k = 0; k < att_nseg_; ++k) g = std::max(g, mx[(size_t)k * h + i]);
                // Pass 2: every slice already summed against `g`, so this is a
                // plain add of values on one grid.
                double sum = 0.0;
                for (int k = 0; k < att_nseg_; ++k) sum += (double)sm[(size_t)k * h + i];
                auto enc = [&](int comp, float v, float scale) {
                        int q = (int)lrintf(v * scale);
                        if (q < -32768) q = -32768;
                        if (q > 32767) q = 32767;
                        if (q < 0) q += 65536;
                        out[(size_t)i * 4 + (size_t)comp * 2] = (uint8_t)(q >> 8);
                        out[(size_t)i * 4 + (size_t)comp * 2 + 1] = (uint8_t)(q & 255);
                };
                enc(0, g, att_score_scale_);
                enc(1, (float)sum, att_sum_scale_);
        }
        glBindTexture(GL_TEXTURE_2D, att_stats_.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
        forget_state();
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

void GPU::att_v(int layer, int pos, GLuint kv){
        // q16 score/stats layout has one packed texel per head; fused decode is mandatory.
        if (att_nseg_ == 1) {
                GLuint p0 = x_v();
                bind_fbo(att_out_.fbo);
                glViewport(0, 0, att_out_.w, 1);
                use_program(p0);
                bind_tex(0, att_scores_.tex);
                set_sampler(p0, "S", 0);
                bind_tex(2, att_stats_.tex);
                set_sampler(p0, "ST", 2);
                bind_tex(1, kv ? kv : kv_cache_[layer].tex);
                set_sampler(p0, "KV", 1);
                if (kv_.vstep) {
                        bind_tex(3, vs_targets_[layer].tex);
                        set_sampler(p0, "VS", 3);
                }
                glUniform1f(uloc(p0, "pos"), (float)pos);
                setup_attrib(p0);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
                att_finish();
                return;
        }
        // Long context: the V accumulation is a sum over positions, so a slice
        // can only produce a partial.  Each slice writes its OWN target -- a
        // texture cannot be sampled while it is the render target -- and
        // att_v_fold() sums them with the format's own X1/XST.
        for (int k = 0; k < att_nseg_; ++k) {
            if (k >= (int)v_parts_.size()) v_parts_.resize(att_nseg_);
            if (v_parts_[k].tex == 0 || v_parts_[k].w != att_out_.w)
                v_parts_[k] = make_target(att_out_.w, 1);
            GLuint p = x_v(k, false);
            bind_fbo(v_parts_[k].fbo);
            glViewport(0, 0, att_out_.w, 1);
            use_program(p);
            bind_tex(0, score_parts_[k].tex);
            set_sampler(p, "S", 0);
            bind_tex(2, att_stats_.tex);
            set_sampler(p, "ST", 2);
            bind_tex(1, (kv && k == 0) ? kv : kv_part(layer, k).tex);
            set_sampler(p, "KV", 1);
            if (kv_.vstep) {
                bind_tex(3, vs_part(layer, k).tex);
                set_sampler(p, "VS", 3);
            }
            glUniform1f(uloc(p, "pos"), (float)pos);
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        att_finish();
        att_v_fold();
    }

// Long context: sum the per-slice value partials into att_out_.  A GPU pass, not
// a host decode: the packed fixed-point layouts are the format's business, and
// re-implementing I24_1 in C++ is how its encoder drifted once already (the
// value lives in .gba with a zero top byte; reading .rgba as a 24-bit number is
// a different format).
void GPU::att_v_fold(){
        if (att_nseg_ <= 1) { att_out_ = v_parts_[0]; return; }
        // Every slice has to be folded in.  A slice left unbound reads unit 0,
        // which sums the first slice again and silently loses the rest --
        // measured at nine slices before this was a loop: PPL 59.0 against 1.2
        // for the same text at one slice.
        //
        // The fold's program carries one sampler a slice, so it can only cover
        // as many slices as there are texture units.  When there are more, the
        // fold runs in rounds and each round adds into att_out_ (the second and
        // later rounds use the accumulate form of the kernel).
        // A round folds at most `units` slices when it only reads, and one fewer
        // when it also adds, because ACC needs a unit of its own: binding ACC to
        // unit `n` with n == units is off the end of 0..units-1 and the driver
        // then reads unit 0, i.e. the first slice again.
        // Eight, not the queried GL_MAX_TEXTURE_IMAGE_UNITS: Xe answers 32, but
        // this engine's own bind path (bind_tex/set_sampler) addresses units
        // 0..7 and the kernels are written for that ceiling.  Using the queried
        // number asked one draw to bind sixteen samplers, most of which never
        // reached the program.
        const int units = std::min(8, max_texture_units());
        const int per_round = std::max(1, units - 1);
        if (att_nseg_ > per_round) {
                if (v_fold_tmp_.tex == 0 || v_fold_tmp_.w != att_out_.w)
                        v_fold_tmp_ = make_target(att_out_.w, 1);
                if (v_fold_tmp2_.tex == 0 || v_fold_tmp2_.w != att_out_.w)
                        v_fold_tmp2_ = make_target(att_out_.w, 1);
        }
        int round = 0;
        for (int base = 0; base < att_nseg_; base += per_round, ++round) {
                const int n = std::min(per_round, att_nseg_ - base);
                const bool add = round > 0;
                GLuint fp = x_v_fold(n, add);
                const bool last = (base + n >= att_nseg_);
                Target& dst = last ? att_out_
                                   : ((round & 1) ? v_fold_tmp2_ : v_fold_tmp_);
                const Target& src = (round & 1) ? v_fold_tmp_ : v_fold_tmp2_;
                bind_fbo(dst.fbo);
                glViewport(0, 0, att_out_.w, 1);
                use_program(fp);
                forget_state();
                for (int k = 0; k < n; ++k) {
                        const string nm = "V" + std::to_string(k);
                        bind_tex(k, v_parts_[base + k].tex);
                        set_sampler(fp, nm.c_str(), k);
                }
                if (add) {
                        bind_tex(n, src.tex);
                        set_sampler(fp, "ACC", n);
                }
                glUniform1f(uloc(fp, "VW"), (float)att_out_.w);
                setup_attrib(fp);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
                att_finish();
        }
}

void GPU::kv_rope_write(int layer, int pos){
        // Long context: the row is written into the segment that owns `pos`, at
        // its offset inside that segment.  The rope table and the V step plane
        // carry the same cut, so every sampler this kernel reads is bound for
        // the segment it is writing.
        const int sg = seg_of(pos);
        // The program is the one for *this* segment: SEGK is a compile-time
        // constant in its text, so the rope table's y and the V step plane's y
        // are both `pos - SEGK*SEG`.  Asking for segment 0's program while
        // writing segment k's row is how every position past the first segment
        // read the wrong rope pair (a silent, plausible-looking wrong number).
        GLuint p = x_kv_rope(sg);
        bind_fbo(kv_part(layer, sg).fbo);
        // The viewport's y is where the row goes, and the kernel's rope/V-step
        // fetch must read the *same* row of its segment-sized plane.  A
        // one-segment session has `seg_off(pos) == pos` and the shipped shader
        // text, so this is the identical call; a split session writes into the
        // segment's own texture at the offset, and SEGK in the text makes the
        // rope table agree.
        glViewport(0, seg_off(pos), kv_.texels, 1);
        use_program(p);
        // Bind every sampler to the unit the *program* declares.  The shipped
        // text puts QKV on unit 0 and RT on 1; a segmented text declares its
        // planes first, so the driver numbers QKV 1 and RT 0.  bind_tex caches
        // by unit and set_sampler by (program, name), so a different numbering
        // has to forget the unit cache or the bind is skipped as "already done"
        // while the unit really holds the previous program's texture.
        // Ask only for the uniforms this build actually emits: uloc() prints a
        // "not found" line for a name the compiler dropped, and KN/VS are
        // inside `if(QKNORM==1)` / `if(KVSTEP==1)` in the kernel, so a q16
        // container legitimately has no VS at all.
        const int u_qkv = uloc(p, "QKV");
        // With NOROPE the rotation is dead code to the compiler, so RT (and the
        // write kernel's pos) are pruned out of the program: asking for them
        // prints "uniform not found", which for a *deliberately* removed fetch
        // reads like a typo.  The host knows, so it does not ask.
        const int u_rt = att_rope_none_ ? -1 : uloc(p, "RT");
        if (u_qkv != 0 || u_rt != 1) forget_state();
        bind_tex(u_qkv < 0 ? 0 : u_qkv, qkv_target_.tex);
        set_sampler(p, "QKV", u_qkv < 0 ? 0 : u_qkv);
        if (!att_rope_none_) {
                bind_tex(u_rt < 0 ? 1 : u_rt, rope_parts_[sg]);
                set_sampler(p, "RT", u_rt < 0 ? 1 : u_rt);
        }
        if (att_qk_norm_) {
            const int u_kn = uloc(p, "KN");
            bind_tex(u_kn < 0 ? 2 : u_kn, k_norm_tex_[layer]);
            set_sampler(p, "KN", u_kn < 0 ? 2 : u_kn);
        }
        if (kv_.vstep) {
            const int u_vs = uloc(p, "VS");
            bind_tex(u_vs < 0 ? 3 : u_vs, vs_part(layer, sg).tex);
            set_sampler(p, "VS", u_vs < 0 ? 3 : u_vs);
        }
        // The write kernel's `pos` is the row inside its own segment: the
        // viewport already put the draw there, and the rope table and V step
        // plane it reads are segment-sized.  Handing it the global position
        // read past the end of both planes for every position past the first
        // segment.
        if (!att_rope_none_) glUniform1f(uloc(p, "pos"), (float)seg_off(pos));
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }

// ---- Prefill query batching (PREFILL_BATCH) --------------------------------
// Phase 1 stages each position's rotated Q into row b of qrot_batch_ and its
// K/V into the cache (att_qkv_draw + kv_rope_write, the serial draws).  Phase 2
// scores the whole group in ONE draw: the grid is seq/2 x heads*n, row r =
// h*n + b, and adjacent rows of a head share its K slice -- the 2x2 fragment
// quads coalesce their KV fetches, which is the measured win (att_out_probe:
// qN_serial 5.3 ns/MAC against q1's 15.8, same per-fragment arithmetic).  Phase
// 3 runs stats and v per position against the batched score rows; the residual
// chain (wo, FFN) stays serial in Session::step_group.

void GPU::att_stage(const Mat& qkv, GLuint x, int layer, int pos, int b, int n){
        att_qkv_draw(qkv, x);
        att_finish();
        if (kv_.vstep) kv_vscale(layer, pos);
        kv_rope_write(layer, pos);
        att_qrot_stage(layer, pos, b, n);
}

void GPU::att_qrot_stage(int layer, int pos, int b, int n){
        if (qrot_batch_.tex == 0 || qrot_batch_.w != att_dim_ / 2 || qrot_batch_.h != n)
            qrot_batch_ = make_target(att_dim_ / 2, n);
        if (att_qk_norm_) {
            if (qscale_target_.tex == 0 || qscale_target_.w != att_heads_)
                qscale_target_ = make_target(att_heads_, 1);
            GLuint pq = x_qnorm();
            bind_fbo(qscale_target_.fbo);
            glViewport(0, 0, att_heads_, 1);
            use_program(pq);
            bind_tex(0, qkv_target_.tex);
            set_sampler(pq, "R", 0);
            setup_attrib(pq);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            att_finish();
        }
        GLuint pr = x_qrot();
        bind_fbo(qrot_batch_.fbo);
        // The qrot kernel reads gl_FragCoord.x only, so rendering at y = b
        // stages row b with the serial text unchanged.
        glViewport(0, b, att_dim_ / 2, 1);
        use_program(pr);
        bind_tex(0, qkv_target_.tex);
        set_sampler(pr, "R", 0);
        if (!att_rope_none_) {
                bind_tex(1, rope_tex_);
                set_sampler(pr, "RT", 1);
        }
        if (att_qk_norm_) {
            bind_tex(2, q_norm_tex_[layer]);
            set_sampler(pr, "QN", 2);
            bind_tex(3, qscale_target_.tex);
            set_sampler(pr, "QS", 3);
        }
        if (!att_rope_none_) glUniform1f(uloc(pr, "pos"), (float)pos);
        setup_attrib(pr);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}

void GPU::alloc_cross_kv(int n){
        cross_kv_.assign((size_t)kv_cache_.size(), Target());
        // Height `att_seq_`, not `n`: every attention kernel addresses the KV plane
        // with the compile-time `SEQ` (`(pos+0.5)/SEQ`), so a plane six rows tall
        // put the source positions in the top tenth of a 64-row grid and every
        // query read row 0.  The self-attention's cache has this height for the
        // same reason.
        (void)n;
        for (size_t l = 0; l < cross_kv_.size(); ++l)
                cross_kv_[l] = make_target(kv_.texels, att_seq_);
}

// One half of one row of the cross plane.  The viewport is where the values go --
// K at x=0, V at x=kv_dim/2 -- so one draw is one half of one position's row, and
// the kernel is the attention's own KV encoder (same grid, so nothing can drift).
void GPU::cross_kv_store(int layer, GLuint x, int b, bool v_half, GLuint bias){
        if (layer < 0 || (size_t)layer >= cross_kv_.size() || cross_kv_[layer].tex == 0) return;
        const int half = kv_.texels / 2;
        GLuint p = x_prog(string("xkv_") + (v_half ? "v" : "k") + "_" +
                              to_string(kv_.texels) + "_" + to_string((int)kv_.codec),
                          ::att_cross_kv(att_cfg(), v_half, XFormat::I24_1, bias != 0));
        bind_fbo(cross_kv_[layer].fbo);
        glViewport(v_half ? half : 0, b, half, 1);
        use_program(p);
        bind_tex(0, x);
        set_sampler(p, "XS", 0);
        if (bias) { bind_tex(1, bias); set_sampler(p, "BS", 1); }
        glUniform1f(uloc(p, "XOFF"), v_half ? (float)half : 0.0f);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
}

void GPU::att_scores_batch(int layer, int pos0, int n){
        if (scores_batch_.tex == 0 || scores_batch_.w != att_seq_ / 2 ||
            scores_batch_.h != att_heads_ * n)
            scores_batch_ = make_target(att_seq_ / 2, att_heads_ * n);
        GLuint p = x_score_batch(n);
        bind_fbo(scores_batch_.fbo);
        glViewport(0, 0, att_seq_ / 2, att_heads_ * n);
        use_program(p);
        bind_tex(0, qrot_batch_.tex);
        set_sampler(p, "Q", 0);
        bind_tex(1, kv_cache_[layer].tex);
        set_sampler(p, "KV", 1);
        // The bidirectional build has no clip to make, so the compiler prunes
        // `pos` out and uloc() prints "uniform not found" -- which is the
        // warning a typo produces, so a deliberate absence has to be silent.
        if (att_causal_) glUniform1f(uloc(p, "pos"), (float)pos0);
        glUniform1f(uloc(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}

void GPU::att_stats_row(int pos, int b, int n){
        GLuint p = x_stats_row(n);
        bind_fbo(att_stats_.fbo);
        glViewport(0, 0, att_heads_, 1);
        use_program(p);
        bind_tex(0, scores_batch_.tex);
        set_sampler(p, "S", 0);
        // `pos` is live in both modes: causal, it is this query's position;
        // bidirectional, the last *valid* key (the scan is what bounds which
        // keys enter the softmax either way).  Gating this on `causal` left the
        // uniform at its default of 0, so every query saw exactly one key and
        // the encoder's attention returned V of position 0.
        glUniform1f(uloc(p, "pos"), (float)pos);
        glUniform1f(uloc(p, "qrow0"), (float)b);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}

void GPU::att_v_row(int layer, int pos, int b, int n){
        GLuint p = x_v_row(n);
        bind_fbo(att_out_.fbo);
        glViewport(0, 0, att_out_.w, 1);
        use_program(p);
        bind_tex(0, scores_batch_.tex);
        set_sampler(p, "S", 0);
        bind_tex(2, att_stats_.tex);
        set_sampler(p, "ST", 2);
        bind_tex(1, kv_cache_[layer].tex);
        set_sampler(p, "KV", 1);
        if (kv_.vstep) {
            bind_tex(3, vs_targets_[layer].tex);
            set_sampler(p, "VS", 3);
        }
        // `pos` is live in both modes: causal, it is this query's position;
        // bidirectional, the last *valid* key (the scan is what bounds which
        // keys enter the softmax either way).  Gating this on `causal` left the
        // uniform at its default of 0, so every query saw exactly one key and
        // the encoder's attention returned V of position 0.
        glUniform1f(uloc(p, "pos"), (float)pos);
        glUniform1f(uloc(p, "qrow0"), (float)b);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
}
