#include "session.h"
#include "switches.h"
#include "sampling.h"   // argmax_token: the greedy fallback path
#include "jinja/json.h"   // the cache sidecar: JSON is built, never concatenated
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <string>
#include <vector>

// Default-off probe (PASS_DUP=embed,rms_att,rms_ffn,rms_final,logits,logits_draw
// or all): replay one fixed per-token pass.  Every shader involved is
// deterministic and writes the same target from unchanged inputs, so the replay
// reproduces the same texels and the output stays byte-identical; the end-to-end
// delta is that pass's true marginal cost.  Prices what is left after the
// matvec kernels: the fixed passes, the read-back and the per-token sync.
// Default-off probe (MATVEC_DUP=wo,ffn,w2 or all): replay one weight-matrix
// pass per layer.  It exists because ATTN_DUP and PASS_DUP between them leave
// the four matvec kernels as a single ~200 ms residual, and that residual is
// where the q4/q8 gap lives (FFN gate + w2 carry ~57 of the 102 ms).  Same
// contract as the other two: every shader involved is deterministic and writes
// the same target from unchanged inputs, so the replay reproduces the same
// texels and the output stays byte-identical; the end-to-end delta is that
// kernel's true marginal cost.  `ffn` prices the whole split-chunked FFN
// (gate + swiglu), `w2` prices both column halves.
static bool dup_env(const char* envname, const char* name){
    const char* e = getenv(envname);
    if (!e || !*e) return false;
    const std::string env(e), want(name);
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
static bool pass_dup(const char* name){ return dup_env("PASS_DUP", name); }
// Scratch destination for the w2 replays (see the note at the call site); the
// contents are never read, so one target serves every layer and token.
static Target w2_replay_;
static bool matvec_dup(const char* name){ return dup_env("MATVEC_DUP", name); }


// KV8_SIM probe (default off): quantise the CPU float KV cache to 8 bits at
// read time.  The cache holds the same float K/V the GPU path quantises, and
// both paths consume the dequantised value, so this prices the grid without a
// GPU kernel.  Modes: 1 global scale on K and V, 2 per-head dynamic scale on
// both, 3 global K only, 4 global V only, 5 per-head V only, 6 per-head K only.
// KV8_SIM_SCALE sets the global grid (default 16, the KV_PACK=8 default).
static int kv8_sim_mode(){
    static const int m = env_int("KV8_SIM", 0);
    return m;
}
static float kv8_sim_scale(){
    static const float s = getenv("KV8_SIM_SCALE") ? (float)atof(getenv("KV8_SIM_SCALE")) : 16.0f;
    return s > 0.0f ? s : 16.0f;
}
static inline float kv8_round(float v, float s){
    float q = floorf(v * s + 0.5f);
    q = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
    return q / s;
}

// The FFN matrices one layer dispatch uses: the dense layer's, or one expert's.
// Pointers, so the CPU and GPU FFN paths below stay a single copy of the
// chunked/whole logic.
struct FfnMats {
    const Mat* w1 = nullptr; const Mat* w3 = nullptr; const Mat* w2 = nullptr;
    const Mat* w13 = nullptr;
    const Mat* w1a = nullptr; const Mat* w3a = nullptr; const Mat* w2a = nullptr;
    const Mat* w1b = nullptr; const Mat* w3b = nullptr; const Mat* w2b = nullptr;
    int width = 0;
};

static FfnMats ffn_mats(const Model& m, int l, int expert){
    FfnMats f;
    if (expert >= 0) {
        const Model::MoeExpert& e = m.moe[l][expert];
        f.w1 = &e.w1; f.w3 = &e.w3; f.w2 = &e.w2;
        f.w1a = &e.w1a; f.w3a = &e.w3a; f.w2a = &e.w2a;
        f.w1b = &e.w1b; f.w3b = &e.w3b; f.w2b = &e.w2b;
        f.width = m.moe_hidden;
    } else {
        // A fused-W13 container (feng, stories15M, TinyChat15M) has no w1/w3
        // at all.  Taking the address of an empty vector's element left them
        // null, and the callers' `f.w1->rows` then dereferenced it; every
        // fused-W13 model crashed from 607cf77 (2026-11-11) until this guard.
        if (!m.w1.empty()) { f.w1 = &m.w1[l]; f.w3 = &m.w3[l]; }
        if (!m.w2.empty()) f.w2 = &m.w2[l];
        if (!m.w13.empty()) f.w13 = &m.w13[l];
        if (!m.w1a.empty()) {
            f.w1a = &m.w1a[l]; f.w3a = &m.w3a[l]; f.w2a = &m.w2a[l];
            f.w1b = &m.w1b[l]; f.w3b = &m.w3b[l]; f.w2b = &m.w2b[l];
        }
        f.width = m.hidden;
    }
    return f;
}

// Top-1 routing.  norm_topk_prob is enforced by the loader, so the selected
// expert's output is used with weight exactly 1.0 and the argmax of the router
// logits is the whole decision (softmax is monotone).
static int pick_expert(const vector<float>& gate){
    int best = 0;
    for (size_t i = 1; i < gate.size(); ++i)
        if (gate[i] > gate[best]) best = (int)i;
    return best;
}

void Session::reset(){
        // Capture the referents, destroy, reconstruct in place: the references
        // cannot be rebound, and the two objects this points at outlive the
        // session (they are main()'s Model and GPU).
        Model* m = &m_;
        GPU* g = &gpu_;
        this->~Session();
        new (this) Session(*m, *g);
    }

Session::Session(Model& model, GPU& gpu) : m_(model), gpu_(gpu){
        // The format plan is decided from the model and the env, and the GPU
        // outlives the session: start from the defaults so a previous model's
        // choices (ACT8, RES_PACK) cannot leak into this one.
        gpu_.reset_format_plan();
        gpu_layer_ = gpu_.gpu_layer_enabled();
        gpu_attn_ = gpu_layer_ || gpu_.gpu_attn_enabled();
        // RWKV-7 has no attention block, no FFN gate pair and no KV cache, so
        // none of the fused path's state applies: its layer loop is matvec +
        // host elementwise (docs/rwkv7.md), and everything
        // below -- the conv history, the KV caches, the rope tables, the
        // rmsnorm weight textures, init_attention() -- would be state nothing
        // reads.  GPU_LAYER=1 is honoured by ignoring it: forward_gpu_hidden()
        // runs a transformer block.
        if (m_.is_rwkv()) {
                // gpu_attn_ = false and kv_cache_ empty also decide the disk
                // prefix cache: prefix_cache.cpp refuses to save or load when
                // either is unset, which is the answer the plan's O30 wants for
                // P2 -- the state is a WKV snapshot (version 5), and that is a
                // P6 item, not a cache with a stale shape in it.
                gpu_layer_ = false;
                gpu_attn_ = false;
                // The recurrence and both token-shift vectors: zero here, and
                // again through reset() (the REPL's /model reconstructs the
                // session, which is what drops a previous model's state).
                zero_rwkv_state();
                // P6: RWKV_GPU=1 selects the fused GPU layer loop.  The CPU
                // serial path stays the default (and the numeric reference).
                rwkv_gpu_ = env_flag("RWKV_GPU");
                // The bit-cast f32 WKV state needs ESSL 3.00; without it (or with
                // RWKV_WKV_CPU=1) the recurrence falls back to the host f32 state.
                rwkv_wkv_cpu_ = env_flag("RWKV_WKV_CPU") || !gpu_es3_ok();
                if (rwkv_gpu_) gpu_.init_rwkv(m_);
                if (rwkv_gpu_)
                    fprintf(stderr, "formats: rwkv act %s res %s ffn %s | kv none (recurrent)\n",
                            x_format_name(gpu_.act_fmt_), x_format_name(gpu_.res_fmt_),
                            x_format_name(gpu_.ffn_fmt_));
                return;
        }
        // Both loops are correct for a hybrid model since 2026-11-12: the q8
        // conv block was reading its matvec output 256 too large (a q24 encode
        // bug in matvec_q8's I16_2 branch) and the i16 residual format was
        // decided from an empty wo.  No fallback is needed.
        if (m_.hybrid) {
            if (gpu_layer_) {
                gpu_.init_conv(m_.layers, m_.dim);
                for (int l = 0; l < m_.layers; ++l)
                    if (m_.is_conv(l))
                        gpu_.set_conv_kernel(l, m_.conv_w[l].data(), m_.dim, m_.conv_k);
            } else {
                conv_state_.assign(m_.layers, vector<float>());
                for (int l = 0; l < m_.layers; ++l)
                    if (m_.is_conv(l))
                        conv_state_[l].assign((size_t)(m_.conv_k - 1) * m_.dim, 0.0f);
            }
        }
        // RWKV-7 keeps no KV cache and no RoPE: its cross-token state is the
        // fixed-size WKV matrix plus two shift vectors (RwkvState).  Allocating
        // seq*kv_dim KV rows and a seq*half RoPE table for it would waste RAM
        // (and startup time) for buffers no RWKV kernel ever reads.
        if (!m_.is_rwkv()) {
            if (!gpu_attn_) {
                kv_cache_k_.resize(m_.layers, vector<float>((size_t)m_.seq * m_.kv_dim, 0));
                kv_cache_v_.resize(m_.layers, vector<float>((size_t)m_.seq * m_.kv_dim, 0));
            }
            const int half = m_.head_dim / 2;
            rope_cos_.resize((size_t)m_.seq * half);
            rope_sin_.resize((size_t)m_.seq * half);
            for (int p = 0; p < m_.seq; ++p) {
                for (int i = 0; i < half; ++i) {
                    float freq = 1.0f / powf(m_.rope_theta, (2.0f * i) / m_.head_dim);
                    // A `longrope` model scales pair i's angle by 1/factor_i
                    // (metadata key 18): 32 different bases on a 64-wide head.
                    // Empty for every plain-theta container.
                    if (!m_.rope_factors.empty()) freq /= m_.rope_factors[(size_t) i];
                    float a = p * freq;
                    rope_cos_[(size_t)p * half + i] = cosf(a);
                    rope_sin_[(size_t)p * half + i] = sinf(a);
                }
            }
        }
        // Paired qkv matvec: two output rows per fragment, half the activation
        // decode.  Only the int8 checkpoint uses it; QKV_PAIR=0 restores the
        // single-row kernel.  The int4 pair kernel was tried and rejected: it
        // shifts q4 PPL 325.3751 -> 331.6324 and decodes 1.2% slower.
        // Decided before the format block below because the v5 qkv bias is
        // uploaded in the qkv target's own layout, which this flag picks.
        bool qkv_pair = true;
        for (const Mat& m : m_.qkv)
            if ((m.type != WeightType::INT8 && m.type != WeightType::T3) || (m.rows & 1))
                qkv_pair = false;
        if (env_zero("QKV_PAIR")) qkv_pair = false;
        if (gpu_layer_) {
            // A stream's format is checked against the weight-format table, not
            // a hand-written "model family" list: `act_mask` owns which kernels
            // can read which XFormat, and a value it refuses stops the run with
            // the matrix's name.  The old rule ("qkv/wo/w2 all q8") also gated
            // FFN_HID on matrices that read a different stream, and it had no
            // record for the families it happened to allow or refuse.
            const auto refuse = [&](XFormat f, const char* stream, const char* label,
                                    const Mat& m) {
                if (m.rows <= 0 || wf_accepts(m.type, f)) return;
                cerr << stream << ": " << label << " is " << wf(m.type).name
                     << ", which cannot read " << x_format_name(f) << "\n";
                exit(2);
            };
            const auto refusev = [&](XFormat f, const char* stream, const char* label,
                                     const vector<Mat>& v) {
                for (const Mat& m : v) refuse(f, stream, label, m);
            };
            XFormat act_want = XFormat::I16_2;
            if (const char* a = sw_enum("ACT")) {
                if (!strcmp(a, "f32")) act_want = XFormat::F32;
                else if (!strcmp(a, "bf16")) act_want = XFormat::BF16;
                else if (!strcmp(a, "i8")) act_want = XFormat::I8_4;
            }
            // Every matrix the ACT stream feeds.  w2 is here only for v5: its
            // post-norm FFN writes swish(fc1) in ACT and reads it back in ACT;
            // a dense w2 reads FFN_HID, the other axis.
            refusev(act_want, "ACT", "qkv", m_.qkv);
            refusev(act_want, "ACT", "wo", m_.wo);
            refusev(act_want, "ACT", "w13", m_.w13);
            refusev(act_want, "ACT", "w1", m_.w1);
            refusev(act_want, "ACT", "w3", m_.w3);
            refusev(act_want, "ACT", "w1a", m_.w1a);
            refusev(act_want, "ACT", "w1b", m_.w1b);
            refusev(act_want, "ACT", "w3a", m_.w3a);
            refusev(act_want, "ACT", "w3b", m_.w3b);
            refusev(act_want, "ACT", "router", m_.router);
            refusev(act_want, "ACT", "logits_tiles", m_.logits_tiles);
            refusev(act_want, "ACT", "conv_in", m_.conv_in);
            refusev(act_want, "ACT", "conv_in_b", m_.conv_in_b);
            refusev(act_want, "ACT", "conv_in_c", m_.conv_in_c);
            refusev(act_want, "ACT", "conv_in_x", m_.conv_in_x);
            refusev(act_want, "ACT", "conv_out", m_.conv_out);
            refusev(act_want, "ACT", "u_enc_qkv", m_.u_enc_qkv);
            refusev(act_want, "ACT", "u_enc_wo", m_.u_enc_wo);
            refusev(act_want, "ACT", "u_enc_w1", m_.u_enc_w1);
            refusev(act_want, "ACT", "u_enc_w2", m_.u_enc_w2);
            refusev(act_want, "ACT", "u_cross_q", m_.u_cross_q);
            refusev(act_want, "ACT", "u_cross_k", m_.u_cross_k);
            refusev(act_want, "ACT", "u_cross_v", m_.u_cross_v);
            refusev(act_want, "ACT", "u_cross_o", m_.u_cross_o);
            if (m_.is_encdec()) {
                refusev(act_want, "ACT", "w2", m_.w2);
                refusev(act_want, "ACT", "w2a", m_.w2a);
                refusev(act_want, "ACT", "w2b", m_.w2b);
            }
            for (const auto& layer : m_.moe)
                for (const auto& e : layer) {
                    refuse(act_want, "ACT", "moe.w1", e.w1);
                    refuse(act_want, "ACT", "moe.w3", e.w3);
                    refuse(act_want, "ACT", "moe.w1a", e.w1a);
                    refuse(act_want, "ACT", "moe.w1b", e.w1b);
                    refuse(act_want, "ACT", "moe.w3a", e.w3a);
                    refuse(act_want, "ACT", "moe.w3b", e.w3b);
                }
            gpu_.act_fmt_ = act_want;
            // FFN_HID: the w2 input.  `f32` exists because the q16 grid rails at
            // +-128 and a model whose hidden state exceeds it loses the tail of
            // every FFN output -- MNN-Pocket peaks at 1005 (probe_residuals.py),
            // 8x the rail, which is why it is the first container to ask for it.
            // A v5 block's FFN hidden is the ACT stream (swish(fc1) -> fc2), so
            // the axis does not reach it; say that once rather than accept a
            // value that changes nothing.
            if (const char* fh = sw_enum("FFN_HID")) {
                XFormat want = XFormat::I16_2;
                if (!strcmp(fh, "i8")) want = XFormat::I8_4;
                else if (!strcmp(fh, "f32")) want = XFormat::F32;
                else if (!strcmp(fh, "bf16")) want = XFormat::BF16;
                if (m_.is_encdec()) {
                    cerr << "FFN_HID=" << fh << " has no effect on a v5 container: "
                            "its FFN hidden is the ACT stream\n";
                } else {
                    refusev(want, "FFN_HID", "w2", m_.w2);
                    refusev(want, "FFN_HID", "w2a", m_.w2a);
                    refusev(want, "FFN_HID", "w2b", m_.w2b);
                    for (const auto& layer : m_.moe)
                        for (const auto& e : layer) {
                            refuse(want, "FFN_HID", "moe.w2", e.w2);
                            refuse(want, "FFN_HID", "moe.w2a", e.w2a);
                            refuse(want, "FFN_HID", "moe.w2b", e.w2b);
                        }
                    gpu_.ffn_fmt_ = want;
                }
            }
            // Residual stream packed two int16 per texel: halves the fetch count in
            // every rmsnorm pass (stats and norm both read the residual) and lets
            // matvec_add produce two rows per fragment.  Measured -11% decode on
            // the q8 model, perplexity unchanged.  RES_PACK=0 opts out.
            // A hybrid model's layer 0 is a conv layer with no wo, so wo[0] is an
            // empty Mat whose type defaults to INT8.  Testing it claimed the q8/q4
            // packed residual for an i16 container, whose add kernels then wrote
            // one q16 per texel into a target the rmsnorm read as pairs -- the
            // LFM2 i16 packs answered with 0.996 constants (2026-11-12).  Use the
            // first attention layer's matrix, like the fused-loop check above.
            bool wo_packs = false;
            for (const Mat& w : m_.wo)
                if (w.rows > 0) {
                    // A t3 matrix writes the packed pair too (matvec_t3_add_pair),
                    // so it takes the same two-values-a-texel residual the q8 and
                    // q4 paths use: half the fetch count in every rmsnorm pass.
                    wo_packs = w.type == WeightType::INT8 || w.type == WeightType::INT4 ||
                               w.type == WeightType::T3;
                    break;
                }
            // The residual axis, one value: `i16` keeps the default decision
            // (two q16 a texel where the wo kernels can pack, which halves the
            // fetch count in every rmsnorm pass: -11% decode on the q8 model,
            // perplexity unchanged), `i16_1` the one-a-texel q16 stream (what
            // RES_PACK=0 used to force), `i24` and `f32` the wide ones.
            // A container that declares a 24-bit residual gets one value per
            // texel: packing two values into a texel is a q16-only idea, and
            // the residual-consuming kernels store q16, so a container's own
            // declaration wins over the default but not over an explicit RES.
            const char* res = sw_enum("RES");
            // bf16: f32's exponent range at q16's two bytes a value -- the rung
            // between "q16 rails at +-128" and "f32 costs 21% more decode"
            // (docs/performance.md, 2026-11-18).
            if (res && !strcmp(res, "bf16")) gpu_.res_fmt_ = XFormat::BF16;
            else if (res && !strcmp(res, "f32")) gpu_.res_fmt_ = XFormat::F32;
            else if (res && !strcmp(res, "i24")) gpu_.res_fmt_ = XFormat::I24_1;
            else if (res && !strcmp(res, "i16_1")) gpu_.res_fmt_ = XFormat::I16_1;
            else if (m_.res_q24) gpu_.res_fmt_ = XFormat::I24_1;
            else gpu_.res_fmt_ = wo_packs ? XFormat::I16_2 : XFormat::I16_1;
            gpu_.res_pack_ = x_per_texel(gpu_.res_fmt_) == 2;
            if (res && strcmp(res, "i16") && m_.res_q24 && strcmp(res, "i24"))
                cerr << "RES overrides the container's q24 residual\n";
            // A projection's output: the single-value matvec store (the pair
            // kernels keep their packed q16).  `f32` is exact and costs the
            // same to decode (probes/codec_probe: 6.08 ns against 6.08) but
            // needs ESSL 3.00 for the bit-cast; `i24` is the shipped default.
            const char* proj = sw_enum("PROJ");
            if (proj && !strcmp(proj, "f32")) gpu_.proj_fmt_ = XFormat::F32;
            else gpu_.proj_fmt_ = XFormat::I24_1;
            if (gpu_.proj_fmt_ == XFormat::F32 && !gpu_es3_ok()) {
                cerr << "PROJ=f32: the projection store is a bit-cast and needs an "
                        "ESSL 3.00 context; this driver reports less\n";
                exit(2);
            }
            // The v5 bias and LayerNorm rows are packed *in the formats chosen
            // above*, so they cannot be uploaded before this block runs.  They
            // were uploaded from App::upload_weights(), i.e. while res_fmt_ was
            // still its I16_1 default: every row went out one-value-a-texel
            // (alpha = 255) and the row passes read the odd half of an I16_2
            // row as that alpha byte -- a constant +255/256 on every odd
            // component, which is what the "odd components are wrong" hunt was
            // chasing for six commits.
            if (m_.is_encdec()) {
                // The v5 LayerNorm's statistics are f32 (a bit-cast), so every
                // norm pass is an ESSL 3.00 kernel.  The ES 3.0 context request
                // and the Mesa override the q4 path sets normally provide it;
                // say it here rather than let the first layer die in a shader
                // compile with a line number.
                if (!gpu_es3_ok()) {
                    cerr << "encoder-decoder: the v5 LayerNorm statistics are f32 and "
                            "need an ESSL 3.00 context; this driver reports less "
                            "(on Mesa, Q4_SHIFT=0 turns off the version override)\n";
                    exit(2);
                }
                // Every v5 add reads its base as a *residual* stream (the
                // previous LayerNorm's output, see matvec_add_rf), so the
                // activation format has to be one the residual codec covers.
                // ACT=i8 does not, and without this it dies in a link error
                // ("no function with name Y1Q") at the first layer.
                if (!x_can_residual(gpu_.act_fmt_)) {
                    cerr << "encoder-decoder: ACT=" << x_format_name(gpu_.act_fmt_)
                         << " cannot be a residual stream, and a v5 block adds into "
                            "the previous LayerNorm's output\n";
                    exit(2);
                }
                // The qkv bias rides in the qkv target's own layout, which is
                // the pair it packs into or -- unpaired -- the projection's.
                gpu_.upload_v5_rows(m_, (qkv_pair && ((m_.dim + 2 * m_.kv_dim) % 2 == 0))
                                            ? XFormat::I16_2 : gpu_.proj_fmt_);
            }
        }
        if (gpu_attn_)
            gpu_.init_attention(m_.dim, m_.kv_dim, m_.heads, m_.kv_heads, m_.head_dim,
                                m_.seq, m_.rope_half, m_.qk_norm, m_.layers,
                                m_.q_norm, m_.k_norm, rope_cos_, rope_sin_, qkv_pair,
                                m_.hybrid, m_.hybrid ? &m_.layer_type : nullptr,
                                /*rope_none=*/m_.is_encdec());
        // What this run actually asked the kernels for.  One line, because every
        // investigation here starts with "which stream is this texture in".
        if (gpu_attn_) {
            const char* kvs = gpu_.kv_.codec == KvCodec::Bf16 ? "bf16"
                            : gpu_.kv_.codec == KvCodec::F32 ? "f32"
                            : gpu_.kv_.bits == 16 ? "q16"
                            : gpu_.kv_.bits == 8 ? "q8" : "i24";
            const char* rms = sw_str("RMS_STORE");
            fprintf(stderr, "formats: act %s res %s ffn %s proj %s | kv %s width %d | "
                            "qkv_pair %d | rms %s\n",
                    x_format_name(gpu_.act_fmt_), x_format_name(gpu_.res_fmt_),
                    x_format_name(gpu_.ffn_fmt_), x_format_name(gpu_.proj_fmt_), kvs,
                    gpu_.kv_.texels, (int)gpu_.att_qkv_pair_, rms ? rms : "i24");
            // The parse above is inside `if (gpu_layer_)`; saying so is the
            // difference between "the axis was applied" and "the axis was
            // ignored and the defaults ran".
            if (!gpu_layer_)
                fprintf(stderr, "formats: ACT/RES/FFN_HID/PROJ apply only with "
                                "GPU_LAYER=1; this run uses the defaults above\n");
        }
        // A v5 container has no model-wide norm: every layer carries its own
        // LayerNorm inside the bias rows, so init_persistent would memcpy from
        // three empty vectors (and did: SIGSEGV in memmove).
        if ((gpu_layer_ || getenv("CPU_RMS_GPU")) && !m_.is_encdec())
            gpu_.init_persistent(m_.dim, m_.layers, m_.rms_att, m_.rms_ffn, m_.rms_final);
    }

// The block's whole history: one head_size x head_size recurrence per head per
// layer (f32, see session.h) and the two token-shift rows per layer.  Sized from
// the model, so the constructor and any future caller agree.
void Session::zero_rwkv_state(){
        if (!m_.is_rwkv()) return;
        const size_t hs = (size_t)m_.rwkv.head_size;
        rwkv_state_.wkv.assign((size_t)m_.layers * m_.heads * hs * hs, 0.0f);
        rwkv_state_.shift_att.assign((size_t)m_.layers * m_.dim, 0.0f);
        rwkv_state_.shift_ffn.assign((size_t)m_.layers * m_.dim, 0.0f);
        rwkv_state_.v_first.assign(m_.dim, 0.0f);
        rwkv_state_.have_v_first = false;
}

void Session::zero_conv_state(){
        if (!m_.hybrid) return;
        if (gpu_layer_) { gpu_.clear_conv_state(); return; }
        for (auto& v : conv_state_)
                for (float& f : v) f = 0.0f;
}


// ACT_CPU=f32 (was ACT_F32=1) keeps the CPU layer loop's activations in f32
// instead of storing them on the q16 grid.  The grid is 1/256, which is fine for a residual of
// order 1 and fatal for one of order 0.01: LFM2.5's embedding row peaks at
// 0.05, so quantising it left 2-3 significant bits per component and the whole
// model drifted (probe: xb 8.8% off after the first norm).  The GPU path is
// unaffected -- its textures are q16 by construction.
static bool act_f32() {
    static const bool on = [] {
        const char* v = sw_enum("ACT_CPU");
        return v && !strcmp(v, "f32");
    }();
    return on;
}

vector<float> Session::embedding(int token, bool exact_f32){
        vector<float> x(m_.dim);
        const bool f32 = exact_f32 || act_f32();
        auto q = [&](float v) { return f32 ? v : quantize_act_f(v); };
        if (m_.has_i16_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = q((float)m_.emb_i16[(size_t)token * m_.dim + i] * s);
        } else if (m_.has_q6_emb) {
            // q6 row: 4-bit low plane, 2-bit high plane, both row-major.
            const float s = m_.emb_scale[token];
            const uint8_t* lo = m_.emb_q6_lo.data() + (size_t)token * (m_.dim / 2);
            const uint8_t* hi = m_.emb_q6_hi.data() + (size_t)token * (m_.dim / 4);
            for (int i = 0; i < m_.dim; ++i) {
                const int lb = (i & 1) ? (lo[i / 2] >> 4) : (lo[i / 2] & 15);
                const int hb = (hi[i / 4] >> (2 * (i & 3))) & 3;
                int v = lb | (hb << 4);
                if (v >= 32) v -= 64;
                x[i] = q((float)v * s);
            }
        } else if (m_.has_q8_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = q((float)m_.emb_q8[(size_t)token * m_.dim + i] * s);
        } else {
            for (int i = 0; i < m_.dim; ++i)
                x[i] = q(m_.emb_f32[(size_t)token * m_.dim + i]);
        }
        return x;
    }

vector<float> Session::forward(int token, int pos){
        vector<float> x = embedding(token);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return x;
            vector<float> xb = getenv("CPU_RMS_GPU")
                    ? rms_gpu(x, l, false)
                    : rms_norm(x, m_.rms_att.data() + (size_t)l * m_.dim);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(xb, gpu_.act_fmt_, "xb", 4096);
            vector<float> wo;
            if (m_.is_conv(l)) {
                // LFM2 conv block: in_proj -> gate -> depthwise causal conv
                // (k taps, state in conv_state_) -> gate -> out_proj.  Every
                // stage that the GPU stores as q16 is quantised here too, so
                // this is the reference the GPU kernel must reproduce.
                const int dim = m_.dim, k = m_.conv_k;
                vector<float> gb = gpu_.matvec_s(m_.conv_in_b[l], xb);
                vector<float> gc = gpu_.matvec_s(m_.conv_in_c[l], xb);
                vector<float> gx = gpu_.matvec_s(m_.conv_in_x[l], xb);
                vector<float> gated(dim), cv(dim);
                for (int i = 0; i < dim; ++i)
                    gated[i] = act_f32() ? gb[i] * gx[i] : quantize_act_f(gb[i] * gx[i]);
                vector<float>& st = conv_state_[l];
                for (int i = 0; i < dim; ++i) {
                    float acc = 0.0f;
                    for (int j = 0; j < k; ++j) {
                        const float v = (j + 1 < k) ? st[(size_t)j * dim + i] : gated[i];
                        acc += m_.conv_w[l][(size_t)i * k + j] * v;
                    }
                    cv[i] = act_f32() ? acc : quantize_act_f(acc);
                }
                if (k > 1) {
                    for (int j = 0; j + 2 < k; ++j)
                        memcpy(st.data() + (size_t)j * dim, st.data() + (size_t)(j + 1) * dim,
                               sizeof(float) * dim);
                    memcpy(st.data() + (size_t)(k - 2) * dim, gated.data(), sizeof(float) * dim);
                }
                vector<float> y(dim);
                for (int i = 0; i < dim; ++i)
                    y[i] = act_f32() ? gc[i] * cv[i] : quantize_act_f(gc[i] * cv[i]);
                if (getenv("CONV_DEBUG") && l == 0) {
                    cerr << "CV pos=" << pos << " xb";
                    for (int i = 0; i < 5; ++i) cerr << " " << xb[i];
                    cerr << " | B";
                    for (int i = 0; i < 5; ++i) cerr << " " << gb[i];
                    cerr << " | C";
                    for (int i = 0; i < 3; ++i) cerr << " " << gc[i];
                    cerr << " | x";
                    for (int i = 0; i < 3; ++i) cerr << " " << gx[i];
                    cerr << " | gated";
                    for (int i = 0; i < 5; ++i) cerr << " " << gated[i];
                    cerr << " | cv";
                    for (int i = 0; i < 5; ++i) cerr << " " << cv[i];
                    cerr << " | y";
                    for (int i = 0; i < 5; ++i) cerr << " " << y[i];
                    cerr << " | w0..2 " << m_.conv_w[l][0] << " " << m_.conv_w[l][1]
                         << " " << m_.conv_w[l][2] << "\n";
                }
                if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(gated, gpu_.act_fmt_, "cvg", 4096);
                wo = gpu_.matvec_s(m_.conv_out[l], y);
            } else if (gpu_attn_) {
                GLuint xtex = gpu_.upload_input(xb);
                GLuint atex = gpu_.attention_block(m_.qkv[l], xtex, l, pos);
                if (getenv("CPU_ADD_GPU")) {
                        // Probe: the attention output's residual add with the
                        // fused arithmetic (one rounding), like matvec_add.
                        GLuint r = gpu_.upload_hidden(x);
                        gpu_.matvec_add(m_.wo[l], atex, r, gpu_.hidden_a_, gpu_.ffn_fmt_);
                        wo = read_hidden(gpu_.hidden_a_);
                        wo.resize(x.size());
                        x = wo;                     // already x + wo
                        wo.assign(x.size(), 0.0f);
                } else {
                        wo = gpu_.matvec_from(m_.wo[l], atex);
                }
            } else {
                vector<float> qkv = gpu_.matvec_s(m_.qkv[l], xb);
                vector<float> q(qkv.begin(), qkv.begin() + m_.dim);
                vector<float> k(qkv.begin() + m_.dim, qkv.begin() + m_.dim + m_.kv_dim);
                vector<float> v(qkv.begin() + m_.dim + m_.kv_dim, qkv.begin() + m_.dim + 2 * m_.kv_dim);
                if (m_.qk_norm) {
                    q = head_rms_norm(q, m_.heads, m_.q_norm.data() + (size_t)l * m_.head_dim);
                    k = head_rms_norm(k, m_.kv_heads, m_.k_norm.data() + (size_t)l * m_.head_dim);
                }
                q = rope(q, m_.heads, pos);
                k = rope(k, m_.kv_heads, pos);
                memcpy(kv_cache_k_[l].data() + (size_t)pos * m_.kv_dim, k.data(), m_.kv_dim * 4);
                memcpy(kv_cache_v_[l].data() + (size_t)pos * m_.kv_dim, v.data(), m_.kv_dim * 4);
                vector<float> att = attention(q, l, pos);
                vector<float> att_q16(att.size());
                for (size_t i = 0; i < att.size(); ++i)
                    att_q16[i] = act_f32() ? att[i] : quantize_act_f(att[i]);
                if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(att_q16, gpu_.act_fmt_, "att", 4096);
                wo = gpu_.matvec_s(m_.wo[l], att_q16);
            }
            vector<float> x1 = add_q16(x, wo);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(x1, gpu_.act_fmt_, "x1", 4096);
            vector<float> xbf = getenv("CPU_RMS_GPU")
                    ? rms_gpu(x1, l, true)
                    : rms_norm(x1, m_.rms_ffn.data() + (size_t)l * m_.dim);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(xbf, gpu_.act_fmt_, "xbf", 4096);
            vector<float> o2;
            int expert = -1;
            if (m_.is_moe() && getenv("MOE_FIXED")) {
                // Probe: skip the router and its read-back, pin one expert.
                expert = atoi(getenv("MOE_FIXED"));
            } else if (m_.is_moe()) {
                vector<float> gate = gpu_.matvec_s(m_.router[l], xbf);
                expert = pick_expert(gate);
                if (getenv("MOE_DUMP"))
                    cerr << "MOE l=" << l << " route=" << expert << " gate="
                         << gate[0] << "," << gate[1] << "," << gate[2] << "," << gate[3] << "\n";
            }
            const FfnMats f = ffn_mats(m_, l, expert);
            // Metadata key 15: the serial FFN kernels write the SwiGLU hidden
            // *2^-hs (and w2's scale was folded by 2^hs), so the CPU FFN path
            // divides the float hidden before its matvec upload.
            const int hs = m_.hid_shift(l);
            const float hsm = hs > 0 ? ldexpf(1.0f, -hs) : 1.0f;
            bool gpu_ffn = m_.is_moe() || !m_.w1.empty();
            if (getenv("CPU_FFN")) gpu_ffn = false;
            else if (getenv("GPU_FFN")) gpu_ffn = atoi(getenv("GPU_FFN")) != 0;
            if (!gpu_ffn) {
                vector<float> h;
                if (m_.ffn_chunked) {
                    vector<float> h1 = gpu_.matvec_s(*f.w1a, xbf);
                    vector<float> h3 = gpu_.matvec_s(*f.w3a, xbf);
                    vector<float> hh;
                    hh.reserve(2 * m_.ffn_chunk);
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> ha = swiglu(hh, m_.ffn_chunk);
                    if (hs) for (float& v : ha) v *= hsm;
                    h1 = gpu_.matvec_s(*f.w1b, xbf);
                    h3 = gpu_.matvec_s(*f.w3b, xbf);
                    hh.clear(); hh.reserve(2 * (f.width - m_.ffn_chunk));
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> hb = swiglu(hh, f.width - m_.ffn_chunk);
                    if (hs) for (float& v : hb) v *= hsm;
                    o2 = add_q16(gpu_.matvec_s(*f.w2a, ha), gpu_.matvec_s(*f.w2b, hb));
                } else {
                    if (f.w1 && f.w1->rows) {
                        vector<float> h1 = gpu_.matvec_s(*f.w1, xbf);
                        vector<float> h3 = gpu_.matvec_s(*f.w3, xbf);
                        h.reserve(2 * f.width);
                        h.insert(h.end(), h1.begin(), h1.end());
                        h.insert(h.end(), h3.begin(), h3.end());
                    } else {
                        h = gpu_.matvec_s(*f.w13, xbf);
                    }
                    vector<float> hid = swiglu(h);
                    if (hs) for (float& v : hid) v *= hsm;
                    o2 = gpu_.matvec_s(*f.w2, hid);
                }
            } else {
                GLuint xtex = gpu_.upload_input(xbf);
                GLuint htex = 0;
                if (m_.ffn_chunked) {
                    GLuint ha, hb;
                    gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xtex,
                                           m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb, hs);
                    if (getenv("CPU_ADD_GPU")) {
                        // Probe: fold the residual the way the fused path does
                        // (one rounding per stage) instead of two read-backs
                        // plus a CPU add.
                        GLuint r1 = gpu_.upload_hidden(x1);
                        gpu_.matvec_add(*f.w2a, ha, r1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                        gpu_.matvec_add(*f.w2b, hb, gpu_.hidden_a_.tex, gpu_.hidden_b_, gpu_.ffn_fmt_);
                        o2 = read_hidden(gpu_.hidden_b_);
                        o2.resize(x1.size());
                        x1 = o2;
                        o2.assign(x1.size(), 0.0f);
                    } else {
                        o2 = add_q16(gpu_.matvec_from(*f.w2a, ha), gpu_.matvec_from(*f.w2b, hb));
                    }
                } else if (f.w1 && f.w1->rows) htex = gpu_.ffn_split(*f.w1, *f.w3, xtex, f.width, hs);
                else htex = gpu_.ffn_fused(*f.w13, xtex, f.width, hs);
                if (!m_.ffn_chunked) o2 = gpu_.matvec_from(*f.w2, htex);
            }
            x = add_q16(x1, o2);
            if (getenv("HIDDEN_DUMP")) gpu_.dump_cpu_row(x, gpu_.act_fmt_, "hidA", 4096);
        }
        return x;
    }

// ---- RWKV-7: the block's layer loop (docs/engine-internals.md) ------
//
// A single token through one RWKV-7 (Goose) layer, transcribed from
// engine/tools/ref_rwkv.py's Rwkv7.forward() -- the oracle this path is accepted
// against, cos 1.00000000 against the official rwkv package on the same
// checkpoint (docs/rwkv7.md).  The *order* is the
// oracle's and has to stay the oracle's: the block is a chain, so a swapped pair
// is a different model rather than a different rounding.
//
// What runs where: r/k/v, the four low-rank pairs, W_o, the channel mix's key
// and value, and the logits cross a texture (GPU::matvec); the token shift, the
// six lerps, tanh/sigmoid/exp, the per-head L2 norm, the WKV recurrence, the
// per-head group norm, the r_k bonus and the gating are f32 on the host.  That
// is P2's split: the matvec family already exists (no new shader), and the
// recurrence is the one tensor whose f32-ness is a measured requirement -- S-1
// puts the q16 grid on the state at +0.1543 nats/token and greedy 6/8.
static const float kRwkvDecayC = 0.60653065971263342f;  // 1/sqrt(e), rwkv7-base.cpp:69
static const float kRwkvGroupEps = 64e-5f;              // rwkv7-base.cpp:118

// 1/(1+exp(-z)) without the overflow.  g1d's decay and gate logits reach +-80,
// where exp(-z) is not representable; the oracle suppresses the warning and
// keeps the saturated 0/1, which is what this returns.
static inline float rwkv_sigmoid(float z){
    if (z >= 0.0f) return 1.0f / (1.0f + expf(-z));
    const float e = expf(z);
    return e / (1.0f + e);
}

vector<float> Session::rwkv_layernorm(const vector<float>& x, const float* wb, float eps){
        const int n = (int)x.size();
        double sum = 0.0;
        for (float v : x) sum += v;
        const float mu = (float)(sum / n);
        double vs = 0.0;
        for (float v : x) { const double d = (double)v - (double)mu; vs += d * d; }
        const float inv = 1.0f / sqrtf((float)(vs / n) + eps);
        vector<float> o((size_t)n);
        for (int i = 0; i < n; ++i) o[i] = (x[i] - mu) * inv * wb[i] + wb[n + i];
        return o;
    }

// A matvec whose input may sit past the q16 activation rail (+-127.996).  The
// token-shift lerps do: g1d's xv reaches 225 and layers 3-9 cross the rail,
// where the upload clamps *silently* -- that was worth cos 0.9948 against the
// oracle until it was found.  Dividing the input by 2^k and multiplying the
// matrix's read-back scale by 2^k is exact (both are powers of two) and leaves
// the *output* unchanged, because the kernel computes scale*dot; it only moves
// the input rail to 128*2^k.
//
// k normalises the peak into 32..63 -- the same target GPU::matvec_s aims at --
// rather than being the smallest value that fits under the rail: both forms fix
// the overflow, and they differ only in which quantisation of the same row the
// kernel sees (measured: cos 0.99605 against the container-weight oracle for
// this form, 0.99547 for the threshold form; at that level it is the grid's
// noise floor, not a quality ranking).
//
// GPU::matvec_s is the opposite operation -- it scales the input *up* to buy
// resolution, which multiplies the output before the kernel's own clamp -- so
// it cannot be used here: this block's matvec outputs run to 90.
vector<float> Session::rwkv_mv(const Mat& m, const vector<float>& x){
        float mx = 0.0f;
        for (float v : x) mx = std::max(mx, fabsf(v));
        int e = 0;
        frexpf(mx, &e);                  // mx = f * 2^e, f in [0.5, 1)
        const int k = e - 6;             // put the peak in 32..63
        if (!(mx > 0.0f) || k <= 0) return gpu_.matvec(m, x);
        Mat ms = m;                      // a view: borrowed bytes are not copied
        if (ms.type == WeightType::INT4) ms.q4_param_scale = ldexpf(m.q4_param_scale, k);
        else ms.scale = ldexpf(m.scale, k);
        const float inv = ldexpf(1.0f, -k);
        vector<float> xs(x.size());
        for (size_t i = 0; i < x.size(); ++i) xs[i] = x[i] * inv;
        // No correction on the way out: the kernel's own scale uniform undoes
        // the input scaling exactly, so the readback is already in model units.
        return gpu_.matvec(ms, xs);
    }

// A group of matvecs sharing one read-back (docs/rwkv7.md).  The
// per-matrix work is rwkv_mv's, verbatim: the same 2^k input normalisation, the
// same Mat view with the scale folded in, the same draw and the same decode.
// What changes is only the number of host syncs -- one per group instead of one
// per matrix -- because on this path the read-back, not the draw, is the cost.
vector<vector<float>> Session::rwkv_mv_group_impl(const vector<const Mat*>& ms,
                                                  const vector<const vector<float>*>& xs,
                                                  int slot0, int mode){
        vector<Mat> views(ms.size());
        vector<vector<float>> scaled(ms.size());
        vector<GPU::MvRow> rows(ms.size());
        for (size_t i = 0; i < ms.size(); ++i) {
                const Mat& m = *ms[i];
                const vector<float>& x = *xs[i];
                float mx = 0.0f;
                for (float v : x) mx = std::max(mx, fabsf(v));
                int e = 0;
                frexpf(mx, &e);
                const int k = e - 6;
                views[i] = m;              // a view: borrowed bytes are not copied
                rows[i].m = &views[i];
                if (!(mx > 0.0f) || k <= 0) {
                        rows[i].x = xs[i];
                        continue;
                }
                if (views[i].type == WeightType::INT4) views[i].q4_param_scale = ldexpf(m.q4_param_scale, k);
                else views[i].scale = ldexpf(m.scale, k);
                const float inv = ldexpf(1.0f, -k);
                scaled[i].resize(x.size());
                for (size_t j = 0; j < x.size(); ++j) scaled[i][j] = x[j] * inv;
                rows[i].x = &scaled[i];
        }
        const int read_mode = (mode == 2) ? 1 : (mode == 3 ? 2 : 0);
        vector<vector<float>> out = gpu_.matvec_rows(rows, slot0, read_mode);
        // RWKV_GROUP_CHECK: the grouped result against the pre-P7 path, in the
        // same process and the same run.  A read-back quirk shows up here with
        // the member, the index and the magnitude, which is what a drift that
        // only VC4 has needs in order to be named rather than bisected.
        if (env_flag("RWKV_GROUP_CHECK")) {
                static int nmis = 0, nprinted = 0;
                for (size_t i = 0; i < ms.size(); ++i) {
                        const vector<float> one = rwkv_mv(*ms[i], *xs[i]);
                        int ndiff = 0, first = -1;
                        float mx = 0.0f;
                        for (size_t j = 0; j < one.size() && j < out[i].size(); ++j) {
                                const float d = fabsf(one[j] - out[i][j]);
                                if (d > 0.0f) { if (first < 0) first = (int)j; ++ndiff; mx = std::max(mx, d); }
                        }
                        if (ndiff) {
                                ++nmis;
                                if (nprinted++ < 12)
                                        fprintf(stderr, "RWKV_GROUP_CHECK member=%zu rows=%d cols=%d ndiff=%d/%d maxdiff=%g first=%d (call %d)\n",
                                                i, ms[i]->rows, ms[i]->cols, ndiff, (int)one.size(),
                                                (double)mx, first, nmis);
                        }
                }
        }
        return out;
}

vector<vector<float>> Session::rwkv_mv_group(const vector<const Mat*>& ms,
                                             const vector<const vector<float>*>& xs,
                                             int slot0){
        // RWKV_GROUP (docs/test-hooks.md): 1 groups the whole call, 4 groups
        // only the runs whose matrices share a row count (the VC4-safe arm if
        // the unequal-width draws turn out to be the problem), 0 is the pre-P7
        // path verbatim, 2 and 3 are the read-back bisect.
        // The default arm is the safe one wherever the read-back behaviour is not
        // ours to assume: VC4 (needs_finish) keeps the pre-P7 path until the board
        // bisect names an arm that is bit-identical there, every other driver keeps
        // the grouping.  RWKV_GROUP overrides either way.
        static const int mode = env_int("RWKV_GROUP", gpu_.needs_finish() ? 0 : 1);
        if (mode == 0) {
                vector<vector<float>> out(ms.size());
                for (size_t i = 0; i < ms.size(); ++i) out[i] = rwkv_mv(*ms[i], *xs[i]);
                return out;
        }
        if (mode == 4) {
                vector<vector<float>> out(ms.size());
                for (size_t i = 0; i < ms.size();) {
                        size_t j = i + 1;
                        while (j < ms.size() && ms[j]->rows == ms[i]->rows) ++j;
                        vector<const Mat*> gm(ms.begin() + (long)i, ms.begin() + (long)j);
                        vector<const vector<float>*> gx(xs.begin() + (long)i, xs.begin() + (long)j);
                        vector<vector<float>> part = rwkv_mv_group_impl(gm, gx, slot0 + (int)i, 1);
                        for (size_t k = 0; k < part.size(); ++k) out[i + k] = std::move(part[k]);
                        i = j;
                }
                return out;
        }
        return rwkv_mv_group_impl(ms, xs, slot0, mode);
}

vector<float> Session::forward_rwkv(int token){
        const int dim = m_.dim, L = m_.layers, H = m_.heads, hs = m_.rwkv.head_size;
        const float eps = m_.rwkv.norm_eps;
        const size_t hdim = (size_t)hs * (size_t)hs;
        // The token norm is layer 0's pre_norm, which the container carries
        // model-wide (id 0x22): one vector, not a per-layer group.  llama.cpp
        // reads it as token_embd_norm and the oracle applies it as layer 0's
        // pre_norm, which is the same function at every position.
        // The embedding stays off the q16 grid: g1d's rows peak at 0.08 and the
        // grid's step is 1/256, so rounding here would leave two or three bits
        // per component -- and the block's first op is a LayerNorm, which
        // divides by the row's deviation and hands that error straight to every
        // matvec.  Everything after the token norm is an O(1) row, so the grid
        // there is the one A-1/K-1/Q-1 priced.
        vector<float> x = embedding(token, true);
        if (!m_.rwkv_tok_norm.empty()) x = rwkv_layernorm(x, m_.rwkv_tok_norm.data(), eps);
        vector<float> sa((size_t)hs);           // <S[i], kk>, per head, reused
        // RWKV_PROFILE (docs/test-hooks.md): read-only.  The matvec's three GL
        // steps come from GPU's counters (upload/draw/read-back); `elem` is the
        // layer's wall time minus those, i.e. the host side of the block.  On a
        // needs_finish() driver the upload finishes inside its own step, so the
        // read-back is what carries the GPU's matvec work.
        static const bool prof = env_set("RWKV_PROFILE");
        double t_up = 0.0, t_draw = 0.0, t_read = 0.0, t_tot = 0.0;
        for (int l = 0; l < L; ++l) {
            if (g_stop) return x;
            const auto tl0 = Clock::now();
            const double up0 = gpu_.rwkv_prof.upload_ms;
            const double dr0 = gpu_.rwkv_prof.draw_ms;
            const double rd0 = gpu_.rwkv_prof.read_ms;
            // O1/O2/O3: att_norm, then the six token-shifted inputs.  Position
            // t's previous input is the *normalised* row of t-1, which is why
            // the shift state holds xn and not x.
            vector<float> xn = rwkv_layernorm(x, m_.rwkv_att_norm.data() + (size_t)l * 2 * dim, eps);
            float* sh = rwkv_state_.shift_att.data() + (size_t)l * dim;
            const float* lp = m_.rwkv_lerp.data() + (size_t)l * 6 * dim;
            vector<float> xr((size_t)dim), xw((size_t)dim), xk((size_t)dim);
            vector<float> xv((size_t)dim), xa((size_t)dim), xg((size_t)dim);
            for (int i = 0; i < dim; ++i) {
                const float sx = sh[i] - xn[i];
                xr[i] = xn[i] + sx * lp[i];
                xw[i] = xn[i] + sx * lp[dim + i];
                xk[i] = xn[i] + sx * lp[2 * dim + i];
                xv[i] = xn[i] + sx * lp[3 * dim + i];
                xa[i] = xn[i] + sx * lp[4 * dim + i];
                xg[i] = xn[i] + sx * lp[5 * dim + i];
            }
            memcpy(sh, xn.data(), sizeof(float) * dim);
            // O4-O6: the three projections.  The container stores them as one
            // 3*dim x dim matrix under the dense qkv id; split_rwkv() cut it
            // into the three Mats each of these needs.  One read-back for the
            // three draws: r, k and v are independent, so sharing the host sync
            // changes nothing but the number of syncs (P7, A2).
            const Mat* rkv_m[3] = {&m_.rwkv_r[l], &m_.rwkv_k[l], &m_.rwkv_v[l]};
            const vector<float>* rkv_x[3] = {&xr, &xk, &xv};
            vector<vector<float>> rkv = rwkv_mv_group(
                vector<const Mat*>(rkv_m, rkv_m + 3),
                vector<const vector<float>*>(rkv_x, rkv_x + 3), 0);
            vector<float> r = std::move(rkv[0]), k = std::move(rkv[1]), v = std::move(rkv[2]);
            const float* w0 = m_.rwkv_bias.data() + (size_t)l * 3 * dim;
            const float* a0 = w0 + dim;
            const float* v0 = a0 + dim;
            // O7-O10: the four low-rank branches (decay, value residual, ICLR,
            // the optional gate).  Four down projections and four up projections
            // become two read-backs instead of eight: the branches are
            // independent of each other, so the draws and their arithmetic are
            // unchanged and only the shared host sync differs (P7, A3).
            const bool have_gate = m_.rwkv.gate && l < (int)m_.rwkv_g1.size() &&
                                   m_.rwkv_g1[l].rows > 0;
            const bool have_vres = l > 0 && rwkv_state_.have_v_first &&
                                   m_.rwkv_v1[l].rows > 0;
            vector<const Mat*> dm, um;
            vector<const vector<float>*> dx, ux;
            dm.push_back(&m_.rwkv_w1[l]); dx.push_back(&xw);
            if (have_vres) { dm.push_back(&m_.rwkv_v1[l]); dx.push_back(&xv); }
            dm.push_back(&m_.rwkv_a1[l]); dx.push_back(&xa);
            if (have_gate) { dm.push_back(&m_.rwkv_g1[l]); dx.push_back(&xg); }
            vector<vector<float>> down = rwkv_mv_group(dm, dx, 4);
            size_t bi = 0;
            vector<float> tw = std::move(down[bi++]);          // w down: tanh
            vector<float> tv;
            if (have_vres) tv = std::move(down[bi++]);         // v down: pass-through
            vector<float> ta = std::move(down[bi++]);          // a down: pass-through
            vector<float> tg;
            if (have_gate) tg = std::move(down[bi++]);         // g down: sigmoid
            // The two elementwise ops the down side owns.  Order and formula are
            // the per-branch code's: tanh on the decay branch, sigmoid on the
            // gate's (O10 applies its sigmoid on the down projection).
            for (float& z : tw) z = tanhf(z);
            for (float& z : tg) z = rwkv_sigmoid(z);
            um.push_back(&m_.rwkv_w2[l]); ux.push_back(&tw);
            if (have_vres) { um.push_back(&m_.rwkv_v2[l]); ux.push_back(&tv); }
            um.push_back(&m_.rwkv_a2[l]); ux.push_back(&ta);
            if (have_gate) { um.push_back(&m_.rwkv_g2[l]); ux.push_back(&tg); }
            vector<vector<float>> up = rwkv_mv_group(um, ux, 8);
            size_t ui = 0;
            const vector<float> uu = std::move(up[ui++]);      // decay logits
            vector<float> uv;
            if (have_vres) uv = std::move(up[ui++]);           // vres logits
            const vector<float>& ua = up[ui++];                // ICLR logits
            // O7: w = exp(-sigmoid(tanh(xw W_w1) W_w2 + w0) / sqrt(e)) is in
            // (0.545, 1]; it multiplies the state once per token, which is why
            // W-1 stores 1-w rather than w.
            vector<float> w((size_t)dim);
            for (int i = 0; i < dim; ++i)
                w[i] = expf(-rwkv_sigmoid(uu[i] + w0[i]) * kRwkvDecayC);
            // O8: the value residual.  Layer 0 records v_first (its v_lora is
            // absent by construction); every later layer pulls v towards it.
            if (l == 0) {
                rwkv_state_.v_first = v;
                rwkv_state_.have_v_first = true;
            } else if (have_vres) {
                const float* vf = rwkv_state_.v_first.data();
                for (int i = 0; i < dim; ++i)
                    v[i] += (vf[i] - v[i]) * rwkv_sigmoid(uv[i] + v0[i]);
            }
            // O9: ICLR's gate, sigmoid on the *up* projection (unlike O10).
            vector<float> a((size_t)dim);
            for (int i = 0; i < dim; ++i) a[i] = rwkv_sigmoid(ua[i] + a0[i]);
            // O10: the optional time-mix gate.  Its sigmoid is on the down
            // projection and key 16 says whether the model has one.
            vector<float> g;
            if (have_gate) g = std::move(up[ui++]);
            // O11/O12: kk is the L2-normalised key, k the a-modulated one.
            const float* sk = m_.rwkv_scale.data() + (size_t)l * 3 * dim;
            const float* kk_s = sk;
            const float* ka_s = sk + dim;
            const float* rk_s = sk + 2 * dim;
            vector<float> kk((size_t)dim);
            for (int i = 0; i < dim; ++i) kk[i] = k[i] * kk_s[i];
            for (int i = 0; i < dim; ++i) k[i] += (a[i] - 1.0f) * k[i] * ka_s[i];
            for (int h = 0; h < H; ++h) {
                float* z = kk.data() + (size_t)h * hs;
                double ss = 0.0;
                for (int i = 0; i < hs; ++i) ss += (double)z[i] * (double)z[i];
                const float nrm = (float)sqrt(ss);
                const float inv = 1.0f / (nrm > 1e-12f ? nrm : 1e-12f);
                for (int i = 0; i < hs; ++i) z[i] *= inv;
            }
            // O13/O14: the recurrence.  Per head, with S[i][j] indexed the way
            // the oracle and llama.cpp's wkv7 kernel index it -- the row i
            // pairs with v and r, the column j with w, kk and k:
            //   sa_i  = <S[i], kk>              (pre-update state)
            //   S[i]  = S[i]*w + v_i*k + sa_i*(kk*a)
            //   out_i = <S[i], r>
            vector<float> y((size_t)dim);
            {
                float* S = rwkv_state_.wkv.data() + (size_t)l * H * hdim;
                for (int h = 0; h < H; ++h) {
                    float* Sh = S + (size_t)h * hdim;
                    const float* kkh = kk.data() + (size_t)h * hs;
                    const float* ah = a.data() + (size_t)h * hs;
                    const float* kh = k.data() + (size_t)h * hs;
                    const float* vh = v.data() + (size_t)h * hs;
                    const float* wh = w.data() + (size_t)h * hs;
                    const float* rh = r.data() + (size_t)h * hs;
                    for (int i = 0; i < hs; ++i) {
                        double acc = 0.0;
                        const float* Si = Sh + (size_t)i * hs;
                        for (int j = 0; j < hs; ++j) acc -= (double)Si[j] * (double)kkh[j];
                        sa[i] = (float)acc;
                    }
                    for (int i = 0; i < hs; ++i) {
                        float* Si = Sh + (size_t)i * hs;
                        const float vi = vh[i], s = sa[i];
                        for (int j = 0; j < hs; ++j)
                            Si[j] = Si[j] * wh[j] + vi * kh[j] + s * (kkh[j] * ah[j]);
                    }
                    float* yh = y.data() + (size_t)h * hs;
                    for (int i = 0; i < hs; ++i) {
                        const float* Si = Sh + (size_t)i * hs;
                        double acc = 0.0;
                        for (int j = 0; j < hs; ++j) acc += (double)Si[j] * (double)rh[j];
                        yh[i] = (float)acc;
                    }
                }
            }
            // O15: per-head group norm over the 64 channels, eps 64e-5 (the
            // architecture's own, not key 16's), then the ln_x affine.
            {
                const float* lnw = m_.rwkv_ln_x.data() + (size_t)l * 2 * dim;
                const float* lnb = lnw + dim;
                for (int h = 0; h < H; ++h) {
                    float* z = y.data() + (size_t)h * hs;
                    double sum = 0.0;
                    for (int i = 0; i < hs; ++i) sum += z[i];
                    const float mu = (float)(sum / hs);
                    double vs = 0.0;
                    for (int i = 0; i < hs; ++i) { const double d = (double)z[i] - (double)mu; vs += d * d; }
                    const float inv = 1.0f / sqrtf((float)(vs / hs) + kRwkvGroupEps);
                    for (int i = 0; i < hs; ++i)
                        z[i] = (z[i] - mu) * inv * lnw[(size_t)h * hs + i] + lnb[(size_t)h * hs + i];
                }
            }
            // O16: the r_k bonus.  One scalar per head, sum(k*r*r_k) over the
            // head's channels, added to every channel as v*rk.
            for (int h = 0; h < H; ++h) {
                double acc = 0.0;
                for (int i = 0; i < hs; ++i) {
                    const size_t o = (size_t)h * hs + i;
                    acc += (double)k[o] * (double)r[o] * (double)rk_s[o];
                }
                const float rk = (float)acc;
                for (int i = 0; i < hs; ++i) {
                    const size_t o = (size_t)h * hs + i;
                    y[o] += v[o] * rk;
                }
            }
            // O17-O19: gate, W_o, and the residual on the engine's grid.
            if (have_gate)
                for (int i = 0; i < dim; ++i) y[i] *= g[i];
            x = add_q16(x, rwkv_mv(m_.wo[l], y));
            // O20-O25: the channel mix.  attn_norm_2 has its own token shift,
            // one lerp vector, then key matvec -> relu^2 -> value matvec.
            vector<float> xn2 = rwkv_layernorm(x, m_.rwkv_att_norm_2.data() + (size_t)l * 2 * dim, eps);
            float* sh2 = rwkv_state_.shift_ffn.data() + (size_t)l * dim;
            const float* cml = m_.rwkv_cm_lerp.data() + (size_t)l * dim;
            vector<float> xk2((size_t)dim);
            for (int i = 0; i < dim; ++i) {
                const float sx = sh2[i] - xn2[i];
                xk2[i] = xn2[i] + sx * cml[i];
            }
            memcpy(sh2, xn2.data(), sizeof(float) * dim);
            // relu^2: the square is the one place the block amplifies instead
            // of attenuating (Q-1: peak 91.95 against the q16 rail's 128).
            // On VC4 the key is chunked (split_rwkv) so each matvec texture is
            // <=2048 rows; the two relu^2 halves are recombined by the value halves.
            float hmax = 0.0f;
            vector<float> cm_out;
            if (m_.ffn_chunked) {
                vector<float> ha = rwkv_mv(m_.w1a[l], xk2);
                vector<float> hb = rwkv_mv(m_.w1b[l], xk2);
                for (float& z : ha) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                for (float& z : hb) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                cm_out = add_q16(rwkv_mv(m_.w2a[l], ha), rwkv_mv(m_.w2b[l], hb));
            } else {
                vector<float> h = rwkv_mv(m_.w1[l], xk2);
                for (float& z : h) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                cm_out = rwkv_mv(m_.w2[l], h);
            }
            x = add_q16(x, cm_out);
            // RWKV_DUMP: the peaks the block has to keep inside its rails --
            // the recurrence's |S|max (f32 today, S-1), the residual (q16 rail
            // 128, R-1) and the relu^2 hidden (Q-1, the tensor closest to the
            // rail) -- plus the decay and gate extremes, which P6's GPU port
            // needs as per-layer references.  The same numbers are in
            // ref_rwkv.py --measure's per-layer table, so a layer that drifts
            // says where.  P2 parked these under STAGE_DUMP; P4 gave them
            // their own switch (docs/test-hooks.md).
            static const bool rwkv_diag = env_set("RWKV_DUMP");
            if (rwkv_diag) {
                const float* S = rwkv_state_.wkv.data() + (size_t)l * H * hdim;
                float smax = 0.0f, xmax = 0.0f, wmax = 0.0f, amax = 0.0f, gmax = 0.0f;
                for (size_t i = 0; i < (size_t)H * hdim; ++i) smax = std::max(smax, fabsf(S[i]));
                for (float v : x) xmax = std::max(xmax, fabsf(v));
                for (float v : w) wmax = std::max(wmax, v);      // w in (0.545, 1]
                for (float v : a) amax = std::max(amax, fabsf(v));
                if (have_gate)
                    for (float v : g) gmax = std::max(gmax, fabsf(v));
                cerr << "RWKV l=" << l << " state_max=" << smax
                     << " resid_max=" << xmax << " sqrelu_max=" << hmax
                     << " w_max=" << wmax << " a_max=" << amax;
                if (have_gate) cerr << " g_max=" << gmax;
                cerr << "\n";
            }
            if (prof) {
                const double up = gpu_.rwkv_prof.upload_ms - up0;
                const double dr = gpu_.rwkv_prof.draw_ms - dr0;
                const double rd = gpu_.rwkv_prof.read_ms - rd0;
                const double tot = elapsed_ms(tl0, Clock::now());
                t_up += up; t_draw += dr; t_read += rd; t_tot += tot;
                fprintf(stderr, "RWKV_PROFILE cpu layer=%d total=%.2f mv_up=%.2f mv_draw=%.2f "
                                "mv_read=%.2f elem=%.2f\n", l, tot, up, dr, rd, tot - up - dr - rd);
            }
        }
        if (prof)
            fprintf(stderr, "RWKV_PROFILE cpu token total=%.2f mv_up=%.2f mv_draw=%.2f "
                            "mv_read=%.2f elem=%.2f\n", t_tot, t_up, t_draw, t_read,
                    t_tot - t_up - t_draw - t_read);
        return x;
    }

// ---- RWKV-7 chunked prefill (P1, task-13) ----------------------------------
//
// The serial loop above walks one token through all L layers, so every
// projection is drawn and read back once per token: on Xe that is ~170 host
// syncs and on the board ~170 uploads per token, and a 512-token prompt is
// minutes.  Nothing in a projection depends on the previous token, so the *same*
// block can be evaluated one layer at a time for a batch of tokens: one draw per
// weight for the whole batch (GPU::matvec_batch_group) and one read-back.
//
// What stays serial, and must: the token shift (a running state per layer), the
// WKV7 recurrence and everything elementwise (tanh/sigmoid/exp, the L2 and group
// norms, the r_k bonus, relu^2, the residual adds).  The plan's split -- the
// recurrence is 1.1-2.0% of the MACs -- is why this is where the work goes.
//
// The numbers are the serial loop's, bit for bit: the batch kernel is
// matvec_q8's text with the input row substituted, and each input keeps its own
// 2^k normalisation (the rows are grouped by k inside matvec_batch_group, one
// draw per group, exactly the uniform the serial path binds).  So the layer
// arithmetic below is transcribed from forward_rwkv unchanged, and the state
// this leaves behind (WKV, both shift vectors, v_first) is the serial loop's.
vector<vector<float>> Session::forward_rwkv_batch(const vector<int>& toks){
        const int N = (int)toks.size();
        const int dim = m_.dim, L = m_.layers, H = m_.heads, hs = m_.rwkv.head_size;
        const float eps = m_.rwkv.norm_eps;
        const size_t hdim = (size_t)hs * (size_t)hs;
        // O1/O2 pre-roll: the embedding stays off the q16 grid (see forward_rwkv)
        // and layer 0's pre_norm is applied here, once per token.
        vector<vector<float>> x(N);
        for (int t = 0; t < N; ++t) {
                x[t] = embedding(toks[t], true);
                if (!m_.rwkv_tok_norm.empty())
                        x[t] = rwkv_layernorm(x[t], m_.rwkv_tok_norm.data(), eps);
        }
        // O8's cross-layer state is per *token*: layer 0 records v_first for the
        // token it is running and every later layer of that same token reads it.
        // One vector for the batch would mix in another token (and after the
        // batch the last token's value is what a decode continues from).
        vector<vector<float>> vfirst(N);
        vector<float> sa((size_t)hs);
        for (int l = 0; l < L; ++l) {
            if (g_stop) break;
            const float* w0 = m_.rwkv_bias.data() + (size_t)l * 3 * dim;
            const float* a0 = w0 + dim;
            const float* v0 = a0 + dim;
            const bool have_gate = m_.rwkv.gate && l < (int)m_.rwkv_g1.size() &&
                                   m_.rwkv_g1[l].rows > 0;
            const bool have_vres = l > 0 && rwkv_state_.have_v_first &&
                                   m_.rwkv_v1[l].rows > 0;
            // O1-O3, per token and in token order (the shift is a running state)
            vector<vector<float>> xr(N), xw(N), xk(N), xv(N), xa(N), xg(N);
            vector<const vector<float>*> pxr(N), pxw(N), pxk(N), pxv(N), pxa(N), pxg(N);
            for (int t = 0; t < N; ++t) {
                vector<float> xn = rwkv_layernorm(x[t], m_.rwkv_att_norm.data() + (size_t)l * 2 * dim, eps);
                float* sh = rwkv_state_.shift_att.data() + (size_t)l * dim;
                const float* lp = m_.rwkv_lerp.data() + (size_t)l * 6 * dim;
                xr[t].resize((size_t)dim); xw[t].resize((size_t)dim); xk[t].resize((size_t)dim);
                xv[t].resize((size_t)dim); xa[t].resize((size_t)dim); xg[t].resize((size_t)dim);
                for (int i = 0; i < dim; ++i) {
                    const float sx = sh[i] - xn[i];
                    xr[t][i] = xn[i] + sx * lp[i];
                    xw[t][i] = xn[i] + sx * lp[dim + i];
                    xk[t][i] = xn[i] + sx * lp[2 * dim + i];
                    xv[t][i] = xn[i] + sx * lp[3 * dim + i];
                    xa[t][i] = xn[i] + sx * lp[4 * dim + i];
                    xg[t][i] = xn[i] + sx * lp[5 * dim + i];
                }
                memcpy(sh, xn.data(), sizeof(float) * dim);
                pxr[t] = &xr[t]; pxw[t] = &xw[t]; pxk[t] = &xk[t];
                pxv[t] = &xv[t]; pxa[t] = &xa[t]; pxg[t] = &xg[t];
            }
            // O4-O6: r/k/v, one draw per projection for the whole batch.
            const GPU::MvBatch rkv[3] = {{&m_.rwkv_r[l], &pxr},
                                         {&m_.rwkv_k[l], &pxk},
                                         {&m_.rwkv_v[l], &pxv}};
            vector<vector<vector<float>>> rkv_out =
                gpu_.matvec_batch_group(vector<GPU::MvBatch>(rkv, rkv + 3));
            vector<vector<float>> r = std::move(rkv_out[0]), k = std::move(rkv_out[1]);
            vector<vector<float>> v = std::move(rkv_out[2]);
            // O7-O10: the four low-rank down projections, one read-back.
            vector<GPU::MvBatch> dg;
            vector<const vector<float>*> ptw, ptv, pta, ptg;
            dg.push_back({&m_.rwkv_w1[l], &pxw});
            if (have_vres) dg.push_back({&m_.rwkv_v1[l], &pxv});
            dg.push_back({&m_.rwkv_a1[l], &pxa});
            if (have_gate) dg.push_back({&m_.rwkv_g1[l], &pxg});
            vector<vector<vector<float>>> down = gpu_.matvec_batch_group(dg);
            size_t bi = 0;
            vector<vector<float>> tw = std::move(down[bi++]);          // w down: tanh
            vector<vector<float>> tv;
            if (have_vres) tv = std::move(down[bi++]);                 // v down: pass-through
            vector<vector<float>> ta = std::move(down[bi++]);          // a down: pass-through
            vector<vector<float>> tg;
            if (have_gate) tg = std::move(down[bi++]);                 // g down: sigmoid
            for (int t = 0; t < N; ++t) {
                for (float& z : tw[t]) z = tanhf(z);
                if (have_gate)
                    for (float& z : tg[t]) z = rwkv_sigmoid(z);
            }
            ptw.resize(N); ptv.resize(N); pta.resize(N); ptg.resize(N);
            for (int t = 0; t < N; ++t) {
                ptw[t] = &tw[t]; pta[t] = &ta[t];
                if (have_vres) ptv[t] = &tv[t];
                if (have_gate) ptg[t] = &tg[t];
            }
            // O7-O10 up projections, one read-back.
            vector<GPU::MvBatch> ug;
            ug.push_back({&m_.rwkv_w2[l], &ptw});
            if (have_vres) ug.push_back({&m_.rwkv_v2[l], &ptv});
            ug.push_back({&m_.rwkv_a2[l], &pta});
            if (have_gate) ug.push_back({&m_.rwkv_g2[l], &ptg});
            vector<vector<vector<float>>> up = gpu_.matvec_batch_group(ug);
            size_t ui = 0;
            vector<vector<float>> uu = std::move(up[ui++]);            // decay logits
            vector<vector<float>> uv;
            if (have_vres) uv = std::move(up[ui++]);                   // vres logits
            vector<vector<float>> ua = std::move(up[ui++]);            // ICLR logits
            vector<vector<float>> gg;
            if (have_gate) gg = std::move(up[ui++]);
            // O7-O19, per token: the decay, the value residual, the ICLR gate,
            // the key norm and delta, the recurrence, the group norm, the r_k
            // bonus, the gate and W_o's input row.
            vector<vector<float>> y(N);
            vector<const vector<float>*> py(N);
            float hmax = 0.0f, smax = 0.0f, xmax = 0.0f, wmax = 0.0f, amax = 0.0f, gmax = 0.0f;
            static const bool rwkv_diag = env_set("RWKV_DUMP");
            for (int t = 0; t < N; ++t) {
                vector<float> w((size_t)dim);
                for (int i = 0; i < dim; ++i)
                    w[i] = expf(-rwkv_sigmoid(uu[t][i] + w0[i]) * kRwkvDecayC);
                if (l == 0) {
                    vfirst[t] = v[t];
                } else if (have_vres) {
                    const float* vf = vfirst[t].data();
                    for (int i = 0; i < dim; ++i)
                        v[t][i] += (vf[i] - v[t][i]) * rwkv_sigmoid(uv[t][i] + v0[i]);
                }
                vector<float> a((size_t)dim);
                for (int i = 0; i < dim; ++i) a[i] = rwkv_sigmoid(ua[t][i] + a0[i]);
                const float* sk = m_.rwkv_scale.data() + (size_t)l * 3 * dim;
                const float* kk_s = sk;
                const float* ka_s = sk + dim;
                const float* rk_s = sk + 2 * dim;
                vector<float> kk((size_t)dim);
                for (int i = 0; i < dim; ++i) kk[i] = k[t][i] * kk_s[i];
                for (int i = 0; i < dim; ++i) k[t][i] += (a[i] - 1.0f) * k[t][i] * ka_s[i];
                for (int h = 0; h < H; ++h) {
                    float* z = kk.data() + (size_t)h * hs;
                    double ss = 0.0;
                    for (int i = 0; i < hs; ++i) ss += (double)z[i] * (double)z[i];
                    const float nrm = (float)sqrt(ss);
                    const float inv = 1.0f / (nrm > 1e-12f ? nrm : 1e-12f);
                    for (int i = 0; i < hs; ++i) z[i] *= inv;
                }
                vector<float> yt((size_t)dim);
                {
                    float* S = rwkv_state_.wkv.data() + (size_t)l * H * hdim;
                    for (int h = 0; h < H; ++h) {
                        float* Sh = S + (size_t)h * hdim;
                        const float* kkh = kk.data() + (size_t)h * hs;
                        const float* ah = a.data() + (size_t)h * hs;
                        const float* kh = k[t].data() + (size_t)h * hs;
                        const float* vh = v[t].data() + (size_t)h * hs;
                        const float* wh = w.data() + (size_t)h * hs;
                        const float* rh = r[t].data() + (size_t)h * hs;
                        for (int i = 0; i < hs; ++i) {
                            double acc = 0.0;
                            const float* Si = Sh + (size_t)i * hs;
                            for (int j = 0; j < hs; ++j) acc -= (double)Si[j] * (double)kkh[j];
                            sa[i] = (float)acc;
                        }
                        for (int i = 0; i < hs; ++i) {
                            float* Si = Sh + (size_t)i * hs;
                            const float vi = vh[i], s = sa[i];
                            for (int j = 0; j < hs; ++j)
                                Si[j] = Si[j] * wh[j] + vi * kh[j] + s * (kkh[j] * ah[j]);
                        }
                        float* yh = yt.data() + (size_t)h * hs;
                        for (int i = 0; i < hs; ++i) {
                            const float* Si = Sh + (size_t)i * hs;
                            double acc = 0.0;
                            for (int j = 0; j < hs; ++j) acc += (double)Si[j] * (double)rh[j];
                            yh[i] = (float)acc;
                        }
                    }
                }
                {
                    const float* lnw = m_.rwkv_ln_x.data() + (size_t)l * 2 * dim;
                    const float* lnb = lnw + dim;
                    for (int h = 0; h < H; ++h) {
                        float* z = yt.data() + (size_t)h * hs;
                        double sum = 0.0;
                        for (int i = 0; i < hs; ++i) sum += z[i];
                        const float mu = (float)(sum / hs);
                        double vs = 0.0;
                        for (int i = 0; i < hs; ++i) { const double d = (double)z[i] - (double)mu; vs += d * d; }
                        const float inv = 1.0f / sqrtf((float)(vs / hs) + kRwkvGroupEps);
                        for (int i = 0; i < hs; ++i)
                            z[i] = (z[i] - mu) * inv * lnw[(size_t)h * hs + i] + lnb[(size_t)h * hs + i];
                    }
                }
                for (int h = 0; h < H; ++h) {
                    double acc = 0.0;
                    for (int i = 0; i < hs; ++i) {
                        const size_t o = (size_t)h * hs + i;
                        acc += (double)k[t][o] * (double)r[t][o] * (double)rk_s[o];
                    }
                    const float rk = (float)acc;
                    for (int i = 0; i < hs; ++i) {
                        const size_t o = (size_t)h * hs + i;
                        yt[o] += v[t][o] * rk;
                    }
                }
                if (have_gate)
                    for (int i = 0; i < dim; ++i) yt[i] *= gg[t][i];
                y[t] = std::move(yt);
                py[t] = &y[t];
                if (rwkv_diag) {
                    const float* S = rwkv_state_.wkv.data() + (size_t)l * H * hdim;
                    for (size_t i = 0; i < (size_t)H * hdim; ++i) smax = std::max(smax, fabsf(S[i]));
                    for (float z : x[t]) xmax = std::max(xmax, fabsf(z));
                    for (float z : w) wmax = std::max(wmax, z);
                    for (float z : a) amax = std::max(amax, fabsf(z));
                    if (have_gate) for (float z : gg[t]) gmax = std::max(gmax, fabsf(z));
                }
            }
            // O18-O19: W_o for the whole batch, then the residual on the q16 grid.
            const GPU::MvBatch wo[1] = {{&m_.wo[l], &py}};
            vector<vector<vector<float>>> wo_out =
                gpu_.matvec_batch_group(vector<GPU::MvBatch>(wo, wo + 1));
            for (int t = 0; t < N; ++t) x[t] = add_q16(x[t], wo_out[0][t]);
            // O20-O25: the channel mix, per-token norm and lerp, batched key.
            vector<vector<float>> xk2(N);
            vector<const vector<float>*> pxk2(N);
            for (int t = 0; t < N; ++t) {
                vector<float> xn2 = rwkv_layernorm(x[t], m_.rwkv_att_norm_2.data() + (size_t)l * 2 * dim, eps);
                float* sh2 = rwkv_state_.shift_ffn.data() + (size_t)l * dim;
                const float* cml = m_.rwkv_cm_lerp.data() + (size_t)l * dim;
                xk2[t].resize((size_t)dim);
                for (int i = 0; i < dim; ++i) {
                    const float sx = sh2[i] - xn2[i];
                    xk2[t][i] = xn2[i] + sx * cml[i];
                }
                memcpy(sh2, xn2.data(), sizeof(float) * dim);
                pxk2[t] = &xk2[t];
            }
            vector<vector<float>> cm_out(N);
            if (m_.ffn_chunked) {
                // VC4: the key is split so each weight texture is <= 2048 rows;
                // the two relu^2 halves are recombined by the value halves, on
                // the q16 grid (add_q16), exactly as the serial loop does.
                const GPU::MvBatch ka[1] = {{&m_.w1a[l], &pxk2}};
                const GPU::MvBatch kb[1] = {{&m_.w1b[l], &pxk2}};
                vector<vector<vector<float>>> ha = gpu_.matvec_batch_group(vector<GPU::MvBatch>(ka, ka + 1));
                vector<vector<vector<float>>> hb = gpu_.matvec_batch_group(vector<GPU::MvBatch>(kb, kb + 1));
                vector<const vector<float>*> pha(N), phb(N);
                for (int t = 0; t < N; ++t) {
                    for (float& z : ha[0][t]) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                    for (float& z : hb[0][t]) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                    pha[t] = &ha[0][t];
                    phb[t] = &hb[0][t];
                }
                const GPU::MvBatch va[1] = {{&m_.w2a[l], &pha}};
                const GPU::MvBatch vb[1] = {{&m_.w2b[l], &phb}};
                vector<vector<vector<float>>> ca = gpu_.matvec_batch_group(vector<GPU::MvBatch>(va, va + 1));
                vector<vector<vector<float>>> cb = gpu_.matvec_batch_group(vector<GPU::MvBatch>(vb, vb + 1));
                for (int t = 0; t < N; ++t) cm_out[t] = add_q16(ca[0][t], cb[0][t]);
            } else {
                const GPU::MvBatch kk1[1] = {{&m_.w1[l], &pxk2}};
                vector<vector<vector<float>>> h = gpu_.matvec_batch_group(vector<GPU::MvBatch>(kk1, kk1 + 1));
                vector<const vector<float>*> ph(N);
                for (int t = 0; t < N; ++t) {
                    for (float& z : h[0][t]) { z = z > 0.0f ? z * z : 0.0f; hmax = std::max(hmax, z); }
                    ph[t] = &h[0][t];
                }
                const GPU::MvBatch w2g[1] = {{&m_.w2[l], &ph}};
                cm_out = gpu_.matvec_batch_group(vector<GPU::MvBatch>(w2g, w2g + 1))[0];
            }
            for (int t = 0; t < N; ++t) x[t] = add_q16(x[t], cm_out[t]);
            rwkv_state_.have_v_first = l == 0 ? true : rwkv_state_.have_v_first;
            if (rwkv_diag)
                cerr << "RWKV l=" << l << " n=" << N << " state_max=" << smax
                     << " resid_max=" << xmax << " sqrelu_max=" << hmax
                     << " w_max=" << wmax << " a_max=" << amax;
            if (rwkv_diag && have_gate) cerr << " g_max=" << gmax;
            if (rwkv_diag) cerr << "\n";
        }
        // The state the next decode continues from is the last token's.
        if (!vfirst.empty() && rwkv_state_.have_v_first) rwkv_state_.v_first = vfirst.back();
        return x;
    }

bool Session::prefill_batch_ok() const {
        static const bool on = env_flag("RWKV_PREFILL");
        if (!on || !m_.is_rwkv()) return false;
        // The batched kernel is the q8 one (matvec_q8_batch): a container whose
        // matrices do not advertise WStage::Batch keeps the serial prefill
        // rather than answering from a kernel that cannot decode its weights.
        if (rwkv_gpu_) return false;   // the fused loop is one draw per op; no batch kernel
        const auto batch_ok = [](const vector<Mat>& v) {
            for (const Mat& m : v) if (m.rows > 0 && !wf_can(m.type, WStage::Batch)) return false;
            return true;
        };
        if (!batch_ok(m_.rwkv_r) || !batch_ok(m_.rwkv_k) || !batch_ok(m_.rwkv_v) || !batch_ok(m_.wo)) return false;
        if (!batch_ok(m_.rwkv_w1) || !batch_ok(m_.rwkv_w2) || !batch_ok(m_.rwkv_a1) || !batch_ok(m_.rwkv_a2)) return false;
        if (!batch_ok(m_.rwkv_v1) || !batch_ok(m_.rwkv_v2) || !batch_ok(m_.rwkv_g1) || !batch_ok(m_.rwkv_g2)) return false;
        if (m_.ffn_chunked && (!batch_ok(m_.w1a) || !batch_ok(m_.w1b) || !batch_ok(m_.w2a) || !batch_ok(m_.w2b))) return false;
        return true;
    }

int Session::prefill_batch_size() const {
        static const int n = env_int("RWKV_PREFILL_N", 16);
        // Clamped so the staging block (groups * n texture rows, groups <= 4)
        // stays inside the driver's target limit on every machine: VC4's is
        // 2048 rows, and the widest group is the four low-rank downs.
        if (n < 1) return 16;
        return n > 128 ? 128 : n;
    }

// One prefill call: the batch through the block, and the last token's logits
// when the prompt ends here (the caller's only consumer).  A batch that does not
// contain the last prompt token returns no logits, exactly like step(...,
// false), so the caller's loop and the LOGITS_OUT dump are unchanged.
vector<float> Session::prefill_batch(const vector<int>& toks, bool want_logits){
        if (toks.empty()) return {};
        // The serial fallback has to finish the job, not just move the state: the
        // caller asked for the last token's logits and would otherwise answer
        // from an empty vector.
        const auto serial = [&]() -> vector<float> {
                vector<float> out;
                for (size_t i = 0; i < toks.size(); ++i) {
                        const bool want = want_logits && i + 1 == toks.size();
                        vector<float> lg = step(toks[i], (int)i, want);
                        if (want) out = std::move(lg);
                }
                return out;
        };
        if (!prefill_batch_ok()) return serial();
        // The one-time guard: one real projection through both paths before any
        // token is committed.  RWKV_PREFILL defaults off, but a machine where the
        // batched kernel is wrong (the VC4 tall-draw question, or a driver whose
        // 2^k fold differs) would answer with wrong numbers the moment someone
        // sets it -- so the kernel has to prove itself on this machine first, and
        // a refusal costs the serial path rather than the numbers.
        if (rwkv_batch_state_ == 0) {
                vector<float> probe = embedding(toks[0], true);
                if (!m_.rwkv_tok_norm.empty())
                        probe = rwkv_layernorm(probe, m_.rwkv_tok_norm.data(), m_.rwkv.norm_eps);
                const vector<float> one = rwkv_mv(m_.rwkv_r[0], probe);
                const vector<const vector<float>*> in(1, &probe);
                const GPU::MvBatch g[1] = {{&m_.rwkv_r[0], &in}};
                const vector<vector<vector<float>>> bat =
                        gpu_.matvec_batch_group(vector<GPU::MvBatch>(g, g + 1));
                int ndiff = 0;
                float mxd = 0.0f;
                if (!bat.empty() && !bat[0].empty())
                        for (size_t j = 0; j < one.size() && j < bat[0][0].size(); ++j) {
                                const float d = fabsf(one[j] - bat[0][0][j]);
                                if (d > 0.0f) { ++ndiff; mxd = std::max(mxd, d); }
                        }
                else
                        ndiff = (int)one.size();
                // A tolerance, not exactness.  On Xe the batched row is the
                // serial row bit for bit; on VC4 the same kernel text reads its
                // staging row through a *uniform* coordinate, which the driver
                // rounds differently from the literals the serial kernel bakes in
                // (the effect AGENTS.md records as "a constant is not a uniform"),
                // so the board measures max|d| 0.37 on a projection whose rows are
                // O(10) -- with top-1 and the whole top-5 order unchanged -- where
                // the broken row mapping this guards against was 47.  The line is
                // therefore drawn at 1.0, and an accepted difference is reported
                // rather than hidden.
                rwkv_batch_state_ = (mxd <= 1.0f) ? 1 : -1;
                if (mxd > 0.0f)
                        cerr << "RWKV_PREFILL: the batched matvec differs from the serial one on this "
                             << "machine by max|d| " << mxd << " over " << one.size() << " rows ("
                             << ndiff << " differ); "
                             << (rwkv_batch_state_ > 0
                                     ? "within the 1.0 tolerance, so batching is used"
                                     : "past the 1.0 tolerance, so the serial prefill is used")
                             << "\n";
        }
        if (rwkv_batch_state_ < 0) return serial();
        vector<vector<float>> x = forward_rwkv_batch(toks);
        if (g_stop || !want_logits || x.empty()) return {};
        return gpu_.logits(m_.logits_tiles, rwkv_head_input(x.back()));
    }

vector<float> Session::rwkv_head_input(const vector<float>& x){
        vector<float> xf = rwkv_layernorm(x, m_.rwkv_out_norm.data(), m_.rwkv.norm_eps);
        // RWKV_DUMP: the final LayerNorm's peak is the number metadata key 17
        // exists for (g1d: 315.6 = 2.5x the q16 rail); print it before the
        // *2^-k store so a container whose k no longer fits is visible.
        static const bool rwkv_head_diag = env_set("RWKV_DUMP");
        if (rwkv_head_diag) {
            float mx = 0.0f;
            for (float v : xf) mx = std::max(mx, fabsf(v));
            cerr << "RWKV final_ln_max=" << mx << "\n";
        }
        // Metadata key 17: the final LayerNorm's output reaches 315.6 on g1d --
        // 2.5x the q16 rail -- so the container states an exponent k, the head's
        // input is written x*2^-k, and every logits tile's scale already carries
        // the 2^k (folded at pack time).  Applying it to the tile scale here as
        // well would count it twice.
        if (m_.logit_shift > 0) {
                const float s = ldexpf(1.0f, -m_.logit_shift);
                for (float& v : xf) v *= s;
        }
        return xf;
    }

// P6: the fused GPU layer loop.  The embedding and token norm stay on the host
// in f32 (g1d's embedding rows peak at 0.08, where the q16 grid leaves 2-3 bits
// -- the trap the CPU path's exact_f32 embedding avoids); after the token norm
// the row is O(1), so the q16 residual store is the grid the CPU uses too.  The
// whole layer loop then runs on textures; only the final residual crosses back.
vector<float> Session::forward_gpu_rwkv(int token){
        vector<float> x = embedding(token, true);
        if (!m_.rwkv_tok_norm.empty())
            x = rwkv_layernorm(x, m_.rwkv_tok_norm.data(), m_.rwkv.norm_eps);
        GLuint resid = gpu_.rwkv_upload_x0(x);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) break;
            float* cs = rwkv_wkv_cpu_
                ? rwkv_state_.wkv.data() + (size_t)l * m_.heads * m_.rwkv.head_size * m_.rwkv.head_size
                : nullptr;
            resid = gpu_.rwkv_block(l, m_, resid, cs);
        }
        gpu_.rwkv_flip();
        gpu_.rwkv_prof_fused_summary();
        return gpu_.rwkv_read_residual(resid);
    }

vector<float> Session::step_rwkv(int token, bool want_logits){
        vector<float> x = rwkv_gpu_ ? forward_gpu_rwkv(token) : forward_rwkv(token);
        if (g_stop || !want_logits) return {};
        // RWKV_PROFILE: the head is its own pass (43 tiles x draw + one read-back).
        static const bool prof = env_set("RWKV_PROFILE");
        const auto th0 = Clock::now();
        vector<float> out = gpu_.logits(m_.logits_tiles, rwkv_head_input(x));
        if (prof)
                fprintf(stderr, "RWKV_PROFILE head total=%.2f (logits draw+readback)\n",
                        elapsed_ms(th0, Clock::now()));
        return out;
    }

// Defined next to read_hidden() further down; the fused loop dumps through it
// so its stages are diffable against the CPU path's float dumps.
static vector<float> decode_act_row(const vector<uint8_t>& b, int texels, XFormat f);

GLuint Session::forward_gpu_hidden(int token, int pos, bool want_logits){
        bool prof = env_set("LAYER_PROFILE");
        auto pt = Clock::now();
        // The laps print the draw count as well as the time: on VC4 a pass costs
        // a fixed ~1 ms of draw overhead before its arithmetic, so "which phase"
        // and "how many draws" are the same question when the profile says
        // attention is 45% of a block whose FLOPs are tiny.
        long draws_at_last_mark = (long)gpu_.draws();
        // One counter region per stage.  A mark already synchronises, so its ms
        // is QPU work plus submit plus finish; the counters split that into the
        // part an optimisation can reach (fragment cycles, TMU stalls) and the
        // part it cannot (the per-draw overhead).  A stage whose GPU share is
        // near zero is not slow, it is expensive to launch.
        const bool counters = prof && gpu_.pc_available();
        auto mark = [&](const char* name, int l) {
            if (!prof) return;
            gpu_.sync();
            auto now = Clock::now();
            const long d = (long)gpu_.draws();
            const double ms = elapsed_ms(pt, now);
            unsigned long long idle = 0, frag = 0, exec = 0, quads = 0;
            bool have = false;
            if (counters) have = gpu_.pc_stage_end(idle, frag, exec, quads);
            cerr << "PROF l=" << l << " " << name << " " << ms
                 << " draws=" << (d - draws_at_last_mark);
            if (have) {
                // cycles / (qpus * MHz) = ms; the divisor is measured, not
                // assumed (12 QPU-equivalents at 300 MHz on this board).
                const double to_ms = 1.0 / (GPU::pc_qpus() * (double)GPU::pc_mhz() * 1000.0);
                const double gpu_ms = (double)frag * to_ms;
                cerr << " gpu=" << gpu_ms
                     << " (" << (ms > 0 ? (int)(100.0 * gpu_ms / ms) : 0) << "%)"
                     << " exec=" << ((double)exec * to_ms)
                     << " idle=" << ((double)idle * to_ms)
                     << " quads=" << quads;
            }
            cerr << "\n";
            // Arranged so a stage's region covers that stage only: begin here,
            // after the current pipeline is drained.
            if (counters) gpu_.pc_stage_begin();
            draws_at_last_mark = d;
            pt = now;
        };
        if (counters) gpu_.pc_stage_begin();

        // A replay probe's delta is a wall-time delta, and a wall-time delta
        // cannot say whether the extra pass cost QPU cycles or a finish.  This
        // brackets the *replay* with its own counter region and prints the same
        // columns the stage line does, so the two numbers can be compared:
        // dup_gpu ~= delta_wall means the pass is arithmetic, dup_gpu << delta
        // means the pass is its own synchronisation and belongs to fusion work.
        // DUP_PROFILE=1 turns it on; without it the replay is untouched.
        static const bool dup_prof = env_set("DUP_PROFILE");
        // DUP_PROFILE does not need LAYER_PROFILE: it is the profiler of a
        // replay, and a replay run with the stage marks off is the cleanest
        // case (no other region is open to return early).
        const bool dup_counters = dup_prof && gpu_.pc_available();
        auto replay = [&](const char* name, int layer, auto&& fn) {
            if (!dup_prof) { fn(); return; }
            const auto t0 = Clock::now();
            gpu_.sync();                       // start from a drained pipeline
            if (dup_counters) gpu_.pc_stage_begin();
            fn();
            gpu_.sync();
            const double ms = elapsed_ms(t0, Clock::now());
            unsigned long long idle = 0, frag = 0, exec = 0, quads = 0;
            const bool have = dup_counters && gpu_.pc_stage_end(idle, frag, exec, quads);
            if (have) {
                const double to_ms = 1.0 / (GPU::pc_qpus() * (double)GPU::pc_mhz() * 1000.0);
                const double gpu_ms = (double)frag * to_ms;
                cerr << "DUP l=" << layer << " " << name << " wall=" << ms
                     << " gpu=" << gpu_ms
                     << " (" << (ms > 0 ? (int)(100.0 * gpu_ms / ms) : 0) << "%)"
                     << " exec=" << ((double)exec * to_ms)
                     << " idle=" << ((double)idle * to_ms)
                     << " quads=" << quads << "\n";
            } else {
                cerr << "DUP l=" << layer << " " << name << " wall=" << ms << "\n";
            }
        };

        vector<float> emb = embedding(token);
        GLuint h = gpu_.upload_hidden(emb);
        if (pass_dup("embed")) replay("embed", -1, [&]{ emb = embedding(token); h = gpu_.upload_hidden(emb); });
        mark("embed", -1);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return h;
            GLuint xb = gpu_.rmsnorm(h, gpu_.rms_att_tex_[l], gpu_.xb_);
            if (pass_dup("rms_att")) replay("rms_att", l, [&]{ xb = gpu_.rmsnorm(h, gpu_.rms_att_tex_[l], gpu_.xb_); });
            mark("rms_att", l);
            // Dump as floats with the CPU path's printer, so GPU_LAYER=0 and
            // GPU_LAYER=1 dumps are diffable even though the GPU keeps its own
            // residual format (the byte dump of a target is layout-specific).
            if (getenv("STAGE_DUMP")) {
                vector<uint8_t> db = gpu_.read_bytes(gpu_.xb_, gpu_.xb_.w, 1);
                gpu_.dump_cpu_row(decode_act_row(db, gpu_.xb_.w, gpu_.act_fmt_), gpu_.act_fmt_, "xb", 4096);
            }
            if (m_.is_conv(l) && getenv("CONV_SKIP")) {
                gpu_.conv_copy(h, gpu_.hidden_b_);   // probe: conv layer = identity
            } else if (m_.is_conv(l)) {
                gpu_.conv_block(m_.conv_in_b[l], m_.conv_in_c[l], m_.conv_in_x[l],
                                m_.conv_out[l], xb, h, l, gpu_.hidden_b_);
                if (getenv("STAGE_DUMP"))
                    gpu_.dump_cpu_row(read_hidden(gpu_.hidden_b_), gpu_.act_fmt_, "x1", 4096);
                mark("conv", l);
            } else {
                GLuint att_tex = gpu_.attention_block(m_.qkv[l], xb, l, pos);
                if (getenv("STAGE_DUMP")) {
                    vector<uint8_t> db = gpu_.read_bytes(gpu_.att_out_, gpu_.att_out_.w, 1);
                    gpu_.dump_cpu_row(decode_act_row(db, gpu_.att_out_.w, gpu_.act_fmt_), gpu_.act_fmt_, "att", 4096);
                }
                mark("attn", l);
                gpu_.matvec_add(m_.wo[l], att_tex, h, gpu_.hidden_b_, gpu_.act_fmt_);
                if (matvec_dup("wo"))
                    replay("wo", l, [&]{ gpu_.matvec_add(m_.wo[l], att_tex, h, gpu_.hidden_b_, gpu_.act_fmt_); });
                if (getenv("STAGE_DUMP"))
                    gpu_.dump_cpu_row(read_hidden(gpu_.hidden_b_), gpu_.act_fmt_, "x1", 4096);
                mark("wo", l);
            }
            GLuint x1 = gpu_.hidden_b_.tex;
            GLuint xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
            if (getenv("STAGE_DUMP")) {
                vector<uint8_t> db = gpu_.read_bytes(gpu_.xbf_, gpu_.xbf_.w, 1);
                gpu_.dump_cpu_row(decode_act_row(db, gpu_.xbf_.w, gpu_.act_fmt_), gpu_.act_fmt_, "xbf", 4096);
            }
            if (pass_dup("rms_ffn")) replay("rms_ffn", l, [&]{ xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_); });
            mark("rms_ffn", l);
            int expert = -1;
            if (m_.is_moe() && getenv("MOE_FIXED")) {
                // Probe: skip the router and its read-back, pin one expert.
                expert = atoi(getenv("MOE_FIXED"));
            } else if (m_.is_moe()) {
                // The route is a 4-value read-back: one sync per layer per
                // token, the price of a discrete choice on a GLES2 pipeline.
                vector<float> gate = gpu_.matvec_from(m_.router[l], xbf);
                expert = pick_expert(gate);
                if (getenv("MOE_DUMP"))
                    cerr << "MOE l=" << l << " route=" << expert << " gate="
                         << gate[0] << "," << gate[1] << "," << gate[2] << "," << gate[3] << "\n";
            }
            const FfnMats f = ffn_mats(m_, l, expert);
            const int hs = m_.hid_shift(l);   // FFN hidden store scale (key 15)
            if (m_.ffn_chunked) {
                GLuint ha, hb;
                gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xbf,
                                       m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb, hs);
                if (matvec_dup("ffn"))
                    replay("ffn", l, [&]{ gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xbf,
                                           m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb, hs); });
                if (getenv("FFN_DUMP") && l == 0) {
                    gpu_.sync();
                    gpu_.dump_target(gpu_.ffn_h_a_, "ha", 8);
                    gpu_.dump_target(gpu_.ffn_h_b_, "hb", 8);
                }
                mark("ffn", l);
                // The two w2 halves are NOT replay-safe in place: w2b writes
                // gpu_.hidden_b_, which is exactly the texture `x1` names as
                // w2a's residual.  Replaying the pair after the fact fed w2a a
                // residual of the layer's *output* and doubled it (the greedy
                // answer moved).  Each replay therefore goes into its own
                // scratch target, and w2a's happens before w2b clobbers x1.
                gpu_.matvec_add(*f.w2a, ha, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                if (matvec_dup("w2"))
                    replay("w2a", l, [&]{ gpu_.matvec_add(*f.w2a, ha, x1, w2_replay_, gpu_.ffn_fmt_); });
                gpu_.matvec_add(*f.w2b, hb, gpu_.hidden_a_.tex, gpu_.hidden_b_, gpu_.ffn_fmt_);
                if (matvec_dup("w2"))
                    replay("w2b", l, [&]{ gpu_.matvec_add(*f.w2b, hb, gpu_.hidden_a_.tex, w2_replay_, gpu_.ffn_fmt_); });
                mark("w2", l);
                h = gpu_.hidden_b_.tex;
            if (getenv("HIDDEN_DUMP"))
                gpu_.dump_cpu_row(read_hidden(gpu_.hidden_b_), gpu_.act_fmt_, "hidA", 4096);
            } else {
                GLuint hffn;
                if (f.w1 && f.w1->rows) hffn = gpu_.ffn_split(*f.w1, *f.w3, xbf, f.width, hs);
                else hffn = gpu_.ffn_fused(*f.w13, xbf, f.width, hs);
                mark("ffn", l);
                gpu_.matvec_add(*f.w2, hffn, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                mark("w2", l);
                h = gpu_.hidden_a_.tex;
            if (getenv("HIDDEN_DUMP"))
                gpu_.dump_cpu_row(read_hidden(gpu_.hidden_a_), gpu_.act_fmt_, "hidA", 4096);
            }
        }
        if (!want_logits) return h;
        GLuint xf = gpu_.rmsnorm(h, gpu_.rms_final_tex_, gpu_.xf_);
        if (pass_dup("rms_final")) replay("rms_final", -1, [&]{ xf = gpu_.rmsnorm(h, gpu_.rms_final_tex_, gpu_.xf_); });
        mark("rms_final", -1);
        return xf;
    }

// Prefill batching, steps 7c-7e.  The packed path handles two consecutive
// token positions in one layer pass: the FFN reads each weight texel once and
// feeds both positions.  On q4 and t3 it is byte-identical to the serial path
// (every hidden state, the logits, the PPL and the greedy strings; Iris Xe
// 2026-11-25), and since step 7d it is also cheaper: the FFN input is built in
// P2 and the second w2 chunk residual stays there, so a layer needs 14 draws
// against 18 serial and warm prefill CPU drops from 6.81-6.96 s to 5.66-5.74 s
// (21-token prompt); wall time is neutral within this board noise (+1.5%
// median over 14 warm samples, against a 6.30-7.50 s spread), so CPU is what
// the default buys.  Step 7e makes it the default; PREFILL_PAIR=0 restores the
// serial loop.  Attention and rmsnorm stay per position (latency, not weight
// traffic); ACT8 / int16 / fused-w13 models keep the serial loop, and so does
// a q4 model in the packed nibble view (Q4_FMT=0).  The q8 weight half is
// exact but its row-pair qkv kernel adds a 1-LSB drift on Xe at late
// positions, so byte-exactness is a per-format measurement
// (docs/prefill-batching.md).
bool Session::pair_ok() const {
        if (!gpu_layer_) return false;
        // Two positions per layer pass is a property of the attention and FFN
        // kernels.  An RWKV layer's two positions cannot share a pass: the WKV
        // recurrence and both token shifts are strictly sequential, so the
        // second position has to see the first one's state (plan O31).  A pair
        // variant is P6's problem; the serial loop is correct today.
        if (m_.is_rwkv()) return false;
        // A format with no pair-family kernel is refused by the table at the
        // end of this function (WStage::P2); a missing pair kernel is a wrong
        // shader, not a slow one.
        // The two-position prefill implements a dense transformer block: it
        // calls attention_block() for every layer, and a hybrid model's conv
        // layers carry no qkv (the LFM2-350M q8 container aborted in
        // matvec_q8_pair_logits with rows=0, 2026-11-12).  There is no conv
        // pair kernel, so a hybrid model always takes the serial prefill.
        if (m_.hybrid) return false;
        // The paired path runs one FFN for two positions.  A MoE layer routes
        // each position on its own (and the chosen expert is a host-side
        // uniform), so the pair cannot share the FFN; the serial loop costs
        // the ~17% measured in the comment above, not 2x.
        if (m_.is_moe()) return false;
        // A per-layer FFN-hidden store scale (metadata key 15) is applied by the
        // serial FFN kernels only: the two-position kernels encode the hidden
        // with the fixed q16 scale, so a scaled model has to stay serial.
        for (int l = 0; l < m_.layers; ++l)
                if (m_.hid_shift(l) > 0) return false;
        // The two-position kernels have wrow variants (one fragment is one row
        // for both positions, so the plane read is a single fetch) and produce
        // byte-identical tokens.  Measured with PREFILL_REPEAT on a Pi Zero,
        // void.1 q8r, 17-byte prompt: warm prefill 7.0-7.2 s paired against
        // 8.2-8.4 s single, so the paired path is the faster one.  Its first
        // prefill is dearer (15.9 against 14.1 s) because it compiles the P2
        // program set; that is startup, not per-token work.  PAIR_WROW=0 keeps
        // the single path for an A/B inside one binary.
        {
                const char* w = getenv("PAIR_WROW");
                if (m_.row_scale && w && atoi(w) == 0) return false;
        }
        const char* e = getenv("PREFILL_PAIR");
        if (e && strcmp(e, "0") == 0) return false;
        // The pair kernels read and write I16_2 activations (q8 and q4's 4-bit
        // view).
        if (gpu_.act_fmt_ != XFormat::I16_2 || gpu_.ffn_fmt_ != XFormat::I16_2) return false;
        // The pair-prefill path stores two positions into one texel; a 24-bit
        // residual is one value per texel, so it cannot take part.
        if (x_is_int24(gpu_.res_fmt_) || x_is_f32(gpu_.res_fmt_)) return false;
        if ((m_.dim & 1) || (m_.hidden & 1)) return false;
        const vector<Mat>& w1 = !m_.w1a.empty() ? m_.w1a : m_.w1;
        const vector<Mat>& w3 = !m_.w3a.empty() ? m_.w3a : m_.w3;
        const vector<Mat>& w2 = !m_.w2a.empty() ? m_.w2a : m_.w2;
        if (w1.empty() || w3.empty() || w2.empty()) return false;
        // The weight half is the table's answer; the stream half (I16_2 pairs,
        // a residual that is not I24/F32) is checked above.  q8, q4 and t3
        // have the two-position family; i16/q6 take the serial loop.
        if (!(wf_can(w1[0].type, WStage::P2) && wf_can(w3[0].type, WStage::P2) &&
              wf_can(w2[0].type, WStage::P2)))
                return false;
        // q4's P2 kernels read the 4-bit view (Q4_FMT=rgba4), which only the
        // chunked FFN marks; the packed nibble view keeps the serial loop.
        // The check is here, not in the table: the view is a property of the
        // loaded Mats, not of the format.
        if (w1[0].type == WeightType::INT4 && !(w1[0].fmt4 && w2[0].fmt4)) return false;
        // The q4 and t3 two-position kernels are an Iris Xe result.  On the
        // board every pass is finished before the next, and the pair's four
        // live accumulators and eight activation fetches a group cost more
        // than the draws it removes: q4 warm prefill is 16.5 s against 12.4 s
        // on a 72-token prompt (2026-11-25, three interleaved pairs), and the
        // t3 gate is not even exact there (its LOGITS differ from the serial
        // arm in 71943 of 73448 words).  The q8 family predates these and is
        // the board's status quo (wall-neutral, CPU-cheaper), so it stays.
        if (gpu_.needs_finish() &&
            (w1[0].type == WeightType::INT4 || w1[0].type == WeightType::T3))
                return false;
        return true;
    }

vector<float> Session::step_pair(int tok_a, int pos_a, int tok_b, int pos_b, bool want_logits){
        if (!pair_ok()) {
            step(tok_a, pos_a, false);
            return step(tok_b, pos_b, want_logits);
        }
        GLuint h_a = gpu_.upload_hidden(embedding(tok_a));
        GLuint h_b = gpu_.upload_hidden_b(embedding(tok_b));
        for (int l = 0; l < m_.layers; ++l) {
            auto pt = Clock::now();
            auto plap = [&](const char* n) { if (getenv("PAIR_PROFILE")) { gpu_.sync(); cerr << "PAIRP " << n << " " << elapsed_ms(pt, Clock::now()) << "\n"; pt = Clock::now(); } };
            if (g_stop) return {};
            // A writes KV slot pos_a and B reads it; the command stream is
            // ordered, so attention can stay per position and single-draw.
            GLuint xba = gpu_.rmsnorm(h_a, gpu_.rms_att_tex_[l], gpu_.xb_);
            GLuint att_a = gpu_.attention_block(m_.qkv[l], xba, l, pos_a);
            if (getenv("STAGE_DUMP")) { gpu_.sync(); gpu_.dump_target(gpu_.att_out_, "att", 4096); }
            gpu_.matvec_add(m_.wo[l], att_a, h_a, gpu_.pp_a2_, gpu_.act_fmt_);
            GLuint xbb = gpu_.rmsnorm(h_b, gpu_.rms_att_tex_[l], gpu_.xb2_);
            GLuint att_b = gpu_.attention_block(m_.qkv[l], xbb, l, pos_b);
            gpu_.matvec_add(m_.wo[l], att_b, h_b, gpu_.pp_b2_, gpu_.act_fmt_);
            // The same dump names the serial loop uses for position A, so the
            // two paths' layer 0 can be diffed; "b" is the pair's second
            // position.  pp_in2_ is the P2 layout (A in .rg, B in .ba), not the
            // serial I16_2 pair, so it has its own name.
            if (getenv("STAGE_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.pp_a2_, "x1", 4096);
                gpu_.dump_target(gpu_.pp_b2_, "x1b", 4096);
            }
            plap("att");
            // 7d: the FFN input is built straight into P2 - one stats draw for
            // both positions, one normalising draw - so there is no pack pass,
            // and the second w2 chunk takes its residual from the first chunk P2
            // output, which removes the four splits 7c needed.  Only the two
            // splits that hand the layer output back as per-position residuals
            // remain.
            GLuint xin2 = gpu_.rmsnorm_pair2(gpu_.pp_a2_.tex, gpu_.pp_b2_.tex,
                                             gpu_.rms_ffn_tex_[l], gpu_.pp_in2_);
            if (getenv("STAGE_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.pp_in2_, "p2in", 4096);
                gpu_.dump_target(gpu_.rms_partial2_, "p2part", 64);
            }
            if (m_.ffn_chunked) {
                gpu_.ffn_pair_swiglu_chunk2_p2(m_.w1a[l], m_.w3a[l], xin2,
                                               m_.ffn_chunk, gpu_.pp_g_);
                plap("gate0");
                gpu_.matvec_p2_add(m_.w2a[l], gpu_.pp_g_.tex,
                                   gpu_.pp_a2_.tex, gpu_.pp_b2_.tex, gpu_.pp_w2_);
                plap("w20");
                gpu_.ffn_pair_swiglu_chunk2_p2(m_.w1b[l], m_.w3b[l], xin2,
                                               m_.hidden - m_.ffn_chunk, gpu_.pp_g_);
                gpu_.matvec_p2_add_p2(m_.w2b[l], gpu_.pp_g_.tex, gpu_.pp_w2_.tex, gpu_.pp_w2b_);
                gpu_.p2_split(gpu_.pp_w2b_.tex, gpu_.pp_a_, 0);
                gpu_.p2_split(gpu_.pp_w2b_.tex, gpu_.pp_b_, 1);
            } else {
                gpu_.ffn_pair_swiglu_chunk2_p2(m_.w1[l], m_.w3[l], xin2,
                                               m_.hidden, gpu_.pp_g_);
                gpu_.matvec_p2_add(m_.w2[l], gpu_.pp_g_.tex,
                                   gpu_.pp_a2_.tex, gpu_.pp_b2_.tex, gpu_.pp_w2_);
                gpu_.p2_split(gpu_.pp_w2_.tex, gpu_.pp_a_, 0);
                gpu_.p2_split(gpu_.pp_w2_.tex, gpu_.pp_b_, 1);
            }

            plap("ffn");
            h_a = gpu_.pp_a_.tex;
            h_b = gpu_.pp_b_.tex;
            if (getenv("HIDDEN_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.pp_a_, "hidA", 4096);
                gpu_.dump_target(gpu_.pp_b_, "hidB", 4096);
            }
        }
        if (!want_logits || g_stop) return {};
        GLuint xf = gpu_.rmsnorm(h_b, gpu_.rms_final_tex_, gpu_.xf_);
        return gpu_.logits_from(m_.logits_tiles, xf);
    }

// ---- Prefill query batching (PREFILL_BATCH) --------------------------------
// The attention-internal passes of N positions fuse into one score draw; the
// residual chain stays serial.  Unlike the pair path this changes no weight
// kernel, so it is format agnostic: t3, q8, q4, fused-W13, every KV axis and
// the hid_shift FFN scale all take it.  What it refuses is what has no dense
// attention block to fuse (RWKV, hybrid conv) or routes per position (MoE).

int Session::group_size() const {
        int n = env_int("PREFILL_BATCH", 0);
        if (n < 2) return 0;
        if (n > 16) n = 16;
        return n;
}

bool Session::group_ok() const {
        if (!gpu_layer_) return false;
        if (group_size() < 2) return false;
        if (m_.is_rwkv()) return false;
        if (m_.hybrid) return false;
        if (m_.is_moe()) return false;
        return true;
}

vector<float> Session::step_group(const vector<int>& toks, int pos0, bool want_logits){
        const int n = (int)toks.size();
        if (n < 2 || !group_ok()) {
            vector<float> lg;
            for (int b = 0; b < n; ++b)
                lg = step(toks[b], pos0 + b, want_logits && b + 1 == n);
            return lg;
        }
        // Every position's embedding, packed before the layer loop.
        vector<GLuint> h(n);
        for (int b = 0; b < n; ++b)
            h[b] = gpu_.upload_hidden_grp(embedding(toks[b]), b, n);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return {};
            // Phase 1: stage each position's K/V and rotated Q.  The residual
            // h[b] is untouched here, so the order across positions is free;
            // each position's own draw sequence is the serial block's.
            for (int b = 0; b < n; ++b) {
                GLuint xb = gpu_.rmsnorm(h[b], gpu_.rms_att_tex_[l], gpu_.xb_);
                gpu_.att_stage(m_.qkv[l], xb, l, pos0 + b, b, n);
            }
            // Phase 2: one score draw for the whole group, causal per row.
            gpu_.att_scores_batch(l, pos0, n);
            // Phase 3: the serial per-position sequence.  stats and v read the
            // batched score rows; wo and the FFN are the serial kernels, and
            // the residual ping-pongs grp_h(b) <-> grp_tmp exactly as the
            // serial path ping-pongs hidden_a_ <-> hidden_b_.
            for (int b = 0; b < n; ++b) {
                if (g_stop) return {};
                gpu_.att_stats_row(pos0 + b, b, n);
                gpu_.att_v_row(l, pos0 + b, b, n);
                gpu_.matvec_add(m_.wo[l], gpu_.att_out_.tex, h[b], gpu_.grp_tmp(), gpu_.act_fmt_);
                GLuint x1 = gpu_.grp_tmp().tex;
                GLuint xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
                const FfnMats f = ffn_mats(m_, l, -1);
                const int hs = m_.hid_shift(l);
                if (m_.ffn_chunked) {
                    GLuint ha, hb;
                    gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xbf,
                                           m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb, hs);
                    gpu_.matvec_add(*f.w2a, ha, x1, gpu_.grp_w2a(), gpu_.ffn_fmt_);
                    gpu_.matvec_add(*f.w2b, hb, gpu_.grp_w2a().tex, gpu_.grp_h(b, n), gpu_.ffn_fmt_);
                } else {
                    GLuint hffn;
                    if (f.w1 && f.w1->rows) hffn = gpu_.ffn_split(*f.w1, *f.w3, xbf, f.width, hs);
                    else hffn = gpu_.ffn_fused(*f.w13, xbf, f.width, hs);
                    gpu_.matvec_add(*f.w2, hffn, x1, gpu_.grp_h(b, n), gpu_.ffn_fmt_);
                }
                h[b] = gpu_.grp_h(b, n).tex;
            }
        }
        if (g_stop || !want_logits) return {};
        GLuint xf = gpu_.rmsnorm(h[n - 1], gpu_.rms_final_tex_, gpu_.xf_);
        return gpu_.logits_from(m_.logits_tiles, xf);
}

vector<float> Session::step(int token, int pos, bool want_logits){
        // RWKV-7 runs its own loop.  There is no attention block to fuse and no
        // KV row to write per position, so the position only matters to the
        // caller's context limit; the block's history is the per-head state.
        if (m_.is_rwkv()) return step_rwkv(token, want_logits);
        if (gpu_layer_) {
            GLuint xf = forward_gpu_hidden(token, pos, want_logits);
            if (g_stop || !want_logits) return {};
            auto a = Clock::now();
            vector<float> out = gpu_.logits_from(m_.logits_tiles, xf);
            if (pass_dup("logits_draw")) gpu_.logits_pass_only(m_.logits_tiles, xf);
            if (pass_dup("logits")) { vector<float> dup = gpu_.logits_from(m_.logits_tiles, xf); (void)dup; }
            if (getenv("LAYER_PROFILE")) {
                gpu_.sync();
                cerr << "PROF logits " << elapsed_ms(a, Clock::now()) << "\n";
            }
            return out;
        }
        vector<float> x = forward(token, pos);
        if (g_stop || !want_logits) return {};
        vector<float> xb = getenv("CPU_RMS_GPU")
                ? rms_final_gpu(x)
                : rms_norm(x, m_.rms_final.data());
        return gpu_.logits(m_.logits_tiles, xb);
    }

bool Session::greedy_fast_ok() const {
        return gpu_layer_ && gpu_.logits_argmax_ok(m_.logits_tiles);
    }

int Session::step_greedy(int token, int pos){
        if (greedy_fast_ok()) {
            GLuint xf = forward_gpu_hidden(token, pos, true);
            if (g_stop) return -1;
            int t = gpu_.logits_argmax_from(m_.logits_tiles, xf);
            if (t >= 0) return t;
            // The reduction found no valid winner (cannot happen for a real
            // vocabulary): the logits are already drawn, so read them back
            // rather than drawing them twice.
            vector<float> lg = gpu_.read_logits_full(m_.logits_tiles);
            return argmax_token(lg);
        }
        vector<float> lg = step(token, pos, true);
        if (g_stop || lg.empty()) return -1;
        return argmax_token(lg);
    }

CandidateResult Session::step_candidates(int token, int pos, int topk, float temp, float top_p){
        if (m_.is_rwkv()) {
                vector<float> x = rwkv_gpu_ ? forward_gpu_rwkv(token) : forward_rwkv(token);
                if (g_stop) return {};
                return gpu_.logits_candidates(m_.logits_tiles, rwkv_head_input(x), topk, temp, top_p);
        }
        if (gpu_layer_) {
            GLuint xf = forward_gpu_hidden(token, pos, true);
            if (g_stop) return {};
            return gpu_.logits_candidates_from(m_.logits_tiles, xf, topk, temp, top_p);
        }
        vector<float> x = forward(token, pos);
        if (g_stop) return {};
        vector<float> xb = rms_norm(x, m_.rms_final.data());
        return gpu_.logits_candidates(m_.logits_tiles, xb, topk, temp, top_p);
    }

vector<float> Session::rms_gpu(const vector<float>& x, int layer, bool ffn){
        // rmsnorm reads the residual layout (res_fmt_), not the activation one:
        // the fused path feeds it upload_hidden()'s texture.
        GLuint xt = gpu_.upload_hidden(x);
        Target& t = ffn ? gpu_.xbf_ : gpu_.xb_;
        gpu_.rmsnorm(xt, ffn ? gpu_.rms_ffn_tex_[layer] : gpu_.rms_att_tex_[layer], t);
        vector<uint8_t> b = gpu_.read_bytes(t, t.w, 1);
        vector<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
                const size_t o = (i / 2) * 4 + (i % 2) * 2;
                int u = (int)b[o] * 256 + (int)b[o + 1];
                if (u >= 32768) u -= 65536;
                out[i] = (float)u / ACT_SCALE;
        }
        return out;
}

vector<float> Session::rms_final_gpu(const vector<float>& x){
        GLuint xt = gpu_.upload_hidden(x);
        gpu_.rmsnorm(xt, gpu_.rms_final_tex_, gpu_.xf_);
        vector<uint8_t> b = gpu_.read_bytes(gpu_.xf_, gpu_.xf_.w, 1);
        vector<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
                const size_t o = (i / 2) * 4 + (i % 2) * 2;
                int u = (int)b[o] * 256 + (int)b[o + 1];
                if (u >= 32768) u -= 65536;
                out[i] = (float)u / ACT_SCALE;
        }
        return out;
}

// One texture row as floats.  The layout is the *GPU's* residual format: a
// container whose wo is q8/q4 packs two q16 per texel (I16_2), a q8/i16
// checkpoint uses one q16 per texel (I16_1), and a q24 container carries the
// 24-bit encode.  Reading with the wrong width silently returns a shifted row,
// which is how a probe can "prove" a bug that is not there.
static vector<float> decode_act_row(const vector<uint8_t>& b, int texels, XFormat f){
        const int per = x_per_texel(f);
        vector<float> out((size_t)texels * per);
        for (size_t i = 0; i < out.size(); ++i) {
                const size_t o = (i / per) * 4;
                if (f == XFormat::I16_2) {
                        const size_t q = o + (i % per) * 2;
                        int u = (int)b[q] * 256 + (int)b[q + 1];
                        if (u >= 32768) u -= 65536;
                        out[i] = (float)u / ACT_SCALE;
                } else if (f == XFormat::I16_1) {
                        int u = (int)b[o] * 256 + (int)b[o + 1];
                        if (u >= 32768) u -= 65536;
                        out[i] = (float)u / ACT_SCALE;
                } else if (f == XFormat::I24_1) {
                        int v = (int)b[o + 1] * 65536 + (int)b[o + 2] * 256 + (int)b[o + 3] - 8388608;
                        out[i] = (float)v / ACT_SCALE;
                } else if (f == XFormat::F32) {
                        uint32_t u = ((uint32_t)b[o] << 24) | ((uint32_t)b[o + 1] << 16) |
                                     ((uint32_t)b[o + 2] << 8) | (uint32_t)b[o + 3];
                        float v = 0.0f;
                        memcpy(&v, &u, 4);
                        out[i] = v;
                } else {
                        out[i] = ((float)b[o + (i % 4)] - 128.0f) / 16.0f;
                }
        }
        return out;
}

// Decode one texture with an explicit format: the debugging counterpart of
// read_hidden(), which is only right for a residual-format stream.
// Needs a Target with an FBO, so it cannot read a bare texture: the LN weights
// are one, which is why the dump below registers a throwaway target first.
vector<float> dump_tex(GPU& g, GLuint tex, int dim, XFormat f) {
        Target t;
        t.tex = tex; t.w = x_texels(dim, f); t.h = 1;
        g.attach_dump_fbo(t);
        vector<uint8_t> b = g.read_bytes(t, t.w, 1);
        return decode_act_row(b, t.w, f);
}

vector<float> Session::read_hidden(const Target& t){
        vector<uint8_t> b = gpu_.read_bytes(const_cast<Target&>(t), t.w, 1);
        return decode_act_row(b, t.w, gpu_.res_fmt_);
}

vector<float> Session::rms_norm(const vector<float>& x, const float* w){
        float ss = 0;
        for (float v : x) ss += v * v;
        ss /= x.size();
        // The container states its eps in metadata key 11 and the GPU path
        // passes it to the shader (engine.cpp set_rms_eps); the CPU layer loop
        // used to hardcode 1e-5, which is 10x MiniMind3's 1e-6 and made
        // GPU_LAYER=0 disagree with the fused path by more than the activation
        // grid.  `rms_eps` defaults to 1e-5, so a container without the key
        // keeps the old arithmetic.
        const float eps = m_.rms_eps > 0.0f ? m_.rms_eps : 1e-5f;
        float r = 1.0f / sqrtf(ss + eps);
        vector<float> o(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            o[i] = quantize_act_f(w[i] * (r * x[i]));
        return o;
    }

vector<float> Session::head_rms_norm(const vector<float>& v, int n_heads, const float* w){
        vector<float> o(v.size());
        const int hd = m_.head_dim;
        for (int h = 0; h < n_heads; ++h) {
            const int base = h * hd;
            float ss = 0;
            for (int i = 0; i < hd; ++i) { float x = v[base + i]; ss += x * x; }
            float r = 1.0f / sqrtf(ss / hd + 1e-6f);
            for (int i = 0; i < hd; ++i) o[base + i] = w[i] * (r * v[base + i]);
        }
        return o;
    }

vector<float> Session::rope(const vector<float>& v, int n_heads, int pos){
        vector<float> o = v;
        int half = m_.head_dim / 2;
        for (int h = 0; h < n_heads; ++h) {
            int base = h * m_.head_dim;
            if (m_.rope_half) {
                for (int i = 0; i < half; ++i) {
                    float c = rope_cos_[(size_t)pos * half + i];
                    float s = rope_sin_[(size_t)pos * half + i];
                    float a = o[base + i], b = o[base + i + half];
                    o[base + i] = a * c - b * s;
                    o[base + i + half] = a * s + b * c;
                }
            } else {
                for (int i = 0; i < half; ++i) {
                    float c = rope_cos_[(size_t)pos * half + i];
                    float s = rope_sin_[(size_t)pos * half + i];
                    float a = o[base + 2 * i], b = o[base + 2 * i + 1];
                    o[base + 2 * i] = a * c - b * s;
                    o[base + 2 * i + 1] = a * s + b * c;
                }
            }
        }
        return o;
    }

vector<float> Session::attention(const vector<float>& q, int layer, int pos){
        int group = m_.heads / m_.kv_heads;
        vector<float> out(m_.dim);
        vector<float> scores(pos + 1);
        const int qmode = kv8_sim_mode();
        const float qscale = kv8_sim_scale();
        vector<float> kbuf, vbuf;
        if (qmode) {
            kbuf.resize((size_t)(pos + 1) * m_.kv_dim);
            vbuf.resize((size_t)(pos + 1) * m_.kv_dim);
        }
        for (int h = 0; h < m_.heads; ++h) {
            int kvh = h / group;
            const float* kb = kv_cache_k_[layer].data() + (size_t)kvh * m_.head_dim;
            const float* vb = kv_cache_v_[layer].data() + (size_t)kvh * m_.head_dim;
            if (qmode) {
                const bool qk = qmode == 1 || qmode == 2 || qmode == 3 || qmode == 6;
                const bool qv = qmode == 1 || qmode == 2 || qmode == 4 || qmode == 5;
                const bool ph = qmode == 2 || qmode == 5 || qmode == 6;
                float kmx0 = 0.0f, vmx0 = 0.0f;
                for (int t = 0; t <= pos; ++t) {
                    float mk = 0.0f, mv = 0.0f;
                    for (int i = 0; i < m_.head_dim; ++i) {
                        mk = std::max(mk, fabsf(kb[(size_t)t * m_.kv_dim + i]));
                        mv = std::max(mv, fabsf(vb[(size_t)t * m_.kv_dim + i]));
                    }
                    if (t == 0) { kmx0 = mk; vmx0 = mv; }
                    const float sk = (ph && qk && mk > 0.0f) ? 127.0f / mk : qscale;
                    const float sv = (ph && qv && mv > 0.0f) ? 127.0f / mv : qscale;
                    for (int i = 0; i < m_.head_dim; ++i) {
                        const size_t o = (size_t)t * m_.kv_dim + i;
                        kbuf[o] = qk ? kv8_round(kb[o], sk) : kb[o];
                        vbuf[o] = qv ? kv8_round(vb[o], sv) : vb[o];
                    }
                }
                if (pos == 0 && getenv("KV8_STATS"))
                    cerr << "KV8STAT layer=" << layer << " head=" << kvh
                         << " kmax=" << kmx0 << " vmax=" << vmx0 << "\n";
                kb = kbuf.data();
                vb = vbuf.data();
            }
            float mx = -1e30f;
            for (int t = 0; t <= pos; ++t) {
                float s = 0;
                for (int i = 0; i < m_.head_dim; ++i)
                    s += q[h * m_.head_dim + i] * kb[(size_t)t * m_.kv_dim + i];
                s /= sqrtf((float)m_.head_dim);
                scores[t] = s;
                mx = max(mx, s);
            }
            float sum = 0;
            for (int t = 0; t <= pos; ++t) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
            for (int t = 0; t <= pos; ++t) scores[t] /= sum;
            for (int i = 0; i < m_.head_dim; ++i) {
                float s = 0;
                for (int t = 0; t <= pos; ++t)
                    s += scores[t] * vb[(size_t)t * m_.kv_dim + i];
                out[h * m_.head_dim + i] = s;
            }
        }
        return out;
    }

vector<float> Session::add_q16(const vector<float>& a, const vector<float>& b){
        vector<float> o(a.size());
        if (m_.res_q24) {
                // Same grid, 256x the range: the clamp moves from 128 to 32768.
                for (size_t i = 0; i < a.size(); ++i) {
                        float v = a[i] + b[i];
                        float q = roundf(v * 256.0f);
                        q = fmaxf(-8388608.0f, fminf(8388607.0f, q));
                        o[i] = q / 256.0f;
                }
                return o;
        }
        for (size_t i = 0; i < a.size(); ++i)
            o[i] = act_f32() ? a[i] + b[i] : quantize_act_f(a[i] + b[i]);
        return o;
    }

vector<float> Session::swiglu(const vector<float>& h, int n){
        vector<float> o(n);
        for (int i = 0; i < n; ++i) {
            float a = h[i], b = h[n + i];
            o[i] = act_f32() ? (a / (1.0f + expf(-a))) * b : quantize_act_f((a / (1.0f + expf(-a))) * b);
        }
        return o;
    }

vector<float> Session::swiglu(const vector<float>& h){
        return swiglu(h, m_.hidden);
    }

// ---- v5: the encoder-decoder path -----------------------------------------
// The encoder is run layer-outer over every source position, which is the same
// order `step_group` uses for a prefill batch: stage each position's K/V for
// this layer, one batched score draw, then each position's stats/V and its
// residual chain.  What differs from the dense loop is the block, not the
// machinery: post-norm (the LayerNorm follows each residual add), a bias on
// every projection, a non-gated FFN, and no rotation.
//
// The cross-attention reads its K/V on the host for now.  That is a deliberate
// bring-up step, not the shape of the finished path: the encoder output and the

// The v5 profile lap.  Same line and the same reason as the dense loop's `mark`:
// a lap synchronises, so its wall time is QPU work plus submit plus finish, and
// only the performance counters separate them.  A stage whose GPU share is near
// zero is not slow, it is expensive to launch -- which on this board is most of
// them, and is the thing a "move it to the GPU" plan has to know.
namespace {
struct EncLap {
        GPU& g;
        bool prof;
        bool counters;
        Clock::time_point pt = Clock::now();
        long draws = 0;
        EncLap(GPU& gg, bool p, bool c) : g(gg), prof(p), counters(c) {
                if (!prof) return;
                draws = (long)g.draws();
                if (counters) g.pc_stage_begin();
        }
        void mark(const char* name, int l) {
                if (!prof) return;
                g.sync();
                auto now = Clock::now();
                const long d = (long)g.draws();
                const double ms = elapsed_ms(pt, now);
                unsigned long long idle = 0, frag = 0, exec = 0, quads = 0;
                bool have = false;
                if (counters) have = g.pc_stage_end(idle, frag, exec, quads);
                cerr << "PROF l=" << l << " " << name << " " << ms
                     << " draws=" << (d - draws);
                if (have) {
                        // cycles / (qpus * MHz) = ms, with the divisor measured
                        // on the board rather than assumed.
                        const double to_ms = 1.0 / (GPU::pc_qpus() * (double)GPU::pc_mhz() * 1000.0);
                        const double gpu_ms = (double)frag * to_ms;
                        cerr << " gpu=" << gpu_ms
                             << " (" << (ms > 0 ? (int)(100.0 * gpu_ms / ms) : 0) << "%)"
                             << " exec=" << ((double)exec * to_ms)
                             << " idle=" << ((double)idle * to_ms)
                             << " quads=" << quads;
                }
                cerr << "\n";
                if (counters) g.pc_stage_begin();
                draws = d;
                pt = now;
        }
};
}

void Session::encdec_encode(const vector<int>& src) {
        const Model& m = m_;
        const int n = (int)src.size();
        if (n <= 0 || n > m.enc_seq || n > m.seq) {
                // A refusal, not a truncation.  `cd_n_ = 0` makes the decode loop
                // stop instead of running on the previous sentence's state, which
                // is what "return here" did: a silent redirect of the model's
                // answer to something unrelated.
                cerr << "encoder: " << n << " positions do not fit (enc_seq "
                     << m.enc_seq << ", seq " << m.seq << "); the container was "
                     << "packed for a shorter source\n";
                cd_n_ = 0;
                return;
        }
        cd_n_ = n;
        const float s = m.enc_scale_embed ? sqrtf((float)m.dim) : 1.0f;
        const int E = m.enc_layers, D = m.dim, F = m.enc_ffn;
        // The embedding plus the position table, on the host: this block adds
        // its positions to the embedding rather than rotating q/k, so there is
        // no kernel to add them later.
        // `h[b]` is the live handle for position b, and it moves as the layer
        // writes its residual -- the same ping-pong `step_group` keeps.  Reading
        // `grp_h(b,n)` instead (which is what this loop did first) reads a fresh
        // *zero* target: upload_hidden_grp writes grp_in_[b], not grp_h_[b], and
        // every value after that is a normalised zero.  That is the whole
        // "maxabs 0.00390625" -- one q16 step off zero.
        vector<GLuint> h(n);
        for (int b = 0; b < n; ++b) {
                // exact_f32: `embedding()` otherwise rounds the row onto the q16
                // activation grid *before* the sqrt(d) scale.  A Marian
                // embedding is ~0.03 and that grid's step is 1/256, so the row
                // would arrive 13% off and the block would normalise the error
                // in.  The scale belongs on the real values; the upload that
                // follows quantises once, at the right magnitude.
                vector<float> x = embedding(src[b], /*exact_f32=*/true);
                const float* p = m.pos_table.data() + (size_t)b * D;
                for (int i = 0; i < D; ++i) x[i] = x[i] * s + p[i];
                h[b] = gpu_.upload_into_fmt(gpu_.v5_h(b, n), x, gpu_.act_fmt_);
        }
        gpu_.set_causal(false);            // the encoder sees every position
        // POST-norm, which is the order `normalize_before: false` states and the
        // one the numpy reference's correct translations came from:
        //
        //     r1 = x + attn(x)        attention reads the *raw* residual
        //     x1 = LN1(r1)            the normalisation follows the add
        //     r2 = x1 + fc2(swish(fc1(x1)))
        //     x  = LN2(r2)
        //
        // Both loops were written pre-norm (LN before attention, no LN on the
        // output).  It looks like a small ordering choice and it is not: with
        // nothing normalising the stream, the residual compounds every layer,
        // which is the 110.8 against the reference's 1.68.
        EncLap lap(gpu_, env_set("LAYER_PROFILE"),
                   env_set("LAYER_PROFILE") && gpu_.pc_available());
        const bool enc_ffn_fused = env_flag("ENC_FFN_FUSED");
        for (int l = 0; l < E; ++l) {
                lap.mark("encLN1", l);
                gpu_.set_qkv_bias(gpu_.enc_rows_.enc_qkv_b[l]);
                for (int b = 0; b < n; ++b) {
                        lap.mark("encPhase1", l);
                        gpu_.att_stage(m.u_enc_qkv[l], h[b], l, b, b, n);
                        lap.mark("encAttStage", l);
                }
                gpu_.set_qkv_bias(0);
                lap.mark("encQKV", l);
                gpu_.att_scores_batch(l, n - 1, n);
                lap.mark("encScoresBatch", l);
                lap.mark("encScores", l);
                for (int b = 0; b < n; ++b) {
                        lap.mark("encPhase3", l);
                        // `pos` is the last *valid* key under NOCAUSAL (the
                        // scan bound), not this query's position: the encoder
                        // fills n slots of a SEQ-wide cache and the rest must
                        // not enter the softmax.
                        gpu_.att_stats_row(n - 1, b, n);
                        lap.mark("encStatsRow", l);
                        gpu_.att_v_row(l, n - 1, b, n);
                        lap.mark("encVRow", l);
                        lap.mark("encStatsV", l);
                        // r1 = x + wo*att + o_b, then x1 = LN1(r1)
                        gpu_.matvec_add_rf(m.u_enc_wo[l], gpu_.att_out_.tex,
                                           h[b], gpu_.cd_b(), gpu_.act_fmt_, gpu_.act_fmt_);
                        gpu_.bias_add(gpu_.cd_b().tex, gpu_.enc_rows_.enc_o_b[l], gpu_.cd_c());
                        lap.mark("encWo", l);
                        if (env_flag("ENCDEC_DUMP") && l == 0 && b == 0) {
                                // Even components of r1 match the numpy reference
                                // and odd ones do not, which is the signature of
                                // the q16 *pair* layout going wrong on one of the
                                // two halves -- the same family of bug the comment
                                // in GPU::matvec_add records ("every odd row gained
                                // exactly 255/256").
                                vector<float> rv = read_hidden(gpu_.cd_c());
                                cerr << "ENCDEC r1[0..7]";
                                for (int i = 0; i < 8 && i < (int)rv.size(); ++i)
                                        cerr << " " << rv[i];
                                cerr << "  (reference 0.309 -0.245 1.004 -0.198"
                                        " -0.751 -0.066 0.612 -0.231)\n";
                        }
                        GLuint x1 = gpu_.ln_q16(gpu_.cd_c().tex, gpu_.enc_rows_.enc_ln1[l],
                                                gpu_.enc_rows_.enc_ln1b[l],
                                                gpu_.cd_a(), m.rms_eps, "enc", l, 1, b);
                        lap.mark("encLN1b", l);
                        // f = fc2(swish(fc1(x1) + b1)) + b2, then r2 = x1 + f.
                        // cd_b holds the fc1 side, so x1 (cd_a) survives as the
                        // residual this adds into.
                        gpu_.matvec_to(m.u_enc_w1[l], x1, gpu_.cd_d());
                        lap.mark("encFC1", l);
                        // X is what `matvec_q8` wrote, and that kernel's
                        // output store is the projection's (PROJ), not the
                        // ACT axis's.  Reading it as act_fmt_ is the
                        // "FFN hidden is fine in one half only" bug.
                        // swish(fc1 + b) in one pass: the bias add was 0.59 ms
                        // against the 0.50 ms matvec that fed it, and both were
                        // their own draw.
                        if (env_flag("ENCDEC_DUMP") && l == 0 && b == 0) {
                                // The projection store itself: the tensor the
                                // PROJ axis changes, decoded by the same codec
                                // the GPU reads it with.
                                Target& t = gpu_.cd_d();
                                vector<uint8_t> wb = gpu_.read_bytes(t, 8, 1);
                                cerr << "ENCDEC w1store[0..7] " << x_format_name(gpu_.proj_fmt_);
                                for (int i = 0; i < 8; ++i)
                                        cerr << " " << x_unpack(wb.data(), i, gpu_.proj_fmt_);
                                cerr << "\n";
                        }
                        GLuint hf = gpu_.act_swish_biased(gpu_.cd_d().tex,
                                                          gpu_.enc_rows_.enc_fc1_b[l],
                                                          gpu_.cd_b(), F, gpu_.proj_fmt_,
                                                          gpu_.act_fmt_);
                        lap.mark("encSwishB", l);
                        // hf lives in cd_b, so the add must not write there: a
                        // matvec reads X and draws into its target, and one
                        // texture for both is a hazard the strip split makes
                        // real (measured: the reply collapsed to 0 tokens).
                        // ENC_FFN_FUSED folds the fc2 bias into the w2 add's
                        // write-back: one draw instead of two, and a *different
                        // rounding* -- once instead of twice, by up to 1/256 of
                        // a value.  It is a dev arm because of that, not a
                        // refactor.  The bias belongs to exactly one arm: the
                        // kernel's or the `bias_add` below.  Giving it to both
                        // added it twice (measured: the non-fused r2 differs
                        // from the fused row by exactly fc2.bias).
                        const GLuint fb = gpu_.enc_rows_.enc_fc2_b[l];
                        gpu_.matvec_add_bias_rf(m.u_enc_w2[l], hf, x1,
                                                enc_ffn_fused ? gpu_.enc_rows_.enc_fc2_b_rf[l] : 0,
                                                gpu_.cd_d(), gpu_.act_fmt_, gpu_.act_fmt_);
                        lap.mark("encW2", l);
                        if (enc_ffn_fused) gpu_.cd_c() = gpu_.cd_d();
                        else gpu_.bias_add(gpu_.cd_d().tex, fb, gpu_.cd_c());
                        if (env_flag("ENCDEC_DUMP") && l == 0 && b == 0) {
                                vector<float> rv = read_hidden(gpu_.cd_c());
                                cerr << "ENCDEC r2[0..7]";
                                for (int i = 0; i < 8 && i < (int)rv.size(); ++i)
                                        cerr << " " << rv[i];
                                cerr << "\n";
                        }
                        lap.mark("encFC2b", l);
                        // x = LN2(r2), straight into the group slot the next layer reads.
                        h[b] = gpu_.ln_q16(gpu_.cd_c().tex, gpu_.enc_rows_.enc_ln2[l],
                                           gpu_.enc_rows_.enc_ln2b[l],
                                           gpu_.v5_h(b, n), m.rms_eps, "enc", l, 2, b);
                }
                lap.mark("encFFN", l);
        }
        // The encoder output and each decoder layer's cross K/V, read back once.
        cd_enc_.assign((size_t)n * D, 0.0f);
        // The host copy of the encoder's output is a **dump**, not an operand:
        // the cross-attention's K/V is built on the GPU and nothing else reads
        // this.  It used to be filled unconditionally, which is `n` finishes and
        // `n` read-backs a sentence -- the fixed cost that made the encode pass
        // look superlinear in the source length.
        if (env_flag("ENCDEC_DUMP")) {
                for (int b = 0; b < n; ++b) {
                        // An ACT row: `read_hidden` would decode it as the
                        // residual, which is a different format as soon as the
                        // two axes differ (RES=f32 printed 3e38 against an
                        // encoder that was exact).
                        const Target& t = gpu_.v5_h(b, n);
                        vector<uint8_t> bytes = gpu_.read_bytes(const_cast<Target&>(t), t.w, 1);
                        vector<float> row = decode_act_row(bytes, t.w, gpu_.act_fmt_);
                        memcpy(cd_enc_.data() + (size_t)b * D, row.data(),
                               sizeof(float) * D);
                }
        }
        // ENCDEC_DUMP: the two numbers that decide whether the encode pass is
        // sane -- the output's magnitude (a post-norm stack ends O(1), so a
        // value in the tens means a residual is going un-normalised) and the
        // first decoder step's argmax, which is what a reference comparison
        // quotes.  Both were the entire diagnosis of this block.
        if (env_flag("ENCDEC_DUMP")) {
                float mx = 0.0f, ss = 0.0f;
                for (float v : cd_enc_) { mx = fabsf(v) > mx ? fabsf(v) : mx; ss += v * v; }
                cerr << "ENCDEC enc_out maxabs " << mx << " rms "
                     << sqrtf(ss / (float)cd_enc_.size()) << "\n";
                cerr << "ENCDEC enc_out[0..7]";
                for (int i = 0; i < 8 && i < (int)cd_enc_.size(); ++i)
                        cerr << " " << cd_enc_[i];
                cerr << "\n";
        }
        // The cross-attention's K/V live on the GPU now: one plane per decoder
        // layer, `[K | V]` a row, written from each projection's output.  No
        // read-back -- the host kept two vectors a layer and paid `n * layers`
        // round trips to fill them, and the decoder then paid two more per token.
        gpu_.alloc_cross_kv(n);
        lap.mark("encAllocKV", 0);
        for (int l = 0; l < m.layers; ++l) {
                lap.mark("encCrossKV", l);
                for (int b = 0; b < n; ++b) {
                        // The bias rides the store: `att_cross_kv` reads the
                        // projection's I24 row and its bias row with the same
                        // reader, so the separate bias-add pass is gone
                        // and this is two draws a position instead of four.
                        gpu_.matvec_to(m.u_cross_k[l], gpu_.v5_h(b, n).tex, gpu_.cd_b());
                        gpu_.cross_kv_store(l, gpu_.cd_b().tex, b, /*v_half=*/false,
                                            gpu_.enc_rows_.ck_b[l]);
                        gpu_.matvec_to(m.u_cross_v[l], gpu_.v5_h(b, n).tex, gpu_.cd_b());
                        gpu_.cross_kv_store(l, gpu_.cd_b().tex, b, /*v_half=*/true,
                                            gpu_.enc_rows_.cv_b[l]);
                }
        }
        lap.mark("encCrossKVEnd", 0);
        gpu_.set_causal(true);
}

vector<float> Session::encdec_step(int token, int pos) {
        // `encdec_encode` refuses a source that does not fit and zeroes this.
        if (cd_n_ <= 0) return vector<float>();
        const Model& m = m_;
        const int D = m.dim, F = m.enc_ffn;
        const float s = m.enc_scale_embed ? sqrtf((float)D) : 1.0f;
        vector<float> y = embedding(token, /*exact_f32=*/true);
        const float* p = m.pos_table.data() + (size_t)pos * D;
        for (int i = 0; i < D; ++i) y[i] = y[i] * s + p[i];
        GLuint h = gpu_.upload_hidden_fmt(y, gpu_.act_fmt_);
        const int n = cd_n_;
        for (int l = 0; l < m.layers; ++l) {
                // Same post-norm order as the encoder: self-attention on the raw
                // residual, then LN1; cross-attention from LN1's output, then
                // LN2; the FFN from LN2's output, then LN3.
                gpu_.set_qkv_bias(gpu_.enc_rows_.qkv_b[l]);
                GLuint at = gpu_.attention_block(m.qkv[l], h, l, pos);
                gpu_.set_qkv_bias(0);
                gpu_.matvec_add_rf(m.wo[l], at, h, gpu_.cd_b(), gpu_.act_fmt_, gpu_.act_fmt_);
                gpu_.bias_add(gpu_.cd_b().tex, gpu_.enc_rows_.o_b[l], gpu_.cd_c());
                GLuint y1 = gpu_.ln_q16(gpu_.cd_c().tex, gpu_.enc_rows_.ln1[l],
                                        gpu_.enc_rows_.ln1b[l], gpu_.cd_a(), m.rms_eps,
                                        "dec", l, 1, pos);
                // cross-attention: q from y1, k/v from the encoder (read-back
                // planes; the cross kernel is the remaining host step)
                // The cross-attention runs where the self-attention runs: the
                // query is projected on the GPU, the three passes read the plane
                // `encdec_encode` built, and the context stays in a texture.  What
                // used to be here was a read-back, a host softmax with a nested
                // loop over heads and keys, and an upload -- per layer, per token.
                // The cross-attention runs where the self-attention runs: the
                // same block, pointed at the plane `encdec_encode` built and told
                // not to touch this layer's cache.  It used to be a read-back, a
                // host softmax over heads and keys, and an upload -- per layer,
                // per token.
                //
                // The bias add leaves `qkv_target_` at `3*dim`, which is the width
                // `RQKVQ` divides by (`QKVTW`); a `dim`-wide add shrank it to a
                // third and every three q samples landed on the same texel.  The
                // uploaded bias is zero-padded to that width for the same reason.
                gpu_.set_qkv_bias(gpu_.enc_rows_.cq_b[l]);
                GLuint ctx_tex = gpu_.attention_block(m.u_cross_q[l], y1, l, n - 1,
                                                      gpu_.cross_kv_tex(l),
                                                      /*stage_kv=*/false);
                gpu_.set_qkv_bias(0);
                gpu_.matvec_add_rf(m.u_cross_o[l], ctx_tex, y1, gpu_.cd_b(),
                                   gpu_.act_fmt_, gpu_.act_fmt_);
                gpu_.bias_add(gpu_.cd_b().tex, gpu_.enc_rows_.co_b[l], gpu_.cd_c());
                GLuint y2 = gpu_.ln_q16(gpu_.cd_c().tex, gpu_.enc_rows_.ln2[l],
                                        gpu_.enc_rows_.ln2b[l], gpu_.cd_a(), m.rms_eps,
                                        "dec", l, 2, pos);
                gpu_.matvec_to(m.w1[l], y2, gpu_.cd_d());
                GLuint hid = gpu_.act_swish_biased(gpu_.cd_d().tex, gpu_.enc_rows_.fc1_b[l],
                                                   gpu_.cd_b(), F, gpu_.proj_fmt_,
                                                   gpu_.act_fmt_);
                gpu_.matvec_add_rf(m.w2[l], hid, y2, gpu_.cd_d(), gpu_.act_fmt_, gpu_.act_fmt_);
                gpu_.bias_add(gpu_.cd_d().tex, gpu_.enc_rows_.fc2_b[l], gpu_.cd_c());
                h = gpu_.ln_q16(gpu_.cd_c().tex, gpu_.enc_rows_.ln3[l],
                                gpu_.enc_rows_.ln3b[l], gpu_.cd_d(), m.rms_eps,
                                "dec", l, 3, pos);
        }
        vector<float> lg = gpu_.logits_from(m.logits_tiles, h);
        // The container's head bias: a Marian head is `final_logits_bias` plus
        // the tied embedding, and the bias is per vocabulary row.  It has to
        // go in before anything reads the argmax: the bias is per row, so it
        // can *pick* the token (minecraft's step-0 argmax is 6590, while the
        // pre-bias argmax this dump used to print was 8).
        for (size_t i = 0; i < lg.size() && i < m_.logits_bias.size(); ++i) lg[i] += m_.logits_bias[i];
        if (env_flag("ENCDEC_DUMP") && pos == 0 && !lg.empty()) {
                int best = 0;
                for (int v = 1; v < (int)lg.size(); ++v) if (lg[v] > lg[best]) best = v;
                cerr << "ENCDEC step0 argmax " << best << "\n";
        }
        return lg;
}
