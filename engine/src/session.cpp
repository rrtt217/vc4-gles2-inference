#include "session.h"
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
static bool pass_dup(const char* name){
    static const std::string env = getenv("PASS_DUP") ? getenv("PASS_DUP") : "";
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


// KV8_SIM probe (default off): quantise the CPU float KV cache to 8 bits at
// read time.  The cache holds the same float K/V the GPU path quantises, and
// both paths consume the dequantised value, so this prices the grid without a
// GPU kernel.  Modes: 1 global scale on K and V, 2 per-head dynamic scale on
// both, 3 global K only, 4 global V only, 5 per-head V only, 6 per-head K only.
// KV8_SIM_SCALE sets the global grid (default 16, the KV_PACK=8 default).
static int kv8_sim_mode(){
    static const int m = getenv("KV8_SIM") ? atoi(getenv("KV8_SIM")) : 0;
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
        f.w1 = &m.w1[l]; f.w3 = &m.w3[l]; f.w2 = &m.w2[l];
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

Session::Session(Model& model, GPU& gpu) : m_(model), gpu_(gpu){
        gpu_layer_ = gpu_.gpu_layer_enabled();
        gpu_attn_ = gpu_layer_ || gpu_.gpu_attn_enabled();
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
                float a = p * freq;
                rope_cos_[(size_t)p * half + i] = cosf(a);
                rope_sin_[(size_t)p * half + i] = sinf(a);
            }
        }
        if (gpu_layer_) {
            const auto all_int8 = [](const vector<Mat>& v) {
                if (v.empty()) return false;
                for (const Mat& m : v) if (m.type != WeightType::INT8) return false;
                return true;
            };
            const vector<Mat>& w2 = !m_.w2a.empty() ? m_.w2a : m_.w2;
            const bool q8_model = all_int8(m_.qkv) && all_int8(m_.wo) && all_int8(w2);
            const char* a8 = getenv("ACT8");
            const int act8_mode = a8 ? atoi(a8) : 0;
            if (q8_model && (act8_mode == 1 || act8_mode == 2)) {
                if (act8_mode == 1) gpu_.act_fmt_ = XFormat::I8_4;
                gpu_.ffn_fmt_ = XFormat::I8_4;
            }
            // Residual stream packed two int16 per texel: halves the fetch count in
            // every rmsnorm pass (stats and norm both read the residual) and lets
            // matvec_add produce two rows per fragment.  Measured -11% decode on
            // the q8 model, perplexity unchanged.  RES_PACK=0 opts out.
            const char* rp = getenv("RES_PACK");
            gpu_.res_pack_ = !(rp && strcmp(rp, "0") == 0) &&
                             !m_.wo.empty() &&
                             (m_.wo[0].type == WeightType::INT8 || m_.wo[0].type == WeightType::INT4);
            // A container that declares a 24-bit residual gets one value per
            // texel; packing two values into a texel is a q16-only idea, so it
            // is switched off rather than reinterpreted.
            // RES_FMT=i16_1|i24_1 overrides the container's choice, so a
            // format can be A/B'd on the same file.
            if (const char* rf = getenv("RES_FMT")) {
                gpu_.res_pack_ = false;
                gpu_.res_fmt_ = strcmp(rf, "i24_1") == 0 ? XFormat::I24_1 : XFormat::I16_1;
                if (m_.res_q24 && gpu_.res_fmt_ != XFormat::I24_1)
                    cerr << "RES_FMT overrides the container's q24 residual\n";
            } else if (m_.res_q24) {
                // The residual-consuming kernels still store q16 (x_st1q), so
                // the GPU path would read a 24-bit texel holding a 16-bit
                // encode.  Refuse instead of answering from garbage; the CPU
                // path implements the format and is what this declares.
                gpu_.res_pack_ = false;
                gpu_.res_fmt_ = XFormat::I24_1;
            } else {
                gpu_.res_fmt_ = gpu_.res_pack_ ? XFormat::I16_2 : XFormat::I16_1;
            }
        }
        // Paired qkv matvec: two output rows per fragment, half the activation
        // decode.  Only the int8 checkpoint uses it; QKV_PAIR=0 restores the
        // single-row kernel.  The int4 pair kernel was tried and rejected: it
        // shifts q4 PPL 325.3751 -> 331.6324 and decodes 1.2% slower.
        bool qkv_pair = true;
        for (const Mat& m : m_.qkv)
            if (m.type != WeightType::INT8 || (m.rows & 1)) qkv_pair = false;
        if (getenv("QKV_PAIR") && strcmp(getenv("QKV_PAIR"), "0") == 0) qkv_pair = false;
        if (gpu_attn_)
            gpu_.init_attention(m_.dim, m_.kv_dim, m_.heads, m_.kv_heads, m_.head_dim,
                                m_.seq, m_.rope_half, m_.qk_norm, m_.layers,
                                m_.q_norm, m_.k_norm, rope_cos_, rope_sin_, qkv_pair,
                                m_.hybrid, m_.hybrid ? &m_.layer_type : nullptr);
        if (gpu_layer_)
            gpu_.init_persistent(m_.dim, m_.layers, m_.rms_att, m_.rms_ffn, m_.rms_final);
    }

void Session::zero_conv_state(){
        if (!m_.hybrid) return;
        if (gpu_layer_) { gpu_.clear_conv_state(); return; }
        for (auto& v : conv_state_)
                for (float& f : v) f = 0.0f;
}

bool Session::save_prefix_cache(const string& path, const vector<int>& toks){
        // A hybrid model's cache carries the conv history too (version 4, flag
        // bit 32); without it a reload would resume from a state that never
        // existed.  That needs the GPU conv path, which owns the state textures.
        if (m_.hybrid && (!gpu_layer_ || gpu_.conv_state_row_bytes() == 0)) return false;
        if (!gpu_attn_ || gpu_.kv_cache_.empty() || toks.empty() || (int)toks.size() > m_.seq) return false;
        ofstream f(path, ios::binary);
        if (!f) return false;
        const char magic[8] = {'M','M','K','V','1','\0','\0','\0'};
        f.write(magic, 8);
        auto u32 = [&](uint32_t v) { f.write((char*)&v, 4); };
        auto u64 = [&](uint64_t v) { f.write((char*)&v, 8); };
        // Version 3 adds an optional per-head V step plane (8-bit KV with
        // KV_VSTEP); without it the file is byte-for-byte a version 2 file
        // with a new version number, so the default path keeps writing 2.
        const int vsw = gpu_.att_kv_vstep_ ? gpu_.att_kv_heads_ : 0;
        u32(m_.hybrid ? 4u : vsw > 0 ? 3u : 2u);
        u64(m_.source_size); u64(m_.source_mtime);
        u32(m_.dim); u32(m_.hidden); u32(m_.layers); u32(m_.heads);
        u32(m_.kv_heads); u32(m_.vocab); u32(m_.seq); u32(m_.head_dim);
        f.write((char*)&m_.rope_theta, 4);
        u32((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
            (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u) |
            (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u) |
            (m_.hybrid ? 32u : 0u));
        u32((uint32_t)toks.size());
        u32((uint32_t)gpu_.att_kv_width_);
        // The field exists in every version >= 3 file, so a version 4 file
        // (hybrid) writes it even when the V-step plane is absent -- otherwise
        // the load's read shifts and the whole header mismatches.
        if (m_.hybrid || vsw > 0) u32((uint32_t)vsw);
        f.write((char*)&gpu_.att_kv_scale_, 4);
        for (int t : toks) u32((uint32_t)t);
        gpu_.sync();
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        const int width = gpu_.att_kv_width_, n = (int)toks.size();
        vector<uint8_t> data((size_t)width * n * 4);
        for (int l = 0; l < m_.layers; ++l) {
            glBindFramebuffer(GL_FRAMEBUFFER, gpu_.kv_cache_[l].fbo);
            glReadPixels(0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            f.write((char*)data.data(), data.size());
        }
        // The conv history: two q16 rows per conv layer (version 4 only).
        if (m_.hybrid) {
            vector<uint8_t> st;
            for (int l = 0; l < m_.layers; ++l) {
                if (!m_.is_conv(l)) continue;
                for (int which = 0; which < 2; ++which) {
                    if (!gpu_.read_conv_state(l, which, st)) return false;
                    f.write((char*)st.data(), st.size());
                }
            }
        }
        // 8-bit per-head V steps: without this plane a reloaded cache would
        // dequantise every V row on the wrong grid, so the file must carry it.
        for (int l = 0; l < m_.layers && vsw > 0; ++l) {
            glBindFramebuffer(GL_FRAMEBUFFER, gpu_.vs_targets_[l].fbo);
            glReadPixels(0, 0, vsw, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            f.write((char*)data.data(), (size_t)vsw * n * 4);
        }
        gpu_.forget_state();
        return (bool)f;
    }

bool Session::load_prefix_cache(const string& path, vector<int>& toks){
        if (m_.hybrid && (!gpu_layer_ || gpu_.conv_state_row_bytes() == 0)) return false;
        if (!gpu_attn_ || gpu_.kv_cache_.empty()) return false;
        ifstream f(path, ios::binary);
        if (!f) return false;
        char magic[8] {};
        f.read(magic, 8);
        if (!f || memcmp(magic, "MMKV1", 5) != 0) return false;
        auto u32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
        auto u64 = [&](uint64_t& v) { f.read((char*)&v, 8); };
        uint32_t version = 0, dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
        uint32_t vocab = 0, seq = 0, head_dim = 0, flags = 0, n = 0, width = 0, vsw = 0;
        uint64_t size = 0, mtime = 0; float theta = 0, kscale = 0;
        u32(version); u64(size); u64(mtime);
        u32(dim); u32(hidden); u32(layers); u32(heads); u32(kv_heads);
        u32(vocab); u32(seq); u32(head_dim);
        f.read((char*)&theta, 4);
        u32(flags); u32(n); u32(width);
        if (version >= 3) u32(vsw);   // per-head V step plane present
        f.read((char*)&kscale, 4);
        if (!f || (version != 2 && version != 3 && version != 4) || dim != (uint32_t)m_.dim ||
            hidden != (uint32_t)m_.hidden ||
            layers != (uint32_t)m_.layers || heads != (uint32_t)m_.heads ||
            kv_heads != (uint32_t)m_.kv_heads || vocab != (uint32_t)m_.vocab ||
            seq != (uint32_t)m_.seq || head_dim != (uint32_t)m_.head_dim ||
            size != m_.source_size || mtime != m_.source_mtime ||
            theta != m_.rope_theta || width != (uint32_t)gpu_.att_kv_width_ ||
            vsw != (uint32_t)(gpu_.att_kv_vstep_ ? gpu_.att_kv_heads_ : 0) ||
            n == 0 || n > (uint32_t)m_.seq ||
            flags != ((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
                      (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u) |
                      (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u) |
                      (m_.hybrid ? 32u : 0u))) return false;
        if (memcmp(&kscale, &gpu_.att_kv_scale_, sizeof(float)) != 0) {
            cerr << "PREFIX_CACHE KV_SCALE mismatch: cache " << kscale
                 << ", process " << gpu_.att_kv_scale_ << endl;
            return false;
        }
        vector<int> tmp(n);
        for (uint32_t i = 0; i < n; ++i) { uint32_t v = 0; u32(v); tmp[i] = (int)v; }
        if (!f) return false;
        vector<uint8_t> data((size_t)width * n * 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        for (int l = 0; l < m_.layers; ++l) {
            f.read((char*)data.data(), data.size());
            if (!f) return false;
            glBindTexture(GL_TEXTURE_2D, gpu_.kv_cache_[l].tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        }
        if (vsw > 0) {
            for (int l = 0; l < m_.layers; ++l) {
                f.read((char*)data.data(), (size_t)vsw * n * 4);
                if (!f) return false;
                glBindTexture(GL_TEXTURE_2D, gpu_.vs_targets_[l].tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vsw, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            }
        }
        if (version == 4) {
            vector<uint8_t> st((size_t)gpu_.conv_state_row_bytes());
            for (int l = 0; l < m_.layers; ++l) {
                if (!m_.is_conv(l)) continue;
                for (int which = 0; which < 2; ++which) {
                    f.read((char*)st.data(), st.size());
                    if (!f) return false;
                    if (!gpu_.write_conv_state(l, which, st)) return false;
                }
            }
        }
        gpu_.forget_state();
        gpu_.sync();
        toks = std::move(tmp);
        return true;
    }

vector<float> Session::embedding(int token){
        vector<float> x(m_.dim);
        if (m_.has_i16_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f((float)m_.emb_i16[(size_t)token * m_.dim + i] * s);
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
                x[i] = quantize_act_f((float)v * s);
            }
        } else if (m_.has_q8_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f((float)m_.emb_q8[(size_t)token * m_.dim + i] * s);
        } else {
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f(m_.emb_f32[(size_t)token * m_.dim + i]);
        }
        return x;
    }

vector<float> Session::forward(int token, int pos){
        vector<float> x = embedding(token);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return x;
            vector<float> xb = rms_norm(x, m_.rms_att.data() + (size_t)l * m_.dim);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(xb, gpu_.act_fmt_, "xb", 4096);
            vector<float> wo;
            if (m_.is_conv(l)) {
                // LFM2 conv block: in_proj -> gate -> depthwise causal conv
                // (k taps, state in conv_state_) -> gate -> out_proj.  Every
                // stage that the GPU stores as q16 is quantised here too, so
                // this is the reference the GPU kernel must reproduce.
                const int dim = m_.dim, k = m_.conv_k;
                vector<float> gb = gpu_.matvec(m_.conv_in_b[l], xb);
                vector<float> gc = gpu_.matvec(m_.conv_in_c[l], xb);
                vector<float> gx = gpu_.matvec(m_.conv_in_x[l], xb);
                vector<float> gated(dim), cv(dim);
                for (int i = 0; i < dim; ++i)
                    gated[i] = quantize_act_f(gb[i] * gx[i]);
                vector<float>& st = conv_state_[l];
                for (int i = 0; i < dim; ++i) {
                    float acc = 0.0f;
                    for (int j = 0; j < k; ++j) {
                        const float v = (j + 1 < k) ? st[(size_t)j * dim + i] : gated[i];
                        acc += m_.conv_w[l][(size_t)i * k + j] * v;
                    }
                    cv[i] = quantize_act_f(acc);
                }
                if (k > 1) {
                    for (int j = 0; j + 2 < k; ++j)
                        memcpy(st.data() + (size_t)j * dim, st.data() + (size_t)(j + 1) * dim,
                               sizeof(float) * dim);
                    memcpy(st.data() + (size_t)(k - 2) * dim, gated.data(), sizeof(float) * dim);
                }
                vector<float> y(dim);
                for (int i = 0; i < dim; ++i)
                    y[i] = quantize_act_f(gc[i] * cv[i]);
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
                wo = gpu_.matvec(m_.conv_out[l], y);
            } else if (gpu_attn_) {
                GLuint xtex = gpu_.upload_input(xb);
                GLuint atex = gpu_.attention_block(m_.qkv[l], xtex, l, pos);
                wo = gpu_.matvec_from(m_.wo[l], atex);
            } else {
                vector<float> qkv = gpu_.matvec(m_.qkv[l], xb);
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
                for (size_t i = 0; i < att.size(); ++i) att_q16[i] = quantize_act_f(att[i]);
                if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(att_q16, gpu_.act_fmt_, "att", 4096);
                wo = gpu_.matvec(m_.wo[l], att_q16);
            }
            vector<float> x1 = add_q16(x, wo);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(x1, gpu_.act_fmt_, "x1", 4096);
            vector<float> xbf = rms_norm(x1, m_.rms_ffn.data() + (size_t)l * m_.dim);
            if (getenv("STAGE_DUMP")) gpu_.dump_cpu_row(xbf, gpu_.act_fmt_, "xbf", 4096);
            vector<float> o2;
            int expert = -1;
            if (m_.is_moe() && getenv("MOE_FIXED")) {
                // Probe: skip the router and its read-back, pin one expert.
                expert = atoi(getenv("MOE_FIXED"));
            } else if (m_.is_moe()) {
                vector<float> gate = gpu_.matvec(m_.router[l], xbf);
                expert = pick_expert(gate);
                if (getenv("MOE_DUMP"))
                    cerr << "MOE l=" << l << " route=" << expert << " gate="
                         << gate[0] << "," << gate[1] << "," << gate[2] << "," << gate[3] << "\n";
            }
            const FfnMats f = ffn_mats(m_, l, expert);
            bool gpu_ffn = m_.is_moe() || !m_.w1.empty();
            if (getenv("CPU_FFN")) gpu_ffn = false;
            else if (getenv("GPU_FFN")) gpu_ffn = atoi(getenv("GPU_FFN")) != 0;
            if (!gpu_ffn) {
                vector<float> h;
                if (m_.ffn_chunked) {
                    vector<float> h1 = gpu_.matvec(*f.w1a, xbf);
                    vector<float> h3 = gpu_.matvec(*f.w3a, xbf);
                    vector<float> hh;
                    hh.reserve(2 * m_.ffn_chunk);
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> ha = swiglu(hh, m_.ffn_chunk);
                    h1 = gpu_.matvec(*f.w1b, xbf);
                    h3 = gpu_.matvec(*f.w3b, xbf);
                    hh.clear(); hh.reserve(2 * (f.width - m_.ffn_chunk));
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> hb = swiglu(hh, f.width - m_.ffn_chunk);
                    o2 = add_q16(gpu_.matvec(*f.w2a, ha), gpu_.matvec(*f.w2b, hb));
                } else {
                    if (f.w1->rows) {
                        vector<float> h1 = gpu_.matvec(*f.w1, xbf);
                        vector<float> h3 = gpu_.matvec(*f.w3, xbf);
                        h.reserve(2 * f.width);
                        h.insert(h.end(), h1.begin(), h1.end());
                        h.insert(h.end(), h3.begin(), h3.end());
                    } else {
                        h = gpu_.matvec(*f.w13, xbf);
                    }
                    o2 = gpu_.matvec(*f.w2, swiglu(h));
                }
            } else {
                GLuint xtex = gpu_.upload_input(xbf);
                GLuint htex = 0;
                if (m_.ffn_chunked) {
                    GLuint ha, hb;
                    gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xtex,
                                           m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb);
                    o2 = add_q16(gpu_.matvec_from(*f.w2a, ha), gpu_.matvec_from(*f.w2b, hb));
                } else if (f.w1->rows) htex = gpu_.ffn_split(*f.w1, *f.w3, xtex, f.width);
                else htex = gpu_.ffn_fused(*f.w13, xtex, f.width);
                if (!m_.ffn_chunked) o2 = gpu_.matvec_from(*f.w2, htex);
            }
            x = add_q16(x1, o2);
            if (getenv("HIDDEN_DUMP")) gpu_.dump_cpu_row(x, gpu_.act_fmt_, "hidA", 4096);
        }
        return x;
    }

GLuint Session::forward_gpu_hidden(int token, int pos, bool want_logits){
        bool prof = getenv("LAYER_PROFILE") != nullptr;
        auto pt = Clock::now();
        auto mark = [&](const char* name, int l) {
            if (!prof) return;
            gpu_.sync();
            auto now = Clock::now();
            cerr << "PROF l=" << l << " " << name << " " << elapsed_ms(pt, now) << "\n";
            pt = now;
        };
        vector<float> emb = embedding(token);
        GLuint h = gpu_.upload_hidden(emb);
        if (pass_dup("embed")) { emb = embedding(token); h = gpu_.upload_hidden(emb); }
        mark("embed", -1);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return h;
            GLuint xb = gpu_.rmsnorm(h, gpu_.rms_att_tex_[l], gpu_.xb_);
            if (pass_dup("rms_att")) xb = gpu_.rmsnorm(h, gpu_.rms_att_tex_[l], gpu_.xb_);
            mark("rms_att", l);
            if (getenv("STAGE_DUMP")) gpu_.dump_target(gpu_.xb_, "xb", 4096);
            if (m_.is_conv(l) && getenv("CONV_SKIP")) {
                gpu_.conv_copy(h, gpu_.hidden_b_);   // probe: conv layer = identity
            } else if (m_.is_conv(l)) {
                gpu_.conv_block(m_.conv_in_b[l], m_.conv_in_c[l], m_.conv_in_x[l],
                                m_.conv_out[l], xb, h, l, gpu_.hidden_b_);
                if (getenv("STAGE_DUMP")) gpu_.dump_target(gpu_.hidden_b_, "x1", 4096);
                mark("conv", l);
            } else {
                GLuint att_tex = gpu_.attention_block(m_.qkv[l], xb, l, pos);
                if (getenv("STAGE_DUMP")) gpu_.dump_target(gpu_.att_out_, "att", 4096);
                mark("attn", l);
                gpu_.matvec_add(m_.wo[l], att_tex, h, gpu_.hidden_b_, gpu_.act_fmt_);
                if (getenv("STAGE_DUMP")) gpu_.dump_target(gpu_.hidden_b_, "x1", 4096);
                mark("wo", l);
            }
            GLuint x1 = gpu_.hidden_b_.tex;
            GLuint xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
            if (getenv("STAGE_DUMP")) gpu_.dump_target(gpu_.xbf_, "xbf", 4096);
            if (pass_dup("rms_ffn")) xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
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
            if (m_.ffn_chunked) {
                GLuint ha, hb;
                gpu_.ffn_split_chunked(*f.w1a, *f.w3a, *f.w1b, *f.w3b, xbf,
                                       m_.ffn_chunk, f.width - m_.ffn_chunk, ha, hb);
                if (getenv("FFN_DUMP") && l == 0) {
                    gpu_.sync();
                    gpu_.dump_target(gpu_.ffn_h_a_, "ha", 8);
                    gpu_.dump_target(gpu_.ffn_h_b_, "hb", 8);
                }
                mark("ffn", l);
                gpu_.matvec_add(*f.w2a, ha, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                gpu_.matvec_add(*f.w2b, hb, gpu_.hidden_a_.tex, gpu_.hidden_b_, gpu_.ffn_fmt_);
                mark("w2", l);
                h = gpu_.hidden_b_.tex;
            if (getenv("HIDDEN_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.hidden_b_, "hidA", 4096);
            }
            } else {
                GLuint hffn;
                if (f.w1->rows) hffn = gpu_.ffn_split(*f.w1, *f.w3, xbf, f.width);
                else hffn = gpu_.ffn_fused(*f.w13, xbf, f.width);
                mark("ffn", l);
                gpu_.matvec_add(*f.w2, hffn, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                mark("w2", l);
                h = gpu_.hidden_a_.tex;
            if (getenv("HIDDEN_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.hidden_a_, "hidA", 4096);
            }
            }
        }
        if (!want_logits) return h;
        GLuint xf = gpu_.rmsnorm(h, gpu_.rms_final_tex_, gpu_.xf_);
        if (pass_dup("rms_final")) xf = gpu_.rmsnorm(h, gpu_.rms_final_tex_, gpu_.xf_);
        mark("rms_final", -1);
        return xf;
    }

// Prefill batching, steps 7c-7e.  The packed path handles two consecutive
// token positions in one layer pass: the FFN reads each q8 weight texel once
// and feeds both positions.  It is byte-identical to the serial path (every
// hidden state, the logits, the greedy strings and the KV bytes), and since
// step 7d it is also cheaper: the FFN input is built in P2 and the second w2
// chunk residual stays there, so a layer needs 14 draws against 18 serial and
// warm prefill CPU drops from 6.81-6.96 s to 5.66-5.74 s (21-token prompt);
// wall time is neutral within this board noise (+1.5% median over 14 warm
// samples, against a 6.30-7.50 s spread), so CPU is what the default buys.
// Step 7e makes it the default; PREFILL_PAIR=0 restores the serial loop.
// Attention and rmsnorm stay per position (latency, not weight traffic) and
// q4 / ACT8 / int16 / fused-w13 models keep the serial loop.
bool Session::pair_ok() const {
        if (!gpu_layer_) return false;
        // The paired path runs one FFN for two positions.  A MoE layer routes
        // each position on its own (and the chosen expert is a host-side
        // uniform), so the pair cannot share the FFN; the serial loop costs
        // the ~17% measured in the comment above, not 2x.
        if (m_.is_moe()) return false;
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
        // The packed kernels are INT8-only and read and write I16_2 activations.
        if (gpu_.act_fmt_ != XFormat::I16_2 || gpu_.ffn_fmt_ != XFormat::I16_2) return false;
        // The pair-prefill path stores two positions into one texel; a 24-bit
        // residual is one value per texel, so it cannot take part.
        if (x_is_int24(gpu_.res_fmt_)) return false;
        if ((m_.dim & 1) || (m_.hidden & 1)) return false;
        const vector<Mat>& w1 = !m_.w1a.empty() ? m_.w1a : m_.w1;
        const vector<Mat>& w3 = !m_.w3a.empty() ? m_.w3a : m_.w3;
        const vector<Mat>& w2 = !m_.w2a.empty() ? m_.w2a : m_.w2;
        if (w1.empty() || w3.empty() || w2.empty()) return false;
        return w1[0].type == WeightType::INT8 && w3[0].type == WeightType::INT8 &&
               w2[0].type == WeightType::INT8;
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
            gpu_.matvec_add(m_.wo[l], att_a, h_a, gpu_.pp_a2_, gpu_.act_fmt_);
            GLuint xbb = gpu_.rmsnorm(h_b, gpu_.rms_att_tex_[l], gpu_.xb2_);
            GLuint att_b = gpu_.attention_block(m_.qkv[l], xbb, l, pos_b);
            gpu_.matvec_add(m_.wo[l], att_b, h_b, gpu_.pp_b2_, gpu_.act_fmt_);
            plap("att");
            // 7d: the FFN input is built straight into P2 - one stats draw for
            // both positions, one normalising draw - so there is no pack pass,
            // and the second w2 chunk takes its residual from the first chunk P2
            // output, which removes the four splits 7c needed.  Only the two
            // splits that hand the layer output back as per-position residuals
            // remain.
            GLuint xin2 = gpu_.rmsnorm_pair2(gpu_.pp_a2_.tex, gpu_.pp_b2_.tex,
                                             gpu_.rms_ffn_tex_[l], gpu_.pp_in2_);
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

vector<float> Session::step(int token, int pos, bool want_logits){
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
        vector<float> xb = rms_norm(x, m_.rms_final.data());
        return gpu_.logits(m_.logits_tiles, xb);
    }

CandidateResult Session::step_candidates(int token, int pos, int topk, float temp, float top_p){
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

vector<float> Session::rms_norm(const vector<float>& x, const float* w){
        float ss = 0;
        for (float v : x) ss += v * v;
        ss /= x.size();
        float r = 1.0f / sqrtf(ss + 1e-5f);
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
        for (size_t i = 0; i < a.size(); ++i) o[i] = quantize_act_f(a[i] + b[i]);
        return o;
    }

vector<float> Session::swiglu(const vector<float>& h, int n){
        vector<float> o(n);
        for (int i = 0; i < n; ++i) {
            float a = h[i], b = h[n + i];
            o[i] = quantize_act_f((a / (1.0f + expf(-a))) * b);
        }
        return o;
    }

vector<float> Session::swiglu(const vector<float>& h){
        return swiglu(h, m_.hidden);
    }
