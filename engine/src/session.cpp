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

Session::Session(Model& model, GPU& gpu) : m_(model), gpu_(gpu){
        gpu_layer_ = gpu_.gpu_layer_enabled();
        gpu_attn_ = gpu_layer_ || gpu_.gpu_attn_enabled();
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
            gpu_.res_fmt_ = gpu_.res_pack_ ? XFormat::I16_2 : XFormat::I16_1;
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
                                m_.q_norm, m_.k_norm, rope_cos_, rope_sin_, qkv_pair);
        if (gpu_layer_)
            gpu_.init_persistent(m_.dim, m_.layers, m_.rms_att, m_.rms_ffn, m_.rms_final);
    }

bool Session::save_prefix_cache(const string& path, const vector<int>& toks){
        if (!gpu_attn_ || gpu_.kv_cache_.empty() || toks.empty() || (int)toks.size() > m_.seq) return false;
        ofstream f(path, ios::binary);
        if (!f) return false;
        const char magic[8] = {'M','M','K','V','1','\0','\0','\0'};
        f.write(magic, 8);
        auto u32 = [&](uint32_t v) { f.write((char*)&v, 4); };
        auto u64 = [&](uint64_t v) { f.write((char*)&v, 8); };
        u32(2);
        u64(m_.source_size); u64(m_.source_mtime);
        u32(m_.dim); u32(m_.hidden); u32(m_.layers); u32(m_.heads);
        u32(m_.kv_heads); u32(m_.vocab); u32(m_.seq); u32(m_.head_dim);
        f.write((char*)&m_.rope_theta, 4);
        u32((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
            (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u) |
            (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u));
        u32((uint32_t)toks.size());
        u32((uint32_t)gpu_.att_kv_width_);
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
        gpu_.forget_state();
        return (bool)f;
    }

bool Session::load_prefix_cache(const string& path, vector<int>& toks){
        if (!gpu_attn_ || gpu_.kv_cache_.empty()) return false;
        ifstream f(path, ios::binary);
        if (!f) return false;
        char magic[8] {};
        f.read(magic, 8);
        if (!f || memcmp(magic, "MMKV1", 5) != 0) return false;
        auto u32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
        auto u64 = [&](uint64_t& v) { f.read((char*)&v, 8); };
        uint32_t version = 0, dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
        uint32_t vocab = 0, seq = 0, head_dim = 0, flags = 0, n = 0, width = 0;
        uint64_t size = 0, mtime = 0; float theta = 0, kscale = 0;
        u32(version); u64(size); u64(mtime);
        u32(dim); u32(hidden); u32(layers); u32(heads); u32(kv_heads);
        u32(vocab); u32(seq); u32(head_dim);
        f.read((char*)&theta, 4);
        u32(flags); u32(n); u32(width); f.read((char*)&kscale, 4);
        if (!f || version != 2 || dim != (uint32_t)m_.dim || hidden != (uint32_t)m_.hidden ||
            layers != (uint32_t)m_.layers || heads != (uint32_t)m_.heads ||
            kv_heads != (uint32_t)m_.kv_heads || vocab != (uint32_t)m_.vocab ||
            seq != (uint32_t)m_.seq || head_dim != (uint32_t)m_.head_dim ||
            size != m_.source_size || mtime != m_.source_mtime ||
            theta != m_.rope_theta || width != (uint32_t)gpu_.att_kv_width_ ||
            n == 0 || n > (uint32_t)m_.seq ||
            flags != ((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
                      (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u) |
                      (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u))) return false;
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
            vector<float> wo;
            if (gpu_attn_) {
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
                wo = gpu_.matvec(m_.wo[l], att_q16);
            }
            vector<float> x1 = add_q16(x, wo);
            vector<float> xbf = rms_norm(x1, m_.rms_ffn.data() + (size_t)l * m_.dim);
            vector<float> o2;
            bool gpu_ffn = !m_.w1.empty();
            if (getenv("CPU_FFN")) gpu_ffn = false;
            else if (getenv("GPU_FFN")) gpu_ffn = atoi(getenv("GPU_FFN")) != 0;
            if (!gpu_ffn) {
                vector<float> h;
                if (m_.ffn_chunked) {
                    vector<float> h1 = gpu_.matvec(m_.w1a[l], xbf);
                    vector<float> h3 = gpu_.matvec(m_.w3a[l], xbf);
                    vector<float> hh;
                    hh.reserve(2 * m_.ffn_chunk);
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> ha = swiglu(hh, m_.ffn_chunk);
                    h1 = gpu_.matvec(m_.w1b[l], xbf);
                    h3 = gpu_.matvec(m_.w3b[l], xbf);
                    hh.clear(); hh.reserve(2 * (m_.hidden - m_.ffn_chunk));
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> hb = swiglu(hh, m_.hidden - m_.ffn_chunk);
                    o2 = add_q16(gpu_.matvec(m_.w2a[l], ha), gpu_.matvec(m_.w2b[l], hb));
                } else {
                    if (!m_.w1.empty()) {
                        vector<float> h1 = gpu_.matvec(m_.w1[l], xbf);
                        vector<float> h3 = gpu_.matvec(m_.w3[l], xbf);
                        h.reserve(2 * m_.hidden);
                        h.insert(h.end(), h1.begin(), h1.end());
                        h.insert(h.end(), h3.begin(), h3.end());
                    } else {
                        h = gpu_.matvec(m_.w13[l], xbf);
                    }
                    o2 = gpu_.matvec(m_.w2[l], swiglu(h));
                }
            } else {
                GLuint xtex = gpu_.upload_input(xbf);
                GLuint htex = 0;
                if (m_.ffn_chunked) {
                    GLuint ha, hb;
                    gpu_.ffn_split_chunked(m_.w1a[l], m_.w3a[l], m_.w1b[l], m_.w3b[l], xtex,
                                           m_.ffn_chunk, m_.hidden - m_.ffn_chunk, ha, hb);
                    o2 = add_q16(gpu_.matvec_from(m_.w2a[l], ha), gpu_.matvec_from(m_.w2b[l], hb));
                } else if (!m_.w1.empty()) htex = gpu_.ffn_split(m_.w1[l], m_.w3[l], xtex, m_.hidden);
                else htex = gpu_.ffn_fused(m_.w13[l], xtex, m_.hidden);
                if (!m_.ffn_chunked) o2 = gpu_.matvec_from(m_.w2[l], htex);
            }
            x = add_q16(x1, o2);
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
            GLuint att_tex = gpu_.attention_block(m_.qkv[l], xb, l, pos);
            mark("attn", l);
            gpu_.matvec_add(m_.wo[l], att_tex, h, gpu_.hidden_b_, gpu_.act_fmt_);
            mark("wo", l);
            GLuint x1 = gpu_.hidden_b_.tex;
            GLuint xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
            if (pass_dup("rms_ffn")) xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
            mark("rms_ffn", l);
            if (m_.ffn_chunked) {
                GLuint ha, hb;
                gpu_.ffn_split_chunked(m_.w1a[l], m_.w3a[l], m_.w1b[l], m_.w3b[l], xbf,
                                       m_.ffn_chunk, m_.hidden - m_.ffn_chunk, ha, hb);
                if (getenv("FFN_DUMP") && l == 0) {
                    gpu_.sync();
                    gpu_.dump_target(gpu_.ffn_h_a_, "ha", 8);
                    gpu_.dump_target(gpu_.ffn_h_b_, "hb", 8);
                }
                mark("ffn", l);
                gpu_.matvec_add(m_.w2a[l], ha, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
                gpu_.matvec_add(m_.w2b[l], hb, gpu_.hidden_a_.tex, gpu_.hidden_b_, gpu_.ffn_fmt_);
                mark("w2", l);
                h = gpu_.hidden_b_.tex;
            if (getenv("HIDDEN_DUMP")) {
                gpu_.sync();
                gpu_.dump_target(gpu_.hidden_b_, "hidA", 4096);
            }
            } else {
                GLuint hffn;
                if (!m_.w1.empty()) hffn = gpu_.ffn_split(m_.w1[l], m_.w3[l], xbf, m_.hidden);
                else hffn = gpu_.ffn_fused(m_.w13[l], xbf, m_.hidden);
                mark("ffn", l);
                gpu_.matvec_add(m_.w2[l], hffn, x1, gpu_.hidden_a_, gpu_.ffn_fmt_);
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
        const char* e = getenv("PREFILL_PAIR");
        if (e && strcmp(e, "0") == 0) return false;
        // The packed kernels are INT8-only and read and write I16_2 activations.
        if (gpu_.act_fmt_ != XFormat::I16_2 || gpu_.ffn_fmt_ != XFormat::I16_2) return false;
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
        for (int h = 0; h < m_.heads; ++h) {
            int kvh = h / group;
            const float* kb = kv_cache_k_[layer].data() + (size_t)kvh * m_.head_dim;
            const float* vb = kv_cache_v_[layer].data() + (size_t)kvh * m_.head_dim;
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
