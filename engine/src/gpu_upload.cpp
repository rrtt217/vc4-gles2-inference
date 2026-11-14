// The upload/staging layer: host float vectors and Mats into the textures the
// kernels read, plus the two normalisations and the pair splitters that feed them.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  Every texture it
// makes goes through `make_texture`/`make_target` (gpu_state.cpp), which is what
// keeps the scene ledger complete for `release_scene()`.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

void GPU::upload(Mat& m){
        // Weights may be borrowed straight from the checkpoint mmap, so every
        // reader goes through the w8()/w16()/w4() accessors; the +128 bias of INT8
        // cannot be applied in place to a read-only map.
        //
        // Per-row q8 scales (MMCF tensor id part 3) travel as one 24-bit value
        // per row: the row's scale divided by the matrix's unit `scale`, which
        // the shader multiplies back.  24 bits put the smallest row within
        // 6e-8 of the largest, far below the 0.79% weight error this exists to
        // remove.  Uploaded before the type branches because the prebiased q8
        // path returns early.
        if (!m.row_scale.empty()) {
                // The plane already holds s_row / unit; `scale` is the unit and
                // the shader multiplies it back.  Dividing again here would put
                // every row past 2^16 and clamp the plane to 1.0, which looks
                // exactly like "per-row scales do nothing".
                //
                // Two rows per texel, 16 bits each: .rg is row 2t and .ba is
                // row 2t+1.  A paired kernel then reads the texel once, because
                // two fetches from one sampler in one fragment come back
                // undefined on VC4 (probes/wrow_probe: 88 of 896 rows wrong on
                // the qkv shape, exact on Xe).  Four texels wide keeps the
                // texture out of the 1-texel-wide corner as well.
                const int W = 4;
                const int np = (m.rows + 1) / 2;
                vector<uint8_t> rs((size_t)np * W * 4, 0);
                for (int r = 0; r < m.rows; ++r) {
                        double v = (double)m.row_scale[(size_t)r];
                        long q = lrint(v * 65536.0);
                        if (q < 0) q = 0;
                        if (q > 65535) q = 65535;
                        // (2t, 2t+1) normally; (t, t+rows/2) for a fused W13
                        const int t = m.row_scale_hi
                                      ? (r < m.rows / 2 ? r : r - m.rows / 2) : r / 2;
                        const bool second = m.row_scale_hi ? (r >= m.rows / 2) : ((r & 1) != 0);
                        uint8_t* o = &rs[((size_t)t * W) * 4];
                        if (!second) { o[0] = (uint8_t)(q >> 8); o[1] = (uint8_t)q; }
                        else         { o[2] = (uint8_t)(q >> 8); o[3] = (uint8_t)q; }
                }
                m.row_scale_tex = make_texture(W, np, rs.data());
        }
        if (m.type == WeightType::INT6) {
            // Two planes: nibbles (cols/2 bytes a row) and two-bit values
            // (cols/4 bytes a row).  `tex` carries the low plane so the generic
            // draw() binding works; the high plane gets unit 2 there.
            m.tex_lo = make_texture(m.cols / 8, m.rows, m.lo6());
            m.tex_hi = make_texture(m.cols / 16, m.rows, m.hi6());
            m.tex = m.tex_lo;
            return;
        }
        if (m.type == WeightType::INT8) {
            const int8_t* src = m.w8();
            const size_t n = (size_t)m.rows * m.cols;
            vector<uint8_t> data(n);
            // +128 on int8 is a bit flip: (v + 128) & 255 == v ^ 0x80, so the
            // matrix converts four bytes at a time (UPLOAD_XOR=0 for an A/B).
            if (m.prebiased) {
                // Already biased at pack time: no arithmetic, just the copy.
                memcpy(data.data(), src, n);
                m.tex = make_texture(m.cols / 4, m.rows, data.data());
                return;
            }
            size_t i = 0;
            const char* xo = getenv("UPLOAD_XOR");
            if (!(xo && strcmp(xo, "0") == 0)) {
                while (i < n && ((size_t)(src + i) & 3u)) { data[i] = (uint8_t)((int)src[i] + 128); ++i; }
                for (; i + 4 <= n; i += 4) {
                    uint32_t v; memcpy(&v, src + i, 4); v ^= 0x80808080u;
                    memcpy(data.data() + i, &v, 4);
                }
            }
            for (; i < n; ++i) data[i] = (uint8_t)((int)src[i] + 128);
            m.tex = make_texture(m.cols / 4, m.rows, data.data());
        } else if (m.type == WeightType::INT16) {
            const int16_t* src = m.w16();
            const int tex_w = (m.cols + 1) / 2;
            vector<uint8_t> data((size_t)m.rows * tex_w * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int c = 0; c < m.cols; ++c) {
                    uint16_t u = (uint16_t)src[(size_t)r * m.cols + c];
                    int o = ((size_t)r * tex_w + c / 2) * 4;
                    if ((c & 1) == 0) { data[o] = u & 255; data[o + 1] = u >> 8; }
                    else               { data[o + 2] = u & 255; data[o + 3] = u >> 8; }
                }
            }
            m.tex = make_texture(tex_w, m.rows, data.data());
        } else {
            const uint8_t* src4 = m.w4();
            const int16_t* s4 = m.s4();
            const int16_t* mn4 = m.m4();
            m.tex = make_texture((m.cols + 7) / 8, m.rows, src4);
            vector<uint8_t> pd((size_t)m.rows * m.groups * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int g = 0; g < m.groups; ++g) {
                    uint16_t sc = (uint16_t)s4[(size_t)r * m.groups + g];
                    uint16_t mn = (uint16_t)mn4[(size_t)r * m.groups + g];
                    size_t o = ((size_t)r * m.groups + g) * 4;
                    pd[o + 0] = sc >> 8; pd[o + 1] = sc & 255;
                    pd[o + 2] = mn >> 8; pd[o + 3] = mn & 255;
                }
            }
            m.q4_param_tex = make_texture(m.groups, m.rows, pd.data());
        }
        if (!getenv("KEEP_CPU_WEIGHTS")) {
            vector<int8_t>().swap(m.q8);
            vector<int16_t>().swap(m.q16);
            if (m.type == WeightType::INT4) {
                vector<uint8_t>().swap(m.q4);
                vector<int16_t>().swap(m.q4_scale);
                vector<int16_t>().swap(m.q4_min);
            }
        }
    }

GLuint GPU::upload_input(const vector<float>& x){ return upload_activations(x); }

void GPU::init_persistent(int dim, int layers, const vector<float>& rms_att,
                        const vector<float>& rms_ffn, const vector<float>& rms_final){
        layer_fast_finish_ = true;
        // rmsnorm() sizes its target from att_dim_, which init_attention sets
        // on the fused path; the CPU loop's A/B probe calls rmsnorm without it.
        att_dim_ = dim;
        const int rw = x_texels(dim, res_fmt_);
        const int aw = x_texels(dim, act_fmt_);
        hidden_a_ = make_target(rw, 1);
        hidden_b_ = make_target(rw, 1);
        xb_ = make_target(aw, 1);
        xbf_ = make_target(aw, 1);
        xf_ = make_target(aw, 1);
        rms_stats_ = make_target(1, 1);
        hidden_in_ = make_texture(rw, 1, nullptr);
        hidden_in_b_ = make_texture(rw, 1, nullptr);
        rms_att_tex_.clear(); rms_ffn_tex_.clear();
        for (int l = 0; l < layers; ++l) {
            vector<float> a(rms_att.begin() + (size_t)l * dim, rms_att.begin() + (size_t)(l + 1) * dim);
            vector<float> f(rms_ffn.begin() + (size_t)l * dim, rms_ffn.begin() + (size_t)(l + 1) * dim);
            rms_att_tex_.push_back(make_q16_pair_texture(a));
            rms_ffn_tex_.push_back(make_q16_pair_texture(f));
        }
        rms_final_tex_ = make_q16_pair_texture(rms_final);
    }

GLuint GPU::upload_hidden(const vector<float>& x){
        const int tw = x_texels((int)x.size(), res_fmt_);
        vector<uint8_t> data = x_pack(x, res_fmt_);
        bind_tex(0, hidden_in_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return hidden_in_;
    }

GLuint GPU::upload_hidden_b(const vector<float>& x){
        const int tw = x_texels((int)x.size(), res_fmt_);
        vector<uint8_t> data = x_pack(x, res_fmt_);
        bind_tex(0, hidden_in_b_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return hidden_in_b_;
    }

GLuint GPU::rmsnorm(GLuint x, GLuint w, Target& out){
        const int aw = x_texels(att_dim_, act_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        if (rms_stats_.tex == 0) rms_stats_ = make_target(1, 1);
        if (rms_partial_.tex == 0 || rms_partial_.w != RMS_PARTIALS)
            rms_partial_ = make_target(RMS_PARTIALS, 1);
        GLuint ps = x_rms_split();
        bind_fbo(rms_partial_.fbo);
        glViewport(0, 0, RMS_PARTIALS, 1);
        use_program(ps);
        bind_tex(0, x);
        set_sampler(ps, "X", 0);
        setup_attrib(ps);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        // RMS_FUSED folds the 32-way reduce into every normalising fragment:
        // same summation order, so bit-identical, one draw less per rmsnorm.
        // VC4 defaults to the fused reduce (one draw less per rmsnorm, same
        // summation order so bit-identical); other backends keep the split pass,
        // where a 32-fetch fragment per output texel may cost more than a draw.
        // ACT8=1 (int8 activations) does not accept the fused form.
        const bool fused_reduce = rms_fused() && act_fmt_ != XFormat::I8_4;
        if (!fused_reduce) {
            GLuint pr = x_rms_reduce();
            bind_fbo(rms_stats_.fbo);
            glViewport(0, 0, 1, 1);
            use_program(pr);
            bind_tex(0, rms_partial_.tex);
            set_sampler(pr, "S", 0);
            setup_attrib(pr);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_RMS");

        GLuint pn = x_rms_norm(rms_eps_stated_);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(pn);
        if (rms_eps_stated_) glUniform1f(uloc(pn, "eps"), rms_eps_);
        bind_tex(0, x);
        set_sampler(pn, "X", 0);
        bind_tex(1, w);
        set_sampler(pn, "W", 1);
        bind_tex(2, fused_reduce ? rms_partial_.tex : rms_stats_.tex);
        set_sampler(pn, "S", 2);
        setup_attrib(pn);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        layer_finish("NF_RMS");
        return out.tex;
    }

GLuint GPU::upload_input2(const vector<float>& a, const vector<float>& b){
        const int n = (int)a.size();
        const int tw = x_texels(n, act_fmt_);
        vector<uint8_t> data = x_pack(a, act_fmt_);
        vector<uint8_t> db = x_pack(b, act_fmt_);
        if ((int)data.size() != tw * 4 || (int)db.size() != tw * 4) {
            cerr << "upload_input2: activation size mismatch\n";
            exit(1);
        }
        data.insert(data.end(), db.begin(), db.end());
        auto it = input_textures2_.find(n);
        if (it == input_textures2_.end()) {
            input_textures2_[n] = make_texture(2 * tw, 1, data.data());
            return input_textures2_[n];
        }
        bind_tex(0, it->second);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2 * tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return it->second;
    }

GLuint GPU::rmsnorm_pair2(GLuint xa, GLuint xb, GLuint w, Target& out){
        const int aw = att_dim_;   // P2: one texel per value per position
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        if (rms_partial2_.tex == 0 || rms_partial2_.w != 64) rms_partial2_ = make_target(64, 1);
        GLuint ps = x_rms_stats2();
        bind_fbo(rms_partial2_.fbo);
        glViewport(0, 0, 64, 1);
        use_program(ps);
        bind_tex(0, xa); set_sampler(ps, "XA", 0);
        bind_tex(1, xb); set_sampler(ps, "XB", 1);
        setup_attrib(ps);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        layer_finish("NF_RMS");
        GLuint pn = x_rms_norm_pair2(rms_eps_stated_);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(pn);
        if (rms_eps_stated_) glUniform1f(uloc(pn, "eps"), rms_eps_);
        bind_tex(0, xa); set_sampler(pn, "XA", 0);
        bind_tex(1, xb); set_sampler(pn, "XB", 1);
        bind_tex(2, w); set_sampler(pn, "W", 2);
        bind_tex(3, rms_partial2_.tex); set_sampler(pn, "S", 3);
        setup_attrib(pn);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        layer_finish("NF_RMS");
        return out.tex;
    }

void GPU::p2_split(GLuint xp2, Target& out, int which){
        const int ow = x_texels(att_dim_, res_fmt_);
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        GLuint p = x_p2_split(att_dim_);
        bind_fbo(out.fbo);
        glViewport(0, 0, ow, 1);
        use_program(p);
        bind_tex(0, xp2); set_sampler(p, "X", 0);
        glUniform1f(uloc(p, "which"), (float)which);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

GLuint GPU::ffn_pair_i4_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out, int hs){
        const int aw = x_texels(rows, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        GLuint p = x_i4_pair_swiglu(rows, w1.cols, hs);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x); set_sampler(p, "X", 2);
        bind_tex(3, w1.q4_param_tex); set_sampler(p, "P1", 3);
        bind_tex(4, w3.q4_param_tex); set_sampler(p, "P2", 4);
        glUniform1f(uloc(p, "ps1"), w1.q4_param_scale);
        glUniform1f(uloc(p, "ps2"), w3.q4_param_scale);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
}

GLuint GPU::upload_activations(const vector<float>& x){
        const int n = (int)x.size();
        const int texels = x_texels(n, act_fmt_);
        vector<uint8_t> data = x_pack(x, act_fmt_);
        auto it = input_textures_.find(n);
        if (it == input_textures_.end()) {
            GLuint t = make_texture(texels, 1, data.data());
            input_textures_[n] = t;
            return t;
        }
        bind_tex(0, it->second);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texels, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return it->second;
    }
