// The upload/staging layer: host float vectors and Mats into the textures the
// kernels read, plus the two normalisations and the pair splitters that feed them.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  Every texture it
// makes goes through `make_texture`/`make_target` (gpu_state.cpp), which is what
// keeps the scene ledger complete for `release_scene()`.

#include "gpu.h"
#include "switches.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"

#include <algorithm>
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
        if (m.type == WeightType::T3) {
            // The container's bytes *are* the texture: 16 weights an RGBA8
            // texel, four a byte.  No repack, no bias, no copy -- the upload is
            // the one place the 2-bit kind pays off twice (167 MB of file
            // becomes 167 MB of texture, where q8 needs 4x).
            // GL_MAX_TEXTURE_SIZE is 2048 on VC4 and the FFN gate is 4096 rows.
            // A taller body is folded into a 2D tile -- texel (x, y) of the
            // matrix lives at (x + (y/TH)*TW, y % TH) -- because the alternative
            // is what the board actually did: glTexImage2D at 4096 rows fails,
            // the texture keeps its previous contents, and the model answers
            // `ovarian` twelve times.  probes/t3mv_probe reproduces the edge: the
            // engine's own text is exact at 1024/1536/1792 rows and wrong in 1692
            // of 2048 rows at 2048 (2026-11-18).
            // A texel is 2 rows x 8 columns, so the texture is (cols/8) x
            // (rows/2) -- half the matrix's height, which is also what keeps
            // every shape in this model under the driver's 2048-texel limit.
            const int W = (m.cols + 7) / 8;
            const int prow = (m.rows + 1) / 2;
            // The tile geometry is decided HERE and only here: the shader reads
            // the two numbers back off the Mat.  Computing the count in the
            // uploader over texel rows and again in the Mat over matrix rows is
            // how a 2048-row chunk became a one-tile texture its own shader
            // fetched as two (2026-11-19).
            //
            // 1024, not the driver's 2048: the probe's edge is at exactly 2048
            // (exact at 1792, 1692 of 2048 rows wrong at 2048), the same limit
            // kDrawCap encodes for the draw strips.
            const int lim = std::min(kDrawCap, max_target_dim() > 0 ? max_target_dim() : kDrawCap);
            const int TH = std::min(prow, lim);
            const int tiles = (prow + TH - 1) / TH;
            m.t3_tile_w = W;
            m.t3_tile_h = TH;
            m.t3_tiles_n = tiles;
            if (getenv("T3_GEOM"))
                    fprintf(stderr, "T3GEOM up rows=%d cols=%d prow=%d W=%d TH=%d tiles=%d "
                            "lim=%d\n", m.rows, m.cols, prow, W, TH, tiles, lim);
            if (tiles == 1) {
                    // One tile is the body itself: no reshuffle, which is every
                    // matrix of this model (the tallest is the 2048-row FFN
                    // chunk, i.e. 1024 texel rows).
                    m.tex = make_texture(W, prow, m.w3());
            } else {
                    // The body is a row-major W x rows block; the tile wants
                    // row r at (x + (r/TH)*W, r%TH), so consecutive tiles sit
                    // side by side and each row has to move.  Uploading the
                    // block as W*tiles wide instead reads the next row's texels
                    // for every row past the first tile (cost: a wrong model
                    // with no error anywhere).
                    vector<uint8_t> tile((size_t)W * tiles * TH * 4, 0);
                    for (int r = 0; r < prow; ++r) {
                            const uint8_t* src = m.w3() + (size_t)r * W * 4;
                            uint8_t* dst = &tile[((size_t)(r % TH) * W * tiles +
                                                  (size_t)(r / TH) * W) * 4];
                            memcpy(dst, src, (size_t)W * 4);
                    }
                    m.tex = make_texture(W * tiles, TH, tile.data());
            }
            return;
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
            m.tex = m.fmt4 ? make_q4_texture(m.cols / 4, m.rows, src4)
                           : make_texture((m.cols + 7) / 8, m.rows, src4);
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
        // Allocate on demand: the two slots used to be created by
        // init_persistent(), which a v5 container skips (it has no model-wide
        // norm), so every decoder upload went to texture 0 and every read of it
        // came back black -- while the encoder, whose slots are Targets, was
        // fine.  An upload helper that hands back a name it never created is a
        // trap for the next caller too.
        if (hidden_in_ == 0 || hidden_in_w_ != tw) {
                hidden_in_ = make_texture(tw, 1, nullptr);
                hidden_in_w_ = tw;
        }
        vector<uint8_t> data = x_pack(x, res_fmt_);
        bind_upload_tex(hidden_in_);
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

// ---- Prefill query batching: the group's residual store --------------------
// N upload slots (one per position, packed before the layer loop) and N
// residual targets the group loop ping-pongs through grp_tmp_.  The width is
// the widest of the three stream formats, so a wo (act) or w2 (ffn) write into
// a residual slot never outruns the texture whatever the axis settings.

static int grp_width(int dim, XFormat res, XFormat act, XFormat ffn){
        return std::max(x_texels(dim, res), std::max(x_texels(dim, act), x_texels(dim, ffn)));
}

GLuint GPU::upload_hidden_grp(const vector<float>& x, int b, int n){
        const int rw = grp_width(att_dim_, res_fmt_, act_fmt_, ffn_fmt_);
        while ((int)grp_in_.size() < n) grp_in_.push_back(make_texture(rw, 1, nullptr));
        const int tw = x_texels((int)x.size(), res_fmt_);
        vector<uint8_t> data = x_pack(x, res_fmt_);
        bind_upload_tex(grp_in_[b]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return grp_in_[b];
}

Target& GPU::grp_h(int b, int n){
        const int rw = grp_width(att_dim_, res_fmt_, act_fmt_, ffn_fmt_);
        if ((int)grp_h_.size() < n) grp_h_.resize(n);
        if (grp_h_[b].tex == 0 || grp_h_[b].w != rw) grp_h_[b] = make_target(rw, 1);
        return grp_h_[b];
}

Target& GPU::grp_tmp(){
        const int rw = grp_width(att_dim_, res_fmt_, act_fmt_, ffn_fmt_);
        if (grp_tmp_.tex == 0 || grp_tmp_.w != rw) grp_tmp_ = make_target(rw, 1);
        return grp_tmp_;
}

Target& GPU::grp_w2a(){
        const int rw = grp_width(att_dim_, res_fmt_, act_fmt_, ffn_fmt_);
        if (grp_w2a_.tex == 0 || grp_w2a_.w != rw) grp_w2a_ = make_target(rw, 1);
        return grp_w2a_;
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
        // RMS_DUMP=1: the two scalars every layer's normalisation divides by.
        // A cross-machine difference that is a *pure scale* on the normalised
        // row (measured 1.00212 on layer 0, 2026-11-18) can only come from here
        // or from the weight vector, and the weight is the same bytes.
        if (const char* rd = getenv("RMS_DUMP")) {
                // The 32 lane sums as the shader wrote them (24-bit fixed point,
                // step 1/4096, biased by 2^23) -- the finest thing the rmsnorm
                // produces.  `RMS_DUMP=<layer>` keeps the log to one layer.
                static int layer_no = 0;
                const bool want = !*rd || atoi(rd) == layer_no;
                if (want) {
                        sync();
                        const vector<uint8_t> b = read_bytes(rms_partial_, RMS_PARTIALS, 1);
                        fprintf(stderr, "RMSD %d %d", layer_no, RMS_PARTIALS);
                        static const bool f32_store = [] {
                                const char* v = sw_str("RMS_STORE");
                                return v && !strcmp(v, "f32");
                        }();
                        for (int i = 0; i < RMS_PARTIALS && (size_t)(i * 4 + 3) < b.size(); ++i) {
                                if (f32_store) {
                                        const uint32_t u = ((uint32_t)b[i * 4] << 24) |
                                                           ((uint32_t)b[i * 4 + 1] << 16) |
                                                           ((uint32_t)b[i * 4 + 2] << 8) |
                                                           (uint32_t)b[i * 4 + 3];
                                        float f = 0;
                                        memcpy(&f, &u, 4);
                                        fprintf(stderr, " %.6g", (double)f);
                                } else {
                                        const long q = (long)b[i * 4] * 16777216L + (long)b[i * 4 + 1] * 65536L +
                                                       (long)b[i * 4 + 2] * 256L + (long)b[i * 4 + 3] - 8388608L;
                                        fprintf(stderr, " %ld", q);
                                }
                        }
                        fprintf(stderr, "\n");
                }
                ++layer_no;
        }

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
        GLuint p = x_i4_pair_swiglu(rows, w1.cols, hs, w1.fmt4);
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
