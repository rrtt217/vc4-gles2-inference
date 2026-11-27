// The upload/staging layer: host float vectors and Mats into the textures the
// kernels read, plus the two normalisations and the pair splitters that feed them.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  Every texture it
// makes goes through `make_texture`/`make_target` (gpu_state.cpp), which is what
// keeps the scene ledger complete for `release_scene()`.

#include "gpu.h"
#include "model.h"
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
        // Exhaustive on purpose (weight_format.h): a type without a case would
        // otherwise fall into the q4 `else` and read an empty body.
        switch (m.type) {
        case WeightType::T3: {
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
        case WeightType::INT6: {
            // Two planes: nibbles (cols/2 bytes a row) and two-bit values
            // (cols/4 bytes a row).  `tex` carries the low plane so the generic
            // draw() binding works; the high plane gets unit 2 there.
            m.tex_lo = make_texture(m.cols / 8, m.rows, m.lo6());
            m.tex_hi = make_texture(m.cols / 16, m.rows, m.hi6());
            m.tex = m.tex_lo;
            return;
        }
        case WeightType::INT8: {
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
            break;
        }
        case WeightType::INT16: {
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
            break;
        }
        case WeightType::INT4: {
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
            break;
        }
        }
        // A type that reached this point without a case has no texture; stop
        // rather than hand the session an empty one.
        if (m.tex == 0) wf_no_kernel("upload", m.type);
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
        return upload_hidden_fmt(x, res_fmt_);
    }

GLuint GPU::upload_hidden_fmt(const vector<float>& x, XFormat f){
        const int tw = x_texels((int)x.size(), f);
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
        vector<uint8_t> data = x_pack(x, f);
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

Target& GPU::v5_h(int b, int n){
        // A textured row is addressed by act_fmt_'s texel count, so the
        // texture has to *be* that wide.  grp_width's max(res, act, ffn)
        // sized these 512 texels for a 256-texel f32 row, and the fetch at
        // u=(i+0.5)/256 then read every other texel: RES=f32 alone gave an
        // encoder r1[0] of 1.45 against the reference's 0.31.
        const int w = x_texels(att_dim_, act_fmt_);
        if ((int)v5_h_.size() < n) v5_h_.resize(n);
        if (v5_h_[b].tex == 0 || v5_h_[b].w != w) v5_h_[b] = make_target(w, 1);
        return v5_h_[b];
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

// ---- the container's bias and LayerNorm rows (v5) -------------------------
// The v5 f32 rows, uploaded once per container.  A bias is a weight vector to
// the store (one q16 pair row), and a LayerNorm is the same packing two rows
// tall with the bias underneath -- so `ln_q16` reads y=0.25 for w and y=0.75
// for b, and `bias_add` reads a one-row texture.
void GPU::upload_v5_rows(const Model& m, XFormat qkf){
        EncdecRows& r = enc_rows_;
        // A bias rides the format of the tensor it is added to: a matvec writes
        // act_fmt_ (I24_1 by default) and a residual add reads res_fmt_, so the
        // same bias vector may need both encodings.  `one` is the residual one,
        // `one_act` the matvec's.
        auto one = [&](const float* v, int n) {
                return upload_bias(vector<float>(v, v + n), res_fmt_);
        };
        auto one_act = [&](const float* v, int n) {
                return upload_bias(vector<float>(v, v + n), act_fmt_);
        };
        // One row a vector, both through upload_bias: see EncdecRows.  A
        // LayerNorm's weight and bias are kLnPairFmt, the same layout
        // ln_apply_q16 reads (shaders.h owns the name).
        auto wrow = [&](const float* w, int n) { return upload_bias(
                vector<float>(w, w + n), kLnPairFmt); };
        const int L = m.layers, E = m.enc_layers, D = m.dim, F = m.enc_ffn;
        r.ln1.assign(L, 0); r.ln1b.assign(L, 0);
        r.ln2.assign(L, 0); r.ln2b.assign(L, 0);
        r.ln3.assign(L, 0); r.ln3b.assign(L, 0);
        r.enc_ln1.assign(E, 0); r.enc_ln1b.assign(E, 0);
        r.enc_ln2.assign(E, 0); r.enc_ln2b.assign(E, 0);
        r.qkv_b.assign(L, 0); r.o_b.assign(L, 0);
        r.o_b_rf.assign(L, 0); r.co_b_rf.assign(L, 0); r.fc2_b_rf.assign(L, 0);
        r.cq_b.assign(L, 0); r.ck_b.assign(L, 0); r.cv_b.assign(L, 0); r.co_b.assign(L, 0);
        r.fc1_b.assign(L, 0); r.fc2_b.assign(L, 0);
        r.enc_qkv_b.assign(E, 0); r.enc_o_b.assign(E, 0);
        r.enc_fc1_b.assign(E, 0); r.enc_fc2_b.assign(E, 0);
        r.enc_fc2_b_rf.assign(E, 0);
        r.enc_o_b_rf.assign(E, 0);
        for (int l = 0; l < L; ++l) {
                const int B = m.dec_bias_row(l, Model::kRowQ);
                r.ln1[l] = wrow(m.bias_dim_row(B + Model::kRowLn1W), D);
                r.ln1b[l] = wrow(m.bias_dim_row(B + Model::kRowLn1B), D);
                r.ln2[l] = wrow(m.bias_dim_row(B + Model::kRowLn2W), D);
                r.ln2b[l] = wrow(m.bias_dim_row(B + Model::kRowLn2B), D);
                r.ln3[l] = wrow(m.bias_dim_row(B + Model::kRowLn3W), D);
                r.ln3b[l] = wrow(m.bias_dim_row(B + Model::kRowLn3B), D);
                // q/k/v are one 3*dim row in the qkv texture, so their three
                // biases are concatenated rather than added one at a time.
                vector<float> qkv(m.bias_dim_row(B + Model::kRowQ),
                                  m.bias_dim_row(B + Model::kRowQ) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + Model::kRowK),
                           m.bias_dim_row(B + Model::kRowK) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + Model::kRowV),
                           m.bias_dim_row(B + Model::kRowV) + D);
                // The q/k/v bias is added into the qkv target, whose layout is
                // the pair/I24 one the rope, score and KV readers decode -- not
                // an axis format.  It was uploaded as ACT and added as RES, which
                // only agreed while both were x16_2.
                r.qkv_b[l] = upload_bias(qkv, qkf);
                r.o_b[l] = one(m.bias_dim_row(B + Model::kRowO), D);
                // The cross query's bias rides `qkv_target_`'s width, not `dim`:
                // `att_scores` reads the q out of that target through `RQKVQ`,
                // whose texel divisor is the compile-time `QKVTW` (3*dim).  A
                // `dim`-wide bias add shrank the target to a third and every
                // three samples landed on the same texel -- the q came out
                // replicated three times and the translation repeated one token.
                {
                        vector<float> pad(m.bias_dim_row(B + Model::kRowCQ),
                                          m.bias_dim_row(B + Model::kRowCQ) + D);
                        pad.resize((size_t)3 * D, 0.0f);
                        r.cq_b[l] = upload_bias(pad, qkf);
                }
                // The projection's format (`PROJ`), not a literal: the
                // matvec store is proj_fmt_ and `att_cross_kv` reads both
                // with one `X1`.
                r.ck_b[l] = upload_bias(vector<float>(m.bias_dim_row(B + Model::kRowCK),
                                               m.bias_dim_row(B + Model::kRowCK) + D),
                                 proj_fmt_);
                r.cv_b[l] = upload_bias(vector<float>(m.bias_dim_row(B + Model::kRowCV),
                                               m.bias_dim_row(B + Model::kRowCV) + D),
                                 proj_fmt_);
                r.co_b[l] = one(m.bias_dim_row(B + Model::kRowCO), D);
                r.fc1_b[l] = one_act(m.bias_ffn_row(l), F);
                r.fc2_b[l] = one(m.bias_dim_row(B + Model::kRowFc2B), D);
                // The fused decoder-bias arm reads these in the activation format.
                r.o_b_rf[l] = one_act(m.bias_dim_row(B + Model::kRowO), D);
                r.co_b_rf[l] = one_act(m.bias_dim_row(B + Model::kRowCO), D);
                r.fc2_b_rf[l] = one_act(m.bias_dim_row(B + Model::kRowFc2B), D);
        }
        for (int l = 0; l < E; ++l) {
                const int B = m.enc_row(l, 0);
                r.enc_ln1[l] = wrow(m.bias_dim_row(B + 4), D);
                r.enc_ln1b[l] = wrow(m.bias_dim_row(B + 5), D);
                r.enc_ln2[l] = wrow(m.bias_dim_row(B + 6), D);
                r.enc_ln2b[l] = wrow(m.bias_dim_row(B + 7), D);
                vector<float> qkv(m.bias_dim_row(B + 0), m.bias_dim_row(B + 0) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + 1), m.bias_dim_row(B + 1) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + 2), m.bias_dim_row(B + 2) + D);
                r.enc_qkv_b[l] = upload_bias(qkv, qkf);
                r.enc_o_b[l] = one(m.bias_dim_row(B + 3), D);
                r.enc_fc1_b[l] = one_act(m.bias_ffn_row(L + l), F);
                r.enc_fc2_b[l] = one(m.bias_dim_row(B + 8), D);
                // The fused w2 arm reads its base in the activation format, so
                // the bias it folds must be encoded the same way.
                r.enc_fc2_b_rf[l] = one_act(m.bias_dim_row(B + 8), D);
                // The fused wo arm reads its base in the activation format too.
                r.enc_o_b_rf[l] = one_act(m.bias_dim_row(B + 3), D);
        }
        // No logits-bias texture: a 65k-row bias is 32,501 texels wide, over
        // every GL target limit, and the head's bias is added on the host where
        // the logits already are (`Session::encdec_step`).
        r.logits_b = 0;
        // No position-table texture either: the table is added to the embedding
        // on the host (`Session::encdec_encode`), because a block with no
        // rotation has no kernel that would need it on the GPU.
        r.pos = 0;
}

// Upload a row into a *Target* rather than into a bare texture: a v5 position
// needs its residual to be readable and to have an FBO, and `upload_hidden_grp`
// hands back a make_texture() with no framebuffer at all -- which is fine for a
// draw source and wrong for anything a caller later reads or draws into.  The
// group slots (`grp_h(b, n)`) are Targets, so the encoder writes there and the
// ping-pong is the same one the prefill group loop keeps.
GLuint GPU::upload_into(Target& t, const vector<float>& x){
        return upload_into_fmt(t, x, res_fmt_);
    }

GLuint GPU::upload_into_fmt(Target& t, const vector<float>& x, XFormat f){
        const int tw = x_texels((int)x.size(), f);
        if (t.tex == 0 || t.w < tw) t = make_target(tw, 1);
        vector<uint8_t> data = x_pack(x, f);
        bind_upload_tex(t.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return t.tex;
}

GLuint GPU::upload_bias(const vector<float>& b, XFormat f){
        vector<uint8_t> data = x_pack(b, f);
        return make_texture(x_texels((int)b.size(), f), 1, data.data());
}
