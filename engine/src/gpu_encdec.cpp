// The encoder-decoder (MMCF v5) GPU passes.
//
// A separate translation unit on purpose: this is not RWKV and not the dense
// decoder-only chain.  What lives here is the row-level plumbing a post-norm
// block needs and no other block does -- a LayerNorm whose stats travel as
// bit-cast f32, a bias add that rides whatever format its caller uses, the
// non-gated FFN's activation, and the one-time upload of the container's bias
// rows and LayerNorm (w,b) pairs.
//
// The shader *text* stays in src/shaders.cpp (v5 section): this file only binds
// programs, sizes targets and draws.  The block's structure -- encode pass,
// cross-attention, decode loop -- is Session's (session.cpp), because that is
// where the model's layer order lives.
#include "gpu.h"
#include "model.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::to_string;

// The v5 f32 rows, uploaded once per container.  A bias is a weight vector to
// the store (one q16 pair row), and a LayerNorm is the same packing two rows
// tall with the bias underneath -- so `ln_q16` reads y=0.25 for w and y=0.75
// for b, and `bias_add` reads a one-row texture.
void GPU::upload_encdec_rows(const Model& m){
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
        auto pair2 = [&](const float* w, const float* b, int n) {
                vector<float> v(w, w + n);
                v.insert(v.end(), b, b + n);
                return make_q16_pair2_texture(v);
        };
        const int L = m.layers, E = m.enc_layers, D = m.dim, F = m.enc_ffn;
        r.ln1.assign(L, 0); r.ln2.assign(L, 0); r.ln3.assign(L, 0);
        r.enc_ln1.assign(E, 0); r.enc_ln2.assign(E, 0);
        r.qkv_b.assign(L, 0); r.o_b.assign(L, 0);
        r.cq_b.assign(L, 0); r.ck_b.assign(L, 0); r.cv_b.assign(L, 0); r.co_b.assign(L, 0);
        r.fc1_b.assign(L, 0); r.fc2_b.assign(L, 0);
        r.enc_qkv_b.assign(E, 0); r.enc_o_b.assign(E, 0);
        r.enc_fc1_b.assign(E, 0); r.enc_fc2_b.assign(E, 0);
        for (int l = 0; l < L; ++l) {
                const int B = m.dec_bias_row(l, Model::kRowQ);
                r.ln1[l] = pair2(m.bias_dim_row(B + Model::kRowLn1W),
                                 m.bias_dim_row(B + Model::kRowLn1B), D);
                r.ln2[l] = pair2(m.bias_dim_row(B + Model::kRowLn2W),
                                 m.bias_dim_row(B + Model::kRowLn2B), D);
                r.ln3[l] = pair2(m.bias_dim_row(B + Model::kRowLn3W),
                                 m.bias_dim_row(B + Model::kRowLn3B), D);
                // q/k/v are one 3*dim row in the qkv texture, so their three
                // biases are concatenated rather than added one at a time.
                vector<float> qkv(m.bias_dim_row(B + Model::kRowQ),
                                  m.bias_dim_row(B + Model::kRowQ) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + Model::kRowK),
                           m.bias_dim_row(B + Model::kRowK) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + Model::kRowV),
                           m.bias_dim_row(B + Model::kRowV) + D);
                r.qkv_b[l] = upload_bias(qkv, act_fmt_);
                r.o_b[l] = one(m.bias_dim_row(B + Model::kRowO), D);
                r.cq_b[l] = one(m.bias_dim_row(B + Model::kRowCQ), D);
                r.ck_b[l] = one(m.bias_dim_row(B + Model::kRowCK), D);
                r.cv_b[l] = one(m.bias_dim_row(B + Model::kRowCV), D);
                r.co_b[l] = one(m.bias_dim_row(B + Model::kRowCO), D);
                r.fc1_b[l] = one_act(m.bias_ffn_row(l), F);
                r.fc2_b[l] = one(m.bias_dim_row(B + Model::kRowFc2B), D);
        }
        for (int l = 0; l < E; ++l) {
                const int B = m.enc_row(l, 0);
                r.enc_ln1[l] = pair2(m.bias_dim_row(B + 4), m.bias_dim_row(B + 5), D);
                r.enc_ln2[l] = pair2(m.bias_dim_row(B + 6), m.bias_dim_row(B + 7), D);
                vector<float> qkv(m.bias_dim_row(B + 0), m.bias_dim_row(B + 0) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + 1), m.bias_dim_row(B + 1) + D);
                qkv.insert(qkv.end(), m.bias_dim_row(B + 2), m.bias_dim_row(B + 2) + D);
                r.enc_qkv_b[l] = upload_bias(qkv, act_fmt_);
                r.enc_o_b[l] = one(m.bias_dim_row(B + 3), D);
                r.enc_fc1_b[l] = one_act(m.bias_ffn_row(L + l), F);
                r.enc_fc2_b[l] = one(m.bias_dim_row(B + 8), D);
        }
        // No logits-bias texture: a 65k-row bias is 32,501 texels wide, over
        // every GL target limit, and the head's bias is added on the host where
        // the logits already are (`Session::encdec_step`).
        r.logits_b = 0;
        // No position-table texture either: the table is added to the embedding
        // on the host (`Session::encdec_encode`), because a block with no
        // rotation has no kernel that would need it on the GPU.
        r.pos = 0;
        r.ready = true;
}

// ---- v5 (encoder-decoder) row passes -------------------------------------
// They live here because they are the same "one fullscreen draw, one row"
// shape as the RWKV passes, and they share make_q16_pair_texture with the
// rmsnorm weights, so a bias vector needs no new upload path.
GLuint GPU::ln_q16(GLuint x, GLuint wb, Target& out, float eps){
        const int dim = att_dim_;
        const int ow = x_texels(dim, act_fmt_);
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        if (ln_stat_.tex == 0 || ln_stat_.w != 2) ln_stat_ = make_target(2, 1);
        GLuint ps = x_prog("lns_" + to_string(dim) + "_" + to_string((int)res_fmt_),
                           ln_stats_q16(dim, res_fmt_));
        bind_fbo(ln_stat_.fbo);
        glViewport(0, 0, 2, 1);
        use_program(ps);
        bind_tex(0, x); set_sampler(ps, "X", 0);
        glUniform1f(uloc(ps, "eps"), eps);
        setup_attrib(ps);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        GLuint pn = x_prog("lna_" + to_string(dim) + "_" + to_string((int)res_fmt_) +
                           "_" + to_string((int)act_fmt_),
                           ln_apply_q16(dim, res_fmt_, act_fmt_));
        bind_fbo(out.fbo);
        glViewport(0, 0, ow, 1);
        use_program(pn);
        bind_tex(0, x); set_sampler(pn, "X", 0);
        bind_tex(1, ln_stat_.tex); set_sampler(pn, "S", 1);
        bind_tex(2, wb); set_sampler(pn, "W", 2);
        setup_attrib(pn);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        return out.tex;
}

GLuint GPU::bias_add(GLuint x, GLuint b, Target& out){
        return bias_add_wide(x, b, out, att_dim_);
}

GLuint GPU::upload_bias(const vector<float>& b, XFormat f){
        vector<uint8_t> data = x_pack(b, f);
        return make_texture(x_texels((int)b.size(), f), 1, data.data());
}

GLuint GPU::bias_add_wide(GLuint x, GLuint b, Target& out, int dim){
        return bias_add_f(x, b, out, dim, res_fmt_);
}

GLuint GPU::bias_add_f(GLuint x, GLuint b, Target& out, int dim, XFormat f){
        const int tw = x_texels(dim, f);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("badd_" + to_string(dim) + "_" + to_string((int)f),
                          bias_add_q16(dim, f));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        bind_tex(1, b); set_sampler(p, "B", 1);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        return out.tex;
}

GLuint GPU::act_swish(GLuint x, Target& out, int dim, XFormat f){
        const int tw = x_texels(dim, f);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("swish_" + to_string(dim) + "_" + to_string((int)f),
                          ::act_swish(dim, f));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        return out.tex;
}
