// The encoder-decoder (MMCF v5) GPU passes.
//
// A separate translation unit on purpose: this is not RWKV and not the dense
// decoder-only chain.  What lives here is the row-level plumbing a post-norm
// block needs and no other block does:
//
//   ln_q16        (mean, invstd) as ONE q16 pair in one texel, then the apply
//                 pass; the weight and its bias are two one-row textures
//                 because no XFORMAT macro addresses a second row
//   bias_add_*    a bias on whatever wrote x -- `bias_add_matvec` for a
//                 projection's I24 output, `bias_add_wide` for a residual
//   act_swish     the non-gated FFN's activation
//   upload_encdec_rows  the container's bias rows and LayerNorm pairs, uploaded
//                 from Session's constructor because they are packed *in the
//                 formats that constructor decides*
//
// Three of those comments used to describe the first shape of each pass (f32
// stats, a two-row weight texture, a single-format bias add) and stayed after the
// shape changed -- so read the code, not this block, and keep this block's list
// of *what is here* rather than of how it was.
//
// The shader *text* stays in src/shaders_block.cpp (post-norm row passes): this
// file only binds
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
        // One row a vector, both through upload_bias: see EncdecRows.
        auto wrow = [&](const float* w, int n) { return upload_bias(
                vector<float>(w, w + n), XFormat::I16_2); };
        const int L = m.layers, E = m.enc_layers, D = m.dim, F = m.enc_ffn;
        r.ln1.assign(L, 0); r.ln1b.assign(L, 0);
        r.ln2.assign(L, 0); r.ln2b.assign(L, 0);
        r.ln3.assign(L, 0); r.ln3b.assign(L, 0);
        r.enc_ln1.assign(E, 0); r.enc_ln1b.assign(E, 0);
        r.enc_ln2.assign(E, 0); r.enc_ln2b.assign(E, 0);
        r.qkv_b.assign(L, 0); r.o_b.assign(L, 0);
        r.cq_b.assign(L, 0); r.ck_b.assign(L, 0); r.cv_b.assign(L, 0); r.co_b.assign(L, 0);
        r.fc1_b.assign(L, 0); r.fc2_b.assign(L, 0);
        r.enc_qkv_b.assign(E, 0); r.enc_o_b.assign(E, 0);
        r.enc_fc1_b.assign(E, 0); r.enc_fc2_b.assign(E, 0);
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
                r.qkv_b[l] = upload_bias(qkv, act_fmt_);
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
                        r.cq_b[l] = upload_bias(pad, res_fmt_);
                }
                // I24 to match the projection it rides: `att_cross_kv` reads
                // both with one `X1`.
                r.ck_b[l] = upload_bias(vector<float>(m.bias_dim_row(B + Model::kRowCK),
                                               m.bias_dim_row(B + Model::kRowCK) + D),
                                 XFormat::I24_1);
                // I24 to match the projection it rides: `att_cross_kv` reads
                // both with one `X1`.
                r.cv_b[l] = upload_bias(vector<float>(m.bias_dim_row(B + Model::kRowCV),
                                               m.bias_dim_row(B + Model::kRowCV) + D),
                                 XFormat::I24_1);
                r.co_b[l] = one(m.bias_dim_row(B + Model::kRowCO), D);
                r.fc1_b[l] = one_act(m.bias_ffn_row(l), F);
                r.fc2_b[l] = one(m.bias_dim_row(B + Model::kRowFc2B), D);
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
}

// ---- v5 (encoder-decoder) row passes -------------------------------------
// They live here because they are the same "one fullscreen draw, one row"
// shape as the RWKV passes, and they share make_q16_pair_texture with the
// rmsnorm weights, so a bias vector needs no new upload path.
// Upload a row into a *Target* rather than into a bare texture: a v5 position
// needs its residual to be readable and to have an FBO, and `upload_hidden_grp`
// hands back a make_texture() with no framebuffer at all -- which is fine for a
// draw source and wrong for anything a caller later reads or draws into.  The
// group slots (`grp_h(b, n)`) are Targets, so the encoder writes there and the
// ping-pong is the same one the prefill group loop keeps.
GLuint GPU::upload_into(Target& t, const vector<float>& x){
        const int tw = x_texels((int)x.size(), res_fmt_);
        if (t.tex == 0 || t.w < tw) t = make_target(tw, 1);
        vector<uint8_t> data = x_pack(x, res_fmt_);
        bind_upload_tex(t.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return t.tex;
}

void GPU::attach_dump_fbo(Target& t){
        if (t.fbo) return;
        glGenFramebuffers(1, &t.fbo);
        bind_fbo(t.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
}

GLuint GPU::ln_q16(GLuint x, GLuint w, GLuint b, Target& out, float eps){
        const int dim = att_dim_;
        const int ow = x_texels(dim, act_fmt_);
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        // Two texels: the stats kernel writes mean at x<1 and invstd at x>=1,
        // and ln_apply reads them at uv 0.25 / 0.75 (texel 0 / texel 1).  A
        // one-wide target draws only the mean and leaves invstd at zero, which
        // makes the LayerNorm return its bias alone.
        // ONE texel: (mean, invstd) is an x_st2q pair in .rg/.ba.
        if (ln_stat_.tex == 0 || ln_stat_.w != 1) ln_stat_ = make_target(1, 1);
        if (getenv("ENCDEC_DUMP")) {
                static bool once = false;
                if (!once) { once = true;
                        cerr << "ENCDEC ln formats dim=" << dim
                             << " res=" << x_format_name(res_fmt_)
                             << " texels=" << x_texels(dim, res_fmt_)
                             << " act=" << x_format_name(act_fmt_) << "\n"; }
        }
        GLuint ps = x_prog("lns_" + to_string(dim) + "_" + to_string((int)res_fmt_),
                           ln_stats_q16(dim, res_fmt_));
        bind_fbo(ln_stat_.fbo);
        glViewport(0, 0, 1, 1);
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
        bind_tex(2, w); set_sampler(pn, "W", 2);
        bind_tex(3, b); set_sampler(pn, "B", 3);
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
        return bias_add_f(x, b, out, dim, res_fmt_, res_fmt_);
}

// A bias on a *projection's* output: the matvec wrote I24 (its store is hardcoded,
// 34 emitted sites), and the bias and the result ride the residual format
// because that is what the next reader decodes.  This is the pair of formats
// every `matvec_to` + bias site needs, named once so the three of them cannot
// drift apart.
GLuint GPU::bias_add_matvec(GLuint x, GLuint b, Target& out, int dim){
        return bias_add_f(x, b, out, dim, XFormat::I24_1, res_fmt_);
}

GLuint GPU::bias_add_f(GLuint x, GLuint b, Target& out, int dim, XFormat xf, XFormat f){
        const int tw = x_texels(dim, f);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("badd_" + to_string(dim) + "_" + to_string((int)xf) + "_" +
                              to_string((int)f),
                          bias_add_q16(dim, xf, f));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        bind_tex(1, b); set_sampler(p, "B", 1);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        return out.tex;
}

// swish(x + b): one draw where the two passes it replaces took three (the
// 2048-wide bias add splits into strips), and one texture round trip fewer.
GLuint GPU::act_swish_biased(GLuint x, GLuint b, Target& out, int dim, XFormat xf, XFormat f){
        const int tw = x_texels(dim, f);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("swishb_" + to_string(dim) + "_" + to_string((int)xf) + "_" +
                              to_string((int)f),
                          ::act_swish_biased(dim, xf, f));
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
