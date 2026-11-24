// The row-pass drivers: one fullscreen draw over one row.
//
// LayerNorm, a bias add and the non-gated FFN's swish.  They were introduced
// with the encoder-decoder container, but none of them knows what a transformer
// is: bias_add_wide is already called by the decoder-only qkv path
// (gpu_attn.cpp), and a decoder-only post-norm model would call the same
// passes.  The shader text is in shaders_block.cpp; the block's layer order is
// Session's.
#include "gpu.h"
#include "switches.h"

#include <string>
#include <vector>

using std::string;
using std::to_string;

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
