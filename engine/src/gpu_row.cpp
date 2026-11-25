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
        // S is two rows: (mean*2^-k, sd*2^-k) and k.  The statistics are a
        // split scan (64 lanes: sum(x), sum(x^2)) and a reduce, not one serial
        // fragment: the pair is scale-free, so a residual far from O(1) no
        // longer loses the whole row to a fixed 1/256 step.
        if (ln_stat_.tex == 0 || ln_stat_.w != 1 || ln_stat_.h != 2)
                ln_stat_ = make_target(1, 2);
        if (ln_partial_.tex == 0 || ln_partial_.w != 64)
                ln_partial_ = make_target(64, 1);
        {
                GLuint ps = x_prog("lnsp_" + to_string(dim) + "_" + x_format_name(res_fmt_),
                                   ln_stats_split(dim, res_fmt_));
                bind_fbo(ln_partial_.fbo);
                glViewport(0, 0, 64, 1);
                use_program(ps);
                bind_tex(0, x); set_sampler(ps, "X", 0);
                setup_attrib(ps);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_LN");
        {
                GLuint pr = x_prog("lnrd_" + to_string(dim), ln_stats_reduce(dim));
                bind_fbo(ln_stat_.fbo);
                glViewport(0, 0, 1, 2);
                use_program(pr);
                bind_tex(0, ln_partial_.tex); set_sampler(pr, "S", 0);
                glUniform1f(uloc(pr, "eps"), eps);
                setup_attrib(pr);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_LN");
        if (getenv("ENCDEC_DUMP")) {
                static int ln_no = 0;
                sync();
                const vector<uint8_t> b = read_bytes(ln_stat_, 1, 2);
                const float ms = x_unpack(b.data(), 0, kLnPairFmt);
                const float ss = x_unpack(b.data(), 1, kLnPairFmt);
                const float kf = x_unpack(b.data(), 2, kLnPairFmt);
                const float sc = exp2f(kf);
                fprintf(stderr, "LNS %d mean %.3f sd %.3f (k %d)\n", ln_no++,
                        (double)(ms * sc), (double)(ss * sc), (int)kf);
        }
        if (getenv("ENCDEC_DUMP")) {
                static bool once = false;
                if (!once) { once = true;
                        cerr << "ENCDEC ln formats dim=" << dim
                             << " res=" << x_format_name(res_fmt_)
                             << " texels=" << x_texels(dim, res_fmt_)
                             << " act=" << x_format_name(act_fmt_) << "\n"; }
        }
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
                // The 32 lane sums, decoded by the format RMS_STORE selected --
                // through the same accessor the shader's codec comes from, so a
                // dump cannot read the bytes as the other format.  (It did: a
                // hardcoded 24-bit decode read the f32 store as fixed point and
                // the PPL came back 6399 instead of 270.32, 2026-11-18.)
                static int layer_no = 0;
                const bool want = !*rd || atoi(rd) == layer_no;
                if (want) {
                        sync();
                        const vector<uint8_t> b = read_bytes(rms_partial_, RMS_PARTIALS, 1);
                        const XFormat rf = rms_store_format();
                        fprintf(stderr, "RMSD %d %d", layer_no, RMS_PARTIALS);
                        for (int i = 0; i < RMS_PARTIALS && (size_t)(i * 4 + 3) < b.size(); ++i)
                                fprintf(stderr, " %.6g", (double)x_unpack(b.data(), i, rf));
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

GLuint GPU::act_sqrelu(GLuint x, Target& out){
        GLuint p = x_prog("act_sqrelu_" + to_string(out.w), ::act_sqrelu(out.w));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        draw_row(p, out.w);
        return out.tex;
}
