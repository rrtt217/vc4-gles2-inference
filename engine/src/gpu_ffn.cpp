// The FFN: split and fused SwiGLU, the 2-row packing, and the pair variants.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  The FFN is where
// the draw cap and the flush hooks earn their keep: `ffn_flush_wanted`/`ffn_sync`
// are the measurement hooks (14 sites, now two functions in gpu_state.cpp) and
// `kDrawCap` splits a 1216-wide chunk into strips.  The GLSL is in shaders.cpp.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"

#include <cstdio>
#include <string>
#include <vector>

GLuint GPU::ffn_split(const Mat& w1, const Mat& w3, GLuint x, int hidden, int hs){
        // The old non-paired q8 path (q24 matvec_to_row + swiglu_2row) has been
        // observed to miscompute w1/w3 for hidden=1408; the paired q16 path is
        // correct and also faster.  FFN_PAIR=0 keeps the old path for A/B.
        static const bool pair_env = !(env_zero("FFN_PAIR"));
        if (pair_env && w1.type == WeightType::INT8 && w3.type == WeightType::INT8)
            return ffn_pair_chunk(w1, w3, x, hidden, ffn_pair_a_, ffn_h_a_, hs);
        if (ffn_ab_.tex == 0 || ffn_ab_.w != hidden || ffn_ab_.h != 2) ffn_ab_ = make_target(hidden, 2);
        matvec_to_row(w1, x, ffn_ab_, 0);
        matvec_to_row(w3, x, ffn_ab_, 1);
        return swiglu_2row(ffn_ab_, hidden, hs);
    }

GLuint GPU::ffn_fused(const Mat& w13, GLuint x, int hidden, int hs){
        // q8 W13 uses the paired q16 path: the old q24 path miscomputes
        // fused-W13 FFN.  q4/i16 keep the q24 path.
        if (w13.type == WeightType::INT8) {
            const int aw = x_texels(hidden, ffn_fmt_);
            if (ffn_pair_a_.tex == 0 || ffn_pair_a_.w != hidden) ffn_pair_a_ = make_target(hidden, 1);
            if (ffn_h_a_.tex == 0 || ffn_h_a_.w != aw) ffn_h_a_ = make_target(aw, 1);
            const bool wr = w13.has_row_scale();
            GLuint p = x_q8_pair_fused(hidden, w13.cols, wr);
            bind_fbo(ffn_pair_a_.fbo);
            glViewport(0, 0, hidden, 1);
            use_program(p);
            bind_tex(0, w13.tex);
            set_sampler(p, "W", 0);
            bind_tex(1, x);
            set_sampler(p, "X", 1);
            glUniform1f(uloc(p, "ws"), w13.scale);
            if (wr) { bind_tex(2, w13.row_scale_tex); set_sampler(p, "WS", 2); }
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            GLuint ps = x_swiglu_pair(hidden, hs);
            bind_fbo(ffn_h_a_.fbo);
            glViewport(0, 0, aw, 1);
            use_program(ps);
            bind_tex(0, ffn_pair_a_.tex);
            set_sampler(ps, "A", 0);
            setup_attrib(ps);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
            else layer_finish("NF_FFN");
            ffn_sync();
            return ffn_h_a_.tex;
        }
        matvec_to(w13, x, ffn_a_);
        return swiglu_fused(ffn_a_, hidden, hs);
    }

void GPU::ffn_split_chunked(const Mat& w1a, const Mat& w3a, const Mat& w1b, const Mat& w3b,
                           GLuint x, int n1, int n2, GLuint& ha, GLuint& hb, int hs){
        static const bool pair_env = !(env_zero("FFN_PAIR"));
        // The paired shader assumes int8 weights; q4/i16 keep the old path.
        // It also assumes the *q16 pair* hidden layout: `matvec_i4_pair` and
        // `matvec_q8_pair` write (gate_r, up_r) as two q16 values in one texel,
        // and swiglu_pair reads that layout.  An f32 hidden is one value a
        // texel, so the pair path cannot express it -- f32 takes the
        // two-buffer path below (`swiglu_2row`, which has an f32 output).
        // Without this condition FFN_HID=f32 produced a garbage half-width
        // buffer and the model answered in one repeated token (2026-11-17).
        // The pair path's buffer is a q16 pair layout by construction; f32 and
        // bf16 both take the two-buffer path below (bf16 measured 60.93 against
        // 7148610 on pocket when it was let through, 2026-11-18).
        const bool pair = pair_env && ffn_fmt_ == XFormat::I16_2 &&
                          ((w1a.type == WeightType::INT8 && w3a.type == WeightType::INT8) ||
                           (w1a.type == WeightType::INT4 && w3a.type == WeightType::INT4));
        if (pair) {
            // Fused gate+swiglu: one draw per chunk instead of two.  Only the
            // q8 path has the fused kernel, and only I16_2 output is wired to w2.
            // Default ON (measured -20 ms/token).  The fused kernel differs from the two
            // pass sequence in the last bits of the swiglu output (PPL 312.1366 vs
            // 311.1636, 0.3% - far inside the 254-token corpus noise, see WORKLOG), but
            // the greedy gates are byte-identical.  FFN_FUSE=0 restores the two-pass
            // path for A/B and for any future bit-exactness work.
            static const bool fuse_env = !(env_zero("FFN_FUSE"));
            if (fuse_env && w1a.type == WeightType::INT8 && ffn_fmt_ == XFormat::I16_2 &&
                (n1 & 1) == 0 && (n2 & 1) == 0) {
                ha = ffn_pair_swiglu_chunk(w1a, w3a, x, n1, ffn_h_a_, hs);
                hb = ffn_pair_swiglu_chunk(w1b, w3b, x, n2, ffn_h_b_, hs);
                return;
            }
            // q4 has no fused gate: matvec_i4_pair writes one row per fragment
            // and a separate swiglu draw packs it.  The fused form computes the
            // row pair in one fragment, swiglus both rows in-kernel and writes
            // the packed q16 pair, so it is one draw per chunk and bit-identical
            // (same group order, same qz/sl rounding).
            if (fuse_env && w1a.type == WeightType::INT4 && ffn_fmt_ == XFormat::I16_2 &&
                (n1 & 1) == 0 && (n2 & 1) == 0) {
                ha = ffn_pair_i4_swiglu_chunk(w1a, w3a, x, n1, ffn_h_a_, hs);
                hb = ffn_pair_i4_swiglu_chunk(w1b, w3b, x, n2, ffn_h_b_, hs);
                return;
            }

            ha = ffn_pair_chunk(w1a, w3a, x, n1, ffn_pair_a_, ffn_h_a_, hs);
            hb = ffn_pair_chunk(w1b, w3b, x, n2, ffn_pair_b_, ffn_h_b_, hs);
            return;
        }
        if (ffn_ab_a_.tex == 0 || ffn_ab_a_.w != n1) ffn_ab_a_ = make_target(n1, 2);
        if (ffn_ab_b_.tex == 0 || ffn_ab_b_.w != n2) ffn_ab_b_ = make_target(n2, 2);
        matvec_to_row(w1a, x, ffn_ab_a_, 0);
        matvec_to_row(w3a, x, ffn_ab_a_, 1);
        ha = swiglu_2row(ffn_ab_a_, n1, ffn_h_a_, hs);
        matvec_to_row(w1b, x, ffn_ab_b_, 0);
        matvec_to_row(w3b, x, ffn_ab_b_, 1);
        hb = swiglu_2row(ffn_ab_b_, n2, ffn_h_b_, hs);
    }

GLuint GPU::swiglu_2row(Target& src, int hidden, int hs){
        return swiglu_2row(src, hidden, ffn_h_, hs);
    }

GLuint GPU::swiglu_2row(Target& src, int hidden, Target& out, int hs){
        const int aw = x_texels(hidden, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        GLuint p = x_swiglu_2row(hidden, hs);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, src.tex); set_sampler(p, "A", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
    }

GLuint GPU::swiglu_split(Target& a, Target& b){
        const int aw = x_texels(a.w, ffn_fmt_);
        if (ffn_h_.tex == 0 || ffn_h_.w != aw) ffn_h_ = make_target(aw, 1);
        GLuint p = x_swiglu(a.w, false);
        bind_fbo(ffn_h_.fbo);
        glViewport(0, 0, ffn_h_.w, 1);
        use_program(p);
        bind_tex(0, a.tex); set_sampler(p, "A", 0);
        bind_tex(1, b.tex); set_sampler(p, "B", 1);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return ffn_h_.tex;
    }

GLuint GPU::swiglu_fused(Target& src, int hidden, int hs){
        const int aw = (hidden + 1) / 2;
        if (ffn_h_.tex == 0 || ffn_h_.w != aw) ffn_h_ = make_target(aw, 1);
        GLuint p = x_swiglu(hidden, true, hs);
        bind_fbo(ffn_h_.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, src.tex); set_sampler(p, "A", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return ffn_h_.tex;
    }

GLuint GPU::ffn_pair_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out, int hs){
        const int aw = x_texels(rows, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        const bool wr = w1.has_row_scale() && w3.has_row_scale();
        GLuint p = x_q8_pair_swiglu(rows, w1.cols, wr, hs);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x); set_sampler(p, "X", 2);
        glUniform1f(uloc(p, "ws1"), w1.scale);
        glUniform1f(uloc(p, "ws2"), w3.scale);
        if (wr) {
            bind_tex(3, w1.row_scale_tex); set_sampler(p, "WS1", 3);
            bind_tex(4, w3.row_scale_tex); set_sampler(p, "WS2", 4);
        }
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
    }

GLuint GPU::ffn_pair_swiglu_chunk2(const Mat& w1, const Mat& w3, GLuint x2, int rows, Target& out){
        if (out.tex == 0 || out.w != rows) out = make_target(rows, 1);
        GLuint p = x_q8_gate2(rows, w1.cols);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x2); set_sampler(p, "X", 2);
        glUniform1f(uloc(p, "ws1"), w1.scale);
        glUniform1f(uloc(p, "ws2"), w3.scale);
        setup_attrib(p);
        for (int x0 = 0; x0 < rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
    }

GLuint GPU::ffn_pair_swiglu_chunk2_p2(const Mat& w1, const Mat& w3, GLuint xp2, int rows, Target& out){
        if (out.tex == 0 || out.w != rows) out = make_target(rows, 1);
        const bool wr = w1.has_row_scale() && w3.has_row_scale();
        GLuint p = x_q8_gate2_p2(rows, w1.cols, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, xp2); set_sampler(p, "X", 2);
        glUniform1f(uloc(p, "ws1"), w1.scale);
        glUniform1f(uloc(p, "ws2"), w3.scale);
        if (wr) {
            bind_tex(3, w1.row_scale_tex); set_sampler(p, "WS1", 3);
            bind_tex(4, w3.row_scale_tex); set_sampler(p, "WS2", 4);
        }
        setup_attrib(p);
        for (int x0 = 0; x0 < rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
    }

// Paired gate matvec + swiglu for one FFN chunk.
GLuint GPU::ffn_pair_chunk(const Mat& w1, const Mat& w3, GLuint x, int hidden, Target& pair_buf, Target& out, int hs){
        if (pair_buf.tex == 0 || pair_buf.w != hidden) pair_buf = make_target(hidden, 1);
        const int aw = x_texels(hidden, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        const bool i4 = (w1.type == WeightType::INT4);
        const bool wr = !i4 && w1.has_row_scale() && w3.has_row_scale();
        GLuint pp = i4 ? x_i4_pair(hidden, w1.cols) : x_q8_pair(hidden, w1.cols, wr);
        bind_fbo(pair_buf.fbo);
        glViewport(0, 0, hidden, 1);
        use_program(pp);
        bind_tex(0, w1.tex);
        set_sampler(pp, "W1", 0);
        bind_tex(1, w3.tex);
        set_sampler(pp, "W2", 1);
        bind_tex(2, x);
        set_sampler(pp, "X", 2);
        if (i4) {
            bind_tex(3, w1.q4_param_tex);
            set_sampler(pp, "P1", 3);
            bind_tex(4, w3.q4_param_tex);
            set_sampler(pp, "P2", 4);
            glUniform1f(uloc(pp, "ps1"), w1.q4_param_scale);
            glUniform1f(uloc(pp, "ps2"), w3.q4_param_scale);
        } else {
            glUniform1f(uloc(pp, "ws1"), w1.scale);
            glUniform1f(uloc(pp, "ws2"), w3.scale);
            if (wr) {
                bind_tex(3, w1.row_scale_tex); set_sampler(pp, "WS1", 3);
                bind_tex(4, w3.row_scale_tex); set_sampler(pp, "WS2", 4);
            }
        }
        setup_attrib(pp);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();

        GLuint ps = x_swiglu_pair(hidden, hs);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(ps);
        bind_tex(0, pair_buf.tex);
        set_sampler(ps, "A", 0);
        setup_attrib(ps);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
        return out.tex;
    }

void GPU::layer_finish(const char* skip_env){
        bool skip = skip_env && getenv(skip_env);
        if (skip_env && layer_fast_finish_ && !getenv("LAYER_SAFE_FINISH")) {
            string s(skip_env);
            if (s == "NF_RMS" || s == "NF_MAT" || s == "NF_ATT") skip = true;
        }
        if (skip) return;
        if (needs_finish()) { glFlush(); glFinish(); }
    }
