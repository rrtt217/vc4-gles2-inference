// The two-position prefill (P2): the gate/w2 matvecs and the pair split.
//
// The weight kernels are q8's P2 family (shaders_q8.cpp); this file is the
// driver.  It is a separate TU from the projection drivers because the P2
// scheme -- one texel per value per position -- is its own mechanism.
#include "gpu.h"
#include "gpu_internal.h"

#include <string>

void GPU::matvec_p2(const Mat& m, GLuint xp2, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        // No q4 kernel on this entry (the pair loop's w2 always adds); a loud
        // refusal beats decoding nibbles with the q8 kernel.
        if (m.type != WeightType::INT8) wf_no_kernel("matvec_p2", m.type);
        GLuint p = x_q8_p2(m.rows, m.cols, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

void GPU::matvec_p2_add_p2(const Mat& m, GLuint xp2, GLuint rp2, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        GLuint p = 0;
        switch (m.type) {
            case WeightType::INT8:
                p = x_q8_p2_add_p2(m.rows, m.cols, wr);
                break;
            case WeightType::INT4:
                p = x_i4_p2_add_p2(m.rows, m.cols);
                break;
            case WeightType::T3:
                p = x_t3_p2_add_p2(m, wr);
                break;
            case WeightType::INT16:
            case WeightType::INT6:
                wf_no_kernel("matvec_p2_add_p2", m.type);
        }
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        bind_tex(2, rp2); set_sampler(p, "RA", 2);
        if (m.type == WeightType::INT4) {
            bind_tex(3, m.q4_param_tex); set_sampler(p, "P", 3);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        } else {
            glUniform1f(uloc(p, "ws"), m.scale);
            if (wr) { bind_tex(4, m.row_scale_tex); set_sampler(p, "WS", 4); }
        }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

void GPU::matvec_p2_add(const Mat& m, GLuint xp2, GLuint ra, GLuint rb, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        GLuint p = 0;
        switch (m.type) {
            case WeightType::INT8:
                p = x_q8_p2_add(m.rows, m.cols, wr);
                break;
            case WeightType::INT4:
                p = x_i4_p2_add(m.rows, m.cols);
                break;
            case WeightType::T3:
                p = x_t3_p2_add(m, wr);
                break;
            case WeightType::INT16:
            case WeightType::INT6:
                wf_no_kernel("matvec_p2_add", m.type);
        }
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        bind_tex(2, ra); set_sampler(p, "RA", 2);
        bind_tex(3, rb); set_sampler(p, "RB", 3);
        if (m.type == WeightType::INT4) {
            bind_tex(4, m.q4_param_tex); set_sampler(p, "P", 4);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        } else {
            glUniform1f(uloc(p, "ws"), m.scale);
            if (wr) { bind_tex(4, m.row_scale_tex); set_sampler(p, "WS", 4); }
        }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
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
