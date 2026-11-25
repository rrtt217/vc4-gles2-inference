// The hybrid (conv) block: LFM2.5's convolution layers replace attention, so they
// own their own state (three slots per layer), their own kernel fetch and their
// own three-draw sequence.
//
// Split out of gpu.cpp (2026-11-14).  The GLSL is not here -- `conv3_act`,
// `mul_act` and `copy_act` live in shaders_block.cpp with every other kernel, and
// `dump_shaders.cpp` lists them; this file is the driver, exactly like
// gpu_state.cpp is the object layer.  `conv_state_`/`conv_k_tex_` stay in gpu.h
// and `prefix_cache.cpp` reads and writes the state through the public
// read_conv_state()/write_conv_state() pair, so nothing changed outside.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"

#include <cstring>
#include <string>
#include <vector>

void GPU::init_conv(int layers, int dim){
        const int tw = x_texels(dim, act_fmt_);
        conv_state_.assign((size_t)layers * 3, Target());
        conv_cur_.assign(layers, 0);
        for (int l = 0; l < layers; ++l)
                for (int s = 0; s < 3; ++s) {
                        // The state starts at zero; a texture's contents are
                        // undefined, so clear every slot through the FBO.
                        conv_slot(l, s) = make_target(tw, 1);
                }
        clear_conv_state();
        conv_t_b_ = conv_t_c_ = conv_t_x_ = conv_cv_ = conv_y_ = Target();
        conv_texels_ = tw;
}

bool GPU::read_conv_state(int layer, int which, vector<uint8_t>& out){
        if (layer < 0 || (size_t)layer * 3 + 2 >= conv_state_.size() || conv_texels_ <= 0) return false;
        // which 0 is x[pos-2] (the older row), which 1 is x[pos-1].  The slot
        // that holds x[pos-1] is conv_cur_; the rotation writes forward, so the
        // older row is the next slot around.
        const int slot = which == 0 ? (conv_cur_[layer] + 2) % 3 : conv_cur_[layer];
        const Target& t = conv_slot(layer, slot);
        if (t.tex == 0) return false;
        out.resize((size_t)conv_texels_ * 4);
        glFinish();
        bind_fbo(t.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, conv_texels_, 1, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
        gl_state_forget_bindings();
        return true;
}

bool GPU::write_conv_state(int layer, int which, const vector<uint8_t>& in){
        if (layer < 0 || (size_t)layer * 3 + 2 >= conv_state_.size() || conv_texels_ <= 0) return false;
        if (in.size() != (size_t)conv_texels_ * 4) return false;
        // A reload puts the history where a fresh sequence would start: slot 0
        // is x[pos-1] and slot 2 is x[pos-2], so the rotation continues from 0.
        // Slot 1 is stale and is the first one the next token overwrites.
        const int slot = which == 0 ? 2 : 0;
        conv_cur_[layer] = 0;
        const Target& t = conv_slot(layer, slot);
        if (t.tex == 0) return false;
        glBindTexture(GL_TEXTURE_2D, t.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, conv_texels_, 1, GL_RGBA, GL_UNSIGNED_BYTE, in.data());
        gl_state_forget_bindings();
        return true;
}

void GPU::clear_conv_state(){
        // Alpha must be 0 too: I16_2 stores the second value in .ba, so a 1.0
        // alpha clear leaves that value at 255 (0.996) and the conv starts from
        // a state that is not zero -- measured as layer 0 cv = 62 instead of 0.
        for (auto& v : conv_state_) {
                if (!v.tex) continue;
                bind_fbo(v.fbo);
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
        }
        for (auto& c : conv_cur_) c = 0;
        // The clears bind FBOs behind the engine's back; a stale binding would
        // make the first draw of the next prefill read the wrong target.
        gl_state_forget_bindings();
}

GLuint GPU::conv_mul(GLuint a, XFormat af, GLuint b, XFormat bf, Target& out){
        const int tw = x_texels(att_dim_, act_fmt_);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        // The program text carries both source layouts, so the cache key must
        // too (and the width is a literal in the text, not a uniform).
        GLuint p = x_prog(string("conv_mul_") + x_format_name(af) + "_" + x_format_name(bf),
                          mul_act(att_dim_, act_fmt_, af, bf));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, a);
        set_sampler(p, "A", 0);
        bind_tex(1, b);
        set_sampler(p, "B", 1);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        ++draws_;
        return out.tex;
}

void GPU::set_conv_kernel(int layer, const float* w, int dim, int k){
        if ((int)conv_k_tex_.size() <= layer) conv_k_tex_.resize(layer + 1, 0);
        const int tw = (dim + 1) / 2;
        vector<uint8_t> px((size_t)tw * k * 4, 0);
        for (int t = 0; t < k; ++t) {
                for (int i = 0; i < dim; ++i) {
                        long q = lrintf(w[(size_t)i * k + t] * ACT_SCALE);
                        if (q < -32768) q = -32768;
                        if (q > 32767) q = 32767;
                        const uint16_t u = (uint16_t)(int16_t)q;
                        const size_t o = ((size_t)t * tw + (i / 2)) * 4 + (i % 2) * 2;
                        px[o] = (uint8_t)(u >> 8);
                        px[o + 1] = (uint8_t)(u & 0xFF);
                }
        }
        GLuint t = make_texture(tw, k, px.data());
        if (conv_k_tex_[layer]) glDeleteTextures(1, &conv_k_tex_[layer]);
        conv_k_tex_[layer] = t;
}

GLuint GPU::conv3(GLuint s0, GLuint s1, GLuint g, GLuint ktex, Target& out){
        // The output is a q16 pair (two channels per texel): the second gate
        // reads it as I16_2, and the matvec_add after that reads the gate's own
        // I16_2 output.
        const int tw = x_texels(att_dim_, act_fmt_);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("conv3", conv3_act(att_dim_, act_fmt_));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, s0);
        set_sampler(p, "S0", 0);
        bind_tex(1, s1);
        set_sampler(p, "S1", 1);
        bind_tex(2, g);
        set_sampler(p, "G", 2);
        bind_tex(3, ktex);
        set_sampler(p, "K", 3);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        ++draws_;
        return out.tex;
}

GLuint GPU::conv_copy(GLuint src, Target& out){
        const int tw = x_texels(att_dim_, act_fmt_);
        if (out.tex == 0 || out.w != tw) out = make_target(tw, 1);
        GLuint p = x_prog("conv_copy", copy_act(att_dim_, act_fmt_));
        bind_fbo(out.fbo);
        glViewport(0, 0, tw, 1);
        use_program(p);
        bind_tex(0, src);
        set_sampler(p, "A", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        ++draws_;
        return out.tex;
}

GLuint GPU::conv_block(const Mat& in_b, const Mat& in_c, const Mat& in_x,
                       const Mat& out_proj, GLuint xb, GLuint residual,
                       int layer, Target& out){
        matvec_to(in_b, xb, conv_t_b_);
        matvec_to(in_c, xb, conv_t_c_);
        matvec_to(in_x, xb, conv_t_x_);
        if (getenv("CONV_DEBUG") && layer == 0) {
            auto dump_q24 = [&](Target& t, const char* nm) {
                vector<int32_t> raw = read_i32(t, t.w, 1);
                fprintf(stderr, "GPUConvVec %s w=%d:", nm, t.w);
                for (int i = 0; i < 8 && i < (int)raw.size(); ++i)
                    fprintf(stderr, " %.4f", (float)raw[i] / ACT_SCALE);
                fprintf(stderr, "\n");
            };
            dump_q24(conv_t_b_, "in_proj_b");
            dump_q24(conv_t_c_, "in_proj_c");
            dump_q24(conv_t_x_, "in_proj_x");
        }
        // Three rotating state slots: conv_cur_ is x[pos-1], the slot one step
        // further round is x[pos-2], and the third (x[pos-3]) is dead.  The
        // gate's output *is* x[pos] for this layer, so it is written straight
        // into that dead slot and the shift costs nothing.  Indices are
        // distinct mod 3, so no draw reads a target another one writes.
        const int cur = conv_cur_[layer];
        const int prev = (cur + 2) % 3;
        const int nxt = (cur + 1) % 3;
        // conv_t_b_/conv_t_x_ are matvec_to outputs, so they are read in the
        // projection's format (PROJ), not a literal: an f32 store has no i24
        // texel to decode, and the misread turns the whole conv block to noise.
        const GLuint gated = conv_mul(conv_t_b_.tex, proj_fmt_, conv_t_x_.tex,
                                      proj_fmt_, conv_slot(layer, nxt));
        const GLuint cv = conv3(conv_slot(layer, prev).tex, conv_slot(layer, cur).tex, gated,
                                conv_k_tex_[layer], conv_cv_);
        conv_cur_[layer] = (uint8_t)nxt;
        const GLuint y = conv_mul(conv_t_c_.tex, proj_fmt_, cv, act_fmt_, conv_y_);
        matvec_add(out_proj, y, residual, out, act_fmt_);
        if (getenv("CONV_DEBUG") && layer == 0 && out.tex) {
            vector<uint8_t> b = read_bytes(out, out.w, 1);
            fprintf(stderr, "GPUConv x1r");
            for (int i = 0; i < 5; ++i) {
                int u = (int)b[i * 4] * 256 + (int)b[i * 4 + 1];
                if (u >= 32768) u -= 65536;
                fprintf(stderr, " %.4f", (float)u / 256.0f);
            }
            fprintf(stderr, " (w=%d)\n", out.w);
        }
        if (getenv("CONV_DEBUG") && layer == 0) {
            auto dump_act = [&](const Target& t, const char* nm) {
                vector<uint8_t> b = read_bytes(const_cast<Target&>(t), t.w, 1);
                fprintf(stderr, "GPUConv %s", nm);
                for (int i = 0; i < 5; ++i) {
                    const int o = (i / 2) * 4 + (i % 2) * 2;
                    int u = (int)b[o] * 256 + (int)b[o + 1];
                    if (u >= 32768) u -= 65536;
                    fprintf(stderr, " %.4f", (float)u / 256.0f);
                }
                fprintf(stderr, "\n");
            };
            dump_act(conv_slot(layer, nxt), "gated");
            dump_act(conv_cv_, "cv");
            dump_act(conv_y_, "y");
        }
        return out.tex;
}

