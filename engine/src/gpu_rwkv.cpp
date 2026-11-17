// The RWKV-7 (Goose) fused GPU layer loop (P6, docs/rwkv7-plan.zh-CN.md).
//
// The third layer loop's elementwise ops and WKV7 recurrence run on textures;
// the matvec family is reused unchanged (matvec_to / matvec_add).  Activations
// are I24_1, the residual stays q16 (I16_1) so it matches the CPU add_q16 grid
// bit-for-bit under the rail.  The only host read-back is the final residual,
// once per token (Session::forward_gpu_rwkv).  The GLSL is in shaders.cpp
// (rwkv_* builders); this file is the driver, like gpu_conv.cpp.
//
// WKV state is a per-head hs x hs block of bit-cast f32 in RGBA8 ping-pong
// textures (native ESSL 3.00: floatBitsToUint / uintBitsToFloat), so the
// recurrence runs in exact f32 like the CPU reference -- the S-1 fixed-point
// grid cost disappears.  Three passes (sa / update / out); no per-head scale and
// no read-back.  The norm statistics are f32 bit-cast too.  When the driver
// offers no ESSL 3.00 the session falls back to the host f32 state (hybrid).

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"
#include "model.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

void GPU::init_rwkv(const Model& m){
        act_fmt_ = XFormat::I24_1;
        ffn_fmt_ = XFormat::I24_1;
        res_fmt_ = XFormat::I16_1;
        res_pack_ = false;
        layer_fast_finish_ = true;
        const int dim = m.dim, L = m.layers, H = m.heads, hs = m.rwkv.head_size;
        rwkv_dim_ = dim; rwkv_layers_ = L; rwkv_heads_ = H; rwkv_hs_ = hs;
        rwkv_eps_ = m.rwkv.norm_eps > 0.0f ? m.rwkv.norm_eps : 1e-5f;
        rwkv_cur_ = 0; rwkv_have_vfirst_ = false;

        // Low-rank widths (the t intermediates); uniform across layers, but take
        // the max so a ragged container cannot undersize a scratch target.
        int rwm = 0, ram = 0, rvm = 0, rgm = 0, cmw = 0;
        for (int l = 0; l < L; ++l) {
            rwm = std::max(rwm, m.rwkv_w1[l].rows);
            ram = std::max(ram, m.rwkv_a1[l].rows);
            if (l < (int)m.rwkv_v1.size()) rvm = std::max(rvm, m.rwkv_v1[l].rows);
            if (l < (int)m.rwkv_g1.size()) rgm = std::max(rgm, m.rwkv_g1[l].rows);
            cmw = std::max(cmw, m.ffn_chunked ? m.w1a[l].rows : m.w1[l].rows);
        }
        rwkv_tmp_.assign(RT_COUNT, Target());
        auto mk = [&](RwkvTmp i, int w, int h = 1) { rwkv_tmp_[i] = make_target(w, h); };
        for (RwkvTmp i : {RT_XN, RT_XN2, RT_XR, RT_XW, RT_XK, RT_XV, RT_XA, RT_XG,
                          RT_R, RT_K, RT_KD, RT_V, RT_V2, RT_U, RT_DECAY,
                          RT_A, RT_G, RT_KK, RT_Y, RT_Y2, RT_RES, RT_RES2})
            mk(i, dim);
        mk(RT_TW, rwm); mk(RT_TW2, rwm); mk(RT_TA, ram); mk(RT_TV, rvm);
        mk(RT_TG, rgm); mk(RT_TG2, rgm);
        mk(RT_CMH, cmw); mk(RT_CMH2, cmw); mk(RT_CMH3, cmw); mk(RT_CMH4, cmw);
        mk(RT_LNSTAT, 2); mk(RT_GNSTAT, H, 2);
        mk(RT_L2STAT, H); mk(RT_RK, H); mk(RT_SA, H * hs);

        rwkv_state_.assign((size_t)L * 2, Target());
        rwkv_shift_.assign((size_t)L * 2, Target());
        for (int l = 0; l < L; ++l) {
            for (int b = 0; b < 2; ++b)
                rwkv_state_[(size_t)l * 2 + b] = make_target(hs, H * hs);
            rwkv_shift_[(size_t)l * 2 + 0] = make_target(dim, 1);
            rwkv_shift_[(size_t)l * 2 + 1] = make_target(dim, 1);
        }
        rwkv_vfirst_ = make_target(dim, 1);
        rwkv_x0_ = make_target(dim, 1);
        // upload_hidden's staging texture: init_persistent makes it on the
        // transformer fused path, which RWKV does not run.
        hidden_in_ = make_texture(x_texels(dim, res_fmt_), 1, nullptr);
        clear_rwkv_state();

        // Per-layer constant vectors, *64 into I24_1.  The multi-row ones
        // (lerp 6, scale 3, the norm affines 2) are 2D dim x rows textures so the
        // WIDTH is always dim: a flat 6*dim=4608-wide 1D texture exceeds VC4's
        // GL_MAX_TEXTURE_SIZE=2048 and glTexImage2D fails GL_INVALID_VALUE there
        // (Xe's 16384 hid it).  bias is split into its w0/a0/v0 dim-wide rows;
        // cm_lerp is one dim-wide row (the lerp kernel reads it as rows=1).
        if (dim > max_target_dim_) {
            cerr << "RWKV GPU: dim " << dim << " exceeds the GL target limit "
                 << max_target_dim_ << "; the fused path needs dim <= that\n";
            exit(1);
        }
        auto up = [&](const float* p, int n) {
            vector<float> s(p, p + n);
            for (float& v : s) v *= 64.0f;
            return make_q24_texture(s, 256.0f);
        };
        auto up2 = [&](const float* p, int rows) {
            vector<uint8_t> data((size_t)dim * rows * 4);
            for (int r = 0; r < rows; ++r)
                for (int i = 0; i < dim; ++i) {
                    long q = lrintf(p[(size_t)r * dim + i] * 64.0f * 256.0f) + 8388608;
                    if (q < 0) q = 0;
                    if (q > 16777215) q = 16777215;
                    const size_t o = ((size_t)r * dim + i) * 4;
                    data[o] = 0; data[o+1] = (uint8_t)(q >> 16);
                    data[o+2] = (uint8_t)(q >> 8); data[o+3] = (uint8_t)q;
                }
            return make_texture(dim, rows, data.data());
        };
        rwkv_const_.assign((size_t)L * RC_COUNT, 0);
        for (int l = 0; l < L; ++l) {
            const size_t b6 = (size_t)l * 6 * dim, b3 = (size_t)l * 3 * dim, b2 = (size_t)l * 2 * dim;
            rwkv_const_[(size_t)l * RC_COUNT + RC_LERP]     = up2(m.rwkv_lerp.data() + b6, 6);
            rwkv_const_[(size_t)l * RC_COUNT + RC_W0]       = up(m.rwkv_bias.data() + b3, dim);
            rwkv_const_[(size_t)l * RC_COUNT + RC_A0]       = up(m.rwkv_bias.data() + b3 + dim, dim);
            rwkv_const_[(size_t)l * RC_COUNT + RC_V0]       = up(m.rwkv_bias.data() + b3 + 2 * dim, dim);
            rwkv_const_[(size_t)l * RC_COUNT + RC_SCALE]    = up2(m.rwkv_scale.data() + b3, 3);
            rwkv_const_[(size_t)l * RC_COUNT + RC_ATTNorm]  = up2(m.rwkv_att_norm.data() + b2, 2);
            rwkv_const_[(size_t)l * RC_COUNT + RC_ATTNorm2] = up2(m.rwkv_att_norm_2.data() + b2, 2);
            rwkv_const_[(size_t)l * RC_COUNT + RC_LNX]      = up2(m.rwkv_ln_x.data() + b2, 2);
            rwkv_const_[(size_t)l * RC_COUNT + RC_CMLERP]   = up(m.rwkv_cm_lerp.data() + (size_t)l * dim, dim);
        }
}

// One host sync for a group of matvecs (docs/rwkv7-plan.zh-CN.md P7).  Each
// matrix keeps its own draw, its own program and its own uniforms -- the strip
// split, the q8 row-scale fetch and the 2^k input scaling all happen exactly as
// in matvec() -- and only the *read-back* is shared: the draws land in the rows
// of one target and a single glReadPixels takes the block.  The CPU path's cost
// is that sync (one per matvec, 14 per layer), not the draw.
GLuint GPU::rwkv_upload_slot(const vector<float>& x, int slot){
        const int n = (int)x.size();
        const int texels = x_texels(n, act_fmt_);
        vector<uint8_t> data = x_pack(x, act_fmt_);
        if ((int)rwkv_in_tex_.size() <= slot) {
                rwkv_in_tex_.resize((size_t)slot + 1, 0);
                rwkv_in_w_.resize((size_t)slot + 1, 0);
        }
        // The width has to be exactly x_texels(n): the kernels compute the
        // sample coordinate as (i+0.5)/axw from the *matrix's* column count, so a
        // wider texture silently samples other texels.  (Found the hard way: the
        // up projections, whose inputs are the rank-long t vectors, read a 384-
        // texel staging texture that had been made for a 768-long one.)
        GLuint t = rwkv_in_tex_[(size_t)slot];
        if (t == 0 || rwkv_in_w_[(size_t)slot] != texels) {
                t = make_texture(texels, 1, data.data());
                rwkv_in_tex_[(size_t)slot] = t;
                rwkv_in_w_[(size_t)slot] = texels;
                return t;
        }
        bind_tex(0, t);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texels, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return t;
}

vector<vector<float>> GPU::matvec_rows(const vector<MvRow>& rows, int slot0){
        vector<vector<float>> out(rows.size());
        if (rows.empty()) return out;
        int w = 0;
        for (const MvRow& r : rows) w = std::max(w, r.m->rows);
        // One target for every group: grow monotonically, never shrink.  A
        // per-call make_target would put a texture in the release ledger per
        // layer per token (the create sites are not guarded, on purpose).
        Target& t = rwkv_mvrows_;
        if (t.tex == 0 || t.w < w || t.h < (int)rows.size())
                t = make_target(std::max(w, t.w), std::max((int)rows.size(), t.h));
        for (size_t i = 0; i < rows.size(); ++i) {
                GLuint xt = rwkv_upload_slot(*rows[i].x, slot0 + (int)i);
                draw(*rows[i].m, xt, t, (int)i);
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        vector<int32_t> raw = read_i32(t, w, (int)rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
                const int n = rows[i].m->rows;
                out[i].resize((size_t)n);
                for (int j = 0; j < n; ++j) {
                        int v = raw[(size_t)i * (size_t)w + (size_t)j];
                        if (v >= 32768) v -= 65536;
                        out[i][(size_t)j] = (float)v / ACT_SCALE;
                }
        }
        return out;
}
void GPU::clear_rwkv_state(){
        // The WKV state is bit-cast f32: +0.0 is the texel (0,0,0,0), so black
        // is the correct zero.  The token-shift rows and v_first are I24_1
        // (offset-binary), whose zero is 0x800000 (g=128), NOT black.
        for (auto& t : rwkv_state_) { if (t.tex) { bind_fbo(t.fbo); glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT); } }
        const float z = 128.0f / 255.0f;
        for (auto& t : rwkv_shift_) { if (t.tex) { bind_fbo(t.fbo); glClearColor(0,z,0,0); glClear(GL_COLOR_BUFFER_BIT); } }
        if (rwkv_vfirst_.tex) { bind_fbo(rwkv_vfirst_.fbo); glClearColor(0,z,0,0); glClear(GL_COLOR_BUFFER_BIT); }
        rwkv_cur_ = 0; rwkv_have_vfirst_ = false;
        gl_state_forget_bindings();
}

std::vector<float> GPU::rwkv_read_residual(GLuint tex){
        Target* t = nullptr;
        if (tex == rwkv_tmp(RT_RES).tex) t = &rwkv_tmp(RT_RES);
        else if (tex == rwkv_tmp(RT_RES2).tex) t = &rwkv_tmp(RT_RES2);
        if (!t) return {};
        sync();
        vector<uint8_t> b = read_bytes(*t, t->w, 1);
        vector<float> out((size_t)t->w);
        for (int i = 0; i < t->w; ++i) {
            int u = (int)b[(size_t)i*4] * 256 + (int)b[(size_t)i*4+1];
            if (u >= 32768) u -= 65536;
            out[(size_t)i] = (float)u / ACT_SCALE;
        }
        return out;
}

// One full-target draw of an already-bound program (viewport w x h).
void GPU::rwkv_draw1(GLuint prog, int w, int h){
        setup_attrib(prog);
        if (needs_finish()) {
            // VC4 silently drops a single draw wider than kDrawCap (the sqrelu of
            // a 1536-wide chunked cm hidden left its target at the clear color).
            // Split into strips; gl_FragCoord.x is absolute, so the shader is
            // unchanged.  Xe drives the wide draw in one piece (its own rounding).
            for (int x0 = 0; x0 < w; x0 += kDrawCap) {
                glViewport(x0, 0, std::min(kDrawCap, w - x0), h);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            }
        } else {
            glViewport(0, 0, w, h);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
}

// A matvec into a pre-sized scratch target (no read-back).  Unlike matvec_to it
// does not recreate the target when the row count differs from a previous use:
// the RWKV scratch is sized once at init, and a narrower draw just leaves the
// tail texels unread.
GLuint GPU::rwkv_mv(const Mat& m, GLuint x, Target& out){
        if (out.tex == 0) out = make_target(m.rows, 1);
        draw(m, x, out, 0);
        return out.tex;
}

void GPU::rwkv_copy(GLuint src, Target& dst){
        const int tw = dst.w;
        if (dst.tex == 0) dst = make_target(tw, 1);
        GLuint p = x_prog("rwkv_copy_" + to_string(tw), copy_act(tw, XFormat::I24_1));
        bind_fbo(dst.fbo);
        use_program(p);
        bind_tex(0, src); set_sampler(p, "A", 0);
        rwkv_draw1(p, tw);
}

GLuint GPU::rwkv_ln(GLuint x, GLuint wb, Target& out){
        const int dim = rwkv_dim_;
        Target& stat = rwkv_tmp(RT_LNSTAT);
        GLuint ps = x_prog("rwkv_lns_" + to_string(dim), rwkv_ln_stats(dim, res_fmt_));
        bind_fbo(stat.fbo); use_program(ps);
        bind_tex(0, x); set_sampler(ps, "X", 0);
        glUniform1f(uloc(ps, "eps"), rwkv_eps_);
        rwkv_draw1(ps, 2);
        GLuint pn = x_prog("rwkv_lnn_" + to_string(dim), rwkv_ln_norm(dim, res_fmt_));
        bind_fbo(out.fbo); use_program(pn);
        bind_tex(0, x); set_sampler(pn, "X", 0);
        bind_tex(1, stat.tex); set_sampler(pn, "S", 1);
        bind_tex(2, wb); set_sampler(pn, "W", 2);
        rwkv_draw1(pn, dim);
        return out.tex;
}

GLuint GPU::rwkv_upload_x0(const vector<float>& x0){
        const int dim = rwkv_dim_;
        if (rwkv_x0_.tex == 0) rwkv_x0_ = make_target(dim, 1);
        vector<float> s(x0.size());
        for (size_t i = 0; i < x0.size(); ++i) s[i] = x0[i] * 256.0f;
        vector<uint8_t> data = x_pack(s, XFormat::I24_1);
        bind_tex(0, rwkv_x0_.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        return rwkv_x0_.tex;
}

GLuint GPU::rwkv_ln_p(GLuint x, GLuint wb, Target& out){
        const int dim = rwkv_dim_;
        Target& stat = rwkv_tmp(RT_LNSTAT);
        GLuint ps = x_prog("rwkv_lnsp_" + to_string(dim), ::rwkv_ln_stats_p(dim));
        bind_fbo(stat.fbo); use_program(ps);
        bind_tex(0, x); set_sampler(ps, "X", 0);
        glUniform1f(uloc(ps, "eps"), rwkv_eps_);
        rwkv_draw1(ps, 2);
        GLuint pn = x_prog("rwkv_lnp_" + to_string(dim), ::rwkv_ln_norm_p(dim));
        bind_fbo(out.fbo); use_program(pn);
        bind_tex(0, x); set_sampler(pn, "X", 0);
        bind_tex(1, stat.tex); set_sampler(pn, "S", 1);
        bind_tex(2, wb); set_sampler(pn, "W", 2);
        rwkv_draw1(pn, dim);
        return out.tex;
}

GLuint GPU::rwkv_res_add(GLuint resid, GLuint delta, Target& out){
        const int dim = rwkv_dim_;
        if (out.tex == 0) out = make_target(dim, 1);
        GLuint p = x_prog("rwkv_res_add_" + to_string(dim), ::rwkv_res_add(dim));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, resid); set_sampler(p, "R", 0);
        bind_tex(1, delta); set_sampler(p, "D", 1);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_first_add(GLuint x0, GLuint delta, Target& out){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_fadd_" + to_string(dim), ::rwkv_first_add(dim));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, x0); set_sampler(p, "P", 0);
        bind_tex(1, delta); set_sampler(p, "D", 1);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_gn(GLuint y, GLuint wb, Target& out){
        const int dim = rwkv_dim_, hs = rwkv_hs_, H = rwkv_heads_;
        Target& stat = rwkv_tmp(RT_GNSTAT);
        GLuint ps = x_prog("rwkv_gns_" + to_string(dim) + "_" + to_string(hs), rwkv_gn_stats(dim, hs));
        bind_fbo(stat.fbo); use_program(ps);
        bind_tex(0, y); set_sampler(ps, "Y", 0);
        rwkv_draw1(ps, H, 2);
        GLuint pn = x_prog("rwkv_gnn_" + to_string(dim) + "_" + to_string(hs), rwkv_gn_norm(dim, hs));
        bind_fbo(out.fbo); use_program(pn);
        bind_tex(0, y); set_sampler(pn, "Y", 0);
        bind_tex(1, stat.tex); set_sampler(pn, "S", 1);
        bind_tex(2, wb); set_sampler(pn, "W", 2);
        rwkv_draw1(pn, dim);
        return out.tex;
}

GLuint GPU::rwkv_lerp(GLuint xn, GLuint sh, GLuint lp, int which, int nrows, Target& out){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_lerp_" + to_string(dim) + "_" + to_string(which) + "_" + to_string(nrows),
                          ::rwkv_lerp(dim, which, nrows));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, xn); set_sampler(p, "XN", 0);
        bind_tex(1, sh); set_sampler(p, "SH", 1);
        bind_tex(2, lp); set_sampler(p, "L", 2);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_l2(GLuint k, GLuint sc, Target& out){
        const int dim = rwkv_dim_, hs = rwkv_hs_, H = rwkv_heads_;
        Target& stat = rwkv_tmp(RT_L2STAT);
        GLuint ps = x_prog("rwkv_l2s_" + to_string(dim) + "_" + to_string(hs), rwkv_l2_stats(dim, hs));
        bind_fbo(stat.fbo); use_program(ps);
        bind_tex(0, k); set_sampler(ps, "K", 0);
        bind_tex(1, sc); set_sampler(ps, "SC", 1);
        rwkv_draw1(ps, H);
        GLuint pn = x_prog("rwkv_l2n_" + to_string(dim) + "_" + to_string(hs), rwkv_l2_norm(dim, hs));
        bind_fbo(out.fbo); use_program(pn);
        bind_tex(0, k); set_sampler(pn, "K", 0);
        bind_tex(1, sc); set_sampler(pn, "SC", 1);
        bind_tex(2, stat.tex); set_sampler(pn, "N", 2);
        rwkv_draw1(pn, dim);
        return out.tex;
}

GLuint GPU::rwkv_keydelta(GLuint k, GLuint a, GLuint sc, Target& out){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_kd_" + to_string(dim), ::rwkv_keydelta(dim));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, k); set_sampler(p, "K", 0);
        bind_tex(1, a); set_sampler(p, "A", 1);
        bind_tex(2, sc); set_sampler(p, "SC", 2);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_tanh(GLuint x, Target& out){
        const int w = out.w;
        GLuint p = x_prog("rwkv_tanh_" + to_string(w), ::rwkv_tanh(w));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        rwkv_draw1(p, w);
        return out.tex;
}

GLuint GPU::rwkv_sigmoid(GLuint x, GLuint bias, Target& out, int oscale){
        const int w = out.w;
        const bool hb = bias != 0;
        GLuint p = x_prog("rwkv_sig_" + to_string(w) + (hb ? "_b" : "") + "_" + to_string(oscale), ::rwkv_sigmoid(w, hb, oscale));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        if (hb) { bind_tex(1, bias); set_sampler(p, "B", 1); }
        rwkv_draw1(p, w);
        return out.tex;
}

GLuint GPU::rwkv_decay(GLuint u, GLuint w0, Target& out){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_decay_" + to_string(dim), ::rwkv_decay(dim));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, u); set_sampler(p, "X", 0);
        bind_tex(1, w0); set_sampler(p, "B", 1);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_vres(GLuint v, GLuint vf, GLuint u, GLuint v0, Target& out){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_vres_" + to_string(dim), ::rwkv_vres(dim));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, v); set_sampler(p, "V", 0);
        bind_tex(1, vf); set_sampler(p, "VF", 1);
        bind_tex(2, u); set_sampler(p, "U", 2);
        bind_tex(3, v0); set_sampler(p, "B", 3);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_sqrelu(GLuint x, Target& out){
        GLuint p = x_prog("rwkv_sqrelu_" + to_string(out.w), ::rwkv_sqrelu(out.w));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, x); set_sampler(p, "X", 0);
        rwkv_draw1(p, out.w);
        return out.tex;
}

GLuint GPU::rwkv_mul(GLuint a, GLuint b, Target& out, int adiv){
        const int dim = rwkv_dim_;
        GLuint p = x_prog("rwkv_mul_" + to_string(dim) + "_" + to_string(adiv), ::rwkv_mul(dim, adiv));
        bind_fbo(out.fbo); use_program(p);
        bind_tex(0, a); set_sampler(p, "A", 0);
        bind_tex(1, b); set_sampler(p, "B", 1);
        rwkv_draw1(p, dim);
        return out.tex;
}

GLuint GPU::rwkv_rk(GLuint k, GLuint r, GLuint sc, GLuint v, GLuint y, Target& out){
        const int dim = rwkv_dim_, hs = rwkv_hs_, H = rwkv_heads_;
        Target& rk = rwkv_tmp(RT_RK);
        GLuint pr = x_prog("rwkv_rkr_" + to_string(dim) + "_" + to_string(hs), rwkv_rk_reduce(dim, hs));
        bind_fbo(rk.fbo); use_program(pr);
        bind_tex(0, k); set_sampler(pr, "K", 0);
        bind_tex(1, r); set_sampler(pr, "R", 1);
        bind_tex(2, sc); set_sampler(pr, "SC", 2);
        rwkv_draw1(pr, H);
        GLuint pa = x_prog("rwkv_rka_" + to_string(dim) + "_" + to_string(hs), rwkv_rk_add(dim, hs));
        bind_fbo(out.fbo); use_program(pa);
        bind_tex(0, y); set_sampler(pa, "Y", 0);
        bind_tex(1, v); set_sampler(pa, "V", 1);
        bind_tex(2, rk.tex); set_sampler(pa, "RK", 2);
        rwkv_draw1(pa, dim);
        return out.tex;
}

GLuint GPU::rwkv_wkv(int layer, GLuint r, GLuint k, GLuint v, GLuint a, GLuint kk,
                     GLuint decay, Target& out){
        const int H = rwkv_heads_, hs = rwkv_hs_, ROWS = H * hs;
        const int cur = rwkv_cur_, nxt = cur ^ 1;
        Target& st_old = rwkv_state_[(size_t)layer * 2 + cur];
        Target& st_new = rwkv_state_[(size_t)layer * 2 + nxt];
        Target& sa = rwkv_tmp(RT_SA);
        const string shape = to_string(H) + "_" + to_string(hs);
        // sa: one fragment per (head, row); the state is bit-cast f32.
        GLuint ps = x_prog("rwkv_wkv_sa_" + shape, ::rwkv_wkv_sa(H, hs));
        bind_fbo(sa.fbo); use_program(ps);
        bind_tex(0, st_old.tex); set_sampler(ps, "ST", 0);
        bind_tex(1, kk); set_sampler(ps, "KK", 1);
        rwkv_draw1(ps, ROWS);
        // update: one fragment per state element, kDrawCap-bounded strips.
        GLuint pu = x_prog("rwkv_wkv_up_" + shape, ::rwkv_wkv_update(H, hs));
        bind_fbo(st_new.fbo); use_program(pu);
        bind_tex(0, st_old.tex); set_sampler(pu, "ST", 0);
        bind_tex(1, sa.tex); set_sampler(pu, "SA", 1);
        bind_tex(2, decay); set_sampler(pu, "DC", 2);
        bind_tex(3, v); set_sampler(pu, "V", 3);
        bind_tex(4, k); set_sampler(pu, "K", 4);
        bind_tex(5, kk); set_sampler(pu, "KK", 5);
        bind_tex(6, a); set_sampler(pu, "A", 6);
        setup_attrib(pu);
        if (needs_finish()) {
            // VC4 drops a draw wider than kDrawCap *fragments*, and the state is
            // one fragment per f32 element (H*hs*hs = 49152 here), so the update
            // is 48 cap-sized strips there -- that is the largest draw count in
            // the whole path and it is VC4-specific.  Every other driver takes it
            // in one piece: the shader derives both indices from gl_FragCoord, so
            // the fragments and their arithmetic are identical either way.
            const int rps = kDrawCap / hs;
            for (int y0 = 0; y0 < ROWS; y0 += rps) {
                glViewport(0, y0, hs, std::min(rps, ROWS - y0));
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            }
        } else {
            glViewport(0, 0, hs, ROWS);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        // out: dot the updated state with r.
        GLuint po = x_prog("rwkv_wkv_out_" + shape, ::rwkv_wkv_out(H, hs));
        bind_fbo(out.fbo); use_program(po);
        bind_tex(0, st_new.tex); set_sampler(po, "ST", 0);
        bind_tex(1, r); set_sampler(po, "R", 1);
        rwkv_draw1(po, ROWS);
        return out.tex;
}

GLuint GPU::rwkv_block(int l, const Model& m, GLuint resid, float* cpu_wkv_state){
        const bool have_gate = m.rwkv.gate && l < (int)m.rwkv_g1.size() && m.rwkv_g1[l].rows > 0;
        const GLuint lp = rwkv_const(l, RC_LERP), sc = rwkv_const(l, RC_SCALE);
        // O1-O3: att_norm, then the six token-shifted inputs; shift := xn.
        GLuint xn = (l == 0) ? rwkv_ln_p(resid, rwkv_const(l, RC_ATTNorm), rwkv_tmp(RT_XN))
                             : rwkv_ln(resid, rwkv_const(l, RC_ATTNorm), rwkv_tmp(RT_XN));
        Target& shA = rwkv_shift_[(size_t)l * 2 + 0];
        GLuint xr = rwkv_lerp(xn, shA.tex, lp, 0, 6, rwkv_tmp(RT_XR));
        GLuint xw = rwkv_lerp(xn, shA.tex, lp, 1, 6, rwkv_tmp(RT_XW));
        GLuint xk = rwkv_lerp(xn, shA.tex, lp, 2, 6, rwkv_tmp(RT_XK));
        GLuint xv = rwkv_lerp(xn, shA.tex, lp, 3, 6, rwkv_tmp(RT_XV));
        GLuint xa = rwkv_lerp(xn, shA.tex, lp, 4, 6, rwkv_tmp(RT_XA));
        GLuint xg = rwkv_lerp(xn, shA.tex, lp, 5, 6, rwkv_tmp(RT_XG));
        rwkv_copy(xn, shA);
        // O4-O6: the three projections.
        GLuint r = rwkv_mv(m.rwkv_r[l], xr, rwkv_tmp(RT_R));
        GLuint k = rwkv_mv(m.rwkv_k[l], xk, rwkv_tmp(RT_K));
        GLuint v = rwkv_mv(m.rwkv_v[l], xv, rwkv_tmp(RT_V));
        RwkvTmp vidx = RT_V;
        // O7: decay branch (tanh on the down projection, then exp/sigmoid).
        GLuint t = rwkv_mv(m.rwkv_w1[l], xw, rwkv_tmp(RT_TW));
        t = rwkv_tanh(t, rwkv_tmp(RT_TW2));
        GLuint u = rwkv_mv(m.rwkv_w2[l], t, rwkv_tmp(RT_U));
        GLuint decay = rwkv_decay(u, rwkv_const(l, RC_W0), rwkv_tmp(RT_DECAY));
        // O8: value residual (layer 0 records v_first).
        if (l == 0) {
            rwkv_copy(v, rwkv_vfirst_);
            rwkv_have_vfirst_ = true;
        } else if (rwkv_have_vfirst_ && l < (int)m.rwkv_v1.size() && m.rwkv_v1[l].rows > 0) {
            t = rwkv_mv(m.rwkv_v1[l], xv, rwkv_tmp(RT_TV));
            u = rwkv_mv(m.rwkv_v2[l], t, rwkv_tmp(RT_U));
            v = rwkv_vres(v, rwkv_vfirst_.tex, u, rwkv_const(l, RC_V0), rwkv_tmp(RT_V2));
            vidx = RT_V2;
        }
        // O9: ICLR gate a = sigmoid(a2(a1(xa)) + a0).
        t = rwkv_mv(m.rwkv_a1[l], xa, rwkv_tmp(RT_TA));
        u = rwkv_mv(m.rwkv_a2[l], t, rwkv_tmp(RT_U));
        GLuint a = rwkv_sigmoid(u, rwkv_const(l, RC_A0), rwkv_tmp(RT_A), 16);
        // O10: optional time-mix gate g = g2(sigmoid(g1(xg))).
        GLuint g = 0;
        if (have_gate) {
            t = rwkv_mv(m.rwkv_g1[l], xg, rwkv_tmp(RT_TG));
            t = rwkv_sigmoid(t, 0, rwkv_tmp(RT_TG2), 1);
            g = rwkv_mv(m.rwkv_g2[l], t, rwkv_tmp(RT_G));
        }
        // O11/O12: kk = l2norm(k*k_k) (*16 store), k' = k + (a-1)*k*k_a.
        GLuint kk = rwkv_l2(k, sc, rwkv_tmp(RT_KK));
        GLuint kd = rwkv_keydelta(k, a, sc, rwkv_tmp(RT_KD));
        // O13/O14: the recurrence.
        GLuint y;
        if (cpu_wkv_state) {
            // Hybrid: the elementwise chain stays on the GPU, but the WKV
            // recurrence runs here in f32 on the host state (the S-1 reference),
            // so the state itself adds no fixed-point error.  The six inputs
            // cross back once per layer; kk is stored *16 and decay (1-w)*256.
            sync();
            auto readv = [&](RwkvTmp idx) {
                Target& t = rwkv_tmp(idx);
                vector<uint8_t> b = read_bytes(t, t.w, 1);
                vector<float> o((size_t)t.w);
                for (int i = 0; i < t.w; ++i) {
                    int u = (int)b[(size_t)i*4+1]*65536+(int)b[(size_t)i*4+2]*256+(int)b[(size_t)i*4+3]-8388608;
                    o[(size_t)i] = u / 256.0f;
                }
                return o;
            };
            vector<float> rr = readv(RT_R), kdv = readv(RT_KD), vv = readv(vidx);
            vector<float> aa = readv(RT_A), kkv = readv(RT_KK), dcv = readv(RT_DECAY);
            const int H = rwkv_heads_, hs = rwkv_hs_, dim = rwkv_dim_;
            for (float& z : kkv) z *= 0.0625f;          // kk *16 -> natural
            for (float& z : aa) z *= 0.0625f;           // a *16 -> natural
            vector<float> wv((size_t)dim);
            for (int i = 0; i < dim; ++i) wv[(size_t)i] = 1.0f - dcv[(size_t)i] * 0.00390625f;
            vector<float> yv((size_t)dim);
            vector<float> sa((size_t)hs);
            float* S = cpu_wkv_state;
            for (int h = 0; h < H; ++h) {
                float* Sh = S + (size_t)h * hs * hs;
                const float* kkh = kkv.data() + (size_t)h * hs;
                const float* ah = aa.data() + (size_t)h * hs;
                const float* kh = kdv.data() + (size_t)h * hs;
                const float* vh = vv.data() + (size_t)h * hs;
                const float* wh = wv.data() + (size_t)h * hs;
                const float* rh = rr.data() + (size_t)h * hs;
                for (int i = 0; i < hs; ++i) {
                    double acc = 0.0; const float* Si = Sh + (size_t)i * hs;
                    for (int j = 0; j < hs; ++j) acc -= (double)Si[j] * (double)kkh[j];
                    sa[(size_t)i] = (float)acc;
                }
                for (int i = 0; i < hs; ++i) {
                    float* Si = Sh + (size_t)i * hs; const float vi = vh[i], s = sa[(size_t)i];
                    for (int j = 0; j < hs; ++j)
                        Si[j] = Si[j] * wh[j] + vi * kh[j] + s * (kkh[j] * ah[j]);
                }
                float* yh = yv.data() + (size_t)h * hs;
                for (int i = 0; i < hs; ++i) {
                    const float* Si = Sh + (size_t)i * hs; double acc = 0.0;
                    for (int j = 0; j < hs; ++j) acc += (double)Si[j] * (double)rh[j];
                    yh[i] = (float)acc;
                }
            }
            for (float& z : yv) z *= 16.0f;             // match the *16 wkv_out store
            vector<uint8_t> px = x_pack(yv, XFormat::I24_1);
            bind_tex(0, rwkv_tmp(RT_Y).tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dim, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            y = rwkv_tmp(RT_Y).tex;
        } else {
            y = rwkv_wkv(l, r, kd, v, a, kk, decay, rwkv_tmp(RT_Y));
        }
        // O15: per-head group norm + ln_x affine.
        y = rwkv_gn(y, rwkv_const(l, RC_LNX), rwkv_tmp(RT_Y2));
        // O16: r_k bonus (uses the delta key).
        y = rwkv_rk(kd, r, sc, v, y, rwkv_tmp(RT_Y));
        // O17: gate.
        if (have_gate) y = rwkv_mul(y, g, rwkv_tmp(RT_Y2), 256);
        // O18/O19: W_o + residual.  Two-step (matvec to I24, then q16 add) like
        // the CPU's read-back + add_q16; the fused matvec_add drops its dot on
        // VC4's I24_1 chain, and the split keeps the add's input fetch ordered.
        if (l == 0) {
            GLuint wo_t = rwkv_mv(m.wo[l], y, rwkv_tmp(RT_U));
            resid = rwkv_first_add(rwkv_x0_.tex, wo_t, rwkv_tmp(RT_RES));
        } else {
            Target& wo_out = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
            if (needs_finish()) {
                // VC4: the fused matvec_add drops its dot on the I24_1 chain;
                // split into matvec + q16 add (the CPU's two-rounding order).
                GLuint wo_t = rwkv_mv(m.wo[l], y, rwkv_tmp(RT_U));
                resid = rwkv_res_add(resid, wo_t, wo_out);
            } else {
                matvec_add(m.wo[l], y, resid, wo_out, act_fmt_);
                resid = wo_out.tex;
            }
        }
        // O20-O22: ffn norm, its token shift and the channel-mix lerp.
        GLuint xn2 = rwkv_ln(resid, rwkv_const(l, RC_ATTNorm2), rwkv_tmp(RT_XN2));
        Target& shF = rwkv_shift_[(size_t)l * 2 + 1];
        GLuint xk2 = rwkv_lerp(xn2, shF.tex, rwkv_const(l, RC_CMLERP), 0, 1, rwkv_tmp(RT_XR));
        rwkv_copy(xn2, shF);
        // O23-O25: key matvec -> relu^2 -> value matvec + residual.  The key is
        // chunked on VC4 (split_rwkv); the two halves recombine at the value add.
        if (m.ffn_chunked) {
            GLuint ha = rwkv_mv(m.w1a[l], xk2, rwkv_tmp(RT_CMH));
            ha = rwkv_sqrelu(ha, rwkv_tmp(RT_CMH2));
            GLuint hb = rwkv_mv(m.w1b[l], xk2, rwkv_tmp(RT_CMH3));
            hb = rwkv_sqrelu(hb, rwkv_tmp(RT_CMH4));
            // mid != resid; the final add reuses resid's (now dead) buffer.
            Target& mid = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
            Target& fin = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES) : rwkv_tmp(RT_RES2);
            if (needs_finish()) {
                GLuint ca = rwkv_mv(m.w2a[l], ha, rwkv_tmp(RT_U));
                GLuint mid_r = rwkv_res_add(resid, ca, mid);
                GLuint cb = rwkv_mv(m.w2b[l], hb, rwkv_tmp(RT_U));
                return rwkv_res_add(mid_r, cb, fin);
            }
            matvec_add(m.w2a[l], ha, resid, mid, act_fmt_);
            matvec_add(m.w2b[l], hb, mid.tex, fin, act_fmt_);
            return fin.tex;
        }
        GLuint h = rwkv_mv(m.w1[l], xk2, rwkv_tmp(RT_CMH));
        h = rwkv_sqrelu(h, rwkv_tmp(RT_CMH2));
        Target& cm_out = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
        if (needs_finish()) {
            GLuint cm_t = rwkv_mv(m.w2[l], h, rwkv_tmp(RT_U));
            return rwkv_res_add(resid, cm_t, cm_out);
        }
        matvec_add(m.w2[l], h, resid, cm_out, act_fmt_);
        return cm_out.tex;
}


