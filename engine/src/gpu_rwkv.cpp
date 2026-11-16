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
        // Split into kDrawCap-wide strips: VC4 silently drops a single draw wider
        // than the cap (the sqrelu of a 1536-wide chunked cm hidden left its
        // target at the clear color).  gl_FragCoord.x is absolute, so the shader
        // is unchanged across strips.
        for (int x0 = 0; x0 < w; x0 += kDrawCap) {
            glViewport(x0, 0, std::min(kDrawCap, w - x0), h);
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
        const int rps = kDrawCap / hs;
        for (int y0 = 0; y0 < ROWS; y0 += rps) {
            glViewport(0, y0, hs, std::min(rps, ROWS - y0));
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
        if (env_set("RWKV_FORCE_REBIND")) gl_state_forget_bindings();
        const bool have_gate = m.rwkv.gate && l < (int)m.rwkv_g1.size() && m.rwkv_g1[l].rows > 0;
        const GLuint lp = rwkv_const(l, RC_LERP), sc = rwkv_const(l, RC_SCALE);
        auto D = [&](const char* nm, RwkvTmp idx, float div) {
            if (!env_set("RWKV_STAGE_DUMP") || l > 1) return;
            sync(); Target& t = rwkv_tmp(idx);
            vector<uint8_t> b = read_bytes(t, t.w, 1);
            auto val=[&](int i){int u=(int)b[(size_t)i*4+1]*65536+(int)b[(size_t)i*4+2]*256+(int)b[(size_t)i*4+3]-8388608;return (u/256.0f)/div;};
            float mx=0; for(int i=0;i<t.w;++i) mx=std::max(mx,fabsf(val(i)));
            fprintf(stderr,"OP GPU l%d %s max=%.4f first=%.4f %.4f %.4f\n",l,nm,mx,val(0),val(1),val(2));
        };
        auto DQ = [&](const char* nm, RwkvTmp idx) {
            if (!env_set("RWKV_STAGE_DUMP") || l > 1) return;
            sync(); Target& t = rwkv_tmp(idx);
            vector<uint8_t> b = read_bytes(t, t.w, 1);
            auto val=[&](int i){int u=(b[(size_t)i*4]<<8)|b[(size_t)i*4+1];return (u>=32768?u-65536:u)/256.0f;};
            float mx=0; for(int i=0;i<t.w;++i) mx=std::max(mx,fabsf(val(i)));
            fprintf(stderr,"OPQ GPU l%d %s max=%.4f first=%.4f %.4f %.4f\n",l,nm,mx,val(0),val(1),val(2));
        };
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
        D("xn", RT_XN, 256); D("xr", RT_XR, 1);
        rwkv_copy(xn, shA);
        // O4-O6: the three projections.
        GLuint r = rwkv_mv(m.rwkv_r[l], xr, rwkv_tmp(RT_R));
        GLuint k = rwkv_mv(m.rwkv_k[l], xk, rwkv_tmp(RT_K));
        GLuint v = rwkv_mv(m.rwkv_v[l], xv, rwkv_tmp(RT_V));
        RwkvTmp vidx = RT_V;
        D("r", RT_R, 1); D("k", RT_K, 1); D("v", RT_V, 1);
        // O7: decay branch (tanh on the down projection, then exp/sigmoid).
        GLuint t = rwkv_mv(m.rwkv_w1[l], xw, rwkv_tmp(RT_TW));
        t = rwkv_tanh(t, rwkv_tmp(RT_TW2));
        GLuint u = rwkv_mv(m.rwkv_w2[l], t, rwkv_tmp(RT_U));
        GLuint decay = rwkv_decay(u, rwkv_const(l, RC_W0), rwkv_tmp(RT_DECAY));
        D("onemw", RT_DECAY, 256);
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
        D("vpost", vidx, 1);
        // O9: ICLR gate a = sigmoid(a2(a1(xa)) + a0).
        t = rwkv_mv(m.rwkv_a1[l], xa, rwkv_tmp(RT_TA));
        u = rwkv_mv(m.rwkv_a2[l], t, rwkv_tmp(RT_U));
        GLuint a = rwkv_sigmoid(u, rwkv_const(l, RC_A0), rwkv_tmp(RT_A), 16);
        D("a", RT_A, 16);
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
        D("kk", RT_KK, 16); D("kd", RT_KD, 1);
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
        D("wkv", RT_Y, 16);
        // O15: per-head group norm + ln_x affine.
        y = rwkv_gn(y, rwkv_const(l, RC_LNX), rwkv_tmp(RT_Y2));
        D("gn", RT_Y2, 256);
        // O16: r_k bonus (uses the delta key).
        y = rwkv_rk(kd, r, sc, v, y, rwkv_tmp(RT_Y));
        D("rk", RT_Y, 256);
        // O17: gate.
        if (have_gate) y = rwkv_mul(y, g, rwkv_tmp(RT_Y2), 256);
        D("gate", have_gate ? RT_Y2 : RT_Y, 1);
        // O18/O19: W_o + residual (fused q16 add, matches the CPU grid).
        if (l == 0) {
            GLuint wo_t = rwkv_mv(m.wo[l], y, rwkv_tmp(RT_U));
            D("wo_t", RT_U, 1);
            resid = rwkv_first_add(rwkv_x0_.tex, wo_t, rwkv_tmp(RT_RES));
            DQ("resid_after_wo", RT_RES);
        } else {
            Target& wo_out = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
            GLuint wo_t = rwkv_mv(m.wo[l], y, rwkv_tmp(RT_U));
            resid = rwkv_res_add(resid, wo_t, wo_out);
            if (env_set("RWKV_STAGE_DUMP") && l == 1) {
                rwkv_mv(m.wo[l], y, rwkv_tmp(RT_U));
                sync();
                Target& ut = rwkv_tmp(RT_U);
                vector<uint8_t> b = read_bytes(ut, ut.w, 1);
                auto val=[&](int i){int u=(int)b[(size_t)i*4+1]*65536+(int)b[(size_t)i*4+2]*256+(int)b[(size_t)i*4+3]-8388608;return u/256.0f;};
                float mx=0; for(int i=0;i<ut.w;++i) mx=std::max(mx,fabsf(val(i)));
                fprintf(stderr,"OP GPU l1 wo_pure max=%.4f first=%.4f %.4f %.4f\n",mx,val(0),val(1),val(2));
                vector<uint8_t> rb = read_bytes(wo_out, wo_out.w, 1);
                auto qv=[&](int i){int u=(rb[(size_t)i*4]<<8)|rb[(size_t)i*4+1];return (u>=32768?u-65536:u)/256.0f;};
                float rmx=0; for(int i=0;i<wo_out.w;++i) rmx=std::max(rmx,fabsf(qv(i)));
                fprintf(stderr,"OP GPU l1 resid_after_wo max=%.4f first=%.4f %.4f %.4f\n",rmx,qv(0),qv(1),qv(2));
            }
        }
        // O20-O22: ffn norm, its token shift and the channel-mix lerp.
        GLuint xn2 = rwkv_ln(resid, rwkv_const(l, RC_ATTNorm2), rwkv_tmp(RT_XN2));
        Target& shF = rwkv_shift_[(size_t)l * 2 + 1];
        GLuint xk2 = rwkv_lerp(xn2, shF.tex, rwkv_const(l, RC_CMLERP), 0, 1, rwkv_tmp(RT_XR));
        D("xn2", RT_XN2, 256); D("xk2", RT_XR, 1);
        rwkv_copy(xn2, shF);
        // O23-O25: key matvec -> relu^2 -> value matvec + residual.  The key is
        // chunked on VC4 (split_rwkv); the two halves recombine at the value add.
        if (m.ffn_chunked) {
            GLuint ha = rwkv_mv(m.w1a[l], xk2, rwkv_tmp(RT_CMH));
            D("cmkey_a", RT_CMH, 1);
            ha = rwkv_sqrelu(ha, rwkv_tmp(RT_CMH2));
            D("sqrelu_a", RT_CMH2, 1);
            GLuint hb = rwkv_mv(m.w1b[l], xk2, rwkv_tmp(RT_CMH3));
            D("cmkey_b", RT_CMH3, 1);
            hb = rwkv_sqrelu(hb, rwkv_tmp(RT_CMH4));
            D("sqrelu_b", RT_CMH4, 1);
            // mid != resid; the final add reuses resid's (now dead) buffer.
            Target& mid = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
            Target& fin = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES) : rwkv_tmp(RT_RES2);
            GLuint ca = rwkv_mv(m.w2a[l], ha, rwkv_tmp(RT_U));
            D("cmt_a", RT_U, 1);
            GLuint mid_r = rwkv_res_add(resid, ca, mid);
            GLuint cb = rwkv_mv(m.w2b[l], hb, rwkv_tmp(RT_U));
            D("cmt_b", RT_U, 1);
            GLuint fr = rwkv_res_add(mid_r, cb, fin);
            DQ("resid_final", (fin.tex == rwkv_tmp(RT_RES).tex) ? RT_RES : RT_RES2);
            return fr;
        }
        GLuint h = rwkv_mv(m.w1[l], xk2, rwkv_tmp(RT_CMH));
        h = rwkv_sqrelu(h, rwkv_tmp(RT_CMH2));
        Target& cm_out = (resid == rwkv_tmp(RT_RES).tex) ? rwkv_tmp(RT_RES2) : rwkv_tmp(RT_RES);
        GLuint cm_t = rwkv_mv(m.w2[l], h, rwkv_tmp(RT_U));
        return rwkv_res_add(resid, cm_t, cm_out);
}


