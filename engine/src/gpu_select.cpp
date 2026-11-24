// The program selectors: builder text -> a cached program.
//
// An x_* is a key plus x_prog().  They live together because a key is not a
// program text: causal, rope_none, qbatch, the activation/residual formats and
// the row-scale flag all change the emitted text, so each has to be in the key
// (x_score_batch records a real collision from getting this wrong).  The
// drivers that call them are split by mechanism.
#include "gpu.h"
#include "gpu_internal.h"
#include "glsl_io.h"
#include "shaders.h"
#include "xformat.h"

#include <string>

using std::string;
using std::to_string;

GLuint GPU::x_q8(int rows, int cols, bool wrow){
        return x_prog("xq8_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_t3(const Mat& m, bool wrow){
        // The tile shape is part of the program key: a 4096-row body folds into
        // two tiles of 2048 and the same (rows, cols) drawn as one tile is a
        // different shader.
        // A Mat that has not been through the uploader has no tile shape yet;
        // one tile is the fallback and is what every fitting body uses.
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : m.rows;
        const int tiles = m.t3_tile_h > 0 ? m.t3_tiles() : 1;
        if (getenv("T3_GEOM"))
                fprintf(stderr, "T3GEOM mv rows=%d cols=%d tile_h=%d th=%d tiles=%d\n",
                        m.rows, m.cols, m.t3_tile_h, th, tiles);
        return x_prog("xt3_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(act_fmt_) + "_" + to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3(m.rows, m.cols, act_fmt_, wrow, th, tiles));
    }

GLuint GPU::x_t3_add(const Mat& m, XFormat xf, XFormat rf, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : m.rows;
        const int tiles = m.t3_tile_h > 0 ? m.t3_tiles() : 1;
        // `xf` is the format of the X texture, which is NOT always act_fmt_: a
        // w2 add reads the FFN hidden stream (FFN_HID), and the two axes differ
        // as soon as ACT=i8 leaves the hidden on q16.  The q8 selectors have
        // always taken xf for this reason; the t3 ones hardcoded act_fmt_ and
        // only worked while the two formats coincided (found 2026-11-21 with
        // ACT=i8 on the t3 body: layer 0's FFN was decoded as I8_4).
        return x_prog("xt3a_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(xf) + "_" + x_format_name(rf) + "_" +
                      to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_add(m.rows, m.cols, xf, rf, wrow, th, tiles));
    }

GLuint GPU::x_q6(int rows, int cols, bool wrow){
        return x_prog("xq6_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(act_fmt_) + (wrow ? "_wr" : ""),
                      matvec_i6(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_q8_pair(int rows, int cols, bool wrow){
        return x_prog("xq8p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_q8_pair_swiglu(int rows, int cols, bool wrow, int hs){
        // Inline I16_2 decode is the default; FFN_SW_INLINE=0 restores the
        // generic X4Q form in the same binary, so both can be A/B-ed
        // interleaved on one binary.
        const char* v = getenv("FFN_SW_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        return x_prog("xq8ps_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(act_fmt_) + (inl ? "" : "_gen") + (wrow ? "_wr" : "") +
                      (hs ? "_h" + to_string(hs) : ""),
                      matvec_q8_pair_swiglu(rows, cols, act_fmt_, inl, wrow, hs));
    }

GLuint GPU::x_q8_gate2(int rows, int cols){
        return x_prog("xq8g2_" + to_string(rows) + "_" + to_string(cols),
                      matvec_q8_gate2(rows, cols));
    }

GLuint GPU::x_q8_p2(int rows, int cols, bool wrow){
        return x_prog("xq8p2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      matvec_q8_p2(rows, cols, wrow));
    }

GLuint GPU::x_rms_stats2(){
        return x_prog("xrms2s_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_),
                      ::rms_stats2(att_dim_, res_fmt_));
    }

GLuint GPU::x_rms_norm_pair2(bool use_eps_uniform){
        return x_prog("xrms2n_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) +
                          (use_eps_uniform ? "_e" : ""),
                      ::rms_norm_pair2(att_dim_, res_fmt_, use_eps_uniform));
    }

GLuint GPU::x_q8_gate2_p2(int rows, int cols, bool wrow){
        return x_prog("xq8g2p2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      ::matvec_q8_gate2_p2(rows, cols, wrow));
    }

GLuint GPU::x_q8_p2_add_p2(int rows, int cols, bool wrow){
        return x_prog("xq8p2a2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      ::matvec_q8_p2_add_p2(rows, cols, wrow));
    }

GLuint GPU::x_q8_p2_add(int rows, int cols, bool wrow){
        return x_prog("xq8p2a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(res_fmt_) + (wrow ? "_wr" : ""),
                      ::matvec_q8_p2_add(rows, cols, res_fmt_, wrow));
    }

GLuint GPU::x_p2_split(int rows){
        return x_prog("xp2sp_" + to_string(rows) + "_" + x_format_name(res_fmt_),
                      ::p2_split(rows, res_fmt_));
    }

GLuint GPU::x_q8_pair_fused(int rows, int cols, bool wrow){
        return x_prog("xq8pf_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair_fused(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_i4_pair(int rows, int cols){
        const string key = "xi4p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key;
        // This site builds a program without x_prog's text hash, so the two
        // runtime GLSL switches have to be asked here too (glsl_io.h).
        const string text = glsl_text(key, matvec_i4_pair(rows, cols, act_fmt_));
        GLuint p = link_program(text);
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_i4_pair_swiglu(int rows, int cols, int hs, bool fmt4){
        const bool inl = q4_inline("gate");
        const string tail = string(inl ? "" : "_gen") + (hs ? "_h" + to_string(hs) : "");
        if (fmt4)
                return x_prog("xi4ps4_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + tail,
                              matvec_i4_pair_swiglu_4x4(rows, cols, act_fmt_, inl, hs));
        return x_prog("xi4ps_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + tail,
                      matvec_i4_pair_swiglu(rows, cols, act_fmt_, inl, hs));
    }

GLuint GPU::x_t3_pair(const Mat& m, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : (m.rows + 1) / 2;
        const int tiles = m.t3_tiles();
        return x_prog("xt3p_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(act_fmt_) + "_" + to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_pair(m.rows, m.cols, act_fmt_, wrow, th, tiles));
    }

GLuint GPU::x_t3_add_pair(const Mat& m, XFormat xf, XFormat rf, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : (m.rows + 1) / 2;
        const int tiles = m.t3_tiles();
        return x_prog("xt3ap_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(xf) + "_" + x_format_name(rf) + "_" +
                      to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_add_pair(m.rows, m.cols, xf, rf, wrow, th, tiles));
    }

GLuint GPU::x_q8_add(int rows, int cols, XFormat xf, bool wrow){
        return x_prog("xq8a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" +
                      x_format_name(res_fmt_) + (wrow ? "_wr" : ""),
                      matvec_q8_add(rows, cols, xf, res_fmt_, wrow));
    }

GLuint GPU::x_q8_add_pair(int rows, int cols, XFormat xf, bool wrow, bool bias){
        // The inline I16_2 decode is the default; Q8AP_INLINE=0 restores the
        // generic X4Q form in the same binary, so the two can be A/B-ed
        // interleaved without a second build.
        const char* v = getenv("Q8AP_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        // The residual's format belongs in the key as well as in the text: a
        // bf16 residual takes the generic path even when the fast I16_2 one is
        // available, and the two are different programs.
        return x_prog("xq8ap_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(xf) + (inl ? "" : "_gen") + (wrow ? "_wr" : "") +
                      (bias ? "_b" : "") +
                      (res_fmt_ == XFormat::I16_2 ? std::string() : std::string("_") + x_format_name(res_fmt_)),
                      matvec_q8_add_pair(rows, cols, xf, res_fmt_, inl, wrow, bias));
    }

GLuint GPU::x_i4_add_pair(int rows, int cols, XFormat xf, bool fmt4){
        const bool inl = q4_inline("w2");
        // The residual's format is part of the key: this kernel writes two rows
        // into one texel, so a bf16 residual and a q16 one are different texts.
        const string key = string(fmt4 ? "xi4ap4_" : "xi4ap_") + to_string(rows) + "_" +
                           to_string(cols) + "_" + x_format_name(xf) + "_" +
                           x_format_name(res_fmt_) + (inl ? "" : "_gen");
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key;
        const string text = glsl_text(key,
                                     fmt4 ? matvec_i4_add_pair_4x4(rows, cols, xf, res_fmt_, inl)
                                          : matvec_i4_add_pair(rows, cols, xf, res_fmt_, inl));
        GLuint p = link_program(text);
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_i16(int rows, int cols){
        return x_prog("xi16_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_i16(rows, cols, act_fmt_));
    }

GLuint GPU::x_i16_add(int rows, int cols, XFormat xf){
        return x_prog("xi16a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      matvec_i16_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_i4(int rows, int cols, bool fmt4){
        const bool inl = q4_inline("qkv");
        return x_prog(string(fmt4 ? "xi44_" : "xi4_") + to_string(rows) + "_" + to_string(cols) + "_" +
                          x_format_name(act_fmt_) + (inl ? "" : "_gen"),
                      fmt4 ? matvec_i4_4x4(rows, cols, act_fmt_, inl)
                           : matvec_i4(rows, cols, act_fmt_, inl));
    }

GLuint GPU::x_i4_add(int rows, int cols, XFormat xf, bool fmt4){
        return x_prog(string(fmt4 ? "xi4a4_" : "xi4a_") + to_string(rows) + "_" + to_string(cols) + "_" +
                          x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      fmt4 ? matvec_i4_add_4x4(rows, cols, xf, res_fmt_)
                           : matvec_i4_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_swiglu_pair(int hidden, int hs){
        return x_prog("xswp_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      swiglu_pair(hidden, ffn_fmt_, hs));
    }

GLuint GPU::x_swiglu_2row(int hidden, int hs){
        return x_prog("xsw2_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      ::swiglu_2row(hidden, ffn_fmt_, hs));
    }

GLuint GPU::x_swiglu(int hidden, bool fused, int hs){
        return x_prog(string("xsw_") + (fused ? "f_" : "s_") + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      swiglu(hidden, fused, ffn_fmt_, hs));
    }

GLuint GPU::x_rms_split(){
        return x_prog("xrmss_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_),
                      rms_stats_split(att_dim_, res_fmt_));
    }

GLuint GPU::x_rms_reduce(){ return x_prog("xrmsr", rms_stats_reduce()); }

GLuint GPU::x_rms_norm(bool use_eps_uniform){
        const bool fused = rms_fused() && act_fmt_ != XFormat::I8_4;
        return x_prog("xrmsn_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) + "_" +
                          x_format_name(act_fmt_) + (fused ? "_f" : "") +
                          (use_eps_uniform ? "_e" : ""),
                      rms_norm(att_dim_, res_fmt_, act_fmt_, fused, use_eps_uniform));
    }

// Long context: one program a segment.  The key carries `segk` because the
// program text does (`SEGK`), and the program cache validates the text hash on
// every hit -- a shared key with two texts is the swap bug this cache already
// paid for once (docs/swap-drift.md).  segk == 0 keeps the shipped key, so the
// digest table and the cache behaviour do not move for a one-segment session.
static string seg_suffix(int k) { return k > 0 ? "_s" + to_string(k) : string(); }

GLuint GPU::x_v(int segk, bool accumulate){
        return x_prog("xv_" + to_string(att_dim_) + "_" + x_format_name(act_fmt_) +
                          seg_suffix(segk) + (accumulate ? "_add" : ""),
                      ::att_v(att_cfg(segk, accumulate), act_fmt_));
    }

GLuint GPU::x_v_fold(int n, bool accumulate){
        return x_prog("xvf_" + to_string(att_dim_) + "_" + x_format_name(act_fmt_) +
                          "_" + to_string(n) + (accumulate ? "_add" : ""),
                      att_v_fold_shader(att_cfg(), act_fmt_, n, accumulate));
    }

GLuint GPU::x_kv_rope(int segk){ return x_prog("xkv" + seg_suffix(segk), att_kv_rope(att_cfg(segk))); }

GLuint GPU::x_vscale(){ return x_prog("xvscale", ::att_vscale(att_cfg())); }

GLuint GPU::x_qnorm(){ return x_prog("xqn", att_qnorm_scale(att_cfg())); }

GLuint GPU::x_qrot(int segk){ return x_prog("xqr" + seg_suffix(segk), att_qrot(att_cfg(segk))); }

GLuint GPU::x_score(int segk){ return x_prog("xsc" + seg_suffix(segk), att_score(att_cfg(segk))); }

GLuint GPU::x_stats(int segk){ return x_prog("xst" + seg_suffix(segk), ::att_stats(att_cfg(segk))); }

// The qbatch variants bake n into the shader text (QB) and take the position's
// row offset qrow0 as a uniform; the serial getters above stay qbatch = 1.
// A key is not a program: `causal` and `rope_none` both change the emitted
// text, so two modes that share a key get the first one's program for ever.  The
// encoder runs `xscb_<n>` bidirectionally and the decode loop asks for the same
// n, causal -- and got the encoder's text ("uniform 'pos' not found", because
// the bidirectional build prunes it).  The dump's labels are the registry's, not
// these, so the name is free to grow.
GLuint GPU::x_score_batch(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xscb_" + to_string(n) + (c.causal ? "_c" : "_bi") +
                          (c.rope_none ? "_nr" : ""), att_score(c));
}

GLuint GPU::x_stats_row(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xstr_" + to_string(n) + (c.causal ? "_c" : "_bi") +
                          (c.rope_none ? "_nr" : ""), ::att_stats(c));
}

GLuint GPU::x_v_row(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xvr_" + to_string(n) + "_" + x_format_name(act_fmt_) +
                          (c.causal ? "_c" : "_bi") + (c.rope_none ? "_nr" : ""),
                      ::att_v(c, act_fmt_));
}

GLuint GPU::x_prob(){ return x_prog("xpr", att_prob(att_cfg())); }

GLuint GPU::x_top2_fast(int rows, int chunk){
        return x_prog("xt2f_" + to_string(rows) + "_" + to_string(chunk), top2_fast(rows, chunk));
    }

GLuint GPU::x_top2(int rows, int chunk){
        return x_prog("xt2_" + to_string(rows) + "_" + to_string(chunk), top2(rows, chunk));
    }

GLuint GPU::x_q8_pair_logits(int rows, int cols, bool wrow){
        return x_prog("xq8pl_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair_logits(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_top2_fast_packed(int rows, int chunk){
        return x_prog("xt2fp_" + to_string(rows) + "_" + to_string(chunk), top2_fast_packed(rows, chunk));
    }

GLuint GPU::x_top2_packed(int rows, int chunk){
        return x_prog("xt2p_" + to_string(rows) + "_" + to_string(chunk), top2_packed(rows, chunk));
    }

GLuint GPU::x_top1_group(int n){
        return x_prog("xt1g_" + to_string(n), top1_group(n));
    }

GLuint GPU::x_top1_final(int groups){
        return x_prog("xt1f_" + to_string(groups), top1_final(groups));
    }

GLuint GPU::x_q8_batch(int rows, int cols, bool wrow){
        return x_prog("xq8b_" + std::to_string(rows) + "_" + std::to_string(cols) + "_" +
                          x_format_name(act_fmt_) + (wrow ? "_wr" : ""),
                      matvec_q8_batch(rows, cols, act_fmt_, wrow));
}
