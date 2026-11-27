// The projection drivers: a weight matrix times an activation.
//
// matvec_to/add/bias draw one projection, draw() picks the weight kernel for a
// matrix, and matvec/matvec_s are the read-back form the host-side paths use.
// The x_* selectors live in gpu_select.cpp (the key policy in one place);
// logits/top-k in gpu_logits.cpp, the two-position family in gpu_p2.cpp and
// attention in gpu_attn.cpp.  The GLSL is in the shader TUs.
#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "switches.h"
#include "xformat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void GPU::matvec_to(const Mat& m, GLuint x, Target& out){
        matvec_to_row(m, x, out, 0);
    }

void GPU::matvec_to_row(const Mat& m, GLuint x, Target& out, int row){
        if (out.tex == 0 || out.w != m.rows || out.h <= row) out = make_target(m.rows, row + 1);
        draw(m, x, out, row);
        layer_finish("NF_MAT");
        ffn_sync();
    }

vector<float> GPU::matvec_from(const Mat& m, GLuint x){
        Target& t = target_for(m.rows);
        draw(m, x, t);
        return read_proj(t, m.rows, 1);
    }

void GPU::matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out, XFormat xf){
        matvec_add_bias_rf(m, x, residual, 0, out, xf, res_fmt_);
}

void GPU::matvec_add_rf(const Mat& m, GLuint x, GLuint residual, Target& out,
                        XFormat xf, XFormat rf, int out_row){
        matvec_add_bias_rf(m, x, residual, 0, out, xf, rf, out_row);
}

void GPU::matvec_add_rf_xrow(const Mat& m, GLuint x, int xrow, GLuint residual,
                             Target& out, XFormat xf, XFormat rf){
        if (m.type != WeightType::INT8 || xf != XFormat::I16_2) {
                matvec_add_rf(m, x, residual, out, xf, rf);
                return;
        }
        const bool wr = wf(m.type).wrow && m.has_row_scale();
        // Mirror matvec_add_bias_rf's choice *and* its lazy target sizing: the
        // packed residual uses the pair kernel, whose store is the I16_2 layout
        // the next pass reads, and the target is (rows+1)/2 texels wide.
        const bool pair = res_pack_;
        const int ow = pair ? (m.rows + 1) / 2 : m.rows;
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        GLuint p = pair ? x_q8_add_pair_xrow(m.rows, m.cols, xf, wr)
                        : x_q8_add_xrow(m.rows, m.cols, xf, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, x); set_sampler(p, "X", 1);
        bind_tex(2, residual); set_sampler(p, "R", 2);
        glUniform1f(uloc(p, "ws"), m.scale);
        // The pair form emits `xr`; the non-pair xrow form does too.
        // The context plane is att_seq_ tall; the row is the source position.
        glUniform1f(uloc(p, "xr"), ((float)xrow + 0.5f) / (float)att_seq_);
        if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        setup_attrib(p);
        const int strip = std::min(kDrawCap, m.rows);
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
                const int w = std::min(strip, m.rows - x0);
                glViewport(x0, 0, w, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_MAT");
}

// `bias` folds the following `bias_add` into this draw, for the kernels that
// carry the arm (the q8 row-pair one).  Zero is the shipped form, unchanged.
void GPU::matvec_add_bias_rf(const Mat& m, GLuint x, GLuint residual, GLuint bias,
                             Target& out, XFormat xf, XFormat rf, int out_row){
        if (!wf_accepts(m.type, xf)) wf_no_activation("matvec_add", m.type, xf);
        const bool pair = res_pack_ && (m.type == WeightType::INT8 ||
                                        m.type == WeightType::T3);
        const bool pair4 = res_pack_ && m.type == WeightType::INT4;
        const int ow = (pair || pair4) ? (m.rows + 1) / 2 : m.rows;
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        // The packed residual layout is a property of the model, not of the
        // weight format: a row-scale q8 matrix must still write the packed
        // texel the rmsnorm reads, so the paired kernel grows a wrow variant
        // rather than the writer switching layouts.
        const bool wr = wf(m.type).wrow && m.has_row_scale();
        // A t3 matrix must write the layout `res_pack_` says the reader expects:
        // the packed pair when the residual is two values a texel (VC4's I16_2),
        // the single value otherwise.
        // `xf` is the *activation* format (the X texture); the residual is
        // always res_fmt_.  Passing xf here decodes R with the wrong layout --
        // the q8 selectors read res_fmt_ themselves, which is why they never had
        // this bug.  Measured: every odd row gained exactly 255/256 and the reply
        // collapsed to one token.
        // Exhaustive, no default: see weight_format.h.  The old chain ended in
        // x_i16_add, so an unhandled format added to WeightType would have run
        // the i16 decoder on its bytes.
        GLuint p = 0;
        switch (m.type) {
            case WeightType::INT8:
                p = res_pack_ ? x_q8_add_pair(m.rows, m.cols, xf, rf, wr, bias != 0)
                              : x_q8_add(m.rows, m.cols, xf, rf, wr);
                break;
            case WeightType::INT16:
                p = x_i16_add(m.rows, m.cols, xf, rf);
                break;
            case WeightType::INT4:
                p = res_pack_ ? x_i4_add_pair(m.rows, m.cols, xf, rf, m.fmt4)
                              : x_i4_add(m.rows, m.cols, xf, rf, m.fmt4);
                break;
            case WeightType::INT6:
                wf_no_kernel("matvec_add", m.type);
            case WeightType::T3:
                p = res_pack_ ? x_t3_add_pair(m, xf, rf, wr)
                              : x_t3_add(m, xf, rf, wr);
                break;
        }
        if (!p) wf_no_kernel("matvec_add", m.type);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex);
        set_sampler(p, "W", 0);
        bind_tex(1, x);
        set_sampler(p, "X", 1);
        if (m.type == WeightType::INT4) {
            // P takes unit 2 on the q4 kernels, so R moves to 3.
            bind_tex(2, m.q4_param_tex);
            set_sampler(p, "P", 2);
            bind_tex(3, residual);
            set_sampler(p, "R", 3);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        } else {
            bind_tex(2, residual);
            set_sampler(p, "R", 2);
        }
        // The q4 kernels take their scale as "ps"; "ws" does not exist there.
        if (m.type != WeightType::INT4) glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        // The bias arm's sampler: unit 4 is free here (W 0, X 1, R 2, WS 3), and
        // a kernel without the arm has no `B` uniform, so the bind only happens
        // for a program that declares one.
        if (bias) { bind_tex(4, bias); set_sampler(p, "B", 4); }
        setup_attrib(p);
        // Same rule as draw(): a single draw never exceeds 1024 fragments.  A
        // full-target-width draw is what gets dropped (docs/engine-internals.md
        // section 2), and a 1536-row matrix is exactly that for its own target.
        // The old special case split 1536 into 768+768 and left every other
        // width in one pass; the cap covers all of them, and `ow` is the
        // fragment count (rows/2 for the paired kernels), not the row count.
        // Same experimental override as draw(): MATVEC_STRIP caps the fragment
        // count per draw so a suspected full-width draw can be A/B-ed.
        const char* msv = getenv("MATVEC_STRIP");
        const int cap = (msv && atoi(msv) > 0) ? atoi(msv) : kDrawCap;
        const int strip = min(cap, ow);
        for (int x0 = 0; x0 < ow; x0 += strip) {
            int w = min(strip, ow - x0);
            glViewport(x0, out_row, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_MAT");
    }


vector<float> GPU::matvec(const Mat& m, const vector<float>& x){
        // RWKV_PROFILE only: the three GL steps timed apart, so the serial
        // path's read-back can be compared with its draw and its upload.  The
        // cached switch leaves every other model on the same instructions.
        const bool prof = rwkv_prof_on();
        const auto tp0 = Clock::now();
        Target& t = target_for(m.rows);
        GLuint xt = upload_activations(x);
        if (prof) {
                const auto now = Clock::now();
                rwkv_prof.upload_ms += elapsed_ms(tp0, now);
        }
        const auto td0 = Clock::now();
        draw(m, xt, t);
        if (needs_finish()) { glFlush(); glFinish(); }
        if (prof) {
                const auto now = Clock::now();
                rwkv_prof.draw_ms += elapsed_ms(td0, now);
        }
        const auto tr0 = Clock::now();
        vector<float> out = read_proj(t, m.rows, 1);
        if (prof) {
                const auto now = Clock::now();
                rwkv_prof.read_ms += elapsed_ms(tr0, now);
        }
        return out;
    }

// The same matvec, with the activation scaled by a power of two that puts its
// largest component near 64 on the q16 grid, and the result scaled back.
//
// The grid is absolute (step 1/256, rail 128), so a matvec input of order 1e-3
// -- LFM2.5's conv gate output at the start of a sequence, or its attention
// output -- lands on a handful of levels or on zero, and the layer contributes
// nothing.  Scaling by 2^k costs one multiply on the host, is exact (a power of
// two), and the *result* keeps its own resolution because the readback is q16
// at scale 1 after the divide.  This is the per-tensor scale the activation
// format never had, applied where the engine crosses the grid.
vector<float> GPU::matvec_s(const Mat& m, const vector<float>& x){
        float mx = 0.0f;
        for (float v : x) mx = std::max(mx, fabsf(v));
        if (!(mx > 0.0f)) return matvec(m, x);
        int e = 0;
        frexpf(mx, &e);                 // mx = f * 2^e, f in [0.5, 1)
        const int k = 6 - e;            // target 32..64
        const float s = ldexpf(1.0f, k);
        const float inv = ldexpf(1.0f, -k);
        vector<float> xs(x.size());
        for (size_t i = 0; i < x.size(); ++i) xs[i] = x[i] * s;
        vector<float> out = matvec(m, xs);
        for (float& v : out) v *= inv;
        return out;
    }


void GPU::draw(const Mat& m, GLuint x, Target& t, int row ){
        if (!wf_accepts(m.type, act_fmt_)) wf_no_activation("draw", m.type, act_fmt_);
        const bool wr = wf(m.type).wrow && m.has_row_scale();
        // Exhaustive on purpose, with no default: -Wswitch names a WeightType
        // added without a kernel here, and the guard below stops the run if one
        // is ever compiled past the warning.  A trailing `: x_i16` used to do
        // this silently (weight_format.h).
        GLuint p = 0;
        switch (m.type) {
            case WeightType::INT8:  p = x_q8(m.rows, m.cols, wr); break;
            case WeightType::INT16: p = x_i16(m.rows, m.cols); break;
            case WeightType::INT4:  p = x_i4(m.rows, m.cols, m.fmt4); break;
            case WeightType::INT6:  p = x_q6(m.rows, m.cols, wr); break;
            case WeightType::T3:    p = x_t3(m, wr); break;
        }
        if (!p) wf_no_kernel("draw", m.type);
        bind_fbo(t.fbo);
        // DRAW_CLEAR=1 fills the target with a known pattern before the draw, so
        // "the draw did not execute" and "the draw read stale input" are
        // distinguishable in a readback.  0x33/0x66/0x99/0xff is not a valid q16
        // pair for any small value.  Cached: this sits on the per-draw path.
        static const bool draw_clear = env_set("DRAW_CLEAR");
        if (draw_clear) { glClearColor(0.2f, 0.4f, 0.6f, 1.0f); glClear(GL_COLOR_BUFFER_BIT); }
        use_program(p);
        bind_tex(0, m.tex);
        set_sampler(p, "W", 0);
        bind_tex(1, x);
        set_sampler(p, "X", 1);
        if (m.type == WeightType::INT6) {
            bind_tex(2, m.tex_hi);
            set_sampler(p, "W2", 2);
        }
        if (m.type == WeightType::INT4) {
            bind_tex(2, m.q4_param_tex);
            set_sampler(p, "P", 2);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        } else {
            glUniform1f(uloc(p, "ws"), m.scale);
            if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        }
        setup_attrib(p);
        // A single draw must stay below GL_MAX_TEXTURE_SIZE: at exactly 2048
        // fragments wide vc4 silently produces wrong bytes (2040 is fine,
        // 2048 differs in 5029/8192 bytes and finishes 3x too fast), while the
        // same matrix drawn as two viewport-offset halves is exact.  Splitting
        // costs one draw (~10 us), so cap every strip at 1024 and keep the
        // gl_FragCoord-absolute convention the shader relies on.
        //
        // That cap is what protects the 1536-wide qkv today: a single
        // 1536-wide draw is byte-exact in a readback probe but leaves the qkv
        // target unwritten when the next pass samples it as a texture -
        // different garbage every run, PPL 271.2855 -> 71669 on the Pi
        // (docs/engine-internals.md section 2).  matvec_add() uses the same cap.
        // MATVEC_STRIP overrides the cap for experiments.
        const char* sv = getenv("MATVEC_STRIP");
        const int env_strip = sv ? atoi(sv) : 0;
        const int cap = (env_strip > 0) ? env_strip : kDrawCap;
        const int strip = std::min(cap, m.rows);
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
            int w = min(strip, m.rows - x0);
            glViewport(x0, row, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
    }
