#include "gpu.h"
#include "gpu_internal.h"
#include "logits_grid.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <string>
#include <unordered_map>
#include <vector>

// Defined next to the bind cache helpers further down; declared here because the
// debug dump and the prefix cache invalidate the cache from earlier in the file.
// Startup upload accounting (SHADER_TIME=1 / UPLOAD_TIME=1): the weight
// upload is the last large, controllable part of a cold start on this board.
// Scene ledgers: everything this context created that belongs to a model or a
// session rather than to a shader program.  File-static for the same reason the
// bind cache is - one GL context per process - and they are what lets a model
// be replaced without rebuilding the context (docs/performance.md: 1.89 s of
// program links and 4.23 s of EGL context on the board, per process).
int gpu_upload_textures(){ return g_up_tex; }
double gpu_upload_texture_ms(){ return g_up_tex_ms; }

// Default-off probe (ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v or all):
// replay one attention pass per layer.  Every shader involved is deterministic
// and writes the same FBO from unchanged inputs, so the second draw reproduces
// the same texels and the output stays byte-identical.  The end-to-end delta is
// the true marginal cost of that pass, unlike the LAYER_PROFILE laps, whose glFinish
// also retires the draw issued before the lap started.


// Q4_INLINE selects the INT4 activation-decode form per kernel group, so the
// generic and the inline build can be A/B-ed from one binary (the program
// cache key carries the choice).  Unset = all inline; Q4_INLINE=0 = all
// generic; otherwise a comma list of gate,w2,qkv enables only those.

GPU::GPU(){
        init_egl();
        init_gl();
        GLint max_tex = 0, max_rb = 0, max_vp[2] = {0, 0};
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
        glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_rb);
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_vp);
        max_target_dim_ = max_tex;
        if (max_rb > 0 && max_rb < max_target_dim_) max_target_dim_ = max_rb;
        if (max_vp[0] > 0 && max_vp[0] < max_target_dim_) max_target_dim_ = max_vp[0];
        if (max_vp[1] > 0 && max_vp[1] < max_target_dim_) max_target_dim_ = max_vp[1];
        if (max_target_dim_ <= 0) max_target_dim_ = 2048;
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







// The paired kernel is q8-only; a q4 checkpoint keeps q4 vocabulary tiles and
// therefore the unpacked path (byte-identical to before this change).

// Logits pass.  The paired path renders two vocabulary rows per fragment and
// packs both q16 logits into one texel, so the target is half as wide.

// Read the logits target back as floats, unpacking the texel pairs when the
// paired pass ran.



bool GPU::gpu_attn_enabled() const{
        return env_flag("GPU_ATTN");
    }

bool GPU::gpu_layer_enabled() const{
        return env_flag("GPU_LAYER");
    }





// Same upload into the second slot: the prefill pair has two positions in
// flight, and hidden_in_ would be overwritten by the second upload.






void GPU::sync(){ glFinish(); }

void GPU::forget_state(){ gl_state_forget_bindings(); }
// One row of an activation target, in the exact bytes a GL read-back returns,
// plus a hash of the whole row.  The GPU probe (dump_target) and the CPU probe
// (dump_cpu_row) both print through here, so a GPU_LAYER=0 dump and a
// GPU_LAYER=1 dump of the same model are diffable byte for byte.
// The hash covers the row, not the printed prefix: n may be shorter than w.


// Decode the qkv target to per-row q16 integers.  The decode is over the
// integers, not the texels, so the one-row (24-bit) and the paired (packed q16)
// layouts decode alike: a QKV_PAIR=0 run and a QKV_PAIR=1 run of the same model
// must produce the same vector for every layer and position.  QKV_DUMP uses it.
vector<int> GPU::qkv_values(int rows){
        vector<int> out(rows, 0);
        if (qkv_target_.tex == 0) return out;
        const int tw = att_qkv_pair_ ? (rows + 1) / 2 : rows;
        vector<uint8_t> b((size_t)tw * 4);
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, qkv_target_.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        gl_state_forget_bindings();
        for (int i = 0; i < rows; ++i) {
            if (att_qkv_pair_) {
                const int o = (i / 2) * 4 + ((i & 1) ? 2 : 0);
                int v = (int)b[o] * 256 + (int)b[o + 1];
                if (v >= 32768) v -= 65536;
                out[i] = v;
            } else {
                const int o = i * 4;
                const int64_t u = (int64_t)b[o] * 16777216 + (int64_t)b[o + 1] * 65536 +
                                  (int64_t)b[o + 2] * 256 + b[o + 3] - 8388608;
                out[i] = (int)u;
            }
        }
        return out;
    }

unsigned long long GPU::qkv_hash(int rows){
        const vector<int> v = qkv_values(rows);
        unsigned long long h = 1469598103934665603ull;
        for (int x : v) {
            h ^= (unsigned long long)(uint32_t)x;
            h *= 1099511628211ull;
        }
        return h;
    }

bool GPU::rms_fused() const{
        // RMS_FUSED forces it on, RMS_SPLIT forces the old two-pass form.
        if (getenv("RMS_FUSED")) return true;
        if (getenv("RMS_SPLIT")) return false;
        return needs_finish();   // VC4 / VideoCore
    }

bool GPU::needs_finish() const{
        if (getenv("FORCE_GLFINISH")) return true;
        return renderer_.find("VC4") != string::npos || renderer_.find("VideoCore") != string::npos;
    }

AttnConfig GPU::att_cfg() const{
        AttnConfig c;
        c.dim = att_dim_; c.kv_dim = att_kv_dim_; c.heads = att_heads_;
        c.kv_heads = att_kv_heads_; c.head_dim = att_head_dim_; c.seq = att_seq_;
        c.rope_half = att_rope_half_; c.qk_norm = att_qk_norm_;
        c.kv_bits = att_kv_bits_; c.kv_scale = att_kv_scale_; c.kv_width = att_kv_width_;
        c.kv_vstep = att_kv_vstep_;
        c.qkv_pair = att_qkv_pair_;
        return c;
    }

// Default-off probe (SHADER_TIME=1): time every fragment-program link and
// report when it happened relative to process start.  First-token latency is
// a compile-count question first: x_prog also merges keys that build
// byte-identical program text, so a repeat compile cannot hide here.
// every later number is wrong with no diagnostic.  Draw each new program once
// into a sentinel-filled 1x1 target and stop if nothing was written.






// Gate pair matvec for two rows per fragment plus swiglu, straight into a q16
// pair texture that the w2 matvec can read: one draw instead of two.



// Position A then position B in one texture, one row wide: the 2-position
// kernels read A at texel [0,axw) and B at [axw,2*axw).  Keyed by length like
// upload_input, but kept in its own map: a single-position upload of the same
// length would otherwise overwrite one half.

// Two-position FFN gate: one draw for both positions (see matvec_q8_gate2).

// Two-position w2: reads the P2 gate output, writes one P2 q16 texel per row.







// Prefill step 7d: two per-position FFN residuals -> one P2 texture in two
// draws (sum of squares for both positions, then one normalising pass).


// Two-position w2 whose residual is the previous chunk P2 output: one texel
// carries both positions value r, so the add reads one texel, and the
// expression order (add inside the kernel, before the q16 encode) is the
// serial one that step 7c had to preserve.

// Two-position w2 including the residual add: one draw per FFN chunk for both
// positions, reading each position residual stream (step 7c).

// One position of a P2 row into its residual stream (which = 0 A, 1 B).

// q4 gate chunk: row pair per fragment, w1/w3 dots and both swiglu results in
// one draw (see matvec_i4_pair_swiglu).




// GL binding state is context-wide (GLES2 has no VAOs and the engine owns the
// context), so these live at file scope: the static texture helpers below bind
// on their own and can forget the cache without an instance.


































// The q4 decode wants ESSL 3.00 for its bitwise nibble split (b>>4 / b&15;
// ESSL 1.00 forbids them).  It is on by default and resolved in two steps: the
// environment says whether it is wanted (Q4_SHIFT=0 turns it off), and
// init_egl records whether the context really offers ESSL 3.00.  A driver that
// cannot gets the float decode back with a warning instead of a fatal error.










    void GPU::layer_finish(const char* skip_env){
        bool skip = skip_env && getenv(skip_env);
        if (skip_env && layer_fast_finish_ && !getenv("LAYER_SAFE_FINISH")) {
            string s(skip_env);
            if (s == "NF_RMS" || s == "NF_MAT" || s == "NF_ATT") skip = true;
        }
        if (skip) return;
        if (needs_finish()) { glFlush(); glFinish(); }
    }


    void GPU::kv_rope_write(int layer, int pos){
        GLuint p = x_kv_rope();
        bind_fbo(kv_cache_[layer].fbo);
        glViewport(0, pos, att_kv_width_, 1);
        use_program(p);
        bind_tex(0, qkv_target_.tex);
        set_sampler(p, "QKV", 0);
        bind_tex(1, rope_tex_);
        set_sampler(p, "RT", 1);
        if (att_qk_norm_) {
            bind_tex(2, k_norm_tex_[layer]);
            set_sampler(p, "KN", 2);
        }
        if (att_kv_vstep_) {
            bind_tex(3, vs_targets_[layer].tex);
            set_sampler(p, "VS", 3);
        }
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        att_finish();
    }








// --- ATT_DUMP probe -----------------------------------------------------------------
// The attention block is five dependent passes (qrot, kv_rope, score, stats, v).
// This probe prints the exact integers each one consumed and produced for one
// layer and position, so the whole block is checkable offline with a dot
// product and a softmax: no second engine path and no float32 reference.
namespace {


// Signed 16-bit big-endian pair component: comp 0 is rg, comp 1 is ba.
}  // namespace

