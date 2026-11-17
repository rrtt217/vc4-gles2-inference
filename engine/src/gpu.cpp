// The GL object's own state: the constructor, the capability queries and the
// little probes that read the model's layout back.
//
// This file used to be 2674 lines.  The translation units around it took the work
// that belongs to a layer -- `gpu_gl.cpp` (EGL/GL, draw, binds, read-back),
// `gpu_state.cpp` (textures/targets and the release ledger), `gpu_prog.cpp` (the
// program cache), `gpu_attn.cpp`, `gpu_conv.cpp`, `gpu_ffn.cpp`, `gpu_matvec.cpp`,
// `gpu_upload.cpp` and `gpu_diag.cpp` -- and what is left here is the object
// itself.  `gpu_internal.h` is the seam: the bind cache, the draw cap, the shared
// program-layer symbols and the two measurement hooks.
//
// The invariants did not move with the code: a program key is not a program text
// (`gpu_prog.cpp`), everything the GPU owns is released and reset
// (`gpu_state.cpp`, gate/check_target_reset.sh) and a draw never covers more
// than `kDrawCap` fragments (`gpu_internal.h`).

#include "gpu.h"
#include "gpu_internal.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Startup upload accounting (SHADER_TIME=1 / UPLOAD_TIME=1): the weight upload is
// the last large, controllable part of a cold start on this board.  The counters
// live with the factories in gpu_state.cpp; these are the accessors the CLI prints.
int gpu_upload_textures(){ return g_up_tex; }
double gpu_upload_texture_ms(){ return g_up_tex_ms; }

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

bool GPU::gpu_attn_enabled() const{
        return env_flag("GPU_ATTN");
    }

bool GPU::gpu_layer_enabled() const{
        return env_flag("GPU_LAYER");
    }

void GPU::sync(){ note_sync(); glFinish(); }

void GPU::forget_state(){ gl_state_forget_bindings(); }

// Decode the qkv target to per-row q16 integers.  The decode is over the integers,
// not the texels, so the one-row (24-bit) and the paired (packed q16) layouts
// decode alike: a QKV_PAIR=0 run and a QKV_PAIR=1 run of the same model must
// produce the same vector for every layer and position.  QKV_DUMP uses it.
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
        c.kv_fmt = att_kv_fmt_;
        c.score_scale = att_score_scale_; c.sum_scale = att_sum_scale_;
        c.sum_log = att_sum_log_;
        c.exp_poly = exp_poly_enabled();   // EXP=drv|poly, one axis (shaders.cpp)
        c.qkv_pair = att_qkv_pair_;
        return c;
    }
