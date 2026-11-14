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
vector<float> GPU::read_logits_full(const vector<Mat>& tiles){
        const int rows = (int)tiles.size();
        vector<float> out;
        out.reserve((size_t)rows * kLogitsGrid);
        if (logits_packed_) {
            const int w = logits_target_.w;
            bind_fbo(logits_target_.fbo);
            vector<uint8_t> b((size_t)w * rows * 4);
            glReadPixels(0, 0, w, rows, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
            for (int i = 0; i < rows; ++i) {
                const int n = tiles[i].rows;
                for (int j = 0; j < n; ++j) {
                    const int t = j / 2, half = j - t * 2;
                    const size_t o = ((size_t)i * w + t) * 4 + (half ? 2 : 0);
                    int v = (int)b[o] * 256 + b[o + 1];
                    if (v >= 32768) v -= 65536;
                    out.push_back((float)v / ACT_SCALE);
                }
            }
            return out;
        }
        vector<int32_t> raw = read_i32(logits_target_, kLogitsGrid, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < tiles[i].rows; ++j) {
                int v = raw[(size_t)i * kLogitsGrid + j];
                if (v >= 32768) v -= 65536;
                out.push_back((float)v / ACT_SCALE);
            }
        }
        return out;
    }



bool GPU::gpu_attn_enabled() const{
        return env_flag("GPU_ATTN");
    }

bool GPU::gpu_layer_enabled() const{
        return env_flag("GPU_LAYER");
    }





// Same upload into the second slot: the prefill pair has two positions in
// flight, and hidden_in_ would be overwritten by the second upload.





vector<uint8_t> GPU::read_bytes(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        return b;
    }

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
void GPU::check_program_runs(GLuint p, const string& key){
        if (getenv("NO_SHADER_SELFTEST")) return;
        if (probe_.tex == 0) probe_ = make_target(1, 1);
        bind_fbo(probe_.fbo);
        glViewport(0, 0, 1, 1);
        glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        use_program(p);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        glFinish();
        vector<uint8_t> b = read_bytes(probe_, 1, 1);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        if (b.size() == 4 && b[0] == 51 && b[1] == 102 && b[2] == 153) {
                cerr << "program '" << key << "' did not draw: the driver skipped the\n"
                        "  draw call, most likely because vc4's register allocator failed\n"
                        "  (run with MESA_DEBUG=1 to see it).  Refusing to continue with a\n"
                        "  kernel that would silently return stale data.\n";
                exit(1);
        }
    }






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

void GPU::bind_tex(int unit, GLuint tex) {
        if (unit < 0 || unit > 7 || g_bound_tex[unit] == tex) return;
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, tex);
        g_bound_tex[unit] = tex;
    }

void GPU::bind_fbo(GLuint fbo) {
        if (g_cur_fbo == fbo) return;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        g_cur_fbo = fbo;
    }

void GPU::use_program(GLuint prog) {
        if (g_cur_prog == prog) return;
        glUseProgram(prog);
        g_cur_prog = prog;
    }

void GPU::set_sampler(GLuint prog, const char* name, int unit) {
        // Usually a sampler keeps one unit for the life of its program, so the
        // uniform is written once; the q4 kernels re-point R from unit 2 to 3, so
        // the cached unit is verified rather than assumed.
        auto& v = sampler_set_[prog];
        for (auto& e : v) {
            if (e.name != name) continue;
            if (e.unit == unit) return;
            e.unit = unit;
            glUniform1i(uloc(prog, name), unit);
            return;
        }
        glUniform1i(uloc(prog, name), unit);
        v.push_back({name, unit});
    }

void GPU::setup_attrib(GLuint prog) {
        // GLES2 has no VAOs, so the single "p" attribute is context state: set it
        // once per (distinct) location instead of three calls per draw.
        GLint loc = aloc(prog, "p");
        if (loc < 0) return;
        if (attrib_ready_ && loc == attrib_p_) return;
        glEnableVertexAttribArray((GLuint)loc);
        glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        attrib_p_ = loc;
        attrib_ready_ = true;
    }

GLint GPU::uloc(GLuint prog, const char* name) {
        // Uniform names are string literals, so the pointer is a stable identity:
        // a per-program linear scan over a handful of entries beats hashing.
        if (prog != uloc_prog_ || !uloc_tab_) {
            uloc_tab_ = &uloc_cache_[prog];
            uloc_prog_ = prog;
        }
        for (const auto& e : *uloc_tab_) if (e.name == name) return e.loc;
        GLint loc = glGetUniformLocation(prog, name);
        // -1 is what a misspelt name and a name the compiler dropped both look
        // like, and glUniform* then silently does nothing: the kernel runs with
        // whatever the uniform held (0), which is a wrong number, not an error.
        if (loc < 0)
                fprintf(stderr, "program '%s': uniform '%s' not found (typo, or the "
                                "compiler removed it)\n", g_prog_key.c_str(), name);
        uloc_tab_->push_back({name, loc});
        return loc;
    }




























// The q4 decode wants ESSL 3.00 for its bitwise nibble split (b>>4 / b&15;
// ESSL 1.00 forbids them).  It is on by default and resolved in two steps: the
// environment says whether it is wanted (Q4_SHIFT=0 turns it off), and
// init_egl records whether the context really offers ESSL 3.00.  A driver that
// cannot gets the float decode back with a warning instead of a fatal error.


void GPU::init_egl(){
        // Q4_SHIFT emits ESSL 3.00 kernels, and Mesa only accepts those when the
        // front end is told the context is ES 3.0.  The overrides change the
        // version the *compiler* is offered, not the driver's capabilities (vc4
        // still has no MRT, no integer textures, and no fround_even), so they
        // are set here, before eglInitialize, and never overwrite a value the
        // caller picked.  Q4_SHIFT_MESA=0 skips this for an A/B.
        if (q4_shift_wanted() && !getenv("Q4_SHIFT_NO_MESA")) {
                setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
                setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
        }
        auto get_platform_display = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        dpy_ = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        EGLint major = 0, minor = 0;
        if (!eglInitialize(dpy_, &major, &minor)) { cerr << "eglInitialize failed\n"; exit(1); }
        eglBindAPI(EGL_OPENGL_ES_API);
        EGLint attrs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        EGLConfig cfg; EGLint n = 0;
        eglChooseConfig(dpy_, attrs, &cfg, 1, &n);
        EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
        EGLContext ctx = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, ctx_attrs);
        if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            cerr << "EGL context creation failed\n"; exit(1);
        }
        ctx_ = ctx;
        const char* r = (const char*)glGetString(GL_RENDERER);
        renderer_ = r ? r : "";
        if (q4_shift_wanted()) {
                const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
                g_q4_shift_ok = glsl && strncmp(glsl, "OpenGL ES GLSL ES 3", 19) == 0;
                if (g_q4_shift_ok) {
                        if (getenv("SHADER_TIME") || getenv("Q4_SHIFT_VERBOSE"))
                                fprintf(stderr, "Q4_SHIFT: integer nibble split on %s (ESSL %s)\n",
                                        renderer_.c_str(), glsl + 18);
                } else
                        fprintf(stderr, "Q4_SHIFT: %s offers %s; using the float q4 decode\n",
                                renderer_.c_str(), glsl ? glsl : "(unknown)");
        }
    }

void GPU::init_gl(){
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
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






void GPU::draw(const Mat& m, GLuint x, Target& t, int row ){
        const bool wr = (m.type == WeightType::INT8 || m.type == WeightType::INT6) &&
                        m.has_row_scale();
        GLuint p = (m.type == WeightType::INT8) ? x_q8(m.rows, m.cols, wr)
                  : (m.type == WeightType::INT4) ? x_i4(m.rows, m.cols)
                  : (m.type == WeightType::INT6) ? x_q6(m.rows, m.cols, wr)
                                                 : x_i16(m.rows, m.cols);
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

vector<int32_t> GPU::read_i32(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        vector<int32_t> out((size_t)w * h);
        for (int i = 0; i < w * h; ++i) {
            int j = i * 4;
            int64_t v = (int64_t)b[j] * 16777216 + (int64_t)b[j + 1] * 65536 +
                        (int64_t)b[j + 2] * 256 + b[j + 3] - 8388608;
            out[i] = (int32_t)v;
        }
        return out;
    }

// --- ATT_DUMP probe -----------------------------------------------------------------
// The attention block is five dependent passes (qrot, kv_rope, score, stats, v).
// This probe prints the exact integers each one consumed and produced for one
// layer and position, so the whole block is checkable offline with a dot
// product and a softmax: no second engine path and no float32 reference.
namespace {


// Signed 16-bit big-endian pair component: comp 0 is rg, comp 1 is ba.
}  // namespace

