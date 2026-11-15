// The GL context and the draw/read-back primitives: everything that talks to
// EGL/GLES2 directly, plus the bind cache the rest of the GPU code goes through.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  `init_egl`
// carries the two rules that were paid for on VC4: the surfaceless context and the
// MESA_GLES_VERSION_OVERRIDE dance for the q4 kernels (docs/engine-internals.md).

#include "gpu.h"
#include "gpu_internal.h"
#include "logits_grid.h"
#include "xformat.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

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

vector<uint8_t> GPU::read_bytes(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        return b;
    }

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
        gpu_dmabuf_init(dpy_);
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

void GPU::draw(const Mat& m, GLuint x, Target& t, int row ){
        const bool wr = (m.type == WeightType::INT8 || m.type == WeightType::INT6) &&
                        m.has_row_scale();
        GLuint p = (m.type == WeightType::INT8) ? x_q8(m.rows, m.cols, wr)
                  : (m.type == WeightType::INT4) ? x_i4(m.rows, m.cols, m.fmt4)
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
