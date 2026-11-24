// The GL context and the read-back primitives: everything that talks to
// EGL/GLES2 directly, plus the bind cache the rest of the GPU code goes through.
// The projection draw is in gpu_matvec.cpp; logits read-back in gpu_logits.cpp.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  `init_egl`
// carries the two rules that were paid for on VC4: the surfaceless context and the
// MESA_GLES_VERSION_OVERRIDE dance for the q4 kernels (docs/engine-internals.md).

#include "gpu.h"
#include "switches.h"
#include "gpu_internal.h"
#include "xformat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x00000040
#endif
#include <GLES2/gl2.h>

vector<uint8_t> GPU::read_bytes(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        note_sync();
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

// Bind a texture for a *client upload*.  `bind_tex()` skips the glActiveTexture
// when the unit already holds that texture, which is right for a draw and wrong
// here: glTexSubImage2D writes into the texture bound on the *active* unit, and
// the active unit is whatever the last bind left.  A caller that uploaded the
// same texture twice in a row (the decoder's residual, then the cross-attention
// context) therefore wrote the second upload into the previous unit's texture
// and kept reading the old bytes.
void GPU::bind_upload_tex(GLuint tex) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        g_bound_tex[0] = tex;
}

GLuint GPU::bound_tex(int u) { return (u >= 0 && u < 8) ? g_bound_tex[u] : 0; }

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
        // The cache key is the name itself, not its address: several callers
        // pass the .c_str() of a temporary ("V" + to_string(k)), and a pointer
        // comparison then either misses every time or, worse, matches a stale
        // entry that a freed string's address was reused for.
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
        // Q4_SHIFT's nibble split and RWKV_GPU's bit-cast f32 state kernels both
        // need the ESSL 3.00 compiler, so either one requests the MESA override.
        const char* rf = sw_enum("RES");
        const char* af = sw_enum("ACT");
        const char* hf = sw_enum("FFN_HID");
        const char* rs = sw_enum("RMS_STORE");
        // The two bit-cast formats need the same compiler as Q4_SHIFT's nibble
        // split: f32 streams (uintBitsToFloat) and bf16 streams (which rebuild
        // a float from its top 16 bits the same way).  bf16 was missing from
        // this test -- it worked only because Q4_SHIFT is on by default, so
        // Q4_SHIFT=0 turned every bf16 axis into a compile error.
        const auto es3_fmt = [](const char* v) {
                return v && (!strcmp(v, "f32") || !strcmp(v, "bf16"));
        };
        const bool bitcast_axes = es3_fmt(rf) || es3_fmt(af) || es3_fmt(hf) || es3_fmt(rs);
        if ((q4_shift_wanted() || env_flag("RWKV_GPU") || bitcast_axes) && !getenv("Q4_SHIFT_NO_MESA")) {
                setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
                setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
        }
        auto get_platform_display = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        dpy_ = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        EGLint major = 0, minor = 0;
        if (!eglInitialize(dpy_, &major, &minor)) { cerr << "eglInitialize failed\n"; exit(1); }
        eglBindAPI(EGL_OPENGL_ES_API);
        // Ask for an ES 3.0 context first, and fall back to 2.0.
        //
        // The ESSL 3.00 kernels (the bit-cast f32 streams, the q4 nibble split,
        // bf16 when it lands) need an ES 3.00 compiler, and *requesting* one is
        // the only portable way to get it: the Mesa version overrides above are
        // a Mesa-only prerequisite (vc4 does not advertise ES 3.0 on its own),
        // and on a driver that is not Mesa there is no such variable to lean on.
        // An ES 3.0 context still accepts ESSL 1.00 shaders -- backward
        // compatibility is in the specification -- so every kernel that does not
        // ask for ES3 keeps its text and its numerics (measured: the context
        // version alone costs nothing; docs/performance.md 2026-11-18).
        EGLint attrs3[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        EGLint attrs2[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        EGLConfig cfg = nullptr; EGLint n = 0;
        const EGLint want3[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        const EGLint want2[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
        EGLContext ctx = EGL_NO_CONTEXT;
        if (eglChooseConfig(dpy_, attrs3, &cfg, 1, &n) && n > 0)
                ctx = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, want3);
        if (ctx == EGL_NO_CONTEXT) {
                cfg = nullptr; n = 0;
                eglChooseConfig(dpy_, attrs2, &cfg, 1, &n);
                ctx = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, want2);
        }
        if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            cerr << "EGL context creation failed\n"; exit(1);
        }
        ctx_ = ctx;
        gpu_dmabuf_init(dpy_);
        const char* r = (const char*)glGetString(GL_RENDERER);
        renderer_ = r ? r : "";
        // A software rasteriser is not a machine the pins apply to: the golden
        // strings and PPL are per GPU, and llvmpipe means the sandbox lost
        // /dev/dri (2026-11-15).  Say it once, loudly, so a green gate cannot be
        // read as a hardware run.  scripts/regression.sh greps this line.
        if (renderer_.find("llvmpipe") != string::npos ||
            renderer_.find("softpipe") != string::npos ||
            renderer_.find("swrast") != string::npos)
            fprintf(stderr, "renderer: %s -- software rasteriser: timings and machine pins "
                            "do not apply\n", renderer_.c_str());
        {
                const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
                g_es3_ok = glsl && strncmp(glsl, "OpenGL ES GLSL ES 3", 19) == 0;
                g_q4_shift_ok = g_es3_ok && q4_shift_wanted();
                if (q4_shift_wanted()) {
                        if (g_q4_shift_ok) {
                                if (getenv("SHADER_TIME") || getenv("Q4_SHIFT_VERBOSE"))
                                        fprintf(stderr, "Q4_SHIFT: integer nibble split on %s (ESSL %s)\n",
                                                renderer_.c_str(), glsl + 18);
                        } else
                                fprintf(stderr, "Q4_SHIFT: %s offers %s; using the float q4 decode\n",
                                        renderer_.c_str(), glsl ? glsl : "(unknown)");
                }
                if (env_flag("RWKV_GPU") && !g_es3_ok)
                        fprintf(stderr, "RWKV_GPU: %s offers no ESSL 3.00; the WKV state falls back to CPU f32\n",
                                renderer_.c_str());
        }
    }

void GPU::init_gl(){
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    }


vector<int32_t> GPU::read_i32_rows(Target& t, int w, int y, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        note_sync();
        glReadPixels(0, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        vector<int32_t> out((size_t)w * h);
        for (int i = 0; i < w * h; ++i) {
                const int j = i * 4;
                int64_t v = (int64_t)b[j] * 16777216 + (int64_t)b[j + 1] * 65536 +
                            (int64_t)b[j + 2] * 256 + b[j + 3] - 8388608;
                out[(size_t)i] = (int32_t)v;
        }
        return out;
}
vector<int32_t> GPU::read_i32(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        note_sync();
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

// One row-sized full-target draw of an already-bound program (viewport w x h).
void GPU::draw_row(GLuint prog, int w, int h){
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
