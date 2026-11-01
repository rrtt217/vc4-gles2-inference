// Synthetic matvec probe: how much of the engine's matvec time is int8 decode?
//
// Builds one weight matrix in three encodings and times the same shape:
//   q8   : one byte per weight, 4 per RGBA8 texel (what the engine ships)
//   q8x2 : same weights, but activations packed two-per-texel (engine today)
//   f16  : one half per weight, 4 per RGBA half-float texel (no decode at all)
//
// Usage: ./matvec_probe [rows] [cols] [iters]
// Nothing here is used by the engine; it exists to answer "would I16 weights be
// worth the memory?" before risking a 128 MB model on a 426 MB board.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return chrono::duration<double, milli>(Clock::now() - t).count();
}

// float -> IEEE half bits (armv6 has no _Float16 support in gcc)
static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#ifndef PFNEGLGETPLATFORMDISPLAYEXTPROC
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum, void*, const EGLint*);
#endif

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log); printf("SHADER FAIL: %s\n", log); }
    return s;
}

static GLuint program(const char* fs) {
    const char* vs = "attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof(log), nullptr, log); printf("LINK FAIL: %s\n", log); }
    return p;
}

struct Target { GLuint tex = 0, fbo = 0; int w = 0, h = 0; };

static Target make_target(int w, int h, GLenum filter = GL_NEAREST) {
    Target t; t.w = w; t.h = h;
    glGenTextures(1, &t.tex);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) printf("FBO incomplete: 0x%x (%dx%d)\n", st, w, h);
    return t;
}

// ---------------------------------------------------------------- q8 weights
// rows x cols int8 in [-128,127]; four per RGBA8 texel, one row per texture row.
static GLuint upload_q8(const vector<float>& w, int rows, int cols) {
    const int texw = cols / 4;
    vector<unsigned char> data((size_t)texw * rows * 4);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float v = w[(size_t)r * cols + c] * 127.0f;
            int q = (int)lrintf(v);
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            size_t texel = (size_t)r * texw + c / 4;
            data[texel * 4 + (c % 4)] = (unsigned char)(q + 128);
        }
    GLuint t; glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texw, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    return t;
}

// --------------------------------------------------------------- f16 weights
static GLuint upload_f16(const vector<float>& w, int rows, int cols) {
    const int texw = cols / 4;
    vector<uint16_t> data((size_t)texw * rows * 4);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            data[(size_t)r * texw * 4 + c] = f2h(w[(size_t)r * cols + c]);
    GLuint t; glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texw, rows, 0, GL_RGBA, GL_HALF_FLOAT_OES, data.data());
    return t;
}

static const char* VS = "attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}";

// q8 weights + activations 1 per texel (the pre-packing engine layout)
static GLuint prog_q8(int rows, int cols) {
    static char buf[4096];
    snprintf(buf, sizeof buf,
        "precision highp float;uniform sampler2D W,X;"
        "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(%d),0.5));"
        "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
        "return u>=32768.0?u-65536.0:u;}"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<%d;q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(%d),(float(r)+0.5)/float(%d)));"
        "vec4 w=floor(c*255.0+0.5)-128.0;float k=float(4*q);"
        "s+=w.x*dx(k)+w.y*dx(k+1.0)+w.z*dx(k+2.0)+w.w*dx(k+3.0);}"
        "gl_FragColor=vec4(clamp(s/2048.0,-1.0,1.0),0.0,0.0,1.0);}",
        cols, cols / 4, cols / 4, rows);
    return program(buf);
}

// q8 weights + activations 2 per texel (the engine today)
static GLuint prog_q8x2(int rows, int cols) {
    static char buf[4096];
    const int axw = (cols + 1) / 2;
    snprintf(buf, sizeof buf,
        "precision highp float;uniform sampler2D W,X;"
        "float dA(vec4 c){float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
        "return u>=32768.0?u-65536.0:u;}"
        "float dB(vec4 c){float u=floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5);"
        "return u>=32768.0?u-65536.0:u;}"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<%d;q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(%d),(float(r)+0.5)/float(%d)));"
        "vec4 w=floor(c*255.0+0.5)-128.0;"
        "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(%d),0.5));"
        "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(%d),0.5));"
        "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
        "gl_FragColor=vec4(clamp(s/2048.0,-1.0,1.0),0.0,0.0,1.0);}",
        cols / 4, cols / 4, rows, axw, axw);
    return program(buf);
}

// f16 weights + f16 activations: fetch and multiply, no decode
static GLuint prog_f16(int rows, int cols) {
    static char buf[4096];
    snprintf(buf, sizeof buf,
        "precision highp float;uniform sampler2D W,X;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<%d;q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(%d),(float(r)+0.5)/float(%d)));"
        "vec4 x=texture2D(X,vec2((float(q)+0.5)/float(%d),0.5));"
        "s+=dot(c,x);}"
        "gl_FragColor=vec4(clamp(s/2048.0,-1.0,1.0),0.0,0.0,1.0);}",
        cols / 4, cols / 4, rows, cols / 4);
    return program(buf);
}


// q8 weights + packed activations, without the redundant rounding steps.
// float32(b/255)*255 == b exactly for all 256 byte values, so floor(x+0.5) can
// be dropped; this measures what the ALU saved is worth before changing the engine.
static GLuint prog_fast(int rows, int cols) {
    static char buf[4096];
    const int axw = (cols + 1) / 2;
    snprintf(buf, sizeof buf,
        "precision highp float;uniform sampler2D W,X;"
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;return u>=32768.0?u-65536.0:u;}"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;return u>=32768.0?u-65536.0:u;}"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<%d;q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(%d),(float(r)+0.5)/float(%d)));"
        "vec4 w=c*255.0-128.0;"
        "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(%d),0.5));"
        "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(%d),0.5));"
        "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
        "gl_FragColor=vec4(clamp(s/2048.0,-1.0,1.0),0.0,0.0,1.0);}",
        cols / 4, cols / 4, rows, axw, axw);
    return program(buf);
}

static void draw(GLuint prog, GLuint wt, GLuint xt, Target& out, int rows) {
    // GLES2 forbids client-side vertex arrays: the full-screen triangle must
    // come from a bound VBO or the driver faults.
    static GLuint vbo = 0;
    if (!vbo) {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
    glViewport(0, 0, rows, 1);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, wt);
    glUniform1i(glGetUniformLocation(prog, "W"), 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, xt);
    glUniform1i(glGetUniformLocation(prog, "X"), 1);
    GLuint loc = glGetAttribLocation(prog, "p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: a crash must not eat the log
    const int rows = argc > 1 ? atoi(argv[1]) : 2432;
    const int cols = argc > 2 ? atoi(argv[2]) : 768;
    const int iters = argc > 3 ? atoi(argv[3]) : 10;

    // Headless: the same surfaceless-Mesa platform the engine uses (a pbuffer
    // surface fails here, there is no display server on the Pi).
    auto get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!get_platform_display) { printf("no eglGetPlatformDisplayEXT\n"); return 1; }
    EGLDisplay dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint major = 0, minor = 0;
    if (!eglInitialize(dpy, &major, &minor)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint cfg_attr[] = { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                EGL_ALPHA_SIZE, 8, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE };
    EGLConfig cfg; EGLint n = 0;
    eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n);
    const EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) { printf("context creation failed\n"); return 1; }
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        printf("eglMakeCurrent(surfaceless) failed\n"); return 1;
    }
    printf("GL: %s | %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    GLint maxtex = 0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtex);
    printf("GL_MAX_TEXTURE_SIZE=%d\n", maxtex);
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    const bool have_half = ext && (strstr(ext, "OES_texture_half_float") || strstr(ext, "GL_EXT_texture_half_float"));
    printf("half-float textures: %s\n", have_half ? "yes" : "NO");
    if (getenv("PROBE_EXT")) printf("extensions: %s\n", ext ? ext : "(null)");

    vector<float> w((size_t)rows * cols);
    for (size_t i = 0; i < w.size(); ++i) w[i] = sinf((float)(i % 977) * 0.013f) * 0.5f;

    vector<unsigned char> act8((size_t)cols * 4);
    vector<unsigned char> actx2((size_t)((cols + 1) / 2) * 4);
    vector<uint16_t> actf((size_t)((cols / 4) * 4));
    for (int i = 0; i < cols; ++i) {
        int v = (int)lrintf(sinf((float)i * 0.017f) * 4000.0f) & 0xFFFF;
        act8[(size_t)i * 4 + 0] = (v >> 8) & 255;
        act8[(size_t)i * 4 + 1] = v & 255;
        act8[(size_t)i * 4 + 3] = 255;
        size_t t = (size_t)(i / 2) * 4 + ((i & 1) ? 2 : 0);
        actx2[t + 0] = (v >> 8) & 255;
        actx2[t + 1] = v & 255;
        actf[i] = f2h(sinf((float)i * 0.017f) * 0.9f);
    }

    GLuint wq8 = upload_q8(w, rows, cols);
    printf("q8 upload err=0x%x\n", glGetError());
    GLuint wf16 = 0;
    if (have_half) {
        wf16 = upload_f16(w, rows, cols);
        printf("f16 upload err=0x%x\n", glGetError());
    }

    auto tex = [](int tw, int th, GLenum type, const void* p) {
        GLuint t; glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, type, p);
        return t;
    };
    GLuint x1 = tex(cols, 1, GL_UNSIGNED_BYTE, act8.data());
    GLuint x2 = tex((cols + 1) / 2, 1, GL_UNSIGNED_BYTE, actx2.data());
    GLuint xf = tex(cols / 4, 1, GL_HALF_FLOAT_OES, actf.data());

    // VC4 exposes GL_MAX_TEXTURE_SIZE 2048, so a 2432-row matvec has to be
    // tiled exactly like the engine's logits pass does.
    const int tile = 1024;
    Target out = make_target(tile, 1);

    struct Case { const char* name; GLuint prog, wt, xt; };
    Case cases[] = {
        {"q8  act1/texel", prog_q8(rows, cols),   wq8,  x1},
        {"q8  act2/texel", prog_q8x2(rows, cols), wq8,  x2},
        {"q8  fast decode", prog_fast(rows, cols), wq8, x2},
        {"f16 f16/texel",  prog_f16(rows, cols),  wf16, xf},
    };
    const int nc = have_half ? 4 : 3;

    printf("shape %dx%d, %d iterations, %d rows per draw\n", rows, cols, iters, tile);
    for (int ci = 0; ci < nc; ++ci) {
        Case& c = cases[ci];
        auto matvec = [&]() {
            for (int r0 = 0; r0 < rows; r0 += tile) {
                int n = rows - r0 < tile ? rows - r0 : tile;
                draw(c.prog, c.wt, c.xt, out, n);
            }
        };
        matvec();   // warm up
        glFinish();
        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) matvec();
        glFinish();
        double ms = ms_since(t0) / iters;
        double macs = (double)rows * cols;
        printf("%-16s %8.2f ms/matvec   %6.2f MMAC/s\n", c.name, ms, macs / ms / 1000.0);
        fflush(stdout);
    }
    return 0;
}
