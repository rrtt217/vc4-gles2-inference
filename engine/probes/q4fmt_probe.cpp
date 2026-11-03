// What does the VC4 texture unit actually return for a small UNORM component?
//
// The q4 split layout (Q4_SPLIT) turned nibbles into floats through an RGBA4
// texture and the cheap-decode idea needed the sample to be exact.  This probe
// answers that for both formats, and it is the evidence behind the "q4 decode is
// FLOOR-bound" conclusion in docs/performance.md:
//
//   pass 1  RGBA4, texel v = (v,v,v,v) for v = 0..15:
//             round(c*255) - the 4->8 bit replication (17v if the unit expands)
//             round(c*15)  - the nibble, if the unit normalizes by 15
//             (c*15 - v)*2^20 - the residual, i.e. how far the decode is from
//                            the integer in units of 2^-20
//   pass 2  the same for an RGBA8 texture holding v<<4
//   pass 3  RGBA4 again, with the two decodes a split kernel would use:
//             (c*255 - 17v)*2^20 and (c*255*(1/17) - v)*2^20
//   pass 4  all 256 byte values of an RGBA8 texture: is c exactly b/255, i.e.
//           does floor(c*255+0.5) == b and floor(c*255) == b?
//
// Build (Pi cross, same flags as the engine):
//   make -C engine probes/q4fmt_probe_pi CROSS_SYSROOT=~/raspi-root4
// Run on the Pi: ./q4fmt_probe_pi
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log); printf("SHADER FAIL: %s\n", log); }
    return s;
}

static GLuint program(const char* fs) {
    const char* vs = "attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof(log), nullptr, log); printf("LINK FAIL: %s\n", log); }
    return p;
}

static void params() {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static GLuint tex_rgba4(int w, const std::vector<uint16_t>& d) {
    GLuint t = 0; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA4, w, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, d.data());
    return t;
}

static GLuint tex_rgba8(int w, const std::vector<uint8_t>& d) {
    GLuint t = 0; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
    return t;
}

struct Target { GLuint tex = 0, fbo = 0; };

static Target make_target(int w) {
    Target t;
    glGenTextures(1, &t.tex); glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    params();
    glGenFramebuffers(1, &t.fbo); glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("FBO incomplete %dx1\n", w);
    return t;
}

// texture2D at x/N with N texels; gl_FragCoord.x is the pixel index on VC4.
static const char* FS_TABLE =
    "precision highp float;uniform sampler2D W;uniform float n;"
    "void main(){vec4 c=texture2D(W,vec2((gl_FragCoord.x+0.5)/n,0.5));"
    "float n15=c.r*15.0;float v=floor(n15+0.5);"
    "gl_FragColor=vec4(floor(c.r*255.0+0.5)/255.0,v/255.0,"
    "clamp(128.0+(n15-v)*1048576.0,0.0,255.0)/255.0,1.0);}";

static const char* FS17 =
    "precision highp float;uniform sampler2D W;uniform float n;"
    "void main(){vec4 c=texture2D(W,vec2((gl_FragCoord.x+0.5)/n,0.5));"
    "float v=gl_FragCoord.x;float t=c.r*255.0;float u=t*0.0588235294117647;"
    "gl_FragColor=vec4(clamp(128.0+(t-17.0*v)*1048576.0,0.0,255.0)/255.0,"
    "clamp(128.0+(u-v)*1048576.0,0.0,255.0)/255.0,c.r,u/255.0);}";

static const char* FS_BYTES =
    "precision highp float;uniform sampler2D W;"
    "void main(){vec4 c=texture2D(W,vec2((gl_FragCoord.x+0.5)/256.0,0.5));"
    "float b=gl_FragCoord.x;float t=c.r*255.0;"
    "gl_FragColor=vec4(clamp(128.0+(t-b)*1048576.0,0.0,255.0)/255.0,"
    "floor(t+0.5)/255.0,floor(t)/255.0,1.0);}";

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
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
    printf("GL_RENDERER %s\nGL_VERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

    static GLuint vbo = 0;
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float verts[] = { -1, -1, 3, -1, -1, 3 };
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    Target t16 = make_target(16), t256 = make_target(256);

    auto draw = [&](GLuint prog, GLuint tex, Target& out, int w, float ntex) {
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, w, 1);
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "W"), 0);
        glUniform1f(glGetUniformLocation(prog, "n"), ntex);
        GLuint loc = glGetAttribLocation(prog, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
    };

    std::vector<uint16_t> a4(16);
    std::vector<uint8_t> a8(16 * 4);
    for (int v = 0; v < 16; ++v) {
        a4[v] = (uint16_t)((v << 12) | (v << 8) | (v << 4) | v);
        for (int ch = 0; ch < 4; ++ch) a8[v * 4 + ch] = (uint8_t)(v << 4);
    }
    GLuint t4 = tex_rgba4(16, a4), t8 = tex_rgba8(16, a8);

    uint8_t px[16 * 4];
    GLuint pt = program(FS_TABLE);
    for (int pass = 0; pass < 2; ++pass) {
        draw(pt, pass ? t8 : t4, t16, 16, 16.0f);
        glReadPixels(0, 0, 16, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        printf("%s: v  c*255  c*15  resid(c*15-v, 2^-20)\n", pass ? "RGBA8 (v<<4)" : "RGBA4");
        for (int v = 0; v < 16; ++v) {
            const uint8_t* q = px + v * 4;
            printf("%2d  %3d  %3d  %+4d\n", v, q[0], q[1], (int)q[2] - 128);
        }
    }

    GLuint p17 = program(FS17);
    draw(p17, t4, t16, 16, 16.0f);
    glReadPixels(0, 0, 16, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("RGBA4 decodes: v  resid(c*255-17v)  resid(c*255/17-v)  (2^-20)\n");
    for (int v = 0; v < 16; ++v) {
        const uint8_t* q = px + v * 4;
        printf("%2d  %+4d  %+4d\n", v, (int)q[0] - 128, (int)q[1] - 128);
    }

    std::vector<uint8_t> a256(256 * 4);
    for (int b = 0; b < 256; ++b) memset(&a256[b * 4], b, 4);
    GLuint t256tex = tex_rgba8(256, a256);
    uint8_t p256[256 * 4];
    draw(program(FS_BYTES), t256tex, t256, 256, 256.0f);
    glReadPixels(0, 0, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, p256);
    int lo = 0, hi = 0, bad_half = 0, bad_floor = 0;
    for (int b = 0; b < 256; ++b) {
        const uint8_t* q = p256 + b * 4;
        int r = (int)q[0] - 128;
        if (r < lo) lo = r;
        if (r > hi) hi = r;
        if (q[1] != (uint8_t)b) ++bad_half;
        if (q[2] != (uint8_t)b) ++bad_floor;
    }
    printf("RGBA8 over 256 bytes: c*255-b %+d..%+d (2^-20); floor(t+0.5)!=b: %d/256; floor(t)!=b: %d/256\n",
           lo, hi, bad_half, bad_floor);
    return 0;
}
