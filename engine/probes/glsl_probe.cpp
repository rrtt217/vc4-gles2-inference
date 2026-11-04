// Which shader language does the vc4 ES context really give us, and can the q4
// nibble unpack use integer ops?  Built for WORKLOG item 0, which proposed
// int(roundEven(c*255)) -> FTOIN plus b>>4 / b&15.  Answer on Mesa 26.2.2:
// bitwise ops need ESSL 3.00 (the GLES override does provide it), but roundEven
// reaches the backend as fround_even and ABORTS the process, and the ESSL 1.00
// integer form (b/16, b-h*16) is exact but costs more ops than the float decode.
// Each linked variant decodes all 256 byte values on four channels: bad=N/1024.
// See docs/performance.md.
// modes: es2 (client 2), es3 (client 2 + MESA_GLES_VERSION_OVERRIDE=3.0),
//        es3ctx (client 3).  The overrides must be set before eglInitialize.
// build: make -C engine probes/glsl_probe_pi CROSS_SYSROOT=/home/ubuntu/raspi-root4
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNGPD)(EGLenum, void*, const EGLint*);
using std::string;

static GLuint comp(GLenum t, const string& s) {
    GLuint h = glCreateShader(t);
    const char* p = s.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(h, sizeof(log), nullptr, log);
        printf("    compile error: %s\n", log);
    }
    return h;
}

static GLuint prog(const string& vs, const string& fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, comp(GL_VERTEX_SHADER, vs));
    glAttachShader(p, comp(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        printf("    link error: %s\n", log);
        return 0;
    }
    return p;
}

static const char* D_FLOOR = "float dec(float c){return floor(c*255.0+0.5);}";
// ESSL 1.00 integer form: int() truncates, +0.5 rounds, /16 and -h*16 are int.
static const char* D_INT = "float dec(float c){int b=int(c*255.0+0.5);int h=b/16;int l=b-h*16;return float(h*16+l);}";
// Truncation without +0.5: expected off by one on some multiples (Q4_BFAST).
static const char* D_TRUNC = "float dec(float c){int b=int(c*255.0);int h=b>>4;int l=b&15;return float(h*16+l);}";
static const char* D_BITS = "float dec(float c){int b=int(c*255.0+0.5);int h=b>>4;int l=b&15;return float(h*16+l);}";
static const char* D_REVEN = "float dec(float c){int b=int(roundEven(c*255.0));int h=b>>4;int l=b&15;return float(h*16+l);}";

static string fs_src(const char* dec, bool es3, bool legacy_names) {
    string s = es3 ? "#version 300 es\n" : "";
    s += "precision highp float;\n";
    s += es3 ? "out vec4 o;\n" : "";
    s += "uniform highp sampler2D W;uniform float n;\n";
    s += dec;
    s += legacy_names ? "void main(){vec4 c=texture2D(W,vec2((gl_FragCoord.x+0.5)/n,0.5));"
                      : "void main(){vec4 c=texture(W,vec2((gl_FragCoord.x+0.5)/n,0.5));";
    s += "vec4 v=vec4(dec(c.r),dec(c.g),dec(c.b),dec(c.a))/255.0;";
    s += legacy_names ? "gl_FragColor=v;}" : "o=v;}";
    return s;
}

struct Variant {
    const char* name;
    const char* vs;
    const char* dec;
    bool es3;
    bool legacy_names;
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const string mode = argc > 1 ? argv[1] : "es2";
    const int client = mode == "es3ctx" ? 3 : 2;
    PFNGPD gpd = (PFNGPD)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!gpd) { printf("no eglGetPlatformDisplayEXT\n"); return 1; }
    EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint maj = 0, min = 0;
    if (!eglInitialize(d, &maj, &min)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ca[] = { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                          EGL_ALPHA_SIZE, 8, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                          EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE };
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(d, ca, &cfg, 1, &n) || n < 1) { printf("eglChooseConfig n=%d\n", n); return 1; }
    const EGLint xa[] = { EGL_CONTEXT_CLIENT_VERSION, client, EGL_NONE };
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, xa);
    if (!ctx) { printf("context failed (client %d)\n", client); return 1; }
    if (!eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) { printf("makeCurrent failed\n"); return 1; }
    printf("%s: renderer %s\n", mode.c_str(), (const char*)glGetString(GL_RENDERER));
    printf("%s: version  %s\n", mode.c_str(), (const char*)glGetString(GL_VERSION));
    printf("%s: glsl     %s\n", mode.c_str(), (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    static GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float verts[] = { -1, -1, 3, -1, -1, 3 };
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // 256 RGBA8 texels; every channel holds a different byte value.
    std::vector<uint8_t> src(256 * 4);
    for (int i = 0; i < 256; ++i) {
        src[i * 4 + 0] = (uint8_t)i;
        src[i * 4 + 1] = (uint8_t)(255 - i);
        src[i * 4 + 2] = (uint8_t)((i * 7 + 3) & 255);
        src[i * 4 + 3] = (uint8_t)((i * 13 + 5) & 255);
    }
    GLuint wtex = 0;
    glGenTextures(1, &wtex);
    glBindTexture(GL_TEXTURE_2D, wtex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, src.data());

    GLuint otex = 0, fbo = 0;
    glGenTextures(1, &otex);
    glBindTexture(GL_TEXTURE_2D, otex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, otex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { printf("FBO incomplete\n"); return 1; }

    static const char* VS0 = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    static const char* VS3 = "#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    const Variant tab[] = {
        {"floor_100", VS0, D_FLOOR, false, true},
        {"int_100", VS0, D_INT, false, true},
        {"bits_100", VS0, D_BITS, false, true},
        {"trunc_100", VS0, D_TRUNC, false, true},
        {"ver300", VS3, D_BITS, true, false},
        {"trunc300", VS3, D_TRUNC, true, false},
        {"ver300_legacy", VS3, D_BITS, true, true},
        {"reven300", VS3, D_REVEN, true, false},
    };
    std::vector<uint8_t> got(256 * 4);
    for (const Variant& v : tab) {
        printf("VAR %-14s ", v.name);
        GLuint p = prog(v.vs, fs_src(v.dec, v.es3, v.legacy_names));
        if (!p) { printf("LINK_FAIL\n"); continue; }
        glViewport(0, 0, 256, 1);
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "W"), 0);
        glUniform1f(glGetUniformLocation(p, "n"), 256.0f);
        GLint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, wtex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        glReadPixels(0, 0, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, got.data());
        int bad = 0, first = -1;
        for (int i = 0; i < 1024; ++i)
            if (got[i] != src[i]) { ++bad; if (first < 0) first = i; }
        printf("OK bad=%d/1024", bad);
        if (bad) printf(" first texel %d ch %d got %d want %d", first / 4, first % 4, got[first], src[first]);
        printf("\n");
    }
    return 0;
}
