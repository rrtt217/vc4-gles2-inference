// Can a linked program be serialized on vc4, and does reloading it skip the
// compile?  Item 6 (first-token latency) needs to know this before anyone
// builds a program cache: Mesa exposes GL_OES_get_program_binary on the ES2
// profile, but the vc4 backend compiles QPU code at draw time, so the answer
// is not obvious from the extension list.
//
// The probe links a matvec-shaped shader, draws once (forcing the vc4
// compile), then asks for the program binary, reloads those bytes into a
// second program and draws again.  A working cache shows up as: status OK,
// a non-zero length, a non-zero format and identical pixels.  Timing the two
// first draws says whether the reload actually skipped the backend compile.
//
// Build: make -C engine probes/progbinary_probe_pi CROSS_SYSROOT=~/raspi-root4
// Run on the Pi: ./progbinary_probe_pi
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#ifndef PFNEGLGETPLATFORMDISPLAYEXTPROC
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum, void*, const EGLint*);
#endif
#ifndef GL_PROGRAM_BINARY_LENGTH_OES
#define GL_PROGRAM_BINARY_LENGTH_OES 0x8741
#endif

typedef void (*PFNGETPROGRAMBINARYOES)(GLuint, GLsizei, GLsizei*, GLenum*, void*);
typedef void (*PFNPROGRAMBINARYOES)(GLuint, GLenum, const void*, GLint);

static double now_ms(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char* kFS =
    "precision highp float;\n"
    "uniform sampler2D W;\n"
    "uniform sampler2D X;\n"
    "void main(){\n"
    "  vec2 uv = gl_FragCoord.xy;\n"
    "  float acc = 0.0;\n"
    "  for (int i = 0; i < 32; ++i) {\n"
    "    vec4 w = texture2D(W, vec2((uv.x + float(i) * 4.0) / 256.0, uv.y / 4.0));\n"
    "    vec4 x = texture2D(X, vec2((float(i) * 4.0) / 256.0, 0.5));\n"
    "    acc += dot(w, x);\n"
    "  }\n"
    "  gl_FragColor = vec4(acc);\n"
    "}\n";

static GLuint compile(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log); printf("SHADER FAIL: %s\n", log); }
    return s;
}

static GLuint link_prog(){
    const char* vs = "attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, kFS));
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof(log), nullptr, log); printf("LINK FAIL: %s\n", log); }
    return p;
}

static GLuint make_tex(int w, int h, int seed){
    std::vector<unsigned char> d((size_t)w * h * 4);
    for (size_t i = 0; i < d.size(); ++i) d[i] = (unsigned char)((i * 37 + seed * 11) & 255);
    GLuint t = 0; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static void draw(GLuint prog, GLuint w, GLuint x){
    static const float verts[] = { -1, -1, 3, -1, -1, 3 };
    GLuint vbo = 0; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glUseProgram(prog);
    GLint loc = glGetAttribLocation(prog, "p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, w);
    glUniform1i(glGetUniformLocation(prog, "W"), 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, x);
    glUniform1i(glGetUniformLocation(prog, "X"), 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glDeleteBuffers(1, &vbo);
}

int main(){
    auto get_platform_display = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint major = 0, minor = 0;
    if (!eglInitialize(dpy, &major, &minor)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint attrs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                       EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE };
    EGLConfig cfg; EGLint n = 0;
    eglChooseConfig(dpy, attrs, &cfg, 1, &n);
    EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) { printf("context failed\n"); return 1; }
    printf("renderer: %s\n", (const char*)glGetString(GL_RENDERER));
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    printf("OES_get_program_binary advertised: %s\n", (ext && strstr(ext, "GL_OES_get_program_binary")) ? "yes" : "no");

    auto getbin = (PFNGETPROGRAMBINARYOES)eglGetProcAddress("glGetProgramBinaryOES");
    auto setbin = (PFNPROGRAMBINARYOES)eglGetProcAddress("glProgramBinaryOES");
    printf("proc addresses: get=%p set=%p\n", (void*)getbin, (void*)setbin);
    if (!getbin || !setbin) { printf("RESULT: no entry points, program binary caching unavailable\n"); return 0; }

    GLuint texw = make_tex(256, 4, 1), texx = make_tex(256, 1, 2);
    GLuint fbo = 0, rbo = 0;
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(1, &rbo); glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, 64, 4);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    glViewport(0, 0, 64, 4);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { printf("fbo incomplete\n"); return 1; }

    double t0 = now_ms();
    GLuint a = link_prog();
    double t1 = now_ms();
    draw(a, texw, texx);
    double t2 = now_ms();
    printf("A: link %.1f ms, first draw %.1f ms\n", t1 - t0, t2 - t1);
    std::vector<unsigned char> pa(64 * 4 * 4);
    glReadPixels(0, 0, 64, 4, GL_RGBA, GL_UNSIGNED_BYTE, pa.data());

    GLint len = 0; GLenum fmt = 0;
    glGetProgramiv(a, GL_PROGRAM_BINARY_LENGTH_OES, &len);
    printf("GL_PROGRAM_BINARY_LENGTH_OES = %d\n", len);
    std::vector<unsigned char> bin(len > 0 ? len : 1);
    GLsizei got = 0;
    getbin(a, (GLsizei)bin.size(), &got, &fmt, bin.data());
    GLenum err = glGetError();
    printf("getProgramBinary: err=0x%04x format=0x%04x length=%d\n", err, fmt, got);
    if (err != GL_NO_ERROR || got <= 0) { printf("RESULT: vc4 cannot export a program binary\n"); return 0; }

    double t3 = now_ms();
    GLuint b = glCreateProgram();
    setbin(b, fmt, bin.data(), got);
    GLint ok = 0; glGetProgramiv(b, GL_LINK_STATUS, &ok);
    double t4 = now_ms();
    printf("programBinary: link status %s, %.1f ms\n", ok ? "OK" : "FAIL", t4 - t3);
    if (!ok) { printf("RESULT: binary reload rejected\n"); return 0; }
    draw(b, texw, texx);
    double t5 = now_ms();
    std::vector<unsigned char> pb(64 * 4 * 4);
    glReadPixels(0, 0, 64, 4, GL_RGBA, GL_UNSIGNED_BYTE, pb.data());
    printf("B: first draw %.1f ms, pixels %s\n", t5 - t4, memcmp(pa.data(), pb.data(), pa.size()) == 0 ? "identical" : "DIFFER");
    printf("RESULT: program binary round trip works; reload+first draw %.1f ms vs link+first draw %.1f ms\n", (t4 - t3) + (t5 - t4), t2 - t0);
    return 0;
}
