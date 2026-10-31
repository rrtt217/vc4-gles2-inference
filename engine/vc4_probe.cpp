// Minimal VC4 GLES2 FBO/texture probe. No model code.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace std;

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
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
struct Target { GLuint tex = 0, fbo = 0; int w = 0, h = 0; };
static Target make_target(int w, int h, const vector<unsigned char>& data = {}) {
    Target t; t.w = w; t.h = h;
    glGenTextures(1, &t.tex); glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.empty() ? nullptr : data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &t.fbo); glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("FBO incomplete\n");
    return t;
}
static void draw(GLuint p, Target& t, int x = 0, int y = 0, int w = -1, int h = -1) {
    if (w < 0) w = t.w; if (h < 0) h = t.h;
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(x, y, w, h);
    glUseProgram(p);
    GLuint loc = glGetAttribLocation(p, "p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
static vector<unsigned char> read(Target& t, int w = -1, int h = -1) {
    if (w < 0) w = t.w; if (h < 0) h = t.h;
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(0, 0, t.w, t.h);
    vector<unsigned char> b((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
    return b;
}
static int maxdiff(const vector<unsigned char>& a, const vector<unsigned char>& b) {
    if (a.size() != b.size()) return 999;
    int d = 0; for (size_t i = 0; i < a.size(); ++i) d = max(d, abs((int)a[i] - (int)b[i]));
    return d;
}
static void init_gl() {
    auto getpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay d = getpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint a, b; eglInitialize(d, &a, &b); eglBindAPI(EGL_OPENGL_ES_API);
    EGLint at[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    EGLConfig c; EGLint n; eglChooseConfig(d, at, &c, 1, &n);
    EGLint ca[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(d, c, EGL_NO_CONTEXT, ca);
    eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    printf("RENDERER: %s\nVERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
    GLuint vbo; glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float v[] = {-1,-1, 3,-1, -1,3}; glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
}
int main() {
    init_gl();
    const char* fsPattern =
        "precision highp float;"
        "void main(){gl_FragColor=vec4(gl_FragCoord.x/64.0, gl_FragCoord.y/2.0, 0.5, 1.0);}";
    const char* fsSample =
        "precision highp float;uniform sampler2D A;"
        "void main(){vec2 uv=vec2((float(int(gl_FragCoord.x))+0.5)/64.0,(float(int(gl_FragCoord.y))+0.5)/2.0);"
        "gl_FragColor=texture2D(A,uv);}";
    const char* fsSample2 =
        "precision highp float;uniform sampler2D A,B;"
        "void main(){vec2 uv=vec2((float(int(gl_FragCoord.x))+0.5)/64.0,(float(int(gl_FragCoord.y))+0.5)/2.0);"
        "vec4 a=texture2D(A,uv),b=texture2D(B,uv);gl_FragColor=vec4(a.r,b.g,a.b,b.a);}";
    GLuint pPat = program(fsPattern), pS = program(fsSample), pS2 = program(fsSample2);

    // 1. direct render + read
    Target A = make_target(64, 2);
    draw(pPat, A);
    glFinish();
    auto direct = read(A);
    printf("T1 direct: %d %d %d %d\n", direct[0], direct[1], direct[2], direct[3]);

    // 2. sample FBO texture after finish only
    Target B = make_target(64, 2);
    glBindTexture(GL_TEXTURE_2D, A.tex);
    glUseProgram(pS);
    glUniform1i(glGetUniformLocation(pS, "A"), 0);
    glActiveTexture(GL_TEXTURE0);
    draw(pS, B);
    glFinish();
    auto sampled = read(B);
    auto direct2 = direct; // compare sampled against direct
    printf("T2 sample finish-only diff=%d first=%d,%d,%d,%d\n", maxdiff(direct2, sampled), sampled[0], sampled[1], sampled[2], sampled[3]);

    // 3. sample after flush+finish
    Target C = make_target(64, 2);
    glBindTexture(GL_TEXTURE_2D, A.tex);
    glUseProgram(pS); glUniform1i(glGetUniformLocation(pS, "A"), 0);
    glActiveTexture(GL_TEXTURE0);
    draw(pS, C);
    glFlush(); glFinish();
    auto sampled2 = read(C);
    printf("T3 sample flush+finish diff=%d first=%d,%d,%d,%d\n", maxdiff(direct2, sampled2), sampled2[0], sampled2[1], sampled2[2], sampled2[3]);

    // 4. two FBO textures sampled together
    Target D = make_target(64, 2);
    const char* fsB = "precision highp float;void main(){gl_FragColor=vec4(0.1,0.7,0.3,1.0);}";
    GLuint pB = program(fsB);
    draw(pB, D); glFlush(); glFinish();
    Target E = make_target(64, 2);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, A.tex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, D.tex);
    glUseProgram(pS2); glUniform1i(glGetUniformLocation(pS2, "A"), 0); glUniform1i(glGetUniformLocation(pS2, "B"), 1);
    draw(pS2, E); glFlush(); glFinish();
    auto both = read(E);
    printf("T4 two-texture first=%d,%d,%d,%d\n", both[0], both[1], both[2], both[3]);

    // 5. row1 direct vs sampled (2-row)
    printf("T5 direct row1 first=%d,%d,%d,%d sampled row1=%d,%d,%d,%d\n",
        direct[64*4], direct[64*4+1], direct[64*4+2], direct[64*4+3],
        sampled2[64*4], sampled2[64*4+1], sampled2[64*4+2], sampled2[64*4+3]);

    printf("GL_ERROR=0x%x\n", glGetError());
    return 0;
}
