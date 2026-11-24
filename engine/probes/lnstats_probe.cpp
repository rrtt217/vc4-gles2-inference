// What do the LayerNorm statistics cost, and at which residual scale?
//
// The v5 block stores (mean, invstd) as two q16 halves of one texel.  Those two
// numbers have opposite dimensions: mean scales with the residual's spread, and
// invstd is its reciprocal, so one fixed 1/256 step cannot serve both as the
// spread moves.  This probe drives the shipped `ln_stats_q16`/`ln_apply_q16`
// and two candidate encodings on the same rows, at spreads from 1e-2 to 1e3,
// and compares each to a double-precision reference:
//
//   A   the shipped (mean, invstd) q16 pair
//   B1  (mean, sd) scaled by 2^-k, k passed as a uniform (the encoding with a
//       free per-row exponent)
//   B2  the same, with k packed in the texel's third byte and int8 mantissas
//       (a self-contained one-texel form)
//   C   (mean, sd) as bfloat16 -- a per-value exponent, no k (host-computed;
//       this context is ES2)
//
// The input row is I24 (the q16 stream would clamp before the statistics do at
// large spreads), and the reference is computed on the *quantized* input the
// kernels actually read, so the only difference between the three rows is the
// statistics encoding.
//
// build: make -C engine probes/lnstats_probe
#include "shaders.h"
#include "xformat.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using std::string;
using std::vector;

bool gpu_q4_shift() { return false; }

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNGPD)(EGLenum, void*, const EGLint*);

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

static GLuint prog(const string& fs) {
    static const char* VS = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, comp(GL_VERTEX_SHADER, VS));
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

static GLuint tex(int w, int h, const vector<uint8_t>& px) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return t;
}

static GLuint target_tex(int w) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    GLuint f = 0;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("    FBO incomplete at w=%d\n", w);
    return t;
}

static void tri(GLuint p, int w, int h) {
    static GLuint vbo = 0;
    if (!vbo) {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float v[] = {-1, -1, 3, -1, -1, 3};
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    glViewport(0, 0, w, h);
    GLint loc = glGetAttribLocation(p, "p");
    glEnableVertexAttribArray((GLuint)loc);
    glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static const char* kQ16 =
    "vec2 q16(float v){float q=clamp(floor(v*256.0+0.5),-32768.0,32767.0);"
    "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
    "return vec2(hi/255.0,(q-hi*256.0)/255.0);}";
static const char* kD16 =
    "float d16(vec2 c){float u=c.x*65280.0+c.y*255.0;"
    "if(u>=32768.0)u-=65536.0;return u/256.0;}";
static const char* kD24 =
    "float d24(vec4 c){return (float(int(c.g*255.0+0.5))*65536.0+"
    "float(int(c.b*255.0+0.5))*256.0+float(int(c.a*255.0+0.5))-8388608.0)/256.0;}";

static const int D = 512;
static const int OW = D / 2;

static void make_row(double mu, double sig, vector<float>& x) {
    vector<double> z(D);
    double m = 0;
    for (int i = 0; i < D; ++i) { z[(size_t)i] = std::sin(i * 0.37) + 0.3 * std::cos(i * 1.11); m += z[(size_t)i]; }
    m /= D;
    double v = 0;
    for (int i = 0; i < D; ++i) v += (z[(size_t)i] - m) * (z[(size_t)i] - m);
    const double sd = std::sqrt(v / D);
    x.resize(D);
    for (int i = 0; i < D; ++i) x[(size_t)i] = (float)(mu + sig * (z[(size_t)i] - m) / sd);
}

static void stats_ref(const vector<float>& xq, double& mu, double& sd) {
    double s = 0, s2 = 0;
    for (float v : xq) { s += v; s2 += (double)v * v; }
    mu = s / D;
    sd = std::sqrt(std::max(0.0, s2 / D - mu * mu) + 1e-5);
}

struct Err { double maxabs = 0, rms = 0; };
static Err compare(const vector<float>& yg, const vector<double>& yr) {
    Err e;
    double acc = 0, ref = 0;
    for (int i = 0; i < D; ++i) {
        const double d = yg[(size_t)i] - yr[(size_t)i];
        e.maxabs = std::max(e.maxabs, std::fabs(d));
        acc += d * d;
        ref += yr[(size_t)i] * yr[(size_t)i];
    }
    e.rms = std::sqrt(acc / std::max(ref, 1e-30));
    return e;
}

// stat_tex: 1x1; out reads OW texels of I16_2.
static vector<float> apply(GLuint p, GLuint x_tex, GLuint s_tex, GLuint w_tex,
                           GLuint b_tex, GLuint out_tex, float iscale) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);
    glUseProgram(p);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, x_tex); glUniform1i(glGetUniformLocation(p, "X"), 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, s_tex); glUniform1i(glGetUniformLocation(p, "S"), 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, w_tex); glUniform1i(glGetUniformLocation(p, "W"), 2);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, b_tex); glUniform1i(glGetUniformLocation(p, "B"), 3);
    glUniform1f(glGetUniformLocation(p, "iscale"), iscale);
    tri(p, OW, 1);
    glFinish();
    vector<uint8_t> px((size_t)OW * 4);
    glReadPixels(0, 0, OW, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glDeleteFramebuffers(1, &fbo);
    vector<float> y(D);
    for (int i = 0; i < D; ++i) y[(size_t)i] = x_unpack(px.data(), i, XFormat::I16_2);
    return y;
}

static void run_stats(GLuint p, GLuint x_tex, GLuint s_tex, float sc) {
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s_tex, 0);
    glUseProgram(p);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, x_tex); glUniform1i(glGetUniformLocation(p, "X"), 0);
    glUniform1f(glGetUniformLocation(p, "eps"), 1e-5f);
    glUniform1f(glGetUniformLocation(p, "sc"), sc);
    tri(p, 1, 1);
    glFinish();
    glDeleteFramebuffers(1, &fbo);
}

int main() {
    PFNGPD gpd = (PFNGPD)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!gpd) { printf("no eglGetPlatformDisplayEXT\n"); return 1; }
    EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint maj = 0, min = 0;
    if (!eglInitialize(d, &maj, &min)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ca[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                         EGL_ALPHA_SIZE, 8, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                         EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(d, ca, &cfg, 1, &n) || n < 1) { printf("eglChooseConfig n=%d\n", n); return 1; }
    const EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, xa);
    if (!ctx || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        printf("context failed\n");
        return 1;
    }
    printf("renderer %s\n\n", (const char*)glGetString(GL_RENDERER));

    // A: the shipped builders, exactly as the v5 block builds them.
    GLuint a_stats = prog(ln_stats_q16(D, XFormat::I24_1));
    GLuint a_apply = prog(ln_apply_q16(D, XFormat::I24_1, XFormat::I16_2));

    // B1: (mean, sd) q16, scaled by a host-supplied 2^-k.
    GLuint b1_stats = prog(
        string("precision highp float;uniform sampler2D X;uniform float sc;\n") + kD24 + kQ16 +
        "void main(){float s=0.0,s2=0.0;"
        "for(int i=0;i<512;i++){float v=d24(texture2D(X,vec2((float(i)+0.5)/512.0,0.5)));s+=v;s2+=v*v;}"
        "float m=s/512.0;float sd=sqrt(max(s2/512.0-m*m,0.0)+1e-5);"
        "vec2 a=q16(m*sc),b=q16(sd*sc);gl_FragColor=vec4(a,b);}");
    GLuint b1_apply = prog(
        string("precision highp float;uniform sampler2D X,S,W,B;uniform float iscale;\n") + kD24 + kD16 + kQ16 +
        "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
        "vec4 st=texture2D(S,vec2(0.5,0.5));"
        "float m=d16(st.rg)*iscale,sd=d16(st.ba)*iscale;float inv=1.0/sd;"
        "float y0=0.0,y1=0.0;"
        "for(int q=0;q<2;q++){int i=i0+q;"
        "float x=d24(texture2D(X,vec2((float(i)+0.5)/512.0,0.5)));"
        "vec4 wc=texture2D(W,vec2((float(i/2)+0.5)/256.0,0.5));"
        "vec4 bc=texture2D(B,vec2((float(i/2)+0.5)/256.0,0.5));"
        "float wv=((i/2)*2==i)?d16(wc.rg):d16(wc.ba);"
        "float bv=((i/2)*2==i)?d16(bc.rg):d16(bc.ba);"
        "float y=(x-m)*inv*wv+bv;if(q==0)y0=y;else y1=y;}"
        "vec2 p=q16(y0),r=q16(y1);gl_FragColor=vec4(p,r);}");

    // B2: (mean, sd) int8 pair plus the per-row exponent in the third byte.
    GLuint b2_stats = prog(
        string("precision highp float;uniform sampler2D X;\n") + kD24 +
        "void main(){float s=0.0,s2=0.0;"
        "for(int i=0;i<512;i++){float v=d24(texture2D(X,vec2((float(i)+0.5)/512.0,0.5)));s+=v;s2+=v*v;}"
        "float m=s/512.0;float sd=sqrt(max(s2/512.0-m*m,0.0)+1e-5);"
        "float smax=max(abs(m),sd);float kf=clamp(ceil(log2(max(smax,1e-20)/64.0)),-16.0,16.0);"
        "float sc=exp2(-kf);"
        "float ms=clamp(floor(m*sc+0.5),-127.0,127.0);"
        "float ss=clamp(floor(sd*sc+0.5),-127.0,127.0);"
        "gl_FragColor=vec4((ms+128.0)/255.0,(ss+128.0)/255.0,(kf+32.0)/255.0,0.0);}");
    GLuint b2_apply = prog(
        string("precision highp float;uniform sampler2D X,S,W,B;\n") + kD24 + kD16 + kQ16 +
        "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
        "vec4 st=texture2D(S,vec2(0.5,0.5));"
        "float ms=floor(st.r*255.0+0.5)-128.0,ss=floor(st.g*255.0+0.5)-128.0;"
        "float kf=floor(st.b*255.0+0.5)-32.0;float sc=exp2(kf);"
        "float m=ms*sc,sd=ss*sc;float inv=1.0/sd;"
        "float y0=0.0,y1=0.0;"
        "for(int q=0;q<2;q++){int i=i0+q;"
        "float x=d24(texture2D(X,vec2((float(i)+0.5)/512.0,0.5)));"
        "vec4 wc=texture2D(W,vec2((float(i/2)+0.5)/256.0,0.5));"
        "vec4 bc=texture2D(B,vec2((float(i/2)+0.5)/256.0,0.5));"
        "float wv=((i/2)*2==i)?d16(wc.rg):d16(wc.ba);"
        "float bv=((i/2)*2==i)?d16(bc.rg):d16(bc.ba);"
        "float y=(x-m)*inv*wv+bv;if(q==0)y0=y;else y1=y;}"
        "vec2 p=q16(y0),r=q16(y1);gl_FragColor=vec4(p,r);}");
    if (!a_stats || !a_apply || !b1_stats || !b1_apply || !b2_stats || !b2_apply) return 1;

    vector<float> w(D), b(D);
    for (int i = 0; i < D; ++i) {
        w[(size_t)i] = (float)(0.5 + 0.5 * std::sin(i * 0.13));
        b[(size_t)i] = (float)(0.1 * std::cos(i * 0.07));
    }
    const GLuint w_tex = tex(OW, 1, x_pack(w, XFormat::I16_2));
    const GLuint b_tex = tex(OW, 1, x_pack(b, XFormat::I16_2));
    const GLuint s_tex = target_tex(1);
    const GLuint o_tex = target_tex(OW);

    printf("A = shipped (mean, invstd) q16; B1 = (mean, sd) q16 + free 2^-k; "
           "B2 = int8 pair + k in the texel\n");
    for (double sg : {1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3}) {
        vector<float> x;
        make_row(0.7, sg, x);
        const vector<uint8_t> xb = x_pack(x, XFormat::I24_1);
        vector<float> xq(D);
        for (int i = 0; i < D; ++i) xq[(size_t)i] = x_unpack(xb.data(), i, XFormat::I24_1);
        double mu, sd;
        stats_ref(xq, mu, sd);
        vector<double> yr(D);
        for (int i = 0; i < D; ++i)
            yr[(size_t)i] = (xq[(size_t)i] - mu) / sd * w[(size_t)i] + b[(size_t)i];

        const GLuint x_tex = tex(D, 1, xb);
        const double smax = std::max(std::fabs(mu), sd);
        int k = (int)std::ceil(std::log2(std::max(smax, 1e-20) / 64.0));
        k = std::max(-16, std::min(16, k));

        run_stats(a_stats, x_tex, s_tex, 0.0f);
        const Err ea = compare(apply(a_apply, x_tex, s_tex, w_tex, b_tex, o_tex, 0.0f), yr);
        run_stats(b1_stats, x_tex, s_tex, (float)std::ldexp(1.0, -k));
        const Err eb1 = compare(apply(b1_apply, x_tex, s_tex, w_tex, b_tex, o_tex,
                                      (float)std::ldexp(1.0, k)), yr);
        run_stats(b2_stats, x_tex, s_tex, 0.0f);
        const Err eb2 = compare(apply(b2_apply, x_tex, s_tex, w_tex, b_tex, o_tex, 0.0f), yr);

        // C: the same (mean, sd) pair as bfloat16 -- a per-value exponent and no
        // k at all.  Computed on the host (this context is ES2) with the same
        // rounding x_pack uses, and the result rounded to the output's q16 so
        // the comparison is like for like.
        const vector<uint8_t> bp = x_pack(vector<float>{(float)mu, (float)sd}, XFormat::BF16);
        const double mb = x_unpack(bp.data(), 0, XFormat::BF16);
        const double sb = x_unpack(bp.data(), 1, XFormat::BF16);
        vector<float> yc(D);
        for (int i = 0; i < D; ++i) {
            const float v = (float)((xq[(size_t)i] - mb) / sb * w[(size_t)i] + b[(size_t)i]);
            yc[(size_t)i] = x_unpack(x_pack(vector<float>{v}, XFormat::I16_2).data(), 0, XFormat::I16_2);
        }
        const Err ec = compare(yc, yr);

        printf("  sigma %7.3g  mu %7.3g  k %+3d | A max %.3g rms %.2g | "
               "B1 max %.3g rms %.2g | B2 max %.3g rms %.2g | C(bf16) max %.3g rms %.2g\n",
               sg, mu, k, ea.maxabs, ea.rms, eb1.maxabs, eb1.rms, eb2.maxabs, eb2.rms,
               ec.maxabs, ec.rms);
        glDeleteTextures(1, &x_tex);
    }
    return 0;
}