// How should the v5 LayerNorm's two statistics be encoded?
//
// The apply pass needs, per row: the mean (same units as x) and the
// inverse standard deviation (a dimensionless scale, unbounded as the row
// flattens).  Those have different dimensions, and that is the whole
// question: one fixed-point pair cannot serve both as the spread moves.
//
//   K   the shipped pair: (mean, sd) * 2^-k as two q16 halves of one texel,
//       k in the second row (`ln_stats_split` + `ln_stats_reduce` +
//       `ln_apply_q16`, exactly the engine's text).  The k exists because the
//       pair is absolute (+-128); x is scaled by 2^-k to match.
//   B   mean in the residual format (it is a value of the row), invstd as f32
//       (a per-value exponent, so no k and no rail).
//   C   the same with invstd as bf16 (7-bit mantissa, 0.4% relative).
//   D   (mean, sd) both bf16 in one texel -- a per-value exponent, no k.
//   E   (mean, invstd) both f32: no encoding at all, the floor.
//
// All arms read the same I24 input row (the q16 stream would clamp before the
// statistics do at large spreads), sum in f32, and write an f32 output, so
// the only difference measured is the statistics encoding.  The reference is
// double precision on the *quantized* input the kernels read.
//
// The rows: a sweep of spreads with a fixed mean, then the shapes a v5 block
// actually produces -- a modest row (sd 2, mean 0.05) and OPUS-MT's dec1 FFN
// row (one 5959 dim in a 512-wide row, sd ~263).
//
// build: make -C engine probes/lnstats_probe
#include "shaders.h"
#include "shaders_internal.h"
#include "xformat.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using std::string;
using std::to_string;
using std::vector;

bool gpu_q4_shift() { return false; }

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNGPD)(EGLenum, void*, const EGLint*);

// The f32 and bf16 codecs are ESSL 3.00 text (uintBitsToFloat and friends),
// so the probe runs in an ES3 context and wraps every shader the way
// GPU::link_program does: no #version -> 300 es, gl_FragColor renamed,
// texture2D aliased, and the fragment stage's default precision injected.
static GLuint comp(GLenum t, const string& src) {
    string s = src;
    if (s.find("#version") == string::npos) {
        if (t == GL_VERTEX_SHADER) {
            s = "#version 300 es\n#define attribute in\n" + s;
        } else {
            for (size_t at = s.find("gl_FragColor"); at != string::npos;
                 at = s.find("gl_FragColor", at + 12))
                s.replace(at, 12, "frag_color");
            s = "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
                "#define texture2D texture\n#define textureCube texture\n" + s;
        }
    }
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

static GLuint target_tex(int w, int h = 1) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    GLuint f = 0;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("    FBO incomplete at %dx%d\n", w, h);
    return t;
}

static GLuint g_fbo = 0;

static void draw_to(GLuint p, GLuint out_tex, int w, int h) {
    if (!g_fbo) glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);
    glUseProgram(p);
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

static void bind1(GLuint p, const char* name, int unit, GLuint tex_name) {
    glActiveTexture((GLenum)(GL_TEXTURE0 + unit));
    glBindTexture(GL_TEXTURE_2D, tex_name);
    glUniform1i(glGetUniformLocation(p, name), unit);
}

static const int D = 512;

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

// One fragment a texel of the output; reads the statistics the arm wrote.
static string apply_shader(XFormat rf, const string& stats) {
    const string RW = to_string(x_texels(D, rf));
    const string WW = to_string(x_texels(D, kLnPairFmt));
    return "precision highp float;\n" + head(rf) + x_residual_glsl(rf) +
           x_codec_glsl(XFormat::F32) + x_codec_glsl(XFormat::BF16) +
           "uniform sampler2D X,S,W,B;\n"
           "float wv(int i){vec4 c=texture2D(W,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
           "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n"
           "float bv(int i){vec4 c=texture2D(B,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
           "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n"
           "void main(){int i=int(gl_FragCoord.x);"
           "float x=Y1(X,i," + RW + ".0);\n" + stats + "\n}";
}

// The candidate statistics pass: one fragment per row of the 1x2 target, each
// recomputing the sums (the probe measures the encoding, not the scan).
static string cand_stats(XFormat rf, const string& row0, const string& row1) {
    const string RW = to_string(x_texels(D, rf));
    return "precision highp float;\n" + head(rf) + x_residual_glsl(rf) +
           x_codec_glsl(XFormat::F32) + x_codec_glsl(XFormat::BF16) +
           "uniform sampler2D X;uniform float eps;"
           "void main(){float s=0.0,s2=0.0;"
           "for(int i=0;i<" + to_string(D) + ";i++){float v=Y1(X,i," + RW + ".0);s+=v;s2+=v*v;}"
           "float mean=s/float(" + to_string(D) + ");"
           "float sd=sqrt(max(s2/float(" + to_string(D) + ")-mean*mean,0.0)+eps);"
           "if(gl_FragCoord.y<1.0){" + row0 + "}else{" + row1 + "}}";
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
    const EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, xa);
    if (!ctx || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        printf("context failed\n");
        return 0;
    }
    printf("renderer %s\n", (const char*)glGetString(GL_RENDERER));

    const XFormat RF = XFormat::I24_1;
    const int OW = D;  // f32 output: one value a texel

    // K: the shipped text, exactly as the engine builds it.
    GLuint k_split = prog(ln_stats_split(D, RF));
    GLuint k_reduce = prog(ln_stats_reduce(D));
    GLuint k_apply = prog(ln_apply_q16(D, RF, XFormat::F32));
    // B: mean in the residual format, invstd f32.
    GLuint b_stats = prog(cand_stats(RF, "gl_FragColor=YST1(mean);",
                                     "gl_FragColor=x_f32enc(1.0/sd);"));
    GLuint b_apply = prog(apply_shader(RF,
        "float mu=x_q24(texture2D(S,vec2(0.5,0.25)));float inv=x_f32(texture2D(S,vec2(0.5,0.75)));"
        "gl_FragColor=x_f32enc((x-mu)*inv*wv(i)+bv(i));"));
    // C: the same with invstd as bf16.
    GLuint c_stats = prog(cand_stats(RF, "gl_FragColor=YST1(mean);",
                                     "gl_FragColor=x_bf16enc(1.0/sd);"));
    GLuint c_apply = prog(apply_shader(RF,
        "float mu=x_q24(texture2D(S,vec2(0.5,0.25)));float inv=x_bf16(texture2D(S,vec2(0.5,0.75)).rg);"
        "gl_FragColor=x_f32enc((x-mu)*inv*wv(i)+bv(i));"));
    // D: (mean, sd) both bf16 in one texel.
    GLuint d_stats = prog(cand_stats(RF,
        "gl_FragColor=vec4(x_bf16enc(mean).rg,x_bf16enc(sd).rg);", "gl_FragColor=vec4(0.0);"));
    GLuint d_apply = prog(apply_shader(RF,
        "vec4 st=texture2D(S,vec2(0.5,0.25));float mu=x_bf16(st.rg),sd=x_bf16(st.ba);"
        "gl_FragColor=x_f32enc((x-mu)/sd*wv(i)+bv(i));"));
    // E: (mean, invstd) f32.
    GLuint e_stats = prog(cand_stats(RF, "gl_FragColor=x_f32enc(mean);",
                                     "gl_FragColor=x_f32enc(1.0/sd);"));
    GLuint e_apply = prog(apply_shader(RF,
        "float mu=x_f32(texture2D(S,vec2(0.5,0.25)));"
        "float inv=x_f32(texture2D(S,vec2(0.5,0.75)));"
        "gl_FragColor=x_f32enc((x-mu)*inv*wv(i)+bv(i));"));
    if (!k_split || !k_reduce || !k_apply || !b_stats || !b_apply || !c_stats || !c_apply ||
        !d_stats || !d_apply || !e_stats || !e_apply)
        return 1;

    vector<float> w(D), b(D);
    for (int i = 0; i < D; ++i) {
        w[(size_t)i] = (float)(0.5 + 0.5 * std::sin(i * 0.13));
        b[(size_t)i] = (float)(0.1 * std::cos(i * 0.07));
    }
    const GLuint w_tex = tex(x_texels(D, kLnPairFmt), 1, x_pack(w, kLnPairFmt));
    const GLuint b_tex = tex(x_texels(D, kLnPairFmt), 1, x_pack(b, kLnPairFmt));
    const GLuint partial = target_tex(64, 1);   // the shipped split's 64 lanes
    const GLuint stat = target_tex(1, 2);
    const GLuint out = target_tex(OW, 1);

    printf("K = shipped (mean, sd)*2^-k q16 pair; B = mean in res + invstd f32; "
           "C = same, invstd bf16; D = (mean, sd) bf16; E = (mean, invstd) f32\n");

    auto run = [&](const char* tag, const vector<float>& x) {
        const vector<uint8_t> xb = x_pack(x, RF);
        vector<float> xq(D);
        for (int i = 0; i < D; ++i) xq[(size_t)i] = x_unpack(xb.data(), i, RF);
        double mu, sd;
        stats_ref(xq, mu, sd);
        vector<double> yr(D);
        for (int i = 0; i < D; ++i) yr[(size_t)i] = (xq[(size_t)i] - mu) / sd * w[(size_t)i] + b[(size_t)i];
        const GLuint xt = tex(x_texels(D, RF), 1, xb);

        vector<float> y(D);

        // K: split -> reduce -> apply.  Program, samplers and uniforms are set
        // *before* the draw: a glUniform after glDrawArrays lands on the next
        // draw of that program.
        glUseProgram(k_split);
        bind1(k_split, "X", 0, xt);
        draw_to(k_split, partial, 64, 1);
        glFinish();
        glUseProgram(k_reduce);
        bind1(k_reduce, "S", 0, partial);
        glUniform1f(glGetUniformLocation(k_reduce, "eps"), 1e-5f);
        draw_to(k_reduce, stat, 1, 2);
        glFinish();
        glUseProgram(k_apply);
        bind1(k_apply, "X", 0, xt);
        bind1(k_apply, "S", 1, stat);
        bind1(k_apply, "W", 2, w_tex);
        bind1(k_apply, "B", 3, b_tex);
        draw_to(k_apply, out, OW, 1);
        glFinish();
        {
            vector<uint8_t> px((size_t)OW * 4);
            glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
            glReadPixels(0, 0, OW, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            for (int i = 0; i < D; ++i) y[(size_t)i] = x_unpack(px.data(), i, XFormat::F32);
        }
        const Err ek = compare(y, yr);

        auto cand = [&](GLuint sp, GLuint ap, int rows) {
            glUseProgram(sp);
            bind1(sp, "X", 0, xt);
            glUniform1f(glGetUniformLocation(sp, "eps"), 1e-5f);
            draw_to(sp, stat, 1, rows);
            glFinish();
            glUseProgram(ap);
            bind1(ap, "X", 0, xt);
            bind1(ap, "S", 1, stat);
            bind1(ap, "W", 2, w_tex);
            bind1(ap, "B", 3, b_tex);
            draw_to(ap, out, OW, 1);
            glFinish();
            vector<uint8_t> px((size_t)OW * 4);
            glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
            glReadPixels(0, 0, OW, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            vector<float> yy(D);
            for (int i = 0; i < D; ++i) yy[(size_t)i] = x_unpack(px.data(), i, XFormat::F32);
            return compare(yy, yr);
        };
        const Err eb = cand(b_stats, b_apply, 2);
        const Err ec = cand(c_stats, c_apply, 2);
        const Err ed = cand(d_stats, d_apply, 1);
        const Err ee = cand(e_stats, e_apply, 2);

        printf("  %-22s mu %8.3f sd %8.3f | K %.2g/%.2g | B %.2g/%.2g | C %.2g/%.2g | "
               "D %.2g/%.2g | E %.2g/%.2g\n",
               tag, mu, sd, ek.maxabs, ek.rms, eb.maxabs, eb.rms, ec.maxabs, ec.rms,
               ed.maxabs, ed.rms, ee.maxabs, ee.rms);
        glDeleteTextures(1, &xt);
    };

    printf("max / rms relative error of the LayerNorm output (f32 out, so this is the "
           "statistics' own error)\n");
    for (double sg : {1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3}) {
        vector<float> x;
        make_row(0.7, sg, x);
        char tag[64];
        snprintf(tag, sizeof tag, "sweep sd %.0e", sg);
        run(tag, x);
    }
    {
        vector<float> x;
        make_row(0.05, 2.0, x);
        run("v5 enc/dec ln1/ln2 (sd 2)", x);
    }
    {
        vector<float> x;
        make_row(0.7, 1.0, x);
        x[295] = 5959.0f;  // OPUS-MT dec1's massive-activation dim
        run("v5 dec1 ln3 (spike 5959)", x);
    }
    return 0;
}