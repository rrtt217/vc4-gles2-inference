// Does the conv block's texture fetching work on this driver?
//
// The fused conv kernels were written against the shapes the layer needs and
// verified end to end ("layer 0's x1/xbf are byte-identical to the CPU path"),
// which is a strong check of the *composition* and a weak one of each fetch.
// Three patterns in particular are outside what the engine does elsewhere:
//
//   * conv3_act samples the depthwise kernel texture (tw x 3, NPOT height)
//     three times in one fragment, once per tap, and picks .rg or .ba by a
//     per-fragment branch.  probes/wrow_probe recorded that two fetches from
//     one sampler in one fragment returned undefined data on VC4 for a plane
//     texture; the engine's other kernels never do it.
//   * conv3_act reads S0/S1/G through X1Q, whose I16_2 form samples the same
//     texel twice inside a ternary.
//   * the state textures are one row high, the kernel is three, and neither is
//     a power of two.
//
// So this probe drives each pattern with data the CPU knows, at the engine's
// real widths, and names the values that differ.  Section B and C run the
// *shipped* shader text (shaders.cpp is linked in, not copied), so a pass is
// evidence about the engine and a failure is evidence about the engine too.
//
//   A  the kernel fetch alone: 1, 2, 3 and 5 fetches per fragment, .rg and
//      .ba, height 3 and height 4, at width 512 and width 4
//   B  conv3_act(1024) vs a CPU reference (the shipped text)
//   C  mul_act(1024) vs a CPU reference (the shipped text)
//
// build: make -C engine probes/conv_probe  /  probes/conv_probe_pi
#include "shaders.h"
#include "xformat.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

// shaders.cpp asks the GPU whether the vertex shader can do integer ops; this
// probe builds no q4 kernel, so the answer is "no".
bool gpu_q4_shift() { return false; }

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNGPD)(EGLenum, void*, const EGLint*);

static int failures = 0;

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

// One triangle covering the viewport, like the engine's setup_attrib().
static void tri(GLuint p) {
    static GLuint vbo = 0;
    if (!vbo) {
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float v[] = {-1, -1, 3, -1, -1, 3};
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    GLint loc = glGetAttribLocation(p, "p");
    glEnableVertexAttribArray((GLuint)loc);
    glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static GLuint out_tex = 0, out_fbo = 0;
static vector<uint8_t> got;
static int out_w = 0;

static void out_begin(int w) {
    if (out_w != w) {
        if (!out_tex) {
            glGenTextures(1, &out_tex);
            glGenFramebuffers(1, &out_fbo);
        }
        glBindTexture(GL_TEXTURE_2D, out_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            printf("    FBO incomplete at w=%d\n", w);
        got.assign((size_t)w * 4, 0);
        out_w = w;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
    glViewport(0, 0, w, 1);
}

static const uint8_t* out_end() {
    glFinish();
    glReadPixels(0, 0, out_w, 1, GL_RGBA, GL_UNSIGNED_BYTE, got.data());
    return got.data();
}

// q16 pairs in the target, read back as the integers the kernel wrote.
static vector<int> read_q16(int n) {
    const uint8_t* g = out_end();
    vector<int> v(2 * n);
    for (int i = 0; i < 2 * n; ++i) {
        int hi = g[i * 2], lo = g[i * 2 + 1];
        int q = (hi << 8) | lo;
        v[i] = q >= 32768 ? q - 65536 : q;
    }
    return v;
}

// 24-bit targets, read back as raw q24 units (the bias removed).
static vector<int> read_q24(int n) {
    const uint8_t* g = out_end();
    vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        int raw = (g[i * 4 + 1] << 16) | (g[i * 4 + 2] << 8) | g[i * 4 + 3];
        v[i] = raw - 8388608;
    }
    return v;
}

static void report(const char* name, const vector<int>& got_v, const vector<int>& want,
                   int tol = 0) {
    int bad = 0, first = -1, worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        int d = std::abs(got_v[i] - want[i]);
        if (d > worst) worst = d;
        if (d > tol) { ++bad; if (first < 0) first = (int)i; }
    }
    printf("  %-30s %s  bad=%d/%zu  worst |d| %d", name, bad ? "FAIL" : "PASS",
           bad, want.size(), worst);
    if (bad) printf("  [%d] got %d want %d", first, got_v[first], want[first]);
    printf("\n");
    if (bad) ++failures;
}

// ------------------------------------------------------- the kernel fetch

// Frame and channel data for the depthwise kernel, in the layout
// GPU::set_conv_kernel writes: texel q of row t holds tap t for channels 2q
// (.rg) and 2q+1 (.ba), all q16 raw.
struct Kernel {
    int dim = 0, k = 3, tw = 0;
    vector<int> q;        // [t][i] raw q16
    vector<uint8_t> px;   // the texture bytes
};

static Kernel make_kernel(int dim, int k, int height, bool ba_shift = false) {
    Kernel kn;
    kn.dim = dim; kn.k = k; kn.tw = (dim + 1) / 2;
    kn.q.assign((size_t)k * dim, 0);
    kn.px.assign((size_t)kn.tw * height * 4, 0);
    for (int t = 0; t < k; ++t)
        for (int i = 0; i < dim; ++i) {
            // Per-channel taps that are distinguishable and exercise the sign
            // path (negative for odd taps) without leaving the q16 range.
            // Weights are O(0.5) real: the conv kernel of a trained block is
            // small, and a q16 product then stays where the shader's float32
            // accumulation is exact.  Values are spread over the whole byte
            // range so a wrong fetch still shows.
            long v = (long)((i * 2654435761u + t * 7919u) % 256u) - 128;
            v = ba_shift ? -v : v;
            kn.q[(size_t)t * dim + i] = (int)v;
            const uint16_t u = (uint16_t)(int16_t)v;
            const size_t o = ((size_t)t * kn.tw + (i / 2)) * 4 + (i % 2) * 2;
            kn.px[o] = (uint8_t)(u >> 8);
            kn.px[o + 1] = (uint8_t)(u & 0xFF);
        }
    return kn;
}

// A. How many fetches from one kernel texture does this driver deliver?
//
// Every variant samples the kernel texture at the same coordinates the engine
// uses ((q+0.5)/tw, (t+0.5)/3) and copies *bytes* out: a texel byte that comes
// back from a NEAREST UNORM8 fetch re-quantises to itself exactly, so the
// comparison is byte exact and no decode can hide a wrong row.  Three fetches
// per fragment land in .r/.g/.b of the output, one per tap.
static void test_kernel_fetch(int dim, int k, int height, int nfetch, int chan,
                              const char* name) {
    Kernel kn = make_kernel(dim, k, height);
    GLuint kt = tex(kn.tw, height, kn.px);
    const string TW = std::to_string(kn.tw), HH = std::to_string(height);
    // The divisor must be a float literal: this driver rejects `(0.5)/3`
    // inside a vec2 constructor (int/float mixing in a constant expression).
    auto vr = [&](int t) { return "(" + std::to_string(t) + ".5)/" + HH + ".0"; };
    string fs = "precision highp float;\nuniform sampler2D K;uniform sampler2D J1;";
    fs += "void main(){int q=int(gl_FragCoord.x);float u=(float(q)+0.5)/" + TW + ".0;";
    if (nfetch >= 5)
        fs += "vec4 j=texture2D(J1,vec2(u,0.5));if(j.r<-1e9)gl_FragColor=vec4(1.0);";
    fs += "vec4 c0=texture2D(K,vec2(u," + vr(0) + "));"
          "vec4 c1=texture2D(K,vec2(u," + vr(1) + "));"
          "vec4 c2=texture2D(K,vec2(u," + vr(2) + "));";
    // Which source bytes each output channel must reproduce.
    int wrow[4] = {0, 0, 0, 0}, wbyte[4] = {0, 0, 0, 0};
    switch (nfetch) {
        case 1:
            fs += chan == 0 ? "gl_FragColor=vec4(c0.r,c0.g,c0.b,c0.a);"
                            : "gl_FragColor=vec4(c0.b,c0.a,0.0,1.0);";
            wrow[0] = 0; wbyte[0] = chan == 0 ? 0 : 2;
            wrow[1] = 0; wbyte[1] = chan == 0 ? 1 : 3;
            wrow[2] = 0; wbyte[2] = chan == 0 ? 2 : -1;
            wrow[3] = 0; wbyte[3] = chan == 0 ? 3 : -1;
            break;
        case 2:
            fs += chan == 0 ? "gl_FragColor=vec4(c0.r,c0.g,c1.r,c1.g);"
                            : "gl_FragColor=vec4(c0.b,c0.a,c1.b,c1.a);";
            wrow[0] = 0; wbyte[0] = chan == 0 ? 0 : 2;
            wrow[1] = 0; wbyte[1] = chan == 0 ? 1 : 3;
            wrow[2] = 1; wbyte[2] = chan == 0 ? 0 : 2;
            wrow[3] = 1; wbyte[3] = chan == 0 ? 1 : 3;
            break;
        case 3:
            fs += chan == 0 ? "gl_FragColor=vec4(c0.r,c1.g,c2.b,1.0);"
                            : "gl_FragColor=vec4(c0.b,c1.a,c2.b,1.0);";
            wrow[0] = 0; wbyte[0] = chan == 0 ? 0 : 2;
            wrow[1] = 1; wbyte[1] = chan == 0 ? 1 : 3;
            wrow[2] = 2; wbyte[2] = 2;
            wrow[3] = -1; wbyte[3] = -1;
            break;
        default:   // 5: three taps plus a second sampler in the same fragment
            fs += "gl_FragColor=vec4(c0.r,c1.g,c2.b,j.r);";
            wrow[0] = 0; wbyte[0] = 0;
            wrow[1] = 1; wbyte[1] = 1;
            wrow[2] = 2; wbyte[2] = 2;
            // J1 is the same texture, sampled at v=0.5: row 1 of 3.
            wrow[3] = 1; wbyte[3] = 0;
            break;
    }
    fs += "}";
    if (getenv("SHOW_FS")) printf("---- %s\n%s\n", name, fs.c_str());
    GLuint p = prog(fs);
    if (!p) { printf("  %-30s LINK_FAIL\n", name); ++failures; return; }
    out_begin(kn.tw);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "K"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, kt);
    if (nfetch >= 5) {
        glUniform1i(glGetUniformLocation(p, "J1"), 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, kt);
    }
    tri(p);
    out_end();

    int bad = 0, first = -1;
    for (int q = 0; q < kn.tw; ++q)
        for (int c = 0; c < 4; ++c) {
            int want = wbyte[c] < 0 ? 255 : kn.px[((size_t)wrow[c] * kn.tw + q) * 4 + wbyte[c]];
            if (got[q * 4 + c] != want) { ++bad; if (first < 0) first = q * 4 + c; }
        }
    printf("  %-30s %s  bad=%d/%d", name, bad ? "FAIL" : "PASS", bad, kn.tw * 4);
    if (bad) {
        const int q = first / 4, c = first % 4;
        printf("  texel %d ch %d: got %d want %d", q, c, got[first],
               wbyte[c] < 0 ? 255 : kn.px[((size_t)wrow[c] * kn.tw + q) * 4 + wbyte[c]]);
    }
    printf("\n");
    if (bad) ++failures;
    glDeleteProgram(p);
    glDeleteTextures(1, &kt);
}

// ------------------------------------------------- conv3 / mul_act vs CPU

// I16_2 q16 values the CPU knows, packed exactly as x_pack does.
static vector<int> make_q16(vector<uint8_t>& px) {
    const int n = (int)px.size() / 2;
    vector<int> q(n);
    for (int i = 0; i < n; ++i) {
        int hi = px[i * 2], lo = px[i * 2 + 1];
        int v = (hi << 8) | lo;
        q[i] = v >= 32768 ? v - 65536 : v;
    }
    return q;
}

static vector<uint8_t> pack_q16(const vector<int>& q) {
    vector<uint8_t> px((size_t)q.size() * 2);
    for (size_t i = 0; i < q.size(); ++i) {
        const uint16_t u = (uint16_t)(int16_t)q[i];
        px[i * 2] = (uint8_t)(u >> 8);
        px[i * 2 + 1] = (uint8_t)(u & 0xFF);
    }
    return px;
}

// Activations of the size the model actually has: mostly O(1) with a few big
// channels, which is the shape the residual stream has at layer 5-6.
static vector<int> make_act(int dim, int seed) {
    vector<int> q(dim);
    for (int i = 0; i < dim; ++i) {
        // O(1) activations with the occasional large channel, which is what
        // the residual stream looks like at layer 5-6.
        double v = 1.7 * sin(i * 0.017 + seed) + 0.4 * cos(i * 0.31 + seed * 2);
        if (i % 337 == 0) v *= 8.0;
        long x = lrint(v * 256.0);
        q[i] = (int)std::max(-32768L, std::min(32767L, x));
    }
    return q;
}

// B. The shipped conv3_act: out_raw24[i] = round((k0*a + k1*b + k2*c)/256)
static void test_conv3(int dim) {
    const int tw = (dim + 1) / 2;
    Kernel kn = make_kernel(dim, 3, 3);
    GLuint kt = tex(kn.tw, 3, kn.px);
    vector<int> a = make_act(dim, 1), b = make_act(dim, 2), c = make_act(dim, 3);
    vector<uint8_t> ap = pack_q16(a), bp = pack_q16(b), cp = pack_q16(c);
    GLuint at = tex(tw, 1, ap), bt = tex(tw, 1, bp), ct = tex(tw, 1, cp);

    GLuint p = prog(conv3_act(dim, XFormat::I16_2));
    if (!p) { printf("  conv3 LINK_FAIL\n"); ++failures; return; }
    out_begin(dim);
    glUseProgram(p);
    const char* nm[4] = {"S0", "S1", "G", "K"};
    GLuint tx[4] = {at, bt, ct, kt};
    for (int u = 0; u < 4; ++u) {
        glUniform1i(glGetUniformLocation(p, nm[u]), u);
        glActiveTexture(GL_TEXTURE0 + u);
        glBindTexture(GL_TEXTURE_2D, tx[u]);
    }
    tri(p);
    vector<int> g = read_q24(dim);

    // CPU: the same three taps, in the same order, in float32 - a q16 product
    // reaches 8e8, which float32 cannot hold exactly, so emulating the shader's
    // own precision is what makes the comparison a test of the fetch and not of
    // the arithmetic.
    vector<int> want(dim);
    for (int i = 0; i < dim; ++i) {
        float s = (float)kn.q[0 * dim + i] * (float)a[i] +
                  (float)kn.q[1 * dim + i] * (float)b[i] +
                  (float)kn.q[2 * dim + i] * (float)c[i];
        want[i] = (int)floorf(s / 256.0f + 0.5f);   // x_i24encq: floor(x+0.5)
    }
    char nameb[64];
    snprintf(nameb, sizeof nameb, "B conv3_act(%d) shipped", dim);
    // tol 1: the emulation above matches the shader's expression order; one
    // unit absorbs a driver that keeps the accumulate in a different order.
    report(nameb, g, want, 1);
    glDeleteProgram(p);
    glDeleteTextures(1, &at); glDeleteTextures(1, &bt);
    glDeleteTextures(1, &ct); glDeleteTextures(1, &kt);
}

// C. The shipped mul_act: inputs are 24-bit q24 raw, output a q16 pair.
static void test_mul(int dim) {
    const int tw = x_texels(dim, XFormat::I16_2);
    vector<int> a24(dim), b24(dim);
    for (int i = 0; i < dim; ++i) {
        a24[i] = (int)lrint(1.7 * sin(i * 0.021 + 0.3) * 256.0);
        b24[i] = (int)lrint(1.1 * cos(i * 0.013 + 1.1) * 256.0);
    }
    auto pack24 = [](const vector<int>& v) {
        vector<uint8_t> px((size_t)v.size() * 4, 0);
        for (size_t i = 0; i < v.size(); ++i) {
            long q = (long)v[i] + 8388608L;
            px[i * 4 + 0] = 0;
            px[i * 4 + 1] = (uint8_t)((q >> 16) & 0xFF);
            px[i * 4 + 2] = (uint8_t)((q >> 8) & 0xFF);
            px[i * 4 + 3] = (uint8_t)(q & 0xFF);
        }
        return px;
    };
    vector<uint8_t> ap = pack24(a24), bp = pack24(b24);
    GLuint at = tex(dim, 1, ap), bt = tex(dim, 1, bp);

    GLuint p = prog(mul_act(dim, XFormat::I16_2));
    if (!p) { printf("  mul LINK_FAIL\n"); ++failures; return; }
    out_begin(tw);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "A"), 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, at);
    glUniform1i(glGetUniformLocation(p, "B"), 1);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, bt);
    glUniform1f(glGetUniformLocation(p, "aw"), (float)dim);
    tri(p);
    vector<int> g = read_q16(tw);

    // a and b are q24 raw (real*256); their product is real^2*65536, which the
    // shader stores as a q16 pair (real*256) -> raw = round(a*b/256).
    vector<int> want(dim);
    for (int i = 0; i < dim; ++i) {
        // x_st2 -> x_i16enc(v) -> x_q16enc(v*256) -> clamp(floor(x+0.5))
        float v = ((float)a24[i] * (float)b24[i]) / 65536.0f;
        long q = (long)floorf(v * 256.0f + 0.5f);
        want[i] = (int)std::max(-32768L, std::min(32767L, q));
    }
    char nameb[64];
    snprintf(nameb, sizeof nameb, "C mul_act(%d) shipped", dim);
    report(nameb, g, want, 0);
    glDeleteProgram(p);
    glDeleteTextures(1, &at); glDeleteTextures(1, &bt);
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
    GLint maxu = 0, maxt = 0;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxu);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxt);
    printf("renderer %s\n", (const char*)glGetString(GL_RENDERER));
    printf("version  %s\n", (const char*)glGetString(GL_VERSION));
    printf("units %d  max texture %d\n\n", maxu, maxt);

    printf("A. the depthwise kernel fetch on its own (width 512, height 3)\n");
    test_kernel_fetch(1024, 3, 3, 1, 0, "A1 one fetch .rg");
    test_kernel_fetch(1024, 3, 3, 2, 0, "A2 two fetches .rg");
    test_kernel_fetch(1024, 3, 3, 3, 0, "A3 three fetches .rg");
    test_kernel_fetch(1024, 3, 3, 3, 1, "A4 three fetches .ba");
    test_kernel_fetch(1024, 3, 3, 5, 0, "A5 three fetches +2 samplers");
    test_kernel_fetch(1024, 3, 4, 3, 0, "A6 three fetches, height 4");
    test_kernel_fetch(8, 3, 3, 3, 0, "A7 three fetches, width 4");

    printf("\nB/C. the shipped conv kernels vs a CPU reference\n");
    test_conv3(1024);
    test_mul(1024);

    printf("\n%s (%d failing)\n", failures ? "FAIL" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
