// What does each XFormat's encode and decode cost per value, on this driver?
//
// xformat.h defines seven streams: I16_1, I16_2, I8_4, I24_1, S24_1, F32 and
// BF16.  Every kernel in the engine reads one of them through x_glsl(f)'s
// X1/X4 macros and writes one through its XST* macros, so the cost of those
// two directions is the floor under every matvec, norm and residual add.  The
// kernels here are built from that same emitted text; nothing is re-typed.
//
// Per format, two timed draws and two checks:
//
//   decode  one fragment a value: `X1(T, i, TW)`.  Its store is a plain
//           fract() write, no codec, shared by every arm -- so the column is
//           the decode alone.  A 2- or 4-value layout decodes one value a
//           fragment, i.e. a value's own fetch; sharing a fetch between two
//           values (x_i16pair) is a kernel decision priced by q16dec_probe
//           and t3act_probe, not a property of the codec.
//   encode  one fragment a texel: `XSTp(v0..)` of a pattern, into a target
//           that *is* the encoded row.
//
// The checks are exact, not a tolerance: the pattern is k/16 in +-7.875,
// which every format represents exactly (i8's rail is +-7.94 at a 1/16 step,
// s24 is 1/4096, bf16 has the 1/16 in three bits, f32 exact).  The encode is
// compared byte for byte against x_pack() of the same host pattern, and the
// decode is checked by one untimed round trip -- decode then re-encode in the
// same format -- against the input texture's bytes.  GL_DITHER is off: an
// 8-bit render target would otherwise round writes the check reads back.
//
// Two clocks, because they answer different questions.  The wall column
// includes the submit and finish around a 1024-fragment draw, which on VC4
// dominates a cheap arm (probes/README.md); the counter column is the QPU's
// own fragment-shading time when PERFCNT=1, divided by the measured 12
// QPU-equivalents and 300 MHz (perfcnt.h).  Compare formats by the gpu
// column, and the wall column by itself over time.
//
// build: make -C engine probes/codec_probe[_pi]
// run:   ./codec_probe [--values N] [--iters K]
#include "egl_surfaceless.h"
#include "perfcnt.h"

#include "shaders.h"
#include "shaders_internal.h"
#include "xformat.h"

#include <GLES2/gl2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::to_string;
using std::vector;

// shaders_q4.cpp asks the GPU-side switch (gpu_prog.cpp) for the nibble-split
// arm; a probe that links SHADERLIB but no GPU has to answer it.  The q4
// builders are not part of this probe.
bool gpu_q4_shift() { return false; }

// The engine's es3ify() (gpu_prog.cpp) for the fragment stage: ESSL 3.00
// removes gl_FragColor and texture2D and has no default float precision.
static string es3ify_fs(const string& src) {
    if (src.find("#version") != string::npos) return src;
    string body = src;
    for (size_t at = body.find("gl_FragColor"); at != string::npos;
         at = body.find("gl_FragColor", at + 12))
        body.replace(at, 12, "frag_color");
    return "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
           "#define texture2D texture\n#define textureCube texture\n" + body;
}

static GLuint compile(GLenum t, const string& s, string& log) {
    GLuint h = glCreateShader(t);
    const char* p = s.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetShaderInfoLog(h, sizeof(b) - 1, nullptr, b);
        log = b;
        return 0;
    }
    return h;
}

static GLuint build(const string& fs_in, bool es3, string& log) {
    const string fs = es3 ? es3ify_fs(fs_in) : fs_in;
    const string vs = es3 ? "#version 300 es\n#define attribute in\n"
                            "attribute vec2 A;void main(){gl_Position=vec4(A,0.,1.);}"
                          : "attribute vec2 A;void main(){gl_Position=vec4(A,0.,1.);}";
    GLuint v = compile(GL_VERTEX_SHADER, vs, log);
    if (!v) return 0;
    GLuint f = compile(GL_FRAGMENT_SHADER, fs, log);
    if (!f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "A");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetProgramInfoLog(p, sizeof(b) - 1, nullptr, b);
        log = b;
        return 0;
    }
    return p;
}

static GLuint make_tex(int w, int h, const vector<uint8_t>& px) {
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

static GLuint make_target(int w, int h) {
    GLuint t = make_tex(w, h, vector<uint8_t>((size_t)w * h * 4, 0));
    GLuint f = 0;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("    FBO incomplete at %dx%d\n", w, h);
    return t;
}

// The pattern: k/16 with k = (37*i mod 253) - 126, in +-7.875.  Computed in
// float on both sides (integer `%` is ESSL 3.00 only) and exact in f32 for
// these magnitudes, so the host and the shader agree on every value.
static inline float pat(int i) {
    const float a = (float)i * 37.0f;
    const float k = a - floorf(a / 253.0f) * 253.0f - 126.0f;
    return k * 0.0625f;
}

static string pat_glsl_float() {
    return "float pat(float i){float a=i*37.0;"
           "return (a-floor(a/253.0)*253.0-126.0)*0.0625;}";
}

// One fragment a value; the store is a shared, codec-free fract write.
static string dec_shader(XFormat f, int n) {
    return head(f) +
        "uniform sampler2D T;uniform float sc1;uniform float sc2;" + pat_glsl_float() +
        "void main(){int i=int(gl_FragCoord.x);"
        "float v=X1(T,i," + to_string(x_texels(n, f)) + ".0);"
        "gl_FragColor=vec4(fract(v*sc1),fract(v*sc2),0.0,1.0);}";
}

// One fragment a texel.  `src` is either the pattern (the encode arm) or the
// decoded input (the round-trip check), so the two programmes differ only in
// where a value comes from.
static string enc_shader(XFormat f, int n, bool from_texture) {
    const int p = x_per_texel(f);
    const string TW = to_string(x_texels(n, f));
    string vals;
    for (int j = 0; j < p; ++j) {
        const string k = to_string(j);
        vals += "float v" + k + "=" + (from_texture ? "X1(T,i0+" + k + "," + TW + ".0)"
                                                    : "pat(float(i0+" + k + "))") + ";";
    }
    string args;
    for (int j = 0; j < p; ++j) args += (j ? "," : "") + string("v") + to_string(j);
    return head(f) + (from_texture ? string("uniform sampler2D T;") : string()) +
        pat_glsl_float() +
        "void main(){int i0=int(gl_FragCoord.x)*" + to_string(p) + ";" + vals +
        "gl_FragColor=XST" + to_string(p) + "(" + args + ");}";
}

struct Arm {
    const char* name;
    XFormat f;
    bool es3;   // the codec is a bit-cast (F32, BF16): ESSL 3.00 text
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int N = 1024, iters = 2000;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--values") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--values N] [--iters K]\n", argv[0]); return 2; }
    }
    // One fragment a value at most, so N is the fragment count and also a row
    // width; the engine's own cap is 1024 (kDrawCap) and VC4 breaks at 2048.
    if (N < 64 || N > 1024 || (N % 16)) {
        printf("--values must be a multiple of 16 in 64..1024\n");
        return 2;
    }

    // The Mesa overrides the engine sets for its q4/f32 kernels; the ES 3.0
    // context request is what a non-Mesa driver needs.  vc4 reports 2.0
    // without the override.
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
    EglSurfaceless e = egl_surfaceless_init(3);
    if (!e.ok()) return 1;
    const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    const bool es3_ctx = glsl && !strncmp(glsl, "OpenGL ES GLSL ES 3", 19);
    printf("== codec_probe: %d values a draw, %d draws an arm ==\n", N, iters);
    printf("renderer %s\nglsl     %s\n", (const char*)glGetString(GL_RENDERER),
           glsl ? glsl : "?");
    glDisable(GL_DITHER);  // the checks read 8-bit render targets back
    GLuint g_fbo = 0;
    glGenFramebuffers(1, &g_fbo);

    // The shared 64/64 vertex buffer and a unit quad.
    GLuint vb = 0;
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    const float tri[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // The input row, packed by the same x_pack the engine's uploads use.
    vector<float> src((size_t)N);
    for (int i = 0; i < N; ++i) src[(size_t)i] = pat(i);

    const Arm arms[] = {
        {"i16_1", XFormat::I16_1, false},
        {"i16_2", XFormat::I16_2, false},
        {"i8_4",  XFormat::I8_4,  false},
        {"i24_1", XFormat::I24_1, false},
        {"s24_1", XFormat::S24_1, false},
        {"f32",   XFormat::F32,   true},
        {"bf16",  XFormat::BF16,  true},
    };
    const int na = (int)(sizeof arms / sizeof arms[0]);

    PerfMon pm;
    const bool want_c = perfcnt_enabled() && !pm.missing();
    if (want_c)
        pm.pick({"QPU-total-clk-cycles-executing-valid-instr"});
    const size_t EXEC = 0;

    printf("\n%-7s %3s %7s %11s %11s %11s %11s  %s\n",
           "format", "pv", "texels", "dec wall ns", "dec gpu ns", "enc wall ns",
           "enc gpu ns", "check");

    for (int a = 0; a < na; ++a) {
        const Arm& A = arms[a];
        if (A.es3 && !es3_ctx) {
            printf("%-7s %3d %7d  skipped: the codec is ESSL 3.00 text and the "
                   "context is not\n", A.name, x_per_texel(A.f), x_texels(N, A.f));
            continue;
        }
        const int p = x_per_texel(A.f);
        const int tw = x_texels(N, A.f);
        string log;
        GLuint pdec = build(dec_shader(A.f, N), A.es3, log);
        GLuint penc = build(enc_shader(A.f, N, false), A.es3, log);
        GLuint prt = build(enc_shader(A.f, N, true), A.es3, log);
        if (!pdec || !penc || !prt) {
            printf("%-7s COMPILE FAIL: %s\n", A.name, log.c_str());
            continue;
        }

        const vector<uint8_t> bytes = x_pack(src, A.f);
        GLuint xt = make_tex(tw, 1, bytes);
        GLuint out_dec = make_target(N, 1);
        GLuint out_enc = make_target(tw, 1);
        auto attach = [&](GLuint tex) {
            glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        };
        auto read_back = [&](GLuint tex, int w, int h) {
            attach(tex);
            vector<uint8_t> px((size_t)w * h * 4);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            return px;
        };
        auto use = [&](GLuint prog, int w, int h, bool with_tex) {
            glViewport(0, 0, w, h);
            glUseProgram(prog);
            if (with_tex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, xt);
                glUniform1i(glGetUniformLocation(prog, "T"), 0);
            }
        };

        // ---- check 1: the round trip, byte for byte -----------------------
        use(prt, tw, 1, true);
        attach(out_enc);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> rt = read_back(out_enc, tw, 1);
        int bad_rt = 0;
        for (size_t b = 0; b < bytes.size() && b < rt.size(); ++b)
            if (bytes[b] != rt[b]) ++bad_rt;

        // ---- check 2: the encode against x_pack of the same pattern -------
        use(penc, tw, 1, false);
        attach(out_enc);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> en = read_back(out_enc, tw, 1);
        int bad_en = 0;
        for (size_t b = 0; b < bytes.size() && b < en.size(); ++b)
            if (bytes[b] != en[b]) ++bad_en;

        // ---- the two timed regions ---------------------------------------
        // Decode: N fragments; the fract write is validated against x_unpack
        // of the same bytes, so a decode that returns the wrong stream is
        // named rather than measured as fast.
        static const float SC1 = 13.0f, SC2 = 7.389f;
        use(pdec, N, 1, true);
        glUniform1f(glGetUniformLocation(pdec, "sc1"), SC1);
        glUniform1f(glGetUniformLocation(pdec, "sc2"), SC2);
        attach(out_dec);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> dc = read_back(out_dec, N, 1);
        int bad_dc = 0;
        for (int i = 0; i < N; ++i) {
            const double v = x_unpack(bytes.data(), i, A.f);
            for (int ch = 0; ch < 2; ++ch) {
                const double sc = ch ? SC2 : SC1;
                double want = std::fmod(v * sc, 1.0);
                if (want < 0) want += 1.0;
                const double got = dc[(size_t)i * 4 + (size_t)ch] / 255.0;
                double d = std::fabs(got - want);
                if (d > 0.5) d = 1.0 - d;  // fract wraps
                if (d > 3.0 / 255.0) ++bad_dc;
            }
        }

        auto time_dec = [&]() {
            use(pdec, N, 1, true);
            glUniform1f(glGetUniformLocation(pdec, "sc1"), SC1);
            glUniform1f(glGetUniformLocation(pdec, "sc2"), SC2);
            attach(out_dec);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count() / iters;
        };
        auto time_enc = [&]() {
            use(penc, tw, 1, false);
            attach(out_enc);
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count() / iters;
        };

        const double dec_ms = time_dec();
        const double enc_ms = time_enc();

        double dec_gpu = 0, enc_gpu = 0;
        if (want_c) {
            const double to_ms = 1.0 / (pm.qpus() * PERFCNT_CLOCK_MHZ * 1000.0);
            use(pdec, N, 1, true);
            glUniform1f(glGetUniformLocation(pdec, "sc1"), SC1);
            glUniform1f(glGetUniformLocation(pdec, "sc2"), SC2);
            attach(out_dec);
            pm.begin();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            const PerfMon::Sample s = pm.end();
            if (!s.value.empty()) dec_gpu = (double)s.value[EXEC] * to_ms / iters;

            use(penc, tw, 1, false);
            attach(out_enc);
            pm.begin();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            const PerfMon::Sample s2 = pm.end();
            if (!s2.value.empty()) enc_gpu = (double)s2.value[EXEC] * to_ms / iters;
        }

        char check[96];
        if (bad_rt || bad_en || bad_dc)
            snprintf(check, sizeof check, "MISMATCH rt %d, enc %d, dec %d",
                     bad_rt, bad_en, bad_dc);
        else
            snprintf(check, sizeof check, "exact");
        const double per_val = 1e6 / (double)N;  // ms a draw -> ns a value
        printf("%-7s %3d %7d %11.3f %11.3f %11.3f %11.3f  %s\n",
               A.name, p, tw, dec_ms * per_val, dec_gpu * per_val,
               enc_ms * per_val, enc_gpu * per_val, check);

        glDeleteTextures(1, &xt);
        glDeleteTextures(1, &out_dec);
        glDeleteTextures(1, &out_enc);
        glDeleteProgram(pdec);
        glDeleteProgram(penc);
        glDeleteProgram(prt);
    }

    printf("\nwall ns is a 1024-fragment draw's submit+finish divided by the work; "
           "gpu ns is the QPU's fragment-shading cycles when PERFCNT=1\n");
    if (!want_c)
        printf("PERFCNT=1 adds the gpu column (the wall column alone is "
               "submission-dominated on VC4)\n");
    return 0;
}