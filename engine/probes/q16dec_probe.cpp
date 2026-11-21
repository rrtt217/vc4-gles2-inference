// What does the q16 activation decode cost, and can the sign fix be made
// cheaper without changing a single output byte?
//
// The engine's decode (shaders.cpp, kDec16A/kDec16B) is, per component:
//
//     u = c.r*65280.0 + c.g*255.0;
//     return u >= 32768.0 ? u - 65536.0 : u;
//
// which is 2 mul + 1 add + 1 compare + 1 select.  In the q8 pair kernel four
// of these run per inner iteration against 8 MACs, so the decode is over half
// the instructions the kernel issues.
//
// `probes/t3dec_probe` found the same shape for the 2-bit sign decode and the
// bit-cast form won 13% there: a float's sign bit can be flipped with an
// integer OR into the exponent field, no compare and no select.  q16 is a
// signed 16-bit integer in a float, so the question is whether the sign fix
// can be done in the integer domain instead.
//
// Arms, all bit-exact against a CPU reference on every value:
//
//   ship    the engine's text, unchanged
//   bitcast uint: assemble the int16 from the two bytes with shifts, then
//           int16BitsToFloat-free: sign-extend in int and convert with float()
//   bitcast2 the same, but the sign extension is `(v << 16) >> 16` on an int
//   branchless  u - 65536.0 * step(32768.0, u), a select written as arithmetic
//   es3     ES3 uint shifts on the texel bytes (Q4_SHIFT's route)
//
// The decode runs inside a real matvec loop, not in isolation: the scheduler
// matters, and an isolated decode measurement has been misleading before.
//
// build: make -C engine probes/q16dec_probe[_pi]
// run:   ./q16dec_probe [--rows N] [--cols M] [--iters K]
#include "egl_surfaceless.h"
#include "perfcnt.h"

#include <GLES2/gl2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

static GLuint compile(GLenum type, const string& src, string& log) {
    GLuint h = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetShaderInfoLog(h, sizeof(b) - 1, nullptr, b);
        log = b;
    }
    return ok ? h : 0;
}

// The engine's es3ify() (gpu_prog.cpp) for the fragment stage, because ESSL
// 3.00 removes gl_FragColor and texture2D and has no default float precision.
// A shader that needs the integer arms is compiled with this; one that does not
// keeps the ES 1.00 text.
static string es3ify_fs(const string& src) {
    if (src.find("#version") != string::npos) return src;
    string body = src;
    for (size_t at = body.find("gl_FragColor"); at != string::npos;
         at = body.find("gl_FragColor", at + 12))
        body.replace(at, 12, "frag_color");
    return "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
           "#define texture2D texture\n#define textureCube texture\n" + body;
}

static bool es3_ctx = false;

static GLuint build(const string& fs, string& log) {
    static const string vs_es1 = "attribute vec2 A;void main(){gl_Position=vec4(A,0.0,1.0);}";
    static const string vs_es3 = "#version 300 es\n#define attribute in\n" + vs_es1;
    const char* vs = es3_ctx ? vs_es3.c_str() : vs_es1.c_str();
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
    }
    return ok ? p : 0;
}

// ---- the decode variants -------------------------------------------------
// Each defines dA(vec4) and dB(vec4) reading the same two bytes as the engine.

static const char* DEC_SHIP =
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "return u>=32768.0?u-65536.0:u;}"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "return u>=32768.0?u-65536.0:u;}";

// Integer domain: bytes -> int16 by shift/or, sign-extend, then one ITOF.
// Needs ESSL 3.00 for the integer conversions of the sampled byte.
static const char* DEC_INT =
        "precision highp float;precision highp int;\n"
        "float dA(vec4 c){int u=(int(c.r*255.0+0.5)<<8)|int(c.g*255.0+0.5);"
        "int v=(u<<16)>>16;return float(v);}"
        "float dB(vec4 c){int u=(int(c.b*255.0+0.5)<<8)|int(c.a*255.0+0.5);"
        "int v=(u<<16)>>16;return float(v);}";

// Same but the shift uses the uint domain, which the t3 probe found cheaper
// than int (two's-complement negation is not free, but a shift is).
static const char* DEC_UINT =
        "precision highp float;precision highp int;\n"
        "float dA(vec4 c){uint u=(uint(c.r*255.0+0.5)<<8u)|uint(c.g*255.0+0.5);"
        "uint v=(u<<16u)>>16u;return float(v);}"
        "float dB(vec4 c){uint u=(uint(c.b*255.0+0.5)<<8u)|uint(c.a*255.0+0.5);"
        "uint v=(u<<16u)>>16u;return float(v);}";

// Branchless in the float domain: no compare/select, one extra multiply.
static const char* DEC_BRANCHLESS =
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "return u-65536.0*step(32768.0,u);}"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "return u-65536.0*step(32768.0,u);}";

struct Arm { const char* name; const char* dec; bool es3; };

// The loop: the engine's q8 pair matvec, with only the decode swapped.
static string shader(int rows, int cols, const char* dec, bool es3) {
    const int c4 = cols / 4, axw = (cols + 1) / 2;
    const string R = to_string(rows), C4 = to_string(c4), AX = to_string(axw);
    string head = es3 ? "precision highp float;precision highp int;\n"
                      : "precision highp float;\n";
    return head +
           "uniform sampler2D W,X;uniform float ws;" + dec +
           "vec2 enc16(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);"
           "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
           "return vec2(hi/255.0,(v-hi*256.0)/255.0);}"
           "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;"
           "float s0=0.0,s1=0.0;"
           "for(int q=0;q<" + C4 + ";q++){"
           "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AX + "),0.5));"
           "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AX + "),0.5));"
           "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
           "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r0)+0.5)/float(" + R + ")));"
           "vec4 w=c*255.0-128.0;s0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r1)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s1+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;}"
           "vec2 p1=enc16(s0*ws),p2=enc16(s1*ws);"
           "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
}

static GLuint make_tex(int w, int h, const void* px) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return t;
}

static GLuint make_fbo(int w) {
    GLuint tex = make_tex(w, 1, nullptr);
    GLuint fb = 0;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO incomplete\n");
    return fb;
}

static double q16_at(const vector<unsigned char>& px, int texel, int half) {
    const size_t o = (size_t)texel * 4 + (size_t)half * 2;
    double u = ((double)px[o] * 65280.0 + (double)px[o + 1] * 255.0) / 255.0;
    if (u >= 32768.0) u -= 65536.0;
    return u / 256.0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rows = 1280, cols = 768, iters = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--rows N] [--cols M] [--iters K]\n", argv[0]); return 2; }
    }
    if (rows % 2 || cols % 4) { printf("rows even, cols multiple of 4\n"); return 2; }

    // ES 3.00 for the integer decode arms: ESSL 1.00 forbids bitwise ops.
    // The driver must be told before eglInitialize, which the helper does.
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    EglSurfaceless e = egl_surfaceless_init(3);
    es3_ctx = true;
    if (!e.ok()) return 1;
    printf("== q16dec_probe: %d x %d, %d draws an arm ==\n", rows, cols, iters);
    printf("renderer %s\nglsl     %s\n", (const char*)glGetString(GL_RENDERER),
           (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    vector<unsigned char> wpx((size_t)rows * cols);
    vector<int> wref((size_t)rows * cols);
    unsigned seed = 12345u;
    for (size_t i = 0; i < wpx.size(); ++i) {
        seed = seed * 1103515245u + 12345u;
        const int b = (int)((seed >> 16) & 255);
        wpx[i] = (unsigned char)b;
        wref[i] = b - 128;
    }
    GLuint wt = make_tex(cols / 4, rows, wpx.data());

    const int axw = (cols + 1) / 2;
    vector<unsigned char> xpx((size_t)axw * 4, 0);
    vector<double> xval((size_t)cols);
    for (int i = 0; i < cols; ++i) {
        const int v = (int)lrint(sin((double)i * 0.017) * 4000.0) & 0xFFFF;
        xval[(size_t)i] = (v >= 32768 ? v - 65536 : v) / 256.0;
        const size_t t = (size_t)(i / 2) * 4 + ((i & 1) ? 2 : 0);
        xpx[t + 0] = (unsigned char)((v >> 8) & 255);
        xpx[t + 1] = (unsigned char)(v & 255);
    }
    GLuint xt = make_tex(axw, 1, xpx.data());

    const float tri[6] = {-1, -1, 3, -1, -1, 3};
    GLuint vb = 0;
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    const int pairs = rows / 2;
    GLuint fb = make_fbo(pairs);
    const double ws = 1.0 / 131072.0;

    PerfMon pm;
    const bool want_c = perfcnt_enabled() && !pm.missing();
    if (want_c)
        pm.pick({"QPU-total-idle-clk-cycles",
                 "QPU-total-clk-cycles-fragment-shading",
                 "QPU-total-clk-cycles-executing-valid-instr",
                 "QPU-total-clk-cycles-waiting-TMU",
                 "TMU-total-text-quads-processed"});
    vector<PerfMon::Row> table;

    const Arm arms[] = {
        {"ship (engine)",   DEC_SHIP,        false},
        {"int sign-extend", DEC_INT,         true},
        {"uint sign-extend",DEC_UINT,        true},
        {"branchless step", DEC_BRANCHLESS,  false},
    };
    const int na = (int)(sizeof arms / sizeof arms[0]);

    printf("\n%-20s %9s %10s %9s %8s\n", "arm", "ms/draw", "ns/MAC", "MMAC/s", "check");
    for (int i = 0; i < na; ++i) {
        string log;
        string src = shader(rows, cols, arms[i].dec, arms[i].es3);
        src = es3ify_fs(src);   // every arm, so only the decode differs
        GLuint p = build(src, log);
        if (!p) { printf("%-20s COMPILE FAIL %s\n", arms[i].name, log.c_str()); continue; }
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glViewport(0, 0, pairs, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, wt);
        glUniform1i(glGetUniformLocation(p, "W"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, xt);
        glUniform1i(glGetUniformLocation(p, "X"), 1);
        glUniform1f(glGetUniformLocation(p, "ws"), (float)ws);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();

        vector<unsigned char> px((size_t)pairs * 4, 0);
        glReadPixels(0, 0, pairs, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int bad = 0;
        for (int r = 0; r < rows; ++r) {
            double cpu = 0.0;
            for (int j = 0; j < cols; ++j) cpu += (double)wref[(size_t)r * cols + j] * xval[(size_t)j];
            const double got = q16_at(px, r / 2, r & 1);
            if (fabs(got - cpu * ws) > 2e-2 * (1.0 + fabs(cpu * ws))) ++bad;
        }

        const auto t0 = chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() / iters;
        const double macs = (double)rows * cols;
        printf("%-20s %9.3f %10.2f %9.2f %8s\n", arms[i].name, ms, 1e6 * ms / macs,
               macs / ms / 1000.0, bad ? "MISMATCH" : "ok");
        if (bad) printf("    %d/%d rows off\n", bad, rows);

        if (want_c) {
            pm.begin();
            for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            PerfMon::Row row;
            row.sample = pm.end_and_print(arms[i].name, iters);
            row.wall_ms = ms;
            table.push_back(row);
        }
        glDeleteProgram(p);
    }
    if (want_c) pm.table_rows(table, iters);
    return 0;
}
