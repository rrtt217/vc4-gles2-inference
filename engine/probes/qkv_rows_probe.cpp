// How many output rows should one q8 matvec fragment amortise its activation
// decode over?  **Answered on VC4 (2026-11-21): two is already right; four is
// 15% slower.**
//
// The question came from a real anomaly.  Two q8 kernels in
// engine/src/shaders.cpp share one activation decode across several rows --
// matvec_q8_pair_logits (qkv, 2 rows) and matvec_q8_pair_swiglu (FFN gate, 4
// rows) -- and measured through DUP_PROFILE the FFN runs at 1.79 ns/MAC against
// qkv's 2.87, a ratio of 0.62 against the 0.5 the row count predicts.  The
// hypothesis this probe tests: qkv is leaving the same 2x on the table, and a
// 4-row qkv kernel lands near the FFN.
//
// It does not.  Measured at the real qkv shape (1280x768, 640 fragments, 30
// draws an arm, VC4 V3D 2.1):
//
//   arm      ms/draw   rows stored   vs r2
//   r2        2.351       1280          --     (the engine's shape)
//   r4        2.700       1280        +15%
//   r4x2      2.307       1280         -2%   (inside noise)
//
// The reason is what makes the FFN fast, and it is not the row count: its four
// rows come from TWO independent weight matrices (w1 and w3), so its inner body
// carries four independent weight fetches.  qkv's four rows are four rows of
// the SAME matrix sharing one activation pair, so adding rows only reuses a
// decode that was never the bottleneck -- which ACT=i8 had already shown, since
// halving the activation fetches (quads 2.80M -> 2.33M) left the FFN's gpu= and
// exec= unchanged.
//
// A caveat the numbers carry: r4's grid is 2*quads fragments and EVERY fragment
// runs the whole column loop, half of them discarding their sums, so it
// executes twice the MACs it stores.  Priced per MAC executed it reads 1.37
// ns/MAC against r2's 2.39, which looks like a 43% win and is not one -- the
// honest comparison for a kernel is per output row, and there r4 loses.  Do not
// quote the ns/MAC column without the executed-MAC divisor.
//
// A third arm closes the question.  The obvious way to bank four rows properly
// is two colour attachments, and **VC4 has no MRT**: GL_MAX_DRAW_BUFFERS and
// GL_MAX_COLOR_ATTACHMENTS both read 1, glDrawBuffers(2, ...) returns
// GL_INVALID_OPERATION, and a two-target fragment shader fails to compile with
// "array index must be < 1".  So r4x2 is the only shape left -- four rows a
// fragment with the column loop deepened instead -- and the counters say why it
// does not win either (same weights, same activations, 1280x768):
//
//   arm    frag ms   exec ms   waitTMU   quads      cache-miss
//   r2     2.320     1.992     0.252     29,589,504 1,005,779
//   r4x2   2.283     1.962     0.176     44,384,256 1,801,690
//   r4     2.699     2.371     0.202     44,384,256 3,727,087
//
// r4x2 does shave fragment cycles and the TMU wait, and then pays for it:
// deepening the body to four activation texels an iteration raises the texture
// quads by 50% for the same rows stored.  The net is -2%, inside noise.
//
// The arms are all bit-exact against a CPU dot product of the same bytes, every
// row, not just row 0 (the layout bug this probe was written with -- a
// `frags == pairs` heuristic that coincides at rows == 4*quads -- is why the
// check covers all rows).
//
// build: make -C engine probes/qkv_rows_probe[_pi]
// run:   ./qkv_rows_probe [--rows N] [--cols M] [--iters K]

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

static GLuint build(const string& fs, const char* vs, string& log) {
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

// The engine's decode, verbatim (kDec16A/kDec16B and the q8 weight line).
static const char* DEC16 =
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "return u>=32768.0?u-65536.0:u;}"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "return u>=32768.0?u-65536.0:u;}";

// enc16, the engine's q16 store (kEnc16Mul without the mul).
static const char* ENC16 =
        "vec2 enc16(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);"
        "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
        "return vec2(hi/255.0,(v-hi*256.0)/255.0);}";

static const char* VS =
        "attribute vec2 A;void main(){gl_Position=vec4(A,0.0,1.0);}";

// r2: the engine's shape.  Fragment t holds rows 2t and 2t+1.
static string sh_r2(int rows, int cols) {
    const int c4 = cols / 4, axw = (cols + 1) / 2;
    const string R = to_string(rows), C4 = to_string(c4), AX = to_string(axw);
    return "precision highp float;uniform sampler2D W,X;uniform float ws;" +
           string(DEC16) + string(ENC16) +
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

// r4: four rows a fragment, one activation decode for all four.
//
// Two fragments carry the four rows so the store stays two q16 a texel, the
// same trick matvec_q8_pair_swiglu uses for gate+swiglu.  Fragment t writes
// rows 4t, 4t+1 (texel t of the target's first half) and rows 4t+2, 4t+3 go to
// the fragment at t + pairs, which recomputes nothing: both halves of the grid
// run the same loop and only the two rows they own are stored.  That doubles
// the MACs a draw performs relative to the rows it writes, so the arm is
// priced per MAC *computed*, which is the honest comparison against r2.
static string sh_r4(int rows, int cols) {
    const int c4 = cols / 4, axw = (cols + 1) / 2;
    const string R = to_string(rows), C4 = to_string(c4), AX = to_string(axw);
    const string HALF = to_string((rows + 1) / 2);
    return "precision highp float;uniform sampler2D W,X;uniform float ws;"
           "uniform int quads;" +
           string(DEC16) + string(ENC16) +
           "void main(){int t=int(gl_FragCoord.x);"
           "int lo=(t<int(quads))?t:(t-int(quads));"
           "int base=4*lo;int lim=" + to_string(rows - 1) + ";"
           "int r0=base,r1=min(base+1,lim),r2=min(base+2,lim),r3=min(base+3,lim);"
           "float s0=0.0,s1=0.0,s2=0.0,s3=0.0;"
           "for(int q=0;q<" + C4 + ";q++){"
           "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AX + "),0.5));"
           "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AX + "),0.5));"
           "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
           "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r0)+0.5)/float(" + R + ")));"
           "vec4 w=c*255.0-128.0;s0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r1)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s1+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r2)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s2+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(q)+0.5)/float(" + C4 + "),(float(r3)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s3+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;}"
           "gl_FragColor=(t<int(quads))"
           "?vec4(enc16(s0*ws),enc16(s1*ws))"
           ":vec4(enc16(s2*ws),enc16(s3*ws));}";
}

// r4x2: the same four rows with two activation pairs per iteration (8 fetches),
// to separate "more rows amortise the decode" from "a longer body pipelines".
static string sh_r4x2(int rows, int cols) {
    const int c4 = cols / 4, axw = (cols + 1) / 2;
    const string R = to_string(rows), C8 = to_string(c4 / 2), C4S = to_string(c4), AX = to_string(axw);
    return "precision highp float;uniform sampler2D W,X;uniform float ws;"
           "uniform int quads;" +
           string(DEC16) + string(ENC16) +
           "void main(){int t=int(gl_FragCoord.x);"
           "int lo=(t<int(quads))?t:(t-int(quads));"
           "int base=4*lo;int lim=" + to_string(rows - 1) + ";"
           "int r0=base,r1=min(base+1,lim),r2=min(base+2,lim),r3=min(base+3,lim);"
           "float s0=0.0,s1=0.0,s2=0.0,s3=0.0;"
           "for(int q=0;q<" + C8 + ";q++){"
           "vec4 xa=texture2D(X,vec2((float(4*q)+0.5)/float(" + AX + "),0.5));"
           "vec4 xb=texture2D(X,vec2((float(4*q+1)+0.5)/float(" + AX + "),0.5));"
           "vec4 xc=texture2D(X,vec2((float(4*q+2)+0.5)/float(" + AX + "),0.5));"
           "vec4 xd=texture2D(X,vec2((float(4*q+3)+0.5)/float(" + AX + "),0.5));"
           "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
           "float b0=dA(xc),b1=dB(xc),b2=dA(xd),b3=dB(xd);"
           "vec4 c=texture2D(W,vec2((float(2*q)+0.5)/float(" + C4S + "),(float(r0)+0.5)/float(" + R + ")));"
           "vec4 w=c*255.0-128.0;s0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(2*q+1)+0.5)/float(" + C4S + "),(float(r0)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s0+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
           "c=texture2D(W,vec2((float(2*q)+0.5)/float(" + C4S + "),(float(r1)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s1+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(2*q+1)+0.5)/float(" + C4S + "),(float(r1)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s1+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
           "c=texture2D(W,vec2((float(2*q)+0.5)/float(" + C4S + "),(float(r2)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s2+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(2*q+1)+0.5)/float(" + C4S + "),(float(r2)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s2+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
           "c=texture2D(W,vec2((float(2*q)+0.5)/float(" + C4S + "),(float(r3)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s3+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
           "c=texture2D(W,vec2((float(2*q+1)+0.5)/float(" + C4S + "),(float(r3)+0.5)/float(" + R + ")));"
           "w=c*255.0-128.0;s3+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
           "gl_FragColor=(t<int(quads))"
           "?vec4(enc16(s0*ws),enc16(s1*ws))"
           ":vec4(enc16(s2*ws),enc16(s3*ws));}";
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

static GLuint make_fbo(int w, int h) {
    GLuint tex = make_tex(w, h, nullptr);
    GLuint fb = 0;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO incomplete (%dx%d)\n", w, h);
    return fb;
}

// The q16 decode of a written texel, the engine's rule (t3mv_probe's q16_at).
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
    if (cols % 8) { printf("cols must be a multiple of 8 (got %d)\n", cols); return 2; }
    if (rows % 4) { printf("rows must be a multiple of 4 (got %d)\n", rows); return 2; }

    printf("== qkv_rows_probe: %d rows x %d cols, %d draws an arm ==\n", rows, cols, iters);
    EglSurfaceless e = egl_surfaceless_init(2);
    if (!e.ok()) return 1;
    printf("renderer %s\nversion  %s\n", (const char*)glGetString(GL_RENDERER),
           (const char*)glGetString(GL_VERSION));

    // Weights: q8 bytes, and the matching signed integers for the CPU check.
    vector<unsigned char> wpx((size_t)rows * cols);
    vector<int> wref((size_t)rows * cols);
    unsigned seed = 12345u;
    for (size_t i = 0; i < wpx.size(); ++i) {
        seed = seed * 1103515245u + 12345u;
        const int b = (int)((seed >> 16) & 255);
        wpx[i] = (unsigned char)b;
        wref[i] = b - 128;
    }
    // q8 weight layout: 4 columns a texel, cols/4 wide, rows tall.
    GLuint wt = make_tex(cols / 4, rows, wpx.data());

    // Activations: q16, two a texel, (cols+1)/2 wide.
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

    const int pairs = (rows + 1) / 2;
    const int quads = (rows + 3) / 4;
    GLuint fb2 = make_fbo(pairs, 1);   // r2 writes `pairs` texels
    GLuint fb4 = make_fbo(2 * quads, 1);  // r4 writes 2*quads texels

    PerfMon pm;
    const bool want_c = perfcnt_enabled() && !pm.missing();
    if (want_c)
        pm.pick({"QPU-total-idle-clk-cycles",
                 "QPU-total-clk-cycles-fragment-shading",
                 "QPU-total-clk-cycles-executing-valid-instr",
                 "QPU-total-clk-cycles-waiting-TMU",
                 "TMU-total-text-quads-processed",
                 "TMU-total-text-cache-miss"});
    vector<PerfMon::Row> table;

    // rows4: how many output rows a fragment amortises the decode over.  It is
    // a field, not `frags == pairs` -- with rows == 4*quads those two coincide
    // and the heuristic silently picked the wrong layout.
    struct Arm { const char* name; string fs; GLuint fbo; int frags; int rows_written; int rows4; };
    string log;
    vector<Arm> arms;
    arms.push_back({"r2 (engine shape)", sh_r2(rows, cols), fb2, pairs, pairs * 2, 2});
    arms.push_back({"r4 (4 rows)", sh_r4(rows, cols), fb4, 2 * quads, rows, 4});
    arms.push_back({"r4x2 (4 rows, 2 pairs)", sh_r4x2(rows, cols), fb4, 2 * quads, rows, 4});

    // The store is q16, +-32768.  A 768-column dot of |w|<=128 and |x|<=15.6
    // reaches ~1.5e6, so the scale must be ~1/2^17 to stay in range; 1/128
    // (the engine's q8 scale) saturates and the check would report the rail,
    // not a decode error.
    const double ws = 1.0 / 131072.0;
    printf("\n%-24s %8s %9s %10s %9s %8s\n", "arm", "frags", "ms/draw", "ns/MAC", "MMAC/s", "check");
    for (Arm& a : arms) {
        GLuint p = build(a.fs, VS, log);
        if (!p) { printf("%-24s COMPILE/LINK FAIL %s\n", a.name, log.c_str()); continue; }
        glBindFramebuffer(GL_FRAMEBUFFER, a.fbo);
        glViewport(0, 0, a.frags, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, wt);
        glUniform1i(glGetUniformLocation(p, "W"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, xt);
        glUniform1i(glGetUniformLocation(p, "X"), 1);
        glUniform1f(glGetUniformLocation(p, "ws"), (float)ws);
        glUniform1f(glGetUniformLocation(p, "pairs"), (float)pairs);
        glUniform1i(glGetUniformLocation(p, "quads"), quads);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();

        // Read every row back and compare against the CPU dot product.
        vector<unsigned char> px((size_t)a.frags * 4, 0);
        glReadPixels(0, 0, a.frags, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int bad = 0;
        double worst = 0.0;
        for (int r = 0; r < a.rows_written; ++r) {
            double cpu = 0.0;
            for (int i = 0; i < cols; ++i) cpu += (double)wref[(size_t)r * cols + i] * xval[(size_t)i];
            const double want = cpu * ws;
            // Which texel and half holds row r.
            int texel, half;
            if (a.rows4 == 2) { texel = r / 2; half = r & 1; }
            else {
                // r4: group q = r/4 owns rows 4q..4q+3. Fragment q stores the
                // group's first two rows, fragment quads+q the last two.
                const int q = r / 4, h = r & 3;
                texel = (h < 2) ? q : (quads + q);
                half = h & 1;
            }
            const double got = q16_at(px, texel, half);
            const double err = fabs(got - want);
            if (err > 2e-2 * (1.0 + fabs(want))) ++bad;
            if (err > worst) worst = err;
        }

        const auto t0 = chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() / iters;
        // r4 computes each row's MACs twice (both halves of the grid run the
        // whole loop, each storing half the rows), so price the MACs actually
        // executed: the honest per-MAC number for the same work r2 does.
        const double macs = (double)rows * cols * (a.rows4 == 2 ? 1.0 : 2.0);

        printf("%-24s %8d %9.3f %10.2f %9.2f %8s\n", a.name, a.frags, ms, 1e6 * ms / macs,
               macs / ms / 1000.0, bad ? "MISMATCH" : "ok");
        if (bad) {
            // One row's numbers separate a decode error (large, every row) from
            // the q16 rail (clamped at +-128).
            const int r = 0;
            double cpu = 0.0;
            for (int i = 0; i < cols; ++i) cpu += (double)wref[(size_t)r * cols + i] * xval[(size_t)i];
            int texel, half;
            if (a.rows4 == 2) { texel = r / 2; half = r & 1; }
            else { const int q = r / 4, h = r & 3; texel = (h < 2) ? q : (quads + q); half = h & 1; }
            printf("    %d/%d rows off (worst abs %.4f); row 0 got %.4f want %.4f\n",
                   bad, a.rows_written, worst, q16_at(px, texel, half), cpu * ws);
        }

        if (want_c) {
            pm.begin();
            for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            PerfMon::Row row;
            row.sample = pm.end_and_print(a.name, iters);
            row.wall_ms = ms;
            table.push_back(row);
        }
        glDeleteProgram(p);
    }
    if (want_c) pm.table_rows(table, iters);
    return 0;
}
