// What does a 2-bit ternary weight decode cost per MAC, next to the q8 one?
//
// This is the pricing probe for a ternary (b1.58) container kind.  The packer
// side is settled -- a per-row `{0, +s, -s}` matrix packs to 2 bits with
// reconstruction error 0 (docs/bitcpm.md) -- so the open question is whether
// reading 16 weights out of one RGBA8 texel and unpacking them is cheaper than
// reading 4 int8 and using them directly.  The body is the engine's q8 matvec
// (`engine/src/shaders_q8.cpp`, matvec_q8): one fragment per output row, a loop over
// weight texels, four weights against four activations per iteration.  Only the
// packing and the decode differ between variants, so the difference is the
// decode's price and nothing else.
//
// Two decodes exist because ESSL 1.00 has no integer bitwise ops:
//
//   t3f       float: mod()/floor() on the byte -- ESSL 1.00, any driver
//   t3u       uint:  shift/mask on the byte -- ESSL 3.00, the Q4_SHIFT route
//   t3u_fetch the same loop and the same fetches, weights replaced by a
//             constant: the pricing arm that separates the fetch from the
//             unpack (`Q4_NOUNPACK` is the q4 equivalent)
//
// The bit layout is bit0 = magnitude, bit1 = sign, so a weight is
// `mag * (1 - 2*sign)` and neither decode needs a compare or a branch.  A code
// table (0, +1, -1) would cost a `step()` pair per weight on ES2 and an
// equality test on ES3.
//
//   build:  make -C engine probes/t3mv_probe          (native)
//           make -C engine probes/t3mv_probe_pi CROSS_SYSROOT=<sysroot>
//   run:    ./t3mv_probe [--rows N] [--cols M] [--iters K] [--variant NAME]
//
// Each variant checks its row-0 result against a CPU dot product over the same
// bytes first -- a decode that is fast and wrong is the failure mode this
// catches -- then times K draws after one discarded warm-up.  Compare ns per
// MAC: multiply by a token's MAC count for the weights' share of decode.
#include "egl_surfaceless.h"
#include "perfcnt.h"

#include <GLES2/gl2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

// The quiet form of the ES3 request the engine makes in GPU::init_egl(): Mesa
// only offers the ESSL 3.00 compiler when the front end is told so, and VC4
// advertises ES 2.0 only.  Set before eglInitialize, never overriding the user.
static void request_es3() {
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
}

enum class Kind {
    Q8,       // one int8 a weight, 4 a texel (the baseline)
    T3F,      // 2-bit, float decode
    T3U,      // 2-bit, uint decode
    T3Fetch,  // 2-bit layout, fetches kept, unpack removed (pricing arm)
    T3E,      // the engine's own kernel: row-scale plane, q24 store, one fetch
    T3T,      // the same, with the body folded into tiles of --tile rows
    T3P,      // 2 ROWS a texel: bytes b0,b1 = row r cols 8c+0..7, b2,b3 = row r+1
};

struct Variant {
    const char* name;
    Kind kind;
    int bits;     // bits a weight in the texture
    bool es3;     // needs ESSL 3.00
};

static const Variant VARIANTS[] = {
    {"q8 (4 weights/texel)", Kind::Q8, 8, false},
    {"t3f (2-bit, float)", Kind::T3F, 2, false},
    {"t3u (2-bit, uint)", Kind::T3U, 2, true},
    {"t3u_fetch (no unpack)", Kind::T3Fetch, 2, true},
    {"t3e (engine text, row scale)", Kind::T3E, 2, true},
    {"t3t (engine text, tiled)", Kind::T3T, 2, true},
    {"t3p (2 rows/texel)", Kind::T3P, 2, true},
};

static GLuint compile(GLenum t, const string& src, string& log) {
    GLuint h = glCreateShader(t);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[4096];
        glGetShaderInfoLog(h, sizeof b, nullptr, b);
        log = b;
    }
    return ok ? h : 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rows = 1024, cols = 1024, iters = 20, tile = 0, x4 = 0, strip = 0;
    const char* only = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--x4")) x4 = 1;
        else if (!strcmp(argv[i], "--strip") && i + 1 < argc) strip = atoi(argv[++i]);
        else {
            printf("usage: %s [--rows N] [--cols M] [--iters K] [--variant NAME]\n", argv[0]);
            return 2;
        }
    }
    if (cols % 16) { printf("cols must be a multiple of 16 (got %d)\n", cols); return 2; }

    printf("== t3mv_probe: %d rows x %d cols, %d draws a variant ==\n", rows, cols, iters);
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\n", (const char*) glGetString(GL_RENDERER));
    printf("version  %s\n", (const char*) glGetString(GL_VERSION));
    printf("glsl     %s\n", (const char*) glGetString(GL_SHADING_LANGUAGE_VERSION));

    // One activation row: cols values, four a texel, the engine's byte encoding.
    vector<unsigned char> xtex((size_t) cols, 0);
    vector<float> xval((size_t) cols);
    for (int i = 0; i < cols; i++) {
        // Integers, so the byte encoding is exact and the CPU reference is the
        // shader's own input (a half-integer value would truncate in the cast).
        // Small integers, because one arm stores q16: that grid holds +-128 and
        // a 1024-term dot of +-30 rails it at the first row (the engine's own
        // residual sits around 1, so this is also the realistic amplitude).
        xval[i] = (float) ((i % 5) - 2);                   // -2 .. 2
        xtex[(size_t) i] = (unsigned char) ((int) xval[i] + 128);
    }
    GLuint xtex_id = 0;
    glGenTextures(1, &xtex_id);
    glBindTexture(GL_TEXTURE_2D, xtex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, xtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rows, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("fbo incomplete (%dx1 RGBA8)\n", rows);
        return 1;
    }

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    // Opt-in counters (PERFCNT=1); without them the probe prints what it always
    // did and the monitor is never even created.
    PerfMon pm_storage;
    PerfMon* pm = nullptr;
    bool pm_on = false;
    if (perfcnt_enabled() && !pm_storage.missing()) {
        pm = &pm_storage;
        pm_on = pm->pick({"QPU-total-idle-clk-cycles",
                          "QPU-total-clk-cycles-fragment-shading",
                          "QPU-total-clk-cycles-executing-valid-instr",
                          "QPU-total-clk-cycles-waiting-TMU",
                          "QPU-total-clk-cycles-waiting-scoreboard",
                          "TMU-total-text-quads-processed",
                          "TMU-total-text-cache-miss",
                          "L2C-total-cache-hit",
                          "L2C-total-cache-miss"});
    }
    std::vector<PerfMon::Row> rows_out;
    int strip_draws = 1;  // draws per call in the last row, for the table divisor

    printf("\n%-24s %6s %9s %10s %9s %8s\n", "variant", "texels", "ms/draw",
           "ns/MAC", "MB/draw", "check");
    for (const Variant& v : VARIANTS) {
        if (only && !strstr(v.name, only)) continue;
        const int TH = (tile > 0) ? std::min(tile, rows) : rows;
        const int TILES = (rows + TH - 1) / TH;
        const int per = 32 / v.bits;                      // weights a texel
        const int nq = cols / per;                        // weight texels a row
        const int groups = per / 4;                       // activation texels an iteration

        // The pair layout: ONE texel carries 2 rows x 8 columns, so a fragment
        // that writes 2 rows (vec4 = two q16 values, x_st2q) can consume all
        // four bytes of the fetch it makes.  Texel (c, t) holds row 2t cols
        // 8c+0..3 in b0, row 2t cols 8c+4..7 in b1, row 2t+1 the same in b2, b3.
        const bool pair_arm = (v.kind == Kind::T3P);
        const int pnq = pair_arm ? cols / 8 : nq;             // texels a row (pair)
        const int prows = pair_arm ? (rows + 1) / 2 : rows;   // texel rows
        // ---- the weights, and the CPU reference over the same bytes
        vector<unsigned char> wtex((size_t) pnq * prows * 4, 0);
        vector<float> wref((size_t) rows * cols, 0.0f);
        unsigned seed = 12345;
        auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
        if (pair_arm) {
            for (int t = 0; t < prows; t++) {
                for (int c = 0; c < pnq; c++) {
                    for (int ch = 0; ch < 4; ch++) {
                        const int r = 2 * t + (ch >= 2 ? 1 : 0);
                        const int base = 8 * c + (ch & 1) * 4;
                        const float wsrow = 1.0f + (float) (r % 7) * 0.25f;
                        unsigned acc = 0;
                        for (int k = 0; k < 4; k++) {
                            if (r >= rows) break;             // an odd tail row
                            const int code = (int) (rnd() % 3);
                            const float w = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                            wref[(size_t) r * cols + base + k] = w * wsrow;
                            acc |= ((code == 0 ? 0u : 1u) | (code == 2 ? 2u : 0u)) << (2 * k);
                        }
                        wtex[(size_t) (t * pnq + c) * 4 + ch] = (unsigned char) (acc & 0xff);
                    }
                }
            }
        }
        for (int r = 0; pair_arm ? false : (r < rows); r++) {
            const float wsrow = 1.0f + (float) (r % 7) * 0.25f;   // one scale a row
            for (int q = 0; q < nq; q++) {
                for (int ch = 0; ch < 4; ch++) {
                    unsigned acc = 0;
                    if (v.bits == 8) {
                        // q8: the four channels of a texel are four consecutive
                        // weights, and the byte *is* value + 128.
                        const int col = q * 4 + ch;
                        const int code = (int) (rnd() % 3);
                        const float w = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                        wref[(size_t) r * cols + col] = w * wsrow;
                        acc = (unsigned) (int) (w + 128.0f);
                    } else {
                        // 2-bit: four weights a channel, bit0 magnitude, bit1 sign.
                        for (int k = 0; k < 4; k++) {
                            const int col = q * per + ch * 4 + k;
                            const int code = (int) (rnd() % 3);
                            const float w = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                            wref[(size_t) r * cols + col] = w * wsrow;
                            acc |= ((code == 0 ? 0u : 1u) | (code == 2 ? 2u : 0u)) << (2 * k);
                        }
                    }
                    wtex[(size_t) (r * nq + q) * 4 + ch] = (unsigned char) (acc & 0xff);
                }
            }
        }
        // The pair arm stores through q16, which holds +-128: pick the store
        // scale from the reference's own peak so the check can never report the
        // grid instead of the kernel.  Rows are 4096 columns wide at the
        // down-projection shape, where a fixed 1/4 still rails (measured 139.5).
        double pair_div = 1.0;
        if (pair_arm) {
            double amax = 0.0;
            for (int r = 0; r < rows; ++r) {
                double w = 0.0;
                for (int i = 0; i < cols; ++i) w += wref[(size_t) r * cols + i] * xval[i];
                if (std::fabs(w) > amax) amax = std::fabs(w);
            }
            while (amax / pair_div > 120.0) pair_div *= 2.0;
            if (getenv("T3MV_RANGE"))
                printf("    # pair store scale 1/%g (peak %.1f, q16 rails at 128)\n",
                       pair_div, amax);
        }
        GLuint wtex_id = 0;
        std::vector<unsigned char> wtiled;
        if (v.kind == Kind::T3T && TILES > 1) {
            // The engine's upload: row r moves to (x + (r/TH)*W, r%TH).
            wtiled.assign((size_t)nq * TILES * TH * 4, 0);
            for (int r = 0; r < rows; ++r)
                std::memcpy(&wtiled[((size_t)(r % TH) * nq * TILES + (size_t)(r / TH) * nq) * 4],
                            &wtex[(size_t)r * nq * 4], (size_t)nq * 4);
        }
        // The per-row scale plane the engine carries as part 3: two rows a texel,
        // 16 bits each, s_row / unit.  Only the t3e arm reads it, but every arm
        // gets the texture so the binding does not depend on the variant.
        const int np = (rows + 1) / 2;
        std::vector<unsigned char> wsx((size_t)np * 4 * 4, 0);
        for (int r = 0; r < rows; ++r) {
            const float srow = 1.0f + (float) (r % 7) * 0.25f;
            // The plane holds s_row/unit and the kernel multiplies `ws` back;
            // both arms set ws = 3.0 below, so the plane carries s_row/3.
            const long qv = (long) (srow / 3.0f * 65536.0);
            unsigned char* o = &wsx[(size_t)(r / 2) * 16];
            if ((r & 1) == 0) { o[0] = (unsigned char) (qv >> 8); o[1] = (unsigned char) qv; }
            else { o[2] = (unsigned char) (qv >> 8); o[3] = (unsigned char) qv; }
        }
        GLuint wsx_id = 0;
        glGenTextures(1, &wsx_id);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, wsx_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, np, 0, GL_RGBA, GL_UNSIGNED_BYTE, wsx.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glActiveTexture(GL_TEXTURE0);
        glGenTextures(1, &wtex_id);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, wtex_id);
        if (pair_arm)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pnq, prows, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, wtex.data());
        else if (v.kind == Kind::T3T)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nq * TILES, TH, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, (TILES > 1 ? wtiled.data() : wtex.data()));
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nq, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         wtex.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glActiveTexture(GL_TEXTURE0);

        // ---- the shader
        const string ver = v.es3 ? "#version 300 es\n" : "#version 100\n";
        string vs = ver +
                    (v.es3 ? "in vec2 pos;\n" : "attribute vec2 pos;\n") +
                    "void main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver + "precision highp float;\n";
        if (v.es3) fs += "precision highp int;\nout vec4 frag_color;\n";
        fs += "uniform sampler2D W,X;uniform float ws;\n";
        if (v.kind == Kind::T3P) {
            // Two output rows a fragment, one weight fetch an iteration, and all
            // four bytes of that fetch used: b0,b1 are row 2t's two column
            // groups, b2,b3 are row 2t+1's.  16 MACs per iteration for 4 decodes
            // and 3 fetches, against the one-row kernel's 4 MACs for the same 4
            // decodes and 3 fetches.
            const int np = (rows + 1) / 2;
            fs += "uniform sampler2D WS;\n";
            fs += "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));"
                  "return vec4(float(u&1u),float((u>>2)&1u),float((u>>4)&1u),float((u>>6)&1u))"
                  "*(1.0-2.0*vec4(float((u>>1)&1u),float((u>>3)&1u),float((u>>5)&1u),"
                  "float((u>>7)&1u)));}\n";
            fs += "vec2 wsr2(int t){vec4 c=texture2D(WS,vec2(0.125,(float(t)+0.5)/float(" +
                  std::to_string(np) + ")));"
                  "return vec2(c.r*65280.0+c.g*255.0,c.b*65280.0+c.a*255.0)*(1.0/65536.0);}\n";
            fs += "vec2 enc16(float v){float q=clamp(floor(v+0.5),-32768.0,32767.0);"
                  "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
                  "return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n";
            fs += "void main(){int t=int(gl_FragCoord.x);float s0=0.0,s1=0.0;"
                  "for(int c=0;c<" + std::to_string(cols / 8) + ";c++){"
                  "vec4 w=texture2D(W,vec2((float(c)+0.5)/float(" + std::to_string(pnq) + "),"
                  "(float(t)+0.5)/float(" + std::to_string(prows) + ")));"
                  "vec4 xa=texture2D(X,vec2((float(2*c)+0.5)/float(" +
                  std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 xb=texture2D(X,vec2((float(2*c+1)+0.5)/float(" +
                  std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "s0+=dot(dec(w.r),xa)+dot(dec(w.g),xb);"
                  "s1+=dot(dec(w.b),xa)+dot(dec(w.a),xb);}"
                  // enc16 takes raw q16: the value grid is 1/256, so the product
                  // is multiplied in before the round (x_i16enc's shape).
                  "vec2 sc=wsr2(t)*ws;vec2 p0=enc16(s0*sc.x*256.0),p1=enc16(s1*sc.y*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
        }
        else if (v.kind == Kind::T3T) {
            // The engine's own tiled text: tile TH rows, `tiles` of them side by
            // side, one weight fetch an iteration, plus the row-scale plane.
            const int np = (rows + 1) / 2;
            fs += "uniform sampler2D WS;\n";
            fs += "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));"
                  "return vec4(float(u&1u),float((u>>2)&1u),float((u>>4)&1u),float((u>>6)&1u))"
                  "*(1.0-2.0*vec4(float((u>>1)&1u),float((u>>3)&1u),float((u>>5)&1u),"
                  "float((u>>7)&1u)));}\n";
            fs += "float wsr(int r){vec4 c=texture2D(WS,vec2(0.125,(float(r/2)+0.5)/float(" +
                  std::to_string(np) + ")));"
                  "return (mod(float(r),2.0)<0.5?c.r*65280.0+c.g*255.0"
                  ":c.b*65280.0+c.a*255.0)*(1.0/65536.0);}\n";
            fs += "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
                  "for(int q=0;q<" + std::to_string(cols / 4) + ";q++){"
                  "vec4 c=texture2D(W,vec2((float(q/4)+float(int(float(r)/" +
                  std::to_string(TH) + ".0))*float(" + std::to_string(nq) + ")+0.5)/float(" +
                  std::to_string(nq * TILES) + ".0),(mod(float(r)," +
                  std::to_string(TH) + ".0)+0.5)/" + std::to_string(TH) + ".0));"
                  + std::string("vec4 x=") +
                  (x4 ? "vec4(texture2D(X,vec2((float(q*4+0)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+1)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+2)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+3)+0.5)/float(" + std::to_string(cols) + "),0.5)).r)*255.0-128.0;"
                      : "texture2D(X,vec2((float(q)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;") +
                  "int j=q-4*(q/4);vec4 w4=vec4(0.0);"
                  "vec4 d0=dec(c.r),d1=dec(c.g),d2=dec(c.b),d3=dec(c.a);"
                  "w4=d0*step(abs(float(j)),0.5)+d1*step(abs(float(j)-1.0),0.5)+"
                  "d2*step(abs(float(j)-2.0),0.5)+d3*step(abs(float(j)-3.0),0.5);"
                  "s+=dot(w4,x);}"
                  "float v=clamp(floor(s*wsr(r)*ws*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
                  "float g=floor(v/65536.0);float b=floor(v/256.0)-g*256.0;"
                  "float a=v-g*65536.0-b*256.0;"
                  "frag_color=vec4(g/255.0,b/255.0,a/255.0,1.0);}\n";
        }
        else if (v.kind == Kind::T3E) {
            // The engine's kernel, transcribed, so the probe runs the exact text
            // the board runs: `q` walks activation texels, the weight texel is
            // q/4 and its byte is chosen by a compare chain, the per-row plane is
            // Mat::row_scale's part-3 texel, and the store is x_i24encq's.
            const int np = (rows + 1) / 2;
            fs += "uniform sampler2D WS;\n";
            fs += "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));"
                  "return vec4(float(u&1u),float((u>>2)&1u),float((u>>4)&1u),float((u>>6)&1u))"
                  "*(1.0-2.0*vec4(float((u>>1)&1u),float((u>>3)&1u),float((u>>5)&1u),"
                  "float((u>>7)&1u)));}\n";
            fs += "float wsr(int r){vec4 c=texture2D(WS,vec2(0.125,(float(r/2)+0.5)/float(" +
                  std::to_string(np) + ")));"
                  "return (mod(float(r),2.0)<0.5?c.r*65280.0+c.g*255.0"
                  ":c.b*65280.0+c.a*255.0)*(1.0/65536.0);}\n";
            fs += "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
                  "for(int q=0;q<" + std::to_string(cols / 4) + ";q++){"
                  "vec4 c=texture2D(W,vec2((float(q/4)+0.5)/float(" + std::to_string(nq) + "),"
                  "(float(r)+0.5)/float(" + std::to_string(rows) + ")));"
                  + std::string("vec4 x=") +
                  (x4 ? "vec4(texture2D(X,vec2((float(q*4+0)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+1)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+2)+0.5)/float(" + std::to_string(cols) + "),0.5)).r,"
                        "texture2D(X,vec2((float(q*4+3)+0.5)/float(" + std::to_string(cols) + "),0.5)).r)*255.0-128.0;"
                      : "texture2D(X,vec2((float(q)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;") +
                  "int j=q-4*(q/4);vec4 w4=vec4(0.0);"
                  "if(j==0)w4=dec(c.r);else if(j==1)w4=dec(c.g);"
                  "else if(j==2)w4=dec(c.b);else w4=dec(c.a);"
                  "s+=dot(w4,x);}"
                  "float v=clamp(floor(s*wsr(r)*ws*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
                  "float g=floor(v/65536.0);float b=floor(v/256.0)-g*256.0;"
                  "float a=v-g*65536.0-b*256.0;"
                  "frag_color=vec4(g/255.0,b/255.0,a/255.0,1.0);}\n";
        }
        if (v.kind != Kind::T3E && v.kind != Kind::T3T && v.kind != Kind::T3P) {
        fs += "void main(){int r=int(gl_FragCoord.x);float s=0.0;\n";
        fs += "for(int q=0;q<" + std::to_string(nq) + ";q++){\n";
        fs += "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + std::to_string(nq) +
              "),(float(r)+0.5)/float(" + std::to_string(rows) + ")));\n";
        for (int g = 0; g < groups; g++)
            // ESSL 1.00 has no int->float conversion, so every literal here is
            // written with a decimal point ("float(q)*1" is a compile error).
            fs += "vec4 x" + std::to_string(g) + "=texture2D(X,vec2((float(q)*" +
                  std::to_string(groups) + ".0+" + std::to_string(g) + ".0+0.5)/float(" +
                  std::to_string(cols / 4) + "),0.5))*255.0-128.0;\n";
        if (v.kind == Kind::Q8) {
            // The engine's own q8 body, verbatim: a vec4 of int8 a texel.
            fs += "vec4 w=c*255.0-128.0;\ns+=dot(w,x0);\n";
        }
        for (int ch = 0; ch < (v.kind == Kind::Q8 || v.kind == Kind::T3E ? 0 : 4); ch++) {
            const char* cn = ch == 0 ? "c.r" : ch == 1 ? "c.g" : ch == 2 ? "c.b" : "c.a";
            const string b = "b" + std::to_string(ch);
            const string xg = "x" + std::to_string(ch);
            fs += "float " + b + "=floor(" + cn + "*255.0+0.5);\n";
            if (v.kind == Kind::T3Fetch) {
                // Same fetches, no unpack: what the memory side alone costs.
                fs += "s+=dot(vec4(1.0,0.0,-1.0,0.0)," + xg + ")*(" + b + "*0.0+1.0);\n";
            } else if (v.kind == Kind::T3F) {
                const string m = "m" + std::to_string(ch), g = "g" + std::to_string(ch);
                fs += "vec4 " + m + "=vec4(mod(" + b + ",2.0),mod(floor(" + b + "*0.25),2.0),"
                      "mod(floor(" + b + "*0.0625),2.0),mod(floor(" + b + "*0.015625),2.0));\n";
                fs += "vec4 " + g + "=vec4(mod(floor(" + b + "*0.5),2.0),mod(floor(" + b +
                      "*0.125),2.0),mod(floor(" + b + "*0.03125),2.0),mod(floor(" + b +
                      "*0.0078125),2.0));\n";
                fs += "s+=dot(" + m + "*(1.0-2.0*" + g + ")," + xg + ");\n";
            } else {
                const string u = "u" + std::to_string(ch), m = "m" + std::to_string(ch),
                             g = "g" + std::to_string(ch);
                fs += "uint " + u + "=uint(" + b + ");\n";
                fs += "vec4 " + m + "=vec4(float(" + u + "&1u),float((" + u + ">>2)&1u),"
                      "float((" + u + ">>4)&1u),float((" + u + ">>6)&1u));\n";
                fs += "vec4 " + g + "=vec4(float((" + u + ">>1)&1u),float((" + u + ">>3)&1u),"
                      "float((" + u + ">>5)&1u),float((" + u + ">>7)&1u));\n";
                fs += "s+=dot(" + m + "*(1.0-2.0*" + g + ")," + xg + ");\n";
            }
        }
        fs += "}\n";
        fs += "float v=clamp(floor(s*ws*256.0+0.5),-8388608.0,8388607.0)+8388608.0;\n";
        fs += "float gg=floor(v/65536.0);float bb=floor(v/256.0)-gg*256.0;"
              "float aa=v-gg*65536.0-bb*256.0;\n";
        fs += v.es3 ? "frag_color=vec4(gg/255.0,bb/255.0,aa/255.0,1.0);}\n"
                    : "gl_FragColor=vec4(gg/255.0,bb/255.0,aa/255.0,1.0);}\n";
        }

        if (getenv("T3MV_DUMP")) {
            printf("---- %s\n%s\n", v.name, fs.c_str());
        }
        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) {
            printf("%-24s COMPILE FAIL  %s\n", v.name, log.c_str());
            if (hvs) glDeleteShader(hvs);
            if (hfs) glDeleteShader(hfs);
            glDeleteTextures(1, &wtex_id);
            continue;
        }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs);
        glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char b[4096];
            glGetProgramInfoLog(p, sizeof b, nullptr, b);
            printf("%-24s LINK FAIL     %s\n", v.name, b);
            glDeleteProgram(p);
            glDeleteShader(hvs);
            glDeleteShader(hfs);
            glDeleteTextures(1, &wtex_id);
            continue;
        }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, xtex_id);
        glUniform1i(glGetUniformLocation(p, "X"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, wtex_id);
        glUniform1i(glGetUniformLocation(p, "W"), 1);
        glUniform1f(glGetUniformLocation(p, "ws"), 1.0f);
        if (v.kind == Kind::T3E || v.kind == Kind::T3T || v.kind == Kind::T3P) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, wsx_id);
            glUniform1i(glGetUniformLocation(p, "WS"), 2);
            glActiveTexture(GL_TEXTURE0);
            // The unit multiplies the plane back.  The pair arm stores through
            // q16, which holds +-128, so it uses a quarter of the scale and its
            // reference is a quarter too -- otherwise a 1024-term dot rails the
            // grid at the first row and the check reports the format, not the
            // kernel.
            glUniform1f(glGetUniformLocation(p, "ws"), pair_arm ? (float) (3.0 / pair_div) : 3.0f);
        }
        glActiveTexture(GL_TEXTURE0);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        // --strip N draws in N-row viewport chunks, the shape the engine's
        // kDrawCap forces on the board (a 1280- or 1536-row target is two
        // draws).  gl_FragCoord.x stays absolute, so the arithmetic is the
        // same either way -- if the numbers change, the cause is the driver.
        const int frags = pair_arm ? prows : rows;
        auto draw_all = [&]() {
            if (strip <= 0) {
                glViewport(0, 0, frags, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                return;
            }
            for (int x0 = 0; x0 < frags; x0 += strip) {
                const int w = (frags - x0 < strip) ? frags - x0 : strip;
                glViewport(x0, 0, w, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
        };
        draw_all();                                        // warm-up, discarded
        glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) draw_all();
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        // PERFCNT=1: the same iters draws again under the monitor.  t3's story
        // is "the 2-bit decode is cheaper than a q8 fetch", which is a claim
        // about ALU against TMU; the two counters that separate them are
        // executing-valid-instr and text-quads-processed.
        if (pm && pm_on) {
            pm->begin();
            for (int k = 0; k < iters; k++) draw_all();
            glFinish();
            // --strip N splits one draw into ceil(frags/N) draws, so the
            // region's draw count and the per-draw wall time both have to be
            // divided by that factor or the columns are off by it.
            const int per_call = strip > 0 ? (frags + strip - 1) / strip : 1;
            PerfMon::Row row;
            row.sample = pm->end_and_print(v.name, iters * per_call);
            row.wall_ms = ms / per_call;
            rows_out.push_back(row);
            strip_draws = per_call;
        }

        // ---- the readback against the CPU
        //
        // The single-row arms store a 24-bit value in one texel (g,b,a); the pair
        // arm stores two q16 values in one texel (hi0,lo0,hi1,lo1), the engine's
        // x_st2q layout, which is what matvec_*_add_pair already writes.
        vector<unsigned char> px((size_t) frags * 4, 0);
        glReadPixels(0, 0, frags, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        // q16 is offset-free with a negative rail at 0x8000: u is hi*256+lo and
        // a u at or above 32768 is a negative value (x_i16raw's rule, which the
        // CPU reference needs here or every negative row reads 256 too high).
        auto q16_at = [&](int t, int half) {
            const size_t o = (size_t) t * 4 + (size_t) half * 2;
            double u = ((double) px[o] * 65280.0 + (double) px[o + 1] * 255.0) / 255.0;
            if (u >= 32768.0) u -= 65536.0;
            return u / 256.0;
        };
        auto q24_at = [&](int r) {
            return ((double) px[(size_t) r * 4] * 65536.0 + (double) px[(size_t) r * 4 + 1] * 256.0 +
                    px[(size_t) r * 4 + 2] - 8388608.0) / 256.0;
        };
        auto dot_row = [&](int r) {
            double w = 0.0;
            for (int i = 0; i < cols; ++i) w += wref[(size_t) r * cols + i] * xval[i];
            return w;
        };
        if (getenv("T3MV_RANGE")) {
            double mx = 0.0;
            for (int r = 0; r < rows; ++r) {
                const double w = dot_row(r) / pair_div;
                if (std::fabs(w) > mx) mx = std::fabs(w);
            }
            printf("    # max |reference| = %.1f (q16 rails at 128)\n", mx);
        }
        const double got = pair_arm ? q16_at(0, 0) : q24_at(0);
        double ref = dot_row(0) / pair_div;
        if (v.kind == Kind::T3Fetch) {
            // No unpack: the reference is the shader's own constant pattern.
            ref = 0.0;
            for (int i = 0; i < cols; i++) {
                const int k = i & 3;
                ref += (k == 0 ? 1.0 : k == 2 ? -1.0 : 0.0) * xval[i];
            }
        }
        // Relative: VC4's UNORM8 -> float conversion is (b+0.5)/256 rather than
        // b/255 (the q8 variant decodes `c*255.0-128.0` and lands 0.04% low
        // there, while both 2-bit decoders add 0.5 and recover the byte
        // exactly), so an absolute tolerance flags a correct kernel on the board.
        const double err = std::fabs(got - ref);
        const bool good = err < 2e-3 * (1.0 + std::fabs(ref));
        // The MAC count a draw really does: a single-row fragment covers |rows|
        // outputs, a pair fragment covers two (and the arm still walks the same
        // columns), so the pair arm does twice the MACs in half the fragments.
        const double macs = (pair_arm ? 2.0 : 1.0) * (double) rows * (double) cols;
        printf("%-24s %6d %9.3f %10.2f %9.2f %8s\n", v.name, nq, ms, 1e6 * ms / macs,
               (double) pnq * prows * 4 / 1e6, good ? "ok" : "MISMATCH");
        if (!good) printf("    row 0: got %.6f want %.6f (err %.6f)\n", got, ref, err);
        if (v.kind == Kind::T3E || v.kind == Kind::T3T || pair_arm) {
            // Every row: the row scale plane and the weight row change with r, so
            // "row 0 is right" says almost nothing (that was the wide kernel's
            // blind spot on the board).
            int bad = 0; double worst = 0.0; int worst_row = -1;
            for (int r = 0; r < rows; ++r) {
                const double g = pair_arm ? q16_at(r / 2, r & 1) : q24_at(r);
                const double w = dot_row(r) / pair_div;
                const double e = std::fabs(g - w) / (1.0 + std::fabs(w));
                if (e > 2e-3) ++bad;
                if (e > worst) { worst = e; worst_row = r; }
            }
            printf("    every row: %d of %d rows off (worst rel %.3g at row %d)\n",
                   bad, rows, worst, worst_row);
        }
        fflush(stdout);
        glDeleteProgram(p);
        glDeleteShader(hvs);
        glDeleteShader(hfs);
        glDeleteTextures(1, &wtex_id);
    }
    if (pm_on) pm->table_rows(rows_out, iters * strip_draws);

    return 0;
}
