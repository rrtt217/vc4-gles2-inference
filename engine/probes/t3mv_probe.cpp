// What does a 2-bit ternary weight decode cost per MAC, next to the q8 one?
//
// This is the pricing probe for a ternary (b1.58) container kind.  The packer
// side is settled -- a per-row `{0, +s, -s}` matrix packs to 2 bits with
// reconstruction error 0 (docs/bitcpm.md) -- so the open question is whether
// reading 16 weights out of one RGBA8 texel and unpacking them is cheaper than
// reading 4 int8 and using them directly.  The body is the engine's q8 matvec
// (`engine/src/shaders.cpp`, matvec_q8): one fragment per output row, a loop over
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
    int rows = 1024, cols = 1024, iters = 20, tile = 0;
    const char* only = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
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
        xval[i] = (float) ((i % 61) - 30);                 // -30 .. 30
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

    printf("\n%-24s %6s %9s %10s %9s %8s\n", "variant", "texels", "ms/draw",
           "ns/MAC", "MB/draw", "check");
    for (const Variant& v : VARIANTS) {
        if (only && !strstr(v.name, only)) continue;
        const int TH = (tile > 0) ? std::min(tile, rows) : rows;
        const int TILES = (rows + TH - 1) / TH;
        const int per = 32 / v.bits;                      // weights a texel
        const int nq = cols / per;                        // weight texels a row
        const int groups = per / 4;                       // activation texels an iteration

        // ---- the weights, and the CPU reference over the same bytes
        vector<unsigned char> wtex((size_t) nq * rows * 4, 0);
        vector<float> wref((size_t) rows * cols, 0.0f);
        unsigned seed = 12345;
        auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
        for (int r = 0; r < rows; r++) {
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
        if (v.kind == Kind::T3T)
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
        if (v.kind == Kind::T3T) {
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
                  "vec4 x=texture2D(X,vec2((float(q)+0.5)/float(" +
                  std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
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
                  "vec4 x=texture2D(X,vec2((float(q)+0.5)/float(" +
                  std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "int j=q-4*(q/4);vec4 w4=vec4(0.0);"
                  "if(j==0)w4=dec(c.r);else if(j==1)w4=dec(c.g);"
                  "else if(j==2)w4=dec(c.b);else w4=dec(c.a);"
                  "s+=dot(w4,x);}"
                  "float v=clamp(floor(s*wsr(r)*ws*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
                  "float g=floor(v/65536.0);float b=floor(v/256.0)-g*256.0;"
                  "float a=v-g*65536.0-b*256.0;"
                  "frag_color=vec4(g/255.0,b/255.0,a/255.0,1.0);}\n";
        }
        if (v.kind != Kind::T3E && v.kind != Kind::T3T) {
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
        if (v.kind == Kind::T3E || v.kind == Kind::T3T) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, wsx_id);
            glUniform1i(glGetUniformLocation(p, "WS"), 2);
            glActiveTexture(GL_TEXTURE0);
            glUniform1f(glGetUniformLocation(p, "ws"), 3.0f);   // the unit
        }
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, rows, 1);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        glDrawArrays(GL_TRIANGLES, 0, 3);                 // warm-up, discarded
        glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;

        // ---- row 0 of the readback against the CPU
        vector<unsigned char> px((size_t) rows * 4, 0);
        glReadPixels(0, 0, rows, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        const double got = ((double) px[0] * 65536.0 + (double) px[1] * 256.0 + px[2] -
                            8388608.0) / 256.0;
        double ref = 0.0;
        for (int i = 0; i < cols; i++) ref += wref[(size_t) i] * xval[i];
        // (the t3e reference is wref as packed: it already carries the row
        // scale, and the shader multiplies the plane back to the same number)
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
        printf("%-24s %6d %9.3f %10.2f %9.2f %8s\n", v.name, nq, ms, 1e6 * ms / (rows * (double) cols),
               (double) nq * rows * 4 / 1e6, good ? "ok" : "MISMATCH");
        if (!good) printf("    row 0: got %.6f want %.6f (err %.6f)\n", got, ref, err);
        if (v.kind == Kind::T3E || v.kind == Kind::T3T) {
            // Every row: the row scale plane and the weight row change with r, so
            // "row 0 is right" says almost nothing (that was the wide kernel's
            // blind spot on the board).
            int bad = 0; double worst = 0.0; int worst_row = -1;
            for (int r = 0; r < rows; ++r) {
                const double g = ((double) px[(size_t) r * 4 + 0] * 65536.0 +
                                  (double) px[(size_t) r * 4 + 1] * 256.0 +
                                  px[(size_t) r * 4 + 2] - 8388608.0) / 256.0;
                double w = 0.0;
                for (int i = 0; i < cols; ++i) w += wref[(size_t) r * cols + i] * xval[i];
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
    return 0;
}
