// What does the T3 weight DECODE cost, isolated from everything else?
//
// t3mv_probe priced whole kernels; this one holds the layout, the fetches, the
// accumulation order and the store FIXED -- the exact shipped matvec_t3 shape
// (column-major 2-row texel, one fragment a row, bb=mix(c.ba,c.rg,1-h),
// s+=dot(dec(bb.x),xa)+dot(dec(bb.y),xb)) -- and varies ONLY dec(byte)->vec4.
// Every variant decodes to the same {0,+1,-1} vec4 and accumulates in the same
// order, so all of them are byte-identical to each other; the ns/MAC difference
// is the decode's ALU price and nothing else.
//
// The shipped t3_byte writes uint(floor(bb*255.0+0.5)) EIGHT times (4 magnitude
// + 4 sign extracts) and relies on the compiler to CSE it.  The first question
// is whether VC4 actually does: `ship` is that text verbatim, `hoist` computes
// the uint once.  The rest are cheaper per-weight algebras:
//
//   ship     the shipped text, 8 redundant byte->uint conversions
//   hoist    uint u once, then shift/mask (the t3mv_probe dec)
//   code     code=(u>>2k)&3; w=(code&1)*(1-(code&2))   [2-bit code algebra]
//   lut      vec4(0,1,0,-1) indexed by the 2-bit code  [dynamic component index]
//   signbit  build +-1.0f by bit-cast (sign<<31 | 0x3F800000), times magnitude
//   wide     16-wide single row: all four bytes, 16 MACs an iteration (T3_WIDE)
//
//   build:  make -C engine probes/t3dec_probe[_pi]
//   run:    ./t3dec_probe [--rows N] [--cols M] [--iters K] [--variant NAME]
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

static void request_es3() {
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
}

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

// The shipped t3_byte, verbatim (shaders_t3.cpp): eight byte->uint conversions.
static string dec_ship() {
    return "vec4 dec(float bb){return vec4("
           "float(uint(floor(bb*255.0+0.5))&1u),"
           "float((uint(floor(bb*255.0+0.5))>>2)&1u),"
           "float((uint(floor(bb*255.0+0.5))>>4)&1u),"
           "float((uint(floor(bb*255.0+0.5))>>6)&1u))"
           "*(1.0-2.0*vec4("
           "float((uint(floor(bb*255.0+0.5))>>1)&1u),"
           "float((uint(floor(bb*255.0+0.5))>>3)&1u),"
           "float((uint(floor(bb*255.0+0.5))>>5)&1u),"
           "float((uint(floor(bb*255.0+0.5))>>7)&1u)));}\n";
}
static string dec_hoist() {
    return "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));"
           "return vec4(float(u&1u),float((u>>2)&1u),float((u>>4)&1u),float((u>>6)&1u))"
           "*(1.0-2.0*vec4(float((u>>1)&1u),float((u>>3)&1u),float((u>>5)&1u),"
           "float((u>>7)&1u)));}\n";
}
static string dec_code() {
    return "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));vec4 w;"
           "for(int k=0;k<4;k++){uint c=(u>>uint(2*k))&3u;"
           "w[k]=float(c&1u)*(1.0-float(c&2u));}return w;}\n";
}
static string dec_lut() {
    return "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));"
           "vec4 T=vec4(0.0,1.0,0.0,-1.0);vec4 w;"
           "for(int k=0;k<4;k++){uint c=(u>>uint(2*k))&3u;w[k]=T[int(c)];}return w;}\n";
}
static string dec_signbit() {
    return "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));vec4 w;"
           "for(int k=0;k<4;k++){uint m=(u>>uint(2*k))&1u;"
           "uint s=(u>>uint(2*k+1))&1u;"
           "w[k]=uintBitsToFloat(0x3F800000u|(s<<31))*float(m);}return w;}\n";
}
// Unrolled signbit (no dynamic component write), and a no-floor form: uint()
// truncates toward zero and bb*255+0.5 is positive, so it equals uint(floor()).
static string dec_sb_unroll() {
    return "vec4 dec(float bb){uint u=uint(floor(bb*255.0+0.5));return vec4("
           "uintBitsToFloat(0x3F800000u|(((u>>1)&1u)<<31))*float(u&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>3)&1u)<<31))*float((u>>2)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>5)&1u)<<31))*float((u>>4)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>7)&1u)<<31))*float((u>>6)&1u));}\n";
}
static string dec_sb_trunc() {
    return "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
           "uintBitsToFloat(0x3F800000u|(((u>>1)&1u)<<31))*float(u&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>3)&1u)<<31))*float((u>>2)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>5)&1u)<<31))*float((u>>4)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>7)&1u)<<31))*float((u>>6)&1u));}\n";
}
// The sign's &1 is redundant: (u>>(2k+1))<<31 shifts every higher bit out of
// the uint32, so bit31 is already the sign bit alone.  One op less a weight.
static string dec_sb_noand() {
    return "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
           "uintBitsToFloat(0x3F800000u|((u>>1)<<31))*float(u&1u),"
           "uintBitsToFloat(0x3F800000u|((u>>3)<<31))*float((u>>2)&1u),"
           "uintBitsToFloat(0x3F800000u|((u>>5)<<31))*float((u>>4)&1u),"
           "uintBitsToFloat(0x3F800000u|((u>>7)<<31))*float((u>>6)&1u));}\n";
}
// Zero by masking the float bits instead of multiplying by the magnitude:
// (0x3F800000|sign<<31) & (0-mag).  Trades the ITOF(mag)+fmul for a sub+and
// (both add slot); same {0,+-1} out.
static string dec_sb_mask() {
    return "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
           "uintBitsToFloat((0x3F800000u|((u>>1)<<31))&(0u-(u&1u))),"
           "uintBitsToFloat((0x3F800000u|((u>>3)<<31))&(0u-((u>>2)&1u))),"
           "uintBitsToFloat((0x3F800000u|((u>>5)<<31))&(0u-((u>>4)&1u))),"
           "uintBitsToFloat((0x3F800000u|((u>>7)<<31))&(0u-((u>>6)&1u))));}\n";
}
// The unused code 2 (mag0 sign1) is a DON'T-CARE: assign it -2 and the weight
// becomes bit0 - 2*bit1 -- an integer subtract + one ITOF, no bit-cast and no
// float multiply (frees the mul slot for the dot's FMAD).  Valid codes
// {00,01,11} still decode to {0,+1,-1}; code 2 -> -2 never occurs.  Works on the
// shipped mag/sign encoding, no repack.
static string dec_sub() {
    return "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
           "float(int(u&1u)-(int((u>>1)&1u)<<1)),"
           "float(int((u>>2)&1u)-(int((u>>3)&1u)<<1)),"
           "float(int((u>>4)&1u)-(int((u>>5)&1u)<<1)),"
           "float(int((u>>6)&1u)-(int((u>>7)&1u)<<1)));}\n";
}
static string dec_sub_vec() {
    return "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);"
           "ivec4 m=ivec4(u&1u,(u>>2)&1u,(u>>4)&1u,(u>>6)&1u);"
           "ivec4 s=ivec4((u>>1)&1u,(u>>3)&1u,(u>>5)&1u,(u>>7)&1u);"
           "return vec4(m-(s<<1));}\n";
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rows = 1024, cols = 1024, iters = 30;
    const char* only = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) only = argv[++i];
        else { printf("usage: %s [--rows N] [--cols M] [--iters K] [--variant NAME]\n", argv[0]); return 2; }
    }
    if (cols % 8) { printf("cols must be a multiple of 8\n"); return 2; }
    rows += rows & 1;
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\nversion  %s\n", (const char*) glGetString(GL_RENDERER),
           (const char*) glGetString(GL_VERSION));

    // Activations: one row, four int8 a texel (the I8_4 / X4Q shape), small ints.
    vector<float> xval((size_t) cols);
    vector<unsigned char> xtex((size_t) cols, 0);
    for (int i = 0; i < cols; i++) { xval[i] = (float) ((i % 5) - 2); xtex[i] = (unsigned char) ((int) xval[i] + 128); }
    GLuint xid = 0;
    glGenTextures(1, &xid);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, xid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, xtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Weights in the shipped column-major layout: texel (c,t) = row 2t cols
    // 8c+0..3 in b0, 8c+4..7 in b1, row 2t+1 the same in b2,b3.
    const int pnq = cols / 8, prows = rows / 2;
    vector<unsigned char> wtex((size_t) pnq * prows * 4, 0);
    vector<float> wref((size_t) rows * cols, 0.0f);
    unsigned seed = 12345;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
    for (int t = 0; t < prows; t++)
        for (int c = 0; c < pnq; c++)
            for (int ch = 0; ch < 4; ch++) {
                const int r = 2 * t + (ch >= 2 ? 1 : 0);
                const int base = 8 * c + (ch & 1) * 4;
                unsigned acc = 0;
                for (int k = 0; k < 4; k++) {
                    const int code = (int) (rnd() % 3);
                    const float w = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                    wref[(size_t) r * cols + base + k] = w;
                    acc |= ((code == 0 ? 0u : 1u) | (code == 2 ? 2u : 0u)) << (2 * k);
                }
                wtex[((size_t) (t * pnq + c) * 4) + ch] = (unsigned char) acc;
            }
    GLuint wid = 0;
    glGenTextures(1, &wid);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, wid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pnq, prows, 0, GL_RGBA, GL_UNSIGNED_BYTE, wtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rows, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    auto dot_row = [&](int r) {
        double w = 0.0;
        for (int i = 0; i < cols; i++) w += wref[(size_t) r * cols + i] * xval[i];
        return w;
    };
    // The 2-row arm stores q16 (rails at +-128): pick a power-of-2 store scale
    // from the reference peak so the check reports the kernel, not the grid.
    double pair_div = 1.0;
    {
        double amax = 0.0;
        for (int r = 0; r < rows; r++) amax = std::max(amax, std::fabs(dot_row(r)));
        while (amax / pair_div > 120.0) pair_div *= 2.0;
    }

    struct V { const char* name; string (*dec)(); bool wide; bool intacc; };
    const V VS[] = {
        {"ship (8x convert)", dec_ship, false, false},
        {"hoist (uint once)", dec_hoist, false, false},
        {"code (2-bit algebra)", dec_code, false, false},
        {"lut (vec4 index)", dec_lut, false, false},
        {"signbit (bit-cast)", dec_signbit, false, false},
        {"sb_unroll (no loop)", dec_sb_unroll, false, false},
        {"sb_trunc (no floor)", dec_sb_trunc, false, false},
        {"sb_noand (sign no &1)", dec_sb_noand, false, false},
        {"sb_mask (mask not mul)", dec_sb_mask, false, false},
        {"subtract (bit0-2bit1)", dec_sub, false, false},
        {"sub_vec (ivec4)", dec_sub_vec, false, false},
        {"wide (16 a row)", dec_hoist, true, false},
        {"sb_wide (2row+signbit)", dec_sb_trunc, true, false},
        {"sub_wide (2row+sub)", dec_sub, true, false},
        {"sub_vec_wide (2row)", dec_sub_vec, true, false},
        {"int_acc (integer dot)", dec_hoist, false, true},
        {"int_acc_wide (2row int)", dec_hoist, true, true},
    };
    printf("\n%-22s %9s %10s %8s\n", "variant", "ms/draw", "ns/MAC", "check");
    for (const V& v : VS) {
        if (only && !strstr(v.name, only)) continue;
        const string ver = "#version 300 es\n";
        string vs = ver + "in vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver + "precision highp float;\nprecision highp int;\nout vec4 frag_color;\n"
                    "uniform sampler2D W,X;\n" + v.dec();
        if (v.intacc) {
            // Integer-domain dot: keep the activation as raw int8, decode the
            // weight to a sign/mag mask, (a ^ sign<<31) & (0-mag), accumulate
            // in ivec4, one ITOF at the end.  No per-weight float mul.  The
            // small-integer probe data makes this exact (so the check passes);
            // on real q16 data the rounding differs from the float path -- a
            // model run would need a PPL re-baseline, not a byte-compare.
            fs += "ivec4 wmask(uint u,ivec4 a){ivec4 sg=ivec4((u>>1)&1u,(u>>3)&1u,(u>>5)&1u,(u>>7)&1u);"
                  "ivec4 mask=ivec4(0)-sg;ivec4 mv=ivec4(u&1u,(u>>2)&1u,(u>>4)&1u,(u>>6)&1u);"
                  "return ((a^mask)-mask)&(ivec4(0)-mv);}\n";
            if (v.wide) {
                fs += "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
                      "float hi=floor(q/256.0);return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n";
                fs += "void main(){int t=int(gl_FragCoord.x);ivec4 s0=ivec4(0),s1=ivec4(0);"
                      "for(int c=0;c<" + std::to_string(pnq) + ";c++){"
                      "vec4 w=texture2D(W,vec2((float(c)+0.5)/float(" + std::to_string(pnq) +
                      "),(float(t)+0.5)/float(" + std::to_string(prows) + ")));"
                      "ivec4 xa=ivec4(texture2D(X,vec2((float(2*c)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0+0.5)-128;"
                      "ivec4 xb=ivec4(texture2D(X,vec2((float(2*c+1)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0+0.5)-128;"
                      "s0+=wmask(uint(w.r*255.0+0.5),xa)+wmask(uint(w.g*255.0+0.5),xb);"
                      "s1+=wmask(uint(w.b*255.0+0.5),xa)+wmask(uint(w.a*255.0+0.5),xb);}"
                      "float f0=float(s0.x+s0.y+s0.z+s0.w)*" + std::to_string(1.0 / pair_div) +
                      ",f1=float(s1.x+s1.y+s1.z+s1.w)*" + std::to_string(1.0 / pair_div) + ";"
                      "vec2 p0=e16(f0*256.0),p1=e16(f1*256.0);"
                      "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
            } else {
                fs += "void main(){int r=int(gl_FragCoord.x);int t=r/2;float h=float(r-2*t);ivec4 s=ivec4(0);"
                      "for(int g=0;g<" + std::to_string(pnq) + ";g++){"
                      "vec4 c=texture2D(W,vec2((float(g)+0.5)/float(" + std::to_string(pnq) +
                      "),(float(t)+0.5)/float(" + std::to_string(prows) + ")));"
                      "vec2 bb=mix(c.ba,c.rg,1.0-h);"
                      "ivec4 xa=ivec4(texture2D(X,vec2((float(g*2)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0+0.5)-128;"
                      "ivec4 xb=ivec4(texture2D(X,vec2((float(g*2+1)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0+0.5)-128;"
                      "s+=wmask(uint(bb.x*255.0+0.5),xa)+wmask(uint(bb.y*255.0+0.5),xb);}"
                      "float sf=float(s.x+s.y+s.z+s.w);"
                      "float q=clamp(floor(sf*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
                      "float gg=floor(q/65536.0),b=floor(q/256.0)-gg*256.0,a=q-gg*65536.0-b*256.0;"
                      "frag_color=vec4(gg/255.0,b/255.0,a/255.0,1.0);}\n";
            }
        } else if (v.wide) {
            // 16-wide single row: a texel is 16 weights of ONE row (row-major),
            // so this arm needs its own weight layout -- skip the shared texture
            // and decode all four bytes of the column-major texel as 16 weights
            // of row 2t (b0,b1) and 2t+1 (b2,b3) is the pair form, not this.
            // Here: one fragment a row-pair, 16 MACs, all four bytes (the t3p
            // amortization), storing two q16 -- priced as 2 rows a fragment.
            fs += "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
                  "float hi=floor(q/256.0);return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n";
            fs += "void main(){int t=int(gl_FragCoord.x);float s0=0.0,s1=0.0;"
                  "for(int c=0;c<" + std::to_string(pnq) + ";c++){"
                  "vec4 w=texture2D(W,vec2((float(c)+0.5)/float(" + std::to_string(pnq) +
                  "),(float(t)+0.5)/float(" + std::to_string(prows) + ")));"
                  "vec4 xa=texture2D(X,vec2((float(2*c)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 xb=texture2D(X,vec2((float(2*c+1)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "s0+=dot(dec(w.r),xa)+dot(dec(w.g),xb);"
                  "s1+=dot(dec(w.b),xa)+dot(dec(w.a),xb);}"
                  "vec2 p0=e16(s0*" + std::to_string(1.0 / pair_div) + "*256.0),p1=e16(s1*" +
                  std::to_string(1.0 / pair_div) + "*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
        } else {
            // The exact shipped matvec_t3 body: one fragment a row, bb=mix.
            fs += "void main(){int r=int(gl_FragCoord.x);int t=r/2;float h=float(r-2*t);float s=0.0;"
                  "for(int g=0;g<" + std::to_string(pnq) + ";g++){"
                  "vec4 c=texture2D(W,vec2((float(g)+0.5)/float(" + std::to_string(pnq) +
                  "),(float(t)+0.5)/float(" + std::to_string(prows) + ")));"
                  "vec2 bb=mix(c.ba,c.rg,1.0-h);"
                  "vec4 xa=texture2D(X,vec2((float(g*2)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 xb=texture2D(X,vec2((float(g*2+1)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "s+=dot(dec(bb.x),xa)+dot(dec(bb.y),xb);}"
                  "float q=clamp(floor(s*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
                  "float gg=floor(q/65536.0),b=floor(q/256.0)-gg*256.0,a=q-gg*65536.0-b*256.0;"
                  "frag_color=vec4(gg/255.0,b/255.0,a/255.0,1.0);}\n";
        }
        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) { printf("%-22s COMPILE FAIL %s\n", v.name, log.c_str()); continue; }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs); glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char b[4096]; glGetProgramInfoLog(p, sizeof b, nullptr, b); printf("%-22s LINK FAIL %s\n", v.name, b); continue; }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, xid);
        glUniform1i(glGetUniformLocation(p, "X"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, wid);
        glUniform1i(glGetUniformLocation(p, "W"), 1);
        glActiveTexture(GL_TEXTURE0);
        const int frags = v.wide ? prows : rows;
        glViewport(0, 0, frags, 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        vector<unsigned char> px((size_t) frags * 4, 0);
        glReadPixels(0, 0, frags, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int bad = 0; double worst = 0;
        for (int r = 0; r < rows; r++) {
            double got;
            if (v.wide) {
                const size_t o = (size_t) (r / 2) * 4 + (size_t) (r & 1) * 2;
                double u = ((double) px[o] * 65280.0 + (double) px[o + 1] * 255.0) / 255.0;
                if (u >= 32768.0) u -= 65536.0;
                got = u / 256.0;
            } else {
                got = ((double) px[(size_t) r * 4] * 65536.0 + (double) px[(size_t) r * 4 + 1] * 256.0 +
                       px[(size_t) r * 4 + 2] - 8388608.0) / 256.0;
            }
            const double ref = dot_row(r) / (v.wide ? pair_div : 1.0);
            const double e = std::fabs(got - ref) / (1.0 + std::fabs(ref));
            if (e > 2e-3) { if (bad < 3) printf("    row %d got %.4f want %.4f\n", r, got, ref); ++bad; }
            if (e > worst) worst = e;
        }
        const double macs = (double) rows * cols;   // both forms do rows*cols MACs
        printf("%-22s %9.3f %10.2f %8s\n", v.name, ms, 1e6 * ms / macs, bad ? "MISMATCH" : "ok");
        if (bad) printf("    %d/%d rows off (worst %.3g)\n", bad, rows, worst);
        if (getenv("T3DEC_DUMP")) printf("    shader:\n%s\n", fs.c_str());
        glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
    }
    return 0;
}