// What does each XFormat's encode and decode cost per value, on this driver?
//
// xformat.h defines seven streams: I16_1, I16_2, I8_4, I24_1, S24_1, F32 and
// BF16.  Every kernel in the engine reads one of them through x_glsl(f)'s
// X1/X4 macros and writes one through its XST* macros, so the cost of those
// two directions is the floor under every matvec, norm and residual add.  The
// kernels here are built from that same emitted text; nothing is re-typed.
//
// Per format, two timed draws and three checks:
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
// The checks.  The pattern is k/16 in +-7.875, which every format represents
// exactly (i8's rail is +-7.94 at a 1/16 step, s24 is 1/4096, bf16 has the
// 1/16 in three bits, f32 exact), so:
//
//   * the encode is compared byte for byte against x_pack() of the same host
//     pattern (GL_DITHER off: an 8-bit target would otherwise round writes
//     the check reads back);
//   * the shipped decode is checked by one untimed round trip -- decode then
//     re-encode in the same format -- against the input texture's bytes;
//   * every arm's decoded values are written as f32 (a bit-cast, exact) and
//     compared with the *shipped* arm's, so an "optimized" decode that gives
//     a different value is named instead of timed.  The shipped q16 decode is
//     itself the 2-mul approximation (not x_unpack bit for bit), which is why
//     the reference is the shipped text and not the host.
//
// Two clocks, because they answer different questions.  The wall column
// includes the submit and finish around a 1024-fragment draw, which on VC4
// dominates a cheap arm (probes/README.md); the counter column is the QPU's
// own fragment-shading time when PERFCNT=1, divided by the measured 12
// QPU-equivalents and 300 MHz (perfcnt.h).  Compare formats by the gpu
// column, and the wall column by itself over time.
//
// The variant rows exist to answer "how could this format be cheaper": the
// shipped macro against a formulation that does the same arithmetic another
// way, same checks, same clock.
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

// A decode formulation.  Ship is the engine's X1; the others do the same
// arithmetic another way, and the f32 check says whether they agree.
enum class Opt {
    Ship,
    PairHoist,  // one fetch, then select the texel half (the shipped X1's
                // ternary names texture2D twice and both sides are evaluated)
    I8Mask,     // I8_4: one vec4 conversion, then a basis dot for the component
    I8Index,    // I8_4: one vec4 conversion, dynamic component index
    NoSnap,     // I24/S24: the bytes as one dot with the byte weights, no int()
    F32Vec,     // F32: the byte fixes as a vec4 floor, then the shifts
    Recip,      // encode: /255 as * (1/255)
};

struct Arm {
    const char* name;
    XFormat f;
    bool es3;   // the codec is a bit-cast (F32, BF16): ESSL 3.00 text
    Opt dec;
    Opt enc;
};

// The statements that leave the value in `float v`, given `int i`.
static string dec_body(XFormat f, Opt o, int n) {
    const string TW = to_string(x_texels(n, f)) + ".0";
    switch (o) {
        case Opt::Ship:
            return "float v=X1(T,i," + TW + ");";
        case Opt::PairHoist:
            if (f == XFormat::BF16)
                return "vec4 c=texture2D(T,vec2((float(i/2)+0.5)/" + TW + ",0.5));"
                       "float v=(((i/2)*2==i)?x_bf16(c.rg):x_bf16(c.ba));";
            return "vec4 c=texture2D(T,vec2((float(i/2)+0.5)/" + TW + ",0.5));"
                   "float v=x_i16(((i/2)*2==i)?c.rg:c.ba);";
        case Opt::I8Mask:
            return "vec4 c=texture2D(T,vec2((float(i/4)+0.5)/" + TW + ",0.5));"
                   "vec4 q=floor(c*255.0+0.5)-128.0;"
                   "vec4 m=1.0-step(0.5,abs(vec4(float(i-(i/4)*4))-vec4(0.0,1.0,2.0,3.0)));"
                   "float v=dot(q,m)/16.0;";
        case Opt::I8Index:
            return "vec4 c=texture2D(T,vec2((float(i/4)+0.5)/" + TW + ",0.5));"
                   "float v=(floor(c[int(i-(i/4)*4)]*255.0+0.5)-128.0)/16.0;";
        case Opt::NoSnap:
            if (f == XFormat::S24_1)
                return "vec4 c=texture2D(T,vec2((float(i)+0.5)/" + TW + ",0.5));"
                       "float v=c.r*1044480.0+c.g*4080.0+c.b*15.9375"
                       "+c.a*0.062255859375-2048.0;";
            return "vec4 c=texture2D(T,vec2((float(i)+0.5)/" + TW + ",0.5));"
                   "float v=c.g*65280.0+c.b*255.0+c.a*0.99609375-32768.0;";
        case Opt::F32Vec:
            return "vec4 c=texture2D(T,vec2((float(i)+0.5)/" + TW + ",0.5));"
                   "vec4 b=floor(c*255.0+0.5);"
                   "uint u=(uint(b.x)<<24)|(uint(b.y)<<16)|(uint(b.z)<<8)|uint(b.w);"
                   "float v=uintBitsToFloat(u);";
        case Opt::Recip:
            break;
    }
    return "";
}

// One fragment a value.  `f32out` writes the value itself (a bit-cast, so the
// comparison is exact) instead of the fract probe.
static string dec_shader(XFormat f, int n, Opt o, bool f32out) {
    // The f32 codec is a bit-cast: it is only emitted for the check's f32
    // output, and that programme is ESSL 3.00 whatever the codec is.
    return head(f) + (f32out ? x_codec_glsl(XFormat::F32) : std::string()) +
        "uniform sampler2D T;" +
        (f32out ? "" : "uniform float sc1;uniform float sc2;") +
        "void main(){int i=int(gl_FragCoord.x);" + dec_body(f, o, n) +
        (f32out ? "gl_FragColor=x_f32enc(v);"
                : "gl_FragColor=vec4(fract(v*sc1),fract(v*sc2),0.0,1.0);") + "}";
}

// The reciprocal-255 store variants: the same encoder with the divisions
// replaced by a multiply.  A byte is recovered by the target's own rounding,
// so the checks below decide whether it is still the same byte.
static string enc_prelude(XFormat f) {
    const string IV = "0.00392156862745098";  // f32(1/255)
    switch (f) {
        case XFormat::I8_4:
            return "vec4 xs_enc(float a,float b,float c,float d){"
                   "vec4 q=clamp(floor(vec4(a,b,c,d)*16.0+0.5),-127.0,127.0);"
                   "return (q+128.0)*" + IV + ";}\n";
        case XFormat::I24_1:
            return "vec4 xs_enc(float a){float q=floor(clamp(a*256.0,-8388608.0,"
                   "8388607.0)+0.5)+8388608.0;float g=floor(q/65536.0),bb=floor(q/256.0)"
                   "-g*256.0,aa=q-g*65536.0-bb*256.0;"
                   "return vec4(0.0,g*" + IV + ",bb*" + IV + ",aa*" + IV + ");}\n";
        case XFormat::S24_1:
            return "vec4 xs_enc(float a){float q=floor(a*4096.0+0.5)+8388608.0;"
                   "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
                   "bb=floor(q/256.0)-r0*65536.0-g*256.0,"
                   "aa=q-r0*16777216.0-g*65536.0-bb*256.0;"
                   "return vec4(r0*" + IV + ",g*" + IV + ",bb*" + IV + ",aa*" + IV + ");}\n";
        case XFormat::I16_1:
            return "vec2 xq(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
                   "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
                   "return vec2(hi*" + IV + ",(q-hi*256.0)*" + IV + ");}\n"
                   "vec4 xs_enc(float a){vec2 p=xq(a*256.0);"
                   "return vec4(p.x,p.y,0.0,1.0);}\n";
        case XFormat::I16_2:
            return "vec2 xq(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
                   "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
                   "return vec2(hi*" + IV + ",(q-hi*256.0)*" + IV + ");}\n"
                   "vec4 xs_enc(float a,float b){vec2 p=xq(a*256.0),q=xq(b*256.0);"
                   "return vec4(p.x,p.y,q.x,q.y);}\n";
        default:
            return "";
    }
}

// One fragment a texel.  `from_texture` is the round-trip check (values come
// from the input row); otherwise the pattern, so the target *is* the encoded
// row that the check compares with x_pack.
static string enc_shader(XFormat f, int n, bool from_texture, Opt o) {
    const int p = x_per_texel(f);
    const string TW = to_string(x_texels(n, f));
    string vals;
    for (int j = 0; j < p; ++j) {
        const string k = to_string(j);
        vals += "float v" + k + "=" + (from_texture ? "X1(T,i0+" + k + "," + TW + ".0)"
                                                    : "pat(float(i0+" + k + "))") + ";";
    }
    // xs_enc takes (a,b) for every format but I8_4, which takes four: the
    // shape of the call follows the layout, not the variant.
    string args;
    if (o == Opt::Ship) {
        for (int j = 0; j < p; ++j) args += (j ? "," : "") + string("v") + to_string(j);
        return head(f) + (from_texture ? string("uniform sampler2D T;") : string()) +
            pat_glsl_float() +
            "void main(){int i0=int(gl_FragCoord.x)*" + to_string(p) + ";" + vals +
            "gl_FragColor=XST" + to_string(p) + "(" + args + ");}";
    }
    for (int j = 0; j < p; ++j) args += (j ? "," : "") + string("v") + to_string(j);
    return head(f) + pat_glsl_float() + enc_prelude(f) +
        "void main(){int i0=int(gl_FragCoord.x)*" + to_string(p) + ";" + vals +
        "gl_FragColor=xs_enc(" + args + ");}";
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int N = 1024, iters = 2000;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--values") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--values N] [--iters K]\n", argv[0]); return 2; }
    }
    if (N < 64 || N > 1024 || (N % 16)) {
        printf("--values must be a multiple of 16 in 64..1024\n");
        return 2;
    }

    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
    EglSurfaceless e = egl_surfaceless_init(3);
    if (!e.ok()) return 1;
    const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    const bool es3_ctx = glsl && !strncmp(glsl, "OpenGL ES GLSL ES 3", 19);
    printf("== codec_probe: %d values a draw, %d draws an arm ==\n", N, iters);
    printf("renderer %s\nglsl     %s\n", (const char*)glGetString(GL_RENDERER),
           glsl ? glsl : "?");
    glDisable(GL_DITHER);
    GLuint g_fbo = 0;
    glGenFramebuffers(1, &g_fbo);

    GLuint vb = 0;
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    const float tri[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    vector<float> src((size_t)N);
    for (int i = 0; i < N; ++i) src[(size_t)i] = pat(i);

    const Arm arms[] = {
        {"i16_1",        XFormat::I16_1, false, Opt::Ship,      Opt::Ship},
        {"i16_2",        XFormat::I16_2, false, Opt::Ship,      Opt::Ship},
        {"i8_4",         XFormat::I8_4,  false, Opt::Ship,      Opt::Ship},
        {"i24_1",        XFormat::I24_1, false, Opt::Ship,      Opt::Ship},
        {"s24_1",        XFormat::S24_1, false, Opt::Ship,      Opt::Ship},
        {"f32",          XFormat::F32,   true,  Opt::Ship,      Opt::Ship},
        {"bf16",         XFormat::BF16,  true,  Opt::Ship,      Opt::Ship},
        // decode variants
        {"i16_2 hoist",  XFormat::I16_2, false, Opt::PairHoist, Opt::Ship},
        {"bf16 hoist",   XFormat::BF16,  true,  Opt::PairHoist, Opt::Ship},
        {"i8_4 mask",    XFormat::I8_4,  false, Opt::I8Mask,    Opt::Ship},
        {"i8_4 index",   XFormat::I8_4,  false, Opt::I8Index,   Opt::Ship},
        {"i24_1 nosnap", XFormat::I24_1, false, Opt::NoSnap,    Opt::Ship},
        {"s24_1 nosnap", XFormat::S24_1, false, Opt::NoSnap,    Opt::Ship},
        {"f32 vec",      XFormat::F32,   true,  Opt::F32Vec,    Opt::Ship},
        // encode variants
        {"i16_1 recip",  XFormat::I16_1, false, Opt::Ship,      Opt::Recip},
        {"i16_2 recip",  XFormat::I16_2, false, Opt::Ship,      Opt::Recip},
        {"i8_4 recip",   XFormat::I8_4,  false, Opt::Ship,      Opt::Recip},
        {"i24_1 recip",  XFormat::I24_1, false, Opt::Ship,      Opt::Recip},
        {"s24_1 recip",  XFormat::S24_1, false, Opt::Ship,      Opt::Recip},
    };
    const int na = (int)(sizeof arms / sizeof arms[0]);

    PerfMon pm;
    const bool want_c = perfcnt_enabled() && !pm.missing() &&
                        pm.pick({"QPU-total-clk-cycles-executing-valid-instr"});
    const size_t EXEC = 0;

    printf("\n%-13s %3s %7s %11s %11s %11s %11s  %s\n",
           "arm", "pv", "texels", "dec wall ns", "dec gpu ns", "enc wall ns",
           "enc gpu ns", "check");

    // The shipped decode's f32 output per format: the reference every
    // variant's decode is compared against, bit for bit.
    vector<float> shipped_dec[8];

    for (int a = 0; a < na; ++a) {
        const Arm& A = arms[a];
        if (A.es3 && !es3_ctx) {
            printf("%-13s skipped: the codec is ESSL 3.00 text and the context "
                   "is not\n", A.name);
            continue;
        }
        const int p = x_per_texel(A.f);
        const int tw = x_texels(N, A.f);
        const size_t fi = (size_t)A.f;
        string log;
        const string s_dec = dec_shader(A.f, N, A.dec, false);
        const string s_ex = dec_shader(A.f, N, A.dec, true);
        const string s_out = enc_shader(A.f, N, false, A.enc);
        const string s_rt = enc_shader(A.f, N, true, Opt::Ship);
        // The f32 check writes through the bit-cast, so *that* programme is
        // ESSL 3.00 even for an ES2 codec.
        GLuint pdec = build(s_dec, A.es3, log);
        GLuint pex = build(s_ex, true, log);
        GLuint pout = build(s_out, A.es3, log);
        GLuint prt = build(s_rt, A.es3, log);
        if (!pdec || !pex || !pout || !prt) {
            printf("%-13s COMPILE FAIL: %s\n", A.name, log.c_str());
            const string& bad = !pdec ? s_dec : !pex ? s_ex : !pout ? s_out : s_rt;
            int n = 1;
            printf("  --- source ---\n");
            for (size_t i = 0; i < bad.size(); ++i) {
                if (i == 0 || bad[i - 1] == '\n') printf("  %d| ", n++);
                putchar(bad[i]);
            }
            printf("\n  --- end ---\n");
            continue;
        }

        const vector<uint8_t> bytes = x_pack(src, A.f);
        GLuint xt = make_tex(tw, 1, bytes);
        GLuint out_dec = make_tex(N, 1, vector<uint8_t>((size_t)N * 4, 0));
        GLuint out_f32 = make_tex(N, 1, vector<uint8_t>((size_t)N * 4, 0));
        GLuint out_enc = make_tex(tw, 1, vector<uint8_t>((size_t)tw * 4, 0));

        auto attach = [&](GLuint tex) {
            glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        };
        auto read_back = [&](GLuint tex, int w) {
            attach(tex);
            vector<uint8_t> px((size_t)w * 4);
            glReadPixels(0, 0, w, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            return px;
        };
        auto use = [&](GLuint prog, int w, bool with_tex) {
            glViewport(0, 0, w, 1);
            glUseProgram(prog);
            if (with_tex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, xt);
                glUniform1i(glGetUniformLocation(prog, "T"), 0);
            }
        };
        static const float SC1 = 13.0f, SC2 = 7.389f;

        // ---- the round trip (shipped only) --------------------------------
        int bad_rt = 0;
        if (A.dec == Opt::Ship && A.enc == Opt::Ship) {
            use(prt, tw, true);
            attach(out_enc);
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            const vector<uint8_t> rt = read_back(out_enc, tw);
            for (size_t b = 0; b < bytes.size() && b < rt.size(); ++b)
                if (bytes[b] != rt[b]) ++bad_rt;
        }

        // ---- the encode against x_pack ------------------------------------
        use(pout, tw, false);
        attach(out_enc);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> en = read_back(out_enc, tw);
        int bad_en = 0;
        for (size_t b = 0; b < bytes.size() && b < en.size(); ++b)
            if (bytes[b] != en[b]) ++bad_en;

        // ---- the decoded values, exactly ----------------------------------
        use(pex, N, true);
        attach(out_f32);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> fx = read_back(out_f32, N);
        vector<float> vals((size_t)N);
        for (int i = 0; i < N; ++i)
            vals[(size_t)i] = x_unpack(fx.data(), i, XFormat::F32);
        if (A.dec == Opt::Ship) shipped_dec[fi] = vals;
        int bad_bit = 0;
        double maxdiff = 0;
        const vector<float>& ref = shipped_dec[fi];
        if (ref.size() == (size_t)N) {
            for (int i = 0; i < N; ++i) {
                const uint32_t ua = (uint32_t)fx[(size_t)i * 4] << 24 |
                                    (uint32_t)fx[(size_t)i * 4 + 1] << 16 |
                                    (uint32_t)fx[(size_t)i * 4 + 2] << 8 |
                                    (uint32_t)fx[(size_t)i * 4 + 3];
                uint32_t ub = 0;
                memcpy(&ub, &ref[(size_t)i], 4);
                if (ua != ub) ++bad_bit;
                const double d = std::fabs((double)vals[(size_t)i] - (double)ref[(size_t)i]);
                if (d > maxdiff) maxdiff = d;
            }
        }

        // ---- the fract probe, checked and timed ---------------------------
        use(pdec, N, true);
        glUniform1f(glGetUniformLocation(pdec, "sc1"), SC1);
        glUniform1f(glGetUniformLocation(pdec, "sc2"), SC2);
        attach(out_dec);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const vector<uint8_t> dc = read_back(out_dec, N);
        int bad_dc = 0;
        for (int i = 0; i < N; ++i) {
            const double v = x_unpack(bytes.data(), i, A.f);
            for (int ch = 0; ch < 2; ++ch) {
                const double sc = ch ? SC2 : SC1;
                double want = std::fmod(v * sc, 1.0);
                if (want < 0) want += 1.0;
                const double got = dc[(size_t)i * 4 + (size_t)ch] / 255.0;
                double d = std::fabs(got - want);
                if (d > 0.5) d = 1.0 - d;
                if (d > 3.0 / 255.0) ++bad_dc;
            }
        }

        auto time_dec = [&]() {
            use(pdec, N, true);
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
            use(pout, tw, false);
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
            use(pdec, N, true);
            glUniform1f(glGetUniformLocation(pdec, "sc1"), SC1);
            glUniform1f(glGetUniformLocation(pdec, "sc2"), SC2);
            attach(out_dec);
            pm.begin();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            const PerfMon::Sample s = pm.end();
            if (!s.value.empty()) dec_gpu = (double)s.value[EXEC] * to_ms / iters;

            use(pout, tw, false);
            attach(out_enc);
            pm.begin();
            for (int k = 0; k < iters; ++k) glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            const PerfMon::Sample s2 = pm.end();
            if (!s2.value.empty()) enc_gpu = (double)s2.value[EXEC] * to_ms / iters;
        }

        char check[128];
        if (bad_rt || bad_en || bad_dc)
            snprintf(check, sizeof check, "MISMATCH rt %d, enc %d, fract %d",
                     bad_rt, bad_en, bad_dc);
        else if (bad_bit)
            snprintf(check, sizeof check, "dec differs in %d/%d (max %.3g)",
                     bad_bit, N, maxdiff);
        else
            snprintf(check, sizeof check, "exact");
        const double per_val = 1e6 / (double)N;
        printf("%-13s %3d %7d %11.3f %11.3f %11.3f %11.3f  %s\n",
               A.name, p, tw, dec_ms * per_val, dec_gpu * per_val,
               enc_ms * per_val, enc_gpu * per_val, check);

        glDeleteTextures(1, &xt);
        glDeleteTextures(1, &out_dec);
        glDeleteTextures(1, &out_f32);
        glDeleteTextures(1, &out_enc);
        glDeleteProgram(pdec);
        glDeleteProgram(pex);
        glDeleteProgram(pout);
        glDeleteProgram(prt);
    }

    printf("\nwall ns is a 1024-fragment draw's submit+finish divided by the work; "
           "gpu ns is the QPU's fragment-shading cycles when PERFCNT=1\n");
    if (!want_c)
        printf("PERFCNT=1 adds the gpu column (the wall column alone is "
               "submission-dominated on VC4)\n");
    return 0;
}