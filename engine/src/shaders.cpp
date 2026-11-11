#include "shaders.h"
#include "gpu.h"

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using std::string;
using std::to_string;

namespace {

string head(XFormat xf) {
    return "precision highp float;\n" + x_glsl(xf);
}

string q24_helpers() {
    return
        "float x_i24raw(vec4 c){return float(int(c.r*255.0+0.5))*16777216.0+"
        "float(int(c.g*255.0+0.5))*65536.0+float(int(c.b*255.0+0.5))*256.0+"
        "float(int(c.a*255.0+0.5))-8388608.0;}\n"
        "float x_i24q(vec4 c){return x_i24raw(c)/256.0;}\n"
        "vec4 x_i24enc(float v){float q=floor(v*256.0+0.5)+8388608.0;"
        "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
        "b=floor(q/256.0)-r0*65536.0-g*256.0,"
        "a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n"
        "vec4 x_i24encq(float q){q=clamp(floor(q+0.5),-32768.0,32767.0)+8388608.0;"
        "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
        "b=floor(q/256.0)-r0*65536.0-g*256.0,"
        "a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n";
}

// Shared GLSL fragments.  These are constants rather than GLSL helper
// functions on purpose: the hot matvec loops must keep the decode inline (VC4
// compiles a helper that takes a texture fetch as an argument much slower -
// 177 against 283 MMAC/s, docs/performance.md), so the emitted program carries
// the text either way.  Defining it once means a change to the decode is made
// once, and --dump-shaders proves the emitted bytes did not move.
//
// The q4 kernels route through q4_head(), which has always put each decoder on
// its own line; the newline-separated join is kept there so the text is
// unchanged.
const string kDec16A =
    "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
    "return u>=32768.0?u-65536.0:u;}";
const string kDec16B =
    "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
    "return u>=32768.0?u-65536.0:u;}";

// Round/clamp to q16, then swiglu: the fused gate kernels.
const string kQzSl =
    "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
    "float sl(float q,float g){float a=q/256.0;return (a/(1.0+exp(-a)))*(g/256.0);}";

// q16 pair from a scaled value (s*w) and from a value already in q16 units.
const string kEnc16Mul =
    "vec2 enc16(float s,float w){float v=clamp(floor(s*w+0.5),-32768.0,32767.0);"
    "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
    "return vec2(hi/255.0,(v-hi*256.0)/255.0);}";
const string kEnc16Raw =
    "vec2 enc16(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);"
    "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
    "return vec2(hi/255.0,(v-hi*256.0)/255.0);}";

// Sum of squares in 24-bit fixed point, scale 4096 (the rms stats passes).
const string kEncs4096 =
    "vec4 encs(float s){float q=floor(s*4096.0+0.5)+8388608.0;"
    "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
    "b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
    "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";

// Shape contract.  The q8 kernels step the column loop by four; the q4 kernels
// take eight weights per texel in groups of 32.  A width that is not a multiple
// of that reads past the activation texture and the shader still compiles - it
// just returns a wrong number a layer later, which is how a mis-sized call
// survives to the PPL run.  Check where the text is made, once per program.
static void need_shape(const char* who, int rows, int cols, int unit) {
    if (rows <= 0 || cols <= 0 || cols % unit != 0) {
        fprintf(stderr, "shader %s: rows=%d cols=%d (cols must be a positive multiple of %d)\n",
                who, rows, cols, unit);
        exit(2);
    }
}

static void need_dim(const char* who, int n) {
    if (n <= 0) {
        fprintf(stderr, "shader %s: size=%d\n", who, n);
        exit(2);
    }
}

static string q4_head(XFormat xf, bool inl) {
    if (!inl || xf != XFormat::I16_2) return head(xf);
    return head(xf) + kDec16A + "\n" + kDec16B + "\n";
}

// INT4 kernels all decode eight I16_2 activations per 32-weight group.  The
// generic X4Q macro hands two texel fetches to a helper and VC4 compiles that
// loop much slower (q8 matvec_probe: 177 vs 283 MMAC/s at 1536x768).  Emitting
// the four fetches and the dA/dB decoders in the loop body is the q4 analogue
// of the q8 inline decode; the arithmetic and the order are unchanged.
static string q4_act(XFormat xf, int cols, bool inl) {
    const string AXW = to_string(x_texels(cols, xf));
    if (!inl || xf != XFormat::I16_2)
        return "vec4 xa=X4Q(X,c," + AXW + ".0);vec4 xb=X4Q(X,c+4," + AXW + ".0);";
    return
        "vec4 t0=texture2D(X,vec2((float(c/2)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t1=texture2D(X,vec2((float(c/2+1)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t2=texture2D(X,vec2((float(c/2+2)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t3=texture2D(X,vec2((float(c/2+3)+0.5)/" + AXW + ".0,0.5));"
        "float a0=dA(t0),a1=dB(t0),a2=dA(t1),a3=dB(t1),"
        "a4=dA(t2),a5=dB(t2),a6=dA(t3),a7=dB(t3);"
        "vec4 xa=vec4(a0,a1,a2,a3);vec4 xb=vec4(a4,a5,a6,a7);";
}

// Default-off probe (Q4_NOMIN=1): drop the per-group min term (mn*xsum) and
// the seven adds that build xsum from the INT4 kernels.  The output is wrong
// on purpose - this prices the ALU that a precomputed per-group activation
// sum would remove, before any pass or texture plumbing is built for it.
static bool q4_nomin() {
    static const char* e = getenv("Q4_NOMIN");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

static string q4_mint() {
    return q4_nomin() ? "" : "+mn*(xa.x+xa.y+xa.z+xa.w+xb.x+xb.y+xb.z+xb.w)";
}

static string q4_xsum() {
    return q4_nomin() ? "" : "xsum+=xa.x+xa.y+xa.z+xa.w+xb.x+xb.y+xb.z+xb.w;";
}

// Vec4 form of the fused-gate tap decode (Q4_MV4=1, default off).  It is
// byte-identical to the scalar form but 21% slower at the model shape
// (q4mv_probe: 11.35 against 9.39 ms per 2432-row draw), so it ships off and
// stays as the A/B that killed the ALU hypothesis.  Every tap is an integer
// in [0,15] and every activation an integer in [-32768,32767], so each
// product is exact and any sum of the eight per group stays below 2^24: the
// lane split and the horizontal add cannot move a bit.  The caller declares
// bb/ib/hi/lo once per loop body, so the statements below are assignments.
static bool q4_mv4() {
    static const char* e = getenv("Q4_MV4");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

static string q4_taps4(const char* W) {
    const string w(W);
    return "bb=floor(" + w + "*255.0+0.5);ib=ivec4(bb);"
           "hi=vec4(ib>>4);lo=vec4(ib&15);";
}

// The eight activations of one 32-weight group, even columns in xe and odd in
// xo, so lo*xe+hi*xo is the same term order as the scalar l0..h3 chain.
static string q4_act8(XFormat xf, int cols, bool inl) {
    const string AXW = to_string(x_texels(cols, xf));
    if (!inl || xf != XFormat::I16_2)
        return "vec4 xa=X4Q(X,c," + AXW + ".0);vec4 xb=X4Q(X,c+4," + AXW + ".0);"
               "vec4 xe=vec4(xa.x,xa.z,xb.x,xb.z),xo=vec4(xa.y,xa.w,xb.y,xb.w);";
    return
        "vec4 t0=texture2D(X,vec2((float(c/2)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t1=texture2D(X,vec2((float(c/2+1)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t2=texture2D(X,vec2((float(c/2+2)+0.5)/" + AXW + ".0,0.5));"
        "vec4 t3=texture2D(X,vec2((float(c/2+3)+0.5)/" + AXW + ".0,0.5));"
        "vec4 xe=vec4(dA(t0),dA(t1),dA(t2),dA(t3));"
        "vec4 xo=vec4(dB(t0),dB(t1),dB(t2),dB(t3));";
}

static string q4_xsum8() {
    return q4_nomin() ? "" : "vec4 xs=xe+xo;xsum+=xs.x+xs.y+xs.z+xs.w;";
}


// Default-off probe (Q4_NOUNPACK=1): swap the nibble unpack for a handful of
// multiplies (the weights are nonsense on purpose) to price the four fetches
// plus the floor chain.  A large win here is the only thing that would justify
// a hi/lo split weight format, which needs a re-quantised model.
static bool q4_nounpack() {
    static const char* e = getenv("Q4_NOUNPACK");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

// Q4_BFAST=1 drops the floor(x+0.5) on the byte decode: an 8-bit component
// comes back as a float within 1e-5 of the byte, and the q8 kernels have
// never needed a floor because they multiply instead of flooring.  This probe
// floors afterwards (h = floor(b/16)), so the ~1e-5 deficit lands under every
// multiple of 16 and the weights are wrong on purpose: it prices the four byte
// floors, which is the ceiling any exact byte decode could reach.
static bool q4_bfast() {
    static const char* e = getenv("Q4_BFAST");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

static string q4_byte(const string& w, char ch) {
    const string t = w + "." + ch;
    if (q4_bfast()) return t + "*255.0";
    return "floor(" + t + "*255.0+0.5)";
}

// The unpack of one 8-weight texel, shared verbatim by the INT4 kernels so the
// probe swaps it in one place.  q4_h is defined below; W is the texel name.
static string q4_h(const string& w);
static string q4_taps(const char* W) {
    const string w(W);
    if (gpu_q4_shift() && !q4_nounpack()) {
        // Bitwise split: ASR + AND, one QPU instruction each (vc4_program.c
        // handles nir_op_ishr/iand directly).  Needs ESSL 3.00, which
        // GPU::link_program emits when it sees the "&15" in this text.
        return "int b0=int(" + w + ".r*255.0+0.5),b1=int(" + w + ".g*255.0+0.5),"
               "b2=int(" + w + ".b*255.0+0.5),b3=int(" + w + ".a*255.0+0.5);"
               "float h0=float(b0>>4),h1=float(b1>>4),h2=float(b2>>4),h3=float(b3>>4);"
               "float l0=float(b0&15),l1=float(b1&15),l2=float(b2&15),l3=float(b3&15);";
    }
    if (q4_nounpack())
        return "float h0=" + w + ".r*15.0,l0=" + w + ".g*15.0,h1=" + w + ".b*15.0,"
               "l1=" + w + ".a*15.0,h2=" + w + ".r*7.0,l2=" + w + ".g*7.0,"
               "h3=" + w + ".b*7.0,l3=" + w + ".a*7.0;";
    return "float b0=" + q4_byte(w, 'r') + ",b1=" + q4_byte(w, 'g') + ","
           "b2=" + q4_byte(w, 'b') + ",b3=" + q4_byte(w, 'a') + ";"
           + q4_h(w) +
           "float l0=b0-h0*16.0,l1=b1-h1*16.0,l2=b2-h2*16.0,l3=b3-h3*16.0;";
}

// Q4_HFAST=1 builds the high nibble straight from the texel instead of from
// the byte: 255/16 is exact, so floor(w*15.9375 + 1/32) == floor(b/16) and the
// value is unchanged.  Same two ops, but the chain from the fetch to h drops
// from four ops to two.
static bool q4_hfast() {
    static const char* e = getenv("Q4_HFAST");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

static string q4_h(const string& w) {
    if (q4_hfast())
        return "float h0=floor(" + w + ".r*15.9375+0.03125),h1=floor(" + w + ".g*15.9375+0.03125),"
               "h2=floor(" + w + ".b*15.9375+0.03125),h3=floor(" + w + ".a*15.9375+0.03125);";
    return "float h0=floor(b0*0.0625),h1=floor(b1*0.0625),"
           "h2=floor(b2*0.0625),h3=floor(b3*0.0625);";
}
}  // namespace

// Per-row q8 weight scale.  A container may carry one f32 per output row
// (tensor id part 3) instead of a single per-matrix scale, because one scale
// for a 640x1440 matrix leaves 58-66% of void.1's layer-0 FFN weights at zero
// and the model then misspells bytes (max|d| 14.4 logits against f32; per-row
// brings it to 0.4).  The plane is one texel wide, one row per texel, so the
// row a fragment writes is the texel it reads.  Emitted only when the matrix
// has one: a model without it links exactly the text it linked before.
string wrow_scale(int rows, const char* smp = "WS", const char* fn = "wsr",
                  const char* unit = "ws", bool hi = false) {
    // One texel per *pair* of rows: .rg holds row 2t, .ba holds row 2t+1, each
    // 16 bits of the row scale divided by the matrix unit (which travels in
    // the kernel's existing scale uniform).  A kernel that needs both rows of
    // a fragment therefore reads the texel once.
    //
    // Pairing is not an optimisation.  Asking VC4 for two texture2D fetches
    // from the same sampler in one fragment returns undefined data once the
    // matrix is tall: probes/wrow_probe measures matvec_q8_pair_logits with
    // wrow off by 33126 at 88 of 896 rows, while the same shader shape reading
    // two different textures (matvec_q8_pair) is exact at 1440 rows.  void.1's
    // qkv and logits pass through that kernel, which is why the whole model
    // answered with <pad> and invalid bytes on the card and was correct on Xe.
    const int np = (rows + 1) / 2;
    if (hi) {
        // hi pairs (t, t+rows/2): the fused W13 kernel writes row r and row
        // r+rows from one fragment, so those are the two the texel carries.
        // One fetch, same as the (2t, 2t+1) form.
        return string("uniform sampler2D ") + smp +
               ";vec2 " + fn + "2(int t){vec4 c=texture2D(" + smp +
               ",vec2(0.125,(float(t)+0.5)/float(" + to_string(np) + ")));"
               "return vec2(c.r*65280.0+c.g*255.0,c.b*65280.0+c.a*255.0)*" + unit +
               "*(1.0/65536.0);}"
               "float " + fn + "(int r){return 0.0;}\n";
    }
    return string("uniform sampler2D ") + smp +
           ";vec4 " + fn + "p(int r){return texture2D(" + smp +
           ",vec2(0.125,(float(r/2)+0.5)/float(" + to_string(np) + ")));}"
           "float " + fn + "(int r){vec4 c=" + string(fn) + "p(r);"
           "return (mod(float(r),2.0)<0.5?c.r*65280.0+c.g*255.0:c.b*65280.0+c.a*255.0)*"
           + unit + "*(1.0/65536.0);}"
           "vec2 " + fn + "2(int r){vec4 c=" + string(fn) + "p(r);"
           "return vec2(c.r*65280.0+c.g*255.0,c.b*65280.0+c.a*255.0)*" + unit +
           "*(1.0/65536.0);}";
}


string matvec_q8(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_q8", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    // VC4's shader compiler generates a slower loop when the activation decode
    // goes through a helper that takes two texel fetches as arguments.  Emitting
    // the fetches and the two scalar decoders inline is measurably faster, so
    // the I16_2 fast path keeps the inline form.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;" + WD +
               kDec16A + kDec16B +
               "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r)+0.5)/float(" + ROWS + ")));"
               "vec4 w=c*255.0-128.0;"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
               "float v=clamp(floor(s*" + SC + "+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;"
               "v+=8388608.0;"
               "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
               "float b=floor(v/256.0)-r0*65536.0-g*256.0;"
               "float a=v-r0*16777216.0-g*65536.0-b*256.0;"
               "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
    }
    return head(xf) + q24_helpers() +
        "uniform sampler2D W,X;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_i24encq(s*" + SC + ");}";
}

string matvec_q8_pair(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_q8_pair", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WD = wrow ? (wrow_scale(rows, "WS1", "wsr1", "ws1") +
                              wrow_scale(rows, "WS2", "wsr2", "ws2")) : "";
    const string SC1 = wrow ? "wsr1(r)" : "ws1";
    const string SC2 = wrow ? "wsr2(r)" : "ws2";
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
               kDec16A + kDec16B
               + kEnc16Mul +
               "void main(){int r=int(gl_FragCoord.x);float s1=0.0,s2=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 c1=texture2D(W1,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r)+0.5)/float(" + ROWS + ")));"
               "vec4 c2=texture2D(W2,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r)+0.5)/float(" + ROWS + ")));"
               "vec4 w1=c1*255.0-128.0;vec4 w2=c2*255.0-128.0;"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "s1+=w1.x*a0+w1.y*a1+w1.z*a2+w1.w*a3;"
               "s2+=w2.x*a0+w2.y*a1+w2.z*a2+w2.w*a3;}"
               "vec2 p1=enc16(s1," + SC1 + "),p2=enc16(s2," + SC2 + ");"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s1=0.0,s2=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c1=texture2D(W1,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 c2=texture2D(W2,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w1=c1*255.0-128.0;vec4 w2=c2*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s1+=w1.x*x.x+w1.y*x.y+w1.z*x.z+w1.w*x.w;"
        "s2+=w2.x*x.x+w2.y*x.y+w2.z*x.z+w2.w*x.w;}"
        "gl_FragColor=x_st2q(s1*" + SC1 + ",s2*" + SC2 + ");}";
}

// q8 vocabulary projection, two output rows per fragment.  One activation decode
// feeds two dot products, so both the fragment count and the activation decode
// halve: 1536 vocabulary rows become 768 fragments.  Each accumulator sums in
// exactly the order matvec_q8 uses, and x_st2q applies the same q16 rounding and
// clamp as x_i24encq, so the packed pair holds the same two integers that the
// unpacked pass writes.
string matvec_q8_pair_logits(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_q8_pair_logits", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WR0 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r0)+0.5)/float(" + ROWS + "))";
    const string WR1 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r1)+0.5)/float(" + ROWS + "))";
    const string WD = wrow ? wrow_scale(rows) : "";
    // One texel holds both rows of the fragment; two fetches from the same
    // sampler are what VC4 gets wrong (probes/wrow_probe).
    const string SC0 = wrow ? "s.x" : "ws", SC1 = wrow ? "s.y" : "ws";
    const string SP = wrow ? "vec2 s=wsr2(r0);" : "";
    // Inline I16_2 decode.  The generic X4Q helper takes two texel fetches as
    // arguments and VC4 compiles that loop much slower (matvec_probe measured
    // 177 vs 283 MMAC/s at 1536x768).  Decode, arithmetic and the q16 rounding
    // are identical, so the packed output is byte-identical.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;" + WD
               + kDec16A + kDec16B
               + kEnc16Mul +
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
        "float s0=0.0,s1=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "vec4 c=texture2D(W," + WR0 + ");vec4 w=c*255.0-128.0;"
               "s0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
               "c=texture2D(W," + WR1 + ");w=c*255.0-128.0;"
               "s1+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;}"
               "vec2 p1=enc16(s0," + SC0 + "),p2=enc16(s1," + SC1 + ");"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;" + WD +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
        "float s0=0.0,s1=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "vec4 c=texture2D(W," + WR0 + ");vec4 w=c*255.0-128.0;"
        "s0+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;"
        "c=texture2D(W," + WR1 + ");w=c*255.0-128.0;"
        "s1+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_st2q(s0*" + SC0 + ",s1*" + SC1 + ");}";
}

// q8 FFN gate + swiglu fused: one fragment evaluates two consecutive gate rows
// for both w1 and w3 (four dots sharing one activation decode) and writes the two
// swiglu results as one q16 pair.  This collapses the two-draw sequence
// (pair matvec -> swiglu) into a single draw, and halving the fragment count keeps
// the viewport at rows/2.  The gate quantisation (int16 of s*ws) and the final q16
// rounding are the same operations the separate passes perform, so the output is
// bit-identical.
string matvec_q8_pair_swiglu(int rows, int cols, XFormat xf, bool inl, bool wrow) {
    need_shape("matvec_q8_pair_swiglu", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WD = wrow ? (wrow_scale(rows, "WS1", "wsr1", "ws1") +
                              wrow_scale(rows, "WS2", "wsr2", "ws2")) : "";
    // r0 is even and r1 is r0+1, so one plane texel carries both rows.
    const string A10 = wrow ? "s1.x" : "ws1", A20 = wrow ? "s2.x" : "ws2";
    const string A11 = wrow ? "s1.y" : "ws1", A21 = wrow ? "s2.y" : "ws2";
    const string SP = wrow ? "vec2 s1=wsr12(r0),s2=wsr22(r0);" : "";
    const string WR0 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r0)+0.5)/float(" + ROWS + "))";
    const string WR1 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r1)+0.5)/float(" + ROWS + "))";
    // Inline I16_2 decode.  The generic X4Q macro takes its two texel fetches as
    // arguments and VC4 compiles that loop measurably slower (matvec_probe
    // measured 177 vs 283 MMAC/s).  dA/dB are exactly the decode x_i16quadq
    // performs and the four accumulators keep the same summation order, so the
    // packed output is byte-identical.  FFN_SW_INLINE=0 restores the generic
    // form in the same binary for an interleaved A/B.
    if (inl && xf == XFormat::I16_2) {
        return head(xf) + "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
               kDec16A + kDec16B
               + kQzSl +
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
               "int r1=min(r0+1," + to_string(rows - 1) + ");" + SP +
               "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "vec4 c=texture2D(W1," + WR0 + ");vec4 w=c*255.0-128.0;"
               "s10+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
               "c=texture2D(W2," + WR0 + ");w=c*255.0-128.0;"
               "s20+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
               "c=texture2D(W1," + WR1 + ");w=c*255.0-128.0;"
               "s11+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
               "c=texture2D(W2," + WR1 + ");w=c*255.0-128.0;"
               "s21+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;}"
               "float v1=(r0+1<" + to_string(rows) + ")?sl(qz(s11*" + A11 + "),qz(s21*" + A21 + ")):0.0;"
               "gl_FragColor=x_st2(sl(qz(s10*" + A10 + "),qz(s20*" + A20 + ")),v1);}";
    }
    return head(xf) +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
               kQzSl +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
        "int r1=min(r0+1," + to_string(rows - 1) + ");" + SP +
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "vec4 c=texture2D(W1," + WR0 + ");vec4 w=c*255.0-128.0;"
        "s10+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;"
        "c=texture2D(W2," + WR0 + ");w=c*255.0-128.0;"
        "s20+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;"
        "c=texture2D(W1," + WR1 + ");w=c*255.0-128.0;"
        "s11+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;"
        "c=texture2D(W2," + WR1 + ");w=c*255.0-128.0;"
        "s21+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "float v1=(r0+1<" + to_string(rows) + ")?sl(qz(s11*" + A11 + "),qz(s21*" + A21 + ")):0.0;"
        "gl_FragColor=x_st2(sl(qz(s10*" + A10 + "),qz(s20*" + A20 + ")),v1);}";
}

// Two-position gate packing (WORKLOG item 7b).  One fragment evaluates one
// gate row for BOTH positions: the two weight texels (W1 and W3, row r) are
// fetched once and each feeds two activation quads, so a weight byte is used
// twice and the weight traffic per position halves.  A fragment writes one
// RGBA8 texel, i.e. two q16 values, which is one row for two positions - that
// is why this kernel is one row per fragment (viewport rows wide) and not the
// row pair the single-position kernel uses: four q16 would need two texels and
// vc4 has no MRT.  The activation input is one texture holding position A in
// [0,axw) and position B in [axw,2*axw); the output is the P2 layout
// (xformat.h).  Per (row, position) the fetch order, the four MACs and the
// qz/sl/encode steps are the same operations in the same order as
// matvec_q8_pair_swiglu, so the bytes must match two single-position draws.
string matvec_q8_gate2(int rows, int cols) {
    need_shape("matvec_q8_gate2", rows, cols, 4);
    const int axw = x_texels(cols, XFormat::I16_2);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4);
    const string AXW = to_string(axw), AXW2 = to_string(2 * axw);
    const string WR = "vec2((float(q)+0.5)/float(" + COLS4 + "),"
                      "(float(r)+0.5)/float(" + ROWS + "))";
    return head(XFormat::I16_2) + x_pair2_glsl() +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;"
               + kDec16A + kDec16B
               + kQzSl +
        "void main(){int r=int(gl_FragCoord.x);"
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW2 + "),0.5));"
        "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW2 + "),0.5));"
        "vec4 xc=texture2D(X,vec2((float(2*q+" + AXW + ")+0.5)/float(" + AXW2 + "),0.5));"
        "vec4 xd=texture2D(X,vec2((float(2*q+" + AXW + "+1)+0.5)/float(" + AXW2 + "),0.5));"
        "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
        "float b0=dA(xc),b1=dB(xc),b2=dA(xd),b3=dB(xd);"
        "vec4 c=texture2D(W1," + WR + ");vec4 w=c*255.0-128.0;"
        "s10+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "s11+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
        "c=texture2D(W2," + WR + ");w=c*255.0-128.0;"
        "s20+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "s21+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
        "gl_FragColor=x_p2_st(sl(qz(s10*ws1),qz(s20*ws2)),"
        "sl(qz(s11*ws1),qz(s21*ws2)));}";
}

// Two-position w2 matvec: one fragment evaluates output row r for both
// positions.  The P2 gate output gives .rg = position A and .ba = position B
// for the same column, so a four-weight texel needs four P2 texels and feeds
// eight MACs (four per position): the weight byte is used twice.  The output
// is the P2 layout, which the CPU read-back deinterleaves.
string matvec_q8_p2(int rows, int cols, bool wrow) {
    need_shape("matvec_q8_p2", rows, cols, 4);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4);
    const string CW = to_string(cols);
    // One fragment is one output row for BOTH positions, so the row scale is a
    // single value and one fetch; there is no row pair to read here.
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "float s0=wsr(r);" : "";
    const string SC = wrow ? "s0" : "ws";
    return head(XFormat::I16_2) + x_pair2_glsl() +
        "uniform sampler2D W,X;uniform float ws;"
        + (wrow ? "\n" : "") + WD +
        "void main(){int r=int(gl_FragCoord.x);" + SP + "float sa=0.0,sb=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 t0=texture2D(X,vec2((float(4*q)+0.5)/float(" + CW + "),0.5));"
        "vec4 t1=texture2D(X,vec2((float(4*q+1)+0.5)/float(" + CW + "),0.5));"
        "vec4 t2=texture2D(X,vec2((float(4*q+2)+0.5)/float(" + CW + "),0.5));"
        "vec4 t3=texture2D(X,vec2((float(4*q+3)+0.5)/float(" + CW + "),0.5));"
        "float a0=x_p2_a(t0),a1=x_p2_a(t1),a2=x_p2_a(t2),a3=x_p2_a(t3);"
        "float b0=x_p2_b(t0),b1=x_p2_b(t1),b2=x_p2_b(t2),b3=x_p2_b(t3);"
        "sa+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "sb+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
        "gl_FragColor=x_p2_stq(sa*" + SC + ",sb*" + SC + ");}";
}

// Prefill batching, step 7d.  Step 7c needed a pack pass because the FFN input
// was two per-position I16_2 textures and the packed gate wanted one texture.
// The fix is to build that texture in the layout the gate reads: one stats draw
// over both residuals plus one normalising draw that writes the P2 layout
// (value k of A in .rg, value k of B in .ba).  On VC4 every pass ends in a
// glFinish, so the passes 7c added (two rmsnorms, the pack and four splits) are
// what made the pair path slower than serial; this is two draws for the FFN
// input and no split at all.  The gate also reads P2 directly, which keeps the
// fetch count of matvec_q8_gate2 (four activation texels per iteration) while
// dropping the concatenated-half addressing.
//
// Exactness: the per-position sum of squares is the same 32 partials of 24
// values in the same order as rms_stats_split, and the normalising expression
// is the same R1(X,i)*inv*w[i] encoded by the same x_st2.  Only the layout the
// result lands in changes, so the gate sees the q16 integers the serial
// rmsnorm would have produced.
string rms_stats2(int dim, XFormat rf) {
    need_dim("rms_stats2", dim);
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, rf));
    return head(XFormat::I16_2) + x_residual_glsl(rf) + q24_helpers() +
        "uniform sampler2D XA,XB;"
               + kEncs4096 +
        "void main(){int pid=int(gl_FragCoord.x);int per=(" + D + "+31)/32;"
        "int i0=(pid-(pid/32)*32)*per;"
        "float s=0.0;"
        "for(int k=0;k<per;k++){int i=i0+k;if(i<" + D + "){"
        "float v=(pid<32)?R1(XA,i," + RW + ".0):R1(XB,i," + RW + ".0);s+=v*v;}}"
        "gl_FragColor=encs(s);}";
}

string rms_norm_pair2(int dim, XFormat rf, bool use_eps_uniform) {
    need_dim("rms_norm_pair2", dim);
    const string eps_src = use_eps_uniform ? "eps" : "1e-5";
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, rf));
    const string DW = to_string(x_texels(dim, XFormat::I16_2));
    return head(XFormat::I16_2) + x_residual_glsl(rf) + x_pair2_glsl() + q24_helpers() +
        (use_eps_uniform ? "uniform sampler2D XA,XB,W,S;uniform float eps;"
                         : "uniform sampler2D XA,XB,W,S;") +
        "void main(){int k=int(gl_FragCoord.x);"
        "float ssa=0.0,ssb=0.0;"
        "for(int p=0;p<32;p++){"
        "ssa+=x_i24raw(texture2D(S,vec2((float(p)+0.5)/64.0,0.5)))/4096.0;"
        "ssb+=x_i24raw(texture2D(S,vec2((float(p+32)+0.5)/64.0,0.5)))/4096.0;}"
        "float inva=inversesqrt(ssa/float(" + D + ")+" + eps_src + ");"
        "float invb=inversesqrt(ssb/float(" + D + ")+" + eps_src + ");"
        "vec4 wc=texture2D(W,vec2((float(k/2)+0.5)/float(" + DW + "),0.5));"
        "float wk=((k/2)*2==k)?x_i16(wc.rg):x_i16(wc.ba);"
        "float va=R1(XA,k," + RW + ".0)*inva*wk;"
        "float vb=R1(XB,k," + RW + ".0)*invb*wk;"
        "gl_FragColor=x_p2_st(va,vb);}";
}


// The packed gate on the P2 FFN input: four activation texels per iteration,
// each carrying one value per position, so the fetch count and the MAC order
// are exactly those of matvec_q8_gate2 and the output bytes match it.
string matvec_q8_gate2_p2(int rows, int cols, bool wrow) {
    need_shape("matvec_q8_gate2_p2", rows, cols, 4);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4);
    const string CW = to_string(cols);
    const string WR = "vec2((float(q)+0.5)/float(" + COLS4 + ")," 
                      "(float(r)+0.5)/float(" + ROWS + "))";
    // Two different planes, i.e. two samplers: that is fine on VC4 (the
    // undefined case is two fetches from the *same* sampler).
    const string WD = wrow ? (wrow_scale(rows, "WS1", "wsr1", "ws1") +
                              wrow_scale(rows, "WS2", "wsr2", "ws2")) : "";
    const string SP = wrow ? "float s1=wsr1(r),s2=wsr2(r);" : "";
    const string C1 = wrow ? "s1" : "ws1", C2 = wrow ? "s2" : "ws2";
    // The scale uniforms have to be declared before the helpers that read
    // them (GLSL is not hoisted), which is why WD comes after this string.
    return head(XFormat::I16_2) + x_pair2_glsl() +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD + kQzSl +
        "void main(){int r=int(gl_FragCoord.x);" + SP + 
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 t0=texture2D(X,vec2((float(4*q)+0.5)/float(" + CW + "),0.5));"
        "vec4 t1=texture2D(X,vec2((float(4*q+1)+0.5)/float(" + CW + "),0.5));"
        "vec4 t2=texture2D(X,vec2((float(4*q+2)+0.5)/float(" + CW + "),0.5));"
        "vec4 t3=texture2D(X,vec2((float(4*q+3)+0.5)/float(" + CW + "),0.5));"
        "float a0=x_p2_a(t0),a1=x_p2_a(t1),a2=x_p2_a(t2),a3=x_p2_a(t3);"
        "float b0=x_p2_b(t0),b1=x_p2_b(t1),b2=x_p2_b(t2),b3=x_p2_b(t3);"
        "vec4 c=texture2D(W1," + WR + ");vec4 w=c*255.0-128.0;"
        "s10+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "s11+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
        "c=texture2D(W2," + WR + ");w=c*255.0-128.0;"
        "s20+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "s21+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
        "gl_FragColor=x_p2_st(sl(qz(s10*" + C1 + "),qz(s20*" + C2 + ")),"
        "sl(qz(s11*" + C1 + "),qz(s21*" + C2 + ")));}";
}

// Two-position w2 whose residual is the previous w2 chunk output, which is
// already P2: one texel carries both positions value r, so the add reads one
// texel instead of two and the add itself is the same expression.
string matvec_q8_p2_add_p2(int rows, int cols, bool wrow) {
    need_shape("matvec_q8_p2_add_p2", rows, cols, 4);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), CW = to_string(cols);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "float s0=wsr(r);" : "";
    const string SC = wrow ? "s0" : "ws";
    return head(XFormat::I16_2) + x_pair2_glsl() +
        "uniform sampler2D W,X,RA;uniform float ws;"
        + (wrow ? "\n" : "") + WD +
        "void main(){int r=int(gl_FragCoord.x);" + SP + "float sa=0.0,sb=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + ")," 
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 t0=texture2D(X,vec2((float(4*q)+0.5)/float(" + CW + "),0.5));"
        "vec4 t1=texture2D(X,vec2((float(4*q+1)+0.5)/float(" + CW + "),0.5));"
        "vec4 t2=texture2D(X,vec2((float(4*q+2)+0.5)/float(" + CW + "),0.5));"
        "vec4 t3=texture2D(X,vec2((float(4*q+3)+0.5)/float(" + CW + "),0.5));"
        "float a0=x_p2_a(t0),a1=x_p2_a(t1),a2=x_p2_a(t2),a3=x_p2_a(t3);"
        "float b0=x_p2_b(t0),b1=x_p2_b(t1),b2=x_p2_b(t2),b3=x_p2_b(t3);"
        "sa+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "sb+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
        "vec4 rc=texture2D(RA,vec2((float(r)+0.5)/float(" + ROWS + "),0.5));"
        "gl_FragColor=x_p2_stq(sa*" + SC + "+x_p2_a(rc),sb*" + SC + "+x_p2_b(rc));}";
}


// Two-position w2 with the residual add (step 7c).  The single-position path
// adds the residual *before* rounding (enc16(s*ws+res)); rounding the matvec
// first and adding afterwards is not the same expression and it shows: at
// layer 0 of MiniMind3 (token 0) 2 of 768 q16 values moved by one LSB, where
// the float sum lands inside one ULP of a 0.5 boundary.  So the add has to
// happen inside the kernel, in the same expression order as the serial path.
string matvec_q8_p2_add(int rows, int cols, XFormat rf, bool wrow) {
    need_shape("matvec_q8_p2_add", rows, cols, 4);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), CW = to_string(cols);
    const string RW = to_string(x_texels(rows, rf));
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "float s0=wsr(r);" : "";
    const string SC = wrow ? "s0" : "ws";
    return head(XFormat::I16_2) + x_pair2_glsl() + x_residual_glsl(rf) +
        "uniform sampler2D W,X,RA,RB;uniform float ws;"
        + (wrow ? "\n" : "") + WD +
        "void main(){int r=int(gl_FragCoord.x);" + SP + "float sa=0.0,sb=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + ")," 
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 t0=texture2D(X,vec2((float(4*q)+0.5)/float(" + CW + "),0.5));"
        "vec4 t1=texture2D(X,vec2((float(4*q+1)+0.5)/float(" + CW + "),0.5));"
        "vec4 t2=texture2D(X,vec2((float(4*q+2)+0.5)/float(" + CW + "),0.5));"
        "vec4 t3=texture2D(X,vec2((float(4*q+3)+0.5)/float(" + CW + "),0.5));"
        "float a0=x_p2_a(t0),a1=x_p2_a(t1),a2=x_p2_a(t2),a3=x_p2_a(t3);"
        "float b0=x_p2_b(t0),b1=x_p2_b(t1),b2=x_p2_b(t2),b3=x_p2_b(t3);"
        "sa+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "sb+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;}"
        "float ra=R1Q(RA,r," + RW + ".0),rb=R1Q(RB,r," + RW + ".0);"
        "gl_FragColor=x_p2_stq(sa*" + SC + "+ra,sb*" + SC + "+rb);}";
}

// Split a P2 row back into one position residual stream.  The values are q16
// integers, so this is a re-encode and cannot move a value; it exists because
// a draw has one target and the next pass wants the per-position layout.
string p2_split(int rows, XFormat rf) {
    need_dim("p2_split", rows);
    const string ROWS = to_string(rows), RW = to_string(x_texels(rows, rf));
    const string H = head(XFormat::I16_2) + x_pair2_glsl() + x_residual_glsl(rf);
    if (rf != XFormat::I16_2)
        return H +
            "uniform sampler2D X;uniform float which;"
            "void main(){int i=int(gl_FragCoord.x);"
            "vec4 c=texture2D(X,vec2((float(i)+0.5)/float(" + ROWS + "),0.5));"
            "float q=(which<0.5)?x_p2_a(c):x_p2_b(c);"
            "gl_FragColor=x_st1q(q);}";
    return H +
        "uniform sampler2D X;uniform float which;"
        "void main(){int t=int(gl_FragCoord.x);"
        "int t1=min(2*t+1," + to_string(rows - 1) + ");"
        "vec4 ca=texture2D(X,vec2((float(2*t)+0.5)/float(" + ROWS + "),0.5));"
        "vec4 cb=texture2D(X,vec2((float(t1)+0.5)/float(" + ROWS + "),0.5));"
        "float q0=(which<0.5)?x_p2_a(ca):x_p2_b(ca);"
        "float q1=(which<0.5)?x_p2_a(cb):x_p2_b(cb);"
        "gl_FragColor=x_st2q(q0,q1);}";
}

string matvec_q8_pair_fused(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_q8_pair_fused", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const int rows2 = 2 * rows;
    // W13 holds the gate rows and then the up rows, so a fragment writes row r
    // and row r+rows of one plane.
    const string WD = wrow ? wrow_scale(rows2, "WS", "wsr", "ws", /*hi=*/true) : "";
    const string SP = wrow ? "vec2 s=wsr2(r);" : "";
    const string SC1 = wrow ? "s.x" : "ws", SC2 = wrow ? "s.y" : "ws";
    // The FFN gate/swiglu kernel is the hottest matvec in the token (4864x768
    // per layer) and it was still decoding through the generic X4Q helper.
    // Inline I16_2 decode is the same arithmetic with the fetches emitted in
    // the loop body; matvec_probe measures 283 vs 177 MMAC/s for that shape.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;" + WD
               + kDec16A + kDec16B
               + kEnc16Mul +
               "void main(){int r=int(gl_FragCoord.x);" + SP + "float s1=0.0,s2=0.0;"
               "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + to_string(axw) + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + to_string(axw) + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "vec4 c1=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
               "(float(r)+0.5)/float(" + to_string(rows2) + ")));"
               "vec4 c2=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
               "(float(r+" + to_string(rows) + ")+0.5)/float(" + to_string(rows2) + ")));"
               "vec4 w1=c1*255.0-128.0;vec4 w2=c2*255.0-128.0;"
               "s1+=w1.x*a0+w1.y*a1+w1.z*a2+w1.w*a3;"
               "s2+=w2.x*a0+w2.y*a1+w2.z*a2+w2.w*a3;}"
               "vec2 p1=enc16(s1," + SC1 + "),p2=enc16(s2," + SC2 + ");"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);" + SP + "float s1=0.0,s2=0.0;"
        "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
        "vec4 c1=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(r)+0.5)/float(" + to_string(rows2) + ")));"
        "vec4 c2=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(r+" + to_string(rows) + ")+0.5)/float(" + to_string(rows2) + ")));"
        "vec4 w1=c1*255.0-128.0;vec4 w2=c2*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + to_string(axw) + ".0);"
        "s1+=w1.x*x.x+w1.y*x.y+w1.z*x.z+w1.w*x.w;"
        "s2+=w2.x*x.x+w2.y*x.y+w2.z*x.z+w2.w*x.w;}"
        "gl_FragColor=x_st2q(s1*" + SC1 + ",s2*" + SC2 + ");}";
}

string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf, bool wrow) {
    need_shape("matvec_q8_add", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    if (xf == XFormat::I16_2 && rf == XFormat::I16_1) {
        // Inline form: VC4 compiles this noticeably faster than the generic
        // helper-based version (the helper takes two texel fetches as arguments).
        return "precision highp float;uniform sampler2D W,X,R;uniform float ws;" + WD +
               kDec16A + kDec16B +
               "float dr(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "if(u>=32768.0)u-=65536.0;return u/256.0;}"
               "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r)+0.5)/float(" + ROWS + ")));"
               "vec4 w=c*255.0-128.0;"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
               "float v=clamp(floor(s*" + SC + "+0.5),-32768.0,32767.0);"
               "vec4 rc=texture2D(R,vec2((float(r)+0.5)/float(" + ROWS + "),0.5));"
               "float oq=v+dr(rc)*256.0;"
               "float q=clamp(floor(oq+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
               "float hi=floor(q/256.0),lo=q-hi*256.0;"
               "gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
    }
    return head(xf) + x_residual_glsl(rf) +
        "uniform sampler2D W,X,R;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "float res=R1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=XST1Q(s*" + SC + "+res);}";
}

string matvec_q8_add_pair(int rows, int cols, XFormat xf, bool inl, bool wrow) {
    need_shape("matvec_q8_add_pair", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const int rw = (rows + 1) / 2;
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    // Two rows per fragment, so two scale rows: r0 even, r1 odd, exactly the
    // rows the packed residual texel carries.
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "vec2 sc=wsr2(r0);" : "";
    const string SC0 = wrow ? "sc.x" : "ws", SC1 = wrow ? "sc.y" : "ws";
    // Inline I16_2 decode, the form matvec_q8_pair already uses: the generic
    // X4Q helper takes its two texel fetches as macro arguments and VC4 compiles
    // that loop measurably slower.  r0 is always even and r1 always odd, so the
    // two residuals are the halves of one I16_2 texel.  Accumulation order and
    // the q16 rounding are unchanged, so the packed output is bit-identical.
    if (inl && xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X,R;uniform float ws;" + WD +
               kDec16A + kDec16B
               + kEnc16Raw +
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
               "vec4 rc=texture2D(R,vec2((float(t)+0.5)/float(" + to_string(rw) + "),0.5));"
               "float res0=dA(rc),res1=dB(rc);"
               "float s0=0.0,s1=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 c0=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r0)+0.5)/float(" + ROWS + ")));"
               "vec4 c1=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(min(r1," + to_string(rows - 1) + "))+0.5)/float(" + ROWS + ")));"
               "vec4 w0=c0*255.0-128.0;vec4 w1=c1*255.0-128.0;"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "s0+=w0.x*a0+w0.y*a1+w0.z*a2+w0.w*a3;"
               "s1+=w1.x*a0+w1.y*a1+w1.z*a2+w1.w*a3;}"
               "vec2 p1=enc16(s0*" + SC0 + "+res0),p2=enc16(s1*" + SC1 + "+res1);"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) + x_residual_glsl(XFormat::I16_2) +
        "uniform sampler2D W,X,R;uniform float ws;" + WD +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
        "float res0=R1Q(R,r0," + to_string(rw) + ".0);"
        "float res1=R1Q(R,r1," + to_string(rw) + ".0);"
        "float s0=0.0,s1=0.0;"
        "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
        "vec4 c0=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(r0)+0.5)/float(" + to_string(rows) + ")));"
        "vec4 c1=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(min(r1," + to_string(rows - 1) + "))+0.5)/float(" + to_string(rows) + ")));"
        "vec4 w0=c0*255.0-128.0;vec4 w1=c1*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + to_string(axw) + ".0);"
        "s0+=w0.x*x.x+w0.y*x.y+w0.z*x.z+w0.w*x.w;"
        "s1+=w1.x*x.x+w1.y*x.y+w1.z*x.z+w1.w*x.w;}"
        "gl_FragColor=x_st2q(s0*" + SC0 + "+res0,s1*" + SC1 + "+res1);}";
}

string matvec_i16(int rows, int cols, XFormat xf) {
    need_shape("matvec_i16", rows, cols, 1);
    const int tw = (cols + 1) / 2;
    const int axw = x_texels(cols, xf);
    return head(xf) + q24_helpers() +
        "uniform sampler2D W,X;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + to_string(tw) + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float u0=x_i16raw(c.gr);float u1=x_i16raw(c.ab);"
        "vec2 x=X2Q(X,2*q," + to_string(axw) + ".0);"
        "s+=u0*x.x;if(2*q+1<" + to_string(cols) + ")s+=u1*x.y;}"
        "gl_FragColor=x_i24encq(s*ws);}";
}

string matvec_i16_add(int rows, int cols, XFormat xf, XFormat rf) {
    need_shape("matvec_i16_add", rows, cols, 1);
    const int tw = (cols + 1) / 2;
    const int axw = x_texels(cols, xf);
    return head(xf) + x_residual_glsl(rf) +
        "uniform sampler2D W,X,R;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + to_string(tw) + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float u0=x_i16raw(c.gr);float u1=x_i16raw(c.ab);"
        "vec2 x=X2Q(X,2*q," + to_string(axw) + ".0);"
        "s+=u0*x.x;if(2*q+1<" + to_string(cols) + ")s+=u1*x.y;}"
        "float res=R1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=XST1Q(s*ws+res);}";
}

string matvec_i4(int rows, int cols, XFormat xf, bool inl) {
    need_shape("matvec_i4", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    return q4_head(xf, inl) + q24_helpers() +
        "uniform sampler2D W,P,X;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int g=0;g<" + to_string(groups) + ";g++){"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float sc=x_i16raw(p.rg)*ps;float mn=x_i16raw(p.ba)*ps;"
        "for(int t=0;t<4;t++){int c=g*32+t*8;"
        "vec4 w=texture2D(W,vec2((float(g*4+t)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        + q4_taps("w")
        + q4_act(xf, cols, inl) +
        // (n*sc + mn)*x sums to sc*sum(n*x) + mn*sum(x): the min term is paid
        // once per group instead of once per weight, and the activation sum
        // reuses the fetches already made.
        "float d0=l0*xa.x+h0*xa.y;"
        "float d1=l1*xa.z+h1*xa.w;"
        "float d2=l2*xb.x+h2*xb.y;"
        "float d3=l3*xb.z+h3*xb.w;"
        "s+=sc*(d0+d1+d2+d3)" + q4_mint() + ";}}"
        "gl_FragColor=x_i24encq(s);}";
}

string matvec_i6(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_i6", rows, cols, 8);
    if (xf != XFormat::I16_2)
        { fprintf(stderr, "shader matvec_i6: q6 head needs I16_2 activations\n"); exit(2); }
    const int axw = x_texels(cols, xf);
    const int lo_w = cols / 8;    // one texel = 4 bytes = 8 nibbles
    const int hi_w = cols / 16;   // one texel = 4 bytes = 16 two-bit values
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    // The scale uniforms must be declared before the wrow helper that uses
    // them (the q8 kernels order it that way; GLSL has no forward declaration).
    return head(xf) + q24_helpers() +
        "uniform sampler2D W,W2,X;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;float sc=" + SC + ";"
        "for(int q=0;q<" + to_string(lo_w) + ";q++){"
        "vec4 lo=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(lo_w) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "vec4 hc=texture2D(W2,vec2((float(q/2)+0.5)/float(" + to_string(hi_w) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float b0=floor(lo.r*255.0+0.5),b1=floor(lo.g*255.0+0.5),"
        "b2=floor(lo.b*255.0+0.5),b3=floor(lo.a*255.0+0.5);"
        "float l0=mod(b0,16.0),l1=floor(b0/16.0),l2=mod(b1,16.0),l3=floor(b1/16.0),"
        "l4=mod(b2,16.0),l5=floor(b2/16.0),l6=mod(b3,16.0),l7=floor(b3/16.0);"
        // even q takes the hi texel's .rg (values 0..7), odd q its .ba (8..15)
        "vec2 hb=(q-(q/2)*2==0)?hc.rg:hc.ba;"
        "float h0=floor(hb.x*255.0+0.5),h1=floor(hb.y*255.0+0.5);"
        "float g0=mod(h0,4.0),g1=mod(floor(h0/4.0),4.0),g2=mod(floor(h0/16.0),4.0),"
        "g3=floor(h0/64.0),g4=mod(h1,4.0),g5=mod(floor(h1/4.0),4.0),"
        "g6=mod(floor(h1/16.0),4.0),g7=floor(h1/64.0);"
        "float v0=l0+g0*16.0,v1=l1+g1*16.0,v2=l2+g2*16.0,v3=l3+g3*16.0,"
        "v4=l4+g4*16.0,v5=l5+g5*16.0,v6=l6+g6*16.0,v7=l7+g7*16.0;"
        "v0=v0>=32.0?v0-64.0:v0;v1=v1>=32.0?v1-64.0:v1;"
        "v2=v2>=32.0?v2-64.0:v2;v3=v3>=32.0?v3-64.0:v3;"
        "v4=v4>=32.0?v4-64.0:v4;v5=v5>=32.0?v5-64.0:v5;"
        "v6=v6>=32.0?v6-64.0:v6;v7=v7>=32.0?v7-64.0:v7;"
        "vec4 xa=X4Q(X,q*8," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,q*8+4," + to_string(axw) + ".0);"
        "s+=v0*xa.x+v1*xa.y+v2*xa.z+v3*xa.w+v4*xb.x+v5*xb.y+v6*xb.z+v7*xb.w;}"
        "gl_FragColor=x_i24encq(s*sc);}";
}

// q4 residual-add, two rows per fragment: rows 2t and 2t+1 share one activation
// decode and one residual fetch, and are written as a raw q16 pair (x_st2q, since
// the q4 accumulator works in raw q16 units, not activation units).
string matvec_i4_add_pair(int rows, int cols, XFormat xf, bool inl) {
    need_shape("matvec_i4_add_pair", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int rw = (rows + 1) / 2;
    return q4_head(xf, inl) + x_residual_glsl(XFormat::I16_2) +
        "uniform sampler2D W,P,X,R;uniform float ps;"
        "float dot4(vec4 w0,vec4 xa,vec4 xb){"
        + q4_taps("w0") +
        "return l0*xa.x+h0*xa.y+l1*xa.z+h1*xa.w+l2*xb.x+h2*xb.y+l3*xb.z+h3*xb.w;}"
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;"
        "float s0=0.0,s1=0.0;"
        "for(int g=0;g<" + to_string(groups) + ";g++){"
        "vec4 pa=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r0)+0.5)/float(" + to_string(rows) + ")));"
        "vec4 pb=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r1)+0.5)/float(" + to_string(rows) + ")));"
        "float sa=x_i16raw(pa.rg)*ps,ma=x_i16raw(pa.ba)*ps;"
        "float sb=x_i16raw(pb.rg)*ps,mb=x_i16raw(pb.ba)*ps;"
        "float xsum=0.0,dota=0.0,dotb=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;"
        + q4_act(xf, cols, inl) +
        q4_xsum() +
        "vec4 w0=texture2D(W,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r0)+0.5)/float(" + to_string(rows) + ")));"
        "dota+=dot4(w0,xa,xb);"
        "w0=texture2D(W,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r1)+0.5)/float(" + to_string(rows) + ")));"
        "dotb+=dot4(w0,xa,xb);}"
        "s0+=sa*dota+ma*xsum;s1+=sb*dotb+mb*xsum;}"
        "float res0=R1Q(R,r0," + to_string(rw) + ".0),res1=R1Q(R,r1," + to_string(rw) + ".0);"
        "gl_FragColor=x_st2q(s0+res0,s1+res1);}";
}

// q4 FFN gate, paired: one fragment computes w1[r] and w3[r], sharing the
// activation decode, and writes both as a raw q16 pair for swiglu.
string matvec_i4_pair(int rows, int cols, XFormat xf) {
    need_shape("matvec_i4_pair", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    return head(xf) +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;"
        "float dot4(vec4 w0,vec4 xa,vec4 xb){"
        + q4_taps("w0") +
        "return l0*xa.x+h0*xa.y+l1*xa.z+h1*xa.w+l2*xb.x+h2*xb.y+l3*xb.z+h3*xb.w;}"
        "void main(){int r=int(gl_FragCoord.x);float s1=0.0,s2=0.0;"
        "for(int g=0;g<" + to_string(groups) + ";g++){"
        "vec4 q1=texture2D(P1,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "vec4 q2=texture2D(P2,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float sa1=x_i16raw(q1.rg)*ps1,ma1=x_i16raw(q1.ba)*ps1;"
        "float sa2=x_i16raw(q2.rg)*ps2,ma2=x_i16raw(q2.ba)*ps2;"
        "float xsum=0.0,d1=0.0,d2=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;"
        "vec4 xa=X4Q(X,c," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,c+4," + to_string(axw) + ".0);"
        + q4_xsum() +
        "vec4 w0=texture2D(W1,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "d1+=dot4(w0,xa,xb);"
        "w0=texture2D(W2,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "d2+=dot4(w0,xa,xb);}"
        "s1+=sa1*d1+ma1*xsum;s2+=sa2*d2+ma2*xsum;}"
        "gl_FragColor=x_st2q(s1,s2);}";
}

// q4 FFN gate + swiglu fused, two rows per fragment.  The row pair r0/r1
// shares one activation decode and one xsum; each row w1/w3 dots are
// swiglu-d in-kernel and the two results are written as one q16 pair, so the
// chunk costs one draw instead of (matvec_i4_pair -> swiglu_pair) and the
// fragment count halves.  The accumulator keeps the raw-q16 unit convention
// and the per-group order (scale*dot + min*xsum) of matvec_i4_pair; qz/sl
// reproduce x_st2q -> x_i16pair -> x_st2 exactly, so the packed output is
// bit-identical to the two-pass path.
string matvec_i4_pair_swiglu(int rows, int cols, XFormat xf, bool inl) {
    need_shape("matvec_i4_pair_swiglu", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), GROUPS = to_string(groups),
                 TEXW = to_string(texw), AXW = to_string(axw);
    // Q4_MV4=1 selects the vec4 tap/dot form.  The default scalar path emits
    // exactly the GLSL this kernel shipped with: same strings, same order.
    const bool mv4 = q4_mv4() && gpu_q4_shift() && !q4_nounpack();
    const string ACC = mv4
        ? ";vec4 d1=vec4(0.0),d2=vec4(0.0),e1=vec4(0.0),e2=vec4(0.0);"
          "vec4 w0;vec4 bb;ivec4 ib;vec4 hi,lo;"
        : ",d1=0.0,d2=0.0,e1=0.0,e2=0.0;";
    // mv4 flattens the four vec4 accumulators once per group, so the
    // horizontal add costs four sums per group instead of twelve per texel.
    const string SUM = mv4
        ? "float f1=d1.x+d1.y+d1.z+d1.w,f2=d2.x+d2.y+d2.z+d2.w,"
          "f3=e1.x+e1.y+e1.z+e1.w,f4=e2.x+e2.y+e2.z+e2.w;"
          "s10+=sa1*f1+ma1*xsum;s20+=sa2*f2+ma2*xsum;"
          "s11+=sb1*f3+mb1*xsum;s21+=sb2*f4+mb2*xsum;}"
        : "s10+=sa1*d1+ma1*xsum;s20+=sa2*d2+ma2*xsum;"
          "s11+=sb1*e1+mb1*xsum;s21+=sb2*e2+mb2*xsum;}";
    // The first scalar tap declares w0 and the rest assign it: that is the
    // exact text of the kernel before Q4_MV4 existed, so the default path
    // compiles to the same program (argument order is not evaluation order,
    // hence the explicit index).
    auto wtap = [&](const char* dst, const char* W, const char* row, int idx) {
        const bool decl = !mv4 && idx == 0;
        string s = string(decl ? "vec4 w0=texture2D(" : "w0=texture2D(") + W +
                   ",vec2((float(g*4+q)+0.5)/float(" + TEXW + "),(u" + row +
                   "+0.5)/float(" + ROWS + ")));";
        if (mv4) return s + q4_taps4("w0") + dst + "+=lo*xe+hi*xo;";
        return s + dst + "+=dot4(w0,xa,xb);";
    };
    return q4_head(xf, inl) +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;" +
        (mv4 ? string()
             : "float dot4(vec4 w0,vec4 xa,vec4 xb){" + q4_taps("w0") +
               "return l0*xa.x+h0*xa.y+l1*xa.z+h1*xa.w+l2*xb.x+h2*xb.y+l3*xb.z+h3*xb.w;}")
        + kQzSl +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
        "int r1=min(r0+1," + to_string(rows - 1) + ");"
        "float u0=float(r0),u1=float(r1);"
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int g=0;g<" + GROUPS + ";g++){"
        "vec4 q1=texture2D(P1,vec2((float(g)+0.5)/float(" + GROUPS + "),(u0+0.5)/float(" + ROWS + ")));"
        "vec4 q2=texture2D(P2,vec2((float(g)+0.5)/float(" + GROUPS + "),(u0+0.5)/float(" + ROWS + ")));"
        "vec4 p1=texture2D(P1,vec2((float(g)+0.5)/float(" + GROUPS + "),(u1+0.5)/float(" + ROWS + ")));"
        "vec4 p2=texture2D(P2,vec2((float(g)+0.5)/float(" + GROUPS + "),(u1+0.5)/float(" + ROWS + ")));"
        "float sa1=x_i16raw(q1.rg)*ps1,ma1=x_i16raw(q1.ba)*ps1;"
        "float sa2=x_i16raw(q2.rg)*ps2,ma2=x_i16raw(q2.ba)*ps2;"
        "float sb1=x_i16raw(p1.rg)*ps1,mb1=x_i16raw(p1.ba)*ps1;"
        "float sb2=x_i16raw(p2.rg)*ps2,mb2=x_i16raw(p2.ba)*ps2;"
        "float xsum=0.0" + ACC +
        "for(int q=0;q<4;q++){int c=g*32+q*8;" +
        (mv4 ? q4_act8(xf, cols, inl) + q4_xsum8()
             : q4_act(xf, cols, inl) + q4_xsum()) +
        wtap("d1", "W1", "0", 0) + wtap("d2", "W2", "0", 1) +
        wtap("e1", "W1", "1", 2) + wtap("e2", "W2", "1", 3) + "}" + SUM +
        "float v0=sl(qz(s10),qz(s20));"
        "float v1=(r0+1<" + ROWS + ")?sl(qz(s11),qz(s21)):0.0;"
        "gl_FragColor=x_st2(v0,v1);}";
}


string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf) {
    need_shape("matvec_i4_add", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    return head(xf) + x_residual_glsl(rf) +
        "uniform sampler2D W,P,X,R;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int g=0;g<" + to_string(groups) + ";g++){"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float sc=x_i16raw(p.rg)*ps;float mn=x_i16raw(p.ba)*ps;"
        "for(int t=0;t<4;t++){int c=g*32+t*8;"
        "vec4 w=texture2D(W,vec2((float(g*4+t)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        + q4_taps("w") +
        "vec4 xa=X4Q(X,c," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,c+4," + to_string(axw) + ".0);"
        // (n*sc + mn)*x sums to sc*sum(n*x) + mn*sum(x): the min term is paid
        // once per group instead of once per weight, and the activation sum
        // reuses the fetches already made.
        "float d0=l0*xa.x+h0*xa.y;"
        "float d1=l1*xa.z+h1*xa.w;"
        "float d2=l2*xb.x+h2*xb.y;"
        "float d3=l3*xb.z+h3*xb.w;"
        "s+=sc*(d0+d1+d2+d3)" + q4_mint() + ";}}"
        "float res=R1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=XST1Q(s+res);}";
}


string swiglu_pair(int hidden, XFormat out) {
    need_dim("swiglu_pair", hidden);
    const string H = to_string(hidden);
    string act =
        "float actv(int i){if(i>=" + H + ")return 0.0;"
        "vec2 g=X2(A,2*i," + H + ".0);"
        "return (g.x/(1.0+exp(-g.x)))*g.y;}";
    string body;
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(actv(i0),actv(i0+1),actv(i0+2),actv(i0+3));}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(actv(i0),actv(i0+1));}";
    }
    return head(XFormat::I16_2) + "uniform sampler2D A;" + act + body;
}

string swiglu_2row(int hidden, XFormat out) {
    need_dim("swiglu_2row", hidden);
    const string H = to_string(hidden);
    string read =
        "float yat(int i){float x=float(i);"
        "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + H + "),0.25));"
        "vec4 cb=texture2D(A,vec2((x+0.5)/float(" + H + "),0.75));"
        "float a=x_i24q(ca),b=x_i24q(cb);"
        "return (a/(1.0+exp(-a)))*b;}";
    string body;
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(yat(i0),yat(i0+1),yat(i0+2),yat(i0+3));}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(yat(i0),yat(i0+1));}";
    }
    return head(XFormat::I16_2) + q24_helpers() + "uniform sampler2D A;" + read + body;
}

string swiglu(int hidden, bool fused, XFormat out) {
    need_dim("swiglu", hidden);
    const string H = to_string(hidden);
    string read;
    if (fused) {
        read = "uniform sampler2D A;"
               "float yat(int i){float x=float(i);"
               "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + to_string(2 * hidden) + "),0.5));"
               "vec4 cb=texture2D(A,vec2((x+0.5+float(" + H + "))/float(" + to_string(2 * hidden) + "),0.5));"
               "float a=x_i24q(ca),b=x_i24q(cb);"
               "return (a/(1.0+exp(-a)))*b;}";
    } else {
        read = "uniform sampler2D A,B;"
               "float yat(int i){float x=float(i);"
               "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + H + "),0.5));"
               "vec4 cb=texture2D(B,vec2((x+0.5)/float(" + H + "),0.5));"
               "float a=x_i24q(ca),b=x_i24q(cb);"
               "return (a/(1.0+exp(-a)))*b;}";
    }
    string body;
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(yat(i0),yat(i0+1),yat(i0+2),yat(i0+3));}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(yat(i0),yat(i0+1));}";
    }
    return head(XFormat::I16_2) + q24_helpers() + read + body;
}

string rms_stats_split(int dim, XFormat residual) {
    need_dim("rms_stats_split", dim);
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, residual));
    return head(residual) + x_residual_glsl(residual) +
        "uniform sampler2D X;"
               + kEncs4096 +
        "void main(){int pid=int(gl_FragCoord.x);int per=(" + D + "+31)/32;"
        "int i0=pid*per;float s=0.0;"
        "for(int k=0;k<per;k++){int i=i0+k;if(i<" + D + "){"
        "float v=R1(X,i," + RW + ".0);s+=v*v;}}"
        "gl_FragColor=encs(s);}";
}

string rms_stats_reduce() {
    return "precision highp float;\n"
        "uniform sampler2D S;"
        "float raw(vec4 c){return floor(c.r*255.0+0.5)*16777216.0+"
        "floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+"
        "floor(c.a*255.0+0.5)-8388608.0;}"
               + kEncs4096 +
        "void main(){float s=0.0;for(int p=0;p<32;p++){"
        "s+=raw(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)))/4096.0;}"
        "gl_FragColor=encs(s);}";
}

// One elementwise multiply of two conv-block activations, two values per
// fragment.  A and B are not always the same width: the in_proj gate reads two
// matvec outputs (24-bit, one value per texel), while the second gate reads a
// matvec output and the convolution's result (q16, two values per texel).  The
// decoder is chosen per operand, so an I16_2 side costs one fetch where a
// 24-bit side costs two.
static string mul_act_read(const char* tex, const char* var, XFormat f, int dim, int tw) {
    // GLSL for `vec2 var` = the two values at output texel q, in activation
    // units: an I16_2 texel already holds the pair, a 24-bit one holds a value
    // per texel, so it takes two fetches and no /256.
    const string TW = to_string(tw), AW = to_string(dim);
    if (f == XFormat::I16_2)
        return string("vec2 ") + var + "=x_i16pairq(texture2D(" + tex +
               ",vec2((float(q)+0.5)/" + TW + ".0,0.5)))/256.0;";
    // Both operands can be 24-bit, so the coordinate gets a per-operand name.
    return string("vec2 ") + var + "=vec2(0.0);{float t=(float(2*q)+0.5)/" + AW + ".0;"
                  + var + "=vec2(x_i24q(texture2D(" + tex + ",vec2(t,0.5))),"
               "x_i24q(texture2D(" + tex + ",vec2(t+1.0/" + AW + ".0,0.5))));}";
}

string mul_act(int dim, XFormat xf, XFormat af, XFormat bf) {
    if (xf != XFormat::I16_2)
        { fprintf(stderr, "shader mul_act: the conv block is q16 only\n"); exit(2); }
    const int tw = x_texels(dim, xf);
    // Output is the q16 pair the next matvec_add reads.  A 24-bit operand is
    // real-valued (x_i24q), so the product is real and x_st2 is the right store
    // for both mixes; an I16_2 operand is raw q16, hence the /256.
    const string ra = mul_act_read("A", "av", af, dim, x_texels(dim, af));
    const string rb = mul_act_read("B", "bv", bf, dim, x_texels(dim, bf));
    return head(xf) + q24_helpers() +
        "uniform sampler2D A,B;"
        "void main(){int q=int(gl_FragCoord.x);" + ra + rb +
        "gl_FragColor=x_st2(av.x*bv.x,av.y*bv.y);}";
}

string conv3_act(int dim, XFormat xf) {
    if (xf != XFormat::I16_2)
        { fprintf(stderr, "shader conv3_act: the conv block is q16 only\n"); exit(2); }
    const int tw = x_texels(dim, xf);
    // Two channels per fragment: every input is an I16_2 texel holding the pair
    // (2q, 2q+1), and so is the depthwise kernel row, so each of the four
    // textures is fetched once instead of once per channel.  The result is the
    // q16 pair the second gate reads -- one rounding here instead of a 24-bit
    // store and a decode in the gate, at the cost of the sub-q16 fraction that
    // the gate then had to multiply anyway.
    return head(xf) + q24_helpers() +
        "uniform sampler2D S0,S1,G,K;"
        "void main(){int q=int(gl_FragCoord.x);"
        "float u=(float(q)+0.5)/float(" + to_string(tw) + ");"
        "vec2 a=x_i16pairq(texture2D(S0,vec2(u,0.5)));"
        "vec2 b=x_i16pairq(texture2D(S1,vec2(u,0.5)));"
        "vec2 c=x_i16pairq(texture2D(G,vec2(u,0.5)));"
        // Row t of K holds tap t for channels 2q and 2q+1, so the pair decode
        // is the kernel row's own layout.
        "vec2 k0=x_i16pairq(texture2D(K,vec2(u,0.5/3.0)));"
        "vec2 k1=x_i16pairq(texture2D(K,vec2(u,1.5/3.0)));"
        "vec2 k2=x_i16pairq(texture2D(K,vec2(u,2.5/3.0)));"
        // Raw q16 units: a raw product is real*256, so three of them come back
        // down by 256; 0.00390625 is exact scaling (a power of two), and it
        // avoids the reciprocal the driver emits for a division.
        "vec2 cv=(k0*a+k1*b+k2*c)*0.00390625;"
        "gl_FragColor=x_st2q(cv.x,cv.y);}";
}

string copy_act(int dim, XFormat xf) {
    const int tw = x_texels(dim, xf);
    return head(xf) +
        "uniform sampler2D A;"
        "void main(){int q=int(gl_FragCoord.x);"
        "gl_FragColor=texture2D(A,vec2((float(q)+0.5)/float(" + to_string(tw) + "),0.5));}";
}

string rms_norm(int dim, XFormat residual, XFormat out, bool fused_reduce,
                bool use_eps_uniform) {
    need_dim("rms_norm", dim);
    const string eps_src = use_eps_uniform ? "eps" : "1e-5";
    // Fusing the 32-way reduce into every normalising fragment keeps the exact
    // summation order, so the result is bit-identical to the separate pass; it
    // trades one draw for 32 texture reads per output fragment.
    const string ss_read = fused_reduce
        ? "float ss=0.0;for(int p=0;p<32;p++){"
          "ss+=x_i24raw(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)))/4096.0;}"
        : "float ss=x_i24raw(texture2D(S,vec2(0.5,0.5)))/4096.0;";
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, residual));
    const string DW = to_string(x_texels(dim, XFormat::I16_2));
    // W is packed two q16 per texel, matching the output granularity.
    const string wsplit =
        "vec4 wc=texture2D(W,vec2((float(t)+0.5)/float(" + DW + "),0.5));"
        "float w0=x_i16(wc.rg);float w1=x_i16(wc.ba);";
    string body;
    if (out == XFormat::I8_4) {
        body =
            "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
            + ss_read +
            "float inv=inversesqrt(ss/float(" + D + ")+" + eps_src + ");"
            "vec4 wa=texture2D(W,vec2((float(2*t)+0.5)/float(" + DW + "),0.5));"
            "vec4 wb=texture2D(W,vec2((float(2*t+1)+0.5)/float(" + DW + "),0.5));"
            "float v0=R1(X,i0," + RW + ".0)*inv*x_i16(wa.rg);"
            "float v1=R1(X,i0+1," + RW + ".0)*inv*x_i16(wa.ba);"
            "float v2=R1(X,i0+2," + RW + ".0)*inv*x_i16(wb.rg);"
            "float v3=R1(X,i0+3," + RW + ".0)*inv*x_i16(wb.ba);"
            "gl_FragColor=x_i8enc4(v0,v1,v2,v3);}";
    } else {
        body =
            "void main(){int t=int(gl_FragCoord.x);int i0=2*t;int i1=i0+1;"
            + ss_read +
            "float inv=inversesqrt(ss/float(" + D + ")+" + eps_src + ");" + wsplit +
            "float v0=R1(X,i0," + RW + ".0)*inv*w0;"
            "float v1=0.0;if(i1<" + D + ")v1=R1(X,i1," + RW + ".0)*inv*w1;"
            "gl_FragColor=x_st2(v0,v1);}";
    }
    return head(residual) + x_residual_glsl(residual) + q24_helpers() +
        "uniform sampler2D X,W,S;uniform float dummy;" + (use_eps_uniform ? "uniform float eps;" : string()) + body;
}
