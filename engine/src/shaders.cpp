#include "shaders.h"

#include <string>
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
        "float x_i24raw(vec4 c){return floor(c.r*255.0+0.5)*16777216.0+"
        "floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+"
        "floor(c.a*255.0+0.5)-8388608.0;}\n"
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

static string q4_head(XFormat xf, bool inl) {
    if (!inl || xf != XFormat::I16_2) return head(xf);
    return head(xf) +
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "return u>=32768.0?u-65536.0:u;}\n"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "return u>=32768.0?u-65536.0:u;}\n";
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

// Default-off probe (Q4_NOUNPACK=1): swap the nibble unpack for a handful of
// multiplies (the weights are nonsense on purpose) to price the four fetches
// plus the floor chain.  A large win here is the only thing that would justify
// a hi/lo split weight format, which needs a re-quantised model.
static bool q4_nounpack() {
    static const char* e = getenv("Q4_NOUNPACK");
    static const bool v = e && strcmp(e, "0") != 0;
    return v;
}

// The unpack of one 8-weight texel, shared verbatim by the INT4 kernels so the
// probe swaps it in one place.  q4_h is defined below; W is the texel name.
static string q4_h(const string& w);
static string q4_taps(const char* W) {
    const string w(W);
    if (q4_nounpack())
        return "float h0=" + w + ".r*15.0,l0=" + w + ".g*15.0,h1=" + w + ".b*15.0,"
               "l1=" + w + ".a*15.0,h2=" + w + ".r*7.0,l2=" + w + ".g*7.0,"
               "h3=" + w + ".b*7.0,l3=" + w + ".a*7.0;";
    return "float b0=floor(" + w + ".r*255.0+0.5),b1=floor(" + w + ".g*255.0+0.5),"
           "b2=floor(" + w + ".b*255.0+0.5),b3=floor(" + w + ".a*255.0+0.5);"
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

string matvec_q8(int rows, int cols, XFormat xf) {
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    // VC4's shader compiler generates a slower loop when the activation decode
    // goes through a helper that takes two texel fetches as arguments.  Emitting
    // the fetches and the two scalar decoders inline is measurably faster, so
    // the I16_2 fast path keeps the inline form.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
               "(float(r)+0.5)/float(" + ROWS + ")));"
               "vec4 w=c*255.0-128.0;"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
               "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;"
               "v+=8388608.0;"
               "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
               "float b=floor(v/256.0)-r0*65536.0-g*256.0;"
               "float a=v-r0*16777216.0-g*65536.0-b*256.0;"
               "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
    }
    return head(xf) + q24_helpers() +
        "uniform sampler2D W,X;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_i24encq(s*ws);}";
}

string matvec_q8_pair(int rows, int cols, XFormat xf) {
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W1,W2,X;uniform float ws1,ws2;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "vec2 enc16(float s,float w){float v=clamp(floor(s*w+0.5),-32768.0,32767.0);"
               "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
               "return vec2(hi/255.0,(v-hi*256.0)/255.0);}"
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
               "vec2 p1=enc16(s1,ws1),p2=enc16(s2,ws2);"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;"
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
        "gl_FragColor=x_st2q(s1*ws1,s2*ws2);}";
}

// q8 vocabulary projection, two output rows per fragment.  One activation decode
// feeds two dot products, so both the fragment count and the activation decode
// halve: 1536 vocabulary rows become 768 fragments.  Each accumulator sums in
// exactly the order matvec_q8 uses, and x_st2q applies the same q16 rounding and
// clamp as x_i24encq, so the packed pair holds the same two integers that the
// unpacked pass writes.
string matvec_q8_pair_logits(int rows, int cols, XFormat xf) {
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WR0 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r0)+0.5)/float(" + ROWS + "))";
    const string WR1 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r1)+0.5)/float(" + ROWS + "))";
    // Inline I16_2 decode.  The generic X4Q helper takes two texel fetches as
    // arguments and VC4 compiles that loop much slower (matvec_probe measured
    // 177 vs 283 MMAC/s at 1536x768).  Decode, arithmetic and the q16 rounding
    // are identical, so the packed output is byte-identical.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "vec2 enc16(float s,float w){float v=clamp(floor(s*w+0.5),-32768.0,32767.0);"
               "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
               "return vec2(hi/255.0,(v-hi*256.0)/255.0);}"
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;float s0=0.0,s1=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){"
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/float(" + AXW + "),0.5));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/float(" + AXW + "),0.5));"
               "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
               "vec4 c=texture2D(W," + WR0 + ");vec4 w=c*255.0-128.0;"
               "s0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
               "c=texture2D(W," + WR1 + ");w=c*255.0-128.0;"
               "s1+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;}"
               "vec2 p1=enc16(s0,ws),p2=enc16(s1,ws);"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;"
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;float s0=0.0,s1=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "vec4 c=texture2D(W," + WR0 + ");vec4 w=c*255.0-128.0;"
        "s0+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;"
        "c=texture2D(W," + WR1 + ");w=c*255.0-128.0;"
        "s1+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_st2q(s0*ws,s1*ws);}";
}

// q8 FFN gate + swiglu fused: one fragment evaluates two consecutive gate rows
// for both w1 and w3 (four dots sharing one activation decode) and writes the two
// swiglu results as one q16 pair.  This collapses the two-draw sequence
// (pair matvec -> swiglu) into a single draw, and halving the fragment count keeps
// the viewport at rows/2.  The gate quantisation (int16 of s*ws) and the final q16
// rounding are the same operations the separate passes perform, so the output is
// bit-identical.
string matvec_q8_pair_swiglu(int rows, int cols, XFormat xf, bool inl) {
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WR0 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r0)+0.5)/float(" + ROWS + "))";
    const string WR1 = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r1)+0.5)/float(" + ROWS + "))";
    // Inline I16_2 decode.  The generic X4Q macro takes its two texel fetches as
    // arguments and VC4 compiles that loop measurably slower (matvec_probe
    // measured 177 vs 283 MMAC/s).  dA/dB are exactly the decode x_i16quadq
    // performs and the four accumulators keep the same summation order, so the
    // packed output is byte-identical.  FFN_SW_INLINE=0 restores the generic
    // form in the same binary for an interleaved A/B.
    if (inl && xf == XFormat::I16_2) {
        return head(xf) + "uniform sampler2D W1,W2,X;uniform float ws1,ws2;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
               "float sl(float q,float g){float a=q/256.0;return (a/(1.0+exp(-a)))*(g/256.0);}"
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
               "int r1=min(r0+1," + to_string(rows - 1) + ");float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
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
               "float v1=(r0+1<" + to_string(rows) + ")?sl(qz(s11*ws1),qz(s21*ws2)):0.0;"
               "gl_FragColor=x_st2(sl(qz(s10*ws1),qz(s20*ws2)),v1);}";
    }
    return head(xf) +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;"
        "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
        "float sl(float q,float g){float a=q/256.0;return (a/(1.0+exp(-a)))*(g/256.0);}"
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
        "int r1=min(r0+1," + to_string(rows - 1) + ");float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
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
        "float v1=(r0+1<" + to_string(rows) + ")?sl(qz(s11*ws1),qz(s21*ws2)):0.0;"
        "gl_FragColor=x_st2(sl(qz(s10*ws1),qz(s20*ws2)),v1);}";
}

string matvec_q8_pair_fused(int rows, int cols, XFormat xf) {
    const int axw = x_texels(cols, xf);
    const int rows2 = 2 * rows;
    // The FFN gate/swiglu kernel is the hottest matvec in the token (4864x768
    // per layer) and it was still decoding through the generic X4Q helper.
    // Inline I16_2 decode is the same arithmetic with the fetches emitted in
    // the loop body; matvec_probe measures 283 vs 177 MMAC/s for that shape.
    if (xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X;uniform float ws;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "vec2 enc16(float s,float w){float v=clamp(floor(s*w+0.5),-32768.0,32767.0);"
               "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
               "return vec2(hi/255.0,(v-hi*256.0)/255.0);}"
               "void main(){int r=int(gl_FragCoord.x);float s1=0.0,s2=0.0;"
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
               "vec2 p1=enc16(s1,ws),p2=enc16(s2,ws);"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s1=0.0,s2=0.0;"
        "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
        "vec4 c1=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(r)+0.5)/float(" + to_string(rows2) + ")));"
        "vec4 c2=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
        "(float(r+" + to_string(rows) + ")+0.5)/float(" + to_string(rows2) + ")));"
        "vec4 w1=c1*255.0-128.0;vec4 w2=c2*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + to_string(axw) + ".0);"
        "s1+=w1.x*x.x+w1.y*x.y+w1.z*x.z+w1.w*x.w;"
        "s2+=w2.x*x.x+w2.y*x.y+w2.z*x.z+w2.w*x.w;}"
        "gl_FragColor=x_st2q(s1*ws,s2*ws);}";
}

string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf) {
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    if (xf == XFormat::I16_2 && rf == XFormat::I16_1) {
        // Inline form: VC4 compiles this noticeably faster than the generic
        // helper-based version (the helper takes two texel fetches as arguments).
        return "precision highp float;uniform sampler2D W,X,R;uniform float ws;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
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
               "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);"
               "vec4 rc=texture2D(R,vec2((float(r)+0.5)/float(" + ROWS + "),0.5));"
               "float oq=v+dr(rc)*256.0;"
               "float q=clamp(floor(oq+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
               "float hi=floor(q/256.0),lo=q-hi*256.0;"
               "gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
    }
    return head(xf) + x_residual_glsl(rf) +
        "uniform sampler2D W,X,R;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "float res=R1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=x_st1q(s*ws+res);}";
}

string matvec_q8_add_pair(int rows, int cols, XFormat xf, bool inl) {
    const int axw = x_texels(cols, xf);
    const int rw = (rows + 1) / 2;
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    // Inline I16_2 decode, the form matvec_q8_pair already uses: the generic
    // X4Q helper takes its two texel fetches as macro arguments and VC4 compiles
    // that loop measurably slower.  r0 is always even and r1 always odd, so the
    // two residuals are the halves of one I16_2 texel.  Accumulation order and
    // the q16 rounding are unchanged, so the packed output is bit-identical.
    if (inl && xf == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X,R;uniform float ws;"
               "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
               "return u>=32768.0?u-65536.0:u;}"
               "vec2 enc16(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);"
               "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
               "return vec2(hi/255.0,(v-hi*256.0)/255.0);}"
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;"
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
               "vec2 p1=enc16(s0*ws+res0),p2=enc16(s1*ws+res1);"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) + x_residual_glsl(XFormat::I16_2) +
        "uniform sampler2D W,X,R;uniform float ws;"
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;"
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
        "gl_FragColor=x_st2q(s0*ws+res0,s1*ws+res1);}";
}

string matvec_i16(int rows, int cols, XFormat xf) {
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
        "gl_FragColor=x_st1q(s*ws+res);}";
}

string matvec_i4(int rows, int cols, XFormat xf, bool inl) {
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
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

// q4 residual-add, two rows per fragment: rows 2t and 2t+1 share one activation
// decode and one residual fetch, and are written as a raw q16 pair (x_st2q, since
// the q4 accumulator works in raw q16 units, not activation units).
string matvec_i4_add_pair(int rows, int cols, XFormat xf, bool inl) {
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
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
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    return head(xf) +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;"
        "float dot4(vec4 w0,vec4 xa,vec4 xb){"
        "float b0=floor(w0.r*255.0+0.5),b1=floor(w0.g*255.0+0.5),"
        "b2=floor(w0.b*255.0+0.5),b3=floor(w0.a*255.0+0.5);"
        "float h0=floor(b0*0.0625),h1=floor(b1*0.0625),"
        "h2=floor(b2*0.0625),h3=floor(b3*0.0625);"
        "float l0=b0-h0*16.0,l1=b1-h1*16.0,l2=b2-h2*16.0,l3=b3-h3*16.0;"
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
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), GROUPS = to_string(groups),
                 TEXW = to_string(texw), AXW = to_string(axw);
    return q4_head(xf, inl) +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;"
        "float dot4(vec4 w0,vec4 xa,vec4 xb){"
        + q4_taps("w0") +
        "return l0*xa.x+h0*xa.y+l1*xa.z+h1*xa.w+l2*xb.x+h2*xb.y+l3*xb.z+h3*xb.w;}"
        "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
        "float sl(float q,float g){float a=q/256.0;return (a/(1.0+exp(-a)))*(g/256.0);}"
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
        "float xsum=0.0,d1=0.0,d2=0.0,e1=0.0,e2=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;"
        + q4_act(xf, cols, inl) +
        q4_xsum() +
        "vec4 w0=texture2D(W1,vec2((float(g*4+q)+0.5)/float(" + TEXW + "),(u0+0.5)/float(" + ROWS + ")));"
        "d1+=dot4(w0,xa,xb);"
        "w0=texture2D(W2,vec2((float(g*4+q)+0.5)/float(" + TEXW + "),(u0+0.5)/float(" + ROWS + ")));"
        "d2+=dot4(w0,xa,xb);"
        "w0=texture2D(W1,vec2((float(g*4+q)+0.5)/float(" + TEXW + "),(u1+0.5)/float(" + ROWS + ")));"
        "e1+=dot4(w0,xa,xb);"
        "w0=texture2D(W2,vec2((float(g*4+q)+0.5)/float(" + TEXW + "),(u1+0.5)/float(" + ROWS + ")));"
        "e2+=dot4(w0,xa,xb);}"
        "s10+=sa1*d1+ma1*xsum;s20+=sa2*d2+ma2*xsum;"
        "s11+=sb1*e1+mb1*xsum;s21+=sb2*e2+mb2*xsum;}"
        "float v0=sl(qz(s10),qz(s20));"
        "float v1=(r0+1<" + ROWS + ")?sl(qz(s11),qz(s21)):0.0;"
        "gl_FragColor=x_st2(v0,v1);}";
}

string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf) {
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
        "float b0=floor(w.r*255.0+0.5),b1=floor(w.g*255.0+0.5),"
        "b2=floor(w.b*255.0+0.5),b3=floor(w.a*255.0+0.5);"
        "float h0=floor(b0*0.0625),h1=floor(b1*0.0625),"
        "h2=floor(b2*0.0625),h3=floor(b3*0.0625);"
        "float l0=b0-h0*16.0,l1=b1-h1*16.0,l2=b2-h2*16.0,l3=b3-h3*16.0;"
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
        "gl_FragColor=x_st1q(s+res);}";
}


string swiglu_pair(int hidden, XFormat out) {
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
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, residual));
    return head(residual) + x_residual_glsl(residual) +
        "uniform sampler2D X;"
        "vec4 encs(float s){float q=floor(s*4096.0+0.5)+8388608.0;"
        "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
        "b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}"
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
        "vec4 encs(float s){float q=floor(s*4096.0+0.5)+8388608.0;"
        "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
        "b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}"
        "void main(){float s=0.0;for(int p=0;p<32;p++){"
        "s+=raw(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)))/4096.0;}"
        "gl_FragColor=encs(s);}";
}

string rms_norm(int dim, XFormat residual, XFormat out, bool fused_reduce) {
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
            "float inv=1.0/sqrt(ss/float(" + D + ")+1e-5);"
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
            "float inv=1.0/sqrt(ss/float(" + D + ")+1e-5);" + wsplit +
            "float v0=R1(X,i0," + RW + ".0)*inv*w0;"
            "float v1=0.0;if(i1<" + D + ")v1=R1(X,i1," + RW + ".0)*inv*w1;"
            "gl_FragColor=x_st2(v0,v1);}";
    }
    return head(residual) + x_residual_glsl(residual) + q24_helpers() +
        "uniform sampler2D X,W,S;uniform float dummy;" + body;
}
