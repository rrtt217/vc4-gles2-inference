#include "shaders_internal.h"
#include "shaders.h"
#include "gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using std::string;
using std::to_string;

namespace {
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

string matvec_i4(int rows, int cols, XFormat xf, XFormat of, bool inl) {
    need_shape("matvec_i4", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    return q4_head(xf, inl) + x_proj_glsl(of) +
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
        "gl_FragColor=x_projenc(s);}";
}

// q4 residual-add, two rows per fragment: rows 2t and 2t+1 share one activation
// decode and one residual fetch, and are written as a raw q16 pair (x_st2q, since
// the q4 accumulator works in raw q16 units, not activation units).
string matvec_i4_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool inl) {
    need_shape("matvec_i4_add_pair", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
        return q4_head(xf, inl) + x_pair2_glsl(of) + x_residual_glsl(rf) +
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
        "float res0=Y1Q(R,r0," + to_string(x_texels(rows, rf)) + ".0),""res1=Y1Q(R,r1," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=x_p2_stq(s0+res0,s1+res1);}";
}

// Q4_FMT=rgba4 residual-add: w2 (and wo) with the weights bound as 4-bit
// texels, two per eight weights, no decode, and the group scale folded as in
// the gate.  The nibble reversal at upload (make_q4_texture) is what lets the
// dot read the texel in natural column order.
string matvec_i4_add_pair_4x4(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool inl) {
    need_shape("matvec_i4_add_pair_4x4", rows, cols, 32);
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
        const string ROWS = to_string(rows), GROUPS = to_string(groups), TEXW = to_string(texw);
    return q4_head(xf, inl) + x_pair2_glsl(of) + x_residual_glsl(rf) +
        "uniform sampler2D W,P,X,R;uniform float ps;"
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;"
        "float s0=0.0,s1=0.0;"
        "float v0r=(float(r0)+0.5)/float(" + ROWS + "),v1r=(float(r1)+0.5)/float(" + ROWS + ");"
        "float itw=1.0/float(" + TEXW + ");"
        "for(int g=0;g<" + GROUPS + ";g++){"
        "vec4 pa=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),v0r));"
        "vec4 pb=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),v1r));"
        "float u1s=65280.0*15.0*ps,v1s=255.0*15.0*ps;"
        "float sa=pa.r*u1s+pa.g*v1s,sb=pb.r*u1s+pb.g*v1s;"
        "float ma=x_i16raw(pa.ba)*ps,mb=x_i16raw(pb.ba)*ps;"
        "float xsum=0.0,dota=0.0,dotb=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;" +
        q4_act(xf, cols, inl) + q4_xsum() +
        "float c0=(float(g*8+q*2)+0.5)*itw,c1=c0+itw;"
        "vec4 wq0=texture2D(W,vec2(c0,v0r)),wq1=texture2D(W,vec2(c1,v0r));"
        "dota+=wq0.x*xa.x+wq0.y*xa.y+wq0.z*xa.z+wq0.w*xa.w+wq1.x*xb.x+wq1.y*xb.y+wq1.z*xb.z+wq1.w*xb.w;"
        "vec4 wr0=texture2D(W,vec2(c0,v1r)),wr1=texture2D(W,vec2(c1,v1r));"
        "dotb+=wr0.x*xa.x+wr0.y*xa.y+wr0.z*xa.z+wr0.w*xa.w+wr1.x*xb.x+wr1.y*xb.y+wr1.z*xb.z+wr1.w*xb.w;}"
        "s0+=sa*dota+ma*xsum;s1+=sb*dotb+mb*xsum;}"
        "float res0=Y1Q(R,r0," + to_string(x_texels(rows, rf)) + ".0),""res1=Y1Q(R,r1," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=x_p2_stq(s0+res0,s1+res1);}";
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
        // The group's 32 taps are summed as a TREE, not by accumulating two
        // long runs: d1/d2 used to hold 32 terms each while s1/s2/xsum stayed
        // live, i.e. five f32 accumulators whose magnitude is set by the
        // activations rather than by the result.  VC4's fragment ALU keeps less
        // precision in that shape than Xe does -- probes/sumorder_probe measures
        // 44% error at magnitude 2300 against 0.000% for a tree, bit-identical
        // to Xe (2026-11-17).  Same adds, same multiplies, different order.
        "float xsum=0.0,t1=0.0,t2=0.0,t3=0.0,t4=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;"
        "vec4 xa=X4Q(X,c," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,c+4," + to_string(axw) + ".0);"
        + q4_xsum() +
        "vec4 w0=texture2D(W1,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float p1=dot4(w0,xa,xb);"
        "w0=texture2D(W2,vec2((float(g*4+q)+0.5)/float(" + to_string(texw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float p2=dot4(w0,xa,xb);"
        // four pairwise trees, combined pairwise, added to s once per group
        "if(q==0){t1=p1;t2=p2;}else if(q==1){t1+=p1;t2+=p2;}"
        "else if(q==2){t3=p1;t4=p2;}else{t3+=p1;t4+=p2;}}"
        "s1+=sa1*((t1+t3))+ma1*xsum;s2+=sa2*((t2+t4))+ma2*xsum;}"
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
string matvec_i4_pair_swiglu(int rows, int cols, XFormat xf, bool inl, int hs) {
    need_shape("matvec_i4_pair_swiglu", rows, cols, 32);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
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
    return q4_head(xf, inl) + HD +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;" +
        (mv4 ? string()
             : "float dot4(vec4 w0,vec4 xa,vec4 xb){" + q4_taps("w0") +
               "return l0*xa.x+h0*xa.y+l1*xa.z+h1*xa.w+l2*xb.x+h2*xb.y+l3*xb.z+h3*xb.w;}")
        + qz_sl() +
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
        "float v0=sl(qz(s10),qz(s20))" + HM + ";"
        "float v1=(r0+1<" + ROWS + ")?sl(qz(s11),qz(s21))" + HM + ":0.0;"
        "gl_FragColor=x_st2(v0,v1);}";
}

// Q4_FMT=rgba4 (default off): the same fused gate with the INT4 matrices bound
// as 4-bit texels.  Four weights per texel and *no decode at all* -- the
// sampler returns n/15, the group scale carries the 15, and the four weights of
// a texel come back in reverse column order, so each texel pairs with
// `xa.wzyx`.  It exists because the packed gate is 143 ms of a 286 ms q4 token
// and only ~32 ms of that is the floor chain (Q4_NOUNPACK): the rest is eight
// scalar taps and an eight-term scalar dot per texel, which is the shape a
// four-lane texel does not need.
// Two default-off pricing arms for the 4-bit gate, the same methodology as
// Q4_NOUNPACK: each removes one candidate for the 2x it still pays against the
// q8 gate, and the numbers are wrong on purpose.
// Q4_GATE_NESTED=1 restores the nested group loop (the pre-2026-11-16 default)
// for an interleaved A/B inside one binary.  The flat 8-wide gate is the
// default: byte-identical (probe hash matches at 304/608/1216/2048 rows on both
// Xe and VC4) and ~43% faster at the engine's 1216-row chunk (6.55 -> 3.74
// ms/draw).  The legacy pricing arms (NOP/ONEW/ONEROW/NOOUT/NOMIN) are defined
// on the nested body, so they route there too.
static bool q4x4_nested(){ return getenv("Q4_GATE_NESTED") != nullptr; }
// The flat-loop rewrite HELPED the gate (a pair kernel: 8 weight fetches and
// four accumulators per inner iteration) but HURT the single-row qkv and
// w2/wo kernels (2 fetches, one accumulator): measured on the Pi Zero
// 2026-11-16, all-flat 197.5 ms/token against gate-only-flat 185.0 and
// all-nested 227.3 -- the qkv/w2 flat form is ~12.5 ms/token slower than
// nested, so those two stay nested by default.  Q4_QKV_FLAT=1 / Q4_W2_FLAT=1
// opt into the flat forms (kept as the measurement arms that produced the
// negative result); the gate's Q4_GATE_NESTED=1 is unchanged.
static bool q4qkv_flat(){ return getenv("Q4_QKV_FLAT") != nullptr; }
static bool q4w2_flat(){ return getenv("Q4_W2_FLAT") != nullptr; }

static string i4_gate_flat8b(int rows, int cols, XFormat xf, bool inl, int hs) {
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
    const string ROWS = to_string(rows), TEXW = to_string(texw),
                 GROUPS = to_string(groups), ITERS = to_string(cols / 8);
    const string act8 = q4_act(xf, cols, inl);
    const string xsum = q4_xsum();
    const auto dot = [](const char* wt0, const char* wt1) {
        return string(wt0) + ".x*xa.x+" + wt0 + ".y*xa.y+" + wt0 + ".z*xa.z+" + wt0 + ".w*xa.w+"
             + wt1 + ".x*xb.x+" + wt1 + ".y*xb.y+" + wt1 + ".z*xb.z+" + wt1 + ".w*xb.w";
    };
    return q4_head(xf, inl) + HD +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;" + qz_sl() +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
        "int r1=min(r0+1," + to_string(rows - 1) + ");"
        "float v0r=(float(r0)+0.5)/float(" + ROWS + "),v1r=(float(r1)+0.5)/float(" + ROWS + ");"
        "float itw=1.0/float(" + TEXW + ");"
        "float u1s=65280.0*15.0*ps1,v1s=255.0*15.0*ps1;"
        "float u2s=65280.0*15.0*ps2,v2s=255.0*15.0*ps2;"
        "float sa1=0.0,ma1=0.0,sa2=0.0,ma2=0.0,sb1=0.0,mb1=0.0,sb2=0.0,mb2=0.0;"
        "float d1=0.0,d2=0.0,e1=0.0,e2=0.0,xsum=0.0;"
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int q=0;q<" + ITERS + ";q++){"
        "int g=q/4;"
        "if(q-g*4==0){"
        "d1=0.0;d2=0.0;e1=0.0;e2=0.0;xsum=0.0;"
        "float gp=(float(g)+0.5)/float(" + GROUPS + ");"
        "vec4 q1=texture2D(P1,vec2(gp,v0r)),q2=texture2D(P2,vec2(gp,v0r));"
        "vec4 p1=texture2D(P1,vec2(gp,v1r)),p2=texture2D(P2,vec2(gp,v1r));"
        "sa1=q1.r*u1s+q1.g*v1s;sa2=q2.r*u2s+q2.g*v2s;"
        "sb1=p1.r*u1s+p1.g*v1s;sb2=p2.r*u2s+p2.g*v2s;"
        "ma1=x_i16raw(q1.ba)*ps1;ma2=x_i16raw(q2.ba)*ps2;"
        "mb1=x_i16raw(p1.ba)*ps1;mb2=x_i16raw(p2.ba)*ps2;}"
        "int c=q*8;" + act8 + xsum +
        "float c0=(float(q*2)+0.5)*itw,c1=c0+itw;"
        "vec4 wt0=texture2D(W1,vec2(c0,v0r)),wt1=texture2D(W1,vec2(c1,v0r));"
        "d1+=" + dot("wt0", "wt1") + ";"
        "vec4 wt2=texture2D(W2,vec2(c0,v0r)),wt3=texture2D(W2,vec2(c1,v0r));"
        "d2+=" + dot("wt2", "wt3") + ";"
        "vec4 wt4=texture2D(W1,vec2(c0,v1r)),wt5=texture2D(W1,vec2(c1,v1r));"
        "e1+=" + dot("wt4", "wt5") + ";"
        "vec4 wt6=texture2D(W2,vec2(c0,v1r)),wt7=texture2D(W2,vec2(c1,v1r));"
        "e2+=" + dot("wt6", "wt7") + ";"
        "if(q-g*4==3){"
        "s10+=sa1*d1+ma1*xsum;s20+=sa2*d2+ma2*xsum;"
        "s11+=sb1*e1+mb1*xsum;s21+=sb2*e2+mb2*xsum;}}"
        "float v0=sl(qz(s10),qz(s20))" + HM + ";"
        "float v1=(r0+1<" + ROWS + ")?sl(qz(s11),qz(s21))" + HM + ":0.0;"
        "gl_FragColor=x_st2(v0,v1);}";
}


string matvec_i4_pair_swiglu_4x4(int rows, int cols, XFormat xf, bool inl, int hs) {
    need_shape("matvec_i4_pair_swiglu_4x4", rows, cols, 32);
    // Default: the byte-identical flat 8-wide gate.  The nested form below is
    // kept for Q4_GATE_NESTED=1 (the pre-2026-11-16 default) and for
    // Q4_NOMIN=1, whose pricing definition is the nested body.
    // The nine arms this dispatcher used to select (Q4_4X4_FLAT, FLAT8, FLAT8M,
    // FLAT8G, FLAT8B, ONEW, ONEROW, NOOUT, NOP) are gone: they were the road to
    // this body -- stage removals and loop shapes whose conclusion is the code
    // below -- and each one was a second kernel text to keep verified.
    if (!(q4x4_nested() || q4_nomin()))
        return i4_gate_flat8b(rows, cols, xf, inl, hs);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    const int texw = cols / 4;              // one 4-bit texel = four weights
    const int groups = (cols + 31) / 32;
    const string ROWS = to_string(rows), GROUPS = to_string(groups), TEXW = to_string(texw);
    // Q4_NOMIN=1 drops the asymmetric min term and the group sum that feeds it:
    // 28 adds per 32-weight group, 672 per fragment against 768 MACs, which is
    // the last unexplained cost against the q8 gate (q8 has one uniform scale).
    const bool nom = q4_nomin();
    const string MA1 = nom ? "" : "+ma1*xsum", MA2 = nom ? "" : "+ma2*xsum";
    const string MB1 = nom ? "" : "+mb1*xsum", MB2 = nom ? "" : "+mb2*xsum";
    return q4_head(xf, inl) + HD +
        "uniform sampler2D W1,W2,P1,P2,X;uniform float ps1,ps2;" + qz_sl() +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;"
        "int r1=min(r0+1," + to_string(rows - 1) + ");"
        "float u0=float(r0),u1=float(r1);"
        // Precomputed once: the two row coordinates and the reciprocal of the
        // texel width.  Emitting them per tap cost an integer->float conversion
        // and two multiplies per tap, eight taps per 32-weight group.
        "float v0r=(u0+0.5)/float(" + ROWS + "),v1r=(u1+0.5)/float(" + ROWS + ");"
        "float itw=1.0/float(" + TEXW + ");"
        "float s10=0.0,s20=0.0,s11=0.0,s21=0.0;"
        "for(int g=0;g<" + GROUPS + ";g++){"
        // The scale is unsigned, so its 16-bit reconstruction folds into
        // the two constants below and needs no negative-rail select; the
        // min still needs one.  The plane was the gate's last big q4-only
        // cost, and this is the half that costs no precision.
        + "vec4 q1=texture2D(P1,vec2((float(g)+0.5)/float(" + GROUPS + "),(u0+0.5)/float(" + ROWS + ")));"
                 "vec4 q2=texture2D(P2,vec2((float(g)+0.5)/float(" + GROUPS + "),(u0+0.5)/float(" + ROWS + ")));"
                 "vec4 p1=texture2D(P1,vec2((float(g)+0.5)/float(" + GROUPS + "),(u1+0.5)/float(" + ROWS + ")));"
                 "vec4 p2=texture2D(P2,vec2((float(g)+0.5)/float(" + GROUPS + "),(u1+0.5)/float(" + ROWS + ")));"
                 "float u1s=65280.0*15.0*ps1,v1s=255.0*15.0*ps1;"
                 "float u2s=65280.0*15.0*ps2,v2s=255.0*15.0*ps2;"
                 "float sa1=q1.r*u1s+q1.g*v1s,sa2=q2.r*u2s+q2.g*v2s;"
                 "float sb1=p1.r*u1s+p1.g*v1s,sb2=p2.r*u2s+p2.g*v2s;"
                 "float ma1=x_i16raw(q1.ba)*ps1,ma2=x_i16raw(q2.ba)*ps2;"
          "float mb1=x_i16raw(p1.ba)*ps1,mb2=x_i16raw(p2.ba)*ps2;"
        "float xsum=0.0,d1=0.0,d2=0.0,e1=0.0,e2=0.0;"
        "for(int q=0;q<4;q++){int c=g*32+q*8;" +
        q4_act(xf, cols, inl) + q4_xsum() +
        + "float c0=(float(g*8+q*2)+0.5)*itw,c1=c0+itw;"
        // make_q4_texture reversed the nibbles at upload, so the texel is in
        // natural column order and no .wzyx is needed (that operand measured
        // 52 ms of the gate; it was the first thing this decomposition found).
        + "vec4 wt0=texture2D(W1,vec2(c0,v0r)),wt1=texture2D(W1,vec2(c1,v0r));"
          "d1+=wt0.x*xa.x+wt0.y*xa.y+wt0.z*xa.z+wt0.w*xa.w+wt1.x*xb.x+wt1.y*xb.y+wt1.z*xb.z+wt1.w*xb.w;"
        + "vec4 wt2=texture2D(W2,vec2(c0,v0r)),wt3=texture2D(W2,vec2(c1,v0r));"
          "d2+=wt2.x*xa.x+wt2.y*xa.y+wt2.z*xa.z+wt2.w*xa.w+wt3.x*xb.x+wt3.y*xb.y+wt3.z*xb.z+wt3.w*xb.w;"
        + "vec4 wt4=texture2D(W1,vec2(c0,v1r)),wt5=texture2D(W1,vec2(c1,v1r));"
          "e1+=wt4.x*xa.x+wt4.y*xa.y+wt4.z*xa.z+wt4.w*xa.w+wt5.x*xb.x+wt5.y*xb.y+wt5.z*xb.z+wt5.w*xb.w;"
          "vec4 wt6=texture2D(W2,vec2(c0,v1r)),wt7=texture2D(W2,vec2(c1,v1r));"
          "e2+=wt6.x*xa.x+wt6.y*xa.y+wt6.z*xa.z+wt6.w*xa.w+wt7.x*xb.x+wt7.y*xb.y+wt7.z*xb.z+wt7.w*xb.w;"
        + "}" +
        "s10+=sa1*d1" + MA1 + ";s20+=sa2*d2" + MA2 + ";"
        "s11+=sb1*e1" + MB1 + ";s21+=sb2*e2" + MB2 + ";}"
        "float v0=sl(qz(s10),qz(s20))" + HM + ";"
        "float v1=(r0+1<" + ROWS + ")?sl(qz(s11),qz(s21))" + HM + ":0.0;"
        + "gl_FragColor=x_st2(v0,v1);}";
}


// Q4_FMT=rgba4 for qkv (one row a fragment) and wo (residual add).  Same
// transformation as the gate: two 4-bit texels per eight weights, no decode,
// the group scale folded into two constants, nibbles reversed at upload.
// Flat 8-wide qkv matvec, byte-identical to the nested matvec_i4_4x4: one row
// per fragment, d/xsum reset at the group start (q%4==0), accumulate per
// iteration, flush s+=sc*d+mn*xsum at the group end (q%4==3).  Conditional
// float updates, which VC4 preserves (docs/vc4-glsl-costs.md §6).
static string i4_qkv_flat(int rows, int cols, XFormat xf, XFormat of, bool inl) {
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
    const string ROWS = to_string(rows), GROUPS = to_string(groups),
                 TEXW = to_string(texw), ITERS = to_string(cols / 8);
    return q4_head(xf, inl) + x_proj_glsl(of) +
        "uniform sampler2D W,P,X;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "float vr=(float(r)+0.5)/float(" + ROWS + "),itw=1.0/float(" + TEXW + ");"
        "float sc=0.0,mn=0.0,d=0.0,xsum=0.0;"
        "for(int q=0;q<" + ITERS + ";q++){"
        "int g=q/4;"
        "if(q-g*4==0){"
        "d=0.0;xsum=0.0;"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),vr));"
        "float u=65280.0*15.0*ps,v=255.0*15.0*ps;"
        "sc=p.r*u+p.g*v;mn=x_i16raw(p.ba)*ps;}"
        "int c=q*8;" + q4_act(xf, cols, inl) + q4_xsum() +
        "float c0=(float(q*2)+0.5)*itw,c1=c0+itw;"
        "vec4 w0=texture2D(W,vec2(c0,vr)),w1=texture2D(W,vec2(c1,vr));"
        "d+=w0.x*xa.x+w0.y*xa.y+w0.z*xa.z+w0.w*xa.w+w1.x*xb.x+w1.y*xb.y+w1.z*xb.z+w1.w*xb.w;"
        "if(q-g*4==3){s+=sc*d+mn*xsum;}}"
        "gl_FragColor=x_projenc(s);}";
}

string matvec_i4_4x4(int rows, int cols, XFormat xf, XFormat of, bool inl) {
    need_shape("matvec_i4_4x4", rows, cols, 32);
    if (q4qkv_flat() && !q4_nomin()) return i4_qkv_flat(rows, cols, xf, of, inl);
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
    const string ROWS = to_string(rows), GROUPS = to_string(groups), TEXW = to_string(texw);
    const bool nom = q4_nomin();
    const string MA = nom ? "" : "+mn*xsum";
    return q4_head(xf, inl) + x_proj_glsl(of) +
        "uniform sampler2D W,P,X;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "float vr=(float(r)+0.5)/float(" + ROWS + "),itw=1.0/float(" + TEXW + ");"
        "for(int g=0;g<" + GROUPS + ";g++){"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),vr));"
        "float u=65280.0*15.0*ps,v=255.0*15.0*ps;"
        "float sc=p.r*u+p.g*v,mn=x_i16raw(p.ba)*ps;"
        "float xsum=0.0,d=0.0;"
        "for(int t=0;t<4;t++){int c=g*32+t*8;" +
        q4_act(xf, cols, inl) + q4_xsum() +
        "float c0=(float(g*8+t*2)+0.5)*itw,c1=c0+itw;"
        "vec4 w0=texture2D(W,vec2(c0,vr)),w1=texture2D(W,vec2(c1,vr));"
        "d+=w0.x*xa.x+w0.y*xa.y+w0.z*xa.z+w0.w*xa.w+w1.x*xb.x+w1.y*xb.y+w1.z*xb.z+w1.w*xb.w;}"
        "s+=sc*d" + MA + ";}"
        "gl_FragColor=x_projenc(s);}";
}

// Flat 8-wide w2/wo residual-add matvec, byte-identical to the nested
// matvec_i4_add_4x4.  Keeps the generic X4Q activation decode (this kernel
// never used the inline dA/dB form) and the residual add.  Same per-group
// conditional float structure as the flat qkv/gate.
static string i4_w2_flat(int rows, int cols, XFormat xf, XFormat rf, XFormat of) {
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), GROUPS = to_string(groups),
                 TEXW = to_string(texw), ITERS = to_string(cols / 8), AXW = to_string(axw);
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return head(xf) + x_residual_glsl(rf) + (of == rf ? std::string() : x_store_glsl(of)) +
        "uniform sampler2D W,P,X,R;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "float vr=(float(r)+0.5)/float(" + ROWS + "),itw=1.0/float(" + TEXW + ");"
        "float sc=0.0,mn=0.0,d=0.0,xsum=0.0;"
        "for(int q=0;q<" + ITERS + ";q++){"
        "int g=q/4;"
        "if(q-g*4==0){"
        "d=0.0;xsum=0.0;"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),vr));"
        "float u=65280.0*15.0*ps,v=255.0*15.0*ps;"
        "sc=p.r*u+p.g*v;mn=x_i16raw(p.ba)*ps;}"
        "int c=q*8;"
        "vec4 xa=X4Q(X,c," + AXW + ".0);"
        "vec4 xb=X4Q(X,c+4," + AXW + ".0);"
        + q4_xsum() +
        "float c0=(float(q*2)+0.5)*itw,c1=c0+itw;"
        "vec4 w0=texture2D(W,vec2(c0,vr)),w1=texture2D(W,vec2(c1,vr));"
        "d+=w0.x*xa.x+w0.y*xa.y+w0.z*xa.z+w0.w*xa.w+w1.x*xb.x+w1.y*xb.y+w1.z*xb.z+w1.w*xb.w;"
        "if(q-g*4==3){s+=sc*d+mn*xsum;}}"
        "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=" + ST + "(s+res);}";
}

string matvec_i4_add_4x4(int rows, int cols, XFormat xf, XFormat rf, XFormat of) {
    need_shape("matvec_i4_add_4x4", rows, cols, 32);
    if (q4w2_flat() && !q4_nomin()) return i4_w2_flat(rows, cols, xf, rf, of);
    const int texw = cols / 4;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), GROUPS = to_string(groups), TEXW = to_string(texw);
    const bool nom = q4_nomin();
    const string MA = nom ? "" : "+mn*xsum";
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return head(xf) + x_residual_glsl(rf) + (of == rf ? std::string() : x_store_glsl(of)) +
        "uniform sampler2D W,P,X,R;uniform float ps;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "float vr=(float(r)+0.5)/float(" + ROWS + "),itw=1.0/float(" + TEXW + ");"
        "for(int g=0;g<" + GROUPS + ";g++){"
        "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + GROUPS + "),vr));"
        "float u=65280.0*15.0*ps,v=255.0*15.0*ps;"
        "float sc=p.r*u+p.g*v,mn=x_i16raw(p.ba)*ps;"
        "float xsum=0.0,d=0.0;"
        "for(int t=0;t<4;t++){int c=g*32+t*8;"
        "vec4 xa=X4Q(X,c," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,c+4," + to_string(axw) + ".0);"
        + q4_xsum() +
        "float c0=(float(g*8+t*2)+0.5)*itw,c1=c0+itw;"
        "vec4 w0=texture2D(W,vec2(c0,vr)),w1=texture2D(W,vec2(c1,vr));"
        "d+=w0.x*xa.x+w0.y*xa.y+w0.z*xa.z+w0.w*xa.w+w1.x*xb.x+w1.y*xb.y+w1.z*xb.z+w1.w*xb.w;}"
        "s+=sc*d" + MA + ";}"
        "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=" + ST + "(s+res);}";
}

string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of) {
    need_shape("matvec_i4_add", rows, cols, 32);
    const int texw = cols / 8;
    const int groups = (cols + 31) / 32;
    const int axw = x_texels(cols, xf);
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return head(xf) + x_residual_glsl(rf) + (of == rf ? std::string() : x_store_glsl(of)) +
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
        "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=" + ST + "(s+res);}";
}
