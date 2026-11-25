#include "shaders_internal.h"
#include "shaders.h"

#include <string>

using std::string;
using std::to_string;

namespace {
// q16 pair from a scaled value (s*w) and from a value already in q16 units.
const string kEnc16Mul =
    "vec2 enc16(float s,float w){float v=clamp(floor(s*w+0.5),-32768.0,32767.0);"
    "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
    "return vec2(hi/255.0,(v-hi*256.0)/255.0);}";
const string kEnc16Raw =
    "vec2 enc16(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);"
    "if(v<0.0)v+=65536.0;float hi=floor(v/256.0);"
    "return vec2(hi/255.0,(v-hi*256.0)/255.0);}";
}  // namespace

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
               // q24, signed with the 2^23 bias -- the same encode as x_i24encq.  This
               // used to add 65536 for a negative sum first (a copy of the unsigned q16
               // packing), so every negative matvec output came back 256 too large: the
               // LFM2 conv block, the only user of this non-pair q8 path, then read
               // garbage from the second texel on (gated 55.98 against 0.07, 2026-11-12).
               "float v=clamp(floor(s*" + SC + "+0.5),-32768.0,32767.0)+8388608.0;"
               "float g=floor(v/65536.0);"
               "float b=floor(v/256.0)-g*256.0;"
               "float a=v-g*65536.0-b*256.0;"
               "gl_FragColor=vec4(0.0,g/255.0,b/255.0,a/255.0);}";
    }
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_q24encq(s*" + SC + ");}";
}

// The batched matvec: one draw for N inputs (RWKV-7 chunked prefill, P1).
//
// The single-token kernels read texel row 0 of X because their staging texture
// is one row tall.  A chunked prefill uploads N inputs as N rows of one texture
// and takes its row from gl_FragCoord.y; the viewport's y offset is absolute
// (docs/engine-internals.md, "pixel centers"), so no extra uniform is needed.
// Everything else is matvec_q8's text: the same weight fetch, the same
// activation decode, the same loop order, the same q24 encode and the same
// scale uniform.  Fragment (r, b) therefore computes exactly what matvec_q8
// computes for row r of input b, bit for bit -- the host keeps the per-input
// 2^k normalisation (Session::rwkv_mv's rule) and folds 2^k into the matrix
// scale of its k group, so each draw sees one k and the arithmetic is the
// single-token path's.
//
// The texture height is the uniform `xh` rather than a literal so one program
// serves every batch size (a literal would link a program per N); sampling is
// GL_NEAREST at a pixel center, so the row select cannot be off by a texel.
string matvec_q8_batch(int rows, int cols, XFormat xf, bool wrow) {
    need_shape("matvec_q8_batch", rows, cols, 4);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    // The batch row this fragment belongs to, in [0,1) texel space.  The X
    // texture's width is a *uniform* (xw): one call packs rows of different
    // lengths into one texture (the ups are 64/64/32/128 wide), so a literal
    // divisor would scale each row's texels by the wrong stride and read the
    // neighbouring row's data.  Sampling is nearest at a pixel center, so the
    // row and column select are exact -- the same texels a one-row texture
    // would hand matvec_q8.
    const string AY = "int b=int(gl_FragCoord.y);float ay=(float(b)+0.5)/xh;";
    const string WROW =
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;";
    if (xf == XFormat::I16_2) {
        // matvec_q8's I16_2 fast path verbatim (two fetches, dA/dB inline); the
        // only edits are the row coordinate and the encode's local name for the
        // blue channel, which the batch index b occupies here.
        return "precision highp float;uniform sampler2D W,X;uniform float ws;"
               "uniform float xh;uniform float xw;" + WD +
               kDec16A + kDec16B +
               "void main(){int r=int(gl_FragCoord.x);" + AY + "float s=0.0;"
               "for(int q=0;q<" + COLS4 + ";q++){" + WROW +
               "vec4 xa=texture2D(X,vec2((float(2*q)+0.5)/xw,ay));"
               "vec4 xb=texture2D(X,vec2((float(2*q+1)+0.5)/xw,ay));"
               "s+=w.x*dA(xa)+w.y*dB(xa)+w.z*dA(xb)+w.w*dB(xb);}"
               "float v=clamp(floor(s*" + SC + "+0.5),-32768.0,32767.0)+8388608.0;"
               "float g=floor(v/65536.0);"
               "float bl=floor(v/256.0)-g*256.0;"
               "float a=v-g*65536.0-bl*256.0;"
               "gl_FragColor=vec4(0.0,g/255.0,bl/255.0,a/255.0);}";
    }
    // The other layouts keep the generic X4Q decode matvec_q8 uses, with the row
    // substituted: one fetch per value (I24_1, I16_1) or per four (I8_4).
    auto fetch = [&](const string& i) -> string {
        if (xf == XFormat::I24_1)
            return "x_q24raw(texture2D(X,vec2((float(" + i + ")+0.5)/xw,ay)))";
        if (xf == XFormat::I16_1)
            return "x_i16raw(texture2D(X,vec2((float(" + i + ")+0.5)/xw,ay)).rg)";
        return "((texture2D(X,vec2((float((" + i + ")/4)+0.5)/xw,ay))*255.0-128.0)*16.0)";
    };
    return head(xf) +
        "uniform sampler2D W,X;uniform float ws;uniform float xh;uniform float xw;" + WD +
        "void main(){int r=int(gl_FragCoord.x);" + AY + "float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){" + WROW +
        "vec4 x=vec4(" + fetch("q*4") + "," + fetch("q*4+1") + ","
        + fetch("q*4+2") + "," + fetch("q*4+3") + ");"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "gl_FragColor=x_q24encq(s*" + SC + ");}";
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
string matvec_q8_pair_swiglu(int rows, int cols, XFormat xf, bool inl, bool wrow, int hs) {
    need_shape("matvec_q8_pair_swiglu", rows, cols, 4);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
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
        return head(xf) + HD + "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
               kDec16A + kDec16B
               + qz_sl() +
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
               "gl_FragColor=x_st2(sl(qz(s10*" + A10 + "),qz(s20*" + A20 + "))" + HM + ",v1" + HM + ");}";
    }
    return head(xf) + HD +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD +
               qz_sl() +
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
        "gl_FragColor=x_st2(sl(qz(s10*" + A10 + "),qz(s20*" + A20 + "))" + HM + ",v1" + HM + ");}";
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
               + qz_sl() +
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
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2;" + WD + qz_sl() +
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
    return head(XFormat::I16_2) + x_pair2_glsl(rf) + x_residual_glsl(rf) +
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
        "float ra=Y1Q(RA,r," + RW + ".0),rb=Y1Q(RB,r," + RW + ".0);"
        "gl_FragColor=x_p2_stq(sa*" + SC + "+ra,sb*" + SC + "+rb);}";
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

string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool wrow) {
    need_shape("matvec_q8_add", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    if (xf == XFormat::I16_2 && rf == XFormat::I16_1 && of == XFormat::I16_1) {
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
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return head(xf) + x_residual_glsl(rf) + (of == rf ? std::string() : x_store_glsl(of)) +
        "uniform sampler2D W,X,R;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + COLS4 + "),"
        "(float(r)+0.5)/float(" + ROWS + ")));"
        "vec4 w=c*255.0-128.0;"
        "vec4 x=X4Q(X,q*4," + AXW + ".0);"
        "s+=w.x*x.x+w.y*x.y+w.z*x.z+w.w*x.w;}"
        "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=" + ST + "(s*" + SC + "+res);}";
}

// `bias` folds a separate `bias_add` pass into this one.  The bias rides the
// output row, so the two rows a fragment writes are two reads of the bias plane
// and no extra draw is issued.  It is **not** byte-identical to a `bias_add`
// after the matvec: the sum rounds once here and twice there, by up to 1/256.
string matvec_q8_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool inl,
                          bool wrow, bool bias) {
    need_shape("matvec_q8_add_pair", rows, cols, 4);
    const int axw = x_texels(cols, xf);
    const int rw = (rows + 1) / 2;
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4), AXW = to_string(axw);
    // Two rows per fragment, so two scale rows: r0 even, r1 odd, exactly the
    // rows the packed residual texel carries.
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "vec2 sc=wsr2(r0);" : "";
    const string SC0 = wrow ? "sc.x" : "ws", SC1 = wrow ? "sc.y" : "ws";
    // The base and the bias are read in rf, whose texel count is
    // x_texels(rows, rf); rw is the *store*'s (the packed residual).  They
    // coincide for the shipped I16_2 residual and diverge for f32/bf16/i24.
    const string RTW = to_string(x_texels(rows, rf));
    const string BW = RTW + ".0";
    const string BIAS_INL =
        bias ? "vec4 bt=texture2D(B,vec2((float(t)+0.5)/float(" + to_string(rw) + "),0.5));"
               "float b0=dA(bt),b1=dB(bt);"
             : std::string();
    const string BIAS_GEN =
        bias ? "float b0=Y1(B,r0," + BW + ");float b1=Y1(B,r1," + BW + ");"
             : std::string();
    const string B0 = bias ? "+b0" : "", B1 = bias ? "+b1" : "";
    const string BS = bias ? ",B" : "";
    // Inline I16_2 decode, the form matvec_q8_pair already uses: the generic
    // X4Q helper takes its two texel fetches as macro arguments and VC4 compiles
    // that loop measurably slower.  r0 is always even and r1 always odd, so the
    // two residuals are the halves of one I16_2 texel.  Accumulation order and
    // the q16 rounding are unchanged, so the packed output is bit-identical.
    if (inl && xf == XFormat::I16_2 && rf == XFormat::I16_2 && of == XFormat::I16_2) {
        return "precision highp float;uniform sampler2D W,X,R" + BS + ";uniform float ws;" + WD +
               kDec16A + kDec16B
               + kEnc16Raw +
               "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
               "vec4 rc=texture2D(R,vec2((float(t)+0.5)/float(" + to_string(rw) + "),0.5));"
               "float res0=dA(rc),res1=dB(rc);" + BIAS_INL +
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
               "vec2 p1=enc16(s0*" + SC0 + "+res0" + B0 + "),"
               "p2=enc16(s1*" + SC1 + "+res1" + B1 + ");"
               "gl_FragColor=vec4(p1.x,p1.y,p2.x,p2.y);}";
    }
    return head(xf) + x_pair2_glsl(of) + x_residual_glsl(rf) +
        "uniform sampler2D W,X,R" + BS + ";uniform float ws;" + WD +
        "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=r0+1;" + SP +
        "float res0=Y1Q(R,r0," + RTW + ".0);"
        "float res1=Y1Q(R,r1," + RTW + ".0);" + BIAS_GEN +
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
        "gl_FragColor=x_p2_stq(s0*" + SC0 + "+res0" + B0 + ",s1*" + SC1 + "+res1" + B1 + ");}";
}
