#include "shaders_internal.h"
#include "shaders.h"
#include "switches.h"

#include <cstring>
#include <string>

using std::string;
using std::to_string;

namespace {
// ---- how the rmsnorm's partial sums travel (RMS_STORE=i24|f32) --------------
// Each of the 32 lanes holds a sum of squares whose magnitude follows the row's
// rms: ~0.013 for pocket's layer 0 (|x| ~ 0.02, measured 2026-11-18) and ~1e8
// for a 2300-magnitude residual.  The 24-bit encoding below has a *fixed*
// 1/4096 step, so at layer 0 the lane sums land at 55-91 LSB -- about seven
// bits -- and one LSB is 0.9% of a lane.  That is exactly the 0.212% scale
// difference the board shows against Xe on layer 0's normalised row, which then
// compounds over 28 layers.  `f32` stores the same four bytes as a bit-cast
// float: exact at every magnitude.  `i24` stays the default so no pin moves.
static std::string rms_store_glsl() {
    static const bool f32 = [] {
        const char* v = sw_str("RMS_STORE");
        return v && !strcmp(v, "f32");
    }();
    if (f32) {
        return std::string(
            "float s_ld(vec4 c){uint u=(uint(int(c.x*255.0+0.5))<<24)|"
            "(uint(int(c.y*255.0+0.5))<<16)|(uint(int(c.z*255.0+0.5))<<8)|"
            "uint(int(c.w*255.0+0.5));return uintBitsToFloat(u);}\n"
            "vec4 s_st(float v){uint u=floatBitsToUint(v);"
            "return vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
            "float(u&255u))/255.0;}\n"
            "\n#define S_ST(V) s_st(V)\n"
            "#define S_LD(C) s_ld(C)\n");
    }
    return "\n#define S_ST(V) x_s24enc(V)\n"
           "#define S_LD(C) x_s24(C)\n";
}
// Sum of squares in 24-bit fixed point, scale 4096 (the rms stats passes).
const string kEncs4096 =
    "vec4 encs(float s){float q=floor(s*4096.0+0.5)+8388608.0;"
    "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
    "b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
    "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
// The 24-bit encoder plus the S_ST/S_LD pair `RMS_STORE` selects: every builder
// that writes or reads an rms partial sum emits this, once.
static string encs_glsl() { return x_codec_glsl(XFormat::S24_1) + rms_store_glsl(); }
}  // namespace

string swiglu_pair(int hidden, XFormat out, int hs) {
    need_dim("swiglu_pair", hidden);
    const string H = to_string(hidden);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    string act =
        "float actv(int i){if(i>=" + H + ")return 0.0;"
        "vec2 g=X2(A,2*i," + H + ".0);"
        "return (g.x/(1.0+" + EXP("-g.x") + "))*g.y;}";
    string body;
    if (out == XFormat::F32) {
        body = "void main(){int i=int(gl_FragCoord.x);"
               "gl_FragColor=x_f32enc(actv(i)" + HM + ");}";
        // head(I16_2), not head(F32): the pair buffer A holds two q16 values a
        // texel and actv() reads it with X2.  Only the *store* is f32.
        return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::F32) + x_codec_glsl(XFormat::BF16) + HD +
           "uniform sampler2D A;" + act + body;
    }
    if (out == XFormat::BF16) {
        // Two values a texel, same pair index as the q16 form; the store is bf16.
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_bf16enc2(actv(i0)" + HM + ",actv(i0+1)" + HM + ");}";
        return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::BF16) + HD +
               "uniform sampler2D A;" + act + body;
    }
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(actv(i0)" + HM + ",actv(i0+1)" + HM + ",actv(i0+2)" + HM + ",actv(i0+3)" + HM + ");}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(actv(i0)" + HM + ",actv(i0+1)" + HM + ");}";
    }
    return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::BF16) + HD +
           "uniform sampler2D A;" + act + body;
}

string swiglu_2row(int hidden, XFormat out, int hs) {
    need_dim("swiglu_2row", hidden);
    const string H = to_string(hidden);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    string read =
        "float yat(int i){float x=float(i);"
        "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + H + "),0.25));"
        "vec4 cb=texture2D(A,vec2((x+0.5)/float(" + H + "),0.75));"
        "float a=x_q24(ca),b=x_q24(cb);"
        "return (a/(1.0+" + EXP("-a") + "))*b;}";
    string body;
    if (out == XFormat::F32) {
        // One value a texel: the fragment index *is* the hidden index, so
        // there is no i0 offset and no pair to pack.  This is the only output
        // format that can hold a hidden state beyond the q16 rail (-128), and
        // pocket's peaks at 1005 (probe_residuals.py).
        body = "void main(){int i=int(gl_FragCoord.x);"
               "gl_FragColor=x_f32enc(yat(i)" + HM + ");}";
        return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::F32) + x_codec_glsl(XFormat::BF16) +
               HD +
               "uniform sampler2D A;" + read + body;
    }
    if (out == XFormat::BF16) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_bf16enc2(yat(i0)" + HM + ",yat(i0+1)" + HM + ");}";
        return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::BF16) + HD +
               "uniform sampler2D A;" + read + body;
    }
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(yat(i0)" + HM + ",yat(i0+1)" + HM + ",yat(i0+2)" + HM + ",yat(i0+3)" + HM + ");}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(yat(i0)" + HM + ",yat(i0+1)" + HM + ");}";
    }
    return head(XFormat::I16_2) + HD + "uniform sampler2D A;" + read + body;
}

// swiglu over the two rows of a buffer that holds **q16 pairs** (x_st2q): row 0
// is the gate, row 1 the up projection, and texel t of each holds rows 2t and
// 2t+1.  This is the FFN gate's shape when the weights are 2-bit: one texel
// carries both column groups of one row, so a pair kernel writes adjacent rows
// of *one* matrix and the two matrices are two rows of the buffer.
// swiglu over the two rows of a buffer that holds **q16 pairs** (x_st2q): row 0
// is the gate, row 1 the up projection, and texel t of each holds rows 2t and
// 2t+1.  This is the FFN gate's shape when the weights are 2-bit: one texel
// carries both column groups of one row, so a pair kernel writes adjacent rows
// of *one* matrix and the two matrices are two rows of the buffer.
string swiglu_pair2row(int hidden, XFormat out, int hs) {
    need_dim("swiglu_pair2row", hidden);
    // `out` is always I16_2 here (a pair texture is the point of the kernel);
    // the parameter stays so the call sites read like the other swiglu builders
    // and a future hidden format has one place to grow.
    (void)out;
    const string NP = to_string((hidden + 1) / 2);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    const string rd = "float i16(vec2 c){float u=c.x*65280.0+c.y*255.0;"
                      "if(u>=32767.5)u-=65536.0;return u/256.0;}";
    // One texel is two rows, so a fragment reads each texture once and gets both
    // halves: no parity test (which would need ESSL 3.00 bitwise ops), no mix(),
    // and two fetches where reading the rows one at a time took four.
    const string body =
        "uniform sampler2D A;"
        "void main(){int t=int(gl_FragCoord.x);"
        "vec4 ca=texture2D(A,vec2((float(t)+0.5)/float(" + NP + "),0.25));"
        "vec4 cb=texture2D(A,vec2((float(t)+0.5)/float(" + NP + "),0.75));"
        "float g0=i16(ca.rg),g1=i16(ca.ba);"
        "float u0=i16(cb.rg),u1=i16(cb.ba);"
        "float y0=(g0/(1.0+" + EXP("-g0") + "))*u0;"
        "float y1=(g1/(1.0+" + EXP("-g1") + "))*u1;"
        "gl_FragColor=x_st2(y0" + HM + ",y1" + HM + ");}";
    return head(XFormat::I16_2) + HD + rd + body;
}

string swiglu(int hidden, bool fused, XFormat out, int hs) {
    need_dim("swiglu", hidden);
    const string H = to_string(hidden);
    const string HD = hid_scale_def(hs), HM = hid_scale_mul(hs);
    string read;
    if (fused) {
        read = "uniform sampler2D A;"
               "float yat(int i){float x=float(i);"
               "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + to_string(2 * hidden) + "),0.5));"
               "vec4 cb=texture2D(A,vec2((x+0.5+float(" + H + "))/float(" + to_string(2 * hidden) + "),0.5));"
               "float a=x_q24(ca),b=x_q24(cb);"
               "return (a/(1.0+" + EXP("-a") + "))*b;}";
    } else {
        read = "uniform sampler2D A,B;"
               "float yat(int i){float x=float(i);"
               "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + H + "),0.5));"
               "vec4 cb=texture2D(B,vec2((x+0.5)/float(" + H + "),0.5));"
               "float a=x_q24(ca),b=x_q24(cb);"
               "return (a/(1.0+" + EXP("-a") + "))*b;}";
    }
    string body;
    if (out == XFormat::F32) {
        body = "void main(){int i=int(gl_FragCoord.x);"
               "gl_FragColor=x_f32enc(yat(i)" + HM + ");}";
        return head(XFormat::I16_2) + x_codec_glsl(XFormat::F32) + HD + read + body;
    }
    if (out == XFormat::BF16) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_bf16enc2(yat(i0)" + HM + ",yat(i0+1)" + HM + ");}";
        return head(XFormat::I16_2) + x_codec_glsl(XFormat::BF16) + HD + read + body;
    }
    if (out == XFormat::I8_4) {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=4*t;"
               "gl_FragColor=x_i8enc4(yat(i0)" + HM + ",yat(i0+1)" + HM + ",yat(i0+2)" + HM + ",yat(i0+3)" + HM + ");}";
    } else {
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "gl_FragColor=x_st2(yat(i0)" + HM + ",yat(i0+1)" + HM + ");}";
    }
    return head(XFormat::I16_2) + poly_src() + x_codec_glsl(XFormat::BF16) + HD + read + body;
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
// is the same Y1(X,i)*inv*w[i] encoded by the same x_st2.  Only the layout the
// result lands in changes, so the gate sees the q16 integers the serial
// rmsnorm would have produced.
string rms_stats2(int dim, XFormat rf) {
    need_dim("rms_stats2", dim);
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, rf));
    return head(XFormat::I16_2) + x_residual_glsl(rf) +
        "uniform sampler2D XA,XB;"
               + encs_glsl() +
        "void main(){int pid=int(gl_FragCoord.x);int per=(" + D + "+31)/32;"
        "int i0=(pid-(pid/32)*32)*per;"
        "float s=0.0;"
        "for(int k=0;k<per;k++){int i=i0+k;if(i<" + D + "){"
        "float v=(pid<32)?Y1(XA,i," + RW + ".0):Y1(XB,i," + RW + ".0);s+=v*v;}}"
        "gl_FragColor=S_ST(s);}";
}

string rms_norm_pair2(int dim, XFormat rf, bool use_eps_uniform) {
    need_dim("rms_norm_pair2", dim);
    const string eps_src = use_eps_uniform ? "eps" : "1e-5";
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, rf));
    const string DW = to_string(x_texels(dim, XFormat::I16_2));
    return head(XFormat::I16_2) + encs_glsl() + x_residual_glsl(rf) + x_pair2_glsl(rf) +
        (use_eps_uniform ? "uniform sampler2D XA,XB,W,S;uniform float eps;"
                         : "uniform sampler2D XA,XB,W,S;") +
        "void main(){int k=int(gl_FragCoord.x);"
        "float ssa=0.0,ssb=0.0;"
        "for(int p=0;p<32;p++){"
        "ssa+=S_LD(texture2D(S,vec2((float(p)+0.5)/64.0,0.5)));"
        "ssb+=S_LD(texture2D(S,vec2((float(p+32)+0.5)/64.0,0.5)));}"
        "float inva=inversesqrt(ssa/float(" + D + ")+" + eps_src + ");"
        "float invb=inversesqrt(ssb/float(" + D + ")+" + eps_src + ");"
        "vec4 wc=texture2D(W,vec2((float(k/2)+0.5)/float(" + DW + "),0.5));"
        "float wk=((k/2)*2==k)?x_i16(wc.rg):x_i16(wc.ba);"
        "float va=Y1(XA,k," + RW + ".0)*inva*wk;"
        "float vb=Y1(XB,k," + RW + ".0)*invb*wk;"
        "gl_FragColor=x_p2_st(va,vb);}";
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

string rms_stats_split(int dim, XFormat residual) {
    need_dim("rms_stats_split", dim);
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, residual));
    return head(residual) + x_residual_glsl(residual) +
        "uniform sampler2D X;"
               + encs_glsl() +
        "void main(){int pid=int(gl_FragCoord.x);int per=(" + D + "+31)/32;"
        "int i0=pid*per;float s=0.0;"
        "for(int k=0;k<per;k++){int i=i0+k;if(i<" + D + "){"
        "float v=Y1(X,i," + RW + ".0);s+=v*v;}}"
        "gl_FragColor=S_ST(s);}";
}

string rms_stats_reduce() {
    // The 32 partial sums are read through S_LD, not a local 24-bit decoder:
    // RMS_STORE=f32 changes what is in those texels, and a hardcoded decoder
    // here silently read the float bytes as fixed point (PPL 6399 instead of
    // 270.32, found 2026-11-18).
    return std::string("precision highp float;\n") +
        "uniform sampler2D S;" + encs_glsl() +
        "void main(){float s=0.0;for(int p=0;p<32;p++){"
        "s+=S_LD(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)));}"
        "gl_FragColor=S_ST(s);}";
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
                  + var + "=vec2(x_q24(texture2D(" + tex + ",vec2(t,0.5))),"
               "x_q24(texture2D(" + tex + ",vec2(t+1.0/" + AW + ".0,0.5))));}";
}

string mul_act(int dim, XFormat xf, XFormat af, XFormat bf) {
    if (xf != XFormat::I16_2)
        { fprintf(stderr, "shader mul_act: the conv block is q16 only\n"); exit(2); }
    // Output is the q16 pair the next matvec_add reads.  A 24-bit operand is
    // real-valued (x_i24q), so the product is real and x_st2 is the right store
    // for both mixes; an I16_2 operand is raw q16, hence the /256.
    const string ra = mul_act_read("A", "av", af, dim, x_texels(dim, af));
    const string rb = mul_act_read("B", "bv", bf, dim, x_texels(dim, bf));
    return head(xf) +
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
    return head(xf) +
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
          "ss+=S_LD(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)));}"
        : "float ss=S_LD(texture2D(S,vec2(0.5,0.5)));";
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
            "float v0=Y1(X,i0," + RW + ".0)*inv*x_i16(wa.rg);"
            "float v1=Y1(X,i0+1," + RW + ".0)*inv*x_i16(wa.ba);"
            "float v2=Y1(X,i0+2," + RW + ".0)*inv*x_i16(wb.rg);"
            "float v3=Y1(X,i0+3," + RW + ".0)*inv*x_i16(wb.ba);"
            "gl_FragColor=x_i8enc4(v0,v1,v2,v3);}";
    } else if (out == XFormat::F32) {
        // One value a texel: the q16 pair store below would write two values into
        // one texel and the f32 consumer would read them as raw bits.  The norm
        // weight plane stays two q16 a texel, so the index is halved here.
        body =
            "vec4 rms_f32enc(float v){uint u=floatBitsToUint(v);return vec4(float(u>>24),"
            "float((u>>16)&255u),float((u>>8)&255u),float(u&255u))/255.0;}\n"
            "void main(){int t=int(gl_FragCoord.x);"
            + ss_read +
            "float inv=inversesqrt(ss/float(" + D + ")+" + eps_src + ");"
            "int tw=t/2;int lo2=t-tw*2;"
            "vec4 wc=texture2D(W,vec2((float(tw)+0.5)/float(" + DW + "),0.5));"
            "float w=(lo2==0)?x_i16(wc.rg):x_i16(wc.ba);"
            "gl_FragColor=rms_f32enc(Y1(X,t," + RW + ".0)*inv*w);}";
    } else {
        body =
            "void main(){int t=int(gl_FragCoord.x);int i0=2*t;int i1=i0+1;"
            + ss_read +
            "float inv=inversesqrt(ss/float(" + D + ")+" + eps_src + ");" + wsplit +
            "float v0=Y1(X,i0," + RW + ".0)*inv*w0;"
            "float v1=0.0;if(i1<" + D + ")v1=Y1(X,i1," + RW + ".0)*inv*w1;" +
            (out == XFormat::BF16 ? string("gl_FragColor=x_bf16enc2(v0,v1);}")
                                  : string("gl_FragColor=x_st2(v0,v1);}"));
    }
    // x_residual_glsl(BF16) now carries the bf16 helpers itself, so the
    // conditional below is redundant for that one format -- but it is what
    // every other format's text contains, and the emitted text is the
    // numerics (gate/shader_hashes.txt), so it stays exactly as it was.
    return head(residual) + encs_glsl() + x_residual_glsl(residual) +
           (residual == XFormat::BF16 ? string() : x_codec_glsl(XFormat::BF16)) +
        "uniform sampler2D X,W,S;uniform float dummy;" + (use_eps_uniform ? "uniform float eps;" : string()) + body;
}

// ---- post-norm block row passes --------------------------------------------
// A post-norm block needs a LayerNorm (mean removed, not just the rms), a bias
// after a projection, and a non-gated FFN activation.  None of these knows what
// a transformer is: they are row-elementwise on the streams the block already
// uses, so a decoder-only post-norm model would call the same kernels.  The
// block's layer order is Session's.
std::string ln_stats_q16(int dim, XFormat rf) {
    need_dim("ln_stats_q16", dim);
    const string D = to_string(dim), RW = to_string(x_texels(dim, rf));
    // One fragment, one XST2 store: the pair is (mean, invstd) in .rg/.ba of a
    // single I16_2 texel.  XFORMAT's rule is that producers use XST* and
    // consumers use X1/X2/X4; the RWKV bit-cast form (kBC) looked available
    // because it is a helper in this file, but it is a *private* codec of that
    // block -- using it here forces the whole program to ESSL 3.00 while the
    // matching reader below is ESSL 1.00, which is exactly the mixed-language
    // kernel the format module exists to prevent.
    //
    // The q16 grid is fine for this pair because both are O(1) after a
    // normalisation: mean is a *mean* of a normalised row and invstd is
    // 1/sigma with sigma O(1) for a post-norm residual.  A model whose
    // residual is far from O(1) would need ACT=f32 here (the ACT axis), not a
    // second private codec.
    return head(XFormat::I16_2) + x_residual_glsl(rf) +
        "uniform sampler2D X;uniform float eps;"
        "void main(){float s=0.0,s2=0.0;"
        "for(int i=0;i<" + D + ";i++){float v=Y1(X,i," + RW + ".0);s+=v;s2+=v*v;}"
        "float m=s/float(" + D + ");"
        "float inv=inversesqrt(max(s2/float(" + D + ")-m*m,0.0)+eps);"
        "gl_FragColor=XST2(m,inv);}";
}

std::string ln_apply_q16(int dim, XFormat rf, XFormat out) {
    need_dim("ln_apply_q16", dim);
    const string RW = to_string(x_texels(dim, rf));
    const string WW = to_string(x_texels(dim, XFormat::I16_2));
    // XFORMAT-clean: every fetch is a macro, and each tensor is its own texture
    // in its own format.  W (the LayerNorm weight) and B (its bias) used to
    // share one two-row texture, which no macro can read -- `R1` addresses
    // y = 0.5 only -- so that "optimisation" forced two hand-written
    // `texture2D` fetches with a row pitch guessed from a comment, which is
    // exactly what the format module exists to stop.  Two one-row textures cost
    // 2 KiB and read like every other weight.
    // S is ONE I16_2 texel holding both members, so its `TW` is 1 -- the macro
    // halves the index and divides by TW, and both 0 and 1 must land on texel 0
    // (picking .rg and .ba).  Passing the target's width instead puts index 1 at
    // texel 0.5, i.e. outside the pair.
    // The *output* shape follows `x_per_texel(out)`, not a per-format literal:
    // `x_f32enc2` never existed in the format module, and writing it here made
    // every ACT=f32 run of the encoder fail to compile -- a branch that is dead
    // until the axis actually moves, which is why v5 shipped with it.
    const bool pr = x_per_texel(out) == 2;
    const auto term = [&](const string& k) {
        return "(Y1(X," + k + "," + RW + ".0)-m)*inv*Y1(W," + k + "," + WW
               + ".0)+Y1(B," + k + "," + WW + ".0)";
    };
    string body = "void main(){";
    body += "float m=Y1(S,0,1.0);float inv=Y1(S,1,1.0);";
    if (pr)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float v0=" + term("i0") + ";float v1=" + term("i0+1") + ";"
                "gl_FragColor=XST2(v0,v1);";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "gl_FragColor=XST1(" + term("i0") + ");";
    return head(out) + x_residual_glsl(rf) +
        "uniform sampler2D X,S,W,B;" + body + "}";
}

std::string bias_add_q16(int dim, XFormat xf, XFormat f) {
    need_dim("bias_add_q16", dim);
    const string AS = to_string(x_texels(dim, xf));   // X: what the matvec wrote
    const string BS = to_string(x_texels(dim, f));    // B and the output
    // X is **not** the output's format, and this kernel is the one place that
    // has to know it.  `matvec_q8` writes with `x_i24encq` unconditionally
    // (34 emitted sites -- the output half of *every* q8 projection), while the
    // activation axis may say I16_2.  Reading a 2048-texel I24 row as 1024 I16_2
    // pairs is what made the non-gated FFN's first hidden layer come back as
    // "the even values are the bias alone".
    // B rides the output's format, because B is a host-uploaded row packed in it
    // and the pass after this one reads what this writes.
    const bool pr = x_per_texel(f) == 2;
    const auto val = [&](const string& k) {
        return "Y1(X," + k + "," + AS + ".0)+X1(B," + k + "," + BS + ".0)";
    };
    string body = "void main(){";
    if (pr)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "gl_FragColor=XST2(" + val("i0") + "," + val("i0+1") + ");";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "gl_FragColor=XST1(" + val("i0") + ");";
    return head(f) + x_residual_glsl(xf) +
        "uniform sampler2D X,B;" + body + "}";
}

std::string act_swish(int dim, XFormat f) {
    need_dim("act_swish", dim);
    const string AS = to_string(x_texels(dim, f));
    // swish(x) = x*sigmoid(x), two values a fragment where the format holds two
    // and one where it holds one.  The *producer* comes from the format module
    // (`XST1`/`XST2`), never written by hand: `x_f32enc2` was invented here to
    // mean "F32 but two a fragment", the format has no such thing, and every
    // `ACT=f32` run of the encoder failed to compile -- a dead branch that only
    // shows when the axis actually moves.
    const bool pr = x_per_texel(f) == 2;
    string body = "void main(){";
    if (pr)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float x0=Y1(X,i0," + AS + ".0);float x1=Y1(X,i0+1," + AS + ".0);"
                "gl_FragColor=XST2(x0/(1.0+" + EXP("-x0") + "),x1/(1.0+" + EXP("-x1") + "));";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "float x0=Y1(X,i0," + AS + ".0);"
                "gl_FragColor=XST1(x0/(1.0+" + EXP("-x0") + "));";
    return head(f) + x_residual_glsl(f) + "uniform sampler2D X;" + body + "}";
}

// swish(x + b) in one pass.  `bias_add_f` then `act_swish` is two draws and two
// texture round trips for what is one elementwise expression, and the bias add
// alone measured 0.59 ms against the 0.50 ms matvec that fed it -- the pass is
// its own cost on a board where a draw is 1.25 ms.
//
// Two slots, because the operands are two formats: X is what the projection
// wrote (I24 today, `xf`) and B is a host-uploaded row in the output's format.
// `head(f)` names slot X for f, `x_residual_glsl(xf)` names slot Y for xf, so
// nothing collides -- the same arrangement every two-format kernel here uses.
std::string act_swish_biased(int dim, XFormat xf, XFormat f) {
    need_dim("act_swish_biased", dim);
    const string AS = to_string(x_texels(dim, xf));
    const string BS = to_string(x_texels(dim, f));
    // The sum goes into a local first.  Composing `X+B/(1+exp(-(X+B)))` straight
    // from the pieces emitted `x + b/(1+exp(-(x+b)))` -- `/` binds tighter than
    // `+`, so the activation landed on the bias alone and the translation came
    // back empty.  A local also evaluates the sum once, not twice.
    const auto sum = [&](const string& k) {
        return "Y1(X," + k + "," + AS + ".0)+X1(B," + k + "," + BS + ".0)";
    };
    const auto sw = [](const char* v) {
        const string neg = string("-") + v;
        return string(v) + "/(1.0+" + EXP(neg.c_str()) + ")";
    };
    string body = "void main(){";
    if (x_per_texel(f) == 2)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float v0=" + sum("i0") + ";float v1=" + sum("i0+1") + ";"
                "gl_FragColor=XST2(" + sw("v0") + "," + sw("v1") + ");";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "float v0=" + sum("i0") + ";"
                "gl_FragColor=XST1(" + sw("v0") + ");";
    return head(f) + x_residual_glsl(xf) + "uniform sampler2D X,B;" + body + "}";
}

// Squared ReLU on the I24 stream.  The formula is generic; the stream is fixed
// because the RWKV block writes every activation in I24_1.  A caller with
// another stream would take the format as a parameter, as act_swish does.
std::string act_sqrelu(int dim) {
    need_dim("act_sqrelu", dim);
    const string D = to_string(dim);
    return head(XFormat::I24_1) + poly_src() +
        "uniform sampler2D X;"
        "void main(){int i=int(gl_FragCoord.x);float r=max(" + rd24("X", "i", D + ".0") + ",0.0);"
        "gl_FragColor=x_q24enc(r*r);}";
}
