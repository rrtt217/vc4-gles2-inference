#include "shaders_internal.h"
#include "shaders.h"
#include "switches.h"

#include <cstring>
#include <string>

using std::string;
using std::to_string;

namespace {
// ---- how the rmsnorm's partial sums travel (RMS_STORE=i24|f32) --------------
// The stream is one XFormat at a time.  `S24_1` is a lane sum at a *fixed*
// 1/4096 step: at layer 0 the lane sums land at 55-91 LSB -- about seven bits,
// one LSB being 0.9% of a lane -- which is the 0.212% scale difference the
// board shows against Xe on layer 0's normalised row, compounding over 28
// layers.  `F32` stores the same four bytes as a bit-cast float, exact at every
// magnitude.  `i24` stays the default so no pin moves.
//
// Both codecs come from xformat.  The f32 branch used to hand-copy
// `x_f32`/`x_f32enc` under the names `s_ld`/`s_st` -- a second copy of a codec,
// which is the drift this module exists to remove.  Sharing `F32` here cannot
// confuse a scale: the format states no step, so the sum and a value stream
// decode by the same, step-free rule.
// False until a container states metadata key 19; the switch is the second
// input and the container wins, because a model that saturates the 24-bit store
// answers wrongly without f32 and the operator cannot be relied on to set it.
static bool g_rms_f32_container = false;

static XFormat rms_store_fmt() {
    // Not a function-local static: `/model` swaps the container and the answer
    // has to change with it.  Shader builders run once per program (x_prog
    // caches by name and validates the text's hash), so this is not hot, and a
    // latch here would make the second model write the first model's partial
    // sums -- which is the failure `prog_key_hash_` exists to catch one layer
    // further down.
    if (g_rms_f32_container) return XFormat::F32;
    const char* v = sw_str("RMS_STORE");
    return (v && !strcmp(v, "f32")) ? XFormat::F32 : XFormat::S24_1;
}
// The aliases the rms builders use, pointed at the selected format's own
// functions (`x_codec_glsl` below is what defines them).
static std::string rms_store_glsl() {
    return rms_store_fmt() == XFormat::F32
               ? "\n#define S_ST(V) x_f32enc(V)\n#define S_LD(C) x_f32(C)\n"
               : "\n#define S_ST(V) x_s24enc(V)\n#define S_LD(C) x_s24(C)\n";
}
// The selected format's codec plus the S_ST/S_LD aliases: every builder that
// writes or reads an rms partial sum emits this, once.
static string encs_glsl() { return x_codec_glsl(rms_store_fmt()) + rms_store_glsl(); }
}  // namespace

XFormat rms_store_format() { return rms_store_fmt(); }

void set_rms_store_f32(bool on) { g_rms_f32_container = on; }
bool rms_store_f32_from_container() { return g_rms_f32_container; }

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
    const string rd = "float i16(vec2 c){return x_i16raw(c)/256.0;}";
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

// The two-position reduce: rms_stats_reduce's 32-add loop, once per position,
// over the 64-wide partial texture.  A split-reduce backend runs this as its
// own draw and the pair norm reads the two totals, which is what makes the
// pair's divisor the serial norm's to the bit.  Summing the partials inside
// the normalising fragment instead is a different loop shape and the compiler
// lands 2 of 768 q16 values one LSB apart (Xe, 2026-11-25).
string rms_stats_reduce2(int partials) {
    const string P = to_string(partials), P2 = to_string(2 * partials);
    return std::string("precision highp float;\n") +
        "uniform sampler2D S;" + encs_glsl() +
        "void main(){int w=int(gl_FragCoord.x);float s=0.0;"
        "for(int p=0;p<" + P + ";p++){int i=w*" + P + "+p;"
        "s+=S_LD(texture2D(S,vec2((float(i)+0.5)/" + P2 + ".0,0.5)));}"
        "gl_FragColor=S_ST(s);}";
}

string rms_norm_pair2(int dim, XFormat rf, bool use_eps_uniform, bool fused_reduce) {
    need_dim("rms_norm_pair2", dim);
    const string eps_src = use_eps_uniform ? "eps" : "1e-5";
    const string D = to_string(dim);
    const string RW = to_string(x_texels(dim, rf));
    const string DW = to_string(x_texels(dim, XFormat::I16_2));
    // The serial path's total is the backend's: VC4 folds the 32-way reduce
    // into each normalising fragment (raw float), every other backend runs
    // rms_stats_reduce, which stores the total through S_ST and reads it back
    // through S_LD.  Read the same number here -- the fused form sums, the
    // split form takes the two totals rms_stats_reduce2 already made.
    const string SS = fused_reduce
        ? "float ssa=0.0,ssb=0.0;"
          "for(int p=0;p<32;p++){"
          "ssa+=S_LD(texture2D(S,vec2((float(p)+0.5)/64.0,0.5)));"
          "ssb+=S_LD(texture2D(S,vec2((float(p+32)+0.5)/64.0,0.5)));}"
        : "float ssa=S_LD(texture2D(S,vec2(0.25,0.5)));"
          "float ssb=S_LD(texture2D(S,vec2(0.75,0.5)));";
    return head(XFormat::I16_2) + encs_glsl() + x_residual_glsl(rf) + x_pair2_glsl(rf) +
        (use_eps_uniform ? "uniform sampler2D XA,XB,W,S;uniform float eps;"
                         : "uniform sampler2D XA,XB,W,S;") +
        "void main(){int k=int(gl_FragCoord.x);" + SS +
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
    // units: an I16_2 texel already holds the pair, a one-value-per-texel store
    // (the projection's i24 or f32) takes two fetches and no /256.
    const string TW = to_string(tw), AW = to_string(dim);
    if (f == XFormat::I16_2)
        return string("vec2 ") + var + "=x_i16pairq(texture2D(" + tex +
               ",vec2((float(q)+0.5)/" + TW + ".0,0.5)))/256.0;";
    // Both operands can be the projection's, so the coordinate gets a
    // per-operand name.  x_projdec is that store's reader (i24 at 1/256, or the
    // f32 bit-cast under PROJ=f32) -- reading an f32 texel as i24 is noise.
    return string("vec2 ") + var + "=vec2(0.0);{float t=(float(2*q)+0.5)/" + AW + ".0;"
                  + var + "=vec2(x_projdec(texture2D(" + tex + ",vec2(t,0.5))),"
               "x_projdec(texture2D(" + tex + ",vec2(t+1.0/" + AW + ".0,0.5))));}";
}

string mul_act(int dim, XFormat xf, XFormat af, XFormat bf) {
    if (xf != XFormat::I16_2)
        { fprintf(stderr, "shader mul_act: the conv block is q16 only\n"); exit(2); }
    // Output is the q16 pair the next matvec_add reads.  A 24-bit operand is
    // real-valued (x_i24q), so the product is real and x_st2 is the right store
    // for both mixes; an I16_2 operand is raw q16, hence the /256.
    const string ra = mul_act_read("A", "av", af, dim, x_texels(dim, af));
    const string rb = mul_act_read("B", "bv", bf, dim, x_texels(dim, bf));
    return head(xf) + x_proj_glsl(af) + x_proj_glsl(bf) +
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
            "vec4 rms_f32enc(float v){return x_f32enc(v);}\n"
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
// The v5 LayerNorm's statistics, as a split scan: 64 lanes hold sum(x)
// (0..31) and sum(x^2) (32..63) as f32 bit-cast values, so no magnitude rails
// them.  The reduce folds them into (mean, sd), scales both by a per-row
// 2^-k that puts the larger near 96 and writes k in the second row of the
// stats target; the apply multiplies x by 2^-k, so the pair is scale-free.
// The one-fragment (mean, 1/sd) q16 this replaces was safe only near sd=1
// (probes/lnstats_probe: 0.32% at 1, 17% at 100, 99% at 1000).
std::string ln_stats_split(int dim, XFormat rf) {
    need_dim("ln_stats_split", dim);
    const string D = to_string(dim), RW = to_string(x_texels(dim, rf));
    const string PER = to_string((dim + 31) / 32);
    return head(kLnPairFmt) + x_residual_glsl(rf) + x_codec_glsl(XFormat::F32) +
        "uniform sampler2D X;"
        "void main(){int j=int(gl_FragCoord.x);int lane=j-(j/32)*32;bool sq=(j>=32);"
        "int i0=lane*" + PER + ";float s=0.0;"
        "for(int k=0;k<" + PER + ";k++){int i=i0+k;if(i<" + D + "){"
        "float v=Y1(X,i," + RW + ".0);s+=sq?v*v:v;}}"
        "gl_FragColor=x_f32enc(s);}";
}

// The split scan for one row of a batch residual plane: the same lanes, the
// same sum, the row from gl_FragCoord.y.  Written as a separate text so the
// shipped one-row kernel keeps its bytes (gate/shader_hashes.sh); the X decode
// is per format because Y1 hardcodes row 0.5.
std::string ln_stats_split_row(int dim, XFormat rf, int bh) {
    need_dim("ln_stats_split_row", dim);
    const string D = to_string(dim), RW = to_string(x_texels(dim, rf));
    const string PER = to_string((dim + 31) / 32);
    // A macro, not a function: Y1 in x_residual_glsl is a macro whose two-fetch
    // ternary the driver inlines one way; a helper taking the same fetches made
    // the square sums reassociate by one LSB (the mean matched, the inverse
    // standard deviation did not).  Same text shape, row substituted.
    const string ROWV = "(float(gl_FragCoord.y)/float(BH))";
    string xv;
    if (rf == XFormat::I16_2)
        xv = "#define XV(I) ((((I)/2)*2)==(I) ? x_i16(texture2D(X,vec2((float((I)/2)+0.5)/float(" + RW +
             ")," + ROWV + ")).rg) : x_i16(texture2D(X,vec2((float((I)/2)+0.5)/float(" + RW +
             ")," + ROWV + ")).ba))\n";
    else if (rf == XFormat::I16_1)
        xv = "#define XV(I) x_i16(texture2D(X,vec2((float(I)+0.5)/float(" + RW + ")," + ROWV + ")).rg)\n";
    else if (rf == XFormat::I24_1)
        xv = "#define XV(I) x_q24(texture2D(X,vec2((float(I)+0.5)/float(" + RW + ")," + ROWV + ")))\n";
    else
        xv = "#define XV(I) x_f32(texture2D(X,vec2((float(I)+0.5)/float(" + RW + ")," + ROWV + ")))\n";
    return head(kLnPairFmt) + "#define BH " + to_string(bh) + "\n" +
        x_residual_glsl(rf) + x_codec_glsl(XFormat::F32) +
        "uniform sampler2D X;\n" + xv +
        "void main(){int j=int(gl_FragCoord.x);int lane=j-(j/32)*32;bool sq=(j>=32);"
        "int i0=lane*" + PER + ";float s=0.0;"
        "for(int k=0;k<" + PER + ";k++){int i=i0+k;if(i<" + D + "){"
        "float v=XV(i);s+=sq?v*v:v;}}"
        "gl_FragColor=x_f32enc(s);}";
}

// The reduce for every row: grid (2, n), x=0 mean and x=1 1/sd, both f32, into
// a (2, seq) plane.  The loop is ln_stats_reduce's.
std::string ln_stats_reduce_row(int dim, int bh) {
    return head(kLnPairFmt) + "#define BH " + to_string(bh) + "\n" +
        x_codec_glsl(XFormat::F32) +
        "uniform sampler2D S;uniform float eps;"
        "void main(){int b=int(gl_FragCoord.y);float sx=0.0,sxx=0.0;"
        "for(int p=0;p<32;p++){"
        "sx+=x_f32(texture2D(S,vec2((float(p)+0.5)/64.0,(float(b)+0.5)/float(BH))));"
        "sxx+=x_f32(texture2D(S,vec2((float(p+32)+0.5)/64.0,(float(b)+0.5)/float(BH))));}"
        "float mean=sx/float(" + to_string(dim) + ");"
        "float sd=sqrt(max(sxx/float(" + to_string(dim) + ")-mean*mean,0.0)+eps);"
        "if(gl_FragCoord.x<1.0)gl_FragColor=x_f32enc(mean);"
        "else gl_FragColor=x_f32enc(1.0/sd);}";
}

// The apply for one row: X row `xr`, the (mean, 1/sd) at (0/1, xr) of the batch
// statistics plane, the output still this position's one-row target.  The
// expression is ln_apply_q16's.
std::string ln_apply_q16_row(int dim, XFormat rf, XFormat out, int bh) {
    need_dim("ln_apply_q16_row", dim);
    const string RW = to_string(x_texels(dim, rf));
    const string WW = to_string(x_texels(dim, kLnPairFmt));
    const auto term = [&](const string& k) {
        return "(XV(" + k + ")-mu)*inv*wv(" + k + ")+bv(" + k + ")";
    };
    const string xv = rf == XFormat::I16_2
        ? "float XV(int i){vec4 c=texture2D(X,vec2((float(i/2)+0.5)/float(" + RW +
          "),xr));return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}"
        : "float XV(int i){return x_q24(texture2D(X,vec2((float(i)+0.5)/float(" + RW + "),xr)));}";
    string body =
        "float wv(int i){vec4 c=texture2D(W,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
        "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n"
        "float bv(int i){vec4 c=texture2D(B,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
        "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n" + xv +
        "void main(){float mu=x_f32(texture2D(S,vec2(0.25,xr)));"
        "float inv=x_f32(texture2D(S,vec2(0.75,xr)));";
    const int n = x_per_texel(out);
    if (n == 4)
        body += "int t=int(gl_FragCoord.x);int i0=4*t;"
                "gl_FragColor=XST4(" + term("i0") + "," + term("i0+1") + ","
                + term("i0+2") + "," + term("i0+3") + ");}";
    else if (n == 2)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float v0=" + term("i0") + ";float v1=" + term("i0+1") + ";"
                "gl_FragColor=XST2(v0,v1);}";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "gl_FragColor=XST1(" + term("i0") + ");}";
    return head(out) + "#define BH " + to_string(bh) + "\n" +
        x_residual_glsl(rf) + x_codec_glsl(XFormat::F32) +
        "uniform sampler2D X,S,W,B;uniform float xr;" + body;
}

std::string ln_stats_reduce(int dim) {
    const string D = to_string(dim);
    // Row 0 is the mean, row 1 the inverse standard deviation, both f32.  The
    // two have reciprocal scales: the mean lives on the row's own grid, while
    // 1/sd is unbounded as the row flattens, so one fixed-point pair cannot
    // serve both.  The pair this replaced scaled both by 2^-k (k from
    // max(|mean|, sd)) and rescaled x in the apply to match; probes/
    // lnstats_probe measures that pair against this one, and this one is the
    // floor on every row, including the rows where |mean| >> sd and the
    // subtraction cancels.  f32 here means a bit-cast, i.e. an ESSL 3.00
    // kernel -- the f32 and bf16 streams already are.
    return head(kLnPairFmt) + x_codec_glsl(XFormat::F32) +
        "uniform sampler2D S;uniform float eps;"
        "void main(){float sx=0.0,sxx=0.0;"
        "for(int p=0;p<32;p++){"
        "sx+=x_f32(texture2D(S,vec2((float(p)+0.5)/64.0,0.5)));"
        "sxx+=x_f32(texture2D(S,vec2((float(p+32)+0.5)/64.0,0.5)));}"
        "float mean=sx/float(" + D + ");"
        "float sd=sqrt(max(sxx/float(" + D + ")-mean*mean,0.0)+eps);"
        "if(gl_FragCoord.y<1.0)gl_FragColor=x_f32enc(mean);"
        "else gl_FragColor=x_f32enc(1.0/sd);}";
}

std::string ln_apply_q16(int dim, XFormat rf, XFormat out) {
    need_dim("ln_apply_q16", dim);
    const string RW = to_string(x_texels(dim, rf));
    const string WW = to_string(x_texels(dim, kLnPairFmt));
    // S row 0 is the mean and row 1 the inverse std, both f32 (see
    // ln_stats_reduce); W and B are their own layout (kLnPairFmt), not the
    // residual's: `rf` is the format of X only.
    // The *output* shape follows `x_per_texel(out)`, not a per-format literal:
    // `x_f32enc2` never existed in the format module, and writing it here made
    // every ACT=f32 run of the encoder fail to compile.
    const auto term = [&](const string& k) {
        return "(Y1(X," + k + "," + RW + ".0)-mu)*inv*wv(" + k + ")+bv(" + k + ")";
    };
    string body =
        "float wv(int i){vec4 c=texture2D(W,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
        "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n"
        "float bv(int i){vec4 c=texture2D(B,vec2((float(i/2)+0.5)/float(" + WW + "),0.5));"
        "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}\n"
        "void main(){float mu=x_f32(texture2D(S,vec2(0.5,0.25)));"
        "float inv=x_f32(texture2D(S,vec2(0.5,0.75)));";
    const int n = x_per_texel(out);
    if (n == 4)
        body += "int t=int(gl_FragCoord.x);int i0=4*t;"
                "gl_FragColor=XST4(" + term("i0") + "," + term("i0+1") + ","
                + term("i0+2") + "," + term("i0+3") + ");}";
    else if (n == 2)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float v0=" + term("i0") + ";float v1=" + term("i0+1") + ";"
                "gl_FragColor=XST2(v0,v1);}";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "gl_FragColor=XST1(" + term("i0") + ");}";
    return head(out) + x_residual_glsl(rf) + x_codec_glsl(XFormat::F32) +
        "uniform sampler2D X,S,W,B;" + body;
}

// The bias pass for one row of a batch plane: X is read at the row the draw
// writes (gl_FragCoord.y), B is still the one-row bias.  The arithmetic is
// bias_add_q16's; the row substitution is the only change, which is why it is a
// separate text rather than a flag in the hotspot.
std::string bias_add_q16_batch(int dim, XFormat xf, XFormat f, int rows) {
    need_dim("bias_add_q16_batch", dim);
    const string AS = to_string(x_texels(dim, xf));
    const string BS = to_string(x_texels(dim, f));
    const bool pr = x_per_texel(f) == 2;
    // The two X layouts the qkv plane can have (PROJ is I24_1 or F32; the qkv
    // store is the p16 pair).  XV(I) returns X's value at index I for row y.
    string xv;
    if (xf == XFormat::I24_1)
        xv = "float XV(int i){return x_q24(texture2D(X,vec2((float(i)+0.5)/float(" + AS +
             "),(float(gl_FragCoord.y)+0.5)/float(BROWS))));}";
    else if (xf == XFormat::F32)
        xv = "float XV(int i){return x_f32(texture2D(X,vec2((float(i)+0.5)/float(" + AS +
             "),(float(gl_FragCoord.y)+0.5)/float(BROWS))));}";
    else
        xv = "float XV(int i){vec4 c=texture2D(X,vec2((float(i/2)+0.5)/float(" + AS +
             "),float(gl_FragCoord.y)/float(BROWS)));"
             "return ((i/2)*2==i)?x_i16(c.rg):x_i16(c.ba);}";
    const auto val = [&](const string& k) {
        return "XV(" + k + ")+X1(B," + k + "," + BS + ".0)";
    };
    string body = "void main(){";
    if (pr)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "gl_FragColor=XST2(" + val("i0") + "," + val("i0+1") + ");";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "gl_FragColor=XST1(" + val("i0") + ");";
    // x_residual_glsl(xf) already carries the X codec and the X1/XST helpers
    // for this format; x_glsl(xf) on top redefined them.
    return head(f) + "#define BROWS " + to_string(rows) + "\n" + x_residual_glsl(xf) +
        "uniform sampler2D X,B;" + xv + body + "}";
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
    if (x_per_texel(f) == 4)
        body += "int t=int(gl_FragCoord.x);int i0=4*t;"
                "float x0=Y1(X,i0," + AS + ".0);float x1=Y1(X,i0+1," + AS + ".0);"
                "float x2=Y1(X,i0+2," + AS + ".0);float x3=Y1(X,i0+3," + AS + ".0);"
                "gl_FragColor=XST4(x0/(1.0+" + EXP("-x0") + "),x1/(1.0+" + EXP("-x1") + "),"
                "x2/(1.0+" + EXP("-x2") + "),x3/(1.0+" + EXP("-x3") + "));";
    else if (pr)
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
    const int n = x_per_texel(f);
    if (n == 4)
        body += "int t=int(gl_FragCoord.x);int i0=4*t;"
                "float v0=" + sum("i0") + ";float v1=" + sum("i0+1") + ";"
                "float v2=" + sum("i0+2") + ";float v3=" + sum("i0+3") + ";"
                "gl_FragColor=XST4(" + sw("v0") + "," + sw("v1") + ","
                + sw("v2") + "," + sw("v3") + ");";
    else if (n == 2)
        body += "int t=int(gl_FragCoord.x);int i0=2*t;"
                "float v0=" + sum("i0") + ";float v1=" + sum("i0+1") + ";"
                "gl_FragColor=XST2(" + sw("v0") + "," + sw("v1") + ");";
    else
        body += "int i0=int(gl_FragCoord.x);"
                "float v0=" + sum("i0") + ";"
                "gl_FragColor=XST1(" + sw("v0") + ");";
    return head(f) + x_residual_glsl(xf) + "uniform sampler2D X,B;" + body + "}";
}

// ---- the BiCrossGate FFN's elementwise passes ------------------------------
// Three passes no other block needs.  The block is
//
//     q1,q2 = q(x).chunk(2);  a = act1(q1*x);  c = act2(q2*x)
//     y = down( gate(concat(a,c)) * up(x) )
//
// so a kernel has to (a) multiply a matvec output by the activation stream,
// (b) apply gelu to one half and silu to the other, and (c) multiply two
// full-width activation streams.  None of the existing kernels can do it:
// `swiglu*` read both operands from *one* texture (two rows, two lanes or two
// x-halves), `mul_act` is gated to the LFM2 conv block and exits unless its
// output is I16_2, and `rwkv_mul` is I24_1-only and gated to RWKV.
//
// The erf here is the Abramowitz-Stegun 7.1.26 form, |error| < 1.5e-7 -- the
// exact-form GELU PyTorch's F.gelu computes.  `tanh(0.7979(x+0.044715x^3))` is
// the other common spelling and it is a *different function* where this block's
// operand lives (it reaches |15|), so it is not substituted.  Horner on
// `t = 1/(1+p|x|)` keeps it to one reciprocal and no branch, which is what
// VC4 wants (docs/vc4-glsl-costs.md: a data-dependent branch costs a quad).
static string erf_src() {
    // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7 -- the exact-form GELU
    // PyTorch's F.gelu computes.  EXP() is resolved here, at build time, so
    // EXP=poly reaches it like every other kernel's exponential.
    return string("float erf_a(float z){float s=z<0.0?-1.0:1.0;float x=abs(z);"
                  "float t=1.0/(1.0+0.3275911*x);"
                  "float y=t*(0.254829592+t*(-0.284496736+t*(1.421413741+"
                  "t*(-1.453152027+t*1.061405429))));"
                  "return s*(1.0-y*") + EXP("-x*x") + ");}\n";
}
static string gelu_fwd() {
    return "float gelu_a(float v){return 0.5*v*(1.0+erf_a(v*0.70710678));}\n";
}

// One half of the BiCrossGate gate: `act(qk * x)` for the k-th half of a 2*dim
// matvec output.  Two draws, one a half, so each fragment reads one value of the
// matvec result and one of the activation stream and writes one -- no
// cross-fragment addressing and no second program text per half (`half` is a
// uniform).
std::string bicross_act(int dim, XFormat pf, XFormat xf, XFormat out, int act) {
    need_dim("bicross_act", dim);
    const string D = to_string(dim), PW = to_string(x_texels(2 * dim, pf));
    const string XW = to_string(x_texels(dim, xf));
    // `half` picks the slice of the 2*dim matvec output, and `dst` where the
    // result lands: both halves are written into ONE 2*dim activation texture
    // (at i and i+dim), so the following gate matvec reads it as a single row.
    const string q = "x_projdec(texture2D(A,vec2((float(i)+0.5)/float(" + PW + "),0.5)))";
    const string v = q + "*X1(X,i," + XW + ".0)";
    const string neg = "-(" + v + ")";
    const string silu = "(" + v + "/(1.0+" + EXP(neg.c_str()) + "))";
    // `act` carries both codes: low byte for q1*x, high byte for q2*x.
    const int act1 = act & 0xFF, act2 = (act >> 8) & 0xFF;
    const string a = (act1 == 2) ? ("gelu_a(" + v + ")") : silu;
    // The store index is `i + hsel*dim`, so the output texture is 2*dim wide and
    // the two draws fill it without a second target or a concatenate pass.
    // `head(out)` already carries X1/XST1 for the output stream and the
    // projection's own reader, so `x_projdec` comes from there; X1 for the
    // *activation* stream is the one thing it does not define, and asking
    // x_residual_glsl for it when the two streams are the same format would
    // redefine Y1/Y2/YST1 and fail the preprocessor.
    // The store is the *output* stream's, and I16_2 defines XST2 only -- a
    // one-value store there is XST2(v,0) (the .rg half), which is what the
    // gate matvec's X1 reads back.  Writing XST1 unconditionally compiled for
    // i24/f32 and failed for the default i16.
    // The 2*dim output is `concat(a, c)`, so output index o < dim is act1's and
    // o >= dim is act2's.  For I16_2 a fragment at texel t holds values 2t and
    // 2t+1, which always land in the same half (dim is even), so the two lanes
    // share one activation -- no cross-half pairing, which is what an earlier
    // version of this kernel got wrong (it paired i with i+dim and the gate
    // then read a mixture of the two halves).
    const bool pr = x_per_texel(out) == 2;
    // One expression a half; the same text is used for both lanes, so the
    // activation is chosen by the lane's half index at run time.
    auto A1v = [&](const char* v) {
        const string t(v);
        return (act1 == 2) ? ("gelu_a(" + t + ")")
                           : ("(" + t + "/(1.0+" + EXP(("-(" + t + ")").c_str()) + "))");
    };
    auto A2v = [&](const char* v) {
        const string t(v);
        return (act2 == 2) ? ("gelu_a(" + t + ")")
                           : ("(" + t + "/(1.0+" + EXP(("-(" + t + ")").c_str()) + "))");
    };
    string body;
    if (pr)
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;int i1=i0+1;"
               "int h0=i0/" + D + ";int h1=i1/" + D + ";"
               "float v0=x_projdec(texture2D(A,vec2((float(i0)+0.5)/float(" + PW + "),0.5)))"
               "*X1(X,i0-" + D + "*h0," + XW + ".0);"
               "float v1=x_projdec(texture2D(A,vec2((float(i1)+0.5)/float(" + PW + "),0.5)))"
               "*X1(X,i1-" + D + "*h1," + XW + ".0);"
               "float a0=(h0==0)?" + A1v("v0") + ":" + A2v("v0") + ";"
               "float a1=(h1==0)?" + A1v("v1") + ":" + A2v("v1") + ";"
               // The consumer is a matvec, which reads the activation stream in
               // *raw q16* (X2Q).  XST2 stores a value (it scales by 256 inside),
               // so writing it here made every gate dot product 256x too large
               // and the projection saturated at 127.996.  x_st2q takes the raw
               // units the reader expects.
               "gl_FragColor=x_st2q(a0*256.0,a1*256.0);}";
    else
        body = "void main(){int o=int(gl_FragCoord.x);int h=o/" + D + ";"
               "float v=x_projdec(texture2D(A,vec2((float(o)+0.5)/float(" + PW + "),0.5)))"
               "*X1(X,o-" + D + "*h," + XW + ".0);"
               "float a=(h==0)?" + A1v("v") + ":" + A2v("v") + ";"
               "gl_FragColor=x_st1q(a*256.0);}";
    return head(out) + poly_src() + x_proj_glsl(pf) + erf_src() + gelu_fwd() +
        "uniform sampler2D A,X;uniform float dummy;" + body;
}

// A*B, two different samplers, both full-width activation streams, result in
// the activation stream.  The dense path had no such kernel (see the note
// above); this is it, and it stays generic over the two input formats so
// ACT=i16 / i8 / bf16 / f32 all reach it through xformat's own decoders.
// A*B elementwise, two samplers, each operand in its own stream format.  The
// BiCrossGate block needs it because its two operands are the *projection*
// store (matvec outputs) while the result belongs in the activation stream --
// and no existing kernel reads two operands with two non-activation formats.
std::string mul2(int dim, XFormat out, XFormat af, XFormat bf) {
    need_dim("mul2", dim);
    const string AW = to_string(x_texels(dim, af)), BW = to_string(x_texels(dim, bf));
    const string D = to_string(dim);
    const bool pr = x_per_texel(out) == 2;
    // The projection store is addressed by value index -- which is what the
    // multiply below needs; the q16 stream is addressed by pair.
    auto rd = [&](const char* tex, XFormat f, const string& i) {
        if (f == XFormat::I16_2) {
            const string TW = to_string(x_texels(dim, f));
            return string("x_i16(texture2D(") + tex + ",vec2((float(int(" + i +
                   ")/2)+0.5)/" + TW + ".0,0.5))." +
                   "(((" + i + ")&1)==0?rg:ba)";
        }
        return string("x_projdec(texture2D(") + tex + ",vec2((float(" + i +
               ")+0.5)/" + to_string(dim) + ".0,0.5)))";
    };
    string body;
    if (pr)
        body = "void main(){int t=int(gl_FragCoord.x);int i0=2*t;"
               "float a0=" + rd("A", af, "i0") + ";float a1=" + rd("A", af, "i0+1") + ";"
               "float b0=" + rd("B", bf, "i0") + ";float b1=" + rd("B", bf, "i0+1") + ";"
               "gl_FragColor=XST2(a0*b0,a1*b1);}";
    else
        body = "void main(){int i=int(gl_FragCoord.x);"
               "gl_FragColor=XST1(" + rd("A", af, "i") + "*" + rd("B", bf, "i") + ");}";
    // x_projdec comes from x_proj_glsl for whichever operand uses it; the q16
    // macro X1 is NOT used here, because it is defined by the *output* format
    // (head(out)) and reading a projection texel through it produced 0.25 every
    // second value.
    return head(out) + x_codec_glsl(af) + x_proj_glsl(af) + x_proj_glsl(bf) +
        "uniform sampler2D A,B;" + body;
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
