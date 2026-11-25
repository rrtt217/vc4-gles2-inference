#include "shaders_internal.h"
#include "shaders.h"

#include <string>

using std::string;
using std::to_string;

// ---- the 2-bit ternary matvec (dtype 7, `--quant t3`) ---------------------
//
// One RGBA8 texel holds 16 weights, four a byte, two bits a weight: bit 0
// magnitude, bit 1 sign, so a weight is `mag * (1 - 2*sign)` with no compare and
// no branch.  Byte j of a texel covers columns 16q + 4j .. 16q + 4j + 3, which is
// exactly the activation texel the dot product multiplies it by, so the weight
// fetch and the activation fetch advance together -- four weight bytes and four
// activation quads an iteration, against q8's one and one.
//
// The decode needs integer bitwise ops, so these kernels are ESSL 3.00 and carry
// a `//T3` marker that link_program() keys on (the q4 nibble split does the same
// with "&15").  probes/t3mv_probe priced this against q8 on both machines: on
// VC4 the uint decode is 2.44 ns/MAC against q8's 2.98 (the ES2 float decode is
// 7.5, i.e. worse than q8), on Iris Xe it is no faster.  docs/performance.md
// 2026-11-18.
// One iteration reads ONE weight texel and ONE activation texel, the shape the
// q8 kernel has -- not the 16-weights-a-texel shape the bandwidth wants.
//
// That is not an oversight.  VC4 returns undefined data from a fragment that
// samples the same texture twice once the texture is tall, which is why
// wrow_scale() packs a row pair into one texel and why the q6 planes live in two
// samplers; probes/wrow_probe measured 88 of 896 rows wrong on a two-fetch
// kernel that is exact on Xe.  A 16-weight iteration needs four activation
// fetches (X4Q is four texture2D calls on the one-value-a-texel formats), and on
// the board that is exactly the shape that goes wrong: the t3 container answered
// `ovarian` twelve times with it and is correct with this one (2026-11-18).
//
// The cost is the weight texel's other three bytes: four weights are decoded and
// four used, so the kernel reads 4x the weight texels it strictly needs while
// moving 4x fewer bytes than q8 per texel -- i.e. it matches q8's fetch count
// and halves its traffic.  T3_WIDE=1 emits the 16-wide form for a machine where
// the sampler can be sampled twice (Iris Xe: measured correct there).

namespace {
static string t3_dec_def() {
    // One byte -> four ternary weights as a vec4, the uint hoisted once.  The
    // sign is bit-cast into a float: +-1.0f is 0x3F800000 | (sign<<31), times
    // the magnitude bit.  Integer shift/or (1 QIR each, add slot) pair with the
    // final mul (mul slot); the old mag*(1-2*sign) spent two float ops on the
    // sign instead.  uint() truncates the positive bb*255+0.5 exactly as
    // uint(floor()) would, so the ~5-QIR floor is gone.  Measured on VC4:
    // 3.01 ns/MAC against the old form's 3.46 (probes/t3dec_probe, -13%),
    // byte-identical -- the same {0,+-1} vec4 in the same accumulation order.
    return "vec4 t3dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
           "uintBitsToFloat(0x3F800000u|(((u>>1)&1u)<<31))*float(u&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>3)&1u)<<31))*float((u>>2)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>5)&1u)<<31))*float((u>>4)&1u),"
           "uintBitsToFloat(0x3F800000u|(((u>>7)&1u)<<31))*float((u>>6)&1u));}\n";
}
static string t3_byte(const string& byte_expr) {
    return "t3dec(" + byte_expr + ")";
}

// The 2-bit fetch, and the tile fold, in one place.  The body is a texel grid
// of (cols/8) x (rows/2): one texel carries rows 2t and 2t+1, each with two
// column groups in two bytes (mmcf_hf.t3mat, docs/checkpoint-container.md).
static string t3_fetch(const string& g, const string& t, int nq, int th, int tiles) {
    const string NQ = to_string(nq) + ".0";
    // One tile is every matrix in this model; the tiled form costs 1.8x on VC4
    // (probes/t3mv_probe: 9.73 ns/MAC tiled against 5.47 untiled, same body), so
    // it is only emitted when a body really needs folding.
    if (tiles <= 1)
        return string("texture2D(W,vec2((") + g + "+0.5)/" + NQ + ",(" + t + "+0.5)/" +
               to_string(th) + ".0))";
    // `t` and `g` arrive as float expressions: ESSL has no implicit int->float,
    // and `(t+0.5)` with an int t is a compile error, not a promotion.
    return string("texture2D(W,vec2((") + g + "+float(int((" + t + ")/" + to_string(th) +
           ".0))*" + NQ + "+0.5)/(" + NQ + "*" + to_string(tiles) + ".0),(mod(" + t + "," +
           to_string(th) + ".0)+0.5)/" + to_string(th) + ".0))";
}
}  // namespace

string matvec_t3(int rows, int cols, XFormat xf, bool wrow, int t3_th, int t3_tiles) {
    need_shape("matvec_t3", rows, cols, 8);
    const int axw = x_texels(cols, xf);
    const int nq = (cols + 7) / 8;                    // texels a row pair
    const string AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    return "//T3\n" + head(xf) + "precision highp int;\n" + t3_dec_def() +
           "uniform sampler2D W,X;uniform float ws;" + WD +
           "void main(){int r=int(gl_FragCoord.x);int t=r/2;float h=float(r-2*t);float s=0.0;"
           "for(int g=0;g<" + to_string(nq) + ";g++){"
           "vec4 c=" + t3_fetch("float(g)", "float(t)", nq, t3_th, t3_tiles) + ";"
           // The two bytes this row owns: .rg is row 2t, .ba is row 2t+1.  mix()
           // rather than a dynamic index or an if-chain -- both were measured to
           // fail, on one driver or the other.
           "vec2 bb=mix(c.ba,c.rg,1.0-h);"
           "vec4 xa=X4Q(X,g*8," + AXW + ".0);"
           "vec4 xb=X4Q(X,g*8+4," + AXW + ".0);"
           "s+=dot(" + t3_byte("bb.x") + ",xa)+dot(" + t3_byte("bb.y") + ",xb);}"
           "gl_FragColor=x_q24encq(s*" + SC + ");}";
}

string matvec_t3_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool wrow, int t3_th, int t3_tiles) {
    need_shape("matvec_t3_add", rows, cols, 8);
    const int axw = x_texels(cols, xf);
    const int nq = (cols + 7) / 8;
    const string AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return "//T3\n" + head(xf) + "precision highp int;\n" + t3_dec_def() + x_residual_glsl(rf) +
           (of == rf ? std::string() : x_store_glsl(of)) +
           "uniform sampler2D W,X,R;uniform float ws;" + WD +
           "void main(){int r=int(gl_FragCoord.x);int t=r/2;float h=float(r-2*t);float s=0.0;"
           "for(int g=0;g<" + to_string(nq) + ";g++){"
           "vec4 c=" + t3_fetch("float(g)", "float(t)", nq, t3_th, t3_tiles) + ";"
           "vec2 bb=mix(c.ba,c.rg,1.0-h);"
           "vec4 xa=X4Q(X,g*8," + AXW + ".0);"
           "vec4 xb=X4Q(X,g*8+4," + AXW + ".0);"
           "s+=dot(" + t3_byte("bb.x") + ",xa)+dot(" + t3_byte("bb.y") + ",xb);}"
           "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
           "gl_FragColor=" + ST + "(s*" + SC + "+res);}";
}

// The 2-bit pair kernel: two output rows a fragment, so all four bytes of one
// weight fetch are used (b0,b1 = row 2t's two column groups, b2,b3 = row 2t+1's)
// and the activation fetch feeds both dot products.  16 MACs a fetch, against
// the one-row kernel's 8.  The store is the engine's existing x_st2q pair
// layout, so every reader of a paired target (rope/qk-norm on qkv, rmsnorm on
// the residual) is unchanged -- probes/t3mv_probe prices this form at
// 1.21-1.25 ns/MAC on VC4 against 2.5 for the one-row one (2026-11-19).
string matvec_t3_pair(int rows, int cols, XFormat xf, bool wrow, int t3_th, int t3_tiles) {
    need_shape("matvec_t3_pair", rows, cols, 8);
    const int axw = x_texels(cols, xf);
    const int nq = (cols + 7) / 8;
    const string AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "vec2 sc=wsr2(2*t);" : "";
    const string SC0 = wrow ? "sc.x" : "ws", SC1 = wrow ? "sc.y" : "ws";
    return "//T3\n" + head(xf) + "precision highp int;\n" + t3_dec_def() + x_pair2_glsl(xf) +
           "uniform sampler2D W,X;uniform float ws;" + WD +
           "void main(){int t=int(gl_FragCoord.x);" + SP +
           "float s0=0.0,s1=0.0;"
           "for(int g=0;g<" + to_string(nq) + ";g++){"
           "vec4 c=" + t3_fetch("float(g)", "float(t)", nq, t3_th, t3_tiles) + ";"
           "vec4 xa=X4Q(X,g*8," + AXW + ".0);"
           "vec4 xb=X4Q(X,g*8+4," + AXW + ".0);"
           "s0+=dot(" + t3_byte("c.r") + ",xa)+dot(" + t3_byte("c.g") + ",xb);"
           "s1+=dot(" + t3_byte("c.b") + ",xa)+dot(" + t3_byte("c.a") + ",xb);}"
           "gl_FragColor=x_st2q(s0*" + SC0 + ",s1*" + SC1 + ");}";
}

string matvec_t3_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of, bool wrow,
                          int t3_th, int t3_tiles) {
    need_shape("matvec_t3_add_pair", rows, cols, 8);
    const int axw = x_texels(cols, xf);
    const int nq = (cols + 7) / 8;
    const string AXW = to_string(axw);
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SP = wrow ? "vec2 sc=wsr2(2*t);" : "";
    const string SC0 = wrow ? "sc.x" : "ws", SC1 = wrow ? "sc.y" : "ws";
    return "//T3\n" + head(xf) + "precision highp int;\n" + t3_dec_def() + x_pair2_glsl(of) +
           x_residual_glsl(rf) +
           "uniform sampler2D W,X,R;uniform float ws;" + WD +
           "void main(){int t=int(gl_FragCoord.x);int r0=2*t;int r1=min(r0+1," +
           to_string(rows - 1) + ");" + SP +
           "float res0=Y1Q(R,r0," + to_string(x_texels(rows, rf)) + ".0);"
           "float res1=Y1Q(R,r1," + to_string(x_texels(rows, rf)) + ".0);"
           "float s0=0.0,s1=0.0;"
           "for(int g=0;g<" + to_string(nq) + ";g++){"
           "vec4 c=" + t3_fetch("float(g)", "float(t)", nq, t3_th, t3_tiles) + ";"
           "vec4 xa=X4Q(X,g*8," + AXW + ".0);"
           "vec4 xb=X4Q(X,g*8+4," + AXW + ".0);"
           "s0+=dot(" + t3_byte("c.r") + ",xa)+dot(" + t3_byte("c.g") + ",xb);"
           "s1+=dot(" + t3_byte("c.b") + ",xa)+dot(" + t3_byte("c.a") + ",xb);}"
           "gl_FragColor=x_p2_stq(s0*" + SC0 + "+res0,s1*" + SC1 + "+res1);}";
}
