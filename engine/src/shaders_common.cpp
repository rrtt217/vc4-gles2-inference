#include "shaders_internal.h"
#include "shaders.h"
#include "switches.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using std::string;
using std::to_string;

namespace {
// FFN hidden store scale (metadata key 15): the SwiGLU output is written as
// value * 2^-hs so a hidden over the q16 activation rail still fits, and w2's
// read-back multiplies by 2^hs (folded into its scale on the host).  hs = 0
// emits no text at all, so a container without the key produces byte-identical
// GLSL and gate/shader_hashes.sh stays green.
static const char* const kHidScaleLit[] = {"", "0.5", "0.25", "0.125", "0.0625", "0.03125"};
static int hid_scale_max() { return (int)(sizeof(kHidScaleLit) / sizeof(kHidScaleLit[0])) - 1; }
}  // namespace

string head(XFormat xf) {
    return "precision highp float;\n" + x_glsl(xf);
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

// ---- the exponential: EXP=drv|poly -----------------------------------------
// VC4's exp2/log2 are the shared SFU's table lookups and Mesa does not refine
// them, unlike rcp/rsqrt which get a Newton-Raphson step (vc4_program.c); its
// own rpi3 piglit list still fails fs-exp-*/exp2-*/log2-*/pow-*.  Measured on
// the board: 4.6e-05 worst relative error over the attention range against Xe's
// 2.2e-06, and the poly below is 1.5e-06 on the board / 1.1e-06 on Xe for
// +0.4-1.8% decode (docs/performance.md, 2026-11-17).
//
// It is one axis and not one per block because the SiLU carries five of the
// eight sites (`a/(1+exp(-a))` in the FFN kernels) and doubles as the sigmoid
// in the RWKV gate kernels: 71,680 exp calls a token in pocket's FFN alone,
// against att_v's two per position.
static const char* kExpPolySource =
    "float x_exp(float x){"
    "float k=floor(x*1.4426950408889634+0.5);"
    "float r=x-k*0.6931471805599453;"
    "float t=1.0+r*(1.0+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+r/720.0)))));"
    "return t*uintBitsToFloat(uint(int(k)+127)<<23);}\n";

// `exp(<arg>)` as this build should spell it.
string EXP(const char* arg) {
    return std::string(exp_poly_enabled() ? "x_exp(" : "exp(") + arg + ")";
}

// The poly's definition, or nothing, for the shaders that call EXP().
std::string poly_src() { return exp_poly_enabled() ? kExpPolySource : ""; }

// Round/clamp to q16, then swiglu: the fused gate kernels.
// The fused gate kernels' q16 round + SiLU.  A function, not a constant, because
// the SiLU's exponential depends on EXP and a static-initialised string would
// read the registry before sw_init has run.
string qz_sl() {
    return poly_src() +
        "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
        "float sl(float q,float g){float a=q/256.0;return (a/(1.0+" + EXP("-a") + "))*(g/256.0);}";
}

string hid_scale_def(int hs) {
    if (hs <= 0 || hs > hid_scale_max()) return "";
    return "#define HIDSCALE " + string(kHidScaleLit[hs]) + "\n";
}
string hid_scale_mul(int hs) {
    if (hs <= 0 || hs > hid_scale_max()) return "";
    return "*HIDSCALE";
}

// Shape contract.  The q8 kernels step the column loop by four; the q4 kernels
// take eight weights per texel in groups of 32.  A width that is not a multiple
// of that reads past the activation texture and the shader still compiles - it
// just returns a wrong number a layer later, which is how a mis-sized call
// survives to the PPL run.  Check where the text is made, once per program.
void need_shape(const char* who, int rows, int cols, int unit) {
    if (rows <= 0 || cols <= 0 || cols % unit != 0) {
        fprintf(stderr, "shader %s: rows=%d cols=%d (cols must be a positive multiple of %d)\n",
                who, rows, cols, unit);
        exit(2);
    }
}

void need_dim(const char* who, int n) {
    if (n <= 0) {
        fprintf(stderr, "shader %s: size=%d\n", who, n);
        exit(2);
    }
}

// Per-row q8 weight scale.  A container may carry one f32 per output row
// (tensor id part 3) instead of a single per-matrix scale, because one scale
// for a 640x1440 matrix leaves 58-66% of void.1's layer-0 FFN weights at zero
// and the model then misspells bytes (max|d| 14.4 logits against f32; per-row
// brings it to 0.4).  The plane is one texel wide, one row per texel, so the
// row a fragment writes is the texel it reads.  Emitted only when the matrix
// has one: a model without it links exactly the text it linked before.
string wrow_scale(int rows, const char* smp, const char* fn, const char* unit,
                  bool hi) {
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
        return string("uniform sampler2D ") + smp + ";vec2 " + fn + "2(int t){vec4 c=texture2D(" + smp +
               ",vec2(0.125,(float(t)+0.5)/float(" + to_string(np) + ")));"
               "return vec2(c.r*65280.0+c.g*255.0,c.b*65280.0+c.a*255.0)*" + unit +
               "*(1.0/65536.0);}"
               "float " + fn + "(int r){return 0.0;}\n";
    }
    return string("uniform sampler2D ") + smp + ";vec4 " + fn + "p(int r){return texture2D(" + smp +
           ",vec2(0.125,(float(r/2)+0.5)/float(" + to_string(np) + ")));}"
           "float " + fn + "(int r){vec4 c=" + string(fn) + "p(r);"
           "return (mod(float(r),2.0)<0.5?c.r*65280.0+c.g*255.0:c.b*65280.0+c.a*255.0)*"
           + unit + "*(1.0/65536.0);}"
           "vec2 " + fn + "2(int r){vec4 c=" + string(fn) + "p(r);"
           "return vec2(c.r*65280.0+c.g*255.0,c.b*65280.0+c.a*255.0)*" + unit +
           "*(1.0/65536.0);}";
}

// The switch reader and the poly's definition, at file scope: the builders in
// every shader TU call them, and so does shaders_attn.cpp.
bool exp_poly_enabled() {
    static const bool v = [] {
        const char* e = sw_str("EXP");
        return e && !strcmp(e, "poly");
    }();
    return v;
}

std::string exp_poly_source() { return exp_poly_enabled() ? kExpPolySource : ""; }
