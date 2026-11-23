#pragma once

// Activation texture layouts ("X formats").
//
// Every activation texture is RGBA8.  The layout decides how many logical
// activations a texel holds and how they are quantized:
//
//   I16_1  one q16 value in .rg, .ba unused
//   I16_2  two q16 values: .rg = value 0, .ba = value 1
//   I8_4   four signed int8 values in RGBA, scale 1/16
//
// All producers use XST* and all consumers use X1/X2/X4.  Keeping the layout
// here is the whole point: a new format must not require editing every shader.
#include <cstdint>
#include <string>
#include <vector>

enum class XFormat : uint8_t {
    I16_1 = 0,
    I16_2 = 1,
    I8_4 = 2,
    // 24-bit fixed point, one value per texel: the widest value a float32 can
    // hold exactly (24 mantissa bits).  Same 1/256 step as I16 but 256x the
    // range, which is what a model whose residual exceeds the q16 rail needs.
    I24_1 = 3,
    // An RMSNorm's **partial sums**: the same three bytes and the same 2^23 bias
    // as I24_1, a different step (1/4096 of a summed quantity, not 1/256 of a
    // value).  It is its own format because it is its own *quantity* -- a sum of
    // `x^2` reaches 2048 here, an I24_1 payload is a value in +-32768 -- and
    // sharing one meant one byte layout decoded at two scales with nothing in the
    // name to say which.  Same bytes, so no number moves.
    S24_1 = 6,
    // 32-bit float, one value per texel: the four bytes of the IEEE-754 number
    // in the texel's .rgba, rebuilt with uintBitsToFloat.  The only format with
    // no fixed-point grid at all.  It costs what I24_1 costs in memory (one
    // texel a value) and needs the ESSL 3.00 compiler the RWKV bit-cast kernels
    // already use (g_es3_ok, which the Q4_SHIFT MESA override makes the default
    // on both drivers).
    F32 = 4,
    // bfloat16, two values a texel: the top 16 bits of the IEEE-754 float, so
    // the exponent is float32's and the mantissa is 8 bits (0.4% relative).
    // Unlike I16_2 it has no absolute rail -- the same two bytes a texel buy
    // range instead of precision -- which is the band between "q16 overflows"
    // (the residual above +-128) and "f32 costs twice the memory and 21% of the
    // decode" (docs/performance.md, 2026-11-18).  ESSL 3.00 for the bit-cast,
    // like F32.
    BF16 = 5,
};

struct XLayout {
    XFormat fmt = XFormat::I16_2;
    int count = 0;
    int texels() const;
};

const char* x_format_name(XFormat f);
int x_per_texel(XFormat f);
int x_texels(int count, XFormat f);
bool x_is_int8(XFormat f);
bool x_is_int24(XFormat f);
bool x_is_f32(XFormat f);
bool x_is_bf16(XFormat f);
// The bf16 helper functions without the X1/X2 macros, for a kernel that already
// defines those for another format (the attention codec).
std::string x_bf16_glsl();

// CPU side packing.  Values are real activations; q16 is applied here.
std::vector<uint8_t> x_pack(const float* values, int count, XFormat f);
std::vector<uint8_t> x_pack(const std::vector<float>& values, XFormat f);

// GLSL helpers and the only format-specific macros.  Shader builders prepend
// "precision highp float;\n" and then this block.
std::string x_glsl(XFormat f);

// Residual-stream variants use distinct R1/R2 names so a shader can read an
// int8 activation and a q16 residual with different layouts in one pass.
std::string x_residual_glsl(XFormat f);
// One format's value codecs, without its texture macros.  A pass names its own
// formats; another format's two functions travel as text (the rms sum's store is
// S24_1 and the residual of the same shader is not).  Never the macros: a
// builder already defines its own `#define` set and a second copy stops the
// program with "Redefinition of macro".
std::string x_codec_glsl(XFormat f);
// Only the float32 *store* helper (`x_f32enc`), without the X1/X2/X4 read
// macros: a kernel that reads a fixed q16 pair layout and writes an f32
// stream -- the swiglu feeding the FFN hidden -- needs the encoder without
// the decoder, or the read macros would change meaning under it.
std::string x_f32_store_glsl();

// Two-position ("batch pair", P2) target layout, used by the 2-position prefill
// kernels: a vector of n values occupies n texels, texel k holding value k of
// position A in .rg and value k of position B in .ba, both q16.  It is a
// *target* layout, not an input XFormat - one texel carries one value per
// position rather than two values of one vector, so x_per_texel() does not
// apply and the width is x_pair2_texels(n) == n.
int x_pair2_texels(int count);

// x_p2_a/x_p2_b return raw q16 units (what a matvec accumulates).  x_p2_st packs
// two *real* values as one P2 texel (x_st2 multiplies by 256), x_p2_stq packs two
// values already in q16 units.  Call after x_glsl(): both use its encoders.
std::string x_pair2_glsl(XFormat f = XFormat::I16_2);
