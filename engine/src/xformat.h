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

// ===================================================================
// The XFormat contract.  A format is not "a 4-byte layout": it is a
// *quantity at a stated step* (see the enum), and it must provide every
// row of this table or say why in the row it does not.  The six
// consumers each used to keep its own list, and they had already drifted
// -- `x_pack` had no I8_4 branch (the last `else` *was* I8_4, so a new
// format silently wrote 8-bit), `x_residual_glsl` had no I8_4 at all,
// and `x_is_int24` answered false for a 24-bit format added later.
//
//   capability          provided by              type of the store
//   ---------------------------------------------------------------
//   value decode        X1 / R1                  a real value
//   q16 decode          X1Q / R1Q                an integer on the step
//   value store         XST1 / XST2 / XST4 (X)    a real value
//   q16 store           XST1Q (R)                 an integer, *no* step
//   host pack           x_pack                   a real value
//   GLSL                x_glsl / x_residual_glsl  (the two APIs)
//
// The last column is what nothing in the code says, and it is the one
// that cost three rounds: XST1 multiplies by the format's step and
// XST1Q does not, under names that differ by one letter.  A pass writes
// through the one that matches what it already has (a value or an
// integer) and reads through either.
//
// X and R are **two format slots**, not two APIs for one.  Every builder
// in shaders.cpp calls `head(...)` for the stream it decodes and
// `x_residual_glsl(rf)` for the second one it touches, and the two
// names carry the fact that they must not collide: `head(I16_2)` is the
// q16 activation, `x_residual_glsl(I24_1)` the 24-bit residual, and one
// kernel decodes both.  A single call could not say which is which.
//
// What is *not* honest is the prefix: `X` and `R` name the slot, and the
// store kind is then tied to it (`XST1` = a value, `XST1Q` = a q, and
// the R slot only offers the q one).  Anything that needs a value store
// in the second slot has none, and a reader has to know the coupling
// from memory.  The X1/R1 decoders underneath are the *same* formula,
// which is why the coupling reads as an accident rather than a design.
//
// The next step is the orthogonal spelling: slot in the prefix (X/ Y),
// store kind in the suffix (ST1 = a value, ST1Q = a q), both slots
// offering both.  It is a rename only -- same bytes, same numbers -- and
// it moves every digest, so it is its own commit with its own
// re-baseline.
// ===================================================================

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
// Two granularities, and only two: a whole codec (`x_codec_glsl`, both
// functions) and a store alone (`x_store_glsl`).  `x_bf16_glsl()` used to be a
// third -- a whole codec under a specific name -- because bf16 was the first
// format to need one and it was written before this dispatcher existed; its 13
// call sites all wanted the whole thing.

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
std::string x_store_glsl(XFormat f);

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
