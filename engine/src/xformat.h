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

// CPU side packing.  Values are real activations; q16 is applied here.
std::vector<uint8_t> x_pack(const float* values, int count, XFormat f);
std::vector<uint8_t> x_pack(const std::vector<float>& values, XFormat f);

// GLSL helpers and the only format-specific macros.  Shader builders prepend
// "precision highp float;\n" and then this block.
std::string x_glsl(XFormat f);

// Residual-stream variants use distinct R1/R2 names so a shader can read an
// int8 activation and a q16 residual with different layouts in one pass.
std::string x_residual_glsl(XFormat f);

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
std::string x_pair2_glsl();
