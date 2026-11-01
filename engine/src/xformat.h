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
