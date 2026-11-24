// The CPU half of the X formats: what a value becomes in a texel.
//
// x_pack() is the producer side of every activation texture, and the GLSL
// decoders in x_glsl() are written to undo exactly these bytes.  A change to
// one without the other is a numerics bug that the shader digest cannot see
// (it hashes the text, not the layout), so this pins the bytes for each format
// and the texel arithmetic around them.  No GPU, no model.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "checks.h"
#include "xformat.h"

namespace {

std::vector<uint8_t> pack(std::initializer_list<float> values, XFormat f) {
    const std::vector<float> v(values);
    return x_pack(v, f);
}

void test_shapes_and_names() {
    CHECK_STR(x_format_name(XFormat::I16_1), "x16_1");
    CHECK_STR(x_format_name(XFormat::I16_2), "x16_2");
    CHECK_STR(x_format_name(XFormat::I8_4), "x8_4");
    CHECK_STR(x_format_name(XFormat::I24_1), "x24_1");
    CHECK_STR(x_format_name(XFormat::S24_1), "s24_1");
    CHECK_STR(x_format_name(XFormat::F32), "f32");
    CHECK_STR(x_format_name(XFormat::BF16), "bf16");

    CHECK_LL(x_per_texel(XFormat::I16_1), 1);
    CHECK_LL(x_per_texel(XFormat::I16_2), 2);
    CHECK_LL(x_per_texel(XFormat::I8_4), 4);
    CHECK_LL(x_per_texel(XFormat::I24_1), 1);
    CHECK_LL(x_per_texel(XFormat::S24_1), 1);
    CHECK_LL(x_per_texel(XFormat::F32), 1);
    CHECK_LL(x_per_texel(XFormat::BF16), 2);

    // A partial texel is still a texel: the count rounds up.
    CHECK_LL(x_texels(5, XFormat::I16_2), 3);
    CHECK_LL(x_texels(5, XFormat::I8_4), 2);
    CHECK_LL(x_texels(7, XFormat::BF16), 4);
    CHECK_LL(x_texels(1, XFormat::I24_1), 1);
    CHECK_LL(x_texels(0, XFormat::F32), 0);
    // The pair layout is one value per texel per position, not two a texel.
    CHECK_LL(x_pair2_texels(5), 5);

    CHECK(x_is_int8(XFormat::I8_4));
    CHECK(!x_is_int8(XFormat::I16_2));
    CHECK(x_is_int24(XFormat::I24_1));
    CHECK(!x_is_int24(XFormat::I16_1));
    CHECK(x_is_f32(XFormat::F32));
    CHECK(x_is_bf16(XFormat::BF16));
}

void test_i16_1() {
    // q16 is round(v * 256), clamped at the int16 rail, big-endian in .rg;
    // .b is 0 and .a is 255 (I16_1's unused lanes are not left undefined).
    CHECK_BYTES(pack({0.0f, 1.0f, -1.0f, -128.0f, 127.99609375f, 1000.0f},
                     XFormat::I16_1),
                0x00, 0x00, 0x00, 0xFF,   // 0
                0x01, 0x00, 0x00, 0xFF,   // 1.0   -> 256
                0xFF, 0x00, 0x00, 0xFF,   // -1.0  -> -256
                0x80, 0x00, 0x00, 0xFF,   // -128  -> -32768, the rail
                0x7F, 0xFF, 0x00, 0xFF,   // 127.99609375 -> 32767
                0x7F, 0xFF, 0x00, 0xFF);  // 1000 clamps to the same rail
}

void test_i16_2() {
    // Two values a texel: value 0 in .rg, value 1 in .ba.  An odd count
    // leaves the second value of the last texel at zero, not garbage.
    CHECK_BYTES(pack({1.0f, -1.0f, 2.0f}, XFormat::I16_2),
                0x01, 0x00, 0xFF, 0x00,
                0x02, 0x00, 0x00, 0x00);
}

void test_i8_4() {
    // Scale 1/16, signed, clamped to -127..127 and stored offset by 128 so the
    // byte is never a signed texture sample.
    CHECK_BYTES(pack({0.0f, 1.0f, -1.0f, 8.0f, -8.0f}, XFormat::I8_4),
                128, 144, 112, 255,        // 0, 1.0, -1.0, 8.0 (clamped +127)
                1, 128, 128, 128);         // -8.0 (clamped -127), then zero fill
}

void test_i24_1() {
    // Offset-binary int24: u = clamp(v*256) + 2^23, big-endian in g,b,a; the
    // red byte stays 0 so the decode is still a plain float.
    CHECK_BYTES(pack({0.0f, 1.0f, -1.0f}, XFormat::I24_1),
                0x00, 0x80, 0x00, 0x00,
                0x00, 0x80, 0x01, 0x00,
                0x00, 0x7F, 0xFF, 0x00);
}

void test_f32() {
    // IEEE-754 bits, big-endian in rgba: no grid, no clamp, no scale.
    CHECK_BYTES(pack({1.0f, -2.0f}, XFormat::F32),
                0x3F, 0x80, 0x00, 0x00,
                0xC0, 0x00, 0x00, 0x00);
}

void test_bf16() {
    // The top 16 bits after round-to-nearest-even.  1.00390625 is exactly
    // halfway between two bfloat16 values and ties to the even one (0x3F80);
    // 1.01171875 is halfway again and ties up to 0x3F82.
    CHECK_BYTES(pack({1.0f, -2.0f, 1.00390625f}, XFormat::BF16),
                0x3F, 0x80, 0xC0, 0x00,
                0x3F, 0x80, 0x00, 0x00);
    CHECK_BYTES(pack({1.01171875f}, XFormat::BF16),
                0x3F, 0x82, 0x00, 0x00);
}

void test_unpack() {
    // x_unpack is the host half of every codec: the same bytes x_pack wrote,
    // decoded by the format's own rule.  These values are exact in all seven
    // formats (multiples of 1/16, |v| <= 7, at most 8 significant bits), so the
    // round trip is exact and a drift on either side shows as a mismatch.
    const std::vector<float> v{0.0f, 0.5f, -1.25f, 2.0f, -3.5f, 7.0f, -0.75f, 1.0f};
    for (XFormat f : {XFormat::I16_1, XFormat::I16_2, XFormat::I8_4,
                      XFormat::I24_1, XFormat::S24_1, XFormat::F32, XFormat::BF16}) {
        const std::vector<uint8_t> b = x_pack(v, f);
        for (size_t i = 0; i < v.size(); ++i) {
            const float got = x_unpack(b.data(), (int)i, f);
            if (!(std::fabs(got - v[i]) <= 1e-6f)) {
                std::fprintf(stderr, "x_unpack %s element %zu: %.9g != %.9g\n",
                             x_format_name(f), i, (double)got, (double)v[i]);
                ++checks::g_fail;
            }
        }
    }
}

void test_empty() {
    CHECK(x_pack(std::vector<float>{}, XFormat::I16_2).empty());
    CHECK(x_pack(std::vector<float>{}, XFormat::I8_4).empty());
}

}  // namespace

int main() {
    test_shapes_and_names();
    test_i16_1();
    test_i16_2();
    test_i8_4();
    test_i24_1();
    test_f32();
    test_bf16();
    test_unpack();
    test_empty();
    if (checks::g_fail) {
        std::fprintf(stderr, "test_xformat: %d check(s) failed\n", checks::g_fail);
        return 1;
    }
    std::printf("test_xformat: texel shapes and every format's packed bytes pass\n");
    return 0;
}
