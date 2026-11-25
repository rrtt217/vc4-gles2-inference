#include "xformat.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

int XLayout::texels() const { return x_texels(count, fmt); }

const char* x_format_name(XFormat f) {
    switch (f) {
        case XFormat::I16_1: return "x16_1";
        case XFormat::I16_2: return "x16_2";
        case XFormat::I8_4: return "x8_4";
        case XFormat::I24_1: return "x24_1";
        case XFormat::S24_1: return "s24_1";
        case XFormat::F32: return "f32";
        case XFormat::BF16: return "bf16";
    }
    return "x?";
}

int x_per_texel(XFormat f) {
    switch (f) {
        case XFormat::I16_1: return 1;
        case XFormat::I16_2: return 2;
        case XFormat::I8_4: return 4;
        case XFormat::I24_1: return 1;
        case XFormat::S24_1: return 1;
        case XFormat::F32: return 1;
        case XFormat::BF16: return 2;
    }
    return 0;
}

int x_texels(int count, XFormat f) {
    const int n = x_per_texel(f);
    return n ? (count + n - 1) / n : 0;
}

bool x_is_int8(XFormat f) { return f == XFormat::I8_4; }
// A 24-bit *integer in an RGBA8*: the value form and the sum form share that
// packing (four bytes, a byte an integer), so the question is about the texture's
// storage, not about which quantity it holds.
bool x_is_int24(XFormat f) { return f == XFormat::I24_1 || f == XFormat::S24_1; }
bool x_is_f32(XFormat f) { return f == XFormat::F32; }
bool x_is_bf16(XFormat f) { return f == XFormat::BF16; }

std::vector<uint8_t> x_pack(const float* values, int count, XFormat f) {
    const int texels = x_texels(count, f);
    std::vector<uint8_t> out((size_t)texels * 4, 0);
    for (int t = 0; t < texels; ++t) {
        uint8_t* p = out.data() + (size_t)t * 4;
        if (f == XFormat::I16_1) {
            const int16_t q = quantize_act(values[t]);
            p[0] = (uint8_t)((uint16_t)q >> 8);
            p[1] = (uint8_t)q;
            p[2] = 0;
            p[3] = 255;
        } else if (f == XFormat::I24_1) {
            // offset-binary int24: u = clamp(v*256) + 2^23, big-endian in g,b,a
            // (the top byte stays 0, which keeps the decode a plain float).
            long q = lrintf(values[t] * 256.0f);
            q = std::max(-8388608L, std::min(8388607L, q)) + 8388608L;
            p[0] = 0;
            p[1] = (uint8_t)((q >> 16) & 0xFF);
            p[2] = (uint8_t)((q >> 8) & 0xFF);
            p[3] = (uint8_t)(q & 0xFF);
        } else if (f == XFormat::F32) {
            // IEEE-754 bits, big-endian in rgba.  No grid, no clamp, no scale.
            uint32_t u = 0;
            const float v = values[t];
            memcpy(&u, &v, 4);
            p[0] = (uint8_t)(u >> 24);
            p[1] = (uint8_t)(u >> 16);
            p[2] = (uint8_t)(u >> 8);
            p[3] = (uint8_t)u;
        } else if (f == XFormat::BF16) {
            // Two bfloat16 values a texel, round-to-nearest-even: the top 16
            // bits of the float after adding 0x7FFF + the bit that makes ties
            // even.  Matches x_bf16i in the GLSL below, so a host-packed row and
            // a shader-written row decode the same way.
            for (int k = 0; k < 2; ++k) {
                const int i = 2 * t + k;
                uint32_t u = 0;
                if (i < count) memcpy(&u, &values[i], 4);
                const uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                p[2 * k] = (uint8_t)(r >> 8);
                p[2 * k + 1] = (uint8_t)(r & 0xFF);
            }
        } else if (f == XFormat::S24_1) {
            // Played on a value a texel, 1/4096: the sum form's step.  Nothing
            // host-packs a sum today (the rms partial sums are written by the
            // shader), and it is here because "a format has every row of the
            // contract" is the rule -- the alternative is this function silently
            // returning I8_4 bytes, which is what it did before.
            long q = lrintf(values[t] * 4096.0f);
            q = std::max(0L, std::min(16777215L, q + 8388608L));
            p[0] = (uint8_t)((q >> 24) & 0xFF);
            p[1] = (uint8_t)((q >> 16) & 0xFF);
            p[2] = (uint8_t)((q >> 8) & 0xFF);
            p[3] = (uint8_t)(q & 0xFF);
        } else if (f == XFormat::I16_2) {
            const int16_t q0 = quantize_act(values[2 * t]);
            const int16_t q1 = 2 * t + 1 < count ? quantize_act(values[2 * t + 1]) : 0;
            p[0] = (uint8_t)((uint16_t)q0 >> 8);
            p[1] = (uint8_t)q0;
            p[2] = (uint8_t)((uint16_t)q1 >> 8);
            p[3] = (uint8_t)q1;
        } else if (f == XFormat::I8_4) {
            for (int k = 0; k < 4; ++k) {
                const int i = 4 * t + k;
                int q = i < count ? (int)lrintf(values[i] * 16.0f) : 0;
                q = std::max(-127, std::min(127, q));
                p[k] = (uint8_t)(q + 128);
            }
        } else {
            // An explicit stop, not a default: `else` meant `I8_4` and a format
            // the packer did not know wrote 8-bit bytes that every reader then
            // decoded as its own -- the one failure mode this API exists to
            // make impossible.  The contract in xformat.h is the list.
            fprintf(stderr, "x_pack: %s has no host pack\n", x_format_name(f));
            abort();
        }
    }
    return out;
}

std::vector<uint8_t> x_pack(const std::vector<float>& values, XFormat f) {
    return x_pack(values.data(), (int)values.size(), f);
}

float x_unpack(const uint8_t* row, int index, XFormat f) {
    switch (f) {
        case XFormat::I16_1: {
            const uint8_t* p = row + (size_t)index * 4;
            return (float)(int16_t)((uint16_t)p[0] << 8 | p[1]) / 256.0f;
        }
        case XFormat::I16_2: {
            const uint8_t* p = row + (size_t)(index / 2) * 4 + (index & 1) * 2;
            return (float)(int16_t)((uint16_t)p[0] << 8 | p[1]) / 256.0f;
        }
        case XFormat::I8_4: {
            const uint8_t* p = row + (size_t)(index / 4) * 4 + (index & 3);
            return ((int)*p - 128) / 16.0f;
        }
        case XFormat::I24_1: {
            const uint8_t* p = row + (size_t)index * 4;
            const long u = (long)p[1] << 16 | (long)p[2] << 8 | p[3];
            return (float)(u - 8388608L) / 256.0f;
        }
        case XFormat::S24_1: {
            const uint8_t* p = row + (size_t)index * 4;
            const long u = (long)p[0] << 24 | (long)p[1] << 16 |
                           (long)p[2] << 8 | p[3];
            return (float)(u - 8388608L) / 4096.0f;
        }
        case XFormat::F32: {
            const uint8_t* p = row + (size_t)index * 4;
            const uint32_t u = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 |
                               (uint32_t)p[2] << 8 | p[3];
            float v = 0.0f;
            memcpy(&v, &u, 4);
            return v;
        }
        case XFormat::BF16: {
            const uint8_t* p = row + (size_t)(index / 2) * 4 + (index & 1) * 2;
            const uint32_t u = ((uint32_t)p[0] << 8 | p[1]) << 16;
            float v = 0.0f;
            memcpy(&v, &u, 4);
            return v;
        }
    }
    return 0.0f;
}

// The q16 decode comes in two spellings.  The default is the fast float
// reconstruction with the sign classified at 32767.5: two multiplies and an
// add, no conversions, which is the cost the pre-6234328 kernels had.  It
// reconstructs every value with the 8-bit texture sample's small error on VC4,
// so the board's PPL pins move with it; Q16_DECODE=exact restores 6234328's
// integer-exact decode (float(int(c*255+0.5)) per byte, bit-identical to floor
// for these positive arguments) when the exact board values are wanted.
// docs/test-hooks.md has the switch, docs/vc4-glsl-costs.md the numbers.
static bool q16_decode_exact() {
        const char* v = getenv("Q16_DECODE");
        return v && (!strcmp(v, "exact") || !strcmp(v, "1"));
}

std::string x_scaled_glsl() {
    return "\n#ifndef X_SCALED_GLSL\n#define X_SCALED_GLSL\n"
        "float x_q16raw_at(vec2 c){float u=c.x*65280.0+c.y*255.0;"
        "if(u>=32767.5)u-=65536.0;return u;}\n"
        "float x_q16dec_at(vec2 c,float s){return x_q16raw_at(c)/s;}\n"
        "vec2 x_q16enc_at(float v,float s){"
        "float q=clamp(floor(v*s+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
        "float hi=floor(q/256.0),lo=q-hi*256.0;"
        "return vec2(hi/255.0,lo/255.0);}\n"
        "float x_i24raw_at(vec4 c){return float(int(c.g*255.0+0.5))*65536.0+"
        "float(int(c.b*255.0+0.5))*256.0+float(int(c.a*255.0+0.5))-8388608.0;}\n"
        "float x_i24dec_at(vec4 c,float s){return x_i24raw_at(c)/s;}\n"
        "vec4 x_i24enc_at(float v,float s){float q=floor(v*s+0.5)+8388608.0;"
        "float g=floor(q/65536.0),b=floor(q/256.0)-g*256.0,a=q-g*65536.0-b*256.0;"
        "return vec4(0.0,g/255.0,b/255.0,a/255.0);}\n"
        // The signed byte itself, then the q16-unit form (byte*16): the KV
        // grid is 1/KVSCALE of the *byte*, not of the I8_4 value.  The byte
        // is offset-binary like the rest of I8_4 (q+128), not two's
        // complement -- the attention's hand copy wrote the two's-complement
        // form, so its bytes and this decoder's disagreed by 128.
        "float x_i8byte_at(float c){return floor(c*255.0+0.5)-128.0;}\n"
        "float x_i8raw1_at(float c){return x_i8byte_at(c)*16.0;}\n"
        "float x_i8raw_at(vec4 c,int i){return x_i8byte_at(c[i])*16.0;}\n"
        "vec4 x_i8enc4_at(float a,float b,float c3,float d,float s){"
        "vec4 q=clamp(floor(vec4(a,b,c3,d)*s+0.5),-127.0,127.0);"
        "return (q+128.0)/255.0;}\n"
        "#endif\n";
}

static std::string q16_raw_glsl() {
        if (q16_decode_exact())
                return x_scaled_glsl() +
                    "float x_i16raw(vec2 c){float hi=float(int(c.x*255.0+0.5));"
                    "float lo=float(int(c.y*255.0+0.5));float u=hi*256.0+lo;"
                    "if(u>=32768.0)u-=65536.0;return u;}\n";
        return x_scaled_glsl() +
            "float x_i16raw(vec2 c){return x_q16raw_at(c);}\n";
}

static const char* q16_pair_glsl() {
        if (q16_decode_exact())
                return
                    "vec2 x_i16pairq(vec4 c){float h0=float(int(c.x*255.0+0.5)),l0=float(int(c.y*255.0+0.5));"
                    "float h1=float(int(c.z*255.0+0.5)),l1=float(int(c.w*255.0+0.5));"
                    "float u0=h0*256.0+l0;if(u0>=32768.0)u0-=65536.0;"
                    "float u1=h1*256.0+l1;if(u1>=32768.0)u1-=65536.0;return vec2(u0,u1);}\n"
                    "vec4 x_i16quadq(vec4 a,vec4 b){float h0=float(int(a.x*255.0+0.5)),l0=float(int(a.y*255.0+0.5));"
                    "float h1=float(int(a.z*255.0+0.5)),l1=float(int(a.w*255.0+0.5));"
                    "float h2=float(int(b.x*255.0+0.5)),l2=float(int(b.y*255.0+0.5));"
                    "float h3=float(int(b.z*255.0+0.5)),l3=float(int(b.w*255.0+0.5));"
                    "float u0=h0*256.0+l0;if(u0>=32768.0)u0-=65536.0;"
                    "float u1=h1*256.0+l1;if(u1>=32768.0)u1-=65536.0;"
                    "float u2=h2*256.0+l2;if(u2>=32768.0)u2-=65536.0;"
                    "float u3=h3*256.0+l3;if(u3>=32768.0)u3-=65536.0;"
                    "return vec4(u0,u1,u2,u3);}\n";
        return
            "vec2 x_i16pairq(vec4 c){float u0=c.x*65280.0+c.y*255.0;"
            "if(u0>=32767.5)u0-=65536.0;float u1=c.z*65280.0+c.w*255.0;"
            "if(u1>=32767.5)u1-=65536.0;return vec2(u0,u1);}\n"
            "vec4 x_i16quadq(vec4 a,vec4 b){float u0=a.x*65280.0+a.y*255.0;"
            "if(u0>=32767.5)u0-=65536.0;float u1=a.z*65280.0+a.w*255.0;"
            "if(u1>=32767.5)u1-=65536.0;float u2=b.x*65280.0+b.y*255.0;"
            "if(u2>=32767.5)u2-=65536.0;float u3=b.z*65280.0+b.w*255.0;"
            "if(u3>=32767.5)u3-=65536.0;return vec4(u0,u1,u2,u3);}\n";
}


std::string x_codec_glsl(XFormat f) {
    if (f == XFormat::BF16) {
    // Only the bf16 functions, no X1/X2 macros: a kernel that already carries
    // its own X-macros (the attention codec's activation format) still wants
    // these, and defining the macros twice is a preprocessor error.
    //
    // The include guard makes the emission idempotent, which matters because a
    // kernel can want these from two directions at once -- an activation stream
    // and a residual stream that are both bf16 each ask for them (found as
    // "function x_bf16 redefined" with RES=bf16 ACT=bf16, 2026-11-18).
    return std::string("\n#ifndef X_BF16_GLSL\n#define X_BF16_GLSL\n") +
        "uint x_bf16i(float v){uint u=floatBitsToUint(v);"
        "u=(u+0x7FFFu+((u>>16)&1u))>>16;return u&0xFFFFu;}\n"
        "float x_bf16(vec2 c){uint hi=uint(int(c.x*255.0+0.5)),"
        "lo=uint(int(c.y*255.0+0.5));return uintBitsToFloat((hi<<24)|(lo<<16));}\n"
        "vec2 x_bf16pair(vec4 c){return vec2(x_bf16(c.rg),x_bf16(c.ba));}\n"
        "vec4 x_bf16enc(float v){uint u=x_bf16i(v);"
        "return vec4(float(u>>8)/255.0,float(u&255u)/255.0,0.0,1.0);}\n"
        "vec4 x_bf16enc2(float a,float b){uint ua=x_bf16i(a),ub=x_bf16i(b);"
        "return vec4(float(ua>>8)/255.0,float(ua&255u)/255.0,"
        "float(ub>>8)/255.0,float(ub&255u)/255.0);}\n"
        "#endif\n";    }
    if (f == XFormat::F32) {
        // The two functions this format *is*, guarded so that both slots can
        // carry one.  `x_residual_glsl(F32)` used to hand-copy them as
        // `x_f32r`/`x_f32renc` because a second `x_f32` would not compile
        // (its own comment said so).  A copy is a thing to keep in sync.
        return
            "\n#ifndef X_F32_GLSL\n#define X_F32_GLSL\n"
            "float x_f32(vec4 c){uint u=(uint(int(c.x*255.0+0.5))<<24)|"
            "(uint(int(c.y*255.0+0.5))<<16)|(uint(int(c.z*255.0+0.5))<<8)|"
            "uint(int(c.w*255.0+0.5));return uintBitsToFloat(u);}\n"
            "vec4 x_f32enc(float v){uint u=floatBitsToUint(v);"
            "return vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
            "float(u&255u))/255.0;}\n"
            "#endif\n";
    }
    if (f != XFormat::S24_1) return std::string();
    // The sum of squares the rms stats passes store.  Faithful to the encoder it
    // replaces (`encs` in shaders_block.cpp): *four* bytes, because a lane sum of
    // `x^2` reaches 2048 and 2048*4096 already fills the 24 the value stream
    // uses -- `.r` is the 25th.
    // Guarded, because this is the *only* place the format's two functions come
    // from: `x_glsl(S24_1)` and `x_residual_glsl(S24_1)` both call it, and a
    // kernel that carries one texture of each kind in the same format would
    // otherwise define them twice.  That is bf16's answer (`X_BF16_GLSL`) rather
    // than f32's (a hand-copied second name, `x_f32r`), because a copy is a
    // thing to keep in sync and a guard is a thing that cannot drift.
    return
        "\n#ifndef X_S24_GLSL\n#define X_S24_GLSL\n"
        "float x_s24(vec4 c){return (float(int(c.r*255.0+0.5))*16777216.0+"
        "float(int(c.g*255.0+0.5))*65536.0+float(int(c.b*255.0+0.5))*256.0+"
        "float(int(c.a*255.0+0.5))-8388608.0)/4096.0;}\n"
        "vec4 x_s24enc(float v){float q=floor(v*4096.0+0.5)+8388608.0;"
        "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,"
        "b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n"
        "#endif\n";
}

std::string x_glsl(XFormat f) {
    // Values crossing this boundary are real activations.  XST* converts to
    // q16/q8 on write.  X1/X2/X4 return real values; the Q-suffixed variants
    // return q16 units for hot matvec loops.  No shader needs to know the byte
    // layout or the 1/16 scale.
    // The q16 decode rebuilds 256*hi+lo from two normalised bytes.  The only
    // value where a plain float reconstruction disagrees with the integer one
    // is the negative rail: (128,0) is 32768.0 on Xe and 32767.99x on VC4, so
    // testing ">= 32768" flips its sign there (6234328).  Testing ">= 32767.5"
    // classifies it as negative on both (the encoder clamps to 32767, so no
    // positive value can land in [32767.5, 32768)), and the value it decodes
    // to is short of -32768 by at most the reconstruction error: < 0.03 q16
    // units = 1.2e-4 activation units, which the next store clamps back to the
    // rail anyway.  That keeps the decode at the pre-6234328 cost - two
    // multiplies and an add, no conversions, no floor (docs/vc4-glsl-costs.md)
    // - which is what the board's i16 model pays for: a bisect put 41% of its
    // 2026-11-09 regression on that commit's floor-based exact decode.
    std::string s = std::string(q16_raw_glsl()) +
        "float x_i16(vec2 c){return x_i16raw(c)/256.0;}\n"
        "vec2 x_i16pair(vec4 c){return vec2(x_i16(c.rg),x_i16(c.ba));}\n" +
        q16_pair_glsl() +
        "vec2 x_q16enc(float q){return x_q16enc_at(q/256.0,256.0);}\n"
        "vec2 x_i16enc(float v){return x_q16enc_at(v,256.0);}\n"
        // x_i8q is in q16 units like every other decoder's raw form: the grid
        // is 1/16, so value*256 = (byte-128)*16.  The real-value forms divide
        // by 256, not by 16 -- the 16 is the grid factor and is already in
        // x_i8q; dividing by it again returned the value *16, which no hot
        // kernel reads (they all use X1Q) and probes/codec_probe caught on the
        // first real-value read of an I8_4 stream.
        "float x_i8q(float c){return x_i8raw1_at(c);}\n"
        // One component conversion and a dynamic index, not a four-way if:
        // every branch of the if was evaluated (18.3 ns a value against 8.2
        // for this form, same value; probes/codec_probe, VC4 2026-11-25).
        "float x_i8cq(vec4 c,int i){return x_i8raw_at(c,i);}\n"
        "float x_i8(float c){return x_i8q(c)/256.0;}\n"
        "float x_i8c(vec4 c,int i){return x_i8cq(c,i)/256.0;}\n"
        "vec4 x_i8enc4(float a,float b,float c,float d){"
        "return x_i8enc4_at(a,b,c,d,16.0);}\n"
        "vec4 x_st1(float v){vec2 p=x_i16enc(v);return vec4(p.x,p.y,0.0,1.0);}\n"
        "vec4 x_st2(float a,float b){vec2 p=x_i16enc(a),q=x_i16enc(b);"
        "return vec4(p.x,p.y,q.x,q.y);}\n"
        "vec4 x_st1q(float q){vec2 p=x_q16enc(q);return vec4(p.x,p.y,0.0,1.0);}\n"
        "vec4 x_st2q(float a,float b){vec2 p=x_q16enc(a),q=x_q16enc(b);"
        "return vec4(p.x,p.y,q.x,q.y);}\n";

    // A 24-bit residual needs its own decode/encode, and `q24_helpers()` may
    // already be in the same shader, so these carry distinct names.
    s +=
        "float x_q24raw(vec4 c){return x_i24raw_at(c);}\n"
        "float x_q24(vec4 c){return x_i24dec_at(c,256.0);}\n"
        "vec4 x_q24enc(float v){return x_i24enc_at(v,256.0);}\n"
        "vec4 x_q24encq(float q){return x_q24enc(q/256.0);}\n";

    switch (f) {
        case XFormat::S24_1:
            s += x_codec_glsl(XFormat::S24_1);
            s +=
                "#define X1(T,I,TW) x_s24(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define X2(T,BASE,TW) vec2(X1(T,(BASE),TW),X1(T,(BASE)+1,TW))\n"
                "#define X4(T,BASE,TW) vec4(X1(T,(BASE),TW),X1(T,(BASE)+1,TW),"
                "X1(T,(BASE)+2,TW),X1(T,(BASE)+3,TW))\n"
                "#define XST1(V) x_s24enc(V)\n"
                "#define XST2(A,B) vec4(0.0)\n"
                "#define XST4(A,B,C,D) vec4(0.0)\n"
                "#define X1Q(T,I,TW) X1(T,I,TW)\n"
                "#define X2Q(T,B,TW) X2(T,B,TW)\n"
                "#define X4Q(T,B,TW) X4(T,B,TW)\n";
            break;
        case XFormat::I24_1:
            s +=
                "#define X1(T,I,TW) x_q24(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define X2(T,BASE,TW) vec2(X1(T,(BASE),TW),X1(T,(BASE)+1,TW))\n"
                "#define X4(T,BASE,TW) vec4(X1(T,(BASE),TW),X1(T,(BASE)+1,TW),"
                "X1(T,(BASE)+2,TW),X1(T,(BASE)+3,TW))\n"
                "#define XST1(V) x_q24enc(V)\n"
                
                "#define XST2(A,B) vec4(0.0)\n"
                "#define XST4(A,B,C,D) vec4(0.0)\n"
                "#define X1Q(T,I,TW) x_q24raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define X2Q(T,B,TW) vec2(X1Q(T,(B),TW),X1Q(T,(B)+1,TW))\n"
                "#define X4Q(T,B,TW) vec4(X1Q(T,(B),TW),X1Q(T,(B)+1,TW),X1Q(T,(B)+2,TW),X1Q(T,(B)+3,TW))\n";
            break;
        case XFormat::BF16:
            // bfloat16, two values a texel: the top 16 bits of the f32, so the
            // exponent is float32's and the mantissa is 8 bits.  No grid, no
            // scale, and the Q variants return value*256 like every other
            // format's do (the matvec kernels carry the 1/256 in their scale).
            s +=
                x_codec_glsl(XFormat::BF16) +
                "#define X1(T,I,TW) ((((I)/2)*2)==(I) ? "
                "x_bf16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) : "
                "x_bf16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define X2(T,BASE,TW) x_bf16pair(texture2D(T,vec2((float((BASE)/2)+0.5)/(TW),0.5)))\n"
                "#define X4(T,BASE,TW) vec4(X2(T,(BASE),TW),X2(T,(BASE)+2,TW))\n"
                "#define XST1(V) x_bf16enc(V)\n"
                "#define XST2(A,B) x_bf16enc2(A,B)\n"
                "#define XST4(A,B,C,D) vec4(0.0)\n"
                "#define X1Q(T,I,TW) (X1(T,(I),(TW))*256.0)\n"
                "#define X2Q(T,B,TW) (X2(T,(B),(TW))*256.0)\n"
                "#define X4Q(T,B,TW) vec4(X2(T,(B),(TW))*256.0,X2(T,(B)+2,(TW))*256.0)\n";
            break;
        case XFormat::I16_1:
            s +=
                "#define X1(T,I,TW) x_i16(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define X2(T,BASE,TW) vec2(X1(T,(BASE),TW),X1(T,(BASE)+1,TW))\n"
                "#define X4(T,BASE,TW) vec4(X1(T,(BASE),TW),X1(T,(BASE)+1,TW),"
                "X1(T,(BASE)+2,TW),X1(T,(BASE)+3,TW))\n"
                "#define XST1(V) x_st1(V)\n"
                
                "#define X1Q(T,I,TW) x_i16raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define X2Q(T,B,TW) vec2(X1Q(T,(B),TW),X1Q(T,(B)+1,TW))\n"
                "#define X4Q(T,B,TW) vec4(X1Q(T,(B),TW),X1Q(T,(B)+1,TW),X1Q(T,(B)+2,TW),X1Q(T,(B)+3,TW))\n";
            break;
        case XFormat::I16_2:
            s +=
                // Reverted from a one-fetch helper: a function taking the
                // *sampler* as a parameter reads undefined data on VC4 (the
                // engine's q8 decode then produced three different texts in
                // ten identical runs).  probes/codec_probe's isolated check
                // could not see it -- it reads one row per arm.
                "#define X1(T,I,TW) ((((I)/2)*2)==(I) ? x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define X2(T,BASE,TW) x_i16pair(texture2D(T,vec2((float((BASE)/2)+0.5)/(TW),0.5)))\n"
                "#define X4(T,BASE,TW) vec4(X2(T,(BASE),TW),X2(T,(BASE)+2,TW))\n"
                "#define XST2(A,B) x_st2(A,B)\n"
                
                "#define X1Q(T,I,TW) ((((I)/2)*2)==(I) ? x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define X2Q(T,B,TW) x_i16pairq(texture2D(T,vec2((float((B)/2)+0.5)/(TW),0.5)))\n"
                "#define X4Q(T,B,TW) x_i16quadq(texture2D(T,vec2((float((B)/2)+0.5)/(TW),0.5)),"
                "texture2D(T,vec2((float((B)/2+1)+0.5)/(TW),0.5)))\n";
            break;
        case XFormat::F32:
            // One value a texel: the four bytes of the float, rebuilt with a
            // bit-cast.  ESSL 3.00 only (link_program finds the bit-cast and
            // es3ifies the kernel); no grid, no clamp, no scale.
            s += x_codec_glsl(XFormat::F32);
            s +=
                "#define X1(T,I,TW) x_f32(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define X2(T,BASE,TW) vec2(X1(T,(BASE),TW),X1(T,(BASE)+1,TW))\n"
                "#define X4(T,BASE,TW) vec4(X1(T,(BASE),TW),X1(T,(BASE)+1,TW),X1(T,(BASE)+2,TW),X1(T,(BASE)+3,TW))\n"
                "#define XST1(V) x_f32enc(V)\n"
                "#define XST2(A,B) vec4(0.0)\n"
                "#define XST4(A,B,C,D) vec4(0.0)\n"
                "#define X1Q(T,I,TW) (x_f32(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))*256.0)\n"
                "#define X2Q(T,B,TW) (vec2(X1(T,(B),TW),X1(T,(B)+1,TW))*256.0)\n"
                "#define X4Q(T,B,TW) (vec4(X1(T,(B),TW),X1(T,(B)+1,TW),X1(T,(B)+2,TW),X1(T,(B)+3,TW))*256.0)\n";
            break;
        case XFormat::I8_4:
            s +=
                "#define X1(T,I,TW) x_i8c(texture2D(T,vec2((float((I)/4)+0.5)/(TW),0.5)),int((I)-(I)/4*4))\n"
                "#define X2(T,BASE,TW) vec2(X1(T,(BASE),TW),X1(T,(BASE)+1,TW))\n"
                "#define X4(T,BASE,TW) ((texture2D(T,vec2((float((BASE)/4)+0.5)/(TW),0.5))*255.0-128.0)/16.0)\n"
                "#define XST4(A,B,C,D) x_i8enc4(A,B,C,D)\n"
                
                "#define X1Q(T,I,TW) x_i8cq(texture2D(T,vec2((float((I)/4)+0.5)/(TW),0.5)),int((I)-(I)/4*4))\n"
                "#define X2Q(T,B,TW) vec2(X1Q(T,(B),TW),X1Q(T,(B)+1,TW))\n"
                "#define X4Q(T,B,TW) ((texture2D(T,vec2((float((B)/4)+0.5)/(TW),0.5))*255.0-128.0)*16.0)\n";
            break;
    }
    return s;
}

std::string x_residual_glsl(XFormat f) {
    switch (f) {
        case XFormat::S24_1:
            return x_codec_glsl(XFormat::S24_1) +
                "#define Y1(T,I,TW) x_s24(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define Y2(T,BASE,TW) vec2(Y1(T,(BASE),TW),Y1(T,(BASE)+1,TW))\n"
                "#define Y1Q(T,I,TW) Y1(T,I,TW)\n"
                "#define YST1(V) x_s24enc(V)\n"
                "#define YST1Q(Q) x_s24enc(Q)\n";
        case XFormat::I24_1:
            return
                "#define Y1(T,I,TW) x_q24(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define Y2(T,BASE,TW) vec2(Y1(T,(BASE),TW),Y1(T,(BASE)+1,TW))\n"
                "#define Y1Q(T,I,TW) x_q24raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define YST1(V) x_q24enc(V)\n"
                "#define YST1Q(Q) x_q24encq(Q)\n";
        case XFormat::I16_1:
            return
                "#define Y1(T,I,TW) x_i16(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define Y2(T,BASE,TW) vec2(Y1(T,(BASE),TW),Y1(T,(BASE)+1,TW))\n"
                "#define Y1Q(T,I,TW) x_i16raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define YST1(V) x_st1(V)\n"
                "#define YST1Q(Q) x_st1q(Q)\n";
        case XFormat::I16_2:
            return
                "#define Y1(T,I,TW) ((((I)/2)*2)==(I) ? x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define Y2(T,BASE,TW) x_i16pair(texture2D(T,vec2((float((BASE)/2)+0.5)/(TW),0.5)))\n"
                "#define Y1Q(T,I,TW) ((((I)/2)*2)==(I) ? x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define YST1(V) x_st1(V)\n"
                "#define YST1Q(Q) x_st1q(Q)\n";
        case XFormat::F32:
            return x_codec_glsl(XFormat::F32) +
                "#define Y1(T,I,TW) x_f32(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define Y2(T,BASE,TW) vec2(Y1(T,(BASE),TW),Y1(T,(BASE)+1,TW))\n"
                "#define Y1Q(T,I,TW) (Y1(T,(I),TW)*256.0)\n"
                "#define YST1(V) x_f32enc(V)\n"
                "#define YST1Q(Q) x_f32enc((Q)/256.0)\n";
        case XFormat::BF16:
            // Same two-halves-a-texel layout; the residual add stores one value
            // a texel (XST1Q) so the pair is only used for reads.
            //
            // The helper block travels with the macros, like x_pair2_glsl()'s
            // bf16 branch: a kernel that emits only x_residual_glsl(BF16) --
            // rms_norm does, and the shader builder used to assume the helpers
            // came from somewhere else -- had R1 call a function nobody had
            // defined ("no function with name 'x_bf16'", found by running
            // RES=bf16 on a q8 container, 2026-11-18; the q4 grid happened not
            // to reach that builder).  The include guard makes it idempotent.
            return x_codec_glsl(XFormat::BF16) +
                "#define Y1(T,I,TW) ((((I)/2)*2)==(I) ? "
                "x_bf16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) : "
                "x_bf16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define Y2(T,BASE,TW) vec2(Y1(T,(BASE),TW),Y1(T,(BASE)+1,TW))\n"
                "#define Y1Q(T,I,TW) (Y1(T,(I),(TW))*256.0)\n"
                "#define YST1(V) x_bf16enc(V)\n"
                "#define YST1Q(Q) x_bf16enc((Q)/256.0)\n";
        case XFormat::I8_4:
            // No residual in this format, by design and not by omission: four
            // values a texel at 1/16 is a *narrow* stream, and a residual is
            // the one that has to survive every layer unscaled (the q16 residual
            // rails at +-128, which is why the wide formats exist at all).  An
            // empty set here is the answer "this format is not a residual", and
            // a caller that asks for one gets a program that does not compile --
            // the loudest available failure.
            return {};
    }

    return {};
}

bool x_can_residual(XFormat f) { return !x_residual_glsl(f).empty(); }

int x_pair2_texels(int count) { return count; }

std::string x_store_glsl(XFormat f) {
    switch (f) {
        case XFormat::I16_1:
        case XFormat::I16_2:
            // Both are the q16 family's single-value texel (.rg, 2 bytes).
            return "vec4 xs1(float q){return x_st1q(q);}\n";
        case XFormat::I24_1:
            return "vec4 xs1(float q){return x_q24encq(q);}\n";
        case XFormat::BF16:
            return x_codec_glsl(XFormat::BF16) +
                "vec4 xs1(float q){return x_bf16enc(q/256.0);}\n";
        case XFormat::F32:
            return x_codec_glsl(XFormat::F32) +
                "vec4 xs1(float q){return x_f32enc(q/256.0);}\n";
        case XFormat::S24_1:
            return x_codec_glsl(XFormat::S24_1) +
                "vec4 xs1(float q){return x_s24enc(q);}\n";
        case XFormat::I8_4:
            // Not a residual stream; a caller that asks gets no store function
            // and the program fails to compile (the same answer x_residual_glsl
            // gives, for the same reason).
            return {};
    }
    return {};
}

std::string x_pair2_glsl(XFormat f) {
    // Two values a texel, as the paired prefill kernels read and write them.
    // The default is the q16 pair; bf16 uses the same layout with its own
    // rounding, so a residual stream can be bf16 without a second pair kernel.
    if (f == XFormat::BF16) {
        return x_codec_glsl(XFormat::BF16) +
            "float x_p2_a(vec4 c){return x_bf16(c.rg);}\n"
            "float x_p2_b(vec4 c){return x_bf16(c.ba);}\n"
            "vec4 x_p2_st(float a,float b){return x_bf16enc2(a,b);}\n"
            "vec4 x_p2_stq(float qa,float qb){return x_bf16enc2(qa/256.0,qb/256.0);}\n";
    }
    return x_scaled_glsl() +
        // The two-position reader is the same raw decoder as the pair layout:
        // it used the pre-6234328 rail (>= 32768), which on VC4 reads the
        // negative rail as +127.99.
        "float x_p2_a(vec4 c){return x_q16raw_at(c.rg);}\n"
        "float x_p2_b(vec4 c){return x_q16raw_at(c.ba);}\n"
        "vec4 x_p2_st(float a,float b){return x_st2(a,b);}\n"
        "vec4 x_p2_stq(float qa,float qb){vec2 p=x_q16enc(qa),q=x_q16enc(qb);"
        "return vec4(p.x,p.y,q.x,q.y);}\n";
}

// The f32 store helper on its own (see xformat.h).  `x_glsl(F32)` emits this
// plus the read macros; a kernel with a fixed q16 *input* layout and an f32
// *output* (the swiglu) emits this plus `head(XFormat::I16_2)`.
