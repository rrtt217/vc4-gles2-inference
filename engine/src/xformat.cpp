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
        case XFormat::F32: return "f32";
    }
    return "x?";
}

int x_per_texel(XFormat f) {
    switch (f) {
        case XFormat::I16_1: return 1;
        case XFormat::I16_2: return 2;
        case XFormat::I8_4: return 4;
        case XFormat::I24_1: return 1;
        case XFormat::F32: return 1;
    }
    return 0;
}

int x_texels(int count, XFormat f) {
    const int n = x_per_texel(f);
    return n ? (count + n - 1) / n : 0;
}

bool x_is_int8(XFormat f) { return f == XFormat::I8_4; }
bool x_is_int24(XFormat f) { return f == XFormat::I24_1; }
bool x_is_f32(XFormat f) { return f == XFormat::F32; }

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
        } else if (f == XFormat::I16_2) {
            const int16_t q0 = quantize_act(values[2 * t]);
            const int16_t q1 = 2 * t + 1 < count ? quantize_act(values[2 * t + 1]) : 0;
            p[0] = (uint8_t)((uint16_t)q0 >> 8);
            p[1] = (uint8_t)q0;
            p[2] = (uint8_t)((uint16_t)q1 >> 8);
            p[3] = (uint8_t)q1;
        } else {
            for (int k = 0; k < 4; ++k) {
                const int i = 4 * t + k;
                int q = i < count ? (int)lrintf(values[i] * 16.0f) : 0;
                q = std::max(-127, std::min(127, q));
                p[k] = (uint8_t)(q + 128);
            }
        }
    }
    return out;
}

std::vector<uint8_t> x_pack(const std::vector<float>& values, XFormat f) {
    return x_pack(values.data(), (int)values.size(), f);
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

static const char* q16_raw_glsl() {
        if (q16_decode_exact())
                return
                    "float x_i16raw(vec2 c){float hi=float(int(c.x*255.0+0.5));"
                    "float lo=float(int(c.y*255.0+0.5));float u=hi*256.0+lo;"
                    "if(u>=32768.0)u-=65536.0;return u;}\n";
        return
            "float x_i16raw(vec2 c){float u=c.x*65280.0+c.y*255.0;"
            "if(u>=32767.5)u-=65536.0;return u;}\n";
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
        "vec2 x_q16enc(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
        "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
        "return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n"
        "vec2 x_i16enc(float v){return x_q16enc(v*256.0);}\n"
        "float x_i8q(float c){return (floor(c*255.0+0.5)-128.0)*16.0;}\n"
        "float x_i8cq(vec4 c,int i){if(i==0)return x_i8q(c.r);if(i==1)return x_i8q(c.g);"
        "if(i==2)return x_i8q(c.b);return x_i8q(c.a);}\n"
        "float x_i8(float c){return x_i8q(c)/16.0;}\n"
        "float x_i8c(vec4 c,int i){return x_i8cq(c,i)/16.0;}\n"
        "vec4 x_i8enc4(float a,float b,float c,float d){"
        "vec4 q=clamp(floor(vec4(a,b,c,d)*16.0+0.5),-127.0,127.0);"
        "return (q+128.0)/255.0;}\n"
        "vec4 x_st1(float v){vec2 p=x_i16enc(v);return vec4(p.x,p.y,0.0,1.0);}\n"
        "vec4 x_st2(float a,float b){vec2 p=x_i16enc(a),q=x_i16enc(b);"
        "return vec4(p.x,p.y,q.x,q.y);}\n"
        "vec4 x_st1q(float q){vec2 p=x_q16enc(q);return vec4(p.x,p.y,0.0,1.0);}\n"
        "vec4 x_st2q(float a,float b){vec2 p=x_q16enc(a),q=x_q16enc(b);"
        "return vec4(p.x,p.y,q.x,q.y);}\n";

    // A 24-bit residual needs its own decode/encode, and `q24_helpers()` may
    // already be in the same shader, so these carry distinct names.
    s +=
        "float x_q24raw(vec4 c){return float(int(c.g*255.0+0.5))*65536.0+"
        "float(int(c.b*255.0+0.5))*256.0+float(int(c.a*255.0+0.5))-8388608.0;}\n"
        "float x_q24(vec4 c){return x_q24raw(c)/256.0;}\n"
        "vec4 x_q24enc(float v){float q=floor(clamp(v*256.0,-8388608.0,8388607.0)+0.5)+8388608.0;"
        "float g=floor(q/65536.0),b=floor(q/256.0)-g*256.0,a=q-g*65536.0-b*256.0;"
        "return vec4(0.0,g/255.0,b/255.0,a/255.0);}\n"
        "vec4 x_q24encq(float q){return x_q24enc(q/256.0);}\n";

    switch (f) {
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
            s +=
                "float x_f32(vec4 c){uint u=(uint(int(c.x*255.0+0.5))<<24)|"
                "(uint(int(c.y*255.0+0.5))<<16)|(uint(int(c.z*255.0+0.5))<<8)|"
                "uint(int(c.w*255.0+0.5));return uintBitsToFloat(u);}\n" +
                x_f32_store_glsl() +
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
        case XFormat::I24_1:
            return
                "#define R1(T,I,TW) x_q24(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define R2(T,BASE,TW) vec2(R1(T,(BASE),TW),R1(T,(BASE)+1,TW))\n"
                "#define R1Q(T,I,TW) x_q24raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define XST1Q(Q) x_q24encq(Q)\n";
        case XFormat::I16_1:
            return
                "#define R1(T,I,TW) x_i16(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define R2(T,BASE,TW) vec2(R1(T,(BASE),TW),R1(T,(BASE)+1,TW))\n"
                "#define R1Q(T,I,TW) x_i16raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define XST1Q(Q) x_st1q(Q)\n";
        case XFormat::I16_2:
            return
                "#define R1(T,I,TW) ((((I)/2)*2)==(I) ? x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define R2(T,BASE,TW) x_i16pair(texture2D(T,vec2((float((BASE)/2)+0.5)/(TW),0.5)))\n"
                "#define R1Q(T,I,TW) ((((I)/2)*2)==(I) ? x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define XST1Q(Q) x_st1q(Q)\n";
        case XFormat::F32:
            // Distinct helper names: a kernel can carry both x_glsl(F32) and
            // x_residual_glsl(F32), and a second definition of x_f32 would not
            // compile.
            return
                "float x_f32r(vec4 c){uint u=(uint(int(c.x*255.0+0.5))<<24)|"
                "(uint(int(c.y*255.0+0.5))<<16)|(uint(int(c.z*255.0+0.5))<<8)|"
                "uint(int(c.w*255.0+0.5));return uintBitsToFloat(u);}\n"
                "vec4 x_f32renc(float v){uint u=floatBitsToUint(v);"
                "return vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
                "float(u&255u))/255.0;}\n"
                "#define R1(T,I,TW) x_f32r(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n"
                "#define R2(T,BASE,TW) vec2(R1(T,(BASE),TW),R1(T,(BASE)+1,TW))\n"
                "#define R1Q(T,I,TW) (R1(T,(I),TW)*256.0)\n"
                "#define XST1Q(Q) x_f32renc((Q)/256.0)\n";
        case XFormat::I8_4:
            return {};
    }
    return {};
}

int x_pair2_texels(int count) { return count; }

std::string x_pair2_glsl() {
    return
        "float x_p2_a(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "if(u>=32768.0)u-=65536.0;return u;}\n"
        "float x_p2_b(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "if(u>=32768.0)u-=65536.0;return u;}\n"
        "vec4 x_p2_st(float a,float b){return x_st2(a,b);}\n"
        "vec4 x_p2_stq(float qa,float qb){vec2 p=x_q16enc(qa),q=x_q16enc(qb);"
        "return vec4(p.x,p.y,q.x,q.y);}\n";
}

// The f32 store helper on its own (see xformat.h).  `x_glsl(F32)` emits this
// plus the read macros; a kernel with a fixed q16 *input* layout and an f32
// *output* (the swiglu) emits this plus `head(XFormat::I16_2)`.
std::string x_f32_store_glsl() {
    return
        "vec4 x_f32enc(float v){uint u=floatBitsToUint(v);"
        "return vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
        "float(u&255u))/255.0;}\n";
}
