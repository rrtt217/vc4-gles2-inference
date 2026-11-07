#include "xformat.h"
#include "common.h"

#include <algorithm>
#include <cmath>

int XLayout::texels() const { return x_texels(count, fmt); }

const char* x_format_name(XFormat f) {
    switch (f) {
        case XFormat::I16_1: return "x16_1";
        case XFormat::I16_2: return "x16_2";
        case XFormat::I8_4: return "x8_4";
        case XFormat::I24_1: return "x24_1";
    }
    return "x?";
}

int x_per_texel(XFormat f) {
    switch (f) {
        case XFormat::I16_1: return 1;
        case XFormat::I16_2: return 2;
        case XFormat::I8_4: return 4;
        case XFormat::I24_1: return 1;
    }
    return 0;
}

int x_texels(int count, XFormat f) {
    const int n = x_per_texel(f);
    return n ? (count + n - 1) / n : 0;
}

bool x_is_int8(XFormat f) { return f == XFormat::I8_4; }
bool x_is_int24(XFormat f) { return f == XFormat::I24_1; }

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

std::string x_glsl(XFormat f) {
    // Values crossing this boundary are real activations.  XST* converts to
    // q16/q8 on write.  X1/X2/X4 return real values; the Q-suffixed variants
    // return q16 units for hot matvec loops.  No shader needs to know the byte
    // layout or the 1/16 scale.
    std::string s =
        "float x_i16raw(vec2 c){float u=c.x*65280.0+c.y*255.0;"
        "if(u>=32768.0)u-=65536.0;return u;}\n"
        "float x_i16(vec2 c){return x_i16raw(c)/256.0;}\n"
        "vec2 x_i16pair(vec4 c){return vec2(x_i16(c.rg),x_i16(c.ba));}\n"
        "vec2 x_i16pairq(vec4 c){float u0=c.x*65280.0+c.y*255.0;"
        "if(u0>=32768.0)u0-=65536.0;float u1=c.z*65280.0+c.w*255.0;"
        "if(u1>=32768.0)u1-=65536.0;return vec2(u0,u1);}\n"
        "vec4 x_i16quadq(vec4 a,vec4 b){float u0=a.x*65280.0+a.y*255.0;"
        "if(u0>=32768.0)u0-=65536.0;float u1=a.z*65280.0+a.w*255.0;"
        "if(u1>=32768.0)u1-=65536.0;float u2=b.x*65280.0+b.y*255.0;"
        "if(u2>=32768.0)u2-=65536.0;float u3=b.z*65280.0+b.w*255.0;"
        "if(u3>=32768.0)u3-=65536.0;return vec4(u0,u1,u2,u3);}\n"
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
                "#define R1Q(T,I,TW) x_q24raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)))\n";
        case XFormat::I16_1:
            return
                "#define R1(T,I,TW) x_i16(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n"
                "#define R2(T,BASE,TW) vec2(R1(T,(BASE),TW),R1(T,(BASE)+1,TW))\n"
                "#define R1Q(T,I,TW) x_i16raw(texture2D(T,vec2((float(I)+0.5)/(TW),0.5)).rg)\n";
        case XFormat::I16_2:
            return
                "#define R1(T,I,TW) ((((I)/2)*2)==(I) ? x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n"
                "#define R2(T,BASE,TW) x_i16pair(texture2D(T,vec2((float((BASE)/2)+0.5)/(TW),0.5)))\n"
                "#define R1Q(T,I,TW) ((((I)/2)*2)==(I) ? x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).rg) "
                ": x_i16raw(texture2D(T,vec2((float((I)/2)+0.5)/(TW),0.5)).ba))\n";
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
