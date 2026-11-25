#include "shaders_internal.h"
#include "shaders.h"

#include <string>

using std::string;
using std::to_string;

string matvec_q6(int rows, int cols, XFormat xf, XFormat of, bool wrow) {
    need_shape("matvec_q6", rows, cols, 8);
    // X4Q is format-generic, so a float32 activation row works here too; the
    // q16 pair grid is only what the VC4 fast path was tuned for.
    if (xf != XFormat::I16_2 && xf != XFormat::F32 && xf != XFormat::BF16)
        { fprintf(stderr, "shader matvec_q6: q6 head needs I16_2, BF16 or F32 activations\n"); exit(2); }
    const int axw = x_texels(cols, xf);
    const int lo_w = cols / 8;    // one texel = 4 bytes = 8 nibbles
    const int hi_w = cols / 16;   // one texel = 4 bytes = 16 two-bit values
    const string WD = wrow ? wrow_scale(rows) : "";
    const string SC = wrow ? "wsr(r)" : "ws";
    // The scale uniforms must be declared before the wrow helper that uses
    // them (the q8 kernels order it that way; GLSL has no forward declaration).
    return head(xf) + x_proj_glsl(of) +
        "uniform sampler2D W,W2,X;uniform float ws;" + WD +
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;float sc=" + SC + ";"
        "for(int q=0;q<" + to_string(lo_w) + ";q++){"
        "vec4 lo=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(lo_w) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "vec4 hc=texture2D(W2,vec2((float(q/2)+0.5)/float(" + to_string(hi_w) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float b0=floor(lo.r*255.0+0.5),b1=floor(lo.g*255.0+0.5),"
        "b2=floor(lo.b*255.0+0.5),b3=floor(lo.a*255.0+0.5);"
        "float l0=mod(b0,16.0),l1=floor(b0/16.0),l2=mod(b1,16.0),l3=floor(b1/16.0),"
        "l4=mod(b2,16.0),l5=floor(b2/16.0),l6=mod(b3,16.0),l7=floor(b3/16.0);"
        // even q takes the hi texel's .rg (values 0..7), odd q its .ba (8..15)
        "vec2 hb=(q-(q/2)*2==0)?hc.rg:hc.ba;"
        "float h0=floor(hb.x*255.0+0.5),h1=floor(hb.y*255.0+0.5);"
        "float g0=mod(h0,4.0),g1=mod(floor(h0/4.0),4.0),g2=mod(floor(h0/16.0),4.0),"
        "g3=floor(h0/64.0),g4=mod(h1,4.0),g5=mod(floor(h1/4.0),4.0),"
        "g6=mod(floor(h1/16.0),4.0),g7=floor(h1/64.0);"
        "float v0=l0+g0*16.0,v1=l1+g1*16.0,v2=l2+g2*16.0,v3=l3+g3*16.0,"
        "v4=l4+g4*16.0,v5=l5+g5*16.0,v6=l6+g6*16.0,v7=l7+g7*16.0;"
        "v0=v0>=32.0?v0-64.0:v0;v1=v1>=32.0?v1-64.0:v1;"
        "v2=v2>=32.0?v2-64.0:v2;v3=v3>=32.0?v3-64.0:v3;"
        "v4=v4>=32.0?v4-64.0:v4;v5=v5>=32.0?v5-64.0:v5;"
        "v6=v6>=32.0?v6-64.0:v6;v7=v7>=32.0?v7-64.0:v7;"
        "vec4 xa=X4Q(X,q*8," + to_string(axw) + ".0);"
        "vec4 xb=X4Q(X,q*8+4," + to_string(axw) + ".0);"
        "s+=v0*xa.x+v1*xa.y+v2*xa.z+v3*xa.w+v4*xb.x+v5*xb.y+v6*xb.z+v7*xb.w;}"
        "gl_FragColor=x_projenc(s*sc);}";
}
