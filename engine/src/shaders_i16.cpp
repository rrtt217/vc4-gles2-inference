#include "shaders_internal.h"
#include "shaders.h"

#include <string>

using std::string;
using std::to_string;

string matvec_i16(int rows, int cols, XFormat xf, XFormat of) {
    need_shape("matvec_i16", rows, cols, 1);
    const int tw = (cols + 1) / 2;
    const int axw = x_texels(cols, xf);
    return head(xf) + x_proj_glsl(of) +
        "uniform sampler2D W,X;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + to_string(tw) + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float u0=x_i16raw(c.gr);float u1=x_i16raw(c.ab);"
        "vec2 x=X2Q(X,2*q," + to_string(axw) + ".0);"
        "s+=u0*x.x;if(2*q+1<" + to_string(cols) + ")s+=u1*x.y;}"
        "gl_FragColor=x_projenc(s*ws);}";
}

string matvec_i16_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of) {
    need_shape("matvec_i16_add", rows, cols, 1);
    const int tw = (cols + 1) / 2;
    const int axw = x_texels(cols, xf);
    const string ST = (of == rf) ? std::string("YST1Q") : std::string("xs1");
    return head(xf) + x_residual_glsl(rf) + (of == rf ? std::string() : x_store_glsl(of)) +
        "uniform sampler2D W,X,R;uniform float ws;"
        "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
        "for(int q=0;q<" + to_string(tw) + ";q++){"
        "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tw) + "),"
        "(float(r)+0.5)/float(" + to_string(rows) + ")));"
        "float u0=x_i16raw(c.gr);float u1=x_i16raw(c.ab);"
        "vec2 x=X2Q(X,2*q," + to_string(axw) + ".0);"
        "s+=u0*x.x;if(2*q+1<" + to_string(cols) + ")s+=u1*x.y;}"
        "float res=Y1Q(R,r," + to_string(x_texels(rows, rf)) + ".0);"
        "gl_FragColor=" + ST + "(s*ws+res);}";
}
