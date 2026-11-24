#include "shaders_internal.h"
#include "shaders.h"

#include <string>

using std::string;
using std::to_string;

// ---- RWKV-7 (Goose) GPU layer loop (P6) ------------------------------------
// See shaders.h for the format plan.  I24_1 activations, q16 residual, constant
// vectors uploaded *64 (RS = 1/64).  The stats loops keep the rms_stats_split
// shape (bounds sentinel, no data-dependent conditional update) so VC4's rolled
// loop cannot drop an accumulator (docs/vc4-glsl-costs.md section 6).
namespace {
const char* kRwkvRS = "#define RS 0.015625\n";  // 1/64, the constant-vector store scale
string rwkv_head() { return head(XFormat::I24_1); }
// f32 <-> RGBA8 bit-cast for the ES3 kernels (WKV state and the norm stats).
// R is the low byte, A the high; floor(c*255+0.5) recovers the exact byte.
const string kBC =
    "float bcRd(sampler2D t,vec2 uv){vec4 c=texture(t,uv);"
    "uint u=uint(floor(c.r*255.0+0.5))|(uint(floor(c.g*255.0+0.5))<<8)|"
    "(uint(floor(c.b*255.0+0.5))<<16)|(uint(floor(c.a*255.0+0.5))<<24);"
    "return uintBitsToFloat(u);}"
    "vec4 bcWr(float f){uint u=floatBitsToUint(f);"
    "return vec4(float(u&255u),float((u>>8)&255u),float((u>>16)&255u),"
    "float((u>>24)&255u))/255.0;}\n";
// I24_1 read with the ES3 texture() (texture2D is gone in 300 es).
string rd24es(const char* tex, const string& idx, const string& W) {
    return string("x_q24(texture(") + tex + ",vec2((float(" + idx + ")+0.5)/" + W + ",0.5)))";
}
// 2D constant-vector read: texel (idx, row) of a dim x rows texture (the wide
// constants are 2D so the width stays dim <= VC4's 2048; see init_rwkv).
string rd24_2d(const char* tex, const string& idx, const string& row, const string& W, const string& H) {
    return string("x_q24(texture2D(") + tex + ",vec2((float(" + idx + ")+0.5)/" + W +
           ",(float(" + row + ")+0.5)/" + H + ")))";
}
string rd24es_2d(const char* tex, const string& idx, const string& row, const string& W, const string& H) {
    return string("x_q24(texture(") + tex + ",vec2((float(" + idx + ")+0.5)/" + W +
           ",(float(" + row + ")+0.5)/" + H + ")))";
}
string wkv_uv(const string& j, const string& row, const string& HS, const string& ROWS) {
    return "vec2((float(" + j + ")+0.5)/" + HS + ".0,(float(" + row + ")+0.5)/" + ROWS + ".0)";
}
}  // namespace

std::string rwkv_ln_stats(int dim, XFormat rf) {
    need_dim("rwkv_ln_stats", dim);
    const string D = to_string(dim), RW = to_string(x_texels(dim, rf));
    // mean/invstd accumulate in f32 registers and are stored bit-cast f32 (ES3).
    // A fixed-point mean loses precision the invstd then amplifies: the token-
    // normed embedding has std ~0.013, so a 1/4096 mean step is ~18% of a
    // normalized component.  f32 stats match the CPU's double accumulation.
    // The residual is the q16 I16_1 stream; the loop is a plain float sum.
    return rwkv_head() + kBC +
        "uniform sampler2D X;uniform float eps;"
        "void main(){float s=0.0,s2=0.0;"
        "for(int i=0;i<" + D + ";i++){"
        "float v=x_i16(texture(X,vec2((float(i)+0.5)/" + RW + ".0,0.5)).rg);s+=v;s2+=v*v;}"
        "float mean=s/float(" + D + ");"
        "float inv=inversesqrt(s2/float(" + D + ")-mean*mean+eps);"
        "if(gl_FragCoord.x<1.0)gl_FragColor=bcWr(mean);else gl_FragColor=bcWr(inv);}";
}

std::string rwkv_ln_norm(int dim, XFormat rf) {
    need_dim("rwkv_ln_norm", dim);
    const string D = to_string(dim), RW = to_string(x_texels(dim, rf));
    return rwkv_head() + kBC + kRwkvRS +
        "uniform sampler2D X,S,W;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float mean=bcRd(S,vec2(0.25,0.5));"
        "float inv=bcRd(S,vec2(0.75,0.5));"
        "float x=x_i16(texture(X,vec2((float(i)+0.5)/" + RW + ".0,0.5)).rg);"
        "float w=" + rd24es_2d("W", "i", "0", D + ".0", "2.0") + "*RS;"
        "float b=" + rd24es_2d("W", "i", "1", D + ".0", "2.0") + "*RS;"
        "gl_FragColor=x_q24enc(((x-mean)*inv*w+b)*256.0);}";
}

std::string rwkv_ln_stats_p(int dim) {
    need_dim("rwkv_ln_stats_p", dim);
    const string D = to_string(dim);
    return rwkv_head() + kBC +
        "uniform sampler2D X;uniform float eps;"
        "void main(){float s=0.0,s2=0.0;"
        "for(int i=0;i<" + D + ";i++){float v=" + rd24es("X", "i", D + ".0") + "*0.00390625;s+=v;s2+=v*v;}"
        "float mean=s/float(" + D + ");"
        "float inv=inversesqrt(s2/float(" + D + ")-mean*mean+eps);"
        "if(gl_FragCoord.x<1.0)gl_FragColor=bcWr(mean);else gl_FragColor=bcWr(inv);}";
}

std::string rwkv_ln_norm_p(int dim) {
    need_dim("rwkv_ln_norm_p", dim);
    const string D = to_string(dim);
    return rwkv_head() + kBC + kRwkvRS +
        "uniform sampler2D X,S,W;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float mean=bcRd(S,vec2(0.25,0.5));"
        "float inv=bcRd(S,vec2(0.75,0.5));"
        "float x=" + rd24es("X", "i", D + ".0") + "*0.00390625;"
        "float w=" + rd24es_2d("W", "i", "0", D + ".0", "2.0") + "*RS;"
        "float b=" + rd24es_2d("W", "i", "1", D + ".0", "2.0") + "*RS;"
        "gl_FragColor=x_q24enc(((x-mean)*inv*w+b)*256.0);}";
}

// q16 residual + I24_1 delta -> q16 (the CPU add_q16 two-step: the matvec output
// is read at I24 precision, then added to the q16 residual and re-rounded).  Used
// instead of the fused matvec_add, whose dot VC4 drops on the RWKV I24_1 chain.
std::string rwkv_res_add(int dim) {
    need_dim("rwkv_res_add", dim);
    const string D = to_string(dim);
    return head(XFormat::I24_1) + x_residual_glsl(XFormat::I16_1) +
        "uniform sampler2D R,D;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float r=Y1(R,i," + D + ".0);"
        "float d=" + rd24("D", "i", D + ".0") + ";"
        "gl_FragColor=x_st1(r+d);}";
}

std::string rwkv_first_add(int dim) {
    need_dim("rwkv_first_add", dim);
    const string D = to_string(dim);
    // q16(x0/256 + delta): the first residual add, x0 precise (*256 I24_1).
    return rwkv_head() + poly_src() +
        "uniform sampler2D P,D;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float v=" + rd24("P", "i", D + ".0") + "*0.00390625+" + rd24("D", "i", D + ".0") + ";"
        "gl_FragColor=x_st1(v);}";
}

std::string rwkv_gn_stats(int dim, int hs) {
    need_dim("rwkv_gn_stats", dim);
    const string D = to_string(dim), HS = to_string(hs), H = to_string(dim / hs);
    // One fragment per (head, row): row 0 the mean, row 1 the invstd, both
    // bit-cast f32 (the WKV output is small, so a fixed-point mean/invstd would
    // round coarsely relative to it).  eps 64e-5 is the architecture's own.
    return rwkv_head() + kBC +
        "uniform sampler2D Y;"
        "void main(){int h=int(gl_FragCoord.x);float s=0.0,s2=0.0;"
        "for(int i=0;i<" + HS + ";i++){"
        "float v=" + rd24es("Y", "h*" + HS + "+i", D + ".0") + "*0.0625;s+=v;s2+=v*v;}"
        "float mean=s/float(" + HS + ");"
        "float inv=inversesqrt(s2/float(" + HS + ")-mean*mean+64e-5);"
        "if(gl_FragCoord.y<1.0)gl_FragColor=bcWr(mean);else gl_FragColor=bcWr(inv);}";
}

std::string rwkv_gn_norm(int dim, int hs) {
    need_dim("rwkv_gn_norm", dim);
    const string D = to_string(dim), HS = to_string(hs), H = to_string(dim / hs);
    return rwkv_head() + kBC + kRwkvRS +
        "uniform sampler2D Y,S,W;"
        "void main(){int i=int(gl_FragCoord.x);int h=i/" + HS + ";"
        "float mean=bcRd(S,vec2((float(h)+0.5)/" + H + ".0,0.25));"
        "float inv=bcRd(S,vec2((float(h)+0.5)/" + H + ".0,0.75));"
        "float y=" + rd24es("Y", "i", D + ".0") + "*0.0625;"
        "float w=" + rd24es_2d("W", "i", "0", D + ".0", "2.0") + "*RS;"
        "float b=" + rd24es_2d("W", "i", "1", D + ".0", "2.0") + "*RS;"
        "gl_FragColor=x_q24enc(((y-mean)*inv*w+b)*256.0);}";
}

std::string rwkv_lerp(int dim, int which, int nrows) {
    need_dim("rwkv_lerp", dim);
    const string D = to_string(dim), NR = to_string(nrows), WH = to_string(which);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D XN,SH,L;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float x=" + rd24("XN", "i", D + ".0") + "*0.00390625;"
        "float sh=" + rd24("SH", "i", D + ".0") + "*0.00390625;"
        "float lp=" + rd24_2d("L", "i", WH, D + ".0", NR + ".0") + "*RS;"
        "gl_FragColor=x_q24enc(x+(sh-x)*lp);}";
}

std::string rwkv_l2_stats(int dim, int hs) {
    need_dim("rwkv_l2_stats", dim);
    const string D = to_string(dim), HS = to_string(hs);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D K,SC;"
        "void main(){int h=int(gl_FragCoord.x);float ss=0.0;"
        "for(int i=0;i<" + HS + ";i++){int o=h*" + HS + "+i;"
        "float z=" + rd24("K", "o", D + ".0") + "*" + rd24_2d("SC", "o", "0", D + ".0", "3.0") + "*RS;ss+=z*z;}"
        "gl_FragColor=x_q24enc(sqrt(ss));}";
}

std::string rwkv_l2_norm(int dim, int hs) {
    need_dim("rwkv_l2_norm", dim);
    const string D = to_string(dim), HS = to_string(hs), H = to_string(dim / hs);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D K,SC,N;"
        "void main(){int i=int(gl_FragCoord.x);int h=i/" + HS + ";"
        "float z=" + rd24("K", "i", D + ".0") + "*" + rd24_2d("SC", "i", "0", D + ".0", "3.0") + "*RS;"
        "float nrm=x_q24(texture2D(N,vec2((float(h)+0.5)/" + H + ".0,0.5)));"
        "gl_FragColor=x_q24enc((z/max(nrm,1e-12))*16.0);}";
}

std::string rwkv_keydelta(int dim) {
    need_dim("rwkv_keydelta", dim);
    const string D = to_string(dim);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D K,A,SC;"
        "void main(){int i=int(gl_FragCoord.x);"
        "float k=" + rd24("K", "i", D + ".0") + ";"
        "float a=" + rd24("A", "i", D + ".0") + "*0.0625;"
        "float ka=" + rd24_2d("SC", "i", "1", D + ".0", "3.0") + "*RS;"
        "gl_FragColor=x_q24enc(k*(1.0+(a-1.0)*ka));}";
}

std::string rwkv_tanh(int dim) {
    need_dim("rwkv_tanh", dim);
    const string D = to_string(dim);
    // Overflow-safe: exp(-2|x|) in (0,1], the decay/gate logits reach +-80.
    return rwkv_head() + poly_src() +
        "uniform sampler2D X;"
        "float th(float x){float a=abs(x);float e=exp(-2.0*a);float t=(1.0-e)/(1.0+e);"
        "return x<0.0?-t:t;}"
        "void main(){int i=int(gl_FragCoord.x);"
        "gl_FragColor=x_q24enc(th(" + rd24("X", "i", D + ".0") + "));}";
}

std::string rwkv_sigmoid(int dim, bool has_bias, int oscale) {
    need_dim("rwkv_sigmoid", dim);
    const string D = to_string(dim), OS = to_string(oscale);
    const string bias = has_bias ? ("z+=" + rd24("B", "i", D + ".0") + "*RS;") : "";
    return rwkv_head() + (has_bias ? kRwkvRS : "") +
        "uniform sampler2D X;" + (has_bias ? "uniform sampler2D B;" : "") +
        "float sg(float z){if(z<0.0){float e=" + EXP("z") + ";return e/(1.0+e);}return 1.0/(1.0+" + EXP("-z") + ");}"
        "void main(){int i=int(gl_FragCoord.x);float z=" + rd24("X", "i", D + ".0") + ";" + bias +
        "gl_FragColor=x_q24enc(sg(z)*" + OS + ".0);}";
}

std::string rwkv_decay(int dim) {
    need_dim("rwkv_decay", dim);
    const string D = to_string(dim);
    // w = exp(-sigmoid(u+w0)/sqrt(e)) in (0.545,1]; store (1-w)*256 so the I24
    // 1/256 texel step becomes 1/65536 in w (W-1).  0.60653... = 1/sqrt(e).
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D X,B;"
        "float sg(float z){if(z<0.0){float e=" + EXP("z") + ";return e/(1.0+e);}return 1.0/(1.0+" + EXP("-z") + ");}"
        "void main(){int i=int(gl_FragCoord.x);"
        "float u=" + rd24("X", "i", D + ".0") + ";"
        "float w0=" + rd24("B", "i", D + ".0") + "*RS;"
        "float w=exp(-sg(u+w0)*0.60653065971263342);"
        "gl_FragColor=x_q24enc((1.0-w)*256.0);}";
}

std::string rwkv_vres(int dim) {
    need_dim("rwkv_vres", dim);
    const string D = to_string(dim);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D V,VF,U,B;"
        "float sg(float z){if(z<0.0){float e=" + EXP("z") + ";return e/(1.0+e);}return 1.0/(1.0+" + EXP("-z") + ");}"
        "void main(){int i=int(gl_FragCoord.x);"
        "float v=" + rd24("V", "i", D + ".0") + ";"
        "float vf=" + rd24("VF", "i", D + ".0") + ";"
        "float u=" + rd24("U", "i", D + ".0") + ";"
        "float v0=" + rd24("B", "i", D + ".0") + "*RS;"
        "gl_FragColor=x_q24enc(v+(vf-v)*sg(u+v0));}";
}

std::string rwkv_mul(int dim, int adiv) {
    need_dim("rwkv_mul", dim);
    const string D = to_string(dim);
    const string AS = adiv == 256 ? "*0.00390625" : "";
    return rwkv_head() + poly_src() +
        "uniform sampler2D A,B;"
        "void main(){int i=int(gl_FragCoord.x);"
        "gl_FragColor=x_q24enc(" + rd24("A", "i", D + ".0") + AS + "*" + rd24("B", "i", D + ".0") + ");}";
}

std::string rwkv_rk_reduce(int dim, int hs) {
    need_dim("rwkv_rk_reduce", dim);
    const string D = to_string(dim), HS = to_string(hs);
    return rwkv_head() + poly_src() + kRwkvRS +
        "uniform sampler2D K,R,SC;"
        "void main(){int h=int(gl_FragCoord.x);float acc=0.0;"
        "for(int i=0;i<" + HS + ";i++){int o=h*" + HS + "+i;"
        "acc+=" + rd24("K", "o", D + ".0") + "*" + rd24("R", "o", D + ".0") + "*"
        + rd24_2d("SC", "o", "2", D + ".0", "3.0") + "*RS;}"
        "gl_FragColor=x_q24enc(acc);}";
}

std::string rwkv_rk_add(int dim, int hs) {
    need_dim("rwkv_rk_add", dim);
    const string D = to_string(dim), HS = to_string(hs), H = to_string(dim / hs);
    return rwkv_head() + poly_src() +
        "uniform sampler2D Y,V,RK;"
        "void main(){int i=int(gl_FragCoord.x);int h=i/" + HS + ";"
        "float y=" + rd24("Y", "i", D + ".0") + "*0.00390625;"
        "float v=" + rd24("V", "i", D + ".0") + ";"
        "float rk=x_q24(texture2D(RK,vec2((float(h)+0.5)/" + H + ".0,0.5)));"
        "gl_FragColor=x_q24enc((y+v*rk)*256.0);}";
}

// WKV7 state layout: one RGBA8 texture per layer buffer, hs texels wide (the
// column j) and heads*hs rows tall (row = h*hs + i).  Each texel holds the four
// bytes of an IEEE-754 f32 S[h][i][j] (bit-cast, NOT a fixed-point encode), so
// the recurrence runs in exact f32 like the CPU reference -- the S-1 fixed-point
// grid cost (+0.154 nats) disappears.  These kernels are native ESSL 3.00: the
// link layer upgrades them when it sees uintBitsToFloat (gpu_prog.cpp), which
// needs the MESA ES3 override that RWKV_GPU requests (gpu_gl.cpp).  floatBitsToUint
// / uintBitsToFloat are measured ok on VC4 Mesa 300 es (docs/vc4-glsl-costs.md
// section 5).  sa/update/out are the three passes; the loop bounds are the
// literal head_size so Mesa unrolls them (no rolled-loop conditional hazard).


std::string rwkv_wkv_sa(int heads, int hs) {
    const string HS = to_string(hs), ROWS = to_string(heads * hs), D = ROWS;
    // sa_i = -<S[i], kk>, exact f32, one fragment per (head, row).  kk is *16.
    return rwkv_head() + kBC +
        "uniform sampler2D ST,KK;"
        "void main(){int row=int(gl_FragCoord.x);int h=row/" + HS + ";"
        "float sa=0.0;"
        "for(int j=0;j<" + HS + ";j++){"
        "float S=bcRd(ST," + wkv_uv("j", "row", HS, ROWS) + ");"
        "sa-=S*" + rd24es("KK", "h*" + HS + "+j", D + ".0") + "*0.0625;}"
        "gl_FragColor=bcWr(sa);}";
}

std::string rwkv_wkv_update(int heads, int hs) {
    const string HS = to_string(hs), ROWS = to_string(heads * hs), D = ROWS;
    // One fragment per state element (j, row), exact f32:
    //   S_new = S_old*w + v_i*k_j + sa_i*(kk_j*a_j)
    // Seven samplers: state, sa, decay, v, k, kk, a (no scale textures -- the
    // bit-cast f32 carries its own exponent).  w = 1 - decay/256, kk is *16.
    return rwkv_head() + kBC +
        "uniform sampler2D ST,SA,DC,V,K,KK,A;"
        "void main(){int j=int(gl_FragCoord.x);int row=int(gl_FragCoord.y);"
        "int h=row/" + HS + ";int i=row-h*" + HS + ";"
        "float S=bcRd(ST," + wkv_uv("j", "row", HS, ROWS) + ");"
        "float sa=bcRd(SA,vec2((float(row)+0.5)/" + ROWS + ".0,0.5));"
        "float w=1.0-" + rd24es("DC", "h*" + HS + "+j", D + ".0") + "*0.00390625;"
        "float v=" + rd24es("V", "h*" + HS + "+i", D + ".0") + ";"
        "float k=" + rd24es("K", "h*" + HS + "+j", D + ".0") + ";"
        "float kk=" + rd24es("KK", "h*" + HS + "+j", D + ".0") + "*0.0625;"
        "float a=" + rd24es("A", "h*" + HS + "+j", D + ".0") + "*0.0625;"
        "gl_FragColor=bcWr(S*w+v*k+sa*(kk*a));}";
}

std::string rwkv_wkv_out(int heads, int hs) {
    const string HS = to_string(hs), ROWS = to_string(heads * hs), D = ROWS;
    // out_i = <S_new[i], r>, exact f32 dot, stored *16 for the group norm.
    return rwkv_head() + kBC +
        "uniform sampler2D ST,R;"
        "void main(){int row=int(gl_FragCoord.x);int h=row/" + HS + ";"
        "float acc=0.0;"
        "for(int j=0;j<" + HS + ";j++){"
        "float S=bcRd(ST," + wkv_uv("j", "row", HS, ROWS) + ");"
        "acc+=S*" + rd24es("R", "h*" + HS + "+j", D + ".0") + ";}"
        "gl_FragColor=x_q24enc(acc*16.0);}";
}
