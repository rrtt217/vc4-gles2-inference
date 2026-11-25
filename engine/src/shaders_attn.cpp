#include "shaders.h"
#include "logits_grid.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>

using std::string;
using std::to_string;

namespace {

// 24-bit logits decode, shared by the sum and the two Top-K passes.  The
// layout is xformat's (x_i24dec_at at the 1/256 grid), not a copy of it.
const string kDec24Q =
    x_scaled_glsl() +
    "float dec(vec4 c){return x_i24dec_at(c,256.0);}";

// QKV holds the q/k/v matvec output.  When the qkv matvec pairs two output
// rows into one texel the reads become a half-width fetch plus a component
// select; otherwise the original 24-bit fetch is emitted verbatim.
string qkv_read(const AttnConfig& c, const char* tex, const char* macro) {
    const string t(tex), m(macro);
    if (!c.qkv_pair)
        return "\n#define " + m + "(I) dec24q(texture2D(" + t +
               ",vec2((float(I)+0.5)/float(QKVDIM),0.5)))\n";
    const string fn = string("qkvr_") + t;
    return "\nfloat " + fn + "(int i){int t=i/2;vec4 c=texture2D(" + t +
           ",vec2((float(t)+0.5)/float(QKVTW),0.5));"
           "vec2 p=(((i-t*2)==0)?c.rg:c.ba);"
           "return x_q16dec_at(p,256.0);}\n"
           "#define " + m + "(I) " + fn + "(I)\n";
}

}  // namespace

// The attention kernels bake dim/heads/seq/kv layout into #defines, and a
// config that violates the pairing assumptions (GROUP = heads/kv_heads, rope
// pairs of head_dim) compiles to a shader that indexes out of bounds.
static void need_attn(const AttnConfig& c) {
    const bool ok = c.dim > 0 && c.kv_dim > 0 && c.heads > 0 && c.kv_heads > 0 &&
                    c.head_dim > 0 && (c.head_dim % 2) == 0 && c.seq > 0 &&
                    (c.heads % c.kv_heads) == 0 && c.kv.texels > 0 &&
                    (c.kv.bits == 0 || c.kv.bits == 8 || c.kv.bits == 16);
    if (!ok) {
        fprintf(stderr, "attn config: dim=%d kv_dim=%d heads=%d kv_heads=%d head_dim=%d "
                        "seq=%d kv_bits=%d width=%d\n", c.dim, c.kv_dim, c.heads, c.kv_heads,
                c.head_dim, c.seq, c.kv.bits, c.kv.texels);
        exit(2);
    }
}

// A portable exp for the attention weights.
//
// `exp()` and `exp2()` are the *same* approximation on VC4: the QPU's ALU is
// plain fp32 (VideoCore IV 3D Architecture Reference Guide, section 3), but
// exp2/log2 are the shared SFU's table lookups and Mesa does **not** refine
// them -- unlike rcp/rsqrt, which get a Newton-Raphson step in vc4_program.c.
// Measured: 4.6e-05 worst relative error over x in [-30, 0] on the board,
// against 2.2e-06 on Xe (probes/sumorder_probe, 2026-11-17), and Mesa's own
// rpi3 piglit list still fails fs-exp-*/exp2-*/log-*/log2-*/pow-*.
// att_v multiplies that into every position's probability, so it is a
// systematic weight error, 28 layers deep, on a model whose residual reaches
// 2300.  This is a degree-6 minimax on r = x - k ln2 with 2^k built by a
// bit-cast (ESSL 3.00, which RES/ACT=f32 already requires): 1.5e-06 on VC4,
// 1.1e-06 on Xe -- the polynomial's own error, not the driver's.
//
// Selected by ATT_EXP=poly; the default stays the driver's exp so that existing
// pins and shader digests do not move.
// The name the shader bodies call: the driver's exp or the portable one.
static std::string exp_call(const AttnConfig& c){
    return c.exp_poly ? "x_exp(" : "exp(";
}


// The shader-side definition of the portable exp, or nothing: one copy lives in
// shaders_common.cpp (EXP=drv|poly, the same switch the FFN kernels read).
static std::string exp_poly_text(const AttnConfig& c){
    return c.exp_poly ? exp_poly_source() : string();
}

SegRef seg_ref(const AttnConfig& c, const string& base, const string& t,
               const string& single_tex, const string& single_y, const string& div) {
    // nseg == 1: the caller's own text, byte for byte -- this function must not
    // move the shipped GLSL, and gate/shader_hashes.sh is what proves it.
    if (c.nseg <= 1) return SegRef{single_tex, single_y};
    // A segmented program IS segment SEGK, so the plane it samples is the one
    // texture that holds this slice: no sampler chain, which ESSL 1.00 cannot
    // index anyway and which VC4's eight units would cap at eight segments.
    // `div` is that texture's own extent along the sequence axis -- SEG for the
    // KV planes, SEG/2 for the score grid, which packs two positions a texel.
    return SegRef{base, "((float(" + t + ")+0.5)/float(" + div + "))"};
}

string seg_uniforms(const AttnConfig& c, const string& base) {
    // One sampler a plane.  The chain this used to emit is gone with the
    // per-segment program: a program that owns one slice binds one texture.
    (void)c;
    return "uniform sampler2D " + base + ";";
}

string seg_seq_tex(const AttnConfig& c, const string& base) { (void)c; return base; }

string seg_rope_y(const AttnConfig& c) {
    // The rope table and the V step plane hold one texel a position of this
    // segment, so a kernel reading global `pos` fetches pos - SEGK*SEG.
    if (c.nseg <= 1) return "(pos+0.5)/float(SEQ)";
    return "((float(pos)-float(SEGK)*float(SEG)+0.5)/float(SEG))";
}


// The cross-attention K/V plane: a projection's I24 row -> one half of a KV row
// (`[K | V]`, `kv_dim/2` q16 texels each, written with the attention's own
// encoder so the grid is the one `att_scores`/`att_v` decode).
//
// `att_v_half` picks which half; the caller puts the viewport at x=0 or
// x=kv_dim/2 and y=the position, so one draw is one half of one row.  This
// replaces the host round trip a v5 decode used to pay twice per layer per token:
// read the query back, softmax and the value mix in C++, upload the context.
string att_cross_kv(const AttnConfig& c, bool v_half, XFormat xf, bool bias) {
    need_attn(c);
    // `xf` is what the *projection* wrote, and the bias must be uploaded in the
    // same one: one slot, one reader, so `X1` serves both textures.  Reading a
    // fixed format here is what put the rail in the plane once already --
    // `.gba` of a q16 pair texel is an integer in the millions and `encKV`
    // clamped it at +-16.
    const string TW = to_string(x_texels(c.kv_dim, xf));
    (void)v_half;
    // One fragment writes one *texel* of the half row, and the number of
    // projection values that texel holds comes from the KV layout: i24/f32
    // one, q16/bf16 two, q8 four.  This used to assume two and encode with
    // encKV, so an i24 half (one value a texel) read 2x and kept only the even
    // value -- half of every K/V row, silently.  (KV=q16 hid it; i24 is the
    // v5 default.)
    string fs = "precision highp float;\n" + att_defs(c) + att_helpers(c) + x_glsl(xf) +
        "uniform sampler2D XS" + (bias ? ",BS" : "") + ";uniform float XOFF;"
        "float rv(int j){return X1(XS,j," + TW + ".0)" +
        (bias ? "+X1(BS,j," + TW + ".0)" : "") + ";}"
        "void main(){int x=int(gl_FragCoord.x-XOFF);";
    if (c.kv.vpt == 4)
        fs += "int i=4*x;gl_FragColor=encKV8(rv(i),rv(i+1),rv(i+2),rv(i+3));}";
    else if (c.kv.vpt == 2)
        fs += "int i=2*x;gl_FragColor=encKV(rv(i),rv(i+1));}";
    else
        fs += "gl_FragColor=encKV1(rv(x));}";
    return fs;
}

string att_defs(const AttnConfig& c) {
    need_attn(c);
    return exp_poly_text(c) + string("#define DIM ") + to_string(c.dim) + "\n" +
           "#define KVDIM " + to_string(c.kv_dim) + "\n" +
           "#define QKVDIM " + to_string(c.dim + 2 * c.kv_dim) + "\n" +
           "#define QKVTW " + to_string((c.dim + 2 * c.kv_dim + 1) / 2) + "\n" +
           "#define HEADS " + to_string(c.heads) + "\n" +
           "#define KVHEADS " + to_string(c.kv_heads) + "\n" +
           "#define GROUP " + to_string(c.heads / c.kv_heads) + "\n" +
           "#define HEADDIM " + to_string(c.head_dim) + "\n" +
           "#define HALF " + to_string(c.head_dim / 2) + "\n" +
           "#define SEQ " + to_string(c.seq) + "\n" +
           "#define ROPEHALF " + string(c.rope_half ? "1" : "0") + "\n" +
           "#define QKNORM " + string(c.qk_norm ? "1" : "0") + "\n" +
           "#define QKEPS 0.000001\n" +
           "#define KVSCALE " + to_string(c.kv.scale) + "\n" +
           "#define SCS " + to_string(c.score_scale) + "\n" +
           "#define SCSI " + to_string(256.0f / c.score_scale) + "\n" +
           "#define SMS " + to_string(c.sum_scale) + "\n" +
           "#define SMSI " + to_string(256.0f / c.sum_scale) + "\n" +
           (c.sum_log ? "#define SUM_LOG_SCALE " + to_string(c.sum_log_scale) + "\n" +
                       "#define SUM_LOG_INV " + to_string(256.0f / c.sum_log_scale) + "\n" : "") +
           "#define KVVPT " + to_string(c.kv.vpt) + "\n" +
           "#define KVWIDTH " + to_string(c.kv.texels) + "\n" +
           "#define KVSTEP " + string(c.kv.vstep ? "1" : "0") + "\n" +
           (c.nseg > 1 ? "#define NSEG " + to_string(c.nseg) + "\n" +
                         "#define SEG " + to_string(c.seg) + "\n" +
                         "#define SEGK " + to_string(c.segk) + "\n" : "") +
           (c.qbatch > 1 ? "#define QB " + to_string(c.qbatch) + "\n" : "") +
           // A bidirectional (encoder) pass or a cross-attention has no future
           // to hide: every key is visible.  Emitted only when it is off, so
           // every decoder-only program keeps its text byte for byte
           // (gate/shader_hashes.sh).
           (!c.causal ? "#define NOCAUSAL 1\n" : "") +
           // No rotation at all (v5: a Marian block carries its positions as a
           // table added to the embedding).  The same rule as the causal flag:
           // emitted only when off, so no shipped program's text moves.
           (c.rope_none ? "#define NOROPE 1\n" : "");
}

string att_helpers(const AttnConfig& c) {
    // Every fixed-point layout comes from xformat: this file's grids are its
    // own (the score at 1/256, the statistics' sum, the KV at KVSCALE, the
    // rope table at 4096, the V step at 2^24) but the *bytes* are the
    // format's, instantiated at those steps by x_scaled_glsl.  It used to
    // re-type them, and that copy missed the q16 rail fix (32767.5, 6234328)
    // for two years' worth of kernels; gate/check_codecs.py now refuses a
    // second body in a shader TU.
    //
    // The KV value encoding.  bf16 keeps the same two values a texel and
    // gives a uniform 0.1% relative error; the fixed grid is 0.006% at the
    // head max and 2% a decade below it, which is what the attention's small
    // values pay.
    string kv_codec;
    if (c.kv.codec == KvCodec::F32) {
        // No rounding at all, which is the point: a 16-bit grid fails by an
        // LSB flip when the two drivers disagree in the last float bit, and no
        // grid width fixes that.
        kv_codec =
            "vec4 encKV1(float v){return x_f32enc(v);}\n"
            "float decKV1(vec4 c){return x_f32(c);}\n"
            "vec4 encKV(float v0,float v1){return encKV1(v1);}\n"
            "float decKV(vec4 c,int odd){return decKV1(c);}\n";
    } else if (c.kv.codec == KvCodec::Bf16) {
        kv_codec =
            "vec4 encKV1(float v){return x_i24enc_at(v,KVSCALE);}\n"
            "float decKV1(vec4 c){return x_i24dec_at(c,KVSCALE);}\n"
            "vec4 encKV(float v0,float v1){return x_bf16enc2(v0,v1);}\n"
            "float decKV(vec4 c,int odd){return (odd==1)?x_bf16(c.ba):x_bf16(c.rg);}\n";
    } else {
        kv_codec =
            "vec4 encKV1(float v){return x_i24enc_at(v,KVSCALE);}\n"
            "float decKV1(vec4 c){return x_i24dec_at(c,KVSCALE);}\n"
            "vec4 encKV(float v0,float v1){if(KVVPT==2)"
            "return vec4(x_q16enc_at(v0,KVSCALE),x_q16enc_at(v1,KVSCALE));"
            "return x_i24enc_at(v0,KVSCALE);}\n"
            "float decKV(vec4 c,int odd){"
            "if(KVVPT==2)return x_q16dec_at((odd==1)?c.ba:c.rg,KVSCALE);"
            "if(KVVPT==4)return x_i8byte_at(c[odd])/KVSCALE;return decKV1(c);}\n";
    }
    return x_scaled_glsl() +
        x_codec_glsl(XFormat::F32) + x_codec_glsl(XFormat::BF16) +
        "vec4 q16(float v){return vec4(x_q16enc_at(v,256.0),0.0,1.0);}\n"
        "float dec24raw(vec4 c){return x_i24raw_at(c);}\n"
        "float dec24q(vec4 c){return x_i24dec_at(c,256.0);}\n"
        "vec4 enc24s(float v,float s){return x_i24enc_at(v,s);}\n"
        "float dec24s(vec4 c,float s){return x_i24dec_at(c,s);}\n"
        "float dec16(vec2 c){return x_q16dec_at(c,256.0);}\n"
        "vec4 encKV8(float v0,float v1,float v2,float v3){"
        "return x_i8enc4_at(v0,v1,v2,v3,KVSCALE);}\n"
        "float decKV8(vec4 c,int idx){return x_i8byte_at(c[idx])/KVSCALE;}\n"
        "float decKV8d(vec4 c,int idx,float d){return x_i8byte_at(c[idx])*d;}\n"
        "vec4 encKV8s(float v0,float v1,float v2,float v3,float d){"
        "float si=1.0/d;return x_i8enc4_at(v0,v1,v2,v3,si);}\n" + kv_codec;
}

string att_kv_rope(const AttnConfig& c) {
    // Segmented: the rope table and (on KVSTEP) the V step plane hold one texel
    // a position of *this* segment, and the dispatch hands `pos` as the row
    // inside the segment (the viewport already put the draw there).  A write
    // kernel therefore fetches y = pos, not pos - SEGK*SEG.
    const string rt = seg_seq_tex(c, "RT"), vs = seg_seq_tex(c, "VS");
    const string rty = c.nseg > 1 ? "((float(pos)+0.5)/float(SEG))"
                                  : "(pos+0.5)/float(SEQ)";
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        (c.nseg > 1 ? seg_uniforms(c, "RT") + seg_uniforms(c, "VS") +
                          "uniform sampler2D QKV,KN;"
                    : string("uniform sampler2D QKV,RT,KN,VS;")) +
        "uniform float pos;" + qkv_read(c, "QKV", "QKVQ") +
        "void main(){int x=int(gl_FragCoord.x);"
        "if(KVVPT==4){"
        "if(x<KVDIM/4){"
        "int base=4*x;int h=base/HEADDIM;int li0=base-h*HEADDIM;"
        "float rk=1.0;"
        "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
        "float zk=QKVQ(DIM+h*HEADDIM+z);ss+=zk*zk;}"
        "rk=inversesqrt(ss/float(HEADDIM)+QKEPS);}"
        "float o0=0.0,o1=0.0,o2=0.0,o3=0.0;"
        "for(int k=0;k<4;k++){int ii=base+k;int li=li0+k;int j,ti;int even=li-(li/2)*2;"
        "float vv=QKVQ(DIM+ii);"
        "if(QKNORM==1)vv=vv*rk*dec16(texture2D(KN,vec2((float(li)+0.5)/float(HEADDIM),0.5)).rg);"
        "if(ROPEHALF==1){if(li<HALF){j=li+HALF;ti=li;}else{j=li-HALF;ti=li-HALF;}}"
        "else{j=(even==0)?li+1:li-1;ti=li/2;}"
        "int pc=DIM+h*HEADDIM+j;"
        "float bb=QKVQ(pc);"
        "if(QKNORM==1)bb=bb*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
        "vec4 rc=texture2D(" + rt + ",vec2((float(2*ti)+0.5)/float(2*HALF)," + rty + "));"
        "vec4 rs=texture2D(" + rt + ",vec2((float(2*ti+1)+0.5)/float(2*HALF)," + rty + "));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
         + string(c.rope_none ? "float oo=vv;" : "float oo=first?vv*co-bb*sn:bb*sn+vv*co;") +
        "if(k==0)o0=oo;else if(k==1)o1=oo;else if(k==2)o2=oo;else o3=oo;}"
        "gl_FragColor=encKV8(o0,o1,o2,o3);return;}"
        "int vb=4*(x-KVDIM/4);"
        "int vh=vb/HEADDIM;"
        "float v0=QKVQ(DIM+KVDIM+vb);"
        "float v1=QKVQ(DIM+KVDIM+vb+1);"
        "float v2=QKVQ(DIM+KVDIM+vb+2);"
        "float v3=QKVQ(DIM+KVDIM+vb+3);"
        "if(KVSTEP==1){float vd=dec24s(texture2D(" + vs + ",vec2((float(vh)+0.5)/float(KVHEADS)," + rty + ")),16777216.0);"
        "gl_FragColor=encKV8s(v0,v1,v2,v3,vd);return;}"
        "gl_FragColor=encKV8(v0,v1,v2,v3);return;}"
        "if(KVVPT==2){"
        "if(x<KVDIM/2){"
        "int i0=2*x;int h=i0/HEADDIM;int li0=i0-h*HEADDIM;"
        "float rk=1.0;"
        "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
        "float zk=QKVQ(DIM+h*HEADDIM+z);ss+=zk*zk;}"
        "rk=inversesqrt(ss/float(HEADDIM)+QKEPS);}"
        "float o0=0.0,o1=0.0;"
        "for(int k=0;k<2;k++){int ii=i0+k;int li=li0+k;int j,ti;int even=li-(li/2)*2;"
        "float vv=QKVQ(DIM+ii);"
        "if(QKNORM==1)vv=vv*rk*dec16(texture2D(KN,vec2((float(li)+0.5)/float(HEADDIM),0.5)).rg);"
        "if(ROPEHALF==1){if(li<HALF){j=li+HALF;ti=li;}else{j=li-HALF;ti=li-HALF;}}"
        "else{j=(even==0)?li+1:li-1;ti=li/2;}"
        "int pc=DIM+h*HEADDIM+j;"
        "float bb=QKVQ(pc);"
        "if(QKNORM==1)bb=bb*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
        "vec4 rc=texture2D(" + rt + ",vec2((float(2*ti)+0.5)/float(2*HALF)," + rty + "));"
        "vec4 rs=texture2D(" + rt + ",vec2((float(2*ti+1)+0.5)/float(2*HALF)," + rty + "));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
         + string(c.rope_none ? "float oo=vv;" : "float oo=first?vv*co-bb*sn:bb*sn+vv*co;") +
        "if(k==0)o0=oo;else o1=oo;}"
        "gl_FragColor=encKV(o0,o1);return;}"
        "int vb=2*(x-KVDIM/2);"
        "float v0=QKVQ(DIM+KVDIM+vb);"
        "float v1=QKVQ(DIM+KVDIM+vb+1);"
        "gl_FragColor=encKV(v0,v1);return;}"
        "float v=QKVQ(DIM+x);"
        "if(x>=KVDIM){gl_FragColor=encKV1(v);return;}"
        "int h=x/HEADDIM;int i=x-h*HEADDIM;int j,ti;int even=i-(i/2)*2;"
        "float rk=1.0;"
        "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
        "float zk=QKVQ(DIM+h*HEADDIM+z);ss+=zk*zk;}"
        "rk=inversesqrt(ss/float(HEADDIM)+QKEPS);}"
        "float vn=v;if(QKNORM==1)vn=v*rk*dec16(texture2D(KN,vec2((float(i)+0.5)/float(HEADDIM),0.5)).rg);"
        "if(ROPEHALF==1){if(i<HALF){j=i+HALF;ti=i;}else{j=i-HALF;ti=i-HALF;}}"
        "else{j=(even==0)?i+1:i-1;ti=i/2;}"
        "int pc=DIM+h*HEADDIM+j;"
        "float b=QKVQ(pc);"
        "float bn=b;if(QKNORM==1)bn=b*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
        "vec4 rc=texture2D(" + rt + ",vec2((float(2*ti)+0.5)/float(2*HALF)," + rty + "));"
        "vec4 rs=texture2D(" + rt + ",vec2((float(2*ti+1)+0.5)/float(2*HALF)," + rty + "));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
        + string(c.rope_none ? "gl_FragColor=encKV1(vn);}"
                             : "float o=first?vn*co-bn*sn:bn*sn+vn*co;gl_FragColor=encKV1(o);}");
}

string att_vscale(const AttnConfig& c) {
    // Per-head V grid (8-bit KV only): the max of the head values at this
    // position sets the step (max/127), so every V head gets the full 8-bit
    // range.  K keeps the global KVSCALE grid, because it is qk-normed before
    // rope and its range is already about unit RMS, while V head ranges
    // differ by an order of magnitude.
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        "uniform sampler2D QKV;" + qkv_read(c, "QKV", "QKVQ") +
        "void main(){int h=int(gl_FragCoord.x);float mx=0.0;"
        "for(int z=0;z<HEADDIM;z++){"
        "float v=QKVQ(DIM+KVDIM+h*HEADDIM+z);"
        "mx=max(mx,abs(v));}"
        "float d=max(mx,0.00001)/127.0;"
        "gl_FragColor=enc24s(d,16777216.0);}";
}

string att_qnorm_scale(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        "uniform sampler2D R;" + qkv_read(c, "R", "RQKVQ") +
        "void main(){int h=int(gl_FragCoord.x);float ss=0.0;"
        "for(int z=0;z<HEADDIM;z++){"
        "float zq=RQKVQ(h*HEADDIM+z);"
        "ss+=zq*zq;}"
        "float rq=inversesqrt(ss/float(HEADDIM)+QKEPS);"
        "gl_FragColor=enc24s(min(rq,127.0),65536.0);}";
}

string att_qrot(const AttnConfig& c) {
    // Segmented: the rope plane for this position's segment, at the offset
    // inside it (the dispatch passes the global `pos`).
    const string rt = seg_seq_tex(c, "RT");
    const string rty = seg_rope_y(c);
    // Two head dims per fragment, written as a q16 pair.  The scores pass then
    // decodes two q values per fetch instead of one 24-bit texel each, which is
    // the dominant cost of that loop.  The value grid is unchanged: enc24s(q,256)
    // stored the same integer that x_st2/q16 stores.
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        (c.nseg > 1 ? seg_uniforms(c, "RT") + "uniform sampler2D R,QN,QS;"
                    : string("uniform sampler2D R,RT,QN,QS;")) +
        "uniform float pos;" + qkv_read(c, "R", "RQKVQ") +
        "float qval(int h,int i){"
        "float rq=1.0;"
        "if(QKNORM==1){rq=dec24s(texture2D(QS,vec2((float(h)+0.5)/float(HEADS),0.5)),65536.0);}"
        "float qv=RQKVQ(h*HEADDIM+i);"
        "int j,ti;int even=i-(i/2)*2;"
        "if(ROPEHALF==1){if(i<HALF){j=i+HALF;ti=i;}else{j=i-HALF;ti=i-HALF;}}"
        "else{j=(even==0)?i+1:i-1;ti=i/2;}"
        "float b=RQKVQ(h*HEADDIM+j);"
        "if(QKNORM==1){"
        "qv=qv*rq*dec16(texture2D(QN,vec2((float(i)+0.5)/float(HEADDIM),0.5)).rg);"
        "b=b*rq*dec16(texture2D(QN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);}"
        "vec4 rc=texture2D(" + rt + ",vec2((float(2*ti)+0.5)/float(2*HALF)," + rty + "));"
        "vec4 rs=texture2D(" + rt + ",vec2((float(2*ti+1)+0.5)/float(2*HALF)," + rty + "));"
        + string(c.rope_none ? "return qv;}"
                : "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
                  "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
                  "return first?qv*co-b*sn:b*sn+qv*co;}") +
        "void main(){int x=int(gl_FragCoord.x);int h=x/(HEADDIM/2);int i0=(x-h*(HEADDIM/2))*2;"
        "vec4 p0=q16(qval(h,i0)),p1=q16(qval(h,i0+1));"
        "gl_FragColor=vec4(p0.r,p0.g,p1.r,p1.g);}";
}


// The q16 score-pair layout needs the hoisted, rotated q, so this is the only
// form: ATTN_HOIST=0 and the R/RT/QN kernel it used to select are gone.
//
// qbatch > 1 is the prefill batch form: the grid is seq/2 x (heads*QB), row
// r = h*QB + b, and a fragment scores query b of head h against the KV pair at
// x, causal up to pos + b.  The dot product, the decode and the encode are the
// serial kernel's text; only the row addressing moves.  Adjacent rows of one
// head share its K slice, which is where the win is: a 2x2 fragment quad
// coalesces its KV fetches (probes/att_out_probe, qN_serial against q1).
string att_score(const AttnConfig& c) {
    const bool B = c.qbatch > 1;
    const string open = B ? "int r=int(gl_FragCoord.y);int h=r/QB;int b=r-h*QB;"
                          : "int h=int(gl_FragCoord.y);";
    const string ypos = B ? "pos+float(b)" : "pos";
    const string qrow = B ? "(float(b)+0.5)/float(QB)" : "0.5";
    // Segmented: this program owns the pairs of one slice, so its own x is the
    // LOCAL pair index inside its score texture while the causal test needs the
    // GLOBAL position.  Both are derived from T here, once.
    const SegRef kv0 = seg_ref(c, "KV", "t0", "KV", "(float(t0)+0.5)/float(SEQ)");
    const SegRef kv1 = seg_ref(c, "KV", "t1", "KV", "(float(t1)+0.5)/float(SEQ)");
    const string g0 = c.nseg > 1 ? "(SEGK*SEG+2*T)" : "t0";
    const string g1 = c.nseg > 1 ? "(SEGK*SEG+2*T+1)" : "t1";
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        (c.nseg > 1 ? seg_uniforms(c, "KV") + "uniform sampler2D Q;"
                    : string("uniform sampler2D Q,KV;")) +
        "uniform float pos,invsqrt;"
        "vec4 qe(float v){return vec4(x_q16enc_at(v,SCS),0.0,1.0);}"
        "void main(){int T=int(gl_FragCoord.x);" + open +
        "int t0=T*2;int t1=t0+1;int kvh=h/GROUP;float s0=0.0,s1=0.0;"
        // Bidirectional: score every key.  The *scan* in att_stats/att_v is
        // what bounds which keys count, and that is `pos` = last valid key.
        + string(c.causal
                 ? "bool e0=(float(" + g0 + ")<=" + ypos + ");bool e1=(float(" + g1 + ")<=" + ypos + ");"
                 : "bool e0=true;bool e1=true;") +
        
        "if(!e0&&!e1){gl_FragColor=vec4(0.0,0.0,0.0,0.0);return;}"
        "for(int i=0;i<HEADDIM;i++){"
        "vec4 qp=texture2D(Q,vec2((float((h*HEADDIM+i)/2)+0.5)/float(DIM/2)," + qrow + "));"
        "float q=(((h*HEADDIM+i)-(h*HEADDIM+i)/2*2)==0)?dec16(qp.rg):dec16(qp.ba);"
        "int ki=kvh*HEADDIM+i;int tx=ki/KVVPT;int odd=ki-tx*KVVPT;"
        "if(e0)s0+=q*decKV(texture2D(" + kv0.sampler + ",vec2((float(tx)+0.5)/float(KVWIDTH)," + kv0.y + ")),odd);"
        "if(e1)s1+=q*decKV(texture2D(" + kv1.sampler + ",vec2((float(tx)+0.5)/float(KVWIDTH)," + kv1.y + ")),odd);}"
        "vec4 p0=qe(e0?s0*invsqrt:0.0),p1=qe(e1?s1*invsqrt:0.0);"
        "gl_FragColor=vec4(p0.r,p0.g,p1.r,p1.g);}";
}

string att_stats(const AttnConfig& c) {
    // qbatch > 1: the score texture is heads*QB rows tall and this position's
    // block starts at row qrow0 (= b, a uniform); the scan itself is unchanged.
    const bool B = c.qbatch > 1;
    const string srow = B ? "((float(h*QB)+qrow0)+0.5)/float(HEADS*QB)"
                          : "(float(h)+0.5)/float(HEADS)";
    // Segmented: this program scans the GLOBAL pair range it owns -- so the
    // causal break is right -- and fetches with the LOCAL pair index inside its
    // own score texture.  Its (max, sum) is the slice's own; the host fold in
    // GPU::att_stats_fold() combines them in double precision, which is exact
    // for the same reason the softmax is: every segment's sum was taken against
    // a max that differs by a constant, and that constant is exp(max_k - max).
    const SegRef st = seg_ref(c, "S", "T", "S", "(float(T)+0.5)/float(SEQ/2)", "SEG/2");
    // A slice owns the positions [SEGK*SEG, min(SEQ,(SEGK+1)*SEG)); the score
    // grid packs two positions a texel, so the pair range is that position
    // range halved -- floor at the start, ceil at the end.  `SEGK*SEG/2` with
    // integer division drops the last position of every slice whose length is
    // odd (SEG=205 lost one position in each of ten slices).
    const string p_lo = c.nseg > 1 ? "(SEGK*SEG)/2" : "0";
    const string p_hi = c.nseg > 1 ? "min(SEQ,(SEGK+1)*SEG)/2+1" : "SEQ/2";
    const string t_lo = p_lo;
    const string t_hi = p_hi;
    const string tloc = c.nseg > 1 ? "(T-(SEGK*SEG)/2)" : "T";
    const string sdiv = c.nseg > 1 ? "(SEG+1)/2" : "SEQ/2";
    // Pass 2 (use_gmax > 0.5) exponentiates against the folded global max, so
    // the segments' sums add directly and the fold's only rounding is the one
    // the single-draw kernel does.  Pass 1 writes its own max.
    const string gm_decl = c.nseg > 1 ? "uniform sampler2D GM;uniform float use_gmax;" : "";
    // nseg == 1 has no second pass and no GM: the shipped text uses `mx`
    // directly, and this must stay byte-identical (gate/shader_hashes.sh).
    const string mx_use = c.nseg > 1
        ? "float mxg=(use_gmax>0.5)?dec24s(texture2D(GM,vec2((float(h)+0.5)/float(HEADS),0.5)),SCS):mx;"
        : string();
    const string mx_var = c.nseg > 1 ? "mxg" : "mx";
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) + gm_decl +
        (c.nseg > 1 ? seg_uniforms(c, "S") + (B ? "uniform float pos,qrow0;"
                                                : "uniform float pos;")
                    : (B ? string("uniform sampler2D S;uniform float pos,qrow0;")
                         : string("uniform sampler2D S;uniform float pos;"))) +
        // One fragment per head: (max, sum) packed into a single texel as two q16
        // values.  Previously two fragments per head did the identical scan and
        // only differed in which of the two statistics they wrote.
        "vec2 qs(float v,float s){return x_q16enc_at(v,s);}"
        "void main(){int h=int(gl_FragCoord.x);float mx=-1e30;"
        // `pos` bounds the scan in *both* modes: causal, it is this query's
        // position; bidirectional, the host sets it to the last *valid* key.
        // Removing the bound instead scans the whole SEQ, and the positions past
        // the sentence are whatever the KV texture held -- they then enter the
        // softmax and the encoder's attention came out 100x too large.
        "for(int T=" + t_lo + ";T<" + t_hi + ";T++){int t0=T*2;if(float(t0)>pos)break;"
        "vec4 c=texture2D(" + st.sampler + ",vec2((float(" + tloc + ")+0.5)/float(" + sdiv + ")," + srow + "));"
        "mx=max(mx,dec16(c.rg)*SCSI);if(float(t0+1)<=pos)mx=max(mx,dec16(c.ba)*SCSI);}"
        + mx_use +
        "float sum=0.0;"
        "for(int T=" + t_lo + ";T<" + t_hi + ";T++){int t0=T*2;if(float(t0)>pos)break;"
        "vec4 c=texture2D(" + st.sampler + ",vec2((float(" + tloc + ")+0.5)/float(" + sdiv + ")," + srow + "));"
        "sum+=" + exp_call(c) + "dec16(c.rg)*SCSI-" + mx_var + ");"
        "if(float(t0+1)<=pos)sum+=" + exp_call(c) + "dec16(c.ba)*SCSI-" + mx_var + ");}"
        // The sum's domain is a choice, not a decision (`AttnConfig::sum_log`):
        // the linear grid is a 1/32 step, which is 2% of a peaked softmax's sum
        // (~1.5) and a pure *gain* error on every probability in the head, while
        // log2 at 1/2048 is 0.03% relative for any sum from 1 to SEQ at the cost
        // of one exp2 in att_v.  The accurate one is not the better model on
        // MiniMind3 (q8 270.3170 -> 274.4160, q4 326.1483 -> 323.8378), so the
        // linear form is the default and this branch is the A/B.
        + (c.sum_log ? string("vec2 pm=qs(") + mx_var + ",SCS),ps=qs(log2(max(sum,1e-6)),SUM_LOG_SCALE);"
                    "gl_FragColor=vec4(pm.x,pm.y,ps.x,ps.y);}"
                  : string("vec2 pm=qs(") + mx_var + ",SCS),ps=qs(sum,SMS);"
                    "gl_FragColor=vec4(pm.x,pm.y,ps.x,ps.y);}");
}

string att_prob(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers(c) +
        "uniform sampler2D S,ST;uniform float pos;"
        "void main(){int t=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
        "vec4 sm=texture2D(ST,vec2((float(2*h)+0.5)/float(2*HEADS),0.5));"
        "vec4 ss=texture2D(ST,vec2((float(2*h+1)+0.5)/float(2*HEADS),0.5));"
        "float mx=dec24s(sm,SCS);float sum=dec24s(ss,SMS);float p=0.0;"
        "if(" + string(c.causal ? "float(t)<=pos" : "true") + " && sum>0.0)p=" + exp_call(c) + "dec24s(texture2D(S,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),SCS)-mx)/sum;"
        "gl_FragColor=enc24s(p,65536.0);}";
}


string att_v(const AttnConfig& c, XFormat out) {
    // The q16 score/stats layout packs (max,sum) into one texel per head, so the
    // probability decode is fused into this pass; there is no probability
    // texture to read and no unfused variant to keep.
    // qbatch > 1: S is the batched score texture (row h*QB + qrow0); ST stays
    // this position's own (heads,1) stats target, and the V pass is unchanged.
    const bool B = c.qbatch > 1;
    const string srow = B ? "((float(h*QB)+qrow0)+0.5)/float(HEADS*QB)"
                          : "(float(h)+0.5)/float(HEADS)";
    // Segmented: the score texture, the KV cache and the V step plane are all
    // this slice's; ST stays the folded global stats.  The scan walks the global
    // pair range and fetches with the local index.
    const SegRef sp = seg_ref(c, "S", "T", "S", "(float(T)+0.5)/float(SEQ/2)", "SEG/2");
    // The KV cache and the V step plane of segment k hold the positions of that
    // slice only, so a GLOBAL position has to be rebased: t0 - SEGK*SEG.  The
    // score grid is addressed by the local pair index instead, which is why
    // that one goes through tloc.
    const string l0 = c.nseg > 1 ? "(t0-SEGK*SEG)" : "t0";
    const string l1 = c.nseg > 1 ? "(t1-SEGK*SEG)" : "t1";
    const SegRef vs0 = seg_ref(c, "VS", l0, "VS", "(float(t0)+0.5)/float(SEQ)");
    const SegRef vs1 = seg_ref(c, "VS", l1, "VS", "(float(t1)+0.5)/float(SEQ)");
    const SegRef kv0 = seg_ref(c, "KV", l0, "KV", "(float(t0)+0.5)/float(SEQ)");
    const SegRef kv1 = seg_ref(c, "KV", l1, "KV", "(float(t1)+0.5)/float(SEQ)");
    // Same pairing as att_stats: the pair range is floor(start/2) .. ceil(end/2).
    const string t_lo = c.nseg > 1 ? "(SEGK*SEG)/2" : "0";
    const string t_hi = c.nseg > 1 ? "min(SEQ,(SEGK+1)*SEG)/2+1" : "SEQ/2";
    const string tloc = c.nseg > 1 ? "(T-(SEGK*SEG)/2)" : "T";
    const string sdiv = c.nseg > 1 ? "(SEG+1)/2" : "SEQ/2";
    const string prob_bind = "S,ST,KV,VS";
    const string prob_body =
        "vec4 st=texture2D(ST,vec2((float(h)+0.5)/float(HEADS),0.5));"
        "float mx=dec16(st.rg)*SCSI;float sum=" +
        string(c.sum_log ? "exp2(dec16(st.ba)*SUM_LOG_INV)" : "dec16(st.ba)*SMSI") +
        ";"
        "float invs=(sum>0.0)?(1.0/sum):0.0;";

    // Two (or four) output channels per fragment share one probability decode and
    // one pass over the KV rows.  HEADDIM is even, so a fragment never straddles a
    // head boundary and a single head index is enough.
    const int N = (out == XFormat::I8_4) ? 4 : (out == XFormat::F32 ? 1 : 2);
    // The V index depends only on the head/dim, not on t, so it is computed once
    // outside the loop.  The loop is unrolled by two: one score texel already
    // holds the pair (t, t+1), so two positions cost one score fetch.
    string heads_idx;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        heads_idx += "int ci" + K + "=i0+" + K + ";int vi" + K + "=kvh*HEADDIM+ci" + K + ";"
                     "int tx" + K + "=KVDIM/KVVPT+vi" + K + "/KVVPT;"
                     "int od" + K + "=vi" + K + "-(vi" + K + "/KVVPT)*KVVPT;";
    }
    string bodies, accs;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        bodies += "a" + K + "+=p0*v0_" + K + ";";
        accs += "a" + K + (k + 1 < N ? "," : "");
    }
    // 8-bit V carries a per-head, per-position step: one fetch per position,
    // shared by every channel of the fragment (they all sit in one KV head).
    const bool vstep = (c.kv.bits == 8 && c.kv.vstep);
    const string step0 = "float d0=dec24s(texture2D(" + vs0.sampler + ",vec2((float(kvh)+0.5)/float(KVHEADS)," + vs0.y + ")),16777216.0);";
    const string step1 = "float d1=dec24s(texture2D(" + vs1.sampler + ",vec2((float(kvh)+0.5)/float(KVHEADS)," + vs1.y + ")),16777216.0);";
    string loads0, loads1;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        const string s0 = "texture2D(" + kv0.sampler + ",vec2((float(tx" + K + ")+0.5)/float(KVWIDTH)," + kv0.y + "))";
        const string s1 = "texture2D(" + kv1.sampler + ",vec2((float(tx" + K + ")+0.5)/float(KVWIDTH)," + kv1.y + "))";
        loads0 += "float v0_" + K + "=" + (vstep ? "decKV8d(" + s0 + ",od" + K + ",d0)" : "decKV(" + s0 + ",od" + K + ")") + ";";
        loads1 += "float v1_" + K + "=" + (vstep ? "decKV8d(" + s1 + ",od" + K + ",d1)" : "decKV(" + s1 + ",od" + K + ")") + ";";
    }
    string init;
    for (int k = 0; k < N; ++k) init += "float a" + to_string(k) + "=0.0;";
    // Segmented: a slice can only produce a partial, and the packed fixed-point
    // layouts cannot be added by the blender (adding two q16 pairs channel by
    // channel does not carry).  So the kernel reads the destination and adds in
    // float before re-encoding -- one texture fetch, no host round trip, and the
    // decode is the format's own X1, not a second copy of it in C++.
    //
    // The fragment covers N logical values at x0 = N*t .. N*t+N-1, which is
    // exactly what X1(ACC, i, ACCW) addresses, so the add is one call a value.
    // Only the *adding* passes read the destination; the first segment of a
    // group overwrites it, so its program must not accumulate (that would fold
    // in the previous layer's output, which is the 35x error this had).
    string acc_read;
    if (c.accumulate)
        for (int k = 0; k < N; ++k)
            acc_read += "a" + to_string(k) + "+=X1(ACC," + to_string(N) + "*t+" +
                        to_string(k) + ",ACCW);";
    string enc = (N == 4) ? "gl_FragColor=x_i8enc4(" + accs + ");}"
                          : (N == 1) ? "gl_FragColor=XST1(" + accs + ");}"
                          : (out == XFormat::BF16) ? "gl_FragColor=x_bf16enc2(" + accs + ");}"
                          : "gl_FragColor=x_st2(" + accs + ");}";

    string fs = "precision highp float;\n" + x_glsl(out) +
                (out == XFormat::BF16 ? string() : x_codec_glsl(XFormat::BF16)) +
                att_defs(c) + att_helpers(c) +
        (c.nseg > 1 ? seg_uniforms(c, "S") + seg_uniforms(c, "KV") + seg_uniforms(c, "VS") +
                          "uniform sampler2D ST;"
                    : "uniform sampler2D " + prob_bind) +
        (c.accumulate ? ";uniform sampler2D ACC;uniform float ACCW;" : ";") +
        (B ? ";uniform float pos,qrow0;" : ";uniform float pos;") +
        "void main(){int t=int(gl_FragCoord.x);int x0=" + to_string(N) + "*t;"
        "int h=x0/HEADDIM;int i0=x0-h*HEADDIM;int kvh=h/GROUP;" + init + heads_idx +
        prob_body +
        "int mxT=int(pos+0.5)/2+1;"
        "for(int T=" + t_lo + ";T<" + t_hi + ";T++){if(T>=mxT)break;" +
        "int t0=T*2;int t1=t0+1;"
        "vec4 sp=texture2D(" + sp.sampler + ",vec2((float(" + tloc + ")+0.5)/float(" + sdiv + ")," + srow + "));"
        "float p0=" + exp_call(c) + "dec16(sp.rg)*SCSI-mx)*invs;" + (vstep ? step0 : string("")) +
        loads0 + bodies +
        "if(float(t1)<=pos){float p1=" + exp_call(c) + "dec16(sp.ba)*SCSI-mx)*invs;" + (vstep ? step1 : string("")) + loads1 +
        "a0+=p1*v1_0;" + (N > 1 ? "a1+=p1*v1_1;" : "") + (N > 2 ? "a2+=p1*v1_2;a3+=p1*v1_3;" : "") + "}}"
        + acc_read + enc;
    return fs;
}



string att_v_fold_shader(const AttnConfig& c, XFormat out, int nseg, bool accumulate) {
    // One fragment an output texel.  The read is the format's own X1 and the
    // write is its own XST, so a fold cannot disagree with the kernel that
    // produced the slices -- which a hand-written C++ decode did (I24_1 puts
    // its value in .gba with a zero top byte; reading .rgba as a 24-bit number
    // is a different format).
    const int N = (out == XFormat::I8_4) ? 4 : (out == XFormat::F32 ? 1 : 2);
    string reads;
    for (int k = 0; k < nseg; ++k) {
        for (int v = 0; v < N; ++v) {
            const string idx = to_string(N) + "*t+" + to_string(v);
            reads += "a" + to_string(v) + "+=X1(V" + to_string(k) + "," + idx + ",VW);";
        }
    }
    string accs;
    for (int v = 0; v < N; ++v) accs += (v ? "," : "") + string("a") + to_string(v);
    string enc = (N == 4) ? "gl_FragColor=x_i8enc4(" + accs + ");}"
                          : (N == 1) ? "gl_FragColor=XST1(" + accs + ");}"
                          : (out == XFormat::BF16) ? "gl_FragColor=x_bf16enc2(" + accs + ");}"
                          : "gl_FragColor=x_st2(" + accs + ");}";
    // A later round adds its slices to what earlier rounds already wrote, so it
    // reads the destination through the format's own X1 -- the same reason the
    // value kernel cannot let the blender do this: packed fixed point does not
    // carry across channels.
    if (accumulate)
        for (int v = 0; v < N; ++v)
            reads = "a" + to_string(v) + "+=X1(ACC," + to_string(N) + "*t+" +
                    to_string(v) + ",VW);" + reads;
    string decl, init;
    for (int k = 0; k < nseg; ++k) decl += "uniform sampler2D V" + to_string(k) + ";";
    if (accumulate) decl += "uniform sampler2D ACC;";
    for (int v = 0; v < N; ++v) init += "float a" + to_string(v) + "=0.0;";
    return "precision highp float;\n" + x_glsl(out) +
           (out == XFormat::BF16 ? string() : x_codec_glsl(XFormat::BF16)) +
           att_defs(c) + decl + "uniform float VW;"
           "void main(){int t=int(gl_FragCoord.x);" + init + reads + enc;
}

string top2_fast(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr;"
               + kDec24Q +
        "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "vec4 c=texture2D(L,vec2((float(idx)+0.5)/" + to_string(kLogitsGrid) + ".0,(tr+0.5)/nr));float v=dec(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

string top2(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr,invt;"
               + kDec24Q +
        "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "vec4 c=texture2D(L,vec2((float(idx)+0.5)/" + to_string(kLogitsGrid) + ".0,(tr+0.5)/nr));float v=dec(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}"
        "if(v>lm){ls=ls*exp((lm-v)*invt)+1.0;lm=v;}else{ls+=exp((v-lm)*invt);}}"
        "if(rank<2){float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);return;}"
        "float sum=ls;float mv=lm*256.0;if(mv<0.0)mv+=65536.0;float mh=floor(mv/256.0),ml=mv-mh*256.0;"
        "float sv=floor(sum*2048.0+0.5);float sh=floor(sv/256.0),sl=sv-sh*256.0;"
        "gl_FragColor=vec4(mh/255.0,ml/255.0,sh/255.0,sl/255.0);}";
}

// Packed-logits Top-K: the logits target is half width, two q16 logits per texel
// (.rg = even vocabulary index, .ba = odd).  Same chunking and same output
// encoding as the unpacked versions, only the sample changes.
string top2_fast_packed(int rows, int chunk) {
    return "precision highp float;" + x_scaled_glsl() +
        "uniform sampler2D L;uniform float tr,nr;"
        "float dl(vec4 c){return x_q16dec_at(c.rg,256.0);}"
        "float dh(vec4 c){return x_q16dec_at(c.ba,256.0);}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "int h=idx/2;vec4 c=texture2D(L,vec2((float(h)+0.5)/" + to_string(kLogitsGridPacked) + ".0,(tr+0.5)/nr));"
        "float v=((h*2)==idx)?dl(c):dh(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

// Greedy argmax, stage 2a: one fragment per (tile, group of 16 chunk
// winners) reduces the rank-0 texels of the top2_fast_packed output (even x
// of a 2n-wide target) to one winner per group.  The decode is the same
// byte-exact floor form; the value stays the raw signed integer (dividing
// by 256 would not change the ordering).  Strict > keeps the first chunk on
// a tie -- max_element's rule, applied at every level.
//
// Why this shape: measured on VC4 (2026-11-15, ARGMAX_DUMP), a loop the
// compiler does not unroll loses every conditional index update after the
// first iteration -- the float max stays correct, the index sticks.  A
// 96-iteration reduce and a 16-iteration loop with a runtime-bound break
// both showed it; the top2 loop with its literal bound is exact.  So this
// loop has NO break at all: the pass clears the chunk target to the
// sentinel texel (-32768, 65535) before stage 1, the short last tile's
// trailing chunks stay sentinel, and the smallest q16 value can never beat
// a real winner (a tie keeps the earlier, real chunk).
string top1_group(int n) {
    return "precision highp float;" + x_scaled_glsl() +
        "uniform sampler2D L;uniform float nr;"
        "void main(){int g=int(gl_FragCoord.x);int r=int(gl_FragCoord.y);"
        "int start=g*16;"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int k=0;k<16;k++){int c=start+k;"
        "vec4 t=texture2D(L,vec2((float(c*2)+0.5)/" + to_string(n * 2) + ".0,(float(r)+0.5)/nr));"
        "float v=x_q16raw_at(t.rg);"
        "int idx=int(floor(t.b*255.0+0.5)*256.0+floor(t.a*255.0+0.5));"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "if(best1<0.0)best1+=65536.0;"
        "float vh=floor(best1/256.0),vl=best1-vh*256.0;"
        "float ih=floor(float(i1)/256.0),il=float(i1)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

// Greedy argmax, stage 2b: one fragment per tile reduces the group winners
// to the tile winner.  Same encoding and tie rule; the loop is the group
// count (<= 16 for every vocabulary the packer produces), again the proven
// length.  The host reads one texel per tile and keeps the first maximum
// across tiles, so the global winner is the smallest token id among the
// maxima -- byte-identical to the full read-back + max_element.
string top1_final(int groups) {
    return "precision highp float;" + x_scaled_glsl() +
        "uniform sampler2D L;uniform float nr;"
        "void main(){int r=int(gl_FragCoord.y);"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int c=0;c<" + to_string(groups) + ";c++){"
        "vec4 t=texture2D(L,vec2((float(c)+0.5)/" + to_string(groups) + ".0,(float(r)+0.5)/nr));"
        "float v=x_q16raw_at(t.rg);"
        "int idx=int(floor(t.b*255.0+0.5)*256.0+floor(t.a*255.0+0.5));"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "if(best1<0.0)best1+=65536.0;"
        "float vh=floor(best1/256.0),vl=best1-vh*256.0;"
        "float ih=floor(float(i1)/256.0),il=float(i1)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

string top2_packed(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr,invt;"
        "float dl(vec4 c){return x_q16dec_at(c.rg,256.0);}"
        "float dh(vec4 c){return x_q16dec_at(c.ba,256.0);}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "int h=idx/2;vec4 c=texture2D(L,vec2((float(h)+0.5)/" + to_string(kLogitsGridPacked) + ".0,(tr+0.5)/nr));"
        "float v=((h*2)==idx)?dl(c):dh(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}"
        "if(v>lm){ls=ls*exp((lm-v)*invt)+1.0;lm=v;}else{ls+=exp((v-lm)*invt);}}"
        "if(rank<2){float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);return;}"
        "float sum=ls;float mv=lm*256.0;if(mv<0.0)mv+=65536.0;float mh=floor(mv/256.0),ml=mv-mh*256.0;"
        "float sv=floor(sum*2048.0+0.5);float sh=floor(sv/256.0),sl=sv-sh*256.0;"
        "gl_FragColor=vec4(mh/255.0,ml/255.0,sh/255.0,sl/255.0);}";
}
