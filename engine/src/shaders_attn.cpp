#include "shaders.h"

#include <cstdlib>
#include <cstring>

#include <string>

using std::string;
using std::to_string;

namespace {

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
           "float u=floor(p.x*255.0+0.5)*256.0+floor(p.y*255.0+0.5);"
           "if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
           "#define " + m + "(I) " + fn + "(I)\n";
}

}  // namespace

string att_defs(const AttnConfig& c) {
    return string("#define DIM ") + to_string(c.dim) + "\n" +
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
           "#define KVSCALE " + to_string(c.kv_scale) + "\n" +
           "#define KVPACK " + to_string(c.kv_bits) + "\n" +
           "#define KVWIDTH " + to_string(c.kv_width) + "\n" +
           "#define KVSTEP " + string(c.kv_vstep ? "1" : "0") + "\n";
}

string att_helpers() {
    return
        "vec4 q16(float v){float q=clamp(floor(v*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;float hi=floor(q/256.0),lo=q-hi*256.0;return vec4(hi/255.0,lo/255.0,0.0,1.0);}\n"
        "float dec24raw(vec4 c){return floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;}\n"
        "float dec24q(vec4 c){float u=dec24raw(c);if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
        "vec4 enc24s(float v,float s){float q=floor(v*s+0.5)+8388608.0;float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n"
        "float dec24s(vec4 c,float s){return dec24raw(c)/s;}\n"
        "float dec16(vec2 c){float u=c.x*65280.0+c.y*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
        "vec4 encKV(float v0,float v1){if(KVPACK==16){float q0=clamp(floor(v0*KVSCALE+0.5),-32768.0,32767.0);if(q0<0.0)q0+=65536.0;float q1=clamp(floor(v1*KVSCALE+0.5),-32768.0,32767.0);if(q1<0.0)q1+=65536.0;float h0=floor(q0/256.0),l0=q0-h0*256.0,h1=floor(q1/256.0),l1=q1-h1*256.0;return vec4(h0/255.0,l0/255.0,h1/255.0,l1/255.0);}return enc24s(v0,KVSCALE);}\n"
        "vec4 encKV8(float v0,float v1,float v2,float v3){float q0=clamp(floor(v0*KVSCALE+0.5),-128.0,127.0);if(q0<0.0)q0+=256.0;float q1=clamp(floor(v1*KVSCALE+0.5),-128.0,127.0);if(q1<0.0)q1+=256.0;float q2=clamp(floor(v2*KVSCALE+0.5),-128.0,127.0);if(q2<0.0)q2+=256.0;float q3=clamp(floor(v3*KVSCALE+0.5),-128.0,127.0);if(q3<0.0)q3+=256.0;return vec4(q0/255.0,q1/255.0,q2/255.0,q3/255.0);}\n"
        "float decKV8(vec4 c,int idx){float q=(idx==0)?c.r:(idx==1)?c.g:(idx==2)?c.b:c.a;float u=floor(q*255.0+0.5);if(u>=128.0)u-=256.0;return u/KVSCALE;}\n"
        "float decKV(vec4 c,int odd){if(KVPACK==16){float hi=(odd==1)?c.b:c.r;float lo=(odd==1)?c.a:c.g;float u=floor(hi*255.0+0.5)*256.0+floor(lo*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/KVSCALE;}if(KVPACK==8)return decKV8(c,odd);return dec24s(c,KVSCALE);}\n"
        "float decKV8d(vec4 c,int idx,float d){float q=(idx==0)?c.r:(idx==1)?c.g:(idx==2)?c.b:c.a;float u=floor(q*255.0+0.5);if(u>=128.0)u-=256.0;return u*d;}\n"
        "vec4 encKV8s(float v0,float v1,float v2,float v3,float d){float q0=clamp(floor(v0/d+0.5),-128.0,127.0);if(q0<0.0)q0+=256.0;float q1=clamp(floor(v1/d+0.5),-128.0,127.0);if(q1<0.0)q1+=256.0;float q2=clamp(floor(v2/d+0.5),-128.0,127.0);if(q2<0.0)q2+=256.0;float q3=clamp(floor(v3/d+0.5),-128.0,127.0);if(q3<0.0)q3+=256.0;return vec4(q0/255.0,q1/255.0,q2/255.0,q3/255.0);}\n";
}

string att_kv_rope(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D QKV,RT,KN,VS;uniform float pos;" + qkv_read(c, "QKV", "QKVQ") +
        "void main(){int x=int(gl_FragCoord.x);"
        "if(KVPACK==8){"
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
        "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
        "float oo=first?vv*co-bb*sn:bb*sn+vv*co;if(k==0)o0=oo;else if(k==1)o1=oo;else if(k==2)o2=oo;else o3=oo;}"
        "gl_FragColor=encKV8(o0,o1,o2,o3);return;}"
        "int vb=4*(x-KVDIM/4);"
        "int vh=vb/HEADDIM;"
        "float v0=QKVQ(DIM+KVDIM+vb);"
        "float v1=QKVQ(DIM+KVDIM+vb+1);"
        "float v2=QKVQ(DIM+KVDIM+vb+2);"
        "float v3=QKVQ(DIM+KVDIM+vb+3);"
        "if(KVSTEP==1){float vd=dec24s(texture2D(VS,vec2((float(vh)+0.5)/float(KVHEADS),(pos+0.5)/float(SEQ))),16777216.0);"
        "gl_FragColor=encKV8s(v0,v1,v2,v3,vd);return;}"
        "gl_FragColor=encKV8(v0,v1,v2,v3);return;}"
        "if(KVPACK==16){"
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
        "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
        "float oo=first?vv*co-bb*sn:bb*sn+vv*co;if(k==0)o0=oo;else o1=oo;}"
        "gl_FragColor=encKV(o0,o1);return;}"
        "int vb=2*(x-KVDIM/2);"
        "float v0=QKVQ(DIM+KVDIM+vb);"
        "float v1=QKVQ(DIM+KVDIM+vb+1);"
        "gl_FragColor=encKV(v0,v1);return;}"
        "float v=QKVQ(DIM+x);"
        "if(x>=KVDIM){gl_FragColor=enc24s(v,KVSCALE);return;}"
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
        "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
        "float o=first?vn*co-bn*sn:bn*sn+vn*co;gl_FragColor=enc24s(o,KVSCALE);}";
}

string att_vscale(const AttnConfig& c) {
    // Per-head V grid (8-bit KV only): the max of the head values at this
    // position sets the step (max/127), so every V head gets the full 8-bit
    // range.  K keeps the global KVSCALE grid, because it is qk-normed before
    // rope and its range is already about unit RMS, while V head ranges
    // differ by an order of magnitude.
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D QKV;" + qkv_read(c, "QKV", "QKVQ") +
        "void main(){int h=int(gl_FragCoord.x);float mx=0.0;"
        "for(int z=0;z<HEADDIM;z++){"
        "float v=QKVQ(DIM+KVDIM+h*HEADDIM+z);"
        "mx=max(mx,abs(v));}"
        "float d=max(mx,0.00001)/127.0;"
        "gl_FragColor=enc24s(d,16777216.0);}";
}

string att_qnorm_scale(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D R;" + qkv_read(c, "R", "RQKVQ") +
        "void main(){int h=int(gl_FragCoord.x);float ss=0.0;"
        "for(int z=0;z<HEADDIM;z++){"
        "float zq=RQKVQ(h*HEADDIM+z);"
        "ss+=zq*zq;}"
        "float rq=inversesqrt(ss/float(HEADDIM)+QKEPS);"
        "gl_FragColor=enc24s(min(rq,127.0),65536.0);}";
}

string att_qrot(const AttnConfig& c) {
    // Two head dims per fragment, written as a q16 pair.  The scores pass then
    // decodes two q values per fetch instead of one 24-bit texel each, which is
    // the dominant cost of that loop.  The value grid is unchanged: enc24s(q,256)
    // stored the same integer that x_st2/q16 stores.
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D R,RT,QN,QS;uniform float pos;" + qkv_read(c, "R", "RQKVQ") +
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
        "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
        "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
        "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
        "return first?qv*co-b*sn:b*sn+qv*co;}"
        "void main(){int x=int(gl_FragCoord.x);int h=x/(HEADDIM/2);int i0=(x-h*(HEADDIM/2))*2;"
        "vec4 p0=q16(qval(h,i0)),p1=q16(qval(h,i0+1));"
        "gl_FragColor=vec4(p0.r,p0.g,p1.r,p1.g);}";
}


string att_score(const AttnConfig& c, bool hoist) {
    if (!hoist) {
        return "precision highp float;\n" + att_defs(c) + att_helpers() +
            "uniform sampler2D R,KV,RT,QN;uniform float pos,invsqrt;" + qkv_read(c, "R", "RQKVQ") +
            "void main(){int t=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
            "if(float(t)>pos){gl_FragColor=enc24s(0.0,1024.0);return;}int kvh=h/GROUP;float s=0.0;"
            "float rq=1.0;"
            "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
            "float zq=RQKVQ(h*HEADDIM+z);ss+=zq*zq;}"
            "rq=inversesqrt(ss/float(HEADDIM)+QKEPS);}"
            "for(int i=0;i<HEADDIM;i++){"
            "float qv=RQKVQ(h*HEADDIM+i);"
            "int j,ti;int even=i-(i/2)*2;"
            "if(ROPEHALF==1){if(i<HALF){j=i+HALF;ti=i;}else{j=i-HALF;ti=i-HALF;}}"
            "else{j=(even==0)?i+1:i-1;ti=i/2;}"
            "float b=RQKVQ(h*HEADDIM+j);"
            "if(QKNORM==1){qv=qv*rq*dec16(texture2D(QN,vec2((float(i)+0.5)/float(HEADDIM),0.5)).rg);"
            "b=b*rq*dec16(texture2D(QN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);}"
            "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "float co=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
            "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
            "float q=first?qv*co-b*sn:b*sn+qv*co;"
            "int ki=kvh*HEADDIM+i;int tx=ki;int odd=0;"
            "if(KVPACK==16){tx=ki/2;odd=ki-tx*2;}else if(KVPACK==8){tx=ki/4;odd=ki-tx*4;}"
            "float k=decKV(texture2D(KV,vec2((float(tx)+0.5)/float(KVWIDTH),(float(t)+0.5)/float(SEQ))),odd);"
            "s+=q*k;}"
            "gl_FragColor=enc24s(s*invsqrt,1024.0);}";
    }
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D Q,KV;uniform float pos,invsqrt;"
        "vec4 qe(float v){float q=clamp(floor(v*1024.0+0.5),-32768.0,32767.0);"
        "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
        "return vec4(hi/255.0,(q-hi*256.0)/255.0,0.0,1.0);}"
        "void main(){int T=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
        "int t0=T*2;int t1=t0+1;int kvh=h/GROUP;float s0=0.0,s1=0.0;"
        "bool e0=(float(t0)<=pos);bool e1=(float(t1)<=pos);"
        "if(!e0&&!e1){gl_FragColor=vec4(0.0,0.0,0.0,0.0);return;}"
        "for(int i=0;i<HEADDIM;i++){"
        "vec4 qp=texture2D(Q,vec2((float((h*HEADDIM+i)/2)+0.5)/float(DIM/2),0.5));"
        "float q=(((h*HEADDIM+i)-(h*HEADDIM+i)/2*2)==0)?dec16(qp.rg):dec16(qp.ba);"
        "int ki=kvh*HEADDIM+i;int tx=ki;int odd=0;"
        "if(KVPACK==16){tx=ki/2;odd=ki-tx*2;}else if(KVPACK==8){tx=ki/4;odd=ki-tx*4;}"
        "if(e0)s0+=q*decKV(texture2D(KV,vec2((float(tx)+0.5)/float(KVWIDTH),(float(t0)+0.5)/float(SEQ))),odd);"
        "if(e1)s1+=q*decKV(texture2D(KV,vec2((float(tx)+0.5)/float(KVWIDTH),(float(t1)+0.5)/float(SEQ))),odd);}"
        "vec4 p0=qe(e0?s0*invsqrt:0.0),p1=qe(e1?s1*invsqrt:0.0);"
        "gl_FragColor=vec4(p0.r,p0.g,p1.r,p1.g);}";
}

string att_stats(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D S;uniform float pos;"
        // One fragment per head: (max, sum) packed into a single texel as two q16
        // values.  Previously two fragments per head did the identical scan and
        // only differed in which of the two statistics they wrote.
        "vec2 qs(float v,float s){float q=clamp(floor(v*s+0.5),-32768.0,32767.0);"
        "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
        "return vec2(hi/255.0,(q-hi*256.0)/255.0);}"
        "void main(){int h=int(gl_FragCoord.x);float mx=-1e30;"
        "for(int T=0;T<SEQ/2;T++){int t0=T*2;if(float(t0)>pos)break;"
        "vec4 c=texture2D(S,vec2((float(T)+0.5)/float(SEQ/2),(float(h)+0.5)/float(HEADS)));"
        "mx=max(mx,dec16(c.rg)/4.0);if(float(t0+1)<=pos)mx=max(mx,dec16(c.ba)/4.0);}"
        "float sum=0.0;"
        "for(int T=0;T<SEQ/2;T++){int t0=T*2;if(float(t0)>pos)break;"
        "vec4 c=texture2D(S,vec2((float(T)+0.5)/float(SEQ/2),(float(h)+0.5)/float(HEADS)));"
        "sum+=exp(dec16(c.rg)/4.0-mx);if(float(t0+1)<=pos)sum+=exp(dec16(c.ba)/4.0-mx);}"
        "vec2 pm=qs(mx,1024.0),ps=qs(sum,32.0);"
        "gl_FragColor=vec4(pm.x,pm.y,ps.x,ps.y);}";
}

string att_prob(const AttnConfig& c) {
    return "precision highp float;\n" + att_defs(c) + att_helpers() +
        "uniform sampler2D S,ST;uniform float pos;"
        "void main(){int t=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
        "vec4 sm=texture2D(ST,vec2((float(2*h)+0.5)/float(2*HEADS),0.5));"
        "vec4 ss=texture2D(ST,vec2((float(2*h+1)+0.5)/float(2*HEADS),0.5));"
        "float mx=dec24s(sm,1024.0);float sum=dec24s(ss,1024.0);float p=0.0;"
        "if(float(t)<=pos && sum>0.0)p=exp(dec24s(texture2D(S,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),1024.0)-mx)/sum;"
        "gl_FragColor=enc24s(p,65536.0);}";
}

string att_v(const AttnConfig& c, XFormat out) {
    // q16 score/stats layout has one packed texel per head; fused decode is mandatory.
    static const bool fuse_prob = true;
    const string prob_bind = fuse_prob ? "S,ST,KV,VS" : "P,KV,VS";
    const string prob_body = fuse_prob
        ? string("vec4 st=texture2D(ST,vec2((float(h)+0.5)/float(HEADS),0.5));"
                 "float mx=dec16(st.rg)/4.0;float sum=dec16(st.ba)*8.0;"
                 "float invs=(sum>0.0)?(1.0/sum):0.0;")
        : string("");
    const string prob_read = fuse_prob
        ? string("vec4 sp=texture2D(S,vec2((float(t/2)+0.5)/float(SEQ/2),(float(h)+0.5)/float(HEADS)));"
                 "float sc=((t-(t/2)*2)==0)?dec16(sp.rg)/4.0:dec16(sp.ba)/4.0;"
                 "float p=exp(sc-mx)*invs;")
        : string("float p=dec24s(texture2D(P,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),65536.0);");

    // Two (or four) output channels per fragment share one probability decode and
    // one pass over the KV rows.  HEADDIM is even, so a fragment never straddles a
    // head boundary and a single head index is enough.
    const int N = (out == XFormat::I8_4) ? 4 : 2;
    // The V index depends only on the head/dim, not on t, so it is computed once
    // outside the loop.  The loop is unrolled by two: one score texel already
    // holds the pair (t, t+1), so two positions cost one score fetch.
    string heads_idx;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        heads_idx += "int ci" + K + "=i0+" + K + ";int vi" + K + "=kvh*HEADDIM+ci" + K + ";"
                     "int tx" + K + "=KVDIM+vi" + K + ";int od" + K + "=0;"
                     "if(KVPACK==16){tx" + K + "=KVDIM/2+vi" + K + "/2;od" + K + "=vi" + K + "-(vi" + K + "/2)*2;}"
                     "else if(KVPACK==8){tx" + K + "=KVDIM/4+vi" + K + "/4;od" + K + "=vi" + K + "-(vi" + K + "/4)*4;}";
    }
    string bodies, accs;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        bodies += "a" + K + "+=p0*v0_" + K + ";";
        accs += "a" + K + (k + 1 < N ? "," : "");
    }
    // 8-bit V carries a per-head, per-position step: one fetch per position,
    // shared by every channel of the fragment (they all sit in one KV head).
    const bool vstep = (c.kv_bits == 8 && c.kv_vstep);
    const string step0 = "float d0=dec24s(texture2D(VS,vec2((float(kvh)+0.5)/float(KVHEADS),(float(t0)+0.5)/float(SEQ))),16777216.0);";
    const string step1 = "float d1=dec24s(texture2D(VS,vec2((float(kvh)+0.5)/float(KVHEADS),(float(t1)+0.5)/float(SEQ))),16777216.0);";
    string loads0, loads1;
    for (int k = 0; k < N; ++k) {
        string K = to_string(k);
        const string s0 = "texture2D(KV,vec2((float(tx" + K + ")+0.5)/float(KVWIDTH),(float(t0)+0.5)/float(SEQ)))";
        const string s1 = "texture2D(KV,vec2((float(tx" + K + ")+0.5)/float(KVWIDTH),(float(t1)+0.5)/float(SEQ)))";
        loads0 += "float v0_" + K + "=" + (vstep ? "decKV8d(" + s0 + ",od" + K + ",d0)" : "decKV(" + s0 + ",od" + K + ")") + ";";
        loads1 += "float v1_" + K + "=" + (vstep ? "decKV8d(" + s1 + ",od" + K + ",d1)" : "decKV(" + s1 + ",od" + K + ")") + ";";
    }
    string init;
    for (int k = 0; k < N; ++k) init += "float a" + to_string(k) + "=0.0;";
    string enc = (N == 4) ? "gl_FragColor=x_i8enc4(" + accs + ");}"
                          : "gl_FragColor=x_st2(" + accs + ");}";

    string fs = "precision highp float;\n" + x_glsl(out) + att_defs(c) + att_helpers() +
        "uniform sampler2D " + prob_bind + ";uniform float pos;"
        "void main(){int t=int(gl_FragCoord.x);int x0=" + to_string(N) + "*t;"
        "int h=x0/HEADDIM;int i0=x0-h*HEADDIM;int kvh=h/GROUP;" + init + heads_idx +
        prob_body +
        "int mxT=int(pos+0.5)/2+1;"
        "for(int T=0;T<SEQ/2;T++){if(T>=mxT)break;"
        "int t0=T*2;int t1=t0+1;"
        "vec4 sp=texture2D(S,vec2((float(T)+0.5)/float(SEQ/2),(float(h)+0.5)/float(HEADS)));"
        "float p0=exp(dec16(sp.rg)/4.0-mx)*invs;" + (vstep ? step0 : string("")) +
        loads0 + bodies +
        "if(float(t1)<=pos){float p1=exp(dec16(sp.ba)/4.0-mx)*invs;" + (vstep ? step1 : string("")) + loads1 +
        "a0+=p1*v1_0;" + (N > 1 ? "a1+=p1*v1_1;" : "") + (N > 2 ? "a2+=p1*v1_2;a3+=p1*v1_3;" : "") + "}}"
        + enc;
    return fs;
}


string sum_logits() {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr,mrows,invt;"
        "float dec(vec4 c){return (floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+"
        "floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0)/256.0;}"
        "void main(){float mx=-1e30;for(int i=0;i<1536;i++){if(float(i)>=mrows)break;"
        "vec4 c=texture2D(L,vec2((float(i)+0.5)/1536.0,(tr+0.5)/nr));mx=max(mx,dec(c));}"
        "float sum=0.0;for(int i=0;i<1536;i++){if(float(i)>=mrows)break;"
        "vec4 c=texture2D(L,vec2((float(i)+0.5)/1536.0,(tr+0.5)/nr));sum+=exp((dec(c)-mx)*invt);}"
        "float v=floor(sum*4096.0+0.5)+8388608.0;"
        "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
        "float b=floor(v/256.0)-r0*65536.0-g*256.0;float a=v-r0*16777216.0-g*65536.0-b*256.0;"
        "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
}

string top2_fast(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr;"
        "float dec(vec4 c){return (floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+"
        "floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0)/256.0;}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "vec4 c=texture2D(L,vec2((float(idx)+0.5)/1536.0,(tr+0.5)/nr));float v=dec(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

string top2(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr,invt;"
        "float dec(vec4 c){return (floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+"
        "floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0)/256.0;}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "vec4 c=texture2D(L,vec2((float(idx)+0.5)/1536.0,(tr+0.5)/nr));float v=dec(c);"
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
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr;"
        "float dv(float v){if(v>=32768.0)v-=65536.0;return v/256.0;}"
        "float dl(vec4 c){return dv(floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5));}"
        "float dh(vec4 c){return dv(floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5));}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "int h=idx/2;vec4 c=texture2D(L,vec2((float(h)+0.5)/768.0,(tr+0.5)/nr));"
        "float v=((h*2)==idx)?dl(c):dh(c);"
        "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
        "float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
        "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
        "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
        "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
}

string top2_packed(int rows, int chunk) {
    return "precision highp float;"
        "uniform sampler2D L;uniform float tr,nr,invt;"
        "float dv(float v){if(v>=32768.0)v-=65536.0;return v/256.0;}"
        "float dl(vec4 c){return dv(floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5));}"
        "float dh(vec4 c){return dv(floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5));}"
        "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*" + to_string(chunk) + ";"
        "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
        "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(rows) + ")break;"
        "int h=idx/2;vec4 c=texture2D(L,vec2((float(h)+0.5)/768.0,(tr+0.5)/nr));"
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
