// Isolate VC4 Top-2: static texture vs FBO-rendered texture, exact engine packing.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace std;

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[4096];glGetShaderInfoLog(s,sizeof(log),nullptr,log);printf("SHADER: %s\n",log);exit(1);} return s;
}
static GLuint program(const char* fs){
    const char* vs="attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p=glCreateProgram();glAttachShader(p,compile(GL_VERTEX_SHADER,vs));glAttachShader(p,compile(GL_FRAGMENT_SHADER,fs));glLinkProgram(p);
    GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),nullptr,log);printf("LINK: %s\n",log);exit(1);}return p;
}
struct T { GLuint tex=0,fbo=0;int w=0,h=0; };
static T target(int w,int h,const vector<unsigned char>& data={}){
    T t;t.w=w;t.h=h;glGenTextures(1,&t.tex);glBindTexture(GL_TEXTURE_2D,t.tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,data.empty()?nullptr:data.data());
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1,&t.fbo);glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t.tex,0);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){printf("FBO bad %dx%d\n",w,h);exit(1);}return t;
}
static void draw(GLuint p,T&t,int x=0,int y=0,int w=-1,int h=-1){
    if(w<0)w=t.w;if(h<0)h=t.h;glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);glViewport(x,y,w,h);glUseProgram(p);
    GLuint l=glGetAttribLocation(p,"p");glEnableVertexAttribArray(l);glVertexAttribPointer(l,2,GL_FLOAT,GL_FALSE,0,nullptr);glDrawArrays(GL_TRIANGLES,0,3);
}
static vector<unsigned char> rd(T&t,int w=-1,int h=-1){if(w<0)w=t.w;if(h<0)h=t.h;glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);glViewport(0,0,t.w,t.h);vector<unsigned char>b((size_t)w*h*4);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,b.data());return b;}
static int16_t q16(float v){float q=floor(v*256.0f+0.5f);if(q<-32768)q=-32768;if(q>32767)q=32767;return (int16_t)q;}
static void pack(vector<unsigned char>& d,int i,float v){int q=q16(v);unsigned u=(unsigned)(q+8388608);d[i*4]=0;d[i*4+1]=(u>>16)&255;d[i*4+2]=(u>>8)&255;d[i*4+3]=u&255;}
static int unpack(const unsigned char* p){int u=((int)p[1]<<16)|((int)p[2]<<8)|(int)p[3];return u-8388608;}
static float decq(const unsigned char* p){return unpack(p)/256.0f;}
static float dec_cand(const unsigned char* p){int u=(int)p[0]*256+(int)p[1];if(u>=32768)u-=65536;return u/256.0f;}
static const char* DEC24 = "float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}";
static void init(){
    auto g=(PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");EGLDisplay d=g(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);
    EGLint a,b;eglInitialize(d,&a,&b);eglBindAPI(EGL_OPENGL_ES_API);EGLint at[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};EGLConfig c;EGLint n;eglChooseConfig(d,at,&c,1,&n);EGLint ca[]={EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE};EGLContext x=eglCreateContext(d,c,EGL_NO_CONTEXT,ca);eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,x);
    printf("RENDERER: %s\n",glGetString(GL_RENDERER));GLuint v;glGenBuffers(1,&v);glBindBuffer(GL_ARRAY_BUFFER,v);float q[]={-1,-1,3,-1,-1,3};glBufferData(GL_ARRAY_BUFFER,sizeof(q),q,GL_STATIC_DRAW);
}
static const int N=1536;
static vector<float> vals;
static void expect(int chunk,int rank,float& ev,int& ei){
    int s=chunk*16;float b1=-1e30f,b2=-1e30f;int i1=0,i2=0;
    for(int k=0;k<16 && s+k<N;k++){float v=vals[s+k];if(v>b1){b2=b1;i2=i1;b1=v;i1=s+k;}else if(v>b2){b2=v;i2=s+k;}}
    ev=rank==0?b1:b2;ei=rank==0?i1:i2;
}
static void check(const char* tag,const vector<unsigned char>& raw){
    printf("%s raw0=%d,%d,%d,%d raw1=%d,%d,%d,%d raw2=%d,%d,%d,%d\n",tag,raw[0],raw[1],raw[2],raw[3],raw[4],raw[5],raw[6],raw[7],raw[8],raw[9],raw[10],raw[11]);
    int badv=0,badi=0;for(int c=0;c<N/16;c++)for(int r=0;r<2;r++){
        int o=(c*3+r)*4; float ev;int ei;expect(c,r,ev,ei);
        float gv=dec_cand(&raw[o]);int gi=(int)raw[o+2]*256+(int)raw[o+3];
        if(fabsf(gv-ev)>1.01f/256.0f)badv++;if(gi!=ei)badi++;
        if(c<4)printf("%s c%d r%d exp=(%.4f,%d) got=(%.4f,%d)\n",tag,c,r,ev,ei,gv,gi);
    }
    printf("%s badv=%d badi=%d\n",tag,badv,badi);
}

int main(){
    init();
    vals.resize(N);vector<unsigned char> ds(N*4);
    for(int i=0;i<N;i++){vals[i]=((i*37)%2000)/256.0f-4.0f;pack(ds,i,vals[i]);}
    T S=target(N,1,ds), L=target(N,1), C=target((N/16)*3,1);
    // copy static q16 -> L, like the logits matvec pass.
    string cp="precision highp float;uniform sampler2D S;void main(){int x=int(gl_FragCoord.x);gl_FragColor=texture2D(S,vec2((float(x)+0.5)/1536.0,0.5));}";
    GLuint pc=program(cp.c_str());glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,S.tex);glUseProgram(pc);glUniform1i(glGetUniformLocation(pc,"S"),0);draw(pc,L);glFlush();glFinish();
    auto lr=rd(L);printf("L direct first dec=%.4f exp=%.4f\n",decq(&lr[0]),vals[0]);
    int badL=0;for(int i=0;i<N;i++)if(unpack(&lr[i*4])!=q16(vals[i]))badL++;printf("L readback bad=%d/%d\n",badL,N);

    string top2="precision highp float;uniform sampler2D L;uniform float tr,nr,invt;"+string(DEC24)+
      "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*16;"
      "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
      "for(int k=0;k<16;k++){int idx=start+k;if(idx>=1536)break;"
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
    GLuint pt=program(top2.c_str());
    glUseProgram(pt);glUniform1i(glGetUniformLocation(pt,"L"),0);glUniform1f(glGetUniformLocation(pt,"tr"),0);glUniform1f(glGetUniformLocation(pt,"nr"),1);glUniform1f(glGetUniformLocation(pt,"invt"),1.25f);
    // from FBO-rendered L
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,L.tex);draw(pt,C);glFlush();glFinish();
    auto cr=rd(C);check("FBO",cr);
    // from static S
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,S.tex);draw(pt,C);glFlush();glFinish();
    auto cs=rd(C);check("STATIC",cs);
    // stats chunk0
    int o=(0*3+2)*4;float sm=(float)((int)cs[o+2]*256+(int)cs[o+3])/2048.0f;float mx=dec_cand(&cs[o]);
    float es=0;for(int k=0;k<16;k++)es+=expf((vals[k]-vals[5])*1.25f);printf("stats exp max=%.4f sum=%.4f got max=%.4f sum=%.4f\n",vals[5],es,mx,sm);
    return 0;
}
