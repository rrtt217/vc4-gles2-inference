// Minimal VC4 split-FFN / Top2 / q16 probe.
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
    if(!ok){char log[4096];glGetShaderInfoLog(s,sizeof(log),nullptr,log);printf("SHADER: %s\n",log);} return s;
}
static GLuint program(const char* fs){
    const char* vs="attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p=glCreateProgram();glAttachShader(p,compile(GL_VERTEX_SHADER,vs));glAttachShader(p,compile(GL_FRAGMENT_SHADER,fs));glLinkProgram(p);
    GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),nullptr,log);printf("LINK: %s\n",log);}return p;
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
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)printf("FBO bad\n");return t;
}
static void draw(GLuint p,T&t,int x=0,int y=0,int w=-1,int h=-1){
    if(w<0)w=t.w;if(h<0)h=t.h;glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);glViewport(x,y,w,h);glUseProgram(p);
    GLuint l=glGetAttribLocation(p,"p");glEnableVertexAttribArray(l);glVertexAttribPointer(l,2,GL_FLOAT,GL_FALSE,0,nullptr);glDrawArrays(GL_TRIANGLES,0,3);
}
static vector<unsigned char> rd(T&t,int w=-1,int h=-1){if(w<0)w=t.w;if(h<0)h=t.h;glBindFramebuffer(GL_FRAMEBUFFER,t.fbo);glViewport(0,0,t.w,t.h);vector<unsigned char>b((size_t)w*h*4);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,b.data());return b;}
static int16_t dec(const unsigned char* p){return (int16_t)((int)p[0]|((int)p[1]<<8));}
static void enc(vector<unsigned char>& d,int i,int16_t v){uint16_t u=(uint16_t)v;d[i*4]=u&255;d[i*4+1]=u>>8;d[i*4+2]=0;d[i*4+3]=255;}
static const char* DEC = "float dec(vec4 c){float u=floor(c.r*255.0+0.5)+floor(c.g*255.0+0.5)*256.0;if(u>=32768.0)u-=65536.0;return u/256.0;}";
static void init(){
    auto g=(PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");EGLDisplay d=g(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);
    EGLint a,b;eglInitialize(d,&a,&b);eglBindAPI(EGL_OPENGL_ES_API);EGLint at[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};EGLConfig c;EGLint n;eglChooseConfig(d,at,&c,1,&n);EGLint ca[]={EGL_CONTEXT_CLIENT_VERSION,2,EGL_NONE};EGLContext x=eglCreateContext(d,c,EGL_NO_CONTEXT,ca);eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,x);
    printf("RENDERER: %s\n",glGetString(GL_RENDERER));GLuint v;glGenBuffers(1,&v);glBindBuffer(GL_ARRAY_BUFFER,v);float q[]={-1,-1,3,-1,-1,3};glBufferData(GL_ARRAY_BUFFER,sizeof(q),q,GL_STATIC_DRAW);
}
int main(){
    init();
    const int N=64;
    vector<int16_t> va(N),vb(N);
    for(int i=0;i<N;i++){va[i]=(int16_t)(((i*37)%200)-100);vb[i]=(int16_t)(((i*53)%300)-150);}
    vector<unsigned char> da(N*4),db(N*4);for(int i=0;i<N;i++){enc(da,i,va[i]);enc(db,i,vb[i]);}
    T A=target(N,1,da),B=target(N,1,db),C=target(N,1);
    string sw="precision highp float;uniform sampler2D A,B;"+string(DEC)+
      "void main(){int x=int(gl_FragCoord.x);vec2 uv=vec2((float(x)+0.5)/64.0,0.5);float a=dec(texture2D(A,uv)),b=dec(texture2D(B,uv));float y=(a/(1.0+exp(-a)))*b;float q=clamp(floor(y*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;float h=floor(q/256.0),l=q-h*256.0;gl_FragColor=vec4(l/255.0,h/255.0,0.0,1.0);}";
    GLuint ps=program(sw.c_str());
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,A.tex);glActiveTexture(GL_TEXTURE1);glBindTexture(GL_TEXTURE_2D,B.tex);
    glUseProgram(ps);glUniform1i(glGetUniformLocation(ps,"A"),0);glUniform1i(glGetUniformLocation(ps,"B"),1);draw(ps,C);glFlush();glFinish();
    auto oc=rd(C);int bad=0;for(int i=0;i<N;i++){float a=va[i]/256.0f,b=vb[i]/256.0f;float y=(a/(1.0f+expf(-a)))*b;int16_t exp=(int16_t)lrintf(y*256.0f);if(dec(&oc[i*4])!=exp)bad++;}printf("SWIGLU input bad=%d/%d first_exp=%d got=%d\n",bad,N,(int)lrintf(((va[0]/256.0f)/(1+expf(-(va[0]/256.0f))))*(vb[0]/256.0f)*256.0f),(int)dec(&oc[0]));

    // 2-row sampling
    vector<unsigned char> d2(N*2*4);for(int i=0;i<N;i++){enc(d2,i,va[i]);enc(d2,N+i,vb[i]);}
    T R=target(N,2,d2),S=target(N,1);
    string sw2="precision highp float;uniform sampler2D A;"+string(DEC)+"void main(){int x=int(gl_FragCoord.x);float a=dec(texture2D(A,vec2((float(x)+0.5)/64.0,0.25)));float b=dec(texture2D(A,vec2((float(x)+0.5)/64.0,0.75)));float y=(a/(1.0+exp(-a)))*b;float q=clamp(floor(y*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;float h=floor(q/256.0),l=q-h*256.0;gl_FragColor=vec4(l/255.0,h/255.0,0.0,1.0);}";
    GLuint p2=program(sw2.c_str());glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,R.tex);glUseProgram(p2);glUniform1i(glGetUniformLocation(p2,"A"),0);draw(p2,S);glFlush();glFinish();auto o2=rd(S);bad=0;for(int i=0;i<N;i++){float a=va[i]/256.0f,b=vb[i]/256.0f;float y=(a/(1.0f+expf(-a)))*b;int16_t exp=(int16_t)lrintf(y*256.0f);if(dec(&o2[i*4])!=exp)bad++;}printf("SWIGLU 2row bad=%d/%d\n",bad,N);

    // top2 on input texture
    vector<unsigned char> dl(N*4);for(int i=0;i<N;i++)enc(dl,i,va[i]);
    T L=target(N,1,dl),O=target(8,1);
    string st="precision highp float;uniform sampler2D L;"+string(DEC)+
      "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*16;float b1=-1e30,b2=-1e30;int i1=0,i2=0;for(int k=0;k<16;k++){int idx=start+k;if(idx>=64)break;float v=dec(texture2D(L,vec2((float(idx)+0.5)/64.0,0.5)));if(v>b1){b2=b1;i2=i1;b1=v;i1=idx;}else if(v>b2){b2=v;i2=idx;}}float bv=(rank==0?b1:b2)*256.0;int bi=rank==0?i1:i2;if(bv<0.0)bv+=65536.0;float bh=floor(bv/256.0),bl=bv-bh*256.0;float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;gl_FragColor=vec4(bl/255.0,bh/255.0,il/255.0,ih/255.0);}";
    GLuint pt=program(st.c_str());glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,L.tex);glUseProgram(pt);glUniform1i(glGetUniformLocation(pt,"L"),0);draw(pt,O);glFlush();glFinish();auto ot=rd(O);
    for(int c=0;c<4;c++){int16_t v=dec(&ot[c*8]);int idx=(int)ot[c*8+2]|((int)ot[c*8+3]<<8);printf("TOP2 c%d v=%d idx=%d\n",c,(int)v,idx);}
    return 0;
}
