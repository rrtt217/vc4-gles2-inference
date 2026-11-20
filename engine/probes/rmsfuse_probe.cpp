// What does the rmsnorm NORMALIZE draw cost, and what would folding it into the
// consumer matvec save?  (Lever 1 from the non-decode overhead investigation.)
//
// The VC4 rmsnorm is two draws: a 32-lane stats split, then a normalize pass of
// dim/2 fragments where EACH fragment fetches all 32 partials (the e7cc839
// fused reduce), sums them, inversesqrt, and normalizes its two output values.
// For dim=1024 that is 512 fragments x 32 partial fetches = 16384 fetches a pass.
// Folding the normalize into the matvec that reads it (qkv / w1 / w3) would
// eliminate this whole draw: the matvec fetches a pre-reduced scalar (1 fetch)
// and normalizes inline.  This probe prices the pieces at the model's dim:
//
//   stats   the 32-lane split (32 fragments, dim/32 values each)
//   reduce  the 32-way reduce to one scalar (1 fragment)
//   norm32  the normalize draw, fused 32-partial fetch (the current form)
//   norm1   the normalize draw, pre-reduced 1-scalar fetch (the split form)
//
// norm32 is the upper bound of what the fusion removes; norm1 - reduce is what
// the split form already pays.  The matvec's added inline cost (1 scalar fetch
// + 2 muls per input, amortized over its weight loop) is small beside norm32.
//
//   build:  make -C engine probes/rmsfuse_probe[_pi]
//   run:    ./rmsfuse_probe [--dim N] [--iters K]
#include "egl_surfaceless.h"

#include <GLES2/gl2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

static void request_es3() {
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
}
static GLuint compile(GLenum t, const string& src, string& log) {
    GLuint h = glCreateShader(t);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) { char b[4096]; glGetShaderInfoLog(h, sizeof b, nullptr, b); log = b; }
    return ok ? h : 0;
}
// i24 fixed-point store (step 1/4096, biased 2^23), the RMS_STORE=i24 default.
static void s_st(float s, unsigned char* o) {
    float q = std::floor(s * 4096.0f + 0.5f) + 8388608.0f;
    unsigned u = (unsigned) q;
    o[0] = (unsigned char) (u >> 24); o[1] = (unsigned char) (u >> 16);
    o[2] = (unsigned char) (u >> 8); o[3] = (unsigned char) u;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int dim = 1024, iters = 50;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--dim N] [--iters K]\n", argv[0]); return 2; }
    }
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\nversion  %s\ndim %d\n", (const char*) glGetString(GL_RENDERER),
           (const char*) glGetString(GL_VERSION), dim);

    const int RW = dim / 2;       // I16_2 residual texels
    // residual: small q16 values; weight: q16 near 1.
    vector<float> xval((size_t) dim), wval((size_t) dim);
    vector<unsigned char> xtex((size_t) RW * 4, 0), wtex((size_t) RW * 4, 0);
    auto q16enc = [](float v, unsigned char* o) {
        float q = std::max(-32768.0f, std::min(32767.0f, std::floor(v * 256.0f + 0.5f)));
        if (q < 0) q += 65536.0f;
        unsigned u = (unsigned) q;
        o[0] = (unsigned char) (u >> 8); o[1] = (unsigned char) (u & 0xff);
    };
    unsigned seed = 7;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) & 0x7fff); };
    for (int i = 0; i < dim; i++) {
        xval[i] = (float) (rnd() % 7) - 3.0f;
        wval[i] = 1.0f + (float) (rnd() % 5) * 0.1f;
        q16enc(xval[i], &xtex[(size_t) (i / 2) * 4 + (i % 2) * 2]);
        q16enc(wval[i], &wtex[(size_t) (i / 2) * 4 + (i % 2) * 2]);
    }
    // host partials: 32 lanes of sum(x^2), stored i24.
    vector<unsigned char> stex(32 * 4, 0);
    const int per = (dim + 31) / 32;
    double ss_total = 0;
    for (int p = 0; p < 32; p++) {
        float s = 0;
        for (int k = 0; k < per; k++) { int i = p * per + k; if (i < dim) { s += xval[i] * xval[i]; ss_total += xval[i] * xval[i]; } }
        s_st(s, &stex[(size_t) p * 4]);
    }
    const float inv_ref = 1.0f / std::sqrt((float) (ss_total / dim) + 1e-5f);

    auto tex = [](const vector<unsigned char>& d, int w, int h) {
        GLuint id = 0; glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        return id;
    };
    GLuint xid = tex(xtex, RW, 1), wid = tex(wtex, RW, 1), sid = tex(stex, 32, 1);
    // a 1x1 reduced-scalar texture (ss_total) for norm1
    vector<unsigned char> sc(4, 0); s_st((float) ss_total, sc.data());
    GLuint scid = tex(sc, 1, 1);

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RW, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    // shared GLSL helpers (I16_2 residual, i24 partial store)
    const string helpers =
        "uniform sampler2D X,W,S;\n"
        "float x_i16(vec2 c){float u=c.x*65280.0+c.y*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
        "float R1(int i){int t=i/2;vec4 c=texture2D(X,vec2((float(t)+0.5)/float(" + std::to_string(RW) + "),0.5));"
        "return ((i-t*2)==0)?x_i16(c.rg):x_i16(c.ba);}\n"
        "float S_LD(vec4 c){return (floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+"
        "floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0)/4096.0;}\n"
        "vec4 S_ST(float s){float q=floor(s*4096.0+0.5)+8388608.0;float r0=floor(q/16777216.0);"
        "float g=floor(q/65536.0)-r0*256.0;float b=floor(q/256.0)-r0*65536.0-g*256.0;float a=q-r0*16777216.0-g*65536.0-b*256.0;"
        "return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n"
        "vec4 x_st2(float a,float b){float qa=clamp(floor(a*256.0+0.5),-32768.0,32767.0);if(qa<0.0)qa+=65536.0;"
        "float qb=clamp(floor(b*256.0+0.5),-32768.0,32767.0);if(qb<0.0)qb+=65536.0;"
        "float ha=floor(qa/256.0),hb=floor(qb/256.0);"
        "return vec4(ha/255.0,(qa-ha*256.0)/255.0,hb/255.0,(qb-hb*256.0)/255.0);}\n";
    const string D = std::to_string(dim);
    const string ss32 = "float ss=0.0;for(int p=0;p<32;p++){ss+=S_LD(texture2D(S,vec2((float(p)+0.5)/32.0,0.5)));}";
    const string ss1 = "float ss=S_LD(texture2D(S,vec2(0.5,0.5)));";
    const string normtail =
        "float inv=inversesqrt(ss/float(" + D + ")+1e-5);"
        "vec4 wc=texture2D(W,vec2((float(t)+0.5)/float(" + std::to_string(RW) + "),0.5));"
        "float w0=x_i16(wc.rg);float w1=x_i16(wc.ba);"
        "float v0=R1(i0)*inv*w0;float v1=R1(i1)*inv*w1;"
        "frag_color=x_st2(v0,v1);}\n";

    struct Arm { const char* name; int frags; string fs; bool check; };
    const Arm arms[] = {
        {"stats (32 lane)", 32,
         "void main(){int pid=int(gl_FragCoord.x);int per=(" + D + "+31)/32;int i0=pid*per;float s=0.0;"
         "for(int k=0;k<per;k++){int i=i0+k;if(i<" + D + "){float v=R1(i);s+=v*v;}}frag_color=S_ST(s);}\n", false},
        {"reduce (1 frag)", 1,
         "void main(){" + ss32 + "frag_color=S_ST(ss);}\n", false},
        {"norm32 (fused)", RW,
         "void main(){int t=int(gl_FragCoord.x);int i0=2*t;int i1=i0+1;" + ss32 + normtail, true},
        {"norm1 (scalar)", RW,
         "void main(){int t=int(gl_FragCoord.x);int i0=2*t;int i1=i0+1;" + ss1 + normtail, true},
    };
    printf("\n%-18s %7s %9s %10s %8s\n", "arm", "frags", "ms/draw", "us/frag", "check");
    for (const Arm& a : arms) {
        const string ver = "#version 300 es\n";
        string vs = ver + "in vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver + "precision highp float;\nprecision highp int;\nout vec4 frag_color;\n" + helpers + a.fs;
        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) { printf("%-18s COMPILE FAIL %s\n", a.name, log.c_str()); continue; }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs); glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char b[4096]; glGetProgramInfoLog(p, sizeof b, nullptr, b); printf("%-18s LINK FAIL %s\n", a.name, b); continue; }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, xid);
        glUniform1i(glGetUniformLocation(p, "X"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, wid);
        glUniform1i(glGetUniformLocation(p, "W"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, a.check && !strcmp(a.name, "norm1 (scalar)") ? scid : sid);
        glUniform1i(glGetUniformLocation(p, "S"), 2);
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, a.frags, 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        const char* chk = "-";
        if (a.check) {
            vector<unsigned char> px((size_t) RW * 4, 0);
            glReadPixels(0, 0, RW, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            int bad = 0;
            for (int t = 0; t < RW; t++) {
                for (int half = 0; half < 2; half++) {
                    int i = 2 * t + half;
                    const size_t o = (size_t) t * 4 + half * 2;
                    double u = ((double) px[o] * 65280.0 + (double) px[o + 1] * 255.0) / 255.0;
                    if (u >= 32768.0) u -= 65536.0;
                    double got = u / 256.0, ref = xval[i] * inv_ref * wval[i];
                    if (std::fabs(got - ref) > 2e-2 * (1.0 + std::fabs(ref))) ++bad;
                }
            }
            chk = bad ? "MISMATCH" : "ok";
            if (bad) printf("    %d/%d values off\n", bad, dim);
        }
        printf("%-18s %7d %9.4f %10.3f %8s\n", a.name, a.frags, ms, 1000.0 * ms / a.frags, chk);
        glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
    }
    return 0;
}