// Does the ACTIVATION format move the T3 matvec -- I8_4 (ACT=i8) vs I16_2?
//
// The t3 matvec is weight-decode-bound, but it still fetches and decodes the
// activations every iteration.  I16_2 (the default q16 pair) is 2 values a
// texel; I8_4 is 4, so I8_4 halves the activation fetches and decodes a byte
// (c*255-128) instead of a q16 pair (dec16).  This probe holds the weight layout
// (column-major 2-row), the signbit weight decode and the accumulation fixed,
// and varies ONLY the activation format, so the difference is the activation
// fetch+decode price.  The pair kernel (2 rows a fragment, the FFN/wo/w2 shape):
//
//   act8    I8_4 activation, 2 fetches + byte decode per 8 activations
//   act16   I16_2 activation, 4 fetches + dec16 per 8 activations
//
// ACT=i8 is currently gated to all-q8 models (session.cpp); this prices what
// enabling it for t3 would buy, before any quality (PPL) consideration.
//
//   build:  make -C engine probes/t3act_probe[_pi]
//   run:    ./t3act_probe [--rows N] [--cols M] [--iters K]
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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rows = 1024, cols = 1024, iters = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--rows N] [--cols M] [--iters K]\n", argv[0]); return 2; }
    }
    if (cols % 8) { printf("cols must be a multiple of 8\n"); return 2; }
    rows += rows & 1;
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\nversion  %s\n", (const char*) glGetString(GL_RENDERER),
           (const char*) glGetString(GL_VERSION));

    const int pnq = cols / 8, prows = rows / 2;
    vector<float> xval((size_t) cols);
    for (int i = 0; i < cols; i++) xval[i] = (float) ((i % 5) - 2);

    // I8_4 activation texture: cols/4 texels, byte = value+128.
    vector<unsigned char> x8((size_t) cols, 0);
    for (int i = 0; i < cols; i++) x8[i] = (unsigned char) ((int) xval[i] + 128);
    // I16_2 activation texture: cols/2 texels, q16 pair (.rg even, .ba odd).
    vector<unsigned char> x16((size_t) cols * 2, 0);
    for (int i = 0; i < cols; i++) {
        float qf = std::max(-32768.0f, std::min(32767.0f, std::floor(xval[i] * 256.0f + 0.5f)));
        if (qf < 0) qf += 65536.0f;
        unsigned u = (unsigned) qf;
        unsigned char* o = &x16[(size_t) i * 2];
        o[0] = (unsigned char) (u >> 8); o[1] = (unsigned char) (u & 0xff);
    }
    GLuint x8id = 0, x16id = 0;
    glGenTextures(1, &x8id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, x8id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, x8.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenTextures(1, &x16id);
    glBindTexture(GL_TEXTURE_2D, x16id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, x16.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Column-major 2-row weights (the shipped layout).
    vector<unsigned char> wtex((size_t) pnq * prows * 4, 0);
    vector<float> wref((size_t) rows * cols, 0.0f);
    unsigned seed = 12345;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
    for (int t = 0; t < prows; t++)
        for (int c = 0; c < pnq; c++)
            for (int ch = 0; ch < 4; ch++) {
                const int r = 2 * t + (ch >= 2 ? 1 : 0);
                const int base = 8 * c + (ch & 1) * 4;
                unsigned acc = 0;
                for (int k = 0; k < 4; k++) {
                    const int code = (int) (rnd() % 3);
                    wref[(size_t) r * cols + base + k] = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                    acc |= ((code == 0 ? 0u : 1u) | (code == 2 ? 2u : 0u)) << (2 * k);
                }
                wtex[((size_t) (t * pnq + c) * 4) + ch] = (unsigned char) acc;
            }
    GLuint wid = 0;
    glGenTextures(1, &wid);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, wid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pnq, prows, 0, GL_RGBA, GL_UNSIGNED_BYTE, wtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, prows, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    auto dot_row = [&](int r) {
        double w = 0.0;
        for (int i = 0; i < cols; i++) w += wref[(size_t) r * cols + i] * xval[i];
        return w;
    };
    double pair_div = 1.0;
    { double amax = 0; for (int r = 0; r < rows; r++) amax = std::max(amax, std::fabs(dot_row(r)));
      while (amax / pair_div > 120.0) pair_div *= 2.0; }

    const string dec =
        "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
        "uintBitsToFloat(0x3F800000u|(((u>>1)&1u)<<31))*float(u&1u),"
        "uintBitsToFloat(0x3F800000u|(((u>>3)&1u)<<31))*float((u>>2)&1u),"
        "uintBitsToFloat(0x3F800000u|(((u>>5)&1u)<<31))*float((u>>4)&1u),"
        "uintBitsToFloat(0x3F800000u|(((u>>7)&1u)<<31))*float((u>>6)&1u));}\n";
    const string e16 =
        "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
        "float hi=floor(q/256.0);return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n";
    // pair body: 2 rows a fragment, signbit weights, activation fetch varies.
    auto body = [&](bool act16) {
        string axw = std::to_string(act16 ? cols / 2 : cols / 4);
        string xa, xb;
        if (act16) {
            // I16_2: 4 fetches for 8 activations, dec16 each half.
            xa = "vec4 t0=texture2D(X,vec2((float(4*c)+0.5)/float(" + axw + "),0.5));"
                 "vec4 t1=texture2D(X,vec2((float(4*c+1)+0.5)/float(" + axw + "),0.5));"
                 "vec4 xa=vec4(dec16(t0.rg),dec16(t0.ba),dec16(t1.rg),dec16(t1.ba));";
            xb = "vec4 t2=texture2D(X,vec2((float(4*c+2)+0.5)/float(" + axw + "),0.5));"
                 "vec4 t3=texture2D(X,vec2((float(4*c+3)+0.5)/float(" + axw + "),0.5));"
                 "vec4 xb=vec4(dec16(t2.rg),dec16(t2.ba),dec16(t3.rg),dec16(t3.ba));";
        } else {
            xa = "vec4 xa=texture2D(X,vec2((float(2*c)+0.5)/float(" + axw + "),0.5))*255.0-128.0;";
            xb = "vec4 xb=texture2D(X,vec2((float(2*c+1)+0.5)/float(" + axw + "),0.5))*255.0-128.0;";
        }
        return "void main(){int t=int(gl_FragCoord.x);float s0=0.0,s1=0.0;"
               "for(int c=0;c<" + std::to_string(pnq) + ";c++){"
               "vec4 w=texture2D(W,vec2((float(c)+0.5)/float(" + std::to_string(pnq) +
               "),(float(t)+0.5)/float(" + std::to_string(prows) + ")));" + xa + xb +
               "s0+=dot(dec(w.r),xa)+dot(dec(w.g),xb);"
               "s1+=dot(dec(w.b),xa)+dot(dec(w.a),xb);}"
               "vec2 p0=e16(s0*" + std::to_string(1.0 / pair_div) + "*256.0),p1=e16(s1*" +
               std::to_string(1.0 / pair_div) + "*256.0);"
               "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
    };

    struct Arm { const char* name; bool act16; GLuint xid; };
    const Arm arms[] = {{"act8 (I8_4)", false, x8id}, {"act16 (I16_2)", true, x16id}};
    printf("\n%-18s %9s %10s %8s\n", "arm", "ms/draw", "ns/MAC", "check");
    for (const Arm& a : arms) {
        const string ver = "#version 300 es\n";
        string vs = ver + "in vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver + "precision highp float;\nprecision highp int;\nout vec4 frag_color;\n"
                    "uniform sampler2D W,X;\n" + dec + e16 +
                    "float dec16(vec2 c){float u=c.x*65280.0+c.y*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
                    + body(a.act16);
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
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, a.xid);
        glUniform1i(glGetUniformLocation(p, "X"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, wid);
        glUniform1i(glGetUniformLocation(p, "W"), 1);
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, prows, 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        vector<unsigned char> px((size_t) prows * 4, 0);
        glReadPixels(0, 0, prows, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int bad = 0; double worst = 0;
        for (int r = 0; r < rows; r++) {
            const size_t o = (size_t) (r / 2) * 4 + (size_t) (r & 1) * 2;
            double u = ((double) px[o] * 65280.0 + (double) px[o + 1] * 255.0) / 255.0;
            if (u >= 32768.0) u -= 65536.0;
            const double got = u / 256.0, ref = dot_row(r) / pair_div;
            const double e = std::fabs(got - ref) / (1.0 + std::fabs(ref));
            if (e > 2e-3) { if (bad < 3) printf("    row %d got %.4f want %.4f\n", r, got, ref); ++bad; }
            if (e > worst) worst = e;
        }
        printf("%-18s %9.3f %10.2f %8s\n", a.name, ms, 1e6 * ms / ((double) rows * cols),
               bad ? "MISMATCH" : "ok");
        if (bad) printf("    %d/%d rows off (worst %.3g)\n", bad, rows, worst);
        glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
    }
    return 0;
}