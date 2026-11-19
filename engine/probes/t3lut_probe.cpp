// Can the T3 dot be computed WITHOUT decoding each weight -- T-MAC's way?
//
// T-MAC precomputes, for each group of 4 activations, a 256-entry table indexed
// by the 8-bit weight code (4 weights x 2 bits): LUT[g][c] = sum_k w_k(c)*a_{4g+k}.
// The matvec then fetches the weight byte and the table entry and adds -- no
// per-weight decode.  On CPU the table lives in registers/L1 (that is why T-MAC
// is CPU-only).  On VC4 it must be a TEXTURE, so the question is whether the
// extra random-access LUT fetch costs more than the decode it removes.
//
// The fetch arithmetic: the packed t3 format is 1 weight fetch per 16 weights;
// the LUT form is 1 weight fetch + 4 LUT fetches per 16 weights = 5x the fetch
// traffic.  This probe prices both on the board, row-major layout (1 row x 16
// weights a texel) so the comparison is clean:
//
//   decode   fetch the weight texel, decode 16 weights, dot (the t3u body)
//   lut      fetch the weight texel, fetch LUT[g][byte] per group, add
//
// The LUT is built on the HOST here (fixed activations) so the draw timing is
// the matvec alone; a model would pay a GPU build pass per layer (activations
// change per token), noted separately.  Both arms check every row against a CPU
// dot over the same weights.
//
//   build:  make -C engine probes/t3lut_probe[_pi]
//   run:    ./t3lut_probe [--rows N] [--cols M] [--iters K]
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
static float trit(int code) { return code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int rows = 1024, cols = 1024, iters = 30;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--rows N] [--cols M] [--iters K]\n", argv[0]); return 2; }
    }
    if (cols % 16) { printf("cols must be a multiple of 16\n"); return 2; }
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\nversion  %s\n", (const char*) glGetString(GL_RENDERER),
           (const char*) glGetString(GL_VERSION));

    const int nq = cols / 16;          // weight texels a row (16 weights each)
    const int ng = cols / 4;           // activation groups a row (4 weights each)

    vector<float> xval((size_t) cols);
    vector<unsigned char> xtex((size_t) cols, 0);
    for (int i = 0; i < cols; i++) { xval[i] = (float) ((i % 5) - 2); xtex[i] = (unsigned char) ((int) xval[i] + 128); }
    GLuint xid = 0;
    glGenTextures(1, &xid);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, xid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cols / 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, xtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Row-major weights: texel (q, r) = 16 weights of row r, byte ch = weights
    // 16q+4ch .. 16q+4ch+3, bit0 magnitude bit1 sign.
    vector<unsigned char> wtex((size_t) nq * rows * 4, 0);
    vector<float> wref((size_t) rows * cols, 0.0f);
    unsigned seed = 12345;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
    for (int r = 0; r < rows; r++)
        for (int q = 0; q < nq; q++)
            for (int ch = 0; ch < 4; ch++) {
                unsigned acc = 0;
                for (int k = 0; k < 4; k++) {
                    const int col = 16 * q + 4 * ch + k;
                    const int code = (int) (rnd() % 3);
                    wref[(size_t) r * cols + col] = trit(code);
                    // bit0 magnitude, bit1 sign: code 0->00, 1->01, 2->11 (the
                    // signbit decode and the LUT's trit((c>>2k)&3) both read this).
                    acc |= ((code == 0 ? 0u : 1u) | (code == 2 ? 2u : 0u)) << (2 * k);
                }
                wtex[((size_t) (r * nq + q) * 4) + ch] = (unsigned char) acc;
            }
    GLuint wid = 0;
    glGenTextures(1, &wid);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, wid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nq, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, wtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // The LUT: texture (256 codes x ng groups), entry = q16 partial sum in .rg.
    // LUT[g][c] = sum_k trit((c>>2k)&3) * xval[4g+k].  Built on the host.
    vector<unsigned char> ltex((size_t) 256 * ng * 4, 0);
    for (int g = 0; g < ng; g++)
        for (int c = 0; c < 256; c++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += trit((c >> (2 * k)) & 3) * xval[(size_t) 4 * g + k];
            float qf = std::max(-32768.0f, std::min(32767.0f, std::floor(s * 256.0f + 0.5f)));
            if (qf < 0) qf += 65536.0f;
            unsigned u = (unsigned) qf;
            unsigned char* o = &ltex[((size_t) g * 256 + c) * 4];
            o[0] = (unsigned char) (u >> 8); o[1] = (unsigned char) (u & 0xff);
        }
    GLuint lid = 0;
    glGenTextures(1, &lid);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, lid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, ng, 0, GL_RGBA, GL_UNSIGNED_BYTE, ltex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rows, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    auto q24_at = [&](const vector<unsigned char>& px, int r) {
        return ((double) px[(size_t) r * 4] * 65536.0 + (double) px[(size_t) r * 4 + 1] * 256.0 +
                px[(size_t) r * 4 + 2] - 8388608.0) / 256.0;
    };
    const string store =
        "float q=clamp(floor(s*256.0+0.5),-8388608.0,8388607.0)+8388608.0;"
        "float gg=floor(q/65536.0),b=floor(q/256.0)-gg*256.0,a=q-gg*65536.0-b*256.0;"
        "frag_color=vec4(gg/255.0,b/255.0,a/255.0,1.0);}\n";

    struct Arm { const char* name; bool lut; };
    const Arm arms[] = {{"decode (16 a texel)", false}, {"lut (T-MAC texture)", true}};
    printf("\n%-22s %9s %10s %8s\n", "arm", "ms/draw", "ns/MAC", "check");
    for (const Arm& a : arms) {
        const string ver = "#version 300 es\n";
        string vs = ver + "in vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver + "precision highp float;\nprecision highp int;\nout vec4 frag_color;\n"
                    "uniform sampler2D W,X,L;\n";
        if (!a.lut) {
            fs += "vec4 dec(float bb){uint u=uint(bb*255.0+0.5);return vec4("
                  "uintBitsToFloat(0x3F800000u|(((u>>1)&1u)<<31))*float(u&1u),"
                  "uintBitsToFloat(0x3F800000u|(((u>>3)&1u)<<31))*float((u>>2)&1u),"
                  "uintBitsToFloat(0x3F800000u|(((u>>5)&1u)<<31))*float((u>>4)&1u),"
                  "uintBitsToFloat(0x3F800000u|(((u>>7)&1u)<<31))*float((u>>6)&1u));}\n";
            fs += "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
                  "for(int q=0;q<" + std::to_string(nq) + ";q++){"
                  "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + std::to_string(nq) +
                  "),(float(r)+0.5)/float(" + std::to_string(rows) + ")));"
                  "vec4 x0=texture2D(X,vec2((float(q*4)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 x1=texture2D(X,vec2((float(q*4+1)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 x2=texture2D(X,vec2((float(q*4+2)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "vec4 x3=texture2D(X,vec2((float(q*4+3)+0.5)/float(" + std::to_string(cols / 4) + "),0.5))*255.0-128.0;"
                  "s+=dot(dec(c.r),x0)+dot(dec(c.g),x1)+dot(dec(c.b),x2)+dot(dec(c.a),x3);}"
                  + store;
        } else {
            // q16 decode of the LUT entry (.rg), the cheap dec16 form.
            fs += "float dlut(vec4 c){float u=c.r*65280.0+c.g*255.0;if(u>=32768.0)u-=65536.0;return u/256.0;}\n";
            fs += "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
                  "for(int q=0;q<" + std::to_string(nq) + ";q++){"
                  "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + std::to_string(nq) +
                  "),(float(r)+0.5)/float(" + std::to_string(rows) + ")));"
                  "float g0=float(q*4),g1=float(q*4+1),g2=float(q*4+2),g3=float(q*4+3);"
                  "float y0=(g0+0.5)/float(" + std::to_string(ng) + "),y1=(g1+0.5)/float(" + std::to_string(ng) +
                  "),y2=(g2+0.5)/float(" + std::to_string(ng) + "),y3=(g3+0.5)/float(" + std::to_string(ng) + ");"
                  "s+=dlut(texture2D(L,vec2((floor(c.r*255.0+0.5)+0.5)/256.0,y0)));"
                  "s+=dlut(texture2D(L,vec2((floor(c.g*255.0+0.5)+0.5)/256.0,y1)));"
                  "s+=dlut(texture2D(L,vec2((floor(c.b*255.0+0.5)+0.5)/256.0,y2)));"
                  "s+=dlut(texture2D(L,vec2((floor(c.a*255.0+0.5)+0.5)/256.0,y3)));}"
                  + store;
        }
        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) { printf("%-22s COMPILE FAIL %s\n", a.name, log.c_str()); continue; }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs); glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) { char b[4096]; glGetProgramInfoLog(p, sizeof b, nullptr, b); printf("%-22s LINK FAIL %s\n", a.name, b); continue; }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, xid);
        glUniform1i(glGetUniformLocation(p, "X"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, wid);
        glUniform1i(glGetUniformLocation(p, "W"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, lid);
        glUniform1i(glGetUniformLocation(p, "L"), 2);
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, rows, 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3); glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;
        vector<unsigned char> px((size_t) rows * 4, 0);
        glReadPixels(0, 0, rows, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int bad = 0; double worst = 0;
        for (int r = 0; r < rows; r++) {
            const double got = q24_at(px, r), ref = dot_row(r);
            const double e = std::fabs(got - ref) / (1.0 + std::fabs(ref));
            if (e > 2e-3) { if (bad < 3) printf("    row %d got %.4f want %.4f\n", r, got, ref); ++bad; }
            if (e > worst) worst = e;
        }
        printf("%-22s %9.3f %10.2f %8s\n", a.name, ms, 1e6 * ms / ((double) rows * cols),
               bad ? "MISMATCH" : "ok");
        if (bad) printf("    %d/%d rows off (worst %.3g)\n", bad, rows, worst);
        glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
    }
    return 0;
}