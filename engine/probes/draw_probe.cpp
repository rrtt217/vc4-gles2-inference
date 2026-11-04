// What does a draw cost on vc4 before any real work is in it?
//
// WORKLOG item 7 (prefill batching) starts with "measure the per-draw overhead
// with a null shader before committing to batching" (docs/future-optimization.md).
// This probe prices the three terms a layer is made of:
//
//   1. a null draw at 1x1, 384x1 and 1216x1 fragments: per-draw issue cost and
//      per-fragment dispatch, with and without a glFinish (the engine's fast
//      path, init_persistent -> layer_fast_finish_, already skips the per-pass
//      finish for rms/mat/att, so the no-finish number is the steady state);
//   2. a whole rmsnorm pass at dim 768: ~6 KB of data, two draws, one fragment
//      per output texel, 32 dependent partial fetches per fragment;
//   3. the q8 matvec at the shapes the model draws (the engine splits anything
//      wider than 1536), i.e. how fast weights stream;
//   4. the CPU cost of rebuilding a shader string, which x_prog() takes by value
//      and therefore rehashes on every draw.
//
// Measured on the Pi Zero (VC4 V3D 2.1, Mesa 26.2.2), stable across runs:
//
//   NULL    1x1   0.008-0.014 ms/draw   issue+finish 0.39-0.40 ms/draw
//   NULL  384x1   0.011-0.013 ms/draw   issue+finish 0.41-0.43 ms/draw
//   NULL 1216x1   0.018-0.019 ms/draw   issue+finish 0.49-0.50 ms/draw
//   RMS  dim=768  1.80-2.17 ms/call (2 draws, 6 KB; split reduce no better)
//   MV   768x768  2.05 ms/call   0.56 MB  274 MB/s
//   MV  1216x768  3.18 ms/call   0.89 MB  280 MB/s
//   MV  1536x768  4.13 ms/call   1.12 MB  272 MB/s
//   MV  2048x768  1.94 ms/call   1.50 MB  773 MB/s  <-- WRONG, see below
//   CPU shader text: matvec_q8 0.013-0.015 ms, rms_norm 0.042-0.051 ms
//
// Three conclusions:
//
//   * A draw costs ~10 us, so "fewer draws" is not a lever: batching positions
//     cannot pay for itself by removing draw overhead, only by making one weight
//     fetch feed several positions.
//   * Weights stream at ~275 MB/s and the model re-reads all 66 MB per token.
//     66 MB / 275 MB/s = 240 ms/token, i.e. prefill (~260) and decode (~183-244)
//     are both at that streaming limit.  Splitting the weight read across B
//     positions is therefore the whole prefill win.
//   * A 2048-fragment-wide draw silently produces wrong bytes on vc4 (3x "faster"
//     than linear, 5030/8192 bytes different from two half-width draws).  Keep
//     every draw at <=1536 wide; the halves check below is what catches this,
//     the same failure q4mv_probe found in the register allocator.  1536 is what
//     the engine already uses (ffn_chunk 1216, logits tile 1536), so the engine
//     itself is clean - but batching must not push a packed draw over it.
//
// build: make -C engine probes/draw_probe_pi CROSS_SYSROOT=/home/ubuntu/raspi-root4
// run:   ./draw_probe_pi              (RMS_SPLIT=1 switches the rmsnorm variant)
#include "gpu.h"

#include <GLES2/gl2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static GLuint compile(GLenum type, const char* src) {
    GLuint h = glCreateShader(type);
    glShaderSource(h, 1, &src, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {0};
        glGetShaderInfoLog(h, sizeof(log) - 1, nullptr, log);
        fprintf(stderr, "compile failed: %s\n", log);
        exit(1);
    }
    return h;
}

// Full-screen triangle from one attribute, so the viewport gets real fragments
// (a degenerate triangle would be culled and price nothing).
static GLuint null_program() {
    const char* vs = "attribute vec2 A;void main(){gl_Position=vec4(A,0.0,1.0);}";
    const char* fs = "precision mediump float;void main(){gl_FragColor=vec4(1.0,0.5,0.25,1.0);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glBindAttribLocation(p, 0, "A");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024] = {0};
        glGetProgramInfoLog(p, sizeof(log) - 1, nullptr, log);
        fprintf(stderr, "link failed: %s\n", log);
        exit(1);
    }
    return p;
}

// 384x1 RGBA8, two q16 weights per texel (hi byte first), the layout
// rms_norm() reads as W.
static GLuint rms_weight_texture(int dim) {
    const int tw = dim / 2;
    vector<uint8_t> px((size_t)tw * 4);
    for (int t = 0; t < tw; ++t) {
        for (int k = 0; k < 2; ++k) {
            int q = (int)lrintf(256.0f);  // weight 1.0 in q16
            if (q < 0) q += 65536;
            px[(size_t)t * 4 + k * 2 + 0] = (uint8_t)(q >> 8);
            px[(size_t)t * 4 + k * 2 + 1] = (uint8_t)(q & 255);
        }
    }
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return t;
}

static Mat make_q8_mat(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.type = WeightType::INT8;
    m.q8.resize((size_t)rows * cols);
    unsigned r = 12345u;
    for (size_t i = 0; i < m.q8.size(); ++i) {
        r = r * 1103515245u + 12345u;
        m.q8[i] = (int8_t)((r >> 16) & 255);
    }
    m.scale = 1.0f / 128.0f;
    return m;
}

int main() {
    setenv("KEEP_CPU_WEIGHTS", "1", 1);  // upload() frees q8 by default; the halves check re-slices it
    GPU gpu;
    printf("renderer %s\n", (const char*)glGetString(GL_RENDERER));

    // (1) null draws -------------------------------------------------------
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2048, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    GLuint np = null_program();
    const float tri[6] = {-1.f, -1.f, 3.f, -1.f, -1.f, 3.f};
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glUseProgram(np);
    const int n = 300;
    const int vps[3] = {1, 384, 1216};
    for (int s = 0; s < 3; ++s) {
        glViewport(0, 0, vps[s], 1);
        for (int i = 0; i < 20; ++i) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        Clock::time_point a = Clock::now();
        for (int i = 0; i < n; ++i) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double issue = elapsed_ms(a, Clock::now()) / n;
        a = Clock::now();
        for (int i = 0; i < n; ++i) { glDrawArrays(GL_TRIANGLES, 0, 3); glFinish(); }
        const double sync = elapsed_ms(a, Clock::now()) / n;
        printf("NULL %4dx1 %.4f ms/draw  issue+finish %.4f ms/draw\n", vps[s], issue, sync);
    }
    gpu.forget_state();

    // (2) rmsnorm ----------------------------------------------------------
    const int dim = 768, layers = 8;
    vector<float> rms_att((size_t)dim * layers, 1.0f), rms_ffn((size_t)dim * layers, 1.0f);
    vector<float> rms_final(dim, 1.0f);
    vector<float> qn((size_t)dim), kn((size_t)dim / 3);
    vector<float> rc((size_t)768 * 32), rs((size_t)768 * 32);
    for (size_t i = 0; i < qn.size(); ++i) qn[i] = 1.0f;
    for (size_t i = 0; i < kn.size(); ++i) kn[i] = 1.0f;
    for (size_t i = 0; i < rc.size(); ++i) { rc[i] = cosf(0.01f * (float)i); rs[i] = sinf(0.01f * (float)i); }
    gpu.init_attention(dim, 256, 12, 4, 64, 768, true, true, layers, qn, kn, rc, rs, true);
    gpu.init_persistent(dim, layers, rms_att, rms_ffn, rms_final);
    GLuint wtex = rms_weight_texture(dim);
    gpu.forget_state();

    vector<float> h(dim);
    for (int i = 0; i < dim; ++i) h[i] = sinf(i * 0.37f);
    GLuint hx = gpu.upload_hidden(h);
    Target nb;
    for (int i = 0; i < 3; ++i) gpu.rmsnorm(hx, wtex, nb);
    glFinish();
    const int nr = 200;
    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < nr; ++i) gpu.rmsnorm(hx, wtex, nb);
    glFinish();
    const double rms_ms = elapsed_ms(t0, Clock::now()) / nr;
    printf("RMS  dim=%d %.4f ms/call (2 draws, %s reduce)\n", dim, rms_ms,
           getenv("RMS_SPLIT") ? "split" : "fused");
    GLuint xb = gpu.rmsnorm(hx, wtex, nb);

    // (3) q8 matvec at the shapes the model draws -------------------------
    // Wide draws are checked against two half-width draws of the same weights:
    // a row's arithmetic does not depend on the row count, so the bytes must
    // match exactly.  A silently broken draw shows up both as a time that does
    // not scale with rows and as a byte diff here.
    static const int shapes[4][2] = {{768, 768}, {1216, 768}, {1536, 768}, {2048, 768}};
    for (int s = 0; s < 4; ++s) {
        const int rows = shapes[s][0], cols = shapes[s][1];
        Mat m = make_q8_mat(rows, cols);
        gpu.upload(m);
        Target out;
        const int iters = 20;
        for (int i = 0; i < 3; ++i) gpu.matvec_to(m, xb, out);
        glFinish();
        t0 = Clock::now();
        for (int i = 0; i < iters; ++i) gpu.matvec_to(m, xb, out);
        glFinish();
        const double ms = elapsed_ms(t0, Clock::now()) / iters;
        const double mb = (double)rows * cols / 1048576.0;
        const int half = rows / 2;
        Mat m0, m1;
        m0.rows = half; m0.cols = cols; m0.type = WeightType::INT8; m0.scale = m.scale;
        m1.rows = rows - half; m1.cols = cols; m1.type = WeightType::INT8; m1.scale = m.scale;
        m0.q8.assign(m.q8.begin(), m.q8.begin() + (size_t)half * cols);
        m1.q8.assign(m.q8.begin() + (size_t)half * cols, m.q8.end());
        gpu.upload(m0);
        gpu.upload(m1);
        Target o0, o1;
        gpu.matvec_to(m0, xb, o0);
        gpu.matvec_to(m1, xb, o1);
        glFinish();
        vector<uint8_t> b = gpu.read_bytes(out, rows, 1);
        vector<uint8_t> b0 = gpu.read_bytes(o0, half, 1);
        vector<uint8_t> b1 = gpu.read_bytes(o1, rows - half, 1);
        int diff = 0;
        for (size_t i = 0; i < b0.size(); ++i) if (b[i] != b0[i]) ++diff;
        for (size_t i = 0; i < b1.size(); ++i) if (b[half * 4 + i] != b1[i]) ++diff;
        printf("MV   %dx%d %.3f ms/call  %.2f MB  %.0f MB/s  halves %s (diff %d/%zu)\n",
               rows, cols, ms, mb, mb / (ms / 1000.0), diff ? "DIFFER" : "identical",
               diff, b0.size() + b1.size());
    }

    // (4) CPU side: x_prog() takes the shader text by value, so every draw
    // rebuilds and rehashes it before the cache lookup.
    {
        const int iters = 200;
        Clock::time_point a = Clock::now();
        size_t sz = 0;
        for (int i = 0; i < iters; ++i) sz += matvec_q8(1536, 768, XFormat::I16_2).size();
        const double mv1536 = elapsed_ms(a, Clock::now()) / iters;
        a = Clock::now();
        for (int i = 0; i < iters; ++i) sz += matvec_q8(768, 768, XFormat::I16_2).size();
        const double mv768 = elapsed_ms(a, Clock::now()) / iters;
        a = Clock::now();
        for (int i = 0; i < iters; ++i) sz += rms_norm(768, XFormat::I16_1, XFormat::I16_2, true).size();
        const double rmsb = elapsed_ms(a, Clock::now()) / iters;
        printf("CPU  matvec_q8 1536x768 %.4f ms/draw (chars %zu)\n", mv1536, sz);
        printf("CPU  matvec_q8  768x768 %.4f ms/draw\n", mv768);
        printf("CPU  rms_norm           %.4f ms/draw\n", rmsb);
    }
    return 0;
}
