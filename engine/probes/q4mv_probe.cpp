// Synthetic INT4 kernels, no model: the plain matvec (matvec_i4) at 64x64 and
// the fused FFN gate (matvec_i4_pair_swiglu) at 2432x768, timed and dumped.
// Built to test WORKLOG item 0 (Q4_EXACTFAST): with the integer unpack the gate
// kernel ran 0.24 ms instead of 11.92 ms with 16/32 bytes different, i.e. the
// vc4 backend dropped the loop and the result was wrong.  The engine no longer
// has that path; the probe stays as a kernel-level timing and determinism check
// (two runs must print identical M/G lines) and as the evidence for that result.
#include "gpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace std;

static unsigned rng = 12345u;
static unsigned next() { rng = rng * 1103515245u + 12345u; return rng >> 8; }

static Mat make_mat(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.type = WeightType::INT4;
    m.group = 32;
    m.groups = (cols + m.group - 1) / m.group;
    m.q4.assign((size_t)rows * cols / 2, 0);
    for (size_t i = 0; i < m.q4.size(); ++i) m.q4[i] = (uint8_t)next();
    m.q4_scale.resize((size_t)rows * m.groups);
    m.q4_min.resize((size_t)rows * m.groups);
    for (size_t i = 0; i < m.q4_scale.size(); ++i) {
        m.q4_scale[i] = (int16_t)(next() & 0x7fff);
        m.q4_min[i] = (int16_t)(next() & 0x7fff);
    }
    m.q4_param_scale = 1.0f / 4096.0f;
    m.scale = 1.0f;
    return m;
}

int main() {
    GPU gpu;
    // 1. plain matvec, 64x64
    {
        Mat m = make_mat(64, 64);
        gpu.upload(m);
        vector<float> x(64);
        for (int i = 0; i < 64; ++i) x[i] = sinf(i * 0.37f) * 2.0f;
        vector<float> y = gpu.matvec(m, x);
        for (int r = 0; r < 64; ++r) printf("M %d %.9g\n", r, y[r]);
    }
    // 2. fused FFN gate at the model shape, rows packed two per texel
    {
        const int rows = 2432, cols = 768;
        Mat w1 = make_mat(rows, cols), w3 = make_mat(rows, cols);
        gpu.upload(w1);
        gpu.upload(w3);
        vector<float> x(cols);
        for (int i = 0; i < cols; ++i) x[i] = sinf(i * 0.37f);
        GLuint xt = gpu.upload_input(x);
        Target out;
        for (int i = 0; i < 3; ++i) gpu.ffn_pair_i4_swiglu_chunk(w1, w3, xt, rows, out);
        glFinish();
        Clock::time_point t0 = Clock::now();
        const int iters = 20;
        for (int i = 0; i < iters; ++i) gpu.ffn_pair_i4_swiglu_chunk(w1, w3, xt, rows, out);
        glFinish();
        const double ms = elapsed_ms(t0, Clock::now()) / iters;
        vector<uint8_t> b = gpu.read_bytes(out, (rows + 1) / 2, 1);
        printf("GATE %.3f ms/iter bytes %zu\n", ms, b.size());
        for (size_t i = 0; i < b.size() && i < 32; ++i) printf("G %zu %u\n", i, b[i]);
    }
    return 0;
}
