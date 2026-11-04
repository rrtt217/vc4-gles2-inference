// Synthetic INT4 kernels, no model: the plain matvec (matvec_i4) at 64x64 and
// the fused FFN gate (matvec_i4_pair_swiglu) at 304-2432 rows x 768 cols,
// timed and hashed.  The row sweep is the instrument behind WORKLOG 8b: one
// 2432-row draw against the two 1216-row draws the engine chunk split issues,
// in both the pipelined and the one-sync-per-draw regime.  Q4_MV4 selects the
// vec4 tap/dot form; at 2432x768 the two forms must print the same hash.
// It started as the test for WORKLOG item 0 (Q4_EXACTFAST): with the integer
// unpack the gate ran 0.24 ms instead of 11.92 ms with 16/32 bytes different,
// i.e. the vc4 backend dropped the loop and the result was wrong.  The engine
// no longer has that path; the probe stays as a kernel-level timing and
// determinism check and as the evidence for that result.
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
    // 2. fused FFN gate: throughput (20 pipelined draws) and latency (one
    // glFinish per draw, the engine regime) against fragment count.  The
    // engine issues two chunk draws of 1216 rows per layer; 2432 rows is the
    // same work in one draw, and the probe reaches it directly.
    {
        const int cols = 768;
        const int sizes[] = {304, 608, 1216, 2432};
        vector<float> x(cols);
        for (int i = 0; i < cols; ++i) x[i] = sinf(i * 0.37f);
        GLuint xt = gpu.upload_input(x);
        for (int si = 0; si < 4; ++si) {
            const int rows = sizes[si];
            Mat w1 = make_mat(rows, cols), w3 = make_mat(rows, cols);
            gpu.upload(w1);
            gpu.upload(w3);
            Target out;
            for (int i = 0; i < 2; ++i) gpu.ffn_pair_i4_swiglu_chunk(w1, w3, xt, rows, out);
            glFinish();
            const int iters = 20, frags = (rows + 1) / 2;
            Clock::time_point t0 = Clock::now();
            for (int i = 0; i < iters; ++i) gpu.ffn_pair_i4_swiglu_chunk(w1, w3, xt, rows, out);
            glFinish();
            const double thr = elapsed_ms(t0, Clock::now()) / iters;
            t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                gpu.ffn_pair_i4_swiglu_chunk(w1, w3, xt, rows, out);
                glFinish();
            }
            const double lat = elapsed_ms(t0, Clock::now()) / iters;
            vector<uint8_t> b = gpu.read_bytes(out, frags, 1);
            unsigned h = 2166136261u;
            for (size_t i = 0; i < b.size(); ++i) { h ^= b[i]; h *= 16777619u; }
            printf("GATE rows %d frags %d thr %.3f lat %.3f bytes %zu hash %08x\n",
                   rows, frags, thr, lat, b.size(), h);
            if (getenv("Q4MV_ALL"))
                for (size_t i = 0; i < b.size(); ++i) printf("D %d %zu %u\n", rows, i, b[i]);
        }
    }
    return 0;
}
