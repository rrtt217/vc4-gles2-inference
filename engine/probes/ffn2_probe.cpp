// Does 2-position packing make one weight fetch serve two positions, and does
// it stay byte-identical to two single-position draws?
//
// WORKLOG item 7b prototypes the batch-prefill kernel here before any session
// plumbing: one fragment evaluates one gate row for both positions, so the W1/
// W3 texels fetched for that row are used twice (once per position) and the
// weight traffic per position halves.  VC4 writes one RGBA8 texel per
// fragment (two q16 values) and has no MRT, so two positions x two rows does
// not fit: the kernel runs one row per fragment and the output is the P2
// layout (texel k = value k of A in .rg, of B in .ba).
//
// The gate and w2 kernels are checked against the serial path by bytes, not by
// tolerance: the fetch order, the four MACs, the q16 rounding and the swiglu
// are the same operations in the same order, so any difference is a packing
// bug.  Timings are the min of three interleaved blocks on the real driver.
//
// build: make -C engine probes/ffn2_probe_pi CROSS_SYSROOT=/home/ubuntu/raspi-root4
// run:   ./ffn2_probe_pi
//
// Measured on the Pi Zero (VC4 V3D 2.1, Mesa 26.2.2), two runs, same numbers:
//
//   GATE 1216x768   q16 differ A 0/1216  B 0/1216     (bit-identical)
//   W2   768x1216   q16 differ A 0/768   B 0/768      (bit-identical)
//   gate, 2 positions    8.83 ms -> 7.16 ms   1.23x
//   w2,   2 positions    7.85 ms -> 5.86 ms   1.33x
//   one layer FFN (2 gate + 2 w2 chunks, 2 positions)  33.4 -> 26.1 ms  1.28x
//                       i.e. 16.7 -> 13.0 ms per position
//
// Two things this settles before any session plumbing:
//
//   * The packing is exact.  Both kernels reproduce the serial path byte for
//     byte, and the serial path is itself 768/768 rows within 1 LSB of a
//     double-precision CPU dot product, so the 1.28x costs no numerics.
//   * 2x was never available.  A fragment writes one RGBA8 texel = two q16,
//     so two positions x two rows (four q16) cannot be packed and the row
//     pairing the serial gate uses has to be dropped: the packed kernel does
//     the same six fetches per iteration as the serial one (2 W + 4 A instead
//     of 4 W + 2 A).  vc4 is fetch-count bound here, so halving the weight
//     bytes only buys the weight share of the time - 1.23x on the gate and
//     1.33x on w2.  Prefill lands near 240/1.3 ms/token, not 240/2.
//
// Trap for anyone adding the next packed kernel: x_st2 takes *real* values and
// multiplies by 256 inside, so a matvec that already accumulates in q16 units
// must use x_p2_stq.  With x_p2_st the output is 256x too large and every row
// still looks plausible; only the CPU reference below caught it.
#include "gpu.h"
#include "xformat.h"

#include <GLES2/gl2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static Mat make_q8_mat(int rows, int cols, unsigned seed, float scale) {
    Mat m;
    m.rows = rows; m.cols = cols; m.type = WeightType::INT8;
    m.q8.resize((size_t)rows * cols);
    unsigned r = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < m.q8.size(); ++i) {
        r = r * 1103515245u + 12345u;
        m.q8[i] = (int8_t)((int)((r >> 16) & 255) - 128);
    }
    m.scale = scale;
    return m;
}

// One activation vector in a texture of its own: the serial kernels read from
// texel 0, so they cannot be handed the two-position upload (A then B).
static GLuint make_x_texture(const vector<float>& v) {
    vector<uint8_t> d = x_pack(v, XFormat::I16_2);
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x_texels((int)v.size(), XFormat::I16_2), 1,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
    return t;
}

static int q16at(const unsigned char* p) {
    const int v = (int)p[0] * 256 + (int)p[1];
    return v >= 32768 ? v - 65536 : v;
}

static int q24at(const unsigned char* p) {
    int64_t v = (int64_t)p[0] * 16777216 + (int64_t)p[1] * 65536 +
                (int64_t)p[2] * 256 + p[3] - 8388608;
    if (v >= 32768) v -= 65536;
    return (int)v;
}

int main() {
    setenv("KEEP_CPU_WEIGHTS", "1", 1);
    GPU gpu;
    printf("FFN2 renderer %s\n", (const char*)glGetString(GL_RENDERER));

    // MiniMind3 q8 FFN shapes: hidden 2432 split into two 1216-row chunks.
    const int chunk = 1216, cols = 768, dim = 768;
    Mat w1a = make_q8_mat(chunk, cols, 1, 1.0f / 2048.0f), w3a = make_q8_mat(chunk, cols, 2, 1.0f / 2048.0f);
    Mat w1b = make_q8_mat(chunk, cols, 3, 1.0f / 2048.0f), w3b = make_q8_mat(chunk, cols, 4, 1.0f / 2048.0f);
    Mat w2a = make_q8_mat(dim, chunk, 5, 1.0f / 512.0f), w2b = make_q8_mat(dim, chunk, 6, 1.0f / 512.0f);
    gpu.upload(w1a); gpu.upload(w3a); gpu.upload(w1b); gpu.upload(w3b);
    gpu.upload(w2a); gpu.upload(w2b);

    if (getenv("DUMP_P2")) { const string s = matvec_q8_p2(dim, chunk); printf("---- P2 %dx%d ----\n%s\n", dim, chunk, s.c_str()); }
    vector<float> va(cols), vb(cols);
    unsigned r = 99991u;
    for (int i = 0; i < cols; ++i) {
        r = r * 1103515245u + 12345u;
        va[i] = (float)((int)((r >> 16) & 255) - 128) / 256.0f;
        r = r * 1103515245u + 12345u;
        vb[i] = (float)((int)((r >> 16) & 255) - 128) / 256.0f;
    }
    GLuint ta = make_x_texture(va), tb = make_x_texture(vb);
    GLuint t2 = gpu.upload_input2(va, vb);
    gpu.forget_state();

    Target ga, gb, g2, oa, ob, o2;

    // (1) gate, two serial draws versus one packed draw: same bytes or a bug.
    gpu.ffn_pair_swiglu_chunk(w1a, w3a, ta, chunk, ga);
    gpu.ffn_pair_swiglu_chunk(w1a, w3a, tb, chunk, gb);
    gpu.ffn_pair_swiglu_chunk2(w1a, w3a, t2, chunk, g2);
    glFinish();
    vector<uint8_t> ba = gpu.read_bytes(ga, chunk / 2, 1);
    vector<uint8_t> bb = gpu.read_bytes(gb, chunk / 2, 1);
    vector<uint8_t> b2 = gpu.read_bytes(g2, chunk, 1);
    int da = 0, db = 0;
    for (int k = 0; k < chunk; ++k) {
        const int off = 4 * (k / 2) + 2 * (k & 1);
        if (q16at(&ba[off]) != q16at(&b2[4 * k])) ++da;
        if (q16at(&bb[off]) != q16at(&b2[4 * k + 2])) ++db;
    }
    printf("GATE %dx%d  q16 differ: A %d/%d  B %d/%d  (0 = identical)\n",
           chunk, cols, da, chunk, db, chunk);
    double gmin = 1e9, gmax = -1e9;
    for (int k = 0; k < chunk; ++k)
        for (int s = 0; s < 2; ++s) {
            const double v = (double)q16at(&b2[4 * k + 2 * s]) / 256.0;
            gmin = std::min(gmin, v); gmax = std::max(gmax, v);
        }
    printf("GATE value range %.3f .. %.3f (a dead kernel would read 0/1)\n", gmin, gmax);

    // (2) w2 over the gate output: serial q24 rows versus packed q16 texels.
    gpu.matvec_to(w2a, ga.tex, oa);
    gpu.matvec_to(w2a, gb.tex, ob);
    gpu.matvec_p2(w2a, g2.tex, o2);
    glFinish();
    vector<uint8_t> ra = gpu.read_bytes(oa, dim, 1);
    vector<uint8_t> rb = gpu.read_bytes(ob, dim, 1);
    vector<uint8_t> r2 = gpu.read_bytes(o2, dim, 1);
    int wa = 0, wb = 0;
    for (int i = 0; i < dim; ++i) {
        if (q24at(&ra[4 * i]) != q16at(&r2[4 * i])) ++wa;
        if (q24at(&rb[4 * i]) != q16at(&r2[4 * i + 2])) ++wb;
    }
    printf("W2   %dx%d  q16 differ: A %d/%d  B %d/%d\n", dim, chunk, wa, dim, wb, dim);
    int maxd = 0, ones = 0, big = 0;
    for (int i = 0; i < dim; ++i) {
        const int d = abs(q24at(&ra[4 * i]) - q16at(&r2[4 * i]));
        if (d > maxd) maxd = d;
        if (d == 1) ++ones;
        if (d > 1) ++big;
    }
    int maxdb = 0;
    for (int i = 0; i < dim; ++i) maxdb = std::max(maxdb, abs(q24at(&rb[4 * i]) - q16at(&r2[4 * i + 2])));
    printf("W2 delta: max %d q16 LSB (A), %d (B); |d|==1 %d, |d|>1 %d, of %d\n",
           maxd, maxdb, ones, big, dim);
    // both kernels.  A q16 LSB of slack covers float32 vs double accumulation.
    {
        int ok_s = 0, ok_p = 0, sat = 0, worst_s = 0, worst_p = 0;
        for (int i = 0; i < dim; ++i) {
            double acc = 0.0;
            for (int c = 0; c < chunk; ++c)
                acc += (double)w2a.w8()[(size_t)i * chunk + c] * (double)q16at(&b2[4 * c]);
            double v = std::floor(acc * w2a.scale + 0.5);
            v = std::max(-32768.0, std::min(32767.0, v));
            if (v > 32760.0 || v < -32760.0) ++sat;
            const int s = q24at(&ra[4 * i]), p = q16at(&r2[4 * i]);
            const int ds = (int)fabs((double)(s - v)), dp = (int)fabs((double)(p - v));
            if (ds <= 1) ++ok_s; else worst_s = std::max(worst_s, ds);
            if (dp <= 1) ++ok_p; else worst_p = std::max(worst_p, dp);
        }
        printf("W2 vs CPU: serial %d/%d within 1 LSB (worst %d), packed %d/%d (worst %d), %d rows saturate\n",
               ok_s, dim, worst_s, ok_p, dim, worst_p, sat);
    }
    // (3) timing: three interleaved blocks, min of each, both sides flushed
    // the way the engine does it in the steady state (FFN_FLUSH style glFlush).
    const int iters = 6;
    double sg = 1e9, bg = 1e9, sw = 1e9, bw = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        Clock::time_point t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            gpu.ffn_pair_swiglu_chunk(w1a, w3a, ta, chunk, ga);
            gpu.ffn_pair_swiglu_chunk(w1a, w3a, tb, chunk, gb);
        }
        glFinish();
        sg = std::min(sg, elapsed_ms(t0, Clock::now()) / iters);

        t0 = Clock::now();
        for (int i = 0; i < iters; ++i) gpu.ffn_pair_swiglu_chunk2(w1a, w3a, t2, chunk, g2);
        glFinish();
        bg = std::min(bg, elapsed_ms(t0, Clock::now()) / iters);

        t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            gpu.matvec_to(w2a, ga.tex, oa);
            gpu.matvec_to(w2a, gb.tex, ob);
        }
        glFinish();
        sw = std::min(sw, elapsed_ms(t0, Clock::now()) / iters);

        t0 = Clock::now();
        for (int i = 0; i < iters; ++i) gpu.matvec_p2(w2a, g2.tex, o2);
        glFinish();
        bw = std::min(bw, elapsed_ms(t0, Clock::now()) / iters);
    }
    printf("TIME gate %dx%d, 2 positions: serial %.3f ms  packed %.3f ms  %.2fx\n",
           chunk, cols, sg, bg, sg / bg);
    printf("TIME w2   %dx%d, 2 positions: serial %.3f ms  packed %.3f ms  %.2fx\n",
           dim, chunk, sw, bw, sw / bw);
    const double ser_l = 2 * sg + 2 * sw, bat_l = 2 * bg + 2 * bw;
    printf("TIME one layer FFN (2 gate + 2 w2 chunks), 2 positions: "
           "serial %.1f ms  packed %.1f ms  %.2fx\n", ser_l, bat_l, ser_l / bat_l);
    printf("TIME per position, one layer FFN: serial %.1f ms  packed %.1f ms\n",
           ser_l / 2, bat_l / 2);
    return 0;
}

