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
// WORKLOG 7f (2026-11-05): B=4 priced, and it does not pay.  A fragment writes
// one RGBA8 texel, so four positions of a gate row (4 q16) cannot leave one
// draw and vc4 has no MRT; the only two encodings are two B=2 passes, which is
// exactly the work per position of B=2 plus one draw, or one pass that writes
// the four gate values as int8.  The int8 gate is 1.57x faster per position
// (2.28 against 3.59 ms) yet only 1.26x on the FFN layer, because the w2 output
// is the residual and must stay q16: for four positions 52.5 -> 41.8 ms and w2
// alone is 23.6 of the 41.8.  With 8 layers (dim 768, hidden 2432) that caps the
// prefill win near 1.10x, for a gate quantised to 1/127 of its row range.  B=2
// stays the exact default (step 5) and this probe is the evidence.
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

// ---------------------------------------------------------------------------
// WORKLOG 7f: price B=4.
//
// A fragment writes one RGBA8 texel = two q16 values, and vc4 has no MRT, so
// four positions of one gate row (four q16 = 8 bytes) cannot leave one draw.
// There are exactly two encodings for B=4:
//
//   (a) two passes of the B=2 kernel.  The second pass re-fetches the same
//       weights, so the work per position is that of B=2 by construction and
//       the only difference is one extra draw per gate chunk.
//   (b) one pass that writes the four gate values as int8, one byte each.
//       That is the fastest B=4 gate that can exist, so its time is an upper
//       bound for any B=4 gate and its error is a lower bound on the quality
//       cost.  A per-row *scale* cannot be produced in the same fragment
//       either (one texel, no max over the row), so the scale has to be a
//       fixed uniform - the probe uses the observed range, which is the best
//       a fixed scale can do.
//
// The w2 side cannot use (b) at all: its output is the FFN residual and has to
// stay q16 (step 7c showed a 1-LSB move in 2/768 gate values changes the greedy
// text), so a B=4 layer still needs four w2 draws for four positions.  What
// B=4 can win is at most the gate half of the FFN.
// ---------------------------------------------------------------------------
struct Rt { GLuint tex = 0, fbo = 0; int w = 0, h = 0; };

static Rt make_rt(int w, int h) {
    Rt t; t.w = w; t.h = h;
    glGenTextures(1, &t.tex);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "B4: FBO incomplete\n"); exit(1);
    }
    return t;
}

static vector<uint8_t> read_rt(const Rt& t) {
    vector<uint8_t> b((size_t)t.w * 4);
    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, t.w, 1, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
    return b;
}

// Four positions, each I16_2-packed at its own offset in one texture, which is
// what the gate reads for a B=4 draw.
static GLuint make_x4_texture(const vector<vector<float> >& vs) {
    const int axw = x_texels((int)vs[0].size(), XFormat::I16_2);
    vector<uint8_t> d;
    for (size_t i = 0; i < vs.size(); ++i) {
        vector<uint8_t> p = x_pack(vs[i], XFormat::I16_2);
        d.insert(d.end(), p.begin(), p.end());
    }
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4 * axw, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
    return t;
}

// The B=4 gate: one row per fragment, all four positions, int8 out.  Fetch
// count per 4-value block: 8 activation texels + 2 weight texels for four
// positions, against 4 + 2 for two positions at B=2 - i.e. 2.5 fetches per
// position against 3.0, with a quarter of the weight bytes per position.
static string matvec_q8_gate4_i8(int rows, int cols) {
    const int axw = x_texels(cols, XFormat::I16_2);
    const string ROWS = to_string(rows), COLS4 = to_string(cols / 4);
    const string A1 = to_string(axw), A2 = to_string(2 * axw), A3 = to_string(3 * axw);
    const string W4 = to_string(4 * axw);
    const string WR = "vec2((float(q)+0.5)/float(" + COLS4 + "),(float(r)+0.5)/float(" + ROWS + "))";
    return string("precision highp float;\n") + x_glsl(XFormat::I16_2) + x_pair2_glsl() +
        "uniform sampler2D W1,W2,X;uniform float ws1,ws2,vs;"
        "float qz(float v){return clamp(floor(v+0.5),-32768.0,32767.0);}"
        "float dA(vec4 c){float u=c.r*65280.0+c.g*255.0;"
        "return u>=32768.0?u-65536.0:u;}"
        "float dB(vec4 c){float u=c.b*65280.0+c.a*255.0;"
        "return u>=32768.0?u-65536.0:u;}"
        "float sl(float q,float g){float a=q/256.0;"
        "return (a/(1.0+exp(-a)))*(g/256.0);}"
        "float e8(float v){return (clamp(floor(v*vs+0.5),-127.0,127.0)+128.0)/255.0;}"
        "void main(){int r=int(gl_FragCoord.x);"
        "float g0=0.0,g1=0.0,g2=0.0,g3=0.0,h0=0.0,h1=0.0,h2=0.0,h3=0.0;"
        "for(int q=0;q<" + COLS4 + ";q++){int t=2*q;"
        "vec4 xa=texture2D(X,vec2((float(t)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xb=texture2D(X,vec2((float(t+1)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xc=texture2D(X,vec2((float(" + A1 + "+t)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xd=texture2D(X,vec2((float(" + A1 + "+t+1)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xe=texture2D(X,vec2((float(" + A2 + "+t)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xf=texture2D(X,vec2((float(" + A2 + "+t+1)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xg=texture2D(X,vec2((float(" + A3 + "+t)+0.5)/float(" + W4 + "),0.5));"
        "vec4 xh=texture2D(X,vec2((float(" + A3 + "+t+1)+0.5)/float(" + W4 + "),0.5));"
        "float a0=dA(xa),a1=dB(xa),a2=dA(xb),a3=dB(xb);"
        "float b0=dA(xc),b1=dB(xc),b2=dA(xd),b3=dB(xd);"
        "float c0=dA(xe),c1=dB(xe),c2=dA(xf),c3=dB(xf);"
        "float d0=dA(xg),d1=dB(xg),d2=dA(xh),d3=dB(xh);"
        "vec4 cw=texture2D(W1," + WR + ");vec4 w=cw*255.0-128.0;"
        "g0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "g1+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
        "g2+=w.x*c0+w.y*c1+w.z*c2+w.w*c3;"
        "g3+=w.x*d0+w.y*d1+w.z*d2+w.w*d3;"
        "cw=texture2D(W2," + WR + ");w=cw*255.0-128.0;"
        "h0+=w.x*a0+w.y*a1+w.z*a2+w.w*a3;"
        "h1+=w.x*b0+w.y*b1+w.z*b2+w.w*b3;"
        "h2+=w.x*c0+w.y*c1+w.z*c2+w.w*c3;"
        "h3+=w.x*d0+w.y*d1+w.z*d2+w.w*d3;}"
        "gl_FragColor=vec4(e8(sl(qz(g0*ws1),qz(h0*ws2))),"
        "e8(sl(qz(g1*ws1),qz(h1*ws2))),"
        "e8(sl(qz(g2*ws1),qz(h2*ws2))),"
        "e8(sl(qz(g3*ws1),qz(h3*ws2))));}";
}

struct B4 {
    GLuint prog = 0, vbo = 0;
    GLint W1 = -1, W2 = -1, X = -1, ws1 = -1, ws2 = -1, vs = -1, p = -1;
};

static B4 make_b4(const string& fs) {
    B4 b;
    const char* vs_src = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }";
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vs_src, nullptr); glCompileShader(vs);
    GLint ok = 0; glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(vs, sizeof(log), nullptr, log);
               fprintf(stderr, "B4 vertex: %s\n", log); exit(1); }
    const char* f = fs.c_str();
    GLuint fsh = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fsh, 1, &f, nullptr); glCompileShader(fsh);
    glGetShaderiv(fsh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[8192]; glGetShaderInfoLog(fsh, sizeof(log), nullptr, log);
               fprintf(stderr, "B4 fragment: %s\n", log); exit(1); }
    b.prog = glCreateProgram();
    glAttachShader(b.prog, vs); glAttachShader(b.prog, fsh);
    glLinkProgram(b.prog);
    glGetProgramiv(b.prog, GL_LINK_STATUS, &ok);
    if (!ok) { char log[8192]; glGetProgramInfoLog(b.prog, sizeof(log), nullptr, log);
               fprintf(stderr, "B4 link: %s\n", log); exit(1); }
    b.W1 = glGetUniformLocation(b.prog, "W1");
    b.W2 = glGetUniformLocation(b.prog, "W2");
    b.X = glGetUniformLocation(b.prog, "X");
    b.ws1 = glGetUniformLocation(b.prog, "ws1");
    b.ws2 = glGetUniformLocation(b.prog, "ws2");
    b.vs = glGetUniformLocation(b.prog, "vs");
    float verts[] = { -1, -1, 3, -1, -1, 3 };
    glGenBuffers(1, &b.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, b.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    b.p = glGetAttribLocation(b.prog, "p");
    return b;
}

static void draw_b4(const B4& b, GLuint w1, GLuint w3, GLuint x4, int rows,
                    float ws1, float ws2, float vs, const Rt& out) {
    glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
    glUseProgram(b.prog);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, w1); glUniform1i(b.W1, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, w3); glUniform1i(b.W2, 1);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, x4); glUniform1i(b.X, 2);
    glUniform1f(b.ws1, ws1); glUniform1f(b.ws2, ws2); glUniform1f(b.vs, vs);
    glBindBuffer(GL_ARRAY_BUFFER, b.vbo);
    glEnableVertexAttribArray((GLuint)b.p);
    glVertexAttribPointer((GLuint)b.p, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glViewport(0, 0, rows, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
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
    // (4) WORKLOG 7f: price B=4.  Both encodings exist (see the comment above
    // matvec_q8_gate4_i8); measure the lossy one against the B=2 path.
    {
        vector<float> vc(cols), vd(cols);
        for (int i = 0; i < cols; ++i) {
            r = r * 1103515245u + 12345u;
            vc[i] = (float)((int)((r >> 16) & 255) - 128) / 256.0f;
            r = r * 1103515245u + 12345u;
            vd[i] = (float)((int)((r >> 16) & 255) - 128) / 256.0f;
        }
        GLuint tcd = gpu.upload_input2(vc, vd);
        Target gcd;
        gpu.ffn_pair_swiglu_chunk2(w1a, w3a, tcd, chunk, gcd);
        glFinish();
        vector<uint8_t> bcd = gpu.read_bytes(gcd, chunk, 1);
        vector<int> ref((size_t)4 * chunk);
        for (int k = 0; k < chunk; ++k) {
            ref[0 * chunk + k] = q16at(&b2[4 * k]);
            ref[1 * chunk + k] = q16at(&b2[4 * k + 2]);
            ref[2 * chunk + k] = q16at(&bcd[4 * k]);
            ref[3 * chunk + k] = q16at(&bcd[4 * k + 2]);
        }
        int maxq = 1;
        for (size_t i = 0; i < ref.size(); ++i) maxq = max(maxq, abs(ref[i]));
        const float vs = 127.0f / ((float)maxq / 256.0f);
        printf("B4 gate range %d q16 = %.3f real; one int8 step = %.1f q16 LSB\n",
               maxq, maxq / 256.0, (double)maxq / 127.0);
        vector<vector<float> > v4;
        v4.push_back(va); v4.push_back(vb); v4.push_back(vc); v4.push_back(vd);
        GLuint x4 = make_x4_texture(v4);
        B4 b4 = make_b4(matvec_q8_gate4_i8(chunk, cols));
        Rt g4 = make_rt(chunk, 1);
        draw_b4(b4, w1a.tex, w3a.tex, x4, chunk, w1a.scale, w3a.scale, vs, g4); gpu.forget_state();
        vector<uint8_t> b4r = read_rt(g4);
        double sum2 = 0.0; int maxe = 0, within1 = 0, dead = 1;
        for (int k = 0; k < chunk; ++k) {
            for (int p = 0; p < 4; ++p) {
                const int e = (int)b4r[4 * k + p];
                if (e != 128) dead = 0;
                const double v = ((int)e - 128) / (double)vs;
                const double q = ref[(size_t)p * chunk + k] / 256.0;
                const double d = fabs(v - q) * 256.0;
                sum2 += d * d;
                if ((int)lrint(d) > maxe) maxe = (int)lrint(d);
                if (d <= 1.0) ++within1;
            }
        }
        printf("B4 int8 gate vs B=2 q16: rms %.2f LSB, max %d LSB, within 1 LSB %d/%d, dead=%d\n",
               sqrt(sum2 / (4.0 * chunk)), maxe, within1, 4 * chunk, dead);
        double t_one = 1e9, t_alt = 1e9, b4t = 1e9, w2p = 1e9, ser1 = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            Clock::time_point t0 = Clock::now();
            for (int i = 0; i < iters; ++i) gpu.ffn_pair_swiglu_chunk2(w1a, w3a, t2, chunk, g2);
            glFinish();
            t_one = std::min(t_one, elapsed_ms(t0, Clock::now()) / iters);
            t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                gpu.ffn_pair_swiglu_chunk2(w1a, w3a, t2, chunk, g2);
                gpu.ffn_pair_swiglu_chunk2(w1a, w3a, tcd, chunk, gcd);
            }
            glFinish();
            t_alt = std::min(t_alt, elapsed_ms(t0, Clock::now()) / (2 * iters));
            t0 = Clock::now();
            for (int i = 0; i < iters; ++i) gpu.matvec_p2(w2a, g2.tex, o2);
            glFinish();
            w2p = std::min(w2p, elapsed_ms(t0, Clock::now()) / iters);
            t0 = Clock::now();
            for (int i = 0; i < iters; ++i) gpu.ffn_pair_swiglu_chunk(w1a, w3a, ta, chunk, ga);
            glFinish();
            ser1 = std::min(ser1, elapsed_ms(t0, Clock::now()) / iters);
            t0 = Clock::now();
            for (int i = 0; i < iters; ++i) { draw_b4(b4, w1a.tex, w3a.tex, x4, chunk, w1a.scale, w3a.scale, vs, g4); gpu.forget_state(); }
            glFinish();
            b4t = std::min(b4t, elapsed_ms(t0, Clock::now()) / iters);
        }
        printf("PHASE4 gate per position: serial %.2f, B=2 same %.2f, B=2 alt %.2f, B=4 int8 %.2f; w2 B=2 %.2f per 2\n",
               ser1, t_one / 2, t_alt / 2, b4t / 4, w2p);
        printf("TIME one FFN layer gate (2 chunks), 4 positions: B=2 %.2f ms  B=4 int8 %.2f ms  %.2fx\n",
               4 * t_alt, b4t, (4 * t_alt) / b4t);
        const double layer_b2_4p = 4 * t_alt + 4 * w2p;
        const double layer_b4_4p = 2 * b4t + 4 * w2p;
        printf("TIME one FFN layer, 4 positions: B=2 %.1f ms  gate-only B=4 int8 %.1f ms  %.2fx\n",
               layer_b2_4p, layer_b4_4p, layer_b2_4p / layer_b4_4p);
    }

    return 0;
}
