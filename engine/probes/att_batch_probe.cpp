// Where does a prefill's attention time go, and can batching the queries over
// one KV read actually buy it?
//
// The shipped score kernel (`att_score`, shaders_attn.cpp) is shaped for
// *decode*: `gl_FragCoord.x` is a KV position pair and `gl_FragCoord.y` a head,
// and each fragment reads the query texels it needs for one query row.  With one
// position there is exactly one query row, so every KV position re-reads it --
// the arithmetic intensity is one multiply-add a fetch.
//
// A prefill has N query rows against the same K/V, which is where a batched form
// could win: fetch the pair once per fragment and do N dot products against it.
// This probe prices that, and it prices the two forms against each other at the
// *same* output, so the difference is the query re-read and nothing else.
//
//   q1   one query row a fragment, the shipped shape (decode)
//   qN   N query rows a fragment over one KV fetch (prefill)
//   qf   q1 with the KV decode removed -- the pure fetch floor
//
// Each arm's row 0 is checked against a CPU dot product first; a fast arm that
// is wrong would be the failure this catches.
//
//   build:  make -C engine probes/att_batch_probe
//   run:    ./att_batch_probe [--seq N] [--dim D] [--heads H] [--batch N] [--iters K]
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

// The engine asks for ESSL 3.00 the same way (GPU::init_egl): the query/KV
// decoders use integer shifts when the driver has them.
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
    if (!ok) {
        char b[4096];
        glGetShaderInfoLog(h, sizeof b, nullptr, b);
        log = b;
    }
    return ok ? h : 0;
}

// One query row of `dim` values in q16 pairs, the engine's layout: two texels a
// value (hi/lo), so a row is 2*dim texels wide.
static vector<unsigned char> pack_row(const vector<float>& v) {
    vector<unsigned char> o(v.size() * 2 * 4, 0);
    for (size_t i = 0; i < v.size(); ++i) {
        float x = std::max(-32768.0f, std::min(32767.0f, std::floor(v[i] * 256.0f + 0.5f)));
        if (x < 0) x += 65536.0f;
        unsigned u = (unsigned) x;
        unsigned char hi = (unsigned char) (u >> 8), lo = (unsigned char) (u & 0xff);
        o[(i * 2) * 4 + 0] = hi; o[(i * 2) * 4 + 1] = lo;
        o[(i * 2) * 4 + 4] = hi; o[(i * 2) * 4 + 5] = lo;
    }
    return o;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int seq = 512, dim = 64, heads = 2, batch = 4, iters = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i + 1 < argc) heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--seq N] [--dim D] [--heads H] [--batch N] [--iters K]\n", argv[0]); return 2; }
    }
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    printf("renderer %s\nversion  %s\n", (const char*) glGetString(GL_RENDERER),
           (const char*) glGetString(GL_VERSION));

    // ---- the data: `batch` query rows, `seq` KV rows, one head
    vector<vector<float>> q(batch, vector<float>(dim));
    vector<vector<float>> kv(seq, vector<float>(dim));
    unsigned seed = 7;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) & 0x7fff); };
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < dim; i++) q[b][i] = (float) (rnd() % 7) - 3;   // small integers
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < dim; i++) kv[t][i] = (float) (rnd() % 7) - 3;

    // Q: `batch` rows stacked, 2*dim texels a row
    const int qw = dim * 2;
    vector<unsigned char> qtex((size_t) qw * batch * 4, 0);
    for (int b = 0; b < batch; b++) {
        vector<unsigned char> r = pack_row(q[b]);
        memcpy(&qtex[(size_t) b * qw * 4], r.data(), (size_t) qw * 4);
    }
    GLuint qid = 0;
    glGenTextures(1, &qid);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, qid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, qw, batch, 0, GL_RGBA, GL_UNSIGNED_BYTE, qtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // KV: one value per *pair of texels* -- hi/lo in .r/.g of texel 2i, and the
    // same bytes in texel 2i+1 (the engine's q16 pair packs four values a texel and
    // this probe does not need that; two texels a value keeps the loader simple).
    const int kvw = dim * 2;
    vector<unsigned char> ktex((size_t) kvw * seq * 4, 0);
    for (int t = 0; t < seq; t++) {
        for (int i = 0; i < dim; i++) {
            float x = std::max(-32768.0f, std::min(32767.0f, std::floor(kv[t][i] * 256.0f + 0.5f)));
            if (x < 0) x += 65536.0f;
            unsigned u = (unsigned) x;
            unsigned char* o = &ktex[((size_t) t * kvw + i * 2) * 4];
            o[0] = (unsigned char) (u >> 8); o[1] = (unsigned char) (u & 0xff);
        }
    }
    GLuint kid = 0;
    glGenTextures(1, &kid);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, kid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kvw, seq, 0, GL_RGBA, GL_UNSIGNED_BYTE, ktex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // ---- the output: `batch` scores a KV position, one pair an RGBA8 texel
    // (two q16-pair halves), so the target is seq/2 x batch.
    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (seq + 1) / 2, batch, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("fbo incomplete (%dx%d)\n", (seq + 1) / 2, batch);
        return 1;
    }

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    // ---- the sanity arm: encode 1.5 and 256 and read them back through the same
    // path every timed arm uses.  A probe whose own encode/decode round trip is
    // wrong reports numbers that mean nothing, and that is how three rounds of
    // this went.
    {
        string vs = "#version 300 es\nin vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
                    "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
                    "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
                    "return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n"
                    "void main(){vec2 a=e16(384.0),b=e16(-384.0);"
                    "frag_color=vec4(a.x,a.y,b.x,b.y);}\n";
        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) { printf("sanity arm COMPILE FAIL %s\n", log.c_str()); }
        else {
            GLuint p = glCreateProgram();
            glAttachShader(p, hvs); glAttachShader(p, hfs);
            glBindAttribLocation(p, 0, "pos");
            glLinkProgram(p);
            glUseProgram(p);
            glViewport(0, 0, (seq + 1) / 2, 1);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            unsigned char t[4] = {0, 0, 0, 0};
            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, t);
            auto dec = [](int o, unsigned char* q) {
                double u = (double) q[o] * 256.0 + (double) q[o + 1];
                if (u >= 32768.0) u -= 65536.0;
                return u / 256.0;
            };
            const double a = dec(0, t), b = dec(2, t);
            printf("sanity: wrote q16 384/-384, read %.4f/%.4f  %s\n", a, b,
                   (std::fabs(a - 384.0 / 256.0) < 2e-3 && std::fabs(b + 384.0 / 256.0) < 2e-3)
                       ? "ok" : "BROKEN");
            glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
        }
    }

    printf("\n%-22s %8s %9s %10s %8s\n", "arm", "frags", "ms/draw", "ns/MAC", "check");
    struct Arm { const char* name; int rows; bool decode_kv; };
    const Arm arms[] = {
        {"q1 (decode shape)",   1,    true  },
        {"qN (batched queries)", batch, true },
        {"q1_fetch (no decode)", 1,   false },
        {"qN_fetch (no decode)", batch, false},
    };
    for (const Arm& a : arms) {
        const string ver = "#version 300 es\n";
        string vs = ver + "in vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
        string fs = ver +
            "precision highp float;\nprecision highp int;\nout vec4 frag_color;\n"
            "uniform sampler2D Q,KV;uniform float qh,invsqrt;\n"
            "float d16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);"
            "if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
            "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
            "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
            "return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n";
        // one query row a fragment (the shipped shape) or N rows over one KV fetch
        if (a.rows == 1) {
            // the shipped shape: one query row, `frags` KV position pairs
            fs += "void main(){int T=int(gl_FragCoord.x);int t0=T*2;int t1=t0+1;"
                  "float s0=0.0,s1=0.0;"
                  "for(int i=0;i<" + std::to_string(dim) + ";i++){"
                  "vec4 qv=texture2D(Q,vec2((float(i*2)+0.5)/float(" + std::to_string(qw) + "),0.5));"
                  "float w=d16(qv.rg);";
            if (a.decode_kv) {
                fs += "vec4 k0=texture2D(KV,vec2((float(i*2)+0.5)/float(" + std::to_string(kvw) +
                      "),(float(t0)+0.5)/float(" + std::to_string(seq) + ")));"
                      "vec4 k1=texture2D(KV,vec2((float(i*2)+0.5)/float(" + std::to_string(kvw) +
                      "),(float(t1)+0.5)/float(" + std::to_string(seq) + ")));"
                      "s0+=w*d16(k0.rg);s1+=w*d16(k1.rg);";
            } else {
                fs += "s0+=w;s1+=w;";     // same MACs, no KV fetch
            }
            fs += "}"
                  "vec2 p0=e16(s0*invsqrt*256.0),p1=e16(s1*invsqrt*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
        } else {
            // N query rows a fragment: one KV fetch, N dot products
            fs += "void main(){int T=int(gl_FragCoord.x);int t0=T*2;int t1=t0+1;";
            for (int b = 0; b < a.rows; b++)
                fs += "float s" + std::to_string(b) + "0=0.0,s" + std::to_string(b) + "1=0.0;";
            fs += "for(int i=0;i<" + std::to_string(dim) + ";i++){";
            if (a.decode_kv) {
                fs += "vec4 k0=texture2D(KV,vec2((float(i*2)+0.5)/float(" + std::to_string(kvw) +
                      "),(float(t0)+0.5)/float(" + std::to_string(seq) + ")));"
                      "float v0=d16(k0.rg);"
                      "vec4 k1=texture2D(KV,vec2((float(i*2)+0.5)/float(" + std::to_string(kvw) +
                      "),(float(t1)+0.5)/float(" + std::to_string(seq) + ")));"
                      "float v1=d16(k1.rg);";
            } else {
                fs += "float v0=1.0;float v1=1.0;";
            }
            for (int b = 0; b < a.rows; b++)
                fs += "vec4 qb" + std::to_string(b) + "=texture2D(Q,vec2((float(i*2)+0.5)/float(" +
                      std::to_string(qw) + "),(" + std::to_string(b) + ".5)/float(" +
                      std::to_string(batch) + ")));"
                      "float w" + std::to_string(b) + "=d16(qb" + std::to_string(b) + ".rg);"
                      "s" + std::to_string(b) + "0+=w" + std::to_string(b) + "*v0;"
                      "s" + std::to_string(b) + "1+=w" + std::to_string(b) + "*v1;";
            fs += "}";
            // one row's worth of output is what the target holds; write row 0 and
            // fold the rest in so the compiler cannot drop them
            fs += "float acc0=s00,acc1=s01;";
            for (int b = 1; b < a.rows; b++)
                fs += "acc0+=" + std::to_string(b) + ".0*0.0+s" + std::to_string(b) + "0*0.0;"
                      "acc1+=s" + std::to_string(b) + "1*0.0;";
            fs += "vec2 p0=e16(acc0*invsqrt*256.0),p1=e16(acc1*invsqrt*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}\n";
        }

        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) {
            printf("%-22s COMPILE FAIL  %s\n", a.name, log.c_str());
            continue;
        }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs);
        glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char b[4096];
            glGetProgramInfoLog(p, sizeof b, nullptr, b);
            printf("%-22s LINK FAIL     %s\n", a.name, b);
            continue;
        }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, qid);
        glUniform1i(glGetUniformLocation(p, "Q"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, kid);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glUniform1f(glGetUniformLocation(p, "invsqrt"), 1.0f / std::sqrt((float) dim));
        glUniform1f(glGetUniformLocation(p, "qh"), (float) a.rows);
        glActiveTexture(GL_TEXTURE0);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        const int frags = (seq + 1) / 2;
        glViewport(0, 0, frags, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;

        // ---- check the first fragment against a CPU dot product
        vector<unsigned char> px((size_t) frags * 4, 0);
        glReadPixels(0, 0, frags, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        auto q16 = [&](int o) {
            double u = (double) px[o] * 256.0 + (double) px[o + 1];
            if (u >= 32768.0) u -= 65536.0;      // offset binary, as the shader writes it
            return u / 256.0;
        };
        const double got = q16(0);
        const double got_b = q16(2);   // the .ba half: the trace arm's KV value
        // The shader writes score * invsqrt, so the reference carries the scale too.
        double ref = 0.0;
        for (int i = 0; i < dim; i++)
            ref += a.decode_kv ? q[0][i] * kv[0][i] : q[0][i];
        ref /= std::sqrt((double) dim);
        // the batched arms fold the other rows out of their output, so only row 0
        // is compared -- which is what they write.
        if (getenv("ATT_DEBUG")) {
            printf("    q[0][0..3] = %.1f %.1f %.1f %.1f\n", q[0][0], q[0][1], q[0][2], q[0][3]);
            printf("    kv[0][0..3]= %.1f %.1f %.1f %.1f   kv[1][0..3]= %.1f %.1f %.1f %.1f\n",
                   kv[0][0], kv[0][1], kv[0][2], kv[0][3],
                   kv[1][0], kv[1][1], kv[1][2], kv[1][3]);
            printf("    got %.6f  ref %.6f  (dot of q0.kv0)\n", got, ref);
            printf("    trace: q-half %.6f  kv-half %.6f\n", got, got_b);
        }
        const double err = std::fabs(got - ref * 1.0);
        const bool good = err < 2e-2 * (1.0 + std::fabs(ref));
        // MACs actually issued: rows x dim x fragments
        const double macs = (double) a.rows * dim * frags;
        if (getenv("ATT_DUMP")) {
            printf("    shader:\n%s\n    raw texel0: %d %d %d %d  decoded %.4f\n",
                   fs.c_str(), px[0], px[1], px[2], px[3], got);
        }
        printf("%-22s %8d %9.3f %10.2f %8s\n", a.name, frags, ms, 1e6 * ms / macs,
               good ? "ok" : "MISMATCH");
        if (!good) printf("    got %.4f want %.4f (err %.4f)\n", got, ref, err);
        glDeleteProgram(p);
        glDeleteShader(hvs);
        glDeleteShader(hfs);
    }
    return 0;
}
