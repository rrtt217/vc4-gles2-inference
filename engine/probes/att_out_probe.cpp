// Does the query-batching win survive *writing the output*?
//
// att_batch_probe priced N queries over one KV fetch, but its qN arm folded
// N-1 rows out of the output (multiply by 0.0) -- it priced the compute, not
// the kernel.  A real score kernel must write every score it computes, and a
// VC4 fragment writes ONE texel: two q16 scores.  N queries x a KV pair is 2N
// scores, so one fragment cannot emit them all without MRT.  The shapes that
// can actually be dispatched are:
//
//   q1          the shipped decode shape, one query a head (grid seq/2 x H)
//   qN_serial   grid seq/2 x (H*N), a fragment computes ONLY its own query
//               -- N serial passes fused into one draw, no sharing
//   qN_recomp   grid seq/2 x (H*N), a fragment computes ALL N queries of its
//               head (2N chains, the probe's shape) but writes only its own
//               row -- N x redundant MACs, and the KV/Q re-fetches may hit
//               the texture cache
//   qN_mrt      glDrawBuffers: one fragment, N/2..N attachments, one KV fetch
//               -- the clean form, IF the driver has it at all
//
// Every arm writes real scores and every row is checked against a CPU dot
// product, so a fast arm that drops fragments (the draw cap) or writes the
// wrong row is caught, not priced.  Run at --heads 1 and --heads 16: if q1
// closes most of the gap at 16, the original 16x was a small-grid occupancy
// artifact and the engine's seq/2 x heads grid already has it.
//
//   build:  make -C engine probes/att_out_probe
//   run:    ./att_out_probe [--seq N] [--dim D] [--heads H] [--batch N] [--iters K]
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

// ES3 entry points the ES2 headers do not declare; resolved at run time so an
// ES2-only loader still links and the MRT arm reports "unsupported" instead.
#ifndef GL_MAX_DRAW_BUFFERS
#define GL_MAX_DRAW_BUFFERS 0x8824
#endif
#ifndef GL_COLOR_ATTACHMENT0
#define GL_COLOR_ATTACHMENT0 0x8CE0
#endif
typedef void (GL_APIENTRY *PFN_DRAWBUFFERS)(GLsizei, const GLenum*);
typedef void (GL_APIENTRY *PFN_READBUFFER)(GLenum);

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

// One query row of `dim` values, two texels a value (hi/lo in .r/.g), the same
// layout att_batch_probe uses so the two probes' numbers are comparable.
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

struct Out {
    GLuint fbo = 0, tex = 0;
    int w = 0, h = 0;
};

static Out make_out(int w, int h) {
    Out o;
    o.w = w; o.h = h;
    glGenFramebuffers(1, &o.fbo);
    glGenTextures(1, &o.tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, o.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, o.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, o.tex, 0);
    return o;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int seq = 512, dim = 64, heads = 16, batch = 8, iters = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i + 1 < argc) heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--seq N] [--dim D] [--heads H] [--batch N] [--iters K]\n", argv[0]); return 2; }
    }
    seq += seq & 1;                     // the check reads KV pairs; keep seq even
    if (batch < 1) batch = 1;
    request_es3();
    EglSurfaceless egl = egl_surfaceless_init(3);
    if (!egl.ok()) return 1;
    GLint maxdb = 0, maxtex = 0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxdb);
    while (glGetError() != GL_NO_ERROR) {}      // ES2 context: invalid enum, maxdb stays 0
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtex);
    PFN_DRAWBUFFERS pDrawBuffers = (PFN_DRAWBUFFERS) eglGetProcAddress("glDrawBuffers");
    PFN_READBUFFER pReadBuffer = (PFN_READBUFFER) eglGetProcAddress("glReadBuffer");
    printf("renderer %s\nversion  %s\nmax_draw_buffers %d  max_texture_size %d  drawbuffers %s\n",
           (const char*) glGetString(GL_RENDERER), (const char*) glGetString(GL_VERSION),
           (int) maxdb, (int) maxtex, pDrawBuffers ? "yes" : "no");

    const int R = heads * batch;        // query rows: head-major, batch within
    const int qw = dim * 2;
    const int kvw = dim * 2 * heads;    // head h owns columns [h*2*dim, (h+1)*2*dim)
    if (kvw > maxtex) { printf("KV width %d over max texture %d\n", kvw, (int) maxtex); return 1; }
    const int frags = seq / 2;

    // ---- the data: `batch` queries shared by every head, `seq` KV rows with
    // one `dim` slice a head (the engine's per-head K columns).
    vector<vector<float>> q(batch, vector<float>(dim));
    vector<vector<float>> kv(seq, vector<float>((size_t) heads * dim));
    unsigned seed = 7;
    auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int) ((seed >> 16) & 0x7fff); };
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < dim; i++) q[b][i] = (float) (rnd() % 7) - 3;
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < heads * dim; i++) kv[t][i] = (float) (rnd() % 7) - 3;

    vector<unsigned char> qtex((size_t) qw * R * 4, 0);
    for (int r = 0; r < R; r++) {
        vector<unsigned char> row = pack_row(q[r % batch]);
        memcpy(&qtex[(size_t) r * qw * 4], row.data(), (size_t) qw * 4);
    }
    GLuint qid = 0;
    glGenTextures(1, &qid);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, qid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, qw, R, 0, GL_RGBA, GL_UNSIGNED_BYTE, qtex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    vector<unsigned char> ktex((size_t) kvw * seq * 4, 0);
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < heads * dim; i++) {
            float x = std::max(-32768.0f, std::min(32767.0f, std::floor(kv[t][i] * 256.0f + 0.5f)));
            if (x < 0) x += 65536.0f;
            unsigned u = (unsigned) x;
            unsigned char* o = &ktex[((size_t) t * kvw + i * 2) * 4];
            o[0] = (unsigned char) (u >> 8); o[1] = (unsigned char) (u & 0xff);
        }
    GLuint kid = 0;
    glGenTextures(1, &kid);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, kid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kvw, seq, 0, GL_RGBA, GL_UNSIGNED_BYTE, ktex.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);

    // ---- sanity: the encode/decode round trip every arm relies on
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
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs); glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        glUseProgram(p);
        Out o = make_out(1, 1);
        glViewport(0, 0, 1, 1);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        unsigned char t[4] = {0, 0, 0, 0};
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, t);
        auto dec = [](int off, unsigned char* b) {
            double u = (double) b[off] * 256.0 + (double) b[off + 1];
            if (u >= 32768.0) u -= 65536.0;
            return u / 256.0;
        };
        const double a = dec(0, t), b = dec(2, t);
        printf("sanity: wrote q16 384/-384, read %.4f/%.4f  %s\n", a, b,
               (std::fabs(a - 1.5) < 2e-3 && std::fabs(b + 1.5) < 2e-3) ? "ok" : "BROKEN");
        glDeleteProgram(p); glDeleteShader(hvs); glDeleteShader(hfs);
        glDeleteFramebuffers(1, &o.fbo); glDeleteTextures(1, &o.tex);
    }

    // ---- shared shader preamble: the codecs and the fetch helpers
    const string pre =
        "#version 300 es\nprecision highp float;\nprecision highp int;\n"
        "uniform sampler2D Q,KV;uniform float invsqrt;\n"
        "float d16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);"
        "if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
        "vec2 e16(float q){q=clamp(floor(q+0.5),-32768.0,32767.0);"
        "if(q<0.0)q+=65536.0;float hi=floor(q/256.0);"
        "return vec2(hi/255.0,(q-hi*256.0)/255.0);}\n"
        "vec4 qf(int row,int i){return texture2D(Q,vec2((float(i*2)+0.5)/float(" +
        std::to_string(qw) + "),(float(row)+0.5)/float(" + std::to_string(R) + ")));}\n"
        "vec4 kf(int h,int i,int t){return texture2D(KV,vec2((float(h*" +
        std::to_string(2 * dim) + "+i*2)+0.5)/float(" + std::to_string(kvw) +
        "),(float(t)+0.5)/float(" + std::to_string(seq) + ")));}\n";
    const string vs = "#version 300 es\nin vec2 pos;\nvoid main(){gl_Position=vec4(pos,0.0,1.0);}\n";
    const float invsqrt = 1.0f / std::sqrt((float) dim);

    // the score of query b, head h, KV position t, on the CPU
    auto ref = [&](int h, int b, int t) {
        double s = 0.0;
        for (int i = 0; i < dim; i++) s += (double) q[b][i] * kv[t][(size_t) h * dim + i];
        return s / std::sqrt((double) dim);
    };
    auto q16dec = [](const unsigned char* p, int off) {
        double u = (double) p[off] * 256.0 + (double) p[off + 1];
        if (u >= 32768.0) u -= 65536.0;
        return u / 256.0;
    };

    printf("\nshape: seq %d dim %d heads %d batch %d -> %d query rows, %d frags/draw\n",
           seq, dim, heads, batch, R, frags);
    printf("%-14s %10s %9s %7s %9s %10s %8s\n",
           "arm", "grid", "ms/draw", "draws", "ms/full", "ns/MAC", "check");

    // ATT_OUT_ARMS selects a subset (comma list); default all four.
    const char* only = getenv("ATT_OUT_ARMS");
    auto wanted = [&](const char* n) {
        if (!only) return true;
        string s(only), t(n);
        size_t p = 0;
        while (p <= s.size()) {
            size_t e = s.find(',', p);
            if (s.substr(p, e == string::npos ? string::npos : e - p) == t) return true;
            if (e == string::npos) break;
            p = e + 1;
        }
        return false;
    };

    const double full_macs = (double) R * seq * dim;   // one complete score set
    struct Res { const char* name; double ms_full; bool ran; };
    Res results[4] = {{"q1", 0, false}, {"qN_serial", 0, false},
                      {"qN_recomp", 0, false}, {"qN_mrt", 0, false}};

    for (int arm = 0; arm < 4; arm++) {
        const char* name = results[arm].name;
        if (!wanted(name)) continue;
        const bool mrt = arm == 3;
        const int A = mrt ? std::min(batch, (int) maxdb) : 1;
        if (mrt && (A < 2 || !pDrawBuffers || !pReadBuffer)) {
            printf("%-14s  unsupported (max_draw_buffers %d, procs %s)\n", name, (int) maxdb,
                   (pDrawBuffers && pReadBuffer) ? "yes" : "no");
            continue;
        }
        // grid height: q1 and mrt one row a head; serial/recomp one row a query
        const int gh = (arm == 0 || mrt) ? heads : R;
        string fs = pre;
        if (mrt) {
            for (int j = 0; j < A; j++)
                fs += "layout(location=" + std::to_string(j) + ") out vec4 c" + std::to_string(j) + ";\n";
        } else {
            fs += "out vec4 frag_color;\n";
        }
        fs += "void main(){int T=int(gl_FragCoord.x);int y=int(gl_FragCoord.y);"
              "int t0=T*2;int t1=t0+1;";
        if (arm == 0) {
            // one query (b=0) a head: the shipped decode shape
            fs += "int h=y;float s0=0.0,s1=0.0;"
                  "for(int i=0;i<" + std::to_string(dim) + ";i++){"
                  "float w=d16(qf(h*" + std::to_string(batch) + ",i).rg);"
                  "s0+=w*d16(kf(h,i,t0).rg);s1+=w*d16(kf(h,i,t1).rg);}"
                  "vec2 p0=e16(s0*invsqrt*256.0),p1=e16(s1*invsqrt*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}";
        } else if (arm == 1) {
            // own query only: N serial passes in one draw
            fs += "int h=y/" + std::to_string(batch) + ";float s0=0.0,s1=0.0;"
                  "for(int i=0;i<" + std::to_string(dim) + ";i++){"
                  "float w=d16(qf(y,i).rg);"
                  "s0+=w*d16(kf(h,i,t0).rg);s1+=w*d16(kf(h,i,t1).rg);}"
                  "vec2 p0=e16(s0*invsqrt*256.0),p1=e16(s1*invsqrt*256.0);"
                  "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}";
        } else {
            // all N queries of the head, one KV fetch; recomp writes row y,
            // mrt writes the first A queries to A attachments.  recomp's grid
            // is one row a query (h=y/batch); mrt's is one row a head (h=y).
            fs += mrt ? "int h=y;" : "int h=y/" + std::to_string(batch) + ";";
            if (arm == 2) fs += "int b=y-h*" + std::to_string(batch) + ";";
            for (int b = 0; b < batch; b++)
                fs += "float s" + std::to_string(b) + "0=0.0,s" + std::to_string(b) + "1=0.0;";
            fs += "for(int i=0;i<" + std::to_string(dim) + ";i++){"
                  "float v0=d16(kf(h,i,t0).rg);float v1=d16(kf(h,i,t1).rg);";
            for (int b = 0; b < batch; b++)
                fs += "float w" + std::to_string(b) + "=d16(qf(h*" + std::to_string(batch) + "+" +
                      std::to_string(b) + ",i).rg);"
                      "s" + std::to_string(b) + "0+=w" + std::to_string(b) + "*v0;"
                      "s" + std::to_string(b) + "1+=w" + std::to_string(b) + "*v1;";
            fs += "}";
            if (arm == 2) {
                // select this fragment's own row out of the N computed
                fs += "float a0=0.0,a1=0.0;";
                for (int b = 0; b < batch; b++)
                    fs += (b ? "else " : "") + string("if(b==") + std::to_string(b) +
                          "){a0=s" + std::to_string(b) + "0;a1=s" + std::to_string(b) + "1;}";
                fs += "vec2 p0=e16(a0*invsqrt*256.0),p1=e16(a1*invsqrt*256.0);"
                      "frag_color=vec4(p0.x,p0.y,p1.x,p1.y);}";
            } else {
                for (int j = 0; j < A; j++)
                    fs += "vec2 a" + std::to_string(j) + "=e16(s" + std::to_string(j) +
                          "0*invsqrt*256.0),b" + std::to_string(j) + "=e16(s" + std::to_string(j) +
                          "1*invsqrt*256.0);c" + std::to_string(j) + "=vec4(a" + std::to_string(j) +
                          ".x,a" + std::to_string(j) + ".y,b" + std::to_string(j) + ".x,b" +
                          std::to_string(j) + ".y);";
                fs += "}";
            }
        }

        string log;
        GLuint hvs = compile(GL_VERTEX_SHADER, vs, log);
        GLuint hfs = compile(GL_FRAGMENT_SHADER, fs, log);
        if (!hvs || !hfs) { printf("%-14s COMPILE FAIL %s\n", name, log.c_str()); continue; }
        GLuint p = glCreateProgram();
        glAttachShader(p, hvs); glAttachShader(p, hfs);
        glBindAttribLocation(p, 0, "pos");
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char b[4096];
            glGetProgramInfoLog(p, sizeof b, nullptr, b);
            printf("%-14s LINK FAIL %s\n", name, b);
            continue;
        }
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, qid);
        glUniform1i(glGetUniformLocation(p, "Q"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, kid);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glUniform1f(glGetUniformLocation(p, "invsqrt"), invsqrt);
        glActiveTexture(GL_TEXTURE0);

        // output: one target, or A attachments on one FBO
        GLuint fbo = 0;
        vector<Out> outs;
        if (mrt) {
            glGenFramebuffers(1, &fbo);
            for (int j = 0; j < A; j++) {
                Out o = make_out(frags, heads);
                // make_out leaves its OWN fbo bound; attach on the outer one
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + j,
                                       GL_TEXTURE_2D, o.tex, 0);
                outs.push_back(o);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            GLenum bufs[8];
            for (int j = 0; j < A; j++) bufs[j] = GL_COLOR_ATTACHMENT0 + j;
            pDrawBuffers(A, bufs);
        } else {
            outs.push_back(make_out(frags, gh));
            fbo = outs[0].fbo;
        }
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            printf("%-14s FBO INCOMPLETE\n", name);
            continue;
        }
        // make_out bound the output texture on unit 0: restore the samplers
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, qid);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, kid);
        glActiveTexture(GL_TEXTURE0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glViewport(0, 0, frags, gh);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();

        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; k++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() / iters;

        // ---- check every row at four KV pairs against the CPU dot product
        int nbad = 0, nchk = 0;
        const int Ts[4] = {0, 1, frags / 2, frags - 1};
        for (int j = 0; j < (mrt ? A : 1); j++) {
            if (mrt) pReadBuffer(GL_COLOR_ATTACHMENT0 + j);
            vector<unsigned char> px((size_t) frags * (mrt ? heads : gh) * 4, 0);
            glReadPixels(0, 0, frags, mrt ? heads : gh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            const int rows = mrt ? heads : gh;
            for (int r = 0; r < rows; r++) {
                // which (head, query) this output row holds
                const int h = mrt ? r : (arm == 0 ? r : r / batch);
                const int b = mrt ? j : (arm == 0 ? 0 : r % batch);
                for (int T : Ts) {
                    const size_t off = ((size_t) r * frags + T) * 4;
                    for (int half = 0; half < 2; half++) {
                        const int t = T * 2 + half;
                        const double got = q16dec(px.data(), (int) off + half * 2);
                        const double want = ref(h, b, t);
                        ++nchk;
                        if (std::fabs(got - want) > 2e-2 * (1.0 + std::fabs(want))) {
                            if (nbad < 4)
                                printf("    row %d (h%d q%d) t%d: got %.4f want %.4f\n",
                                       r, h, b, t, got, want);
                            ++nbad;
                        }
                    }
                }
            }
        }
        // draws to cover the whole batch, and the useful-MAC cost of the set
        const int draws = arm == 0 ? batch : mrt ? (batch + A - 1) / A : 1;
        const double ms_full = ms * draws;
        results[arm] = {name, ms_full, true};
        char grid[32];
        snprintf(grid, sizeof grid, "%dx%d", frags, gh);
        printf("%-14s %10s %9.3f %7d %9.3f %10.2f %8s\n",
               name, grid, ms, draws, ms_full, 1e6 * ms_full / full_macs,
               nbad ? "MISMATCH" : "ok");
        if (nbad) printf("    %d/%d checks failed\n", nbad, nchk);
        if (getenv("ATT_DUMP")) printf("    shader:\n%s\n", fs.c_str());
        glDeleteProgram(p);
        glDeleteShader(hvs); glDeleteShader(hfs);
        for (const Out& o : outs) { glDeleteFramebuffers(1, &o.fbo); glDeleteTextures(1, &o.tex); }
        if (mrt) glDeleteFramebuffers(1, &fbo);
    }

    printf("\nms to produce all %d x %d scores (one prefill group, all heads):\n", R, seq);
    for (const Res& r : results)
        if (r.ran) printf("  %-14s %9.3f\n", r.name, r.ms_full);
    return 0;
}