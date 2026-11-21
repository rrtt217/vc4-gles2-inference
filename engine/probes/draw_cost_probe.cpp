// Why does one draw in the engine's decode stream cost 0.48-0.67 ms when
// draw_probe's pipelined null draw costs 0.008-0.019 ms?  **Answered on VC4
// (2026-11-21): the FBO -> texture dependency, and the counters cannot see most
// of it.**
//
// The engine issues 135 draws a token and spends ~126 ms of that token on the
// QPU (PERFCNT regions, docs/benchmarks.md), leaving 65-75 ms over 135 draws.
// This probe isolates the cause by changing one thing at a time on the same
// shader, the same target size and the same draw count:
//
//   arm                ms/draw   frag     scorebd  varyings  waitTMU  idle
//   dep + finish        0.7206   0.12826  0.01942  0.00000   0.01483  0.00000
//   dep, no finish      0.6354   0.12658  0.01881  0.00000   0.01472  0.00000
//   indep + finish      0.6769   0.12742  0.01910  0.00000   0.01481  0.00000
//   indep, no finish    0.1306   0.02133  0.00567  0.00000   0.00205  0.00000
//   (1216 fragments, 8 draws a chain, 10 iterations, VC4 V3D 2.1)
//
// Three things this establishes, two of them negative:
//
//   * **A glFinish is not the cost.**  dep+finish and dep,no-finish differ by
//     0.085 ms -- real, but a seventh of what the dependency costs.  The engine
//     already skips these finishes (layer_fast_finish_), and forcing them back
//     on costs 71% end to end (327 against 191 ms a token), so that lever is
//     spent.
//   * **The dependency is the cost, and it is 5x.**  0.635 against 0.131 ms for
//     identical work; every draw samples the texture the previous draw wrote.
//     Only 0.105 ms of that appears as fragment cycles (TMU reading pixels that
//     have not reached the cache) and 0.013 as scoreboard.
//   * **The other ~0.47 ms is in NO counter.**  `idle` is 0 in every arm, and
//     so is `waiting-varyings` -- both were added to this probe specifically to
//     test the hypothesis that the round trip would show up there, and it does
//     not.  The time is outside the QPU's cycle accounting altogether, which is
//     what a V3D bin/render synchronisation outside the QPU looks like.
//
// That last point is why this probe's result is a *bound* rather than a plan:
// fusion would remove round trips, and the wall time says there is ~0.5 ms a
// draw to win, but no counter can confirm where an individual win lands.  Treat
// a fusion A/B's wall delta as the evidence, not a counter delta.
//
// build: make -C engine probes/draw_cost_probe[_pi]
// run:   PERFCNT=1 ./draw_cost_probe [--frags N] [--iters K] [--draws D]

#include "egl_surfaceless.h"
#include "perfcnt.h"

#include <GLES2/gl2.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

static GLuint compile(GLenum type, const string& src, string& log) {
    GLuint h = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetShaderInfoLog(h, sizeof(b) - 1, nullptr, b);
        log = b;
    }
    return ok ? h : 0;
}

static GLuint build(const string& fs, string& log) {
    const char* vs = "attribute vec2 A;void main(){gl_Position=vec4(A,0.0,1.0);}";
    GLuint v = compile(GL_VERTEX_SHADER, vs, log);
    if (!v) return 0;
    GLuint f = compile(GL_FRAGMENT_SHADER, fs, log);
    if (!f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "A");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetProgramInfoLog(p, sizeof(b) - 1, nullptr, b);
        log = b;
    }
    return ok ? p : 0;
}

// The cheapest fragment program that still samples its input: the shape of a
// pass-to-pass step without any real arithmetic.
static const char* FS =
        "precision highp float;uniform sampler2D S;"
        "void main(){vec2 uv=gl_FragCoord.xy/vec2(WIDTH,1.0);"
        "vec4 v=texture2D(S,uv);"
        "gl_FragColor=vec4(v.x*0.999+0.001,v.y,v.z,v.w);}";

static GLuint make_tex(int w, int h) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    vector<unsigned char> px((size_t)w * 4 * h, 0);
    for (size_t i = 0; i < px.size(); ++i) px[i] = (unsigned char)(i * 37u + 11u);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return t;
}

struct Target2 { GLuint tex = 0, fbo = 0; };

static Target2 make_target(int w) {
    Target2 t;
    t.tex = make_tex(w, 1);
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO incomplete (%dx1)\n", w);
    return t;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int frags = 1216, iters = 30, draws = 8;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frags") && i + 1 < argc) frags = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draws") && i + 1 < argc) draws = atoi(argv[++i]);
        else { printf("usage: %s [--frags N] [--iters K] [--draws D]\n", argv[0]); return 2; }
    }
    if (draws < 2 || draws > 32) { printf("--draws must be 2..32\n"); return 2; }

    EglSurfaceless e = egl_surfaceless_init(2);
    if (!e.ok()) return 1;
    printf("== draw_cost_probe: %d fragments, %d draws a chain, %d iters ==\n",
           frags, draws, iters);
    printf("renderer %s\n", (const char*)glGetString(GL_RENDERER));

    string log;
    string fs(FS);
    const string W = to_string(frags);
    for (size_t pos = fs.find("WIDTH"); pos != string::npos; pos = fs.find("WIDTH"))
        fs.replace(pos, 5, W);
    GLuint prog = build(fs, log);
    if (!prog) { printf("shader failed: %s\n", log.c_str()); return 1; }

    const float tri[6] = {-1, -1, 3, -1, -1, 3};
    GLuint vb = 0;
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    vector<Target2> t(draws);
    for (int i = 0; i < draws; i++) t[(size_t)i] = make_target(frags);
    GLuint fixed_tex = make_tex(frags, 1);

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "S"), 0);
    glViewport(0, 0, frags, 1);

    PerfMon pm;
    const bool want_c = perfcnt_enabled() && !pm.missing();
    if (want_c)
        // The two stalls a FBO -> texture round trip can hide in.  Idle alone
        // cannot tell "the QPU has nothing to do" from "the QPU is waiting for
        // the previous draw's pixels", and those need different fixes.
        pm.pick({"QPU-total-idle-clk-cycles",
                 "QPU-total-clk-cycles-fragment-shading",
                 "QPU-total-clk-cycles-vertex-coord-shading",
                 "QPU-total-clk-cycles-executing-valid-instr",
                 "QPU-total-clk-cycles-waiting-TMU",
                 "QPU-total-clk-cycles-waiting-scoreboard",
                 "QPU-total-clk-cycles-waiting-varyings",
                 "TMU-total-text-quads-processed",
                 "TMU-total-text-cache-miss",
                 "VPM-total-clk-cycles-VDW-stalled",
                 "VPM-total-clk-cycles-VCD-stalled"});
    vector<PerfMon::Row> table;

    // One measured arm: `chain` draw calls per iteration, and a finish policy.
    //   dependent  target i reads target i-1's texture
    //   indep      every target reads the same fixed texture
    //   finish     glFinish after every draw (draw_probe's "issue+finish")
    struct Arm { const char* name; bool dependent; bool per_draw_finish; };
    const Arm arms[] = {
        {"dep + finish",     true,  true},
        {"dep, no finish",   true,  false},
        {"indep + finish",   false, true},
        {"indep, no finish", false, false},
    };

    if (want_c)
        printf("\n%-20s %10s %12s %10s %9s %9s %9s %9s\n", "arm", "ms/chain", "ms/draw",
               "frag", "scorebd", "varyings", "waitTMU", "idle");
    else
        printf("\n%-20s %10s %12s\n", "arm", "ms/chain", "ms/draw");
    for (const Arm& a : arms) {
        // warm up
        for (int k = 0; k < 3; ++k) {
            for (int d = 0; d < draws; d++) {
                glBindFramebuffer(GL_FRAMEBUFFER, t[(size_t)d].fbo);
                glBindTexture(GL_TEXTURE_2D,
                              a.dependent ? (d == 0 ? fixed_tex : t[(size_t)(d - 1)].tex) : fixed_tex);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                if (a.per_draw_finish) glFinish();
            }
        }
        glFinish();
        const auto t0 = chrono::steady_clock::now();
        for (int k = 0; k < iters; ++k) {
            for (int d = 0; d < draws; d++) {
                glBindFramebuffer(GL_FRAMEBUFFER, t[(size_t)d].fbo);
                glBindTexture(GL_TEXTURE_2D,
                              a.dependent ? (d == 0 ? fixed_tex : t[(size_t)(d - 1)].tex) : fixed_tex);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                if (a.per_draw_finish) glFinish();
            }
        }
        glFinish();
        const double chain = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() / iters;
        if (want_c) {
            pm.begin();
            for (int k = 0; k < iters; ++k) {
                for (int d = 0; d < draws; d++) {
                    glBindFramebuffer(GL_FRAMEBUFFER, t[(size_t)d].fbo);
                    glBindTexture(GL_TEXTURE_2D,
                                  a.dependent ? (d == 0 ? fixed_tex : t[(size_t)(d - 1)].tex) : fixed_tex);
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                    if (a.per_draw_finish) glFinish();
                }
            }
            glFinish();
            PerfMon::Row row;
            row.sample = pm.end_and_print(a.name, iters * draws);
            row.wall_ms = chain / draws;
            table.push_back(row);
            const double n = (double)(iters * draws);
            const double to_ms = 1.0 / (12.0 * 300.0 * 1000.0);
            auto val = [&](const char* sub) -> double {
                for (size_t k = 0; k < pm.picked_count(); ++k)
                    if (pm.picked_name(k).find(sub) != string::npos)
                        return (double)row.sample.value[k] * to_ms / n;
                return 0.0;
            };
            printf("%-20s %10.4f %12.4f %10.5f %9.5f %9.5f %9.5f %9.5f\n",
                   a.name, chain, chain / draws,
                   val("clk-cycles-fragment-shading"),
                   val("clk-cycles-waiting-scoreboard"),
                   val("clk-cycles-waiting-varyings"),
                   val("clk-cycles-waiting-TMU"),
                   val("clk-cycles-idle"));
        } else {
            printf("%-20s %10.4f %12.4f\n", a.name, chain, chain / draws);
        }
    }

    // Same arm with the draw count doubled: the marginal cost of ONE more draw
    // in the same stream shape is the number an optimisation can remove.
    {
        for (int mult = 1; mult <= 2; ++mult) {
            const int n = draws * mult;
            for (int k = 0; k < 2; ++k)
                for (int d = 0; d < n; d++) {
                    glBindFramebuffer(GL_FRAMEBUFFER, t[(size_t)(d % draws)].fbo);
                    glBindTexture(GL_TEXTURE_2D, d == 0 ? fixed_tex : t[(size_t)((d - 1) % draws)].tex);
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }
            glFinish();
            const auto t0 = chrono::steady_clock::now();
            for (int k = 0; k < iters; ++k)
                for (int d = 0; d < n; d++) {
                    glBindFramebuffer(GL_FRAMEBUFFER, t[(size_t)(d % draws)].fbo);
                    glBindTexture(GL_TEXTURE_2D, d == 0 ? fixed_tex : t[(size_t)((d - 1) % draws)].tex);
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }
            glFinish();
            const double chain = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() / iters;
            printf("dep,no-finish x%-2d draws: %9.4f ms/chain  %7.4f ms/draw\n", n, chain, chain / n);
        }
    }
    return 0;
}
