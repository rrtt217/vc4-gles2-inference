// When exactly does VC4 drop a draw?  The minimum condition, isolated.
//
// The engine-side trace (VP_TRACE, 2026-11-21) pinned the trigger to a
// *full-target-width* viewport: a 1536-row logits tile drawn with
// glViewport(0, 0, 1536, 1) into a 1536-wide target is silently discarded,
// while the same rows split as 1535 + 1 land.  That is `w == target_w`, not
// "1536 wide" -- the earlier record's "a 1536-wide draw" was a 1536-*row*
// matvec, and qkv (1280 rows) was never the victim.
//
// This probe reduces it to the smallest possible case, so the condition can be
// tested against the driver source without the engine in the way:
//
//   A  w == target_w          (the suspect)
//   B  w == target_w - 1      (the control the engine uses today)
//   C  w == target_w, target a different width (is it 1536-specific?)
//   D  w == target_w twice in a row (does the first land and the rest drop?)
//   E  w == target_w with MESA_DEBUG-style checks (see the notes)
//
// Each arm draws into a fresh target pre-filled with a known pattern, reads it
// back, and reports how many texels changed.  A discarded draw leaves the
// pattern in place -- that is the signature the engine saw with DRAW_CLEAR.
//
// build: make -C engine probes/vc4_draw_drop_probe[_pi]
// run:   ./vc4_draw_drop_probe [--iters K]
#include "egl_surfaceless.h"

#include <GLES2/gl2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

static GLuint compile(GLenum type, const string& src) {
    GLuint h = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char b[4096] = {0};
        glGetShaderInfoLog(h, sizeof(b) - 1, nullptr, b);
        printf("compile failed: %s\n", b);
    }
    return ok ? h : 0;
}

static GLuint build() {
    const char* vs = "attribute vec2 A;void main(){gl_Position=vec4(A,0.0,1.0);}";
    // Writes a constant: the target's contents after the draw say whether it
    // ran, without any dependence on inputs.
    const char* fs =
            "precision highp float;\n"
            "void main(){gl_FragColor=vec4(1.0,0.5,0.25,1.0);}";
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) return 0;
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
        printf("link failed: %s\n", b);
        return 0;
    }
    return p;
}

// The clear pattern: 0x33/0x66/0x99/0xff per channel.  A draw that ran writes
// 255/128/64/255, so "unchanged" is unambiguous.
static const unsigned char kPat[4] = {0x33, 0x66, 0x99, 0xff};
static const unsigned char kOut[4] = {0xff, 0x80, 0x40, 0xff};

struct Target3 { GLuint tex = 0, fbo = 0; };

static Target3 make_target(int w, int h) {
    Target3 t;
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
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO incomplete %dx%d\n", w, h);
    return t;
}

static void fill(Target3& t, int w, int h) {
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    vector<unsigned char> px((size_t)w * h * 4);
    for (size_t i = 0; i < px.size(); i += 4)
        memcpy(&px[i], kPat, 4);
    glViewport(0, 0, w, h);
    glClearColor(kPat[0] / 255.0f, kPat[1] / 255.0f, kPat[2] / 255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

// Returns how many texels of the first row are NOT the clear pattern.
static int changed(Target3& t, int w) {
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glViewport(0, 0, w, 1);
    vector<unsigned char> px((size_t)w * 4, 0);
    glFinish();
    glReadPixels(0, 0, w, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    int n = 0;
    for (int i = 0; i < w; ++i)
        if (px[(size_t)i * 4] != kPat[0] || px[(size_t)i * 4 + 1] != kPat[1]) ++n;
    return n;
}

static void draw_span(int x0, int w) {
    glViewport(x0, 0, w, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// The engine's draw(): full target width, one row tall, at a y offset.
static void draw_span_fullrow(int w, int row) {
    glViewport(0, row, w, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = 5;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else { printf("usage: %s [--iters K]\n", argv[0]); return 2; }

    EglSurfaceless e = egl_surfaceless_init(2);
    if (!e.ok()) return 1;
    printf("== vc4_draw_drop_probe ==\nrenderer %s\nversion %s\n",
           (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION));
    GLint mts = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mts);
    printf("GL_MAX_TEXTURE_SIZE %d\n", (int)mts);

    GLuint prog = build();
    if (!prog) return 1;
    glUseProgram(prog);
    const float tri[6] = {-1, -1, 3, -1, -1, 3};
    GLuint vb = 0;
    glGenBuffers(1, &vb);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Widths to test: the engine's suspect (1536) plus neighbours, so a
    // 1536-only effect is separable from "any full width".
    const int widths[] = {1536, 1535, 1537, 1280, 1024, 2040};
    const int nw = (int)(sizeof widths / sizeof widths[0]);
    // A second geometry, to test whether the trigger is the width or the
    // target height (the logits tile is 1536 wide and `rows` tall).
    const int heights[] = {1, 4};
    const int nh = 2;

    printf("\n%-7s %-7s %-11s %-11s %s\n", "target_w", "target_h", "draw w", "texels changed", "verdict");
    for (int wi = 0; wi < nw; ++wi) {
        for (int hi = 0; hi < nh; ++hi) {
            const int W = widths[wi], H = heights[hi];
            Target3 t = make_target(W, H);
            // (1) full target width, drawn once
            fill(t, W, H);
            glFinish();
            draw_span(0, W);
            glFinish();
            const int full = changed(t, W);
            printf("%-7d %-7d %-11d %-11d %s\n", W, H, W, full,
                   full == 0 ? "DROPPED" : (full == W ? "ok" : "partial"));
            // (2) one pixel narrower: the engine's control
            fill(t, W, H);
            glFinish();
            draw_span(0, W - 1);
            glFinish();
            const int one = changed(t, W);
            printf("%-7d %-7d %-11d %-11d %s\n", W, H, W - 1, one,
                   one == 0 ? "DROPPED" : (one == W - 1 ? "ok" : "partial"));
            glDeleteFramebuffers(1, &t.fbo);
            glDeleteTextures(1, &t.tex);
        }
    }

    // (2b) The engine's exact shape: a tall target, one row per tile, each row
    // drawn at its own y offset.  The logits target is 1536 x (number of
    // tiles), and draw() emits glViewport(x0, row, w, 1) -- so a "1 row tall"
    // draw still lands in a 5-row-tall target, which the arms above do not
    // cover.
    for (int tiles = 1; tiles <= 5; ++tiles) {
        const int W = 1536, H = tiles;
        Target3 t = make_target(W, H);
        fill(t, W, H);
        glFinish();
        for (int row = 0; row < H; ++row) draw_span_fullrow(W, row);
        glFinish();
        // Count changed texels over the whole target, not just row 0.
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        vector<unsigned char> px((size_t)W * H * 4, 0);
        glFinish();
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int changed_rows = 0;
        for (int row = 0; row < H; ++row) {
            int n = 0;
            for (int i = 0; i < W; ++i) {
                const size_t o = ((size_t)row * W + i) * 4;
                if (px[o] != kPat[0] || px[o + 1] != kPat[1]) ++n;
            }
            if (n) ++changed_rows;
            printf("tiles %d row %d: %d/%d changed  (%s)\n", tiles, row, n, W,
                   n == 0 ? "DROPPED" : "landed");
        }
        printf("tiles %d: %d/%d rows landed  (%s)\n", tiles, changed_rows, H,
               changed_rows == H ? "ok" : "DROPPED");
        glDeleteFramebuffers(1, &t.fbo);
        glDeleteTextures(1, &t.tex);
    }

    // (3) the same full-width draw repeated `iters` times into one target.
    // The engine saw "the first lands, every later one is dropped"; this is
    // the direct test of that.
    {
        const int W = 1536, H = 1;
        Target3 t = make_target(W, H);
        fill(t, W, H);
        glFinish();
        for (int k = 0; k < iters; ++k) {
            draw_span(0, W);
            glFinish();
            const int c = changed(t, W);
            printf("repeat %d: %d/%d changed  (%s)\n", k, c, W, c == 0 ? "dropped" : "landed");
            fill(t, W, H);   // reset so each repeat is independent
            glFinish();
        }
        glDeleteFramebuffers(1, &t.fbo);
        glDeleteTextures(1, &t.tex);
    }
    return 0;
}
