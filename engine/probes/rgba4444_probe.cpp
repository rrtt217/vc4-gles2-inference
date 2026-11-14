// Does this driver actually give us a 4-bit-per-channel texture?
//
// The q4 weight path is 4 bits per weight, and every decode of it in the engine
// pays a floor (docs/performance.md, 2026-11-04: the unpack is FLOOR-bound).
// `GL_UNSIGNED_SHORT_4_4_4_4` is the one format where the sampler does the
// split in hardware, so whether it works decides a whole family of designs --
// and whether it works is a driver question, not a spec question.
//
// This probe is STANDALONE ON PURPOSE: EGL + GLES2 only, no engine sources, so
// it builds and runs against the driver while `engine/src` is being split.
// Build: g++ -O2 -std=c++17 -o rgba4444_probe rgba4444_probe.cpp -lEGL -lGLESv2
//
// It answers, in order:
//
//   A  what the driver is (renderer, version, extensions)
//   B  does glTexImage2D accept 4444, in both spellings, and does it say so?
//        ES2 spelling: internalformat GL_RGBA + type GL_UNSIGNED_SHORT_4_4_4_4
//        ES3 spelling: internalformat GL_RGBA4 (what q4fmt_probe used)
//      Each is followed by glGetError(): the 2026-11-04 claim in the docs rests
//      on a probe that never checked this.
//   C  which nibble of the 16-bit texel lands in which shader component
//   D  is the 4-bit value exact: round(c*15) == n for all 16, and the residual
//      in units of 2^-20 (the "one ULP low" fact the docs quote)
//   E  the real shape: a 1216 x 768 q4 matrix as RGBA4, sampled at known
//      positions and compared against the bytes it was built from.  This is
//      where a row-stride or tiling surprise would show up.
//   F  is 4444 *renderable* (GL_RGBA4 renderbuffer).  vc4_formats.c lists the
//      render-target type as RT_NO.  On VC4 the API disagrees -- storage is
//      accepted, the FBO is complete and GL_RENDERBUFFER_INTERNAL_FORMAT even
//      reports GL_RGBA4 -- while a 0.1/0.5/0.9 clear reads back 26/128/230 on
//      VC4 and 34/136/238 on Xe, i.e. VC4 substituted an 8-bit surface.  The
//      clear is the measurement; the three queries above it are not.
#include "egl_surfaceless.h"

#include <GLES2/gl2ext.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const char* err_name(GLenum e) {
    switch (e) {
    case GL_NO_ERROR: return "GL_NO_ERROR";
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    default: return "other";
    }
}

static int gl_errors(const char* what) {
    int n = 0;
    for (GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError()) {
        printf("   %s: %s (0x%x)\n", what, err_name(e), e);
        if (++n > 8) break;
    }
    return n;
}

static GLuint compile(GLenum t, const char* s) {
    GLuint q = glCreateShader(t);
    glShaderSource(q, 1, &s, nullptr);
    glCompileShader(q);
    GLint ok = 0;
    glGetShaderiv(q, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(q, sizeof(log), nullptr, log);
        printf("SHADER FAIL: %s\n", log);
    }
    return q;
}

static GLuint program(const char* name, const char* fs) {
    const char* vs = "attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        printf("LINK FAIL %s: %s\n", name, log);
        return 0;
    }
    printf("PROGRAM %s: linked\n", name);
    return p;
}

static void params() {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

// floor() makes the pixel->texel map portable: gl_FragCoord.x is 0-based
// integer on vc4 and x+0.5 on Xe, and floor() gives 0 for both.
// The sample is reported as round(c*15) in .r, and the raw nibble residual
// (scaled by 2^20 around 128) in .g, so one readback answers both C and D.
static const char* FS_SAMPLE =
    "precision highp float;uniform sampler2D W;uniform float w;"
    "void main(){float i=floor(gl_FragCoord.x);"
    "vec4 c=texture2D(W,vec2((i+0.5)/w,0.5));"
    "float n15=c.r*15.0;float v=floor(n15+0.5);"
    "float res=clamp(128.0+(n15-v)*1048576.0,0.0,255.0);"
    "gl_FragColor=vec4(v/15.0,res/255.0,"
    "floor(c.g*15.0+0.5)/15.0,floor(c.a*15.0+0.5)/15.0);}";

// Four channels of one texel, as nibbles 0..15.
static const char* FS_TEXEL =
    "precision highp float;uniform sampler2D W;uniform float w;"
    "void main(){float i=floor(gl_FragCoord.x);"
    "vec4 c=texture2D(W,vec2((i+0.5)/w,0.5));"
    "float r=floor(c.r*15.0+0.5),g=floor(c.g*15.0+0.5),"
    "b=floor(c.b*15.0+0.5),a=floor(c.a*15.0+0.5);"
    "gl_FragColor=vec4(r,g,b,a)/15.0;}";

// Row r of a tall texture, column i: the q4 use case, one fragment per texel.
static const char* FS_ROW =
    "precision highp float;uniform sampler2D W;uniform float w,row,n;"
    "void main(){float i=floor(gl_FragCoord.x);"
    "vec4 c=texture2D(W,vec2((i+0.5)/w,(row+0.5)/n));"
    "float r=floor(c.r*15.0+0.5),g=floor(c.g*15.0+0.5),"
    "b=floor(c.b*15.0+0.5),a=floor(c.a*15.0+0.5);"
    "gl_FragColor=vec4(r,g,b,a)/15.0;}";

struct Target { GLuint tex = 0, fbo = 0; };
static Target make_target(int w, int h) {
    Target t;
    glGenTextures(1, &t.tex);
    glBindTexture(GL_TEXTURE_2D, t.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    params();
    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) printf("FBO incomplete 0x%x at %dx%d\n", st, w, h);
    return t;
}

static GLuint vbo = 0;
static void draw(GLuint prog, GLuint tex, Target& out, int w, int h) {
    if (!prog) { printf("   draw skipped: program is 0\n"); return; }
    glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
    glViewport(0, 0, w, h);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "W"), 0);
    glUniform1f(glGetUniformLocation(prog, "w"), (float)w);
    GLuint loc = glGetAttribLocation(prog, "p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
}

static std::vector<uint8_t> read_back(int w, int h) {
    std::vector<uint8_t> px((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return px;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    EglSurfaceless egl = egl_surfaceless_init(2);
    if (!egl.ok()) return 1;

    // ---- A: what are we on? ------------------------------------------------
    printf("A renderer  %s\n", glGetString(GL_RENDERER));
    printf("A version   %s\n", glGetString(GL_VERSION));
    printf("A glsl      %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    const char* ext = (const char*)glGetString(GL_EXTENSIONS);
    printf("A ext       %d bytes; texture_integer: %s; draw_buffers: %s\n",
           ext ? (int)strlen(ext) : 0,
           ext && strstr(ext, "EXT_texture_integer") ? "yes" : "no",
           ext && strstr(ext, "EXT_draw_buffers") ? "yes" : "no");

    static GLuint vbo_local = 0;
    glGenBuffers(1, &vbo_local);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_local);
    float verts[] = { -1, -1, 3, -1, -1, 3 };
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    vbo = vbo_local;
    while (glGetError() != GL_NO_ERROR) {}

    Target t16 = make_target(16, 1);
    GLuint prog_sample = program("sample", FS_SAMPLE);
    GLuint prog_texel = program("texel", FS_TEXEL);
    GLuint prog_row = program("row", FS_ROW);
    if (!prog_sample || !prog_texel || !prog_row) { printf("A program did not link\n"); return 1; }
    if (gl_errors("setup")) return 1;

    // ---- B: does the upload work, in both spellings? -----------------------
    // Texel t = (R=t, G=t, B=t, A=t) -- symmetric, so B only answers "accepted".
    std::vector<uint16_t> same(16);
    for (int t = 0; t < 16; ++t) same[t] = (uint16_t)((t << 12) | (t << 8) | (t << 4) | t);

    GLuint tex_es2 = 0, tex_es3 = 0;
    glGenTextures(1, &tex_es2);
    glBindTexture(GL_TEXTURE_2D, tex_es2);
    params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,
                 same.data());
    const int e2 = gl_errors("B ES2 spelling (GL_RGBA + UNSIGNED_SHORT_4_4_4_4)");
    printf("B ES2 spelling accepted: %s\n", e2 ? "NO" : "yes");

    glGenTextures(1, &tex_es3);
    glBindTexture(GL_TEXTURE_2D, tex_es3);
    params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA4, 16, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,
                 same.data());
    const int e3 = gl_errors("B ES3 spelling (internalformat GL_RGBA4)");
    printf("B ES3 spelling accepted: %s\n", e3 ? "NO" : "yes");

    // Sample both through the same shader: a refused upload leaves the texture
    // incomplete, which returns (0,0,0,1) and shows up as v=0 everywhere.
    const char* names[2] = { "ES2", "ES3" };
    GLuint texes[2] = { tex_es2, tex_es3 };
    for (int k = 0; k < 2; ++k) {
        draw(prog_sample, texes[k], t16, 16, 1);
        std::vector<uint8_t> px = read_back(16, 1);
        int zero = 0, exact = 0;
        for (int t = 0; t < 16; ++t) {
            const int v = px[(size_t)t * 4];
            if (v == 0) zero++;
            if (v == (int)(t * 17)) exact++;
        }
        printf("B %s sampled: v=0 at %d/16 texels, exact 17*n at %d/16, v[0..3]=%d,%d,%d,%d\n",
               names[k], zero, exact, px[0], px[4], px[8], px[12]);
    }

    // ---- C: which nibble goes where ---------------------------------------
    // Every texel carries four distinct markers: 1 in bits 12-15, 2 in 8-11,
    // 4 in 4-7, 8 in 0-3.  Whichever component returns which number names the
    // mapping.  This is the fact the "reversed column order" note rests on.
    std::vector<uint16_t> mark(16, (uint16_t)((1u << 12) | (2u << 8) | (4u << 4) | 8u));
    GLuint tex_mark = 0;
    glGenTextures(1, &tex_mark);
    glBindTexture(GL_TEXTURE_2D, tex_mark);
    params();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,
                 mark.data());
    gl_errors("C upload");
    draw(prog_texel, tex_mark, t16, 16, 1);
    {
        std::vector<uint8_t> px = read_back(16, 1);
        const int r = px[0] / 17, g = px[1] / 17, b = px[2] / 17, a = px[3] / 17;
        auto where = [](int m) {
            switch (m) {
            case 1: return "bits 12-15";
            case 2: return "bits 8-11";
            case 4: return "bits 4-7";
            case 8: return "bits 0-3";
            default: return "??";
            }
        };
        printf("C mapping   R<-%s G<-%s B<-%s A<-%s\n", where(r), where(g), where(b), where(a));
        printf("C verdict   %s\n",
               (r == 1 && g == 2 && b == 4 && a == 8)
                   ? "R is the MOST significant nibble (GL pack order, MSB first)"
                   : "not the MSB-first packing assumed by the docs");
    }

    // ---- D: exactness over all 16 values ----------------------------------
    draw(prog_sample, tex_es2, t16, 16, 1);
    {
        std::vector<uint8_t> px = read_back(16, 1);
        int bad = 0;
        printf("D   n  round(c*15)  resid(2^-20)\n");
        for (int t = 0; t < 16; ++t) {
            const int v = px[(size_t)t * 4];
            const int resid = (int)px[(size_t)t * 4 + 1] - 128;
            if (v != t * 17) bad++;
            printf("D %3d  %3d          %+4d%s\n", t, v, resid, v == t * 17 ? "" : "  MISMATCH");
        }
        printf("D verdict   round(c*15)==n at %d/16; a decode needs floor if this is not 16/16\n",
               16 - bad);
    }

    // ---- E: the shape the engine would actually use -----------------------
    // 1216 rows x 768 weights = 192 RGBA4 texels per row (384 bytes, cols/2).
    // Build the bytes exactly as mmcf_hf.py does (even column in the low
    // nibble), then check that texel (r, 2j..2j+1) returns what the bytes say.
    {
        const int rows = 1216, cols = 768, tw = cols / 4;   // 192 texels per row
        std::vector<uint8_t> bytes((size_t)rows * cols / 2);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols / 2; ++c) {
                const int even = (r * 7 + c * 13) & 15, odd = (r * 5 + c * 11) & 15;
                bytes[(size_t)r * (cols / 2) + c] = (uint8_t)(even | (odd << 4));
            }
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        params();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, rows, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,
                     bytes.data());
        const int eb = gl_errors("E 1216x768 upload");
        printf("E upload    1216 rows x %d texels (%zu bytes): %s\n", tw, bytes.size(),
               eb ? "REFUSED" : "accepted");
        if (!eb) {
            int bad = 0, checked = 0;
            const int sample_rows[] = { 0, 1, 5, 608, 1214, 1215 };
            for (int sr : sample_rows) {
                Target tout = make_target(tw, 1);
                glBindFramebuffer(GL_FRAMEBUFFER, tout.fbo);
                glViewport(0, 0, tw, 1);
                glUseProgram(prog_row);
                glUniform1i(glGetUniformLocation(prog_row, "W"), 0);
                glUniform1f(glGetUniformLocation(prog_row, "w"), (float)tw);
                glUniform1f(glGetUniformLocation(prog_row, "row"), (float)sr);
                glUniform1f(glGetUniformLocation(prog_row, "n"), (float)rows);
                GLuint loc = glGetAttribLocation(prog_row, "p");
                glEnableVertexAttribArray(loc);
                glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, tex);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glFinish();
                std::vector<uint8_t> px = read_back(tw, 1);
                for (int j : { 0, 1, 7, 95, 190, 191 }) {
                    const uint8_t b0 = bytes[(size_t)sr * (cols / 2) + 2 * j];
                    const uint8_t b1 = bytes[(size_t)sr * (cols / 2) + 2 * j + 1];
                    const int want[4] = { (b1 >> 4) & 15, b1 & 15, (b0 >> 4) & 15, b0 & 15 };
                    const int got[4] = { px[(size_t)j * 4] / 17, px[(size_t)j * 4 + 1] / 17,
                                         px[(size_t)j * 4 + 2] / 17, px[(size_t)j * 4 + 3] / 17 };
                    checked += 4;
                    for (int k = 0; k < 4; ++k)
                        if (want[k] != got[k]) {
                            if (bad < 6)
                                printf("E mismatch  row %d texel %d chan %d: want %d got %d\n",
                                       sr, j, k, want[k], got[k]);
                            bad++;
                        }
                }
                glDeleteFramebuffers(1, &tout.fbo);
                glDeleteTextures(1, &tout.tex);
            }
            printf("E verdict   %d/%d sampled nibbles match the packed bytes\n", checked - bad,
                   checked);
        }
    }

    // ---- F: is 4444 renderable? -------------------------------------------
    // "accepted" is not the answer.  ES2 requires RGBA4 as a renderbuffer
    // format, so a driver that cannot render to 4444 has three legal-looking
    // options and only one honest one: refuse, substitute a wider format
    // silently, or render to it.  Mesa's state tracker substitutes, so the
    // *storage* counter below is the measurement and the acceptance is only a
    // hint.  0.1/0.5/0.9 through a real 4444 surface quantises to a multiple of
    // 1/15 (17, 136, 238 in bytes); through an 8-bit one it does not (26, 128,
    // 230).
    {
        GLuint rb = 0, fb = 0;
        glGenRenderbuffers(1, &rb);
        glBindRenderbuffer(GL_RENDERBUFFER, rb);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA4, 64, 4);
        const int ef = gl_errors("F glRenderbufferStorage(GL_RGBA4)");
        GLint got = 0;
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &got);
        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
        const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        const char* gn = got == GL_RGBA4 ? "GL_RGBA4" : got == GL_RGBA ? "GL_RGBA" : "something else";
        printf("F renderable  RGBA4 renderbuffer: %s; FBO %s (0x%x); storage 0x%x = %s\n",
               ef ? "REFUSED" : "accepted",
               st == GL_FRAMEBUFFER_COMPLETE ? "complete" : "INCOMPLETE", st, got, gn);
        if (st == GL_FRAMEBUFFER_COMPLETE) {
            glViewport(0, 0, 4, 4);
            glClearColor(0.1f, 0.5f, 0.9f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glFinish();
            std::vector<uint8_t> px = read_back(4, 4);
            const int r = px[0], g = px[1], b = px[2];
            const bool q4 = (r % 17 == 0 && g % 17 == 0 && b % 17 == 0);
            printf("F precision   clear 0.1/0.5/0.9 reads %d/%d/%d -> %s\n", r, g, b,
                   q4 ? "4-bit surface (a multiple of 1/15)" : "8-bit surface (format substituted)");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    printf("done\n");
    return 0;
}
