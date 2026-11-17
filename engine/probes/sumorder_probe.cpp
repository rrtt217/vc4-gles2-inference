// Does the summation *order* matter on this driver, at the magnitudes a small
// model's residual actually reaches?
//
// The question came from MNN-Pocket on the Pi: the same binary writes a full
// sentence on Xe and repeats one token after two or three tokens on the board.
// A per-layer dump localized it to the layers whose residual reaches ~2300,
// where a dot product's terms are ~100x its result -- the sum cancels, and the
// order of the additions decides the answer.  Layer 0 was identical to five
// decimals, so no kernel is wrong: the difference is arithmetic.
//
// This probe measures that arithmetic directly, with no model, no weights and
// no packing.  One dot product of N=1024 terms whose exact value is known in
// closed form, at magnitudes from 1 to 2300, through four accumulation shapes:
//
//   ltr       one accumulator, `s += w*x`                       (naive)
//   ltr8      one accumulator over eight lanes                  (flat q4 form)
//   engine4   four accumulators over eight lanes, combined pairwise per 32-wide
//             group -- `matvec_i4_add`'s actual shape
//   pairwise  a tree over the whole vector                      (best case)
//
// The case: terms in pairs,
//     x[2k] = +M      w[2k] = 1
//     x[2k+1] = -M    w[2k+1] = 1 - 2^-20
// so each pair contributes exactly M * 2^-20, and 512 pairs give M/2048 --
// exactly representable in binary32 and 2^40 times smaller than the terms.  A
// left-to-right f32 sum has to cancel +2300 against -2299.9998 to find 1.12,
// which is the case where the order is the whole story.  The signs come from a
// texture of exact 0/1 bytes (VC4 returns n/255 exactly -- q4fmt_probe), and
// the magnitudes are compile-time literals, so nothing else can round.
//
// The result comes back as a **bit-cast f32** (RGBA8 = the four bytes), which
// needs ESSL 3.00: the probe sets the Mesa version overrides before
// eglInitialize, exactly as `GPU::init_egl()` does.  A driver without it says so
// and exits, rather than reporting a number at 8-bit resolution.
//
// build:  make -C engine probes/sumorder_probe
//         make -C engine probes/sumorder_probe_pi CROSS_SYSROOT=<sysroot>
// Run it on both machines and put the tables side by side.
#include <GLES2/gl2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "egl_surfaceless.h"

static const int N = 1024;                          // terms per dot product
static const int NSHAPE = 6, NMAG = 5;
static const float MAGS[NMAG] = {1.0f, 16.0f, 256.0f, 1024.0f, 2300.0f};
static const float S = 9.5367431640625e-07f;        // 2^-20, exact

static GLuint compile(GLenum type, const std::string& src, const char* what) {
    GLuint h = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0;
    glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {0};
        glGetShaderInfoLog(h, sizeof(log) - 1, nullptr, log);
        printf("%s: %s\n", what, log);
        return 0;
    }
    return h;
}

// x = shape, y = magnitude.  Every shape reads all N texels, so no shape can be
// constant-folded away, and the four differ only in the order they add.
static std::string fragment_src() {
    char defs[256];
    snprintf(defs, sizeof(defs),
             "#version 300 es\nprecision highp float;\n"
             "#define N %d\n#define W1 (1.0-%g)\n"
             "#define MAT0 1.0\n#define MAT1 16.0\n#define MAT2 256.0\n"
             "#define MAT3 1024.0\n#define MAT4 2300.0\n", N, (double)S);
    return std::string(defs) +
        "uniform sampler2D T;\n"
        "out vec4 fragColor;\n"
        // one fetch per term: r = x sign, g = w is 1 (255) or 1-2^-20 (0)
        "void main(){\n"
        "  int shape=int(gl_FragCoord.x);\n"
        "  int row=int(gl_FragCoord.y);\n"
        // No uniform array indexing here: VC4 returns 0 for it (measured), so
        // the magnitude is a compile-time constant per row.
        "  float M=(row==0)?MAT0:(row==1)?MAT1:(row==2)?MAT2:(row==3)?MAT3:MAT4;\n"
        "  float s=0.0;\n"
        "  if(shape==0){\n"
        "    for(int i=0;i<N;i++){vec4 c=texture(T,vec2((float(i)+0.5)/float(N),0.5));"
        "      s+=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);}\n"
        "  } else if(shape==1){\n"
        // eight accumulators, one per lane of an eight-wide fetch -- the flat
        // q4 kernel's form, combined pairwise at the end
        "    float d[8];for(int k=0;k<8;k++)d[k]=0.0;"
        "    for(int i=0;i<N;i+=8){for(int k=0;k<8;k++){"
        "      vec4 c=texture(T,vec2((float(i+k)+0.5)/float(N),0.5));"
        "      float m=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);"
        "      if(k==0)d[0]+=m;else if(k==1)d[1]+=m;else if(k==2)d[2]+=m;else if(k==3)d[3]+=m;"
        "      else if(k==4)d[4]+=m;else if(k==5)d[5]+=m;else if(k==6)d[6]+=m;else d[7]+=m;}}"
        "    s=(((((d[0]+d[1])+(d[2]+d[3]))+((d[4]+d[5])+(d[6]+d[7])))));\n"
        "  } else if(shape==2){\n"
        "    for(int g=0;g<N;g+=32){float d0=0.0,d1=0.0,d2=0.0,d3=0.0;"
        "      for(int q=0;q<4;q++){"
        "        for(int lane=0;lane<4;lane++){int i=g+q*8+lane;"
        "          vec4 c=texture(T,vec2((float(i)+0.5)/float(N),0.5));"
        "          float m=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);"
        "          if(lane==0)d0+=m;else if(lane==1)d1+=m;else if(lane==2)d2+=m;else d3+=m;}}"
        "      s+=(d0+d2)+(d1+d3);}\n"
        "  } else if(shape==3){\n"
        // two accumulators, each fed by a 2-wide tree -- the cheapest shape that
        // still has a lane structure, and the one this probe exists to price
        "    float a=0.0,b=0.0;"
        "    for(int g=0;g<N;g+=8){float t0,t1,t2,t3;"
        "      for(int q=0;q<4;q++){int i=g+q*2;"
        "        vec4 c=texture(T,vec2((float(i)+0.5)/float(N),0.5));"
        "        vec4 d=texture(T,vec2((float(i+1)+0.5)/float(N),0.5));"
        "        float m0=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);"
        "        float m1=((d.r>0.5)?M:-M)*((d.g>0.5)?1.0:W1);"
        "        float t=m0+m1;"
        "        if(q==0)t0=t;else if(q==1)t1=t;else if(q==2)t2=t;else t3=t;}"
        "      a+=(t0+t1);b+=(t2+t3);}"
        "    s=a+b;\n"
        "  } else if(shape==4){\n"
        // one accumulator, 8-wide pairwise tree per group: no lane survives
        // across groups, so no accumulator ever holds a same-sign run
        "    for(int g=0;g<N;g+=8){float t0,t1,t2,t3;"
        "      for(int q=0;q<4;q++){int i=g+q*2;"
        "        vec4 c=texture(T,vec2((float(i)+0.5)/float(N),0.5));"
        "        vec4 d=texture(T,vec2((float(i+1)+0.5)/float(N),0.5));"
        "        float m0=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);"
        "        float m1=((d.r>0.5)?M:-M)*((d.g>0.5)?1.0:W1);"
        "        float t=m0+m1;"
        "        if(q==0)t0=t;else if(q==1)t1=t;else if(q==2)t2=t;else t3=t;}"
        "      s+=(t0+t1)+(t2+t3);}\n"
        "  } else {\n"
        "    for(int g=0;g<N;g+=8){float d0=0.0,d1=0.0,d2=0.0,d3=0.0;"
        "      for(int q=0;q<2;q++){"
        "        for(int lane=0;lane<4;lane++){int i=g+q*4+lane;"
        "          vec4 c=texture(T,vec2((float(i)+0.5)/float(N),0.5));"
        "          float m=((c.r>0.5)?M:-M)*((c.g>0.5)?1.0:W1);"
        "          if(lane==0)d0+=m;else if(lane==1)d1+=m;else if(lane==2)d2+=m;else d3+=m;}}"
        "      s+=(d0+d1)+(d2+d3);}\n"
        "  }\n"
        "  uint u=floatBitsToUint(s);\n"
        "  fragColor=vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
        "float(u&255u))/255.0;\n"
        "}\n";
}

int main() {
    // ESSL 3.00 for the bit-cast, before eglInitialize: the same two variables
    // GPU::init_egl() sets, and set overridably so a caller can force ES2.
    setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
    EglSurfaceless egl = egl_surfaceless_init(2);
    if (!egl.ok()) return 1;
    const char* r = (const char*)glGetString(GL_RENDERER);
    const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    printf("renderer: %s\nglsl: %s\n", r ? r : "?", glsl ? glsl : "?");
    if (!glsl || strncmp(glsl, "OpenGL ES GLSL ES 3", 19) != 0) {
        printf("\nno ESSL 3.00 here: the result comes back as a bit-cast f32, so\n"
               "this probe cannot report a number at 8-bit resolution.  Set\n"
               "MESA_GLES_VERSION_OVERRIDE=3.0 (and the GLSL one) before the\n"
               "context is created, which is what init_egl does.\n");
        return 1;
    }

    // Pattern: one texel a term.  r = sign of x (255 = +M), g = 255 for w=1 and
    // 0 for w=1-2^-20.  Exact bytes either way.
    std::vector<uint8_t> pat((size_t)N * 4, 0);
    for (int i = 0; i < N; ++i) {
        const bool pos = (i % 2) == 0;
        pat[i * 4 + 0] = pos ? 255 : 0;
        pat[i * 4 + 1] = pos ? 255 : 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pat.data());

    GLuint fbo = 0, out = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, NSHAPE, NMAG, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("framebuffer incomplete\n");
        return 1;
    }

    GLuint vs = compile(GL_VERTEX_SHADER,
                        "#version 300 es\n"
                        "in vec2 p;void main(){gl_Position=vec4(p,0.0,1.0);}", "vs");
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragment_src(), "fs");
    if (!vs || !fs) return 1;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[4096] = {0};
        glGetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
        printf("link: %s\n", log);
        return 1;
    }
    const float verts[] = {-1, -1, 3, -1, -1, 3};
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glViewport(0, 0, NSHAPE, NMAG);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(prog, "T"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    std::vector<uint8_t> px((size_t)NSHAPE * NMAG * 4);
    glReadPixels(0, 0, NSHAPE, NMAG, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

    auto gpu_val = [&](int shape, int row) {
        const uint8_t* p = &px[((size_t)row * NSHAPE + shape) * 4];
        const uint32_t u = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        float f = 0;
        memcpy(&f, &u, 4);
        return f;
    };

    // ---- exp() precision: att_v evaluates exp(score*SCSI - mx) per position,
    // so a relative difference here is a multiplicative weight error on every
    // value in the sum, 28 times a token.  Range: the attention's exponent runs
    // from 0 down to about -30; below -87 the term underflows f32 and
    // contributes nothing, so the wider range would measure denormals.
    {
        const int NX = 64;
        std::string fs_src =
            "#version 300 es\nprecision highp float;\n"
            "uniform sampler2D T;out vec4 fragColor;\n"
            "void main(){int i=int(gl_FragCoord.x);"
            "int j=i%16;float x=-30.0*(float(j)+0.5)/float(16);"
            // columns 0..15 exp(), 16..31 exp2(), 32..47 a portable polynomial,
            // 48..63 the same polynomial with its range reduction removed (so a
            // reader can see which half carries the accuracy)
            "float v;"
            "if(i<16)v=exp(x);"
            "else if(i<32)v=exp2(x*1.4426950408889634);"
            "else{"
            "float k=floor(x*1.4426950408889634+0.5);"
            "float r=x-k*0.6931471805599453;"
            "float t=1.0+r*(1.0+r*(0.5+r*(1.0/6.0+r*(1.0/24.0+r*(1.0/120.0+r/720.0)))));"
            "uint u=uint(int(k)+127)<<23;"
            "float p=t*uintBitsToFloat(u);"
            "v=(i<48)?p:t;}"
            "uint u=floatBitsToUint(v);"
            "fragColor=vec4(float(u>>24),float((u>>16)&255u),float((u>>8)&255u),"
            "float(u&255u))/255.0;}\n";
        GLuint fs2 = compile(GL_FRAGMENT_SHADER, fs_src, "fs-exp");
        if (fs2) {
            GLuint p2 = glCreateProgram();
            glAttachShader(p2, vs);
            glAttachShader(p2, fs2);
            glBindAttribLocation(p2, 0, "p");
            glLinkProgram(p2);
            GLuint fbo2 = 0, out2 = 0;
            glGenFramebuffers(1, &fbo2);
            glGenTextures(1, &out2);
            glBindTexture(GL_TEXTURE_2D, out2);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, NX, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out2, 0);
            glViewport(0, 0, NX, 1);
            glUseProgram(p2);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();
            std::vector<uint8_t> ex((size_t)NX * 4);
            glReadPixels(0, 0, NX, 1, GL_RGBA, GL_UNSIGNED_BYTE, ex.data());
            double worst = 0, worst_x = 0, worst2 = 0, worst2_x = 0;
            double worst3 = 0, worst3_x = 0, worst4 = 0, worst4_x = 0;
            int ulp_worst = 0;
            for (int i = 0; i < NX; ++i) {
                const uint8_t* q = &ex[(size_t)i * 4];
                const uint32_t u = ((uint32_t)q[0] << 24) | ((uint32_t)q[1] << 16) |
                                   ((uint32_t)q[2] << 8) | (uint32_t)q[3];
                float f = 0;
                memcpy(&f, &u, 4);
                const int j = i % 16;
                const double x = -30.0 * (j + 0.5) / 16.0;
                const double ref = exp(x);
                const double rel = fabs((double)f - ref) / ref;
                const int which = i / 16;   // 0 exp, 1 exp2, 2 poly, 3 poly-no-reduction
                if (which == 0) { if (rel > worst) { worst = rel; worst_x = x; } }
                else if (which == 1) { if (rel > worst2) { worst2 = rel; worst2_x = x; } }
                else if (which == 2) { if (rel > worst3) { worst3 = rel; worst3_x = x; } }
                else { if (rel > worst4) { worst4 = rel; worst4_x = x; } }
            }
            (void)ulp_worst;
            printf("\nexp(): worst relative error over x in [-30,0], 32 points each:\n"
                   "  exp(x)          %.3e  at x = %.2f\n"
                   "  exp2(x*log2 e)  %.3e  at x = %.2f\n"
                   "  poly + 2^k       %.3e  at x = %.2f\n"
                   "  poly, no 2^k     %.3e  at x = %.2f  (what the reduction is worth)\n"
                   "  (host uses double exp as the reference; the third row is a\n"
                   "   degree-6 minimax on r = x - k ln2 with the exponent built\n"
                   "   by a bit-cast)\n",
                   worst, worst_x, worst2, worst2_x, worst3, worst3_x,
                   worst4, worst4_x);
            glDeleteFramebuffers(1, &fbo2);
            glDeleteTextures(1, &out2);
        }
    }

    const char* names[NSHAPE] = {"ltr", "ltr8", "engine4", "two-tree", "tree8", "pairwise"};
    // Host references, same terms and magnitudes, f32 and f64.
    printf("\n%-6s %-9s %14s %14s %10s %10s %10s\n", "M", "shape", "exact",
           "gpu", "err/exact", "gpu-host", "host/exact");
    for (int row = 0; row < NMAG; ++row) {
        const double M = MAGS[row];
        // 512 pairs, each contributing M * 2^-20 exactly.
        const double exact = (double)(N / 2) * M * (double)S;
        float hf32 = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float x = (i % 2 == 0) ? (float)M : -(float)M;
            const float w = (i % 2 == 0) ? 1.0f : (1.0f - S);
            hf32 += x * w;
        }
        for (int sh = 0; sh < NSHAPE; ++sh) {
            const double g = gpu_val(sh, row);
            printf("%-6.0f %-9s %14.6g %14.6g %9.3f%% %9.3f%% %9.3f%%\n", M, names[sh],
                   exact, g, 100.0 * fabs(g - exact) / fabs(exact),
                   100.0 * fabs(g - hf32) / (fabs(hf32) > 0 ? fabs(hf32) : 1.0),
                   100.0 * fabs(hf32 - exact) / fabs(exact));
        }
    }
    printf("\nexact = 512 * M * 2^-20 (closed form).  'gpu-host' is what a kernel\n"
           "adds on top of the CPU path; 'host/exact' is the CPU's own loss.\n");

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &out);
    glDeleteTextures(1, &tex);
    return 0;
}
