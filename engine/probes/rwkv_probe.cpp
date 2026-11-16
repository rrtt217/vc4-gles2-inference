// Does the RWKV-7 GPU elementwise chain's texture traffic work on this driver?
//
// The fused RWKV path (P6) is fine on Xe but collapses to garbage on VC4, and
// the hybrid (CPU f32 WKV state) collapses too -- so the culprit is upstream of
// the state: the 2D NPOT constant textures and/or the I24_1 elementwise
// render->fetch round trip.  Xe's lenient NPOT / no-dither hides it.  This probe
// drives each pattern with data the CPU knows, at the engine's real widths, and
// names the first diverging case.  Sections A/B/C run the *shipped* shader text
// (shaders.cpp linked in); D is a minimal bit-cast round trip.
//
//   A  2D NPOT q24 constant fetch (768 x 6/3/2): byte-exact via a raw fetch and
//      via the shipped rwkv_lerp
//   B  I24_1 render -> readback and render -> texture fetch, byte-exact,
//      GL_DITHER on and off
//   C  shipped rwkv_sqrelu / rwkv_lerp vs a CPU f32 reference
//   D  bit-cast f32 render -> fetch round trip (ES3), GL_DITHER on and off
//
// build: make -C engine probes/rwkv_probe  /  probes/rwkv_probe_pi
// run:   ./rwkv_probe [gles2|es3]   (default es3: the MESA override RWKV_GPU asks)
#include "shaders.h"
#include "xformat.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

bool gpu_q4_shift() { return false; }

#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
typedef EGLDisplay (*PFNGPD)(EGLenum, void*, const EGLint*);

static int failures = 0;
static bool g_es3 = false;

static GLuint comp(GLenum t, const string& s) {
    GLuint h = glCreateShader(t);
    const char* p = s.c_str();
    glShaderSource(h, 1, &p, nullptr);
    glCompileShader(h);
    GLint ok = 0; glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(h, sizeof log, nullptr, log);
        printf("    compile error: %s\n", log); }
    return h;
}

// es3ify mirrors gpu_prog.cpp: add the #version line, the out declaration and
// the gl_FragColor alias; the vertex stage is upgraded to match.
static string es3ify(const string& src, bool vertex) {
    if (src.find("#version") != string::npos) return src;
    if (vertex) return "#version 300 es\n#define attribute in\n" + src;
    return "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
           "#define gl_FragColor frag_color\n" + src;
}

static GLuint prog(const string& fs_in, bool es3 = false) {
    string fs = es3 ? es3ify(fs_in, false) : fs_in;
    string vs = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    if (es3) vs = es3ify(vs, true);
    GLuint p = glCreateProgram();
    glAttachShader(p, comp(GL_VERTEX_SHADER, vs));
    glAttachShader(p, comp(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, nullptr, log);
        printf("    link error: %s\n", log); return 0; }
    return p;
}

static GLuint tex(int w, int h, const vector<uint8_t>& px) {
    GLuint t = 0; glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    return t;
}

static void tri(GLuint p) {
    static GLuint vbo = 0;
    if (!vbo) { glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float v[] = {-1, -1, 3, -1, -1, 3};
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    } else glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLint loc = glGetAttribLocation(p, "p");
    glEnableVertexAttribArray((GLuint)loc);
    glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

static GLuint out_tex = 0, out_fbo = 0; static vector<uint8_t> got; static int out_w = 0, out_h = 0;
static void out_begin(int w, int h = 1) {
    if (out_w != w || out_h != h) {
        if (!out_tex) { glGenTextures(1, &out_tex); glGenFramebuffers(1, &out_fbo); }
        glBindTexture(GL_TEXTURE_2D, out_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            printf("    FBO incomplete %dx%d\n", w, h);
        got.assign((size_t)w * h * 4, 0); out_w = w; out_h = h;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);
    glViewport(0, 0, w, h);
}
static const uint8_t* out_end() {
    glFinish();
    glReadPixels(0, 0, out_w, out_h, GL_RGBA, GL_UNSIGNED_BYTE, got.data());
    return got.data();
}

// q24 raw units (bias removed) from a 1D target.
static vector<int> read_q24(int n) {
    const uint8_t* g = out_end(); vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        int raw = (g[i*4+1] << 16) | (g[i*4+2] << 8) | g[i*4+3];
        v[i] = raw - 8388608;
    }
    return v;
}
static void report(const char* name, const vector<int>& gv, const vector<int>& want, int tol = 0) {
    int bad = 0, first = -1, worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        int d = std::abs(gv[i] - want[i]); if (d > worst) worst = d;
        if (d > tol) { ++bad; if (first < 0) first = (int)i; }
    }
    printf("  %-34s %s bad=%d/%zu worst|d| %d", name, bad ? "FAIL" : "PASS", bad, want.size(), worst);
    if (bad) printf("  [%d] got %d want %d", first, gv[first], want[first]);
    printf("\n"); if (bad) ++failures;
}
// q24 texture bytes for raw value r (x_i24enc layout: top byte 0).
static vector<uint8_t> q24_bytes(const vector<int>& raw, int w, int h) {
    vector<uint8_t> px((size_t)w * h * 4, 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        long q = (long)raw[i] + 8388608L;
        px[i*4+0] = 0; px[i*4+1] = (uint8_t)((q >> 16) & 255);
        px[i*4+2] = (uint8_t)((q >> 8) & 255); px[i*4+3] = (uint8_t)(q & 255);
    }
    return px;
}

static void bind(GLuint p, const char* nm, int unit, GLuint t) {
    glUniform1i(glGetUniformLocation(p, nm), unit);
    glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, t);
}

// A. 2D NPOT q24 constant fetch: copy texel (i, row)'s bytes straight out, so a
// wrong row/column on a NEAREST NPOT fetch shows as a byte mismatch.
static void test_const2d(int dim, int rows) {
    vector<int> raw((size_t)dim * rows);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < dim; ++i)
            raw[(size_t)r*dim+i] = (int)((i*2654435761u + r*7919u) % 16777216u) - 8388608;
    vector<uint8_t> px = q24_bytes(raw, dim, rows);
    GLuint kt = tex(dim, rows, px);
    for (int r = 0; r < rows; ++r) {
        string fs = "precision highp float;uniform sampler2D K;"
            "void main(){int i=int(gl_FragCoord.x);"
            "gl_FragColor=texture2D(K,vec2((float(i)+0.5)/" + std::to_string(dim) + ".0,(float(" +
            std::to_string(r) + ")+0.5)/" + std::to_string(rows) + ".0));}";
        GLuint p = prog(fs); if (!p) { printf("  A link fail\n"); ++failures; return; }
        out_begin(dim); glUseProgram(p); bind(p, "K", 0, kt); tri(p);
        const uint8_t* g = out_end();
        int bad = 0, first = -1;
        for (int i = 0; i < dim; ++i) {
            const uint8_t* w = &px[(size_t)(r*dim+i)*4]; const uint8_t* o = &g[(size_t)i*4];
            if (w[0]!=o[0]||w[1]!=o[1]||w[2]!=o[2]||w[3]!=o[3]) { ++bad; if(first<0)first=i; }
        }
        char nm[64]; snprintf(nm, sizeof nm, "A const2d %dx%d row%d", dim, rows, r);
        printf("  %-34s %s bad=%d/%d", nm, bad?"FAIL":"PASS", bad, dim);
        if (bad) printf("  [%d] got %02x%02x%02x%02x want %02x%02x%02x%02x", first,
            g[first*4],g[first*4+1],g[first*4+2],g[first*4+3],
            px[(size_t)(r*dim+first)*4],px[(size_t)(r*dim+first)*4+1],px[(size_t)(r*dim+first)*4+2],px[(size_t)(r*dim+first)*4+3]);
        printf("\n"); if (bad) ++failures;
        glDeleteProgram(p);
    }
    glDeleteTextures(1, &kt);
}

// A2. The shipped rwkv_lerp reads L as 2D: with xn=0, shift=1.0 the output is
// exactly the lerp constant, so this names a wrong L row through the real kernel.
static void test_lerp_const(int dim, int which, int nrows) {
    vector<int> lraw((size_t)nrows*dim);
    // raw q24 units: texture value = lraw/256 = lp*64 (the *64 convention), so
    // lp = lraw/16384 in [0,1).  With xn=0, shift=1 the lerp output is lp, stored
    // as raw lp*256 = lraw/64.
    for (int r = 0; r < nrows; ++r) for (int i = 0; i < dim; ++i)
        lraw[(size_t)r*dim+i] = (int)((i*40503u + r*65537u) % 16384u);
    GLuint lt = tex(dim, nrows, q24_bytes(lraw, dim, nrows));
    vector<int> zero(dim, 0), one(dim, 65536); // xn=0, shift=1.0 (lerp reads XN/SH /256, stored *256 -> raw=1*65536)
    GLuint xnt = tex(dim, 1, q24_bytes(zero, dim, 1));
    GLuint sht = tex(dim, 1, q24_bytes(one, dim, 1));
    GLuint p = prog(rwkv_lerp(dim, which, nrows)); if (!p) { printf("  lerp link fail\n"); ++failures; return; }
    out_begin(dim); glUseProgram(p);
    bind(p, "XN", 0, xnt); bind(p, "SH", 1, sht); bind(p, "L", 2, lt); tri(p);
    vector<int> g = read_q24(dim);
    vector<int> want(dim);
    for (int i = 0; i < dim; ++i) want[i] = lraw[(size_t)which*dim+i] / 64; // lp*256 = lraw/64
    char nm[64]; snprintf(nm, sizeof nm, "A2 rwkv_lerp(%d) which%d", dim, which);
    report(nm, g, want, 1);   // +-1 is the q24 encode rounding; a wrong L row is >>1
    glDeleteProgram(p); glDeleteTextures(1,&lt); glDeleteTextures(1,&xnt); glDeleteTextures(1,&sht);
}

// B. I24_1 render->readback and upload->fetch->render, byte exact, dither A/B.
static void test_i24(int dim, bool dither) {
    if (dither) glEnable(GL_DITHER); else glDisable(GL_DITHER);
    // B1: render x_q24enc(v(i)), read back raw, compare to CPU round(v*256).
    string fs1 = string("precision highp float;\n") + x_glsl(XFormat::I24_1) +
        "void main(){int i=int(gl_FragCoord.x);float v=sin(float(i)*0.021)*50.0;"
        "gl_FragColor=x_q24enc(v);}";
    GLuint p1 = prog(fs1); out_begin(dim); glUseProgram(p1); tri(p1);
    vector<int> rb = read_q24(dim);
    vector<int> want(dim);
    for (int i = 0; i < dim; ++i) {
        long q = lrintf(sinf(i*0.021f)*50.0f*256.0f);
        want[i] = (int)std::max(-8388608L, std::min(8388607L, q));
    }
    char nm[64]; snprintf(nm,sizeof nm,"B1 i24 render->readback dither%d",dither);
    report(nm, rb, want, 0);
    glDeleteProgram(p1);
    // B2: upload known q24, fetch via x_q24, re-encode, read back.
    vector<int> raw(dim);
    for (int i = 0; i < dim; ++i) raw[i] = (int)((i*2654435761u) % 16777216u) - 8388608;
    GLuint it = tex(dim, 1, q24_bytes(raw, dim, 1));
    string fs2 = string("precision highp float;\n") + x_glsl(XFormat::I24_1) +
        "uniform sampler2D X;void main(){int i=int(gl_FragCoord.x);"
        "float v=x_q24(texture2D(X,vec2((float(i)+0.5)/" + std::to_string(dim) + ".0,0.5)));"
        "gl_FragColor=x_q24enc(v);}";
    GLuint p2 = prog(fs2); out_begin(dim); glUseProgram(p2); bind(p2,"X",0,it); tri(p2);
    vector<int> g2 = read_q24(dim);
    snprintf(nm,sizeof nm,"B2 i24 fetch->render dither%d",dither);
    report(nm, g2, raw, 0);
    glDeleteProgram(p2); glDeleteTextures(1,&it);
}

// C. The shipped rwkv_sqrelu vs a CPU f32 reference (I24_1 in and out).
static void test_sqrelu(int dim) {
    vector<int> raw(dim);
    for (int i = 0; i < dim; ++i) raw[i] = (int)lrintf(sinf(i*0.017f)*30.0f*256.0f);
    GLuint it = tex(dim, 1, q24_bytes(raw, dim, 1));
    GLuint p = prog(rwkv_sqrelu(dim)); if (!p) { printf("  sqrelu link fail\n"); ++failures; return; }
    out_begin(dim); glUseProgram(p); bind(p,"X",0,it); tri(p);
    vector<int> g = read_q24(dim);
    vector<int> want(dim);
    for (int i = 0; i < dim; ++i) {
        float v = raw[i]/256.0f; float r = v > 0 ? v*v : 0.0f;
        long q = lrintf(r*256.0f); want[i] = (int)std::max(-8388608L, std::min(8388607L, q));
    }
    report("C rwkv_sqrelu shipped", g, want, 1);
    glDeleteProgram(p); glDeleteTextures(1,&it);
}

// D. bit-cast f32 render->fetch round trip (ES3).  Byte layout: R low, A high.
static const char* BC =
    "float bcRd(sampler2D t,vec2 uv){vec4 c=texture(t,uv);"
    "uint u=uint(floor(c.r*255.0+0.5))|(uint(floor(c.g*255.0+0.5))<<8)|"
    "(uint(floor(c.b*255.0+0.5))<<16)|(uint(floor(c.a*255.0+0.5))<<24);"
    "return uintBitsToFloat(u);}"
    "vec4 bcWr(float f){uint u=floatBitsToUint(f);"
    "return vec4(float(u&255u),float((u>>8)&255u),float((u>>16)&255u),float((u>>24)&255u))/255.0;}";
static vector<uint8_t> f32_bytes(const vector<float>& f) {
    vector<uint8_t> px(f.size()*4);
    for (size_t i = 0; i < f.size(); ++i) {
        uint32_t u; memcpy(&u, &f[i], 4);
        px[i*4+0]=u&255; px[i*4+1]=(u>>8)&255; px[i*4+2]=(u>>16)&255; px[i*4+3]=(u>>24)&255;
    }
    return px;
}
static void test_bitcast(int dim, bool dither) {
    if (dither) glEnable(GL_DITHER); else glDisable(GL_DITHER);
    vector<float> f(dim);
    for (int i = 0; i < dim; ++i) f[i] = i*0.5f;   // exact in f32: GPU/CPU agree
    // D1: render bcWr(f(i)), read back the bytes, compare to CPU floatBits.
    string fs1 = string("precision highp float;\n") + BC +
        "void main(){int i=int(gl_FragCoord.x);gl_FragColor=bcWr(float(i)*0.5);}";
    GLuint p1 = prog(fs1, true); if (!p1) { printf("  D1 link fail\n"); ++failures; return; }
    out_begin(dim); glUseProgram(p1); tri(p1);
    const uint8_t* g = out_end();
    vector<uint8_t> want = f32_bytes(f);
    int bad=0, first=-1;
    for (int i = 0; i < dim*4; ++i) if (g[i]!=want[i]) { ++bad; if(first<0)first=i; }
    char nm[64]; snprintf(nm,sizeof nm,"D1 bc render->readback dither%d",dither);
    printf("  %-34s %s badbytes=%d/%d", nm, bad?"FAIL":"PASS", bad, dim*4);
    if (bad) printf("  [%d] got %02x want %02x", first, g[first], want[first]);
    printf("\n"); if (bad) ++failures;
    glDeleteProgram(p1);
    // D2: upload f bits, bcRd -> bcWr, read back, compare bits.
    GLuint it = tex(dim, 1, f32_bytes(f));
    string fs2 = string("precision highp float;\n") + BC +
        "uniform sampler2D X;void main(){int i=int(gl_FragCoord.x);"
        "gl_FragColor=bcWr(bcRd(X,vec2((float(i)+0.5)/" + std::to_string(dim) + ".0,0.5)));}";
    GLuint p2 = prog(fs2, true); out_begin(dim); glUseProgram(p2); bind(p2,"X",0,it); tri(p2);
    const uint8_t* g2 = out_end();
    bad=0; first=-1;
    for (int i = 0; i < dim*4; ++i) if (g2[i]!=want[i]) { ++bad; if(first<0)first=i; }
    snprintf(nm,sizeof nm,"D2 bc fetch->render dither%d",dither);
    printf("  %-34s %s badbytes=%d/%d", nm, bad?"FAIL":"PASS", bad, dim*4);
    if (bad) printf("  [%d] got %02x want %02x", first, g2[first], want[first]);
    printf("\n"); if (bad) ++failures;
    glDeleteProgram(p2); glDeleteTextures(1,&it);
}

// E. The matvec activation fetch: I24_1 X4Q does FOUR texture2D fetches from
// sampler X in one fragment (I16_2's inline form does two).  wrow_probe found
// multi-fetch from one sampler undefined on VC4 for a tall plane; this names
// whether the I24_1 matvec (the RWKV path's act_fmt_) delivers on VC4.
static void test_matvec(int dim, XFormat xf) {
    const int cols = dim, rows = dim;
    vector<int8_t> w((size_t)rows*cols);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
        w[(size_t)r*cols+c] = (int8_t)(((r*7+c*13)%9)-4);
    // q8 weight texture: texel (q,r) holds weights 4q..4q+3 as (w+128) bytes.
    vector<uint8_t> wp((size_t)(cols/4)*rows*4);
    for (int r = 0; r < rows; ++r) for (int q = 0; q < cols/4; ++q)
        for (int j = 0; j < 4; ++j)
            wp[((size_t)r*(cols/4)+q)*4+j] = (uint8_t)(w[(size_t)r*cols+4*q+j]+128);
    GLuint wt = tex(cols/4, rows, wp);
    vector<int> xraw(cols);
    for (int c = 0; c < cols; ++c) xraw[c] = (int)((c*2654435761u)%512u)-256;
    GLuint xt;
    if (xf == XFormat::I24_1) xt = tex(cols, 1, q24_bytes(xraw, cols, 1));
    else { vector<uint8_t> px((size_t)(cols/2)*4);
        for (int c = 0; c < cols; ++c) { uint16_t u=(uint16_t)(int16_t)xraw[c];
            px[(c/2)*4+(c%2)*2]=u>>8; px[(c/2)*4+(c%2)*2+1]=u&255; }
        xt = tex(cols/2, 1, px); }
    GLuint p = prog(matvec_q8(rows, cols, xf, false)); if (!p) { printf("  E link fail\n"); ++failures; return; }
    out_begin(rows); glUseProgram(p);
    bind(p,"W",0,wt); bind(p,"X",1,xt);
    glUniform1f(glGetUniformLocation(p,"ws"), 1.0f);
    tri(p);
    vector<int> g = read_q24(rows);
    vector<int> want(rows);
    for (int r = 0; r < rows; ++r) {
        float s = 0; for (int c = 0; c < cols; ++c) s += (float)w[(size_t)r*cols+c]*(float)xraw[c];
        long q = lrintf(s); want[r] = (int)std::max(-32768L, std::min(32767L, q));
    }
    char nm[64]; snprintf(nm,sizeof nm,"E matvec_q8 %s %dx%d", x_format_name(xf), rows, cols);
    report(nm, g, want, 1);
    glDeleteProgram(p); glDeleteTextures(1,&wt); glDeleteTextures(1,&xt);
}

int main(int argc, char** argv) {
    bool want_es3 = !(argc > 1 && strcmp(argv[1], "gles2") == 0);
    if (want_es3) { setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
                    setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0); }
    PFNGPD gpd = (PFNGPD)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!gpd) { printf("no eglGetPlatformDisplayEXT\n"); return 1; }
    EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint maj=0, min=0;
    if (!eglInitialize(d, &maj, &min)) { printf("eglInitialize failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ca[] = {EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,
        EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES2_BIT,EGL_NONE};
    EGLConfig cfg; EGLint n=0;
    if (!eglChooseConfig(d, ca, &cfg, 1, &n) || n<1) { printf("eglChooseConfig n=%d\n", n); return 1; }
    const EGLint xa[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, xa);
    if (!ctx || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) { printf("context failed\n"); return 1; }
    const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    g_es3 = glsl && strncmp(glsl, "OpenGL ES GLSL ES 3", 19) == 0;
    GLint maxt=0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxt);
    GLint dith=0; glGetIntegerv(GL_DITHER, &dith);
    printf("renderer %s\nversion  %s\nglsl     %s  (es3=%d)\nmaxtex %d  default GL_DITHER=%d\n\n",
        (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION), glsl, g_es3, maxt, dith);

    const int D = 768;
    printf("A. 2D NPOT q24 constant fetch (width %d)\n", D);
    test_const2d(D, 6); test_const2d(D, 3); test_const2d(D, 2);
    test_lerp_const(D, 0, 6); test_lerp_const(D, 3, 6); test_lerp_const(D, 5, 6);
    test_lerp_const(D, 0, 1);
    printf("\nB. I24_1 render/fetch round trip (dither A/B)\n");
    test_i24(D, false); test_i24(D, true);
    printf("\nC. shipped elementwise\n");
    test_sqrelu(D);
    printf("\nE. matvec activation fetch (I24_1 X4Q 4-fetch vs I16_2 2-fetch)\n");
    test_matvec(D, XFormat::I16_2);
    test_matvec(D, XFormat::I24_1);
    if (g_es3) { printf("\nD. bit-cast f32 round trip (dither A/B)\n");
        test_bitcast(D, false); test_bitcast(D, true); }
    else printf("\nD. skipped (no ESSL 3.00)\n");
    printf("\n%s (%d failing)\n", failures ? "FAIL" : "ALL PASS", failures);
    return failures ? 1 : 0;
}


