// Does a 4-bit (B4G4R4A4) dma-buf imported as a T-tiled EGLImage sample back
// exactly, at the widths the engine uses?
//
// Why this exists: glTexImage2D cannot give a two-byte surface on this driver.
// `st_ChooseTextureFormat` adds PIPE_BIND_RENDER_TARGET for GL_RGBA4, vc4 cannot
// render to B4G4R4A4 (`vc4_formats.c`: rt = RT_NO), so `find_supported_format`
// skips both two-byte candidates and takes DEFAULT_RGBA_FORMATS; the retry that
// would drop the render-target binding only runs when nothing matches, and RGBA8
// always matches.  docs/performance.md (2026-11-15) has the measurements.
//
// Importing our own buffer goes around that function.  Three things are needed
// and each one cost a round:
//   1. the modifier is *queried* (`eglQueryDmaBufModifiersEXT`); this board
//      reports 0x0700000000000001, not a value to be spelled out by hand;
//   2. the BO carries it (`DRM_IOCTL_VC4_SET_TILING`), or the import is
//      EGL_BAD_MATCH;
//   3. the plane pitch is present even for a tiled image, because Mesa's
//      `dri2_check_dma_buf_format` requires it.
//
// One shape a process (`EGL_IMG_SHAPE=WxH`, default 192x1536): a loop shares GL
// state and the glTexImage2D control below starts failing on the second shape,
// which made every later reading meaningless.  The control runs first, so a
// broken harness cannot be mistaken for a broken import.

#include "egl_surfaceless.h"

#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <drm_fourcc.h>
#include <vc4_drm.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- the VC4 T-tiled layout, from Mesa's own arithmetic --------------------
// vc4_tiling.c::t_utile_address: 4 KB tiles of 8x8 utiles, odd tile rows
// mirrored, 1 KB stiles permuted by the even/odd maps.
static uint32_t t_utile_address(uint32_t ux, uint32_t uy, uint32_t utile_stride) {
    const uint32_t tile_stride = utile_stride >> 3;
    uint32_t tile_x = ux >> 3, tile_y = uy >> 3;
    const bool odd = (tile_y & 1) != 0;
    if (odd) tile_x = tile_stride - tile_x - 1;
    const uint32_t tile_offset = 4096u * (tile_y * tile_stride + tile_x);
    const uint32_t stile_x = (ux >> 2) & 1, stile_y = (uy >> 2) & 1;
    const uint32_t idx = (stile_y << 1) + stile_x;
    static const uint32_t odd_map[4] = { 2, 1, 3, 0 };
    static const uint32_t even_map[4] = { 0, 3, 1, 2 };
    return tile_offset + 1024u * (odd ? odd_map[idx] : even_map[idx]);
}

// vc4_tiling_lt.c::swizzle_lt_{x,y} for cpp = 2: 8x4 utiles inside a stile.
// Which arrangement the driver uses: settled by measurement below.
static char g_map = 'a';

static size_t t_offset(int x, int y, int w) {
    const uint32_t utile_stride = (uint32_t)(w / 8);
    const uint32_t ux = (uint32_t)x >> 3, uy = (uint32_t)y >> 2;
    const uint32_t base = t_utile_address(ux & ~3u, uy & ~3u, utile_stride);
    // Inside a stile (4x4 utiles = 32x16 texels at cpp 2) the layout is dense:
    // vc4_t_image_helper hands the whole stile to the LT writer with a stride of
    // 64 bytes, so a row is 32 texels = 64 bytes.  The utile *order* is in
    // t_utile_address above; this used to apply the utile-internal swizzle here
    // as well, which is what put every texel in the wrong place.
    const uint32_t within = (((uint32_t)y & 15u) << 6) | (((uint32_t)x & 31u) << 1);
    if (g_map == 'a') return base + within;
    // Variants, to be settled by measurement rather than by reading: 'b' drops
    // the stile permutation, 'c' keeps it but not the odd-tile-row mirroring,
    // 'd' drops both.
    const uint32_t tile_stride = utile_stride >> 3;
    const uint32_t tile_x = ux >> 3, tile_y = uy >> 3;
    const uint32_t stile_x = (ux >> 2) & 1u, stile_y = (uy >> 2) & 1u;
    const uint32_t idx = (stile_y << 1) + stile_x;
    const uint32_t tile_base = 4096u * (tile_y * tile_stride + tile_x);
    if (g_map == 'b' || g_map == 'd') return tile_base + 1024u * idx + within;
    static const uint32_t om[4] = { 2, 1, 3, 0 }, em[4] = { 0, 3, 1, 2 };
    const bool odd = (tile_y & 1) != 0;
    return tile_base + 1024u * (odd ? om[idx] : em[idx]) + within;
}

static size_t t_bytes(int w, int h) { return 4096u * (size_t)(w / 64) * (size_t)((h + 31) / 32); }

// Nibbles n0..n3 in a word come back as (n2, n1, n0, n3) through an imported
// surface -- measured, not assumed.
static uint16_t iwz(uint16_t v) {
    return (uint16_t)((v & 0xF0F0u) | ((v & 0x000Fu) << 8) | ((v & 0x0F00u) >> 8));
}

// Texel (x,y) is filled with the same nibble in all four components, value
// v(x,y) = ((x + 192*y) % 15) + 1, so a wrong *position* reads a wrong value
// even though the component order is invisible to it.  (0,0) carries 0x3210 to
// check the component order separately.
static int v_at(int x, int y) { return ((x + 192 * y) % 15) + 1; }
static uint16_t word_at(int x, int y) {
    return (x == 0 && y == 0) ? (uint16_t)0x3210 : (uint16_t)(v_at(x, y) * 0x1111);
}
static int expect_comp(int x, int y, int c) {
    if (x == 0 && y == 0) return 3 - c;   // 0x3210 samples as 3 2 1 0? no: see below
    return v_at(x, y);
}

static GLuint make_prog() {
    static const char* vs = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
    // highp on the uniform *and* FragCoord: a fragment uniform defaults to
    // mediump, and 191.5 through a 10-bit mantissa lands a texel away.
    static const char* fs =
        "precision highp float;uniform sampler2D T;uniform highp vec2 UV;"
        "void main(){"
        "vec4 c=texture2D(T,UV);gl_FragColor=vec4(c.x*15.0,c.y*15.0,c.z*15.0,c.w*15.0)/255.0;}";
    auto sh = [](GLenum t, const char* src) {
        GLuint h = glCreateShader(t);
        glShaderSource(h, 1, &src, nullptr);
        glCompileShader(h);
        GLint ok = 0; glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[512]; glGetShaderInfoLog(h, sizeof log, nullptr, log); printf("shader: %s\n", log); }
        return h;
    };
    GLuint pr = glCreateProgram();
    glAttachShader(pr, sh(GL_VERTEX_SHADER, vs));
    glAttachShader(pr, sh(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pr);
    GLint ok = 0; glGetProgramiv(pr, GL_LINK_STATUS, &ok);
    if (!ok) { char log[512]; glGetProgramInfoLog(pr, sizeof log, nullptr, log); printf("link: %s\n", log); }
    return pr;
}

// Sample one texel per draw, at a centre handed in as a uniform: no per-fragment
// coordinate arithmetic, which is where the earlier versions of this probe went
// wrong (a one-texel shift that only appeared at 192 wide).
static int compare(GLuint prog, GLuint tex, int W, int H, const char* what) {
    GLuint fbo = 0, rt = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(prog, "T"), 0);
    GLuint vbo = 0;
    static const float quad[] = { -1, -1, 3, -1, -1, 3 };
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    GLint loc = glGetAttribLocation(prog, "p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glViewport(0, 0, 1, 1);
    const GLint uvloc = glGetUniformLocation(prog, "UV");
    int bad = 0, shown = 0;
    const int pts[][2] = { {0,0}, {1,0}, {2,0}, {7,0}, {8,0}, {63,0}, {64,0}, {0,1}, {0,3}, {0,4},
                           {3,3}, {W-1,0}, {0,H-1}, {W/2,H/2}, {W-1,H-1} };
    for (auto& pt : pts) {
        const int x = pt[0], y = pt[1];
        if (x < 0 || y < 0 || x >= W || y >= H) continue;
        glUniform2f(uvloc, (x + 0.5f) / W, (y + 0.5f) / H);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        uint8_t got[4] = {};
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
        const uint16_t w0 = word_at(x, y);
        const int n[4] = { w0 & 0xF, (w0 >> 4) & 0xF, (w0 >> 8) & 0xF, (w0 >> 12) & 0xF };
        const int want[4] = { n[2], n[1], n[0], n[3] };
        int here = 0;
        for (int c = 0; c < 4; ++c) if (got[c] != want[c]) ++here;
        if (here) bad += here;
        if (shown < 18) {
            printf("  %-14s (%3d,%3d) read %2d %2d %2d %2d   want %2d %2d %2d %2d%s\n", what, x, y,
                   got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3],
                   here ? "" : "  ok");
            ++shown;
        }
    }
    printf("  %-22s %d of %d sampled components wrong%s\n", what, bad, 15 * 4, bad ? "" : "  <- exact");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &rt);
    glDeleteBuffers(1, &vbo);
    return bad;
}

int main() {
    EglSurfaceless egl = egl_surfaceless_init(2);
    if (!egl.ok()) return 1;

    int W = 192, H = 1536;
    if (const char* sh = getenv("EGL_IMG_SHAPE")) {
        int w = 0, h = 0;
        if (sscanf(sh, "%dx%d", &w, &h) == 2) { W = w; H = h; }
    }
    if (const char* m = getenv("EGL_IMG_MAP")) g_map = m[0];
    printf("shape        %dx%d  (%zu bytes, map %c)\n", W, H, t_bytes(W, H), g_map);
    if (W % 64) printf("  note       width must be a multiple of 64 texels (utile_stride %% 8)\n");

    GLuint prog = make_prog();

    // ---- control: the same pattern through glTexImage2D -------------------
    // If this is not exact the harness is wrong and nothing below means
    // anything.
    {
        std::vector<uint8_t> ref((size_t)W * H * 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const uint16_t w0 = word_at(x, y);
                const int n[4] = { w0 & 0xF, (w0 >> 4) & 0xF, (w0 >> 8) & 0xF, (w0 >> 12) & 0xF };
                uint8_t* p4 = &ref[((size_t)y * W + x) * 4];
                p4[0] = (uint8_t)(n[2] * 17);
                p4[1] = (uint8_t)(n[1] * 17);
                p4[2] = (uint8_t)(n[0] * 17);
                p4[3] = (uint8_t)(n[3] * 17);
            }
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, ref.data());
        compare(prog, tex, W, H, "control glTexImage2D");
        glDeleteTextures(1, &tex);
    }

    // ---- the modifier the display actually offers -------------------------
    auto query = (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
    if (!query) { printf("no eglQueryDmaBufModifiersEXT\n"); return 1; }
    EGLuint64KHR mods[8] = {};
    EGLint n = 0;
    if (!query(egl.dpy, (EGLint)DRM_FORMAT_ARGB4444, 8, mods, nullptr, &n) || n == 0) {
        printf("ARGB4444: no modifier reported\n");
        return 1;
    }
    uint64_t tiled = 0;
    for (int i = 0; i < n; ++i) if (mods[i] != 0) tiled = mods[i];
    printf("modifier     0x%llx%s\n", (unsigned long long)tiled, tiled ? "" : "  (only linear!)");
    if (!tiled) return 1;

    // ---- BO: create, tag with the modifier, fill, export -------------------
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) fd = open("/dev/dri/renderD128", O_RDWR);
    if (fd < 0) { printf("no DRM node: %s\n", strerror(errno)); return 1; }
    const size_t bytes = t_bytes(W, H);
    struct drm_vc4_create_bo cbo = {};
    cbo.size = (uint32_t)bytes;
    if (ioctl(fd, DRM_IOCTL_VC4_CREATE_BO, &cbo) != 0) { printf("CREATE_BO: %s\n", strerror(errno)); return 1; }
    struct drm_vc4_set_tiling st = {};
    st.handle = cbo.handle;
    st.modifier = tiled;
    if (ioctl(fd, DRM_IOCTL_VC4_SET_TILING, &st) != 0) printf("SET_TILING: %s\n", strerror(errno));
    struct drm_vc4_mmap_bo mbo = {};
    mbo.handle = cbo.handle;
    if (ioctl(fd, DRM_IOCTL_VC4_MMAP_BO, &mbo) != 0) { printf("MMAP_BO: %s\n", strerror(errno)); return 1; }
    void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mbo.offset);
    if (map == MAP_FAILED) { printf("mmap: %s\n", strerror(errno)); return 1; }
    memset(map, 0, bytes);
    // EGL_IMG_FILL=tiled (the T layout) or linear (texel i at byte 2i).  Running
    // both tells us which layout the driver is actually reading, instead of
    // guessing at the offset between them.
    const char* fill = getenv("EGL_IMG_FILL");
    const bool linear_fill = fill && strcmp(fill, "linear") == 0;
    // EGL_IMG_FILL=driver leaves the buffer alone and uploads through
    // glTexSubImage2D once the image is bound, so *the driver* does the tiling.
    // No layout to know, which is the whole point of trying it.
    const bool driver_fill = fill && strcmp(fill, "driver") == 0;
    printf("fill         %s\n", driver_fill ? "driver (glTexSubImage2D)" : linear_fill ? "linear" : "tiled");
    if (!driver_fill) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const size_t off = linear_fill ? ((size_t)y * W + x) * 2 : t_offset(x, y, W);
                *(uint16_t*)((char*)map + off) = iwz(word_at(x, y));
            }
        msync(map, bytes, MS_SYNC);
    }
    munmap(map, bytes);
    struct drm_prime_handle ph = {};
    ph.handle = cbo.handle;
    ph.flags = DRM_CLOEXEC;
    if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph) != 0) { printf("PRIME: %s\n", strerror(errno)); return 1; }

    // ---- import and compare ------------------------------------------------
    auto create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    auto image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!create_image || !image_target) { printf("no EGLImage entry points\n"); return 1; }
    EGLint a[] = {
        EGL_WIDTH, W,
        EGL_HEIGHT, H,
        EGL_LINUX_DRM_FOURCC_EXT, (EGLint)DRM_FORMAT_ARGB4444,
        EGL_DMA_BUF_PLANE0_FD_EXT, ph.fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, W * 2,     // required even when tiled
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(tiled & 0xffffffffu),
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(tiled >> 32),
        EGL_NONE
    };
    EGLImageKHR img = create_image(egl.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, a);
    if (img == EGL_NO_IMAGE_KHR) { printf("import rejected: 0x%x\n", eglGetError()); return 1; }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    image_target(GL_TEXTURE_2D, img);
    if (GLenum e = glGetError()) { printf("bind rejected: 0x%x\n", e); return 1; }
    if (driver_fill) {
        std::vector<uint16_t> cpu((size_t)W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) cpu[(size_t)y * W + x] = word_at(x, y);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, cpu.data());
        printf("upload       glTexSubImage2D into the imported image: err 0x%x\n", glGetError());
    }
    int bad = compare(prog, tex, W, H, "dma-buf T-tiled");
    return bad ? 2 : 0;
}
