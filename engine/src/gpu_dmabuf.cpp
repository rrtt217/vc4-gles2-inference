// A 4-bit weight texture the driver cannot turn into RGBA8.
//
// Why glTexImage2D cannot do it: st_ChooseTextureFormat adds
// PIPE_BIND_RENDER_TARGET for GL_RGBA4, vc4 cannot render to B4G4R4A4
// (vc4_formats.c: rt = RT_NO), so find_supported_format skips both two-byte
// candidates and takes DEFAULT_RGBA_FORMATS; the retry that would drop the
// render-target binding only runs when nothing matches, and RGBA8 always
// matches.  docs/performance.md, 2026-11-15.
//
// Importing our own buffer goes around that function.  Three things are needed:
// the modifier has to be *queried* (this board says 0x0700000000000001, not a
// value to spell out), the BO has to carry it (VC4_SET_TILING, or the import is
// EGL_BAD_MATCH), and the plane pitch has to be present even though the image is
// tiled (dri2_check_dma_buf_format insists).

#include "gpu_internal.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The DRM entry points, pinned: libdrm's headers are not in the cross sysroot and
// these numbers are the kernel's stable ABI (uapi/drm/{drm,vc4_drm}.h).  A wrong
// number fails the ioctl and the caller keeps the upload path.
#define VC4_DRM_IOCTL_BASE 'd'
#define VC4_DRM_COMMAND_BASE 0x40
struct vc4_drm_create_bo { uint32_t size, flags, handle, pad; };
struct vc4_drm_mmap_bo { uint32_t handle, flags; uint64_t offset; };
struct vc4_drm_prime_handle { uint32_t handle, flags; int32_t fd; };
struct vc4_drm_gem_close { uint32_t handle, pad; };
struct vc4_drm_set_tiling { uint32_t handle, flags; uint64_t modifier; };
#define VC4_IOCTL_CREATE_BO \
        _IOWR(VC4_DRM_IOCTL_BASE, VC4_DRM_COMMAND_BASE + 0x03, struct vc4_drm_create_bo)
#define VC4_IOCTL_MMAP_BO \
        _IOWR(VC4_DRM_IOCTL_BASE, VC4_DRM_COMMAND_BASE + 0x04, struct vc4_drm_mmap_bo)
#define VC4_IOCTL_SET_TILING \
        _IOWR(VC4_DRM_IOCTL_BASE, VC4_DRM_COMMAND_BASE + 0x08, struct vc4_drm_set_tiling)
#define VC4_IOCTL_PRIME_HANDLE_TO_FD _IOWR(VC4_DRM_IOCTL_BASE, 0x2d, struct vc4_drm_prime_handle)
#define VC4_IOCTL_GEM_CLOSE _IOW(VC4_DRM_IOCTL_BASE, 0x09, struct vc4_drm_gem_close)

// DRM_FORMAT_ARGB4444 = fourcc_code('A','R','1','2'); 0x34325241 ("AR24") is a
// different format and the driver says so.
#define VC4_FOURCC_ARGB4444 0x32315241u
#ifndef DRM_CLOEXEC
#define DRM_CLOEXEC 0x00000002u
#endif

namespace {
EGLDisplay g_dpy = EGL_NO_DISPLAY;
struct ImportedTexture { GLuint tex; EGLImageKHR img; uint32_t handle; int dmabuf; };
std::vector<ImportedTexture> g_imported;
int g_drm = -2;
uint64_t g_modifier = 0;
PFNEGLCREATEIMAGEKHRPROC p_create_image;
PFNEGLDESTROYIMAGEKHRPROC p_destroy_image;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_image_target;

// ---- the surface size ------------------------------------------------------
// The surface is made of 4 KB tiles of 8x8 utiles, a utile being 8x4 texels at
// two bytes each.  The arrangement *inside* them is the driver's business now
// that the upload goes through glTexSubImage2D; only the size is needed here.
//
// The tiled layout's pitch is the 64-texel column grid, not the requested
// width: the kernel's modifier says so, and the import checks it.  `w / 64`
// truncated, so a width that is not a multiple of 64 (896 columns at four
// weights a texel is 224) got a pitch of w*2 = 448 and a surface one tile
// column short; the driver refused the image ("unsupported stride 448 instead
// of 512") and the q4 upload silently fell back to glTexImage2D, where VC4's
// only renderable choice is RGBA8 -- four bytes a texel, *twice* the memory.
// The fallback was invisible (one Q4_TEX_STATS line) and the whole model then
// needed 2x its weights.  Pad the pitch and the tile count together.
static int t_pitch_texels(int w) { return (w + 63) & ~63; }
static size_t t_bytes(int w, int h) {
        return 4096u * (size_t)(t_pitch_texels(w) / 64) * (size_t)((h + 31) / 32);
}

int drm_fd() {
    if (g_drm != -2) return g_drm;
    g_drm = -1;
    // card0 is the KMS node (root:video), renderD128 the render node
    // (root:render, never needs DRM master).  Both accept VC4_CREATE_BO on the
    // board; the render node is preferred for a headless compute process.
    static const char* nodes[] = { "/dev/dri/renderD128", "/dev/dri/card0", "/dev/dri/card1" };
    for (const char* n : nodes) {
        int fd = open(n, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        struct vc4_drm_create_bo probe = {};
        probe.size = 4096;
        if (ioctl(fd, VC4_IOCTL_CREATE_BO, &probe) == 0) {
            struct vc4_drm_gem_close c = {};
            c.handle = probe.handle;
            ioctl(fd, VC4_IOCTL_GEM_CLOSE, &c);
            g_drm = fd;
            return g_drm;
        }
        close(fd);
    }
    return g_drm;
}

uint64_t tiled_modifier() {
    if (g_modifier) return g_modifier;
    if (g_dpy == EGL_NO_DISPLAY) return 0;
    auto query = (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
    if (!query) return 0;
    EGLuint64KHR mods[8] = {};
    EGLint n = 0;
    if (!query(g_dpy, (EGLint)VC4_FOURCC_ARGB4444, 8, mods, nullptr, &n)) return 0;
    for (int i = 0; i < n; ++i) if (mods[i] != 0) { g_modifier = mods[i]; break; }
    return g_modifier;
}
}  // namespace

void gpu_dmabuf_init(EGLDisplay dpy) { g_dpy = dpy; }

bool gpu_dmabuf_available() {
    if (drm_fd() < 0) return false;
    if (!p_create_image) p_create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    if (!p_destroy_image) p_destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    if (!p_image_target)
        p_image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!tiled_modifier()) return false;
    return p_create_image && p_destroy_image && p_image_target;
}

GLuint gpu_dmabuf_texture(int w, int h, const uint16_t* data) {
    if (!gpu_dmabuf_available()) return 0;
    const int fd = drm_fd();
    const size_t bytes = t_bytes(w, h);
    struct vc4_drm_create_bo cbo = {};
    cbo.size = (uint32_t)bytes;
    if (ioctl(fd, VC4_IOCTL_CREATE_BO, &cbo) != 0) return 0;
    struct vc4_drm_set_tiling st = {};
    st.handle = cbo.handle;
    st.modifier = g_modifier;
    if (ioctl(fd, VC4_IOCTL_SET_TILING, &st) != 0) goto close_bo;
    {
        struct vc4_drm_mmap_bo mbo = {};
        mbo.handle = cbo.handle;
        if (ioctl(fd, VC4_IOCTL_MMAP_BO, &mbo) != 0) goto close_bo;
        void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mbo.offset);
        if (map == MAP_FAILED) goto close_bo;
        // No CPU-side fill: the layout of a T-tiled surface is the driver's
        // business, and a glTexSubImage2D into the imported image makes it do
        // the work.  The CPU only has to produce the container's own nibble
        // order, which is what `data` already holds.  Measured on the board:
        // with the driver doing it, 64x16 / 192x16 / 192x1536 / 256x1024 all
        // sample back exactly (probes/egl_image_probe, EGL_IMG_FILL=driver).
        munmap(map, bytes);
    }
    {
        struct vc4_drm_prime_handle ph = {};
        ph.handle = cbo.handle;
        ph.flags = DRM_CLOEXEC;
        if (ioctl(fd, VC4_IOCTL_PRIME_HANDLE_TO_FD, &ph) != 0) goto close_bo;
        EGLint a[] = {
            EGL_WIDTH, w,
            EGL_HEIGHT, h,
            EGL_LINUX_DRM_FOURCC_EXT, (EGLint)VC4_FOURCC_ARGB4444,
            EGL_DMA_BUF_PLANE0_FD_EXT, ph.fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, t_pitch_texels(w) * 2,
            EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(g_modifier & 0xffffffffu),
            EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(g_modifier >> 32),
            EGL_NONE
        };
        EGLImageKHR img = p_create_image(g_dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, a);
        if (img == EGL_NO_IMAGE_KHR) { close(ph.fd); goto close_bo; }
        GLuint tex = 0;
        glGenTextures(1, &tex);
        // The engine caches which texture is bound to which unit; binding here
        // behind its back would make its later bind_tex() calls no-ops and the
        // kernels would sample whatever it still believes is there.
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        gl_state_forget_bindings();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        while (glGetError() != GL_NO_ERROR) {}
        p_image_target(GL_TEXTURE_2D, img);
        if (glGetError() != GL_NO_ERROR) {
            glDeleteTextures(1, &tex);
            p_destroy_image(g_dpy, img);
            close(ph.fd);
            goto close_bo;
        }
        {
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
                                GL_UNSIGNED_SHORT_4_4_4_4, data);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                if (glGetError() != GL_NO_ERROR) {
                        glDeleteTextures(1, &tex);
                        p_destroy_image(g_dpy, img);
                        close(ph.fd);
                        goto close_bo;
                }
        }
        // Q4_IMPORT_CHECK=1: sample texel (0,0) once, through the same kind of
        // 1x1 render the probe uses, and print it against the word handed in.
        // It separates "the surface is wrong" from "the kernels read it wrong".
        if (getenv("Q4_IMPORT_CHECK") && g_imported.empty()) {
                GLuint fbo = 0, rt = 0;
                glGenFramebuffers(1, &fbo);
                glGenTextures(1, &rt);
                glBindTexture(GL_TEXTURE_2D, rt);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
                static const char* vs = "attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}";
                static const char* fs =
                    "precision highp float;uniform sampler2D T;uniform highp vec2 UV;"
                    "void main(){vec4 c=texture2D(T,UV);gl_FragColor=vec4(c.x*15.0,c.y*15.0,c.z*15.0,c.w*15.0)/255.0;}";
                auto mk = [](GLenum t, const char* src) {
                        GLuint h = glCreateShader(t);
                        glShaderSource(h, 1, &src, nullptr);
                        glCompileShader(h);
                        return h;
                };
                GLuint pr = glCreateProgram();
                glAttachShader(pr, mk(GL_VERTEX_SHADER, vs));
                glAttachShader(pr, mk(GL_FRAGMENT_SHADER, fs));
                glLinkProgram(pr);
                glUseProgram(pr);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, tex);
                gl_state_forget_bindings();
                glUniform1i(glGetUniformLocation(pr, "T"), 0);
                glUniform2f(glGetUniformLocation(pr, "UV"), 0.5f / w, 0.5f / h);
                GLuint vbo = 0;
                static const float quad[] = { -1, -1, 3, -1, -1, 3 };
                glGenBuffers(1, &vbo);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
                GLint loc = glGetAttribLocation(pr, "p");
                glEnableVertexAttribArray(loc);
                glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
                glViewport(0, 0, 1, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glFinish();
                uint8_t got[4] = {};
                glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
                fprintf(stderr, "import check: %dx%d word0=0x%04x nibbles %d %d %d %d -> sampled %d %d %d %d\n",
                        w, h, data[0], data[0] & 0xF, (data[0] >> 4) & 0xF, (data[0] >> 8) & 0xF,
                        (data[0] >> 12) & 0xF, got[0], got[1], got[2], got[3]);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glDeleteFramebuffers(1, &fbo);
                glDeleteTextures(1, &rt);
                glDeleteBuffers(1, &vbo);
                glDeleteProgram(pr);
                gl_state_forget_bindings();
        }
        g_imported.push_back({ tex, img, cbo.handle, ph.fd });
        return tex;
    }
close_bo:
    {
        struct vc4_drm_gem_close c = {};
        c.handle = cbo.handle;
        ioctl(fd, VC4_IOCTL_GEM_CLOSE, &c);
    }
    return 0;
}

void gpu_dmabuf_release_all() {
    const int fd = drm_fd();
    for (ImportedTexture& it : g_imported) {
        if (it.tex) glDeleteTextures(1, &it.tex);
        if (it.img != EGL_NO_IMAGE_KHR && p_destroy_image) p_destroy_image(g_dpy, it.img);
        if (it.dmabuf >= 0) close(it.dmabuf);
        if (it.handle && fd >= 0) {
            struct vc4_drm_gem_close c = {};
            c.handle = it.handle;
            ioctl(fd, VC4_IOCTL_GEM_CLOSE, &c);
        }
    }
    g_imported.clear();
}

int gpu_dmabuf_count() { return (int)g_imported.size(); }
