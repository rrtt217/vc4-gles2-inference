// Can a 4-bit texture reach the TMU *without* Mesa choosing the format?
//
// st_ChooseTextureFormat adds PIPE_BIND_RENDER_TARGET for GL_RGBA4 (GL textures
// may become render targets and Mesa cannot know), vc4 cannot render to
// B4G4R4A4 (vc4_formats.c: rt = RT_NO), so find_supported_format skips the two
// two-byte candidates and lands on RGBA8.  The retry that would drop the render
// target binding only runs when *nothing* matches, and DEFAULT_RGBA_FORMATS
// always matches.  So glTexImage2D/glTexStorage2D cannot give a 2-byte surface
// whatever the spelling (measured: docs/performance.md 2026-11-15).
//
// EGLImage import goes around that function entirely: we allocate the BO
// ourselves at two bytes a texel, export it as a dma-buf and hand it to the
// sampler.  This probe asks whether that path exists on this board and whether
// the sampler reads the nibbles correctly.
//
// Build on the board:  g++ -O2 -std=c++17 -o eglimg egl_image_probe.cpp -lEGL -lGLESv2

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
#include <cstring>
#include <vector>

static const char* has(const char* list, const char* name) {
    if (!list) return nullptr;
    size_t n = strlen(name);
    for (const char* p = list; (p = strstr(p, name)); p += n)
        if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0')) return p;
    return nullptr;
}

int main() {
    EglSurfaceless egl = egl_surfaceless_init(2);
    if (!egl.ok()) return 1;

    const char* gext = (const char*)glGetString(GL_EXTENSIONS);
    const char* eext = eglQueryString(egl.dpy, EGL_EXTENSIONS);
    printf("A renderer   %s\n", (const char*)glGetString(GL_RENDERER));
    printf("A GL         OES_EGL_image=%s  OES_EGL_image_external=%s\n",
           has(gext, "GL_OES_EGL_image") ? "yes" : "no",
           has(gext, "GL_OES_EGL_image_external") ? "yes" : "no");
    printf("A EGL        image_base=%s  dma_buf_import=%s  dma_buf_modifiers=%s\n",
           has(eext, "EGL_KHR_image_base") ? "yes" : "no",
           has(eext, "EGL_EXT_image_dma_buf_import") ? "yes" : "no",
           has(eext, "EGL_EXT_image_dma_buf_import_modifiers") ? "yes" : "no");
    if (!has(gext, "GL_OES_EGL_image") || !has(eext, "EGL_EXT_image_dma_buf_import")) {
        printf("A verdict    this stack has no dma-buf image import; stop here\n");
        return 0;
    }

    // ---- B: a two-byte-per-texel BO we allocate ourselves -------------------
    const int W = 64, H = 16;
    int fd = open("/dev/dri/card0", O_RDWR);
    if (fd < 0) { printf("B drm        open card0 failed: %s\n", strerror(errno)); return 0; }
    struct drm_vc4_create_bo cbo = {};
    cbo.size = (uint32_t)(W * H * 2);
    if (ioctl(fd, DRM_IOCTL_VC4_CREATE_BO, &cbo) != 0) {
        printf("B drm        VC4_CREATE_BO failed: %s\n", strerror(errno));
        return 0;
    }
    struct drm_vc4_mmap_bo mbo = {};
    mbo.handle = cbo.handle;
    if (ioctl(fd, DRM_IOCTL_VC4_MMAP_BO, &mbo) != 0) {
        printf("B drm        VC4_MMAP_BO failed: %s\n", strerror(errno));
        return 0;
    }
    void* map = mmap(nullptr, cbo.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mbo.offset);
    if (map == MAP_FAILED) { printf("B drm        mmap failed: %s\n", strerror(errno)); return 0; }
    // Nibbles the way the engine packs them: R is the most significant, and the
    // probe's kernel-side convention is texel .x..w = columns 0..3 reversed at
    // upload.  For a feasibility answer, one value per texel is enough.
    uint16_t* px = (uint16_t*)map;
    for (int i = 0; i < W * H; ++i) px[i] = (uint16_t)(0x1111 * ((i % 15) + 1));
    munmap(map, cbo.size);
    int dmabuf = -1;
    if (drmPrimeHandleToFD(fd, cbo.handle, DRM_CLOEXEC, &dmabuf) != 0) {
        printf("B drm        PRIME_HANDLE_TO_FD failed: %s\n", strerror(errno));
        return 0;
    }
    printf("B drm        %dx%d at 2 bytes a texel = %u B, dma-buf fd %d\n",
           W, H, cbo.size, dmabuf);

    // ---- C: import it and sample -------------------------------------------
    auto create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    auto image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!create_image || !image_target) {
        printf("C import     eglCreateImageKHR/glEGLImageTargetTexture2DOES missing\n");
        return 0;
    }
    // B4G4R4A4 (Mesa's PIPE_FORMAT_B4G4R4A4_UNORM) is DRM's ARGB4444 family;
    // try both ends of it, since which one matches is a table on Mesa's side.
    const uint32_t fourccs[] = { DRM_FORMAT_ARGB4444, DRM_FORMAT_ABGR4444 };
    const char* names[] = { "ARGB4444", "ABGR4444" };
    for (int k = 0; k < 2; ++k) {
        EGLint attrs[] = {
            EGL_WIDTH, W,
            EGL_HEIGHT, H,
            EGL_LINUX_DRM_FOURCC_EXT, (EGLint)fourccs[k],
            EGL_DMA_BUF_PLANE0_FD_EXT, dmabuf,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, W * 2,
            EGL_NONE
        };
        EGLImageKHR img = create_image(egl.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, attrs);
        if (img == EGL_NO_IMAGE_KHR) {
            printf("C import     %s rejected (0x%x)\n", names[k], eglGetError());
            continue;
        }
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        while (glGetError() != GL_NO_ERROR) {}
        image_target(GL_TEXTURE_2D, img);
        GLenum err = glGetError();
        printf("C import     %s accepted%s\n", names[k],
               err ? " (but the bind raised an error)" : "");
        if (err) { glDeleteTextures(1, &tex); continue; }

        // Sample texel (0,0) and (1,0) through a 1x1 render and read back.
        GLuint fbo = 0, rt = 0;
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &rt);
        glBindTexture(GL_TEXTURE_2D, rt);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
        static const char* vs = "attribute vec2 p;varying vec2 uv;void main(){uv=p*0.5+0.5;gl_Position=vec4(p,0.,1.);}";
        static const char* fs =
            // Sample the centres of texels 0 and 1 exactly, so the read-back is
            // a comparison against the pattern and not an interpolation.
            "precision highp float;uniform sampler2D T;uniform vec2 TS;"
            "void main(){vec2 uv=(gl_FragCoord.xy-0.5)*TS+0.5*TS;"
            "vec4 c=texture2D(T,uv);gl_FragColor=vec4(c.x*15.0,c.y*15.0,c.z*15.0,c.w*15.0)/255.0;}";
        auto sh = [](GLenum t, const char* s) { GLuint h = glCreateShader(t); glShaderSource(h, 1, &s, nullptr); glCompileShader(h); return h; };
        GLuint pr = glCreateProgram();
        glAttachShader(pr, sh(GL_VERTEX_SHADER, vs));
        glAttachShader(pr, sh(GL_FRAGMENT_SHADER, fs));
        glLinkProgram(pr);
        glUseProgram(pr);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(pr, "T"), 0);
        glUniform2f(glGetUniformLocation(pr, "TS"), 1.0f / W, 1.0f / H);
        GLuint vbo = 0;
        static const float tri[] = { -1, -1, 3, -1, -1, 3 };
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
        GLint loc = glGetAttribLocation(pr, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glViewport(0, 0, 2, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        uint8_t back[8] = {};
        glReadPixels(0, 0, 2, 1, GL_RGBA, GL_UNSIGNED_BYTE, back);
        printf("C sample     texel0 = %d %d %d %d   texel1 = %d %d %d %d  (expect 1 1 1 1 / 2 2 2 2)\n",
               back[0], back[1], back[2], back[3], back[4], back[5], back[6], back[7]);
        break;
    }

    close(fd);
    return 0;
}
