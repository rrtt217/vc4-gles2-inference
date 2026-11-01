// What texture height does this VC4 driver actually accept?
// The engine allocates weight textures with height = rows (2432 for w1/w3), so
// if the limit were 2048 those uploads would fail silently.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <cstdio>
#include <vector>
#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    auto gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay d = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (!eglInitialize(d, nullptr, nullptr)) { printf("egl init failed\n"); return 1; }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint attrs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                       EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE };
    EGLConfig cfg; EGLint n = 0; eglChooseConfig(d, attrs, &cfg, 1, &n);
    EGLint ca[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ca);
    eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

    GLint mx = 0; glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mx);
    printf("GL_MAX_TEXTURE_SIZE = %d\n", mx);
    while (glGetError() != GL_NO_ERROR) {}

    for (int h : {512, 1024, 2048, 2432, 4096}) {
        const int w = 192;
        std::vector<unsigned char> data((size_t)w * h * 4, 7);
        GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        GLenum e = glGetError();
        printf("w=%d h=%d -> err=0x%x\n", w, h, e);
        glDeleteTextures(1, &t);
    }
    return 0;
}
