// One surfaceless EGL + GLES2 context, for probes.
//
// `engine/probes/README.md` records that eight probes carry their own copy of
// this set-up, four of them from one commit, and asks the next person to add a
// probe to extract it instead of copying it a ninth time.  This is that
// extraction, deliberately a header and not a library: a probe is one
// translation unit, and two of them are cross-built without linking anything
// but EGL and GLESv2.
//
//     EglSurfaceless egl = egl_surfaceless_init(2);
//     if (!egl.ok()) return 1;
//
// It prints its own diagnostics on failure, so the caller only has to decide
// what a failed context means.
#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <cstdio>

struct EglSurfaceless {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext ctx = EGL_NO_CONTEXT;
    bool ok() const { return dpy != EGL_NO_DISPLAY && ctx != EGL_NO_CONTEXT; }
};

// version is EGL_CONTEXT_CLIENT_VERSION: 2 asks for ES2, which is all vc4
// reports; asking for 2 on a driver that offers more is harmless.
static inline EglSurfaceless egl_surfaceless_init(int version = 2) {
    EglSurfaceless e;
    auto get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!get_platform_display) {
        printf("egl: no eglGetPlatformDisplayEXT\n");
        return e;
    }
    e.dpy = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    EGLint major = 0, minor = 0;
    if (!eglInitialize(e.dpy, &major, &minor)) {
        printf("egl: eglInitialize failed\n");
        e.dpy = EGL_NO_DISPLAY;
        return e;
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint cfg_attr[] = { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                EGL_ALPHA_SIZE, 8, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE };
    EGLConfig cfg;
    EGLint n = 0;
    if (!eglChooseConfig(e.dpy, cfg_attr, &cfg, 1, &n) || n == 0) {
        printf("egl: no matching config\n");
        return e;
    }
    const EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, version, EGL_NONE };
    e.ctx = eglCreateContext(e.dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (e.ctx == EGL_NO_CONTEXT) {
        printf("egl: context creation failed\n");
        return e;
    }
    if (!eglMakeCurrent(e.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, e.ctx)) {
        printf("egl: eglMakeCurrent(surfaceless) failed\n");
        e.ctx = EGL_NO_CONTEXT;
        return e;
    }
    return e;
}
