#pragma once
// State that gpu.cpp and its split-out translation units share.
//
// The GL bind cache and the upload counters are file-static today and the engine
// has exactly one GL context per process, so they are process state in all but
// name.  `gl_state_forget_bindings()` invalidates the cache: anything that
// creates, rebinds or deletes a texture/FBO/program must call it, which is why it
// cannot stay private to one file once the factories move.
//
// gpu_state.cpp owns the definitions; gpu.cpp (and any future gpu_*.cpp) reads
// them through here.

#include "gpu.h"

extern int g_up_tex;             // count, for UPLOAD_TIME
extern double g_up_tex_ms;

extern GLuint g_bound_tex[8];    // bind cache: unit -> texture, 0 = unknown
extern GLuint g_cur_fbo;
extern GLuint g_cur_prog;

void gl_state_forget_bindings();
