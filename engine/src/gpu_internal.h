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

// The largest number of fragments one draw may cover.  VC4 processes a fragment
// per cycle and the driver's scratch target is 2048 wide, so a wider strip either
// exceeds the target or stalls the block; 1024 is what keeps `draw()` and
// `matvec_add()` splitting (docs/engine-internals.md, "draw splitting").  It was a
// bare 1024 in seven places with no name and no check, and a wrong width fails
// *silently* -- the read-back returns the previous contents of the target, which
// looks like a plausible number.  scripts/check_draw_cap.sh keeps it named.
constexpr int kDrawCap = 1024;
