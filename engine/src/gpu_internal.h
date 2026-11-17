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

#include <string>

extern int g_up_tex;             // count, for UPLOAD_TIME
extern double g_up_tex_ms;

extern GLuint g_bound_tex[8];    // bind cache: unit -> texture, 0 = unknown
extern GLuint g_cur_fbo;
extern GLuint g_cur_prog;

void gl_state_forget_bindings();

// The program layer (gpu_prog.cpp) shares three things with the kernel builders:
// the key of the program being built (diagnostics name it), whether the q4 nibble
// unpack should be emitted inline, and whether the driver accepted ESSL 3.00.
extern std::string g_prog_key;
extern bool g_q4_shift_ok;
// The driver offers ESSL 3.00 (the MESA override was requested and accepted).
// Generalizes g_q4_shift_ok: the q4 nibble split and the RWKV bit-cast f32 state
// kernels both need it.  g_q4_shift_ok = g_es3_ok && Q4_SHIFT wanted.
extern bool g_es3_ok;
bool gpu_es3_ok();
bool q4_inline(const char* name);
bool q4_shift_wanted();   // Q4_SHIFT: ask for ESSL 3.00 q4 kernels

// FFN_FLUSH / FFN_SYNC are measurement hooks: flushing the command stream after
// each FFN pass makes a per-pass timing meaningful, and FFN_SYNC adds the wait.
// They used to be written out 14 times in gpu.cpp, each pair doing two getenv()
// per pass, per layer, per token (docs/test-hooks.md).  These read them once.
bool ffn_flush_wanted(bool fast_finish);  // fast_finish || FFN_FLUSH is set
void ffn_sync();                          // glFlush+glFinish if FFN_SYNC is set

// The largest number of fragments one draw may cover.  VC4 processes a fragment
// per cycle and the driver's scratch target is 2048 wide, so a wider strip either
// exceeds the target or stalls the block; 1024 is what keeps `draw()` and
// `matvec_add()` splitting (docs/engine-internals.md, "draw splitting").  It was a
// bare 1024 in seven places with no name and no check, and a wrong width fails
// *silently* -- the read-back returns the previous contents of the target, which
// looks like a plausible number.  gate/check_draw_cap.sh keeps it named.
constexpr int kDrawCap = 1024;

// gpu_dmabuf.cpp: a 4-bit texture allocated and imported by hand -- the only
// way to get a two-byte-per-texel surface out of this driver.
void gpu_dmabuf_init(EGLDisplay dpy);
bool gpu_dmabuf_available();
GLuint gpu_dmabuf_texture(int w, int h, const uint16_t* data);
void gpu_dmabuf_release_all();
int gpu_dmabuf_count();
