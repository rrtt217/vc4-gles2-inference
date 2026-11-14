#pragma once
#include <iostream>

// Shared vocabulary for the engine: quantized matrices, render targets and the
// fixed-point conventions used by every kernel.
#include <GLES2/gl2.h>

#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

static constexpr float ACT_SCALE = 256.0f;  // q16 activation scale


enum class WeightType { INT8, INT16, INT4, INT6 };

struct Mat {
    int rows = 0, cols = 0;
    WeightType type = WeightType::INT8;
    vector<int8_t> q8;
    vector<int16_t> q16;
    vector<uint8_t> q4;                 // packed nibbles, low nibble first
    vector<int16_t> q4_scale, q4_min;   // per row/group, quantized
    int group = 32, groups = 0;
    float q4_param_scale = 1.0f;
    float scale = 1.0f;
    // One weight scale per output row (MMCF tensor id part 3).  Empty when the
    // matrix has a single per-matrix scale, which is every container written
    // before this existed.  The plane is uploaded as one 24-bit value per row
    // and the shader multiplies the row's dot product by it instead of `scale`.
    // q6 stores a 4-bit low plane and a 2-bit high plane in two textures (VC4
    // cannot fetch one sampler twice in a fragment).  Only the tied head uses
    // it: the logits kernel decodes lo | hi<<4 and the rows carry a scale.
    vector<uint8_t> q6_lo, q6_hi;
    const uint8_t* ext6_lo = nullptr;
    const uint8_t* ext6_hi = nullptr;
    GLuint tex_lo = 0, tex_hi = 0;
    vector<float> row_scale;
    // A fused W13 draw evaluates rows r and r+rows in one fragment, so its
    // plane is packed as (t, t+rows/2) instead of (2t, 2t+1) and the kernel
    // reads it once.  Two fetches from one sampler are undefined on VC4.
    bool row_scale_hi = false;
    bool has_row_scale() const { return !row_scale.empty(); }
    GLuint tex = 0, q4_param_tex = 0, row_scale_tex = 0;
    // Borrowed weights: when one of these is set the matching vector above is
    // empty and the bytes live in the checkpoint mmap (page cache) instead of
    // anonymous memory.  A 69 MB q8 checkpoint then costs no RAM, stays
    // cached across runs and reloads in tens of milliseconds.  Readers must
    // go through the accessors below.
    const int16_t* ext16 = nullptr;
    const int8_t* ext8 = nullptr;
    const uint8_t* ext4 = nullptr;
    const int16_t* ext_scale4 = nullptr;
    const int16_t* ext_min4 = nullptr;
    const int8_t* w8() const { return ext8 ? ext8 : q8.data(); }
    const int16_t* w16() const { return ext16 ? ext16 : q16.data(); }
    const uint8_t* w4() const { return ext4 ? ext4 : q4.data(); }
    const int16_t* s4() const { return ext_scale4 ? ext_scale4 : q4_scale.data(); }
    const int16_t* m4() const { return ext_min4 ? ext_min4 : q4_min.data(); }
    const uint8_t* lo6() const { return ext6_lo ? ext6_lo : q6_lo.data(); }
    const uint8_t* hi6() const { return ext6_hi ? ext6_hi : q6_hi.data(); }
    bool borrowed() const { return ext8 || ext16 || ext4; }
    // MMCF q8u8 tensors carry the +128 bias already applied, so the upload
    // copies the bytes to the texture instead of flipping the sign bit.
    bool prebiased = false;
    // Default-off probe (Q4_FMT=rgba4): bind an INT4 matrix as GL_RGBA4 at
    // twice the width, so one texel is four weights and the kernel decodes
    // nothing.  The bytes are the container's own; only the view changes.
    bool fmt4 = false;
};

struct Target {
    GLuint tex = 0, fbo = 0;
    int w = 0, h = 0;
};

extern volatile sig_atomic_t g_stop;
void install_stop_handlers();
double elapsed_ms(Clock::time_point a, Clock::time_point b);

// TIMING has to survive every way a turn can end: Ctrl-C during prefill, Ctrl-C
// during decode, a context that filled up, or the normal stop.  The early
// returns that mark an interruption are exactly the runs someone wants numbers
// for, which is why the old inline print - at the end of the happy path - was
// silent in every case that mattered.  A destructor runs on all of them; `cut`
// names the reason when the turn did not finish on its own, so a short `gen=`
// is never ambiguous.
struct TurnTiming {
        Clock::time_point start = Clock::now();
        Clock::time_point decode_start {};
        int generated = 0;
        bool decoding = false;
        const char* cut = nullptr;

        void begin_decode() {
                decode_start = Clock::now();
                decoding = true;
        }
        ~TurnTiming() {
                const char* on = getenv("TIMING");
                if (!on || !*on) return;
                const auto now = Clock::now();
                const double prefill = elapsed_ms(start, decoding ? decode_start : now) / 1000.0;
                const double avg = (decoding && generated)
                                       ? elapsed_ms(decode_start, now) / generated
                                       : 0.0;
                cerr << "TIMING prefill=" << prefill << "s decode_avg=" << avg
                     << "ms/token gen=" << generated;
                if (cut) cerr << " cut=" << cut;
                cerr << "\n";
        }
};


inline int16_t quantize_act(float x) {
    long v = lrintf(x * ACT_SCALE);
    if (v < -32768) v = -32768;
    if (v > 32767) v = 32767;
    return (int16_t)v;
}
inline float dequantize_act(float q) { return q / ACT_SCALE; }
inline float quantize_act_f(float x) { return (float)quantize_act(x) / ACT_SCALE; }

bool read_text_file(const string& path, string& out);

// 64-bit FNV-1a, hex in the sidecars: provenance, not cryptography.  It only has
// to notice that an input changed (the system prompt, the tool list, the tokens).
uint64_t fnv64(const std::string& s);

// A prompt segment can contain newlines and control characters; printing it raw
// splits one logical line and scatters the PROMPT tags of CHAT_DUMP=1.
std::string dump_escape(const std::string& t);

// Environment switches, read one way.  The engine has 93 of them and 206 call
// sites, and 21 of those read the same variable twice
// (`getenv(X) ? atoi(getenv(X)) : d`), which is both slow in a per-layer loop and
// the shape where the two reads could in principle disagree.  New code uses these;
// the remaining one-shot reads are converted as they are touched.
//
// `env_flag` is "set and not 0", which is the convention every switch follows
// (docs/test-hooks.md); `env_set` is only set-ness, for switches that take a value
// (`STAGE_DUMP=1`) or ignore it (`CHAT_DUMP=1`).
bool env_set(const char* name);
// Set and equal to "0": the switch was explicitly turned off.  This is *not*
// `!env_flag` -- that is also true when the variable is unset, and inverting an
// off-by-default switch that way moved q4 PPL on 2026-11-14 (330.5302 -> 326.5032,
// caught by the gate).  `FFN_PAIR=0` disables pairing; unset keeps the default.
bool env_zero(const char* name);
bool env_flag(const char* name);
int env_int(const char* name, int def);
double env_double(const char* name, double def);
