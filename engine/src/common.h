#pragma once

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


enum class WeightType { INT8, INT16, INT4 };

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
    bool borrowed() const { return ext8 || ext16 || ext4; }
    // MMCF q8u8 tensors carry the +128 bias already applied, so the upload
    // copies the bytes to the texture instead of flipping the sign bit.
    bool prebiased = false;
};

struct Target {
    GLuint tex = 0, fbo = 0;
    int w = 0, h = 0;
};

extern volatile sig_atomic_t g_stop;
void install_stop_handlers();
double elapsed_ms(Clock::time_point a, Clock::time_point b);

inline int16_t quantize_act(float x) {
    long v = lrintf(x * ACT_SCALE);
    if (v < -32768) v = -32768;
    if (v > 32767) v = 32767;
    return (int16_t)v;
}
inline float dequantize_act(float q) { return q / ACT_SCALE; }
inline float quantize_act_f(float x) { return (float)quantize_act(x) / ACT_SCALE; }

bool read_text_file(const string& path, string& out);
