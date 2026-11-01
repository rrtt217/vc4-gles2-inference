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

static constexpr uint32_t MAGIC_Q8 = 0x51384D31;      // Q8M1
static constexpr uint32_t MAGIC_I16 = 0x4931364D;     // I16M1
static constexpr uint32_t MAGIC_Q8_V2 = 0x51384D32;   // Q8M2
static constexpr uint32_t MAGIC_I16_V2 = 0x4931364E;  // I16N
static constexpr uint32_t MAGIC_Q4 = 0x51344D31;      // Q4M1

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
    GLuint tex = 0, q4_param_tex = 0;
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
