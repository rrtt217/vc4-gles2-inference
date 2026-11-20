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


enum class WeightType { INT8, INT16, INT4, INT6, T3 };

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
    // A 2-bit ternary body (dtype 7): 16 weights a texel, four a byte, bit 0
    // magnitude and bit 1 sign.  Borrowed from the container mapping -- the
    // whole point of the format is that this is the texture's bytes already.
    const uint8_t* ext3 = nullptr;
    vector<uint8_t> q3;
    // The 2-bit body reaches the GPU as a 2D tile of at most `t3_tile_h` rows
    // (VC4's GL_MAX_TEXTURE_SIZE is 2048 and the FFN gate is 4096 rows): matrix
    // texel (x, y) lives at (x + (y/t3_tile_h)*t3_tile_w, y % t3_tile_h).  A body
    // that fits is one tile, which is the whole model except those two matrices.
    int t3_tile_w = 0, t3_tile_h = 0, t3_tiles_n = 0;
    // Both numbers come from gpu_upload.cpp, which is the only place that knows
    // the texture it made; `t3_tiles_n` is its tile count, not a recomputation.
    int t3_tiles() const { return t3_tiles_n > 0 ? t3_tiles_n : 1; }                 // only when the mapping is not borrowed
    bool is_t3() const { return type == WeightType::T3; }
    const uint8_t* w3() const { return ext3 ? ext3 : q3.data(); }
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

// The bytes a matrix's weights occupy, for accounting (the upload progress bar).
// Packed layouts are counted packed; a q4 matrix also carries one int16 scale and
// minimum per group.  A per-row scale plane (part 3) is not counted for any kind,
// so the total is the container's weight bytes, not its file size.
//
// The switch is exhaustive on purpose, with no `default:`.  A t3 body was added
// to WeightType without a case here and fell through to `n` -- four times its
// real size -- so BitCPM's 167 MB container printed "413.7/413.7 MB" while the
// load line next to it said 85 MB of t3 texture (2026-11-20).  -Wswitch now names
// any kind added later and `warncheck` fails the gate on it.
inline size_t mat_bytes(const Mat& m) {
    const size_t n = (size_t)m.rows * (size_t)m.cols;
    switch (m.type) {
        case WeightType::INT8: return n;
        case WeightType::INT16: return n * 2;
        case WeightType::INT4: return n / 2 + (size_t)m.rows * (size_t)m.groups * 2 * sizeof(int16_t);
        // Two planes, a 4-bit low and a 2-bit high one (`mmcf_hf.q6mat`): 6 bits
        // a value, not one byte.  Only the tied head uses it.
        case WeightType::INT6: return n / 2 + n / 4;
        // 2 bits a weight, in the container's own texel grid: an RGBA8 texel is
        // 2 rows by 8 columns, so ((rows+1)/2) * ((cols+7)/8) * 4 bytes -- n/4
        // for a shape the packer accepts.  This is the length `as_t3()` borrows.
        case WeightType::T3: return (((size_t)m.rows + 1) / 2) * ((((size_t)m.cols + 7) / 8) * 4);
    }
    return n;
}

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
// What the REPL's one-line status shows after a turn.  TurnTiming's destructor is
// the only code that sees every exit, so it copies the numbers here rather than
// letting the caller re-derive them (and miss the interrupted turns).
struct TurnReport {
        double prefill_s = 0;
        double decode_ms_per_token = 0;
        int generated = 0;
        const char* cut = nullptr;   // nullptr = the turn ended on its own
        // The decode loop's own launch counters (DRAWCOUNT / SYNCCOUNT): the
        // prefill's draws are a prompt-specific extra, so a draws-per-token
        // figure has to start after the last prompt token.  See
        // docs/test-hooks.md.
        long draws_decode = 0;
        long syncs_decode = 0;
};

struct TurnTiming {
        Clock::time_point start = Clock::now();
        Clock::time_point decode_start {};
        int generated = 0;
        bool decoding = false;
        TurnReport* out = nullptr;   // optional: set `out->...` at the end
        const char* cut = nullptr;

        void begin_decode() {
                decode_start = Clock::now();
                decoding = true;
        }
        ~TurnTiming() {
                const auto now = Clock::now();
                const double prefill = elapsed_ms(start, decoding ? decode_start : now) / 1000.0;
                const double avg = (decoding && generated)
                                       ? elapsed_ms(decode_start, now) / generated
                                       : 0.0;
                if (out) {
                        out->prefill_s = prefill;
                        out->decode_ms_per_token = avg;
                        out->generated = generated;
                        out->cut = cut;
                }
                const char* on = getenv("TIMING");
                if (!on || !*on) return;
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

// A field out of /proc/meminfo by name prefix (CmaFree, MemFree); -1 if absent.
long long proc_meminfo_kb(const char* key);

// A one-line progress bar on stderr for the waits a start or a turn spends time
// in: uploading the weights, prefilling a prompt, a --ppl run's windows.  Off
// unless stderr is a terminal, so a pipe, the worker protocol and the regression
// scripts never see a '\r' or half a line -- and off when SHADER_TIME is set,
// whose STARTUP lines are grep targets.  The engine owns exactly one (g_progress):
// the shader links run *inside* a prefill, from gpu_prog.cpp, and count
// themselves on it as a side figure -- a local instance would draw a bar that
// never showed them.
//
// A phase has to last kShowAfterMs before the first line is drawn: a two-token
// prefill on a desktop is ~30 ms, and a bar that appears and erases for 30 ms is
// a blink, not information.
class Progress {
public:
        Progress() = default;
        Progress(const Progress&) = delete;
        Progress& operator=(const Progress&) = delete;
        // A phase with a size: bytes (shown in MB) or whole units (tokens).
        void begin(const char* what, uint64_t total);
        void begin_count(const char* what, uint64_t total, const char* unit);
        // A phase whose size nobody can state before running it -- how many
        // windows a --ppl run needs depends on its own warm-up arithmetic -- so
        // it shows a count and a clock instead of a percentage that would be
        // wrong.  A bar that ends at 90% is worse than no bar.
        void begin_open(const char* what, const char* unit);
        void side_bump(const char* label);            // "shaders 12": a count, no total
        void set(uint64_t done);
        void end();
        ~Progress() { end(); }
        bool active() const { return on_; }
private:
        void init(const char* what, uint64_t total, const char* unit, bool count, bool open);
        void maybe_draw(bool final);
        void draw(uint64_t done);
        bool on_ = false;
        bool count_ = false;        // whole units, not MB
        bool open_ = false;         // no total: never "done"
        bool drawn_ = false;
        const char* what_ = "";
        const char* unit_ = "MB";
        const char* side_label_ = nullptr;
        uint64_t total_ = 0;
        uint64_t done_ = 0;
        uint64_t side_ = 0;
        Clock::time_point start_ {}, last_ {};
};
extern Progress g_progress;

// A phase's bar, in a scope: begun at construction, erased at destruction, so the
// early returns a Ctrl-C and a full context take cannot leave half a line on
// stderr.  It drives the global because that is the object the shader links are
// counted on.
class ProgressScope {
public:
        ProgressScope(const char* what, uint64_t total);                     // bytes, in MB
        ProgressScope(const char* what, uint64_t total, const char* unit);   // whole units
        ~ProgressScope() { g_progress.end(); }
        ProgressScope(const ProgressScope&) = delete;
        ProgressScope& operator=(const ProgressScope&) = delete;
        void set(uint64_t done) { g_progress.set(done); }
        void end() { g_progress.end(); }
};

// Environment switches, read one way.  Reading the same variable twice in one
// expression (`getenv(X) ? atoi(getenv(X)) : d`) is both slow in a per-layer loop
// and the shape where the two reads could in principle disagree; 21 sites did
// that when these helpers landed.  New code uses these.  Every switch has a row
// in switches.cpp, which is the inventory (a count here would be a promise
// nobody can keep -- `./engine_mm3 --dump-switches` prints the current one).
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
