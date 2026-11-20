#include "common.h"
#include "switches.h"

#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

volatile sig_atomic_t g_stop = 0;
Progress g_progress;

static void on_stop_signal(int) { g_stop = 1; }

void install_stop_handlers() {
    struct sigaction sa {};
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: interrupt getline()
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return chrono::duration<double, milli>(b - a).count();
}

bool read_text_file(const string& path, string& out) {
    ifstream f(path, ios::binary);
    if (!f) return false;
    stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Every read goes through the switch registry (switches.cpp): it is where a
// legacy spelling is rewritten to its axis name, where a closed set is
// validated, and where `--dump-switches` gets what this process resolved.
bool env_set(const char* name) { return sw_isset(name); }

bool env_flag(const char* name) {
    const char* v = sw_str(name);
    return v && strcmp(v, "0") != 0;
}

bool env_zero(const char* name) {
    const char* v = sw_str(name);
    return v && strcmp(v, "0") == 0;
}

int env_int(const char* name, int def) {
    const char* v = sw_str(name);
    return v ? atoi(v) : def;
}

double env_double(const char* name, double def) {
    const char* v = sw_str(name);
    return v ? atof(v) : def;
}

uint64_t fnv64(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ull; }
    return h;
}

std::string dump_escape(const std::string& t) {
    std::string o;
    for (unsigned char c : t) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o += (char)c;
    }
    return o;
}

long long proc_meminfo_kb(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    const size_t n = strlen(key);
    while (std::getline(f, line))
        if (line.compare(0, n, key) == 0) return atoll(line.c_str() + n + 1);
    return -1;
}


// ---------------------------------------------------------------------------
// Progress: one line, on stderr, only for a human at a terminal.  The bar is
// drawn at most ten times a second; on the Pi a container read is 6 s of SD card
// and a redraw per record would cost more than the load.
// ---------------------------------------------------------------------------
static constexpr double kShowAfterMs = 150.0;   // a shorter phase never draws
static constexpr double kRedrawMs = 100.0;

void Progress::init(const char* what, uint64_t total, const char* unit, bool count, bool open) {
        end();
        on_ = false;
        // A side count belongs to one phase: a swap's upload must not inherit
        // the links the previous model compiled during its last prefill.
        side_ = 0;
        side_label_ = nullptr;
        count_ = count;
        open_ = open;
        what_ = what;
        unit_ = unit;
        total_ = total;
        // A phase starts at zero: a bump can draw before the first `set`, and an
        // inherited count from the previous phase (the upload's 60.9 MB on the
        // prefill's first frame) is not a number anyone can read.
        done_ = 0;
        // A total of zero is a phase with nothing to do, not a phase of unknown
        // size -- that is begin_open(), which keeps the line.
        if (total == 0 && !open) return;
        if (!isatty(STDERR_FILENO)) return;
        if (getenv("SHADER_TIME")) return;   // those STARTUP lines are grep targets
        start_ = last_ = Clock::now();
        drawn_ = false;
        on_ = true;
}

void Progress::begin(const char* what, uint64_t total) { init(what, total, "MB", false, false); }

void Progress::begin_count(const char* what, uint64_t total, const char* unit) {
        init(what, total, unit, true, false);
}

void Progress::begin_open(const char* what, const char* unit) { init(what, 0, unit, true, true); }

ProgressScope::ProgressScope(const char* what, uint64_t total) { g_progress.begin(what, total); }

ProgressScope::ProgressScope(const char* what, uint64_t total, const char* unit) {
        g_progress.begin_count(what, total, unit);
}

void Progress::side_bump(const char* label) {
        if (!on_) return;
        side_label_ = label;
        ++side_;
        // Redraw here too, not only on the phase's own `set`.  A link can take a
        // second on the board -- the first prefill token pays every program
        // before it can report anything -- so a count that only moved at token
        // boundaries stayed invisible for exactly the wait it describes (measured
        // on the Pi: the first frame appeared at 4.0 s with `shaders 16` already
        // there, 16 links that the bar never showed arriving).
        maybe_draw(false);
}

void Progress::set(uint64_t done) {
        if (!on_) return;
        done_ = done;
        maybe_draw(!open_ && done >= total_);
}

void Progress::maybe_draw(bool final) {
        const auto now = Clock::now();
        if (!drawn_) {
                // The line is for a wait: a phase that is already over before it
                // is visible would only have blinked.
                if (elapsed_ms(start_, now) < kShowAfterMs) return;
        } else if (!final && elapsed_ms(last_, now) < kRedrawMs) {
                return;
        }
        last_ = now;
        drawn_ = true;
        draw(done_);
}

void Progress::draw(uint64_t done) {
        const double secs = elapsed_ms(start_, Clock::now()) / 1000.0;
        char side[40] = "";
        if (side_label_)
                snprintf(side, sizeof side, "  %s %llu", side_label_, (unsigned long long)side_);
        if (open_) {
                fprintf(stderr, "\r[%-8s %6llu %-3s %5.1f s%s\x1b[K", what_,
                        (unsigned long long)done, unit_, secs, side);
                return;
        }
        double pct = 100.0 * (double)done / (double)total_;
        if (pct > 100.0) pct = 100.0;
        char bar[25];
        const int filled = (int)(pct / 100.0 * 24.0 + 0.5);
        for (int i = 0; i < 24; ++i) bar[i] = i < filled ? '#' : '-';
        bar[24] = 0;
        // A counted phase prints whole units (a token is not 14.0 tokens); a byte
        // phase prints MB with one decimal, which is how a container's size is
        // quoted everywhere else in this tree.
        const int dec = count_ ? 0 : 1;
        const double dv = count_ ? (double)done : (double)done / (1024.0 * 1024.0);
        const double tv = count_ ? (double)total_ : (double)total_ / (1024.0 * 1024.0);
        fprintf(stderr, "\r[%-8s %3.0f%% %s %6.*f/%6.*f %-3s %5.1f s%s\x1b[K", what_, pct, bar,
                dec, dv, dec, tv, unit_, secs, side);
}

void Progress::end() {
        if (!on_) return;
        fprintf(stderr, "\r\x1b[K");
        on_ = false;
}
