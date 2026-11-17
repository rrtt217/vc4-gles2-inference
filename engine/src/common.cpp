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
void Progress::begin(const char* what, uint64_t total) {
        end();
        if (total == 0 || !isatty(STDERR_FILENO)) return;
        if (getenv("SHADER_TIME")) return;   // those STARTUP lines are grep targets
        on_ = true;
        what_ = what;
        total_ = total;
        last_ = Clock::now() - chrono::milliseconds(500);
        set(0);
}

void Progress::set(uint64_t done) {
        if (!on_) return;
        const auto now = Clock::now();
        const bool last = done >= total_;
        if (!last && elapsed_ms(last_, now) < 100.0) return;
        last_ = now;
        double pct = 100.0 * (double)done / (double)total_;
        if (pct > 100.0) pct = 100.0;
        char bar[25];
        const int filled = (int)(pct / 100.0 * 24.0 + 0.5);
        for (int i = 0; i < 24; ++i) bar[i] = i < filled ? '#' : '-';
        bar[24] = 0;
        fprintf(stderr, "\r[%-6s %3.0f%% %s %6.1f/%6.1f MB\x1b[K", what_, pct, bar,
                (double)done / (1024.0 * 1024.0), (double)total_ / (1024.0 * 1024.0));
}

void Progress::end() {
        if (!on_) return;
        fprintf(stderr, "\r\x1b[K");
        on_ = false;
}
