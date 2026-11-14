#include "common.h"

#include <csignal>
#include <fstream>
#include <sstream>

volatile sig_atomic_t g_stop = 0;

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

bool env_set(const char* name) { return getenv(name) != nullptr; }

bool env_flag(const char* name) {
    const char* v = getenv(name);
    return v && strcmp(v, "0") != 0;
}

bool env_zero(const char* name) {
    const char* v = getenv(name);
    return v && strcmp(v, "0") == 0;
}

int env_int(const char* name, int def) {
    const char* v = getenv(name);
    return v ? atoi(v) : def;
}

double env_double(const char* name, double def) {
    const char* v = getenv(name);
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
