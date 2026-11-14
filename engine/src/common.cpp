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
