// Shared assertion helpers for the compiled tests.
//
// There is no test framework here on purpose.  Each test is a program with a
// main() that calls these helpers and returns checks::g_fail ? 1 : 0, and
// `make -C engine test` runs the binaries.  The features a framework would add
// -- discovery, fixtures, parameterised cases, mocking -- are not used by a
// test whose job is "pack this known vector and compare the bytes"; the compiler
// is the runner.  If that changes, a single-header framework can replace this
// file without rewriting the test bodies.
//
// A failure prints file:line and both values, and keeps going, so one run lists
// every broken check instead of the first.
#pragma once

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

namespace checks {

inline int g_fail = 0;

inline std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 32 || c == 127) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02x", c);
            o += b;
        } else {
            o.push_back((char)c);
        }
    }
    return o;
}

inline void fail(const char* file, int line, const std::string& msg) {
    ++g_fail;
    std::fprintf(stderr, "FAIL %s:%d  %s\n", file, line, msg.c_str());
}

inline void check_true(bool ok, const char* what, const char* file, int line) {
    if (!ok) fail(file, line, what);
}

inline void check_ll(long long got, long long want, const char* what,
                     const char* file, int line) {
    if (got == want) return;
    fail(file, line, std::string(what) + ": got " + std::to_string(got) +
                         ", want " + std::to_string(want));
}

inline void check_str(const std::string& got, const std::string& want, const char* what,
                      const char* file, int line) {
    if (got == want) return;
    fail(file, line, std::string(what) + ": got \"" + esc(got) + "\", want \"" +
                         esc(want) + "\"");
}

inline std::string list_ints(const std::vector<int>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ", ";
        o += std::to_string(v[i]);
    }
    return o + "]";
}

inline void check_ints(const std::vector<int>& got, std::initializer_list<int> want,
                       const char* what, const char* file, int line) {
    const std::vector<int> w(want);
    if (got == w) return;
    fail(file, line, std::string(what) + ": got " + list_ints(got) + ", want " +
                         list_ints(w));
}

inline std::string list_bytes(const std::vector<uint8_t>& v) {
    std::string o = "[";
    char b[8];
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ", ";
        std::snprintf(b, sizeof b, "%02x", v[i]);
        o += b;
    }
    return o + "]";
}

inline void check_bytes(const std::vector<uint8_t>& got,
                        std::initializer_list<uint8_t> want, const char* what,
                        const char* file, int line) {
    const std::vector<uint8_t> w(want);
    if (got == w) return;
    fail(file, line, std::string(what) + ": got " + list_bytes(got) + ", want " +
                         list_bytes(w));
}

}  // namespace checks

#define CHECK(cond) ::checks::check_true((cond), #cond, __FILE__, __LINE__)
#define CHECK_LL(got, want) \
    ::checks::check_ll((long long)(got), (long long)(want), #got, __FILE__, __LINE__)
#define CHECK_STR(got, want) ::checks::check_str((got), (want), #got, __FILE__, __LINE__)
#define CHECK_INTS(got, ...) \
    ::checks::check_ints((got), {__VA_ARGS__}, #got, __FILE__, __LINE__)
#define CHECK_BYTES(got, ...) \
    ::checks::check_bytes((got), {__VA_ARGS__}, #got, __FILE__, __LINE__)
