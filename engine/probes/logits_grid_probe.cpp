// Does the logits grid stay in one place?
//
// The vocabulary matrix is stored in container tiles of kLogitsGrid rows, and
// three separate languages have to agree on that number:
//
//   tools/mmcf_grid.py     LOGITS_GRID: how many T_LOGITS rows per tile
//   src/gpu.cpp            the logits target width, the read-back stride, and
//                          token = tile * kLogitsGrid + idx
//   shaders_attn.cpp       the L texture divisor in the four Top-K kernels
//
// None of them can see the others: a change in one place is a wrong token id,
// not a compile error, a link error or a GL error.  Measured before the
// constant existed (2026-11-08): with only the shader divisor changed, greedy
// output and PPL were bit-identical - the gate never runs the Top-K path - and
// TOPK=8 sampling changed from "我是由Jingyaogong创建的..." to "the manuscript
// is the speaker...".
//
// This probe builds the kernels, reads the packer, and fails if any of the
// three disagrees with src/logits_grid.h.
//
// Build (no GL, no model):
//   make -C engine probes/logits_grid_probe
//   ./probes/logits_grid_probe [--grid N]
//
// --grid overrides what the *host* is assumed to use, which is how the probe
// demonstrates the hazard: with --grid 1024 the builders still bake the header
// value and the run fails.
#include "shaders.h"
#include "logits_grid.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace {

string read_file(const char* path, bool* ok) {
    FILE* f = fopen(path, "rb");
    if (!f) { *ok = false; return string(); }
    string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    *ok = true;
    return s;
}

// Every "/<number>.0" that normalises a coordinate for the L sampler.  The four
// kernels differ in what they index (a logit, a pair of logits), not in how
// they scale that index, so the divisor is the grid.
void divisors(const string& src, vector<int>& out) {
    const string pat = "texture2D(L,vec2((float(";
    size_t p = 0;
    while ((p = src.find(pat, p)) != string::npos) {
        const size_t q = src.find(")/", p);
        if (q == string::npos) break;
        int n = 0;
        if (sscanf(src.c_str() + q + 2, "%d.0", &n) != 1) {
            fprintf(stderr, "probe: cannot parse the L divisor\n");
            exit(2);
        }
        out.push_back(n);
        p = q;
    }
}

// Whole-number match: "768" must not fire inside "32768".
bool has_number(const string& s, int n) {
    char buf[32];
    const int len = snprintf(buf, sizeof buf, "%d", n);
    for (size_t p = s.find(buf); p != string::npos; p = s.find(buf, p + 1)) {
        const bool left = p == 0 || !isdigit((unsigned char)s[p - 1]);
        const bool right = p + len >= s.size() || !isdigit((unsigned char)s[p + len]);
        if (left && right) return true;
    }
    return false;
}

// Remaining bare occurrences in a file, with line numbers.  Evidence, not a
// pass condition: the header and the packer's definition are expected to spell
// the number out, everything else should have stopped.
int census(const char* path) {
    bool ok = false;
    const string s = read_file(path, &ok);
    if (!ok) {
        fprintf(stderr, "probe: missing %s (run from engine/)\n", path);
        return -1;
    }
    int line = 0, hits = 0;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t e = s.find('\n', i);
        const string l = s.substr(i, e == string::npos ? string::npos : e - i);
        ++line;
        if (has_number(l, kLogitsGrid) || has_number(l, kLogitsGridPacked)) {
            ++hits;
            printf("    %-24s %4d  %s\n", path, line, l.c_str());
        }
        if (e == string::npos) break;
        i = e + 1;
    }
    return hits;
}

// Python cannot include a C++ header, so its constant is checked by text.
// It lives in tools/mmcf_grid.py; mmcf_pack.py and mmcf_hf.py import it.
bool python_grid(int* out) {
    bool ok = false;
    const string s = read_file("tools/mmcf_grid.py", &ok);
    if (!ok) {
        fprintf(stderr, "probe: missing tools/mmcf_grid.py (run from engine/)\n");
        return false;
    }
    const size_t p = s.find("LOGITS_GRID =");
    if (p == string::npos) {
        fprintf(stderr, "probe: tools/mmcf_grid.py has no LOGITS_GRID definition\n");
        return false;
    }
    *out = atoi(s.c_str() + p + strlen("LOGITS_GRID ="));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int host_grid = kLogitsGrid;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--grid") && i + 1 < argc) host_grid = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--grid N]\n", argv[0]);
            return 2;
        }
    }

    printf("header: kLogitsGrid=%d (packed %d), kMatvecStripRows=%d\n",
           kLogitsGrid, kLogitsGridPacked, kMatvecStripRows);
    printf("host assumes: %d\n\n", host_grid);
    bool ok = host_grid == kLogitsGrid;

    printf("kernels:\n");
    struct Case { const char* name; int rows; bool packed; };
    const Case cases[] = {{"top2_fast", kLogitsGrid, false}, {"top2", kLogitsGrid, false},
                          {"top2_fast_packed", kLogitsGrid, true}, {"top2_packed", kLogitsGrid, true},
                          {"top2_fast", 256, false}, {"top2_packed", 256, true}};
    for (const Case& c : cases) {
        const string src = !strcmp(c.name, "top2_fast")   ? top2_fast(c.rows, 256)
                         : !strcmp(c.name, "top2")        ? top2(c.rows, 256)
                         : !strcmp(c.name, "top2_fast_packed") ? top2_fast_packed(c.rows, 256)
                                                               : top2_packed(c.rows, 256);
        vector<int> d;
        divisors(src, d);
        if (d.empty()) {
            printf("  %-16s rows=%-5d FAIL: no L sampler found\n", c.name, c.rows);
            ok = false;
            continue;
        }
        const int want = c.packed ? host_grid / 2 : host_grid;
        for (int got : d) {
            const bool good = got == want;
            printf("  %-16s rows=%-5d divides by %-5d (host wants %-5d) %s\n",
                   c.name, c.rows, got, want, good ? "ok" : "MISMATCH");
            ok = ok && good;
        }
    }

    int py = 0;
    const bool py_ok = python_grid(&py);
    printf("\npacker: tools/mmcf_grid.py LOGITS_GRID=%d %s\n", py,
           !py_ok ? "MISSING" : py == kLogitsGrid ? "ok" : "MISMATCH");
    ok = ok && py_ok && py == kLogitsGrid;

    printf("\nremaining literals (should be none in C++):\n");
    const int a = census("src/gpu.cpp");
    const int b = census("src/shaders_attn.cpp");
    const int c = census("tools/mmcf_grid.py");
    const int d = census("tools/mmcf_hf.py");
    const int e = census("tools/mmcf_pack.py");
    printf("    %d in the host, %d in the builders, %d in the Python loaders\n", a, b, c + d + e);

    printf("\n");
    if (ok) {
        printf("PASS: header, packer and all four kernels agree on %d\n", kLogitsGrid);
        return 0;
    }
    printf("FAIL: the vocabulary grid disagrees somewhere; a container packed at\n"
           "  one stride would decode with shifted tokens and no error at all\n");
    return 1;
}
