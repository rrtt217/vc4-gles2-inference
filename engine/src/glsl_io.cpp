// The two runtime GLSL-text switches; glsl_io.h says what they are for.
//
// `GLSL_DUMP` is the seed and `GLSL_OVERRIDE` is the edit, so the workflow is
// one run with the directory, then one path a program: no rebuild in between, and
// no file that silently does nothing (the exit report names an unused key, and a
// malformed value or an unreadable file stops the run).
#include "glsl_io.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include "switches.h"

using std::string;

namespace {

// One `<key>=<file>` entry.  `used` is what lets the exit report name a key that
// never matched: an override that silently does nothing is the TOKENIZER= trap
// this registry keeps paying for.
struct Override {
    string key;
    string path;
    string text;
    bool used = false;
};

string trim(const string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

bool read_file(const string& path, string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    const bool bad = ferror(f) != 0;
    fclose(f);
    return !bad;
}

struct State {
    const char* dump_dir = nullptr;
    std::vector<Override> over;
    // key -> the texts already written.  A model swap rebuilds a key with
    // another text; writing that as <key>.2.glsl keeps both snapshots, where
    // overwriting <key>.glsl would leave an override pointing at the wrong one.
    std::map<string, std::vector<size_t>> dumped;
    long wrote = 0;

    State() {
        dump_dir = sw_str("GLSL_DUMP");
        const char* v = sw_str("GLSL_OVERRIDE");
        if (!v) return;
        const string s = v;
        size_t p = 0;
        while (p <= s.size()) {
            const size_t q = s.find(',', p);
            const string e = trim(s.substr(p, q == string::npos ? string::npos : q - p));
            const size_t eq = e.find('=');
            if (eq == string::npos || eq == 0 || eq + 1 == e.size()) {
                fprintf(stderr,
                        "GLSL_OVERRIDE: expected <key>=<file>[,<key>=<file>...], got '%s'\n",
                        e.c_str());
                exit(2);
            }
            Override o;
            o.key = e.substr(0, eq);
            o.path = e.substr(eq + 1);
            if (!read_file(o.path, o.text)) {
                fprintf(stderr, "GLSL_OVERRIDE: cannot read %s for %s: %s\n",
                        o.path.c_str(), o.key.c_str(), strerror(errno));
                exit(2);
            }
            if (o.text.empty()) {
                fprintf(stderr, "GLSL_OVERRIDE: %s is empty (for %s)\n",
                        o.path.c_str(), o.key.c_str());
                exit(2);
            }
            for (const Override& prev : over)
                if (prev.key == o.key) {
                    fprintf(stderr, "GLSL_OVERRIDE: %s is named twice\n", o.key.c_str());
                    exit(2);
                }
            over.push_back(std::move(o));
            p = q == string::npos ? s.size() + 1 : q + 1;
        }
    }

    ~State() {
        for (const Override& o : over)
            if (!o.used)
                fprintf(stderr, "GLSL_OVERRIDE: %s never built; %s was not compiled\n",
                        o.key.c_str(), o.path.c_str());
        if (dump_dir && wrote)
            fprintf(stderr, "GLSL_DUMP: %ld programs to %s\n", wrote, dump_dir);
    }
};

State& state() {
    static State s;
    return s;
}

bool mkdir_p(const string& dir) {
    if (dir.empty()) return false;
    string cur;
    for (size_t i = 0; i <= dir.size(); ++i) {
        if (i != dir.size() && dir[i] != '/') continue;
        if (i == 0) {
            cur = "/";
            continue;
        }
        cur = dir.substr(0, i);
        if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) return false;
    }
    return true;
}

// A key becomes a file name under the caller's directory: a key carrying a path
// is refused rather than written outside it.  The keys are builder names today
// (`xq8ps_1216_768_x16_2`), and the check is what keeps it true.
bool good_key(const string& key) {
    return !key.empty() && key.find('/') == string::npos &&
           key.find('\\') == string::npos && key.find("..") == string::npos;
}

Override* find_override(const string& key) {
    for (Override& o : state().over)
        if (o.key == key) return &o;
    return nullptr;
}

}  // namespace

string glsl_text(const string& key, const string& src) {
    State& st = state();
    string text = src;
    if (Override* o = find_override(key)) {
        text = o->text;
        if (!o->used) {
            o->used = true;
            fprintf(stderr, "GLSL_OVERRIDE: %s <- %s (%zu B)\n",
                    key.c_str(), o->path.c_str(), text.size());
        }
    }
    if (!st.dump_dir) return text;
    if (!good_key(key)) {
        fprintf(stderr, "GLSL_DUMP: refusing key '%s' as a file name\n", key.c_str());
        exit(2);
    }
    const size_t h = std::hash<string>{}(text);
    std::vector<size_t>& hs = st.dumped[key];
    if (std::find(hs.begin(), hs.end(), h) != hs.end()) return text;
    hs.push_back(h);
    const string path = string(st.dump_dir) + "/" + key +
                        (hs.size() > 1 ? "." + std::to_string(hs.size()) : string()) +
                        ".glsl";
    if (!mkdir_p(st.dump_dir)) {
        fprintf(stderr, "GLSL_DUMP: cannot create %s: %s\n", st.dump_dir, strerror(errno));
        exit(2);
    }
    FILE* f = fopen(path.c_str(), "wb");
    const bool ok = f && fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) fclose(f);
    if (!ok) {
        fprintf(stderr, "GLSL_DUMP: cannot write %s: %s\n", path.c_str(), strerror(errno));
        exit(2);
    }
    ++st.wrote;
    fprintf(stderr, "GLSL_DUMP: %s (%zu B)\n", path.c_str(), text.size());
    return text;
}
