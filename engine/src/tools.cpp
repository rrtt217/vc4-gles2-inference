#include "tools.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <sys/statvfs.h>

string json_escape(const string& s) {
    string o;
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8]; snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c); o += b;
                } else o += c;
                break;
        }
    }
    return o;
}

void json_skip_ws(const string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
}

bool json_key_pos(const string& obj, const string& key, size_t& p) {
    string pat = "\"" + key + "\"";
    size_t k = obj.find(pat);
    if (k == string::npos) return false;
    p = k + pat.size();
    json_skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != ':') return false;
    ++p;
    json_skip_ws(obj, p);
    return true;
}

bool json_get_string(const string& obj, const string& key, string& out) {
    size_t p = 0;
    if (!json_key_pos(obj, key, p)) return false;
    if (p >= obj.size() || obj[p] != '"') return false;
    ++p;
    out.clear();
    while (p < obj.size()) {
        char c = obj[p++];
        if (c == '"') return true;
        if (c != '\\') { out += c; continue; }
        if (p >= obj.size()) return false;
        char e = obj[p++];
        switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                if (p + 4 > obj.size()) return false;
                unsigned v = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = obj[p++]; v <<= 4;
                    if (h >= '0' && h <= '9') v |= h - '0';
                    else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
                    else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
                    else return false;
                }
                if (v < 0x80) out += (char)v;
                else if (v < 0x800) {
                    out += (char)(0xC0 | (v >> 6));
                    out += (char)(0x80 | (v & 0x3F));
                } else {
                    out += (char)(0xE0 | (v >> 12));
                    out += (char)(0x80 | ((v >> 6) & 0x3F));
                    out += (char)(0x80 | (v & 0x3F));
                }
                break;
            }
            default: return false;
        }
    }
    return false;
}

bool json_get_object(const string& obj, const string& key, string& out) {
    size_t p = 0;
    if (!json_key_pos(obj, key, p) || p >= obj.size() || obj[p] != '{') return false;
    size_t start = p;
    int depth = 0; bool in_str = false, esc = false;
    for (; p < obj.size(); ++p) {
        char c = obj[p];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) { out = obj.substr(start, p - start + 1); return true; }
        }
    }
    return false;
}

namespace {

string trim_copy(const string& s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Split on commas that are not inside quotes or brackets.
vector<string> split_top(const string& s) {
    vector<string> out;
    string cur;
    int depth = 0;
    char q = 0;
    for (char c : s) {
        if (q) { cur += c; if (c == q) q = 0; continue; }
        if (c == '"' || c == '\'') { q = c; cur += c; continue; }
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if (c == ')' || c == ']' || c == '}') --depth;
        if (c == ',' && depth == 0) { out.push_back(trim_copy(cur)); cur.clear(); continue; }
        cur += c;
    }
    if (!trim_copy(cur).empty()) out.push_back(trim_copy(cur));
    return out;
}

// One Python-ish literal -> JSON text.  The LFM2.5 tool calls are Pythonic
// (`f(city="Paris", n=3)`), so a quoted string, a number, a bool/None or one
// level of list/dict is all that shows up in practice; anything else becomes a
// JSON string rather than an invalid document.
string py_json(const string& in) {
    string v = trim_copy(in);
    if (v.empty()) return "null";
    const char c = v[0];
    if (c == '"' || c == '\'') {
        string body;
        for (size_t i = 1; i < v.size(); ++i) {
            char ch = v[i];
            if (ch == '\\' && i + 1 < v.size()) {
                char nx = v[++i];
                if (nx == 'n') body += '\n';
                else if (nx == 't') body += '\t';
                else if (nx == 'r') body += '\r';
                else body += nx;
                continue;
            }
            if (ch == c) break;
            body += ch;
        }
        return "\"" + json_escape(body) + "\"";
    }
    if (v == "True" || v == "true") return "true";
    if (v == "False" || v == "false") return "false";
    if (v == "None" || v == "null") return "null";
    if (c == '[' || c == '(') {
        string inner = v.substr(1, v.size() > 2 ? v.size() - 2 : 0);
        string out = "[";
        bool first = true;
        for (auto& item : split_top(inner)) {
            if (!first) out += ",";
            first = false;
            out += py_json(item);
        }
        return out + "]";
    }
    if (c == '{') {
        string inner = v.substr(1, v.size() > 2 ? v.size() - 2 : 0);
        string out = "{";
        bool first = true;
        for (auto& kv : split_top(inner)) {
            size_t colon = kv.find(':');
            if (colon == string::npos) continue;
            if (!first) out += ",";
            first = false;
            out += py_json(kv.substr(0, colon)) + ":" + py_json(kv.substr(colon + 1));
        }
        return out + "}";
    }
    // A bare number stays a number; anything else is quoted so the JSON
    // document is still valid.
    bool numeric = !v.empty();
    for (char ch : v)
        if (!(isdigit((unsigned char)ch) || ch == '-' || ch == '+' || ch == '.' ||
              ch == 'e' || ch == 'E'))
            numeric = false;
    return numeric ? v : "\"" + json_escape(v) + "\"";
}

// <|tool_call_start|>[f(a="x"), g(b=1)]<|tool_call_end|>
vector<ToolCall> parse_lfm_calls(const string& text) {
    vector<ToolCall> out;
    const string open = "<|tool_call_start|>", close = "<|tool_call_end|>";
    size_t pos = 0;
    while ((pos = text.find(open, pos)) != string::npos) {
        size_t a = pos + open.size();
        size_t b = text.find(close, a);
        if (b == string::npos) break;
        string body = trim_copy(text.substr(a, b - a));
        if (!body.empty() && body.front() == '[' && body.back() == ']')
            body = body.substr(1, body.size() - 2);
        for (auto& call : split_top(body)) {
            size_t p = call.find('(');
            if (p == string::npos) continue;
            string name = trim_copy(call.substr(0, p));
            size_t q = call.rfind(')');
            string argv = (q == string::npos || q <= p) ? string() : call.substr(p + 1, q - p - 1);
            string js = "{";
            bool first = true;
            for (auto& kv : split_top(argv)) {
                size_t eq = kv.find('=');
                if (eq == string::npos) continue;
                if (!first) js += ",";
                first = false;
                js += "\"" + json_escape(trim_copy(kv.substr(0, eq))) + "\":" +
                      py_json(kv.substr(eq + 1));
            }
            js += "}";
            if (!name.empty()) out.push_back(ToolCall{name, js});
        }
        pos = b + close.size();
    }
    return out;
}

}  // namespace

vector<ToolCall> parse_tool_calls(const string& text) {
    vector<ToolCall> lfm = parse_lfm_calls(text);
    if (!lfm.empty()) return lfm;
    vector<ToolCall> out;
    size_t pos = 0;
    const string open = "<tool_call>", close = "</tool_call>";
    while ((pos = text.find(open, pos)) != string::npos) {
        size_t a = pos + open.size();
        size_t b = text.find(close, a);
        if (b == string::npos) break;
        string body = text.substr(a, b - a);
        ToolCall c;
        if (json_get_string(body, "name", c.name)) {
            if (!json_get_object(body, "arguments", c.args)) c.args = "{}";
            out.push_back(std::move(c));
        }
        pos = b + close.size();
    }
    return out;
}

string base64_decode(const string& in) {
    static int tbl[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) tbl[i] = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tbl[(unsigned char)a[i]] = i;
        init = true;
    }
    string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (tbl[c] < 0) continue;
        val = (val << 6) + tbl[c];
        bits += 6;
        if (bits >= 0) {
            out += (char)((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return out;
}

size_t utf8_complete_prefix(const string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int need = 0;
        if (c < 0x80) need = 0;
        else if ((c & 0xE0) == 0xC0) need = 1;
        else if ((c & 0xF0) == 0xE0) need = 2;
        else if ((c & 0xF8) == 0xF0) need = 3;
        else { ++i; continue; }
        if (i + (size_t)need >= s.size()) break;
        bool ok = true;
        for (int j = 1; j <= need; ++j)
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { ++i; continue; }
        i += (size_t)need + 1;
    }
    return i;
}

string utf8_take(string& pending) {
    size_t n = utf8_complete_prefix(pending);
    if (n == 0) return {};
    string out = pending.substr(0, n);
    pending.erase(0, n);
    return out;
}

struct ExprParser {
    const char* p;
    bool ok = true;
    explicit ExprParser(const string& s) : p(s.c_str()) {}
    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }
    double expr() {
        double v = term();
        for (;;) {
            ws();
            if (*p == '+') { ++p; v += term(); }
            else if (*p == '-') { ++p; v -= term(); }
            else return v;
        }
    }
    double term() {
        double v = power();
        for (;;) {
            ws();
            if (*p == '*') {
                if (p[1] == '*') { ++p; ++p; v = pow(v, power()); }
                else { ++p; v *= power(); }
            } else if (*p == '/') { ++p; v /= power(); }
            else if (*p == '%') { ++p; v = fmod(v, power()); }
            else return v;
        }
    }
    double power() {
        double v = factor();
        ws();
        if (*p == '^') { ++p; v = pow(v, power()); }
        return v;
    }
    double factor() {
        ws();
        if (*p == '+') { ++p; return factor(); }
        if (*p == '-') { ++p; return -factor(); }
        if (*p == '(') {
            ++p; double v = expr(); ws();
            if (*p != ')') { ok = false; return 0; }
            ++p; return v;
        }
        char* end = nullptr;
        double v = strtod(p, &end);
        if (end == p) { ok = false; return 0; }
        p = end;
        return v;
    }
};

string run_tool(const string& name, const string& args) {
    if (name == "get_current_time") {
        string tz;
        json_get_string(args, "timezone", tz);
        string old_tz;
        const char* old = getenv("TZ");
        if (old) old_tz = old;
        if (!tz.empty()) { setenv("TZ", tz.c_str(), 1); tzset(); }
        time_t now = time(nullptr);
        struct tm tmv {};
        localtime_r(&now, &tmv);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        if (!tz.empty()) {
            if (old) setenv("TZ", old_tz.c_str(), 1);
            else unsetenv("TZ");
            tzset();
        }
        return "{\"datetime\":\"" + json_escape(buf) + "\",\"timezone\":\"" +
               json_escape(tz.empty() ? string("local") : tz) + "\"}";
    }
    if (name == "calculate_math") {
        string expr;
        if (!json_get_string(args, "expression", expr)) return "{\"error\":\"missing expression\"}";
        ExprParser ep(expr);
        double v = ep.expr();
        ep.ws();
        if (!ep.ok || *ep.p != '\0') return "{\"error\":\"invalid expression\"}";
        char buf[64];
        if (fabs(v - round(v)) < 1e-9) snprintf(buf, sizeof(buf), "%.0f", v);
        else snprintf(buf, sizeof(buf), "%.6g", v);
        return "{\"result\":\"" + string(buf) + "\"}";
    }
    if (name == "system_status" || name.find("status") != string::npos) {
        double mem_free = 0, mem_avail = 0;
        {
            ifstream in("/proc/meminfo");
            string key;
            while (in >> key) {
                long value = 0;
                string unit;
                if (!(in >> value)) break;
                getline(in, unit);
                if (key == "MemFree:") mem_free = value / 1024.0;
                else if (key == "MemAvailable:") mem_avail = value / 1024.0;
            }
        }
        double temp_c = 0;
        {
            ifstream in("/sys/class/thermal/thermal_zone0/temp");
            long milli = 0;
            if (in >> milli) temp_c = milli / 1000.0;
        }
        double uptime_s = 0, load1 = 0;
        {
            ifstream in("/proc/uptime");
            in >> uptime_s;
        }
        {
            ifstream in("/proc/loadavg");
            in >> load1;
        }
        double disk_free_mb = 0;
        struct statvfs st {};
        if (statvfs("/", &st) == 0) disk_free_mb = (double)st.f_bavail * st.f_frsize / 1048576.0;
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"temp_c\":%.1f,\"mem_free_mb\":%.1f,\"mem_avail_mb\":%.1f,"
                 "\"disk_free_mb\":%.1f,\"uptime_s\":%.0f,\"load1\":%.2f}",
                 temp_c, mem_free, mem_avail, disk_free_mb, uptime_s, load1);
        return string(buf);
    }
    return "{\"error\":\"unknown tool\"}";
}
