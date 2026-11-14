#include "tools.h"

#include "common.h"

// Tool arguments and results are JSON documents.  common_json is the vendored
// wrapper the chat-template engine already carries: a pimpl over the same
// nlohmann build, so no other translation unit pays its compile cost.
#include "jinja/json.h"

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

// A tool argument document arrives from a model, so a parse failure has to
// degrade to "no arguments" instead of throwing into the generation loop.
common_json parse_object_tolerant(const string& text) {
    common_json j = common_json::parse_no_throw(text);
    return j.is_discarded() || !j.is_object() ? common_json::object() : j;
}

string json_arg(const common_json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key)) return string();
    const common_json& v = obj.at(key);
    return v.is_string() ? v.get<string>() : string();
}

// `arguments` is normally an object; a few templates emit it as a JSON string,
// and both shapes have to reach the tool as a document.
string json_arg_document(const common_json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key)) return "{}";
    const common_json& v = obj.at(key);
    if (v.is_string()) return v.get<string>();
    return v.is_object() ? v.dump() : string("{}");
}

// One Python-ish literal -> a JSON value.  The LFM2.5 tool calls are Pythonic
// (`f(city="Paris", n=3)`), so a quoted string, a number, a bool/None or one
// level of list/dict is all that shows up in practice; anything else becomes a
// JSON string rather than an invalid document.
common_json py_json(const string& in) {
    const string v = trim_copy(in);
    if (v.empty()) return common_json();
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
        return common_json(body);
    }
    if (v == "True" || v == "true") return common_json(true);
    if (v == "False" || v == "false") return common_json(false);
    if (v == "None" || v == "null") return common_json();
    if (c == '[' || c == '(') {
        common_json arr = common_json::array();
        for (auto& item : split_top(v.substr(1, v.size() > 2 ? v.size() - 2 : 0)))
            arr.push_back(py_json(item));
        return arr;
    }
    if (c == '{') {
        common_json obj = common_json::object();
        for (auto& kv : split_top(v.substr(1, v.size() > 2 ? v.size() - 2 : 0))) {
            const size_t colon = kv.find(':');
            if (colon == string::npos) continue;
            const common_json key = py_json(kv.substr(0, colon));
            obj[key.is_string() ? key.get<string>() : key.dump()] = py_json(kv.substr(colon + 1));
        }
        return obj;
    }
    // A bare number stays a number; anything else is quoted so the JSON
    // document is still valid.
    // A bare number stays a number; anything else is quoted so the document is
    // still valid.  Pasting the text through, as this used to, turned `+1` into
    // invalid JSON; now the contract is "the result parses".
    const common_json num = common_json::parse_no_throw(v);
    if (!num.is_discarded() && num.is_number()) return num;
    return common_json(v);
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
            common_json args = common_json::object();
            for (auto& kv : split_top(argv)) {
                const size_t eq = kv.find('=');
                if (eq == string::npos) continue;
                args[trim_copy(kv.substr(0, eq))] = py_json(kv.substr(eq + 1));
            }
            if (!name.empty()) out.push_back(ToolCall{name, args.dump()});
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
        const common_json call = parse_object_tolerant(body);
        const string name = json_arg(call, "name");
        if (!name.empty()) {
            ToolCall c;
            c.name = name;
            c.args = json_arg_document(call, "arguments");
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
    const common_json a = parse_object_tolerant(args);
    if (name == "get_current_time") {
        const string tz = json_arg(a, "timezone");
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
        return common_json::object({{"datetime", string(buf)},
                                    {"timezone", tz.empty() ? string("local") : tz}}).dump();
    }
    if (name == "calculate_math") {
        const bool have_expr =
            a.is_object() && a.contains("expression") && a.at("expression").is_string();
        if (!have_expr)
            return common_json::object({{"error", string("missing expression")}}).dump();
        const string expr = a.at("expression").get<string>();
        ExprParser ep(expr);
        double v = ep.expr();
        ep.ws();
        if (!ep.ok || *ep.p != '\0')
            return common_json::object({{"error", string("invalid expression")}}).dump();
        char buf[64];
        if (fabs(v - round(v)) < 1e-9) snprintf(buf, sizeof(buf), "%.0f", v);
        else snprintf(buf, sizeof(buf), "%.6g", v);
        return common_json::object({{"result", string(buf)}}).dump();
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
        // Same precision the printf had, so the numbers a model sees do not
        // change; only the trailing zeros are the library's business now.
        const auto r1 = [](double v) { return std::round(v * 10.0) / 10.0; };
        const auto r2 = [](double v) { return std::round(v * 100.0) / 100.0; };
        return common_json::object({
            {"temp_c", r1(temp_c)},
            {"mem_free_mb", r1(mem_free)},
            {"mem_avail_mb", r1(mem_avail)},
            {"disk_free_mb", r1(disk_free_mb)},
            {"uptime_s", (long long) llround(uptime_s)},
            {"load1", r2(load1)},
        }).dump();
    }
    return common_json::object({{"error", string("unknown tool")}}).dump();
}
