// The JSON constraint's contract, from a synthetic vocabulary: no model, no GPU.
//
// Ids 0..255 are the single raw bytes (so a byte can be fed by its own value),
// then a handful of multi-byte tokens that span construct boundaries, an EOS, a
// chat marker, an empty entry and a <0xNN> fallback.  The tests pin:
//
//   1. valid JSON is a legal byte prefix and reaches done(); invalid JSON has a
//      first illegal byte; an incomplete prefix is legal but not done();
//   2. numbers, strings (escapes / \u / control chars), nesting, trailing commas;
//   3. a multi-byte token may cross a construct boundary ({"  ":  ,"  true  \u0041);
//   4. EOS is legal only in an accepting state; markers / empty tokens never are;
//   5. mask() writes -1e30f on exactly the forbidden ids.
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "checks.h"
#include "grammar.h"
#include "tokenizer.h"

namespace {

struct Entry { std::string text; float score; };

constexpr int EOS = 256, MARK = 257, EMPTY = 258, FALLBACK = 259;
constexpr int T_OBJKEY = 260, T_COLON = 261, T_COMMAKEY = 262;   // {"  ":  ,"
constexpr int T_TRUE = 263, T_NULL = 264, T_NUM = 265, T_KEYA = 266;
constexpr int T_ARR_OPEN = 267, T_CLOSE = 268, T_FRAC = 269, T_NEG = 270;
constexpr int T_ESC = 271, T_UNICODE = 272;                      // \n  \u0041
constexpr int VOCAB = 273;

std::vector<Entry> build_vocab() {
    std::vector<Entry> v;
    for (int i = 0; i < 256; ++i) v.push_back({std::string(1, (char)i), 0});
    v.push_back({"<|eos|>", 0});      // EOS
    v.push_back({"<|im_end|>", 0});   // MARK
    v.push_back({"", 0});             // EMPTY
    v.push_back({"<0x7B>", 0});       // FALLBACK -> '{'
    v.push_back({"{\"", 0});          // T_OBJKEY
    v.push_back({"\":", 0});          // T_COLON
    v.push_back({",\"", 0});          // T_COMMAKEY
    v.push_back({"true", 0});         // T_TRUE
    v.push_back({"null", 0});         // T_NULL
    v.push_back({"123", 0});          // T_NUM
    v.push_back({"\"a\"", 0});        // T_KEYA
    v.push_back({"[{", 0});           // T_ARR_OPEN
    v.push_back({"}]", 0});           // T_CLOSE
    v.push_back({"0.5", 0});          // T_FRAC
    v.push_back({"-1", 0});           // T_NEG
    v.push_back({"\\n", 0});          // T_ESC
    v.push_back({"\\u0041", 0});      // T_UNICODE
    return v;
}

std::vector<uint8_t> make_blob(const std::vector<Entry>& vocab) {
    std::vector<uint8_t> b;
    auto put_u32 = [&](uint32_t u) { uint8_t x[4]; std::memcpy(x, &u, 4); b.insert(b.end(), x, x + 4); };
    auto put_f32 = [&](float f) { uint32_t u = 0; std::memcpy(&u, &f, 4); put_u32(u); };
    int maxlen = 0;
    for (const Entry& e : vocab) if ((int)e.text.size() > maxlen) maxlen = (int)e.text.size();
    put_u32((uint32_t)maxlen);
    for (const Entry& e : vocab) {
        put_f32(e.score);
        put_u32((uint32_t)e.text.size());
        b.insert(b.end(), e.text.begin(), e.text.end());
    }
    return b;
}

std::vector<float> masked(const JsonConstraint& jc) {
    std::vector<float> lg(VOCAB, 0.0f);
    jc.mask(lg);
    return lg;
}
bool allowed(const std::vector<float>& lg, int id) { return lg[(size_t)id] > -1e29f; }

// Feed a string one byte at a time; false at the first illegal byte.
bool feed_all(JsonConstraint& jc, const std::string& s) {
    for (unsigned char c : s) {
        if (!jc.allows(c)) return false;
        jc.accept(c);
    }
    return true;
}

// --- random *valid* JSON, valid by construction ---------------------------
// Every string this emits is legal JSON, so the automaton must accept all of it
// and reach done(); and every prefix must be a legal prefix.  A rejection here
// is an over-restriction bug the fixed cases above would not reach.
struct Gen {
    std::mt19937 rng;
    explicit Gen(uint32_t seed) : rng(seed) {}
    int pick(int n) { return (int)(rng() % (uint32_t)n); }
    std::string ws() {
        if (pick(3)) return "";
        const char* w = " \t\n\r";
        return std::string(1, w[pick(4)]);
    }
    std::string str() {
        std::string o = "\"";
        int n = pick(6);
        for (int i = 0; i < n; ++i) {
            int k = pick(10);
            if (k == 0) o += "\\\"";
            else if (k == 1) o += "\\\\";
            else if (k == 2) o += "\\n";
            else if (k == 3) o += "\\u00e9";
            else if (k == 4) o += "\xc3\xa9";        // raw UTF-8 'é'
            else o += (char)('a' + pick(26));
        }
        return o + "\"";
    }
    std::string num() {
        std::string o;
        if (pick(2)) o += "-";
        if (pick(2)) o += "0";
        else { o += (char)('1' + pick(9)); int n = pick(3); while (n--) o += (char)('0' + pick(10)); }
        if (pick(2)) { o += "."; int n = 1 + pick(3); while (n--) o += (char)('0' + pick(10)); }
        if (pick(2)) { o += pick(2) ? "e" : "E"; if (pick(2)) o += pick(2) ? "+" : "-";
                       int n = 1 + pick(2); while (n--) o += (char)('0' + pick(10)); }
        return o;
    }
    std::string value(int depth) {
        int t = depth <= 0 ? 2 + pick(5) : pick(7);
        switch (t) {
        case 0: {                                   // object
            std::string o = "{" + ws();
            int n = pick(4);
            for (int i = 0; i < n; ++i) {
                if (i) o += "," + ws();
                o += str() + ws() + ":" + ws() + value(depth - 1);
            }
            return o + ws() + "}";
        }
        case 1: {                                   // array
            std::string o = "[" + ws();
            int n = pick(4);
            for (int i = 0; i < n; ++i) {
                if (i) o += "," + ws();
                o += value(depth - 1);
            }
            return o + ws() + "]";
        }
        case 2: return str();
        case 3: return num();
        case 4: return "true";
        case 5: return "false";
        default: return "null";
        }
    }
};

}  // namespace

int main() {
    Tokenizer tok;
    auto blob = make_blob(build_vocab());
    CHECK(tok.load_memory(blob.data(), blob.size(), VOCAB));
    JsonConstraint jc;
    jc.init(tok, VOCAB, {EOS});

    // --- valid, complete JSON ---------------------------------------------
    const char* valid[] = {
        "{}", "[]", "{\"a\":1}", "{\"a\":1,\"b\":2}", "[1,2,3]",
        "[true,false,null]", "{\"a\":\"b\"}", "\"hello\"", "123", "-123", "0",
        "0.5", "-0.5", "1e3", "1E+3", "1e-3", "1.5e2", "true", "false", "null",
        "{\"a\":[1,{\"b\":null}]}", "[[],{},[[]]]", "{\"a\":\"\\u0041\"}",
        "{\"a\":\"line\\nbreak\"}", " { \"a\" : 1 } ", "[\n1,\t2\n]",
        "{\"a\":-0.0e-0}", "[\"\\\"\\\\\"]", "1.2e3", "123.456e+78", "0.0e0",
    };
    for (const char* s : valid) {
        jc.reset();
        CHECK(feed_all(jc, s));
        CHECK(jc.done());
        CHECK(allowed(masked(jc), EOS));
    }

    // --- legal but incomplete prefixes (not done, EOS refused) ------------
    const char* prefix[] = {
        "{", "{\"a\":", "{\"a\":1", "[1,", "\"abc", "tru", "nul", "{\"a\":\"b\"",
        "1.", "1e", "1e+", "\"abc\\", "\"\\u00", "[1,2", "\"unterminated",
    };
    for (const char* s : prefix) {
        jc.reset();
        CHECK(feed_all(jc, s));
        CHECK(!jc.done());
        CHECK(!allowed(masked(jc), EOS));
    }

    // --- invalid JSON: feed_all must hit an illegal byte ------------------
    const char* invalid[] = {
        "{,}", "{\"a\"}", "{\"a\":}", "{\"a\":1,}", "[1,]", "[1 2]", "01",
        "1.a", "1e+a", "\"abc\\q\"", "\"\\u00G1\"", "truX", "nulll", "}{",
        "1 2", "{\"a\":1 \"b\":2}", "[1,,2]",
        // a number's fraction and exponent each appear at most once: a second
        // '.' or 'e' ends the number and the stray byte is rejected (the bug the
        // MoE / RWKV cross-model run exposed: "1.2.3.4..." was being accepted).
        "1.2.3", "1e2e3", "0.0.0", "1.2e3.4", "1..2", "1e2e3e4", "12.34.56", "1e2.3",
    };
    for (const char* s : invalid) {
        jc.reset();
        CHECK(!feed_all(jc, s));
    }
    // a raw control byte inside a string is illegal
    jc.reset();
    CHECK(feed_all(jc, "\""));
    CHECK(!allowed(masked(jc), 0x01));

    // --- multi-byte tokens crossing construct boundaries ------------------
    jc.reset();                                   // {"a":1,"b":2}
    for (int id : {T_OBJKEY, (int)'a', T_COLON, (int)'1', T_COMMAKEY, (int)'b', T_COLON, (int)'2', (int)'}'}) {
        CHECK(allowed(masked(jc), id));
        jc.accept(id);
    }
    CHECK(jc.done());
    jc.reset();                                   // ["a"]
    for (int id : {(int)'[', T_KEYA, (int)']'}) { CHECK(allowed(masked(jc), id)); jc.accept(id); }
    CHECK(jc.done());
    jc.reset(); CHECK(allowed(masked(jc), T_TRUE)); jc.accept(T_TRUE); CHECK(jc.done());
    jc.reset(); CHECK(allowed(masked(jc), T_NULL)); jc.accept(T_NULL); CHECK(jc.done());
    jc.reset(); CHECK(allowed(masked(jc), T_NUM));  jc.accept(T_NUM);  CHECK(jc.done());
    jc.reset(); CHECK(allowed(masked(jc), T_FRAC)); jc.accept(T_FRAC); CHECK(jc.done());
    jc.reset(); CHECK(allowed(masked(jc), T_NEG));  jc.accept(T_NEG);  CHECK(jc.done());
    jc.reset();                                   // [{}]
    for (int id : {T_ARR_OPEN, (int)'}', (int)']'}) { CHECK(allowed(masked(jc), id)); jc.accept(id); }
    CHECK(jc.done());
    jc.reset();                                   // [{}] via the }] token
    for (int id : {(int)'[', (int)'{', T_CLOSE}) { CHECK(allowed(masked(jc), id)); jc.accept(id); }
    CHECK(jc.done());
    jc.reset();                                   // "a\nb"
    for (int id : {(int)'"', (int)'a', T_ESC, (int)'b', (int)'"'}) { CHECK(allowed(masked(jc), id)); jc.accept(id); }
    CHECK(jc.done());
    jc.reset();                                   // "\u0041"
    for (int id : {(int)'"', T_UNICODE, (int)'"'}) { CHECK(allowed(masked(jc), id)); jc.accept(id); }
    CHECK(jc.done());

    // --- EOS / marker / empty / fallback ----------------------------------
    jc.reset();
    CHECK(!allowed(masked(jc), EOS));      // nothing parsed yet
    CHECK(!allowed(masked(jc), MARK));     // a chat marker is never JSON
    CHECK(!allowed(masked(jc), EMPTY));    // an empty token cannot advance
    CHECK(allowed(masked(jc), FALLBACK));  // <0x7B> decodes to '{'
    jc.accept(FALLBACK);
    CHECK(allowed(masked(jc), (int)'}'));  // ...so it opened an object
    jc.accept('}');
    CHECK(jc.done());
    CHECK(allowed(masked(jc), EOS));
    jc.accept(' ');                        // trailing whitespace keeps done
    CHECK(jc.done());
    CHECK(allowed(masked(jc), EOS));

    // --- mask writes -1e30f on exactly the forbidden ids ------------------
    jc.reset();
    auto lg = masked(jc);
    CHECK(lg[(size_t)'}'] == -1e30f);      // '}' cannot start a value
    CHECK(lg[(size_t)'{'] == 0.0f);        // '{' can
    CHECK(lg[EOS] == -1e30f);
    CHECK(lg[MARK] == -1e30f);
    CHECK(lg[EMPTY] == -1e30f);
    CHECK(jc.allowed_count() > 0);

    // --- stress: random valid JSON must be accepted, prefixes legal --------
    Gen gen(20261122);
    for (int iter = 0; iter < 400; ++iter) {
        std::string s = gen.value(3);
        jc.reset();
        if (!feed_all(jc, s)) {
            checks::fail(__FILE__, __LINE__, "random valid JSON rejected: " + checks::esc(s));
            continue;
        }
        CHECK(jc.done());
        // every prefix is a legal prefix (the language is prefix-closed)
        for (size_t k = 0; k < s.size(); ++k) {
            jc.reset();
            if (!feed_all(jc, s.substr(0, k))) {
                checks::fail(__FILE__, __LINE__,
                             "prefix rejected at " + std::to_string(k) + ": " + checks::esc(s));
                break;
            }
        }
    }

    if (checks::g_fail) { std::fprintf(stderr, "%d check(s) failed\n", checks::g_fail); return 1; }
    std::printf("test_grammar: all checks passed\n");
    return 0;
}
