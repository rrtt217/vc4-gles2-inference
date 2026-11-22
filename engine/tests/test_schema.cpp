// The generic grammar matcher (gbnf) and the JSON-Schema compiler (schema).
//
// Part 1 builds the plain-JSON grammar and runs the same valid/invalid/prefix
// corpus as test_grammar, so the deterministic matcher is checked against the
// hand-written automaton's behaviour.  Part 2 compiles schemas and feeds valid
// and invalid instances, plus random instances generated from a schema.
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "checks.h"
#include "gbnf.h"
#include "schema.h"
#include "tokenizer.h"

namespace {
constexpr int EOS = 256, MARK = 257, EMPTY = 258, VOCAB = 259;

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t u) { uint8_t x[4]; std::memcpy(x, &u, 4); b.insert(b.end(), x, x + 4); };
    auto f32 = [&](float f) { uint32_t u = 0; std::memcpy(&u, &f, 4); u32(u); };
    auto rec = [&](const std::string& s) { f32(0); u32((uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end()); };
    u32(10);
    for (int i = 0; i < 256; ++i) rec(std::string(1, (char)i));
    rec("<|eos|>"); rec("<|im_end|>"); rec("");
    return b;
}

bool feed(GrammarConstraint& jc, const std::string& s) {
    for (unsigned char c : s) {
        if (!jc.allows(c)) return false;
        jc.accept(c);
    }
    return true;
}

GrammarConstraint make_constraint(const Grammar& g, Tokenizer& tok) {
    GrammarConstraint jc;
    jc.init(tok, VOCAB, {EOS}, g);
    return jc;
}

// Random *valid* JSON (valid by construction): the generic matcher must accept
// all of it, the same stress the DPDA gets in test_grammar.
struct Gen {
    std::mt19937 rng;
    explicit Gen(uint32_t seed) : rng(seed) {}
    int pick(int n) { return (int)(rng() % (uint32_t)n); }
    std::string ws() { if (pick(3)) return ""; const char* w = " \t\n\r"; return std::string(1, w[pick(4)]); }
    std::string str() {
        std::string o = "\"";
        int n = pick(6);
        for (int i = 0; i < n; ++i) {
            int k = pick(10);
            if (k == 0) o += "\\\""; else if (k == 1) o += "\\\\"; else if (k == 2) o += "\\n";
            else if (k == 3) o += "\\u00e9"; else if (k == 4) o += "\xc3\xa9";
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
        if (t == 0) {
            std::string o = "{" + ws(); int n = pick(4);
            for (int i = 0; i < n; ++i) { if (i) o += "," + ws(); o += str() + ws() + ":" + ws() + value(depth - 1); }
            return o + ws() + "}";
        }
        if (t == 1) {
            std::string o = "[" + ws(); int n = pick(4);
            for (int i = 0; i < n; ++i) { if (i) o += "," + ws(); o += value(depth - 1); }
            return o + ws() + "]";
        }
        if (t == 2) return str();
        if (t == 3) return num();
        if (t == 4) return "true";
        if (t == 5) return "false";
        return "null";
    }
};
}  // namespace

int main() {
    Tokenizer tok;
    auto blob = make_blob();
    CHECK(tok.load_memory(blob.data(), blob.size(), VOCAB));

    // --- Part 1: the JSON grammar reproduces the automaton corpus -----------
    Grammar gj = build_json_grammar();
    const char* valid[] = {
        "{}", "[]", "{\"a\":1}", "{\"a\":1,\"b\":2}", "[1,2,3]", "[true,false,null]",
        "{\"a\":\"b\"}", "\"hello\"", "123", "-123", "0", "0.5", "-0.5", "1e3",
        "1E+3", "1e-3", "1.5e2", "true", "false", "null", "{\"a\":[1,{\"b\":null}]}",
        "[[],{},[[]]]", "{\"a\":\"\\u0041\"}", " { \"a\" : 1 } ", "1.2e3", "123.456e+78",
    };
    for (const char* s : valid) {
        auto jc = make_constraint(gj, tok);
        CHECK(feed(jc, s));
        CHECK(jc.done());
        CHECK(jc.allows(EOS));
    }
    const char* invalid[] = {
        "{,}", "{\"a\"}", "{\"a\":}", "{\"a\":1,}", "[1,]", "[1 2]", "01", "1.a",
        "1e+a", "\"abc\\q\"", "\"\\u00G1\"", "truX", "nulll", "}{", "1 2",
        "1.2.3", "1e2e3", "0.0.0", "1.2e3.4", "1..2", "[1,,2]",
    };
    for (const char* s : invalid) {
        auto jc = make_constraint(gj, tok);
        CHECK(!feed(jc, s));
    }
    const char* prefix[] = {"{", "{\"a\":", "{\"a\":1", "[1,", "\"abc", "tru", "1.", "1e", "1e+"};
    for (const char* s : prefix) {
        auto jc = make_constraint(gj, tok);
        CHECK(feed(jc, s));
        CHECK(!jc.done());
        CHECK(!jc.allows(EOS));
    }

    // --- Part 2: schemas ----------------------------------------------------
    auto compile = [&](const char* json) {
        Grammar g;
        std::string err;
        common_json sch = common_json::parse(json);
        bool okc = compile_schema(sch, g, err);
        if (!okc) checks::fail(__FILE__, __LINE__, "compile failed: " + err + " for " + json);
        return g;
    };
    auto accepts = [&](const Grammar& g, const std::string& s) {
        auto jc = make_constraint(g, tok);
        return feed(jc, s) && jc.done();
    };

    // simple typed object
    Grammar g1 = compile(R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false})");
    CHECK(accepts(g1, "{\"name\":\"Alice\",\"age\":30}"));
    CHECK(!accepts(g1, "{\"age\":30,\"name\":\"Alice\"}"));  // required keys keep a fixed order
    CHECK(!accepts(g1, "{\"name\":\"Alice\"}"));             // missing required age
    CHECK(!accepts(g1, "{\"name\":1,\"age\":30}"));          // name must be string
    CHECK(!accepts(g1, "{\"name\":\"Alice\",\"age\":30.5}")); // age must be integer
    CHECK(!accepts(g1, "{\"name\":\"Alice\",\"age\":30,\"x\":1}")); // closed object

    // enum + nested array of objects
    Grammar g2 = compile(R"({"type":"object","properties":{"color":{"enum":["red","green","blue"]},"tags":{"type":"array","items":{"type":"string"}}},"required":["color"],"additionalProperties":false})");
    CHECK(accepts(g2, "{\"color\":\"red\"}"));
    CHECK(accepts(g2, "{\"color\":\"blue\",\"tags\":[\"a\",\"b\"]}"));
    CHECK(!accepts(g2, "{\"color\":\"yellow\"}"));           // not in enum
    CHECK(!accepts(g2, "{\"color\":\"red\",\"tags\":[1]}")); // items must be string
    CHECK(accepts(g2, "{\"color\":\"green\",\"tags\":[]}")); // empty array ok

    // nested object + boolean + number
    Grammar g3 = compile(R"({"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"},"active":{"type":"boolean"}},"required":["id"],"additionalProperties":false},"score":{"type":"number"}},"required":["user"],"additionalProperties":false})");
    CHECK(accepts(g3, "{\"user\":{\"id\":1,\"active\":true},\"score\":-1.5e2}"));
    CHECK(accepts(g3, "{\"user\":{\"id\":1}}"));
    CHECK(!accepts(g3, "{\"user\":{\"active\":true}}"));     // missing required id
    CHECK(!accepts(g3, "{\"user\":{\"id\":1,\"active\":\"yes\"}}")); // boolean

    // array of integers at top level
    Grammar g4 = compile(R"({"type":"array","items":{"type":"integer"}})");
    CHECK(accepts(g4, "[1,2,3]"));
    CHECK(accepts(g4, "[]"));
    CHECK(!accepts(g4, "[1,\"x\"]"));
    CHECK(!accepts(g4, "[1.5]"));

    // enum of numbers (shared-prefix literals: 1 vs 12 vs 123)
    Grammar g5 = compile(R"({"enum":[1,12,123]})");
    CHECK(accepts(g5, "1"));
    CHECK(accepts(g5, "12"));
    CHECK(accepts(g5, "123"));
    CHECK(!accepts(g5, "1234"));
    CHECK(!accepts(g5, "2"));

    // unsupported combinators are refused
    {
        Grammar g; std::string err;
        CHECK(!compile_schema(common_json::parse(R"({"anyOf":[{"type":"string"}]})"), g, err));
        CHECK(err.find("anyOf") != std::string::npos);
        CHECK(!compile_schema(common_json::parse(R"({"$ref":"#/x"})"), g, err));
    }

    // --- stress: random valid JSON through the generic matcher --------------
    Gen gen(20261122);
    for (int iter = 0; iter < 400; ++iter) {
        std::string s = gen.value(3);
        auto jc = make_constraint(gj, tok);
        if (!feed(jc, s) || !jc.done()) {
            checks::fail(__FILE__, __LINE__, "random valid JSON rejected: " + checks::esc(s));
        }
    }

    if (checks::g_fail) { std::fprintf(stderr, "%d check(s) failed\n", checks::g_fail); return 1; }
    std::printf("test_schema: all checks passed\n");
    return 0;
}
