// Full GBNF: the general multi-stack matcher and the text parser.
//
// Unlike test_schema (LL(1)), this exercises grammars the deterministic matcher
// cannot handle: ambiguous shared-prefix alternatives (members ::= pair | pair
// "," members), repetition * + ? {n,m}, inverse ranges, and groups.  Plus parser
// refusals: left recursion, undefined rules, and (for now) unicode escapes.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "checks.h"
#include "gbnf_full.h"
#include "tokenizer.h"

namespace {
constexpr int EOS = 256, VOCAB = 257;

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t u) { uint8_t x[4]; std::memcpy(x, &u, 4); b.insert(b.end(), x, x + 4); };
    auto f32 = [&](float f) { uint32_t u = 0; std::memcpy(&u, &f, 4); u32(u); };
    auto rec = [&](const std::string& s) { f32(0); u32((uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end()); };
    u32(6);
    for (int i = 0; i < 256; ++i) rec(std::string(1, (char)i));
    rec("<|eos|>");
    return b;
}

bool feed(GbnfConstraint& jc, const std::string& s) {
    for (unsigned char c : s) {
        if (!jc.allows(c)) return false;
        jc.accept(c);
    }
    return true;
}

GbnfGrammar parse_ok(const char* gbnf) {
    GbnfGrammar g;
    std::string err;
    if (!g.parse(gbnf, err)) checks::fail(__FILE__, __LINE__, "parse failed: " + err);
    return g;
}

bool accepts(const GbnfGrammar& g, Tokenizer& tok, const std::string& s) {
    GbnfConstraint jc;
    jc.init(tok, VOCAB, {EOS}, g);
    return feed(jc, s) && jc.done();
}
}  // namespace

int main() {
    Tokenizer tok;
    auto blob = make_blob();
    CHECK(tok.load_memory(blob.data(), blob.size(), VOCAB));

    // --- arithmetic.gbnf (groups, * +, char sets) ---------------------------
    GbnfGrammar arith = parse_ok(R"GBNF(
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
)GBNF");
    CHECK(accepts(arith, tok, "x=1\n"));
    CHECK(accepts(arith, tok, "x+y=1\n"));         // lhs is an expr, rhs a term
    CHECK(accepts(arith, tok, "(a+b)=c\n"));
    CHECK(accepts(arith, tok, "a=1\nb=2\n"));      // the + on root
    CHECK(!accepts(arith, tok, "x=1"));            // missing trailing newline
    CHECK(!accepts(arith, tok, "=1\n"));           // no lhs
    CHECK(!accepts(arith, tok, "x==1\n"));

    // --- a JSON grammar with an AMBIGUOUS shared-prefix members rule --------
    // members ::= pair | pair "," members  -- both alts start with a string,
    // which the LL(1) matcher cannot dispatch; the NFA explores both.
    GbnfGrammar json = parse_ok(R"GBNF(
root   ::= ws value ws
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws "}" | "{" ws members ws "}"
members ::= pair | pair ws "," ws members
pair   ::= string ws ":" ws value
array  ::= "[" ws "]" | "[" ws elements ws "]"
elements ::= value | value ws "," ws elements
string ::= "\"" chars "\""
chars  ::= char chars |
char   ::= [^"\\\x00-\x1F] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex    ::= [0-9a-fA-F]
number ::= "-"? int frac? exp?
int    ::= "0" | [1-9] [0-9]*
frac   ::= "." [0-9]+
exp    ::= [eE] [+-]? [0-9]+
ws     ::= [ \t\n\r]*
)GBNF");
    const char* valid[] = {
        "{}", "[]", "{\"a\":1}", "{\"a\":1,\"b\":2}", "[1,2,3]", "[true,false,null]",
        "\"hello\"", "123", "-0.5", "1.5e2", "{\"a\":[1,{\"b\":null}]}",
        "{\"a\":\"\\u0041\"}", " { \"a\" : 1 } ", "[[],{},[[]]]",
    };
    for (const char* s : valid) CHECK(accepts(json, tok, s));
    const char* invalid[] = {
        "{,}", "{\"a\"}", "{\"a\":1,}", "[1,]", "01", "1.2.3", "1e2e3", "truX",
        "{\"a\":1 \"b\":2}", "[1,,2]", "\"abc\\q\"",
    };
    for (const char* s : invalid) CHECK(!accepts(json, tok, s));

    // --- ambiguity, repetition, inverse ranges, groups ----------------------
    GbnfGrammar amb = parse_ok("root ::= \"a\" | \"ab\"");
    CHECK(accepts(amb, tok, "a"));
    CHECK(accepts(amb, tok, "ab"));
    CHECK(!accepts(amb, tok, "abc"));
    CHECK(!accepts(amb, tok, "b"));

    GbnfGrammar plus = parse_ok("root ::= \"a\"+");
    CHECK(accepts(plus, tok, "a"));
    CHECK(accepts(plus, tok, "aaaa"));
    CHECK(!accepts(plus, tok, ""));

    GbnfGrammar star = parse_ok("root ::= \"a\"*");
    CHECK(accepts(star, tok, ""));
    CHECK(accepts(star, tok, "aaa"));
    CHECK(!accepts(star, tok, "aab"));

    GbnfGrammar bounded = parse_ok("root ::= \"a\"{2,4}");
    CHECK(!accepts(bounded, tok, "a"));
    CHECK(accepts(bounded, tok, "aa"));
    CHECK(accepts(bounded, tok, "aaaa"));
    CHECK(!accepts(bounded, tok, "aaaaa"));

    GbnfGrammar exact = parse_ok("root ::= [0-9]{3}");
    CHECK(accepts(exact, tok, "123"));
    CHECK(!accepts(exact, tok, "12"));
    CHECK(!accepts(exact, tok, "1234"));

    GbnfGrammar inv = parse_ok("root ::= [^a\r\n]+");
    CHECK(accepts(inv, tok, "xyz"));
    CHECK(!accepts(inv, tok, "a"));
    CHECK(!accepts(inv, tok, "xa"));

    GbnfGrammar grp = parse_ok("root ::= (\"ab\" | \"cd\")+");
    CHECK(accepts(grp, tok, "abcd"));
    CHECK(accepts(grp, tok, "cdab"));
    CHECK(!accepts(grp, tok, "ac"));

    // --- parser refusals ----------------------------------------------------
    {
        GbnfGrammar g; std::string err;
        CHECK(!g.parse("root ::= root \"a\" | \"a\"", err));   // left recursion
        CHECK(err.find("recursion") != std::string::npos);
        CHECK(!g.parse("root ::= undefined_rule", err));        // undefined rule
        CHECK(!g.parse("root ::= \"\\u0041\"", err));           // unicode deferred
        CHECK(err.find("unicode") != std::string::npos);
        CHECK(!g.parse("foo ::= \"a\"", err));                  // no root
        CHECK(!g.parse("root ::= \"unterminated", err));        // syntax error
    }

    if (checks::g_fail) { std::fprintf(stderr, "%d check(s) failed\n", checks::g_fail); return 1; }
    std::printf("test_gbnf_full: all checks passed\n");
    return 0;
}
