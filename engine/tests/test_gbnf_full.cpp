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
constexpr int EOS = 256, CJK_ONE = 257, CJK_TWO = 258, CJK_MID = 259,
              CJK_TAIL = 260, CJK_BAD = 261, ASCII_X = 262, VOCAB = 263;

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t u) { uint8_t x[4]; std::memcpy(x, &u, 4); b.insert(b.end(), x, x + 4); };
    auto f32 = [&](float f) { uint32_t u = 0; std::memcpy(&u, &f, 4); u32(u); };
    auto rec = [&](const std::string& s) { f32(0); u32((uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end()); };
    u32(6);
    for (int i = 0; i < 256; ++i) rec(std::string(1, (char)i));
    rec("<|eos|>");
    rec("\xe4\xb8\x80");  // CJK_ONE: complete 一
    rec("\xe4\xb8\x8c");  // CJK_TWO: complete 二
    rec("\xe4");          // CJK_MID: first byte of a three-byte CJK character
    rec("\xb8\x80");      // CJK_TAIL: its continuation bytes
    rec("\xef\xbf\xbd");  // CJK_BAD: U+FFFD, outside the CJK range
    rec("x");             // ASCII_X
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
        CHECK(!g.parse("foo ::= \"a\"", err));                  // no root
        CHECK(!g.parse("root ::= \"unterminated", err));        // syntax error
    }

    // --- Unicode: literals, code-point ranges and split UTF-8 tokens --------
    GbnfGrammar unicode = parse_ok("root ::= \"\\u4E00\" | [一-鿿] | [^\\r\\n\\u2028]+");
    CHECK(accepts(unicode, tok, "\xe4\xb8\x80"));             // 一
    CHECK(accepts(unicode, tok, "\xe9\xbe\xbf"));             // 鿿
    CHECK(accepts(unicode, tok, "xyz"));
    CHECK(!accepts(unicode, tok, "\n"));
    CHECK(accepts(unicode, tok, "\xef\xbf\xbd"));             // inverse Unicode range
    GbnfGrammar ascii_only = parse_ok("root ::= [a-z]+");
    CHECK(!accepts(ascii_only, tok, "\xe4\xb8\x80"));
    GbnfGrammar literal_jp = parse_ok("root ::= \"日本\"");
    CHECK(accepts(literal_jp, tok, "日本"));
    CHECK(!accepts(literal_jp, tok, "日x"));

    // A three-byte character may arrive as two tokens.  The first byte must be
    // legal, but EOS remains forbidden until its continuation bytes arrive.
    {
        GbnfGrammar one = parse_ok("root ::= [一-鿿]");
        GbnfConstraint jc;
        jc.init(tok, VOCAB, {EOS}, one);
        CHECK(jc.allows(CJK_ONE));
        CHECK(jc.allows(CJK_TWO));
        CHECK(jc.allows(CJK_MID));
        CHECK(!jc.allows(CJK_BAD));
        CHECK(!jc.allows(ASCII_X));
        jc.accept(CJK_MID);
        CHECK(!jc.done());
        CHECK(!jc.allows(EOS));
        CHECK(jc.allows(CJK_TAIL));
        jc.accept(CJK_TAIL);
        CHECK(jc.done());
        CHECK(jc.allows(EOS));
    }
    {
        // Invalid continuation, and a first byte whose possible range misses
        // [一-鿿], must be rejected rather than buffered.
        GbnfGrammar one = parse_ok("root ::= [一-鿿]");
        GbnfConstraint jc;
        jc.init(tok, VOCAB, {EOS}, one);
        CHECK(!jc.allows(0xf0));   // four-byte code points are outside the range
        jc.accept(CJK_MID);
        CHECK(!jc.allows((int)'x'));
        CHECK(!jc.allows(0xff));
    }
    {
        // Worst case: 一 arrives as three one-byte tokens (0xe4, 0xb8, 0x80).
        // Each step must keep the rule alive and EOS refused until the last byte.
        GbnfGrammar one = parse_ok("root ::= [一-鿿]");
        GbnfConstraint jc;
        jc.init(tok, VOCAB, {EOS}, one);
        CHECK(jc.allows(0xe4));
        jc.accept(0xe4);
        CHECK(!jc.allows(EOS));
        CHECK(jc.allows(0xb8));
        jc.accept(0xb8);
        CHECK(!jc.allows(EOS));
        CHECK(jc.allows(0x80));
        jc.accept(0x80);
        CHECK(jc.done());
        CHECK(jc.allows(EOS));
    }
    {
        // An alternative that dies on the partial bytes must not kill the one
        // that is still alive: [一-鿿] | [\\u4e00-\\u9fff] share the range, while
        // [^a-z]+ takes any code point and must survive a bare first byte.
        GbnfGrammar alt = parse_ok("root ::= [a-z]+ | [^a-z]+");
        GbnfConstraint jc;
        jc.init(tok, VOCAB, {EOS}, alt);
        CHECK(jc.allows(0xe4));            // [^a-z] branch still covers it
        jc.accept(0xe4);
        CHECK(jc.allows(0xb8));
        jc.accept(0xb8);
        CHECK(jc.allows(0x80));
        jc.accept(0x80);
        CHECK(jc.done());
        CHECK(jc.allows(EOS));
    }

    {
        // A *finished* parse accepts EOS and nothing else.  A token that ends
        // in the middle of a character must therefore be refused there: the
        // empty stack that says "done" cannot also be waiting for continuation
        // bytes.  When it was allowed (the `st.empty() ||` guard in
        // prune_partial), a drawn lead byte dead-ended the matcher -- every
        // later step found nothing legal -- and the turn continued on argmax
        // over all -1e30, i.e. one token repeated to max_tokens.  Seen from a
        // client as "the forced prefix came out, then the model rambled".
        GbnfGrammar done_g = parse_ok("root ::= \"ab\"");
        GbnfConstraint jc;
        jc.init(tok, VOCAB, {EOS}, done_g);
        CHECK(feed(jc, "ab"));
        CHECK(jc.done());
        CHECK(jc.allows(EOS));
        CHECK(!jc.allows(CJK_MID));        // 0xe4: a lead byte, not a new token
        CHECK(!jc.allows(0xe4));
        CHECK(!jc.allows((int)'a'));
        // The same token stays legal while a character is still expected, so
        // the fix is "no partial at a done state", not "no partial at all".
        GbnfGrammar wait_g = parse_ok("root ::= \"ab\" .");
        GbnfConstraint wc;
        wc.init(tok, VOCAB, {EOS}, wait_g);
        CHECK(feed(wc, "ab"));
        CHECK(!wc.done());
        CHECK(wc.allows(0xe4));
    }

    if (checks::g_fail) { std::fprintf(stderr, "%d check(s) failed\n", checks::g_fail); return 1; }
    std::printf("test_gbnf_full: all checks passed\n");
    return 0;
}
