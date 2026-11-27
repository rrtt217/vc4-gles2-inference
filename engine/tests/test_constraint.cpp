// The CLI/REPL constraint layer: the /constrain spec parser, the choice
// shortcut's GBNF text, and CliConstraint installing each tier and resetting it
// per reply.
//
// The matchers themselves are covered by test_grammar / test_schema /
// test_gbnf_full; this is the surface above them -- what a flag or a slash
// command can say, and that "a typo costs nothing" holds (a failed build leaves
// the previous constraint active).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "checks.h"
#include "constrain.h"
#include "tokenizer.h"

namespace {
constexpr int EOS = 256, CJK_ONE = 257, CJK_TWO = 258, ASCII_X = 262, VOCAB = 263;

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b;
    auto u32 = [&](uint32_t u) { uint8_t x[4]; std::memcpy(x, &u, 4); b.insert(b.end(), x, x + 4); };
    auto f32 = [&](float f) { uint32_t u = 0; std::memcpy(&u, &f, 4); u32(u); };
    auto rec = [&](const std::string& s) { f32(0); u32((uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end()); };
    u32(6);
    for (int i = 0; i < 256; ++i) rec(std::string(1, (char)i));
    rec("<|eos|>");
    rec("\xe4\xb8\x80");  // CJK_ONE: 一
    rec("\xe4\xba\x8c");  // CJK_TWO: 二
    rec("\xe4");
    rec("\xb8\x80");
    rec("\xef\xbf\xbd");
    rec("x");             // ASCII_X
    return b;
}

// Feed the token ids that spell `s` one ASCII byte at a time.
bool feed(CliConstraint& c, const std::string& s) {
    for (unsigned char ch : s) {
        if (!c.matcher()->allows(ch)) return false;
        c.matcher()->accept(ch);
    }
    return true;
}

std::vector<int> eos_ids() { return {EOS}; }

bool write_file(const char* path, const char* text) {
    std::ofstream f(path);
    if (!f) return false;
    f << text;
    return (bool) f;
}
}  // namespace

int main() {
    Tokenizer tok;
    auto blob = make_blob();
    CHECK(tok.load_memory(blob.data(), blob.size(), VOCAB));

    // --- the choice shortcut -------------------------------------------------
    {
        std::string gbnf, err;
        CHECK(choice_grammar("yes|no", gbnf, err));
        CHECK_STR(gbnf, "root ::= \"yes\" | \"no\"");
        CHECK(!choice_grammar("yes||no", gbnf, err));   // an empty item
        CHECK(!choice_grammar("", gbnf, err));
        // Quotes, backslashes and whitespace are literal.
        CHECK(choice_grammar("a\"b|c\\d|e f", gbnf, err));
        CHECK_STR(gbnf, "root ::= \"a\\\"b\" | \"c\\\\d\" | \"e f\"");
    }

    // --- the spec parser -----------------------------------------------------
    {
        ConstraintSpec s;
        std::string err;
        CHECK(constraint_spec_parse("json", s, err) && s.kind == ConstraintSpec::JsonObject);
        CHECK(constraint_spec_parse("off", s, err) && s.kind == ConstraintSpec::None);
        CHECK(constraint_spec_parse("gbnf a.gbnf", s, err) && s.kind == ConstraintSpec::Gbnf &&
              s.arg == "a.gbnf");
        CHECK(constraint_spec_parse("schema  dir/my schema.json", s, err) &&
              s.kind == ConstraintSpec::JsonSchema && s.arg == "dir/my schema.json");
        CHECK(constraint_spec_parse("choice a|b c", s, err) && s.kind == ConstraintSpec::Choice &&
              s.arg == "a|b c");
        CHECK(!constraint_spec_parse("bogus", s, err));
        CHECK(!constraint_spec_parse("schema", s, err));     // no path
        CHECK(!constraint_spec_parse("json extra", s, err)); // takes no argument
        CHECK(!constraint_spec_parse("", s, err));
        CHECK_STR(constraint_spec_describe(ConstraintSpec{}), "none");
    }

    // --- choice, end to end: a literal list is a whole reply -----------------
    {
        CliConstraint c;
        std::string err;
        ConstraintSpec spec;
        spec.kind = ConstraintSpec::Choice;
        spec.arg = "yes|no";
        CHECK(c.build(spec, tok, VOCAB, eos_ids(), err));
        CHECK(c.active());
        CHECK(!c.matcher()->allows(EOS));      // not before the literal is complete
        CHECK(c.matcher()->allows((int)'y'));
        CHECK(c.matcher()->allows((int)'n'));
        CHECK(!c.matcher()->allows(ASCII_X));
        CHECK(feed(c, "yes"));
        CHECK(c.matcher()->done());
        CHECK(c.matcher()->allows(EOS));
        CHECK(!c.matcher()->allows(ASCII_X));
        // The next reply starts from the start state.
        c.reset();
        CHECK(!c.matcher()->done());
        CHECK(feed(c, "no"));
        CHECK(c.matcher()->done());
    }

    // --- a choice of CJK literals survives the GBNF literal path -------------
    {
        CliConstraint c;
        std::string err;
        ConstraintSpec spec;
        spec.kind = ConstraintSpec::Choice;
        spec.arg = "一|二";
        CHECK(c.build(spec, tok, VOCAB, eos_ids(), err));
        CHECK(c.matcher()->allows(CJK_ONE));
        CHECK(c.matcher()->allows(CJK_TWO));
        c.matcher()->accept(CJK_ONE);
        CHECK(c.matcher()->done());
        CHECK(c.matcher()->allows(EOS));
    }

    // --- json_object and the per-reply reset --------------------------------
    {
        CliConstraint c;
        std::string err;
        ConstraintSpec spec;
        spec.kind = ConstraintSpec::JsonObject;
        CHECK(c.build(spec, tok, VOCAB, eos_ids(), err));
        CHECK(c.matcher()->allows((int)'{'));
        CHECK(!c.matcher()->allows(ASCII_X));
        CHECK(feed(c, "{\"a\":1}"));
        CHECK(c.matcher()->done());
        CHECK(c.matcher()->allows(EOS));
        c.reset();
        CHECK(!c.matcher()->done());
        CHECK(c.matcher()->allows((int)'{'));
    }

    // --- a grammar file, the schema file, and the errors ---------------------
    {
        CliConstraint c;
        std::string err;
        CHECK(write_file("test_constraint.gbnf", "root ::= \"ab\"\n"));
        ConstraintSpec g;
        g.kind = ConstraintSpec::Gbnf;
        g.arg = "test_constraint.gbnf";
        CHECK(c.build(g, tok, VOCAB, eos_ids(), err));
        CHECK(feed(c, "ab"));
        CHECK(c.matcher()->done());

        // A failed build keeps the previous constraint: the typo costs nothing.
        ConstraintSpec bad;
        bad.kind = ConstraintSpec::Gbnf;
        bad.arg = "test_constraint_missing.gbnf";
        CHECK(!c.build(bad, tok, VOCAB, eos_ids(), err));
        CHECK(!err.empty());
        CHECK(c.active());
        CHECK(c.spec().arg == "test_constraint.gbnf");

        CHECK(write_file("test_constraint.json",
                         "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},"
                         "\"required\":[\"name\"],\"additionalProperties\":false}"));
        ConstraintSpec s;
        s.kind = ConstraintSpec::JsonSchema;
        s.arg = "test_constraint.json";
        CHECK(c.build(s, tok, VOCAB, eos_ids(), err));
        CHECK(feed(c, "{\"name\":\"a\"}"));
        CHECK(c.matcher()->done());
        CHECK(!c.matcher()->allows((int)'x'));

        // A grammar that cannot compile is refused by name, not approximated.
        ConstraintSpec badg;
        badg.kind = ConstraintSpec::Gbnf;
        CHECK(write_file("test_constraint_bad.gbnf", "root ::= left\nleft ::= root\n"));
        badg.arg = "test_constraint_bad.gbnf";
        CHECK(!c.build(badg, tok, VOCAB, eos_ids(), err));   // left recursion

        std::remove("test_constraint.gbnf");
        std::remove("test_constraint.json");
        std::remove("test_constraint_bad.gbnf");
    }

    // --- None clears ---------------------------------------------------------
    {
        CliConstraint c;
        std::string err;
        ConstraintSpec spec;
        spec.kind = ConstraintSpec::Choice;
        spec.arg = "a";
        CHECK(c.build(spec, tok, VOCAB, eos_ids(), err));
        CHECK(c.active());
        CHECK(c.build(ConstraintSpec{}, tok, VOCAB, eos_ids(), err));
        CHECK(!c.active());
        CHECK_STR(constraint_spec_describe(c.spec()), "none");
    }

    if (checks::g_fail) { std::fprintf(stderr, "%d check(s) failed\n", checks::g_fail); return 1; }
    std::printf("test_constraint: spec, choice, json, schema and reset pass\n");
    return 0;
}