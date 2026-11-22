#pragma once

// Full GBNF support: a GBNF text parser and a *general* context-free grammar
// matcher, for arbitrary user grammars (the json_schema tier's LL(1) matcher in
// gbnf.h cannot handle ambiguous alternatives or shared prefixes).
//
// The matcher is llama.cpp's multi-stack NFA (subset construction), adapted
// self-contained and byte-based: the state is a SET of pushdown stacks, and
// advance_stack explores every alternative of a rule reference in parallel,
// deduplicating so nullable recursion terminates.  A token is legal iff some
// stack survives consuming all its bytes.  This subsumes the LL(1) matcher but
// is slower (per-token simulation through the stack set), so the schema tier
// keeps its fast path; this one serves arbitrary GBNF text.
//
// Step 1 is byte-based (0..255): literals, char ranges [a-z] / [^...] / \xNN,
// groups, rule refs, and repetition * + ? {n,m}.  Unicode code-point ranges and
// \uNNNN escapes (and partial-UTF-8 matching) are a later step; they are refused
// by name here rather than silently mis-matched.  Left recursion is detected at
// parse time and refused (it would loop advance_stack forever).
//
// Adapted from llama.cpp src/llama-grammar.cpp (MIT, (c) llama.cpp contributors).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "grammar.h"   // Constraint base
#include "tokenizer.h"

struct GbnfElem {
    enum Type : uint8_t {
        END = 0, ALT = 1, RULE_REF = 2, CHAR = 3, CHAR_NOT = 4,
        RNG_UPPER = 5, CHAR_ALT = 6, CHAR_ANY = 7,
    };
    uint8_t type = END;
    uint32_t value = 0;   // byte (CHAR*) or rule id (RULE_REF)
};
using GbnfRule = std::vector<GbnfElem>;

class GbnfGrammar {
public:
    std::vector<GbnfRule> rules;   // flat: ALT-separated alternatives, END-terminated
    size_t start = 0;

    // Parse GBNF text.  false + err on a syntax error, undefined rule, left
    // recursion, or an unsupported (Unicode) construct.
    bool parse(const std::string& gbnf, std::string& err);

    using Stack = std::vector<const GbnfElem*>;
    using Stacks = std::vector<Stack>;

    Stacks start_stacks() const;
    void accept(Stacks& s, uint8_t c) const;                 // consume one byte
    bool accepts(const Stacks& from, const std::string& bytes, Stacks& out) const;
    bool done(const Stacks& s) const;                        // some stack empty

private:
    void advance(const Stack& stack, Stacks& out) const;
    static std::pair<bool, const GbnfElem*> match_char(const GbnfElem* pos, uint8_t c);
    bool has_left_recursion(std::string& err) const;
};

// The token-table + mask wrapper, same Constraint contract as the other tiers.
class GbnfConstraint : public Constraint {
public:
    void init(const Tokenizer& tok, int vocab, const std::vector<int>& eos_ids,
              const GbnfGrammar& g);
    void reset();
    void mask(std::vector<float>& logits) const override;
    void accept(int token) override;
    bool done() const override;
    bool allows(int id) const;

private:
    bool is_eos(int id) const;
    const GbnfGrammar* g_ = nullptr;
    GbnfGrammar::Stacks stacks_;
    std::vector<std::string> bytes_;
    std::vector<int> eos_;
};
