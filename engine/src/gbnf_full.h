#pragma once

// Full GBNF support: a GBNF text parser and a *general* context-free grammar
// matcher, for arbitrary user grammars (the json_schema tier's LL(1) matcher in
// gbnf.h cannot handle ambiguous alternatives or shared prefixes).
//
// The matcher is llama.cpp's multi-stack NFA (subset construction), adapted
// self-contained: the state is a SET of pushdown stacks, and advance_stack
// explores every alternative of a rule reference in parallel, deduplicating so
// nullable recursion terminates.  Character rules match Unicode code points;
// tokens are decoded incrementally so a multi-byte character split across tokens
// remains legal while its partial sequence can still fit a range.  This subsumes
// the LL(1) matcher but is slower (per-token simulation through the stack set),
// so the schema tier keeps its fast path; this one serves arbitrary GBNF text.
//
// Literals, ranges [a-z] / [^...] / [一-鿿], escapes \xNN / \uNNNN / \UNNNNNNNN,
// groups, rule refs and repetition * + ? {n,m} are supported.  Left recursion is
// detected at parse time and refused (it would loop advance_stack forever).
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
    uint32_t value = 0;   // Unicode code point (CHAR*) or rule id (RULE_REF)
};
using GbnfRule = std::vector<GbnfElem>;

class GbnfGrammar {
public:
    std::vector<GbnfRule> rules;   // flat: ALT-separated alternatives, END-terminated
    size_t start = 0;

    // Parse GBNF text.  false + err on a syntax error, undefined rule, or left
    // recursion.
    bool parse(const std::string& gbnf, std::string& err);

    using Stack = std::vector<const GbnfElem*>;
    using Stacks = std::vector<Stack>;

    struct PartialUtf8 {
        uint32_t value = 0;   // decoded bits so far, unshifted
        int n_remain = 0;     // continuation bytes expected; -1 = invalid
    };
    struct MatchState {
        Stacks stacks;
        PartialUtf8 partial;
    };

    MatchState start_state() const;
    // Consume one complete code point; stacks with a different active branch die.
    void accept(Stacks& s, uint32_t cp) const;
    bool accepts(const MatchState& from, const std::string& bytes, MatchState& out) const;
    bool done(const MatchState& s) const;   // complete UTF-8 and an empty stack
    // Test-only introspection: the residual partial sequence after a token.
    static PartialUtf8 partial_after(const MatchState& state);

private:
    void advance(const Stack& stack, Stacks& out) const;
    static std::pair<bool, const GbnfElem*> match_char(const GbnfElem* pos, uint32_t cp);
    static bool match_partial(const GbnfElem* pos, PartialUtf8 partial);
    bool has_left_recursion(std::string& err) const;
    void prune_partial(Stacks& s, uint32_t value, int remain) const;
    bool stacks_done(const Stacks& s) const;
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
    GbnfGrammar::MatchState state_;
    std::vector<std::string> bytes_;
    std::vector<int> eos_;
};
