#pragma once

// A generic grammar matcher for constrained decoding, and the JSON-Schema tier.
//
// Where grammar.cpp's JsonConstraint is a hand-written JSON automaton, this is a
// grammar-driven matcher: a schema is compiled to a small rule set (see
// schema.h) and the same mask/accept/done contract applies.  It stays above the
// logits boundary exactly like the JSON automaton, so it is architecture
// agnostic and changes no numerics.
//
// The matcher is a *deterministic* pushdown machine, not llama.cpp's multi-stack
// NFA.  That is sound for the grammars the schema compiler emits: every choice
// is LL(1) -- type dispatch is by first byte, and string sets (property names,
// enums) are trie-factored so alternatives never share a first byte.  A nullable
// alternative is the catch-all fallback (the '?' / '*' of the source grammar).
// Keeping one stack makes a per-token trial a cheap memcpy, which matters at a
// 65536-entry vocabulary.
//
// End-of-input is a sentinel byte (kEnd = 256) matched only by an END element
// the compiler puts at the end of the root rule.  done() is "the sentinel is
// accepted from here", i.e. the parse can finish with no more real bytes -- the
// same accepting-state idea as the JSON automaton's ROOT_AFTER.

#include <cstdint>
#include <string>
#include <vector>

#include "grammar.h"
#include "tokenizer.h"

class Grammar {
public:
    struct Elem {
        enum Type : uint8_t { RANGE, REF, END } type = RANGE;
        uint8_t lo = 0, hi = 0;   // RANGE: inclusive byte range
        uint16_t rule = 0;        // REF: target rule index
    };
    struct Alt { std::vector<Elem> elems; };
    struct Rule { std::vector<Alt> alts; };

    std::vector<Rule> rules;
    uint16_t start = 0;

    static constexpr uint16_t kEnd = 256;   // sentinel: end of input (not a byte)

    // Rule-building helpers used by the compiler.
    uint16_t add_rule();                                   // empty rule, returns id
    void set_alts(uint16_t r, std::vector<Alt> alts);
    static Elem range(uint8_t lo, uint8_t hi);
    static Elem ref(uint16_t r);
    static Elem end();
    static Alt seq(std::vector<Elem> elems);

    void prepare();          // compute nullable[]; call once after building

    // Matcher state: a stack of (rule, alt, elem) positions.  Fixed array so a
    // trial copy is a memcpy of the frames in use, not a heap allocation.
    static constexpr int kMaxDepth = 256;
    struct Frame { uint16_t rule, alt, elem; };
    struct State { Frame f[kMaxDepth]; int n = 0; };

    State start_state() const;
    // Feed one byte (0..255) or kEnd.  false = cannot continue.  Expanding a REF
    // advances the parent first, so a failed trial can be discarded whole.
    bool step(State& s, uint16_t c) const;
    bool done(const State& s) const;                 // kEnd accepted from here
    bool accepts(const State& from, const std::string& bytes, State& out) const;

private:
    std::vector<bool> nullable_;
    bool alt_nullable(const Alt& a) const;
    bool alt_starts(const Alt& a, uint16_t c, int depth) const;
    int choose_alt(uint16_t rule, uint16_t c) const;
};

// The token-table + mask wrapper around a Grammar: the same contract as
// JsonConstraint, so the worker treats both identically.
class GrammarConstraint : public Constraint {
public:
    void init(const Tokenizer& tok, int vocab, const std::vector<int>& eos_ids,
              const Grammar& g);
    void reset();
    void mask(std::vector<float>& logits) const;
    void accept(int token);
    bool done() const;
    bool allows(int id) const;
    int allowed_count() const;
    bool ready() const { return grammar_ != nullptr && !bytes_.empty(); }

private:
    bool is_eos(int id) const;
    const Grammar* grammar_ = nullptr;
    Grammar::State state_;
    std::vector<std::string> bytes_;
    std::vector<int> eos_;
};
