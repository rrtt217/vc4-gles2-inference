#pragma once

// Constrained decoding: a byte-level JSON automaton that masks the vocabulary.
//
// The hook is the same one the sampler already uses for penalties: a host-side
// `vector<float>` of the whole vocabulary, where "forbidden" is `-1e30f`
// (sampling.cpp's no_repeat_ngram and chat.cpp's mask_markers both do exactly
// this).  A constraint only needs to live above the logits boundary -- it never
// touches attention, the KV cache or the forward pass -- so it is architecture
// agnostic: dense, MoE, hybrid and RWKV all return the same per-step logits.
//
// Matching is over the *bytes a token contributes to the output*, not over the
// decoded display text.  The .tok blob already stores raw bytes per entry
// (make_tok.py inverts the ByteLevel / SentencePiece alphabets), so id_text(id)
// is the byte string to feed the automaton; the legacy <0xNN> fallback spelling
// is decoded the same way tokenizer.cpp's decode() does it.  A token may end
// mid-construct (half a number, an open string), so the automaton state persists
// across tokens -- that is the whole reason it is a pushdown machine and not a
// per-token regex.
//
// JSON is LL(1): the next byte always decides the transition, so this is a
// deterministic pushdown automaton with no backtracking.  The one non-consuming
// transition is a finished number handing its terminating byte (',' '}' ']' or
// whitespace) back to the enclosing context.

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.h"

// The contract the three matchers present: a host-side vocabulary mask applied
// above the logits boundary.  json_object uses the JSON automaton below;
// json_schema uses the grammar-driven matcher in gbnf.h.  The worker builds one
// per request; the CLI's CliConstraint (constrain.h) resets instead.
struct Constraint {
    virtual ~Constraint() = default;
    virtual void reset() = 0;
    virtual void mask(std::vector<float>& logits) const = 0;
    virtual void accept(int token) = 0;
    virtual bool done() const = 0;
    // The same test mask() applies, for one id (EOS only when done()).
    virtual bool allows(int id) const = 0;
};

class JsonConstraint : public Constraint {
public:
    // Build the token->bytes table once per loaded model.  `eos_ids` are the
    // ids that end a turn (model.id_eos / id_stop); they are allowed only in an
    // accepting state, never as ordinary bytes.  A second init() replaces the
    // table (model swap), matching Tokenizer::reset()'s contract.
    void init(const Tokenizer& tok, int vocab, const std::vector<int>& eos_ids);

    // Back to "expect one top-level value".  Called at the start of a request.
    void reset() override;

    // Set every token the automaton cannot continue with to -1e30f.  EOS is
    // passed through only when done(); empty-byte tokens are always masked (they
    // cannot advance the parse and could only spin).  O(vocab * token_len).
    void mask(std::vector<float>& logits) const;

    // Advance the state by the sampled token.  Must be a token mask() allowed;
    // the transition is deterministic, so this reproduces mask()'s end state.
    void accept(int token);

    // True when a complete top-level value has been parsed and only whitespace
    // (or EOS) may follow.  This is when EOS becomes legal.
    bool done() const;

    // True when the automaton can continue with this token from the current
    // state (EOS only when done()).  mask() is this applied to the whole vocab.
    bool allows(int id) const override;

    // Number of tokens mask() would keep at the current state (diagnostics/tests).
    int allowed_count() const;

    bool ready() const { return !bytes_.empty(); }

private:
    // A parse frame.  `kind` selects the construct; `phase` is its position;
    // `aux` is a small counter (string \u hex digits); `lit` selects which of
    // true/false/null a LITERAL frame is matching.  Four bytes so a state copy
    // is a cheap memcpy of the frames actually in use.
    struct Frame {
        uint8_t kind = 0, phase = 0, aux = 0, lit = 0;
    };
    enum Kind : uint8_t {
        ROOT_AFTER = 0,   // top-level value finished: whitespace then EOS only
        VALUE,            // dispatch on the first byte of a value
        OBJECT, ARRAY, STRING, NUMBER, LITERAL,
    };
    static constexpr int kMaxDepth = 128;   // JSON nesting cap; refuse deeper
    struct State {
        Frame f[kMaxDepth];
        int n = 0;
    };

    // Feed one byte; false = the byte cannot continue the parse.  May pop a
    // finished number and re-process the byte against the parent (the only
    // non-consuming transition), hence the loop.
    static bool step(State& s, uint8_t c);
    static bool push(State& s, uint8_t kind, uint8_t phase);
    static bool num_stop(uint8_t phase);
    // Run a whole token's bytes from a state; returns the end state in `out`.
    bool run(const std::string& bytes, State& out) const;
    bool is_done(const State& s) const;
    bool is_eos(int id) const;

    State state_;
    std::vector<std::string> bytes_;   // per id: the raw output bytes
    std::vector<int> eos_;
};
