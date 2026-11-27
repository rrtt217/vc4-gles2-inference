#pragma once

// The CLI and REPL side of constrained decoding.
//
// The matchers themselves and their contract live in grammar.h (JSON automaton),
// gbnf.h (the deterministic LL(1) grammar the schema tier compiles to) and
// gbnf_full.h (full GBNF); the worker builds them per request from the OpenAI
// `response_format` object.  This is the same three tiers for a human at a
// terminal -- one spec a flag or a slash command fills, one holder that owns the
// grammar objects and names the active matcher -- plus the shortcut that keeps
// the common case from needing a file: a fixed set of literals compiles to a
// GBNF root rule.
//
// A constraint is a host-side vocabulary mask above the logits boundary, so it
// forces the full-logits path (no GPU argmax, no top-k candidate read-back) and
// changes no numerics, exactly as the worker's comment says.

#include <string>
#include <vector>

#include "gbnf.h"
#include "gbnf_full.h"
#include "grammar.h"
#include "tokenizer.h"

struct ConstraintSpec {
    enum Kind { None = 0, JsonObject, JsonSchema, Gbnf, Choice };
    Kind kind = None;
    // JsonSchema / Gbnf: the path to read; Choice: the "A|B|C" list.  Unused for
    // JsonObject.  The file is read when the constraint is built, so a spec
    // holds a path, not the text.
    std::string arg;
};

// Parse a `/constrain` argument: "off" (or "none"), "json", "schema PATH",
// "gbnf PATH", "choice A|B|C".  Empty text is the caller's "show me" and is
// refused here.  false + err on an unknown kind or a missing argument.
bool constraint_spec_parse(const std::string& text, ConstraintSpec& out, std::string& err);

// One line for /stats and the echo after a change: "none", "json_object",
// "gbnf person.gbnf", "choice yes|no".
std::string constraint_spec_describe(const ConstraintSpec& s);

// `A|B|C` -> `root ::= "A" | "B" | "C"`, each literal escaped.  An empty list,
// an empty item or an empty-looking grammar is an error.  Items are literal,
// spaces included; '|' separates them and cannot appear inside one.
bool choice_grammar(const std::string& list, std::string& gbnf, std::string& err);

class CliConstraint {
public:
    // Compile `spec` against this model's tokenizer and install it, replacing
    // whatever was active (Kind::None clears).  false + err leaves the previous
    // constraint untouched, so a typo costs nothing -- except a model swap,
    // where the caller must clear on failure because the old token table
    // belongs to the old vocabulary.  `eos_ids` are the ids that end a turn;
    // they are released only once the parse can finish, the worker's rule.
    bool build(const ConstraintSpec& spec, const Tokenizer& tok, int vocab,
               const std::vector<int>& eos_ids, std::string& err);
    void clear();
    // Back to the start state, once per reply.
    void reset();
    bool active() const { return active_ != nullptr; }
    const ConstraintSpec& spec() const { return spec_; }
    Constraint* matcher() { return active_; }

private:
    ConstraintSpec spec_;
    JsonConstraint json_;
    Grammar schema_grammar_;      // GrammarConstraint points into this
    GrammarConstraint schema_;
    GbnfGrammar gbnf_grammar_;    // GbnfConstraint points into this
    GbnfConstraint gbnf_;
    Constraint* active_ = nullptr;
};