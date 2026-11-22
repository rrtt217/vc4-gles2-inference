#include "gbnf.h"

#include <cstdio>
#include <cstring>

namespace {
constexpr auto RANGE = Grammar::Elem::RANGE;
constexpr auto REF = Grammar::Elem::REF;
constexpr auto END = Grammar::Elem::END;
}  // namespace

// ---------------------------------------------------------------------------
// Rule building
// ---------------------------------------------------------------------------

uint16_t Grammar::add_rule() {
    rules.emplace_back();
    return (uint16_t)(rules.size() - 1);
}
void Grammar::set_alts(uint16_t r, std::vector<Alt> alts) { rules[r].alts = std::move(alts); }
Grammar::Elem Grammar::range(uint8_t lo, uint8_t hi) { Elem e; e.type = RANGE; e.lo = lo; e.hi = hi; return e; }
Grammar::Elem Grammar::ref(uint16_t r) { Elem e; e.type = REF; e.rule = r; return e; }
Grammar::Elem Grammar::end() { Elem e; e.type = END; return e; }
Grammar::Alt Grammar::seq(std::vector<Elem> elems) { Alt a; a.elems = std::move(elems); return a; }

// nullable[r]: some alternative of r matches the empty string.  A fixpoint over
// the rule graph; the compiler emits no left recursion, so it converges fast.
void Grammar::prepare() {
    nullable_.assign(rules.size(), false);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t r = 0; r < rules.size(); ++r) {
            if (nullable_[r]) continue;
            for (const Alt& a : rules[r].alts) {
                if (alt_nullable(a)) { nullable_[r] = true; changed = true; break; }
            }
        }
    }
}

bool Grammar::alt_nullable(const Alt& a) const {
    for (const Elem& e : a.elems) {
        if (e.type != REF || !nullable_[e.rule]) return false;
    }
    return true;   // empty alt, or all-nullable refs
}

// Can this alternative begin with byte c?  Walk its leading elements: a range
// decides immediately; a nullable ref may be skipped to reach the next element.
bool Grammar::alt_starts(const Alt& a, uint16_t c, int depth) const {
    if (depth > 64) return false;   // guard; compiler grammars are far shallower
    for (const Elem& e : a.elems) {
        if (e.type == RANGE) return c >= e.lo && c <= e.hi;
        if (e.type == END) return c == kEnd;
        // REF: does the target have a non-empty alt starting with c?
        for (const Alt& sub : rules[e.rule].alts) {
            if (!alt_nullable(sub) && alt_starts(sub, c, depth + 1)) return true;
        }
        if (!nullable_[e.rule]) return false;   // ref cannot be empty: decided
        // nullable ref: it may match nothing, so try the next element
    }
    return false;   // empty alt starts with nothing (it is the fallback)
}

// Pick the alternative of `rule` to expand for byte c.  LL(1): at most one
// non-empty alt starts with c; a nullable alt is the catch-all fallback.
int Grammar::choose_alt(uint16_t rule, uint16_t c) const {
    int fallback = -1;
    const Rule& r = rules[rule];
    for (size_t a = 0; a < r.alts.size(); ++a) {
        if (alt_nullable(r.alts[a])) { if (fallback < 0) fallback = (int)a; continue; }
        if (alt_starts(r.alts[a], c, 0)) return (int)a;
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

Grammar::State Grammar::start_state() const {
    State s;
    s.n = 1;
    s.f[0] = {start, 0, 0};
    return s;
}

bool Grammar::step(State& s, uint16_t c) const {
    while (s.n > 0) {
        Frame& f = s.f[s.n - 1];
        const Rule& r = rules[f.rule];
        if (f.elem >= r.alts[f.alt].elems.size()) { --s.n; continue; }   // alt done
        const Elem& e = r.alts[f.alt].elems[f.elem];
        if (e.type == REF) {
            ++f.elem;                       // parent advances past the ref first
            int a = choose_alt(e.rule, c);
            if (a < 0) return false;
            if (s.n >= kMaxDepth) return false;
            s.f[s.n++] = {e.rule, (uint16_t)a, 0};
            continue;                       // re-process c against the child
        }
        if (e.type == END) {
            if (c == kEnd) { --s.n; continue; }
            return false;                   // end-of-input is not a real byte
        }
        if (c >= e.lo && c <= e.hi) { ++f.elem; return true; }
        return false;
    }
    return c == kEnd;   // stack empty: only the sentinel is accepted
}

bool Grammar::done(const State& s) const {
    State t = s;
    return step(t, kEnd);
}

bool Grammar::accepts(const State& from, const std::string& bytes, State& out) const {
    out.n = from.n;
    memcpy(out.f, from.f, (size_t)from.n * sizeof(Frame));
    for (unsigned char c : bytes) if (!step(out, c)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// GrammarConstraint: the token table + mask, identical contract to JsonConstraint
// ---------------------------------------------------------------------------

void GrammarConstraint::init(const Tokenizer& tok, int vocab,
                             const std::vector<int>& eos_ids, const Grammar& g) {
    grammar_ = &g;
    bytes_.assign((size_t)vocab, std::string());
    for (int i = 0; i < vocab; ++i) {
        std::string s = tok.id_text(i);
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            unsigned v = 0;
            if (sscanf(s.c_str(), "<0x%02X>", &v) == 1) { bytes_[(size_t)i] = std::string(1, (char)v); continue; }
        }
        bytes_[(size_t)i] = s;
    }
    eos_.clear();
    for (int e : eos_ids) if (e >= 0) eos_.push_back(e);
    reset();
}

void GrammarConstraint::reset() { state_ = grammar_->start_state(); }

bool GrammarConstraint::is_eos(int id) const {
    for (int e : eos_) if (e == id) return true;
    return false;
}

bool GrammarConstraint::done() const { return grammar_->done(state_); }

bool GrammarConstraint::allows(int id) const {
    if (id < 0 || id >= (int)bytes_.size()) return false;
    if (is_eos(id)) return done();
    const std::string& b = bytes_[(size_t)id];
    if (b.empty()) return false;
    Grammar::State out;
    return grammar_->accepts(state_, b, out);
}

void GrammarConstraint::mask(std::vector<float>& logits) const {
    for (size_t i = 0; i < logits.size(); ++i)
        if (!allows((int)i)) logits[i] = -1e30f;
}

void GrammarConstraint::accept(int token) {
    if (is_eos(token)) return;
    Grammar::State out;
    if (grammar_->accepts(state_, bytes_[(size_t)token], out)) state_ = out;
}

int GrammarConstraint::allowed_count() const {
    int n = 0;
    for (size_t i = 0; i < bytes_.size(); ++i) if (allows((int)i)) ++n;
    return n;
}
