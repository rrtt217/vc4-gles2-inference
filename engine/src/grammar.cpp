#include "grammar.h"

#include <cstdio>
#include <cstring>

// Character classes of the JSON grammar (RFC 8259).  Whitespace is exactly
// space / tab / LF / CR; everything else is dispatched by the automaton.
namespace {
inline bool is_ws(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
inline bool is_hex(uint8_t c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
}  // namespace

bool JsonConstraint::push(State& s, uint8_t kind, uint8_t phase) {
    if (s.n >= kMaxDepth) return false;
    Frame& f = s.f[s.n++];
    f.kind = kind; f.phase = phase; f.aux = 0; f.lit = 0;
    return true;
}

// A number may end after its integer, fraction or exponent digits; at those
// phases a byte that does not continue it is handed back to the parent.  After
// a lone '-', a '.', or an exponent sign the number is incomplete and such a
// byte is a hard error instead.
bool JsonConstraint::num_stop(uint8_t p) {
    return p == 2 || p == 3 || p == 5 || p == 8;
}

bool JsonConstraint::step(State& s, uint8_t c) {
    while (s.n > 0) {
        Frame& f = s.f[s.n - 1];
        switch (f.kind) {
        case ROOT_AFTER:
            // The top-level value is done: only whitespace may follow, then EOS
            // (which is not a byte and is handled by done()/mask()).
            return is_ws(c);

        case VALUE:
            if (is_ws(c)) return true;
            if (c == '{') { f.kind = OBJECT; f.phase = 0; return true; }
            if (c == '[') { f.kind = ARRAY; f.phase = 0; return true; }
            if (c == '"') { f.kind = STRING; f.phase = 0; f.aux = 0; return true; }
            if (c == '-') { f.kind = NUMBER; f.phase = 1; return true; }
            if (c == '0') { f.kind = NUMBER; f.phase = 2; return true; }
            if (c >= '1' && c <= '9') { f.kind = NUMBER; f.phase = 3; return true; }
            if (c == 't') { f.kind = LITERAL; f.lit = 0; f.aux = 1; return true; }
            if (c == 'f') { f.kind = LITERAL; f.lit = 1; f.aux = 1; return true; }
            if (c == 'n') { f.kind = LITERAL; f.lit = 2; f.aux = 1; return true; }
            return false;

        case OBJECT:
            if (is_ws(c)) return true;
            switch (f.phase) {
            case 0:   // after '{': a key string or the empty object
                if (c == '"') { if (!push(s, STRING, 0)) return false; f.phase = 1; return true; }
                if (c == '}') { --s.n; return true; }
                return false;
            case 1:   // after the key: ':' then a value
                if (c == ':') { if (!push(s, VALUE, 0)) return false; f.phase = 2; return true; }
                return false;
            case 2:   // after a member value: ',' or '}'
                if (c == ',') { f.phase = 3; return true; }
                if (c == '}') { --s.n; return true; }
                return false;
            case 3:   // after ',': the next key (no trailing comma)
                if (c == '"') { if (!push(s, STRING, 0)) return false; f.phase = 1; return true; }
                return false;
            }
            return false;

        case ARRAY:
            if (is_ws(c)) return true;
            switch (f.phase) {
            case 0:   // after '[': a value or the empty array
                if (c == ']') { --s.n; return true; }
                if (!push(s, VALUE, 0)) return false;
                f.phase = 1;
                continue;   // re-process c as the first byte of the value
            case 1:   // after an element: ',' or ']'
                if (c == ',') { f.phase = 2; return true; }
                if (c == ']') { --s.n; return true; }
                return false;
            case 2:   // after ',': a value (a ']' here is a trailing comma)
                if (!push(s, VALUE, 0)) return false;
                f.phase = 1;
                continue;
            }
            return false;

        case STRING:
            switch (f.phase) {
            case 0:   // body
                if (c == '"') { --s.n; return true; }
                if (c == '\\') { f.phase = 1; return true; }
                if (c < 0x20) return false;   // unescaped control character
                return true;                  // any other byte, incl. UTF-8 >= 0x80
            case 1:   // after a backslash
                if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                    c == 'n' || c == 'r' || c == 't') { f.phase = 0; return true; }
                if (c == 'u') { f.phase = 2; f.aux = 0; return true; }
                return false;
            case 2:   // \u followed by exactly four hex digits
                if (is_hex(c)) { if (++f.aux >= 4) { f.phase = 0; f.aux = 0; } return true; }
                return false;
            }
            return false;

        case NUMBER: {
            const uint8_t p = f.phase;
            if (p == 1) {   // after '-': a digit is required
                if (c == '0') { f.phase = 2; return true; }
                if (c >= '1' && c <= '9') { f.phase = 3; return true; }
                return false;
            }
            if (p == 4) {   // after '.': at least one fraction digit
                if (is_digit(c)) { f.phase = 5; return true; }
                return false;
            }
            if (p == 6) {   // after 'e'/'E': optional sign then a digit
                if (c == '+' || c == '-') { f.phase = 7; return true; }
                if (is_digit(c)) { f.phase = 8; return true; }
                return false;
            }
            if (p == 7) {   // after the exponent sign: a digit is required
                if (is_digit(c)) { f.phase = 8; return true; }
                return false;
            }
            // Valid-stop phases 2/3/5/8: continue, or hand c back to the parent.
            // Each part appears at most once and in order: more digits within the
            // int/frac/exp; a '.' starts the fraction only after the integer; an
            // 'e' starts the exponent only after the integer or fraction.  A second
            // '.' or 'e' is not a continuation -- the number ends and the parent
            // rejects the stray byte (this is what makes "1.2.3" / "1e2e3" invalid).
            if ((p == 3 || p == 5 || p == 8) && is_digit(c)) return true;
            if ((p == 2 || p == 3) && c == '.') { f.phase = 4; return true; }
            if ((p == 2 || p == 3 || p == 5) && (c == 'e' || c == 'E')) { f.phase = 6; return true; }
            --s.n;
            continue;   // number complete; the parent consumes c
        }

        case LITERAL: {
            const char* target = f.lit == 0 ? "true" : f.lit == 1 ? "false" : "null";
            if (c == (uint8_t)target[f.aux]) {
                ++f.aux;
                if (target[f.aux] == '\0') --s.n;
                return true;
            }
            return false;
        }
        }
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Public interface: the token table, the per-request state, and the mask.
// ---------------------------------------------------------------------------

void JsonConstraint::init(const Tokenizer& tok, int vocab, const std::vector<int>& eos_ids) {
    bytes_.assign((size_t)vocab, std::string());
    for (int i = 0; i < vocab; ++i) {
        std::string s = tok.id_text(i);
        // The legacy byte-fallback spelling "<0xNN>" is six characters of text
        // but one byte of output; decode() reads it the same way.  Every other
        // entry is already raw bytes in the blob (make_tok.py inverts the
        // ByteLevel / SentencePiece alphabets), so it is used verbatim.
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            unsigned v = 0;
            if (sscanf(s.c_str(), "<0x%02X>", &v) == 1) {
                bytes_[(size_t)i] = std::string(1, (char)v);
                continue;
            }
        }
        bytes_[(size_t)i] = s;
    }
    eos_.clear();
    for (int e : eos_ids) if (e >= 0) eos_.push_back(e);
    reset();
}

void JsonConstraint::reset() {
    state_.n = 0;
    push(state_, ROOT_AFTER, 0);   // bottom: what follows the top-level value
    push(state_, VALUE, 0);        // top: parse one value
}

bool JsonConstraint::run(const std::string& bytes, State& out) const {
    out.n = state_.n;
    memcpy(out.f, state_.f, (size_t)state_.n * sizeof(Frame));
    for (unsigned char c : bytes) if (!step(out, c)) return false;
    return true;
}

bool JsonConstraint::is_eos(int id) const {
    for (int e : eos_) if (e == id) return true;
    return false;
}

bool JsonConstraint::is_done(const State& s) const {
    State t;
    t.n = s.n;
    memcpy(t.f, s.f, (size_t)s.n * sizeof(Frame));
    // A finished top-level number lingers on the stack until its terminating
    // byte arrives; for the purpose of "may EOS follow" it is already complete.
    while (t.n > 0 && t.f[t.n - 1].kind == NUMBER && num_stop(t.f[t.n - 1].phase)) --t.n;
    return t.n > 0 && t.f[t.n - 1].kind == ROOT_AFTER;
}

bool JsonConstraint::done() const { return is_done(state_); }

bool JsonConstraint::allows(int id) const {
    if (id < 0 || id >= (int)bytes_.size()) return false;
    if (is_eos(id)) return done();
    const std::string& b = bytes_[(size_t)id];
    if (b.empty()) return false;   // an empty token cannot advance the parse
    State tmp;
    tmp.n = state_.n;
    memcpy(tmp.f, state_.f, (size_t)state_.n * sizeof(Frame));
    for (unsigned char c : b) if (!step(tmp, c)) return false;
    return true;
}

void JsonConstraint::mask(std::vector<float>& logits) const {
    for (size_t i = 0; i < logits.size(); ++i)
        if (!allows((int)i)) logits[i] = -1e30f;
}

void JsonConstraint::accept(int token) {
    if (is_eos(token)) return;   // the turn ends; the state no longer matters
    State out;
    if (run(bytes_[(size_t)token], out)) {
        state_.n = out.n;
        memcpy(state_.f, out.f, (size_t)out.n * sizeof(Frame));
    }
}

int JsonConstraint::allowed_count() const {
    int n = 0;
    for (size_t i = 0; i < bytes_.size(); ++i) if (allows((int)i)) ++n;
    return n;
}
