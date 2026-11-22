#include "gbnf_full.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

namespace {
inline bool is_end(const GbnfElem* pos) {
    return pos->type == GbnfElem::END || pos->type == GbnfElem::ALT;
}
}  // namespace

// ---------------------------------------------------------------------------
// General matcher: a set of pushdown stacks (NFA subset construction)
// ---------------------------------------------------------------------------

// Does byte c satisfy the char range at pos (positive or inverse)?  Walks the
// CHAR / CHAR_ALT / RNG_UPPER chain that encodes [a-z0-9_] or [^...].
std::pair<bool, const GbnfElem*> GbnfGrammar::match_char(const GbnfElem* pos, uint8_t c) {
    bool found = false;
    const bool positive = pos->type == GbnfElem::CHAR || pos->type == GbnfElem::CHAR_ANY;
    do {
        if (pos[1].type == GbnfElem::RNG_UPPER) {
            found = found || (pos->value <= c && c <= pos[1].value);
            pos += 2;
        } else if (pos->type == GbnfElem::CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || (pos->value == c);
            pos += 1;
        }
    } while (pos->type == GbnfElem::CHAR_ALT);
    return {found == positive, pos};
}

// Expand a stack until every resulting stack ends at a terminal (or is empty).
// A RULE_REF forks into one stack per alternative; the `seen` set dedups so
// nullable / epsilon cycles terminate instead of looping.
void GbnfGrammar::advance(const Stack& stack, Stacks& out) const {
    std::vector<Stack> todo;
    todo.push_back(stack);
    auto cmp = [](const Stack& a, const Stack& b) {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
            [](const GbnfElem* pa, const GbnfElem* pb) { return pa < pb; });
    };
    std::set<Stack, decltype(cmp)> seen(cmp);

    while (!todo.empty()) {
        Stack curr = std::move(todo.back());
        todo.pop_back();
        if (!seen.insert(curr).second) continue;

        if (curr.empty()) {
            if (std::find(out.begin(), out.end(), curr) == out.end()) out.push_back(curr);
            continue;
        }
        const GbnfElem* pos = curr.back();
        switch (pos->type) {
        case GbnfElem::RULE_REF: {
            const GbnfElem* subpos = rules[pos->value].data();
            do {
                Stack next(curr.begin(), curr.end() - 1);
                if (!is_end(pos + 1)) next.push_back(pos + 1);   // rest of parent
                if (!is_end(subpos)) next.push_back(subpos);     // this alternative
                todo.push_back(std::move(next));
                while (!is_end(subpos)) ++subpos;                // scan to its end
                if (subpos->type == GbnfElem::ALT) ++subpos;     // next alternative
                else break;
            } while (true);
            break;
        }
        case GbnfElem::CHAR:
        case GbnfElem::CHAR_NOT:
        case GbnfElem::CHAR_ANY:
            if (std::find(out.begin(), out.end(), curr) == out.end()) out.push_back(curr);
            break;
        default:
            break;   // stacks are never left on END/ALT/range modifiers
        }
    }
}

void GbnfGrammar::accept(Stacks& s, uint8_t c) const {
    Stacks out;
    out.reserve(s.size());
    for (const Stack& stack : s) {
        if (stack.empty()) continue;
        const GbnfElem* pos = stack.back();
        auto m = match_char(pos, c);
        if (m.first) {
            Stack next(stack.begin(), stack.end() - 1);
            if (!is_end(m.second)) next.push_back(m.second);
            advance(next, out);
        }
    }
    s = std::move(out);
}

bool GbnfGrammar::accepts(const Stacks& from, const std::string& bytes, Stacks& out) const {
    out = from;
    for (unsigned char c : bytes) {
        accept(out, c);
        if (out.empty()) return false;
    }
    return true;
}

bool GbnfGrammar::done(const Stacks& s) const {
    for (const Stack& st : s) if (st.empty()) return true;
    return false;
}

GbnfGrammar::Stacks GbnfGrammar::start_stacks() const {
    // Fork every alternative of the start rule (advance_stack only forks rule
    // *references*, so the root's own alternatives must be split here).
    Stacks out;
    const GbnfElem* pos = rules[start].data();
    do {
        Stack stack;
        if (!is_end(pos)) stack.push_back(pos);
        advance(stack, out);
        while (!is_end(pos)) ++pos;
        if (pos->type == GbnfElem::ALT) ++pos;
        else break;
    } while (true);
    return out;
}

// ---------------------------------------------------------------------------
// GBNF text parser (byte-based; Unicode escapes refused for a later step)
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kMaxRepetition = 1000;
constexpr uint32_t kMaxDepth = 100;

inline bool is_digit_char(char c) { return c >= '0' && c <= '9'; }
inline bool is_word_char(char c) { return isalnum((unsigned char)c) || c == '-' || c == '_'; }

const char* parse_space(const char* src, bool newline_ok) {
    const char* pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') { while (*pos && *pos != '\r' && *pos != '\n') ++pos; }
        else ++pos;
    }
    return pos;
}
const char* parse_name(const char* src) {
    const char* pos = src;
    while (is_word_char(*pos)) ++pos;
    if (pos == src) throw std::runtime_error(std::string("expecting name at ") + src);
    return pos;
}
const char* parse_int(const char* src) {
    const char* pos = src;
    while (is_digit_char(*pos)) ++pos;
    if (pos == src) throw std::runtime_error(std::string("expecting integer at ") + src);
    return pos;
}
std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
    uint32_t v = 0;
    for (int i = 0; i < size; ++i) {
        char c = src[i];
        uint32_t d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else throw std::runtime_error(std::string("invalid hex at ") + src);
        v = (v << 4) | d;
    }
    return {v, src + size};
}
// One byte of a literal or range.  \u / \U are code points: refused until the
// Unicode step rather than mis-matched as a single byte.
std::pair<uint8_t, const char*> parse_char(const char* src) {
    if (*src == '\\') {
        switch (src[1]) {
        case 'x': { auto p = parse_hex(src + 2, 2); return {(uint8_t)p.first, p.second}; }
        case 'u': case 'U':
            throw std::runtime_error("unicode escapes (\\u/\\U) are not supported yet");
        case 't': return {'\t', src + 2};
        case 'r': return {'\r', src + 2};
        case 'n': return {'\n', src + 2};
        case '\\': case '"': case '[': case ']': case '-':
            return {(uint8_t)src[1], src + 2};
        default: throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    }
    if (*src) return {(uint8_t)*src, src + 1};
    throw std::runtime_error("unexpected end of input");
}

class GbnfParser {
public:
    std::vector<GbnfRule> rules;
    std::map<std::string, uint32_t> symbol_ids;

    uint32_t get_symbol_id(const char* src, size_t len) {
        std::string name(src, len);
        auto it = symbol_ids.find(name);
        if (it != symbol_ids.end()) return it->second;
        uint32_t id = (uint32_t)rules.size();
        symbol_ids[name] = id;
        rules.emplace_back();
        return id;
    }
    uint32_t generate_symbol_id(const std::string& base) {
        std::string name;
        do { name = base + "_" + std::to_string(gen_counter_++); } while (symbol_ids.count(name));
        uint32_t id = (uint32_t)rules.size();
        symbol_ids[name] = id;
        rules.emplace_back();
        return id;
    }
    void add_rule(uint32_t id, const GbnfRule& r) { rules[id] = r; }

    const char* parse_alternates(const char* src, const std::string& name, uint32_t id,
                                 bool nested, uint32_t depth) {
        GbnfRule rule;
        const char* pos = parse_sequence(src, name, rule, nested, depth);
        while (*pos == '|') {
            rule.push_back({GbnfElem::ALT, 0});
            pos = parse_space(pos + 1, nested);
            pos = parse_sequence(pos, name, rule, nested, depth);
        }
        rule.push_back({GbnfElem::END, 0});
        add_rule(id, rule);
        return pos;
    }

    const char* parse_sequence(const char* src, const std::string& name, GbnfRule& rule,
                               bool nested, uint32_t depth) {
        size_t last_sym_start = rule.size();
        const char* pos = src;
        uint64_t n_prev_rules = 1;

        auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
            bool no_max = max_times == UINT64_MAX;
            if (last_sym_start == rule.size())
                throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
            GbnfRule prev_rule(rule.begin() + last_sym_start, rule.end());
            uint64_t total_rules = (!no_max && max_times > 0) ? max_times : (min_times > 0 ? min_times : 1);
            if (n_prev_rules * total_rules > kMaxRepetition)
                throw std::runtime_error("repetition exceeds sane limits");
            if (min_times == 0) rule.resize(last_sym_start);
            else for (uint64_t i = 1; i < min_times; ++i) rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            uint64_t n_opt = no_max ? 1 : max_times - min_times;
            GbnfRule rec_rule(prev_rule);
            uint32_t last_rec_rule_id = 0;
            for (uint64_t i = 0; i < n_opt; ++i) {
                rec_rule.resize(prev_rule.size());
                uint32_t rec_rule_id = generate_symbol_id(name);
                if (i > 0 || no_max)
                    rec_rule.push_back({GbnfElem::RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
                rec_rule.push_back({GbnfElem::ALT, 0});
                rec_rule.push_back({GbnfElem::END, 0});
                add_rule(rec_rule_id, rec_rule);
                last_rec_rule_id = rec_rule_id;
            }
            if (n_opt > 0) rule.push_back({GbnfElem::RULE_REF, last_rec_rule_id});
            n_prev_rules *= total_rules;
        };

        while (*pos) {
            if (*pos == '"') {
                ++pos; last_sym_start = rule.size(); n_prev_rules = 1;
                while (*pos != '"') {
                    if (!*pos) throw std::runtime_error("unexpected end of input");
                    auto cp = parse_char(pos); pos = cp.second;
                    rule.push_back({GbnfElem::CHAR, cp.first});
                }
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '[') {
                ++pos;
                uint8_t start_type = GbnfElem::CHAR;
                if (*pos == '^') { ++pos; start_type = GbnfElem::CHAR_NOT; }
                last_sym_start = rule.size(); n_prev_rules = 1;
                while (*pos != ']') {
                    if (!*pos) throw std::runtime_error("unexpected end of input");
                    auto cp = parse_char(pos); pos = cp.second;
                    uint8_t type = last_sym_start < rule.size() ? (uint8_t)GbnfElem::CHAR_ALT : start_type;
                    rule.push_back({type, cp.first});
                    if (pos[0] == '-' && pos[1] != ']') {
                        if (!pos[1]) throw std::runtime_error("unexpected end of input");
                        auto ec = parse_char(pos + 1); pos = ec.second;
                        rule.push_back({GbnfElem::RNG_UPPER, ec.first});
                    }
                }
                pos = parse_space(pos + 1, nested);
            } else if (is_word_char(*pos)) {
                const char* name_end = parse_name(pos);
                uint32_t ref = get_symbol_id(pos, name_end - pos);
                pos = parse_space(name_end, nested);
                last_sym_start = rule.size(); n_prev_rules = 1;
                rule.push_back({GbnfElem::RULE_REF, ref});
            } else if (*pos == '(') {
                if (depth >= kMaxDepth) throw std::runtime_error("grammar nesting too deep");
                pos = parse_space(pos + 1, true);
                uint32_t n_before = (uint32_t)symbol_ids.size();
                uint32_t sub = generate_symbol_id(name);
                pos = parse_alternates(pos, name, sub, true, depth + 1);
                n_prev_rules = std::max<uint64_t>(1, (uint32_t)symbol_ids.size() - n_before);
                last_sym_start = rule.size();
                rule.push_back({GbnfElem::RULE_REF, sub});
                if (*pos != ')') throw std::runtime_error(std::string("expecting ')' at ") + pos);
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '.') {
                last_sym_start = rule.size(); n_prev_rules = 1;
                rule.push_back({GbnfElem::CHAR_ANY, 0});
                pos = parse_space(pos + 1, nested);
            } else if (*pos == '*') { pos = parse_space(pos + 1, nested); handle_repetitions(0, UINT64_MAX); }
            else if (*pos == '+') { pos = parse_space(pos + 1, nested); handle_repetitions(1, UINT64_MAX); }
            else if (*pos == '?') { pos = parse_space(pos + 1, nested); handle_repetitions(0, 1); }
            else if (*pos == '{') {
                pos = parse_space(pos + 1, nested);
                if (!is_digit_char(*pos)) throw std::runtime_error(std::string("expecting int at ") + pos);
                const char* ie = parse_int(pos);
                uint64_t mn = std::stoull(std::string(pos, ie - pos));
                pos = parse_space(ie, nested);
                uint64_t mx = UINT64_MAX;
                if (*pos == '}') { mx = mn; pos = parse_space(pos + 1, nested); }
                else if (*pos == ',') {
                    pos = parse_space(pos + 1, nested);
                    if (is_digit_char(*pos)) { ie = parse_int(pos); mx = std::stoull(std::string(pos, ie - pos)); pos = parse_space(ie, nested); }
                    if (*pos != '}') throw std::runtime_error(std::string("expecting '}' at ") + pos);
                    pos = parse_space(pos + 1, nested);
                } else throw std::runtime_error(std::string("expecting ',' at ") + pos);
                if (mn > kMaxRepetition) throw std::runtime_error("repetition too large");
                if (mx != UINT64_MAX && mx > kMaxRepetition) mx = UINT64_MAX;
                handle_repetitions(mn, mx);
            } else break;
        }
        return pos;
    }

    const char* parse_rule(const char* src) {
        const char* name_end = parse_name(src);
        const char* pos = parse_space(name_end, false);
        uint32_t id = get_symbol_id(src, name_end - src);
        std::string name(src, name_end - src);
        if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '='))
            throw std::runtime_error(std::string("expecting ::= at ") + pos);
        pos = parse_space(pos + 3, true);
        pos = parse_alternates(pos, name, id, false, 0);
        if (*pos == '\r') pos += pos[1] == '\n' ? 2 : 1;
        else if (*pos == '\n') ++pos;
        else if (*pos) throw std::runtime_error(std::string("expecting newline or end at ") + pos);
        return parse_space(pos, true);
    }

    bool parse(const char* src, std::string& err) {
        try {
            const char* pos = parse_space(src, true);
            while (*pos) pos = parse_rule(pos);
            for (const auto& rule : rules) {
                if (rule.empty()) throw std::runtime_error("undefined rule");
                for (const auto& e : rule)
                    if (e.type == GbnfElem::RULE_REF && (e.value >= rules.size() || rules[e.value].empty()))
                        throw std::runtime_error("undefined rule identifier");
            }
        } catch (const std::exception& e) {
            err = e.what();
            rules.clear();
            return false;
        }
        return true;
    }

private:
    uint32_t gen_counter_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// GbnfGrammar::parse + left-recursion guard
// ---------------------------------------------------------------------------

bool GbnfGrammar::parse(const std::string& gbnf, std::string& err) {
    GbnfParser p;
    if (!p.parse(gbnf.c_str(), err)) return false;
    rules = std::move(p.rules);
    auto it = p.symbol_ids.find("root");
    if (it == p.symbol_ids.end()) { err = "grammar must define a 'root' rule"; return false; }
    start = it->second;
    if (has_left_recursion(err)) return false;
    return true;
}

// GBNF forbids left recursion (it would loop advance_stack forever).  Detect it
// the way llama.cpp does: walk leftmost nonterminals, skipping past nullable
// rules, and flag a rule already in progress.
bool GbnfGrammar::has_left_recursion(std::string& err) const {
    std::vector<bool> in_progress(rules.size(), false), may_be_empty(rules.size(), false);
    std::function<bool(size_t, uint32_t)> detect = [&](size_t ri, uint32_t depth) -> bool {
        if (in_progress[ri]) return true;
        if (depth > kMaxDepth) { err = "grammar rules chained too deep"; return true; }
        in_progress[ri] = true;
        const GbnfRule& rule = rules[ri];
        bool at_start = true;
        for (size_t i = 0; i < rule.size(); ++i) {
            if (is_end(&rule[i])) { if (at_start) { may_be_empty[ri] = true; break; } at_start = true; }
            else at_start = false;
        }
        bool recurse = true;
        for (size_t i = 0; i < rule.size(); ++i) {
            if (rule[i].type == GbnfElem::RULE_REF && recurse) {
                if (detect(rule[i].value, depth + 1)) return true;
                if (!may_be_empty[rule[i].value]) recurse = false;
            } else if (is_end(&rule[i])) recurse = true;
            else recurse = false;
        }
        in_progress[ri] = false;
        return false;
    };
    if (detect(start, 0)) { if (err.empty()) err = "left recursion is not supported"; return true; }
    return false;
}

// ---------------------------------------------------------------------------
// GbnfConstraint: token table + mask, same contract as the other tiers
// ---------------------------------------------------------------------------

void GbnfConstraint::init(const Tokenizer& tok, int vocab, const std::vector<int>& eos_ids,
                          const GbnfGrammar& g) {
    g_ = &g;
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

void GbnfConstraint::reset() { stacks_ = g_->start_stacks(); }

bool GbnfConstraint::is_eos(int id) const {
    for (int e : eos_) if (e == id) return true;
    return false;
}

bool GbnfConstraint::done() const { return g_->done(stacks_); }

bool GbnfConstraint::allows(int id) const {
    if (id < 0 || id >= (int)bytes_.size()) return false;
    if (is_eos(id)) return done();
    const std::string& b = bytes_[(size_t)id];
    if (b.empty()) return false;
    GbnfGrammar::Stacks out;
    return g_->accepts(stacks_, b, out);
}

void GbnfConstraint::mask(std::vector<float>& logits) const {
    for (size_t i = 0; i < logits.size(); ++i)
        if (!allows((int)i)) logits[i] = -1e30f;
}

void GbnfConstraint::accept(int token) {
    if (is_eos(token)) return;
    GbnfGrammar::Stacks out;
    if (g_->accepts(stacks_, bytes_[(size_t)token], out)) stacks_ = std::move(out);
}
