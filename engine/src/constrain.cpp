#include "constrain.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "jinja/json.h"
#include "schema.h"

namespace {

bool read_text_file(const std::string& path, std::string& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    if (out.find_first_not_of(" \t\r\n") == std::string::npos) {
        err = path + " is empty";
        return false;
    }
    return true;
}

// One choice literal -> the body of a GBNF "..." literal.  `\` and `"` are
// escaped, and a control byte takes the escape the parser defines.  Everything
// else, UTF-8 included, goes through verbatim: parse_char() decodes a literal's
// bytes as UTF-8 code points.
void append_literal(const std::string& s, std::string& out) {
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') {
            out += '\\';
            out += (char) c;
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02X", (unsigned) c);
            out += b;
        } else {
            out += (char) c;
        }
    }
}

}  // namespace

bool choice_grammar(const std::string& list, std::string& gbnf, std::string& err) {
    std::vector<std::string> items;
    size_t start = 0;
    for (;;) {
        const size_t bar = list.find('|', start);
        items.push_back(list.substr(start, bar == std::string::npos ? std::string::npos : bar - start));
        if (bar == std::string::npos) break;
        start = bar + 1;
    }
    gbnf = "root ::= ";
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].empty()) {
            err = "choice has an empty item (items are separated by '|')";
            return false;
        }
        if (i) gbnf += " | ";
        gbnf += '"';
        append_literal(items[i], gbnf);
        gbnf += '"';
    }
    return true;
}

bool constraint_spec_parse(const std::string& text, ConstraintSpec& out, std::string& err) {
    const size_t sp = text.find_first_of(" \t");
    const std::string kind = text.substr(0, sp);
    std::string arg;
    if (sp != std::string::npos) {
        const size_t p = text.find_first_not_of(" \t", sp);
        if (p != std::string::npos) arg = text.substr(p);
    }
    out = ConstraintSpec{};
    if (kind == "off" || kind == "none") {
        if (!arg.empty()) {
            err = std::string("/constrain ") + kind + " takes no argument";
            return false;
        }
        out.kind = ConstraintSpec::None;
        return true;
    }
    if (kind == "json" || kind == "json_object") {
        if (!arg.empty()) {
            err = "/constrain " + kind + " takes no argument";
            return false;
        }
        out.kind = ConstraintSpec::JsonObject;
        return true;
    }
    if (kind == "schema" || kind == "gbnf") {
        if (arg.empty()) {
            err = "/constrain " + kind + " needs a path";
            return false;
        }
        out.kind = kind == "schema" ? ConstraintSpec::JsonSchema : ConstraintSpec::Gbnf;
        out.arg = arg;
        return true;
    }
    if (kind == "choice") {
        if (arg.empty()) {
            err = "/constrain choice needs one or more literals separated by '|'";
            return false;
        }
        out.kind = ConstraintSpec::Choice;
        out.arg = arg;
        return true;
    }
    err = "unknown constraint \"" + kind +
          "\" (off, json, schema PATH, gbnf PATH, choice A|B|C)";
    return false;
}

std::string constraint_spec_describe(const ConstraintSpec& s) {
    switch (s.kind) {
        case ConstraintSpec::None: return "none";
        case ConstraintSpec::JsonObject: return "json_object";
        case ConstraintSpec::JsonSchema: return "schema " + s.arg;
        case ConstraintSpec::Gbnf: return "gbnf " + s.arg;
        case ConstraintSpec::Choice: return "choice " + s.arg;
    }
    return "none";
}

bool CliConstraint::build(const ConstraintSpec& spec, const Tokenizer& tok, int vocab,
                          const std::vector<int>& eos_ids, std::string& err) {
    // Everything that can fail is compiled into a local first; the members are
    // only overwritten once the new constraint is known good.
    switch (spec.kind) {
        case ConstraintSpec::None:
            clear();
            return true;
        case ConstraintSpec::JsonObject:
            json_.init(tok, vocab, eos_ids);
            active_ = &json_;
            break;
        case ConstraintSpec::JsonSchema: {
            std::string text;
            if (!read_text_file(spec.arg, text, err)) return false;
            Grammar g;
            std::string perr;
            try {
                const common_json js = common_json::parse(text);
                if (!compile_schema(js, g, perr)) {
                    err = spec.arg + ": " + perr;
                    return false;
                }
            } catch (const std::exception& e) {
                err = spec.arg + ": " + e.what();
                return false;
            }
            schema_grammar_ = std::move(g);
            schema_.init(tok, vocab, eos_ids, schema_grammar_);
            active_ = &schema_;
            break;
        }
        case ConstraintSpec::Gbnf:
        case ConstraintSpec::Choice: {
            std::string text;
            if (spec.kind == ConstraintSpec::Gbnf) {
                if (!read_text_file(spec.arg, text, err)) return false;
            } else if (!choice_grammar(spec.arg, text, err)) {
                return false;
            }
            GbnfGrammar g;
            std::string perr;
            if (!g.parse(text, perr)) {
                err = (spec.kind == ConstraintSpec::Gbnf ? spec.arg + ": " : std::string()) + perr;
                return false;
            }
            gbnf_grammar_ = std::move(g);
            gbnf_.init(tok, vocab, eos_ids, gbnf_grammar_);
            active_ = &gbnf_;
            break;
        }
    }
    spec_ = spec;
    return true;
}

void CliConstraint::clear() {
    active_ = nullptr;
    spec_ = ConstraintSpec{};
}

void CliConstraint::reset() {
    if (active_ != nullptr) active_->reset();
}