#include "schema.h"

#include <functional>
#include <map>
#include <set>
#include <utility>

using Elem = Grammar::Elem;
using Alt = Grammar::Alt;
using G = Grammar;

namespace {

// A literal string as a sequence of single-byte ranges.
Alt lit(const std::string& s) {
    std::vector<Elem> e;
    for (unsigned char c : s) e.push_back(G::range(c, c));
    return G::seq(std::move(e));
}

// The JSON primitives, built once into a grammar.  `value` is any JSON value
// (the json_object tier and the compiler's "no type" fallback).  Every rule is
// LL(1): repetition is factored as `head tail` with a nullable tail, never
// `A | A rest` (which would share a first byte).
struct Primitives {
    uint16_t ws = 0, str = 0, num = 0, integer = 0, boolean = 0, nullr = 0, value = 0;

    void build(Grammar& g) {
        auto rule = [&](std::vector<Alt> alts) {
            uint16_t r = g.add_rule();
            g.set_alts(r, std::move(alts));
            return r;
        };
        auto R = [](uint8_t lo, uint8_t hi) { return G::range(lo, hi); };
        auto ref = [](uint16_t r) { return G::ref(r); };
        Alt empty = G::seq({});

        // ws ::= wschar ws | eps ; wschar ::= [09-0a] | [0d] | [20]
        uint16_t wschar = rule({G::seq({R(0x09, 0x0a)}), G::seq({R(0x0d, 0x0d)}), G::seq({R(0x20, 0x20)})});
        ws = rule({G::seq({ref(wschar), ref(0)}), empty});   // patched below
        // (ws references itself; fix the self-ref now that ws has its id)
        g.rules[ws].alts[0].elems[1] = ref(ws);

        uint16_t hex = rule({G::seq({R('0', '9')}), G::seq({R('a', 'f')}), G::seq({R('A', 'F')})});
        uint16_t esc = rule({
            G::seq({R('"', '"')}), G::seq({R('\\', '\\')}), G::seq({R('/', '/')}),
            G::seq({R('b', 'b')}), G::seq({R('f', 'f')}), G::seq({R('n', 'n')}),
            G::seq({R('r', 'r')}), G::seq({R('t', 't')}),
            G::seq({R('u', 'u'), ref(hex), ref(hex), ref(hex), ref(hex)})});
        // char ::= [20-21] | [23-5b] | [5d-ff] | "\" escape   (not " \ or control)
        uint16_t chr = rule({G::seq({R(0x20, 0x21)}), G::seq({R(0x23, 0x5b)}),
                             G::seq({R(0x5d, 0xff)}), G::seq({R('\\', '\\'), ref(esc)})});
        uint16_t chars = rule({G::seq({ref(chr), ref(0)}), empty});
        g.rules[chars].alts[0].elems[1] = ref(chars);
        str = rule({G::seq({R('"', '"'), ref(chars), R('"', '"')})});

        uint16_t digits = rule({G::seq({R('0', '9'), ref(0)}), empty});
        g.rules[digits].alts[0].elems[1] = ref(digits);
        uint16_t intr = rule({G::seq({R('0', '0')}), G::seq({R('1', '9'), ref(digits)})});
        uint16_t frac = rule({G::seq({R('.', '.'), R('0', '9'), ref(digits)})});
        uint16_t optfrac = rule({G::seq({ref(frac)}), empty});
        uint16_t sign = rule({G::seq({R('+', '+')}), G::seq({R('-', '-')}), empty});
        uint16_t exp = rule({G::seq({R('e', 'e'), ref(sign), R('0', '9'), ref(digits)}),
                             G::seq({R('E', 'E'), ref(sign), R('0', '9'), ref(digits)})});
        uint16_t optexp = rule({G::seq({ref(exp)}), empty});
        uint16_t optminus = rule({G::seq({R('-', '-')}), empty});
        num = rule({G::seq({ref(optminus), ref(intr), ref(optfrac), ref(optexp)})});
        integer = rule({G::seq({ref(optminus), ref(intr)})});
        boolean = rule({lit("true"), lit("false")});
        nullr = rule({lit("null")});

        // any object / array (mutually recursive with value)
        uint16_t anyobj = g.add_rule();
        uint16_t anyarr = g.add_rule();
        value = rule({G::seq({ref(anyobj)}), G::seq({ref(anyarr)}), G::seq({ref(str)}),
                      G::seq({ref(num)}), lit("true"), lit("false"), lit("null")});
        uint16_t anypair = rule({G::seq({ref(str), ref(ws), R(':', ':'), ref(ws), ref(value)})});
        // tail ::= ws ("," ws members)?  -- ws is consumed BEFORE the nullable
        // dispatch, so a space then '}' picks the empty branch, not the comma one
        uint16_t memberscomma = rule({G::seq({R(',', ','), ref(ws), ref(0)}), empty});
        uint16_t memberstail = rule({G::seq({ref(ws), ref(memberscomma)})});
        uint16_t anymembers = rule({G::seq({ref(anypair), ref(memberstail)})});
        g.rules[memberscomma].alts[0].elems[2] = ref(anymembers);
        // factor the body so the two object shapes dispatch on '}' vs a key, not
        // both on '{' (which a deterministic matcher cannot tell apart)
        uint16_t anyobjbody = rule({G::seq({R('}', '}')}),
                                    G::seq({ref(anymembers), ref(ws), R('}', '}')})});
        g.set_alts(anyobj, {G::seq({R('{', '{'), ref(ws), ref(anyobjbody)})});
        uint16_t elementscomma = rule({G::seq({R(',', ','), ref(ws), ref(0)}), empty});
        uint16_t elementstail = rule({G::seq({ref(ws), ref(elementscomma)})});
        uint16_t anyelements = rule({G::seq({ref(value), ref(elementstail)})});
        g.rules[elementscomma].alts[0].elems[2] = ref(anyelements);
        uint16_t anyarrbody = rule({G::seq({R(']', ']')}),
                                    G::seq({ref(anyelements), ref(ws), R(']', ']')})});
        g.set_alts(anyarr, {G::seq({R('[', '['), ref(ws), ref(anyarrbody)})});
    }
};

// A trie node: edges by byte, and (if a string ends here) the continuation that
// follows it.  For property keys the continuation is `ws ":" ws value`; for a
// pure literal set (enum) it is empty.
struct TrieNode {
    std::map<uint8_t, int> children;
    bool terminal = false;
    std::vector<Elem> suffix;
};

}  // namespace

namespace {

class SchemaCompiler {
public:
    Grammar g;
    Primitives prim;
    std::string err;
    bool ok = true;

    SchemaCompiler() { prim.build(g); }

    uint16_t rule(std::vector<Alt> alts) {
        uint16_t r = g.add_rule();
        g.set_alts(r, std::move(alts));
        return r;
    }
    static Elem R(uint8_t lo, uint8_t hi) { return G::range(lo, hi); }
    static Elem ref(uint16_t r) { return G::ref(r); }

    // Trie-factor a set of (string, continuation) pairs into one LL(1) rule.
    uint16_t trie(const std::vector<std::pair<std::string, std::vector<Elem>>>& entries) {
        std::vector<TrieNode> nodes(1);
        for (const auto& en : entries) {
            int cur = 0;
            for (unsigned char c : en.first) {
                auto it = nodes[cur].children.find(c);
                if (it == nodes[cur].children.end()) {
                    int nx = (int)nodes.size();
                    nodes.emplace_back();
                    nodes[cur].children[c] = nx;
                    cur = nx;
                } else {
                    cur = it->second;
                }
            }
            nodes[cur].terminal = true;
            nodes[cur].suffix = en.second;
        }
        std::function<uint16_t(int)> conv = [&](int idx) -> uint16_t {
            uint16_t r = g.add_rule();
            std::vector<Alt> alts;
            for (const auto& ch : nodes[idx].children) {
                uint16_t cr = conv(ch.second);
                alts.push_back(G::seq({R(ch.first, ch.first), ref(cr)}));
            }
            if (nodes[idx].terminal) alts.push_back(G::seq(nodes[idx].suffix));
            g.set_alts(r, std::move(alts));
            return r;
        };
        return conv(0);
    }

    uint16_t visit(const common_json& s) {
        if (!ok) return prim.value;
        if (!s.is_object()) { err = "schema must be an object"; ok = false; return prim.value; }
        for (const char* k : {"$ref", "anyOf", "oneOf", "allOf", "not"}) {
            if (s.contains(k)) { err = std::string("unsupported schema keyword: ") + k; ok = false; return prim.value; }
        }
        if (s.contains("enum")) {
            const common_json& e = s.at("enum");
            if (!e.is_array() || e.empty()) { err = "enum must be a non-empty array"; ok = false; return prim.value; }
            std::vector<std::pair<std::string, std::vector<Elem>>> entries;
            for (size_t i = 0; i < e.size(); ++i) entries.push_back({e.at(i).dump(), {}});
            return trie(entries);
        }
        if (s.contains("type") && !s.at("type").is_string()) {
            err = "type must be a string (multiple types unsupported)"; ok = false; return prim.value;
        }
        std::string type = s.value("type", std::string(""));
        if (type == "object") return object_rule(s);
        if (type == "array") return array_rule(s);
        if (type == "string") return prim.str;
        if (type == "integer") return prim.integer;
        if (type == "number") return prim.num;
        if (type == "boolean") return prim.boolean;
        if (type == "null") return prim.nullr;
        if (type.empty()) return prim.value;
        err = "unsupported type: " + type; ok = false; return prim.value;
    }

    // Closed object (additionalProperties=false, the OpenAI structured-output
    // contract): required keys in a fixed order, then optional keys in any order.
    uint16_t object_rule(const common_json& s) {
        if (s.contains("additionalProperties")) {
            const common_json& ap = s.at("additionalProperties");
            if (!ap.is_boolean() || ap.get<bool>()) {
                err = "objects must set additionalProperties:false"; ok = false; return prim.value;
            }
        }
        std::set<std::string> req;
        if (s.contains("required") && s.at("required").is_array()) {
            const common_json& rq = s.at("required");
            for (size_t i = 0; i < rq.size(); ++i)
                if (rq.at(i).is_string()) req.insert((std::string)rq.at(i));
        }
        std::vector<std::string> req_names, opt_names;
        std::map<std::string, uint16_t> val_rule;
        if (s.contains("properties")) {
            if (!s.at("properties").is_object()) { err = "properties must be an object"; ok = false; return prim.value; }
            for (const auto& kv : s.at("properties").items()) {
                val_rule[kv.key()] = visit(kv.value());
                if (req.count(kv.key())) req_names.push_back(kv.key());
                else opt_names.push_back(kv.key());
            }
        }
        auto suffix = [&](const std::string& name) {
            return std::vector<Elem>{ref(prim.ws), R(':', ':'), ref(prim.ws), ref(val_rule[name])};
        };
        auto quoted = [](const std::string& name) { return common_json::make(name).dump(); };

        if (req_names.empty() && opt_names.empty())
            return rule({G::seq({R('{', '{'), ref(prim.ws), R('}', '}')})});

        // optional members: a trie of quoted keys, each continuing to its value
        uint16_t opt_member = 0;
        if (!opt_names.empty()) {
            std::vector<std::pair<std::string, std::vector<Elem>>> entries;
            for (const auto& n : opt_names) entries.push_back({quoted(n), suffix(n)});
            opt_member = trie(entries);
        }
        uint16_t opt_tail = g.add_rule();
        uint16_t opt_comma = g.add_rule();
        if (opt_member)
            g.set_alts(opt_comma, {G::seq({R(',', ','), ref(prim.ws), ref(opt_member), ref(opt_tail)}), G::seq({})});
        else
            g.set_alts(opt_comma, {G::seq({})});
        g.set_alts(opt_tail, {G::seq({ref(prim.ws), ref(opt_comma)})});

        uint16_t members = g.add_rule();
        if (!req_names.empty()) {
            std::vector<Elem> e;
            for (size_t i = 0; i < req_names.size(); ++i) {
                if (i) { e.push_back(ref(prim.ws)); e.push_back(R(',', ',')); e.push_back(ref(prim.ws)); }
                std::vector<Elem> kv = lit(quoted(req_names[i])).elems;
                for (const Elem& x : suffix(req_names[i])) kv.push_back(x);
                uint16_t kvr = rule({G::seq(kv)});
                e.push_back(ref(kvr));
            }
            e.push_back(ref(opt_tail));
            g.set_alts(members, {G::seq(e)});
        } else {
            g.set_alts(members, {G::seq({ref(opt_member), ref(opt_tail)})});
        }
        uint16_t body = rule({G::seq({R('}', '}')}), G::seq({ref(members), ref(prim.ws), R('}', '}')})});
        return rule({G::seq({R('{', '{'), ref(prim.ws), ref(body)})});
    }

    uint16_t array_rule(const common_json& s) {
        uint16_t item = prim.value;
        if (s.contains("items")) {
            const common_json& it = s.at("items");
            if (it.is_array()) { err = "tuple items not supported"; ok = false; return prim.value; }
            item = visit(it);
        }
        uint16_t arr = g.add_rule();
        uint16_t elements = g.add_rule();
        uint16_t tail = g.add_rule();
        uint16_t comma = g.add_rule();
        g.set_alts(comma, {G::seq({R(',', ','), ref(prim.ws), ref(elements)}), G::seq({})});
        g.set_alts(tail, {G::seq({ref(prim.ws), ref(comma)})});
        g.set_alts(elements, {G::seq({ref(item), ref(tail)})});
        uint16_t body = rule({G::seq({R(']', ']')}), G::seq({ref(elements), ref(prim.ws), R(']', ']')})});
        g.set_alts(arr, {G::seq({R('[', '['), ref(prim.ws), ref(body)})});
        return arr;
    }
};

}  // namespace

Grammar build_json_grammar() {
    SchemaCompiler c;
    uint16_t root = c.g.add_rule();
    c.g.set_alts(root, {G::seq({G::ref(c.prim.ws), G::ref(c.prim.value), G::ref(c.prim.ws), G::end()})});
    c.g.start = root;
    c.g.prepare();
    return c.g;
}

bool compile_schema(const common_json& schema, Grammar& out, std::string& err) {
    SchemaCompiler c;
    uint16_t vr = c.visit(schema);
    if (!c.ok) { err = c.err; return false; }
    uint16_t root = c.g.add_rule();
    c.g.set_alts(root, {G::seq({G::ref(c.prim.ws), G::ref(vr), G::ref(c.prim.ws), G::end()})});
    c.g.start = root;
    c.g.prepare();
    out = std::move(c.g);
    return true;
}
