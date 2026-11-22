#pragma once

// JSON Schema -> grammar, for the json_schema tier of constrained decoding.
//
// Supported subset: type (object/array/string/integer/number/boolean/null),
// properties + required + additionalProperties:false, items, enum, and nested
// schemas.  Annotation keywords (title/description/default/examples) and the
// numeric/string bounds (minimum/maxLength/pattern/format/...) are accepted but
// NOT enforced -- the grammar guarantees shape and types, not every facet.
// Structural combinators ($ref/anyOf/oneOf/allOf/not) are refused by name rather
// than approximated, because a wrong approximation is worse than a clear error.
//
// The emitted grammar is LL(1): type dispatch is by first byte and every string
// set (property keys, enum literals) is trie-factored, so gbnf.cpp's deterministic
// matcher never faces an ambiguous choice.  Required properties are emitted in a
// fixed order (the reference converter does the same); optional ones may appear
// in any order.  server/README.md ("Structured output") carries the user contract.

#include <string>

#include "gbnf.h"
#include "jinja/json.h"

// The plain-JSON grammar (any valid JSON value), used by the json_object tier
// when it runs through the generic matcher, and as the compiler's "any" fallback.
Grammar build_json_grammar();

// Compile `schema` into `out`.  Returns false and sets `err` on an unsupported
// keyword or a malformed schema; `out` is left empty in that case.
bool compile_schema(const common_json& schema, Grammar& out, std::string& err);
