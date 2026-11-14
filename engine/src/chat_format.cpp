// The three text-shaping helpers a chat turn needs: strip a tool-call block,
// separate a <think> block from the answer, and render tool calls for the template.
//
// They were capture-free lambdas inside `main()` (2026-11-14, P2-1 of the
// 2026-11-13 audit); as free functions they can be unit-tested and `main()` no
// longer carries the string handling.

#include "chat_format.h"

#include "tools.h"

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include "jinja/json.h"   // -isystem third_party

std::string strip_tool_calls(const string& text) {
    string out;
    size_t pos = 0;
    while (true) {
        size_t a = text.find("<tool_call>", pos);
        size_t la = text.find("<|tool_call_start|>", pos);
        if (la != string::npos && (a == string::npos || la < a)) {
            out += text.substr(pos, la - pos);
            size_t b = text.find("<|tool_call_end|>", la);
            if (b == string::npos) break;
            pos = b + strlen("<|tool_call_end|>");
            continue;
        }
        if (a == string::npos) { out += text.substr(pos); break; }
        out += text.substr(pos, a - pos);
        size_t b = text.find("</tool_call>", a);
        if (b == string::npos) break;
        pos = b + strlen("</tool_call>");
    }
    string trimmed;
    for (char c : out) if (!isspace((unsigned char)c)) { trimmed += c; }
    return trimmed.empty() ? string() : out;
}
// LFM2.5/similar: the answer may open with <think> ... </think>.  The
// reasoning goes into the message's reasoning field (the template renders
// it as reasoning_content when preserve_thinking is set) and the content
// the caller sees is the answer without it.
std::string take_thinking(const string& text, string& reasoning) {
    size_t a = text.find("<think>");
    if (a == string::npos) return text;
    size_t b = text.find("</think>", a);
    // No closing tag: the model opened a block and stopped.  Keeping the
    // text is the honest choice -- swallowing the tail would print an
    // empty answer for a truncated generation.
    if (b == string::npos) return text;
    size_t cs = a + 7;
    size_t ce = (b == string::npos) ? text.size() : b;
    reasoning = text.substr(cs, ce - cs);
    size_t s0 = reasoning.find_first_not_of(" \t\r\n");
    size_t s1 = reasoning.find_last_not_of(" \t\r\n");
    reasoning = (s0 == string::npos) ? string()
                                    : reasoning.substr(s0, s1 - s0 + 1);
    size_t cut = (b == string::npos) ? text.size() : b + 8;
    return text.substr(0, a) + text.substr(cut);
}
std::string tool_calls_json(const vector<ToolCall>& calls) {
    common_json out = common_json::array();
    for (const auto& c : calls) {
        // `arguments` is an object, not an OpenAI-style JSON string: the
        // LFM2.5 template reads `func_args.items()` and throws on a string
        // (the render aborted the process before this).  A model that emits
        // unparseable arguments therefore gets {} rather than raw text.
        common_json args = common_json::parse_no_throw(c.args);
        common_json call = common_json::object();
        call["type"] = string("function");
        call["function"]["name"] = c.name;
        call["function"]["arguments"] =
            args.is_discarded() || !args.is_object() ? common_json::object() : args;
        out.push_back(call);
    }
    return out.dump();
}
