#include "chattmpl.h"

#include "jinja/json.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <stdexcept>

using namespace jinja;

namespace {

// JSON string escape.  The template context is built as JSON text so that
// `global_from_json` can mark message values as caller input in one place.
std::string jesc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

}  // namespace

struct ChatTemplate::Impl {
    bool ok = false;
    std::string error;
    program prog;
};

ChatTemplate::ChatTemplate(const std::string& src) : src_(src), impl_(new Impl) {
    try {
        lexer lx;
        auto tokens = lx.tokenize(src);
        impl_->prog = parse_from_tokens(tokens);
        impl_->ok = true;
    } catch (const std::exception& e) {
        impl_->ok = false;
        impl_->error = e.what();
    }
}

ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::ok() const { return impl_ && impl_->ok; }

const std::string& ChatTemplate::error() const {
    static const std::string empty;
    return impl_ ? impl_->error : empty;
}

std::vector<PromptPart> ChatTemplate::render(const std::vector<ChatMsg>& msgs,
                                             const std::string& tools,
                                             bool add_generation_prompt,
                                             bool open_thinking,
                                             const std::string& bos_token,
                                             const std::string& eos_token) const {
    if (!ok()) throw std::runtime_error("chat template did not compile: " + error());

    std::string msgs_json = "[";
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (i) msgs_json += ",";
        msgs_json += "{\"role\":\"" + jesc(msgs[i].role) + "\",\"content\":\"" +
                     jesc(msgs[i].content) + "\"";
        if (!msgs[i].reasoning.empty())
            msgs_json += ",\"reasoning_content\":\"" + jesc(msgs[i].reasoning) + "\"";
        if (!msgs[i].tool_calls.empty())
            msgs_json += ",\"tool_calls\":" + msgs[i].tool_calls;
        msgs_json += "}";
    }
    msgs_json += "]";

    const std::string tools_json = tools.empty() ? std::string("[]") : tools;
    const std::string ctx_json =
        "{\"messages\":" + msgs_json + ",\"tools\":" + tools_json +
        ",\"add_generation_prompt\":" + (add_generation_prompt ? "true" : "false") +
        ",\"open_thinking\":" + (open_thinking ? "true" : "false") +
        ",\"bos_token\":\"" + jesc(bos_token) + "\"" +
        ",\"eos_token\":\"" + jesc(eos_token) + "\"}";

    context ctx(src_);
    auto parsed = common_json::parse(ctx_json);
    // mark_input=true is the whole point: everything from the context (message
    // roles and contents, tool definitions) is caller text, never template text.
    global_from_json(ctx, parsed, /*mark_input=*/true);

    runtime rt(ctx);
    auto out = rt.execute(impl_->prog);

    std::vector<PromptPart> parts;
    for (auto& v : out->val_arr) {
        auto gathered = runtime::gather_string_parts(v);
        for (auto& p : gathered->val_str.parts) {
            if (p.val.empty()) continue;
            if (!parts.empty() && parts.back().is_input == p.is_input)
                parts.back().text += p.val;
            else
                parts.push_back(PromptPart{p.val, p.is_input});
        }
    }
    return parts;
}

std::string chat_tools_array(const std::string& tools) {
    size_t i = 0;
    while (i < tools.size() && isspace((unsigned char)tools[i])) ++i;
    if (i >= tools.size()) return "";
    if (tools[i] == '[') return tools;
    // One JSON object per line (minimind3/tools.json): wrap it into an array.
    std::string out = "[";
    bool first = true;
    size_t pos = 0;
    while (pos < tools.size()) {
        size_t eol = tools.find('\n', pos);
        if (eol == std::string::npos) eol = tools.size();
        std::string line = tools.substr(pos, eol - pos);
        pos = eol + 1;
        bool blank = true;
        for (char c : line) if (!isspace((unsigned char)c)) { blank = false; break; }
        if (blank) continue;
        if (!first) out += ",";
        out += line;
        first = false;
    }
    out += "]";
    return out;
}
