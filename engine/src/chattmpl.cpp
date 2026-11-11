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
    // Templates that render a reasoning field (LFM2.5) only keep it in history
    // when preserve_thinking is set; the engine sets it whenever a message
    // carries reasoning, and --thinking asks for it for the current turn.
    bool preserve_thinking = open_thinking;
    for (const auto& m : msgs)
        if (!m.reasoning.empty()) preserve_thinking = true;
    const std::string ctx_json =
        "{\"messages\":" + msgs_json + ",\"tools\":" + tools_json +
        ",\"add_generation_prompt\":" + (add_generation_prompt ? "true" : "false") +
        ",\"open_thinking\":" + (open_thinking ? "true" : "false") +
        ",\"preserve_thinking\":" + (preserve_thinking ? "true" : "false") +
        ",\"bos_token\":\"" + jesc(bos_token) + "\"" +
        ",\"eos_token\":\"" + jesc(eos_token) + "\"}";

    context ctx(src_);
    auto parsed = common_json::parse(ctx_json);
    // mark_input=true is the whole point: everything from the context (message
    // roles and contents, tool definitions) is caller text, never template text.
    global_from_json(ctx, parsed, /*mark_input=*/true);
    // ... but the special-token strings are *template* text, not caller text:
    // marked as input, `{{- bos_token -}}` would be tokenised as the literal
    // characters "<|startoftext|>" (4 tokens) instead of the BOS id.  The
    // template context is the only place this distinction is visible, so it is
    // made here rather than in the tokenizer.
    ctx.set_val("bos_token", mk_val<value_string>(bos_token));
    ctx.set_val("eos_token", mk_val<value_string>(eos_token));

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

bool chat_parse_request(const std::string& json, std::vector<ChatMsg>& msgs,
                        std::string& tools, std::string& err, int& open_thinking) {
    open_thinking = -1;
    try {
        auto root = common_json::parse(json);
        auto get_str = [](const common_json& o, const char* key) -> std::string {
            if (!o.contains(key)) return "";
            const common_json& v = o.at(key);
            if (!v.is_string()) return "";
            return static_cast<std::string>(v);
        };
        if (!root.contains("messages")) { err = "no messages"; return false; }
        const common_json& msv = root.at("messages");
        if (!msv.is_array()) { err = "messages is not an array"; return false; }
        for (common_json m : msv) {
            ChatMsg cm;
            cm.role = get_str(m, "role");
            cm.content = get_str(m, "content");
            if (cm.role.empty()) { err = "message without a role"; return false; }
            if (m.contains("tool_calls")) {
                const common_json& tc = m.at("tool_calls");
                if (tc.is_array() && tc.size() > 0) cm.tool_calls = tc.dump();
            }
            msgs.push_back(std::move(cm));
        }
        if (msgs.empty()) { err = "no messages"; return false; }
        if (root.contains("open_thinking")) {
            const common_json& ot = root.at("open_thinking");
            if (ot.is_boolean()) open_thinking = ot.get<bool>() ? 1 : 0;
        }
        if (root.contains("tools")) {
            const common_json& tv = root.at("tools");
            if (tv.is_array()) tools = tv.size() ? tv.dump() : "";
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
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
