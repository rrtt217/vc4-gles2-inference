#pragma once

// Chat-template rendering (the container's `chat_template`, HuggingFace Jinja
// text) on top of the vendored llama.cpp Jinja engine.
//
// Why a wrapper instead of calling jinja:: directly: the engine only ever
// needs "messages + tools -> prompt", and the prompt has to come back split
// into template text and caller text, because a user message that happens to
// contain `<|im_end|>` must not become a special token.

#include <memory>
#include <string>
#include <vector>

struct ChatMsg {
    std::string role;          // system | user | assistant | tool
    std::string content;
    // Assistant tool calls as a JSON array text ("" when there are none), e.g.
    // [{"type":"function","function":{"name":"f","arguments":"{}"}}]
    std::string tool_calls;
    // Optional reasoning_content for templates that render it separately.
    std::string reasoning;
};

// A rendered prompt piece.  `is_input` is true when the text came from a
// message, so the tokenizer matches special tokens only outside those pieces.
struct PromptPart {
    std::string text;
    bool is_input = false;
};

class ChatTemplate {
public:
    // Compiles `src`.  On a syntax error the object is !ok() and `error()`
    // carries the engine's message, which includes the offending source line.
    explicit ChatTemplate(const std::string& src);
    ~ChatTemplate();
    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;

    bool ok() const;
    const std::string& error() const;

    // `tools` is a JSON array text ("" or "[]" for none).  Throws
    // std::runtime_error when the template fails at render time.
    std::vector<PromptPart> render(const std::vector<ChatMsg>& msgs,
                                   const std::string& tools,
                                   bool add_generation_prompt,
                                   bool open_thinking,
                                   const std::string& bos_token,
                                   const std::string& eos_token) const;

    const std::string& source() const { return src_; }

private:
    struct Impl;
    std::string src_;
    std::unique_ptr<Impl> impl_;
};

// Parse a worker request payload:
//   {"messages":[{"role":..,"content":..,"tool_calls":[..]}], "tools":[..]}
// Used by --worker when the server sends messages instead of a prompt string.
// `open_thinking` is set to -1 when the payload does not state it.
bool chat_parse_request(const std::string& json, std::vector<ChatMsg>& msgs,
                        std::string& tools, std::string& err, int& open_thinking);

// Normalise a `--tools` payload into a JSON array text: an array is passed
// through, a sequence of one-object-per-line is wrapped.  Returns "" on a
// payload that is neither.
std::string chat_tools_array(const std::string& tools);
