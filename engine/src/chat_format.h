#pragma once
// Text shaping for a chat turn (see chat_format.cpp).

#include <string>
#include <vector>

struct ToolCall;

// Remove a <tool_call>...</tool_call> (or <|tool_call_start|>...end) block; the
// answer is what remains, or empty when nothing else was produced.
std::string strip_tool_calls(const std::string& text);

// Split `<think>...</think>` off the front of an answer into `reasoning`.
std::string take_thinking(const std::string& text, std::string& reasoning);

// tool_calls array as the chat template expects it (arguments as an object).
std::string tool_calls_json(const std::vector<ToolCall>& calls);
