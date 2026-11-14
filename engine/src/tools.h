#pragma once

#include "common.h"

struct ToolCall { string name, args; };

string json_escape(const string& s);
vector<ToolCall> parse_tool_calls(const string& text);
string base64_decode(const string& in);
size_t utf8_complete_prefix(const string& s);
string utf8_take(string& pending);
string run_tool(const string& name, const string& args);
