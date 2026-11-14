#pragma once
// `--build-prefix-cache` (see prefix_build.cpp).

#include <string>

class ChatTemplate;
class Model;
class Session;
class Tokenizer;
struct Options;

int build_prefix_cache(Model& model, Tokenizer& tokenizer, const ChatTemplate* chat_tmpl,
                       const Options& opt, const std::string& tools_array, bool enc_bos,
                       const std::string& bos_str, const std::string& eos_str, bool quiet,
                       Session& session);
