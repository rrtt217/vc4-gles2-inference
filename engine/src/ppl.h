#pragma once
// `--ppl PATH`: perplexity of a text file (see ppl.cpp for the two rules).

#include <string>

class GPU;
class Model;
class Tokenizer;

int run_ppl(const std::string& path, Model& model, GPU& gpu, Tokenizer& tokenizer);
