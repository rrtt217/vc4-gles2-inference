#pragma once

#include "common.h"
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    bool load(const string& path, int n);
    // MMCF stores the same blob inside the container; n is the vocab size.
    bool load_memory(const void* data, size_t bytes, int n);
    void set_bos(int id) { bos_id_ = id; }
    void set_qwen(bool q);
    void set_bytelevel(bool b);

    vector<int> encode(const string& text, bool bos = true, bool dummy_prefix = true);

    string decode(const vector<int>& toks, int prev = 0);

private:
    vector<string> vocab_;
    vector<float> scores_;
    unordered_map<string, int> lookup_;
    vector<pair<string, int>> specials_;
    bool qwen_ = false;
    bool bytelevel_ = false;
    int bos_id_ = 1;
};
