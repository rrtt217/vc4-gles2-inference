#pragma once

#include "chattmpl.h"
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
    // Encode a rendered prompt.  Special-token matching is enabled only for
    // template text: a user message that contains "<|im_end|>" is tokenised as
    // ordinary text, so it cannot forge a turn boundary.  The BPE merge pass
    // still runs once over the whole sequence, which is why a single template
    // part tokenises exactly like encode().
    vector<int> encode_parts(const vector<PromptPart>& parts, bool bos = true,
                             bool dummy_prefix = true);

    string decode(const vector<int>& toks, int prev = 0);

private:
    void encode_into(const string& text, bool allow_specials, vector<int>& out) const;
    void bpe_merge(vector<int>& t) const;
    vector<string> vocab_;
    vector<float> scores_;
    unordered_map<string, int> lookup_;
    vector<pair<string, int>> specials_;
public:
    // Ids of the atomic <...> tokens this blob carries.  Generation should not
    // be able to spell one of these as text: a model that starts emitting
    // "<|user|>" has left its own format, and every token after that is noise.
    const vector<pair<string, int>>& specials() const { return specials_; }
private:
    bool qwen_ = false;
    bool bytelevel_ = false;
    int bos_id_ = 1;
};
