#pragma once

#include "chattmpl.h"
#include "common.h"
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    // Same contract as Model::reset(): a second load replaces the vocabulary
    // instead of appending to it (load_memory() pushes into vocab_, so a reload
    // without this would silently double the table).
    void reset() { *this = Tokenizer(); }
    bool load(const string& path, int n);
    // MMCF stores the same blob inside the container; n is the vocab size.
    bool load_memory(const void* data, size_t bytes, int n);
    void set_bos(int id) { bos_id_ = id; }
    void set_qwen(bool q);
    void set_bytelevel(bool b);
    // RWKV World vocabulary (rwkv_vocab_v20230424): greedy longest match over
    // the entries' raw bytes, no BPE merge pass, and decode emits the entry's
    // bytes verbatim.  Set it *after* load(), like set_qwen() -- reset()
    // rebuilds the object and a mode set before a reload is lost.
    void set_rwkv(bool r);
    bool rwkv() const { return rwkv_; }

    vector<int> encode(const string& text, bool bos = true, bool dummy_prefix = true);
    // Encode a rendered prompt.  Special-token matching is enabled only for
    // template text: a user message that contains "<|im_end|>" is tokenised as
    // ordinary text, so it cannot forge a turn boundary.  The BPE merge pass
    // still runs once over the whole sequence, which is why a single template
    // part tokenises exactly like encode().
    vector<int> encode_parts(const vector<PromptPart>& parts, bool bos = true,
                             bool dummy_prefix = true);

    string decode(const vector<int>& toks, int prev = 0);
    // The token's own text as the vocabulary spells it.  A chat template can
    // ask for it by name (LFM2.5's starts with {{ bos_token }}), and passing ""
    // there silently changes the prompt: the model then sees no BOS at all.
    string id_text(int id) const;

private:
    void encode_into(const string& text, bool allow_specials, vector<int>& out) const;
    void bpe_merge(vector<int>& t) const;
    // RWKV mode: longest entry that fits, at every position.
    void encode_greedy(const string& text, vector<int>& out) const;
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
    bool rwkv_ = false;
    // Longest entry in the blob.  Greedy matching scans downwards from
    // min(maxlen, bytes left); taken from the entries themselves, not from the
    // header's i32, so a stale header cannot hide a long entry.
    size_t max_len_ = 0;
    // Longest entry starting with each byte.  An entry that starts with
    // another byte cannot match at this position, so this only shortens the
    // downward scan -- it is exact, and 0 means "no entry starts here".
    size_t first_max_[256] = {};
    int bos_id_ = 1;
};
