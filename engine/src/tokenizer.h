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
    // Metadata key 10's third byte: prefix *every* text run between special
    // tokens with the space marker, not just the prompt.  That is what the
    // reference tokenizer does (its normalizer runs per split), and on a
    // SentencePiece vocabulary it changes tokens -- "user" vs "\u2581user" --
    // so a model trained that way sees a different prompt without it.
    void set_seg_prefix(bool b) { seg_prefix_ = b; }
    // RWKV World vocabulary (rwkv_vocab_v20230424): greedy longest match over
    // the entries' raw bytes, no BPE merge pass, and decode emits the entry's
    // bytes verbatim.  Set it *after* load(), like set_qwen() -- reset()
    // rebuilds the object and a mode set before a reload is lost.
    void set_rwkv(bool r);
    // SentencePiece **unigram** (metadata key 10's fourth byte).  The sentence
    // is the segmentation that maximises the sum of the pieces' log
    // probabilities, so the blob's score column is load-bearing -- which is what
    // `make_tok_spm.py` writes and what a BPE blob (score = 1e6 - rank) would
    // turn into a segmentation by merge order.
    void set_unigram(bool u) { unigram_ = u; }
    bool unigram() const { return unigram_; }
    bool rwkv() const { return rwkv_; }
    // A vocabulary that is a plain list of strings with no merges at all, so the
    // text is segmented by taking the *longest* entry that starts at each
    // position (metadata key 10's fifth byte).  RWKV's mode is the same
    // algorithm, which is why it reuses encode_greedy(), but this one keeps the
    // special-token table: MiniYuanYuan's prompt is literal
    // `<bos><user>…<sep><assistant>` and those markers have to become one id
    // each, where RWKV's vocabulary carries them as ordinary entries and
    // matching them first could only shorten a match.
    //
    // It also does not fold spaces into the next token: a vocabulary like this
    // states its whitespace as named entries (`<space>`, `<newline>`), and no
    // entry in it contains a space at all, so the BPE path's leading-space token
    // would be a token the vocabulary does not have.
    void set_greedy(bool g) { greedy_ = g; }
    bool greedy() const { return greedy_; }

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
    // `run_start` tracks the SentencePiece per-run space marker (seg_prefix_):
    // true means the next *text* character begins a run and needs the marker.
    // A special token ends the run, so the flag comes back true after it.  A
    // null pointer is the plain behaviour (no marker bookkeeping).
    void encode_into(const string& text, bool allow_specials, vector<int>& out,
                     bool* run_start = nullptr) const;
    void bpe_merge(vector<int>& t) const;
    // RWKV mode: longest entry that fits, at every position.
    void encode_greedy(const string& text, vector<int>& out) const;
    // Unigram mode: the maximum-score segmentation of one run, over raw bytes.
    void encode_unigram(const string& text, vector<int>& out) const;
    // Unigram mode with the special-token scan.  `lead` puts the space marker on
    // the first run, `per_run` on every run after a special.
    void encode_into_unigram(const string& text, bool allow_specials,
                             vector<int>& out, bool lead, bool per_run) const;
    bool is_special_id(int id) const;
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
    bool seg_prefix_ = false;
    bool rwkv_ = false;
    bool unigram_ = false;
    // Metadata key 10's fifth byte: longest-match segmentation over a flat
    // vocabulary, with the special-token table kept.
    bool greedy_ = false;
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
