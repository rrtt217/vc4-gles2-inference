#include "tokenizer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include <string>
#include <unordered_map>
#include <vector>

bool Tokenizer::load_memory(const void* data, size_t bytes, int n){
        reset();
        const uint8_t* p = (const uint8_t*)data;
        const uint8_t* end = p + bytes;
        int maxlen = 0;
        if (bytes < 4) return false;
        memcpy(&maxlen, p, 4); p += 4;
        auto need = [&](size_t k) { return p + k <= end; };
        for (int i = 0; i < n; ++i) {
            if (!need(8)) { cerr << "tokenizer blob truncated\n"; return false; }
            float score = 0; int len = 0;
            memcpy(&score, p, 4); memcpy(&len, p + 4, 4); p += 8;
            if (len < 0 || !need((size_t)len)) { cerr << "tokenizer blob truncated\n"; return false; }
            string s((const char*)p, (size_t)len); p += len;
            vocab_.push_back(s); scores_.push_back(score); lookup_[s] = i;
            if (s.size() > max_len_) max_len_ = s.size();
            if (!s.empty() && s.size() > first_max_[(unsigned char)s[0]])
                first_max_[(unsigned char)s[0]] = s.size();
        }
        for (int i = 0; i < (int)vocab_.size(); ++i) {
            const string& s = vocab_[i];
            if (s.size() > 2 && s.front() == '<' && s.back() == '>' && s.compare(0, 3, "<0x") != 0)
                specials_.push_back({s, i});
        }
        sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });
        return true;
    }

bool Tokenizer::load(const string& path, int n){
        reset();
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open tokenizer: " << path << "\n"; return false; }
        int maxlen = 0; f.read((char*)&maxlen, 4);
        for (int i = 0; i < n; ++i) {
            float score = 0; int len = 0;
            f.read((char*)&score, 4); f.read((char*)&len, 4);
            string s(len, '\0'); f.read(&s[0], len);
            vocab_.push_back(s); scores_.push_back(score); lookup_[s] = i;
            if (s.size() > max_len_) max_len_ = s.size();
            if (!s.empty() && s.size() > first_max_[(unsigned char)s[0]])
                first_max_[(unsigned char)s[0]] = s.size();
        }
        for (int i = 0; i < (int)vocab_.size(); ++i) {
            const string& s = vocab_[i];
            // Chat/tool/thinking markers are often added tokens rather than
            // BPE merges (e.g. <think>).  Treat any non-hex <...> token as
            // atomic so the byte-level BPE cannot split it.
            if (s.size() > 2 && s.front() == '<' && s.back() == '>' && s.compare(0, 3, "<0x") != 0)
                specials_.push_back({s, i});
        }
        sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });
        return true;
    }

void Tokenizer::set_qwen(bool q){ qwen_ = q; }

void Tokenizer::set_bytelevel(bool b){ bytelevel_ = b; }

bool Tokenizer::is_special_id(int id) const{
        for (auto& sp : specials_) if (sp.second == id) return true;
        return false;
}

void Tokenizer::set_rwkv(bool r){ rwkv_ = r; }

void Tokenizer::encode_into(const string& text, bool allow_specials, vector<int>& t,
                           bool* run_start) const {
        // The marker is pushed as its own token and the merge pass joins it to
        // the word that follows (the blob spells the word as " user"), which is
        // exactly what writing " " + text would do -- without copying the text
        // once per run.
        bool pending = run_start ? *run_start : false;
        for (size_t i = 0; i < text.size();) {
            bool matched = false;
            if (allow_specials) {
                for (auto& sp : specials_) {
                    if (text.compare(i, sp.first.size(), sp.first) == 0) {
                        t.push_back(sp.second); i += sp.first.size(); matched = true; break;
                    }
                }
            }
            if (matched) {
                // A special token ends the text run: whatever follows starts a
                // new one and gets its own marker.
                if (run_start) *run_start = true;
                pending = run_start != nullptr;
                continue;
            }
            if (pending) {
                auto sp = lookup_.find(" ");
                if (sp != lookup_.end()) t.push_back(sp->second);
                pending = false;
            }
            if (run_start) *run_start = false;
            unsigned char c = text[i];
            if (bytelevel_) {
                string bs(1, (char)c);
                auto bt = lookup_.find(bs);
                t.push_back(bt != lookup_.end() ? bt->second : (int)c + 3);
                ++i;
                continue;
            }
            int len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > text.size()) len = 1;
            string s = text.substr(i, len); i += len;
            auto it = lookup_.find(s);
            if (it != lookup_.end()) t.push_back(it->second);
            else {
                for (unsigned char b : s) {
                    string bs(1, (char)b);
                    auto bt = lookup_.find(bs);
                    t.push_back(bt != lookup_.end() ? bt->second : (int)b + 3);
                }
            }
        }
}

// RWKV World tokenisation is one rule: at every position take the longest
// vocabulary entry that fits in what is left.  There is no merge pass, so the
// scores in the blob are unused -- and an all-zero-score blob fed to
// bpe_merge() is not a no-op, it merges by insertion order (0.0 beats the
// -1e30 initial best), which is why RWKV mode must bypass that call.
//
// The reference (rwkv/utils.py RWKV_TOKENIZER.encodeBytes, a 256-way TRIE)
// matches over raw bytes, so this does too: a match may end mid-UTF-8-
// character, exactly like the reference.  When no entry of any length matches,
// fall back the way the byte-level path does (one UTF-8 character, then single
// bytes) so a hole in the table costs that byte instead of the rest of the
// string.  The RWKV blob carries all 256 single bytes, so that path is a
// guard, not a routine case.
void Tokenizer::encode_greedy(const string& text, vector<int>& t) const {
        static bool warned = false;
        const size_t n = text.size();
        size_t i = 0;
        while (i < n) {
            // No entry starts with this byte -> nothing can match here; skip
            // straight to the fallback instead of 128 failed lookups.
            size_t len = max_len_ < n - i ? max_len_ : n - i;
            size_t cap = first_max_[(unsigned char)text[i]];
            if (cap < len) len = cap;
            size_t hit = (size_t)-1;
            int id = -1;
            for (; len >= 1; --len) {
                auto it = lookup_.find(text.substr(i, len));
                if (it != lookup_.end()) { id = it->second; hit = len; break; }
            }
            if (id >= 0) { t.push_back(id); i += hit; continue; }
            // Nothing matched.  The downward scan above may have stopped short
            // of a UTF-8 character's width (first_max_ caps it), so try the
            // whole character once, then give up on this byte.
            size_t clen = 1;
            unsigned char c = (unsigned char)text[i];
            if ((c & 0xE0) == 0xC0) clen = 2;
            else if ((c & 0xF0) == 0xE0) clen = 3;
            else if ((c & 0xF8) == 0xF0) clen = 4;
            if (i + clen > n) clen = 1;
            auto it = lookup_.find(text.substr(i, clen));
            if (it != lookup_.end()) { t.push_back(it->second); i += clen; continue; }
            if (!warned) {
                warned = true;
                cerr << "tokenizer: byte 0x" << hex << (int)c << dec
                     << " is not in the RWKV vocabulary; skipping it\n";
            }
            ++i;
        }
}

// SentencePiece unigram segmentation: the piece sequence with the largest sum
// of log probabilities.  dp[i] is the best score for text[i..], so the pass runs
// backwards and the emit pass forwards.
//
// The tie rule is part of the contract: the reference walks the lattice in
// piece-length order and keeps the first maximum, so a strict `>` here keeps the
// shorter piece at equal score, which is the same choice.  `>` rather than `>=`
// is therefore deliberate, and a Python Viterbi that uses it agrees token for
// token (engine/scripts/check_unigram.py is the oracle).
//
// A blob built from BPE merges stores score = 1e6 - rank, which is monotone in
// merge order, so this loop on such a blob would segment by rank instead of by
// likelihood.  That is why the container says which kind it is (metadata key
// 10's fourth byte) and why this path is opt-in.
void Tokenizer::encode_unigram(const string& text, vector<int>& out) const {
        const size_t n = text.size();
        if (!n) return;
        const float NEG = -3.0e38f;
        vector<float> dp(n + 1, NEG);
        vector<int> pick(n + 1, -1);
        vector<int> plen(n + 1, 0);
        dp[n] = 0.0f;
        for (size_t i = n; i-- > 0;) {
            size_t cap = max_len_ < n - i ? max_len_ : n - i;
            const size_t first = first_max_[(unsigned char)text[i]];
            if (first < cap) cap = first;
            for (size_t L = 1; L <= cap; ++L) {
                auto it = lookup_.find(text.substr(i, L));
                if (it == lookup_.end()) continue;
                if (dp[i + L] <= NEG) continue;
                const float v = scores_[it->second] + dp[i + L];
                if (v > dp[i]) { dp[i] = v; pick[i] = it->second; plen[i] = L; }
            }
            // A hole in the table would leave dp[i] at NEG and silently drop
            // the rest of the string; cost this byte instead (the greedy path's
            // guard).  Both spm blobs carry every single byte, so this is a
            // guard, not a routine case.
            if (pick[i] < 0 && dp[i + 1] > NEG) { dp[i] = dp[i + 1]; plen[i] = 0; }
        }
        for (size_t i = 0; i < n;) {
            if (plen[i] == 0) { ++i; continue; }
            out.push_back(pick[i]);
            i += (size_t)plen[i];
        }
}

void Tokenizer::encode_into_unigram(const string& text, bool allow_specials,
                                    vector<int>& out, bool lead, bool per_run) const {
        string seg;
        auto flush = [&](bool marker) {
            if (seg.empty()) return;
            // The space marker belongs to the first *piece*: "▁the" is one
            // entry, and Viterbi can only see the marker if it is part of the
            // string being segmented.  Pushing it as its own token -- which is
            // what the BPE path does, relying on the merge pass to put it back
            // -- would make every marker-carrying piece unreachable.
            encode_unigram(marker ? (" " + seg) : seg, out);
            seg.clear();
        };
        for (size_t i = 0; i < text.size();) {
            bool matched = false;
            if (allow_specials) {
                for (auto& sp : specials_) {
                    if (text.compare(i, sp.first.size(), sp.first) == 0) {
                        flush(lead);
                        out.push_back(sp.second);
                        i += sp.first.size();
                        matched = true;
                        lead = per_run;
                        break;
                    }
                }
            }
            if (matched) continue;
            seg.push_back(text[i++]);
        }
        flush(lead);
}

void Tokenizer::bpe_merge(vector<int>& t) const {
        while (true) {
            float best = -1e30f; int bid = -1, ba = -1, bb = -1;
            for (size_t i = 0; i + 1 < t.size(); ++i) {
                string s = vocab_[t[i]] + vocab_[t[i + 1]];
                auto it = lookup_.find(s);
                if (it != lookup_.end() && scores_[it->second] > best) {
                    best = scores_[it->second]; bid = it->second;
                    ba = t[i]; bb = t[i + 1];
                }
            }
            if (bid < 0) break;
            // Merge every non-overlapping occurrence of the winning pair, as
            // the byte-level BPE reference does.
            for (size_t i = 0; i + 1 < t.size(); ) {
                if (t[i] == ba && t[i + 1] == bb) {
                    t[i] = bid;
                    t.erase(t.begin() + i + 1);
                } else {
                    ++i;
                }
            }
        }
}

vector<int> Tokenizer::encode(const string& text, bool bos , bool dummy_prefix ){
        vector<int> t;
        if (bos && bos_id_ >= 0) t.push_back(bos_id_);
        // RWKV: no dummy prefix, no space fold, no special-token table and no
        // merge pass.  The reference tokenizer sees the raw string and nothing
        // else; the vocabulary already carries the <|...|> markers as ordinary
        // entries, so matching them first could only ever shorten a match.
        if (rwkv_) { encode_greedy(text, t); return t; }
        if (unigram_) {
            encode_into_unigram(text, /*allow_specials=*/true, t,
                                /*lead=*/dummy_prefix && !text.empty(),
                                /*per_run=*/seg_prefix_ && dummy_prefix);
            return t;
        }
        // GPT-2/SmolLM/Qwen style tokenizers fold the leading space into the
        // first word (" capital" is one token), so it must be prepended to the
        // text and not emitted as its own token.  TOK_SPACE_FOLD=1 selects that;
        // the default keeps the historical behaviour the MiniMind2/3 gates were
        // baselined on.
        static const bool fold = env_flag("TOK_SPACE_FOLD");
        // HF's tokenizers.encode() does not prepend a space to a raw string, so
        // the engine must be able to skip the prefix entirely (TOK_NO_PREFIX=1).
        static const bool no_prefix = env_flag("TOK_NO_PREFIX");
        if (no_prefix) dummy_prefix = false;
        string body = text;
        const bool per_run = seg_prefix_ && dummy_prefix && !bytelevel_;
        if (dummy_prefix && !text.empty() && !per_run) {
            if (fold) body = " " + text;
            else {
                auto it = lookup_.find(" ");
                if (it != lookup_.end()) t.push_back(it->second);
            }
        }
        bool at_run = per_run;
        encode_into(body, /*allow_specials=*/true, t, per_run ? &at_run : nullptr);
        bpe_merge(t);
        return t;
    }

vector<int> Tokenizer::encode_parts(const vector<PromptPart>& parts, bool bos, bool dummy_prefix) {
        vector<int> t;
        if (bos && bos_id_ >= 0) t.push_back(bos_id_);
        // RWKV mode joins the parts first: they are contiguous slices of the
        // rendered prompt, and greedy longest match is not boundary-
        // independent, so encoding piece by piece would lose a token that
        // spans a template/message seam.  Nothing is given up by joining --
        // this mode does not treat template text as special.
        if (rwkv_) {
            string all;
            for (auto& p : parts) all += p.text;
            encode_greedy(all, t);
            return t;
        }
        if (unigram_) {
            const bool per_run = seg_prefix_ && dummy_prefix;
            bool lead = dummy_prefix;
            for (auto& p : parts) {
                if (p.text.empty()) continue;
                encode_into_unigram(p.text, /*allow_specials=*/!p.is_input, t,
                                    lead, per_run);
                lead = per_run;
            }
            return t;
        }
        bool nonempty = false;
        for (auto& p : parts) if (!p.text.empty()) { nonempty = true; break; }
        // A SentencePiece vocabulary wants the marker on *every* text run: the
        // reference normalizer prefixes each split separately, so the text after
        // a template marker starts a word ("\u2581user") and dropping the marker
        // there tokenises a word-continuation instead (measured on BitCPM:
        // "user" 3060 vs 1836, assistant 16434 vs 43686, logits cosine 0.912).
        // The container says which convention it wants (metadata key 10's third
        // byte), so a byte-level model keeps the single-marker behaviour its
        // gates were baselined with; on those the two produce the same tokens.
        const bool per_run = seg_prefix_ && dummy_prefix && !bytelevel_;
        if (dummy_prefix && nonempty && !per_run) {
            auto it = lookup_.find(" ");
            if (it != lookup_.end()) t.push_back(it->second);
        }
        bool at_run = per_run;
        for (auto& p : parts) {
            if (p.text.empty()) continue;
            // Caller text never becomes a special token.
            encode_into(p.text, /*allow_specials=*/!p.is_input, t, per_run ? &at_run : nullptr);
        }
        bpe_merge(t);
        return t;
    }

string Tokenizer::id_text(int id) const{
    if (id < 0 || id >= (int)vocab_.size()) return string();
    return vocab_[(size_t)id];
}

string Tokenizer::decode(const vector<int>& toks, int prev ){
        string out;
        for (int id : toks) {
            if (id < 0 || id >= (int)vocab_.size()) continue;
            // RWKV entries are raw bytes: emit them as they are.  There is no
            // <0xNN> convention to decode (that is the byte-level BPE
            // fallback spelling) and no leading-space fold.
            if (rwkv_) { out += vocab_[(size_t)id]; prev = id; continue; }
            string p = vocab_[id];
            if (!qwen_ && prev == 1 && !p.empty() && p[0] == ' ') p = p.substr(1);
            if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
                unsigned v = 0;
                if (sscanf(p.c_str(), "<0x%02X>", &v) == 1) {
                    out.push_back((char)v); prev = id; continue;
                }
            }
            out += p; prev = id;
        }
        return out;
    }
