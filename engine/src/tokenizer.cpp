#include "tokenizer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include <string>
#include <unordered_map>
#include <vector>

bool Tokenizer::load_memory(const void* data, size_t bytes, int n){
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
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open tokenizer: " << path << "\n"; return false; }
        int maxlen = 0; f.read((char*)&maxlen, 4);
        for (int i = 0; i < n; ++i) {
            float score = 0; int len = 0;
            f.read((char*)&score, 4); f.read((char*)&len, 4);
            string s(len, '\0'); f.read(&s[0], len);
            vocab_.push_back(s); scores_.push_back(score); lookup_[s] = i;
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

void Tokenizer::encode_into(const string& text, bool allow_specials, vector<int>& t) const {
        for (size_t i = 0; i < text.size();) {
            bool matched = false;
            if (allow_specials) {
                for (auto& sp : specials_) {
                    if (text.compare(i, sp.first.size(), sp.first) == 0) {
                        t.push_back(sp.second); i += sp.first.size(); matched = true; break;
                    }
                }
            }
            if (matched) continue;
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
        // GPT-2/SmolLM/Qwen style tokenizers fold the leading space into the
        // first word (" capital" is one token), so it must be prepended to the
        // text and not emitted as its own token.  TOK_SPACE_FOLD=1 selects that;
        // the default keeps the historical behaviour the MiniMind2/3 gates were
        // baselined on.
        static const bool fold = getenv("TOK_SPACE_FOLD") && atoi(getenv("TOK_SPACE_FOLD")) != 0;
        // HF's tokenizers.encode() does not prepend a space to a raw string, so
        // the engine must be able to skip the prefix entirely (TOK_NO_PREFIX=1).
        static const bool no_prefix = getenv("TOK_NO_PREFIX") && atoi(getenv("TOK_NO_PREFIX")) != 0;
        if (no_prefix) dummy_prefix = false;
        string body = text;
        if (dummy_prefix && !text.empty()) {
            if (fold) body = " " + text;
            else {
                auto it = lookup_.find(" ");
                if (it != lookup_.end()) t.push_back(it->second);
            }
        }
        encode_into(body, /*allow_specials=*/true, t);
        bpe_merge(t);
        return t;
    }

vector<int> Tokenizer::encode_parts(const vector<PromptPart>& parts, bool bos, bool dummy_prefix) {
        vector<int> t;
        if (bos && bos_id_ >= 0) t.push_back(bos_id_);
        bool nonempty = false;
        for (auto& p : parts) if (!p.text.empty()) { nonempty = true; break; }
        if (dummy_prefix && nonempty) {
            auto it = lookup_.find(" ");
            if (it != lookup_.end()) t.push_back(it->second);
        }
        for (auto& p : parts) {
            if (p.text.empty()) continue;
            // Caller text never becomes a special token.
            encode_into(p.text, /*allow_specials=*/!p.is_input, t);
        }
        bpe_merge(t);
        return t;
    }

string Tokenizer::decode(const vector<int>& toks, int prev ){
        string out;
        for (int id : toks) {
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
