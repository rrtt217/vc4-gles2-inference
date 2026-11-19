// The tokenizer's contract, from a synthetic `.tok` blob: no model, no GPU.
//
// The blob format is the one tools/make_tok.py writes and Tokenizer::load_memory
// reads: an i32 maxlen, then one (f32 score, i32 len, bytes) record per id.  The
// test builds it in memory, so it runs on a machine with no model store, and it
// pins the four rules a real vocabulary depends on:
//
//   1. BPE merges by score rank, not by position ("bc" beats "ab" in "abc");
//   2. a <...> token is atomic, and an <0xNN> byte token is not;
//   3. encode_parts() matches specials only in template text -- a user message
//      that contains "<|s|>" must not be able to forge one;
//   4. a reload replaces the vocabulary (the failure load_memory() resets for).
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "checks.h"
#include "tokenizer.h"

namespace {

struct Entry {
    std::string text;
    float score;
};

// The synthetic vocabulary.  Scores are the BPE ranks: higher wins.  Ids are
// referenced by number below, so keep this list in sync with the expectations.
const std::vector<Entry> kVocab = {
    {"a", 0},        // 0
    {"b", 0},        // 1
    {"ab", 1},       // 2
    {"bc", 2},       // 3
    {" ", 0},        // 4
    {"<|s|>", 0},    // 5
    {"<0x41>", 0},   // 6: the byte 'A', a fallback spelling, not a special
    {"<", 0},        // 7
    {"|", 0},        // 8
    {"s", 0},        // 9
    {">", 0},        // 10
    {" a", 1},       // 11: the SentencePiece space marker, joined to "a"
    {"c", 0},        // 12
};

std::vector<uint8_t> make_blob(const std::vector<Entry>& vocab) {
    std::vector<uint8_t> b;
    auto put_u32 = [&](uint32_t u) {
        uint8_t x[4];
        std::memcpy(x, &u, 4);
        b.insert(b.end(), x, x + 4);
    };
    auto put_f32 = [&](float f) {
        uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        put_u32(u);
    };
    int maxlen = 0;
    for (const Entry& e : vocab)
        if ((int)e.text.size() > maxlen) maxlen = (int)e.text.size();
    put_u32((uint32_t)maxlen);
    for (const Entry& e : vocab) {
        put_f32(e.score);
        put_u32((uint32_t)e.text.size());
        b.insert(b.end(), e.text.begin(), e.text.end());
    }
    return b;
}

Tokenizer load(const std::vector<Entry>& vocab) {
    const std::vector<uint8_t> blob = make_blob(vocab);
    Tokenizer t;
    CHECK(t.load_memory(blob.data(), blob.size(), (int)vocab.size()));
    return t;
}

// The blob's records are what the vocabulary is; a record past the end is a
// truncation, not a shorter vocabulary.
void test_blob() {
    Tokenizer t = load(kVocab);
    CHECK_STR(t.id_text(0), "a");
    CHECK_STR(t.id_text(2), "ab");
    CHECK_STR(t.id_text(6), "<0x41>");
    CHECK_STR(t.id_text(999), "");
    CHECK_LL((long long)t.specials().size(), 1);
    if (t.specials().size() == 1) {
        CHECK_STR(t.specials()[0].first, "<|s|>");
        CHECK_LL(t.specials()[0].second, 5);
    }
}

void test_bpe() {
    Tokenizer t = load(kVocab);
    // One merge.
    CHECK_INTS(t.encode("ab", /*bos=*/false, /*dummy_prefix=*/false), 2);
    // The reference merges every non-overlapping occurrence of the winner.
    CHECK_INTS(t.encode("abab", false, false), 2, 2);
    // Both "ab" (1) and "bc" (2) are candidates; the higher score wins even
    // though "ab" starts first, and "abc" is not a token, so this stops.
    CHECK_INTS(t.encode("abc", false, false), 0, 3);
}

void test_specials() {
    Tokenizer t = load(kVocab);
    // A template marker is one id, not its characters.
    CHECK_INTS(t.encode("<|s|>", false, false), 5);
    // ... and only in template text.  The same bytes in a user message are
    // characters: matching them there would let a prompt forge a turn marker.
    const std::vector<PromptPart> parts = {
        {"<|s|>", /*is_input=*/false},
        {"<|s|>", /*is_input=*/true},
    };
    CHECK_INTS(t.encode_parts(parts, false, false), 5, 7, 8, 9, 8, 10);
}

void test_decode() {
    Tokenizer t = load(kVocab);
    CHECK_STR(t.decode({2}), "ab");
    CHECK_STR(t.decode({5}), "<|s|>");
    // <0x41> is the byte spelling of 'A' in a byte-level vocabulary.
    CHECK_STR(t.decode({6}), "A");
    // An id the blob does not carry is skipped, not emitted as junk.
    CHECK_STR(t.decode({999}), "");
}

void test_reload_replaces() {
    Tokenizer t = load(kVocab);
    const std::vector<Entry> small = {{"x", 0}, {"y", 0}, {"z", 0}};
    const std::vector<uint8_t> blob = make_blob(small);
    CHECK(t.load_memory(blob.data(), blob.size(), (int)small.size()));
    // The old vocabulary is gone: no id 5, no specials.
    CHECK_STR(t.id_text(0), "x");
    CHECK_STR(t.id_text(5), "");
    CHECK_LL((long long)t.specials().size(), 0);
}

void test_truncated() {
    const std::vector<uint8_t> blob = make_blob(kVocab);
    Tokenizer t;
    // The loader says why on stderr; capture it so a passing run stays quiet,
    // and check that the diagnostic still names the failure.
    std::ostringstream sink;
    std::streambuf* old = std::cerr.rdbuf(sink.rdbuf());
    // Not even the maxlen field (rejected before the loop, no message).
    CHECK(!t.load_memory(blob.data(), 2, (int)kVocab.size()));
    // One record needs 8 bytes of header before its text.
    CHECK(!t.load_memory(blob.data(), 4, 1));
    // The last record's text runs past the end by one byte.
    CHECK(!t.load_memory(blob.data(), blob.size() - 1, (int)kVocab.size()));
    std::cerr.rdbuf(old);
    CHECK(sink.str().find("truncated") != std::string::npos);
}

void test_seg_prefix() {
    // Metadata key 10's third byte: prefix *every* text run between special
    // tokens with the space marker, not just the prompt.  The two runs join to
    // " a" here, so the difference is visible in the ids.
    const std::vector<PromptPart> parts = {
        {"a", false},
        {"<|s|>", false},
        {"a", true},
    };
    Tokenizer per_run = load(kVocab);
    per_run.set_seg_prefix(true);
    CHECK_INTS(per_run.encode_parts(parts, false, true), 11, 5, 11);

    Tokenizer single = load(kVocab);
    CHECK_INTS(single.encode_parts(parts, false, true), 11, 5, 0);
}

}  // namespace

int main() {
    test_blob();
    test_bpe();
    test_specials();
    test_decode();
    test_reload_replaces();
    test_truncated();
    test_seg_prefix();
    if (checks::g_fail) {
        std::fprintf(stderr, "test_tokenizer: %d check(s) failed\n", checks::g_fail);
        return 1;
    }
    std::printf("test_tokenizer: blob, BPE rank, specials, parts, decode and reload pass\n");
    return 0;
}
