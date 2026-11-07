#!/usr/bin/env python3
"""Convert a HuggingFace tokenizer.json (ByteLevel BPE) to llama2.c tokenizer.bin.

The engine's tokenizer stores raw byte strings.  Scores are used as merge
priorities: the engine picks the highest score, so rank 0 gets the largest
score.  Special/added tokens beginning with '<' are handled by the C++ side.
"""
import argparse, json, struct

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    j = json.load(open(a.input))
    model = j["model"]
    vocab = model["vocab"]
    if isinstance(vocab, list):
        items = list(enumerate(vocab))
    else:
        items = sorted(vocab.items(), key=lambda kv: kv[1])

    # Added tokens (the chat markers) live *above* the BPE vocabulary in some
    # tokenizers - supra2-medium puts <|im_start|> at 16384 with a 16384-token
    # BPE, so a blob built from model["vocab"] alone leaves the engine unable to
    # recognise them and it splits them into characters instead.  Lay the added
    # tokens onto their ids and fill any hole with a placeholder, so the blob's
    # index stays equal to the token id.
    bpe_ids = {i for _, i in items}
    added = {t["id"]: t["content"] for t in j.get("added_tokens", [])}
    top = max([i for _, i in items] + list(added)) if added else len(items) - 1
    table = {i: None for i in range(top + 1)}
    for tok, i in items:
        table[i] = tok
    for i, tok in added.items():
        table[i] = tok
    for i in range(top + 1):
        if table[i] is None:
            table[i] = "<|unused_%d|>" % i
    items = [(table[i], i) for i in range(top + 1)]
    n = len(items)
    added_ids = sorted(i for i in added if i not in bpe_ids)

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    def raw(tok):
        out = bytearray()
        for ch in tok:
            if ch not in u2b:
                raise ValueError(f"character {ch!r} not in byte-level alphabet (token {tok!r})")
            out.append(u2b[ch])
        return bytes(out)

    merge_rank = {}
    for rank, pair in enumerate(model["merges"]):
        if isinstance(pair, str):
            left, right = pair.split()
        else:
            left, right = pair
        merge_rank[left + right] = rank

    # A token above the BPE range is an added token: its text is literal, not
    # byte-level alphabet text.  (Some tokenizers list *every* entry in
    # added_tokens; those ids are in bpe_ids and stay byte-level.)
    raw_tokens = [t.encode("utf-8") if i not in bpe_ids else raw(t) for t, i in items]
    maxlen = max(len(x) for x in raw_tokens)
    with open(a.output, "wb") as f:
        f.write(struct.pack("<i", maxlen))
        for (tok, idx), rb in zip(items, raw_tokens):
            rank = merge_rank.get(tok)
            score = 1.0e6 - float(rank) if rank is not None else 0.0
            f.write(struct.pack("<fi", score, len(rb)))
            f.write(rb)
    print(f"wrote {a.output}: vocab={n} merges={len(merge_rank)} maxlen={maxlen}"
          f" added={len(added_ids)}" + (f" (ids {added_ids[0]}..{added_ids[-1]})" if added_ids else ""))

if __name__ == "__main__":
    main()
