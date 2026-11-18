#!/usr/bin/env python3
"""HuggingFace `tokenizer.json` -> the engine's `.tok` blob, any BPE flavour.

    make_tok.py --input <dir or tokenizer.json> --output model.tok

The engine's tokenizer stores raw bytes per entry and uses `score` as the merge
priority (it keeps the *highest* score, so rank 0 gets the largest).  Three
things bite when this is done ad hoc, one per model family already in this
repository:

* **ByteLevel BPE** (Qwen/Llama: `Ġ` for space) needs `bytes_to_unicode()` back
  before the bytes are written - `Ġ` is a printable stand-in for 0x20.
* **SentencePiece BPE** (`▁` for space, byte fallback) needs only that marker
  rewritten, and must NOT go through the byte-level alphabet.
* **Added tokens** live *above* the BPE range in most recent tokenizers
  (`<|im_start|>` at 16384 next to a 16384-token BPE, `<|im_end|>` at 32000
  next to 32000).  A blob built from `model["vocab"]` alone therefore ends at
  the wrong id, and the engine splits the chat markers into characters.  They
  are laid onto their ids here, with the holes filled, so blob index == token id.

This replaces the per-model copies (`minimind3/make_tokenizer.py`,
`models/tinybrainbot/make_tok.py`); both are supersets of what those did.

One more trap, measured on Minueza's tokenizer: 50 of its merges are written as
`"; \r"`, i.e. the right-hand token is a literal carriage return, and
`str.split()` eats it.  Splitting on the first space only is what the file
means.
"""
import argparse
import json
import os
import struct


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    return json.load(open(path))


def flavour(items):
    """SentencePiece writes space as U+2581; ByteLevel writes it as `Ġ`."""
    for tok, _ in items[:4000]:
        if "\u2581" in tok:
            return "sentencepiece"
    return "bytelevel"


def is_byte_token(tok):
    return len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">")


def dedupe(items, raw_tokens):
    """Make one id per byte string; the others become empty entries.

    Two ids can spell the same bytes, and the engine's map is *last wins* while
    its blob index must stay equal to the token id, so the winner has to be
    decided here and the loser written as an entry that can never match:

    * a SentencePiece vocabulary carries both the marker piece `\u2581` and a
      literal space; rewriting the marker to a space makes them identical, and
      the reference tokenizer only ever emits the marker (BitCPM-CANN-0.5B: the
      real space id 72181 is unreachable, so every word boundary has to be
      59320 -- measured 47 of 124 prompts wrong without this);
    * byte fallback spells a byte as `<0x3F>`, which after the raw write above
      collides with the real `?` piece.  A real piece always wins.

    Priority is (is a byte-fallback entry, id): a real piece before a fallback
    byte, then the lowest id.  Emptying the loser keeps `vocab_[id]` indexed by
    id, so nothing shifts.
    """
    winner = {}
    for (tok, idx), rb in zip(items, raw_tokens):
        if not rb:
            continue
        key = (1 if is_byte_token(tok) else 0, idx)
        if rb not in winner or key < winner[rb]:
            winner[rb] = key
    out = []
    for (tok, idx), rb in zip(items, raw_tokens):
        if rb and winner.get(rb) != (1 if is_byte_token(tok) else 0, idx):
            rb = b""
        out.append(rb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="tokenizer.json, or a model directory containing one")
    ap.add_argument("--output", required=True)
    ap.add_argument("--flavour", choices=("auto", "bytelevel", "sentencepiece"),
                    default="auto")
    a = ap.parse_args()

    j = load(a.input)
    model = j["model"]
    vocab = model["vocab"]
    items = (list(enumerate(vocab)) if isinstance(vocab, list)
             else sorted(vocab.items(), key=lambda kv: kv[1]))

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
    n_added = len([i for i in added if i not in bpe_ids])

    kind = a.flavour if a.flavour != "auto" else flavour(items)

    rank = {}
    for r, pair in enumerate(model.get("merges") or []):
        # "a b" means the pair ("a","b"); split on the first space only, or a
        # merge whose right token is "\r" loses it.
        left, right = pair.split(" ", 1) if isinstance(pair, str) else pair
        rank[left + right] = r

    if kind == "sentencepiece":
        def raw(tok):
            # Byte fallback: the vocabulary spells the byte 0x41 as "<0x41>",
            # but the engine matches raw bytes, so the entry has to *be* that
            # byte.  Written as text it never matches, and the engine's
            # fallback then emits `byte + 3` -- an id it guessed (BitCPM's
            # rare CJK came out as 243 163 131 131 instead of 1329 1249 1217).
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    return bytes([int(tok[3:5], 16)])
                except ValueError:
                    pass
            return tok.replace("\u2581", " ").encode("utf-8")
    else:
        b2u = bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}

        def raw(tok):
            out = bytearray()
            for ch in tok:
                if ch not in u2b:
                    raise SystemExit("character %r is not in the byte-level alphabet "
                                     "(token %r); is this really ByteLevel BPE?  Try "
                                     "--flavour sentencepiece" % (ch, tok))
                out.append(u2b[ch])
            return bytes(out)

    # An added token's text is literal, not alphabet text; a tokenizer that
    # lists every entry in added_tokens keeps its BPE entries byte-level.
    raw_tokens = [t.encode("utf-8") if i not in bpe_ids else raw(t) for t, i in items]
    raw_tokens = dedupe(items, raw_tokens)
    maxlen = max(len(x) for x in raw_tokens)
    with open(a.output, "wb") as f:
        f.write(struct.pack("<i", maxlen))
        for (tok, idx), rb in zip(items, raw_tokens):
            r = rank.get(tok)
            score = 1.0e6 - float(r) if r is not None else 0.0
            f.write(struct.pack("<fi", score, len(rb)))
            f.write(rb)
    print("wrote %s: vocab=%d merges=%d maxlen=%d added=%d (%s)"
          % (a.output, n, len(rank), maxlen, n_added, kind))


if __name__ == "__main__":
    main()
