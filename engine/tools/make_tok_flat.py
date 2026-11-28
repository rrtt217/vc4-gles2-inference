#!/usr/bin/env python3
"""Build a `.tok` blob from a flat, longest-match vocabulary.

    make_tok_flat.py --input corpus/vocab_v2.json --output miniyuanyuan.tok

`make_tok.py` is for a BPE export (merges, `1e6 - rank` scores) and
`make_tok_spm.py` for a SentencePiece unigram model (log probabilities).  A
third kind of vocabulary is a plain list of strings with neither: MiniYuanYuan
ships `vocab_v2.json`, whose `tokens` array *is* the id order and whose only
segmentation rule is "take the longest entry that fits at this position".  The
blob format is the same for all three -- `i32 maxlen`, then `f32 score, i32 len,
bytes` per id -- because the id order is the file order.

Two things this writes that the others do not:

- **Score 0.0 everywhere.**  There are no merges, so there is no rank to encode.
  A blob of zeros fed to the BPE pass is *not* a no-op (0.0 beats the -1e30
  initial best, so it merges by insertion order), which is why the container
  states the algorithm in metadata key 10 rather than leaving the engine to
  guess.  `--tokenizer-kind greedy` on `mmcf_pack.py` writes that byte.
- **Whitespace as named entries.**  The vocabulary states space, tab and newline
  as `<space>`/`<newline>` and no entry contains any of those bytes, so the blob
  must carry the names and the engine maps the bytes to them.  `--check`
  verifies exactly that, because a vocabulary that *did* contain a space would
  need the opposite treatment and would tokenise every prompt as `<unk>`.

`--check` also proves the three invariants the reference's own smoke test does:
every listed character is one token, every single-token word is one token, and
`decode(encode(x)) == x` for text drawn from the vocabulary.
"""

import argparse
import json
import struct
import sys


WS = {" ": "<space>", "\n": "<newline>", "\t": "<space>"}


def load(path):
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    if "tokens" not in v:
        raise SystemExit("%s: no 'tokens' array; is this a flat vocabulary?" % path)
    return v["tokens"], dict(v.get("specials") or {})


def table_of(tokens, specials):
    """Validate the names and return (id -> bytes, maxlen)."""
    seen = {}
    for i, t in enumerate(tokens):
        if not isinstance(t, str) or t == "":
            raise SystemExit("token %d is not a non-empty string" % i)
        if t in seen:
            raise SystemExit("%r appears at ids %d and %d; a lookup table keeps one "
                             "and the other becomes unreachable" % (t, seen[t], i))
        seen[t] = i
    for name in specials:
        if name not in seen:
            raise SystemExit("special %r is not in the token list" % name)
    for name, idx in WS.items():
        if specials.get(WS[name]) is None and WS[name] not in seen:
            raise SystemExit("%r is not in the vocabulary, so no id can stand for a "
                             "space" % WS[name])
    raw = [t.encode("utf-8") for t in tokens]
    return raw, max(len(b) for b in raw), seen


def check(tokens, specials, raw, seen):
    """The reference's three invariants, plus the whitespace rule."""
    sp = set(specials)
    holes = []
    # No entry may contain a whitespace byte: the vocabulary spells those as
    # names, so an entry with a literal space would be a second, unreachable
    # spelling of the same text.
    for t in tokens:
        if t not in sp and any(w in t for w in WS):
            holes.append("entry %r contains whitespace" % t)
    if holes:
        raise SystemExit("; ".join(holes[:3]))

    maxlen = max(len(t) for t in tokens if t not in sp)
    # The header carries the longest entry of any kind, like make_tok.py writes;
    # the engine recomputes its own scan bound from the entries.  Report the
    # non-special one too, because that is the bound the matcher actually scans.
    all_maxlen = max(len(b) for b in raw)

    def encode(text):
        out, i = [], 0
        while i < len(text):
            if text[i] in WS:
                out.append(seen[WS[text[i]]])
                i += 1
                continue
            for n in range(min(maxlen, len(text) - i), 0, -1):
                j = seen.get(text[i:i + n])
                if j is not None:
                    out.append(j)
                    i += n
                    break
            else:
                out.append(seen["<unk>"])
                i += 1
        return out

    bad = []
    # (1) every listed character is exactly one token
    for c in tokens:
        if c in sp or len(c) != 1:
            continue
        if encode(c) != [seen[c]]:
            bad.append("char %r" % c)
    # (2) every non-special entry is exactly one token
    for c in tokens:
        if c in sp:
            continue
        if encode(c) != [seen[c]]:
            bad.append("entry %r" % c)
    if bad:
        raise SystemExit("these entries do not round-trip to their own id: %s"
                         % ", ".join(bad[:8]))
    # (3) decode(encode(x)) == x, over a sample of the vocabulary
    sample = "".join(t for t in tokens if t not in sp)[:4000]
    rt = "".join(tokens[i] for i in encode(sample))
    if rt.replace("<space>", " ").replace("<newline>", "\n") != sample:
        raise SystemExit("decode(encode(x)) != x over a %d-char sample" % len(sample))
    print("check: %d entries, maxlen %d (text) / %d (with specials), %d specials, "
          "3 invariants hold" % (len(tokens), maxlen, all_maxlen, len(sp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a vocab_v2.json with a tokens array")
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", action="store_true",
                    help="validate the vocabulary before writing the blob")
    a = ap.parse_args()

    tokens, specials = load(a.input)
    raw, _, seen = table_of(tokens, specials)
    if a.check:
        check(tokens, specials, raw, seen)

    header_maxlen = max(len(b) for b in raw)
    with open(a.output, "wb") as f:
        f.write(struct.pack("<i", header_maxlen))
        for t, rb in zip(tokens, raw):
            # No merges, so no rank: every score is 0.0.  The engine must not
            # run the BPE pass on this blob (metadata key 10 says so).
            f.write(struct.pack("<fi", 0.0, len(rb)))
            f.write(rb)
    n_special = len(specials)
    print("wrote %s: vocab=%d maxlen=%d specials=%d scores=flat (longest match)"
          % (a.output, len(tokens), header_maxlen, n_special))


if __name__ == "__main__":
    main()
