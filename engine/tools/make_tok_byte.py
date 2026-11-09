#!/usr/bin/env python3
"""A pure byte-level `byte_vocab.json` -> the engine's `.tok` blob.

    make_tok_byte.py --input <HF dir with byte_vocab.json> --output model.tok

`make_tok.py` covers BPE (ByteLevel or SentencePiece) and lays added tokens onto
their ids.  A byte tokenizer has neither: there is no `tokenizer.json`, no merge
table and no `model["vocab"]` to map through `bytes_to_unicode()`.  Its entries
are `<0xNN>` strings standing for *the raw byte NN*, and running them through
make_tok.py would write the six characters `<0x41>` where the engine needs the
single byte `0x41` - every id would still exist, so nothing would fail loudly,
and the model would read a different byte than the one it was trained on.

The blob is otherwise the same format (docs/model-formats.md): i32 maxlen, then
(f32 score, i32 len, bytes) per entry, blob index == token id.  Scores are 0.0
because there is nothing to merge; the engine's `bpe_merge` finds no pair of
entries that concatenates to another entry and stops immediately.

Verify against the model's own tokenizer before packing:

    python3 - <<'EOF'
    import json
    v = json.load(open('byte_vocab.json'))
    assert len(v) == 259 and v['<0x41>'] == 65 and v['<eos>'] == 258
    EOF
"""
import argparse
import json
import os
import struct

BYTE_PREFIX = "<0x"


def entries(vocab):
    """id -> raw bytes, from a `<0xNN>` / `<special>` name table."""
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    top = items[-1][1]
    table = [None] * (top + 1)
    for name, idx in items:
        if name.startswith(BYTE_PREFIX) and name.endswith(">"):
            b = int(name[3:-1], 16)
            if not 0 <= b <= 255:
                raise SystemExit("%r is not a byte" % name)
            table[idx] = bytes([b])
        else:
            table[idx] = name.encode("utf-8")
    missing = [i for i, v in enumerate(table) if v is None]
    if missing:
        raise SystemExit("ids missing from the vocab: %s" % missing[:8])
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="byte_vocab.json, or a model directory containing one")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    path = a.input
    if os.path.isdir(path):
        path = os.path.join(path, "byte_vocab.json")
    vocab = json.load(open(path))
    table = entries(vocab)
    maxlen = max(len(x) for x in table)
    with open(a.output, "wb") as f:
        f.write(struct.pack("<i", maxlen))
        for i, rb in enumerate(table):
            f.write(struct.pack("<fi", 0.0, len(rb)))
            f.write(rb)
    specials = [t.decode("utf-8", "replace") for t in table if len(t) > 1]
    print("wrote %s: vocab=%d maxlen=%d specials=%s"
          % (a.output, len(table), maxlen,
             {i: s for i, s in enumerate(specials) if len(table[i]) > 1}))


if __name__ == "__main__":
    main()
