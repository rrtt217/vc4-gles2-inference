r"""RWKV World vocab (rwkv_vocab_v20230424.txt) -> the engine's .tok blob.

The blob is the format docs/model-formats.md documents: i32 maxlen, then for
each id (f32 score, i32 len, bytes) with blob index == token id.  RWKV has no
BPE merges, so every score is 0; the engine's RWKV mode (P3) ignores scores and
does greedy longest matching against these bytes.

The vocab file is `idx repr(token) len` per line, CRLF or LF.  `repr` is a
Python literal: b'...' is raw bytes, '...' is a *str* literal whose \xNN is a
codepoint (U+0080 -> b'\xc2\x80'), which is why ast.literal_eval is used and
the declared length is asserted (the one trap: id 2422 is b'\xbf\xbd',
id 2423 is '\x80' = b'\xc2\x80', both 2 bytes).

Usage: python3 make_tok_rwkv.py --vocab models/rwkv7/g1d-hf/rwkv_vocab_v20230424.txt \
    --out models/rwkv7/g1d.tok [--vocab-size 65536] [--bos-text <text>]
"""
import argparse, ast, struct

DEFAULT_BOS = "<|rwkv_tokenizer_end_of_text|>"

def parse_vocab(path):
    entries = {}
    for raw in open(path, "rb").read().splitlines():
        if not raw.strip():
            continue
        line = raw.decode("utf-8")
        idx_s, rest = line.split(" ", 1)
        repr_s, len_s = rest.rsplit(" ", 1)
        val = ast.literal_eval(repr_s)
        b = val.encode("utf-8") if isinstance(val, str) else val
        n = int(len_s)
        assert len(b) == n, "%s: id %s declares %d bytes, got %d (%r)" % (path, idx_s, n, len(b), val)
        entries[int(idx_s)] = b
    return entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--bos-text", default=DEFAULT_BOS)
    a = ap.parse_args()
    idx = parse_vocab(a.vocab)
    n = a.vocab_size
    assert 1 + max(idx) <= n, "vocab has id %d, larger than %d" % (max(idx), n)
    table = [None] * n
    table[0] = a.bos_text.encode("utf-8")
    for i in range(1, n):
        table[i] = idx.get(i)
        if table[i] is None:
            table[i] = ("[PAD%d]" % i).encode("utf-8")
    maxlen = max(len(x) for x in table)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<i", maxlen))
        for b in table:
            f.write(struct.pack("<fi", 0.0, len(b)))
            f.write(b)
    print("wrote %s: vocab=%d entries=%d maxlen=%d bos=%r" % (a.out, n, len(idx), maxlen, a.bos_text))

if __name__ == "__main__":
    main()
