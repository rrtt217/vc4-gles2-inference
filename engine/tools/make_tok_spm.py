"""Build the `.tok` blob for a SentencePiece **unigram** model.

`make_tok.py` builds the blob from a `tokenizer.json`'s BPE merges and writes
`score = 1e6 - rank`, so the engine's merge pass and the reference agree on which
pair wins.  A unigram model has no merges at all -- it has a *score* a piece, and
its segmentation is the one that maximises the sum (Viterbi).  Writing such a
table through `make_tok.py` gives every piece score 0.0, and the engine's merge
loop then merges by insertion order: it produces tokens, just not the model's.
So this writes the piece's own log probability instead, which is the number
`Tokenizer::encode_unigram` maximises.

    # one spm, one table (T5: tokenizer.json's `model.vocab` is the same pieces)
    python3 engine/tools/make_tok_spm.py --spm models/t5-small/spiece.model \
        --out models/t5-small/t5.tok

    # two spms and the *merged* table (the OPUS-MT family): vocab.json is the
    # source and target pieces interleaved by frequency, so a piece's id is its
    # position in that file and not in either spm.  A piece's score comes from
    # the source model when it has one and the target model otherwise, which is
    # what the reference does for source text.
    python3 engine/tools/make_tok_spm.py \
        --vocab-json models/opus-mt-en-zh/vocab.json \
        --source-spm models/opus-mt-en-zh/source.spm \
        --target-spm models/opus-mt-en-zh/target.spm \
        --out models/opus-mt-en-zh/opus.tok

Blob layout (what `Tokenizer::load_memory` reads, unchanged): an i32 max piece
length, then per id a f32 score, an i32 byte length and the piece's raw bytes.
SentencePiece's `\\u2581` becomes a plain space and `<0xNN>` becomes that byte,
because the engine matches raw bytes: a piece spelled as text never matches, and
the byte fallback then emits `byte + 3`, an id it guessed.

The pieces are *not* deduplicated while preserving both ids -- the blob is
id-indexed, so `n` must equal the model's vocabulary and ids must not move.
Where two pieces are the same byte string (source and target both spell `▁the`),
the merged table already holds one entry, so nothing collides.
"""
import argparse
import json
import struct
import sys


def spm_pieces(path):
    """[(piece, score)] in id order, from a SentencePiece ModelProto.

    ModelProto.pieces is field 1; each SentencePiece is {piece: 1, score: 2}.
    TrainerSpec (field 2) field 3 is the model type, which is only used to
    refuse a BPE model loudly -- this writer's scores are unigram's.
    """
    b = open(path, "rb").read()

    def varint(b, i):
        r = 0; s = 0
        while True:
            c = b[i]; i += 1
            r |= (c & 0x7f) << s; s += 7
            if not c & 0x80:
                return r, i

    def fields(b):
        i = 0
        while i < len(b):
            k, i = varint(b, i)
            f, w = k >> 3, k & 7
            if w == 0:
                v, i = varint(b, i); yield f, w, v
            elif w == 2:
                n, i = varint(b, i); yield f, w, b[i:i + n]; i += n
            elif w == 5:
                yield f, w, b[i:i + 4]; i += 4
            elif w == 1:
                yield f, w, b[i:i + 8]; i += 8
            else:
                raise SystemExit("%s: protobuf wire type %d" % (path, w))

    out, model_type = [], None
    for f, w, v in fields(b):
        if f == 1 and w == 2:
            piece = score = None
            for f2, w2, v2 in fields(v):
                if f2 == 1:
                    piece = v2
                elif f2 == 2:
                    score = struct.unpack("<f", v2)[0]
            if piece is None or score is None:
                raise SystemExit("%s: a piece has no text or no score" % path)
            out.append((piece.decode("utf-8"), score))
        elif f == 2 and w == 2:
            for f2, w2, v2 in fields(v):
                if f2 == 3 and w2 == 0:
                    model_type = v2
    if model_type not in (None, 1):
        names = {2: "BPE", 3: "WORD", 4: "CHAR"}
        raise SystemExit("%s: model_type %d (%s), not unigram.  A BPE "
                         "SentencePiece model goes through make_tok.py."
                         % (path, model_type, names.get(model_type, "?")))
    return out


def raw_bytes(piece):
    """The bytes the engine matches, not the piece's spelling."""
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            pass
    return piece.replace("\u2581", " ").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm", help="the piece table, or ...")
    ap.add_argument("--vocab-json", help="...the merged id->piece table")
    ap.add_argument("--source-spm")
    ap.add_argument("--target-spm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.spm:
        table = spm_pieces(a.spm)
        n = len(table)
        entries = list(table)
        kind = "one spm"
    elif a.vocab_json:
        if not (a.source_spm and a.target_spm):
            raise SystemExit("--vocab-json needs --source-spm and --target-spm: "
                             "the merged table does not carry the scores")
        vocab = json.load(open(a.vocab_json))
        n = max(vocab.values()) + 1
        if len(vocab) != n:
            raise SystemExit("%s: ids are not 0..%d (n=%d); the blob is "
                             "id-indexed" % (a.vocab_json, n - 1, len(vocab)))
        src = dict(spm_pieces(a.source_spm))
        tgt = dict(spm_pieces(a.target_spm))
        by_id = {i: p for p, i in vocab.items()}
        entries, added = [], 0
        for i in range(n):
            p = by_id.get(i)
            if p is None:
                raise SystemExit("%s: id %d has no piece" % (a.vocab_json, i))
            if p in src:
                entries.append((p, src[p]))
            elif p in tgt:
                entries.append((p, tgt[p]))
            else:
                # Language tags (>>cmn_Hans<<) and <pad> are added tokens: the
                # reference never produces them by segmentation, so the score
                # only has to be low enough that Viterbi never chooses one.
                # It is not an error -- the merged table is the model's own.
                entries.append((p, -1.0e4))
                added += 1
        kind = "merged vocab.json, %d added pieces at -1e4" % added
    else:
        raise SystemExit("give --spm, or --vocab-json with both spms")

    blobs = [raw_bytes(p) for p, _ in entries]
    maxlen = max(len(x) for x in blobs)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<i", maxlen))
        for (p, sc), rb in zip(entries, blobs):
            f.write(struct.pack("<fi", sc, len(rb)))
            f.write(rb)
    nbytes = 4 + sum(8 + len(x) for x in blobs)
    print("wrote %s: vocab=%d maxlen=%d bytes=%d (%s, scores are log probs)"
          % (a.out, len(entries), maxlen, nbytes, kind))


if __name__ == "__main__":
    sys.exit(main())
