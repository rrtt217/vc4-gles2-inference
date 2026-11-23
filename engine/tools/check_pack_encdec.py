"""Is a v5 container the checkpoint it claims to be?

`mmcf_pack.py --verify` proves the *bytes* are what the writer meant; this proves
the bytes are what the model *does*: it dequantizes the container back to f32,
runs it through the same reference forward as `ref_encdec.py` on the source
checkpoint, and compares.  A record written with the wrong id, a row order one
off, or a matrix transposed all pass `--verify` and all fail here.

Three checks, cheapest first:

  * the f32 rows (`bias_dim`, `bias_ffn`, `logits_bias`, `pos_table`) must be
    **bit-exact** -- they are not quantized, so any difference is a layout bug;
  * the q8 bodies must correlate with the source; and
  * a full encoder pass plus the first decoder step's logits: the cosine and
    the **argmax** are what decide whether the translation can match.  A q8
    body is lossy, so the pass/fail is the argmax and a cosine floor, not
    equality.

    uv run engine/tools/check_pack_encdec.py \
        models/opus-mt-en-zh/opus.q8.mmcf models/opus-mt-en-zh "The weather is nice."

Two traps this had to get past, both of which look like a broken packer:

  * a matrix body is **q8u8** (pre-biased +128, dtype 2) while the embedding is
    **q8i8** (dtype 3).  Reading the first as int8 shifts every weight by 128
    and the cosine comes out -0.71, which is the fingerprint to recognise;
  * `pre_biased` in metadata key 3 is a property of the writer, not of every
    record, so the record's own dtype is what decides how to read a body.
"""
import sys

import numpy as np

import mmcf_format as F
from ref_encdec import Tok, attn, encoder, load_marian, logits_last, self_attn

# The container's q8 error is ~0.4% a matrix and it compounds through 6 layers.
ENC_COS_FLOOR = 0.99
LOGITS_COS_FLOOR = 0.99


def read_container(path):
    blob = open(path, "rb").read()
    h = F.Header(blob)
    tbl = (h.meta_off + h.meta_size + 15) // 16 * 16
    T = {}
    for i in range(h.n_tensors):
        tid, dt, fl, grp, rows, cols, off, blen, crc, scale, aux = F.rec_unpack(
            blob[tbl + i * h.rec_size: tbl + (i + 1) * h.rec_size])
        kind, layer, part = tid & 0xFF, (tid >> 8) & 0x3F, tid >> 14
        T[(kind, layer, part)] = dict(dtype=dt, rows=rows, cols=cols, scale=scale,
                                      data=blob[h.data_off + off:
                                                h.data_off + off + blen])
    return h, T


def dequant(r):
    a = np.frombuffer(r["data"], np.uint8 if r["dtype"] == F.DT_Q8U8
                      else np.int8)
    v = (a.astype(np.int16) - 128) if r["dtype"] == F.DT_Q8U8 else a
    return v.reshape(r["rows"], r["cols"]).astype(np.float32) * np.float32(r["scale"])


def f32rows(T, kind):
    r = T[(kind, 0, 0)]
    return np.frombuffer(r["data"], "<f4").reshape(r["rows"], r["cols"])


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    path, src_dir, text = argv[1], argv[2], " ".join(argv[3:])
    h, T = read_container(path)
    M = load_marian(src_dir)
    dim, layers, enc_layers = M["dim"], M["layers"], len(M["enc"])
    bad = []

    def check(name, ok, detail):
        print("%s %-12s %s" % ("ok  " if ok else "FAIL", name, detail))
        if not ok:
            bad.append(name)

    logits_bias = f32rows(T, 0x3B)[0]
    pos = f32rows(T, 0x3C)
    check("logits_bias", np.array_equal(logits_bias, M["final_logits_bias"]),
          "bit-exact" if np.array_equal(logits_bias, M["final_logits_bias"])
          else "deviates")
    check("pos_table", np.array_equal(pos, M["pos"][:pos.shape[0]]),
          "bit-exact, %d rows" % pos.shape[0])

    er = T[(1, 0, 0)]
    emb = np.frombuffer(er["data"], np.int8).reshape(er["rows"], er["cols"])
    emb = emb.astype(np.float32) * np.frombuffer(T[(2, 0, 0)]["data"], "<f4")[:, None]
    check("embedding", float(np.abs(emb - M["shared"]).max()) < 0.02,
          "maxabs %.5f" % float(np.abs(emb - M["shared"]).max()))

    # Rebuild the reference tensors from the container.
    bias = f32rows(T, 0x39)
    bffn = f32rows(T, 0x3A)
    C = dict(M, shared=emb, final_logits_bias=logits_bias, pos=pos, pos_dec=pos)
    C["enc"] = []
    for l in range(enc_layers):
        qkv, B = dequant(T[(0x31, l, 0)]), 90 + l * 9
        C["enc"].append(dict(
            q=qkv[:dim], qb=bias[B + 0], k=qkv[dim:2 * dim], kb=bias[B + 1],
            v=qkv[2 * dim:], vb=bias[B + 2], o=dequant(T[(0x32, l, 0)]),
            ob=bias[B + 3], ln1w=bias[B + 4], ln1b=bias[B + 5],
            ln2w=bias[B + 6], ln2b=bias[B + 7],
            w1=dequant(T[(0x33, l, 0)]), b1=bffn[layers + l],
            w2=dequant(T[(0x34, l, 0)]), b2=bias[B + 8]))
    C["dec"] = []
    for l in range(layers):
        qkv, B = dequant(T[(0x10, l, 0)]), l * 15
        C["dec"].append(dict(
            q=qkv[:dim], qb=bias[B + 0], k=qkv[dim:2 * dim], kb=bias[B + 1],
            v=qkv[2 * dim:], vb=bias[B + 2], o=dequant(T[(0x11, l, 0)]),
            ob=bias[B + 3], cq=dequant(T[(0x35, l, 0)]), cqb=bias[B + 4],
            ck=dequant(T[(0x36, l, 0)]), ckb=bias[B + 5],
            cv=dequant(T[(0x37, l, 0)]), cvb=bias[B + 6],
            co=dequant(T[(0x38, l, 0)]), cob=bias[B + 7],
            ln1w=bias[B + 8], ln1b=bias[B + 9], ln2w=bias[B + 10],
            ln2b=bias[B + 11], ln3w=bias[B + 12], ln3b=bias[B + 13],
            w1=dequant(T[(0x12, l, 0)]), b1=bffn[l],
            w2=dequant(T[(0x15, l, 0)]), b2=bias[B + 14]))

    # One encoder layer on a shared input names the layer that moved, which a
    # whole-stack cosine cannot.
    tok = Tok(src_dir + "/opus.tok")
    src = tok.encode(" " + text) + [M["eos_id"]]
    x0 = M["shared"][np.array(src)] * M["scale_emb"] + M["pos"][:len(src)]
    for l in range(min(enc_layers, 2)):
        a_r = attn(self_attn(M["enc"][l]), x0, dim, M["heads"])
        a_c = attn(self_attn(C["enc"][l]), x0, dim, M["heads"])
        c = float((a_r * a_c).sum() / (np.linalg.norm(a_r) * np.linalg.norm(a_c)))
        check("enc layer %d" % l, c > 0.995, "attn cosine %.6f" % c)

    enc_r, enc_c = encoder(M, np.array(src)), encoder(C, np.array(src))
    c = float((enc_r * enc_c).sum() / (np.linalg.norm(enc_r) * np.linalg.norm(enc_c)))
    check("encoder", c > ENC_COS_FLOOR, "cosine %.6f, maxabs %.4f"
          % (c, float(np.abs(enc_r - enc_c).max())))
    lr = logits_last(M, np.array([M["start_id"]]), enc_r)
    lc = logits_last(C, np.array([M["start_id"]]), enc_c)
    c = float((lr * lc).sum() / (np.linalg.norm(lr) * np.linalg.norm(lc)))
    same = int(lr.argmax()) == int(lc.argmax())
    check("logits", same and c > LOGITS_COS_FLOOR,
          "cosine %.6f, argmax %d/%d %s" % (c, int(lr.argmax()), int(lc.argmax()),
                                            "same" if same else "DIFFERENT"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
