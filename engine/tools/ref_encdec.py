"""A numpy f32 forward pass for a Marian encoder-decoder, and its tokenizer.

The engine's oracle for `docs/<model>.md`'s "the conversion is faithful" question:
this reads the *source* checkpoint, does the reference arithmetic in float32 and
prints the tokens and logits the engine has to reproduce.  An engine bug, a
packer bug and a tokenizer bug all look the same from outside (wrong Chinese);
this separates them, which is the same job `ref_forward.py` does for a
decoder-only safetensors source.

Architecture, from `MarianMTModel` with `normalize_before: false` (this
checkpoint states it), `scale_embedding: true`, `static_position_embeddings:
true` and no `add_final_layer_norm`:

    x = shared[src] * sqrt(d) + pos[:n]                  (no norm on the embed)
    per encoder layer, POST-norm:
        x = LN1(x + self_attn(x))                        (bidirectional)
        x = LN2(x + fc2(swish(fc1(x))))                  (fc1/fc2 both biased)
    y = shared[tgt] * sqrt(d) + pos[:m]
    per decoder layer, POST-norm:
        y = LN1(y + self_attn(y, causal))
        y = LN2(y + cross_attn(y, encoder_out))           (q from y, k/v from enc)
        y = LN3(y + fc2(swish(fc1(y))))
    logits = y @ shared.T + final_logits_bias            (both tied to `shared`)

`final_logits_bias` is *not* zero in this checkpoint (max |b| = 8.7), and the
position table is a learned sinusoid laid out [sin | cos] over j = 0..d/2-1 --
verified against the formula to 3e-8, so a shift there would show up as a
disagreement at the first position.

    python3 engine/tools/ref_encdec.py --in models/opus-mt-en-zh \
        --tok models/opus-mt-en-zh/opus.tok "The weather is nice."

`--logits N` prints the first N logits of the first decoder step instead of the
translation, which is what `cmp_logits.py` compares the engine's dump against.
"""
import argparse
import json
import sys

import numpy as np

from pth_legacy import load as load_legacy


def load_marian(model_dir):
    cfg = json.load(open(model_dir + "/config.json"))
    w = load_legacy(model_dir + "/pytorch_model.bin")
    g = lambda k: np.asarray(w[k], dtype=np.float32)
    dim = cfg["d_model"]
    L = cfg["encoder_layers"]
    Ld = cfg["decoder_layers"]
    ff = cfg["encoder_ffn_dim"]
    H = cfg["encoder_attention_heads"]
    out = {
        "dim": dim, "layers": L, "dec_layers": Ld, "hidden": ff, "heads": H,
        "vocab": cfg["vocab_size"], "seq": cfg["max_position_embeddings"],
        "shared": g("model.shared.weight"),
        "final_logits_bias": g("final_logits_bias").reshape(-1),
        "pos": g("model.encoder.embed_positions.weight"),
        "pos_dec": g("model.decoder.embed_positions.weight"),
        "scale_emb": float(np.sqrt(dim)) if cfg.get("scale_embedding") else 1.0,
        "start_id": cfg["decoder_start_token_id"],
        "eos_id": cfg["eos_token_id"],
        "pad_id": cfg["pad_token_id"],
        "ln_eps": float(cfg.get("layer_norm_epsilon", 1e-5)),
        "swish": cfg.get("activation_function", "swish") == "swish",
        "enc": [], "dec": [],
    }
    for l in range(L):
        p = "model.encoder.layers.%d." % l
        out["enc"].append(dict(
            q=g(p + "self_attn.q_proj.weight"), qb=g(p + "self_attn.q_proj.bias"),
            k=g(p + "self_attn.k_proj.weight"), kb=g(p + "self_attn.k_proj.bias"),
            v=g(p + "self_attn.v_proj.weight"), vb=g(p + "self_attn.v_proj.bias"),
            o=g(p + "self_attn.out_proj.weight"), ob=g(p + "self_attn.out_proj.bias"),
            ln1w=g(p + "self_attn_layer_norm.weight"), ln1b=g(p + "self_attn_layer_norm.bias"),
            w1=g(p + "fc1.weight"), b1=g(p + "fc1.bias"),
            w2=g(p + "fc2.weight"), b2=g(p + "fc2.bias"),
            ln2w=g(p + "final_layer_norm.weight"), ln2b=g(p + "final_layer_norm.bias")))
    for l in range(Ld):
        p = "model.decoder.layers.%d." % l
        out["dec"].append(dict(
            q=g(p + "self_attn.q_proj.weight"), qb=g(p + "self_attn.q_proj.bias"),
            k=g(p + "self_attn.k_proj.weight"), kb=g(p + "self_attn.k_proj.bias"),
            v=g(p + "self_attn.v_proj.weight"), vb=g(p + "self_attn.v_proj.bias"),
            o=g(p + "self_attn.out_proj.weight"), ob=g(p + "self_attn.out_proj.bias"),
            ln1w=g(p + "self_attn_layer_norm.weight"), ln1b=g(p + "self_attn_layer_norm.bias"),
            cq=g(p + "encoder_attn.q_proj.weight"), cqb=g(p + "encoder_attn.q_proj.bias"),
            ck=g(p + "encoder_attn.k_proj.weight"), ckb=g(p + "encoder_attn.k_proj.bias"),
            cv=g(p + "encoder_attn.v_proj.weight"), cvb=g(p + "encoder_attn.v_proj.bias"),
            co=g(p + "encoder_attn.out_proj.weight"), cob=g(p + "encoder_attn.out_proj.bias"),
            ln2w=g(p + "encoder_attn_layer_norm.weight"), ln2b=g(p + "encoder_attn_layer_norm.bias"),
            w1=g(p + "fc1.weight"), b1=g(p + "fc1.bias"),
            w2=g(p + "fc2.weight"), b2=g(p + "fc2.bias"),
            ln3w=g(p + "final_layer_norm.weight"), ln3b=g(p + "final_layer_norm.bias")))
    return out


def ln(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def act(x, swish):
    if swish:
        return x / (1.0 + np.exp(-x))
    return np.maximum(x, 0.0)


def attn(a, x, dim, heads, k_in=None, v_in=None, causal=False):
    """One (multi-head) attention block.  `a` carries the eight tensors it needs,
    so a self-attention and a cross-attention differ only in which set is passed
    -- the cross block's q is its own projection, not the self block's."""
    H, d = heads, dim // heads
    n, nsrc = x.shape[0], (x.shape[0] if k_in is None else k_in.shape[0])
    q = (x @ a["q"].T + a["qb"]).reshape(n, H, d).transpose(1, 0, 2)
    kx = x if k_in is None else k_in
    vx = x if v_in is None else v_in
    k = (kx @ a["k"].T + a["kb"]).reshape(nsrc, H, d).transpose(1, 0, 2)
    v = (vx @ a["v"].T + a["vb"]).reshape(nsrc, H, d).transpose(1, 0, 2)
    s = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(d)
    if causal:
        mask = np.triu(np.ones((n, nsrc), dtype=bool), 1)
        s = np.where(mask[None], -1e30, s)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("hqk,hkd->hqd", p, v).transpose(1, 0, 2).reshape(n, dim)
    return o @ a["o"].T + a["ob"]


def self_attn(m):
    return {k: m[k] for k in ("q", "qb", "k", "kb", "v", "vb", "o", "ob")}


def cross_attn(m):
    return {"q": m["cq"], "qb": m["cqb"], "k": m["ck"], "kb": m["ckb"],
            "v": m["cv"], "vb": m["cvb"], "o": m["co"], "ob": m["cob"]}


def encoder(M, src_ids):
    dim, H = M["dim"], M["heads"]
    x = M["shared"][src_ids] * M["scale_emb"] + M["pos"][:len(src_ids)]
    for m in M["enc"]:
        x = ln(x + attn(self_attn(m), x, dim, H), m["ln1w"], m["ln1b"], M["ln_eps"])
        h = act(x @ m["w1"].T + m["b1"], M["swish"])
        x = ln(x + h @ m["w2"].T + m["b2"], m["ln2w"], m["ln2b"], M["ln_eps"])
    return x


def decoder(M, tgt_ids, enc_out):
    dim, H = M["dim"], M["heads"]
    y = M["shared"][tgt_ids] * M["scale_emb"] + M["pos_dec"][:len(tgt_ids)]
    for m in M["dec"]:
        y = ln(y + attn(self_attn(m), y, dim, H, causal=True),
               m["ln1w"], m["ln1b"], M["ln_eps"])
        y = ln(y + attn(cross_attn(m), y, dim, H, k_in=enc_out, v_in=enc_out),
               m["ln2w"], m["ln2b"], M["ln_eps"])
        h = act(y @ m["w1"].T + m["b1"], M["swish"])
        y = ln(y + h @ m["w2"].T + m["b2"], m["ln3w"], m["ln3b"], M["ln_eps"])
    return y


def logits_last(M, tgt_ids, enc_out):
    y = decoder(M, tgt_ids, enc_out)
    return y[-1] @ M["shared"].T + M["final_logits_bias"]


def residual_peaks(M, src_ids, tgt_ids):
    """Per-layer post-add residual and FFN-hidden peaks, plus the head input.

    A "residual" here is what the engine stores in res_fmt_ between two
    sublayers: x + sublayer(x) *before* the LayerNorm, not the LN's output.
    Its rail metric is the row's plain max, the same quantity rail_peaks()
    measures for a dense model: the store is absolute, and the LN subtracts
    the mean only on the way out, so a large uniform offset is as
    unrecoverable as a large deviation from it.  On OPUS-MT the two maxima
    happen to coincide (dec1's FFN carries one 5959 dim at the start token).
    The hidden is the swish(fc1(.)) a non-gated FFN writes in ACT; nothing
    subtracts its mean, so its plain max is the rail.  Encoder and decoder
    layers are combined by index; the head input is the decoder's final
    LayerNorm output.
    """
    import numpy as np
    dim, H, eps = M["dim"], M["heads"], M["ln_eps"]
    x = M["shared"][src_ids] * M["scale_emb"] + M["pos"][:len(src_ids)]
    def mx(r):
        return float(np.abs(r).max())
    er, eh = [], []
    for m in M["enc"]:
        r = x + attn(self_attn(m), x, dim, H)
        er.append(mx(r))
        x = ln(r, m["ln1w"], m["ln1b"], eps)
        h = act(x @ m["w1"].T + m["b1"], M["swish"])
        eh.append(float(np.abs(h).max()))
        r = x + h @ m["w2"].T + m["b2"]
        er[-1] = max(er[-1], mx(r))
        x = ln(r, m["ln2w"], m["ln2b"], eps)
    enc_out = x
    y = M["shared"][tgt_ids] * M["scale_emb"] + M["pos_dec"][:len(tgt_ids)]
    dr, dh = [], []
    for m in M["dec"]:
        r = y + attn(self_attn(m), y, dim, H, causal=True)
        dr.append(mx(r))
        y = ln(r, m["ln1w"], m["ln1b"], eps)
        r = y + attn(cross_attn(m), y, dim, H, k_in=enc_out, v_in=enc_out)
        dr[-1] = max(dr[-1], mx(r))
        y = ln(r, m["ln2w"], m["ln2b"], eps)
        h = act(y @ m["w1"].T + m["b1"], M["swish"])
        dh.append(float(np.abs(h).max()))
        r = y + h @ m["w2"].T + m["b2"]
        dr[-1] = max(dr[-1], mx(r))
        y = ln(r, m["ln3w"], m["ln3b"], eps)
    n = max(len(er), len(dr))
    res = [max(er[i] if i < len(er) else 0.0, dr[i] if i < len(dr) else 0.0)
           for i in range(n)]
    nh = max(len(eh), len(dh))
    hid = [max(eh[i] if i < len(eh) else 0.0, dh[i] if i < len(dh) else 0.0)
           for i in range(nh)]
    return res, hid, float(np.abs(y).max())


# --- the tokenizer, over the .tok blob the engine loads ---------------------
class Tok:
    """Viterbi over the blob `make_tok_spm.py` writes: (score, bytes) per id."""

    def __init__(self, path):
        b = open(path, "rb").read()
        import struct
        (maxlen,) = struct.unpack("<i", b[:4])
        p, self.ids = 4, []
        while p < len(b):
            sc, ln = struct.unpack("<fi", b[p:p + 8])
            p += 8
            self.ids.append((sc, b[p:p + ln]))
            p += ln
        self.maxlen = maxlen
        self.by_bytes = {}
        for i, (sc, bs) in enumerate(self.ids):
            self.by_bytes.setdefault(bs, i)

    def encode(self, text):
        bs = text.encode("utf-8")
        n = len(bs)
        NEG = -3.0e38
        dp = [NEG] * (n + 1)
        pick = [-1] * (n + 1)
        plen = [0] * (n + 1)
        dp[n] = 0.0
        for i in range(n - 1, -1, -1):
            cap = min(self.maxlen, n - i)
            for L in range(1, cap + 1):
                j = self.by_bytes.get(bs[i:i + L])
                if j is None or dp[i + L] <= NEG:
                    continue
                v = self.ids[j][0] + dp[i + L]
                if v > dp[i]:
                    dp[i], pick[i], plen[i] = v, j, L
            if pick[i] < 0 and dp[i + 1] > NEG:
                dp[i], plen[i] = dp[i + 1], 0
        out, i = [], 0
        while i < n:
            if plen[i]:
                out.append(pick[i])
                i += plen[i]
            else:
                i += 1
        return out

    def decode(self, ids):
        return b"".join(self.ids[i][1] for i in ids).decode("utf-8", "replace")


def greedy_peaks(M, tok, text, max_gen=16):
    """Real-text residual/hidden peaks, for a container's rail check.

    Same source convention as translate() -- a leading space and the source
    <eos> -- and the decoder target is the model's own greedy output.  A random
    vocabulary run is *not* representative of a post-norm block: its first
    attention reads the unscaled embedding, and arbitrary ids drive that
    residual to thousands (4090 on OPUS-MT) where real text reaches 4.  A real
    sentence has its own peak elsewhere -- OPUS-MT's dec1 FFN carries one
    5959 dim at the start token -- and that is a value the model really
    produces, so the calibration keeps it.
    """
    import numpy as np
    src = tok.encode(" " + text) + [int(M["eos_id"])]
    enc_out = encoder(M, np.array(src, dtype=np.int64))
    cur = [int(M["start_id"])]
    for _ in range(max_gen):
        lg = logits_last(M, np.array(cur, dtype=np.int64), enc_out)
        nxt = int(np.argmax(lg))
        if nxt == int(M["eos_id"]):
            break
        cur.append(nxt)
    return residual_peaks(M, src, cur)


def translate(M, tok, text, max_gen=32):
    # The reference appends the source <eos> (id 0) to the source ids: it is
    # the marker the encoder was trained with, and without it the decoder
    # repeats its first token forever (measured on this checkpoint).
    src = tok.encode(" " + text) + [M["eos_id"]]
    enc = encoder(M, np.array(src, dtype=np.int64))
    out, cur = [], [M["start_id"]]
    for _ in range(max_gen):
        lg = logits_last(M, np.array(cur, dtype=np.int64), enc)
        nxt = int(np.argmax(lg))
        if nxt == M["eos_id"]:
            break
        out.append(nxt)
        cur.append(nxt)
    return src, out, tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="model_dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--logits", type=int, default=0,
                    help="print the first N logits of step 1 and a hash")
    ap.add_argument("--max-gen", type=int, default=32)
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    M = load_marian(a.model_dir)
    tok = Tok(a.tok)
    text = " ".join(a.text)
    src, out, res = translate(M, tok, text, a.max_gen)
    print("src ids: %s" % src)
    print("tgt ids: %s" % out)
    print("out    : %s" % res)
    if a.logits:
        enc = encoder(M, np.array(src, dtype=np.int64))
        lg = logits_last(M, np.array([M["start_id"]], dtype=np.int64), enc)
        print("logits[%d]: %s" % (a.logits, np.array2string(
            lg[:a.logits], precision=6, separator=",", max_line_width=1000)))
        print("logits sum %.6f  first-3-dot %.6f" % (float(lg.sum()), float(lg[:3] @ lg[:3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
