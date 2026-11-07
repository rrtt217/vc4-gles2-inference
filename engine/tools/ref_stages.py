#!/usr/bin/env python3
"""Per-stage float32 reference dump, in the engine's own DUMP format.

    ref_stages.py <hf-model-dir> "<prompt text>" [--layers N] [--first-pos]

Prints, for every position and layer, the same stages `STAGE_DUMP=1` prints:

    xb   rms_att(x)                      (the normalised attention input)
    att  the attention block output      (before wo)
    x1   x + wo @ att                    (after the attention residual)
    xbf  rms_ffn(x1)
    hid  silu(gate(xbf)) * up(xbf)       (before w2)
    hidA x1 + w2 @ hid                   (the layer output)

Why this exists: `diff_paths.sh` compares the engine's CPU path with its GPU
path, so it can only say "these two disagree".  Comparing either with a float32
forward says *which stage* left first, and on which layer — the difference
between "the attention kernel is wrong" and "the model was loaded wrong".

The output is `dump_diff.awk`-compatible, so:

    ref_stages.py models/supra2 "hi" > /tmp/ref.txt
    MODEL=... GPU_LAYER=0 STAGE_DUMP=1 ./engine_mm3 --prompt "hi" --gen 0   > /tmp/eng.txt
    awk -f engine/scripts/dump_diff.awk /tmp/ref.txt /tmp/eng.txt

Read the caveat in `diff_paths.sh` first: the engine's CPU path still runs its
matvecs on the GPU with quantised weights, so a few LSB of difference are
expected everywhere.  What matters is where the difference stops being a few
LSB.
"""
import argparse
import json
import struct
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from ref_forward import load, rms


def rope(vec, theta, pos, half=True):
    """Rotate one head vector.  `half` = GPT-NeoX / HF rotate_half."""
    out = np.empty_like(vec)
    hd = vec.shape[-1]
    if half:
        for i in range(hd // 2):
            ang = pos / (theta ** (2.0 * i / hd))
            c, s = np.cos(ang), np.sin(ang)
            out[..., i] = vec[..., i] * c - vec[..., i + hd // 2] * s
            out[..., i + hd // 2] = vec[..., i + hd // 2] * c + vec[..., i] * s
    else:
        for i in range(0, hd, 2):
            ang = pos / (theta ** (i / hd))
            c, s = np.cos(ang), np.sin(ang)
            out[..., i] = vec[..., i] * c - vec[..., i + 1] * s
            out[..., i + 1] = vec[..., i + 1] * c + vec[..., i] * s
    return out


def q16(v):
    return np.clip(np.rint(np.asarray(v, np.float32) * 256.0), -32768, 32767).astype(np.int32)


def dump(name, values, limit):
    """One DUMP line, exactly the engine's layout: two q16 per 4-byte texel."""
    q = q16(values)
    n = len(q)
    w = (n + 1) // 2
    out = ["DUMP %s w=%d h=%016x" % (name, w, 0)]
    show = min(w, limit)
    for t in range(show):
        a = int(q[2 * t]) & 0xFFFF
        b = int(q[2 * t + 1]) & 0xFFFF if 2 * t + 1 < n else 0
        out.append("%02x%02x%02x%02x" % (a & 0xFF, a >> 8, b & 0xFF, b >> 8))
    print(" ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--layers", type=int, default=0, help="stop after N layers (0 = all)")
    ap.add_argument("--first-pos", action="store_true", help="only the last position")
    ap.add_argument("--texels", type=int, default=64, help="texels to print per line")
    ap.add_argument("--stages", choices=["gpu", "cpu"], default="gpu",
                    help="gpu = xb/att/x1/xbf/hidA per layer (what GPU FFN dumps); "
                         "cpu adds the SwiGLU hidden")
    ap.add_argument("--ids", help="comma-separated token ids; skips the tokenizer so the "
                                 "comparison cannot be confused by a prefix-space difference")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    get, hdr = load(a.model + "/model.safetensors")
    cfg = json.load(open(a.model + "/config.json"))
    D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    hd = D // H
    L = cfg["num_hidden_layers"]
    eps = float(cfg.get("rms_norm_eps") or 1e-5)
    rp = cfg.get("rope_parameters") or {}
    theta = float(cfg.get("rope_theta") or rp.get("rope_theta") or 10000.0)
    half = not (cfg.get("rope_interleaved") or cfg.get("rope_layout") == "interleaved")
    tok = Tokenizer.from_file(a.model + "/tokenizer.json")
    ids = [int(x) for x in a.ids.split(",")] if a.ids else tok.encode(a.prompt).ids
    T = len(ids)
    print("# ids %s" % (ids,), file=sys.stderr)

    emb = get("model.embed_tokens.weight")
    x = emb[ids].astype(np.float32)
    n_layers = a.layers if a.layers else L
    rep = H // KV
    qk = "model.layers.0.self_attn.q_norm.weight" in hdr

    # The engine dumps per position, and inside it per layer, so do the same.
    for t in range(T):
        xs = x[: t + 1]
        for l in range(n_layers):
            g = lambda n: get("model.layers.%d.%s" % (l, n))
            an, fn = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
            xb = rms(xs[t], an, eps)
            dump("xb", xb, a.texels)
            h = rms(xs, an, eps)
            q = (h @ g("self_attn.q_proj.weight").T).reshape(t + 1, H, hd)
            k = (h @ g("self_attn.k_proj.weight").T).reshape(t + 1, KV, hd)
            v = (h @ g("self_attn.v_proj.weight").T).reshape(t + 1, KV, hd)
            if qk:
                qn, kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")
                q = q / np.sqrt((q * q).mean(-1, keepdims=True) + eps) * qn
                k = k / np.sqrt((k * k).mean(-1, keepdims=True) + eps) * kn
            q = np.stack([rope(q[i], theta, i, half) for i in range(t + 1)])
            k = np.stack([rope(k[i], theta, i, half) for i in range(t + 1)])
            att = np.zeros((t + 1, H, hd), np.float32)
            for p in range(t + 1):
                for hh in range(H):
                    kh = hh // rep
                    sc = (q[p, hh] @ k[: p + 1, kh].T) / np.sqrt(hd)
                    pr = np.exp(sc - sc.max())
                    pr /= pr.sum()
                    att[p, hh] = pr @ v[: p + 1, kh]
            dump("att", att[t].reshape(D), a.texels)
            x1row = xs[t] + (att[t].reshape(D) @ g("self_attn.o_proj.weight").T)
            dump("x1", x1row, a.texels)
            xbf = rms(x1row, fn, eps)
            dump("xbf", xbf, a.texels)
            gg = xbf @ g("mlp.gate_proj.weight").T
            uu = xbf @ g("mlp.up_proj.weight").T
            hid = (gg / (1 + np.exp(-gg))) * uu
            # The engine only dumps the SwiGLU hidden on its CPU FFN path; a
            # model that takes the GPU FFN (any split checkpoint) has no `hid`
            # line, so keep it off by default or the diff is misaligned.
            if a.stages == "cpu":
                dump("hid", hid, a.texels)
            x = x.copy()
            x[t] = x1row + hid @ g("mlp.down_proj.weight").T
            dump("hidA", x[t], a.texels)


if __name__ == "__main__":
    main()
