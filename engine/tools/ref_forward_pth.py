#!/usr/bin/env python3
"""A numpy forward pass over a torch-zip checkpoint with a non-HF block.

    ref_forward_pth.py models/miniyuanyuan "你好，你是谁？"

`ref_forward.py` answers "is the source model sane" for a safetensors directory
of a Llama/Qwen-shaped block: it reads tensors by HF name and runs RMSNorm ->
GQA -> SwiGLU.  A checkpoint that publishes a torch zip *and* a block the
engine has no kernel for cannot use it, and the question is the same one: does
the f32 source predict a sensible token?  If it does, the weights and the cfg
are fine and anything the engine gets wrong afterwards is the port's bug.

Two things this adds over `ref_forward.py`:

- **The zip and its wrapper.**  `pth_reader.load(..., meta=True)` reads a
  `{"model": state_dict, "cfg": {...}}` export and hands back both halves, so
  the architecture config travels with the weights instead of a config.json
  that such a release usually does not ship.
- **A pluggable block.**  `--arch` picks the FFN, because the whole point is a
  block the engine does not have: `bicross` is the 12-layer BiCrossGate FFN
  (`q1,q2 = q(x).chunk(2)`; `gelu(q1*x)` and `silu(q2*x)` concatenated, gated
  against `up(x)`), and `swiglu` is the ordinary one, so a checkpoint that
  turns out to be a plain transformer can still be checked here.

Greedy decoding is included because one logits row is not evidence: the token
sequence a model produces is what a port is judged against, and `--logits-out`
writes the same `<f4` vocabulary dump `LOGITS_OUT=` writes so
`engine/scripts/cmp_logits.py` can diff the two tile by tile.
"""

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import pth_reader  # noqa: E402


# ---------------------------------------------------------------- numerics
def rmsnorm(x, w, eps):
    ss = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    return (x / np.sqrt(ss + eps)).astype(np.float32) * w


def gelu_erf(x):
    """PyTorch's default F.gelu: the exact erf form, not the tanh approximation.

    The difference is not academic here -- the operand reaches |15|, far past
    the range where tanh(0.7979(x+0.044715x^3)) matches, so a port checked
    against the approximation would be checked against the wrong function.
    """
    return (0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def rope_half(x, cos, sin):
    """Rotate-half, the convention `rope_half=1` means in the container.

    The reference applies `chunk(2)` on the head dim, so pair i rotates against
    i + head_dim/2 -- the half-split form, not llama2's adjacent pairs.
    """
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def freqs(head_dim, positions, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    ang = positions[:, None].astype(np.float32) * inv[None, :]
    return np.cos(ang), np.sin(ang)


# ---------------------------------------------------------------- the block
def ffn_swiglu(h, w1, w3, w2):
    g = h @ w1.T
    return ((g / (1.0 + np.exp(-g))) * (h @ w3.T)) @ w2.T


def ffn_bicross(h, q, gate_proj, up_proj, down_proj, act1="gelu", act2="silu"):
    """The block this tool exists for.

        q1, q2 = q(h).chunk(2)
        a = act1(q1 * h); c = act2(q2 * h)
        return down_proj(gate_proj(cat(a, c)) * up_proj(h))
    """
    d = h.shape[-1]
    qq = h @ q.T
    q1, q2 = qq[..., :d], qq[..., d:]
    a = gelu_erf(q1 * h) if act1 == "gelu" else silu(q1 * h)
    c = gelu_erf(q2 * h) if act2 == "gelu" else silu(q2 * h)
    gate = np.concatenate([a, c], -1) @ gate_proj.T
    return (gate * (h @ up_proj.T)) @ down_proj.T


# ---------------------------------------------------------------- the pass
def forward(ids, t, cfg, arch="bicross", stats=None):
    d = int(cfg["hidden_size"])
    L = int(cfg["num_hidden_layers"])
    eps = float(cfg["rms_norm_eps"])
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    grp = nh // nkv
    share = int(cfg.get("ffn_share_group") or 1)
    T = len(ids)

    x = t["model.embed_tokens.weight"][ids].astype(np.float32)
    pos = np.arange(T)
    c, s = freqs(hd, pos, float(cfg["rope_theta"]))
    c, s = c[:, None, :], s[:, None, :]        # broadcast over heads

    for l in range(L):
        g = l // share
        xn = rmsnorm(x, t["model.layers.%d.input_layernorm.weight" % l], eps)
        q = (xn @ t["model.layers.%d.self_attn.q_proj.weight" % l].T).reshape(T, nh, hd)
        k = (xn @ t["model.layers.%d.self_attn.k_proj.weight" % l].T).reshape(T, nkv, hd)
        v = (xn @ t["model.layers.%d.self_attn.v_proj.weight" % l].T).reshape(T, nkv, hd)
        if bool(cfg.get("use_qk_norm", True)) or (
                "model.layers.%d.self_attn.q_norm.weight" % l) in t:
            q = rmsnorm(q, t["model.layers.%d.self_attn.q_norm.weight" % l], eps)
            k = rmsnorm(k, t["model.layers.%d.self_attn.k_norm.weight" % l], eps)
        q = rope_half(q, c, s)
        k = rope_half(k, c, s)
        kx = np.repeat(k, grp, axis=1)
        vx = np.repeat(v, grp, axis=1)
        sc = np.einsum("thd,shd->hts", q, kx) / math.sqrt(hd)
        sc = sc + np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
        o = np.einsum("hts,shd->thd", softmax(sc), vx).reshape(T, nh * hd)
        x = x + o @ t["model.layers.%d.self_attn.o_proj.weight" % l].T
        if stats is not None:
            stats.setdefault("residual", []).append(float(np.abs(x).max()))
        h = rmsnorm(x, t["model.layers.%d.post_attention_layernorm.weight" % l], eps)
        if arch == "bicross":
            pre = "model.ffn_groups.%d" % g
            y = ffn_bicross(h, t[pre + ".q.weight"], t[pre + ".gate_proj.weight"],
                            t[pre + ".up_proj.weight"], t[pre + ".down_proj.weight"],
                            str(cfg.get("qbilinear_act1") or "gelu"),
                            str(cfg.get("qbilinear_act2") or "silu"))
        else:
            y = ffn_swiglu(h, t["model.layers.%d.mlp.gate_proj.weight" % l],
                           t["model.layers.%d.mlp.up_proj.weight" % l],
                           t["model.layers.%d.mlp.down_proj.weight" % l])
        x = x + y
        if stats is not None:
            stats.setdefault("residual", []).append(float(np.abs(x).max()))
    xn = rmsnorm(x, t["model.norm.weight"], eps)
    if stats is not None:
        stats["final_rms_peak"] = float(np.abs(xn).max())
    head = t.get("lm_head.weight")
    if head is None:
        head = t["model.embed_tokens.weight"]
    return xn @ head.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="a torch zip (.pt/.bin), flat or {model,cfg} wrapped")
    ap.add_argument("text", nargs="?", help="prompt; rendered through --template")
    ap.add_argument("--arch", default="bicross", choices=("bicross", "swiglu"))
    ap.add_argument("--ids", help="comma-separated token ids; skips the tokenizer")
    ap.add_argument("--vocab", help="a vocab_v2.json to tokenize with (greedy longest match)")
    ap.add_argument("--prompt", help="text wrapped in --bos/--user/--sep/--assistant")
    ap.add_argument("--template", default="{bos}{user}{text}{sep}{assistant}",
                    help="literal wrapper; {bos}{eos}{user}{assistant}{sep}{text} are substituted")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to generate")
    ap.add_argument("--stop", default="", help="comma-separated ids that end generation")
    ap.add_argument("--logits-out", help="write float32 vocabulary logits (LOGITS_OUT format)")
    ap.add_argument("--stats", action="store_true", help="print per-layer residual peaks and the lane sums")
    a = ap.parse_args()

    t, rest = pth_reader.load(a.ckpt, meta=True)
    cfg = rest.get("cfg") or {}
    if not cfg:
        raise SystemExit("%s: no cfg in the wrapper; pass a checkpoint that carries one" % a.ckpt)
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta"):
        if k not in cfg:
            raise SystemExit("%s: cfg has no %r" % (a.ckpt, k))

    tk = None
    specials = {}
    if a.vocab:
        import json
        import re
        v = json.load(open(a.vocab))
        toks = v["tokens"]
        stoi = {s: i for i, s in enumerate(toks)}
        specials = dict(v["specials"])
        maxlen = max(len(s) for s in toks if s not in specials)
        sp_re = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))
        ws = {" ": stoi["<space>"], "\n": stoi["<newline>"], "\t": stoi["<space>"]}

        def span(text):
            out, i = [], 0
            while i < len(text):
                if text[i] in ws:
                    out.append(ws[text[i]])
                    i += 1
                    continue
                for n in range(min(maxlen, len(text) - i), 0, -1):
                    j = stoi.get(text[i:i + n])
                    if j is not None:
                        out.append(j)
                        i += n
                        break
                else:
                    out.append(stoi["<unk>"])
                    i += 1
            return out

        def encode(text):
            out, p = [], 0
            for m in sp_re.finditer(text):
                if m.start() > p:
                    out += span(text[p:m.start()])
                out.append(specials[m.group(0)])
                p = m.end()
            if p < len(text):
                out += span(text[p:])
            return out

        tk = (toks, encode)

    if a.ids:
        ids = [int(v) for v in a.ids.split(",") if v.strip()]
    elif tk is not None and (a.prompt or a.text):
        names = {"bos": "<bos>", "eos": "<eos>", "user": "<user>",
                 "assistant": "<assistant>", "sep": "<sep>"}
        text = a.template.format(text=a.prompt if a.prompt is not None else a.text, **names)
        ids = tk[1](text)
    else:
        raise SystemExit("pass --ids or --prompt/--text together with --vocab")

    stats = {} if a.stats else None
    lg = forward(ids, t, cfg, a.arch, stats)
    seq = list(ids)
    stop = set(int(v) for v in a.stop.split(",") if v.strip())

    for _ in range(a.gen):
        nxt = int(np.argmax(lg[-1]))
        seq.append(nxt)
        if nxt in stop:
            break
        lg = forward(seq, t, cfg, a.arch, None)

    if a.logits_out:
        np.asarray(lg[-1], np.float32).astype("<f4").tofile(a.logits_out)

    out_ids = seq[len(ids):]
    print("ids:", ids)
    print("arch:", a.arch, "| layers:", cfg["num_hidden_layers"],
          "| d:", cfg["hidden_size"], "| i:", cfg.get("intermediate_size"),
          "| share:", cfg.get("ffn_share_group", 1))
    if stats:
        r = stats["residual"]
        print("residual peaks:", " ".join("%.1f" % v for v in r))
        print("residual max: %.1f (q16 rail 128.0)" % max(r))
    if a.gen:
        if tk is not None:
            print("greedy:", repr("".join(tk[0][i] for i in out_ids)))
        else:
            print("greedy ids:", out_ids)
    top = np.argsort(-lg[-1])[:5]
    print("next top5:", [(int(i), round(float(lg[-1][i]), 2)) for i in top])
    if tk is not None:
        print("next top5 text:", [tk[0][int(i)] for i in top])


if __name__ == "__main__":
    main()
