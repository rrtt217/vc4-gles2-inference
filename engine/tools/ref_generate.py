#!/usr/bin/env python3
"""Greedy float32 generation from an HF source, as the oracle for a container.

    ref_generate.py <hf-model-dir> "<user message>" [--system TEXT] [--gen N]
                    [--template FILE] [--show-prompt]

Prints the reply the same way `engine_mm3 --chat` does, so a container's output
can be compared with the unquantised model case by case.  `ref_forward.py` gives
you the first token and its top-5; this gives the whole sentence, which is what
the battery in `engine/scripts/supra2-battery.sh` compares.

The prompt is rendered from the same template file the container carries
(`--template`, default `templates/<name>.jinja` beside the model dir is not
assumed: pass it), so a template difference shows up as a prompt difference
rather than as a numerics difference.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from ref_forward import load, rms


def rotate(a, theta, half=True):
    """a: (..., hd) -> rotated, position applied by the caller."""
    return a


def rope_at(v, pos, theta, half=True):
    hd = v.shape[-1]
    o = np.empty_like(v)
    if half:
        for i in range(hd // 2):
            ang = pos / (theta ** (2.0 * i / hd))
            c, s = np.cos(ang), np.sin(ang)
            o[..., i] = v[..., i] * c - v[..., i + hd // 2] * s
            o[..., i + hd // 2] = v[..., i + hd // 2] * c + v[..., i] * s
    else:
        for i in range(0, hd, 2):
            ang = pos / (theta ** (i / hd))
            c, s = np.cos(ang), np.sin(ang)
            o[..., i] = v[..., i] * c - v[..., i + 1] * s
            o[..., i + 1] = v[..., i + 1] * c + v[..., i] * s
    return o


class Model:
    def __init__(self, path):
        self.get, self.hdr = load(path + "/model.safetensors")
        c = json.load(open(path + "/config.json"))
        self.cfg = c
        self.D = c["hidden_size"]
        self.H = c["num_attention_heads"]
        self.KV = c["num_key_value_heads"]
        self.hd = c.get("head_dim") or self.D // self.H
        self.L = c["num_hidden_layers"]
        self.eps = float(c.get("rms_norm_eps") or 1e-5)
        rp = c.get("rope_parameters") or {}
        self.theta = float(c.get("rope_theta") or rp.get("rope_theta") or 10000.0)
        self.half = not (c.get("rope_interleaved") or c.get("rope_layout") == "interleaved")
        self.qk = "model.layers.0.self_attn.q_norm.weight" in self.hdr
        self.emb = self.get("model.embed_tokens.weight")
        self.g = [dict(layer=l) for l in range(self.L)]

    def _w(self, l, n):
        return self.get("model.layers.%d.%s" % (l, n))

    def forward(self, tokens, cache, pos):
        """Append `tokens` at `pos`; `cache` holds K/V for everything before.

        One code path for both the prefill (many tokens, empty cache) and a
        decode step (one token, full cache): the cache is always the history and
        `tokens` is what is new, so the causal mask is simply position <= pos+t.
        """
        xs = self.emb[tokens].astype(np.float32)
        T = len(tokens)
        for l in range(self.L):
            an, fn = self._w(l, "input_layernorm.weight"), self._w(l, "post_attention_layernorm.weight")
            h = xs / np.sqrt((xs * xs).mean(-1, keepdims=True) + self.eps) * an
            q = (h @ self._w(l, "self_attn.q_proj.weight").T).reshape(T, self.H, self.hd)
            k = (h @ self._w(l, "self_attn.k_proj.weight").T).reshape(T, self.KV, self.hd)
            v = (h @ self._w(l, "self_attn.v_proj.weight").T).reshape(T, self.KV, self.hd)
            if self.qk:
                q = q / np.sqrt((q * q).mean(-1, keepdims=True) + self.eps) * self._w(l, "self_attn.q_norm.weight")
                k = k / np.sqrt((k * k).mean(-1, keepdims=True) + self.eps) * self._w(l, "self_attn.k_norm.weight")
            for t in range(T):
                q[t] = rope_at(q[t], pos + t, self.theta, self.half)
                k[t] = rope_at(k[t], pos + t, self.theta, self.half)
            cache[l][0] = np.concatenate([cache[l][0], k], 0)
            cache[l][1] = np.concatenate([cache[l][1], v], 0)
            kk, vv = cache[l]
            rep = self.H // self.KV
            att = np.zeros((T, self.H, self.hd), np.float32)
            for t in range(T):
                upto = pos + t + 1
                for hh in range(self.H):
                    kh = hh // rep
                    sc = (q[t, hh] @ kk[:upto, kh].T) / np.sqrt(self.hd)
                    pr = np.exp(sc - sc.max())
                    pr /= pr.sum()
                    att[t, hh] = pr @ vv[:upto, kh]
            x1 = xs + att.reshape(T, self.D) @ self._w(l, "self_attn.o_proj.weight").T
            h2 = x1 / np.sqrt((x1 * x1).mean(-1, keepdims=True) + self.eps) * fn
            gg = h2 @ self._w(l, "mlp.gate_proj.weight").T
            uu = h2 @ self._w(l, "mlp.up_proj.weight").T
            xs = x1 + ((gg / (1 + np.exp(-gg))) * uu) @ self._w(l, "mlp.down_proj.weight").T
        last = xs[-1]
        last = last / np.sqrt((last * last).mean() + self.eps) * self.get("model.norm.weight")
        return last @ self.emb.T


def render(template, system, user):
    try:
        import jinja2
    except ImportError:
        raise SystemExit("need jinja2 to render the template (pip install jinja2)")
    tpl = open(template).read()
    # our template files carry a leading provenance comment block
    if tpl.lstrip().startswith("{#"):
        # our template files carry a provenance comment; strip the comment *and*
        # the newline after it, or jinja renders a leading blank line and every
        # comparison is against a prompt one token longer than the engine's.
        tpl = tpl[tpl.index("#}") + 2:].lstrip("\n")
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    return env.from_string(tpl).render(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--template", required=True)
    ap.add_argument("--show-prompt", action="store_true")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.model + "/tokenizer.json")
    text = render(a.template, a.system, a.prompt)
    if a.show_prompt:
        print("# prompt %r" % text, file=sys.stderr)
    ids = tok.encode(text).ids
    M = Model(a.model)
    cache = [[np.zeros((0, M.KV, M.hd), np.float32), np.zeros((0, M.KV, M.hd), np.float32)]
             for _ in range(M.L)]
    out = []
    logits = M.forward(ids, cache, 0)
    pos = len(ids)
    stop = {2, 6}   # </s> and <|im_end|>
    for _ in range(a.gen):
        nxt = int(np.argmax(logits))
        if nxt in stop:
            break
        out.append(nxt)
        logits = M.forward([nxt], cache, pos)
        pos += 1
    s = tok.decode(out, skip_special_tokens=False)
    print("Assistant: " + (s[1:] if s.startswith(" ") else s))


if __name__ == "__main__":
    main()
