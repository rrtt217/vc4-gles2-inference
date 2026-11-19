#!/usr/bin/env python3
"""RWKV-7 (Goose) reference forward pass in numpy, f32 only.

Why this file exists: the engine has no RWKV-7 path yet, and the plan's phase 0
(docs/rwkv7.md) needs a reference that can answer two
questions before any C++ is written -- "what does this model actually compute"
and "which fixed-point format survives it".  This is that reference: a
straight-line, dependency-free (numpy only) transcription of the published
weights, plus the measurement modes the plan's section 4 asks for.

Sources of truth, in this order (all checked against each other):
  * refs/llamacpp/src-b12283/src/models/rwkv7-base.cpp:8-136 -- the operator
    order and every constant the shipped GGUF uses;
  * refs/llamacpp/src-b12283/ggml/.../rwkv_wkv7_f32.c:3-9 -- the recurrence;
  * BlinkDL/RWKV-LM RWKV-v7/rwkv_v7_numpy.py -- the single-token oracle shape
    (fetched to models/rwkv7/ref/rwkv_v7_numpy.py).  NOT
    train_temp/rwkv7_train_simplified.py: its decay is -softplus(z)-0.5, a
    different parameterisation, so a mismatch there is not a bug.
  * fla-hub/rwkv7-0.1B-g1 config.json + model.safetensors (BF16, FLA naming).

Tensor mapping, FLA safetensors -> math (lo = low-rank):
  model.embeddings.weight            emb
  model.layers.{i}.pre_norm.{w,b}    ln0, layer 0 only (norm_first)
  model.layers.{i}.attn_norm.{w,b}   ln1
  model.layers.{i}.ffn_norm.{w,b}    ln2
  model.norm.{w,b}                   ln_out
  attn.x_{r,w,k,v,a,g}               the six token-shift lerp vectors (1,1,d)
  attn.{r,k,v,o}_proj.weight         Wo etc, nn.Linear layout [out, in]
  attn.w_lora.lora.0.weight [r,d]    W_w1 (down);  lora.2.weight [d,r] W_w2 (up)
  attn.w_lora.lora.2.bias [d]        w_bias
  attn.a_lora / v_lora / g_lora      same shape convention; v_lora is absent on
                                     layer 0, which is exactly the "v_first"
                                     case in the oracle (v0 = v, no residual)
  attn.k_k, attn.k_a [d]             per-channel scales
  attn.r_k [H, 64]                   per-head bonus scale
  attn.g_norm.{w,b}                  ln_w / ln_b of the per-head group norm
  ffn.key.weight [F,d]               W_cm_key;   ffn.value.weight [d,F] W_cm_value
  ffn.x_k [d]                        channel-mix lerp vector
  lm_head.weight [vocab,d]           untied output matrix

Usage (all paths are relative to the repository root):
  python3 engine/tools/ref_rwkv.py --chat --prompt "hi" --gen 8 --tokens-out /tmp/t.txt
  python3 engine/tools/ref_rwkv.py --prompt-file p.txt --logits-out /tmp/l.npy
  python3 engine/tools/ref_rwkv.py --measure models/rwkv7/phase0-measure.json
  python3 engine/tools/ref_rwkv.py --quant-sweep models/rwkv7/phase0-quant.json
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

SQRT_E = math.sqrt(math.e)
DECAY_C = 1.0 / SQRT_E          # 0.606531, rwkv7-base.cpp:69
LN_EPS_DEFAULT = 1e-5           # config.json norm_eps
GN_EPS = 64e-5                  # rwkv7-base.cpp:118


# --------------------------------------------------------------------------
# safetensors
# --------------------------------------------------------------------------

def _bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)


def read_safetensors(path, keep=None):
    """Minimal safetensors reader: 8-byte header length, JSON header, data.

    Supports F32/F16/BF16 (the fla-hub checkpoint is BF16).  No torch, no
    safetensors package -- the format is two reads and a frombuffer.
    """
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            if keep is not None and not keep(name):
                continue
            b, e = info["data_offsets"]
            f.seek(base + b)
            raw = f.read(e - b)
            dt = info["dtype"]
            if dt == "BF16":
                a = _bf16_to_f32(raw)
            elif dt == "F16":
                a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dt == "F32":
                a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            else:
                raise SystemExit("unsupported dtype %s in %s" % (dt, name))
            out[name] = np.ascontiguousarray(a).reshape(info["shape"])
    return out


# --------------------------------------------------------------------------
# tokenizer: greedy longest match over the World vocab, no merges
# --------------------------------------------------------------------------

_ESC = {"n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34, "0": 0, "a": 7,
        "b": 8, "f": 12, "v": 11}


def unescape_entry(s):
    r"""Parse one vocab entry the way hf_rwkv_tokenizer.py's eval() would.

    The file stores Python repr()s: '\x00', b'\xfe', '\t\t', '   '.  We accept
    both str and bytes literals and the escape forms that actually occur in
    rwkv_vocab_v20230424.txt (\xNN, \t, \n, \r, \\, \').
    """
    is_bytes = s[:2] in ("b'", 'b"')
    if is_bytes:
        s = s[1:]
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    # A str literal's \xNN is a CODEPOINT that is then UTF-8 encoded; a bytes
    # literal's is a raw byte.  The file mixes both: id 2422 is b'\xbf\xbd' (2
    # bytes) while id 2423 is '\x80' (also 2 bytes, U+0080 -> c2 80).  Reading
    # the second as a byte silently builds a 1-byte entry and every id after it
    # is still right, so only the length assert catches it.
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(s):
            break
        e = s[i]
        if e == "x":
            out.append(chr(int(s[i + 1:i + 3], 16)))
            i += 3
        elif e == "u":
            out.append(chr(int(s[i + 1:i + 5], 16)))
            i += 5
        elif e == "U":
            out.append(chr(int(s[i + 1:i + 9], 16)))
            i += 9
        elif e in _ESC:
            out.append(chr(_ESC[e]))
            i += 1
        else:
            out.append(e)
            i += 1
    txt = "".join(out)
    if is_bytes:
        return bytes(ord(ch) for ch in txt)
    return txt.encode("utf-8")


class RwkvVocab:
    """id<->bytes, plus the greedy encoder.

    The reference (hf_rwkv_tokenizer.RWKV_TOKENIZER) walks a byte trie and takes
    the longest match.  A dict probe from the longest possible length down to 1
    is the same function with a different data structure: same ids, and it needs
    no trie, which is what the plan asks the packer to imitate
    (docs/rwkv7.md).
    """

    def __init__(self, vocab_file):
        self.idx2tok = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                sp1 = line.index(" ")
                sp2 = line.rindex(" ")
                idx = int(line[:sp1])
                tok = unescape_entry(line[sp1 + 1:sp2])
                assert len(tok) == int(line[sp2 + 1:]), (idx, tok)
                self.idx2tok[idx] = tok
        self.tok2idx = {t: i for i, t in self.idx2tok.items()}
        self.maxlen = max(len(t) for t in self.tok2idx)
        # id 0 is the added token, not in the file; the reference maps it via
        # added_tokens_decoder to this text (tokenizer_config.json).
        self.idx2tok.setdefault(0, b"<|rwkv_tokenizer_end_of_text|>")

    def encode(self, text):
        src = text.encode("utf-8") if isinstance(text, str) else text
        ids = []
        i = 0
        n = len(src)
        while i < n:
            m = min(self.maxlen, n - i)
            while m >= 1:
                j = self.tok2idx.get(src[i:i + m])
                if j is not None:
                    ids.append(j)
                    i += m
                    break
                m -= 1
            else:
                # Cannot happen for a byte-complete vocab; keep it loud.
                raise SystemExit("no vocab entry covers byte %r at %d" % (src[i], i))
        return ids

    def decode(self, ids):
        return b"".join(self.idx2tok.get(int(i), b"<?%d?>" % int(i)) for i in ids)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def layernorm(x, w, b, eps=LN_EPS_DEFAULT):
    """O1: mean+var over the last axis (biased var, as ggml_norm)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def groupnorm_head(y, ln_w, ln_b, n_head, eps=GN_EPS):
    """O15: per-head LayerNorm over the 64 channels of each head."""
    t = y.shape[0]
    z = y.reshape(t, n_head, -1)
    mu = z.mean(axis=-1, keepdims=True)
    var = ((z - mu) ** 2).mean(axis=-1, keepdims=True)
    z = (z - mu) / np.sqrt(var + eps)
    return z.reshape(t, -1) * ln_w + ln_b


def l2norm_head(kk, n_head, eps=1e-12):
    """O11: per-head L2 normalisation, eps 1e-12 (rwkv7-base.cpp:93)."""
    t = kk.shape[0]
    z = kk.reshape(t, n_head, -1)
    nrm = np.sqrt((z.astype(np.float64) ** 2).sum(axis=-1, keepdims=True))
    return (z / np.maximum(nrm, eps)).astype(np.float32).reshape(t, -1)


def sigmoid(x):
    # g1d's activations are large enough to overflow exp() on the negative side;
    # the result is the same (0.0), the warning is not.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def stats(a):
    a = np.asarray(a, dtype=np.float32).ravel()
    aa = np.abs(a)
    return {
        "n": int(a.size),
        "max": float(aa.max()),
        "mean_abs": float(aa.mean()),
        "rms": float(np.sqrt((a.astype(np.float64) ** 2).mean())),
        "p99": float(np.percentile(aa, 99.0)),
        "p999": float(np.percentile(aa, 99.9)),
    }


# --- fixed point helpers: one function, three formats ---------------------

def qfix(x, step, lo=None, hi=None):
    """Round to a fixed-point grid, optionally clamped to a rail.

    step/rail pairs used below are the engine's own formats
    (docs/engine-internals.md:421 I16_2 = q16 = step 1/256, rail +-128;
     plan line 38 I24_1 = step 1/256, rail +-32768).
    """
    q = np.round(np.asarray(x, dtype=np.float32) / np.float32(step))
    if lo is not None:
        q = np.clip(q, lo, hi)
    return (q * np.float32(step)).astype(np.float32)


Q16 = dict(step=1.0 / 256.0, lo=-32768, hi=32767)          # I16_2
I24 = dict(step=1.0 / 256.0, lo=-8388608, hi=8388607)      # I24_1
W_ABS = dict(step=1.0 / 256.0, lo=0, hi=32767)             # W-1 (a)/(c)
W_1M = dict(step=1.0 / 65536.0, lo=0, hi=65535)            # W-1 (b), on 1-w
W_24B = dict(step=1.0 / 16777216.0, lo=0, hi=16777215)     # W-1 (d) 24-bit abs
Q16 = dict(step=1.0 / 256.0, lo=-32768, hi=32767)          # I16_2
I24 = dict(step=1.0 / 256.0, lo=-8388608, hi=8388607)      # I24_1
W_ABS = dict(step=1.0 / 256.0, lo=0, hi=32767)             # W-1 (a)/(c)
W_1M = dict(mode="w1m", step=1.0 / 65536.0)                # W-1 (b), on 1-w
W_24B = dict(step=1.0 / 16777216.0, lo=0, hi=16777215)     # W-1 (d) 24-bit abs
KK_Q16 = dict(step=1.0 / 256.0, lo=-32768, hi=32767)       # K-1 (a)
KK_X8 = dict(mode="x8", step=1.0 / 256.0, lo=-32768, hi=32767)  # K-1 (b)


def qapply(x, cfg):
    """One tensor through one candidate format.

    mode "plain": round to step, clamp to the rail (I16_2 / I24_1 / abs forms).
    mode "w1m"  : store 1-x instead of x (the W-1 (b) proposal).
    mode "x8"   : pre-scale by 8, store, scale back (the K-1 (b) proposal).
    """
    mode = cfg.get("mode", "plain")
    step, lo, hi = cfg["step"], cfg.get("lo"), cfg.get("hi")
    if mode == "w1m":
        return (1.0 - qfix(1.0 - x, step, lo, hi)).astype(np.float32)
    if mode == "x8":
        return (qfix(x * 8.0, step, lo, hi) * np.float32(0.125)).astype(np.float32)
    return qfix(x, step, lo, hi)

# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class Rwkv7:
    """f32 RWKV-7.  forward() handles both the full-sequence and the
    single-token incremental case: one code path, so "prefill == N steps" is a
    property of the code rather than a hope.  --check-incremental asserts it."""

    def __init__(self, model_dir, verbose=False):
        cfg = json.loads(open(os.path.join(model_dir, "config.json")).read())
        self.dir = model_dir
        self.d = int(cfg["hidden_size"])
        self.L = int(cfg["num_hidden_layers"])
        self.F = int(cfg["intermediate_size"])
        self.hs = int(cfg["head_dim"])
        self.H = self.d // self.hs
        self.vocab = int(cfg["vocab_size"])
        self.eps = float(cfg.get("norm_eps", LN_EPS_DEFAULT))
        self.cfg = cfg
        path = os.path.join(model_dir, "model.safetensors")
        W = read_safetensors(path)
        self.emb = W["model.embeddings.weight"]
        self.head = W["lm_head.weight"]
        self.norm_w = W["model.norm.weight"]
        self.norm_b = W["model.norm.bias"]
        g = lambda n: W[n]
        self.ln0_w = self.ln0_b = None
        self.ln1w, self.ln1b, self.ln2w, self.ln2b = [], [], [], []
        self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g = [], [], [], [], [], []
        self.Wr, self.Wk, self.Wv, self.Wo = [], [], [], []
        self.Ww1, self.Ww2, self.w_b = [], [], []
        self.Wa1, self.Wa2, self.a_b = [], [], []
        self.Wv1, self.Wv2, self.v_b = [], [], []
        self.Wg1, self.Wg2 = [], []
        self.k_k, self.k_a, self.r_k = [], [], []
        self.ln_w, self.ln_b = [], []
        self.Wk_cm, self.Wv_cm, self.mix_k = [], [], []
        for i in range(self.L):
            p = "model.layers.%d." % i
            self.ln1w.append(g(p + "attn_norm.weight"))
            self.ln1b.append(g(p + "attn_norm.bias"))
            self.ln2w.append(g(p + "ffn_norm.weight"))
            self.ln2b.append(g(p + "ffn_norm.bias"))
            if i == 0:
                self.ln0_w = g(p + "pre_norm.weight")
                self.ln0_b = g(p + "pre_norm.bias")
            for lst, nm in ((self.x_r, "x_r"), (self.x_w, "x_w"), (self.x_k, "x_k"),
                            (self.x_v, "x_v"), (self.x_a, "x_a"), (self.x_g, "x_g")):
                lst.append(g(p + "attn." + nm).reshape(-1))
            self.Wr.append(g(p + "attn.r_proj.weight"))
            self.Wk.append(g(p + "attn.k_proj.weight"))
            self.Wv.append(g(p + "attn.v_proj.weight"))
            self.Wo.append(g(p + "attn.o_proj.weight"))
            self.Ww1.append(g(p + "attn.w_lora.lora.0.weight"))
            self.Ww2.append(g(p + "attn.w_lora.lora.2.weight"))
            self.w_b.append(g(p + "attn.w_lora.lora.2.bias"))
            self.Wa1.append(g(p + "attn.a_lora.lora.0.weight"))
            self.Wa2.append(g(p + "attn.a_lora.lora.2.weight"))
            self.a_b.append(g(p + "attn.a_lora.lora.2.bias"))
            if i == 0:
                self.Wv1.append(None); self.Wv2.append(None); self.v_b.append(None)
            else:
                self.Wv1.append(g(p + "attn.v_lora.lora.0.weight"))
                self.Wv2.append(g(p + "attn.v_lora.lora.2.weight"))
                self.v_b.append(g(p + "attn.v_lora.lora.2.bias"))
            if (p + "attn.g_lora.lora.0.weight") in W:
                self.Wg1.append(g(p + "attn.g_lora.lora.0.weight"))
                self.Wg2.append(g(p + "attn.g_lora.lora.2.weight"))
            else:
                self.Wg1.append(None); self.Wg2.append(None)
            self.k_k.append(g(p + "attn.k_k"))
            self.k_a.append(g(p + "attn.k_a"))
            self.r_k.append(g(p + "attn.r_k").reshape(self.H, self.hs))
            self.ln_w.append(g(p + "attn.g_norm.weight"))
            self.ln_b.append(g(p + "attn.g_norm.bias"))
            self.Wk_cm.append(g(p + "ffn.key.weight"))
            self.Wv_cm.append(g(p + "ffn.value.weight"))
            self.mix_k.append(g(p + "ffn.x_k").reshape(-1))
        self.q = {}
        if verbose:
            sys.stderr.write("rwkv7: d=%d L=%d F=%d heads=%d hs=%d vocab=%d eps=%g "
                             "ranks w/a/v/g=%d/%d/%d/%d\n" % (
                                 self.d, self.L, self.F, self.H, self.hs, self.vocab,
                                 self.eps, self.Ww1[0].shape[0], self.Wa1[0].shape[0],
                                 self.Wv1[-1].shape[0], self.Wg1[0].shape[0]))

    # -- state -------------------------------------------------------------
    def new_state(self):
        return {
            "S": np.zeros((self.L, self.H, self.hs, self.hs), np.float32),
            "att": np.zeros((self.L, self.d), np.float32),
            "ffn": np.zeros((self.L, self.d), np.float32),
        }

    def _q(self, name, x):
        """Apply the format under test to one tensor, if this run quantises it."""
        cfg = self.q.get(name)
        if cfg is None:
            return x
        return qapply(x, cfg)

    # -- forward -----------------------------------------------------------
    def forward(self, tokens, state=None, dump=None, all_logits=False):
        """tokens: sequence of ids.  Returns (logits, state).

        logits is (T, vocab) when all_logits else (vocab,) for the last
        position.  dump, if a dict, collects per-stage statistics keyed
        'L00.name' (the plan's O39 dump hook, in Python).
        """
        T = len(tokens)
        if T == 0:
            raise SystemExit("empty token sequence")
        st = state if state is not None else self.new_state()
        x = self.emb[np.asarray(tokens, dtype=np.int64)].astype(np.float32)
        v_first = None
        for i in range(self.L):
            if i == 0:
                x = layernorm(x, self.ln0_w, self.ln0_b, self.eps)
            tag = "L%02d." % i
            xn = layernorm(x, self.ln1w[i], self.ln1b[i], self.eps)
            if dump is not None:
                dump[tag + "ln1"] = stats(xn)
            xp = np.empty_like(xn)
            xp[0] = st["att"][i]
            xp[1:] = xn[:-1]
            sx = xp - xn
            xr = xn + sx * self.x_r[i]
            xw = xn + sx * self.x_w[i]
            xk = xn + sx * self.x_k[i]
            xv = xn + sx * self.x_v[i]
            xa = xn + sx * self.x_a[i]
            xg = xn + sx * self.x_g[i]
            r = xr @ self.Wr[i].T
            w = np.exp(-sigmoid(np.tanh(xw @ self.Ww1[i].T) @ self.Ww2[i].T
                                + self.w_b[i]) * DECAY_C)
            k = xk @ self.Wk[i].T
            v = xv @ self.Wv[i].T
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * sigmoid(xv @ self.Wv1[i].T @ self.Wv2[i].T
                                               + self.v_b[i])
            a = sigmoid(xa @ self.Wa1[i].T @ self.Wa2[i].T + self.a_b[i])
            if self.Wg1[i] is not None:
                g = sigmoid(xg @ self.Wg1[i].T) @ self.Wg2[i].T
            else:
                g = None
            kk = k * self.k_k[i]
            k = k + (a - 1.0) * k * self.k_a[i]
            kk = l2norm_head(kk, self.H)
            if dump is not None:
                dump[tag + "w"] = stats(w)
                dump[tag + "kk"] = stats(kk)
                dump[tag + "a"] = stats(a)
                dump[tag + "v"] = stats(v)
                dump[tag + "r"] = stats(r)
                if g is not None:
                    dump[tag + "g"] = stats(g)
            if self.q.get("av") is not None:
                a = self._q("av", a)
                v = self._q("av", v)
                if g is not None:
                    g = self._q("av", g)
            kk = self._q("kk", kk)
            kk3 = kk.reshape(T, self.H, self.hs)
            a3 = a.reshape(T, self.H, self.hs)
            k3 = k.reshape(T, self.H, self.hs)
            v3 = v.reshape(T, self.H, self.hs)
            w3 = w.reshape(T, self.H, self.hs)
            r3 = r.reshape(T, self.H, self.hs)
            S = st["S"][i].copy()
            y = np.empty((T, self.H, self.hs), np.float32)
            for t in range(T):
                wt = self._q("w", w3[t])
                sa = np.matmul(S, -kk3[t][..., None])[..., 0]
                S = S * wt[:, None, :] + v3[t][:, :, None] * k3[t][:, None, :] \
                    + sa[:, :, None] * (kk3[t] * a3[t])[:, None, :]
                S = self._q("state", S)
                y[t] = np.matmul(S, r3[t][..., None])[..., 0]
            st["S"][i] = S
            if dump is not None:
                dump[tag + "state"] = stats(S)
                dump[tag + "state_headmax"] = [float(np.abs(S[h]).max())
                                              for h in range(self.H)]
            y = groupnorm_head(y.reshape(T, self.d), self.ln_w[i], self.ln_b[i],
                               self.H)
            rk = (k3 * r3 * self.r_k[i][None]).sum(axis=-1)      # (T,H)
            y = y + v * np.repeat(rk, self.hs, axis=1)
            if g is not None:
                y = y * g
            elif dump is not None:
                pass
            y = y @ self.Wo[i].T
            x = self._q("resid", x + y)
            st["att"][i] = xn[-1]
            if dump is not None:
                dump[tag + "x_after_att"] = stats(x)
            xn2 = layernorm(x, self.ln2w[i], self.ln2b[i], self.eps)
            xp2 = np.empty_like(xn2)
            xp2[0] = st["ffn"][i]
            xp2[1:] = xn2[:-1]
            xk2 = xn2 + (xp2 - xn2) * self.mix_k[i]
            h = xk2 @ self.Wk_cm[i].T
            if dump is not None:
                dump[tag + "cm_hidden"] = stats(h)
            h = np.maximum(h, 0.0)
            h = h * h
            if dump is not None:
                dump[tag + "cm_sqrelu"] = stats(h)
                dump[tag + "cm_sqrelu_clamped"] = {
                    "frac_at_rail_q16": float(np.mean(h > 128.0)),
                    "frac_at_rail_i24": float(np.mean(h > 32768.0)),
                }
            h = self._q("sqrelu", h)
            y2 = h @ self.Wv_cm[i].T
            x = self._q("resid", x + y2)
            st["ffn"][i] = xn2[-1]
            if dump is not None:
                dump[tag + "x_after_ffn"] = stats(x)
                if i == 0:
                    dump[tag + "v_first"] = stats(v_first)
        x = layernorm(x, self.norm_w, self.norm_b, self.eps)
        if dump is not None:
            dump["final_ln"] = stats(x)
        logits = x @ self.head.T
        if dump is not None:
            dump["logits"] = stats(logits)
        if not all_logits:
            logits = logits[-1]
            x = x[-1:]
        return logits, st

    # -- convenience -------------------------------------------------------
    def forward_seq(self, tokens, **kw):
        return self.forward(list(tokens), **kw)

    def step(self, token, state, **kw):
        """Single-token incremental API."""
        return self.forward([int(token)], state, **kw)

    def greedy(self, prompt_tokens, n, dump=None):
        """Greedy decode.  Returns (generated ids, state, per-step logits cos
        against the argmax margin)."""
        logits, st = self.forward(list(prompt_tokens), None)
        out = []
        for _ in range(n):
            tok = int(np.argmax(logits))
            out.append(tok)
            logits, st = self.step(tok, st)
        return out, st, logits
# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

SPECIAL_TEXT = "<|rwkv_tokenizer_end_of_text|>"


def render_chat(user, bos=True, think=True):
    """The fla-hub chat template, verbatim.

    tokenizer_config.json / the GGUF's tokenizer.chat_template both end with
    {% if add_generation_prompt %}{% if enable_thinking is defined and
    enable_thinking == False %}'Assistant: <think></think>'{% else %}'Assistant:
    <think'{% endif %}{% endif %} -- i.e. the default generation prompt leaves an
    OPEN <think tag, not the bare 'Assistant:' the plan's A5 row quotes.  The
    model closes it and answers; a prompt that stops at 'Assistant:' is off
    template.
    """
    head = SPECIAL_TEXT if bos else ""
    tail = "Assistant: <think></think>" if not think else "Assistant: <think"
    return "%sUser: %s\n\n%s" % (head, user, tail)


def encode_with_special(vocab, text):
    """Encode, mapping the literal <|rwkv_tokenizer_end_of_text|> to id 0 the
    way PreTrainedTokenizer's added-token pass does (tokenizer_config.json
    added_tokens_decoder{0}).  Without this the template's BOS becomes 26 byte
    tokens."""
    ids = []
    rest = text
    while True:
        k = rest.find(SPECIAL_TEXT)
        if k < 0:
            ids += vocab.encode(rest)
            return ids
        ids += vocab.encode(rest[:k])
        ids.append(0)
        rest = rest[k + len(SPECIAL_TEXT):]


def load_prompt(args, vocab):
    text = args.prompt
    if args.prompt_file:
        text = open(args.prompt_file, "r", encoding="utf-8").read()
    if text is None:
        text = "User: What is the capital of France?" if not args.chat else \
            "Hello, my name is"
    if args.chat:
        text = render_chat(text, bos=not args.no_bos)
    return text, encode_with_special(vocab, text)


def token_stream(args, vocab):
    """The token stream a PPL run uses: a file, or the prompt."""
    if args.ppl:
        raw = open(args.ppl, "r", encoding="utf-8").read()
        ids = vocab.encode(raw)
        if args.ppl_tokens:
            ids = ids[:args.ppl_tokens + 1]
        return ids
    _, ids = load_prompt(args, vocab)
    return ids


def nll_of(model, ids, lo, hi):
    """Teacher-forced mean NLL of ids[lo:hi] given ids[:lo] as context."""
    logits, _ = model.forward(ids[:hi], None, all_logits=True)
    z = logits[lo - 1:hi - 1].astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    tgt = np.asarray(ids[lo:hi], dtype=np.int64)
    lp = logp[np.arange(len(tgt)), tgt]
    return float(-lp.mean()), float(-lp.sum())


def do_measure(model, vocab, ids, text):
    dump = {}
    logits, st = model.forward(ids, None, dump=dump, all_logits=True)
    per_layer = []
    for i in range(model.L):
        tag = "L%02d." % i
        S = st["S"][i]
        headmax = [float(np.abs(S[h]).max()) for h in range(model.H)]
        d = {
            "layer": i,
            "ln1_max": dump[tag + "ln1"]["max"],
            "resid_after_att": dump[tag + "x_after_att"]["max"],
            "resid_after_ffn": dump[tag + "x_after_ffn"]["max"],
            "w_min": float(dump[tag + "w"]["max"]),   # max of |w| = max w
            "kk_max": dump[tag + "kk"]["max"],
            "kk_mean_abs": dump[tag + "kk"]["mean_abs"],
            "kk_p99": dump[tag + "kk"]["p99"],
            "a_mean": None,
            "v_max": dump[tag + "v"]["max"],
            "r_max": dump[tag + "r"]["max"],
            "state_max": float(np.abs(S).max()),
            "state_mean_abs": float(np.abs(S).mean()),
            "state_headmax_max": max(headmax),
            "state_headmax_min": min(headmax),
            "state_headmax": headmax,
            "cm_hidden_max": dump[tag + "cm_hidden"]["max"],
            "cm_sqrelu_max": dump[tag + "cm_sqrelu"]["max"],
            "cm_sqrelu_frac_over_128": dump[tag + "cm_sqrelu_clamped"]["frac_at_rail_q16"],
            "cm_sqrelu_frac_over_32768": dump[tag + "cm_sqrelu_clamped"]["frac_at_rail_i24"],
        }
        if (tag + "g") in dump:
            d["g_max"] = dump[tag + "g"]["max"]
        per_layer.append(d)
    res = {
        "model": model.dir,
        "prompt_text": text,
        "prompt_tokens": len(ids),
        "layers": model.L,
        "resid_max": max(max(p["resid_after_att"], p["resid_after_ffn"]) for p in per_layer),
        "resid_final_ln_max": dump["final_ln"]["max"],
        "logits": dump["logits"],
        "state_absmax": max(p["state_max"] for p in per_layer),
        "sqrelu_peak": max(p["cm_sqrelu_max"] for p in per_layer),
        "cm_hidden_peak": max(p["cm_hidden_max"] for p in per_layer),
        "kk_absmax": max(p["kk_max"] for p in per_layer),
        "kk_mean_abs_min": min(p["kk_mean_abs"] for p in per_layer),
        "kk_mean_abs_max": max(p["kk_mean_abs"] for p in per_layer),
        "w_min_of_max": min(p["w_min"] for p in per_layer),
        "per_layer": per_layer,
        "dump": dump,
    }
    return res


def print_measure(res):
    print("prompt: %d tokens  |  %r" % (res["prompt_tokens"], res["prompt_text"][:70]))
    print("%3s %10s %10s %10s %10s %10s %8s %10s %10s" % (
        "L", "ln1max", "resid@att", "resid@ffn", "|S|max", "hdmax_spr", "w_max",
        "sqrelu_pk", "cmhid_pk"))
    for p in res["per_layer"]:
        print("%3d %10.3f %10.3f %10.3f %10.3f %10.2f %8.5f %10.3f %10.3f" % (
            p["layer"], p["ln1_max"], p["resid_after_att"], p["resid_after_ffn"],
            p["state_max"], p["state_headmax_max"] / max(p["state_headmax_min"], 1e-9),
            p["w_min"], p["cm_sqrelu_max"], p["cm_hidden_max"]))
    print("resid max %.3f | state |max| %.3f | sqrelu peak %.1f | kk mean|.| %.4f..%.4f "
          "| kk |max| %.4f" % (res["resid_max"], res["state_absmax"],
                               res["sqrelu_peak"], res["kk_mean_abs_min"],
                               res["kk_mean_abs_max"], res["kk_absmax"]))
    print("q16 rail 128 (resid) / 32768 (I24): resid over rail = %s, sqrelu over 128 = %s"
          % (res["resid_max"] > 128.0, res["sqrelu_peak"] > 128.0))


def do_quant_sweep(model, ids, args):
    """Measure each candidate format by what it costs at the output.

    Protocol: the token stream is the prompt plus a fixed 24-token greedy
    continuation produced by the f32 model, so every variant is scored on the
    same target and the comparison is teacher-forced.  Metrics per variant:
      dNLL  -- mean NLL(continuation) minus the f32 value, in nats/token
      cos   -- cosine of the last-position logits against f32
      dmax  -- max |logit - logit_f32|
      top1  -- does the last-position argmax agree
      greedy-- length of the greedy prefix that agrees with f32 (8 tokens)
    """
    model.q = {}
    cont = 24
    logits_all, st0 = model.forward(ids, None, all_logits=True)
    top1 = int(np.argmax(logits_all[-1]))
    st = st0
    gen = []
    last = logits_all[-1]
    for _ in range(cont):
        tok = int(np.argmax(last))
        gen.append(tok)
        last, st = model.step(tok, st)
    stream = list(ids) + gen
    lo = len(ids)
    hi = len(stream)
    nll_ref, _ = nll_of(model, stream, lo, hi)
    logits_ref, _ = model.forward(stream, None, all_logits=True)
    ref_last = logits_ref[-1].astype(np.float64)
    _s = np.sort(ref_last)[::-1]
    ref_p1 = float(np.exp(_s[0] - np.log(np.exp(ref_last - _s[0]).sum()) - _s[0]))
    ref_gap = float(_s[0] - _s[1])
    ref_top1 = int(np.argmax(ref_last))
    # f32 greedy from the full stream
    ref_gen = []
    lg, stt = model.forward(stream, None)
    for _ in range(8):
        t = int(np.argmax(lg))
        ref_gen.append(t)
        lg, stt = model.step(t, stt)

    variants = [
        ("f32 baseline", {}),
        ("W-1(a) w @1/256 abs", {"w": W_ABS}),
        ("W-1(b) 1-w @1/65536", {"w1m": W_1M}),
        ("W-1(c) w I24_1 1/256", {"w": I24}),
        ("W-1(d) w 24-bit abs", {"w": W_24B}),
        ("K-1(a) kk @1/256", {"kk": KK_Q16}),
        ("K-1(b) kk x8 then I16_2", {"kkx8": KK_X8}),
        ("S-1(b) state I24_1 1/256", {"state": I24}),
        ("S-1(c) state q16 1/256 +-128", {"state": Q16}),
        ("Q-1 sqrelu q16 +-128", {"sqrelu": Q16}),
        ("Q-1 sqrelu I24 1/256", {"sqrelu": I24}),
        ("A-1 a/v/g q16", {"av": Q16}),
        ("R-1 resid q16 +-128", {"resid": Q16}),
    ]
    rows = []
    for name, q in variants:
        model.q = q
        nll, _ = nll_of(model, stream, lo, hi)
        lg_all, _ = model.forward(stream, None, all_logits=True)
        last = lg_all[-1].astype(np.float64)
        v1 = last / (np.linalg.norm(last) + 1e-30)
        v0 = ref_last / (np.linalg.norm(ref_last) + 1e-30)
        # greedy prefix match
        g, s2 = model.forward(stream, None)
        got = []
        for _ in range(8):
            t = int(np.argmax(g))
            got.append(t)
            g, s2 = model.step(t, s2)
        match = 0
        for x, y in zip(got, ref_gen):
            if x != y:
                break
            match += 1
        rows.append({
            "variant": name,
            "dnll": nll - nll_ref,
            "nll": nll,
            "cos": float(np.dot(v1, v0)),
            "dmax": float(np.abs(last - ref_last).max()),
            "top1": int(np.argmax(last)) == ref_top1,
            "p1_f32": ref_p1,
            "top2gap_f32": ref_gap,
            "greedy_prefix": match,
            "greedy_ids": got,
        })
        print("%-26s dNLL %+.6f  nll %.4f  cos %.8f  dmax %.4f  top1 %-5s  greedy %d/8"
              % (name, rows[-1]["dnll"], nll, rows[-1]["cos"], rows[-1]["dmax"],
                 rows[-1]["top1"], match))
    model.q = {}
    print("context: p1(f32) at the scored position = %.4f, top2 gap = %.4f nats "
          "-- a gap below the dmax column means the argmax is a coin flip"
          % (ref_p1, ref_gap))
    return {"prompt_tokens": len(ids), "cont_tokens": cont, "nll_f32": nll_ref,
            "p1_f32": ref_p1, "top2gap_f32": ref_gap,
            "rows": rows, "ref_greedy": ref_gen, "continuation": gen}
# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="numpy f32 RWKV-7 reference forward")
    ap.add_argument("--model", default="models/rwkv7/0.1B-g1")
    ap.add_argument("--vocab", default=None, help="default: <model>/rwkv_vocab_v20230424.txt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--chat", action="store_true", help="wrap in the HF chat template")
    ap.add_argument("--no-bos", action="store_true", help="chat template without the leading token 0")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to generate")
    ap.add_argument("--tokens-out", default=None, help="write greedy ids, one per line")
    ap.add_argument("--logits-out", default=None, help="write last-position logits .npy")
    ap.add_argument("--all-logits-out", default=None, help="write (T,vocab) logits .npy")
    ap.add_argument("--dump-stages", default=None, help="per-stage stats, JSON")
    ap.add_argument("--state-out", default=None, help="final state, .npz")
    ap.add_argument("--state-in", default=None)
    ap.add_argument("--check-incremental", action="store_true")
    ap.add_argument("--measure", default=None, help="write the phase-0 measurement JSON")
    ap.add_argument("--quant-sweep", default=None, help="write the quantisation sweep JSON")
    ap.add_argument("--ppl", default=None, help="text file for a NLL/PPL run")
    ap.add_argument("--ppl-tokens", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    vocab_file = args.vocab or os.path.join(args.model, "rwkv_vocab_v20230424.txt")
    vocab = RwkvVocab(vocab_file)
    t0 = __import__("time").time()
    model = Rwkv7(args.model, verbose=not args.quiet)
    if not args.quiet:
        sys.stderr.write("rwkv7: loaded in %.2f s\n" % (__import__("time").time() - t0))
    if args.info:
        print(json.dumps({"dir": model.dir, "d": model.d, "L": model.L, "F": model.F,
                          "heads": model.H, "head_size": model.hs,
                          "vocab": model.vocab, "eps": model.eps,
                          "ranks": {"w": model.Ww1[0].shape[0],
                                    "a": model.Wa1[0].shape[0],
                                    "v": model.Wv1[-1].shape[0],
                                    "g": model.Wg1[0].shape[0]},
                          "vocab_entries": len(vocab.idx2tok),
                          "vocab_max_token_len": vocab.maxlen}, indent=1))

    text, ids = load_prompt(args, vocab)
    if args.ppl:
        ids = token_stream(args, vocab)
    if not args.quiet:
        print("prompt: %d tokens: %r" % (len(ids), text[:100]))
        print("ids: %s" % ids[:40])

    state = None
    if args.state_in:
        z = np.load(args.state_in)
        state = {"S": z["S"], "att": z["att"], "ffn": z["ffn"]}

    dump = {} if args.dump_stages else None
    logits, state = model.forward(ids, state, dump=dump, all_logits=bool(args.all_logits_out))
    last = logits if logits.ndim == 1 else logits[-1]
    if not args.quiet:
        top = np.argsort(-last)[:args.topk]
        print("top-%d: %s" % (args.topk, [(int(t), round(float(last[t]), 3),
                                           vocab.decode([int(t)])[:16]) for t in top]))

    if args.logits_out:
        np.save(args.logits_out, last.astype(np.float32))
    if args.all_logits_out:
        np.save(args.all_logits_out, logits.astype(np.float32))
    if args.dump_stages:
        json.dump(dump, open(args.dump_stages, "w"), indent=1)
    if args.state_out:
        np.savez(args.state_out, S=state["S"], att=state["att"], ffn=state["ffn"])

    if args.check_incremental:
        lg2 = []
        s2 = model.new_state()
        for t in ids:
            z, s2 = model.step(t, s2)
            lg2.append(z)
        lg2 = np.asarray(lg2, np.float32)
        full = logits if logits.ndim == 2 else np.asarray([last])
        if logits.ndim == 1:
            d = float(np.abs(lg2[-1] - last).max())
            print("check-incremental: last logits max|diff| = %.3e" % d)
        else:
            d = float(np.abs(lg2 - logits).max())
            print("check-incremental: all logits max|diff| = %.3e (T=%d)" % (d, len(ids)))

    if args.ppl:
        nll, tot = nll_of(model, ids, 1, len(ids))
        print("ppl: %d tokens, mean NLL %.5f nats -> ppl %.4f (sum %.3f)"
              % (len(ids), nll, math.exp(nll), tot))

    if args.measure:
        res = do_measure(model, vocab, ids, text)
        print_measure(res)
        res.pop("dump", None)
        json.dump(res, open(args.measure, "w"), indent=1)
        print("wrote %s" % args.measure)

    if args.quant_sweep:
        rows = do_quant_sweep(model, ids, args)
        json.dump(rows, open(args.quant_sweep, "w"), indent=1)
        print("wrote %s" % args.quant_sweep)

    if args.gen:
        gen = []
        state = None if not args.state_out else None
        # contaminate nothing: re-run the prompt to get a fresh state
        lg, st = model.forward(ids, None)
        for _ in range(args.gen):
            tok = int(np.argmax(lg))
            gen.append(tok)
            lg, st = model.step(tok, st)
        print("greedy[%d]: %s" % (len(gen), gen))
        print("greedy text: %r" % vocab.decode(gen).decode("utf-8", "replace"))
        if args.tokens_out:
            with open(args.tokens_out, "w") as f:
                f.write("\n".join(str(t) for t in gen) + "\n")
            print("wrote %s" % args.tokens_out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))



