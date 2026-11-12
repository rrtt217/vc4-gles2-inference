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

Two additions for the MoE (MMCF v3 / `Qwen3MoeForCausalLM`) checkpoints:

* `--f32` prints the stages as raw IEEE bits — the format `STAGE_DUMP=1
  DUMP_F32=1` produces on the engine's CPU path (`gpu.cpp` `dump_cpu_row`) —
  instead of the q16 texel form.  1/256 is coarser than the drift this model
  shows in its early layers, so a q16 comparison cannot see it.
* a MoE checkpoint also prints one

      GATE l=<layer> t=<position> e=<expert> g=<g0>,<g1>,<g2>,<g3>

  line after each `xbf`: the router logits (= `mlp.gate.weight` @ xbf, the
  post-attention norm) and the argmax expert the engine's `MOE_DUMP` line
  reports.  `norm_topk_prob` with top-1 makes the selected expert's output
  weight exactly 1.0, so the argmax is the whole routing decision.

The MoE path is a whole-sequence forward (one layer at a time, causal attention
over the prefix), which is what makes it correct for prompts longer than one
token.  The dense path below is deliberately left exactly as it was — see the
note at the end of this docstring — because its output is a published baseline
(models/README.md).  Keep the dense path byte-identical when touching this file.

Known limitation, kept for compatibility (reported 2026-11-12): the dense branch
recomputes the prefix attention for every position out of the *already updated*
residual `x`, so for a prompt longer than one token its `att`/`x1`/`xbf`/
`hidA` lines are wrong from the second position on (the prefix K/V are taken
from the position's final, layer-8 state instead of layer l-1).  A one-token
prompt is exact, which is how the tables in models/README.md were measured.
The MoE branch does not have that defect.
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


def dump(name, values, limit, f32=False):
    """One DUMP line, exactly the engine's layout: two q16 per 4-byte texel.

    The engine packs the int16 values **big-endian** (high byte first) — decode
    its dump little-endian and every second value is garbage, which is how a
    first attempt at this comparison produced a bogus 256x scale factor.

    With `f32` the line is what the engine prints under `DUMP_F32=1` on its CPU
    path: the raw IEEE-754 bit pattern of every value, one 8-hex-digit group
    each, after "DUMP <name> w=<count> h=0".  The whole row is printed (the
    engine does too), so `limit` is ignored on that form.
    """
    if f32:
        bits = np.asarray(values, np.float32).view(np.uint32)
        print(" ".join(["DUMP %s w=%d h=0" % (name, len(bits))] +
                       ["%08x" % int(b) for b in bits]))
        return
    q = q16(values)
    n = len(q)
    w = (n + 1) // 2
    out = ["DUMP %s w=%d h=%016x" % (name, w, 0)]
    show = min(w, limit)
    for t in range(show):
        a = int(q[2 * t]) & 0xFFFF
        b = int(q[2 * t + 1]) & 0xFFFF if 2 * t + 1 < n else 0
        out.append("%02x%02x%02x%02x" % (a >> 8, a & 0xFF, b >> 8, b & 0xFF))
    print(" ".join(out))


def silu(v):
    return v / (1.0 + np.exp(-v))


# --- stages the MoE branch needs, also importable by an audit script ---------
#
# The engine runs attention and the expert FFN as *blocks*; these helpers are
# the float32 definitions of those blocks, with no quantisation, so a caller
# can feed them the engine's own dumped input and see whether the block's own
# maths is right independently of any upstream drift.

def qkv_attn(h, wq, wk, wv, qn, kn, eps, theta, half, H, KV, hd):
    """Causal attention over a whole sequence; returns the D-wide mix (pre-wo).

    `h` is the normalised input for every position, so position t attends to
    the first t+1 rows.  GQA map is `head // (H // KV)`, qk-norm is per head
    (Qwen3), rope is applied per position on the head vector.
    """
    T = h.shape[0]
    rep = H // KV
    q = (h @ wq.T).reshape(T, H, hd)
    k = (h @ wk.T).reshape(T, KV, hd)
    v = (h @ wv.T).reshape(T, KV, hd)
    if qn is not None:
        q = q / np.sqrt((q * q).mean(-1, keepdims=True) + eps) * qn
        k = k / np.sqrt((k * k).mean(-1, keepdims=True) + eps) * kn
    q = np.stack([rope(q[i], theta, i, half) for i in range(T)])
    k = np.stack([rope(k[i], theta, i, half) for i in range(T)])
    kk = np.repeat(k, rep, axis=1)
    vv = np.repeat(v, rep, axis=1)
    out = np.zeros((T, H, hd), np.float32)
    for t in range(T):
        sc = np.einsum('hd,jhd->hj', q[t], kk[: t + 1]) / np.sqrt(hd)
        pr = np.exp(sc - sc.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        out[t] = np.einsum('hj,jhd->hd', pr, vv[: t + 1])
    return out.reshape(T, H * hd)


def router_logits(xbf, gate_w):
    """`gate(hidden_states)` of SparseMoeBlock: (T, D) @ (E, D).T -> (T, E)."""
    return np.asarray(xbf, np.float32) @ np.asarray(gate_w, np.float32).T


def expert_hidden(xbf, w1, w3):
    """silu(gate_proj(x)) * up_proj(x); w1 = gate_proj, w3 = up_proj."""
    xbf = np.asarray(xbf, np.float32)
    return silu(xbf @ np.asarray(w1, np.float32).T) * (xbf @ np.asarray(w3, np.float32).T)


def expert_ffn(xbf, w1, w3, w2):
    """One expert's SwiGLU FFN; no routing weight (top-1 + norm_topk_prob: 1.0)."""
    return expert_hidden(xbf, w1, w3) @ np.asarray(w2, np.float32).T


def config_of(cfg):
    """(D, H, KV, hd, L, eps, theta, half) from a HF config.json."""
    D = int(cfg['hidden_size'])
    H = int(cfg['num_attention_heads'])
    KV = int(cfg.get('num_key_value_heads') or H)
    hd = int(cfg.get('head_dim') or D // H)
    L = int(cfg['num_hidden_layers'])
    eps = float(cfg.get('rms_norm_eps') or 1e-5)
    rp = cfg.get('rope_parameters') or {}
    theta = float(cfg.get('rope_theta') or rp.get('rope_theta') or 10000.0)
    half = not (cfg.get('rope_interleaved') or cfg.get('rope_layout') == 'interleaved')
    return D, H, KV, hd, L, eps, theta, half


def is_moe(hdr):
    return any('.experts.' in k for k in hdr)


def moe_stages(get, hdr, cfg, ids, layers=0):
    """Float32 stages of a top-1 MoE checkpoint, one layer at a time.

    Returns a dict of (n_layers, T, ...) arrays: xb/att/x1/xbf/hidA/gate and
    `route` (the argmax expert per layer and position), plus `x` (the final
    residual) so a caller can compute the logits.  Whole-sequence per layer:
    that is the correct causal forward, and it keeps position t's prefix in the
    state layer l-1 actually produced.
    """
    D, H, KV, hd, L, eps, theta, half = config_of(cfg)
    L = layers if layers else L
    qk = 'model.layers.0.self_attn.q_norm.weight' in hdr
    emb = get('model.embed_tokens.weight')
    x = emb[ids].astype(np.float32)
    T = len(ids)
    st = {k: [] for k in ('xb', 'att', 'x1', 'xbf', 'hid', 'hidA', 'gate', 'route')}
    for l in range(L):
        g = lambda n, _l=l: get('model.layers.%d.%s' % (_l, n))
        an, fn = g('input_layernorm.weight'), g('post_attention_layernorm.weight')
        xb = rms(x, an, eps)
        att = qkv_attn(xb, g('self_attn.q_proj.weight'), g('self_attn.k_proj.weight'),
                       g('self_attn.v_proj.weight'),
                       g('self_attn.q_norm.weight') if qk else None,
                       g('self_attn.k_norm.weight') if qk else None,
                       eps, theta, half, H, KV, hd)
        x1 = x + att @ g('self_attn.o_proj.weight').T
        xbf = rms(x1, fn, eps)
        gate = router_logits(xbf, g('mlp.gate.weight'))
        route = np.argmax(gate, axis=-1)
        new_x = x1.copy()
        hid = np.empty((T, int(cfg.get('moe_intermediate_size') or 0)), np.float32)
        cache = {}
        for t in range(T):
            e = int(route[t])
            if e not in cache:      # one expert's three matrices at a time
                cache[e] = (g('mlp.experts.%d.gate_proj.weight' % e),
                            g('mlp.experts.%d.up_proj.weight' % e),
                            g('mlp.experts.%d.down_proj.weight' % e))
            w1, w3, w2 = cache[e]
            hid[t] = expert_hidden(xbf[t:t + 1], w1, w3)[0]
            new_x[t] = x1[t] + (hid[t:t + 1] @ w2.T)[0]
        for k, v in (('xb', xb), ('att', att), ('x1', x1), ('xbf', xbf),
                     ('hid', hid), ('hidA', new_x), ('gate', gate), ('route', route)):
            st[k].append(v)
        x = new_x
    st['x'] = x
    return st


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
    ap.add_argument("--f32", action="store_true",
                    help="DUMP_F32 form: raw IEEE bits instead of q16 texels (the engine "
                         "prints this form on its CPU path under DUMP_F32=1)")
    ap.add_argument("--extra-hidden", action="store_true",
                    help="MoE only: also dump the selected expert's SwiGLU hidden (2432 "
                         "wide), which is where a wrong expert/scale shows up first")
    ap.add_argument("--logits-out", help="MoE only: write the last position's f32 logits "
                                         "(LOGITS_OUT format) so the stages can be checked "
                                         "against a plain ref_forward run")
    a = ap.parse_args()

    get, hdr = load(a.model + "/model.safetensors")
    cfg = json.load(open(a.model + "/config.json"))
    # Only build the tokenizer when the ids are not given: models/minimind3 (the
    # dense MiniMind3 source) ships no tokenizer.json, and --ids is exactly the
    # way to compare against it.
    tok = None
    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.model + "/tokenizer.json")
        ids = tok.encode(a.prompt).ids
    T = len(ids)
    print("# ids %s" % (ids,), file=sys.stderr)

    if is_moe(hdr):
        # Whole-sequence stages, printed in the engine's order: position-major,
        # then layer 0..n, and inside a layer xb, att, x1, xbf, GATE, hidA —
        # the order session.cpp emits STAGE_DUMP, MOE_DUMP and HIDDEN_DUMP in.
        st = moe_stages(get, hdr, cfg, ids, a.layers)
        n = len(st['xb'])
        for t in range(T):
            if a.first_pos and t != T - 1:
                continue
            for l in range(n):
                dump("xb", st['xb'][l][t], a.texels, a.f32)
                dump("att", st['att'][l][t], a.texels, a.f32)
                dump("x1", st['x1'][l][t], a.texels, a.f32)
                dump("xbf", st['xbf'][l][t], a.texels, a.f32)
                g = st['gate'][l][t]
                print("GATE l=%d t=%d e=%d g=%s" % (l, t, int(st['route'][l][t]),
                                                    ",".join("%.9g" % float(v) for v in g)))
                if a.extra_hidden:
                    dump("hid", st['hid'][l][t], a.texels, a.f32)
                dump("hidA", st['hidA'][l][t], a.texels, a.f32)
        if a.logits_out:
            xf = rms(st['x'], get('model.norm.weight'), config_of(cfg)[5])
            logits = xf[-1] @ get('model.embed_tokens.weight').T
            np.asarray(logits, np.float32).astype("<f4").tofile(a.logits_out)
        return

    D, H, KV, hd, L, eps, theta, half = config_of(cfg)
    qk = "model.layers.0.self_attn.q_norm.weight" in hdr
    n_layers = a.layers if a.layers else L

    emb = get("model.embed_tokens.weight")
    x = emb[ids].astype(np.float32)
    rep = H // KV

    # The engine dumps per position, and inside it per layer, so do the same.
    for t in range(T):
        for l in range(n_layers):
            g = lambda n: get("model.layers.%d.%s" % (l, n))
            an, fn = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
            # `x` is the running residual: positions < t have already been
            # through layer l-1, and t itself through l-1 as well, so the
            # normalisation must be recomputed here, not hoisted out of the
            # layer loop (the first version of this tool did hoist it, which
            # made every layer after 0 look broken while layer 0 matched).
            xs = x[: t + 1]
            xb = rms(xs[t], an, eps)
            dump("xb", xb, a.texels, a.f32)
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
            dump("att", att[t].reshape(D), a.texels, a.f32)
            x1row = xs[t] + (att[t].reshape(D) @ g("self_attn.o_proj.weight").T)
            dump("x1", x1row, a.texels, a.f32)
            xbf = rms(x1row, fn, eps)
            dump("xbf", xbf, a.texels, a.f32)
            gg = xbf @ g("mlp.gate_proj.weight").T
            uu = xbf @ g("mlp.up_proj.weight").T
            hid = (gg / (1 + np.exp(-gg))) * uu
            # The engine only dumps the SwiGLU hidden on its CPU FFN path; a
            # model that takes the GPU FFN (any split checkpoint) has no `hid`
            # line, so keep it off by default or the diff is misaligned.
            if a.stages == "cpu":
                dump("hid", hid, a.texels, a.f32)
            x = x.copy()
            x[t] = x1row + hid @ g("mlp.down_proj.weight").T
            dump("hidA", x[t], a.texels, a.f32)


if __name__ == "__main__":
    main()
