#!/usr/bin/env python3
"""HuggingFace safetensors -> the tensor list an MMCF container is built from.

This is the only quantizer left in the repository.  The old
`engine/quantize/make_{i16,q8_from_i16,q4,q4_from_i16}.py` scripts wrote the
legacy checkpoint layouts (I16M1, Q8M1, Q4M1) that the engine no longer reads,
so they are gone; `mmcf_pack.py` builds the container directly and never writes
an intermediate file.

The arithmetic is unchanged, which is what lets a container packed from
safetensors be byte-identical to one packed from the legacy file it replaces:

  i16  per-matrix symmetric scale, `max|w|/32767`, rounded
  q8   i16 first, dequantised, then `max|w|/127` -- that two-step order is why
       the shipped q8 containers look the way they do
  q4   block-32 asymmetric (min/scale) from the float weights; embeddings and
       logits stay q8
"""
import json
import math
import os
import struct

import numpy as np

from mmcf_grid import LOGITS_GRID
# The container's own tables for the RWKV-7 (v4) ids and metadata key 16.
# They live in mmcf_format.py so the packer, the dump tool and the reader
# name the same numbers (the K_ names below are the packer's spelling).
from mmcf_format import (  # noqa: E402
    K_RWKV_A1, K_RWKV_A2, K_RWKV_ATT_NORM, K_RWKV_ATT_NORM_2, K_RWKV_BIAS,
    K_RWKV_CM_LERP, K_RWKV_G1, K_RWKV_G2, K_RWKV_LERP, K_RWKV_LN_X,
    K_RWKV_OUT_NORM, K_RWKV_SCALE, K_RWKV_TOK_NORM, K_RWKV_V1, K_RWKV_V2,
    K_RWKV_W1, K_RWKV_W2)

DT_F32, DT_I16, DT_Q8I8, DT_Q4, DT_Q6, DT_T3 = 0, 1, 3, 4, 6, 7
T_EMB, T_EMB_SCALE, T_RMS_ATT, T_RMS_FFN, T_RMS_FINAL = 1, 2, 3, 4, 5
T_Q_NORM, T_K_NORM = 6, 7
T_QKV, T_WO, T_W1, T_W3, T_W13, T_W2 = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
# Version 5 (encoder-decoder): the encoder stack, the cross-attention, and the
# f32 rows a decoder-only block never needed.  Same names as mmcf_format.KINDS.
(T_ENC_QKV, T_ENC_WO, T_ENC_W1, T_ENC_W2) = (0x31, 0x32, 0x33, 0x34)
(T_CROSS_Q, T_CROSS_K, T_CROSS_V, T_CROSS_O) = (0x35, 0x36, 0x37, 0x38)
T_BIAS_DIM, T_BIAS_FFN, T_LOGITS_BIAS, T_POS = 0x39, 0x3A, 0x3B, 0x3C
T_ROUTER = 0x16   # the per-layer MoE gate: rows = experts, cols = dim
# LFM2's "conv" blocks (the Liquid implicit convolution): a 3*dim projection,
# a depthwise causal kernel, and a dim projection back.  A reader that predates
# them ignores the ids and runs attention on a conv layer, which fails on the
# missing qkv instead of answering wrongly -- the layer-type metadata (key 14)
# is what makes a hybrid file loadable at all.
T_CONV_IN, T_CONV_W, T_CONV_OUT = 0x17, 0x18, 0x19
T_LOGITS = 0x20
T_EMB_I16 = 8


def qmat(w):
    mx = float(np.abs(w).max())
    if mx == 0:
        mx = 1.0
    s = mx / 32767.0
    q = np.clip(np.rint(w / s), -32767, 32767).astype(np.int16)
    return q, float(s)


def q8mat(w):
    mx = float(np.abs(w).max())
    if mx == 0:
        mx = 1.0
    s = mx / 127.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q, float(s)


def q4mat(w):
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    if cols % 32:
        raise SystemExit(f"q4 cols must be divisible by 32: {rows}x{cols}")
    group = 32
    groups = (cols + group - 1) // group
    pad = groups * group - cols
    x = np.pad(w, ((0, 0), (0, pad))) if pad else w
    z = x.reshape(rows, groups, group)
    lo = z.min(axis=2, keepdims=True)
    hi = z.max(axis=2, keepdims=True)
    sc = (hi - lo) / 15.0
    sc[sc == 0] = 1.0
    q = np.clip(np.rint((z - lo) / sc), 0, 15).astype(np.uint8)
    q = q.reshape(rows, groups * group)[:, :cols]
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    flat_sc = sc[:, :, 0]
    flat_lo = lo[:, :, 0]
    ps = float(max(np.abs(flat_sc).max(), np.abs(flat_lo).max(), 1e-12) / 32767.0)
    s16 = np.clip(np.rint(flat_sc / ps), -32767, 32767).astype('<i2')
    m16 = np.clip(np.rint(flat_lo / ps), -32767, 32767).astype('<i2')
    return packed, s16, m16, ps, groups


def row_ternary(w, rtol=1e-3):
    """(is every row {0, +-s}, worst relative residual) for a 2-D matrix.

    `q8mat` is exact on a per-row ternary matrix (the row scale IS the row's
    magnitude, so +-s maps to +-127); a 2-bit body is what makes it small.  The
    check is per row because the scale is per row: a matrix whose rows carry
    different magnitudes is fine, one whose *row* holds two magnitudes is not
    ternary at all.
    """
    a = np.abs(np.asarray(w, np.float32))
    mx = a.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(a - mx) / np.maximum(mx, 1e-30)
    ok = (a == 0) | (rel <= rtol)
    worst = float(np.where(a == 0, 0.0, rel).max()) if a.size else 0.0
    return bool(ok.all()), worst


def t3mat(w):
    """A per-row ternary matrix -> 2-bit codes, with the decode checked.

    Layout: 16 weights an RGBA8 texel, four a byte, two bits a weight, bit 0
    magnitude and bit 1 sign, so a weight is `mag * (1 - 2*sign)` and a decode
    needs no compare and no branch.

    The four bytes of a texel are **two rows by two column groups**, not four
    column groups of one row: texel (c, t) carries row 2t columns 8c..8c+3 (b0),
    row 2t columns 8c+4..8c+7 (b1), and the same two for row 2t+1 (b2, b3).  A
    kernel that computes two output rows a fragment then uses all four bytes it
    fetched -- 16 MACs from one weight fetch -- where the row-major reading threw
    three of them away: using four bytes of one row would need four activation
    fetches from one sampler, and VC4 returns undefined data for that once the
    weight texture is tall.  Measured on the board, ns/MAC: 9.73 for the
    row-major reading against 1.23 for this one (probes/t3mv_probe, the t3p arm).

    The texture is `ceil(cols/8) x ceil(rows/2)` texels -- half as tall as the
    matrix, which also keeps every shape this model has under VC4's 2048-texel
    limit.  `cols` pads to 8 and `rows` to 2 with zeros, which decode to zero
    weights.

    Returns (packed bytes, unit, per-row plane).  `unit` is the matrix's largest
    row scale and the plane is `srow / unit` -- the same convention `q8_per_row`
    writes, so the engine's existing per-row scale path reads it unchanged.
    """
    w = np.asarray(w, np.float32)
    ok, worst = row_ternary(w)
    if not ok:
        raise SystemExit("t3mat: this matrix is not per-row ternary (worst row "
                         "residual %.4g of its own max); a 2-bit body is only "
                         "written for weights that already are" % worst)
    rows, cols = w.shape
    # The container's scale is the *step* a q8 level multiplies (max_row/127),
    # not the row's magnitude: the levels are the same +-127 a q8 body carries,
    # and the engine's kernels reconstruct `level * scale * plane`.  Writing the
    # magnitude here instead made every weight 127x too large.
    s = np.abs(w).max(axis=1) / 127.0
    s[s == 0] = 1.0
    lev = np.rint(w / (s[:, None] * 127.0))
    code = (np.abs(lev) > 0.5).astype(np.uint8) | ((lev < -0.5).astype(np.uint8) << 1)
    pad = (-cols) % 8
    if pad:
        code = np.pad(code, ((0, 0), (0, pad)))
    if rows % 2:                                     # a texel carries a row pair
        code = np.pad(code, ((0, 1), (0, 0)))
    packed = _t3_pack(code)
    unit = float(s.max())
    back = unpack_t3(packed.tobytes(), rows, cols) * (s[:, None] * 127.0)
    err = float(np.abs(back - w).max()) if w.size else 0.0
    if err > 1e-6 * max(1.0, float(np.abs(w).max())):
        raise SystemExit("t3mat: pack/unpack is not exact (max |diff| %.3g): this "
                         "matrix is not representable as 2-bit ternary" % err)
    return packed.tobytes(), unit, (s / unit).astype(np.float32)


def _t3_pack(code):
    """(rows, cols) 2-bit codes in [0,3] -> the byte block t3mat writes.

    Column-major inside a texel on purpose: b0,b1 are row 2t's two column
    groups and b2,b3 are row 2t+1's, so the block's row `t` is texel row `t` and
    the engine uploads it without a repack.
    """
    rows, cols = code.shape
    q = code.reshape(rows // 2, 2, cols // 8, 2, 4)      # pair, which row, group, half, cols
    b = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6))
    return b.transpose(0, 2, 1, 3).reshape(rows // 2, -1).astype(np.uint8)


def _t3_unpack(block, rows, cols):
    """The inverse of _t3_pack: (rows, cols) codes from the byte block."""
    q = np.frombuffer(block, np.uint8).reshape(rows // 2, cols // 8, 2, 2)
    q = q.transpose(0, 2, 1, 3)                          # pair, which row, group, half
    out = np.empty((rows // 2, 2, cols // 8, 2, 4), np.uint8)
    for k in range(4):
        out[..., k] = (q >> (2 * k)) & 3
    return out.reshape(rows, cols)


def unpack_t3(packed, rows, cols):
    """The decode, for t3mat's self-check; the engine's copy is model.cpp's as_t3."""
    rpad = rows + (rows % 2)
    cpad = cols + ((-cols) % 8)
    code = _t3_unpack(packed, rpad, cpad)
    flat = code[:rows, :cols].astype(np.float32)
    mag = flat % 2.0
    sgn = np.floor(flat * 0.5) % 2.0
    return mag * (1.0 - 2.0 * sgn)


def q6mat(w):
    """Signed 6-bit, per-row symmetric, as two planes.

    The value is split into a 4-bit low plane (two nibbles per byte) and a
    2-bit high plane (four values per byte): the GPU cannot fetch two texels of
    one sampler in a fragment on VC4, so the two halves must live in two
    textures.  Storage is exactly 6 bits a value; the per-row scale is the
    embedding's existing f32 scale vector.
    """
    w = np.asarray(w, np.float32)
    if w.ndim == 1:
        lo, hi, s = q6mat(w[None, :])
        return lo[0], hi[0], s[0]
    rows, cols = w.shape
    if cols % 4:
        raise SystemExit("q6 cols must be divisible by 4")
    s = np.abs(w).max(axis=1, keepdims=True) / 31.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -32, 31).astype(np.int8)
    v = (q.astype(np.int16) & 0x3F).astype(np.uint16)
    # uint16 shifts: a uint8 << 4 wraps and silently drops the high nibble of
    # the odd value (the first pack was wrong by exactly that).
    # Only the low four bits go in the low plane: the top two bits live in the
    # high plane, and leaving them in the nibble ORs into the neighbour.
    lo = (((v[:, 0::2] & 0xF) | ((v[:, 1::2] & 0xF) << 4)) & 0xFF).astype(np.uint8)
    # The high plane carries the *top two bits* of each value (v >> 4), not the
    # whole 6-bit code: shifting the full value put v's low nibble into the
    # neighbour's field and decoded to garbage.
    h = (v >> 4).astype(np.uint16)
    hi = ((h[:, 0::4] | (h[:, 1::4] << 2) | (h[:, 2::4] << 4) |
           (h[:, 3::4] << 6)) & 0xFF).astype(np.uint8)                # 4 per byte
    return lo.reshape(rows, cols // 2), hi.reshape(rows, cols // 4), s[:, 0]


def rope_of(cfg):
    """(theta, half) from a HF config; half means "rotate half the head".

    transformers 5 moved the theta into `rope_parameters` (Qwen3 configs use
    it), while older configs put it at the top level or inside `rope_scaling`.
    Reading only the top-level key silently gives 1e6, which is the difference
    between a working feng30m and a model that rambles.
    """
    theta = float(cfg.get("rope_theta", 10000.0))
    for key in ("rope_parameters", "rope_scaling"):
        rs = cfg.get(key) or {}
        if isinstance(rs, dict) and "rope_theta" in rs:
            theta = float(rs["rope_theta"])
    half = 0 if cfg.get("rope_interleaved") or cfg.get("rope_layout") == "interleaved" else 1
    return theta, half


def rope_factors(cfg):
    """The per-dimension rope frequency table a `longrope` model needs, or None.

    MiniCPM4 (and the BitCPM ternary QAT releases of it) divides rope pair i's
    frequency by `factor[i]`: 32 different bases on a 64-wide head, which a
    single theta cannot express, so the table travels in metadata key 18.

    Only the *static* table is representable.  The class computes
    `scaling_factor = sqrt(1 + log(max_pos/orig_max_pos)/log(orig_max_pos))` and
    picks short_factor up to `original_max_position_embeddings`, long_factor past
    it, so the container can carry it only when the model does not stretch
    (max_position == original_max_position, i.e. scaling_factor 1) and the two
    tables agree.  Everything else is refused here rather than silently packed
    with the wrong angles: a rope that changes with the sequence length is a
    lookup the container has no room for.

    BitCPM-CANN-0.5B states both conditions (short == long, 32768 == 32768), so
    its table is 32 fixed numbers.
    """
    rs = cfg.get("rope_scaling") or {}
    if not isinstance(rs, dict):
        return None
    if (rs.get("rope_type") or rs.get("type")) != "longrope":
        return None
    short = rs.get("short_factor")
    longf = rs.get("long_factor")
    if not short or not longf:
        raise SystemExit("rope_scaling longrope without short_factor/long_factor")
    if len(short) != len(longf):
        raise SystemExit("rope_scaling longrope: short_factor is %d long, long_factor %d"
                         % (len(short), len(longf)))
    if any(abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(a)))
           for a, b in zip(short, longf)):
        raise SystemExit(
            "rope_scaling longrope: the short and long tables differ, so the rope "
            "angles change with the sequence length.  The container carries one "
            "static table (metadata key 18); this model needs two.")
    orig = rs.get("original_max_position_embeddings")
    mx = cfg.get("max_position_embeddings")
    if orig and mx and int(mx) != int(orig):
        raise SystemExit(
            "rope_scaling longrope: max_position_embeddings %s != "
            "original_max_position_embeddings %s, so the model also multiplies "
            "cos/sin by a length-dependent scaling_factor; the container carries "
            "one static table." % (mx, orig))
    return [float(x) for x in short]


def sp_prefix_of(model_dir):
    """True when the tokenizer's word marker is SentencePiece's `\u2581`.

    That decides how the prompt is tokenised.  The reference normalizer runs on
    *every* text run between added tokens, so each run starts with the marker,
    while the engine's historical behaviour puts one at the front of the whole
    prompt.  On a chat prompt the difference is real -- "user" instead of
    "\u2581user" after the first `<|im_start|>`, measured as logits cosine 0.912
    -- so the container records the mode (metadata key 10, third byte) instead
    of the engine guessing.

    Detection is from the tokenizer's own pieces, not from the model name: a
    normalizer that prepends or substitutes the marker, a decoder that turns it
    back into a space, or a SentencePiece `.model` next to the weights.
    """
    def has_marker(o):
        if isinstance(o, str):
            return '\u2581' in o
        if isinstance(o, dict):
            return any(has_marker(v) for v in o.values())
        if isinstance(o, list):
            return any(has_marker(v) for v in o)
        return False

    path = model_dir + '/tokenizer.json'
    if os.path.exists(path):
        try:
            with open(path) as f:
                tj = json.load(f)
        except (ValueError, OSError):
            tj = None
        if tj:
            for key in ('normalizer', 'pre_tokenizer', 'decoder'):
                if has_marker(tj.get(key)):
                    return True
            pieces = list((tj.get('model') or {}).get('vocab') or {})
            if any('\u2581' in tok for tok in pieces[:2000]):
                return True
    return os.path.exists(model_dir + '/tokenizer.model')


def st_dtype(e):
    """safetensors dtype -> numpy, so F32 and BF16 are not read as F16."""
    return {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}.get(e.get("dtype", "F16"))


def check_source(cfg, hdr):
    """Refuse what the engine cannot represent, before any tensor is read.

    The engine has one dense block: RMSNorm, silu SwiGLU, RoPE with the angle in
    the shader, grouped-query attention, no bias tensors and no sliding window.
    A checkpoint outside that shape used to pack without complaint and then run
    with part of itself silently dropped - the worst of both worlds, because the
    file looks fine and the answers are wrong.  Everything known to be
    unrepresentable stops here, with the reason, and the model is named.
    """
    arch = (cfg.get('architectures') or [cfg.get('model_type', '?')])[0]
    kind = cfg.get('model_type', '')
    # RWKV-7 is implemented and is not a transformer block at all (no
    # attention, no RoPE, no KV cache, no bias-free-matvec rule), so it leaves
    # this function before the Llama-shaped checks below can misname it.  It is
    # a v4 container: docs/checkpoint-container.md.
    if kind == 'rwkv7' or 'rwkv7' in arch.lower():
        return check_source_rwkv7(cfg, hdr)
    # Name the block before the tensor-level checks: a GPT-2 checkpoint has
    # biases *and* a different block, and "it has biases" would send the reader
    # looking for a bias kernel when the real answer is that the whole block is
    # a different one.
    blocks = {
        'gpt2': "the GPT-2 block (LayerNorm, gelu_new, learned position embeddings)",
        'gpt_bigcode': "the GPT-2 block (LayerNorm, gelu_new, learned position embeddings)",
        'gpt_neox': "the GPT-NeoX block (LayerNorm, gelu, parallel residual)",
        'gptj': "the GPT-J block (LayerNorm, gelu_new, rotary on half the head)",
        'opt': "the OPT block (LayerNorm, learned positions, a bias in every projection)",
        'mamba': "the Mamba SSM block (a convolution and a state-space recurrences, no attention)",
        'rwkv': "the RWKV block (a linear-attention recurrence, no softmax)",
        'falcon': "the Falcon block (parallel attention/FFN with LayerNorm, alibi or multiquery)",
        'gemma': "the Gemma block (a 256k vocabulary and a scaled embedding)",
        'mixtral': "a mixture-of-experts block",
    }
    for key, what in blocks.items():
        if kind == key or kind.startswith(key + '_') or key in arch.lower():
            raise SystemExit(
                "%s: this is %s.  The engine has one dense block - RMSNorm, silu "
                "SwiGLU, RoPE, grouped-query attention, no bias tensors - so it "
                "cannot be converted; see docs/model-formats.md." % (arch, what))
    # MoE is decided by the checkpoint, not by the config: MiniMind's config
    # class carries num_experts on dense releases too (the 2026-10 forks say
    # num_experts 4 with 90 dense tensors and not one expert in the header).
    moe = [k for k in hdr
           if '.experts.' in k or 'block_sparse_moe' in k or '.router.' in k
           or k.endswith('gate_inp.weight')]
    if moe:
        # One MoE block is implemented: the Qwen3-MoE layout, every layer
        # sparse, one router per layer, top-1 with a normalised gate (so the
        # selected expert's output is used with weight exactly 1.0), no shared
        # expert.  Everything else stops here, where the reason can name the
        # model: a wrong MoE is a checkpoint that packs and answers nonsense.
        if kind != 'qwen3_moe' and 'qwen3moe' not in arch.lower():
            raise SystemExit(
                "%s: the checkpoint has mixture-of-experts tensors (%s%s), but the "
                "engine implements one MoE block - the Qwen3-MoE layout.  See "
                "docs/checkpoint-container.md."
                % (arch, ', '.join(sorted(moe)[:2]), ', ...' if len(moe) > 2 else ''))
        n_exp = int(cfg.get('num_experts') or 0)
        top_k = int(cfg.get('num_experts_per_tok') or 0)
        if n_exp < 2:
            raise SystemExit("%s: expert tensors but num_experts=%r"
                             % (arch, cfg.get('num_experts')))
        if top_k != 1:
            raise SystemExit(
                "%s: num_experts_per_tok is %r and the engine routes top-1 only.  "
                "A top-k>1 model needs a weighted sum of k expert FFNs, which the "
                "runtime does not have." % (arch, top_k))
        if not cfg.get('norm_topk_prob'):
            raise SystemExit(
                "%s: norm_topk_prob is false, so the expert output would have to be "
                "scaled by its softmax gate; the runtime applies weight 1.0." % arch)
        if cfg.get('shared_expert_intermediate_size') or any('shared_expert' in k for k in hdr):
            raise SystemExit("%s: this MoE has a shared expert; the engine's block "
                             "is routed experts only." % arch)
        if int(cfg.get('decoder_sparse_step', 1) or 1) != 1 or (cfg.get('mlp_only_layers') or []):
            raise SystemExit(
                "%s: sparse layers are not every layer (decoder_sparse_step=%r, "
                "mlp_only_layers=%r); the container is all-MoE or all-dense."
                % (arch, cfg.get('decoder_sparse_step'), cfg.get('mlp_only_layers')))
        if not int(cfg.get('moe_intermediate_size') or 0):
            raise SystemExit("%s: the expert FFN width (moe_intermediate_size) is "
                             "missing" % arch)
    moe_cfg = [k for k in ('num_experts', 'num_local_experts', 'n_routed_experts',
                           'moe_layer_freq', 'num_experts_per_tok') if cfg.get(k)]
    if moe_cfg and not moe:
        print("note: config declares %s but the checkpoint has no expert tensors; "
              "treating it as dense" % ', '.join(moe_cfg))
    biases = sorted(k for k in hdr if k.endswith('.bias'))
    if biases:
        raise SystemExit(
            "%s: the checkpoint carries bias tensors (%s%s).  The matvec kernels "
            "add no bias, so packing this would drop them silently and the logits "
            "would be wrong."
            % (arch, ', '.join(biases[:3]), ', ...' if len(biases) > 3 else ''))
    if cfg.get('attention_bias') or cfg.get('mlp_bias'):
        raise SystemExit(
            "%s: config declares attention_bias/mlp_bias, which the engine cannot "
            "apply (no bias tensors are read)." % arch)
    act = cfg.get('hidden_act') or cfg.get('activation_function')
    if act and str(act).lower() not in ('silu', 'swish'):
        raise SystemExit(
            "%s: hidden_act is %r; the FFN kernel computes silu only (a GPT-2 "
            "checkpoint says gelu_new, a NeoX one gelu)." % (arch, act))
    rs = cfg.get('rope_scaling')
    if rs:
        kind = rs.get('rope_type') or rs.get('type')
        if kind == 'longrope':
            # Accepted only as a static table: rope_factors() is where the
            # "static" conditions are enforced, and the length is checked
            # against the head width below.
            rf = rope_factors(cfg)
            if rf and (len(rf) * 2) != (int(cfg.get('hidden_size') or 0) //
                                        int(cfg.get('num_attention_heads') or 1)):
                raise SystemExit(
                    "%s: rope_scaling longrope has %d factors, the head is %d wide, "
                    "so the table needs %d." % (arch, len(rf),
                                                int(cfg['hidden_size']) //
                                                int(cfg['num_attention_heads']),
                                                len(rf) * 2))
        elif kind not in (None, 'default'):
            raise SystemExit(
                "%s: rope_scaling %r is not representable; the container carries a "
                "theta, a static per-dimension table (key 18) and a half/plain "
                "convention, nothing else." % (arch, kind))
    dim = cfg.get('hidden_size') or cfg.get('n_embd')
    heads = cfg.get('num_attention_heads') or cfg.get('n_head')
    hd = cfg.get('head_dim')
    if dim and heads and hd and hd != dim // heads:
        raise SystemExit(
            "%s: head_dim %d does not match hidden_size/heads = %d/%d; the engine "
            "derives the head width from the two." % (arch, hd, dim, heads))
    # LFM2's hybrid block is a different name tree (conv layers carry
    # in_proj/conv/out_proj and no qkv, attention layers add q_layernorm), and
    # load_lfm2() checks every tensor it reads with the expected shape.  The
    # checks above still apply: no biases, silu FFN, default rope, matched
    # head_dim.
    if cfg.get('model_type') == 'lfm2':
        # Qwen3 configs also carry `layer_types` (all full_attention), so the
        # hybrid block is keyed on model_type, never on the key's presence.
        return
    # The loader maps Llama/Qwen names only.  A BERT/ModernBERT encoder, a
    # Tinyllm or a MiniMind fork passes every check above (no biases, silu) and
    # then dies on a KeyError halfway through the first tensor read.
    need = ['model.embed_tokens.weight', 'model.norm.weight',
            'model.layers.0.input_layernorm.weight',
            'model.layers.0.post_attention_layernorm.weight',
            'model.layers.0.self_attn.q_proj.weight',
            'model.layers.0.self_attn.o_proj.weight']
    need += (['model.layers.0.mlp.gate.weight',
              'model.layers.0.mlp.experts.0.gate_proj.weight',
              'model.layers.0.mlp.experts.0.down_proj.weight'] if moe else
             ['model.layers.0.mlp.gate_proj.weight',
              'model.layers.0.mlp.down_proj.weight'])
    missing = [n for n in need if n not in hdr]
    if missing:
        raise SystemExit(
            "%s: the tensor names are not the Llama/Qwen layout this loader maps "
            "(missing %s%s).  A BERT encoder, a GPT-2 `transformer.h` tree or an "
            "unusual fork lands here."
            % (arch, ', '.join(missing[:3]), ', ...' if len(missing) > 3 else ''))
    if cfg.get('sliding_window') and cfg.get('use_sliding_window') is not False:
        print("note: %s declares sliding_window %s, which the engine ignores; the "
              "attention is dense, so keep SEQ <= %s"
              % (arch, cfg.get('sliding_window'), cfg.get('sliding_window')))


# The RWKV vocabulary file the pth/HF releases ship
# (rwkv_vocab_v20230424.txt): one entry per line as `id 'repr(token)' length`.
# It is LF in the official HuggingFace copy and CRLF in the file inside the pip
# `rwkv` package, which is the one models/rwkv7/g1d-hf carries; a reader that
# only strips "\n" leaves a 0x0d on every line, the byte lengths stop matching
# and the whole id table shifts.
RWKV_VOCAB_NAME = 'rwkv_vocab_v20230424.txt'
RWKV_EOT_TEXT = '<|rwkv_tokenizer_end_of_text|>'
# The calibration stream is a chat prompt in the model's own format: the peaks
# the rail check looks for live on real text, while three runs of vocabulary ids
# miss them (measured 2026-11-16: sqrelu 58 there against 92 on this prompt).
RWKV_RAIL_TEXT = ("User: Explain in two sentences why the sky is blue, then name "
                  "one country in Europe.\n\nAssistant: <think")

_RWKV_ESCAPES = {'n': 10, 'r': 13, 't': 9, '\\': 92, "'": 39, '"': 34, '0': 0,
                 'a': 7, 'b': 8, 'f': 12, 'v': 11}


def _unescape_rwkv(s, raw):
    """One vocabulary entry (already decoded as UTF-8) to token bytes.

    Escapes: \\xNN (a byte in a b'' entry, a code point otherwise), \\uNNNN,
    and the C one-letter forms.  A line may also carry the character itself
    (id 2431 is a literal inverted exclamation mark), which is why the file is
    decoded as UTF-8 before this runs.
    """
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c != '\\':
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            break
        if s[i] == 'x' and i + 2 < n:
            out.append(chr(int(s[i + 1:i + 3], 16)))
            i += 3
            continue
        if s[i] == 'u' and i + 4 < n:
            out.append(chr(int(s[i + 1:i + 5], 16)))
            i += 5
            continue
        out.append(chr(_RWKV_ESCAPES.get(s[i], ord(s[i]))))
        i += 1
    txt = ''.join(out)
    return bytes(ord(ch) for ch in txt) if raw else txt.encode('utf-8')


def load_rwkv_vocab(path):
    """{token bytes: id} and the longest entry, for the rail calibration.

    Line endings are stripped as one sequence so LF and CRLF both work; the
    stated byte length is checked against the decoded entry, which is what turns
    a wrong line-ending strip or an escape this code does not know into an
    error instead of a token table that is quietly off.
    """
    tok2idx = {}
    maxlen = 0
    with open(path, 'rb') as f:
        for line in f:
            line = line.rstrip(b'\r\n').decode('utf-8')   # LF and CRLF
            if not line:
                continue
            sp = line.index(' ')
            idx = int(line[:sp])
            body = line[sp + 1:]
            sp2 = body.rindex(' ')
            entry = body[:sp2]
            want = int(body[sp2 + 1:])
            # Three spellings: 'x' for most entries, "x" when the token itself
            # contains a single quote (id 40 is one), and b'\x80' for a raw
            # byte string.
            is_bytes = entry[:2] in ("b'", 'b"')
            if is_bytes and entry[-1:] == entry[1:2]:
                entry = entry[2:-1]
            elif len(entry) >= 2 and entry[:1] == entry[-1:] and entry[:1] in ("'", '"'):
                entry = entry[1:-1]
            tok = _unescape_rwkv(entry, is_bytes)
            if len(tok) != want:
                raise SystemExit("%s: id %d decodes to %d B, the line says %d "
                                 "(line endings or an unknown escape)"
                                 % (path, idx, len(tok), want))
            tok2idx[tok] = idx
            maxlen = max(maxlen, len(tok))
    if not tok2idx:
        raise SystemExit("%s: empty vocabulary" % path)
    return tok2idx, maxlen


def encode_rwkv(tok2idx, maxlen, text):
    """Greedy longest match over bytes, plus the added end-of-text token.

    This is the packer's calibration encoder, not the engine's tokenizer (that
    is P3 of the plan): it only has to turn the calibration prompt into the ids
    llama.cpp would see.  The longest match is a dict probe from the longest
    possible length down to one byte, which is the trie walk with another data
    structure.
    """
    src = text.encode('utf-8') if isinstance(text, str) else text
    ids = []
    i = 0
    while i < len(src):
        m = min(maxlen, len(src) - i)
        while m >= 1:
            j = tok2idx.get(src[i:i + m])
            if j is not None:
                ids.append(j)
                i += m
                break
            m -= 1
        else:
            raise SystemExit("no vocabulary entry covers byte %r at %d" % (src[i], i))
    return ids


def rwkv_rail_tokens(model_dir):
    """The calibration stream (ids), or [] when the source has no vocabulary."""
    path = model_dir + '/' + RWKV_VOCAB_NAME
    if not os.path.exists(path):
        print('note: no %s next to the source; the rail calibration falls back '
              'to vocabulary runs only' % RWKV_VOCAB_NAME)
        return []
    tok2idx, maxlen = load_rwkv_vocab(path)
    # The template's BOS is an added token: it is not in the file, and the
    # reference tokenizer maps its text to id 0 (tokenizers_config's
    # added_tokens_decoder).  Without it the template's BOS becomes 26 byte
    # tokens and the calibration measures the wrong prompt.
    head = [0] if maxlen else []
    return head + encode_rwkv(tok2idx, maxlen, RWKV_RAIL_TEXT)

def check_source_rwkv7(cfg, hdr):
    """The RWKV-7 block, checked before a single tensor is read.

    Everything here is a shape the reader would otherwise have to guess: the
    four low ranks (llama.cpp's fallback formula is wrong for 0.4B's gate),
    the head width the WKV state is indexed by, and the activation.  RWKV has
    no bias inside a matmul, so the general 'no bias tensors' rule does not
    apply -- the biases that exist are LayerNorm and the three low-rank output
    vectors, and they travel as f32 rows.
    """
    arch = (cfg.get('architectures') or [cfg.get('model_type', '?')])[0]
    act = str(cfg.get('hidden_act') or '').lower()
    if act not in ('sqrelu',):
        raise SystemExit("%s: hidden_act is %r; the RWKV-7 channel mix is "
                         "relu(x)^2 (sqrelu) only." % (arch, act))
    # A config that does not state norm_bias at all is fine (the official pth
    # conversion writes none); one that says false is not: the container stores
    # every LayerNorm as one w|b row.
    if cfg.get('norm_bias') is False:
        raise SystemExit("%s: norm_bias is false; every RWKV-7 LayerNorm has a "
                         "bias and the container stores w and b in one row." % arch)
    dim = int(cfg.get('hidden_size') or 0)
    hidden = int(cfg.get('intermediate_size') or 0)
    layers = int(cfg.get('num_hidden_layers') or 0)
    if not (dim and hidden and layers):
        raise SystemExit("%s: config is missing hidden_size/intermediate_size/"
                         "num_hidden_layers" % arch)
    head_size = int(cfg.get('head_dim') or cfg.get('head_size') or 0)
    if not head_size:
        raise SystemExit("%s: the config states no head_dim; the WKV state is "
                         "indexed by it and the engine will not guess" % arch)
    if dim % head_size:
        raise SystemExit("%s: hidden_size %d is not a multiple of head_dim %d"
                         % (arch, dim, head_size))
    # num_heads in a fla-hub RWKV7 config is NOT dim/head_dim (0.1B says 32
    # heads for a 768-wide model whose WKV head is 64 wide, i.e. 12 heads), so
    # the head count is derived from the two numbers that matter.
    if cfg.get('max_position_embeddings') is None:
        print("note: %s states no max_position_embeddings; RWKV has no KV cache "
              "and the value is only recorded as the container's seq" % arch)
    for key in ('decay_low_rank_dim', 'a_low_rank_dim', 'v_low_rank_dim',
                'gate_low_rank_dim'):
        if cfg.get(key) is None:
            raise SystemExit(
                "%s: the config has no %s.  The low rank cannot be recovered from "
                "the formula llama.cpp falls back to -- it gives 160 for 0.4B's "
                "gate where the config says 128 -- so a source that does not state "
                "it is refused." % (arch, key))
    need = ['model.embeddings.weight', 'lm_head.weight']
    if 'model.layers.0.pre_norm.weight' in hdr:
        need.append('model.layers.0.pre_norm.weight')
    for l in (0, layers - 1):
        need += ['model.layers.%d.%s' % (l, n) for n in
                 ('attn_norm.weight', 'ffn_norm.weight', 'attn.g_norm.weight',
                  'attn.r_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight',
                  'attn.o_proj.weight', 'ffn.key.weight', 'ffn.value.weight',
                  'attn.w_lora.lora.0.weight', 'attn.w_lora.lora.2.weight',
                  'attn.a_lora.lora.0.weight', 'attn.a_lora.lora.2.weight',
                  'attn.k_k', 'attn.k_a', 'attn.r_k',
                  'attn.x_r', 'attn.x_w', 'attn.x_k', 'attn.x_v', 'attn.x_a',
                  'ffn.x_k')]
    missing = [n for n in need if n not in hdr]
    if missing:
        raise SystemExit(
            "%s: not the fla-hub RWKV-7 tensor tree this loader maps (missing "
            "%s%s).  It reads model.embeddings, lm_head, model.norm, "
            "model.layers.N.{attn_norm,ffn_norm,attn.g_norm}, "
            "model.layers.N.attn.{r,k,v,o}_proj, model.layers.N.attn.{w,a,v,g}_lora "
            "and model.layers.N.ffn.{key,value}."
            % (arch, ', '.join(missing[:3]), ', ...' if len(missing) > 3 else ''))
    if cfg.get('tie_word_embeddings') is not False:
        raise SystemExit("%s: tie_word_embeddings is not false; an RWKV-7 head is "
                         "a separate vocab x dim matrix" % arch)


def load_rwkv7(model_dir, seq=512):
    """Read fla-hub RWKV-7 (Goose) safetensors into float arrays.

    The names are the ones refs/llamacpp/src-b12283/conversion/rwkv.py maps
    for Rwkv7ForCausalLM, and each tensor's role was checked against
    llama.cpp's own conversion of this checkpoint: the GGUF's f32 norm and
    vector tensors match these names byte for byte (mmcf_pack.py --verify
    cannot do that for bf16 weights, so the check was a separate pass).

        model.embeddings.weight            token embedding (vocab x dim)
        lm_head.weight                     untied head (vocab x dim)
        model.layers.0.pre_norm.{w,b}      token norm (token_embd_norm)
        model.norm.{w,b}                   output norm (output_norm)
        model.layers.N.attn_norm.{w,b}     LayerNorm before the time mix
        model.layers.N.ffn_norm.{w,b}      LayerNorm before the channel mix
        model.layers.N.attn.g_norm.{w,b}   per-head group norm (time_mix_ln)
        model.layers.N.attn.x_{r,w,k,v,a,g}  the six token-shift lerps
        model.layers.N.attn.{w,a,v}_lora.lora.2.bias   w0 / a0 / v0
        model.layers.N.attn.k_k,k_a,r_k    per-channel scales (r_k is heads x 64)
        model.layers.N.ffn.x_k             the channel-mix lerp
        model.layers.N.attn.{r,k,v}_proj.weight   W_r / W_k / W_v (dim x dim)
        model.layers.N.attn.o_proj.weight         W_o
        model.layers.N.ffn.key.weight      W_cm_key (hidden x dim)
        model.layers.N.ffn.value.weight    W_cm_value (dim x hidden)
        model.layers.N.attn.{w,a,v,g}_lora.lora.{0,2}.weight
                                           down (rank x dim) / up (dim x rank)

    Every matrix is stored the way the checkpoint states it: rows are the
    output rows, like every other Mat the engine reads.  The LayerNorm and
    lerp vectors keep the HF/GGUF order so a reader can diff them against
    llama.cpp.  Layer 0 ships no v_lora (it records v_first instead of
    mixing it), so the value-residual branch is absent there.
    """
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    with open(model_dir + '/model.safetensors', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    check_source(cfg, hdr)

    def get(name, optional=False):
        if name not in hdr:
            if optional:
                return None
            raise SystemExit("%s: the checkpoint has no %s" % (model_dir, name))
        e = hdr[name]
        a, b = e['data_offsets']
        if e.get('dtype') == 'BF16':
            bits = np.frombuffer(data[a:b], dtype='<u2').astype(np.uint32) << 16
            return bits.view(np.float32).reshape(e['shape'])
        dt = st_dtype(e)
        if dt is None:
            raise SystemExit("%s: unsupported safetensors dtype %s"
                             % (name, e.get('dtype')))
        return np.frombuffer(data[a:b], dtype=dt).astype(np.float32).reshape(e['shape'])

    emb = get('model.embeddings.weight')
    vocab, dim = emb.shape
    wcls = get('lm_head.weight')
    if wcls.shape != emb.shape:
        raise SystemExit("lm_head %s does not match the embedding %s"
                         % (list(wcls.shape), list(emb.shape)))
    layers = int(cfg['num_hidden_layers'])
    hidden = int(cfg['intermediate_size'])
    head_size = int(cfg['head_dim'])
    heads = dim // head_size
    ranks = dict(r_w=int(cfg['decay_low_rank_dim']), r_a=int(cfg['a_low_rank_dim']),
                 r_v=int(cfg['v_low_rank_dim']), r_g=int(cfg['gate_low_rank_dim'] or 0))

    def vec(name, n, optional=False):
        a = get(name, optional)
        if a is None:
            return None
        a = np.asarray(a, np.float32).reshape(-1)
        if a.size != n:
            raise SystemExit("%s is %d values, the block wants %d"
                             % (name, a.size, n))
        return a

    def mat(name, shape):
        a = np.asarray(get(name), np.float32)
        if a.shape != shape:
            raise SystemExit("%s is %s, the block wants %s"
                             % (name, list(a.shape), list(shape)))
        return a

    # One row per layer: [w | b], which is how the container stores them.
    def norm(name):
        return np.concatenate([vec(name + '.weight', dim), vec(name + '.bias', dim)])

    att_norm = np.stack([norm('model.layers.%d.attn_norm' % l) for l in range(layers)])
    att_norm_2 = np.stack([norm('model.layers.%d.ffn_norm' % l) for l in range(layers)])
    ln_x = np.stack([norm('model.layers.%d.attn.g_norm' % l) for l in range(layers)])
    cm_lerp = np.stack([vec('model.layers.%d.ffn.x_k' % l, dim) for l in range(layers)])
    tok_norm = None
    if 'model.layers.0.pre_norm.weight' in hdr:
        tok_norm = np.concatenate([vec('model.layers.0.pre_norm.weight', dim),
                                   vec('model.layers.0.pre_norm.bias', dim)])
    out_norm = np.concatenate([vec('model.norm.weight', dim), vec('model.norm.bias', dim)])
    # Worst case for the head's input, from the weights alone: a LayerNorm
    # output is bounded channel-wise by sqrt(dim)*|w_i| + |b_i| (the sum of the
    # squared normalised values is dim, so each is at most sqrt(dim)).  It is a
    # bound, not a measurement: reaching it needs one channel to carry all the
    # variance.
    ln_bound = float(np.sqrt(dim) * np.abs(out_norm[:dim]).max()
                     + np.abs(out_norm[dim:]).max())

    lerp, bias, scale = [], [], []
    for l in range(layers):
        p = 'model.layers.%d.attn.' % l
        lerp.append(np.concatenate([vec(p + 'x_' + k, dim) for k in 'rwkvag']))
        w0 = vec(p + 'w_lora.lora.2.bias', dim)
        a0 = vec(p + 'a_lora.lora.2.bias', dim)
        v0 = vec(p + 'v_lora.lora.2.bias', dim, optional=True)
        if v0 is None:
            # Layer 0 records v_first instead of mixing it: no v0 there, and
            # the row keeps a zero so all three groups stay the same width.
            v0 = np.zeros(dim, np.float32)
        bias.append(np.concatenate([w0, a0, v0]))
        scale.append(np.concatenate([vec(p + 'k_k', dim), vec(p + 'k_a', dim),
                                     vec(p + 'r_k', dim)]))
    lerp = np.stack(lerp)
    bias = np.stack(bias)
    scale = np.stack(scale)

    rkv, wo, cm_key, cm_value = [], [], [], []
    for l in range(layers):
        rkv.append(np.concatenate([mat('model.layers.%d.attn.r_proj.weight' % l, (dim, dim)),
                                   mat('model.layers.%d.attn.k_proj.weight' % l, (dim, dim)),
                                   mat('model.layers.%d.attn.v_proj.weight' % l, (dim, dim))],
                                  axis=0))
        wo.append(mat('model.layers.%d.attn.o_proj.weight' % l, (dim, dim)))
        cm_key.append(mat('model.layers.%d.ffn.key.weight' % l, (hidden, dim)))
        cm_value.append(mat('model.layers.%d.ffn.value.weight' % l, (dim, hidden)))
    rkv, wo = np.stack(rkv), np.stack(wo)
    cm_key, cm_value = np.stack(cm_key), np.stack(cm_value)

    lora = {}
    for br, r in (('w', ranks['r_w']), ('a', ranks['r_a']),
                  ('v', ranks['r_v']), ('g', ranks['r_g'])):
        if not r:
            lora[br] = None
            continue
        down, up = [], []
        for l in range(layers):
            p = 'model.layers.%d.attn.%s_lora.lora.' % (l, br)
            d = get(p + '0.weight', optional=(br == 'v' and l == 0))
            u = get(p + '2.weight', optional=(br == 'v' and l == 0))
            if d is None or u is None:
                if br != 'v' or l != 0:
                    raise SystemExit("layer %d has half of the %s low-rank pair"
                                     % (l, br))
                down.append(None); up.append(None)
                continue
            down.append(mat(p + '0.weight', (r, dim)))
            up.append(mat(p + '2.weight', (dim, r)))
        lora[br] = (down, up)
    gate = lora['g'] is not None
    if gate:
        # A mix of gated and ungated layers is not an RWKV-7 the runtime can
        # express: the flag in metadata key 16 describes the whole model.
        has = sum(1 for x in lora['g'][0] if x is not None)
        if has != layers:
            raise SystemExit("the gate low rank is present on %d of %d layers"
                             % (has, layers))

    return dict(arch='rwkv7', dim=dim, hidden=hidden, layers=layers,
                heads=heads, kv=heads, vocab=vocab, seq=seq,
                head_size=head_size, theta=10000.0, half=False, qk_norm=0,
                rms_eps=float(cfg.get('norm_eps') or 1e-5),
                rwkv=dict(arch=1, head_size=head_size,
                          norm_eps=float(cfg.get('norm_eps') or 1e-5),
                          gate=gate, **ranks),
                emb=emb, wcls=wcls, untied=True,
                att_norm=att_norm, att_norm_2=att_norm_2, ln_x=ln_x,
                cm_lerp=cm_lerp, lerp=lerp, bias=bias, scale=scale,
                tok_norm=tok_norm, out_norm=out_norm, final_ln_bound=ln_bound,
                rkv=rkv, wo=wo, cm_key=cm_key, cm_value=cm_value, lora=lora,
                moe=None, added_ids=[],
                # The rail calibration's real-text stream: the vocabulary sits
                # next to the weights in every release, and the peaks only show
                # up on text (see RWKV_RAIL_TEXT).
                rail_tokens=rwkv_rail_tokens(model_dir))

def load_minimind(model_dir, seq=512):
    """Read a HF Llama/Qwen-shaped directory into float arrays."""
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    st_path = model_dir + '/model.safetensors'
    pth_path = model_dir + '/pytorch_model.bin'
    tensors = {}
    have_st = os.path.exists(st_path)
    if have_st:
        with open(st_path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(n))
            data = f.read()
    elif os.path.exists(pth_path):
        # Some releases ship only a torch zip (`pytorch_model.bin`): the BitCPM
        # family does, and so does anything older than safetensors.  pth_reader
        # reads its pickle and storages directly, so no torch is needed and no
        # 870 MB intermediate file is written.
        import pth_reader
        tensors = pth_reader.load(pth_path)
        hdr = {k: {'dtype': 'F32', 'shape': list(v.shape),
                   'data_offsets': [0, 0]} for k, v in tensors.items()}
        data = b''
    else:
        raise SystemExit("%s has neither model.safetensors nor pytorch_model.bin"
                         % model_dir)
    check_source(cfg, hdr)

    def get(name):
        if not have_st and name in tensors:
            return tensors[name].astype(np.float32)
        e = hdr[name]
        a, b = e['data_offsets']
        dt = st_dtype(e)
        if dt is None:
            raise SystemExit(f"{name}: unsupported safetensors dtype {e.get('dtype')}")
        if e.get("dtype") == "BF16":
            # bf16 is the top half of an f32; widening is exact.
            bits = np.frombuffer(data[a:b], dtype="<u2").astype(np.uint32) << 16
            return bits.view(np.float32).reshape(e['shape'])
        return np.frombuffer(data[a:b], dtype=dt).astype(np.float32).reshape(e['shape'])

    emb = get('model.embed_tokens.weight')
    vocab, dim = emb.shape
    # The LM head is the embedding unless the checkpoint ships its own.  A
    # `tie_word_embeddings: false` model that only *looks* like a Llama block
    # (Mistral labels, Minueza, pythia) has a separate matrix, and using the
    # embedding for it produces a container that runs and answers nonsense.
    wcls = emb
    if 'lm_head.weight' in hdr:
        wcls = get('lm_head.weight')
        if wcls.shape != emb.shape:
            raise SystemExit("lm_head %s does not match the embedding %s"
                             % (list(wcls.shape), list(emb.shape)))
    elif cfg.get('tie_word_embeddings') is False:
        raise SystemExit("config says tie_word_embeddings: false but there is no "
                         "lm_head.weight in the checkpoint; refusing to pack a model "
                         "whose head would silently become its embedding")
    layers = cfg['num_hidden_layers']
    heads = cfg['num_attention_heads']
    kv = cfg['num_key_value_heads']
    hidden = cfg['intermediate_size']
    wq = np.stack([get(f'model.layers.{l}.self_attn.q_proj.weight') for l in range(layers)])
    wk = np.stack([get(f'model.layers.{l}.self_attn.k_proj.weight') for l in range(layers)])
    wv = np.stack([get(f'model.layers.{l}.self_attn.v_proj.weight') for l in range(layers)])
    wo = np.stack([get(f'model.layers.{l}.self_attn.o_proj.weight') for l in range(layers)])
    # A mixture-of-experts checkpoint has no dense mlp.{gate,down,up}_proj at
    # all: the FFN lives under mlp.experts.E.* and the router under mlp.gate.
    # check_source() has already refused every shape this reader cannot carry.
    is_moe = any('.experts.' in k for k in hdr)
    moe = None
    expert_w1 = expert_w2 = expert_w3 = router = None
    if is_moe:
        n_exp = int(cfg['num_experts'])
        moe_hidden = int(cfg['moe_intermediate_size'])
        expert_w1 = [[get(f'model.layers.{l}.mlp.experts.{e}.gate_proj.weight')
                      for e in range(n_exp)] for l in range(layers)]
        expert_w2 = [[get(f'model.layers.{l}.mlp.experts.{e}.down_proj.weight')
                      for e in range(n_exp)] for l in range(layers)]
        expert_w3 = [[get(f'model.layers.{l}.mlp.experts.{e}.up_proj.weight')
                      for e in range(n_exp)] for l in range(layers)]
        router = np.stack([get(f'model.layers.{l}.mlp.gate.weight')
                           for l in range(layers)])
        for l in range(layers):
            for e in range(n_exp):
                for arr, shape, what in ((expert_w1[l][e], (moe_hidden, dim), 'gate_proj'),
                                         (expert_w3[l][e], (moe_hidden, dim), 'up_proj'),
                                         (expert_w2[l][e], (dim, moe_hidden), 'down_proj')):
                    if arr.shape != shape:
                        raise SystemExit(
                            "layer %d expert %d %s is %s, the block wants %s"
                            % (l, e, what, list(arr.shape), list(shape)))
        if router.shape != (layers, n_exp, dim):
            raise SystemExit("router is %s, expected %s"
                             % (list(router.shape), [layers, n_exp, dim]))
        moe = dict(n_experts=n_exp, top_k=int(cfg['num_experts_per_tok']),
                   hidden=moe_hidden, norm_topk=bool(cfg.get('norm_topk_prob')))
        w1 = w2 = w3 = None
    else:
        w1 = np.stack([get(f'model.layers.{l}.mlp.gate_proj.weight') for l in range(layers)])
        w2 = np.stack([get(f'model.layers.{l}.mlp.down_proj.weight') for l in range(layers)])
        w3 = np.stack([get(f'model.layers.{l}.mlp.up_proj.weight') for l in range(layers)])
    rms_att = np.stack([get(f'model.layers.{l}.input_layernorm.weight') for l in range(layers)])
    rms_ffn = np.stack([get(f'model.layers.{l}.post_attention_layernorm.weight') for l in range(layers)])
    rms_final = get('model.norm.weight')
    qk = 'model.layers.0.self_attn.q_norm.weight' in hdr
    q_norm = np.stack([get(f'model.layers.{l}.self_attn.q_norm.weight') for l in range(layers)]) if qk else None
    k_norm = np.stack([get(f'model.layers.{l}.self_attn.k_norm.weight') for l in range(layers)]) if qk else None
    theta, half = rope_of(cfg)
    rfac = rope_factors(cfg)
    # The template's special tokens are where a chat prompt's activation peaks
    # live (MiniMind3-MoE reaches 250 in the expert hidden at <|im_start|>).
    # tokenizer.json lists them with their ids, so the rail calibration can
    # exercise them without a tokenizer library.
    added_ids = []
    tok_json = model_dir + '/tokenizer.json'
    if os.path.exists(tok_json):
        try:
            with open(tok_json) as f:
                tj = json.load(f)
            added_ids = sorted({int(t['id']) for t in tj.get('added_tokens', [])
                                if isinstance(t, dict) and 'content' in t and 'id' in t})
            added_ids = [i for i in added_ids if 0 <= i < vocab]
        except (ValueError, KeyError, TypeError):
            added_ids = []
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab, seq=seq,
                emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=wcls,
                theta=theta, half=half, rope_factors=rfac, qk_norm=1 if qk else 0,
                scale_emb=float(cfg.get('scale_emb') or 1.0),
                scale_depth=float(cfg.get('scale_depth') or 1.0),
                dim_model_base=float(cfg.get('dim_model_base') or 1.0),
                sp_prefix=1 if sp_prefix_of(model_dir) else 0,
                q_norm=q_norm, k_norm=k_norm, untied='lm_head.weight' in hdr,
                moe=moe, router=router, expert_w1=expert_w1, expert_w2=expert_w2,
                expert_w3=expert_w3, added_ids=added_ids)


def load_lfm2(model_dir, seq=512):
    """Read a LiquidAI LFM2.x hybrid directory (8 conv + 6 attention layers).

    The block is the same SwiGLU/RMSNorm/RoPE attention as the dense reader,
    but the "conv" layers carry three tensors no other model has:
    `conv.in_proj` (3*dim x dim), `conv.conv` (depthwise, dim x 1 x k) and
    `conv.out_proj` (dim x dim).  Attention layers additionally carry per-head
    q/k norms.  Everything is read as float32; the quantizer does the rest.
    """
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    with open(model_dir + '/model.safetensors', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    check_source(cfg, hdr)

    def get(name):
        if name not in hdr:
            raise SystemExit("%s: the checkpoint has no %s; this is not the LFM2 "
                             "layout this loader maps" % (model_dir, name))
        e = hdr[name]
        a, b = e['data_offsets']
        dt = st_dtype(e)
        if dt is None:
            raise SystemExit("%s: unsupported safetensors dtype %s"
                             % (name, e.get('dtype')))
        if e.get('dtype') == 'BF16':
            bits = np.frombuffer(data[a:b], dtype='<u2').astype(np.uint32) << 16
            return bits.view(np.float32).reshape(e['shape'])
        return np.frombuffer(data[a:b], dtype=dt).astype(np.float32).reshape(e['shape'])

    emb = get('model.embed_tokens.weight')
    vocab, dim = emb.shape
    layers = int(cfg['num_hidden_layers'])
    heads = int(cfg['num_attention_heads'])
    kv = int(cfg['num_key_value_heads'])
    hidden = int(cfg['intermediate_size'])
    lt = list(cfg['layer_types'])
    if len(lt) != layers:
        raise SystemExit("layer_types has %d entries for %d layers" % (len(lt), layers))
    head_dim = dim // heads

    rms_att = np.stack([get('model.layers.%d.operator_norm.weight' % l)
                        for l in range(layers)])
    rms_ffn = np.stack([get('model.layers.%d.ffn_norm.weight' % l)
                        for l in range(layers)])
    rms_final = get('model.embedding_norm.weight')
    w1 = np.stack([get('model.layers.%d.feed_forward.w1.weight' % l)
                   for l in range(layers)])
    w2 = np.stack([get('model.layers.%d.feed_forward.w2.weight' % l)
                   for l in range(layers)])
    w3 = np.stack([get('model.layers.%d.feed_forward.w3.weight' % l)
                   for l in range(layers)])
    # LFM2's block_auto_adjust_ff_dim rounds the *nominal* config width to the
    # multiple-of-256 the tensors actually carry: LFM2-350M says 6656 and
    # ships 4608.  The tensors are the truth -- the container's hidden field is
    # what the engine slices w1/w2 with, so taking the config value produces a
    # container that loads and then emits garbage (2026-11-12).
    if w1.shape[1:] != (hidden, dim) or w2.shape[1:] != (dim, hidden):
        actual = int(w1.shape[1])
        if w1.shape[1:] != (actual, dim) or w2.shape[1:] != (dim, actual):
            raise SystemExit('%s: w1 %s / w2 %s do not agree on an FFN width'
                             % (model_dir, list(w1.shape), list(w2.shape)))
        print('ffn hidden %d (config.json says %d, block_auto_adjust_ff_dim)'
              % (actual, hidden))
        hidden = actual
    if w3.shape[1:] != (hidden, dim):
        raise SystemExit('%s: w3 has shape %s, expected (%d, %d, %d)'
                         % (model_dir, list(w3.shape), layers, hidden, dim))

    wq = [None] * layers
    wk = [None] * layers
    wv = [None] * layers
    wo = [None] * layers
    q_norm = [None] * layers
    k_norm = [None] * layers
    conv_in = [None] * layers
    conv_w = [None] * layers
    conv_out = [None] * layers
    ksize = None
    for l, t in enumerate(lt):
        if t == 'conv':
            conv_in[l] = get('model.layers.%d.conv.in_proj.weight' % l)
            cw = get('model.layers.%d.conv.conv.weight' % l)
            if cw.ndim == 3:
                if cw.shape[1] != 1:
                    raise SystemExit("layer %d conv.conv is %s; a depthwise conv "
                                     "wants (dim, 1, k)" % (l, list(cw.shape)))
                cw = cw[:, 0, :]
            conv_w[l] = cw
            conv_out[l] = get('model.layers.%d.conv.out_proj.weight' % l)
            ksize = int(cw.shape[1])
            if conv_in[l].shape != (3 * dim, dim):
                raise SystemExit("layer %d conv.in_proj is %s, expected (%d, %d)"
                                 % (l, list(conv_in[l].shape), 3 * dim, dim))
            if conv_out[l].shape != (dim, dim):
                raise SystemExit("layer %d conv.out_proj is %s, expected (%d, %d)"
                                 % (l, list(conv_out[l].shape), dim, dim))
        elif t in ('full_attention', 'attention'):
            wq[l] = get('model.layers.%d.self_attn.q_proj.weight' % l)
            wk[l] = get('model.layers.%d.self_attn.k_proj.weight' % l)
            wv[l] = get('model.layers.%d.self_attn.v_proj.weight' % l)
            wo[l] = get('model.layers.%d.self_attn.out_proj.weight' % l)
            q_norm[l] = get('model.layers.%d.self_attn.q_layernorm.weight' % l)
            k_norm[l] = get('model.layers.%d.self_attn.k_layernorm.weight' % l)
            for arr, shape, what in ((wq[l], (dim, dim), 'q_proj'),
                                     (wk[l], (kv * head_dim, dim), 'k_proj'),
                                     (wv[l], (kv * head_dim, dim), 'v_proj'),
                                     (wo[l], (dim, dim), 'out_proj')):
                if arr.shape != shape:
                    raise SystemExit("layer %d %s is %s, expected %s"
                                     % (l, what, list(arr.shape), list(shape)))
        else:
            raise SystemExit("layer %d has unknown kind %r" % (l, t))
    # The container stores q_norm/k_norm for every layer; the conv layers never
    # run attention, so their rows are unused and must still be finite.
    if any(x is not None for x in q_norm):
        q_norm = np.stack([q_norm[l] if q_norm[l] is not None
                           else np.ones(head_dim, np.float32) for l in range(layers)])
        k_norm = np.stack([k_norm[l] if k_norm[l] is not None
                           else np.ones(head_dim, np.float32) for l in range(layers)])
        qk = 1
    else:
        q_norm = k_norm = None
        qk = 0
    theta, half = rope_of(cfg)
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv,
                vocab=vocab, seq=seq, emb=emb,
                rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=emb,
                theta=theta, half=half, qk_norm=qk, q_norm=q_norm, k_norm=k_norm,
                layer_types=lt, conv_in=conv_in, conv_w=conv_w, conv_out=conv_out,
                conv_k=ksize, untied=False, moe=None, added_ids=[])


def load_llama2c(path):
    """Read a llama2.c fp32 checkpoint (stories15M/42M, TinyChat, 260K)."""
    b = np.fromfile(path, dtype=np.uint8)
    cfg = np.frombuffer(b[:28], dtype=np.int32).tolist()
    dim, hidden, layers, heads, kv, vsigned, seq = cfg
    vocab = abs(vsigned)
    shared = vsigned > 0
    kv_dim = kv * (dim // heads)
    off = 28

    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(b, dtype=np.float32, count=n, offset=off).reshape(shape)
        off += 4 * n
        return a

    emb = take((vocab, dim))
    rms_att = take((layers, dim))
    wq = take((layers, dim, dim))
    wk = take((layers, kv_dim, dim))
    wv = take((layers, kv_dim, dim))
    wo = take((layers, dim, dim))
    rms_ffn = take((layers, dim))
    w1 = take((layers, hidden, dim))
    w2 = take((layers, dim, hidden))
    w3 = take((layers, hidden, dim))
    rms_final = take((dim,))
    off += 4 * seq * (dim // heads)
    wcls = emb if shared else take((vocab, dim))
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab, seq=seq,
                emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=wcls,
                theta=10000.0, half=0, qk_norm=0, q_norm=None, k_norm=None)


def is_llama2c(path):
    """A llama2.c file has no magic: its first int32 is `dim`, a positive small
    number, where every legacy checkpoint starts with a 'Q8M1'-style magic."""
    import os
    if os.path.getsize(path) < 28:
        return False
    with open(path, 'rb') as f:
        head = f.read(4)
    import struct as _s
    v = _s.unpack('<I', head)[0]
    if v in (0x4931364D, 0x4931364E, 0x51384D31, 0x51384D32, 0x51364D31):
        return False
    d = _s.unpack('<i', head)[0]
    return 0 < d <= 4096 and d % 8 == 0


# The engine's residual stream is q16 fixed point at 1/256, so it cannot hold a
# value beyond 32767/256 = 127.996; `add_q16` clamps there.  MiniMind3 peaks
# around 40, but a model whose residual is larger saturates silently and answers
# from a truncated state (supra2-100M reaches 2300 at layer 4, see
# models/README.md).  Detecting that at pack time is the only place it can be
# caught cheaply.
RESIDUAL_RAIL = 32767.0 / 256.0   # 127.996
RESIDUAL_WARN = 96.0             # 75% of the rail
# The FFN's SwiGLU hidden is written through the same q16 activation format,
# so it clamps at the same rail.  Kept as its own name because the two streams
# are different tensors with different fixes.
ACTIVATION_RAIL = 32767.0 / 256.0
# The rmsnorm reads the residual and accumulates sum(x^2) per lane (32 lanes,
# dim/32 values each) into 24-bit fixed point: 1/4096 step biased by 2^23, so a
# lane pins at 2048.  It is a *squared* quantity, which is why it rails long
# before any residual format does: MiniMind3's residual of ~50 pins it (22x),
# SmolLM2's 26884 pins it by 6e6.  Stored: shaders_block.cpp's S_ST / RMS_DUMP.
RMS_LANE_RAIL = 2048.0


def residual_peaks_lfm2(m, seq=8):
    """max|x| per layer for a hybrid conv/attention block (LFM2).

    Same sampling idea as residual_peaks(): a few contiguous vocabulary runs,
    no tokenizer.  The conv layers are the part the dense calibration cannot
    model, so this is the only rail check that covers LFM2.
    """
    dim, heads, kvh = m['dim'], m['heads'], m['kv']
    layers, vocab = m['layers'], m['vocab']
    hd = dim // heads
    theta = float(m['theta'])
    eps = float(m.get('rms_eps') or 1e-5)
    lt = m['layer_types']
    rep = heads // kvh
    seq = max(1, min(seq, vocab))
    peaks = [0.0] * layers

    def rms(x, w):
        v = (x * x).mean(-1, keepdims=True)
        return x / np.sqrt(v + eps) * w

    def rope(a, pos):
        i = np.arange(hd // 2, dtype=np.float32)
        ang = pos[:, None] / (theta ** (2.0 * i / hd))[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x1, x2 = a[..., :hd // 2], a[..., hd // 2:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    starts = [0, max(0, vocab // 2 - seq // 2), max(0, vocab - seq)]
    for st in starts:
        ids = list(range(st, st + seq))
        x = m['emb'][ids].astype(np.float32)
        T = len(ids)
        pos = np.arange(T, dtype=np.float32)
        for l in range(layers):
            h = rms(x, m['rms_att'][l])
            if lt[l] == 'conv':
                g = h @ m['conv_in'][l].T
                B, C, xh = g[:, :dim], g[:, dim:2 * dim], g[:, 2 * dim:]
                gated = B * xh
                cw = np.asarray(m['conv_w'][l], np.float32)
                cv = np.zeros_like(gated)
                for k in range(cw.shape[1]):
                    sh = cw.shape[1] - 1 - k
                    if sh == 0:
                        cv += gated * cw[:, k]
                    else:
                        cv[sh:] += gated[:-sh] * cw[:, k]
                out = (C * cv) @ m['conv_out'][l].T
            else:
                q = rms((h @ m['wq'][l].T).reshape(T, heads, hd), m['q_norm'][l])
                k = rms((h @ m['wk'][l].T).reshape(T, kvh, hd), m['k_norm'][l])
                v = (h @ m['wv'][l].T).reshape(T, kvh, hd)
                q, k = rope(q, pos), rope(k, pos)
                o = np.zeros((T, heads, hd), np.float32)
                for hq in range(heads):
                    kh = hq // rep
                    raw = q[:, hq, :] @ k[:, kh, :].T / np.sqrt(hd)
                    sc = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, raw)
                    sc = sc - sc.max(-1, keepdims=True)
                    pr = np.exp(sc)
                    pr /= pr.sum(-1, keepdims=True)
                    o[:, hq, :] = pr @ v[:, kh, :]
                out = o.reshape(T, dim) @ m['wo'][l].T
            x1 = x + out
            h2 = rms(x1, m['rms_ffn'][l])
            g1 = h2 @ m['w1'][l].T
            g3 = h2 @ m['w3'][l].T
            x = x1 + ((g1 / (1 + np.exp(-g1))) * g3) @ m['w2'][l].T
            mx = float(np.abs(x).max())
            if not np.isfinite(mx):
                raise SystemExit("residual calibration produced %s at layer %d; "
                                 "the check cannot vouch for this model" % (mx, l))
            peaks[l] = max(peaks[l], mx)
    return peaks


def rail_peaks_rwkv7(m, seq=8):
    """(residual peaks, channel-mix hidden peaks) for RWKV-7; final LN via m.

    Same calibration idea as rail_peaks(): three contiguous vocabulary runs
    of seq ids and a float32 forward, no tokenizer.  Two streams travel
    through the engine's q16 activation format -- the residual x and the
    channel mix's relu(x)^2 hidden, whose square is the one place RWKV
    amplifies instead of attenuating -- so both are measured.  The WKV state
    is measured too (m['state_peaks']) but is not a rail: it is a 64x64
    recurrence per head that is never renormalised, and whether the Pi can
    hold it in fixed point is docs/engine-internals.md's state section, decided from
    that number.

    The recurrence is the one in the official numpy forward
    (models/rwkv7/ref/rwkv_v7_numpy.py) and in llama.cpp's wkv7 kernel: for
    each head, row i and column j, with kk the per-head L2-normalised
        S[i,j] = S[i,j]*w[j] - <S[i], kk> * (kk*a)[j] + v[i]*k[j]
        out[i] = <S[i], r>
    """
    dim, layers, vocab = m['dim'], m['layers'], m['vocab']
    hs = m['head_size']
    nh = dim // hs
    eps = float(m['rwkv']['norm_eps'])
    emb = m['emb']
    seq = max(1, min(seq, vocab))
    res = [0.0] * layers
    hid = [0.0] * layers
    stp = [0.0] * layers
    fin = 0.0

    def ln(x, wb):
        w, b = wb[:dim], wb[dim:]
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    def sig(x):
        # Clipped: a decay logit far from zero overflows exp() and the warning
        # is noise -- the value is saturated either way.
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    starts = [0, max(0, vocab // 2 - seq // 2), max(0, vocab - seq)]
    streams = [list(range(st0, st0 + seq)) for st0 in starts]
    if m.get('rail_tokens'):
        streams.append(list(m['rail_tokens']))
    for ids in streams:
        x = emb[ids].astype(np.float32)
        T = len(ids)
        if m.get('tok_norm') is not None:
            x = ln(x, m['tok_norm'])
        v_first = None
        state = np.zeros((layers, nh, hs, hs), np.float32)
        sh_att = np.zeros(dim, np.float32)
        sh_ffn = np.zeros(dim, np.float32)
        for l in range(layers):
            xn = ln(x, m['att_norm'][l])
            # token shift: position t sees the *normalised* input of t-1
            xprev = np.concatenate([sh_att[None], xn[:-1]], 0)
            lp = m['lerp'][l].reshape(6, dim)                  # r,w,k,v,a,g
            sx = xprev - xn
            xr, xw, xk, xv, xa, xg = [xn + sx * lp[i] for i in range(6)]
            mtx = m['rkv'][l]
            r = xr @ mtx[0:dim].T
            k = xk @ mtx[dim:2 * dim].T
            v = xv @ mtx[2 * dim:3 * dim].T
            bias = m['bias'][l].reshape(3, dim)               # w0, a0, v0
            scale = m['scale'][l].reshape(3, dim)             # k_k, k_a, r_k
            wd, wu = m['lora']['w']
            w = np.exp(-sig(np.tanh(xw @ wd[l].T) @ wu[l].T + bias[0]) / np.e ** 0.5)
            lv = m['lora']['v']
            if v_first is None:
                v_first = v
            elif lv is not None and lv[0][l] is not None:
                v = v + (v_first - v) * sig(xv @ lv[0][l].T @ lv[1][l].T + bias[2])
            ad, au = m['lora']['a']
            a = sig(xa @ ad[l].T @ au[l].T + bias[1])
            lg = m['lora']['g']
            g = None if lg is None else sig(xg @ lg[0][l].T) @ lg[1][l].T
            kk = k * scale[0]
            k = k + (a - 1.0) * k * scale[1]
            kkn = kk.reshape(T, nh, hs)
            kkn = kkn / np.maximum(np.linalg.norm(kkn, axis=2, keepdims=True), 1e-12)
            kk = kkn.reshape(T, dim)
            rw = r.reshape(T, nh, hs)
            kw = k.reshape(T, nh, hs)
            vw = v.reshape(T, nh, hs)
            ww = w.reshape(T, nh, hs)
            aw = kkn * a.reshape(T, nh, hs)                   # kk*a, per head
            S = state[l]
            out = np.empty((T, nh, hs), np.float32)
            for t in range(T):
                # S is (head, i, j): the row index i pairs with v and r, the
                # column index j with w, kk and k.
                p = (S * kkn[t][:, None, :]).sum(-1)           # <S[i], kk>, (head, i)
                S = (S * ww[t][:, None, :]
                     + vw[t][:, :, None] * kw[t][:, None, :]
                     - p[:, :, None] * aw[t][:, None, :])
                out[t] = (S * rw[t][:, None, :]).sum(-1)
            state[l] = S
            stp[l] = max(stp[l], float(np.abs(S).max()))
            # per-head group norm: one mean/var over each 64-wide head (the
            # np.var spelling takes the mean of the squares itself)
            yy = out
            mu = yy.mean(-1, keepdims=True)
            var = yy.var(-1, keepdims=True)
            yy = ((yy - mu) / np.sqrt(var + 64e-5)).reshape(T, dim)
            rkh = (rw * kw * scale[2].reshape(nh, hs)[None]).sum(-1)   # (T, nh)
            yy = yy * m['ln_x'][l][:dim] + m['ln_x'][l][dim:]
            yy = yy + (rkh[:, :, None] * vw).reshape(T, dim)
            if g is not None:
                yy = yy * g
            x = x + yy @ m['wo'][l].T
            res[l] = max(res[l], float(np.abs(x).max()))
            fn = ln(x, m['att_norm_2'][l])
            fprev = np.concatenate([sh_ffn[None], fn[:-1]], 0)
            xk2 = fn + (fprev - fn) * m['cm_lerp'][l]
            h = np.maximum(xk2 @ m['cm_key'][l].T, 0.0) ** 2
            hid[l] = max(hid[l], float(np.abs(h).max()))
            x = x + h @ m['cm_value'][l].T
            res[l] = max(res[l], float(np.abs(x).max()))
            if not np.isfinite(res[l] + hid[l] + stp[l]):
                raise SystemExit("residual calibration produced %s at layer %d; "
                                 "the check cannot vouch for this model" % (res[l], l))
            sh_att, sh_ffn = xn[-1], fn[-1]
        # The head's input: the final LayerNorm, which is what the engine
        # writes into a q16 activation (metadata key 17 covers it).
        z = ln(x, m['out_norm'])
        fin = max(fin, float(np.abs(z).max()))
    m['state_peaks'] = stp
    m['final_ln_peak'] = fin
    return res, hid

def rail_peaks(m, seq=8):
    """(residual peaks, FFN hidden peaks) from a float32 forward pass.

    Two streams travel through the engine's q16 activation format (grid 1/256,
    rail +-127.996) and a value over the rail is clamped silently: the residual
    stream `x` and the FFN's SwiGLU hidden `silu(gate)*up`.  The residual check
    has existed since the supra2 episode; the hidden one was added after the
    MiniMind3-MoE audit (2026-11-12), where the experts' hidden state reaches
    250 at the template's `<|im_start|>` positions.

    Sampling is deterministic and needs no tokenizer library: three runs of
    `seq` consecutive vocabulary ids, plus one run per *added token* parsed
    from the source's tokenizer.json.  Position 0 of a run depends only on its
    own embedding, so a leading special id reproduces the hot hidden state a
    chat prompt produces at its first token.

    Returns (residual_peaks, hidden_peaks); hidden_peaks is None for a hybrid
    model (LFM2), whose conv block has no such check yet.
    """
    if m.get('arch') == 'rwkv7':
        return rail_peaks_rwkv7(m, seq)
    if m.get('layer_types'):
        return residual_peaks_lfm2(m, seq), None
    dim, hidden, layers = m['dim'], m['hidden'], m['layers']
    heads, kv, vocab = m['heads'], m['kv'], m['vocab']
    hd = dim // heads
    theta = float(m['theta'])
    half = bool(m['half'])
    rfac = m.get('rope_factors') or [1.0] * (hd // 2)
    eps = float(m.get('rms_eps') or 1e-5)
    qk = bool(m.get('qk_norm'))
    emb = m['emb']
    peaks = [0.0] * layers
    hpeaks = [0.0] * layers
    seq = max(1, min(seq, vocab))

    def rotate(a, pos):          # a: (T, H, hd)
        # rotate_half pairs (i, i + hd/2); the interleaved convention pairs
        # (2i, 2i+1).  Using `i`/`i+1` for the interleaved case leaves half the
        # lanes stale and mixes the rest, which produced NaN scores on the
        # llama2.c models and a residual of "0.0" once max() swallowed it.
        T = a.shape[0]
        o = np.empty_like(a)
        for i in range(hd // 2):
            if half:
                ang = np.arange(T)[:, None] / (theta ** (2.0 * i / hd)) / float(rfac[i])
                c, s = np.cos(ang), np.sin(ang)
                o[:, :, i] = a[:, :, i] * c - a[:, :, i + hd // 2] * s
                o[:, :, i + hd // 2] = a[:, :, i + hd // 2] * c + a[:, :, i] * s
            else:
                ang = np.arange(T)[:, None] / (theta ** (2.0 * i / hd)) / float(rfac[i])
                c, s = np.cos(ang), np.sin(ang)
                o[:, :, 2 * i] = a[:, :, 2 * i] * c - a[:, :, 2 * i + 1] * s
                o[:, :, 2 * i + 1] = a[:, :, 2 * i + 1] * c + a[:, :, 2 * i] * s
        return o

    rep = heads // kv
    starts = [0, max(0, vocab // 2 - seq // 2), max(0, vocab - seq)]
    samples = [list(range(st, st + seq)) for st in starts]
    # One run per added token: a leading special id puts the template's first
    # token at position 0, and position 0's hidden state depends only on that
    # embedding, so this is where a chat prompt's hot expert hidden lives.
    for aid in (m.get('added_ids') or [])[:16]:
        samples.append([aid] + list(range(0, seq - 1)))
    for ids in samples:
        x = emb[ids].astype(np.float32)
        T = len(ids)
        for l in range(layers):
            an = m['rms_att'][l]
            fn = m['rms_ffn'][l]
            h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * an
            q = (h @ m['wq'][l].T).reshape(T, heads, hd)
            k = (h @ m['wk'][l].T).reshape(T, kv, hd)
            v = (h @ m['wv'][l].T).reshape(T, kv, hd)
            if qk:
                q = q / np.sqrt((q * q).mean(-1, keepdims=True) + eps) * m['q_norm'][l]
                k = k / np.sqrt((k * k).mean(-1, keepdims=True) + eps) * m['k_norm'][l]
            q = rotate(q, 0)
            k = rotate(k, 0)
            out = np.zeros((T, heads, hd), np.float32)
            for t in range(T):
                for hh in range(heads):
                    kh = hh // rep
                    sc = (q[t, hh] @ k[: t + 1, kh].T) / np.sqrt(hd)
                    pr = np.exp(sc - sc.max())
                    pr /= pr.sum()
                    out[t, hh] = pr @ v[: t + 1, kh]
            x1 = x + out.reshape(T, dim) @ m['wo'][l].T
            h2 = x1 / np.sqrt((x1 * x1).mean(-1, keepdims=True) + eps) * fn
            if m.get('moe'):
                # Top-1 with a normalised gate is exactly the selected expert's
                # FFN (weight 1.0), so the router only has to pick the expert.
                scores = h2 @ m['router'][l].T
                for t in range(T):
                    e = int(np.argmax(scores[t]))
                    w1e, w3e, w2e = (m['expert_w1'][l][e], m['expert_w3'][l][e],
                                     m['expert_w2'][l][e])
                    g = h2[t:t + 1] @ w1e.T
                    u = h2[t:t + 1] @ w3e.T
                    hh = (g / (1 + np.exp(-g))) * u
                    hpeaks[l] = max(hpeaks[l], float(np.abs(hh).max()))
                    x[t] = x1[t] + hh @ w2e.T
            else:
                g = h2 @ m['w1'][l].T
                u = h2 @ m['w3'][l].T
                hh = (g / (1 + np.exp(-g))) * u
                hpeaks[l] = max(hpeaks[l], float(np.abs(hh).max()))
                x = x1 + hh @ m['w2'][l].T
            mx = float(np.abs(x).max())
            if not np.isfinite(mx):
                raise SystemExit("residual calibration produced %s at layer %d; the "
                                 "check cannot vouch for this model" % (mx, l))
            peaks[l] = max(peaks[l], mx)
    return peaks, hpeaks


def residual_peaks(m, seq=8):
    """The residual-stream peaks only (backwards-compatible wrapper)."""
    return rail_peaks(m, seq)[0]


def tok_entry_count(path):
    """How many tokens a llama2.c-style .tok blob holds (i32 maxlen, then entries)."""
    import struct as _s
    raw = open(path, 'rb').read()
    if len(raw) < 4:
        return 0
    i, n = 4, 0
    while i + 8 <= len(raw):
        _score, ln = _s.unpack_from('<fi', raw, i)
        i += 8
        if ln < 0 or i + ln > len(raw):
            break
        i += ln
        n += 1
    return n


def fold_scale_constants(m):
    """Fold MiniCPM's scale_emb / scale_depth / dim_model_base into the weights.

    MiniCPM4 (and BitCPM-CANN, its ternary QAT release) applies three model-wide
    constants around a block the engine already implements:

        x0      = scale_emb * E[t]                                       (12)
        x_{l+1} = x_l + c * block(x_l),  c = scale_depth/sqrt(layers)     (1.4/sqrt(24))
        logits  = E_head . rms_final(x) / (hidden / dim_model_base)       (/4)

    Every sublayer input is an RMSNorm and nothing carries a bias, so the block
    output is exactly invariant to a positive rescaling of the residual stream
    (`rms(kx) = k*rms(x)`, and the norm eats the k).  That makes the residual
    stream's *absolute* scale a free parameter, and the three constants can be
    folded into the weights instead of becoming three engine features:

        E'        = k*scale_emb * E      (k is free; see below)
        wo', w2'  = k*c * wo, c * w2
        g'        = g * (dim_model_base/hidden) / (k*scale_emb)   [tied head]
        g'        = g * (dim_model_base/hidden)                  [untied]

    `k` is chosen as 1/scale_emb, i.e. the embedding table stays exactly as the
    checkpoint writes it and the 12 moves into the layer outputs.  With the
    natural choice (fold the 12 into the table) the residual would carry it --
    peak 802 on real text -- and, worse, the *head* would have to divide by 48
    instead of 4, because a tied head reads the same rows the lookup does.  The
    engine stores the head's input in the q16 activation format (1/256 step,
    +-128): at 1/4 the input sits at 0.25 RMS, which is fine, and at 1/48 it
    would sit at 0.02 RMS -- five steps -- and the logits would be noise.  This
    is why k is 1/scale_emb and not 1: measured, cosine 0.88 against a float32
    forward with k = 1, 1.000000 with k = 1/scale_emb (2026-11-18).

    It stays exact because every one of these paths stores an f32 scale of its
    own (embedding rows, projection rows, a plain f32 vector for rms_final).

    A model that states none of the three returns unchanged, so this is a no-op
    for every checkpoint packed before 2026-11-18.
    """
    se = float(m.get('scale_emb') or 1.0)
    sd = float(m.get('scale_depth') or 1.0)
    db = float(m.get('dim_model_base') or 1.0)
    if se == 1.0 and sd == 1.0 and db == 1.0:
        return m
    layers, dim = int(m['layers']), int(m['dim'])
    tied = not m.get('untied')
    k = 1.0 / se                     # the table stays as the checkpoint states it
    c = sd / math.sqrt(layers)
    m['wo'] = [w * (k * c) for w in m['wo']]
    m['w2'] = [w * (k * c) for w in m['w2']]
    g = ((db / dim) / (k * se)) if tied else (db / dim)
    m['rms_final'] = m['rms_final'] * g
    print("fold: scale_emb %g -> layer outputs x%.6f, table kept as written "
          "(k=1/scale_emb) | scale_depth %g -> o_proj/down_proj x%.6f | "
          "dim_model_base %g -> rms_final x%.6f (%s head)"
          % (se, k * c, sd, k * c, db, g, "tied" if tied else "untied"))
    return m



def load_marian(model_dir, seq=0, enc_seq=0):
    """Read a Marian encoder-decoder directory (OPUS-MT, `MarianMTModel`).

    The block is not a transformer decoder: `normalize_before: false` puts the
    LayerNorm *after* every residual add (post-norm), the decoder carries a
    cross-attention whose q is its own projection, the FFN is a plain
    fc1(swish)/fc2 with biases, and the head is `shared` plus a learned
    `final_logits_bias` (max |b| = 8.7, so it is not optional).  Positions are a
    *learned* sinusoid added to the embedding, identical for encoder and
    decoder, so there is no RoPE and no rope metadata.

    The checkpoint is a pre-1.6 bare pickle: `pth_legacy.py` reads it, because
    `zipfile` refuses the file.

    What the container gets, and why:
      * bias vectors and the position table as **f32 rows** (they are read once
        per layer and a quantised LayerNorm weight is a systematic channel
        gain), the matrices through the same q8/q4 path as every other model;
      * one row set for the dim-wide vectors and one for the ffn-wide ones,
        in the fixed order `BIAS_DIM_ORDER` / `BIAS_FFN_ORDER` documents;
      * encoder layers under their own kind ids, so the reader dispatches on
        the version and a v2 reader cannot half-load this.
    """
    import numpy as np
    from pth_legacy import load as load_legacy
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    # 0 means "this model's own capacity", resolved here: the encoder block below
    # is built before the position table is read, so a later default wrote
    # `seq 0` into the container and the engine refused every source.  A
    # hand-passed 64 is not a memory saving either -- it is a silent cap on every
    # source and target sentence, and the 512-row position table made the old
    # guard pass regardless.
    cap = int(cfg.get("max_position_embeddings") or 0)
    if cap:
        if not seq:
            seq = cap
        if not enc_seq:
            enc_seq = cap
    # `model.safetensors` first, `pytorch_model.bin` second.  Marian releases
    # ship both, and preferring the bare pickle meant downloading 296 MB we
    # already had -- or failing outright when the safetensors is the only one
    # fetched.  `pth_legacy` is still the path for a release that ships only the
    # pickle (the OPUS-MT family).
    if os.path.exists(model_dir + '/model.safetensors'):
        with open(model_dir + '/model.safetensors', 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            shdr = json.loads(f.read(n))
            sdata = f.read()

        def st_get(name):
            if name not in shdr:
                raise SystemExit("%s: the checkpoint has no %s; this is not the "
                                 "Marian layout this loader maps"
                                 % (model_dir, name))
            e = shdr[name]
            a, b = e['data_offsets']
            dt = st_dtype(e)
            if dt is None:
                raise SystemExit("%s: unsupported safetensors dtype %s"
                                 % (name, e.get('dtype')))
            if e.get("dtype") == "BF16":
                bits = np.frombuffer(sdata[a:b], dtype="<u2").astype(np.uint32) << 16
                return bits.view(np.float32).reshape(e['shape'])
            return np.frombuffer(sdata[a:b], dtype=dt).astype(np.float32).reshape(
                e['shape'])

        w = {k: st_get(k) for k in shdr if k != '__metadata__'}
    else:
        w = load_legacy(model_dir + '/pytorch_model.bin')

    def g(name):
        if name not in w:
            raise SystemExit("%s: the checkpoint has no %s; this is not the "
                             "Marian layout this loader maps" % (model_dir, name))
        return np.asarray(w[name], np.float32)

    dim = int(cfg['d_model'])
    layers = int(cfg['decoder_layers'])
    enc_layers = int(cfg['encoder_layers'])
    ffn = int(cfg['encoder_ffn_dim'])
    heads = int(cfg['encoder_attention_heads'])
    vocab = int(cfg['vocab_size'])
    post_norm = not bool(cfg.get('normalize_before', False))
    scale_emb = bool(cfg.get('scale_embedding', False))
    swish = cfg.get('activation_function', 'swish') == 'swish'
    if int(cfg['decoder_ffn_dim']) != ffn:
        raise SystemExit("%s: encoder and decoder FFN widths differ (%d/%d); "
                         "the container carries one" % (model_dir, ffn,
                                                        cfg['decoder_ffn_dim']))
    if not post_norm:
        raise SystemExit("%s: normalize_before is true; the post-norm block is "
                         "the only one this reader implements" % model_dir)
    if not swish:
        raise SystemExit("%s: activation %r; the shader carries swish"
                         % (model_dir, cfg.get('activation_function')))

    shared = g('model.shared.weight')
    # `share_encoder_decoder_embeddings: true` is the only form this loader
    # accepts, but a current export drops the redundant `embed_tokens` alias
    # entirely -- the tied matrix *is* `model.shared.weight`, so its absence is
    # the same statement, not a different layout.
    if ('model.encoder.embed_tokens.weight' in w and
            not np.array_equal(shared, g('model.encoder.embed_tokens.weight'))):
        raise SystemExit("%s: the encoder embedding is not `shared`; a tied "
                         "head is what this container assumes" % model_dir)

    m = dict(dim=dim, hidden=ffn, layers=layers, heads=heads, kv=heads,
             vocab=vocab, seq=seq, theta=10000.0, half=False, qk_norm=False,
             emb=shared, wcls=None, untied=False, arch='marian')
    m['enc'] = dict(layers=enc_layers, ffn=ffn, seq=enc_seq)
    m['post_norm'] = post_norm
    m['scale_embed'] = scale_emb
    m['nogate'] = True
    m['encdec'] = dict(enc_layers=enc_layers, enc_ffn=ffn, enc_seq=enc_seq,
                       post_norm=1 if post_norm else 0,
                       scale_embed=1 if scale_emb else 0, nogate=1)

    # Encoder: [q,k,v,o], fc1/fc2, two LayerNorms (post-norm).
    m['enc_qkv'], m['enc_wo'], m['enc_w1'], m['enc_w2'] = [], [], [], []
    m['enc_bias'], m['enc_bias_ffn'] = [], []
    for l in range(enc_layers):
        p = 'model.encoder.layers.%d.' % l
        m['enc_qkv'].append(np.concatenate(
            [g(p + 'self_attn.q_proj.weight'), g(p + 'self_attn.k_proj.weight'),
             g(p + 'self_attn.v_proj.weight')], axis=0))
        m['enc_wo'].append(g(p + 'self_attn.out_proj.weight'))
        m['enc_w1'].append(g(p + 'fc1.weight'))
        m['enc_w2'].append(g(p + 'fc2.weight'))
        m['enc_bias'].append([g(p + 'self_attn.q_proj.bias'),
                              g(p + 'self_attn.k_proj.bias'),
                              g(p + 'self_attn.v_proj.bias'),
                              g(p + 'self_attn.out_proj.bias'),
                              g(p + 'self_attn_layer_norm.weight'),
                              g(p + 'self_attn_layer_norm.bias'),
                              g(p + 'final_layer_norm.weight'),
                              g(p + 'final_layer_norm.bias'),
                              g(p + 'fc2.bias')])
        m['enc_bias_ffn'].append([g(p + 'fc1.bias')])

    # Decoder: self-attention, cross-attention (its own q/k/v/o), fc1/fc2,
    # three LayerNorms.
    m['wq'], m['wk'], m['wv'], m['wo'] = [], [], [], []
    m['cross_q'], m['cross_k'], m['cross_v'], m['cross_o'] = [], [], [], []
    m['w1'], m['w2'] = [], []
    m['dec_bias'], m['dec_bias_ffn'] = [], []
    for l in range(layers):
        p = 'model.decoder.layers.%d.' % l
        m['wq'].append(g(p + 'self_attn.q_proj.weight'))
        m['wk'].append(g(p + 'self_attn.k_proj.weight'))
        m['wv'].append(g(p + 'self_attn.v_proj.weight'))
        m['wo'].append(g(p + 'self_attn.out_proj.weight'))
        m['cross_q'].append(g(p + 'encoder_attn.q_proj.weight'))
        m['cross_k'].append(g(p + 'encoder_attn.k_proj.weight'))
        m['cross_v'].append(g(p + 'encoder_attn.v_proj.weight'))
        m['cross_o'].append(g(p + 'encoder_attn.out_proj.weight'))
        m['w1'].append(g(p + 'fc1.weight'))
        m['w2'].append(g(p + 'fc2.weight'))
        m['dec_bias'].append([g(p + 'self_attn.q_proj.bias'),
                              g(p + 'self_attn.k_proj.bias'),
                              g(p + 'self_attn.v_proj.bias'),
                              g(p + 'self_attn.out_proj.bias'),
                              g(p + 'encoder_attn.q_proj.bias'),
                              g(p + 'encoder_attn.k_proj.bias'),
                              g(p + 'encoder_attn.v_proj.bias'),
                              g(p + 'encoder_attn.out_proj.bias'),
                              g(p + 'self_attn_layer_norm.weight'),
                              g(p + 'self_attn_layer_norm.bias'),
                              g(p + 'encoder_attn_layer_norm.weight'),
                              g(p + 'encoder_attn_layer_norm.bias'),
                              g(p + 'final_layer_norm.weight'),
                              g(p + 'final_layer_norm.bias'),
                              g(p + 'fc2.bias')])
        m['dec_bias_ffn'].append([g(p + 'fc1.bias')])

    # Marian's position embedding is *sinusoidal*, so a checkpoint exported by
    # current transformers carries no `embed_positions` tensor at all
    # (`static_position_embeddings: true` means sinusoidal, not learned; the
    # table is a buffer, regenerated at load).  Synthesising it is exact: this
    # formula reproduces the OPUS-MT table -- which does ship one -- to
    # max|diff| 0.0.
    if 'model.encoder.embed_positions.weight' in w:
        m['pos'] = g('model.encoder.embed_positions.weight')
    else:
        rows = int(cfg.get('max_position_embeddings') or 512)
        half = dim // 2
        inv = 10000.0 ** (-np.arange(0, half, dtype=np.float64) / half)
        x = np.arange(rows, dtype=np.float64)[:, None] * inv[None, :]
        m['pos'] = np.concatenate([np.sin(x), np.cos(x)], axis=1).astype(np.float32)
    # 0 means "this model's own capacity".  A hand-passed 64 used to be the
    # default and it is not a memory saving, it is a silent cap: every source or
    # target longer than it is truncated, and the position table's 512 rows made
    # the guard below pass either way.
    cap = int(m['pos'].shape[0])
    if not seq:
        seq = cap
    if not enc_seq:
        enc_seq = cap
    if ('model.decoder.embed_positions.weight' in w and
            not np.array_equal(m['pos'], g('model.decoder.embed_positions.weight'))):
        raise SystemExit("%s: the two position tables differ; the container "
                         "carries one" % model_dir)
    if m['pos'].shape[0] < max(seq, enc_seq):
        raise SystemExit("%s: the position table has %d rows, the container "
                         "needs %d" % (model_dir, m['pos'].shape[0],
                                       max(seq, enc_seq)))
    m['logits_bias'] = g('final_logits_bias').reshape(-1)
    m['start_id'] = int(cfg['decoder_start_token_id'])
    m['eos_id'] = int(cfg['eos_token_id'])
    m['specials'] = dict(bos=None, eos=m['eos_id'], stop=-1,
                         pad=int(cfg['pad_token_id']), unk=1, im_start=-1)
    return m


def tensors_from_hf(model_dir, scheme, seq, m=None, rail_check=True, vocab_limit=None,
                    q8_row=False, head_dedup=False, head_q6=False, enc_seq=64):
    """Build the tensor list directly from an HF safetensors directory.

    Same list shape the legacy reader produces, so build() cannot tell them
    apart: i16 quantizes to int16 with a per-matrix scale, q8 is i16 then q8
    (that is what make_q8_from_i16 does, and the two-step order is why the
    shipped q8 containers look the way they do), q4 quantizes the float
    weights directly with block-32 min/scale and keeps embeddings and logits q8.
    """
    import numpy as np
    if m is None:
        with open(model_dir + '/config.json') as _f:
            _cfg = json.load(_f)
        if _cfg.get('model_type') == 'rwkv7':
            m = load_rwkv7(model_dir, seq)
        elif _cfg.get('model_type') == 'lfm2':
            m = load_lfm2(model_dir, seq)
        elif _cfg.get('model_type') == 'marian':
            m = load_marian(model_dir, seq, enc_seq=enc_seq)
        else:
            m = load_minimind(model_dir, seq)
    # HF pads the embedding to a multiple of its own choosing (16408 for a
    # 16384-token tokenizer), so the *tokenizer* decides how many rows the
    # container may claim: a row the tokenizer cannot produce is dead weight,
    # and claiming more rows than the .tok holds makes the engine read past the
    # end of the blob.
    if vocab_limit and vocab_limit != m['vocab']:
        if vocab_limit > m['vocab']:
            raise SystemExit("tokenizer has %d entries but the embedding only %d rows"
                             % (vocab_limit, m['vocab']))
        for k in ('emb', 'wcls'):
            if k in m and m[k] is not None:
                m[k] = m[k][:vocab_limit]
        m['vocab'] = vocab_limit
    # MiniCPM's three constants (a no-op for every other checkpoint).
    m = fold_scale_constants(m)
    (dim, hidden, layers, heads, kv, vocab, seq) = (
        m['dim'], m['hidden'], m['layers'], m['heads'], m['kv'], m['vocab'], m['seq'])
    head_dim = dim // heads
    qk = int(m.get('qk_norm', 0))
    moe = m.get('moe')
    # RWKV-7: a different block (docs/checkpoint-container.md, v4).  It reuses
    # 0x10/0x11/0x12/0x15 for its four big matrices and has no w3 at all, so
    # the container always says split_w13 and never writes a fused pair.
    rwkv = m.get('arch') == 'rwkv7'
    encdec = m.get('arch') == 'marian'
    enc_layers = int(m['enc']['layers']) if encdec else 0
    if encdec:
        # No rail measurement: every residual add in a post-norm block is
        # followed by a LayerNorm, so the stream is normalised at every layer
        # boundary and there is no growing sum to probe.  (The decoder-only
        # check exists because a pre-norm stream *does* grow; docs/benchmarks.md
        # has the two cases that had to be scaled.)
        rail_check = False
    # The FFN width that is written to the file is the expert width for a MoE
    # container: the dense mlp does not exist there.
    ffn_w = moe['hidden'] if moe else hidden
    # A MoE container always carries separate w1/w3 (there is no fused-W13
    # expert form) and the loader requires the flag to say so, so derive it
    # from the block, not from the dense 2*hidden > 2048 rule.  Without the
    # `moe` term a Qwen3-MoE whose expert width is <= 1024 packs split_w13 = 0
    # and the engine then refuses the file the packer just wrote.
    split = 1 if (moe or rwkv or encdec or 2 * ffn_w > 2048) else 0
    meta = dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv_heads=kv,
                vocab=vocab, seq=seq, head_dim=head_dim,
                theta=float(m['theta']), half=bool(m['half']), qk_norm=bool(qk),
                scheme=scheme, group=(32 if scheme == 'q4' else 0), split=bool(split),
                # key 18: a longrope model's static per-dimension table, and
                # key 10's third byte: the tokenizer's per-run space marker.
                rope_factors=m.get('rope_factors'),
                sp_prefix=int(m.get('sp_prefix') or 0))
    if m.get('layer_types'):
        meta['layer_types'] = list(m['layer_types'])
    if moe:
        meta['experts'] = int(moe['n_experts'])
        meta['top_k'] = int(moe['top_k'])
        meta['moe_hidden'] = int(moe['hidden'])
        meta['norm_topk'] = bool(moe['norm_topk'])
    if encdec:
        # Metadata key 20: the encoder block and the four block properties the
        # decoder-only path takes for granted.  key 21 is the two bias row
        # counts, whose row order mmcf_format.BIAS_DIM_ORDER documents.
        meta['encdec'] = dict(m['encdec'])
        meta['bias_rows'] = dict(
            n_dim=layers * len(m['dec_bias'][0]) + enc_layers * len(m['enc_bias'][0]),
            n_ffn=layers * 1 + enc_layers * 1)
    if rwkv:
        # Metadata key 16: arch, head width, flags, LayerNorm eps, four ranks.
        # The container is version 4 and its rectype is v3's 64-byte one.
        meta['rwkv'] = dict(m['rwkv'])
    # One q8 scale per matrix spends its 127 levels on the largest weight, so a
    # matrix whose rows differ by 500:1 keeps 127/500 of its detail and voids
    # most small weights to zero (void.1 layer 0: 58-66% of the FFN).  Say which
    # matrices are that skewed instead of leaving the reader to guess; a ratio
    # over ~100 is where --q8-row-scale starts to pay.
    skew = []

    def scan_skew(label, w):
        w = np.asarray(w, np.float32)
        med = float(np.median(np.abs(w)))
        if med <= 0:
            return
        skew.append((float(np.abs(w).max()) / med, label))

    for key in ('wq', 'wk', 'wv', 'wo', 'w1', 'w3', 'w2', 'wcls'):
        arr = m.get(key)
        if arr is None:
            continue
        for l in range(len(arr)):
            if arr[l] is not None:
                scan_skew('%s[%d]' % (key, l), arr[l])
    # The expert matrices were missing from this scan, and they are the largest
    # and least uniform matrices of a q8/q4 MoE: a 2432x768 expert at one scale
    # is exactly the case --q8-row-scale exists for.
    if rwkv:
        # The four big RWKV matrices are (3*dim, dim) / (dim, dim) /
        # (hidden, dim) / (dim, hidden): same question, same scan.
        for key in ('rkv', 'wo', 'cm_key', 'cm_value'):
            arr = m.get(key)
            if arr is None:
                continue
            for l in range(len(arr)):
                scan_skew('%s[%d]' % (key, l), arr[l])
    for key in ('expert_w1', 'expert_w3', 'expert_w2'):
        arr = m.get(key)
        if not arr:
            continue
        for l in range(len(arr)):
            for e in range(len(arr[l])):
                scan_skew('%s[%d].e%d' % (key, l, e), arr[l][e])
    if skew:
        skew.sort(reverse=True)
        worst = skew[0]
        print("q8 skew: worst %s max|w|/median|w| = %.0f (median over matrices %.0f)"
              % (worst[1], worst[0], float(np.median([x[0] for x in skew]))))
        if scheme == 'q8' and not q8_row and worst[0] > 100:
            print("note: that matrix loses most of its small weights at one scale; "
                  "--q8-row-scale writes one scale per output row instead")
    if rail_check:
        res_peaks, hid_peaks = rail_peaks(m)
        meta['res_peaks'] = res_peaks
        # The per-layer RMSNorm weight vectors are read in kLnPairFmt (I16_2:
        # 1/256, rail +-127.996) by the norm passes.  They are O(1) in every
        # container packed so far, but that is an observation; this records the
        # number so mmcf_pack.py can say it rather than assume it.
        if m.get('rms_att'):
            meta['norm_peaks'] = [
                float(max(np.abs(np.asarray(m['rms_att'][l], dtype=np.float32)).max(),
                          np.abs(np.asarray(m['rms_ffn'][l], dtype=np.float32)).max()))
                for l in range(m['layers'])]
        if hid_peaks is not None:
            meta['hid_peaks'] = hid_peaks
            # A hidden over the q16 activation rail is stored *2^-k (key 15)
            # instead of being clamped at run time: k is the smallest power of
            # two whose store fits, and the engine folds 2^k back into w2 after
            # load.  k is capped at 5 (the shader literal table's end).
            k = [0 if peak <= ACTIVATION_RAIL
                 else min(5, max(0, int(math.ceil(math.log2(peak / ACTIVATION_RAIL)))))
                 for peak in hid_peaks]
            if any(k):
                meta['hid_scale'] = k
    if rwkv and m.get('state_peaks'):
        # The WKV state's per-layer |max| is not a rail: it is the number that
        # decides the Pi's state encoding (docs/engine-internals.md).  It
        # is recorded nowhere in the file; mmcf_pack.py prints it.
        meta['state_peaks'] = list(m['state_peaks'])
    if rwkv and rail_check:
        # The head's input is the final LayerNorm's output, and it is written
        # through the q16 activation format like any other vector: 0.1B peaks
        # at 315.6, 2.5x the 127.996 rail.  Storing it *2^-k and multiplying
        # every logits tile's scale by 2^k keeps the dot product in the
        # original units, so metadata key 17 is one byte and the read side is
        # free.  The measured peak decides k; the analytic bound
        # (sqrt(dim)*max|w| + max|b|, i.e. one channel carrying all the
        # variance) is reported as the worst case it cannot reach.
        peak = float(m.get('final_ln_peak') or 0.0)
        meta['final_ln_peak'] = peak
        meta['final_ln_bound'] = m.get('final_ln_bound')
        if peak > ACTIVATION_RAIL:
            meta['logit_shift'] = min(5, max(0, int(
                math.ceil(math.log2(peak / ACTIVATION_RAIL)))))
    T = []

    def f32(a):
        return np.asarray(a, np.float32).tobytes()

    def i16_pair(w):
        q, s = qmat(np.asarray(w, np.float32))
        return q, float(s)

    def q8_from_i16(w):
        q, s = i16_pair(w)
        return q8mat(q.astype(np.float32) * s)

    def q8_per_row(w):
        """One scale per output row instead of one per matrix.

        A 640x1440 matrix whose weights span 500:1 loses 58-66% of its entries
        to zero at a single scale (void.1 layer 0); per row the median relative
        error falls from 2.36% to 0.79% and the worst matrix from 16.6% to 1.9%.
        Returns (q, unit, per_row) where unit is the largest row scale, so the
        plane the container carries is unit-relative and fits 24 bits.
        """
        w = np.asarray(w, np.float32)
        s = np.abs(w).max(axis=1, keepdims=True) / 127.0
        s[s == 0] = 1.0
        q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
        return q, float(s.max()), s[:, 0].astype(np.float32)

    if head_q6:
        # A 6-bit head: the embedding body is the two q6 planes and the logits
        # are rendered from them (dedup).  Only the q8/q4 containers use it; an
        # i16 container keeps i16 rows.
        if scheme == 'i16':
            raise SystemExit("--head-q6 is for q8/q4 containers")
        lo = np.empty((vocab, dim // 2), np.uint8)
        hi = np.empty((vocab, dim // 4), np.uint8)
        es = np.empty(vocab, np.float32)
        for i in range(vocab):
            lo[i], hi[i], es[i] = q6mat(np.asarray(m['emb'][i], np.float32))
        body = lo.tobytes() + hi.tobytes()
        T.append((T_EMB, DT_Q6, vocab, dim, 0, body, 0.0))
        T.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, es.tobytes(), 0.0))
        head_dedup = True
    elif scheme == 'i16':
        er = np.empty((vocab, dim), np.int16)
        es = np.empty(vocab, np.float32)
        for i in range(vocab):
            er[i], es[i] = i16_pair(m['emb'][i])
        T.append((T_EMB_I16, DT_I16, vocab, dim, 0, er.astype('<i2').tobytes(), 0.0))
        T.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, es.tobytes(), 0.0))
    else:
        er = np.empty((vocab, dim), np.int8)
        es = np.empty(vocab, np.float32)
        for i in range(vocab):
            if scheme == 'q8':
                er[i], es[i] = q8_from_i16(m['emb'][i])
            else:
                er[i], es[i] = q8mat(np.asarray(m['emb'][i], np.float32))
        T.append((T_EMB, DT_Q8I8, vocab, dim, 0, er.tobytes(), 0.0))
        T.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, es.tobytes(), 0.0))
    if rwkv:
        # The f32 per-channel vectors: one record per kind, rows = layers (1
        # for the two model-wide norms) and cols = groups * dim in the HF/GGUF
        # order, so a reader can diff a row against llama.cpp's tensor.
        for tid, arr, glob in ((K_RWKV_LERP, m['lerp'], False),
                               (K_RWKV_BIAS, m['bias'], False),
                               (K_RWKV_SCALE, m['scale'], False),
                               (K_RWKV_ATT_NORM, m['att_norm'], False),
                               (K_RWKV_ATT_NORM_2, m['att_norm_2'], False),
                               (K_RWKV_LN_X, m['ln_x'], False),
                               (K_RWKV_CM_LERP, m['cm_lerp'], False),
                               (K_RWKV_TOK_NORM, m.get('tok_norm'), True),
                               (K_RWKV_OUT_NORM, m['out_norm'], True)):
            if arr is None:
                continue
            a = np.asarray(arr, np.float32)
            if glob:
                T.append((tid, DT_F32, 1, int(a.size), 0, f32(a), 0.0))
            else:
                T.append((tid, DT_F32, layers, int(a.size) // layers, 0, f32(a), 0.0))
    elif encdec:
        # A post-norm block has no model-wide norm to carry: its LayerNorm
        # weights and biases are rows of the bias tensor, per layer, because
        # every layer has its own (three on the decoder, two on the encoder).
        pass
    else:
        T.append((T_RMS_ATT, DT_F32, layers, dim, 0, f32(m['rms_att']), 0.0))
        T.append((T_RMS_FFN, DT_F32, layers, dim, 0, f32(m['rms_ffn']), 0.0))
        T.append((T_RMS_FINAL, DT_F32, 1, dim, 0, f32(m['rms_final']), 0.0))
        if qk:
            T.append((T_Q_NORM, DT_F32, layers, head_dim, 0, f32(m['q_norm']), 0.0))
            T.append((T_K_NORM, DT_F32, layers, head_dim, 0, f32(m['k_norm']), 0.0))

    def mat(tid, layer, w, aux=0):
        rows, cols = int(w.shape[0]), int(w.shape[1])
        base = tid | (layer << 8)
        if scheme == 'i16':
            q, s = i16_pair(w)
            T.append((base, DT_I16, rows, cols, 0, q.astype('<i2').tobytes(), float(s), aux))
        elif scheme == 'q8' and q8_row:
            q, unit, srow = q8_per_row(w)
            T.append((base, DT_Q8I8, rows, cols, 0, q.tobytes(), float(unit), aux))
            # part 3 is the per-row plane: one f32 per output row, the row's
            # scale divided by the matrix unit the record already carries.
            T.append((base | (3 << 14), DT_F32, rows, 1, 0,
                      (srow / unit).astype('<f4').tobytes(), 0.0, aux))
        elif scheme == 'q8':
            q, s = q8_from_i16(w)
            T.append((base, DT_Q8I8, rows, cols, 0, q.tobytes(), float(s), aux))
        elif scheme == 't3':
            # A 2-bit ternary body plus the per-row plane the q8 path already
            # uses (part 3), so the engine reads a t3 matrix with the same row
            # scale code it reads a --q8-row-scale one.
            packed, unit, plane = t3mat(w)
            T.append((base, DT_T3, rows, cols, 0, packed, float(unit), aux))
            T.append((base | (3 << 14), DT_F32, rows, 1, 0,
                      plane.astype('<f4').tobytes(), 0.0, aux))
        else:
            packed, s16, mn, ps, groups = q4mat(np.asarray(w, np.float32))
            # The record's group byte is the q4 group size for the body
            # (docs/checkpoint-container.md); it used to be written as 0.  The
            # engine hardcodes 32, but a reader of the format should not have to.
            T.append((base, DT_Q4, rows, cols, 32, packed.tobytes(), float(ps), aux))
            T.append((base | (1 << 14), DT_I16, rows, groups, 0, s16.tobytes(), 0.0, aux))
            T.append((base | (2 << 14), DT_I16, rows, groups, 0, mn.tobytes(), 0.0, aux))

    ltypes = m.get('layer_types') or []
    for l in range(layers):
        if encdec:
            # The decoder half is the dense block minus the gate: w1 is fc1 and
            # there is no w3 (`nogate`), because the FFN is fc2(swish(fc1(x))).
            mat(T_QKV, l, np.concatenate([m['wq'][l], m['wk'][l], m['wv'][l]],
                                         axis=0))
            mat(T_WO, l, m['wo'][l])
            mat(T_W1, l, m['w1'][l])
            mat(T_W2, l, m['w2'][l])
            # Cross-attention: q from the decoder, k/v from the encoder, so all
            # four are the decoder's own matrices.
            mat(T_CROSS_Q, l, m['cross_q'][l])
            mat(T_CROSS_K, l, m['cross_k'][l])
            mat(T_CROSS_V, l, m['cross_v'][l])
            mat(T_CROSS_O, l, m['cross_o'][l])
            continue
        if rwkv:
            # 0x10 holds [W_r; W_k; W_v] as one row block (the plan's rkv
            # slot); 0x12/0x15 are the channel mix's key/value pair.  The four
            # low-rank pairs are q8 or q4 like every other matrix, and layer 0
            # simply has no value-residual pair to write.
            mat(T_QKV, l, m['rkv'][l])
            mat(T_WO, l, m['wo'][l])
            mat(T_W1, l, m['cm_key'][l])
            mat(T_W2, l, m['cm_value'][l])
            for br, kd, ku in (('w', K_RWKV_W1, K_RWKV_W2),
                               ('v', K_RWKV_V1, K_RWKV_V2),
                               ('a', K_RWKV_A1, K_RWKV_A2),
                               ('g', K_RWKV_G1, K_RWKV_G2)):
                pair = m['lora'].get(br)
                if pair is None or pair[0][l] is None:
                    continue
                mat(kd, l, pair[0][l])
                mat(ku, l, pair[1][l])
            continue
        if ltypes and ltypes[l] == 'conv':
            mat(T_CONV_IN, l, m['conv_in'][l])
            # The depthwise kernel is dim x k floats (24 KiB at most): f32, no
            # quantisation -- its per-channel scale is already the weight.
            cw = np.asarray(m['conv_w'][l], np.float32)
            T.append((T_CONV_W | (l << 8), DT_F32, int(cw.shape[0]), int(cw.shape[1]),
                      0, cw.tobytes(), 0.0))
            mat(T_CONV_OUT, l, m['conv_out'][l])
        else:
            mat(T_QKV, l, np.concatenate([m['wq'][l], m['wk'][l], m['wv'][l]], axis=0))
            mat(T_WO, l, m['wo'][l])
        if moe:
            # The router stays i16 in every container, q4 included: it is
            # 4x768 values and a route flip is not a rounding error, it is a
            # different FFN.  Top-1 with norm_topk_prob needs only the argmax.
            q, s = i16_pair(m['router'][l])
            T.append((T_ROUTER | (l << 8), DT_I16, moe['n_experts'], dim, 0,
                      q.astype('<i2').tobytes(), float(s), 0))
            for e in range(moe['n_experts']):
                mat(T_W1, l, m['expert_w1'][l][e], aux=e)
                mat(T_W3, l, m['expert_w3'][l][e], aux=e)
                mat(T_W2, l, m['expert_w2'][l][e], aux=e)
        elif split:
            mat(T_W1, l, m['w1'][l])
            mat(T_W3, l, m['w3'][l])
        else:
            mat(T_W13, l, np.concatenate([m['w1'][l], m['w3'][l]], axis=0))
        if not moe:
            mat(T_W2, l, m['w2'][l])
    if encdec:
        for l in range(m['enc']['layers']):
            mat(T_ENC_QKV, l, m['enc_qkv'][l])
            mat(T_ENC_WO, l, m['enc_wo'][l])
            mat(T_ENC_W1, l, m['enc_w1'][l])
            mat(T_ENC_W2, l, m['enc_w2'][l])
        # The bias rows, in the order mmcf_format.BIAS_DIM_ORDER names: every
        # decoder layer first, then every encoder layer.  An f32 row is 2 KiB,
        # and the engine reads these once at load.
        dim_rows = []
        for l in range(layers):
            dim_rows.extend(m['dec_bias'][l])
        for l in range(m['enc']['layers']):
            dim_rows.extend(m['enc_bias'][l])
        ffn_rows = []
        for l in range(layers):
            ffn_rows.extend(m['dec_bias_ffn'][l])
        for l in range(m['enc']['layers']):
            ffn_rows.extend(m['enc_bias_ffn'][l])
        T.append((T_BIAS_DIM, DT_F32, len(dim_rows), dim, 0,
                  np.asarray(dim_rows, np.float32).tobytes(), 0.0))
        # The fc1 biases are `hidden` wide, which is the container's FFN width.
        T.append((T_BIAS_FFN, DT_F32, len(ffn_rows), int(ffn_rows[0].size), 0,
                  np.asarray(ffn_rows, np.float32).tobytes(), 0.0))
    # Logits: q8 from the float weights in a q4 container (the vocabulary head
    # stays 8-bit there), but i16-then-q8 in a q8 container -- the two-step order
    # again, and it is visible in the shipped files.
    #
    # --head-dedup writes none: a tied head is the embedding matrix, the
    # embedding rows already carry one scale each, and the engine can render
    # the logits from those rows (and their plane) instead of a second copy.
    if head_dedup and m.get('untied'):
        raise SystemExit("--head-dedup needs a tied head: this checkpoint ships a "
                         "separate lm_head, so the logits cannot reuse the "
                         "embedding rows")
    if encdec:
        # The head *is* the embedding, so `--head-dedup` is the only form: the
        # logits come from the embedding rows and the learned bias is one f32
        # row under T_LOGITS_BIAS.  (A separate copy would be 33 MB of the same
        # bytes.)
        head_dedup = True
        T.append((T_LOGITS_BIAS, DT_F32, 1, vocab, 0,
                  f32(m['logits_bias']), 0.0))
        np_pos, dim_p = m['pos'].shape
        T.append((T_POS, DT_F32, int(min(np_pos, max(seq, m['enc']['seq']))),
                  dim, 0, f32(m['pos'][:min(np_pos, max(seq, m['enc']['seq']))]),
                  0.0))
    if not head_dedup:
        for i, st in enumerate(range(0, vocab, LOGITS_GRID)):
            w = m['wcls'][st:st + LOGITS_GRID]
            if scheme != 'q4':
                mat(T_LOGITS, i, w)
            else:
                q, sc = q8mat(np.asarray(w, np.float32))
                T.append((T_LOGITS | (i << 8), DT_Q8I8, int(w.shape[0]), int(w.shape[1]),
                          0, q.tobytes(), float(sc)))
    if meta.get('logit_shift'):
        # Metadata key 17: the head's input is written *2^-k, so every logits
        # matrix reads back *2^k.  Folding it into the per-tile scale is free
        # on the read side (`ws` already exists) and leaves the kernel alone.
        mult = float(1 << int(meta['logit_shift']))
        for i, rec in enumerate(T):
            if (rec[0] & 0xFF) == T_LOGITS and (rec[0] >> 14) == 0:
                T[i] = rec[:6] + (float(rec[6]) * mult,) + rec[7:]
    return meta, T


