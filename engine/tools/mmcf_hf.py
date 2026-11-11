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
import struct

import numpy as np

from mmcf_grid import LOGITS_GRID

DT_F32, DT_I16, DT_Q8I8, DT_Q4 = 0, 1, 3, 4
T_EMB, T_EMB_SCALE, T_RMS_ATT, T_RMS_FFN, T_RMS_FINAL = 1, 2, 3, 4, 5
T_Q_NORM, T_K_NORM = 6, 7
T_QKV, T_WO, T_W1, T_W3, T_W13, T_W2 = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
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
        if kind not in (None, 'default'):
            raise SystemExit(
                "%s: rope_scaling %r is not representable; the container carries a "
                "theta and a half/plain convention, nothing else." % (arch, kind))
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


def load_minimind(model_dir, seq=512):
    """Read a HF Llama/Qwen-shaped directory into float arrays."""
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    with open(model_dir + '/model.safetensors', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    check_source(cfg, hdr)

    def get(name):
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
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab, seq=seq,
                emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=wcls,
                theta=theta, half=half, qk_norm=1 if qk else 0,
                q_norm=q_norm, k_norm=k_norm, untied='lm_head.weight' in hdr,
                moe=moe, router=router, expert_w1=expert_w1, expert_w2=expert_w2,
                expert_w3=expert_w3)


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
                conv_k=ksize, untied=False, moe=None)


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


def residual_peaks(m, seq=8):
    """max|x| of the residual stream per layer, for a few token samples.

    A float32 forward from a deterministic sample of vocabulary ids: three runs
    of `seq` consecutive ids (the start of the vocabulary, the middle and the
    end), taking the maximum over both the samples and the positions.  It needs
    no tokenizer, so it works for every source the packer accepts.
    """
    if m.get('layer_types'):
        return residual_peaks_lfm2(m, seq)
    dim, hidden, layers = m['dim'], m['hidden'], m['layers']
    heads, kv, vocab = m['heads'], m['kv'], m['vocab']
    hd = dim // heads
    theta = float(m['theta'])
    half = bool(m['half'])
    eps = float(m.get('rms_eps') or 1e-5)
    qk = bool(m.get('qk_norm'))
    emb = m['emb']
    peaks = [0.0] * layers
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
                ang = np.arange(T)[:, None] / (theta ** (2.0 * i / hd))
                c, s = np.cos(ang), np.sin(ang)
                o[:, :, i] = a[:, :, i] * c - a[:, :, i + hd // 2] * s
                o[:, :, i + hd // 2] = a[:, :, i + hd // 2] * c + a[:, :, i] * s
            else:
                ang = np.arange(T)[:, None] / (theta ** (2.0 * i / hd))
                c, s = np.cos(ang), np.sin(ang)
                o[:, :, 2 * i] = a[:, :, 2 * i] * c - a[:, :, 2 * i + 1] * s
                o[:, :, 2 * i + 1] = a[:, :, 2 * i + 1] * c + a[:, :, 2 * i] * s
        return o

    rep = heads // kv
    starts = [0, max(0, vocab // 2 - seq // 2), max(0, vocab - seq)]
    for st in starts:
        ids = list(range(st, st + seq))
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
                    x[t] = x1[t] + ((g / (1 + np.exp(-g))) * u) @ w2e.T
            else:
                g = h2 @ m['w1'][l].T
                u = h2 @ m['w3'][l].T
                x = x1 + ((g / (1 + np.exp(-g))) * u) @ m['w2'][l].T
            mx = float(np.abs(x).max())
            if not np.isfinite(mx):
                raise SystemExit("residual calibration produced %s at layer %d; the "
                                 "check cannot vouch for this model" % (mx, l))
            peaks[l] = max(peaks[l], mx)
    return peaks


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


def tensors_from_hf(model_dir, scheme, seq, m=None, rail_check=True, vocab_limit=None,
                    q8_row=False, head_dedup=False):
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
        m = load_lfm2(model_dir, seq) if _cfg.get('model_type') == 'lfm2' \
            else load_minimind(model_dir, seq)
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
    (dim, hidden, layers, heads, kv, vocab, seq) = (
        m['dim'], m['hidden'], m['layers'], m['heads'], m['kv'], m['vocab'], m['seq'])
    head_dim = dim // heads
    qk = int(m.get('qk_norm', 0))
    moe = m.get('moe')
    # The FFN width that is written to the file is the expert width for a MoE
    # container: the dense mlp does not exist there.
    ffn_w = moe['hidden'] if moe else hidden
    split = 1 if 2 * ffn_w > 2048 else 0
    meta = dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv_heads=kv,
                vocab=vocab, seq=seq, head_dim=head_dim,
                theta=float(m['theta']), half=bool(m['half']), qk_norm=bool(qk),
                scheme=scheme, group=(32 if scheme == 'q4' else 0), split=bool(split))
    if m.get('layer_types'):
        meta['layer_types'] = list(m['layer_types'])
    if moe:
        meta['experts'] = int(moe['n_experts'])
        meta['top_k'] = int(moe['top_k'])
        meta['moe_hidden'] = int(moe['hidden'])
        meta['norm_topk'] = bool(moe['norm_topk'])
    # One q8 scale per matrix spends its 127 levels on the largest weight, so a
    # matrix whose rows differ by 500:1 keeps 127/500 of its detail and voids
    # most small weights to zero (void.1 layer 0: 58-66% of the FFN).  Say which
    # matrices are that skewed instead of leaving the reader to guess; a ratio
    # over ~100 is where --q8-row-scale starts to pay.
    skew = []
    for key in ('wq', 'wk', 'wv', 'wo', 'w1', 'w3', 'w2', 'wcls'):
        arr = m.get(key)
        if arr is None:
            continue
        n_rows = len(arr) if isinstance(arr, list) else int(getattr(arr, 'shape', (0,))[0])
        for l in range(n_rows):
            if arr[l] is None:
                continue
            w = np.asarray(arr[l], np.float32)
            med = float(np.median(np.abs(w)))
            if med <= 0:
                continue
            skew.append((float(np.abs(w).max()) / med, key, l))
    if skew:
        skew.sort(reverse=True)
        worst = skew[0]
        print("q8 skew: worst %s[%d] max|w|/median|w| = %.0f (median over matrices %.0f)"
              % (worst[1], worst[2], worst[0], float(np.median([x[0] for x in skew]))))
        if scheme == 'q8' and not q8_row and worst[0] > 100:
            print("note: that matrix loses most of its small weights at one scale; "
                  "--q8-row-scale writes one scale per output row instead")
    if rail_check:
        meta['res_peaks'] = residual_peaks(m)
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

    if scheme == 'i16':
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
        else:
            packed, s16, mn, ps, groups = q4mat(np.asarray(w, np.float32))
            T.append((base, DT_Q4, rows, cols, 0, packed.tobytes(), float(ps), aux))
            T.append((base | (1 << 14), DT_I16, rows, groups, 0, s16.tobytes(), 0.0, aux))
            T.append((base | (2 << 14), DT_I16, rows, groups, 0, mn.tobytes(), 0.0, aux))

    ltypes = m.get('layer_types') or []
    for l in range(layers):
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
    if not head_dedup:
        for i, st in enumerate(range(0, vocab, LOGITS_GRID)):
            w = m['wcls'][st:st + LOGITS_GRID]
            if scheme != 'q4':
                mat(T_LOGITS, i, w)
            else:
                q, sc = q8mat(np.asarray(w, np.float32))
                T.append((T_LOGITS | (i << 8), DT_Q8I8, int(w.shape[0]), int(w.shape[1]),
                          0, q.tobytes(), float(sc)))
    return meta, T


