# -*- coding: utf-8 -*-
"""The BiCrossGate FFN (MiniYuanYuan), as a loader and a tensor emitter.

`mmcf_hf.load_minimind` reads a Llama/Qwen directory whose FFN is
`down(silu(gate(x)) * up(x))`.  MiniYuanYuan's FFN is a different block:

    q1, q2 = q(x).chunk(2)              # q: dim -> 2*dim
    a = act1(q1 * x)                    # elementwise against the block's input
    c = act2(q2 * x)
    y = down( gate(concat(a, c)) * up(x) )

so it needs two matrices a SwiGLU does not have -- `q` (dim -> 2*dim) and
`gate_proj` (2*dim -> hidden) -- while `up`/`down` keep the w1/w2 slots.  The
block is refused by `check_source`'s `hidden_act` rule (gelu is not silu) and by
the tensor-name map (there is no `mlp.*`), so the reader lives here rather than
being bolted onto `load_minimind`.

The checkpoint is a torch zip with a training wrapper (`{"model": state_dict,
"cfg": {...}}`), which `pth_reader.load(..., meta=True)` already unwraps.

    python3 engine/tools/mmcf_pack.py --in models/miniyuanyuan/model.pt \
      --quant q4 --seq 512 --tok models/miniyuanyuan/miniyuanyuan.tok \
      --out models/miniyuanyuan/miniyuanyuan.q4.mmcf \
      --name miniyuanyuan --template miniyuanyuan --chat \
      --chat-template models/miniyuanyuan/chat_template.jinja \
      --tok-greedy --rms-store auto --res auto
"""
import numpy as np

# The activation codes metadata key 23 carries (model.cpp's key-23 branch).
ACT_SILU, ACT_GELU = 1, 2

# The reference's `qbilinear_act1` / `qbilinear_act2` names.
_ACTS = {"silu": ACT_SILU, "swish": ACT_SILU, "gelu": ACT_GELU}


def _act_code(name, where, cfg):
    a = str(name or "").lower()
    if a not in _ACTS:
        raise SystemExit(
            "%s is %r; the BiCrossGate block's kernels implement silu and gelu(erf) "
            "only.  A tanh or sigmoid gate would need a third one." % (where, a))
    return _ACTS[a]


def load_pt(path, seq=None):
    """Read a wrapped `{model, cfg}` torch zip into the canonical dict.

    The dict is the same shape `load_minimind` returns, so `tensors_from_hf`
    carries it through unchanged; the two extra entries are the ones this block
    needs and no other reader produces.
    """
    import pth_reader
    t, rest = pth_reader.load(path, meta=True)
    cfg = rest.get("cfg")
    if not cfg:
        raise SystemExit(
            "%s: no `cfg` in the checkpoint's wrapper.  This loader is for a "
            "training-script export (`{'model': state_dict, 'cfg': {...}}`), not a "
            "bare state dict -- for that, convert to an HF directory first." % path)

    dim = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv = int(cfg["num_key_value_heads"])
    # head_dim is stated separately by this family (32 for a 384-wide head
    # count) and the container derives it as dim/heads, so a disagreement is a
    # real error rather than a preference.
    hd = int(cfg.get("head_dim") or (dim // heads))
    if hd != dim // heads:
        raise SystemExit(
            "cfg says head_dim %d but hidden_size/num_attention_heads is %d/%d = %d.  "
            "The container derives the head width from dim and heads, so this "
            "checkpoint's shape cannot be expressed." % (hd, dim, heads, dim // heads))
    if heads % kv:
        raise SystemExit("num_attention_heads %d is not a multiple of "
                         "num_key_value_heads %d" % (heads, kv))
    if int(cfg.get("mtp") or 0):
        raise SystemExit("cfg has mtp=%r; the multi-token-prediction heads are not "
                         "part of the block this engine runs" % cfg.get("mtp"))

    hidden = int(cfg["intermediate_size"])
    share = int(cfg.get("ffn_share_group") or 1)
    if share < 1:
        raise SystemExit("ffn_share_group is %r" % (cfg.get("ffn_share_group"),))
    n_groups = (layers + share - 1) // share

    def get(name):
        if name not in t:
            raise SystemExit("%s: the checkpoint has no %r; this is not the "
                             "BiCrossGate layout this loader maps" % (path, name))
        return np.asarray(t[name], np.float32)

    emb = get("model.embed_tokens.weight")
    vocab = int(emb.shape[0])
    if tuple(emb.shape) != (vocab, dim):
        raise SystemExit("embedding is %s, expected (%d, %d)" % (list(emb.shape), vocab, dim))

    wcls = emb
    if "lm_head.weight" in t:
        wcls = get("lm_head.weight")
        if wcls.shape != emb.shape:
            raise SystemExit("lm_head %s does not match the embedding %s"
                             % (list(wcls.shape), list(emb.shape)))
    elif cfg.get("tie_word_embeddings") is False:
        raise SystemExit("cfg says tie_word_embeddings: false but there is no "
                         "lm_head.weight; refusing to pack a model whose head would "
                         "silently become its embedding")

    def stack(fmt):
        return np.stack([get(fmt % l) for l in range(layers)])

    wq = stack("model.layers.%d.self_attn.q_proj.weight")
    wk = stack("model.layers.%d.self_attn.k_proj.weight")
    wv = stack("model.layers.%d.self_attn.v_proj.weight")
    wo = stack("model.layers.%d.self_attn.o_proj.weight")
    rms_att = stack("model.layers.%d.input_layernorm.weight")
    rms_ffn = stack("model.layers.%d.post_attention_layernorm.weight")
    rms_final = get("model.norm.weight")

    qk = ("model.layers.0.self_attn.q_norm.weight" in t)
    q_norm = (np.stack([get("model.layers.%d.self_attn.q_norm.weight" % l)
                        for l in range(layers)]) if qk else None)
    k_norm = (np.stack([get("model.layers.%d.self_attn.k_norm.weight" % l)
                        for l in range(layers)]) if qk else None)

    # The FFN groups.  A shared group is stored once and referenced by every
    # layer that uses it: `ffn_groups` is already deduplicated in the
    # checkpoint, so the per-layer arrays below repeat a group's matrices and
    # the container's aliasing is what collapses them back to one blob.
    group = {}
    for g in range(n_groups):
        pre = "model.ffn_groups.%d." % g
        group[g] = dict(q=get(pre + "q.weight"), gate=get(pre + "gate_proj.weight"),
                        up=get(pre + "up_proj.weight"), down=get(pre + "down_proj.weight"))
    ffn_q = np.stack([group[l // share]["q"] for l in range(layers)])
    ffn_gate = np.stack([group[l // share]["gate"] for l in range(layers)])
    w1 = np.stack([group[l // share]["up"] for l in range(layers)])     # up -> w1
    w2 = np.stack([group[l // share]["down"] for l in range(layers)])   # down -> w2

    # Shape checks, because the dense path validates almost nothing and this is
    # the reader that knows what the block is.
    want = dict(q=(layers, 2 * dim, dim), gate=(layers, hidden, 2 * dim),
                up=(layers, hidden, dim), down=(layers, dim, hidden))
    got = dict(q=ffn_q, gate=ffn_gate, up=w1, down=w2)
    for k, shape in want.items():
        if tuple(got[k].shape) != shape:
            raise SystemExit("ffn %s is %s, the block wants %s"
                             % (k, list(got[k].shape), list(shape)))
    if tuple(wq.shape) != (layers, dim, dim) or tuple(wo.shape) != (layers, dim, dim):
        raise SystemExit("q_proj/ o_proj are %s / %s, expected (%d, %d, %d)"
                         % (list(wq.shape), list(wo.shape), layers, dim, dim))
    if tuple(wk.shape) != (layers, kv * hd, dim) or tuple(wv.shape) != (layers, kv * hd, dim):
        raise SystemExit("k_proj/ v_proj are %s / %s, expected (%d, %d, %d)"
                         % (list(wk.shape), list(wv.shape), layers, kv * hd, dim))

    theta = float(cfg.get("rope_theta") or 1e6)
    if cfg.get("rope_scaling"):
        raise SystemExit("cfg has rope_scaling=%r; the container carries a theta and a "
                         "half/plain convention only" % (cfg.get("rope_scaling"),))
    qb1 = cfg.get("qbilinear_act1") or "gelu"
    qb2 = cfg.get("qbilinear_act2") or "silu"

    return dict(
        dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab,
        seq=int(seq or cfg.get("max_position_embeddings") or 512),
        emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
        wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=None, wcls=wcls,
        theta=theta, half=True, rope_factors=None, qk_norm=1 if qk else 0,
        q_norm=q_norm, k_norm=k_norm, untied=("lm_head.weight" in t),
        scale_emb=1.0, scale_depth=1.0, dim_model_base=1.0, sp_prefix=0,
        moe=None, router=None, expert_w1=None, expert_w2=None, expert_w3=None,
        added_ids=[],
        # The two entries no other reader produces.
        bicross=dict(act1=_act_code(qb1, "qbilinear_act1", cfg),
                     act2=_act_code(qb2, "qbilinear_act2", cfg),
                     q=ffn_q, gate=ffn_gate),
        rms_eps=float(cfg["rms_norm_eps"]) if cfg.get("rms_norm_eps") else None,
    )
