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

DT_F32, DT_I16, DT_Q8I8, DT_Q4 = 0, 1, 3, 4
T_EMB, T_EMB_SCALE, T_RMS_ATT, T_RMS_FFN, T_RMS_FINAL = 1, 2, 3, 4, 5
T_Q_NORM, T_K_NORM = 6, 7
T_QKV, T_WO, T_W1, T_W3, T_W13, T_W2 = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
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


def load_minimind(model_dir, seq=512):
    """Read a HF Llama/Qwen-shaped directory into float arrays."""
    with open(model_dir + '/config.json') as f:
        cfg = json.load(f)
    with open(model_dir + '/model.safetensors', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        data = f.read()

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
    layers = cfg['num_hidden_layers']
    heads = cfg['num_attention_heads']
    kv = cfg['num_key_value_heads']
    hidden = cfg['intermediate_size']
    wq = np.stack([get(f'model.layers.{l}.self_attn.q_proj.weight') for l in range(layers)])
    wk = np.stack([get(f'model.layers.{l}.self_attn.k_proj.weight') for l in range(layers)])
    wv = np.stack([get(f'model.layers.{l}.self_attn.v_proj.weight') for l in range(layers)])
    wo = np.stack([get(f'model.layers.{l}.self_attn.o_proj.weight') for l in range(layers)])
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
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=emb,
                theta=theta, half=half, qk_norm=1 if qk else 0,
                q_norm=q_norm, k_norm=k_norm)


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


def residual_peaks(m, seq=8):
    """max|x| of the residual stream per layer, for a few token samples.

    A float32 forward from a deterministic sample of vocabulary ids: three runs
    of `seq` consecutive ids (the start of the vocabulary, the middle and the
    end), taking the maximum over both the samples and the positions.  It needs
    no tokenizer, so it works for every source the packer accepts.
    """
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


def tensors_from_hf(model_dir, scheme, seq, m=None, rail_check=True, vocab_limit=None):
    """Build the tensor list directly from an HF safetensors directory.

    Same list shape the legacy reader produces, so build() cannot tell them
    apart: i16 quantizes to int16 with a per-matrix scale, q8 is i16 then q8
    (that is what make_q8_from_i16 does, and the two-step order is why the
    shipped q8 containers look the way they do), q4 quantizes the float
    weights directly with block-32 min/scale and keeps embeddings and logits q8.
    """
    import numpy as np
    if m is None:
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
    (dim, hidden, layers, heads, kv, vocab, seq) = (
        m['dim'], m['hidden'], m['layers'], m['heads'], m['kv'], m['vocab'], m['seq'])
    head_dim = dim // heads
    qk = int(m.get('qk_norm', 0))
    split = 1 if 2 * hidden > 2048 else 0
    meta = dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv_heads=kv,
                vocab=vocab, seq=seq, head_dim=head_dim,
                theta=float(m['theta']), half=bool(m['half']), qk_norm=bool(qk),
                scheme=scheme, group=(32 if scheme == 'q4' else 0), split=bool(split))
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

    def mat(tid, layer, w):
        rows, cols = int(w.shape[0]), int(w.shape[1])
        base = tid | (layer << 8)
        if scheme == 'i16':
            q, s = i16_pair(w)
            T.append((base, DT_I16, rows, cols, 0, q.astype('<i2').tobytes(), float(s)))
        elif scheme == 'q8':
            q, s = q8_from_i16(w)
            T.append((base, DT_Q8I8, rows, cols, 0, q.tobytes(), float(s)))
        else:
            packed, s16, mn, ps, groups = q4mat(np.asarray(w, np.float32))
            T.append((base, DT_Q4, rows, cols, 0, packed.tobytes(), float(ps)))
            T.append((base | (1 << 14), DT_I16, rows, groups, 0, s16.tobytes(), 0.0))
            T.append((base | (2 << 14), DT_I16, rows, groups, 0, mn.tobytes(), 0.0))

    for l in range(layers):
        mat(T_QKV, l, np.concatenate([m['wq'][l], m['wk'][l], m['wv'][l]], axis=0))
        mat(T_WO, l, m['wo'][l])
        if split:
            mat(T_W1, l, m['w1'][l])
            mat(T_W3, l, m['w3'][l])
        else:
            mat(T_W13, l, np.concatenate([m['w1'][l], m['w3'][l]], axis=0))
        mat(T_W2, l, m['w2'][l])
    # Logits: q8 from the float weights in a q4 container (the vocabulary head
    # stays 8-bit there), but i16-then-q8 in a q8 container -- the two-step order
    # again, and it is visible in the shipped files.
    for i, st in enumerate(range(0, vocab, 1536)):
        w = m['wcls'][st:st + 1536]
        if scheme != 'q4':
            mat(T_LOGITS, i, w)
        else:
            q, sc = q8mat(np.asarray(w, np.float32))
            T.append((T_LOGITS | (i << 8), DT_Q8I8, int(w.shape[0]), int(w.shape[1]),
                      0, q.tobytes(), float(sc)))
    return meta, T


