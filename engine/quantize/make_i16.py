#!/usr/bin/env python3
"""Create I16M1 int16 checkpoints for the GLES2 engine.

Formats:
  llama2c  : llama2.c fp32 checkpoint (stories15M/TinyChat/stories42M)
  minimind : MiniMind2/MiniMind3 safetensors + config.json (auto V2 if q_norm)
"""
import argparse, json, struct
import numpy as np

MAGIC = 0x4931364D  # I16M
MAGIC_V2 = 0x4931364E  # I16N: adds q/k per-head RMSNorm

def qmat(w):
    mx = float(np.abs(w).max())
    if mx == 0: mx = 1.0
    s = mx / 32767.0
    q = np.clip(np.rint(w / s), -32767, 32767).astype(np.int16)
    return q, float(s)

def write_mat(f, w):
    q, s = qmat(np.asarray(w, dtype=np.float32))
    f.write(struct.pack('<iif', q.shape[0], q.shape[1], s))
    f.write(q.astype('<i2').tobytes())

def load_llama2c(path):
    b = np.fromfile(path, dtype=np.uint8)
    cfg = np.frombuffer(b[:28], dtype=np.int32).tolist()
    dim, hidden, layers, heads, kv, vsigned, seq = cfg
    vocab = abs(vsigned); shared = vsigned > 0
    hd = dim // heads; kv_dim = kv * hd
    off = 28
    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(b, dtype=np.float32, count=n, offset=off).reshape(shape)
        off += 4 * n
        return a
    emb = take((vocab, dim))
    rms_att = take((layers, dim))
    wq = take((layers, dim, dim)); wk = take((layers, kv_dim, dim)); wv = take((layers, kv_dim, dim))
    wo = take((layers, dim, dim)); rms_ffn = take((layers, dim))
    w1 = take((layers, hidden, dim)); w2 = take((layers, dim, hidden)); w3 = take((layers, hidden, dim))
    rms_final = take((dim,))
    off += 4 * seq * hd
    wcls = emb if shared else take((vocab, dim))
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab, seq=seq,
                emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=wcls,
                theta=10000.0, half=0)

def rope_of(cfg):
    """(theta, half) from a HF config.

    theta comes from config.json (rope_theta); half means "rotate half the head"
    (GPT-NeoX style, what Llama/HF does) as opposed to the interleaved llama2
    layout.  A model whose config says nothing gets the Llama default 10000 and
    rotate-half, which is what every HF Llama-shaped model does.
    """
    theta = float(cfg.get("rope_theta", 10000.0))
    rs = cfg.get("rope_scaling") or {}
    if isinstance(rs, dict) and "rope_theta" in rs:
        theta = float(rs["rope_theta"])
    # interleaved only if the config explicitly asks for the llama2 layout
    half = 0 if cfg.get("rope_interleaved") or cfg.get("rope_layout") == "interleaved" else 1
    return theta, half


def st_dtype(e):
    """safetensors dtype -> numpy, so an F32 or BF16 checkpoint is not read as F16."""
    return {"F16": "<f2", "F32": "<f4", "BF16": "<u2"}.get(e.get("dtype", "F16"))


def load_minimind(model_dir, seq=512):
    with open(model_dir + '/config.json') as f: cfg = json.load(f)
    with open(model_dir + '/model.safetensors', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n)); data = f.read()
    def get(name):
        e = hdr[name]; a, b = e['data_offsets']
        dt = st_dtype(e)
        if dt is None:
            raise SystemExit(f"{name}: unsupported safetensors dtype {e.get('dtype')}")
        if e.get("dtype") == "BF16":
            # bf16 is the top half of an f32; widening is exact and needs no
            # rounding, so a bf16 checkpoint converts losslessly.
            bits = np.frombuffer(data[a:b], dtype="<u2").astype(np.uint32) << 16
            return bits.view(np.float32).reshape(e['shape'])
        return np.frombuffer(data[a:b], dtype=dt).astype(np.float32).reshape(e['shape'])
    emb = get('model.embed_tokens.weight')
    vocab, dim = emb.shape
    layers = cfg['num_hidden_layers']; heads = cfg['num_attention_heads']; kv = cfg['num_key_value_heads']
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
    return dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv=kv, vocab=vocab, seq=seq,
                emb=emb, rms_att=rms_att, rms_ffn=rms_ffn, rms_final=rms_final,
                wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2, w3=w3, wcls=emb,
                theta=rope_of(cfg)[0], half=rope_of(cfg)[1],
                qk_norm=1 if qk else 0, q_norm=q_norm, k_norm=k_norm)

def write_i16(m, dst):
    dim=m['dim']; hidden=m['hidden']; layers=m['layers']; heads=m['heads']; kv=m['kv']; vocab=m['vocab']; seq=m['seq']
    split = 1 if 2*hidden > 2048 else 0
    qk = int(m.get('qk_norm', 0))
    with open(dst, 'wb') as f:
        if qk:
            f.write(struct.pack('<I7ifiii', MAGIC_V2, dim, hidden, layers, heads, kv, vocab, seq, m['theta'], m['half'], split, qk))
        else:
            f.write(struct.pack('<I7ifii', MAGIC, dim, hidden, layers, heads, kv, vocab, seq, m['theta'], m['half'], split))
        er = np.empty((vocab, dim), np.int16); es = np.empty(vocab, np.float32)
        for i in range(vocab):
            q, s = qmat(m['emb'][i]); er[i] = q; es[i] = s
        f.write(er.astype('<i2').tobytes()); f.write(es.tobytes())
        f.write(np.asarray(m['rms_att'], np.float32).tobytes())
        f.write(np.asarray(m['rms_ffn'], np.float32).tobytes())
        f.write(np.asarray(m['rms_final'], np.float32).tobytes())
        if qk:
            f.write(np.asarray(m['q_norm'], np.float32).tobytes())
            f.write(np.asarray(m['k_norm'], np.float32).tobytes())
        for l in range(layers):
            write_mat(f, np.concatenate([m['wq'][l], m['wk'][l], m['wv'][l]], axis=0))
            write_mat(f, m['wo'][l])
            if split:
                write_mat(f, m['w1'][l]); write_mat(f, m['w3'][l])
            else:
                write_mat(f, np.concatenate([m['w1'][l], m['w3'][l]], axis=0))
            write_mat(f, m['w2'][l])
        for st in range(0, vocab, 1536):
            write_mat(f, m['wcls'][st:st+1536])
    print(dst, 'dim', dim, 'hidden', hidden, 'layers', layers, 'vocab', vocab, 'seq', seq, 'split', split, 'qk_norm', qk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['llama2c','minimind'], required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq', type=int, default=512)
    a = ap.parse_args()
    if a.kind == 'llama2c': m = load_llama2c(a.input)
    else: m = load_minimind(a.input, a.seq)
    write_i16(m, a.output)

if __name__ == '__main__': main()
