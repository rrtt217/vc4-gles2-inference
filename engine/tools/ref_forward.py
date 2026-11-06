#!/usr/bin/env python3
"""An independent numpy forward pass over an HF checkpoint.

    ref_forward.py models/smollm2-135m "The capital of France is"

Prints the token ids the HF tokenizer produces and the top-5 next tokens from
an f32 forward pass.  It exists because "the engine's text is wrong" has three
very different causes - a bad conversion, a bad tokenizer, or an engine bug -
and this separates them: if this tool predicts a sensible token and the engine
does not, the weights are fine and the engine is wrong.

Reads safetensors directly (F16 / F32 / BF16, single file), so it needs only
numpy plus the tokenizer.json."""

import json
import struct
import sys

import numpy as np


def load(path):
    f = open(path,'rb'); n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n)); data = f.read()
    def get(name):
        e = hdr[name]; a,b = e['data_offsets']
        dt = e['dtype']
        if dt == 'BF16':
            return (np.frombuffer(data[a:b], dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(e['shape']).astype(np.float32)
        if dt == 'F16':
            return np.frombuffer(data[a:b], dtype='<f2').astype(np.float32).reshape(e['shape'])
        return np.frombuffer(data[a:b], dtype='<f4').reshape(e['shape'])
    return get, hdr

def rms(x, w, eps):
    return x / np.sqrt((x*x).mean(-1, keepdims=True) + eps) * w

def rope(x, theta, pos):
    # x: (heads, hd); rotate_half (HF Llama)
    hd = x.shape[-1]; half = hd//2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1*c - x2*s, x2*c + x1*s], axis=-1)

def main(model, ids, theta, eps=1e-5):
    get, hdr = load(model + "/model.safetensors")
    cfg = json.load(open(model + "/config.json"))
    D = cfg['hidden_size']; H = cfg['num_attention_heads']; KV = cfg['num_key_value_heads']
    hd = D // H; L = cfg['num_hidden_layers']
    emb = get('model.embed_tokens.weight')
    x = emb[ids].astype(np.float32)          # (T, D)
    T = len(ids)
    for l in range(L):
        an = get(f'model.layers.{l}.input_layernorm.weight')
        fn = get(f'model.layers.{l}.post_attention_layernorm.weight')
        wq = get(f'model.layers.{l}.self_attn.q_proj.weight')   # (D, D)
        wk = get(f'model.layers.{l}.self_attn.k_proj.weight')   # (KV*hd, D)
        wv = get(f'model.layers.{l}.self_attn.v_proj.weight')
        wo = get(f'model.layers.{l}.self_attn.o_proj.weight')
        w1 = get(f'model.layers.{l}.mlp.gate_proj.weight')
        w2 = get(f'model.layers.{l}.mlp.down_proj.weight')
        w3 = get(f'model.layers.{l}.mlp.up_proj.weight')
        h = rms(x, an, eps)
        q = (h @ wq.T).reshape(T, H, hd)
        k = (h @ wk.T).reshape(T, KV, hd)
        v = (h @ wv.T).reshape(T, KV, hd)
        q = np.stack([rope(q[t], theta, t) for t in range(T)])
        k = np.stack([rope(k[t], theta, t) for t in range(T)])
        # GQA: kv head per query head
        rep = H // KV
        kk = np.repeat(k, rep, axis=1); vv = np.repeat(v, rep, axis=1)
        out = np.zeros((T, H, hd), np.float32)
        for t in range(T):
            sc = np.einsum('hd,jhd->hj', q[t], kk[:t+1]) / np.sqrt(hd)
            p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            out[t] = np.einsum('hj,jhd->hd', p, vv[:t+1])
        x = x + (out.reshape(T, D) @ wo.T)
        h = rms(x, fn, eps)
        g = h @ w1.T; u = h @ w3.T
        x = x + ((g / (1 + np.exp(-g))) * u) @ w2.T
    x = rms(x, get('model.norm.weight'), eps)
    logits = x[-1] @ emb.T
    return logits

if __name__ == '__main__':
    from tokenizers import Tokenizer
    model = sys.argv[1]; text = sys.argv[2]
    t = Tokenizer.from_file(model + "/tokenizer.json")
    ids = t.encode(text).ids
    theta = float(json.load(open(model + "/config.json")).get("rope_theta") or 10000.0)
    lg = main(model, ids, theta)
    top = np.argsort(-lg)[:5]
    print("ids:", ids)
    print("argmax:", top[0], repr(t.decode([int(top[0])])))
    print("top5:", [(int(i), repr(t.decode([int(i)]))) for i in top])
