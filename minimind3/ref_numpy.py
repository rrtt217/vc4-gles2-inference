#!/usr/bin/env python3
"""NumPy reference for MiniMind3 64M (Qwen3-style q/k norm + RoPE + SwiGLU)."""
import argparse, json, struct, math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n)); data = f.read()
    out = {}
    for name, e in hdr.items():
        if name == '__metadata__': continue
        a, b = e['data_offsets']
        dt = {'F16': '<f2', 'F32': '<f4'}[e['dtype']]
        out[name] = np.frombuffer(data[a:b], dtype=dt).astype(np.float32).reshape(e['shape'])
    return out

def rms(x, w, eps):
    return (x * (1.0 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps))) * w

def rope(x, pos, theta, head_dim):
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    freq = 1.0 / (theta ** (2.0 * i / head_dim))
    ang = pos * freq
    c = np.cos(ang).astype(np.float32)[None, :]
    s = np.sin(ang).astype(np.float32)[None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1*c - x2*s, x1*s + x2*c], axis=-1)

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument('--model', default=str(here / '../models/minimind3/model.safetensors'))
    ap.add_argument('--tokenizer', default='tokenizer.json')
    ap.add_argument('--system', default='你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。')
    ap.add_argument('--user', default='请用一句话介绍你自己。')
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--ignore-eos', action='store_true')
    ap.add_argument('--variant', default='qk', choices=['qk','none','after','noweight','qonly','konly','no-rope'])
    ap.add_argument('--wbits', type=int, default=0, choices=[0,4,6,8])
    ap.add_argument('--group', type=int, default=0, help='weight quantization group size; 0=per-row')
    ap.add_argument('--quant-embed', action='store_true')
    ap.add_argument('--scope', default='all', choices=['all','ffn','attn'])
    ap.add_argument('--asym', action='store_true', help='blockwise asymmetric min/scale')
    ap.add_argument('--attn-bits', type=int, default=0, choices=[0,4,6,8])
    ap.add_argument('--ffn-bits', type=int, default=0, choices=[0,4,6,8])
    ap.add_argument('--out-prefix', default='ref')
    a = ap.parse_args()

    cfg = json.load(open(here / '../models/minimind3/config.json'))
    w = load_safetensors(a.model)
    if a.wbits or a.attn_bits or a.ffn_bits:
        def qmat(x, bits, group, asym):
            levels = (1 << bits) - 1 if asym else (1 << (bits - 1)) - 1
            orig = x.shape
            y = x.reshape(-1, orig[-1])
            if group and group > 0:
                pad = (-y.shape[1]) % group
                if pad:
                    y = np.pad(y, ((0, 0), (0, pad)))
                z = y.reshape(y.shape[0], -1, group)
                if asym:
                    lo = z.min(axis=2, keepdims=True)
                    hi = z.max(axis=2, keepdims=True)
                    sc = (hi - lo) / levels
                    sc[sc == 0] = 1.0
                    z = np.clip(np.rint((z - lo) / sc), 0, levels) * sc + lo
                else:
                    mx = np.abs(z).max(axis=2, keepdims=True)
                    mx[mx == 0] = 1.0
                    z = np.clip(np.rint(z / (mx / levels)), -levels, levels) * (mx / levels)
                y = z.reshape(y.shape[0], -1)[:, :orig[-1]]
            else:
                if asym:
                    lo = y.min(axis=1, keepdims=True)
                    hi = y.max(axis=1, keepdims=True)
                    sc = (hi - lo) / levels
                    sc[sc == 0] = 1.0
                    y = np.clip(np.rint((y - lo) / sc), 0, levels) * sc + lo
                else:
                    mx = np.abs(y).max(axis=1, keepdims=True)
                    mx[mx == 0] = 1.0
                    y = np.clip(np.rint(y / (mx / levels)), -levels, levels) * (mx / levels)
            return y.reshape(orig)
        attn_keys = ('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight')
        ffn_keys = ('gate_proj.weight','up_proj.weight','down_proj.weight')
        if a.attn_bits or a.ffn_bits:
            ab = a.attn_bits or a.wbits
            fb = a.ffn_bits or a.wbits
            for n in list(w):
                if any(k in n for k in attn_keys) and ab:
                    w[n] = qmat(w[n], ab, a.group, a.asym)
                elif any(k in n for k in ffn_keys) and fb:
                    w[n] = qmat(w[n], fb, a.group, a.asym)
            print(f'mixed weights attn=q{ab} ffn=q{fb} group={a.group} asym={a.asym}')
        else:
            if a.scope == 'attn': keys = attn_keys
            elif a.scope == 'ffn': keys = ffn_keys
            else: keys = attn_keys + ffn_keys
            names = [n for n in w if any(k in n for k in keys)]
            for n in names:
                w[n] = qmat(w[n], a.wbits, a.group, a.asym)
            if a.quant_embed:
                w['model.embed_tokens.weight'] = qmat(w['model.embed_tokens.weight'], a.wbits, a.group, a.asym)
            print(f'quantized {len(names)} matrices to q{a.wbits}, group={a.group}, asym={a.asym}, embed={a.quant_embed}')
    tok = Tokenizer.from_file(a.tokenizer)
    text = ('<|im_start|>system\n' + a.system + '<|im_end|>\n'
            '<|im_start|>user\n' + a.user + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    ids = tok.encode(text, add_special_tokens=False).ids
    print('prompt tokens', len(ids), ids)
    print('prompt decoded:', repr(tok.decode(ids, skip_special_tokens=False)))

    dim = cfg['hidden_size']; layers = cfg['num_hidden_layers']
    heads = cfg['num_attention_heads']; kvh = cfg['num_key_value_heads']
    hd = cfg['head_dim']; hidden = cfg['intermediate_size']; vocab = cfg['vocab_size']
    eps = cfg.get('rms_norm_eps', 1e-6); theta = cfg.get('rope_theta', 1e6)
    emb = w['model.embed_tokens.weight']
    generated = []
    all_logits = []
    pos = 0
    # KV caches: [layers, seq, kvh, hd]
    kcache = np.zeros((layers, len(ids) + a.gen + 8, kvh, hd), np.float32)
    vcache = np.zeros_like(kcache)

    def forward(tokid, pos):
        x = emb[tokid].copy()
        for l in range(layers):
            p = f'model.layers.{l}.'
            xn = rms(x, w[p + 'input_layernorm.weight'], eps)
            q = (xn @ w[p + 'self_attn.q_proj.weight'].T).reshape(heads, hd)
            k = (xn @ w[p + 'self_attn.k_proj.weight'].T).reshape(kvh, hd)
            v = (xn @ w[p + 'self_attn.v_proj.weight'].T).reshape(kvh, hd)
            if a.variant in ('qk','after','noweight','qonly','konly'):
                qw = np.ones(hd, np.float32) if a.variant == 'noweight' else w[p + 'self_attn.q_norm.weight']
                kw = np.ones(hd, np.float32) if a.variant == 'noweight' else w[p + 'self_attn.k_norm.weight']
                if a.variant in ('qk','after','noweight','qonly'): q = rms(q, qw, eps)
                if a.variant in ('qk','after','noweight','konly'): k = rms(k, kw, eps)
            if a.variant != 'no-rope':
                q = rope(q, pos, theta, hd)
                k = rope(k, pos, theta, hd)
            if a.variant == 'after':
                q = rms(q, w[p + 'self_attn.q_norm.weight'], eps)
                k = rms(k, w[p + 'self_attn.k_norm.weight'], eps)
            kcache[l, pos] = k; vcache[l, pos] = v
            K = kcache[l, :pos+1].repeat(heads // kvh, axis=1)
            V = vcache[l, :pos+1].repeat(heads // kvh, axis=1)
            scores = np.einsum('hd,thd->ht', q, K) / math.sqrt(hd)
            scores = scores - scores.max(axis=1, keepdims=True)
            prob = np.exp(scores); prob /= prob.sum(axis=1, keepdims=True)
            att = np.einsum('ht,thd->hd', prob, V).reshape(dim)
            x = x + (att @ w[p + 'self_attn.o_proj.weight'].T)
            xn = rms(x, w[p + 'post_attention_layernorm.weight'], eps)
            gate = xn @ w[p + 'mlp.gate_proj.weight'].T
            up = xn @ w[p + 'mlp.up_proj.weight'].T
            h = (gate / (1.0 + np.exp(-gate))) * up
            x = x + (h @ w[p + 'mlp.down_proj.weight'].T)
        x = rms(x, w['model.norm.weight'], eps)
        return x @ emb.T

    logits = None
    for i, t in enumerate(ids):
        logits = forward(t, i)
    pos = len(ids)
    while len(generated) < a.gen:
        nxt = int(np.argmax(logits))
        generated.append(nxt); all_logits.append(logits.astype(np.float32))
        print(f'{len(generated):02d} token={nxt} {tok.decode([nxt], skip_special_tokens=False)!r}')
        if not a.ignore_eos and nxt in (1, 2):
            break
        if len(generated) >= a.gen:
            break
        logits = forward(nxt, pos)
        pos += 1

    np.savez_compressed(a.out_prefix + '.npz', prompt=np.array(ids, np.int32),
                        generated=np.array(generated, np.int32),
                        logits=np.stack(all_logits))
    open(a.out_prefix + '.txt', 'w').write(' '.join(map(str, generated)) + '\n')
    print('generated', len(generated), generated)
    print('text:', repr(tok.decode(generated, skip_special_tokens=False)))

if __name__ == '__main__':
    main()
