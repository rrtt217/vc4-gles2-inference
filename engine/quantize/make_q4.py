#!/usr/bin/env python3
"""Create a Q4M1 checkpoint from MiniMind safetensors.

q4 block32 asymmetric (min/scale) for all linear matrices; q8 embeddings and
q8 logits tiles.  Quantization is done from the original F16 weights, not from
an already-quantized q8/i16 file.
"""
import argparse, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_i16 import load_minimind

MAGIC_Q4 = 0x51344D31  # Q4M1


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


def write_q4_mat(f, w):
    packed, s16, m16, ps, groups = q4mat(w)
    rows, cols = w.shape
    f.write(struct.pack('<iiif', rows, cols, groups, ps))
    f.write(packed.astype(np.uint8).tobytes())
    f.write(s16.tobytes())
    f.write(m16.tobytes())


def write_q8_mat(f, w):
    q, s = q8mat(np.asarray(w, dtype=np.float32))
    f.write(struct.pack('<iif', q.shape[0], q.shape[1], s))
    f.write(q.astype(np.int8).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='directory with config.json/model.safetensors')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq', type=int, default=512)
    a = ap.parse_args()
    m = load_minimind(a.input, a.seq)
    dim, hidden, layers, heads, kv, vocab, seq = (
        m['dim'], m['hidden'], m['layers'], m['heads'], m['kv'], m['vocab'], m['seq'])
    split = 1 if 2 * hidden > 2048 else 0
    qk = int(m.get('qk_norm', 0))
    with open(a.output, 'wb') as f:
        f.write(struct.pack('<I7ifiii', MAGIC_Q4, dim, hidden, layers, heads, kv, vocab, seq,
                            m['theta'], m['half'], split, qk))
        er = np.empty((vocab, dim), np.int8)
        es = np.empty(vocab, np.float32)
        for i in range(vocab):
            q, s = q8mat(m['emb'][i])
            er[i] = q
            es[i] = s
        f.write(er.tobytes())
        f.write(es.tobytes())
        f.write(np.asarray(m['rms_att'], np.float32).tobytes())
        f.write(np.asarray(m['rms_ffn'], np.float32).tobytes())
        f.write(np.asarray(m['rms_final'], np.float32).tobytes())
        if qk:
            f.write(np.asarray(m['q_norm'], np.float32).tobytes())
            f.write(np.asarray(m['k_norm'], np.float32).tobytes())
        for l in range(layers):
            write_q4_mat(f, np.concatenate([m['wq'][l], m['wk'][l], m['wv'][l]], axis=0))
            write_q4_mat(f, m['wo'][l])
            if split:
                write_q4_mat(f, m['w1'][l])
                write_q4_mat(f, m['w3'][l])
            else:
                write_q4_mat(f, np.concatenate([m['w1'][l], m['w3'][l]], axis=0))
            write_q4_mat(f, m['w2'][l])
        for st in range(0, vocab, 1536):
            write_q8_mat(f, m['wcls'][st:st + 1536])
    print(a.output, 'dim', dim, 'hidden', hidden, 'layers', layers,
          'vocab', vocab, 'seq', seq, 'split', split, 'qk_norm', qk)


if __name__ == '__main__':
    main()
