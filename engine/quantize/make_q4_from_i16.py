#!/usr/bin/env python3
"""Convert an I16M1/I16M2 checkpoint directly to Q4M1.

Q4M1 wants a full-precision source; I16 is the allowed integer reference here.
Do not route this through q8: q8 -> q4 is a second quantization and degrades.

Usage:
  python3 make_q4_from_i16.py models/llama2c/minimind2-small.i16 \
      models/llama2c/minimind2-small.q4
"""
import struct
import sys

import numpy as np

from make_q4 import q4mat, q8mat

MAGIC_Q4 = 0x51344D31  # Q4M1
MAGIC_I16 = 0x4931364D  # I16M1
MAGIC_I16_V2 = 0x4931364E  # I16N


def main(src, dst):
    f = open(src, 'rb')
    magic, dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack('<I7i', f.read(32))
    if magic not in (MAGIC_I16, MAGIC_I16_V2):
        raise SystemExit(f'not an I16 checkpoint: {src}')
    if magic == MAGIC_I16_V2:
        theta, half, split, qk = struct.unpack('<fiii', f.read(16))
    else:
        theta, half, split = struct.unpack('<fii', f.read(12))
        qk = 0
    head_dim = dim // heads

    emb16 = np.frombuffer(f.read(vocab * dim * 2), dtype='<i2').reshape(vocab, dim)
    emb_scale = np.frombuffer(f.read(vocab * 4), dtype=np.float32)
    rms_att = np.frombuffer(f.read(layers * dim * 4), np.float32)
    rms_ffn = np.frombuffer(f.read(layers * dim * 4), np.float32)
    rms_final = np.frombuffer(f.read(dim * 4), np.float32)
    q_norm = k_norm = None
    if qk:
        q_norm = np.frombuffer(f.read(layers * head_dim * 4), np.float32)
        k_norm = np.frombuffer(f.read(layers * head_dim * 4), np.float32)

    def read_mat():
        rows, cols = struct.unpack('<ii', f.read(8))
        scale = struct.unpack('<f', f.read(4))[0]
        q = np.frombuffer(f.read(rows * cols * 2), dtype='<i2').reshape(rows, cols)
        return q.astype(np.float32) * scale

    with open(dst, 'wb') as o:
        o.write(struct.pack('<I7ifiii', MAGIC_Q4, dim, hidden, layers, heads, kv_heads,
                            vocab, seq, theta, half, split, qk))
        emb_q8 = np.empty((vocab, dim), np.int8)
        emb_scales = np.empty(vocab, np.float32)
        for i in range(vocab):
            q, s = q8mat(emb16[i].astype(np.float32) * emb_scale[i])
            emb_q8[i] = q
            emb_scales[i] = s
        o.write(emb_q8.tobytes())
        o.write(emb_scales.tobytes())
        o.write(rms_att.tobytes())
        o.write(rms_ffn.tobytes())
        o.write(rms_final.tobytes())
        if qk:
            o.write(q_norm.tobytes())
            o.write(k_norm.tobytes())

        def write_q4(w):
            packed, s16, m16, param_scale, groups = q4mat(w)
            o.write(struct.pack('<iiif', w.shape[0], w.shape[1], groups, param_scale))
            o.write(packed.astype(np.uint8).tobytes())
            o.write(s16.tobytes())
            o.write(m16.tobytes())

        def write_q8(w):
            q, s = q8mat(w)
            o.write(struct.pack('<iif', w.shape[0], w.shape[1], s))
            o.write(q.astype(np.int8).tobytes())

        for _ in range(layers):
            write_q4(read_mat())  # qkv
            write_q4(read_mat())  # wo
            if split:
                write_q4(read_mat())  # w1
                write_q4(read_mat())  # w3
            else:
                write_q4(read_mat())  # w13
            write_q4(read_mat())  # w2
        for _ in range((vocab + 1535) // 1536):
            write_q8(read_mat())  # logits tiles

    print(dst, 'dim', dim, 'hidden', hidden, 'layers', layers, 'vocab', vocab,
          'seq', seq, 'split', split, 'qk_norm', qk)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
