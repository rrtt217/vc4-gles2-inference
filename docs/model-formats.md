# Model formats

The engine reads four checkpoint families. All are little-endian and use
the same tensor order; only the header and weight encoding differ.

## Supported checkpoints

1. **legacy llama2.c fp32** — 28-byte header + fp32 tensors.  The loader
   quantizes it to I16 by default; `LEGACY_Q8=1` selects the old q8 path for
   memory-constrained use.
2. **Q8M1** — int8 weights, int8 embedding
3. **I16M1** — int16 weights, int16 embedding, rope metadata
4. **Q8M2 / I16M2** — same as Q8M1/I16M1, plus stored theta/half and
   per-head Q/K RMSNorm weights (MiniMind3)

### I16M1

```
u32 magic = 0x4931364D ("I16M")
i32 dim, hidden, layers, heads, kv_heads, vocab, seq
f32 rope_theta
i32 rope_half     // 0 = adjacent pairs, 1 = half-split
i32 split         // 1 = separate W1/W3, 0 = fused W13

i16 emb_q[vocab*dim]
f32 emb_scale[vocab]
f32 rms_att[layers*dim]
f32 rms_ffn[layers*dim]
f32 rms_final[dim]

per layer:
  mat qkv [dim+2*kv_dim, dim]
  mat wo  [dim, dim]
  if split: mat w1 [hidden, dim], mat w3 [hidden, dim]
  else:     mat w13 [2*hidden, dim]
  mat w2 [dim, hidden]

logits tiles: mat [<=1536, dim]

mat:
  i32 rows, i32 cols, f32 scale, i16 q[rows*cols]
```

Q8M1 uses the same layout with int8 data and no rope metadata.  MiniMind q8
needs `ROPE_THETA=1000000 ROPE_HALF=1`; I16M1 stores these fields.

### I16M2 / Q8M2

MiniMind3 uses a Qwen3-style block: per-head `q_norm` and `k_norm` RMSNorm
before RoPE.  These formats keep the I16M1/Q8M1 layout but add:

```
u32 magic (I16N=0x4931364E, Q8M2=0x51384D32)
... same 7 ints, theta, half, split ...
i32 qk_norm          // 1 = q_norm/k_norm present
... embeddings, scales, rms_att, rms_ffn, rms_final ...
f32 q_norm[layers*head_dim]
f32 k_norm[layers*head_dim]
... same per-layer mats ...
```

Q8M2 stores theta/half itself; no `ROPE_THETA`/`ROPE_HALF` environment is
needed for MiniMind3.

## Q4M1 (offline q4)

Q4M1 is generated from F16 safetensors by `make_q4.py`, or from an I16
checkpoint by `make_q4_from_i16.py` (I16 is an allowed source; q8 is not):

```text
u32 magic = 0x51344D31 ("Q4M1")
i32 dim, hidden, layers, heads, kv_heads, vocab, seq
f32 rope_theta
i32 rope_half, split, qk_norm

q8 embedding + f32 row scales
f32 rms_att, rms_ffn, rms_final
f32 q_norm/k_norm (optional)

per layer:
  q4 matrix qkv, wo, W1/W3 or W13, W2
  q8 logits tiles

q4 matrix:
  i32 rows, cols, groups, f32 param_scale
  u8 packed nibbles [rows * cols/2]
  i16 block_scale[rows * groups]
  i16 block_min[rows * groups]
```

groups = ceil(cols/32); scale/min are q16 values multiplied by
`param_scale`. Q4 matrices require `cols % 32 == 0`. Embeddings and logits
stay q8.

