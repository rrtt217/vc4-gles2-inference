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

`make_q8_from_i16.py` writes `split=1` by default and splits a fused W13
into W1/W3.  The split layout keeps separate q8 scales and takes the paired
q16 W1/W3 path.  The runtime also supports fused q8 W13 through
`matvec_q8_pair_fused`, so old fused files remain loadable.

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


## The container is the only format (2026-11-05)

`engine/src/model.cpp` no longer contains `load_q8` / `load_q4` / `load_i16` /
`load_legacy`; `Model::load()` accepts `MMCF` and refuses anything else with a
message that names the packer.  All 15 checkpoints in `models/` were converted
and verified against their source (same greedy text, same PPL to 4 decimals on
Xe and on the Pi, where the numbers are the pinned 271.2855 / 325.3751).

Converting a new model is one command:

```bash
engine/tools/mmcf_pack.py --in model.q4 --tok tokenizer.tok \
  --out model.q4.mmcf --name model --template chatml --chat \
  --theta 1000000 [--rope-half] [--verify]
```

`--template` no longer exists as an engine flag: the kind travels in the file
(0 plain, 1 chatml/qwen, 2 minimind3, 3 tinybrainbot), so the same checkpoint
cannot be run with the wrong one.  `TOKENIZER=`, `ROPE_THETA=` and `ROPE_HALF=`
are likewise historical; the container states all three.

## MMCF containers and the TinyBrainBot family (2026-11-05)

`engine/tools/mmcf_pack.py` converts every legacy checkpoint in `models/`
(Q8M1/Q8M2, Q4M1, I16M1/I16N) into the self-describing container described in
[`checkpoint-container.md`](checkpoint-container.md).  Two implicit conventions
that used to live in shell history are now in the file:

| convention | where it lives now |
|---|---|
| `rope_theta` / rotate-half | metadata key 2; a `Q8M1`/`I16M1` source has no rope at all, so the packer **refuses to guess** and wants `--theta`/`--rope-half`, `--rope-plain` (llama2) or `--assume-llama2-rope` |
| BOS / EOS / stop ids | metadata key 8, derived from the embedded vocabulary; llama2.c stores `"<s>"` as `"\n<s>\n"`, so matching is on the trimmed token |
| chat template | metadata key 4 (kind + default system + name); `--template` still overrides |
| tokenizer | the `.tok` blob is inside the container, so `TOKENIZER=` is optional |

`engine/quantize/make_i16.py` also stopped hardcoding the rope: `rope_of(cfg)`
reads `rope_theta` (and `rope_scaling.rope_theta`) and treats the layout as
rotate-half unless the config says otherwise, and the safetensors loader picks
F16/F32 from the tensor header instead of assuming F16.

### TinyBrainBot-100M-v5.1-Instruct

A HF Llama-shaped model with `rope_theta = 10000`, rotate-half, no qk-norm,
tied embeddings and a SentencePiece BPE tokenizer (32000 entries, `▁` for
space, byte fallback).  Its special tokens are `<s>`=0, `</s>`=1, `<unk>`=3,
`<|user|>`=4, `<|assistant|>`=5, `<|system|>`=6, `<|end|>`=7.  The prompt has
no system turn and no BOS:

```
<|user|> {question}<|assistant|> {answer}<|end|>
```

`--template tinybrainbot` (or template kind 3 in a container) emits
`"<|user|> " + q + "<|assistant|> "` and disables the tokenizer's dummy prefix;
the stop set is `{7}` (EOS 7), not `</s>`.

**Precision matters more here than on MiniMind.**  `Q8M1` stores one scale per
matrix (`make_q8_from_i16.py: q8mat` takes `max(|w|)` over the whole matrix), and
for this checkpoint that scale is too coarse: the q8 build answers with a bare
`<|end|>` while the q4 build (block-32 scale *and* min) answers
`Hello! What can I do for you`.  Use **q4 or i16** for this model; the q8 file is
kept only for the size comparison.
