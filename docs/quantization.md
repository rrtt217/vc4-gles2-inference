# Quantization

This document covers the q4/q6 experiments, the offline Q4M1 path, and the
KV packed variants. The model format itself is in
[model-formats.md](model-formats.md).

## Q8 conversion

q8 is produced by `mmcf_pack.py --quant q8`, and it is a terminal precision:
never a source for q4.  Two things about the arithmetic are worth knowing
because they are visible in the output:

- the packer quantizes **i16 first, then q8** from the dequantised values, which
  is what the old two-step converter did.  Quantizing the float weights directly
  would differ by a few LSB and would not reproduce the shipped containers.
- it always writes `split=1` (separate W1/W3), even from a fused source, so each
  half keeps its own q8 scale and takes the paired q16 FFN path.  The runtime
  still handles fused q8 W13 through `matvec_q8_pair_fused`.

### When one scale per matrix is not enough

`ws` is one number per matrix, so the whole matrix shares the range of its
largest weight.  A matrix whose rows differ by more than ~100:1 then loses most
of its small weights: void.1's layer-0 `down_proj` has max/median 521, and 66%
of its entries round to zero (58% for `gate_proj`).  That model answers with
misspelled bytes at q8 and is fine at q4 or i16, and a p99.9-clipped scale makes
it *worse* — the outliers are signal, not noise.

`--q8-row-scale` writes one scale per output row instead (a `f32` plane per
matrix, tensor id part 3; see [checkpoint-container.md](checkpoint-container.md)):

| void.1, 640/1440/24 layers | q8 (one scale) | q8 per row | i16 |
|---|---:|---:|---:|
| median relative weight error | 2.36% | 0.79% | 0.001% |
| worst matrix | 16.56% | 1.91% | — |
| logits vs f32, the diverging prompt | 14.44 (wrong top-1) | **0.67 (right)** | 0.93 |
| English PPL, seq 1024 | 3.9017 | **3.7979** | 3.7824 |
| greedy 400 B | `everythinteresting` | clean | clean |

Use it when a q8 container is coherent but misspells, and check the packer's
per-matrix error first: `mmcf_pack.py` prints the residual, not the weight
error, so the number to look at is `max|w| / median|w|` per matrix.
`mmcf_requant_q8.py --row-scale` rebuilds the same planes from an i16
container; on void.1 it reproduces the packer's greedy text.

The flag is a flag again: the helper that computes the per-row scales was also
named `q8_row`, which shadowed the parameter, so for one commit window every q8
pack wrote a plane per matrix whether asked or not.  Renamed
(`q8_per_row`); a container's tensor count says which one it is — one tensor per
matrix, or two.

The cost is one texel fetch per fragment in the q8 kernels and a plane of
`rows` 16-bit values (486 KB for the whole of void.1, 0.5%).  The texture holds
**one row pair per texel** (`.rg` = row 2t, `.ba` = row 2t+1) and is 4 texels
wide.

Both of those are VC4 requirements, measured with `engine/probes/wrow_probe`:
one fragment asking the same sampler for two different texels returns undefined
data once the matrix is tall (88 of 896 rows wrong on the qkv shape, 40 of 259
on the logits tile), and a 1-texel-wide texture is a shape that board had never
been asked for.  The probe drives the engine's own `matvec_q8_add`,
`matvec_q8_pair` and `matvec_q8_pair_logits` against a CPU reference, so it is
the check to run before adding any sampler to a shader.  16 bits also keep the
decoded value under 2^16, so the decode stays exact even if a backend demotes
the fragment stage to mediump.  The two-position
prefill kernels have no row-scale variant, so such a model prefills on the
single-position path (`Session::pair_ok()` returns false).

## Weight quantization experiments

The NumPy reference (`minimind3/ref_numpy.py`) supports symmetric and
asymmetric block scales, plus mixed attention/FFN bits:

```bash
python3 ref_numpy.py --wbits 6 --group 32 --asym --scope all --gen 64 ...
python3 ref_numpy.py --attn-bits 8 --ffn-bits 4 --group 32 --asym ...
```

Local results (spring / self-introduction, natural EOS):

| weights | memory | spring | self-intro |
|---|---:|---|---|
| q8 | 66 MB | baseline | baseline |
| q6, group 128 symmetric | ~50 MB | exact (64) | slight drift |
| q6, group 64 asymmetric | ~50 MB | coherent | coherent |
| **q4, group 32 asymmetric** | **~33 MB** | **coherent** | **coherent, correct creator** |
| q4, group 64 asymmetric | ~33 MB | coherent | repetition |
| q4, group 32 symmetric | ~33 MB | repetition | poor |
| q4, group 128 symmetric | ~33 MB | coherent | "由我自己编写" |

The asymmetric q4 block-32 result is the first q4 variant that stays usable on
all four test prompts (spring, self-intro, ML explanation, 1+1).  It is not
bit-exact to q8; it is a plausible Q4_K-style scalar block quant, not IQ vector
quantization.  The engine keeps embeddings and logits q8.

CPU weight copies are freed after texture upload by default; q8 MiniMind3 RSS
drops from ~119 MB to ~56 MB.  This is system RAM, not CMA.

### Offline Q4M1

q4 is quantized directly from the float safetensors, not from q8:

```bash
python3 engine/tools/mmcf_pack.py --in models/minimind3 --quant q4 --seq 512 \
  --tok models/minimind3/minimind3.tok --out models/minimind3/minimind3.q4.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja --verify
```

Embeddings and logits stay **q8** in a q4 container, which is why a q4 file is
not simply "everything at 4 bits".

The Q4M1 file stores q4 block32 asymmetric linear weights (qkv/wo/W1/W3/W2),
q8 embeddings and q8 logits tiles.  It is a *packer input*, so wrap it in a
container first, then load that; do **not** set `Q4_WEIGHTS`:

```bash
python3 engine/tools/mmcf_pack.py --in models/minimind3/minimind3.q4 \
  --tok models/minimind3/minimind3.tok \
  --out models/minimind3/minimind3.q4.mmcf \
  --name minimind3 --template minimind3 --chat --verify

MODEL=../models/minimind3/minimind3.q4.mmcf \
  KV_PACK=16 GPU_LAYER=1 ./engine_mm3 ...
```

Local size/speed:

| model | size | decode |
|---|---:|---:|
| q8 | 66 MB | ~17 ms/token |
| **q4 offline** | **45 MB** | **~16–23 ms/token** |

Offline q4 output is coherent on spring/self-intro/ML; 1+1 remains
prompt-sensitive.  This is the recommended q4 path.  Runtime q4-from-I16
(`Q4_WEIGHTS=1`) is retained for experiments; runtime q4-from-q8 is a second
quantization and is not recommended.

Local q4-from-I16 examples:

```text
我是MiniMind，一个由jingyaogong创造的AI助手。
机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测或决策。
1+1=2
因此，1+1=2。
```

After the 8-nibble shader unroll, q4 matvec is close to q8 on the local Xe
(~16–23 ms/token vs ~14–20 ms/token at 128 generated tokens); the win is
memory, not raw speed. The Pi numbers below were measured before that shader
optimization.

### Pi Zero validation (2026-10-31)

The Pi was tested with the q4-capable engine and `minimind3.q4`:

- packed q16/q8 token sequences are **exact** to their 24-bit
  fixed-point references on the Pi;
- q4 q16 and q4 q8 are exact to their unpacked counterparts;
- q4 file: 45 MB; q8: 66 MB;
- q4 decode: ~0.77 s/token vs ~0.50 s/token for q8 (optimized q4 shader);
- q8 KV output drifts from q16 at longer generations.

Full numbers and caveats: [performance.md](performance.md#pi-zero-minimind3-kv-and-q4-validation).


