# Quantization

This document covers the q4/q6 experiments, the offline Q4M1 path, and the
KV packed variants. The model format itself is in
[model-formats.md](model-formats.md).

## Q8 conversion

Convert an I16 checkpoint with `make_q8_from_i16.py`; q8 is a terminal format,
not a source for q4.  The converter always writes `split=1` (separate W1/W3),
even when the I16 source has a fused W13: the fused q8 GPU FFN path is wrong,
while split W1/W3 uses the paired q16 path.  Reconvert old fused q8 files.

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

`make_q4.py` quantizes directly from `model.safetensors` (F16), not from q8:

```bash
python3 engine/quantize/make_q4.py --input models/minimind3 --output models/minimind3/minimind3.q4 --seq 512
```

For an I16 checkpoint, convert directly without going through q8:

```bash
python3 engine/quantize/make_q4_from_i16.py \
  models/llama2c/minimind2-small.i16 models/llama2c/minimind2-small.q4
```

The Q4M1 file stores q4 block32 asymmetric linear weights (qkv/wo/W1/W3/W2),
q8 embeddings and q8 logits tiles.  Load it directly; do **not** set
`Q4_WEIGHTS`:

```bash
MODEL=../models/minimind3/minimind3.q4 \
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


