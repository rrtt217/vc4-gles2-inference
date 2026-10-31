# Performance

Measurements are from the Pi Zero and the local Xe/GLES2 machine. Treat them
as regression baselines, not portable benchmarks.

## Measured performance

Pi Zero, q8, after the draw fix:

- 15M: ~137–151 ms/token; first process after a build is slower (cold shader
  compilation), later runs ~1.1 s prefill.
- 42M: ~346–355 ms/token; prefill ~2.7 s warm, up to ~4–21 s cold.
- MiniMind q8: coherent Chinese chat; CPU FFN and GPU FFN match.
- MiniMind I16: coherent, ~0.3 tok/s.

Iris Xe:

- 15M I16: ~16–24 ms/token (system-load dependent).
- 42M I16 GPU FFN: ~31.5 ms/token in the earlier clean A/B.
- 42M I16 CPU FFN: ~78 ms/token.

## Iris Xe: q4 matvec optimization

MiniMind3 64M, `SEQ=512`, greedy, 128 generated tokens, warm process:

| path | prefill | decode |
|---|---:|---:|
| q8 + q16 KV | 1.0–1.5 s | 14–20 ms/token |
| q4 before unroll | 2.77 s | 36.6 ms/token |
| **q4 after 8-nibble unroll** | **1.2–1.6 s** | **16–23 ms/token** |

The q4 shader now loads one texel (8 nibbles) per iteration instead of
sampling the weight texture once per weight. That cuts q4 decode by roughly
1.6–2.2x and brings it close to q8.

Two variants were tried and rejected:

- factoring each block into `scale*sum(q*x) + min*sum(x)` was slower
  (26.6 ms/token) due to extra registers;
- a 16-weight/2-texel unroll measured ~17–20 ms/token and ~1.2 s prefill, but
  the spring prompt degraded into "大草 / 草茂" repetition. The 8-weight
  version was kept for quality.



## Pi Zero: MiniMind3 KV and Q4 validation

Measured 2026-10-31 on the Pi Zero with the optimized q4 engine
(`09889c54...`) and the offline Q4M1 file.

Packed-vs-unpacked token equality, greedy, `SEQ=512`:

| model | packed | unpacked reference | result |
|---|---|---|---|
| q8 | `KV_PACK=16`, scale 2048 | 24-bit, scale 2048 | exact token sequence |
| q8 | `KV_PACK=8`, scale 16 | 24-bit, scale 16 | exact token sequence |
| q4 | `KV_PACK=16`, scale 2048 | 24-bit, scale 2048 | exact token sequence |
| q4 | `KV_PACK=8`, scale 16 | 24-bit, scale 16 | exact token sequence |

So the packed encoders match their unpacked fixed-point references on VC4.
Q8 **packing** is exact to q8; q8 itself is still a quality tradeoff.

Longer greedy output (32 tokens, `SEQ=768`):

```text
q8 + q16 KV:
MiniMind是一个高效的小参数AI模型，专注于提供准确、简洁且友好的回答。

q8 + q8 KV:
我是一个由Jingyaogong创建的AI助手，我的名字叫MiniMind。我能够回答各种问题，从简单的事

q4 model + q16 KV:
我是MiniMind，一个由jingyaogong创造的AI助手。我能够回答各种问题，提供信息，协助

q4 model + q8 KV:
我是一个由jingyaogong创建的AI助手，专注于提供高效、准确的信息支持与帮助。
```

Packed q8 matches unpacked q8 exactly, but the q8 KV **mode** changes wording
and can change identity; q16 KV is the safe default.

Worker-load `CmaFree` (one sample per mode, `SEQ=768`; idle `CmaFree`
varies as CMA is released asynchronously, so treat as directional):

| modes | idle before | while worker ready |
|---|---:|---:|
| q8 + q16 KV | 74.3 MB | 50.6 MB |
| q8 + q8 KV | 100.4 MB | 28.1 MB |
| q4 + q16 KV | 70.1 MB | 26.6 MB |
| q4 + q8 KV | 106.8 MB | 27.5 MB |

The q4 weights (45 MB file vs 66 MB q8) free roughly 20–25 MB versus q8 in
the q4+q16 vs q8+q16 comparison. The q8+q8 vs q8+q16 delta is larger than the
~4.7 MB KV difference, so do not use this table as a precise allocator model.

Speed on Pi, prompt `你好`, 1 token generated:

| model / KV | prefill | decode |
|---|---:|---:|
| q8 + q16 | 37.7 s | 503 ms/token |
| q4 + q16 | 56.4 s | 774 ms/token |

After the 8-nibble shader unroll, q4 on Pi is about 1.5x slower than q8
(774 vs 503 ms/token) and prefill is 56 vs 38 s. It is still a
memory-constrained mode, but no longer a 3x penalty.

## Corrected MiniMind Q8

The old MiniMind q8 file was converted with the wrong q/k rope permutation.
Convert from a good I16 checkpoint instead:

```bash
python3 engine/make_q8_from_i16.py models/llama2c/minimind2-small.i16 models/llama2c/minimind2-small.q8
```

The corrected file is 28 MB and needs `ROPE_THETA=1000000 ROPE_HALF=1`.

## Layer-by-layer q8/int16 test

Reference: F16 numpy logits for the same 30-token Qwen prompt.

| Variant | top-1 | cosine | MAE |
|---|---|---:|---:|
| I16 baseline | 5134 | 0.99682 | 0.256 |
| all q8 | 5134 | 0.99498 | 0.306 |
| q8 layer 0 | 5134 | 0.99540 | 0.329 |
| q8 layer 1 | 5134 | 0.99696 | 0.238 |
| q8 layer 2 | 5134 | 0.99665 | 0.249 |
| q8 layer 3 | 5134 | 0.99645 | 0.260 |
| q8 layer 4 | 5134 | 0.99702 | 0.238 |
| q8 layer 5 | 5134 | 0.99664 | 0.259 |
| q8 layer 6 | 5134 | 0.99702 | 0.238 |
| q8 layer 7 | 5134 | 0.99716 | 0.230 |
| q8 qkv | 5134 | 0.99476 | 0.318 |
| q8 wo | 5134 | 0.99649 | 0.275 |
| q8 ffn | 5134 | 0.99522 | 0.296 |
| q8 w2 | 5134 | 0.99651 | 0.263 |
| q8 logits | 5134 | 0.99679 | 0.257 |
| q8 emb | 5134 | 0.99724 | 0.228 |

All variants keep the same top-1 token; no single layer requires int16.
