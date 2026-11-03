# Performance

Measurements are from the Pi Zero and the local Xe/GLES2 machine. Treat them
as regression baselines, not portable benchmarks.

## Current model matrix (2026-11-02)

Pi Zero, `GPU_LAYER=1 KV_PACK=16`, greedy, 16 generated tokens.  The Pi also
runs a Minecraft server, so use the ranges rather than individual decimals:

| model | file | size | prefill | decode | status |
|---|---|---:|---:|---:|---|
| MiniMind3 q8 | `minimind3.q8` | 66 MB | 20.5 s | **238 ms/token** | default Chinese chat |
| MiniMind3 q4 | `minimind3.q4` | 45 MB | 33.0 s | **381 ms/token** | memory mode; slightly lower quality |
| MiniMind2 i16 | `minimind2-small.i16` | 58 MB | 9.2 s | 226 ms/token | reference; weak instruction following |
| MiniMind2 q8 | `minimind2-small.q8` | 29 MB | 8.4 s | 140–190 ms/token | fastest MiniMind2; needs `ROPE_THETA=1000000 ROPE_HALF=1` |
| MiniMind2 q4 | `minimind2-small.q4` | 21 MB | 9.7 s | 193–212 ms/token | memory mode; identity answers unreliable |
| TinyChat15M q8 | `TinyChat15M.q8` | 24 MB | 4.8 s | 139–171 ms/token | loads and runs, but output is not coherent with the repo tokenizer |
| stories15M i16 | `stories15M.i16` | 49 MB | 3.1 s | 167 ms/token | TinyStories completion; coherent |
| stories15M legacy | `stories15M.bin` | 60 MB | 3.3 s | 161 ms/token | coherent after legacy-I16 default |
| stories42M q8 | `stories42M.q8` | 58 MB | 3.8 s | 190 ms/token | TinyStories completion; coherent |
| stories15M q8 | `stories15M.q8` | 24 MB | 3.1 s | 148 ms/token | **gibberish**; use the `.bin`/i16 path instead |
| stories260K legacy | `stories260K.bin` + `tok512.bin` | 1 MB | 5.4 s | 85 ms/token | coherent after legacy-I16 default; 5-layer floor probe |

The Pi store now has TinyStories 15M/42M/260K checkpoints.  They are base
completion models, not chat models, so test them with a plain prompt such as
`Once upon a time` and no `--chat`.  `stories42M.i16` is too large for the
current CMA budget; q8 is the practical 42M path.  TinyChat15M should be
treated as a format/probe model, not as a usable chat model until its
tokenizer/prompt pairing is fixed.

The 260K legacy checkpoint is a useful **floor probe**, not a product model:
its 5-layer/64-dim body reaches 85 ms/token after the legacy-I16 fix, but
prefill is still 5.4 s because the attention passes walk the full 512-position
grid and the shader setup is fixed.  This is below the ~150 ms floor of the
coherent 15M/42M models, but it is not a quality replacement for them.
Legacy fp32 files now default to I16; `LEGACY_Q8=1` keeps the old low-memory
q8 path for large checkpoints, at a quality cost on small TinyStories models.

### Default system prompt

The CLI and server use one neutral English prompt for every chat template:
`You are a helpful assistant.`  It replaces the old model-specific Chinese
MiniMind3 prompt and the server's Chinese default.  Tested behavior:

| model | identity response | note |
|---|---|---|
| MiniMind3 q8 | “我是一个AI助手……” | coherent; does not claim a name |
| MiniMind3 q4 | “我是MiniMind……” | retains the model identity |
| MiniMind2 i16/q8 | “我是一款智能语音助手……” | coherent |
| MiniMind2 q4 | “我是一台电脑/大型语言模型……” | still unreliable |

The spring-writing prompt is unchanged across the old and new prompts.  The
shorter header also removes model-specific identity wording from the default;
use `--system` when a deployment needs a specific persona.

The gates pin this prompt on the command line
(`--system "You are a helpful assistant."`): see the canonical smoke tests in
`AGENTS.md` and `engine/scripts/regression.sh`, which also assert that the
same command without `--system` prints identical text.  Pinning matters
because the strings moved when `c1bb4f6` replaced the old Chinese persona
prompt; the pre-`c1bb4f6` strings (`MiniMind是一个高效的小...` for q8,
`我是MiniMind，一个由jingyaogong...` for q4) still reproduce with that old text
passed to `--system`, but they are not the gate.

## How to read this document

This file is a chronological lab notebook, not a single clean benchmark: the
oldest entries are at the top, the newest at the bottom, and every section is
measured on the machine or driver it names.  Start with the current state, then
walk back through the entries that produced it.

Current state:

- [Decode loop campaign summary (2026-11-02 .. 03)](#decode-loop-campaign-summary-2026-11-02--03)
  — 237.3 -> 183.5 ms decode, the pass budget, and the env knob behind each
  landed change.  Dead ends live in
  [`future-optimization.md`](future-optimization.md#do-not-retry-measured-2026-11-03).
- [Pi Zero model matrix (2026-11-02)](#pi-zero-model-matrix-2026-11-02-vlab-cross-build)
- [Benchmarking caveat](#benchmarking-caveat-the-pi-also-runs-a-minecraft-server)

History:

- [Available-model baseline (2026-11-01)](#available-model-baseline-pi-zero-2026-11-01)
- [2026-11-01 optimization pass](#summary-of-the-2026-11-01-optimization-pass)
- [Attention and refactor work (2026-11-02)](#the-refactors-12-regression-helper-indirection-in-the-hot-matvec-loops)
- [q4 path (2026-11-02)](#q4-path-nibble-decode-without-mod--13)
- [Draw-path work: G1 and G2 (2026-11-02)](#draw-path-work-g1-gl-state-caching-and-g2-rmsnorm-reduce-fusion)
- [The q chain is four dependent passes (2026-11-03)](#2026-11-03-the-q-chain-is-four-dependent-passes---and-qk-norm-is-on)
- [qk-norm scans are fetch-count bound (2026-11-03)](#2026-11-03-the-qk-norm--k-norm-scan-is-fetch-count-bound-not-latency-bound)
- [Inline I16_2 decode in FFN gate and logits (2026-11-03)](#2026-11-03-inline-i16_2-decode-in-the-ffn-gate-and-logits-kernels--24-bit-exact)
- [Inline I16_2 decode in the FFN swiglu kernel (2026-11-03)](#2026-11-03---ffn-gate-inline-i16_2-decode-in-matvec_q8_pair_swiglu--10-ms-bit-exact)

## Summary of the 2026-11-01 optimization pass

MiniMind3-64M, q8, Pi Zero, same 73-token prompt, greedy:

| build | prefill | decode |
|---|---:|---:|
| before | 37.95 s | 507.5 ms/token |
| after the decoder work | 24.2 s | 314.7 ms/token |
| **after attention hoist** | **24.3 s** | **303.1 ms/token** |

**1.56x prefill, 1.67x decode** (2.0 -> 3.3 tokens/s), with the generated text
byte-identical to the pre-change engine for both q8 and q4.

Four changes, in the order they paid:

1. **Two int16 activations per texture texel** (+5.5% decode). Halves the
   activation fetches per MAC; the weights, which still stream every token, do
   not shrink, which is why the predicted 1.67x landed as 1.06x.
2. **Drop the redundant `floor(x*255+0.5)` from every decoder** (+~50%). For all
   256 byte values `float32(b/255)*255 == b`, so the rounding changed nothing —
   except in the q4 nibble path and the KV decoders, where a `floor` division or
   a sign threshold depends on exact integers. Those keep it.
3. **Parallel RMS sum of squares** (32 fragments + reduce, +3%). The serial
   768-iteration fragment was 3.5 ms per call; a draw+finish costs ~1 ms on this
   driver, which caps what pass-splitting can buy.
4. **Hoist the position-only part of attention** (-5% decode at 73 tokens,
   -12% at 267). q rotation, qk-norm and RoPE were recomputed for every
   (head, t); they now run once per token in two small passes.

Rejected after measurement: I16/mixed precision (the driver exposes no
half-float or float textures, and the INT16 checkpoint format also decodes), and
the same rounding trick in the attention decoders (+4% but corrupts output).

The remaining wall is attention: flat across everything else, it grows with
position and is the largest phase by ~250 tokens (15.9 ms/layer vs 14.7 for the
FFN). See "Long prompts did not get faster" at the end.


## Available-model baseline (Pi Zero, 2026-11-01)

> Historical snapshot at engine commit 0d06392.  Use the current matrix above
> for the 2026-11-02 binaries.

Engine commit: 0d06392 (attention hoist enabled by default).
All runs use KV_PACK=16, GPU_LAYER=1, greedy sampling, one warm process.

| model | file | seq | prompt / generation | prefill | decode |
|---|---|---:|---|---:|---:|
| MiniMind3 64M q8 | minimind3.q8 (66 MB) | 768 | short Chinese identity / 16 | 17.56 s | 295.3 ms/token |
| MiniMind3 64M q4 | minimind3.q4 (45 MB) | 768 | same / 16 | 25.99 s | 418.7 ms/token |
| MiniMind2 Small i16 | minimind2-small.i16 (58 MB) | 1024 | Chinese arithmetic / 6 | 8.23 s | 234.5 ms/token |
| TinyChat15M q8 | TinyChat15M.q8 (24 MB) | 512 | English chat / 15 | 4.78 s | 134.6 ms/token |

MiniMind3 q8 is the fastest usable Chinese chat model. q4 is a memory mode,
not a speed mode: it uses 21 MB less weight storage but is about 1.42x slower
per decoded token in this run.

MiniMind2 Small answers simple arithmetic but has weak identity/instruction
following.  TinyChat15M was historically treated as a fast English model, but
the current Pi tests produce incoherent text (see the current matrix above);
do not use it as a chat model without fixing its tokenizer/prompt pairing.
The Pi model store has no stories15M/42M or MiniMind3 i16.

These are single-run baselines, not statistical measurements. Pi timing varies
with thermal state, SD-card activity, and shader compilation; expect several
percent of run-to-run noise.

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

## Server-level wins (2026-11-01)

Measured on the same Pi Zero, `SEQ=768`, `KV_PACK=16`, q8 model, greedy.

### The same tool request, with and without a prefix cache

Request: `tools=[system_status]`, user message `现在几点了？` (299 prompt
tokens), `max_tokens=16`.

| path | wall | cached_tokens | prefill_tokens |
|---|---:|---:|---:|
| cold server, no cache | **184.9 s** | 0 | 299 |
| `--prefix-cache` with the system+tools header (277 tokens) | **25.0 s** | 277 | 22 |

7.4× on an identical request, because only the 22-token user turn still has to
be prefilled.  The header cache is reusable for every conversation that uses
the same system prompt and the same tool list.

### Deterministic tool routes

Same question, answered by the server instead of the model:

| path | wall |
|---|---:|
| `GET /status` | 0.013 s |
| `GET /time` | 0.013 s |
| `GET /math?expr=256*37` | 0.015 s |
| `POST /v1/chat/completions` with `tool_choice: calculate_math` | 0.02 s |
| the same question through the model | 25–185 s |

### Per-layer decode profile

`LAYER_PROFILE=1`, one decode step at pos≈79, milliseconds per layer:

| stage | ms | share |
|---|---:|---:|
| attention block (qkv + rope + KV write + scores + softmax + V) | 15.2 | 22% |
| FFN (w1/w3 + swiglu) | 24.6 | 35% |
| w2 | 18.2 | 26% |
| wo | 6.9 | 10% |
| rmsnorm ×2 | 5.6 | 8% |
| **per layer** | **~70** | |
| logits (once per token) | 33.4 | |

The FFN path (w1/w3 + w2) is 61% of a decode token: that is where any further
decode work has to go.

### Activation packing (two int16 per texel)

The matvec fragments used to fetch the activation texture once per weight
(`dx(k)` x4 per weight texel): 1.0 activation samples per MAC against 0.25
weight samples for q8.  Packing two int16 activations into one RGBA8 texel and
decoding both halves from a single fetch drops that to 0.5 + 0.25 = 0.75
samples per MAC, a 1.67x reduction in texture traffic with no change to the
arithmetic (same int16 values, same accumulation order).

The hypothesis came from the logits pass, which is a single 6400x768 matvec:
6.14M samples / 190 Msamples/s predicts 32 ms and the profile measured 33.4 ms,
so the whole pipeline is texture-sample bound rather than ALU bound.  Batch
prefill does not help for the same reason (4 positions in one draw re-read the
weights 4 times), which is what made packing the right lever instead.

Measured on the Pi Zero (q8, KV_PACK=16, greedy, same 73-token prompt):

Interleaved A/B, three rounds each, same prompt, greedy, `TIMING=1`:

| round | ref prefill | ref decode | new prefill | new decode |
|---|---:|---:|---:|---:|
| 1 | 37.52 s | 505.4 ms/token | 36.23 s | 479.6 ms/token |
| 2 | 37.45 s | 507.8 ms/token | 36.12 s | 481.9 ms/token |
| 3 | 37.47 s | 514.4 ms/token | 35.70 s | 479.9 ms/token |
| **median** | **37.47 s** | **507.8 ms/token** | **36.12 s** | **479.9 ms/token** |

**Decode 5.5% faster, prefill 3.6% faster** — not the 1.67x the sample count
predicted.  The activation vector is only ~3 KB, so its texels were already
resident in the texture cache and most of the removed fetches were latency-hidden.
The logits pass matched the model because it is a single 6400x768 matvec whose
weights stream 19.7 MB per call; the end-to-end gain is bounded by the weight
traffic that packing does not touch.

Correctness is checked by building the pre-packing sources into `/tmp/engine_base`
and diffing the generated text with the new binary:

```
q8  new: MiniMind是一个高效的小参数AI模型，专注于提供准确、
    ref: MiniMind是一个高效的小参数AI模型，专注于提供准确、   MATCH
q4  new: 我是MiniMind，一个由jingyaogong
    ref: 我是MiniMind，一个由jingyaogong                      MATCH
```

The values are identical because packing changes only where a value lives, not
what it decodes to.

End to end the server eval suite still passes 15/15 with the new engine
(`server/eval_baseline.json` is the new reference), and its slowest case drops
from 141.4 s to 134.5 s.

### The decode was the lever — and it is now 1.56x cheaper

Per weight the q8 shader spends ~13 scalar ops: 1 fetch, ~4 to decode the weight,
~8 to decode the activations, 1 MAC.  Two changes attacked that directly:
packing two int16 activations per texel (5.5%) and removing the provably
redundant `floor(x*255+0.5)` from every decoder (the rest, 1.6x on the matvec
alone).  Together prefill went 37.95 s -> 24.25 s and decode 507.5 -> 325.4
ms/token with byte-identical output.

**The mixed-precision idea is dead on this driver.**  `engine/texsize_probe.cpp`
shows Mesa exposes neither `OES_texture_half_float` nor `OES_texture_float`, so
weights can only be RGBA8 and every one of them must be decoded in the fragment
shader.  The INT16 checkpoint format does not help either: it stores two int16
per texel and decodes them (`int16_program`), which costs *more* ops per weight
than int8 — the "42M I16 31.5 ms/token vs q8 346 ms/token" figure quoted in older
revisions of this file does not reproduce on this hardware and should not be used
to justify a format switch.

## KV packing costs nothing in speed

Eight configurations, same prompt, 16 generated tokens:

| config | prefill | decode |
|---|---:|---:|
| q8, KV 24-bit | 37.8 s | 509.7 ms/token |
| q8, KV_PACK=8 | 37.6 s | 506.3 ms/token |
| q8, KV_PACK=16 | 37.5 s | 517.5 ms/token |
| q8, GPU_FFN=0 | 39.9 s | 508.4 ms/token |
| q8, GPU_FFN=1 | 37.3 s | 506.4 ms/token |
| q8, CPU_FFN=1 | 38.6 s | 579.2 ms/token |
| q4, KV_PACK=16 | 59.9 s | 768.4 ms/token |
| q4, KV_PACK=8 | — | — |

So `KV_PACK=16` is free (same speed, less CMA), and the q4 model is ~1.5×
slower per token than q8 on VC4 — q4 is a memory mode, not a speed mode.

### Context length costs, KV packing does not

Same prompt repeated to ~250 tokens (`SEQ=768`, q8, greedy, 8 generated):

| KV mode | prefill | decode |
|---|---:|---:|
| `KV_PACK=16` | 163.0 s | 617.5 ms/token |
| `KV_PACK=8` | 164.0 s | 616.4 ms/token |
| 24-bit | 163.9 s | 620.3 ms/token |

Within noise of each other, so packing is a **memory** knob only.

Comparing against the 79-token prompt (prefill 37.5 s = 474 ms/token, decode
517 ms/token), a 3.2× longer context costs ~38% more per prefill token and ~19%
more per decode token.  The attention block grows with position, but sublinearly
relative to the matvec work: at 250 tokens it is still far from dominating.

## Corrected MiniMind Q8

The old MiniMind q8 file was converted with the wrong q/k rope permutation.
Convert from a good I16 checkpoint instead:

```bash
python3 engine/quantize/make_q8_from_i16.py models/llama2c/minimind2-small.i16 models/llama2c/minimind2-small.q8
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

Measured on the Pi Zero with `LAYER_PROFILE=1` (q8, seq 768).  Profiling syncs
after every phase, so it inflates the total by ~20% (540 ms/token profiled vs
484 real); the shares are what matter:

| phase | ms per layer | x 8 layers | share of token |
|---|---:|---:|---:|
| w1+w3 + SwiGLU | 23.64 | 189.1 | 33% |
| w2 | 16.23 | 129.8 | 22% |
| attention (qkv, rope, scores, V) | 14.69 | 117.5 | 20% |
| wo | 6.14 | 49.1 | 8.5% |
| rms_att | 4.01 | 32.1 | 5.5% |
| rms_ffn | 2.49 | 19.9 | 3.5% |
| logits (once per token) | — | 33.3 | 6% |

Earlier numbers for the same phases (before activation packing) were 24.6 / 18.2
/ 15.2 / 6.9 / 5.6 ms — every matvec phase is a little cheaper now, which is the
5.5% decode gain seen in the A/B above.


## Dropping the redundant rounding in the decoders

Every decoder did `floor(x * 255.0 + 0.5)` to turn a normalised RGBA8 texel back
into its byte.  For all 256 byte values, `float32(b/255) * 255 == b` exactly, so
both the `+0.5` and the `floor` are dead weight — two vector ops per texel, i.e.
about two scalar ops per weight, on top of the ~13 the q8 matvec already spends.

The isolated probe (`engine/matvec_probe.cpp`) measures it on identical weights
and an identical activation layout:

| shape | q8, act 1/texel | q8, act 2/texel | q8, no rounding |
|---|---:|---:|---:|
| 1536x768 | 7.24 ms | 6.73 ms | **4.24 ms** |
| 2048x768 | 9.46 ms | 8.84 ms | **5.38 ms** |

That is 1.6x over the packed layout for the same math — larger than packing
itself, because rounding sat on the hottest straight-line path.

End to end, interleaved A/B against a binary built from the pre-change sources
(three rounds, same prompt, greedy):

| build | prefill | decode |
|---|---:|---:|
| reference (original sources) | 37.95 s | 507.5 ms/token |
| packing + fast decode | **24.25 s** | **325.4 ms/token** |
| + parallel RMS stats | 24.2 s | **314.7 ms/token** |

**1.56x on prefill and 1.61x on decode** (2.0 -> 3.2 tokens/s), with
byte-identical text for q8 and q4.

The server eval suite agrees: 15/15 pass, and its wall-clock cases drop by the
same factor (long prompt 134.5 s -> 95.8 s, multi-turn 30.9 s -> 21.1 s, identity
29.7 s -> 20.7 s).  `server/eval_baseline.json` is refreshed from that run.

**Where it must not be applied:** the q4 nibble unpack does `b - 16*floor(b/16)`.
A byte value 1e-7 below a multiple of 16 flips the `floor` and swaps the two
nibbles.  Removing the rounding there turned q4 output into an immediate EOS
while q8 stayed byte-identical, which is exactly the kind of bug a single-format
test would miss.  The rounding stays in the nibble path; every other decoder
(weights, activations, residual, RMSNorm input, q4 scale/min) uses the fast form.

Correctness: the engine still produces byte-identical text to the pre-change
reference binary for both q8 and q4 (`engine/verify_packing.sh`).

### Probes

Two standalone binaries answer layout questions without touching the model:

```sh
g++ -O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -std=c++17 \
    -o /tmp/matvec_probe engine/matvec_probe.cpp -lEGL -lGLESv2 -lm
/tmp/matvec_probe 1536 768 10        # rows, cols, iterations

g++ ... -o /tmp/texsize_probe engine/texsize_probe.cpp -lEGL -lGLESv2 -lm
/tmp/texsize_probe                  # which texture sizes this driver accepts
```

Both are headless: they use the same surfaceless-Mesa EGL platform as the
engine, bind a VBO for the full-screen triangle (GLES2 rejects client-side
arrays, which shows up as a hard fault rather than an error), and print
unbuffered so a crash cannot swallow the log.  `matvec_probe` tiles its draws at
1024 rows and its weight texture must stay under 2048 rows — the same limits the
engine lives with.

### Rejected: same trick on the attention decoders (-4%, wrong output)

Applying the no-rounding form to `dec24raw`, `decKV` and `decKV8` measured
325 -> 312 ms/token (4%) but produced garbage (`%<tts_text_bos_single>1%...`
repeating) for both q8 and q4.  Reverted.

The difference from the safe cases: those decoders feed a **threshold**,
`if (u >= 32768.0) u -= 65536.0` (and `>= 128` for the 8-bit KV).  With the
rounding removed, a value that should be exactly 32768 can come out 32767.999,
which no longer subtracts and flips the sign of the most negative KV entries.
The matvec decoders are safe because their result is only ever multiplied — no
branch depends on the exact integer.

If the 4% is ever wanted, the minimal change is to keep one floor on the high
byte only, `floor(hi*255+0.5)*256 + lo*255`: the sign boundary then stays exact
because the high byte is integral.

## Parallelising the RMS sum of squares (+3%)

`rms_stats` used to be a single fragment looping `DIM` times, so every one of the
768 iterations paid full texture latency: ~3.5 ms per layer for the attention
norm and ~2 ms for the FFN norm.  It is now split over 32 fragments (24 elements
each) plus a one-fragment reduce.

That only bought 3.1% end to end (325 -> 315 ms/token, three consecutive runs at
313.9 / 314.7 / 315.8), not the ~10% the ALU reduction suggests, because **each
extra draw plus finish costs close to a millisecond on this driver**: the second
pass and its target switch eat most of what the shorter serial chain saves.
Output stays byte-identical for q8 and q4.

That fixed cost is the thing to remember when planning further work here: on this
board, splitting a pass pays only if it removes *more than about a millisecond*
of serial work, which is why the matvec rewrites (tens of ms) were worth it and
small per-layer passes are not.

## Long prompts did not get faster (attention is the wall)

Same binary, 267-token prompt: prefill 91.0-91.8 s, i.e. ~340 ms/token average,
versus 331 ms/token at 73 tokens.  Before the matvec work the same prompt also
took 91.5 s — **the hex improvement is entirely absent there**, which locates the
long-prefill cost in the attention passes rather than the matvecs.

Profiled at position ~267 (`LAYER_PROFILE=1`, gen 1):

| phase | ms/layer at pos 73 | ms/layer at pos 267 |
|---|---:|---:|
| attention block | 11.2 | **15.9** |
| w1+w3+SwiGLU | 14.9 | 14.7 |
| w2 | 10.5 | 10.5 |
| wo | 4.1 | 4.1 |
| rms_att + rms_ffn | 7.1 | 6.7 |

Everything except attention is flat in position; attention grows ~linearly and
becomes the largest single phase around 250 tokens.  The attention block is five
dependent passes (`kv_rope_write`, `att_scores`, `att_stats`, `att_probs`,
`att_v`), each needing its own finish, so at ~1 ms per draw+finish the fixed cost
is already ~5 ms/layer.  Next step would be parallelising `att_stats` (8 serial
fragments today, one per head) and testing whether a tiled score pass beats the
current per-element KV decode — both are multi-pass changes, so the 1 ms/draw
budget has to be respected.

### Per-pass attention breakdown (pos ~73, profiled, ms/layer)

| pass | ms | note |
|---|---:|---|
| qkv matvec | 5.28 | the only part the hex optimizations touched |
| att_scores | 3.51 | q·k, 8 heads x pos, 96-element dot with a KV decode per element |
| att_v | 2.56 | p·V, 768 outputs x pos |
| kv_rope_write | 2.45 | one draw, no loop |
| att_probs | 1.57 | softmax normalise, 8 x pos elements |
| att_stats | 1.54 | max + sum per head, 8 serial fragments |

Five dependent draws, each needing its own finish, so ~5 of the ~17 ms is the
driver's per-draw cost; the rest is decode-bound ALU in the three loop passes.
Merging any two passes would save about a millisecond per layer (~2.5% of a
token) and each merge needs the intermediate result to be complete, which is
exactly what the finish buys — that is why the block is left as it is rather
than micro-merged.


## Hoisting the position-only part of attention (+5% short, +12% long)

`score_program` runs one fragment per (head, t).  For every one of the 96 head
dimensions it decoded q[i], decoded the RoPE partner q[j], **recomputed the qk-norm
factor with a second 96-iteration loop**, fetched cos/sin and decoded K.  All of
that except the K fetch depends only on the current position, so it was recomputed
`pos` times per token.

It is now three passes: `qnorm_scale` (one fragment per head, the norm factor),
`qrot` (one fragment per (head, dim), the rotated+normed q) and a score pass that
only does `q_rot[i] * k[i]`.  That is two extra draws+finishes per layer against a
per-element drop from five fetches / four decodes to two fetches / two decodes.

Interleaved A/B on the Pi, greedy, `TIMING=1`:

| context | path | prefill | decode |
|---|---|---:|---:|
| 73 tokens | inline (old) | 24.2 / 25.2 s | 311.4 / 323.9 ms/token |
| 73 tokens | **hoisted** | 24.9 / 24.1 s | **300.6 / 301.7** |
| 267 tokens | inline (old) | 89.5 / 91.8 s | 413.9 / (1056.7 outlier) |
| 267 tokens | **hoisted** | **83.0 / 84.3 s** | **359.7 / 370.5** |

Short context: **-5% decode**, prefill unchanged.  Long context: **-7.7% prefill,
-12% decode**.  Output stays byte-identical for q8 and q4.

With the hoist as the default path the canonical 16-token prompt now decodes at
**303.1 ms/token** (was 314.7), and the server eval's long-prompt case drops from
94.5 s to **85.5 s** (15/15 still passing, baseline refreshed).

Two extra draws were expected to cost ~2 ms/layer and make short contexts a wash;
they did not, because the per-fragment work they remove is larger than the
draw+finish overhead even at 73 positions.  `ATTN_HOIST=0` keeps the old path
compiled in for comparison.

**The bug worth remembering:** the new qrot texture is `DIM` wide, but the score
shader indexed it with `QKVDIM` (1536, the qkv width).  Sampling the wrong half of
a texture produces *no GL error* — the model simply emitted EOS immediately and
the output was empty.  Any texture you create must be indexed with its own width.


### Cost model correction: draws are pipelined, only GPU work counts

The earlier note in this file ("a draw+finish costs ~1 ms, so splitting a pass
needs to save more than that") was an artefact of `LAYER_PROFILE`, which calls
`glFinish()` between phases to attribute time.  Under `GPU_LAYER=1` the real
`att_finish()` is a **no-op** (`layer_finish` skips `NF_ATT`), so the CPU submits
all passes back to back and the GPU executes them in order: per-draw overhead is
hidden, and only the total GPU work matters.

Evidence: folding the softmax normalisation into `att_v` removes a whole pass
(probabilities) and is **exactly neutral** (301.1 vs 301.6 ms/token), because the
inline `exp` costs about what the extra pass did.  Conversely the attention hoist
added two passes and still won 5%, because it removed far more work than it added.

So the levers are, in order:

1. **Ops per multiply-accumulate.** The q8 matvec spends about 8 scalar ops per
   weight: ~2 decoding the weight, ~5 decoding the two activations, 1 MAC.  The
   activation decode dominates and is paid once per output row, i.e. 2432 times
   for the FFN gate.
2. **Share the activation decode across rows.** A fragment can compute two output
   rows and write them as two int16 in one texel; pairing w1 row r with w3 row r
   would also give swiglu its two inputs from a single texel.  That halves the
   activation decode per weight (8 -> ~4.3 ops) and halves the FFN draw count,
   at the cost of a 1216-row tiling (2432 > the 2048 texture limit) and a new
   packed input layout for w2.
3. **Fewer MACs**, which means a different model or sparsity — out of scope here.


## Paired FFN gate matvec (-6.6% decode, -7.4% prefill)

The q8 matvec spends about five of its eight scalar ops per weight decoding the
two activations, and that decode is repeated for **every output row** — 2432
times per layer for the FFN gate.  One fragment now computes row r of `w1` *and*
row r of `w3`, sharing the decode, and writes both results into one texel as two
int16.  That is one draw instead of two and roughly half the activation traffic.

`engine/matvec_probe.cpp` measures it in isolation (2048x768, two rows per
fragment): 7.30 ms for both matrices against 2 x 5.46 ms separately.

Pi, greedy, interleaved A/B (`FFN_PAIR=0` keeps the old path):

| path | prefill | decode |
|---|---:|---:|
| old (two draws) | 24.2 / 24.4 s | 299.1 / (597.4 outlier) ms/token |
| **paired** | **22.2 / 22.7 s** | **279.6 / 279.2 ms/token** |

**-7.4% prefill, -6.6% decode**, output byte-identical for q8 (q4 keeps the old
path: the paired shader assumes int8 weights, and the code checks the type).

Cumulative against the original sources: 507.5 -> 279.4 ms/token (**1.82x**) and
37.95 -> 22.5 s prefill (**1.69x**).

The same trick applies to `w2`, whose activation is 2432 wide and decoded once
per output row (768 of them).  Pairing its rows needs the residual stream kept
two-per-texel too, because `matvec_add` adds the residual one value per texel and
`rmsnorm` reads it the same way — that is the next step, worth roughly another
10% if it lands.


## Packed residual stream: implemented, measured, neutral

`matvec_add` (the `wo` and `w2` passes) had the same shape of waste as the FFN
gate: one activation decode per output row, and a residual stream stored as one
int16 per texel with the (b, a) bytes unused.  Storing two residual values per
texel lets the same fragment produce rows 2t and 2t+1 and share the decode.

It is implemented (`RES_PACK=1`) and byte-identical, but **neutral**: 280.7 /
280.1 ms/token against 281.7 / 278.8 unpacked.  Unlike the gate, `w2` has only
768 output rows, so there is half as much decode to amortise, and the kernel now
carries two accumulators and two weight fetches per iteration, which this
driver's small register file does not absorb for free.  It stays off by default.

The interesting part is the bug it exposed: `upload_hidden` was rewritten to pack
pairs and indexed `x[2 * t]` unconditionally, so with packing disabled the
embedding upload took every other element.  The GPU path produced garbage while
the CPU path (`GPU_LAYER=0`) stayed correct, which is what localised it.  Lesson:
when a layout change has two variants, check the *old* variant too.


### Where the 280 ms/token goes now (LAYER_PROFILE, per layer)

| phase | ms/layer | notes |
|---|---:|---|
| attention | 18.6 | 7 passes: qkv, qk-norm, q-rot, KV rope, scores, stats, p*V (fused) |
| FFN gate | 12.4 | was 14.7 before the paired matvec |
| w2 | 10.9 | two column-chunked `matvec_add` passes |
| rms_att + rms_ffn + rms_final | 8.3 | three stats+reduce+norm groups |
| wo | 4.3 | one `matvec_add` |

These are inflated by the profiler's per-phase `glFinish` (the sum is ~58 ms/layer
against ~35 ms of real time), so read them as shares, not absolutes.  Attention is
again the largest block; the FFN gate is now the second, and the paired matvec is
what moved it down.


## X-format activation layouts and ACT8 (2026-11-01)

The activation texture layout is now explicit in `src/xformat.*`:

| layout | values / texel | notes |
|---|---:|---|
| `I16_1` | 1 | q16 in `.rg`; default residual stream |
| `I16_2` | 2 | default matvec input and `RES_PACK` residual |
| `I8_4` | 4 | signed int8, 1/16 scale; `ACT8` |

Hot matvec kernels use the q16-domain macros (`X4Q`, `R1Q`) so the inner
loop keeps the original integer arithmetic; producers use `XST*` and generic
attention/RMS code uses the real-valued `X1/X2/X4` variants.  This removed the
per-shader layout strings that caused the earlier ACT8 patch failures.

`--ppl` is the quality gate.  On the fixed 254-token corpus in
`engine/data/ppl_test.txt` and a second 254-token Chinese corpus (q8,
GPU_LAYER, KV_PACK=16):

| path | corpus A PPL | corpus B PPL |
|---|---:|---:|
| baseline | 313.94 | 909.12 |
| `ACT8=2` (FFN hidden only) | 320.77 (+2.2%) | 931.71 (+2.5%) |
| `ACT8=1` (all matvec inputs) | 315.38 (+0.5%) | 1024.38 (+12.7%) |

Corpus B is `README.zh-CN.md` (254 tokens); it deliberately differs in style
from corpus A.  Neither ACT8 mode is a safe default: the full path is nearly
neutral on A and clearly worse on B.  Both stay opt-in; the greedy q8/q4
strings remain byte-identical with `ACT8` unset.

This is the same lesson as q4: a small fixed corpus can rank a lossy format
wrongly.  Keep the harness and compare at least two corpora before changing a
default.


## The refactor's 12% regression: helper indirection in the hot matvec loops

Moving the engine into `src/` with the X-format abstraction made decode ~12%
slower (313 vs 279 ms/token) with **identical** GL work — same draws (14004),
same fragments (10,713,749), same uploads (83), same CPU time (11.2+4.2 s), and
*more* free CMA.  So the cost was in the code the driver generated, not in the
amount of work submitted.

Diffing the emitted GLSL of every shader (dump is one env var away) showed the
hot loops had been rewritten from

```glsl
vec4 xa = texture2D(X, vec2((float(2*q)+0.5)/float(384), 0.5));
vec4 xb = texture2D(X, vec2((float(2*q+1)+0.5)/float(384), 0.5));
s += w.x*dA(xa) + w.y*dB(xa) + w.z*dA(xb) + w.w*dB(xb);
```

to

```glsl
vec4 x = X4Q(X, q*4, 384.0);          // macro -> x_i16quadq(fetch, fetch)
s += w.x*x.x + w.y*x.y + w.z*x.z + w.w*x.w;
```

Semantically identical, but on VC4 the helper that receives two texture fetches
as arguments compiles to a slower loop.  Emitting the inline form again in the
three hot builders (`matvec_q8`, `matvec_q8_pair`, `matvec_q8_add`) restores
both the speed and — notably — the exact numeric output:

| build | decode (interleaved A/B) | PPL | greedy text |
|---|---:|---:|---|
| pre-refactor | 280.6 / 299.2 | 273.05 | `…提供准确、` |
| refactor (helpers) | 313.4 | 274.43 | `…提供精准` |
| refactor + inline hot loops | 282.8 / 296.5 | 273.29 | `…提供准确、` |

The 0.4% perplexity drift disappeared with the inline form, which is a good
reminder that "semantically identical" GLSL is not bit-identical arithmetic on
this GPU.

Rule for `shaders.cpp`: the generic `X4Q` macro form is fine for cold paths, but
the three widest matvec loops keep the explicit two-fetch/decode form.


## Attention: q stored as q16 pairs (-2.6%)

`att_qrot` used to write one 24-bit texel per head dimension and `att_scores`
decoded one q value per fetch.  The stored grid was `enc24s(q, 256)`, i.e. an
integer in +-32767 — exactly what a q16 slot holds — so the format can be
halved with no loss: `att_qrot` now emits two head dims per fragment as a q16
pair and the scores loop decodes two q values from one fetch.

| build | decode (4 interleaved rounds) | PPL |
|---|---|---:|
| pre-refactor | 280.9 / 279.0 / 279.6 / 281.4 | 273.05 |
| + inline matvecs + att_v fusion | 272.8 (min) | 273.29 |
| **+ q16 q-vector** | **273.6 / 271.9 / 272.9 / 273.0** | 273.40 |

Cumulative against the pre-refactor engine: **-2.6% decode**, golden greedy text
byte-identical, q4 unchanged (PPL 328.53 vs 329.16 before, text identical).

The same trick applies to the score texture itself (it is also a 24-bit encoding
of a 1/1024 grid), which would let two positions share one fetch in the scores
pass; that is the next step if more is needed.


## Score texture as q16 pairs (-1 to -3%)

The score matrix was the same story as the q vector: `enc24s(s, 1024)` stores an
integer in +-32767, i.e. exactly a q16 slot.  Packing two positions per texel
(`SEQ/2` wide) lets one fragment compute **two** attention rows, sharing the q
decode, and halves the fetch count in `att_stats` (which scans the scores twice:
once for the max, once for the sum) and in `att_v`.

| build | decode (min of 4 interleaved rounds) | PPL |
|---|---:|---:|
| pre-refactor | 278.2 | 273.05 |
| + inline matvecs, att_v fusion, q16 q vector | 271.9 | 273.40 |
| **+ q16 score pairs** | **270.1** | **272.53** |

Golden greedy text is byte-identical throughout.  q8 PPL ends up slightly *below*
the pre-refactor number (272.53 vs 273.05) because the intermediate values now go
through the same q16 grid the reference quantization used.

Cumulative against the pre-refactor engine: **-2.9% decode** (280 -> 270 ms).


### Tried: online softmax statistics (not landed)

`att_stats` scans the score row twice — once for the max, once for the
exponential sum.  Folding both into one online pass (`sum = sum*exp(mx-a)` when a
new max appears) looked like a clean 2x on that pass, but the shader produced
garbage output (PPL 33619 instead of 272) in both the plain and the
guarded-first-element forms, and the second attempt did not even rebuild.  Left
for a session that can iterate on the GLSL in isolation; the two-pass version is
what ships.


## Attention statistics: one fragment per head (-0 to -1%)

`att_stats` used to run two fragments per head (`kind` 0 and 1) that performed
the *identical* two-pass scan and differed only in which statistic they wrote.
Now one fragment per head writes `(max, sum)` packed into a single texel, with
the max on the 1/1024 grid and the sum on a 1/16 grid (they need different
ranges: the sum reaches `pos+1`, which does not fit a q16 slot at 1/1024).

| build | decode (min of 3) | PPL |
|---|---:|---:|
| pre-refactor | 277.2 | 273.05 |
| + inline matvecs, att_v fusion, q16 q + scores | 270.1 | 272.53 |
| **+ packed stats** | **270.0** | **272.27** |

The pass is now half the fragments and one texel per head, but it was small
enough that the wall-clock difference sits inside the noise; the perplexity is
unchanged and the work is strictly less, so it stays.


## att_v: hoisted V index + unrolled by two (-1%)

The `p*V` loop recomputed the V texel index every iteration even though it only
depends on the head and the output dimension, and it fetched one score texel per
position even though the score texture now holds the pair (t, t+1).  Hoisting the
index out of the loop and unrolling by two halves both counts:

| build | decode (3 rounds) |
|---|---|
| pre-refactor | 278.4 / 287.6 / 280.0 |
| + all previous work | 270.0 (min) |
| **+ hoisted/unrolled att_v** | **269.0 / 271.1 / 273.2** |

Cumulative against the pre-refactor engine: **280 -> 269 ms (-3.9%)**, greedy text
byte-identical, PPL 272.27 (pre-refactor 273.05).

The run-to-run variance also dropped sharply (269-273 vs 278-288 before), which
suggests the loop was bound by the integer index arithmetic rather than memory.


## Packed residual stream, now default (-11%)

`RES_PACK` (two int16 residual values per texel) was implemented early in the
refactor work and measured **neutral**, so it stayed opt-in.  That measurement is
stale: back then the matvec loops were the bottleneck and the residual packing
only helped `matvec_add`.  With the matvec decoders inline and the attention
passes tightened, the rmsnorm passes became a large share of the layer — and both
of them read the residual stream (`stats` and `norm` each fetch all 768 values).
Packing halves those fetches.

| build | decode (3 rounds) | q8 PPL |
|---|---|---:|
| pre-refactor | 277.4 / 276.9 / 278.0 | 273.05 |
| + everything up to att_v unroll | 269.0 | 272.27 |
| **+ packed residual (default)** | **235.4 / 242.3 / 235.4** | **269.61** |

**-15% against the pre-refactor engine**, and perplexity is *better* (269.61 vs
273.05) because the residual now shares the q16 grid the reference uses.

A precision detail found while validating: the softmax denominator was packed on a
1/16 grid, which cost q4 2% perplexity (334.9 vs 329.2).  At 1/32 both models are
back: q8 269.61, q4 329.22 (pre-refactor q4: 329.16).

`RES_PACK=0` restores the unpacked stream.


## Switch sweep on the new baseline (all toggles)

With the default path at ~240 ms the opt-in switches were re-measured; the
relative value of every one of them moved because the baseline they were tuned
against is gone:

| switch | decode | verdict |
|---|---:|---|
| default | 240.0 | — |
| `ACT8=2` | 272.3 | now **13% worse** |
| `ACT8=1` | 282.7 | now **18% worse** |
| `KV_PACK=8` | 238.7 | neutral |
| `FFN_PAIR=0` | 341.3 | paired gate is essential |
| `ATTN_HOIST=0` | 259.1 | hoisting is essential |

The int8 activation modes now lose because the inline fast paths cover `I16_2`
only; `I8_4` falls back to the generic helper shaders.  Worth revisiting only if
the I8_4 path gets the same inline treatment.

An inline `matvec_add` for the packed residual (`I16_2` in and out) was also
tried: 243.4 / 237.3 / 240.1 against 235.4 / 242.3 / 235.4 — inside the noise, so
it was reverted rather than kept for the sake of looking busy.

### Remaining, with estimates

1. `scores` shares one K fetch between two queries during prefill (needs a KV
   write layout change) - the largest structural item left, ~3-5%.
2. `kv_rope_write` folded into the qkv matvec output (one fewer draw per layer).
3. rms weights packed as q16 pairs so the norm pass does one fetch per two
   outputs, ~0.5-1%.
4. Inline `I8_4` variants, which would bring the int8 activation modes back.


## RMS weight vectors packed as q16 pairs

`rms_norm` writes two outputs per fragment but fetched the two weight values from
two separate texels.  Packing the (static) rms vectors two-per-texel makes it one
fetch.  Output text and perplexity are unchanged (269.6078, identical to four
decimals); the win is small and sits under the noise this machine now has.

## Benchmarking caveat: the Pi also runs a Minecraft server

`./pumpkin` has been resident since Nov 1 (~72 CPU-hours).  With it running, the
same binary swings between 236 and 560 ms/token and the load average sits at
2-2.6, which is why several A/B rounds in this log disagree.  Comparisons taken
back-to-back in one session are still valid (both sides see the same
interference), but absolute numbers should be taken with the server idle, and the
best-of-N is the only stable statistic.

## MiniMind2-small Q8 on the laptop (2026-11-02)

Machine: Iris Xe (ADL GT2), Mesa 26.2.5. Model:
`minimind2-small.q8` (28 MB), tokenizer `minimind2-tokenizer.bin`,
`ROPE_THETA=1000000 ROPE_HALF=1`. Prompt `用一句话介绍你自己。` through
`--chat --template qwen`, greedy, 64 generated tokens; prefill is the whole
chat prompt, not a per-token number.

| path | prefill | decode | output |
|---|---:|---:|---|
| q8, GPU layers (fixed, paired FFN) | 0.20 s | 7.7 ms/token | coherent |
| q4, GPU layers (Q4M1 from I16) | 0.20 s | 6.7 ms/token | coherent |
| q8, CPU FFN (`CPU_FFN=1`) | 0.31 s | 11.8 ms/token | coherent |
| i16, GPU layers (`GPU_LAYER=1`) | 0.28 s | 11.1 ms/token | coherent |
| q8, GPU layers, `FFN_PAIR=0` | 0.17 s | 6.2 ms/token* | **garbled** |
| i16, CPU FFN (`CPU_FFN=1`) | 0.73 s | 19.4 ms/token | coherent |

\* the broken path was timed on a 6-token run; it is not a usable baseline.

The q8 failure was not "q8 in general": `Q8_PARTS=ffn` reproduced it while
`w2`, `qkv` and `wo` stayed coherent; `GPU_FFN=1` and `GPU_LAYER=1` both failed
while `CPU_FFN=1` worked; i16 used the same non-paired path and worked;
MiniMind3 (hidden 2432, chunked, paired path) worked.

The fault is the non-paired q8 FFN path (`matvec_to_row` q24 +
`swiglu_2row`) for `hidden=1408`. `GPU::ffn_split` now routes INT8 w1/w3
through `ffn_pair_chunk`, the same q16 paired path already used by chunked
models; `FFN_PAIR=0` keeps the old path for A/B. MiniMind3 q8/q4 canonical
greedy outputs are unchanged after the change.

The Q4M1 file is generated directly from the I16 checkpoint with
`make_q4_from_i16.py` (20.7 MB). It takes the old non-paired path and is
coherent; the failure is q8-specific, not a generic non-paired-path bug. Q4 is
faster than q8 here (6.7 vs 7.7 ms/token) but noticeably lower quality: it
answers `1+1=2` and `中国的首都是北京`, while the self-introduction drifts to
"我是一台电脑".


## Q4 path: nibble decode without mod (-13%)

The q4 matvec decodes two nibbles per weight byte with `mod(b,16)` and
`floor(b/16)`.  GLSL's `mod` is `x - y*floor(x/y)`, i.e. a divide plus a floor,
and it was being paid twice per byte, 768/8 times per output row.

Replacing it with an explicit split — `h = floor(b*0.0625)`, `l = b - h*16` —
keeps the exact same integers (the byte itself is still `floor(w*255+0.5)`) and
cuts the stage from ~8 ops per byte to ~5:

| q4 build | decode | PPL | greedy text |
|---|---:|---:|---|
| before | 463.6 / 463.6 ms | 329.22 | `我是MiniMind，一个由jingyaogong…` |
| **+ split nibble decode** | **403.3 / 407.4 ms** | **329.15** | identical |

Also tried and kept: moving the per-weight `mn` term out into
`sc*sum(n*x) + mn*sum(x)` so the min is paid once per group (neutral on its own,
but it is what makes the nibble split cheap to express).

An early q4 `matvec_add` pair kernel (two rows per fragment, packed residual)
was 20 ms faster but produced garbage (PPL 4e14) because the q4 accumulator is
in raw q16 units while `x_st2` expects activation units.  The fixed kernel uses
`x_st2q`; it is the packed-residual result described below, not a reverted path.

### Why q4 stays slower than q8

Per weight, q4 needs a nibble split (~5 ops for two weights) plus a scale/min
FMA, while int8 needs `c*255-128` (2 ops).  The halved *fetch* count does not pay
for that on VC4, which is ALU-bound: q4 lands at 403 ms against q8's 240 ms.
Closing the rest of the gap needs either the pair kernel above or storing q4
weights pre-split (which doubles the weight footprint and does not fit CMA).


### Q4: paired gate and packed residual (463 -> 380 ms, -18%)

Two more changes landed on the q4 path after the nibble split:

1. **Paired FFN gate** (`matvec_i4_pair`): one fragment computes `w1[r]` and
   `w3[r]`, sharing the activation decode, and writes a raw q16 pair for swiglu.
   The first attempt produced garbage because the q4 accumulator works in *raw*
   q16 units while `x_st2` expects activation units (it multiplies by 256); the
   fix is `x_st2q`.  The same unit trap applies to the q4 `matvec_add` pair
   kernel (`matvec_i4_add_pair`).
2. **Packed residual for q4** (`res_pack_` no longer requires int8 weights): both
   the q4 add pair kernel and the rmsnorm passes now read the stream two values
   per texel.

| q4 build | decode | PPL |
|---|---:|---:|
| baseline | 463.6 | 329.22 |
| + nibble split | 403.3 | 329.15 |
| **+ paired gate + packed residual** | **380.7 / 382.9** | **327.85** |

Phase profile (ms/layer, profiled): FFN 28.5 -> 23.8, w2 16.9 -> 12.7,
wo 6.5 -> 5.4, attention unchanged at ~22.

q4 remains slower than q8 (380 vs 240 ms) because a nibble split costs several
ALU ops per weight against int8's single `c*255-128`, and VC4 is ALU-bound; the
halved fetch count does not pay for it.  Closing that gap would need pre-split
int8 weights, which doubles the weight footprint past what CMA can hold.

## Iris Xe model matrix (2026-11-02, no elevation)

The workspace sandbox now exposes `/dev/dri`; `eglinfo -B` reports
`Mesa Intel(R) Iris(R) Xe Graphics (ADL GT2)` and `Accelerated: yes`.  These
runs use the in-tree `engine_mm3` with no elevated permissions, `GPU_LAYER=1`,
`KV_PACK=16`, greedy, 32 generated tokens, best of two warm runs.  Prefill is
the whole prompt/chat template, not a per-token number; prompts differ by
family, so compare decode more than prefill.

| model | file | size | prefill | decode | output |
|---|---|---:|---:|---:|---|
| MiniMind3 q8 | minimind3.q8 | 66 MB | 0.66 s | 14.1 ms/token | coherent |
| MiniMind3 q4 | minimind3.q4 | 45 MB | 0.88 s | 14.6 ms/token | coherent |
| MiniMind2 q8 | minimind2-small.q8 | 28 MB | 0.39 s | 6.8 ms/token | coherent |
| MiniMind2 q4 | minimind2-small.q4 | 20.7 MB | 0.57 s | 6.4 ms/token | coherent |
| stories15M q8 | stories15M.q8 | 24.5 MB | 0.25 s | 3.8 ms/token | coherent |
| stories42M q8 | stories42M.q8 | 58 MB | 0.25 s | 7.1 ms/token | coherent |
| TinyChat15M q8 | TinyChat15M.q8 | 24.5 MB | 0.23 s | 3.6 ms/token | coherent |
| stories15M fp32 | stories15M.bin | 60 MB | 0.29 s | 9.8 ms/token | coherent |
| TinyChat15M fp32 | TinyChat15M.bin | 60 MB | 0.23 s | 9.2 ms/token | coherent |
| stories15M-enzh fp32 | stories15M-llama2-enzh.bin | 87 MB | 0.19 s | 6.2 ms/token | coherent English |
| stories260K fp32 | stories260K.bin | 1 MB | 0.38 s | 1.5 ms/token | gibberish (expected) |

The q8 15M files initially produced gibberish on the GPU FFN path while their
fp32 and i16 checkpoints were coherent.  The cause is the fused-W13 q8 path,
the same q8 q24-FFN family as the earlier MiniMind2 failure.  The runtime now
has `matvec_q8_pair_fused`, a paired q16 W13 shader; old fused q8 files work
again (stories15M fused: ~5-8 ms/token, coherent).  `make_q8_from_i16.py` still
writes split W1/W3 by default because separate scales and the existing paired
path are faster (stories15M split: ~3.8-4.2 ms/token); the table uses the split
files.  `stories42M` already used split W1/W3.  Q4 is unaffected (fused q4
still uses its own path).

## Pi Zero model matrix (2026-11-02, VLAB cross-build)

The current tree (commit `252c9c4`) was cross-built on VLAB with
`make -C engine -B pi CROSS_SYSROOT=$HOME/raspi-root4` (GCC 12, ARMv6/VFPv2)
and the resulting `engine_mm3_pi_new` was copied to the Pi.  All runs use
`GPU_LAYER=1 KV_PACK=16`, greedy, 32 generated tokens, best of two runs;
`SEQ=768` for MiniMind3.  Prefill is the whole prompt/chat template, not a
per-token number.  The Pi was otherwise idle apart from the resident
`pumpkin` server; `CmaFree` was 117.6 MB before and 93.0 MB after the matrix.

| model | file | prefill | decode | output |
|---|---|---:|---:|---|
| MiniMind3 q8 | minimind3.q8 (66 MB) | 10.95 s | 251.8 ms/token | coherent |
| MiniMind3 q4 | minimind3.q4 (45 MB) | 17.71 s | 387.6 ms/token | coherent |
| MiniMind2 q8 | minimind2-small.q8 (28 MB) | 6.85 s | 143.5 ms/token | coherent |
| MiniMind2 q4 | minimind2-small.q4 (20.7 MB) | 10.01 s | 186.7 ms/token | coherent |
| MiniMind2 i16 | minimind2-small.i16 (58 MB) | 8.89 s | 232.2 ms/token | coherent |
| stories15M q8 split | stories15M-split.q8 (24.5 MB) | 3.26 s | 138.4 ms/token | coherent |
| stories15M q8 fused | stories15M-fused.q8 (24.5 MB) | 3.32 s | 137.2 ms/token | coherent |
| stories42M q8 | stories42M.q8 (58 MB) | 3.57 s | 188.1 ms/token | coherent |
| TinyChat15M q8 split | TinyChat15M-split.q8 (24.5 MB) | 3.35 s | 137.6 ms/token | coherent |
| TinyChat15M q8 fused | TinyChat15M-fused.q8 (24.5 MB) | 3.41 s | 136.4 ms/token | coherent |
| stories15M fp32 | stories15M.bin (60 MB) | 3.04 s | 168.3 ms/token | coherent |
| TinyChat15M fp32 | TinyChat15M.bin (60 MB) | 3.67 s | 161.8 ms/token | coherent |
| stories15M-enzh fp32 | stories15M-llama2-enzh.bin (87 MB) | 4.58 s | 242.4 ms/token | coherent English (toy MT) |
| stories260K fp32 | stories260K.bin (1 MB) | 5.93 s | 81.9 ms/token | gibberish (expected) |

The fused-W13 q8 path is correct and **not slower** on the Pi than the split
path: identical text, 137.2 vs 138.4 ms/token for stories15M, and 136.4 vs
137.6 ms/token for TinyChat15M.  The split files still have separate W1/W3
scales, so they remain the preferred conversion for quality.

MiniMind3 q8 and q4 both stay coherent, but the q8 self-introduction differs
(``我是一个AI助手...`` vs q4 ``我是MiniMind...``); this is present with both the
old and new Pi binaries and is not caused by the fused-W13 change.

## Draw-path work: G1 (GL state caching) and G2 (rmsnorm reduce fusion)

Measured on the Pi Zero with `scripts/bench_plain_ab.sh`, plain mode, greedy,
`--prompt "Once upon a time, there was a little" --gen 32`, min of three
alternating runs.  CPU/token is the rusage difference between the gen-32 and
gen-128 runs, so it excludes prefill and model load.

| build | draws/token | decode (min) | CPU/token | user | sys |
|---|---:|---:|---:|---:|---:|
| deef607 baseline | 204 | 237.3 | 136.4 | 108.5 | 27.9 |
| + G1 state caching | 204 | 234.6 | 123.0 | 96.7 | 26.3 |
| + G2 rms reduce fused | 188 | 229.2 | 114.5 | 89.5 | 25.0 |

Target is 150 ms/token, so this is roughly a fifth of the way there: the draw
count has to come down by another ~90 before the fixed per-draw cost stops
dominating.

### G1: cache the GL state the draws never change

Each draw used to do 3-4 `glGetUniformLocation` calls by name, an
`glGetAttribLocation`, two `glUniform1i` for sampler units that never move, and
`glEnableVertexAttribArray`/`glVertexAttribPointer` for an attribute whose state
is context-global in GLES2.  All of it is now cached: `uloc()`/`aloc()` key the
cache on the string literal's address (names are literals, so the pointer is a
stable identity and beats hashing), `setup_attrib()` sets the attribute once,
`set_sampler()` writes a unit once, and `bind_tex()`/`bind_fbo()`/`use_program()`
skip redundant rebinds.

Two traps found while landing it:

* **A sampler is not fixed per program.** The q4 kernels first point `R` at unit
  2 and then re-point it at unit 3 (because `P` takes unit 2).  Caching the unit
  without verifying it made q4 emit `排排排排...`, while q8 stayed correct —
  a gate that only covered q8 would have shipped it.
* **Static texture helpers bypass the cache.** `make_texture()` binds on its own,
  so the bind cache is file-scope and the helpers call
  `gl_state_forget_bindings()`; a per-instance cache would go stale the first
  time a new input texture is allocated mid-run.

G1 is worth 11.6% of CPU but only 1.1% of decode: the CPU is already hidden
behind the GPU, so submission savings alone cannot move the wall clock.

### G2: fold the rmsnorm reduce into the normalise

`rmsnorm` was three draws (32 partials, 1x1 reduce, normalise).  The reduce
produces one scalar that every normalising fragment reads, so the normalise can
sum the same 32 partials itself with the identical expression and order:

    float ss=0.0;
    for(int p=0;p<32;p++) ss += x_i24raw(texture2D(S,vec2((p+0.5)/32,0.5)))/4096.0;

`regression.sh` reports PPL 311.1636 with and without it and both gates keep
their byte-identical text, so the fusion is numerically free.  It is the default
on VC4/VideoCore and off elsewhere (`RMS_FUSED` / `RMS_SPLIT` override), because
32 fetches per output fragment may cost more than a draw on other GPUs.
`ACT8=1` does not accept the fused form and stays on the split path.

### Two negative results worth keeping

* **Batching the command stream is slower.** The FFN draws each call `glFlush()`.
  Suppressing that flush (`FFN_NOFLUSH=1`) made decode *worse*, 253-266 vs
  234-239 ms/token across three alternating pairs: the flush is what lets the
  GPU start while the CPU builds the next command list, so a bigger batch just
  delays the first tile.
* **The 1536-column strip is required, not conservative.** Drawing the 1536-wide
  qkv matvec in one pass instead of two 768 strips returns garbage tokens
  (`编写一个编写一个...` on q8, `我我我我...` on q4).  The strip stays.

### Where the remaining draw budget goes

Per layer (q8 chunked path): rms_att 2, qkv 2 (strip), rope 1, qnorm 1, qrot 1,
scores 1, stats 1, v 1, wo 1, rms_ffn 2, gate pair 2, swiglu 2, w2 2 = 19 draws,
plus about 15 per token outside the layer loop (embedding, final rmsnorm,
logits tiles, sampling).  The next candidates, in order: fold `att_stats` into
`att_v` (the same trick as the rms reduce), merge the two swiglu draws by
writing both FFN chunks into one texture, and cut the logits tile count.

### Decode is draw-bound, not sequence-bound

Same prompt, same generation length, different `SEQ` (which only changes how far
the attention loops run):

| SEQ | decode |
|---:|---:|
| 192 | 236.9 ms/token |
| 384 | 230.1 ms/token |
| 768 | 228.1 ms/token |

Decode is flat, so the KV scans and score loops are not where the time goes;
each token costs about the same regardless of how much attention work it does.
At 188 draws that is ~1.2 ms per draw, which is why the working order is
"fewer draws" and why shaving arithmetic inside a shader has produced almost
nothing.

### What blocks the remaining 35%

Decode is 229 ms against a 150 ms target, i.e. roughly 70 draws too many at the
current per-draw cost.  The merges that would remove them are blocked by three
concrete things, not by a lack of ideas:

* **The 1536-column strip.**  qkv needs two draws because a single 1536-wide
  pass returns garbage (`编写一个编写一个...`); that is 8 draws per token that
  only the driver can give back.
* **One weight texture per matrix.**  The two FFN chunks are separate Mats, so
  the gate stays two draws and w2 stays two draws (2432 rows also exceeds the
  2048 viewport limit).  Merging them means changing the quantised weight layout,
  which is out of scope for this pass.
* **`att_stats` is quantised, not just computed.**  It stores `(max, sum)` as
  packed q16 and `att_v` reads the *quantised* values, so folding the stats scan
  into `att_v` (the same trick as the rms reduce) is not bit-exact unless the
  fused version reproduces the packing exactly.  Left alone deliberately.

### FFN gate+swiglu fusion: 4 draws -> 2, -20 ms, but not bit-exact (FFN_FUSE, default off)

The two FFN chunks each spend one draw on the paired gate matvec and one on
swiglu.  A single kernel can do both: one fragment evaluates two consecutive gate
rows for w1 and w3 (four dots sharing one activation decode) and writes the two
swiglu results as one q16 pair.  Halving the fragment count keeps the viewport at
rows/2 and the whole chunk becomes one draw, so the FFN gate+swiglu cost drops
from 4 draws to 2 per layer (188 -> 164 draws/token).

Interleaved A/B on the Pi, plain mode, 32 tokens, three alternating pairs:

| run | two-pass | fused |
|---|---:|---:|
| 1 | 229.8 | 206.3 |
| 2 | 229.4 | 213.1 |
| 3 | 229.0 | 207.3 |

Mean -20.4 ms/token (~9%), the largest single win since the GL state caching.

It is not bit-exact yet, so it is behind `FFN_FUSE=1` (default off).  The gate
int16 quantisation and the final q16 rounding are reproduced inline, but PPL moves
from 311.1636 (nll 5.7403) to 312.1366 (nll 5.7434) on the same corpus, while both
greedy gates still print their byte-identical text.  The difference must come from
arithmetic that no longer passes through the RGBA8 intermediate, or from VC4
compiling the four-accumulator loop differently from the two-accumulator one;
the accumulation loop is the first suspect.  With `FFN_FUSE` unset the default
path is unchanged: `regression.sh` gives `q8 OK`, `q4 OK`, PPL 311.1636.

### Note on merging the FFN weight chunks

The plan to merge w1a/w1b, w3a/w3b and w2a/w2b at load time cannot be taken as
written for w1/w3: those Mats are uploaded as `rows`-tall weight textures, so a
merged 2432-row w1 needs `GL_MAX_TEXTURE_SIZE >= 2432` (the board reports 2048 for
targets, and `make_target` refuses anything larger).  Merging the gate therefore
has to keep two weight textures per matrix and do the two chunks in one draw, which
is what the fused kernel above achieves without any repack.  w2 (`768 x 2432`, i.e.
608 wide by 768 tall as a texture) is not affected by that limit; it is a column
split, so a one-draw version would change the summation order and needs its own
bit-exactness check.

### Bisecting the fused FFN drift: it is 1 LSB of codegen noise, not a bug

The fused gate+swiglu kernel above is the fastest change on the board (-20 ms)
but it moved PPL from 311.1636 to 312.1366, so the bisect planned in the
the decode campaign was run.  A new `FFN_DUMP=1` debug hook prints the first 8 texels of
`ha`/`hb` (raw RGBA8 bytes) right after the FFN gate stage, for every layer-0
forward.

**The dumps are byte-identical except for two texels.**  Across ~28 layer-0
forwards (608 texels x 2 chunks each, so ~28k q16 values) exactly two values
differ, and each differs by **1 LSB of q16** (`...ffe4` vs `...ffe5`, `...ff97`
vs `...ff98`).  Gate decode, int16 quantisation, silu input and the q16 store
therefore all match; there is no encoding or logic error in the fused kernel.

Two candidate causes were then eliminated on the board, each with a PPL run
(`--ppl`, 254 tokens):

* **Silu expression form** (`FFN_SILU`): the current one-liner, a split-statement
  version, an explicit-reciprocal version and split+mul all give **312.1366**.
  The silu arithmetic is not the source of the drift.
* **Accumulation association** (`FFN_ACC`): this *is* the lever, but no variant
  reaches the reference.  Serial statements give **309.2918**, explicit
  left-associative gives **312.1366** (identical to the unmodified kernel, i.e.
  the VC4 compiler already picks left-association for `a+b+c+d`), pairwise gives
  **313.5574**.

The spread of those three numbers is the real result: PPL is chaotic with
respect to harmless float reassociation.  The unchanged default path already
shows it - `regression.sh` reports PPL 309.7812 for `ACT8=1` and 314.8067 for
`ACT8=2`, both shipped options with no gate failure.  A 1-ULP difference in a
handful of activations is enough to move PPL by +-2, so the fused kernel cannot
be steered back onto exactly 311.1636 by editing the expression: bit-exactness
here needs the *reference shader text* to survive, and sharing one column loop
between four accumulators changes how the VC4 compiler schedules and contracts
the multiply-adds.

Conclusion: `FFN_FUSE` stays default-off and both debug variants were reverted.
The lesson for the next candidate is to remove draws **without touching a
shader that contains an accumulation chain**; the `FFN_DUMP=1` hook is kept
because any future draw merge has to be checked the same way.

### Measured per-draw cost: ~0.18 ms, so draw merging is not the lever (2026-11-03)

The working assumption since the draw-count analysis has been "188 draws at
0.5-1.2 ms each, so ~94 ms is per-draw overhead, merge draws".  That number was
never measured directly.  It can be, without touching any arithmetic:
`RMS_SPLIT=1` turns off the fused rmsnorm reduce, which re-inserts the 1x1
reduce draw after each partial pass.  Same expressions, same summation order, so
the gates stay byte-identical (that is why the fusion was landed in the first
place) - the only difference is `2 * layers + 1 = 17` extra trivial draws per
token.

Same binary, alternate A/B, plain mode, 32 tokens, SEQ 768:

| run | default (188) | RMS_SPLIT (205) | delta |
|---|---:|---:|---:|
| 1 | 232.7 | 236.4 | +3.8 |
| 2 | 236.4 | 236.4 | +0.0 |
| 3 | 227.9 | 233.3 | +5.4 |

Mean +3.0 ms for 17 draws, i.e. **~0.18 ms per draw** (upper bound ~0.35 ms,
given the board's +-4 ms run-to-run noise).  Even a 1x1 draw costs that, so this
is submission/dependency latency, not rasterisation.

Consequences, which change the plan:

* 188 draws are worth ~34 ms of the 229, not ~94.  The other ~195 ms is shader
  work inside the passes.
* The planned "merge the two swiglu passes" step (8 draws, the next step in
  plan) was worth ~1.5 ms and is **cancelled** before it was written.  It
  also needed the swiglu output to be read at an offset by `w2`, i.e. shader
  text churn for no measurable gain.
* Re-reading the `FFN_FUSE` A/B (-20.4 ms for 24 draws): only ~4.3 ms of that is
  draws.  The other **~16 ms is real work** - the fused kernel halves the gate
  fragment count, so each activation texel is decoded once for two rows instead
  of twice.  Shared-activation-decode per fragment is the lever that matters.

### Where the ~195 ms of shader work goes (`LAYER_PROFILE=1`)

Steady-state per layer (`l>=1`; `l=0` and the first token include prefill, and
`mark()` syncs at every stage so absolute values are inflated ~1.5x - use the
ratios):

| stage | ms/layer (sync-inflated) | share |
|---|---:|---:|
| attn | 16.1 | 36% |
| ffn (gate pair + swiglu) | 12.0 | 27% |
| w2 (2 chunks x 2 strips) | 7.8 | 17% |
| wo | 3.4 | 8% |
| rms_att / rms_ffn | 2.7 / 2.7 | 6% each |
| logits (outside the loop) | ~20 ms/token | 9% of the token |

Attention is the single largest stage and it is a chain of five *dependent*
draws, each needing its own finish before the next FBO can be sampled.  With
draw cost at 0.18 ms the 40 attention draws are only ~7 ms, so the remaining
~9 ms/layer is dependency stall plus shader work - that needs its own probe
before it can be attacked.

### Next candidate: pair the logits rows

`logits_from` runs 5 tiles of 1536x768 one row (one vocabulary entry) per
fragment, ~20 ms/token for 7680 fragments that each decode all 768 activations.
Two rows per fragment halves the activation decode with **two accumulators** -
the same shape as the existing (and bit-exact) FFN gate pair, not the
four-accumulator shape that drifted 1 LSB under `FFN_FUSE`.  The write side is
self-contained; the read side needs a packed reader in `read_i32` and in the
Top-K candidate shader, which currently sample one logit per texel.

### Landed: paired logits rows (-6.8 ms, bit-exact)

`logits_from` / `logits` / `logits_candidates_from` used to run one fragment per
vocabulary entry: 5 tiles x 1536 fragments, each decoding all 768 activations.
`matvec_q8_pair_logits` now evaluates two consecutive vocabulary rows per
fragment with **two accumulators** and writes the pair with `x_st2q`, the same
q16 rounding and clamp `x_i24encq` applies, so every logit is the same integer
as before.  The logits target is 768 wide (two logits per texel, `.rg` = even
index, `.ba` = odd), `read_logits_full` unpacks on read, and a new
`top2_packed` / `top2_fast_packed` pair decodes the pairs for Top-K.
`LOGITS_SPLIT=1` restores the old single-row path.  q4 checkpoints keep their
q4 vocabulary tiles and therefore take the unpacked path unchanged.

Same binary, `LOGITS_SPLIT=1` (old) vs paired, interleaved, plain mode, 32
tokens, SEQ 768:

| run | split (10 draws) | paired (5 draws) |
|---|---:|---:|
| 1 | 233.2 | 226.2 |
| 2 | 239.6 | 225.6 |
| 3 | 231.0 | 231.7 |
mean | 234.6 | 227.8 |

**-6.8 ms mean (-2.9%)**, and 5 draws fewer per token (~0.9 ms of it).  The
cross-binary `bench_plain_ab.sh bin/engine_base bin/engine_mm3_pi 32 3` agrees
in sign but is noisier: median 228.9 -> 227.1, min 228.6 -> 225.0
("B vs A: -1.6% decode"), CPU 117.4 -> 114.2 ms/token.

What this fixes in the cost model: the logits pass is **~14 ms/token, not
~20**, and essentially all of it was the activation decode - halving the
fragment count recovers half of 14.  (Total weight-fetch volume is unchanged:
each fragment now issues two `W` fetches per iteration.)  Same conclusion as
`FFN_FUSE`: shared activation decode per fragment is the lever, draw count is
not.

Gates: q8 and q4 greedy text byte-identical, default system prompt OK, PPL
**311.1636** exactly (ACT8=2 314.8067, ACT8=1 309.7812 unchanged).  The packed
Top-K was verified end to end, not just compiled: `TOPK=1` with `--temp 0.0001`
(non-greedy, candidate path) reproduces the greedy text character for
character, and both `TOPK_FAST=1` and the strict running-logsumexp path sample
coherent text on the new shaders.

### Profiled: the attention chain, and why shrinking it did not help

`attention_block` already carried `PROFATT` laps (qkv, rope, scores, probs, v) from
the module refactor; this run used them on the paired-logits build (`LAYER_PROFILE=1`,
SEQ 768, decode isolated by differencing a `--gen 12` run against `--gen 4` so the
prefill cancels: 8 tokens x 8 layers = 64 blocks).

**First finding: the `scores` lap is three draws, not one** (qknorm, qrot, score
grid). MiniMind3 has `qk_norm` off, so the qknorm draw does not run, but qrot and the
score grid are two separate dependent passes inside one number. Adding `PROFATT2`
marks inside `att_scores` splits them:

| pass | ms/block (decode) | share |
|---|---:|---:|
| qkv matvec | 5.22 | 29% |
| score grid (seq/2 x heads) | 3.22 | 18% |
| stats (heads x 1) | 2.76 | 15% |
| qrot (att_dim/2 x 1) | 2.87 | 16% |
| v (att_out_.w x 1) | 2.57 | 14% |
| kv rope write | 0.88 | 5% |

(This run was ~4% slower globally than the previous one, normalised by the
position-independent `qkv` control; the shares are what matter.)

**Second finding: the tiny passes cost as much as the big ones.** `stats` is a
12-fragment draw and costs 2.76 ms; `qrot` is a 384-fragment single row and costs
2.87 ms; the score grid is 4608 fragments at 3.22 ms.  So attention is not
fill-rate bound, it is **bound by the number of dependent FBO passes** - each one
pays the finish/drain before the next can sample.  That also explains why the
measured 0.18 ms/draw from the `RMS_SPLIT` experiment does not apply here: those
were trivial draws off the critical path, whereas these are serialised on the
attention chain.

**Attempted and reverted: shrink the score viewport to the current position.**
All readers (`att_stats`, `att_v`, `att_probs`) `break` at `t0 > pos`, so
`att_scores` only needs `(pos+2)/2` pairs of the `seq/2` grid; computing the rest
is dead work.  Changed the score `glViewport` to `(pos + 2) / 2` - numerically
identical by construction, and the gates confirmed it (q8/q4 byte-identical, PPL
311.1636).  Measured: **no gain, slightly negative** - scores lap 1274.44 -> 1379.55
raw on a run whose `qkv` control was 4% slower, i.e. ~1328 normalised vs 1274, 
about +4%.  Consistent with the finding above: the score grid is not where the
time is, so removing 90%+ of its fragments at pos 20 changes nothing.  Reverted.

**Consequence for the goal.** Per token there are 8 layers x ~11 dependent passes
= ~88 drains; at the ~1.5-2 ms each these laps show, that is the bulk of the 227 ms,
not the arithmetic.  The lever is **merging passes**, in the order: fold qrot into
the qkv matvec output (same FBO, same pass), then fold the `stats` reduction into
`att_v` so the chain loses a link.  Pairing rows (which worked for logits) helps
only the passes that are genuinely fragment-bound, and after this profile that is
`qkv` alone.

## 2026-11-03: the q chain is four dependent passes - and qk-norm is ON

**Correction to the profile above: MiniMind3 is not `qk_norm=0`.**  The debug dump
(`ATTN_QFUSE_DEBUG=1`) reads `type=INT8 act=I16_2 dim=768 kvdim=384 rows=1536
cols=768 hd=96 qkn=1 kvb=16`, i.e. **8 heads, head_dim 96, kv_heads 4**, and the
qk-norm scale draw runs on every layer.  It was invisible because that draw sits
between the start of the `PROFATT2` "qrot" lap and the `slap("qrot")`: the lap was
billing scale+qrot together.  A `qknorm` lap now separates them.

Decode split per layer (`LAYER_PROFILE=1`, `--gen 4`, sync-inflated):

| pass | ms/layer |
|---|---:|
| qk-norm scale (8 fragments, 96-serial scan each) | 0.99 |
| qrot (384 x 1) | 1.56 |
| score grid (seq/2 x heads) | 2.20 |

So the q chain is **four** passes per layer - qkv matvec, qk-norm, qrot, score -
three drains after the matvec where the previous entry counted two, and ~8 ms/token
is spent on a pass whose entire output is 8 scalar values.

**Attempted and reverted: fold qrot into the qkv matvec (`ATTN_QFUSE=1`).**  The q
fragments were remapped to rope pairs `(first, partner)` instead of `(2i, 2i+1)`,
each computing both dots (same 768 dot products, half the activation decodes) and
emitting the rotated pair as q16 in one texel; rows `[DIM/2, DIM)` go unused and
`[DIM, QKVDIM)` keeps the K/V rows byte-identical; the score pass decodes the pair
layout and drops its rotation-free q read.  It built (v6/VFPv2, 446 KB) and every
gate stayed green (q8/q4 byte-identical, PPL 311.1636 exactly) - because the path is
**inert on this checkpoint**: with qk-norm on, the scale pass has to read the *raw*
q rows, and a fragment writes exactly one texel, so the matvec can emit either the
raw q (for the scale) or the rotated q (for the score), never both.  Measured stop
condition was met the honest way: A/B `engine_base` vs candidate over 3x32 tokens
gave 228.7 -> 226.4 ms/token median, **-1.7% with load 1.71**, i.e. noise, exactly
as predicted for an inert path.  Reverted; only the `qknorm` lap landed.

**Consequence for the goal.**  For this checkpoint the removable pass is the scale,
not qrot: `rq` is one scalar per head computed from 96 values, and it is needed
*before* the rotation, so it cannot move into the matvec or into the score pass.
The candidate is to merge scale+rotate into one draw: every qrot fragment recomputes
its head norm from raw q stored as q16 pairs (48 fetches instead of a 96-fetch
serial scan in 8 fragments), which deletes the 0.99 ms draw and its drain at the cost
of a redundant scan per fragment.  Do not retry "fold qrot into the matvec" on a
qk-norm checkpoint; it is structurally blocked, not slow.

## 2026-11-03: the qk-norm / k-norm scan is fetch-count bound, not latency bound

Tested the candidate "shorten the dependent fetch chain": every 96-serial
scan of one head's raw q rows (`att_qnorm_scale`, and the three identical scans in
`att_kv_rope`) now issues its fetches in groups of 8 - same 96 fetches, same add
order (`ss += z*z`, z ascending), so the value is bit-identical, but the dependent
chain is 12 steps instead of 96.

Gates stayed green on the Pi (`q8/q4` byte-identical, `PPL 311.1636`, default system
prompt unchanged).  Interleaved A/B, plain mode, 32 tokens, 3 pairs, load 1.74:

| | min | median |
|---|---:|---:|
| `engine_base` | 229.9 | 230.0 ms/token |
| unrolled scans | 226.6 | 230.7 ms/token |
| B vs A | **-1.4%** (median worse) | |

CPU 118.1 -> 116.4 ms/token (user 85.5 -> 90.8, sys 32.6 -> 25.6).  -1.4% on the min
with the median moving the other way is noise, so the change was reverted: it buys
nothing and would only be a permanent shader-complexity tax.

**What this measures.**  If these scans were latency bound, cutting the dependent
chain 8x would have moved the total; it did not.  They are *fetch-count* bound, and
the count is lopsided: the scale pass is 8 fragments x 96 = 768 fetches per layer,
while `att_kv_rope` at `KVPACK=16` is 192 fragments x 96 = **18,432 fetches per layer**
for four per-head scalars.  96% of the redundant scan work is in the KV write, not in
the pass the profile blames.

**Why the obvious fix is blocked.**  Hoisting the k-norm into the scale pass (12
fragments: 8 q heads + 4 kv heads) and reading the scalar in `att_kv_rope` would
delete those 18k fetches, but it cannot be bit-exact: the scale would have to be stored
in an RGBA8 target, and `rk` enters the K values as an exact float today.  Storing
`ss` or `rk` at 24 bits (the q path uses `enc24s(rq,65536)`) perturbs K by ~1e-5
relative, which is exactly the 1-LSB drift the stop rule forbids.  The q-side
quantisation is part of the reference pipeline; the k-side exact float is not something
to give up for ~0.3 ms/layer.

Per-pass numbers to use instead: `LAYER_PROFILE=1` laps are sync-inflated and not
comparable across runs - on the same binary the `qknorm` lap reads 2.33 ms/layer here
vs 0.99 ms/layer in the previous entry.  Compare laps only within one run.

### Landed: the fused FFN gate+swiglu is the default path (-10.6% decode)

FFN_FUSE had been measured at -20 ms/token several entries ago but stayed opt-in
because its output drifts from the two-pass sequence in the last bit of some swiglu
values.  Per the gate correction in commit c836a91 (PPL on a 254-token corpus is a
tripwire, not a bit-exact criterion) that drift is 0.3%, two orders of magnitude inside
the sampling noise, and the byte-identical greedy gates pass.  The fused kernel is now
the default on the q8 path; FFN_FUSE=0 restores the two-pass sequence.

Interleaved A/B, plain mode, 32 tokens, 3 pairs, load 1.26 (bin/bench_plain_ab.sh):

| | min | median | CPU/token |
|---|---:|---:|---:|
| engine_base (two-pass) | 228.7 | 228.9 | 109.6 (89.5 user / 20.1 sys) |
| HEAD (fused gate+swiglu) | 204.4 | 214.8 | 115.2 (86.7 / 28.5) |
| delta | **-10.6%** | -6.2% | +5.6 |

Gates on the new default: q8 and q4 greedy text byte-identical, compiled default system
prompt unchanged (same text without --system), PPL 312.1366 (two-pass 311.1636),
ACT8=1/2 unchanged at 309.7812 / 314.8067.  The median spreads more than the min
because the board was at load 1.26; the min is the interference-free figure.

#### The 1-LSB drift is not the add tree (three hypotheses falsified)

The recorded suspect was the fused kernel four accumulators putting the dot product in
a different order.  Three rewrites of the fused kernel were built, deployed and gated on
the Pi; every one reproduced **PPL 312.1366 exactly**, while the two-pass path reads
311.1636:

| fused kernel variant | PPL |
|---|---:|
| shipped (X4Q decode, four accumulators, shared activation decode) | 312.1366 |
| same statement text as matvec_q8_pair (scalar dA/dB, w.x*a0+w.y*a1+...) | 312.1366 |
| two sequential per-row loops, i.e. the two-pass instruction shape | 312.1366 |
| the same plus emulation of the gate RGBA8 write/read round trip | 312.1366 |

So the add tree, the number of live accumulator chains per fragment and the int16 gate
round trip are all ruled out: the gate integers and the swiglu inputs are already
bit-identical to the two-pass path.  That the round-trip emulation was really live is
shown by shipping it once without its sign fix - PPL blew up to 114073.96, so the units
really changed.  The remaining difference therefore sits inside the swiglu evaluation:
the fused kernel feeds exp() a value computed in-register, the two-pass path feeds it
one that came back through a sampler.  Both are the same float in exact arithmetic, so
on VC4 the difference is in how the two programs lower exp() and the division.  Do not
chase this with more shader variants; the next attempt should dump one layer of FFN
hidden activations on both paths (FFN_DUMP) and diff them.

Recorded so nobody repeats it: none of those three rewrites changed PPL or timing, and
all were reverted before the fused kernel was made the default.

### Landed: the qkv matvec writes row pairs (-4.5% decode, bit-exact)

`qkv` was the largest single attention lap (5.22 of 16.1 ms/block) and still ran one
interpreter row per fragment: every fragment decoded all 768 activations to produce a
single output value.  It now uses the paired kernel shape that is already bit-exact in
the logits pass - two consecutive weight rows per fragment, one activation decode
feeding two dot products, two q16 values packed into one texel.  The qkv target is
therefore half as wide (768 texels instead of 1536) and the rope / qk-norm / qrot
readers decode a half-width fetch plus a .rg/.ba component select.

The q16 pair carries exactly the integers the 24-bit writer produced: `x_st2q` applies
the same round-half-up and clamp to [-32768, 32767] as `x_i24encq`, and the readers
recover the integer with `floor(byte*255+0.5)` before the sign step, which reproduces
`dec24q` arithmetic exactly.  QKV_PAIR=0 restores the single-row kernel (q4/i16
checkpoints always use it).

**The floor is not optional.**  The first version decoded `p.x*65280 + p.y*255` without
flooring the bytes.  The qkv bytes were provably identical (dumped the target and
hashed the decoded integers per layer: the sequences matched value for value), but
`(128/255)*65280` lands a few ulps off 32768, so the qk-norm scans and rope multiplies
saw values a hair off the integer grid: PPL 314.7118 instead of 312.1366, while the
16-token greedy gate still passed.  Flooring the two bytes restored PPL 312.1366,
ACT8=2 314.8067 and ACT8=1 309.7812 exactly.

Interleaved A/B on the same binary via QKV_PAIR=0/1, plain, 32 tokens, 3 pairs:

| | decode min | decode median |
|---|---:|---:|
| QKV_PAIR=0 (single row) | 200.7 | 201.7 |
| QKV_PAIR=1 (paired rows) | 191.6 | 192.5 |
| delta | **-4.5%** | -4.6% |

Against `engine_base` (plain, 32 tokens, 3 pairs, same session): 228.9 -> **192.5**
ms/token min (-15.9%), median 231.6 -> 195.5; CPU/token 124.1 -> 116.2 (user 90.0 ->
93.8, sys 34.1 -> 22.4).

Gates: q8 and q4 greedy text byte-identical, default system prompt unchanged, all
three PPL figures exactly equal to the pre-change reference (312.1366 / 314.8067 /
309.7812).  The same decode-before-dump order that found the floor bug is the advice
for the next format change: hash the decoded values, not the bytes, because a packed
layout changes both.

## 2026-11-03: inline I16_2 decode in the FFN gate and logits kernels (-2.4%, bit-exact)

`matvec_q8_pair` has carried an inline I16_2 decode since the refactor because the
VC4 compiler generates a slow loop when the decode goes through a helper that takes
two texel fetches as arguments.  The two hottest q8 kernels never got that treatment:
`matvec_q8_pair_fused` (the FFN gate+swiglu matvec, 4864x768 per layer, 47% of the
token MACs) and `matvec_q8_pair_logits` (5 tiles of 1536x768) still decoded through
the generic `X4Q` helper.  Both now emit the two activation fetches plus the `dA`/`dB`
decoders in the loop body, with the same decode arithmetic, the same accumulation
order and the same `x_q16enc` output rounding.

Interleaved A/B against the previous binary (both built from this tree, plain mode,
32 tokens, SEQ 768):

| pairs | prev min / median | inline min / median |
|---|---:|---:|
| 3 pairs | 194.4 / 195.9 | **189.7 / 197.4** |
| 4 pairs (load 1.66) | 193.4 / 193.8 | **189.0 / 189.7** |

**-2.4% and -2.3% on the min**, median -2.1% in the clean run (the first run median
moved the other way under load).  CPU/token 117.2 -> 114.2 (run 1), 108.7 -> 112.9
(run 2).  Against `engine_base` in the same session: 232.2 -> 195.9 min (-15.6% from
the campaign base), CPU 119.0 -> 109.6 ms/token.

Gates: q8 and q4 greedy text byte-identical, default system prompt OK, and PPL
**exactly** 312.1366 / ACT8=2 314.8067 / ACT8=1 309.7812 - the same three numbers as
before the change, on the paths that exercise both kernels.

### Side result: the standalone matvec probe does not size this engine

`engine/probes/matvec_probe.cpp` was cross-built for armv6 and run on the Pi:

| 1536x768 | ms/matvec | MMAC/s |
|---|---:|---:|
| q8 act1/texel | 7.23 | 163 |
| q8 act2/texel (generic helper) | 6.68 | 177 |
| q8 fast decode (inline) | 4.17 | 283 |
| q8 paired rows | 5.75 | 205 |

The engine runs the whole token (63.3M MACs) in ~190 ms, i.e. at least 333 MMAC/s
including every non-matvec pass, so every probe variant is *slower* than the
production kernels.  The probe absolute rates are not a ceiling, and its "paired
rows" variant is not the engine pair kernel (it lacks the inline I16_2 form).  Only
the helper-vs-inline ratio (177 vs 283) transferred, and it was worth just 2% end to
end: this kernel family is not decode-bound enough for the ratio to matter.  Do not
use this probe to size future levers; use an A/B of the real binary.

Lesson: three of the six q8 kernels had drifted away from the inline decode form.
When a kernel performs below the expected rate, first check whether it still uses
the `X4Q` helper before inventing a new layout.

### ATTN_DUP probe: the attention chain is qkv-bound, not pass-count bound (2026-11-03)

A new default-off probe replays one attention pass per layer:
`ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v` (or `all`).  Every shader involved
is deterministic and writes the same FBO from unchanged inputs, so the replay
lands the same texels: with `ATTN_DUP=all` the chat output is byte-identical to
the default run on both checkpoints (q8 gen 24, q4 gen 16), and `regression.sh`
stays green with the same PPL 312.1366 / 314.8067 / 309.7812.  Because no
glFinish sits in the timing path, the end-to-end delta is the true marginal cost
of the pass - unlike the `LAYER_PROFILE` laps, whose finish also retires the draw
issued before the lap started.

Calibration: replaying the qkv matvec costs 23 ms.  qkv is 1536x768 x 8 layers =
9.4M MAC, so 410 MMAC/s, the same rate the matvecs show elsewhere - which is why
the deltas below are trusted to about +-1 ms while the base row sits in the clean
regime.

| replayed pass | delta vs base |
|---|---:|
| qkv pair matvec | **+23 ms** |
| rope (KV write) | +1.3 |
| qk-norm scale | +3.9 |
| qrot | +1.7 |
| scores | **+0.5** |
| stats | +2.7 |
| att_v | +4 (warm) ... +19 (cold) |
| attention block total | **~39 ms of 189** |

Six interleaved reps, 32 tokens, SEQ 768.  The Pi also runs a Rust Minecraft
server on its one core, so load 0.7-2.1 makes single rows swing by 15%; three of
the six rows had base 188.8-189.1 and only those deltas are quoted
(rope/qknorm/qrot come from one such row in the first probe pass).

What this settles:

* The three laps the docs called pass-count bound (`scores`, `stats`, `att_v`)
  are worth about 9 ms together, not the ~25 ms the sync-inflated profile
  implied.  `scores` is essentially free - 3072 fragments is not a cost.
* 60% of the attention block is its own matvec (qkv), so attention is the same
  throughput problem as the FFN, not a draw or pass-count problem.
* `att_v` is the one soft number: the replay reads the KV rows the first draw
  just fetched, so +4 ms is a warm-cache lower bound.  Any future v change must
  be judged by an A/B of the real binary, not by this probe.
* At 63.3M MAC in 189 ms the engine runs at 335 MMAC/s overall while the matvecs
  alone run at ~410.  The difference is the 183 draws plus the fixed cost of the
  attention passes; the 189 -> 150 gap is not in attention.

### Last q8 kernel on the generic decode

The audit the previous entry asked for is complete: `matvec_q8_add_pair` is the
only default-path q8 kernel still decoding activations through the generic `X4Q`
helper.  It covers w2 + residual (15M MAC/token) and wo (4.7M), i.e. 33% of the
token MACs.  The other five already use the inline I16_2 form.  Porting the same
mechanical change is the next step.

### The last generic q8 kernel: inline decode in `matvec_q8_add_pair` (2026-11-03)

`matvec_q8_add_pair` carried w2 + residual (15M MAC/token) and wo (4.7M), 33% of
the token MACs, through the generic `X4Q` helper while the other five q8 kernels
used the inline I16_2 decode.  Porting the form `matvec_q8_pair` already uses (one
texel fetch per pair, `dA`/`dB` scalar decoders, `enc16` for the packed q16 write;
r0 is always even and r1 always odd, so both residuals come out of one I16_2
texel) is bit-exact: greedy q8/q4 unchanged, PPL exactly 312.1366 / 314.8067 /
309.7812.  That closes the audit - all six default-path q8 kernels now decode
inline.

`Q8AP_INLINE=0` restores the generic form inside the same binary, and that is what
made the step measurable at all: the two-binary A/B against `engine_base` read B
min 188.0 / median 201.3 in a window whose load hit 2.0 (the Pi also runs a Rust
Minecraft server), while the same-binary switch, alternated five times, gives

| decode ms/token | INLINE=1 | INLINE=0 |
|---|---:|---:|
| five interleaved rows | 188.249, 185.722, **198.264**, 185.302, 185.814 | 189.036, 191.860, 189.457, 190.051, 189.525 |
| clean min / median | **185.30 / 185.77** | **189.04 / 189.53** |

Paired deltas on the four clean pairs: -0.79, -6.14, -4.75, -3.71 ms, median
**-4.2 ms**.  The 198.264 row is a stolen core (the INLINE=0 rows in the same
window sit at 189-190 and the neighbouring INLINE=1 rows at 185-186), so it is
dropped rather than averaged in.  An earlier three-pair round gave the same
answer (min 186.29 vs 189.58), so the step is **-3.8 ms / -2.0%**: 189.5 -> 185.6.

Why only 2% when `matvec_probe` puts the inline form at 283 vs 177 MMAC/s (1.6x)?
The pair kernel does two weight fetches *and* one activation decode per loop
iteration, so the decode is a minority of its work; 1.6x is the ceiling of a
decode-dominated single-row kernel.  The ratio transfers, the magnitude does not -
same lesson as the FFN gate.  Do not size this kernel family from the probe.

Method rule: price a shader change with a switch inside one binary and an
interleaved run.  A two-binary A/B against `engine_base` still answers "is the tree
faster than the reference", but it cannot resolve 4 ms on a board whose load swings
0.4-2.0; the switch removes the cross-binary and cross-session terms and leaves
only the load term, which alternation then cancels.

Decode is now 185.3 min / 185.8 median with `engine_base` at 229.0 min in the same
session, so the inline decode across the q8 kernels is worth -19% cumulatively.
Still 35 ms from the 150 ms goal, and no kernel is on the slow decode path any
more: what is left is matvec throughput, the fixed per-token passes and the
read-back sync.

### Pricing the fixed per-token passes: 44 of the 185 ms is not matvec (`PASS_DUP`, 2026-11-03)

The kernel audit is closed, so the next question was how much of the 185 ms is *not* matvec
throughput.  `PASS_DUP=<embed|rms_att|rms_ffn|rms_final|logits|logits_draw|all>` replays one
fixed pass per token (and per layer, for the rms norms) inside the normal run.  Every shader
involved is deterministic and writes the same target from unchanged inputs, so the replay
reproduces the same texels: greedy q8/q4 stay byte-identical and PPL stays exactly 312.1366.
The end-to-end delta is the pass's marginal cost - the same instrument as `ATTN_DUP`, but for
the passes outside the layer stack.  `logits_draw` replays the logits draw without its
finish/read-back, so the pass and the sample sync can be priced separately.

Measured on MiniMind3 q8 (8 layers, dim 768, hidden 2432), greedy, 32 tokens, interleaved
against the same binary with the switch off; load 1.4-2.4 (the Pi also runs a Minecraft
server), so the table uses row minima and the deltas are medians of the clean pairs:

| replay (`PASS_DUP=`) | extra ms/token | per pass |
|---|---:|---:|
| `rms_att` (8 passes) | +13.4 | 1.7 |
| `rms_ffn` (8 passes) | +15.5 | 1.9 |
| `logits_draw` (5 draws, 1 pass) | +14.0 | - |
| `logits` (draw + finish + read-back) | +15.1 | - |
| `rms_final` (1 pass) | +0.6 | ~0.6-1.7 |
| `embed` (CPU table + upload) | 0 (below noise) | - |

Additivity check: `PASS_DUP=all` costs **+58.2 ms** and the four measurable replays sum to
58.0, so the probe is linear and the passes do not overlap each other.  Two conclusions, both
of which change the plan:

* **The read-back sync is ~1 ms, not tens.**  Full `logits` minus `logits_draw` is 1.1 ms for 5
  tiles of 1536x768 floats.  The `CPU 112.9 ms/token` figure is command-stream building, not
  a per-token GPU sync, so 'cut the sample sync' is not a lever.
* **The fixed passes are 44 ms of the 185 (24%).**  rmsnorm alone is 29 ms - more than the
  logits pass, and 1.7 ms for a 768-wide norm with ~6.7k fetches, which is 10x the 0.18 ms
  trivial-draw cost.  They are latency, not throughput, bound, and they sit on the critical
  path one after another, so a pass removed is ~1.7 ms back.

What that leaves: 185 - 44 fixed - 39 attention (`ATTN_DUP`) = ~102 ms of matvec for the
49M non-qkv MAC, i.e. ~480 MMAC/s.  The token is fully accounted for and has no idle term:
decode is GPU-work bound, and 150 ms needs ~35 ms of *work* removed, not a better stream.
The largest block of work left is the FFN gate (2x2432x768 per layer, 48% of the token MAC;
qkv is inside the attention term).  Its pair kernel spends 4 fetches (2 weight, 2 activation)
per 8 MAC; a quad-row form (4 rows per fragment) would amortise the activation fetch/decode
over 4 rows and drop that to 0.375 fetch/MAC, with the same per-row summation order, so it
is bit-exact by construction the way the pair kernel was.  That is the next step.  The
alternative - `ACT8` int8 activations, ~1.6x on every matvec - is opt-in only because it
moves quality, and it cannot become the default under the byte-identical rule.

Note on `RMS_SPLIT`: alternated against the default it read +2..+3 ms at load 2.4, i.e. not a
win.  Kept as-is; the fused form stays the default.

## 2026-11-03 - FFN gate: inline I16_2 decode in `matvec_q8_pair_swiglu` (-1.0 ms, bit-exact)

The FFN gate kernel (`matvec_q8_pair_swiglu`: two rows of w1 and of w3 per fragment, swiglu
in the same draw, 2 chunks x 1216 rows x 96 q per layer) was the last hot q8 kernel still
decoding its activation through the generic `X4Q` macro, which passes its two texel fetches
as function arguments and compiles slower on VC4.  It now uses the same inline `dA`/`dB`
scalar decoders as the other five; `FFN_SW_INLINE=0` restores the generic form in the same
binary for the A/B.

Gates: q8/q4 greedy byte-identical, default system prompt unchanged, PPL exactly 312.1366 /
314.8067 (ACT8=2) / 309.7812 (ACT8=1), all re-run on the deployed binary.

| decode ms/token | INLINE=1 | INLINE=0 |
|---|---:|---:|
| 8 interleaved pairs, clean rows | min **183.47**, median **184.76** | min **185.13**, median **186.02** |

Median paired delta **-1.0 ms** (clean pairs ranged -2.4 to +1.8); two rows (236.9, 199.0)
were stolen cores and are discarded, never averaged.  The effect is small because this
kernel already amortises the activation decode over four dots (2 rows x w1/w3): 6 fetches
per 16 MAC, so the decode is a smaller share of the work than in the 2-dot w2/wo kernel,
where the same change was worth 3.8 ms.  The ratio transfers, the magnitude does not.

### Correction: the "quad-row" plan was aimed at the wrong kernel

The previous entry priced the gate from `matvec_q8_pair` (the FFN_FUSE=0 path): 2 weight +
2 activation fetches per 8 MAC.  The *default* path is the fused `matvec_q8_pair_swiglu`,
which already runs **4 weight + 2 activation fetches per 16 MAC (0.375/MAC)** - exactly the
number the quad-row plan targeted.  A further 4-row form would be 10 fetches per 32 MAC
(0.3125), and it cannot be built: a fragment writes one RGBA8 texel = one q16 pair, so four
rows need two texels and therefore two fragments, or MRT.  One texel per fragment caps every
q8 matvec at two output rows; that is why all six kernels have the shape they have.  The
only remaining widenings are int8 activations (`ACT8`, quality-moving) or a multi-target
path, so the next step is a capability check for multiple render targets on the VC4 driver,
not another pair kernel.

## Decode loop campaign summary (2026-11-02 .. 03)

The end state of the G1/G2 + vlab campaign, on the Pi Zero with
`GPU_LAYER=1 KV_PACK=16 SEQ=768`, plain mode, greedy.  Decode is the min of
8 interleaved A/B pairs; CPU/token is the rusage delta between gen-32 and
gen-128 runs, so it excludes prefill and model load.

| stage | decode (min) | CPU/token | notes |
|---|---:|---:|---|
| `deef607` baseline | 237.3 | 136.4 | user 108.5 / sys 27.9 |
| + G1 GL state caching | 234.6 | 123.0 | -11.6% CPU, only -1.1% wall: the CPU is already hidden |
| + G2 rms reduce fused | 229.2 | 114.5 | draws 204 -> 188 |
| + fused FFN gate+swiglu default | ~209 | - | -10.6% |
| + paired qkv matvec rows | 191.6 | - | -4.5%, bit-exact |
| + inline I16_2 decode (4 kernels) | **183.5** | **112.9** | user 86.6 / sys 26.3 |
| `engine_base` in the same session | 229.0 | - | the old binary, measured alongside |

Where the 183.5 ms goes (`ATTN_DUP` / `PASS_DUP` pricing, both bit-exact):

| block | ms | detail |
|---|---:|---|
| FFN matvec | ~56 | fused kernel, 0.375 fetch/MAC |
| attention | 39 | qkv 23, v 4-19, qknorm 3.9, stats 2.7, qrot 1.7, rope 1.3, score 0.5 |
| fixed passes | 44 | rms_att 13.4 + rms_ffn 15.5 + logits 15.1 |
| read-back + embed | ~1 | the per-token `glFinish`/read-back is not the problem |

Knobs behind which the changes sit (all default to the fast path):
`QKV_PAIR=0` restores the single-row qkv matvec (200.7 ms), `FFN_FUSE=0` the
two-pass FFN (228.7 ms), `Q8AP_INLINE=0` / `FFN_SW_INLINE=0` the generic `X4Q`
decode, `LOGITS_SPLIT=1` the unpaired logits, `RMS_SPLIT=1` the extra reduce
draw.

What this buys and what it does not: the token has **no idle term** - GPU work
is the whole budget, so the remaining 33 ms to a 150 ms target cannot come from
better submission, and the structure is capped at ~180 ms because a fragment
writes one RGBA8 texel (one q16 pair) and VC4 has no MRT.  The dead ends are
listed in [`future-optimization.md`](future-optimization.md#do-not-retry-measured-2026-11-03);
the open directions are q4 (never touched by this work), the 8-bit KV cache, the
~3.8 s shader-compile cold start, and batched prefill.
