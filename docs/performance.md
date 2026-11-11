# Performance

Measurements are from the Pi Zero and the local Xe/GLES2 machine. Treat them
as regression baselines, not portable benchmarks.

This file is append-only history.  Commands inside the entries are reproduced
as they were run, so historical ones may use flags that no longer exist
(`--template`, `TOKENIZER=`, `ROPE_THETA=`, `ROPE_HALF=`, legacy `.q8`/`.q4`
paths); the current invocation is in `AGENTS.md` and `engine/README.md`.

## Current model matrix (2026-11-02 run)

Pi Zero, `GPU_LAYER=1 KV_PACK=16`, greedy, 16 generated tokens.  The Pi also
runs a Minecraft server, so use the ranges rather than individual decimals.

Every row is loaded from the `.mmcf` container named here; the legacy file of
the same name (without the suffix) is the packer input.  These are the
**2026-11-02** numbers: the decode campaign, the fused-FFN default and the
two-position prefill all landed afterwards, so the q8/q4 pair to quote today is
252 / 388 ms/token (see "q4 parity, step 1" below, which explains the different
clock state).  Rows keep their date so the cost model stays auditable.

| model | container | size | prefill | decode | status |
|---|---|---:|---:|---:|---|
| MiniMind3 q8 | `minimind3.q8.mmcf` | 66 MB | 20.5 s | **238 ms/token** | default Chinese chat |
| MiniMind3 q4 | `minimind3.q4.mmcf` | 45 MB | 33.0 s | **381 ms/token** | memory mode; slightly lower quality |
| MiniMind2 i16 | `minimind2-small.i16.mmcf` | 58 MB | 9.2 s | 226 ms/token | reference; weak instruction following |
| MiniMind2 q8 | `minimind2-small.q8.mmcf` | 29 MB | 8.4 s | 140–190 ms/token | fastest MiniMind2; rope comes from the container |
| MiniMind2 q4 | `minimind2-small.q4.mmcf` | 21 MB | 9.7 s | 193–212 ms/token | memory mode; identity answers unreliable |
| TinyChat15M q8 | `TinyChat15M.q8.mmcf` | 24 MB | 4.8 s | 139–171 ms/token | loads and runs, but output is not coherent with the repo tokenizer |
| stories15M i16 | `stories15M.i16.mmcf` | 49 MB | 3.1 s | 167 ms/token | TinyStories completion; coherent |
| stories42M q8 | `stories42M.q8.mmcf` | 58 MB | 3.8 s | 190 ms/token | TinyStories completion; coherent |
| stories15M q8 | `stories15M.q8.mmcf` | 24 MB | 3.1 s | 148 ms/token | **gibberish**; use the i16 container instead |
| stories260K | *(no container yet)* `stories260K.bin` | 1 MB | 5.4 s | 85 ms/token | 5-layer floor probe; pack it first |

The Pi store now has TinyStories 15M/42M/260K checkpoints.  They are base
completion models, not chat models, so test them with a plain prompt such as
`Once upon a time` and no `--chat`.  `stories42M.i16` is too large for the
current CMA budget; q8 is the practical 42M path.  TinyChat15M should be
treated as a format/probe model, not as a usable chat model until its
tokenizer/prompt pairing is fixed.

The 260K checkpoint is a useful **floor probe**, not a product model: its
5-layer/64-dim body reaches 85 ms/token, but prefill is still 5.4 s because the
attention passes walk the full 512-position grid and the shader setup is fixed.
This is below the ~150 ms floor of the coherent 15M/42M models, but it is not a
quality replacement for them.  It is the only row here still in the legacy
llama2.c layout — pack it with `mmcf_pack.py --rope-plain` (and `tok512.bin`)
before the engine can load it.

The `LEGACY_Q8=1` / "legacy fp32 loads as I16" note is historical: those flags
went with the legacy readers in `fcc4095`, and the precision is now decided by
which container you pack.

### Default system prompt

The CLI and server use one neutral English prompt for every chat template:
`You are a helpful assistant.`  It replaces the old model-specific Chinese
MiniMind3 prompt and the server's Chinese default.  Tested behavior:

| model | identity response | note |
|---|---|---|
| MiniMind3 q8 (Pi) | “我是MiniMind，一个由jingyaogong……” | re-baselined 2026-11-09 with the integer q16 decode; see the 2026-11-09 section |
| MiniMind3 q8 (Xe) | “我是一个AI助手，我的存在是为了提供帮助和信息。” | the two GPUs round the fused FFN differently |
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

### Why the q8 identity string changed on 2026-11-03

Making the fused FFN gate+swiglu the default (`b7dce0d`) reassociates the last
bits of the swiglu output, and on VC4 that is enough to flip the first answer:
the pinned gate went from `我是一个AI助手，我的存在是为了提供帮助和信息。` to
`我是一个AI助手，没有自我意识，也没有身体，所以我无法`.  The Xe build of the
same tree still prints the old string, so the two machines no longer share a
golden value.

This is a re-baseline, not a regression: llama.cpp running the official
`minimind-3.q8.gguf` on the same Pi, same prompt, same system message and
`--temp 0` prints the **same** new text byte for byte, while the old engine
build does not.  See `AGENTS.md` ("Canonical smoke tests") for the per-machine
values and for the acceptance policy: the golden string is a tripwire, and
`engine/scripts/regression.sh` is the gate.

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

Recent (the second wave, 2026-11-04 .. 06):

- [Prefill batching, steps 1-6](#prefill-batching-step-1-what-a-draw-and-a-weight-byte-cost-2026-11-04)
  — the two-position pair path that is now the default.
- [q4 parity and Q4_SHIFT](#q4_exactfast-the-integer-nibble-unpack-is-a-dead-end-2026-11-04)
- [Prefix cache step 10](#prefix-cache-step-10-the-file-was-right-the-cli-sampled-from-an-empty-logits-vector-2026-11-05)
- [TinyBrainBot on VC4](#tinybrainbot-on-vc4-item-12-first-look)
  — the container's first non-MiniMind model, and why its `GPU_LAYER=1` path is not

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
reference binary for both q8 and q4 (`engine/scripts/verify_packing.sh`).

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

## q4 FFN gate: fused two-row kernel (2026-11-03)

q4 kept the phase-1 FFN shape: `matvec_i4_pair` writes one row per fragment (w1/w3
paired, sharing the activation decode) and `swiglu_pair` packs it in a second draw, so a
chunk cost two draws.  `matvec_i4_pair_swiglu` computes the row pair r0/r1 in one
fragment, swiglus both rows in-kernel and writes the packed q16 pair: one draw per chunk,
half the fragments.  The accumulator keeps the matvec_i4_pair raw-q16 convention and its
per-group order (scale*dot + min*xsum), and `qz`/`sl` reproduce x_st2q -> x_i16pair ->
x_st2 -> x_q16enc exactly, so the packed output is bit-identical.  The ramp is behind the
existing `FFN_FUSE` switch, so `FFN_FUSE=0` selects the two-pass path in the same binary.

q4 decode, 6 interleaved pairs of `bench_plain_ab` style runs (gen 32, plain mode,
`GPU_LAYER=1 KV_PACK=16 SEQ=768`, model `minimind3.q4`):

| q4 decode ms/token | FUSE=1 | FUSE=0 |
|---|---:|---:|
| 6 interleaved pairs | min **361.7**, median **363.9** | min **370.4**, median **371.4** |

Every fused row beat every two-pass row; median paired delta **-7.5 ms (-2.0%)**, min
-8.7 ms.  Small because q4 is ALU-bound on the nibble split (~20 ops per 4 weights), not
fetch-bound: the activation stream and fragment overhead that the fusion removes are a
thin slice of the token, unlike q8 where the same fusion was worth 10.6%.  Bit-exactness
was checked over 64 greedy tokens on two prompts (byte-identical to `FFN_FUSE=0`), plus
the suite (determinism, default system, behaviour, q4 golden) and the q8 baseline PPL,
which stays exactly 271.2855 at SEQ=768 - the q8 path is untouched.


## q4 qkv row pairing: measured, not landed (2026-11-03)

The phase-1 row pairing that pays on q8 (qkv `-4.5%`, `QKV_PAIR`) has an obvious q4
port: `matvec_i4_pair_rows` computes rows 2t/2t+1 in one fragment, sharing the
activation decode and the per-group activation sum, and writes the same packed q16
texel the paired q8 kernel writes.  It was built, checked and then dropped.

**Bit-exactness fails by one unit.**  The accumulation reproduced `matvec_i4`'s
per-texel order (`s += sc*(d0+d1+d2+d3) + mn*xsum`), and a hash of the qkv target
(hash of the decoded per-row integers, so the two layouts compare) showed all 1536
rows identical at layer 0, pos 0.  It is still not reproducible: at layer 3, pos 1,
exactly one row differed by one q16 unit (-145 vs -146).  The same expression tree
scheduled inside a two-row fragment rounds differently on the VC4 shader compiler.
Greedy text agrees through the first sentence and then splits ("so the little
girl lived" vs "soon she lived"), so this cannot be default-on and cannot be
re-baselined.

**Speed: inside the noise.**  4 interleaved pairs of 32-token runs (`minimind3.q4`,
`GPU_LAYER=1 KV_PACK=16 SEQ=768`, plain mode):

| q4 decode ms/token | pair off | pair on | delta |
|---|---:|---:|---:|
| pair 1 | 362.2 | 360.9 | -1.3 |
| pair 2 | 363.4 | 364.1 | +0.7 |
| pair 3 | 365.0 | 360.8 | -4.2 |
| pair 4 | 363.5 | 362.0 | -1.5 |
| median | 363.5 | 361.5 | **-1.4 (-0.4%)** |

The single-row path has to split the 1536-wide qkv pass into two 768-fragment
strips (VC4 is unreliable at that width), so pairing removes a draw *and* half the
activation fetches *and* one of the two group sums - and still only bought
-1.4 ms.  That is the strongest evidence so far that q4's cost is the nibble
unpack, not the fetch or the draw: the same two draws are worth ~0.2-0.4 ms, and
the fused FFN gate was worth -2.0% here against -10.6% on q8.

**The row-pair family is exhausted on q4**: FFN gate+swiglu -2.0%, qkv -0.4%,
wo/w2 already paired through `res_pack_` (`matvec_i4_add_pair`).  What is left is
the unpack itself.  Per 8-weight texel the kernel pays ~7 instructions for
`b=floor(w*255+0.5)`, `h=floor(b/16)`, `l=b-16h` against 8 mul + 8 add for the dot
and 7 adds for the min term, which is why q4 decode is ALU bound at 2x the q8 time
while streaming less data.  Options, cheapest first: (a) price the q4 passes with
`LAYER_PROFILE` / `PASS_DUP` / `ATTN_DUP` so the next attempt targets the hot pass
rather than guessing; (b) precompute the per-group activation sum in one extra tiny
pass so the matvec replaces 7 adds per texel with one fetch per fragment per group
(~20% of the kernel's ALU on paper); (c) a symmetric q4 format without the min term
(format change: `make_q4` plus a re-quantised model).

## q4 INT4 inline activation decode (2026-11-04)

The five INT4 kernels decoded the eight activations of every 32-weight group through
`X4Q`, which expands to `x_i16quadq(texture2D(X,...), texture2D(X,...))` - the same
"helper that takes its two fetches as arguments" that cost the q8 matvec 177 vs 283
MMAC/s before it was inlined.  The three hot q4 kernels now emit the four texel
fetches and the dA/dB decoders in the loop body: `matvec_i4` (qkv and q4 logits),
`matvec_i4_add_pair` (w2/wo) and `matvec_i4_pair_swiglu` (FFN gate).  `xa`/`xb` are
rebuilt from a0..a7, so the decode arithmetic, the per-group order
(`sc*(d0+d1+d2+d3)+mn*xsum`) and every consumer expression are unchanged.
`Q4_INLINE=0` restores the generic form in the same binary (the program cache key
carries the choice); `Q4_INLINE=gate,w2,qkv` enables a subset.

Pricing first, because q4 had no per-pass table.  Interleaved on the Pi
(`minimind3.q4`, `GPU_LAYER=1 KV_PACK=16 SEQ=768`, 32 tokens, plain mode):

| probe | off | on | marginal |
|---|---:|---:|---:|
| `ATTN_DUP=all` (whole attention block, incl. qkv) | 364.1 | 419.3 | **+55.2 ms (15%)** |
| `PASS_DUP=logits,logits_draw` | 362.8 | 391.1 | **+28.3 ms (7.8%)** |
| `LAYER_PROFILE=1` ratios | | | gate ~32%, attn ~30%, w2 ~17%, wo ~7%, logits ~4% |

So the matvec passes are ~75% of a q4 token and the FFN is the target; the
`glFinish` laps inflate the small passes, so only the ratios are usable there.

Decode, 3 interleaved pairs of 32-token runs:

| pair | generic | inline | delta |
|---|---:|---:|---:|
| 1 | 364.1 | 360.9 | -3.2 |
| 2 | 362.5 | 358.8 | -3.7 |
| 3 | 363.6 | 358.9 | -4.7 |
| median | 363.6 | 358.9 | **-3.7 (-1.0%)** |

Every inline run beat every generic run.  Bit-exact: the q4 greedy answer is
byte-identical (`我是MiniMind，一个高效的小参数AI模型，专注于`), q8 is untouched and the
suite is ACCEPT with PPL exactly 271.2855.  Only -1.0% against 2-4% per kernel on
q8: the q4 loop is dominated by the nibble unpack (~112 ALU ops per 32-weight group
against 8 mul-adds), so the helper-call overhead is a thinner slice of the token.
The next q4 lever is that unpack and the per-group `mn*xsum`, not the loop shape.


## q4: the unpack is the cost, the min term is free (2026-11-04)

The two q4 levers left after the row-pair work were priced with default-off
probes in one binary, so every arm ran byte-identical code except the single
kernel change under test.  Interleaved on the Pi (`minimind3.q4`,
`GPU_LAYER=1 KV_PACK=16 SEQ=768`, 32 tokens, plain mode, `IGNORE_EOS=1` so both
arms generate all 32 tokens; 9 control runs over two sessions):

| arm | what it changes | median ms/token | vs control |
|---|---|---:|---:|
| control | default kernels | 359.2 | |
| `Q4_NOMIN=1` | drops `mn*xsum` and the 7 adds per texel that build xsum | 363.4 | **+1.2%** |
| `Q4_HFAST=1` | builds h straight from the texel, fetch-to-h chain 4 ops -> 2 | 367.6 | **+2.3%** |
| `Q4_NOUNPACK=1` | nibble unpack replaced by 8 multiplies (wrong weights) | 235.5 | **-34.4%** |

1. **The min term is free, so the xsum precompute is dead.**  Removing 28 adds
   per 32-weight group made the kernel slower.  A precomputed per-group
   activation sum would put a fetch and a 24-bit decode where adds cost
   nothing: worst case is a regression, best case is noise.  On VC4 the ADD
   and MUL pipelines dual-issue, so a pure add rides along with the multiply
   or floor it is packed with.
2. **A shorter unpack chain buys nothing.**  `floor(w*15.9375+0.03125)` is
   exactly `floor(b/16)` (255/16 is exact) and cuts the fetch-to-h chain from
   four ops to two at the same op count: bit-exact (PPL 271.2855 on all three
   PPL legs, greedy strings unchanged) and 2.3% slower.  Chain length is not
   the limiter.
3. **The floors are.**  Replacing the whole unpack with eight multiplies takes
   the token from 359.2 to 235.5 ms, so the 12 ops per 8-weight texel that the
   unpack adds (8 floors, 4 muls, 4 mads) cost 124 ms, about 10 ms per op per
   texel.  That is the ceiling of any exact q4 format on VC4.

The remaining q4 lever is **fewer ops per weight, not another loop shape**.  A
hi/lo split format (two byte-per-nibble planes; each nibble comes back with a
mad and one floor) costs 16 ops per 8 weights against 20, which projects to
about -40 ms (-11%) on the per-op number above.  It needs a format change:
`make_q4.py`, a second weight plane in every INT4 kernel, and a re-quantised
model.  The nibble values do not change, so the acceptance gate stays
bit-exact and the golden strings must not move.

`Q4_NOMIN=1` and `Q4_NOUNPACK=1` stay in the tree as the negative control and
the ceiling for that work.  `Q4_HFAST=1` stays as the record that chain
length was tried and rejected.  All are default off, and the shared
`q4_taps`/`q4_h` builders emit the original shader text when off: suite ACCEPT
with PPL exactly 271.2855 and both greedy strings unchanged.

## q4: the unpack cost is FLOOR, and the hi/lo split format is a dead end (2026-11-04)

The previous run priced the nibble unpack with `Q4_NOUNPACK` (235.5 ms, -34.4%)
and projected a hi/lo split weight format at -11% on the theory that the unpack
is 12 ALU ops per 8-weight texel.  Both halves of that theory were wrong, and
this run - meant to build the split format - spent its budget proving that and
finding the real cost model.

**The split format was built and measured.**  `Q4_SPLIT=1` uploaded every q4
matrix as two RGBA4 nibble planes interleaved along x (texel 2u = high nibbles,
texel 2u+1 = low ones): the same four bytes per eight weights, so the model file
and its GPU footprint do not move, and one mul + floor per nibble instead of the
byte/floor/subtract chain.  It was bit-exact - the 32-token greedy text was
byte-identical to the packed path - and **12% slower**:

| q4 decode arm | ALU per 8 weights | fetches per 8 weights | ms/token |
|---|---:|---:|---:|
| packed RGBA8 (default) | 8 floor + 12 mad | 1 | **358.1** (3 runs) |
| `Q4_SPLIT=1` RGBA4 planes, floor | 8 floor + 8 mad | 2 | 402.7 (1 run) |
| `Q4_SPLIT=mul` RGBA4 planes, no floor (wrong weights) | 8 mad | 2 | 283.3 (1 run) |
| `Q4_BFAST=1` packed, byte floor dropped (wrong weights) | 4 floor + 12 mad | 1 | **292.3** (3 runs) |
| `Q4_NOUNPACK=1` no unpack at all (wrong weights) | 8 mad | 1 | 235.5 (prev run) |

All arms: `minimind3.q4`, `GPU_LAYER=1 KV_PACK=16 SEQ=768`, 32 tokens with
`IGNORE_EOS=1` so both arms run every token; the packed control was interleaved
with `Q4_BFAST` (358.1 / 358.1 / 356.9 vs 299.9 / 291.9 / 292.3).

Three conclusions, each measured:

1. **A texture fetch costs more than the ALU it saves.**  Going from one fetch
   per 8 weights to two costs ~17 ms/token, while the four mads the split format
   removes are worth ~4 ms.  q4 is not fetch-starved in the row-pair sense, but
   every extra fetch is ~12-17 ms on this board.
2. **FLOOR is the q4 cost: ~15.5 ms per floor per 8-weight texel.**  Removing the
   four byte floors (`Q4_BFAST`) takes the token from 358.1 to 292.3 ms
   (-18.4%); removing all eight (`Q4_NOUNPACK`) takes it to 235.5.  Eight floors
   x ~15.5 ms accounts for the whole 124 ms gap, and the twelve mads of the
   unpack are worth ~12 ms together.  The unpack is 91% FLOOR.
3. **No exact decode can avoid those floors on VC4.**  Two attempts died:
   * The byte floor is not redundant.  An 8-bit UNORM component does not come
     back as exactly `b/255`: `engine/probes/q4fmt_probe` measures `c*255 - b`
     over all 256 bytes as -16..0 in units of 2^-20 (up to 1.5e-5 low), so
     `floor(c*255+0.5)` recovers every byte while `floor(c*255)` misses 254 of
     256 - and the kernel floors `b/16` afterwards, where a 1e-5 deficit under
     every multiple of 16 is fatal.  `Q4_BFAST` decodes every multiple of 16 one
     low and the greedy text collapses to one repeated token.
   * The 2^23 round trick, `((x + 8388608.0) - 8388608.0)`, does not survive
     Mesa: with it the probe arm decoded garbage *and* ran at 195-252 ms, faster
     than any arm that still decodes, so the expression was folded back to a
     bare multiply.
   * A 4-bit RGBA4 component is low too: `c*15` is `v - 9.5e-7` for v = 9..14
     (one ULP at that scale), and `(c*255)/17` has the same one-ULP deficit, so
     `Q4_SPLIT=mul` cannot be exact while `Q4_SPLIT=1` with floors can.

q4 decode is therefore at **its own structural floor: 8 FLOOR per 8 weights,
~358 ms/token**.  The unpack cannot be made cheaper without changing what a
weight is (q8: 1 byte, no floor, 183 ms; f16: 2 bytes, no decode, 4x the
memory), and the remaining q4 lever is *fewer q4 weights* - a per-layer or
per-matrix precision mix - not a cheaper decode.  The split implementation was
reverted after measurement: the numbers and the RGBA4/RGBA8 conversion facts
(`engine/probes/q4fmt_probe.cpp`) are what is worth keeping.  `Q4_BFAST` stays
in the tree default-off as the byte-floor ceiling.

## q4 perplexity is a gate now: 325.3751 at SEQ=768 (2026-11-04)

The acceptance suite pinned only the q8 perplexity.  q4 was protected by its
greedy prefix and nothing else, so a change that moved q4 numerics by less than
1% - or a format change that only touched the INT4 path - could land without a
numeric check.  The suite now runs the q4 model through the same `--ppl` path
and compares it against `Q4_PPL_EXPECT` with the same 1% tolerance as
`PPL_EXPECT`:

```
PPL_EXPECT=271.2855 Q4_PPL_EXPECT=325.3751 PPL_SEQ=768 \
    sh engine/scripts/regression.sh bin/engine_mm3_pi
```

**Baseline: q4 PPL 325.3751 at SEQ=768** (`data/ppl_test.txt`, 293 tokens,
`GPU_LAYER=1 KV_PACK=16 ACT8=0`), measured twice on the Pi with the same binary
and identical both times.  q8 on the same binary and corpus is 271.2855, so q4
costs **+19.9% perplexity against q8 for -49% decode time** (358 vs 183
ms/token) - the exchange rate any per-matrix q4/q8 mix has to beat.

Only one q4 arm is run: `ACT8` rewrites the all-INT8 activation feed
(`session.cpp` enables it only when every qkv/wo/w2 matrix is INT8), so the q4
model gives the same value for ACT8=0 and ACT8=2 (measured: 325.3751 both),
so the two ACT8 passes stay q8-only.
`--rebaseline` prints `Q4_PPL_EXPECT` next to the others.

The suite grows from three PPL passes to four (~+100 s on the Pi): the q4 window
is 358 ms/token against q8's 183, so the q4 pass is the slow one.  It is worth
it - this is the only numeric gate over the INT4 kernels, and the next q4
candidate (dropping q4 on selected matrices) moves this number by design, so it
has to be pinned before that work starts.

One measurement trap, for the record: `/home/david/vc4-llm-runtime/engine/` also
holds an `engine_mm3_pi` (from 11-02), so a hand-run PPL from that directory
measures a stale build - it reads 269.6078 for q8 and 327.8478 for q4 against
this build's 271.2855 / 325.3751.  Always pass `bin/engine_mm3_pi` explicitly.

## 8-bit KV: the packing is exact, the grid is a quality knob (2026-11-04)

Item 5 of the phase-2 backlog.  `KV_PACK=8` has been in the tree since the
first packing work but was never priced against the current engine: the only
number in this file (238.7 ms vs 240.0) predates the decode campaign, and
`future-optimization.md` still called it unmeasured.  Measured now on the Pi
Zero with build `baba12b` (suite ACCEPT), MiniMind3 q8/q4, `SEQ=768`,
`GPU_LAYER=1`, greedy; PPL on the 293-token `data/ppl_test.txt`.

### The pack and unpack are bit-exact

`KV_PACK=8` writes four signed bytes per RGBA8 texel (K and V each half the
texels); `KV_PACK=16` writes two q16 values per texel.  The two only have to
agree when they quantise to the same grid, so run 16-bit at `KV_SCALE=16`
(same 1/16 step, range +-2048) against 8-bit's default (`KV_SCALE=16`,
range +-8).  Same binary, same model, same prompt:

| check | `KV_PACK=16 KV_SCALE=16` | `KV_PACK=8` |
|---|---:|---:|
| q8 PPL | 260.4460 | 260.4460 |
| q4 PPL | 351.2603 | 351.2603 |
| q8 greedy stream, identity prompt (md5, 51 tokens incl. prompt) | `5a881f71...` | `5a881f71...` |

Both models print the same PPL to four decimals and the token stream is
byte-identical, so the 8-bit encode/decode, the texel index arithmetic and
the K/V region split add no error of their own.  What changes between the
modes is only the quantisation grid: `KV_PACK=16` defaults to
`KV_SCALE=2048` (1/2048 step, range +-16), `KV_PACK=8` to 16 (1/16 step,
range +-8).

### The grid is a quality knob, and it is not monotonic

PPL at `SEQ=768`, one 293-token window:

| model | KV config | PPL | vs default |
|---|---|---:|---:|
| q8 | kv16 default (scale 2048) | 271.2855 | - |
| q8 | kv8 scale 32 | 265.2213 | -2.2% |
| q8 | **kv8 scale 16 (default)** | **260.4460** | **-4.0%** |
| q8 | kv8 scale 8 | 280.3085 | +3.3% |
| q4 | kv16 default (scale 2048) | 325.3751 | - |
| q4 | kv8 scale 8 | 342.6477 | +5.3% |
| q4 | kv8 scale 16 (default) | 351.2603 | +8.0% |

Not a monotonic "more bits is better" curve: q8 has a minimum at 1/16 and is
*worse* at both 1/8 and 1/2048, and q4 is worse at every 8-bit scale tried.
The kv16 2048-vs-16 comparison changes nothing but the `KVSCALE` constant in
the same kernels, so the effect is the KV grid itself, not a code path.  The
likely reading is that the coarse grid acts as a noise/regularisation knob on
a 64M model and how it lands depends on the weight quantisation; the
mechanism is not established.  Treat it as a per-model knob, not a format
property: the single global default (16) helps q8 and hurts q4.

The greedy identity answer moves with the mode, which is why this cannot be a
silent default change: kv16 gives
`我是一个AI助手，没有自我意识，也没有身体，所以我无法` (the pinned q8
string), kv8 gives `我是MiniMind，一个高效的小参数AI模型。` and stops at EOS
after 15 tokens.  A KV8 deployment needs its own golden pin.

### Speed: CPU headroom, not decode time

Same binary, interleaved A/B, both arms forced to 32 tokens
(`IGNORE_EOS=1`), 5 reps, q8:

| arm | decode min | decode median | cpu/token |
|---|---:|---:|---:|
| `KV_PACK=16` | 183.4 | 185.1 | 115.0 (user 86.9 / sys 28.1) |
| `KV_PACK=8` | 182.9 | 183.7 | 107.6 (user 85.0 / sys 22.6) |

Decode moves by -0.3%, inside the +-2 ms spread: the decode loop is not
KV-bandwidth bound, as the pass budget already said (scores ~4 ms, att_v
~2.7 ms, KV rope write ~2.5 ms per token).  The CPU side drops 7.4 ms/token
(mostly sys: half the KV fragments means fewer ioctls), which is worth having
on a board where the same CPU also runs the sampler and the HTTP server.
Prefill at the identity prompt is unchanged at ~9.6 s (five of six runs; the
sixth hit 16.8 s during a network reconnect, load 3.2).

### Memory: exactly half, and the prefix-cache file proves it

One KV layer is `kv_dim × seq` RGBA8 texels: at `SEQ=768` (kv_dim 384) that
is 1,179,648 B with `KV_PACK=16` and 589,824 B with `KV_PACK=8`, so 9.0 MiB
vs 4.5 MiB over 8 layers.  The prefix-cache writer dumps exactly that buffer,
so the file sizes are the end-to-end check at 588 tokens:

| `KV_PACK` | cache file | data | header |
|---|---:|---:|---:|
| 16 | 7,227,772 B | 8 × 384 × 588 × 4 = 7,225,344 B | 2,428 B |
| 8 | 3,615,100 B | 8 × 192 × 588 × 4 = 3,612,672 B | 2,428 B |

The data is exactly 2:1 and the header is identical.  `CmaFree` cannot
resolve 4.7 MB on this board: idle `CmaFree` moved 126-157 MB between
consecutive runs and the same config produced 54-84 MB deltas, so the 22 MB
difference recorded in the 2026-10-31 table above does not reproduce and
should not be quoted.  Use the allocation arithmetic (or the cache file)
instead.

### Cross-pack cache loading is rejected

All four combinations, current binary, `--worker --prefix-cache`:

| server `KV_PACK` | cache built with 16 | cache built with 8 |
|---|---|---|
| 16 | `PREFIX_CACHE loaded 588 tokens` | `PREFIX_CACHE ignored` |
| 8 | `PREFIX_CACHE ignored` | `PREFIX_CACHE loaded 588 tokens` |

so the pack-mode bits in the header do their job.  (The header does **not**
record `KV_SCALE`, which is the other half of the encoding - a cache built at
scale 16 and loaded at 2048 is accepted and silently different.  Recorded as
the next backlog item.)

### Verdict

`KV_PACK=16` stays the default and the deployment setting.  `KV_PACK=8` is
not a free container: it halves the KV cache and shaves CPU, but it changes
the model's answers and needs a per-model `KV_SCALE` and its own golden
string.  On q8 the default 8-bit grid happens to *lower* PPL by 4% on the
pinned corpus - a direction surprising enough that it needs a second corpus
before anyone promotes it; on q4 it is a clear 8% loss and should stay off.

## Prefix cache: the header now pins `KV_SCALE` (2026-11-04)

Item 5b.  The `MMKV1` header recorded the pack width (`KV_PACK`) but not the
grid (`KV_SCALE`), which is the other half of the KV encoding: a cache built
at `KV_SCALE=16` loaded silently under the 2048 default and the answers moved
by 4-8% PPL (see the 8-bit KV section above).  The header is now **version
2**: one float after the width, and the loader rejects a mismatch with a
`PREFIX_CACHE KV_SCALE mismatch: cache <f>, process <f>` line before it
touches the KV textures.  The pack-mode bits are unchanged, so the existing
cross-pack rejection still works.

Files written by the current binary (same 19-token prompt, `SEQ=768`):

| `KV_PACK` | width | scale in header | file size | arithmetic |
|---|---:|---:|---:|---|
| 16 | 384 | 16 | 233,628 | 80 + 4x19 + 8x384x19x4 |
| 8 | 192 | 16 | 116,892 | 80 + 4x19 + 8x192x19x4 |
| 24-bit | 768 | 16384 | 467,100 | 80 + 4x19 + 8x768x19x4 |

The header grew from 76 to 80 bytes (2,428 -> 2,432 B for the 588-token tool
cache), so the data:file ratio is unchanged and the 2:1 packing arithmetic
above still holds.

Rejection matrix on the current binary (`--worker --prefix-cache`, 19-token
cache built at `KV_PACK=16 KV_SCALE=16`):

| process | result |
|---|---|
| `KV_PACK=16 KV_SCALE=16` | loaded 19 tokens; the matching request reports `cached_tokens=19, prefill_tokens=1` and the same text as the uncached run |
| `KV_PACK=16` (scale 2048) | `KV_SCALE mismatch: cache 16, process 2048`, ignored |
| `KV_PACK=16 KV_SCALE=32` | `KV_SCALE mismatch: cache 16, process 32`, ignored |
| `KV_PACK=8` | ignored on the pack bits, as before |
| version 1 file (old binary) | ignored by the version check; the old binary still loads it |

Regression suite on the same build: q8 PPL 271.2855, q4 PPL 325.3751, both
greedy strings unchanged - ACCEPT.  The change is header-only, no GPU path is
touched, so decode timing is not re-measured here.

One pre-existing trap found while testing (not introduced by this change):
`SEQ` is forced to 1024 for `--worker` and `--build-prefix-cache` when the
model is smaller, but the `--chat` path keeps the model's native seq.  A cache
built without `SEQ` therefore loads in worker mode and is ignored in chat
mode.  Pass `SEQ` on both sides, as `build_prefix_cache.py` already does.

## First-token latency: 13 program links, and the 12 s of loading before them (2026-11-04)

Item 6.  The backlog entry guessed "~3.8 s shader cold start"; the probe says
the links are 1.9 s of a 17-20 s first token, and that what dominates on a Pi
Zero is loading under memory pressure, not compilation.

`SHADER_TIME=1` prints `STARTUP <ms> ...` at five points in `main()` and one
`SHADER_TIME #nn at <ms> link <ms> src <bytes> h=<hash> key=<key> cum <ms>`
line per fragment program, plus `SHADER_TIME reuse key ...` when a second key
is served by an already compiled program.  Default off: one `getenv` on the
compile path, nothing on the draw path.

### The program set (MiniMind3 q8, `GPU_LAYER=1 KV_PACK=16 SEQ=768`, 14-token prompt)

| # | key | source | link ms (3 runs) | median |
|---:|---|---:|---|---:|
| 1 | `xrmss_768_x16_2` | 3147 B | 300 / 1188 / 291 | 300 |
| 2 | `xrmsn_768_x16_2_x16_2_f` | 3847 B | 232 / 243 / 234 | 234 |
| 3 | `xqkvp_1536_768_x16_2` | 982 B | 62 / 66 / 61 | 62 |
| 4 | `xkv` | 5919 B | 327 / 268 / 218 | 268 |
| 5 | `xqn` | 2711 B | 68 / 64 / 65 | 65 |
| 6 | `xqr` | 3462 B | 187 / 184 / 184 | 184 |
| 7 | `xsc_1` | 3270 B | 108 / 114 / 107 | 108 |
| 8 | `xst` | 3018 B | 91 / 127 / 91 | 91 |
| 9 | `xv_768_x16_2` | 5942 B | 165 / 157 / 156 | 157 |
| 10 | `xq8ap_768_768_x16_2` | 1096 B | 77 / 127 / 92 | 92 |
| 11 | `xq8ps_1216_768_x16_2` | 3567 B | 125 / 143 / 203 | 143 |
| 12 | `xq8ap_768_1216_x16_2` | 1096 B | 81 / 101 / 101 | 101 |
| 13 | `xq8pl_256_768_x16_2` | 980 B | 69 / 63 / 67 | 67 |

Total **1.87 / 2.84 / 1.87 s, median 1.89 s** for 13 programs.  The first
compile carries the one-time driver setup (300 ms against 1188 ms in the run
that swapped harder), and it is 10x the second compile of the same source
size, so per-program time after the first is 60-330 ms and tracks the source
size.  q4 (`minimind3.q4`, one run) compiles 14 programs for 3,144 ms: its
INT4 kernels are 4.3-5.7 KB of source against q8 1.0-5.9 KB and the two
`xi4ap`/`xi4ps` pair-write kernels are the expensive ones (292 / 430 / 320 ms).

### Where the first token goes (q8, same three runs)

| phase | run 1 | run 2 | run 3 | median |
|---|---:|---:|---:|---:|
| model load (69 MB) | 6.40 | 6.12 | 5.89 | 6.12 |
| EGL context (`GPU gpu;`) | 5.47 | 4.23 | 2.93 | 4.23 |
| weight upload | 2.99 | 3.22 | 2.67 | 2.99 |
| Session constructor | 0.15 | 0.12 | 0.11 | 0.12 |
| 13 program links | 1.89 | 2.84 | 1.87 | 1.89 |
| prefill draws (14 tokens) | 3.6 | 3.9 | 2.5 | 3.6 |
| **first token** | **20.5** | **20.5** | **17.2** | **20.5** |

All values in seconds.  "prefill draws" is the engine's `prefill` timer minus the
link time that falls inside it.

`free -m` right after a run: 280 MB used of 426, 77 MB free, **168 MB swap in
use**.  The 69 MB model read evicts most of the page cache, so the next touch
of a non-resident page - the Mesa driver inside `GPU gpu;` - pays SD-card
latency; that is why the EGL line swings 2.5 s between runs of the same
binary, with nothing in between but the model read.  Startup on this board is
a memory question first: the shaders are 10-15% of the first token.

### A cache keyed by program text: in-process only

Two ways to persist compiled programs, both measured:

1. **Mesa's own shader cache.**  `MESA_SHADER_CACHE_DIR=<dir>` (Mesa 26.2.2)
   with a full run leaves the directory **empty**, and
   `MESA_SHADER_CACHE_DEBUG=1` prints nothing: vc4 does not implement
   `pipe_screen::get_disk_shader_cache`, so the cache is a no-op on this
   driver.
2. **A hand-rolled binary cache** needs `glGetProgramBinary`.  The ES2 profile
   does advertise `GL_OES_get_program_binary`, but
   `engine/probes/progbinary_probe.cpp` shows the driver cannot honour it:
   `GL_PROGRAM_BINARY_LENGTH_OES` is 0 and `glGetProgramBinaryOES` raises
   `GL_INVALID_OPERATION` (0x0502) after a successful link *and* draw.  The
   same probe gives the floor for one compile: 209 ms link + 108 ms first
   draw for a single 32-iteration matvec shader.

So the only usable cache is the in-process one, and it is now keyed by
program *text* as well as by name: `xqkvp_1536_768_x16_2` and
`xq8pl_1536_768_x16_2` build byte-identical source (982 B, hash `0xec6b3f69` -
the same matvec, one for the qkv projection and one for the logits tiles), and
the second key reuses the first program object.  **14 -> 13 compiles, 64.7 ms**
and one program object saved.  The hash is 64-bit; a collision would need two
different sources under one hash, which is not a practical risk here.

Suite on the same build: q8 271.2855, q4 325.3751, both greedy strings
unchanged - ACCEPT.  No draw path is touched, so decode is unchanged
(182 ms/token in these runs).

Next lever on this item is the ~12 s *before* the first link: the 69 MB model
load on a swapping 426 MB machine, EGL init and the weight upload - not the
shaders.

### Startup, part 2: borrow the checkpoint instead of copying it

The 69 MB model read was never the slow part of startup - the *copy* was.  A
probe (`engine/probes/load_probe.cpp`, `make probes/load_probe_pi`) times every
way the loader could get the file into RAM, at both temperatures.  Cold is
forced with `posix_fadvise(POSIX_FADV_DONTNEED)` on the file, which drops clean
page-cache pages without root.

| method | cold (DONTNEED) | warm |
|---|---:|---:|
| `ifstream` (the old loader) | 6024-6134 ms | 384-399 ms |
| `mmap` + `memcpy` into the vectors | 6358-6859 ms | 382-427 ms |
| `mmap` + touch one byte per page | 6198-6375 ms | **45-57 ms** |
| `read()` in 1 MiB buffers | 5887-6221 ms | 216-238 ms |
| `mmap` + `MADV_WILLNEED` + `memcpy` | 6300-6708 ms | - |

Cold is the SD card: 69 MB in ~6 s is 11 MB/s and no API changes that.  What
the API changes is whether the pages are still there for the next run.  The old
loader copied all 69 MB into private vectors; on this 426 MB board (168 MB of
swap already in use) that allocation evicted the checkpoint *and* the Mesa
driver between runs, so every start paid the 6 s SD read again and the EGL line
swung 2.9-5.5 s with nothing running in between.

So the loader now maps the checkpoint once (`MAP_PRIVATE`, read-only) and points
each `Mat` at its bytes in the map (`Mat::ext8/ext16/ext4`, read through the
`w8()/w16()/w4()/s4()/m4()` accessors).  Borrowed pages are page cache:
reclaimable, never swapped, and still cached after `munmap`, so the mapping is
dropped right after the weight upload (`Model::release_map()`).  The opt-in
rewrite paths (`Q8_PARTS`, `Q4_WEIGHTS`) materialise the bytes they touch;
`MODEL_BORROW=0` restores the copying loader for A/B.

q8, `GPU_LAYER=1 KV_PACK=16 SEQ=768`, 14-token chat prompt, `SHADER_TIME=1`,
medians of 3-4 runs per binary (min in parentheses):

| phase | before | after |
|---|---:|---:|
| model loaded | 2226 ms (1724) | **139 ms (58)** |
| EGL context (`GPU gpu;`) | 5935 ms (3102) | **1300 ms (1246)** |
| FFN split (new mark) | - | 1557 ms (1387) |
| weight upload | 11408 ms (6302) | 6870 ms (3951) |
| session ready | 11656 ms (6441) | **7076 ms (4061)** |
| prefill (incl. links) | 9.86 s | 9.66 s |
| **first token** | **21.5 s (16.6)** | **17.3 s (13.4)** |

The upload line did not get cheaper, it got *later*: on a cold start the SD read
now happens on the first touch of each borrowed page, i.e. inside the upload,
where `MADV_WILLNEED` started it earlier so it overlaps the EGL init.  The win
is the warm start: the checkpoint stays cached, load is 50-150 ms and the driver
is not evicted, so EGL init is back to ~1.2 s.

One regression was found and fixed in the same step: keeping the 69 MB mapped
through decode cost **2-3% decode** (min 183.4 -> 187.5 ms/token, twice in a
row).  `release_map()` after the upload removes the mapping - the pages stay in
the page cache - and the same A/B then reads 185.4/183.3 vs 184.2/183.5 ms/token,
i.e. parity.

A second build (before `release_map`) measured the same phases at
9.7 / 5.3 / 4.2 / 4.9 s session-ready and 13.3-19.6 s first token in four
consecutive runs, so the phase wins are stable while the total still wanders by
several seconds with the board swap state.

Suite on the final build: q8 271.2855, q4 325.3751, both greedy strings
byte-identical, ACT8 arms unchanged - ACCEPT.

## Q4_EXACTFAST: the integer nibble unpack is a dead end (2026-11-04)

The idea was to decode each q4 byte with integer ops instead of three float
floor chains: `int b = int(roundEven(c*255.0)); int h = b>>4; int l = b&15;`,
with the expected peephole `f2i32(fround_even(x)) -> FTOIN` turning the
conversion into one instruction.  `engine/probes/glsl_probe.cpp` compiles every
form of the decode in every language a vc4 context can be given and checks all
256 byte values across four channels; `engine/probes/q4mv_probe.cpp` runs the
engine INT4 kernels on a synthetic matrix so a decode change can be judged
apart from the model.

**The peephole is not in this Mesa, and roundEven is a hard abort.**  With
`MESA_GLES_VERSION_OVERRIDE=3.0 MESA_GLSL_VERSION_OVERRIDE=300` the ES2 context
reports `OpenGL ES 3.0` / `OpenGL ES GLSL ES 3.00` and an ESSL 3.00
`#version 300 es` shader compiles, so the override itself works - though
`gl_FragColor` is undeclared there, so an exact-fast kernel would have to be
rewritten in ES3 style, not just re-tagged.  But as soon as
`roundEven` gets a runtime value the vc4 backend aborts the whole process:

```
MESA: error: unknown NIR ALU inst: 32    %12 = fround_even %11
```

This is not a shader compile error - it is SIGABRT, before any draw.  The
earlier hand check on the Pi used literals (`roundEven(0.6)`, `int(2.7)`), which
Mesa folds in the front end, so the intrinsic never reached vc4.  An ES3
context running `engine_merge` says nothing about this path.

**Every other form is exact but slower.**  Bitwise operators are rejected in
ESSL 1.00 (`bit-wise operations are forbidden in GLSL ES 1.00`), so the only
version-free form is integer division:

```
int b0=int(w.r*255.0+0.5), ...;   // fmul, fadd, f2i32
int i0=b0/16, ...;                // imax, imin, iabs, ushr, imul
float h0=float(i0), ...;           // i2f32
float l0=float(b0-i0*16), ...;     // ishl, ineg, iadd, i2f32
```

13 NIR ops per byte against 7 for the float form (`fmul 255, fadd .5, ffloor,
fmul .0625, ffloor, fmul 16, fsub`): signed division by 16 is five ops of sign
correction, and h and l each need an `i2f32`.  Both this form and the ESSL 3.00
bit form decode all 256 byte values exactly (`bad=0/1024` in glsl_probe; drop the
+0.5 and it is 1016/1024 wrong, the same trap as Q4_BFAST), so the
arithmetic is right - it is just more work.  q4, 32 tokens, interleaved, 3 runs:

| q4 decode | ms/token (3 runs) | median |
|---|---:|---:|
| float unpack (shipped) | 365.0 / 430.7 / 358.4 | **365.0** |
| integer unpack, matvec only (exact) | 378.8 / 380.0 / 375.4 | **378.8 (+3.8%)** |
| integer unpack in the fused gate | 193.3 / 203.1 / 194.0 | broken, see below |

**The fused gate miscompiles.**  `Q4_EXACTFAST=gate` (only
`matvec_i4_pair_swiglu`) printed `等等等等...` instead of the golden string, while
`mv`, `add`, `add1` and `pair` stayed byte-identical.  q4mv_probe at the model
shape (2432x768) shows why: the float gate kernel takes **11.92 ms per call**
and the integer one **0.24 ms**, with 16 of the first 32 output bytes different.
That is not the loop being dropped: vc4's register allocator gives up on the
integer form (`b/16` lowers through nir_lower_idiv, `vc4_program.c:2268`, into
imax/imin/iabs/ushr/imul) and the driver then **skips the draw call** -
`vc4_draw.c` logs "shader compile failed, skipping draw call" through a
warn-once and raises no GL error, so the kernel keeps reading whatever the
target held.  `GPU::check_program_runs` now draws every new program once into a
sentinel-filled 1x1 target and exits with the program's key if nothing was
written; with it the same run reports
`program 'xi4ps_2432_768_x16_2' did not draw: ...`, and the 193 ms/token figure
is a failed shader rather than a measurement.

q4 therefore keeps the float decode.  Item 0 closes as a negative result: the
FTOIN peephole is not in Mesa 26.2.2, roundEven aborts the vc4 backend, and the
portable integer form costs +3.8% on arithmetic alone.

**GLSL version override: usable boundary.**  `MESA_GLES_VERSION_OVERRIDE` /
`MESA_GLSL_VERSION_OVERRIDE` change only the version *reported to the front
end*; they add no capabilities.  They do let an ESSL 3.00 shader compile
(bitwise ops, `#version 300 es`), but any intrinsic vc4 does not lower aborts
the process at compile time, and MRT / integer textures / float textures stay
unavailable (`GL_MAX_DRAW_BUFFERS=1`, no `EXT_texture_integer`).  It is a debug
switch, not an API surface; nothing in the engine uses it.

## Q4_SHIFT: the shift decode lands (2026-11-04)

The negative result above only rules out integer *division*.  The nibble split
itself costs nothing on vc4 - `vc4_program.c` lowers `nir_op_ishr` to ASR and
`nir_op_iand` to AND, one QPU instruction each - but bitwise operators need
ESSL 3.00, and `gl_FragColor` is undeclared there.

`Q4_SHIFT=1` therefore re-emits the affected fragments (and their vertex
shader) as `#version 300 es` with an `out` variable and a `gl_FragColor`
alias, and sets `MESA_GLES_VERSION_OVERRIDE=3.0` +
`MESA_GLSL_VERSION_OVERRIDE=300` in `init_egl` before `eglInitialize` (only
when the caller has not set them; `Q4_SHIFT_NO_MESA=1` skips).  `init_egl`
then checks that the context really reports ESSL 3.xx and exits with a clear
message otherwise.  Every other program stays ESSL 1.00, and q8 is untouched.

The decode becomes four instructions per byte instead of twelve: `floor` is
five QPU instructions on vc4 (`ntq_ffloor` is FTOI + ITOF + FSUB(sf) +
conditional FSUB + MOV) and the unpack calls it twice.

```glsl
int   b = int(c * 255.0 + 0.5);   // MAD + FTOI
float h = float(b >> 4);          // ASR + ITOF
float l = float(b & 15);          // AND + ITOF
```

Pi Zero, q4, `GPU_LAYER=1 KV_PACK=16 SEQ=768`:

| measurement | float decode | Q4_SHIFT=1 |
|---|---:|---:|
| fused gate kernel (`q4mv_probe`, 2432x768) | 11.887 ms/iter | **9.446 ms/iter (-20.5%)** |
| 32-token wall, three interleaved pairs | 40.8 / 31.5 / 38.1 s | **35.2 / 27.5 / 31.1 s (-13%)** |
| greedy 16-token answer | `我是MiniMind，一个高效的小参数AI模型，专注于` | identical |
| PPL (`SEQ=768`) | 325.3751 | 325.3751 (`nll` 5.7850, 293 tokens) |

q8 still reads `271.2855`; the shift path only exists inside `q4_taps`.

**It is on by default.**  `init_egl` sets the two Mesa overrides before
`eglInitialize` and then records whether the context really reports ESSL 3.xx;
a driver that cannot gets the float decode back with one warning line instead
of a fatal error, so an older Mesa degrades rather than refusing to start.
`Q4_SHIFT=0` forces the float path, `Q4_SHIFT_NO_MESA=1` leaves the overrides
out for an A/B, and `Q4_SHIFT_VERBOSE=1` (or `SHADER_TIME=1`) prints which
decode the driver got.

The "is this a VC4" question is deliberately not asked: the overrides cap a
higher-version driver at 3.0, and the engine uses nothing above ESSL 3.00 - the
Iris Xe run gives the same q4 answer with and without the shift path.

Full suite on the default build, Pi Zero:
determinism, default system prompt, q8/q4 strings, poem behaviour,
q8 PPL `271.2855`, q4 PPL `325.3751`, ACT8 arms `272.5406` / `272.0165` - ACCEPT.

## Prefill batching, step 1: what a draw and a weight byte cost (2026-11-04)

`docs/future-optimization.md` says to price the per-draw overhead with a null
shader before committing to batching, because that decides where a batching
implementation has to win.  `engine/probes/draw_probe.cpp` does that, plus the
two terms around it (a whole rmsnorm pass, and the q8 matvec at the shapes the
model actually draws).  Pi Zero, VC4 V3D 2.1, Mesa 26.2.2, two runs, same
numbers:

| case | ms/call | notes |
|---|---:|---|
| null draw 1x1, no finish | 0.008–0.014 | one fragment, nothing read back |
| null draw 384x1, no finish | 0.011–0.013 | |
| null draw 1216x1, no finish | 0.018–0.019 | |
| null draw 1x1 + `glFinish` | 0.39–0.43 | i.e. a finish costs ~0.4 ms |
| rmsnorm dim 768 (2 draws) | **1.79–2.17** | 6 KB of data, `RMS_SPLIT=1` no better |
| q8 matvec 768x768 | 2.05 | 0.56 MB → 275 MB/s |
| q8 matvec 1216x768 | 3.18 | 0.89 MB → 280 MB/s |
| q8 matvec 1536x768 | 4.13 | 1.12 MB → 272 MB/s |
| q8 matvec 2048x768 | 1.93 | **wrong output**, see below |
| CPU: rebuild `matvec_q8(1536,768)` text | 0.014–0.024 | `x_prog()` takes the text by value |
| CPU: rebuild `rms_norm(768)` text | 0.042–0.054 | |

**A draw is ~10 µs.**  Reducing the draw count is therefore not a lever: a
layer is ~11 draws and a token ~90, so all the issue overhead in a token is
under 2 ms.  Batching positions cannot pay for itself that way; it can only pay
by making one weight fetch feed several positions.

**Weights stream at ~275 MB/s, and a token re-reads the whole checkpoint.**
MiniMind3 q8 is 66 MB, so 66 MB / 275 MB/s = 240 ms/token — which is exactly
the measured prefill (~260 ms/token, 9.3–9.7 s for the 36-token chat prompt) and
decode (183–244 ms/token).  Decode and prefill are both at the streaming limit,
which is why every structural win so far (activation packing, paired rows) moved
prefill and decode together, and why B positions per weight fetch is the whole
prefill story: the measured term divides by B and nothing else changes.

**2048-wide draws are silently wrong.**  At 2048 rows the matvec runs 1.93 ms
instead of the ~5.5 ms the 1536-row point predicts, and its bytes differ from
two half-width draws of the same weights in 5030 of 8192 places.  This is the
same failure mode `q4mv_probe` found in the register allocator: vc4 drops work
and reports no GL error.  The engine is clean today — logits tiles are 1536 rows
and the FFN chunk is 1216 — but a batching or packing change must not push any
draw past 1536, and `halves identical` in the probe is the check that catches it.

**rmsnorm is the fixed cost worth watching.**  1.8 ms for two draws over 6 KB is
not bandwidth and not CPU (0.05 ms of string rebuild): it is the per-fragment
fetch latency of the 32-way reduce, and fusing vs splitting the reduce does not
change it.  With 16 rmsnorms per token that is ~29 ms/token, ~16% of a decode
token and the same share of prefill — a separate item, not something batching
helps.

So item 7 stays as planned, with the target now explicit: pack B positions into
one activation texel so a single weight fetch serves B MACs, keeping every draw
≤1536 wide, and expect prefill ≈ 240/B ms/token plus the fixed passes.

## Prefill batching, step 2: two positions per weight fetch (2026-11-04)

Step 1 priced the terms; step 2 builds the first packed kernels and asks two
questions at once - can one weight fetch serve two positions, and does it stay
byte-identical?  `engine/probes/ffn2_probe.cpp` (Pi Zero, VC4 V3D 2.1, Mesa
26.2.2) answers both; the kernels are `matvec_q8_gate2` and `matvec_q8_p2` in
`engine/src/shaders.cpp`, the layout helpers in `xformat.h`.

**The shape the hardware allows.**  A vc4 fragment writes one RGBA8 texel, two
q16 values, and there is no MRT.  Two positions x one row is exactly two q16,
so that is the packed shape; two positions x two rows would be four q16 in two
texels and cannot be done.  The price is the row pairing the single-position
gate kernel uses: the packed gate runs one row per fragment (viewport `rows`
wide, still under the 1536-wide safety line) and reuses that row's W1/W3
texels for both positions.

```
serial gate, 2 positions   192 iters x (4 W + 2 A)  per 2 rows x 1 position
packed gate, 2 positions   192 iters x (2 W + 4 A)  per 1 row  x 2 positions
```

The fetch *count* is the same (six per iteration); only the split changes, and
that is exactly what the timings show.

| kernel, two positions | serial | packed | |
|---|---:|---:|---:|
| gate 1216x768 | 8.83 ms | 7.16 ms | **1.23x** |
| w2 768x1216 | 7.85 ms | 5.86 ms | **1.33x** |
| one FFN layer (2 gate + 2 w2 chunks) | 33.4 ms | 26.1 ms | **1.28x** |
| per position | 16.7 ms | 13.0 ms | |

Two runs, the two sets agree to 0.02 ms.

**It is exact.**  Both kernels reproduce the serial path byte for byte (gate
0/1216 and w2 0/768 q16 values differ, for both positions), and the serial path
is itself 768/768 rows within 1 LSB of a double-precision CPU dot product, so
the 1.28x is not bought with numerics.  Fetch order, the q16 rounding of the
gate output, the swiglu and the final rounding are the same operations in the
same order in both forms.

**Why 1.28x and not 2x.**  These kernels are fetch-count bound, not byte bound.
With the same six fetches per iteration the packed form only wins the share of
time that weight bytes were paying: it halves weight fetches and doubles
activation fetches (four texels instead of two, from a cached 4 KB texture).
At these shapes a weight fetch costs roughly 1.8x an activation fetch, so the
trade nets 1.2-1.3x.  The 240 ms/token streaming figure from step 1 is the
*byte* limit of the fully serial form; the packed form moves about 1.3x of it,
so prefill should land near **240/1.3 ms/token**, not 240/2.

**The P2 layout.**  The packed gate output is a third activation layout: texel
k holds value k of position A in `.rg` and value k of position B in `.ba`, both
q16 (`xformat.h`: `x_pair2_texels`, `x_pair2_glsl`, `x_p2_a`/`x_p2_b`/
`x_p2_stq`).  It is a *target* layout, not an `XFormat` - one texel carries one
value per position rather than two values of one vector, so `x_per_texel` does
not apply and the width is the value count.

One trap, worth the sentence: `x_st2` takes *real* values and multiplies by 256
inside, so a matvec that already accumulates in q16 units must use `x_p2_stq`.
With `x_p2_st` the whole output is 256x too large and every row still looks
plausible - only the probe's CPU reference caught it.

**Next (7c).**  Wiring it into `Session`: the layer loop has to produce both
positions' attention before the packed FFN (attention stays per position and
writes its own KV slot in order), the w2 result comes back as one P2 texel per
row and has to be deinterleaved on the CPU before `add_q16`, and q4 / ACT8 /
non-I16_2 keep the serial path.  The win to expect is the measured FFN share at
1.28x; attention and rmsnorm are untouched.

## Prefill batching, step 3: two positions wired into Session (2026-11-04)

Step 2 proved the packed FFN kernels byte-identical but left them unused.  This
step drives them from the layer loop: `Session::step_pair(tok_a, pos_a, tok_b,
pos_b)` runs one layer for two consecutive prompt positions - attention and
rmsnorm stay per position (they are latency, not weight traffic), the FFN runs
once for both - and the three prefill loops (chat, tool, single prompt) consume
two tokens at a time, leaving the last token on the single-position path
because its logits are the ones the caller needs.

Three new pieces were needed:

| pass | why |
|---|---|
| `pair_pack` | the packed gate reads one texture with A in [0,tw) and B in [tw,2tw), the shape `upload_input2` builds on the CPU; on the GPU it is a texel copy of the two I16_2 inputs, so the bytes cannot drift |
| `matvec_q8_p2_add` | w2 for both positions *with the residual add inside the kernel* (see below) |
| `p2_split` | one P2 row back into one position residual stream; a re-encode of q16 integers, exact by construction |

**The trap that cost the first wiring.**  The serial path adds the residual
before rounding, `enc16(s*ws+res)`; the first version of the pair path rounded
the matvec to q16 and added the residual in a later pass.  Both look equivalent
- the residual is an integer - but in float32 the sum can land on the other side
of a 0.5 boundary: at layer 0, token 0, **2 of 768 q16 values moved by one LSB**,
and that was enough to change the greedy continuation.  Moving the add into the
w2 kernel (two extra fetches per fragment, one per position) restored the exact
expression order, and with it exactness: every hidden state of every token and
layer compares byte-identical between `PREFILL_PAIR=0` and `=1`, the `LOGITS_OUT`
dumps are bit-identical, and the suite keeps the same greedy strings and PPL.

**What it costs.**  Per layer for two positions (`PAIR_PROFILE=1` laps, steady
state): gate 7.3 ms against 8.94 ms for one position, w2 6.3 against 6.73, so
the FFN for a pair is 27.9 ms against 31.4 ms serial.  But the pair path needs
five extra passes per layer (one pack and four splits), and they cost more than
the FFN saves.  End to end, on a 45-token prompt, the second (warm) turn
prefills in **10.75-11.24 s with pairs against 9.96-10.41 s serial**: nine
interleaved pairs, min and median both about 6% worse.

So the wiring ships **opt-in** (`PREFILL_PAIR=1`, default off).  It is exact and
gated, but at B=2 the packing win (4 ms per layer) does not cover the extra
target switches.  The next step is to cut the pass count - keep the pair stream
in P2 and let rmsnorm read it, or have the packed gate read the two activation
textures directly and drop `pair_pack` - or to go to B=4, where one weight fetch
serves four positions.

## Prefill batching, step 4: the pair path pays (2026-11-04)

Step 3 wired the two-position FFN into `Session` and judged it **6% slower**,
pricing the loss to the five extra passes per layer.  The passes were real.
The measurement was also carrying something else: the plain-mode `prefill`
timer starts before the first draw, and on VC4 every fragment program is
linked lazily at first use.  `SHADER_TIME` puts **2.5-4.0 s of a cold prefill**
in `glLinkProgram` (14 programs on the serial path, 20 on the pair path), so
two modes that drive different kernels were being compared including their
different compiler time.  Wall-clock prefill on this board is also noisy: the
Pi Zero runs the Minecraft server, so a 21-token prefill measured 15.9-30.6 s
in one afternoon with the engine unchanged.

Step 4 removes the passes and measures the warm path.

**What changed.**  The FFN input is now *built* in the layout the packed gate
reads instead of being packed into it, and the second w2 chunk reads its
residual from the chunk-0 P2 output (a P2 texel already holds value r of both
positions, so one fetch replaces two):

| FFN stage, two positions | step 3 | step 4 |
|---|---|---|
| FFN input | 2 rmsnorm (4 draws) + `pair_pack` | `rms_stats2` + `rms_norm_pair2` (2 draws, writes P2) |
| gate, both chunks | `matvec_q8_gate2` on the A-half-B-half layout | `matvec_q8_gate2_p2` on P2 (same 4 activation fetches per iteration) |
| w2 chunk 0 | `matvec_q8_p2_add`, residual per position | unchanged |
| w2 chunk 1 | 2 splits, then `matvec_q8_p2_add` | `matvec_q8_p2_add_p2`, residual is the chunk-0 P2, output stays P2 |
| layer output | 2 splits | 2 splits |

Per layer for two positions that is **14 draws against 18** for the same work
on the serial path (19 in step 3).  Four of the fourteen are the packed FFN
kernels; the other ten are attention and rmsnorm, which batching does not
touch.

**Exactness.**  The sum of squares keeps the serial order - 32 partials of 24
values, then p=0..31 - and the normalising pass keeps the same
`R1(X,i)*inv*w[i]` expression and the same `x_st2` encoding, so the q16 stream
into the gate is the same bytes, not merely the same value.  The w2 residual
add stays inside the kernel in the serial expression order (`s*ws+res` before
the encode), which step 3 established is what makes the path bit-exact.  On the
Pi: the identity prompt still answers
`我是一个AI助手，没有自我意识，也没有身体，所以我无法` and the q4 prompt
`我是MiniMind，一个高效的小参数AI模型，专注于`; a ~30-token English prompt
with 24 greedy tokens gives the **same text character for character** with
`PREFILL_PAIR=0` and `=1`, so the KV cache the pair path writes is byte-equal
(one LSB anywhere in a hidden state moves greedy output within a few tokens).

**Warm prefill.**  `PREFILL_REPEAT=n` now re-runs the prefill in one process,
which skips every link and, more importantly, every QPU program generation.
Comparing `n=1` with `n=3` and differencing also removes model load and the
decode step, so what is left is two warm prefills.  CPU (user+sys) per 21-token
prefill, two interleaved runs of each mode:

| mode | run 1 | run 2 |
|---|---:|---:|
| serial `PREFILL_PAIR=0` | 6.81 s | 6.96 s |
| pair `PREFILL_PAIR=1` | 5.74 s | 5.66 s |

**-17% for the same prefill.**  CPU rather than wall time is the metric here:
the process totals are stable to ~2% (the board is single-core and shared),
while wall time is not.  Cold prefill in the same set of runs was 15.9-22.2 s
pair against 18.0-30.6 s serial, but cold numbers are the compiler plus the
board and are not a kernel measurement.

**Where the losses were.**  The five draws step 3 added on top of what the layer
needs - three for the FFN input, two splits after chunk 0 - cost about as much as
the FFN packing won (4 ms per layer), which is why step 3 measured slower.
Removing them, and keeping the second w2 residual in P2 so the splits after
chunk 1 also disappear, makes the pair path cheaper than serial in both kernels
and draw count.  The remaining question is the default: `PREFILL_PAIR` is still
opt-in until the suite has been run with it on (item 7e).


## Prefill batching, step 5: the pair path is the default (2026-11-05)

Step 4 left the two-position path opt-in until the acceptance suite had run
with it on.  This step runs the suite and turns it on: `pair_ok()` now returns
true unless `PREFILL_PAIR=0`, which restores the serial loop.  The eligibility
gate is unchanged (INT8 weights, I16_2 activations, even dim and hidden), so
q4, ACT8=1/2, int16 and fused-w13 models still fall back to the serial loop -
the q4 golden run in the suite is what exercises that fallback with the flag
defaulting on.

**Gate.**  Pi suite on the default build: **ACCEPT**.  PPL 271.2855 and q4
325.3751, exactly the serial numbers (the `--ppl` path steps one token at a
time and never takes the pair path, so it is still the same serial
measurement), q8 and q4 golden strings unchanged, determinism and the poem
check pass.  On the same binary `PREFILL_PAIR=0` prints the same q8 string, so
the escape hatch is verified too.

**Wall time.**  The step-4 CPU win does not become a wall win, and this board
cannot resolve one: a 37-token prompt with `PREFILL_REPEAT=9` (two alternating
runs per mode, repeats 3-9 = 14 warm samples each) gives serial **6.49 s
median / 6.63 s mean** against pair **6.59 / 6.71** (+1.5% median), and the
serial samples alone spread 6.30-7.50 s.  A 109-token prompt (4 warm samples
per mode) gives 19.6 s serial mean against 20.3 s pair, and the gap sits in
the first warm repeat after the cold pass; the later repeats are equal
(19.2/20.7 s serial against 19.3/19.6 s pair).  Samples taken in the first
quarter-hour after a board reboot showed a 4-5% pair deficit that the steady
state does not reproduce, so the defensible statement is **wall neutral within
noise, CPU -15-17%**.

That is still the right default here: the Pi Zero is single-core and shared
with the Minecraft server, so 15% less CPU in the prefill loop is real
capacity and it costs no measurable latency.  The path is bit-exact and
`PREFILL_PAIR=0` is a one-variable escape hatch if a workload ever disagrees.


## Prefill batching, step 6: B=4 priced, and it does not pay (2026-11-05)

Step 5 made the two-position path the default.  The open question was whether a
fourth position per weight fetch is worth building.  It is not, and the probe
now carries the measurement (`engine/probes/ffn2_probe.cpp`, last section).

**The encoding constraint comes first.**  A fragment writes one RGBA8 texel -
two q16 values - and vc4 has no MRT, so four positions of one gate row (four
q16 = 8 bytes) cannot come out of a single draw.  There are exactly two
encodings:

* **two passes of the B=2 kernel.**  The second pass re-fetches the same
  weights, so the work per position is the B=2 work by construction (measured:
  3.60 ms per position either way) and the only cost is one more draw per gate
  chunk (8-19 us, step 7a).
* **one pass writing the four gate values as int8.**  A per-row scale cannot be
  produced in the same fragment either (one texel, no max over the row), so the
  scale has to be fixed; this is the fastest B=4 gate that can exist, so it is
  the upper bound for any B=4 design.

**The int8 gate is fast, and it does not reach the layer.**  Pi, 1216x768 gate,
min of three interleaved blocks of six draws, two runs:

| | per position | four positions, one FFN layer |
|---|---:|---:|
| B=2 (two draws per chunk) | 3.60 ms | 52.5 ms |
| B=4 int8 gate | 2.30 ms (**1.57x**) | 42.0 ms (**1.25x**) |

The gate itself is 1.57x per position, but the layer moves 1.25x because the w2
output is the FFN residual and has to stay q16: four positions of a w2 row are
8 bytes again, so w2 cannot batch past B=2, and 23.6 ms of the 42.0 ms is w2.
Batching w2 would quantise the hidden state itself, which is the model, not an
optimisation.

**What that is worth end to end.**  MiniMind3 is 8 layers at dim 768 / hidden
2432, and one FFN layer is 13.1 ms per position at B=2 (26.1 ms for two), so
the FFN is about 105 ms of a ~244 ms prefill token.  A 1.25x FFN is ~21 ms per
token: **at most 1.10x on prefill**, and only where prefill dominates.

**And the price is exactness.**  The int8 gate is quantised to 1/127 of the row
range, q16 is 1/256 absolute.  On the probe data the gate range is 0.363
(93 q16), so one int8 step is 0.7 q16 LSB and the measured error - rms 0.36
LSB, max 1 LSB, 4864/4864 values within 1 LSB, and the byte stream is not dead
(0 values stuck at the zero code) - *cannot* show the loss.  A real layer has a
gate range of a few units, so a step is 5-20 q16 LSB, and step 7c established
that 2 of 768 values moving by 1 LSB changes the greedy text.  An int8 gate can
therefore never be the default under the byte-identical rule.  **Verdict: B=4
rejected; B=2 stays the exact default and item 7 is closed.**

**The six extra programs, explained.**  Step 5 left one thing open: a ~1.5-2 s
gap in the first warm prefill repeat at 109 tokens.  It is not a warm-path cost.
`SHADER_TIME=1` on the deployed binary counts the programs: **20 in pair mode
against 14 serial**, and the six pair-only ones are exactly what step 4 added -
`xrms2s_768_x16_2`, `xrms2n_768_x16_2`, `xq8g2p2_1216_768`,
`xq8p2a_768_1216_x16_2`, `xq8p2a2_768_1216`, `xp2sp_768_x16_2`, 681 ms of link
time between them (link total 2898 against 1821 ms).  Cold prefill on a
10-token prompt differs by 1.7 s (7.16 against 5.46 s); the warm repeats on the
same runs are 2.49/2.53 s against 2.41/2.40 s, **+3.5%**, so the warm path has
no structural gap.  The 7e number was the compiler state of the first process
after the reboot: the same pair binary took 17.1 s cold on its first run and
7.2 s on the next, with one program link falling from 2792 to 301 ms.  Cold
prefill is not a kernel measurement on this board; the six programs are a
first-token cost, not a warm one.


## q4 parity, step 1: where the 1.5x lives (2026-11-05)

Attribution before any rewrite.  Same board, same prompt, q8 and q4 MiniMind3
checkpoints, `LAYER_PROFILE=1` (a `glFinish` after every pass) for 12 greedy
tokens; each number is the sum over the 8 layers of one token, so it is the
same quantity as a decode ms/token figure (the 252 / 388 matrix pair is from
a different clock state, see below).

| stage | q8 | q4 | q4/q8 |
|---|---:|---:|---:|
| rms_att | 17.64 | 17.98 | 1.02 |
| attn (qkv + 4 passes) | 101.56 | 116.47 | 1.15 |
| wo | 19.98 | 26.20 | 1.31 |
| rms_ffn | 17.34 | 18.06 | 1.04 |
| ffn gate | 67.53 | 137.03 | 2.03 |
| w2 | 50.03 | 68.25 | 1.36 |
| **sum** | **273.4** | **383.9** | **1.40** |

The laps add a constant sync cost per pass, so both columns sum above the
unprofiled 188 / 290 ms measured on the same board at a 1-token prompt; the
deltas are the measurement, not the absolutes.  **The FFN gate is 63% of the
gap** (+69.5 of +109.9 ms), w2 is 17% (+18.2) and the whole attention block is
14% (+14.9).  Inside attention the per-pass laps put all of that difference on
the qkv matvec - 5.19 against 3.65 ms per layer (1.42x) - while rope, qk-norm,
q-rot, scores, stats and v stay within 0.06 ms per layer.  That is structural:
q8 draws two qkv rows per fragment (`matvec_q8_pair_logits`), q4 falls back to
the one-row `matvec_i4`.

**Live decode on the same binary.**  Plain mode, 32 greedy tokens, `SEQ=768`,
three interleaved runs each: q8 195.5 / 196.7 / 185.8 and q4 291.7 / 290.9 /
304.1 ms/token (median 195.5 against 291.7, **1.49x**; best 185.8 against
290.9).  The profile delta of +109.9 ms predicts the live +96 ms gap within the
sync constant, which is about 90 ms/token in both columns - that is why the
profile ratio (1.40) is lower than the live one.  The 2026-11-02 matrix quotes
251.8 / 387.6 ms/token for the same models; today both are ~26% faster while
prefill matches it within 2% (10.73 against 10.95 s), and `vcgencmd
measure_clock core` reads 300 MHz, so the matrix looks like a lower GPU clock
state - decode is GPU-bound and prefill is CPU-bound.  The ratio and the
per-stage deltas carry over; the absolute matrix numbers do not.

**Draws are not the problem.**  `DRAWCOUNT` was dead instrumentation until this
step - `draws_` was never incremented - and wired up it gives **159 draws per
decode token for q8 and 143 for q4** (a gen-1 against gen-13 difference, so
prefill drops out): the q4 path issues fewer draws.

**The unpack is 23% of the gate, and removing it does not reach parity.**
`Q4_NOUNPACK=1` (wrong on purpose: the eight nibble taps become eight plain
multiplies, the same four weight fetches stay) prices the unpack on the shipped
binary: gate 137.03 -> 105.0 ms, w2 68.25 -> 56.5.  Under the old float decode
(`Q4_SHIFT=0`) the same probe gives 169.5 -> 105.0, so the integer split is
worth 32 ms of the gate (23%) and the float form costs 64.5 ms (38%).  This
probe only became meaningful in this step: while `Q4_SHIFT` was on it took the
shift branch first and silently did nothing.
At **zero** unpack cost the q4 gate is still 105 ms against q8 67.5, with 25%
fewer texture fetches per fragment (864 against 1152) and the same 3072 MACs.
The rest of the gap is therefore the inner-loop shape: q4 builds eight scalar
taps and an eight-term scalar dot per weight texel, q8 keeps the dot in vec4 lanes
and a four-term dot.  The QPU is 4-wide, so the vec4 form can issue four MACs
per instruction slot while the taps force scalar ops.  That is the hypothesis
step 2 tests, not yet a result.

The per-group min term is not a lever: `Q4_NOMIN=1` measured -4.5% in one run
and +6% in another, i.e. the re-scheduled shader moves more than the
arithmetic.

**Next step (item 8, step 2):** rewrite `matvec_i4_pair_swiglu` to build the
taps in vec4 form (vector floor/shift where the split allows, `dot()` over
vec4) and compare the gate bytes against the current kernel with
`q4mv_probe` before wiring it in - the output is q16 and step 7c showed that a
1-LSB move in 2 of 768 values is enough to change the greedy text.  The qkv
row pairing (the q8 paired kernel applied to q4) is the second, smaller lever
at +14.9 ms.
## q4 parity step 8b: the vec4 gate is not faster, and the gate is draw-bound (2026-11-05)

> **Partly withdrawn by 8c (2026-11-05).**  The 2432-row point in this section
> was a draw over an incomplete texture: this driver caps textures at 2048 per
> side, the 2432-row weight upload fails with `GL_INVALID_VALUE`, and the shader
> sampled nothing.  The "one 2432-row draw is cheaper than two 1216-row draws",
> the "14.4 against 7.7 us per fragment" occupancy cliff and the 21% vec4
> penalty at that size are all artifacts of that.  The vec4 penalty at the
> *valid* sizes (608 and 304 rows) stands, and so does the conclusion that the
> vec4 rewrite is not the lever.  Corrected sweep and the real per-MAC
> comparison are in the 8c section.

**The 8a hypothesis is dead.**  `matvec_i4_pair_swiglu` now has a vec4 form
(`Q4_MV4=1`, default off): the four texel bytes stay a `vec4`, the nibble split
is an `ivec4` shift/mask, the eight activations decode into even/odd vec4 halves,
and each weight texel accumulates `lo*xe+hi*xo` into four vec4 accumulators that
are flattened once per 32-weight group.  At 2432x768 the bytes match -
`q4mv_probe` prints `hash 302bc5c5` for both forms - and the dot itself cannot
move a bit by construction: taps are integers in [0,15] and activations are raw
q16 integers, so every product and every partial sum of eight
(8 x 15 x 32767 < 2^24) is an exact fp32 integer.

**Exact at the model shape, not in general.**  The sweep shows the bytes are
not universally identical: at 608 rows one q16 value in 608 differs (bytes
202-203, 17151 against 17279 units) and at 1216 rows another does.  The dot
cannot be the source - every product and partial sum is an exact integer - so
the difference has to sit in the final scaled accumulate, where
`sa1*d1 + ma1*xsum` may be contracted to an FMA in one form and not the other.
A 1-ULP difference there flips a `qz()` boundary, and a large gate value g turns
that one-unit flip into about g/65536 = 0.5 of output, which is the step the
dump shows.  That is the 7c trap again: the gate is q16, and one value in 608 is
enough to move the greedy text.  Default off is therefore not caution, it is the
measurement.

It is **21% slower**: 11.35 against 9.39 ms per 2432-row gate draw, and 3.58
against 3.05 at 304 rows.  The compiler was already vectorising the scalar taps
(four components of one texel, same op), so the explicit vec4 form added nothing
but four live accumulators.  **The gate is not ALU-bound**, so the 8a reading -
eight scalar taps against four vec4 lanes - was wrong about where the time goes.

**What the gate is: under-occupied per draw.**  The same probe with the scalar
kernel, sweeping the row count, in two regimes: 20 pipelined draws with one
`glFinish`, and one `glFinish` per draw (the engine regime).  The two regimes
agree, so there is no pipelining win to recover; a second build after a board
reboot reproduced all four points within 1%.

| rows | fragments | ms/draw pipelined | ms/draw per-draw sync |
|---:|---:|---:|---:|
| 304 | 152 | 3.05 | 3.02 |
| 608 | 304 | 5.18 | 5.24 |
| 1216 | 608 | 8.71 | 8.73 |
| 2432 | 1216 | 9.39 | 9.37 |

Doubling from 608 to 1216 fragments costs **0.6 ms**, and **one 2432-row draw
(9.39 ms) is cheaper than the two 1216-row draws the engine issues per layer
(2 x 8.71 = 17.4 ms)**.  The profile agrees: the gate stage is 137.03 ms over 8
layers = 17.1 ms/layer, i.e. two 608-fragment draws at 8.7 ms each.  Most of the
2.03x q4/q8 gate ratio is therefore not arithmetic: it is the chunk split
doubling the draw count at a fragment count where this kernel runs at half rate
(14.4 us/fragment at 608 against 7.7 at 1216, which is where the q8 kernel
already sits at 608).

**Live on the shipped build (Q4_MV4 unset).**  32 greedy tokens, `SEQ=768`,
three interleaved runs against `bin/engine_base` with this board just rebooted:
q8 230.7 -> 192.6 median ms/token, q4 377.5 -> 285.9 (-19.2% / -24.3%).  Those
deltas are the cumulative work since the stored baseline, not this step: the
default gate program is byte-identical to the previous build, so this step moves
no default timing at all.  q4 today is 285.9 ms/token against q8 192.6, a 1.48x
ratio that still sits where 8a left it.

**Next step (8c): one gate draw per layer.**  `split_ffn()` splits the FFN at
hidden > 2048 because the *non-fused* two-pass path needs a 2432-wide target;
the fused gate only needs a 1216-texel I16_2 output, which fits.  Plan: keep
w1/w3 full width, draw the gate once (2432 rows, 9.39 ms), and let the two
column-sliced w2 passes read the two halves of that one 1216-wide texture through
an x-offset uniform.  w2 keeps its group order, residual chain and q16 rounding,
and the gate is row-independent, so the path stays byte-identical; expected
-8 ms/layer, about -62 ms/token, gate 2.03x -> ~1.1x.

## q4 parity step 8c: the one-draw gate cannot exist on this driver (2026-11-05)

**The 8b plan is dead at the hardware limit, and 8b itself was measuring an
empty texture.**  8c tried to keep w1/w3 full width and draw the fused q4 gate
once at 2432 rows into a 1216-texel target, with the two w2 column slices
reading the halves through an x-offset.  The engine came up with a gate of
**zeros**: `FFN_DUMP` printed `ha` as eight `00000000` texels at layer 0 and the
hidden stream diverged from the first layer on.

**Root cause: this driver has `GL_MAX_TEXTURE_SIZE` 2048 and a q4 weight matrix
costs one texel row per output row.**  `texsize_probe` on the board today:
`h=2048 -> err=0x0`, `h=2432 -> GL_INVALID_VALUE (0x501)`.  So a 2432-row w1
texture is never allocated, the sampler is incomplete, every fetch returns 0,
and `sl(0,0)` writes a zero gate.  The engine cannot merge the two gate chunks
without a two-sampler kernel that selects the texture by row; the texture-side
merge is not available at all.

**That also voids the 8b measurement.**  The row sweep in `q4mv_probe` went to
2432 without checking `glTexImage2D`, so its 2432-row point was a draw over an
incomplete texture: 9.39 ms was an empty shader run, and `hash 302bc5c5`,
identical in the scalar and vec4 forms, was uninitialised target memory - two
empty draws hash the same because neither reads a weight.  The 8b claims built
on that point are withdrawn: one 2432-row draw is **not** cheaper than two
1216-row draws, and the "14.4 against 7.7 us per fragment" occupancy cliff does
not exist.

**Corrected sweep** (`q4mv_probe`, upload error checked, `nz` = non-zero bytes
read back; 20 pipelined draws and one sync per draw agree):

| rows | fragments | ms/draw pipelined | ms/draw per-draw sync | nz bytes |
|---:|---:|---:|---:|---:|
| 304 | 152 | 2.98 | 3.01 | 251 / 608 |
| 608 | 304 | 5.12 | 5.14 | 531 / 1216 |
| 1216 | 608 | 8.68 | 8.71 | 1029 / 2432 |
| 2048 | 1024 | 13.81 | 13.58 | 1748 / 4096 |

The per-fragment cost is flat once past the smallest size: 16.9, 14.3, 13.3 us
at 304, 608, 1024 fragments.  Fitting the top two points gives **1.6 ms per
draw + 11.7 us per fragment**, so a hypothetical merged 1216-fragment draw
would cost about 15.8 ms against the 17.4 ms the engine pays for its two
608-fragment draws: **1.6 ms/layer, about 13 ms/token**, and it needs the
two-sampler kernel.  That is not worth it while a structural lever exists.

**Where the 2.03x actually lives: ALU per MAC.**  Each fragment does four
768-wide dots (w1 and w3 for two rows) = 3072 MACs.  The q4 gate runs at
3072 / 11.7 us = **263 MMAC/s**; the q8 gate stage (67.53 ms over 8 layers,
2 draws of 608 fragments each) runs at 3072 / 6.94 us = **442 MMAC/s**.  q4 is
59% of q8 per MAC, with the same fetch count per fragment and the same output
shape.  The remaining q4 gate cost is the unpack plus the eight-tap scalar dot,
and 8b already showed the vec4 rewrite does not help - the compiler vectorises
the taps on its own.

**Probe fix (this commit).**  `q4mv_probe` now clears and checks `glGetError`
around both uploads, prints `SKIP upload err ...` instead of timing an
unallocated texture, stops the sweep at 2048 rows, and reports `nz` so a
zero/garbage read-back is visible in the same line as the timing.  The engine
changes for 8c were reverted; the shipped binary is byte-identical to 8b
(`1183f09`), golden q8/q4/poem strings re-checked on the board.

## q4 parity step 8d: the qkv row pair is not bit-exact, and not faster (2026-11-05)

**The 8c hand-off put +14.9 ms/token on the q4 qkv draw because q8 renders two
output rows per fragment and q4 does not.  The q4 pair kernel was built and
measured: it moves the q4 PPL gate and it is slower.  Dead on both counts.**

`matvec_i4_pair_logits(rows, cols, xf)` is the q4 twin of
`matvec_q8_pair_logits`: one weight texel row per output row, rows 2t and 2t+1
in one fragment, both `P` (scale/min) rows fetched once per group, one
activation decode and one per-group activation sum shared, and the two results
written as the packed q16 pair the paired reader already expects.  Per row the
accumulator keeps `matvec_i4` order exactly (`sc*(d0+d1+d2+d3)+mn*xsum`, taps
from the same `q4_taps`), and the taps are exact integers below 2^24, so the
only possible difference is the compiler's reassociation of the final float
combine.

**Not bit-exact.**  A new `QKV_DUMP` probe (below) shows the decoded per-row
q16 integers identical for the first **65 positions x 8 layers**, so the
kernel is functionally right - and yet the q4 PPL at `SEQ=768` moves
**325.3751 -> 331.6324 (+1.92%, gate is 1%)**, while the 16-token golden string
survives.  That combination is the whole argument for keeping both a hash probe
and PPL in the loop: the greedy string is not sensitive enough to see a
1-LSB event, and the hash over the first two tokens is not long enough.
This is the same rare rounding in a two-row fragment that the 2026-11-03
attempt hit; it is not a derivation bug and re-writing the expression does not
move it.

**Slower too.**  Three interleaved plain-mode pairs of 32 greedy tokens on the
same binary (`QKV_PAIR=0` against `QKV_PAIR=1`, MiniMind3 q4, `GPU_LAYER=1
KV_PACK=16 SEQ=768`):

| run | pair off | pair on |
|---|---:|---:|
| 1 | 285.6 | 288.0 |
| 2 | 285.4 | 290.8 |
| 3 | 287.1 | 289.1 |
| median | **285.6** | **289.1** |

+1.2% slower.  q4 is ALU bound (263 against q8 442 MMAC/s, step 8c), so
sharing one activation decode and one per-group sum does not pay for the
doubled `P` fetches and the doubled live accumulators; the draw count that
pairing saves (two strips of a 1536-row pass against one 768-fragment draw) is
worth ~0.2-0.4 ms, exactly as the 2026-11-03 attempt found.  The row-pair family is
now exhausted on q4 - gate+swiglu -2.0%, qkv -0.4% then, +1.2% now, wo/w2
already paired - and the qkv gap in the 8a attribution is **not** a pairing
artifact: q8 pairs **and** is faster because its kernel is fetch-bound, while
q4's is bound by the unpack ALU.

**Landed: the probe, not the kernel.**  `QKV_DUMP=1` decodes the qkv target
back to per-row q16 integers and prints a hash for every layer/position;
`QKV_DUMP=2` also prints the values for the first two positions.  The decode is
layout-agnostic - the one-row 24-bit texel and the packed q16 pair decode to the
same integers - so `QKV_PAIR=0` and `QKV_PAIR=1` runs of the same model can be
diffed line by line, which is how a 1-LSB divergence gets located to a layer
and a position.  The engine default is untouched: the q4 pair path is removed
and the q8 pair path is byte-identical to the previous build.

**Trap for the next person in `attention_block`:** the first cut of the probe
was spliced in ahead of the `kv_rope_write(layer, pos)` line and deleted it, so
the KV cache was never written, every model answered with garbage, and nothing
was logged - the golden strings caught it at once.  Insert debug code around
that call, never over it, and keep the suite in the loop.

## 8-bit KV step 9b: the V grid is what breaks q4, and a per-head grid fixes it (2026-11-05)

Item 9a was already landed before this run: `71cc5c5` made the prefix-cache
header version 2 and pins `KV_SCALE` (section above), so there was nothing left
to do there.  This step prices the other half of item 9: a per-head (64-value)
scale, so the 8-bit grid normalises each KV head instead of the whole tensor.

**The tool is a probe, not a kernel.**  `KV8_SIM` (default off) quantises in the
CPU attention path: with `GPU_LAYER=0` the cache holds float K/V and attention
consumes dequantised values, so quantising at read time is exactly what the GPU
does at write and read.  Modes: 1 global K+V, 2 per-head K+V, 3 global K only,
4 global V only, 5 per-head V only, 6 per-head K only; `KV8_SIM_SCALE` is the
global grid (16 = the `KV_PACK=8` default) and `KV8_STATS=1` prints the
per-head maxima at position 0.

**The CPU path is a fair price for the grid.**  Its float-KV PPL lands on the
GPU KV16 gates (272.1821 against 271.2855 for q8, 327.0226 against 325.3751 for
q4), and q4 reacts to the global 1/16 grid the way the GPU does (+10.3% here,
+8.0% there).  One caveat up front: q8 does **not** react the same way - the
GPU gains 4.0% at that grid while the CPU gains only 0.18% - so the GPU q8 kv8
result is not a pure grid effect (it interacts with the GPU q16 numbering).
The numbers below price *the grid*; a GPU per-head kernel still has to be
measured.

**Full matrix, 293-token corpus, `SEQ=768`, CPU attention (percent against the
same model float-KV line):**

| model | float KV | global 1/16 K+V | global K only | global V only | per-head K+V | per-head V only |
|---|---:|---:|---:|---:|---:|---:|
| q8 | 272.1821 | 271.6947 (-0.18%) | 272.4584 (+0.10%) | 273.1938 (+0.37%) | 270.4710 (-0.63%) | **266.8795 (-1.95%)** |
| q4 | 327.0226 | 360.7004 (+10.3%) | 320.8428 (-1.9%) | 352.9592 (+7.9%) | 321.1931 (-1.8%) | 322.9631 (-1.2%) |

**Reading: K fits the global grid, V does not.**  `KV8_STATS` at position 0
gives per-head maxima of 3.0-6.1 for K on both models, 1.2-2.7 for q8 V and
0.27-3.5 for q4 V with most heads under 0.8.  At a 1/16 step a q4 V head spans
9-26 of the 256 codes (the typical head about 16) while a K head spans 96-192;
K is qk-normalised before rope (per-head RMS 1, max ~4-6), so the global grid
is the right shape for it.  That is why global K is neutral-to-good (q4 -1.9%,
q8 +0.10%) and global V is the whole disaster (q4 +7.9%): one absolute grid
spends its range on K and starves V.  A max-derived per-head grid gives every
head 127 levels, and V per head alone moves q4 from +7.9% to -1.2% and beats
the float-V line on q8 by 1.95%.

**Why per-head V beats float V.**  Both models prefer the quantised V to the
exact one (q8 -1.95%, q4 -1.2%): a per-head max grid puts a step of max/127 on
each head, so the noise is a fixed fraction of that head range (half a step =
1/254 of its max), while the global 1/16 step is 2-20% of a typical V value and
larger than the smallest head range.  That is regularisation, not precision:
the global 1/16 grid on q8 is only -0.18% and on q4 is +10.3%, and the earlier
scan showed the curve is non-monotonic in scale (q8 prefers 1/16 to both 1/8
and 1/32).  Treat the per-head result as a measured property of this model pair,
not as a law.

**What a GPU implementation would cost.**  K keeps the global grid; only V needs
the per-head scale.  The V write needs the max of the 64 values before the
texels are written, so one extra pass per layer per token (4 fragments scanning
64 q16 values, the shape of `att_qnorm_scale`) writes one scale texel per row,
and `att_v` reads one scale per position per head: two per loop iteration
against the four V fetches it already makes, or one if two scales share a q16
texel.  That is a small pass plus a fetch increase in one stage, not a new
format; the 4.5 MiB cache and the 2:1 packing are unchanged.

**Verdict.**  Per-head scaling is the first thing in this campaign that makes
8-bit KV look usable on q4 (+7.9% -> -1.2% on the grid that matters), and on
q8 the V side alone is worth 1.95%.  It is not free, so `KV_PACK=8` stays
opt-in until the GPU version is measured.  Next: per-head V scale in
`att_kv_rope`/`att_v` with K on the global grid, then the q8/q4 PPL pair and
the decode cost.

## 8-bit KV step 9c: per-head V on the GPU helps q4 and costs q8 its gain (2026-11-05)

The GPU version of the 9b per-head V grid, behind `KV_VSTEP=1` (default off).
K keeps the global `KVSCALE` grid: it is qk-normed before rope, so its range
is already about unit RMS, while V head ranges differ by an order of
magnitude.  New `att_vscale` pass, one fragment per KV head and one texel per
position **per layer**, scanning the head (`HEADDIM` q16 values) and writing
`step = max/127` as a 24-bit fixed point value (scale 2^24, so the step is
exact to ~1e-5 relative even at 0.005).  `att_kv_rope` quantises V with that
step and `att_v` dequantises with it, one extra fetch per position in the V
loop (all channels of a fragment share the head).  The plane is
`kv_heads x seq` per layer; the prefix-cache header goes to version 3 only
when the plane exists, so a default build still writes version 2 files
(verified: `KV_PACK=8` PPL is 260.4460 / 351.2603 before and after).

**PPL, `SEQ=768`, 293-token corpus, greedy gates ACCEPT:**

| model | KV16 (default) | kv8 global 1/16 | kv8 + per-head V step |
|---|---:|---:|---:|
| q8 | 271.2855 | **260.4460** (-4.0%) | 277.2030 (+2.2% against KV16, +6.4% against kv8) |
| q4 | 325.3751 | 351.2603 (+8.0%) | **330.3682** (+1.5% against KV16, -6.0% against kv8) |

**Reading: the per-head grid is not a free win on the GPU, and q8 loses the
most.**  The CPU sim (9b) predicted -1.95% for q8 and -1.2% for q4 against
float KV; the GPU reproduces the q4 direction (-6.0% against the global grid,
which is most of the sim gain) but not the q8 one: q8 goes from -4.0% to +2.2%
against KV16.  That is the same asymmetry 9b found from the other side (GPU
q8 kv8 gains 4.0% at that grid, the CPU gains 0.18%), so the q8 kv8 gain is
bound to the specific global 1/16 grid on this engine, not to "8-bit KV" in
general: moving V onto a per-head grid gives the gain back.  q4 behaves like
the sim says it should and is now within 1.5% of KV16 while using half the
cache.

**Cost: one dependent draw per layer per token.**  `LAYER_PROFILE` with its
per-lap `glFinish` (35 tokens x 8 layers, so sync-inflated): vscale 2.0,
rope 4.5, qkv 4.4, scores 7.7, v 2.7 ms/layer.  End-to-end, `KV_PACK=8`,
32 greedy tokens, `IGNORE_EOS=1`, interleaved arms:

| model | `KV_VSTEP=0` median (min) | `KV_VSTEP=1` median (min) | delta |
|---|---:|---:|---:|
| q8 | 190.8 (184.1) | 199.4 (196.1) | +4.5% |
| q4 | 287.9 (287.8) | 304.0 (300.6) | +5.6% |

q4 pays more because its decode is ALU bound; the pass is latency bound
(4 fragments x 96 serial fetches, the shape of `att_qnorm_scale`, which costs
1.0 ms/layer with twice the fragments), so splitting the scan across
partials would halve it, but nothing here is worth that while the grid loses
on q8.

**Verdict: `KV_VSTEP` stays opt-in and `KV_PACK=8` stays opt-in.**  The
per-head grid is a real q4 improvement (-6.0%) but it still does not reach
KV16, and it costs q8 the only thing that made 8-bit KV attractive (-4.0%).
The GPU path is therefore: q8 keeps the global 1/16 grid and remains the one
kv8 configuration worth deploying; q4 stays on KV16.

**Two traps for the next person.**  (1) The step plane must be per layer.  A
single `kv_heads x seq` plane is overwritten by the next layer before the
previous layer attention reads its rows back: the first cut scored 660.2028
on q8 and looked like a broken kernel rather than an aliasing bug.  (2) At
position 0 the attention output is exactly V, and with GQA (8 query heads, 4
KV heads) query heads 0-1 both print KV head 0, so an `att_out` dump at pos 0
is a clean absolute check of encode + step + decode (it matched the qkv V to
the last q16) but says nothing about the per-position path - the aliasing
bug only shows up at pos >= 1.

Observed greedy answers with `KV_PACK=8 KV_VSTEP=1` (unpinned, the kv8 answer
moves with the grid): q8 `我是一个AI助手，我的存在是为了提供帮助和信息。`, q4
`我是Ministry，一个由Oracle公司开发的高级软件工程师。我以高效、`.

## Prefix cache step 10: the file was right, the CLI sampled from an empty logits vector (2026-11-05)

Item 9d left this open: build a cache with `--build-prefix-cache`, reload it
with `--prefix-cache`, and the answer changed in every mode, KV16 included.
This step shows the rows and the header were never the problem: the CLI
skipped every prompt token and then sampled from a logits vector that had
never been filled.

**Repro.**  The cache prompt is the exact rendered ChatML text for
`用一句话介绍你自己。` (36 tokens, `SEQ=768`, `KV_PACK=16`).  Reloaded, the run
prints `PREFIX_CACHE loaded 36 tokens` and
`Assistant: <|endoftext|>，我叫[你的名字]。` against the plain
`我是一个AI助手，没有自我意识，也没有身体，所以我无法`.  Two one-run probes
split the difference: a cache built from only the first 19 of the 36 tokens
gives the byte-identical plain answer, so the rows and the resumed position
are right; the same 36-token cache under `--worker` reports `cached_tokens=36,
prefill_tokens=1` and the plain answer.

**Cause.**  `cache_prefix_len()` in the CLI `run_turn` and `run_tool_turn`
returns the length of the common token prefix, and a fully covered prompt
returns `tokens.size()`: the prefill loop then ran zero steps, `logits` stayed
empty, and `argmax_token` is `max_element(begin, end) - begin` = index 0 =
`<|endoftext|>`, which was printed and stepped.  The worker path has clamped
`prefill_from` to `tokens.size() - 1` since the cache landed; the CLI never
got the same clamp, and the server warmup sends exactly the cache prompt, so
it sat on the same hole.

**Fix.**  One clamp in `cache_prefix_len` (both CLI turns go through it): a
fully covered prompt runs its last token at its own position.  That rewrites
the row the cache already holds with the same bytes - the encode is a
deterministic function of the prefix - and produces the logits.  It costs one
step (~0.19-0.23 s) and only when the prompt is entirely cached.

**Round-trip probe.**  `PREFIX_CACHE_RESAVE=<path>` (default off) writes the
just-loaded rows back out through `save_prefix_cache()`, so a load can be
compared with the build byte for byte:

| build | file | rows | width | resave vs build |
|---|---:|---:|---:|---|
| q8 KV16 | 442,592 B | 36 | 384 | identical |
| q4 KV16 | 1,204,696 B | 98 | 384 | identical |
| q8 `KV_PACK=8` | 221,408 B | 36 | 192 | identical |
| q8 `KV_PACK=8 KV_VSTEP=1` (header v3 + V plane) | 226,020 B | 36 | 192 | identical |

Full-cover answers match plain in all four modes (q8 `我是一个AI助手…`, q4
`我是MiniMind…`, kv8 `我是MiniMind，一个高效的小参数AI模型。`, kv8+vstep
`我是一个AI助手，我的存在是为了提供帮助和信息。`), and the 98-token prompt
returns `我是一个由jingyaogong创建的高效小参数AI模型` from both paths on q8
and q4.  The v3 file is 4,612 B larger than the kv8 file on the same 36 rows:
4,608 B of `kv_heads x seq` step plane plus the 4-byte `vsw` header field.

**What it buys.**  98-token ChatML prompt, `TIMING=1`, `--gen 1`, q8, three
interleaved pairs:

| arm | prefill |
|---|---:|
| plain | 24.7 / 33.7 / 22.6 s (median 24.7) |
| `--prefix-cache` | 13.5 / 13.9 / 14.3 s (median 13.9) |

So ~10.8 s for the 97 tokens (~111 ms/token, the pair path paying two
positions per weight pass).  The cached arm does not fall to zero, and the
floor is not the prefix: a two-character prompt with `--gen 0` costs 12.1 s of
prefill, so what is left is the first-token setup of the chat path.  In that
token 19 programs link lazily (`SHADER_TIME`), with a link total that depends
on the driver shader cache and board load (2.7 s in a clean run, 7.6 s in a
cold and contended one).  Plain prefill is the noisy arm (22.6-33.7 s) because
the board shares its single core with the Minecraft server; the cached arm is
stable because its work is one token plus that setup.  Removing the ~12 s
floor is the next item, not something the cache can do.

**Trap, unchanged:** `--build-prefix-cache` and `--worker` force `SEQ=1024`
when the model is smaller and `SEQ` is unset, while `--chat` keeps the model
seq.  A cache built without `SEQ` loads in worker mode and is ignored in chat
mode.  Pass `SEQ` on both sides.

**Verdict:** item 10 closed.  The disk prefix cache round-trips byte for byte
in all four formats and the CLI now behaves like the worker: one re-seated
token, then generation.  The regression suite on the deployed build is ACCEPT
(q8 PPL 271.2855, q4 PPL 325.3751, greedy strings unchanged).

## First token, step 11: the upload conversion and one vertex shader (2026-11-05)

The first token has three parts on this board: startup (checkpoint map, EGL,
weight upload), the lazy shader work of the first prefill, and the prefill
itself.  `UPLOAD_TIME=1` now prints the upload wall and the time spent inside
`make_texture`, so the split reads from one run.

**Warm start (checkpoint in the page cache), q8, `SEQ=768`, `KV_PACK=16`,
36-token ChatML prompt, `SHADER_TIME=1 UPLOAD_TIME=1`:**

| phase | value |
|---|---:|
| model loaded (borrowed mmap) | 28 ms |
| EGL context (`GPU gpu;`) | 540-700 ms |
| FFN split | 90 ms |
| weight upload (69 textures, 69 MB) | 1.7-2.6 s, of which 0.9-1.1 s inside `make_texture` |
| Session constructor | 120 ms |
| session ready (cumulative) | 2.6-3.5 s |
| first prefill (19 programs) | 9.3-10.6 s |
| second prefill, same process | 3.2 s |

So ~6 s of the first token is one-time shader work: 2.6-3.5 s of GLSL compile
and link, ~3 s of QPU code generation at each program first draw.  Cold
(checkpoint pages dropped) startup adds the 6-7 s SD read of the 69 MB file;
`MADV_WILLNEED` at mmap time already overlaps what it can with EGL init.

**1. INT8 -> uint8 upload conversion, four bytes at a time.**  The +128 bias is
a bit flip, `(v + 128) & 255 == v ^ 0x80`, so the 69 MB conversion is a 32-bit
XOR (`UPLOAD_XOR=0` restores the byte loop).  Warm, four interleaved runs per
arm with `cat` of the checkpoint before each: upload wall median **2.68 s byte
loop against 1.74 s XOR (-0.9 s)**.  All of it is the conversion (wall minus
`make_texture`: 1.8 -> 0.85 s); the texture time itself (0.9-1.1 s) does not
move.  The texture bytes are identical by construction, so the numeric path
cannot drift - the suite below confirms it.

**2. One vertex shader for every program.**  `link_program` compiled the same
full-screen-quad vertex shader once per program because each
`compile_shader()` creates a new object; it is now compiled once per language
(the q4 shift kernels take the ESSL 3.00 form) and attached to all of them.
Link sum over 19 programs, four interleaved runs per binary: **3200 ms median
before, 2967 ms after** (~5 ms per link).

**Rejected: dropping the per-texture `glFinish()`.**  69 uploads serialised
against the GPU looked promising.  It is not: warm, the upload is 1.65-2.04 s
with and without the finish; cold (checkpoint dropped with
`dd iflag=nocache`) both arms scatter 10.3-18.7 s with no order in them.  The
finish stays.

**What is left is the driver, not the engine.**  vc4 has no usable shader
cache: Mesa disk cache is a no-op here (the directory stays empty) and
`glGetProgramBinaryOES` reports length 0 and raises `GL_INVALID_OPERATION`
(see startup part 1 above).  Every process compiles its 19 programs again, so
the only way to keep that off a user request is a warm process - the server
does it with `--warmup`, one 1-token generation at startup.

**Verdict.**  Item 11 lands ~1 s of the first token with the draw path
untouched: upload conversion 0.9 s, shared vertex shader ~0.1 s.  Suite on the
deployed build: **ACCEPT** - determinism, q8 `我是一个AI助手，没有自我意识，也没有身体，所以我无法`,
q4 `我是MiniMind，一个高效的小参数AI模型，专注于`, poem OK, PPL 271.2855 /
325.3751 (both exact).  Plain decode A/B against the pre-step binary, two
interleaved runs of three, min / median ms per token: A 191.8 / 196.4 and
191.9 / 193.7, B 185.8 / 187.4 and 195.2 / 198.0 - no resolvable difference,
and none is possible from a texture upload and a link-time shader object.  The
remaining ~6 s is vc4 code generation; the 19-program count is the only lever
left.

## First token, step 12: the q8u8 container (2026-11-05)

The MMCF container stores q8 matrices pre-biased (`docs/checkpoint-container.md`),
so `GPU::upload` memcpys them into the texture instead of running the +128 pass
over 69 MB.  Upload on the Pi, `UPLOAD_TIME=1`, one token generated, warm page
cache, two runs each:

| model file | upload wall | inside `make_texture` |
|---|---:|---:|
| `minimind3.q8` (legacy, 32-bit XOR conversion) | 6.85 s / 5.90 s | 1439 / 1117 ms |
| `minimind3.q8.mmcf` (q8u8, straight copy) | **5.44 s / 5.32 s** | 1034 / 1334 ms |

The 0.5-1.4 s difference is the conversion loop plus the 69 MB temporary buffer
that has to be allocated, filled and freed while the board is page-cache tight;
`make_texture` itself does not move (the bytes still cross the same path).

Everything else in the container is a correctness wash, verified on both
machines: q8 270.3170 (Xe) / 271.2855 (Pi), q4 330.5302 / 325.3751, i16 280.2156
(Xe), and the pinned greedy strings for each machine.

## TinyBrainBot on VC4 (item 12, first look)

Same binary, same container (`models/tinybrainbot/tinybrainbot.q4.mmcf`),
prompt `Hello, who are you?`, `--gen 16 --greedy`: the CPU path
(`GPU_LAYER=0`) answers `<|end|>Here is a response based on the given
information:` style text, the GPU path (`GPU_LAYER=1`) answers
`<|end|>:belive` / `<|end|>:belive:belive, the truth`.  TinyBrainBot q8 is not
a usable correctness signal: both paths degenerate to `future future ...` on
this prompt (the single global q8 scale, item 15), so the correctness work
here is q4.  The shape table that differs from MiniMind3:
dim 768, hidden 2048 (chunk 1024), 12 layers, heads 12 / kv 4 / head_dim 64,
vocab 32000, qk_norm 0, seq 512, theta 10000.

**The layer probes now work on both paths.**  `HIDDEN_DUMP=1` printed only on
the GPU path.  `GPU::dump_cpu_row` packs the CPU activation vector with
`x_pack` into the same RGBA8 q16 bytes a GPU target returns and prints it
through the same `dump_row`, so the CPU path emits the same
`DUMP hidA w=384 h=<hash>` lines.  A new `STAGE_DUMP=1` dumps the five stages
of every layer, in order: `xb` (rms_att out), `att` (attention block out),
`x1` (after the wo residual), `xbf` (rms_ffn out), `hidA` (layer out).  Every
dump line now carries an FNV-1a hash of the whole row, so two runs are compared
on hashes and only the interesting rows are decoded.

**Where it breaks.**  Both paths emit the same first generated token
(`<|end|>`) and part company on the second (`Here` against `:belive`).  At
position 0 the softmax has one key, so `att` is exactly V and the qk-score
path is not exercised at all; from position 1 it is.  That is the current lead:
the score/softmax path, not the embedding, the rmsnorm, the qkv matvec, the wo
residual or the FFN.  Position 0 agreeing is weak evidence, not proof, but it
is the only split the run gives.

**Negative results from the same run.**  `KV_PACK=0` (24-bit cache) and
`QKV_PAIR=0` produce the identical wrong text, so neither the KV packing nor
the paired qkv kernel is the cause.  `--prompt Hello` (one token) gives the
same split, so it is not the prompt length.

**The CPU path is not a numerical oracle, which changes item 13.**  The CPU
path keeps float32 activations; the GPU path round-trips q16 at every stage.
Per-layer cosine between the two paths, same prompt, same container:

| layer | MiniMind3 q8 (works) | TinyBrainBot q4 (broken) |
|---|---:|---:|
| 0 | 0.9998 | 0.980 |
| 3 | 0.973 | 0.937 |
| 7 | 0.458 | 0.356 |
| 11 | - | -0.03 |

MiniMind3 answers correctly with cos 0.458 at layer 7, so a cross-path delta of
this size is normal (the residual stream saturates at the q16 clamp and the two
numerics part company).  A CPU-vs-GPU dump diff can therefore point at a stage,
but it cannot decide whether a kernel is wrong; the deciding test has to be a
same-path A/B, one flag against one kernel.  `engine/scripts/diff_paths.sh`
prints the cross-path table with that caveat in its header.

**Suite on the deployed build:** ACCEPT - determinism, q8
`我是一个AI助手，没有自我意识，也没有身体，所以我无法`, q4
`我是MiniMind，一个高效的小参数AI模型，专注于`, poem OK, PPL 271.2855 /
325.3751 exact.  Every probe is env-guarded (`STAGE_DUMP`, `HIDDEN_DUMP`) and
the dump path is not in the draw path, so no number can move; the suite is the
proof.

**The kernels are not the bug: attention is exact, the full texture path is not (item 12b).**
`ATT_DUMP=<pos>` (`GPU::dump_attention`) prints, for every layer at that
position, the exact integers the five attention passes consumed and produced:
the q row after rope, every K and V row 0..pos, the score texels per head, the
stats texel and the block output - all in the grids the shaders wrote (q and
att_out int/256, K/V int/KVSCALE, scores and stats max int/1024, stats sum
int/32), with one `ATTD meta` line for the shape.  `engine/scripts/check_att.py`
re-derives each stage with nothing but a dot product and a softmax:

| check | identity |
|---|---|
| score | `round((sum_i q_i k_i) / sqrt(head_dim) * 1024)` |
| stats max | `max_t score[t]`, exact |
| stats sum | `round(32 * sum_t exp((score[t]-max)/1024))` |
| att out | `round(256 * sum_t p_t v_t[i])` |

TinyBrainBot q4, `Hello, who are you?` (9 tokens), KV_PACK=16, all 12 layers:

| position | score dev | stats max | stats sum dev | att_out dev |
|---|---:|---:|---:|---:|
| 1 (prefill) | <=1 LSB | exact | <=0.016/32 | <=1 LSB |
| 9 (first decode) | <=1 LSB | exact | <=0.015/32 | <=1 LSB |

So `att_score`, `att_stats` and `att_v` - the 12-head / head_dim 64 / group-3
mapping, the GQA index math and the score-pair packing - are right on this
shape.  `att_qrot` and `att_kv_rope` are not covered by this check, because it
takes q and K as given.

**Bisection: every GPU kernel is correct in isolation.**  Same container, same
prompt, `--gen 16 --greedy IGNORE_EOS=1`:

| path | output |
|---|---|
| `GPU_LAYER=0` (CPU) | `😊 <|end|> <|end|> Hi! Great to see you` |
| `+ GPU_FFN=1` | identical |
| `+ GPU_ATTN=1 CPU_FFN=1` | identical |
| `+ GPU_ATTN=1 GPU_FFN=1` (hybrid) | identical |
| `GPU_LAYER=1` (full texture path) | `Érica Póralose's name is Maria Lopez Lgas` |

`RES_PACK=0`, `QKV_PAIR=0`, `KV_PACK=0`, `RMS_SPLIT`, `RMS_FUSED`,
`PREFILL_PAIR=0` and `ATTN_DUP=all` all leave the full path wrong (some change
the text), so neither the residual packing nor the qkv pairing, the kv packing
or the rmsnorm variant is the cause.

**What the full path does differently.**  It is the only path that keeps the
residual stream in a texture across layers and runs `rmsnorm` and the residual
adds on the GPU; the CPU paths keep float32 between stages and round to q16
only at the residual add.  Layer-0 stage dumps (q16 ints) agree between the CPU
and the full path to 1-2 units (xb 429/431, att 51/51, x1 196/197), so no stage
is wrong at layer 0; the difference grows with depth (layer 4 att max 114
against 87, layer 11 att max 132 against 367) while the means stay within a
few percent.  That is the profile of accumulated q16 rounding in a fragile
100M chat model, not of a broken kernel.  The prefill logits say the same:
over 32000 vocabulary entries `max|d| = 8.6`, `mean|d| = 2.3`, and every
1536-row tile is hit equally, so the 1280-row tail tile is not a special case.

Consequence: item 12 is a precision wall, not a shape bug.  The deployable
workaround today is the hybrid path (`GPU_LAYER=0 GPU_ATTN=1 GPU_FFN=1`), which
is correct on this prompt in chat mode and still runs both hot kernels on the
GPU; the full path costs 411 ms/token (prefill 9.27 s, 32 greedy tokens, q4)
and is not.  The workaround is priced in the 2026-11-06 subsection below: it
is coherent, it is not identical to the CPU path, and on this model it is not
faster than the CPU path either.

### The workaround, priced (2026-11-06)

TinyBrainBot q4 (`models/tinybrainbot/tinybrainbot.q4.mmcf`), Pi Zero.  Decode
is plain mode, `--prompt "Once upon a time, there was a little" --gen 32
--greedy IGNORE_EOS=1`, three interleaved passes per mode.  Warm prefill is
`PREFILL_REPEAT=3` (the two repeats after the first, four samples per mode):
the first prefill in a process carries the lazy program links, so it is not a
kernel number and the cold figures (CPU 22.8/7.6 s, hybrid 9.9/13.1 s, full
16.2/9.1 s) are link time, not prefill.

| mode | decode ms/token | min | median | warm prefill s (4) | median |
|---|---:|---:|---:|---:|---:|
| `GPU_LAYER=0` (CPU) | 490.5 / 534.2 / 504.1 | 490.5 | 504.1 | 4.06 / 4.49 / 3.60 / 3.53 | 3.83 |
| hybrid (`GPU_LAYER=0 GPU_ATTN=1 GPU_FFN=1`) | 608.2 / 583.8 / 521.2 | 521.2 | 583.8 | 4.54 / 4.06 / 4.07 / 4.27 | 4.17 |
| `GPU_LAYER=1` (full) | 388.0 / 401.0 / 447.5 | 388.0 | 401.0 | 2.95 / 2.96 / 2.98 / 2.96 | 2.96 |

**Hybrid is coherent on the chat path, and only there.**  Chat mode,
`--gen 24 --greedy IGNORE_EOS=1`, three prompts: CPU and hybrid print the
identical string, character for character, on all three; the full path
degenerates on all three.

| prompt | CPU = hybrid | `GPU_LAYER=1` |
|---|---|---|
| `Hello, who are you?` | `😊 <\|end\|> <\|end\|> Hi! Great to see you. How can I help?` | `Érica Póralose Pír Pír ...` |
| `What is the capital of France?` | `Île-de-France. <\|end\|>‐aned. ...` | `�Oge�Oge ...` |
| `Tell me a short story about a dog.` | `‘A dog walks to the park every day. ...` | `⚽ ⚽ ⚽ ...` |

The second and third CPU answers are the model's own degenerate style, not a
harness artefact; the point is that hybrid follows it exactly.

**It is not an identity.**  With no template the same prompt splits them:
plain mode, `--gen 32`, CPU prints `<|end|> <|end|> Here's a response based
on the given information:` and hybrid prints `<|end|> <|end|> <|end|>
nostalgia, I'm your great-grandparent.`  The first two generated tokens agree
and the third does not, so the 12c statement "identical correct string" holds
for the chat path and one prompt, not in general.  The honest form is "a
coherent answer, sometimes a different one", which is exactly what item 12's
acceptance asks of a fixed path.

**Decision: no GPU default for TinyBrainBot.**  The full path is 20% faster
in decode (401 against 504 ms/token) and 23% faster in warm prefill (2.96
against 3.83 s) and is wrong; the hybrid is 16% slower than plain CPU at
decode (584 against 504) and 9% slower at prefill, because at 12 layers x 768
the per-stage q16 round trip costs more than the two GPU kernels save.  The
deployable configuration on this board is therefore plain CPU.  On the
server that is `--no-gpu-layer`: `server.py` sets `GPU_LAYER` from that flag,
there is no hybrid flag, and `GPU_ATTN=1 GPU_FFN=1` would have to come from
the parent environment.

**Range is not the separator, precision is.**  The CPU-side dumps carry the
q16 envelope: TinyBrainBot's residual reaches 30720 (120 of the +-128 range)
in layer 0 and 0.1-0.4% of its texels clip at the ceiling from layer 1 on;
MiniMind3 q8 does the same from layer 0 (0.1-1.2% clipped, measured with the
same `STAGE_DUMP=1 GPU_LAYER=0` run) and still answers correctly.  The two
models have the same headroom problem and different sensitivity to it: one
q16 per activation is enough for MiniMind3's trajectory and not for
TinyBrainBot's.  That rules out a range bug and sizes the real fix:
precision, i.e. more than one int16 per activation on the full path.

### q8 precision is not why TinyBrainBot degrades (item 15, 2026-11-06)

Item 15 said TinyBrainBot's q8 is bad because `Q8M1` stores one scale per
matrix.  The q8 file was measured against its own i16 sibling before changing
any format, and the premise does not hold.

**The file is a faithful quantization.**  For every matrix, `w = i16 * scale`
from an i16 container (regenerate it with `mmcf_pack.py --in models/tinybrainbot
--quant i16 ...`; it is not kept on disk) is the exact source; the shipped q8
tensor is
compared against it row by row (`||w_q8_r - w_r|| / ||w_r||`, 128 rows per
chunk).  The stored scale equals `max|w|/127` to the last digit on all 81
matrices (ratio 1.000), and the whole-tensor cosine is 0.998-0.9999.

| model / file | per-row error p50 (median over matrices) | worst matrix p50 | Frobenius |
|---|---:|---:|---:|
| TinyBrainBot q8 (shipped) | 2.14% | 7.59% (layer 7 w2) | 1.3-6.8% |
| TinyBrainBot q8, scale clipped at p99.9 | **0.92%** | 2.13% | 1.5-3.9% |
| MiniMind3 q8 (works) | 1.10% | 1.56% | 1.2-1.6% |
| TinyBrainBot q4 (works) | **7.83%** | 7.92% | 7.9-8.0% |

**And the text does not follow the error.**  Same binary, same prompt, chat
template from the container, `GPU_LAYER=0` (CPU attention; the matvecs are the
GPU q8 kernels either way), greedy 24 tokens:

| file | `Hello, who are you?` |
|---|---|
| q8 shipped | `contacting contacting contacting ...` |
| q8 clipped at p99.9 (0.92% error) | `RetRetRetRetRetRet ...`; `safely safely ...` on two other prompts |
| q4 (7.83% error) | `Hi! Great to see you. How can I help?` |

A q4 file with **eight times** the weight error answers and a q8 file with
better error than MiniMind3's does not, so the failure is not the scale and
per-row scales (item 15's proposal) would not have fixed it.  It is the q8
*path* at TinyBrainBot's shapes or magnitudes.  The first suspect is the q16
envelope on the matvec output rather than the weights: item 12's `STAGE_DUMP`
already showed this model's residual reaching 30720 of the +-32768 range, and
a fragment writes `s*ws` into that same q16 grid, so a matvec whose output
saturates loses the information regardless of how fine its weights are.  That
is item 15a.

**The tool that made the measurement is kept**:
`engine/tools/mmcf_requant_q8.py --in model.i16.mmcf --out model.q8.mmcf
[--clip P] [--emb keep|q8]` rebuilds every i16 matrix as the pre-biased q8u8
texture dtype, preserving the tokenizer, template and metadata (only the
scheme byte is rewritten; `split` is preserved).  Validation: rebuilding
`stories15M` from its i16 container with the default `max|w|/127` rule gives
the shipped `stories15M.q8.mmcf` **byte-identical greedy text**
(`, there was a little girl named Lily. She loved to play outside in the
sunshine. One day, she saw a big, red ball in`), so the re-quantizer agrees
with the packer that produced the deployed files.

One operational note: the first pass ran numpy over the 250 MB i16 mmap with
whole-tensor `float32` temporaries and rebooted the Pi Zero (426 MB, no swap)
mid-run.  Every analysis script in this section chunks at 128 rows; 250 MB of
mmap plus one 98 MB array is enough to take the board down.

## The Jinja template engine, measured on the Pi (2026-11-07)

The container's chat template is rendered by the vendored llama.cpp Jinja
engine (see `engine/third_party/jinja/README.md`).  What it costs, measured
rather than estimated, because a Pi Zero pays for every byte and every second:

| | before | after |
|---|---:|---:|
| cross build (`107`, `-B pi`) | ~18 s | **58 s** |
| `engine_mm3_pi`, stripped | ~600 KB | **986 KB** |
| native build (`-O2`, one g++ call) | ~18 s | 38 s |
| `engine_mm3`, stripped | 570 KB | 1.19 MB (`--gc-sections`) |

`--gc-sections` is worth 120 KB: the Jinja engine is only reachable from the
chat path, so the linker drops what this build never reaches.

The gate on the Pi, with the shipped (template-carrying) containers, is
**ACCEPT** and the pins are unchanged to four decimals:

```
determinism OK / default system prompt OK
q8 OK: 我是一个AI助手，没有自我意识，也没有身体，所以我无法
q4 OK: 我是MiniMind，一个高效的小参数AI模型，专注于
poem OK         PPL baseline: 271.2855   PPL q4: 325.3751
ACCEPT
```

So 986 KB and 58 s of cross-build buy no numeric change at all on VC4; the
template path is byte-identical to the kind path it replaces (the token-id
comparison on Xe is in the commit that introduced it).

Decode, same board, `SEQ=768 GPU_LAYER=1 KV_PACK=16`, 16 generated tokens:
MiniMind3 q8 **188.6 ms/token**, q4 **295.1 ms/token** — the same range as the
matrix at the top of this file, i.e. no regression from the new code path.

Template compile is one-off and invisible next to shader linking: the
`SHADER_TIME` marks put "model loaded" at 472 ms and "session ready" at
5.6 s, and the 3.9 KB template compiles between them without moving the mark.

## 2026-11-07 — Supra2-Medium q8 + q24 residual (Pi Zero)

First measurement of a model with the 24-bit residual stream (metadata key 12).
Container `medium.q8.q24.mmcf`, 34 MB, 25M parameters, 7 layers, hidden 512,
8/8 heads, vocab 16408, `SEQ=512 GPU_LAYER=1 KV_PACK=16`.

Decode, "Write a long paragraph about cats and dogs", 50 tokens:
**84 ms/token** (11.2 s wall including ~7.0 s of startup).  Shorter runs land at
110-150 ms/token; the board's noise is the usual one.  That is the fastest
model measured here - stories15M q8 is 138 ms and MiniMind3 q8 is 252 ms - and
it is within ~25% of the board's best observed rate (~180 MMAC/s against 228
for MiniMind3, which processes four weight rows per fragment instead of one).

**The q24 residual costs nothing measurable.**  Same container, same run,
`RES_FMT=i16_1` (the q16 grid): 156 ms/token against 147 ms/token, i.e. inside
the noise.  The 24-bit decode is three byte conversions plus two multiplies per
value and the residual is a small fraction of the fetch traffic.

Startup is where the time goes, and none of it is the model:

| mark | at |
|---|---:|
| model loaded (34 MB container) | 163 ms |
| EGL ready | 1889 ms |
| weights uploaded | 2757 ms |
| session ready | 2936 ms |
| first token (shaders linked, prefill) | rest, ~3.5 s |

`SHADER_TIME` puts 14 programs at ~1.8 s of link time, the largest single one
being the logits tile (`xq8pl_1048_512`, a 512-wide unrolled loop).  Mesa's
disk shader cache does not help on this driver: `MESA_SHADER_CACHE_ENABLE=true`
gives 6.9 s and 7.2 s on two consecutive runs, so every start pays the link.

So the decode has no headroom worth chasing on this model; startup would need
fewer or smaller programs, which is a project rather than a tweak.

## The q16 decode flips the sign of the negative rail on VC4 (2026-11-09)

A Pi-vs-Xe sweep over 13 containers, one prompt each, logits dumped after
prefill (`LOGITS_OUT`), `GPU_LAYER=1 KV_PACK=16`, greedy, same container on both
machines.  Most models agree; two do not, and the reason is a **decode bug, not
a model bug**.

| model | residual peak / rail | cosine(Pi, Xe) | max \|d\| logits | entropy Xe -> Pi |
|---|---:|---:|---:|---|
| MiniMind3 q8 | 0.36x | 0.9994 | 0.47 | 0.97 -> 1.00 |
| MiniMind3 q4 | 0.36x | 0.9993 | 0.48 | 0.78 -> 0.77 |
| supra2-medium q8 (q24 residual) | 3.94x | 0.9999 | 0.32 | 2.02 -> 2.07 |
| stories15M q8 | 0.12x | 0.9999 | 1.36 | 4.16 -> 4.08 |
| stories42M q8 | 0.59x | 0.9999 | 0.85 | 4.08 -> 3.87 |
| Vertex-15M q4 | 0.21x | 0.9818 | 2.19 | 0.26 -> 0.43 |
| Vertex-35M q4 | 0.83x | 0.9972 | 0.56 | 0.66 -> 0.82 |
| Vertex-100M q4 | 1.14x | 0.9985 | 0.97 | 2.69 -> 2.74 |
| feng-30m q4 | 1.65x | 0.9846 | 1.57 | 6.47 -> 5.92 |
| **TinyBrainBot q4** | **2.96x** | **0.7233** | **6.16** | **0.75 -> 2.77** |
| **PicoLM-85M q4 (q24 residual, FFN hid 1.07x)** | 1.07x on `hid` | 0.9966 | **13.96** | 0.16 -> 0.25 |

Two rows matter.  Both are models whose residual reaches the q16 rail, and both
are **machine dependent**, not merely degraded.

### The mechanism: `x_i16raw` re-derives the sign from a float compare

`STAGE_DUMP` on TinyBrainBot q4, same prompt, both machines, 779 stage lines in
the same order.  Layers 0-3 agree (cosine 1.00000).  The first line that differs
is the decoder's output at the start of layer 4:

```
line 19  hidA (end of layer 3)   Xe elem133 = -32768    Pi elem133 = -32768   cosine 1.00000
line 20  xb   (rms_att, layer 4) Xe elem133 =   -683    Pi elem133 =   +683   cosine 0.10019
```

The stored value is identical and the norm cannot change a sign, so the sign is
lost in the **read**.  `x_i16raw` reconstructs the 16-bit value as

```glsl
float u = c.x*65280.0 + c.y*255.0;
if (u >= 32768.0) u -= 65536.0;
```

and the encoder clamps to `[-32768, 32767]`, so the most negative value is
stored as bytes `(128, 0)`.  Reconstructing it needs `(128/255)*65280` to land
on the *correct side* of 32768.0.  It does on Xe; on VC4 it lands at or below
32768.0 in the other direction and the value reads back as **+32768 (+128)**.
The next store clamps that to +32767, so the sign flip is permanent.

Whole-dump census of values within 8 units of the rail, same two runs:

| | values at the negative rail | values at the positive rail |
|---|---:|---:|
| Xe | 10 | 0 |
| Pi | 1 | 7 |

Line by line, every line where Xe holds `-32768` either holds nothing unusual on
the Pi or holds `+32767`.  Nothing else in the dump is anywhere near the rail.

### Ruled out, one at a time

| hypothesis | result |
|---|---|
| the Pi's numbers are nondeterministic | two runs, same bytes: `elem133` = 32749 and 32736 both times |
| the store clamps differently | the encoder is identical code and its output is the dump's `hidA` line, which matches exactly |
| the model is just too weak | the **CPU path** on the same container and prompt: cosine **0.9994**, max \|d\| 0.30 (it clamps in `add_q16` and decodes with integers) |
| it is the attention or FFN kernel | the first divergence is the *norm's input*, before any kernel runs again |
| the q24 path has the same bug | `x_q24raw` decodes with `float(int(c.g*255.0+0.5))*65536.0 + ...`, i.e. rounded integers, and is immune — which is why `--res q24` "fixed" supra2 |

So: **the engine's q16 rail is real in the encoder and broken in the decoder on
one driver.**  A model that saturates at the negative rail is not slowly worse on
VC4; it is a different model.

### What it costs, and the patch

Nothing has to change for models that never saturate: every pin in this file is
below 1x the rail (MiniMind3 peaks at 46).  What changes is the advice: a model
at >1x cannot be compared across machines, and `--res q24` is not a luxury, it is
the only correct choice (its decoder is the robust form).

The decode should not re-derive a sign from a float compare.  The robust form is
the one `x_i8q` already uses — round the two channels to integers first, then the
compare is exact:

```glsl
float x_i16raw(vec2 c){
  float hi = floor(c.x*255.0+0.5);
  float lo = floor(c.y*255.0+0.5);
  float u  = hi*256.0 + lo;          // exact for 0..65535
  if (u >= 32768.0) u -= 65536.0;
  return u;
}
```

The same applies to `x_i16pairq` and `x_i16quadq`.  On Xe this is a no-op (the
present form already decodes 0x8000 correctly there), and on VC4 it changes only
the values that are wrong today, so `scripts/regression.sh` should stay on its
pins on both machines; a model that saturates needs its own pin either way.

### The other matrix variables (2026-11-09)

One variable at a time, same container and prompt on both machines.  Numbers are
the cosine and the max |logit difference| between the Pi and the Xe run.

| variable | settings | Pi vs Xe |
|---|---|---|
| FFN kernel | fused (default) vs `FFN_FUSE=0` | same machine: Xe **bit-identical** on both models; Pi q8 max\|d\| **0.19**, q4 bit-identical. Cross-machine: MiniMind3 0.47 -> 0.48, TinyBrainBot 6.160 -> 6.160 (no change) |
| `SEQ` | 256 / 768 / 2048 | 0.9994 / 0.4727 for all three |
| KV format | 24-bit / 16-bit / 8-bit at scale 8 | MiniMind3: 0.9994 / 0.51, 0.9994 / 0.47, **0.9902 / 1.82**. Vertex-100M: 0.9990 / 0.81, **0.9956 / 1.81** |
| path | `GPU_LAYER=1` vs `GPU_LAYER=0` | TinyBrainBot: 0.7233 / 6.16 vs **0.9994 / 0.30** |
| precision | q8 vs q4 vs i16 | MiniMind3 q8 0.9994 vs q4 0.9993; MiniMind2 q8 0.9930 vs i16 **0.9965** |

What that says:

- **The FFN kernel is a VC4-only rounding difference, and a small one.** On Xe
  the fused and two-pass paths are bit-identical; on the Pi the q8 fused kernel
  moves the logits by up to 0.19 (the q4 fused path is bit-identical by
  construction, which `gpu.cpp` states: "same group order, same qz/sl
  rounding").  It explains nothing about the instability — TinyBrainBot's
  cross-machine numbers do not move at all, MiniMind3's move 0.47 -> 0.48 — but
  it does mean a Pi q8 run cannot be made to match an Xe q8 run by switching
  kernels.
- **`SEQ` is numerically inert** at a fixed prompt: the prefill positions and
  the arithmetic are the same, the window is allocation only.  (PPL still must
  not be compared across `SEQ` — the scored window changes; that is a different
  statement from this one.)
- **8-bit KV costs reproducibility, not just quality.**  A 1/16 grid turns every
  disagreement near a rounding boundary into a full step (16 q16 units), so the
  Pi-vs-Xe spread roughly quadruples (0.47 -> 1.82 on MiniMind3, 0.81 -> 1.81 on
  Vertex-100M).  24-bit and 16-bit are a wash.  If the board has to agree with a
  desktop, keep `KV_PACK=16`.
- **i16 weights are the most portable**, but the difference is small and only
  shows on a model that is already borderline (MiniMind2 0.9930 -> 0.9965).
  MiniMind3's i16 container is 137 MB and cannot be loaded on this board
  (CmaFree ~70 MB), so that cell of the matrix stays empty.

Measured, not argued: with that patch applied to the laptop build, MiniMind3 q8
gives the same PPL to four decimals (199.2005 on the 766-token corpus used here,
not the 293-token `data/ppl_test.txt`), the same pinned string, and TinyBrainBot's
logits are **bit-identical** to the unpatched run.  So the change is a no-op where
the decode already worked, which is the shape a fix for this should have.

Landed in `6234328`.  The board then corrected the "on VC4 it changes only the
values that are wrong" reading above: the old decode trusted the 8-bit texture
sample to be exactly `hi/255`, and VC4's sample is not, so **every** q16 value
was reconstructed with a small error, not just the rail.  The Pi therefore
re-baselines:

| on the Pi Zero (VC4) | old decode | integer decode |
|---|---|---:|
| MiniMind3 q8 identity answer | `我是一个AI助手，没有自我意识…` | `我是MiniMind，一个由jingyaogong…` |
| MiniMind3 q8 PPL, 293-token corpus | 271.2855 | 271.6420 |
| MiniMind3 q4 PPL, same corpus | — | 326.8088 |
| TinyBrainBot q4 vs Xe: cosine / top-5 overlap | 0.7233 / 1 of 5 | **0.9971 / 5 of 5** |
| TinyBrainBot q4 vs the board's own CPU path | — | 0.9953 / 5 of 5 |

Xe is unchanged (PPL 270.3170, both pinned strings, TinyBrainBot bit-identical),
and the q4 prefix `我是MiniMind` still matches on both machines.  The board's q8
string and PPL move, which is what a corrected decode looks like: the board now
agrees with the desktop and with its own CPU path, where before the saturating
model agreed with neither.

## Where the last 0.5% lives: the q16 store, not a kernel (2026-11-09)

After the integer decode (6234328) the board and the desktop still disagree by
cosine 0.9994 / max |d| 0.47 on MiniMind3 q8, and by 0.49% PPL (271.6420 vs
270.3170).  This is that gap, attributed stage by stage.

### It compounds, it does not jump

`STAGE_DUMP=1`, same container and prompt, both machines, patched builds.
Per stage: percentage of elements that are bit-identical, the largest difference
in q16 units, and the mean absolute value of the row (real units).

| line | stage | identical | max \|d\| | mean \|d\| | mean \|x\| |
|---|---|---:|---:|---:|---:|
| 0 | xb (layer 0) | 68.1% | 1 | 0.32 | 0.757 |
| 1 | att | 72.3% | 2 | 0.29 | 0.197 |
| 2 | x1 | 78.8% | 1 | 0.21 | 0.100 |
| 3 | xbf | 45.6% | 8 | 1.84 | 0.719 |
| 4 | xb (layer 1) | 40.6% | 17 | 3.23 | 0.692 |
| ... | | | | | |
| 15 | xbf (layer 7) | 7.9% | 30 | 6.68 | 0.784 |
| 16 | xb (layer 8) | 11.3% | 32 | 7.08 | 0.783 |

Two things to read there.  In layer 0 the stages are 68-79% bit-identical and
differ by **one** q16 unit (1/256), inside a value of order 1 — that is the
store's own rounding, not arithmetic that differs.  Then the difference grows
smoothly, roughly 1.5x per layer, with no stage that jumps: it is the residual
chain and the nonlinear attention carrying a last-bit difference forward, not a
kernel doing something different on one machine.  The whole layer-output dump
shows the same shape (line 0 differs by 1 unit, line 31 by up to 72), and it
resets to exact at every new token, because the embedding lookup reads the same
bytes on both machines.

By layer 8 the residual is ~10% off in relative terms (32 units on a 0.78 mean);
the logits are not, because a 768-term dot product averages uncorrelated last-bit
noise down by about sqrt(768): hidden-state cosine 0.9957 against logits cosine
0.9994.

### The grid is the amplifier, the width is the lever

Two single-variable runs decide between "the store's grid" and "a wrong kernel".

| change | Pi vs Xe cosine | max \|d\| | reading |
|---|---:|---:|---|
| baseline (q16 activations, q16 residual) | 0.9991 | 0.5586 | |
| residual q16 -> **q24** (`--res q24`, same weights) | 0.9995 | **0.4180** (-25%) | one store of five widened |
| activations q16 -> int8 (`ACT8=1`, 16x coarser) | 0.9803 | **3.0000** (+437%) | the grid scales the gap |

Coarsening the grid by 16x multiplies the cross-machine difference by 5.4x, and
widening one of the five stores removes a quarter of it.  A kernel that computed
something different would not scale with the grid like that; last-bit rounding
at a quantisation boundary does.

So the 0.5% is the price of 1/256: every store puts values astride a boundary,
the two fragment units round their own way, and the difference compounds through
eight layers.  Nothing in the shader text can make two vendors' fragment units
round a boundary identically.  The levers are all width: the residual is already
switchable (`--res q24`), and the other four stores (xb, att, xbf, hid) would
need an I24 activation format, which is 2x the fetch traffic and a decode VC4
pays for in three byte conversions.

Until then, the pins are per machine, and a 0.5% PPL difference between the
board and the desktop is the floor, not a defect.

## The card under the models: 7.4 MB/s write, 11.4 MB/s read (2026-11-09)

A 64 MiB file on the root filesystem of this Pi Zero (card reports `SD32G`,
08/2024, no speed class), `dd` with `conv=fsync` for writes and `iflag=direct`
to bypass the page cache:

| operation | block | throughput |
|---|---|---:|
| write | 1 MiB | 7.4 MB/s |
| write | 128 KiB | 7.1 MB/s |
| write | 4 KiB | 6.7 MB/s |
| read (O_DIRECT) | 1 MiB | 11.4 MB/s |
| read (O_DIRECT) | 128 KiB | 10.6 MB/s |
| read (O_DIRECT) | 4 KiB | 3.2-3.7 MB/s |
| read (page cache) | 1 MiB | 299 MB/s |

**`/tmp` on this board is a 214 MB tmpfs**, so a card test written there measures
RAM: it reported 107 MB/s write and 178 MB/s read.  The root filesystem is
`/dev/mmcblk0p2`, 87% full (3.6 GB free), which is also the state in which SD
write throughput is worst.

### What it costs per model load

MiniMind3 q8, 65 MiB container, `UPLOAD_TIME=1`, one run with the file's pages
dropped (`posix_fadvise(DONTNEED)`) and one straight after:

| | wall | `UPLOAD` line | texture time |
|---|---:|---:|---:|
| cold (pages dropped) | 46.2 s | 16.00 s | 4.94 s |
| warm (page cache) | 28.6 s | 8.45 s | 1.40 s |

So the 65 MiB read costs **7.6 s of the 16 s upload**, i.e. ~9 MB/s effective —
the sequential figure, because mmap plus the kernel's readahead does the right
thing even though the engine walks the container tensor by tensor.  The rest of
the cold start (about 10 s of the wall-clock difference) is the other pages the
run touches, and the texture work is 5 s of it.

Practical consequences: a cold start pays roughly 8 s per 65 MiB of container
(the 87 MB Vertex-100M q4 costs ~10 s before the GPU does anything).  `--prefix-cache`
saves the prefill but not this.

### The network is slower than the card, not the other way round

The same 64 MiB, moved over the SSH path this workstation actually uses
(100 Mb/s on the board's side, the workstation on WiFi behind CGNAT), timed by
bracketing the transfer; `ssh_download` / `ssh_upload` to RAM to isolate the
link from the card:

| transfer | wall | throughput |
|---|---:|---:|
| board -> workstation | 20.3 s | 3.15 MB/s |
| workstation -> board, into `tmpfs` | 22.5 s | 2.84 MB/s |
| workstation -> board, into the SD card | 26.4 s | 2.42 MB/s |

So the link runs at ~3 MB/s, the card writes at 7.4 and reads at 11.4: **the
network is the slowest part of this system by 2.6-4.7x**, and writing to the
card on top of a transfer costs only ~15% (the two overlap).  A deploy of a
65 MiB container is ~25 s, a cold load from the card is ~8 s.

A caution about the instrument: a plain `nc` between the two machines reported
0.47 MB/s in both directions, 6x worse than SSH.  That test has the board dial
*in* to the workstation's public address, which crosses the workstation's
carrier-grade NAT in the inbound direction; the SSH path is the workstation
dialling out.  Measure the path you deploy over, not a path that merely reaches
the same two hosts.

## Chat REPL: every turn prefills the new exchange, and the tool loop shares the chain (2026-11-09)

Iris Xe (ADL GT2), Mesa 26.2.6, MiniMind3 q8, `GPU_LAYER=1 KV_PACK=16 SEQ=768`,
greedy, `CHAT_CACHE=1 TIMING=1`, one engine process per run.  The prompt is a
469-token system turn followed by eight short user lines, `--gen 8`:

| turn | prompt tokens | cached | prefill tokens | prefill |
|---:|---:|---:|---:|---:|
| 1 | 469 | 0 | 469 | 3.79 s (compiles) |
| 2 | 500 | 477 | 23 | 0.21 s |
| 3 | 530 | 508 | 22 | 0.21 s |
| 4 | 560 | 538 | 22 | 0.21 s |
| 5 | 590 | 568 | 22 | 0.22 s |
| 6 | 620 | 598 | 22 | 0.22 s |
| 7 | 651 | 628 | 23 | 0.23 s |
| 8 | 681 | 659 | 22 | 0.23 s |

`cached` is the whole previous turn (prompt + reply), so the prefill does not
grow with the conversation.  A cold `--worker` replay of turn 8 (681 tokens
from zero) took 7.2 s and produced the same 8 greedy tokens; the warm turn took
0.23 s of prefill.

The tool loop (`--tools`, same model, a 233-token system + tools prompt) used to
re-prefill from zero on every round and to drop the conversation after the first
user line.  It now shares the same chain: a tool-response round caches the
prompt and the generated tool call, and the next user turn caches the whole
previous exchange:

| call | prompt | cached | prefill | prefill |
|---|---:|---:|---:|---:|
| turn 1, round 0 (tool call) | 233 | 0 | 233 | 2.58 s (compiles) |
| turn 1, round 1 (tool response) | 358 | 257 | 101 | 1.19 s |
| turn 2, round 0 (new user line) | 406 | 382 | 24 | 0.26 s |
| turn 2, round 1 (tool response) | 529 | 430 | 99 | 1.22 s |

Before the change the second user line prefilled its whole prompt (1.53 s; the
prompt contained no previous turn) and the first tool-response round prefilled
its whole prompt (2.51 s).  The remaining ~100 tokens are the tool response
itself plus the template markers, i.e. new content.

The same change fixed a REPL-only abort: a Chinese reply cut mid-character by
`--gen` used to put half a UTF-8 sequence into the next turn's re-rendered
history, and `common_json::parse` terminated the process (exit 134).  The REPL
now buffers the tail like the worker does and emits U+FFFD at a cut, so the
third turn prefills that one token again instead of aborting.

## `TOPK` is now `--top-k` + `--top-k-candidates` (2026-11-09)

The sampler's hard top-k is a CLI flag now: `--top-k N` truncates before top-p
and uses `nth_element` + a k-element sort on the full logits (exact).  The old
`TOPK=N` environment switch is gone; the candidate path is opt-in with
`--top-k-candidates` and remains approximate (per-tile top-2, falls back to
full logits when the candidate mass is below `top_p`).  `TOPK_FAST=1` only
matters together with `--top-k-candidates`.

Historical commands above that say `TOPK=...` were measured with the old
switch; the candidate pass is the same one `--top-k-candidates` selects, but
the new flag also applies the hard top-k truncation.

## per-row q8 scales: void.1 becomes usable at q8 size (2026-11-09, Xe)

void.1 (dotlabs, 90.1M params, 259-entry byte tokenizer, 24x640/1440, qk-norm,
theta 1e5) is the first model whose q8 container had to be thrown away: one
scale per matrix leaves its layer-0 FFN with 58-66% of the weights at zero, and
the model answers `everythinteresting` where f32 writes `everything`.  A part-3
plane (one f32 per output row, 486 KB for the whole model) fixes it.

| void.1, Xe/GLES2, seq 1024 | q8 | q8 per row | i16 |
|---|---:|---:|---:|
| logits vs f32 at the prompt that diverged (T=47) | 14.44, top-1 wrong | **0.67, top-1 right** | 0.93 |
| English PPL (1022 tokens) | 3.9017 | **3.7979** | 3.7824 |
| greedy 400 B | misspells | clean | clean |
| decode, ms/token | 21-27 | 19-33 | 37-54 |
| file | 86 MB | 87 MB | 172 MB |

Decode was measured on a contended laptop (host load 2.0, run-to-run spread
2x), so the ms column is a shape, not a baseline.  The MiniMind3 gate is
unchanged (`ACCEPT`, PPL 270.3170, q8/q4 strings identical) and
`shader_hashes.sh` still reports 117 programs: a model without the plane links
exactly the GLSL it linked before.

The two-position prefill kernels have no row-scale variant yet, so a file with
planes prefills on the single-position path (`Session::pair_ok()`); its prefill
is slower than a plain q8 file's, which the table above does not separate.

The plane is 16 bits in a 4-texel-wide texture.  A 1-texel-wide texture is the
one shape VC4 had never been asked to sample: the same container printed
`<pad>` plus invalid bytes on the Pi while it was clean on Xe, and widening the
texture to four texels is what the Pi build below needed.  The 16-bit value is
also what the desktop run now measures: 0.67 logits against f32, better than
i16's 0.93, because a 24-bit decode through a 1-wide fetch had been costing
about 0.4 logits of scale error that the wider 16-bit form does not.

Two bugs on the way, both worth recognising: the upload divided the plane by
the unit a second time, which clamped every row to 1.0 and made per-row scales
look like a no-op (found by replacing the read with `ws` and getting
bit-identical stage dumps); and disabling the packed residual add for these
matrices changed the *residual layout* the rmsnorm reads, which corrupted every
layer after the first.

## per-row q8 on VC4: two fetches from one sampler (2026-11-10, Pi Zero)

The plane shipped correct on Xe and produced `<pad>` plus invalid bytes on the
Pi, with two runs of one binary giving different tokens.  `engine/probes/wrow_probe`
drives each engine kernel that carries a plane against a CPU reference, on the
target, and reports the rows that differ:

| kernel, wrow, on the Pi | before | after |
|---|---|---|
| `matvec_q8_add` 640x640 | pass | pass |
| `matvec_q8_pair` 1440x640 | pass | pass |
| `matvec_q8_pair_logits` 64x64 | pass | pass |
| `matvec_q8_pair_logits` 896x640 (qkv) | **88 of 896 rows, worst 33126** | pass |
| `matvec_q8_pair_logits` 259x640 (void.1's logits tile) | **40 of 259, worst 30835** | pass |

The failing kernel was the only one that fetched the plane twice in one
fragment (row r0 and row r1), and the same shader shape reading *two* textures
is exact at 1440 rows, so the trigger is two `texture2D` fetches from one
sampler, not the fetch count.  qkv and the logits pass both go through it, which
is why the whole model was wrong while its arithmetic was fine.

Fix: the plane holds a row pair per texel, so a paired kernel reads once.  void.1
q8r on the Pi then prints ` there was a little girl named Lily who loved to play
with her friends. One day,` and is byte-identical across runs; MiniMind3's gate
string and PPL are unchanged, and `shader_hashes.sh` still reports 117 programs
because a model without planes links the same GLSL as before.

Prefill on the card is 13-16 s for the 17-byte prompt (1.1 s/token) because a
file with planes prefills on the single-position path; decode settles at
417-466 ms/token against q8's 380.

## Two-position prefill for a per-row scale model: 15% faster, once it is warm (2026-11-10)

The pair kernels got wrow variants (`matvec_q8_p2`, `matvec_q8_p2_add`,
`matvec_q8_p2_add_p2`, `matvec_q8_gate2_p2`), so `pair_ok()` accepts a model with
planes.  Tokens are byte-identical to the single-position path on both machines.

| void.1 q8r, Pi Zero, 17-byte prompt | first prefill | warm prefill |
|---|---:|---:|
| single-position (`PAIR_WROW=0`) | 14.1 s | 8.36, 8.18 s |
| two-position, wrow (default) | 15.9 s | **7.15, 6.97 s** |

**The first numbers are a trap, and this entry first recorded them the wrong
way.**  Comparing two modes on their first prefill measures which program set
each one compiles - the pair path links the P2 kernels, the single path does
not - and on VC4 that is seconds.  The first version of this entry concluded
"correct but slower, 13.7-15.2 s against 12.9-12.9 s" from exactly that
artefact, which is the mistake `engine.cpp` warns about under
`PREFILL_REPEAT`.  Warm, the paired path wins by 15%, so it is the default and
`PAIR_WROW=0` exists for the A/B.

Startup still matters for a one-shot CLI run: the paired path pays about 1.8 s
more compilation than the single path and never gets it back in 17 tokens.  A
server that keeps one process pays it once.

## One small plane atlas is correct and buys nothing (2026-11-10, rejected)

The per-row scale planes were folded into a few atlas textures (four texels wide,
one texel used per row, 2048 rows per texture, a per-draw `woff` row offset and a
`wsh` height), which cuts a token's plane binds from one per matrix to one per
2048-row slab: 121 textures became 38 for void.1.  It is exact - the probe passes
every kernel on Xe, and on the Pi every shape the engine actually uses (640,
896, 1440 and 259 rows) - and it is **not in the tree**, because it does not pay:

| void.1 q8r, Pi Zero, warm prefill | run 1 | run 2 |
|---|---|---|
| planes in 38 atlas textures | 7.02, 7.11 s | 7.48, 6.97 s |
| one texture per matrix | 8.22, 7.07 s | 9.32, 7.08 s |

The second repeat of each run agrees to 1% (7.0 s either way), so the bind count
is not what prefill costs on this board - the first repeat is where the variance
sits.  Complexity without a number is not worth carrying.

Two things were learned on the way and are worth keeping:

* **A 2D grid address is wrong on VC4 from the second grid row on.**  Addressing
  the atlas as `(mod(t, W) + 0.5) / W, (floor(t / W) + 0.5) / H` is exact on Xe
  and returns other rows' data on the card: B2/B4 and every C case in
  `probes/wrow_probe` fail there with it, and pass with a constant column and a
  row offset.  Whatever the driver is doing with `mod`/`floor` on a texture
  coordinate, it is not what the GLSL says, so the atlas form that shipped in
  the experiments keeps `u = 0.125` and varies only `v`.
* **A 64-row toy shape still failed on the Pi** with that form (3 of 22 probe
  cases: B2 and the two 64x64 pair cases, all at row 0), while every larger
  shape passed.  That was the second reason not to ship it: a probe case that
  fails for a reason nobody has explained is a defect, not a size limit.

The obvious explanation - a 4x32 plane texture is too small or too narrow for
VC4's TMU - was then tested and is **false**: A8-A12 in `probes/wrow_probe`
read a 4x8, 4x16, 4x32, 4x320 and 64x32 plane one row at a time and every value
comes back exact on both machines (33 cases, ALL PASS).  So the 64-row failure
in the atlas build was in that build's uniform plumbing, not in the card's
texture handling, and the size hypothesis is closed rather than left hanging.

### Where q8r ended up, after the four experiments (2026-11-10)

The engine is commit 52b52c0 for everything below; the per-row plane layout and
the two paired-row kernels that were tried on top of it are described in the two
sections above and are not in the tree.

| | Xe | Pi Zero |
|---|---|---|
| `probes/wrow_probe` (33 cases: plane fetches, per-row pairing, fused W13, two-position compile) | ALL PASS | ALL PASS |
| void.1 q8r, `--prompt "Once upon a time,"`, greedy | ` in a small town named Harmonyville...` | ` in a small town` (24 tokens, identical to the single-position path) |
| feng q8r, fused W13 **with** planes | `我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。` | same text |
| warm prefill, void.1 q8r, `PREFILL_REPEAT=3` | - | 7.0 s paired / 8.2 s single, 17-byte prompt |
| decode, void.1 q8r | 16-18 ms/token | 415-430 ms/token |
| `regression.sh` | ACCEPT (PPL 270.3170) | q8 pin `我是MiniMind，一个由jingyaogong` |
| `shader_hashes.sh` | 117 programs, unchanged | - |

Three things from the four experiments are in the tree (fused W13 reads its
plane once and the split guard is gone; the two-position kernels have wrow and
are the default for a row-scale model; the packer reports max|w|/median|w|).
The fourth, one small plane atlas, is measured at no gain and left out.

## MoE, first run: MiniMind3-MoE on Xe (2026-11-11)

The engine gained one MoE block (container v3, `docs/checkpoint-container.md`):
a per-layer router and top-1 expert FFNs.  The model is
`jingyaogong/minimind-3-moe`, 198M parameters, 4 experts of width 2432, top-1
with a normalised gate, so the runtime needs no softmax and no weighted combine
— the selected expert's FFN is the layer.

Container sizes from the same F16 source (396.85 MB):

| container | size | tensors |
|---|---:|---:|
| i16 | 406.8 MB | 133 |
| q8 | 203.5 MB | 133 |
| q4 | 131.0 MB | 357 (2 planes per matrix) |

The dense 64M MiniMind3 q8 is 66 MB, so the 198M MoE is 3.1x the file at q8 and
2x at q4.

**Routes and logits against a float32 forward** (`ref_forward.py`, now with
`--routes` and MoE):

| | routes, "Hello" (16 layer-positions) | logits max|d| | mean |d| | top-1 |
|---|---|---:|---:|---|
| i16 | 16/16 identical | 0.70 | 0.13 | same (36) |
| q8 | 16/16 identical | 0.87 | 0.17 | 2968 vs 36, 0.08 apart |
| q4 | 15/16 (l5 flips 2 -> 1) | 7.34 | 1.76 | 37 |

The q8 top-1 flip is a 0.08 margin on 13.7, i.e. inside the q8 error — it is
not a routing bug, and the routes (the discrete part) are identical.  q4's
logits error is an order of magnitude larger, as it is on the dense model.

PPL, `engine/data/ppl_test.txt` (293 tokens), `SEQ=768`, Xe:

| i16 | q8 | q4 |
|---:|---:|---:|
| 389.3532 | 350.8860 | 437.7821 |

q8 below i16 is the same inversion the dense model shows (280.2156 i16 against
270.3170 q8 in the container table), so it is the quantiser, not the MoE path.
The dense 64M model's 270 is a different model on an out-of-domain corpus; do
not read the 350 as "MoE is worse".

**Decode, Xe, chat prompt, 32 greedy tokens:**

| model | decode |
|---|---:|
| dense q8 | 10.9 ms/token |
| MoE q8 | 29.3 ms/token |
| MoE q8, `MOE_FIXED=1` (router skipped) | 10.5 ms/token |
| MoE q4 | 37.2 ms/token |

So the routing costs ~19 ms/token on this machine: eight `glReadPixels` calls of
four texels per token, one per layer, ~2.4 ms each.  The matvec is 4x768; the
read-back is the price, and it is the same on any GLES2 pipeline (the route is a
host-side uniform).  With the router pinned the MoE costs what the dense model
of the same expert width costs, which is the property that matters: only one
expert's weights are streamed.

Whether that is affordable on the Pi is the open question — 8 syncs against a
252 ms dense decode is a smaller fraction than against 10.9 ms — and so is
memory: 131 MB at q4 against a board that runs the 66 MB dense q8 with ~50 MB
of `CmaFree` left.  Both are measured on the board, not inferred from Xe.

### The same model on the Pi Zero (2026-11-11)

`GPU_LAYER=1 KV_PACK=16 SEQ=768`, greedy, 24 generated tokens, three runs
interleaved on one board (the q4 container is 131.0 MB, the q8 is 203.5 MB):

| run | prefill | decode |
|---|---:|---:|
| dense MiniMind3 q4 (45 MB) | 16.9 s | **323.8 ms/token** |
| MiniMind3-MoE q4 (131 MB) | 21.3 s | **341.8 ms/token** |
| MoE q4, `MOE_FIXED=1` (router skipped) | 23.2 s | **312.8 ms/token** |

Three numbers:

* **The routing costs ~29 ms/token**, 8 read-backs of four texels at ~3.6 ms
  each (Xe pays 2.4 ms for the same read-back).  That is 9% of the MoE decode.
  With the router pinned, the MoE is *dense q4 speed* (312.8 against 323.8,
  inside the board's noise): one expert of width 2432 is the dense FFN, so the
  3x parameter count is a memory cost only.
* **Prefill is 26% slower** (21.3 against 16.9 s): the pair path is refused for
  MoE (`pair_ok()`), and every position pays 8 route read-backs.  The
  `pair_ok()` comment's 7.0-vs-8.2 s measurement is the serial penalty alone.
* **q4 fits, q8 does not.**  The board's `CmaTotal` is 256 MB and the container
  uploads are the real test:

  | container | upload | result |
  |---|---:|---|
  | q4 (131.0 MB, 429 textures) | 25.6 s | runs, 342 ms/token |
  | q8 (203.5 MB, 221 textures) | 48.6 s | `FBO incomplete` — the weights took the CMA and the session's KV/logits targets could not be created |

  So a 198M-parameter MoE is q4-or-nothing on this board, and the q4 file is
  2x the dense 64M q8 that is already the default.  The model itself is
  coherent at q4 — `我是MiniMind，一个由jingyaogong创建的高效小参数AI模型。` —
  and its upload is 6x the dense q4's 4 s, because 429 small textures cost more
  per byte than 69 large ones (texture time 18.3 s of the 25.6 s wall).

What a Pi-sized MoE would need: 100M parameters or less at q4, or an expert
count of 2 with a width near 1216.  Nothing in that shape is published for this
family; `stories15M_MOE` is 36M but Mixtral top-2, which needs the weighted
combine the runtime does not have.

## LFM2.5-230M: first numbers (2026-11-11, Xe)

The first hybrid (conv + attention) model this engine runs.  Container
`models/lfm2.5-230m/lfm2.5.q4.mmcf` (235 MB, vocab 64402, `SEQ` 512, tied head
stored twice — see the assessment doc).  The serial path is the acceptance
reference; `GPU_LAYER=1` runs but is **wrong** for this model today and is
opt-in through `CONV_GPU=1`.

| measurement | Xe |
|---|---:|
| PPL (`data/ppl_test.txt`, 333 tokens, q4, serial) | **111.83** |
| decode, serial path (`GPU_LAYER=0` kernels, CPU layer loop) | 172–180 ms/token |
| prefill, 6-token prompt, serial | ~1.15 s |
| packer's residual calibration (rail 128) | peak 33.7 at layer 5 |
| engine q4 logits vs numpy f32 reference (`The capital of France is`) | cos 0.948, top-1 same |
| q4 container, tiled head vs `--head-dedup` | 234.9 MB / 168.9 MB, PPL 111.83 / 112.15 |
| q4 dedup vs `--head-q6` (6-bit rows) | 168.9 MB / 152.4 MB, PPL 112.15 / 128.17 |
| MiniMind3 q8 tiled vs dedup (SEQ=768) | 68.98 MB / 64.07 MB, PPL 271.19 / 270.28, golden string unmoved |

The PPL is not comparable with MiniMind's 270 — different model, tokenizer and
corpus length.  It is a baseline for this container, and the number a change to
the conv block has to hold.

Behaviour (serial path, greedy, `--system "You are a helpful assistant trained
by Liquid AI."`):

```
The capital of France is          -> Paris, and it is a major cultural
What is C. elegans?               -> C. elegans is a well-studied model organism ...
Write a haiku about rain.         -> Rain whispers softly, / A gentle breeze ...
chat, "What is the capital of France?" -> Assistant: The capital of France is Paris.
```

The GPU path (`GPU_LAYER=1 CONV_GPU=1`) now runs the conv block on the GPU and
decodes at **24 ms/token** (7x the serial path), but it is **not** the default:
it costs quality.  Same corpus, same SEQ:

| path | PPL | decode |
|---|---:|---:|
| serial (`GPU_LAYER=0`, CPU attention + CPU conv) | **111.83** | 172–180 ms |
| GPU attention, CPU layer loop (`GPU_LAYER=0 GPU_ATTN=1`) | 119.82 | — |
| fused layer (`GPU_LAYER=1 CONV_GPU=1`) | 126.67 | 24 ms |

So the attention block alone costs ~6% PPL on this model and the fused
residual/conv sequencing another ~6%.  The fused path also ends some plain
prompts on the first token (`Write a haiku about rain.` prints nothing) while
the chat template works, which is the profile of logits that are off enough to
make EOS win, not of a broken kernel.

Two findings worth keeping:

* **The conv block's zero state was not zero.** `init_conv` cleared the state
  targets with alpha 1.0, and I16_2 stores the second value in `.ba`, so every
  state started at 255 (0.996).  The first conv output at layer 0 was 62
  instead of 0.  Clearing alpha to 0 makes layer 0's `x1`/`xbf` byte-identical
  to the CPU path.
* **The 24-bit KV layout is wrong for this shape** (kv_dim 512, head_dim 64):
  the fused path answers garbage with it and is correct with `KV_PACK=16`.  A
  hybrid model now defaults to 16; `KV_PACK` still overrides.  Dense models are
  unaffected (MiniMind3 answers the same with either).

### The 6-bit head is not free either (2026-11-11)

`--head-q6` stores the tied head as signed 6-bit (a 4-bit low plane plus a 2-bit
high plane, per-row scale) and renders the logits from it: 152.4 MB, 16.5 MB
smaller than the q8 dedup container.  It costs **PPL 112.15 -> 128.17 (+14%)**,
because the head is also the *input* embedding: 1.61% row error at the input is
amplified by 14 layers exactly as the q4 head was (q4: 2.1% -> +7.5 logits).
The logits side alone would be cheap; the input side is what decides, and both
6 and 4 bits fail there.  So the Pi candidates are 168.9 MB at q8 quality or
152.4 MB at -14%, and the q8 one is the reference.
