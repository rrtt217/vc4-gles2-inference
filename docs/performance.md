# Performance

Measurements are from the Pi Zero and the local Xe/GLES2 machine. Treat them
as regression baselines, not portable benchmarks.

This file is append-only history.  Commands inside the entries are reproduced
as they were run, so historical ones may use flags that no longer exist
(`--template`, `TOKENIZER=`, `ROPE_THETA=`, `ROPE_HALF=`, legacy `.q8`/`.q4`
paths); the current invocation is in `AGENTS.md` and `engine/README.md`.

## Historical model matrix (2026-11-02 run)

**The current Pi matrix is the 2026-11-13 campaign near the bottom of this
file** (HEAD `71cb1e0`, protocol v1; the very last section is a documentation
addendum, not a measurement).  This table stays as history: it was measured on
an older build in a lower GPU-clock state, and the 2026-11-13 section lists what
moved.  Quote a Pi number only with its date and protocol.

Pi Zero, `GPU_LAYER=1 KV_PACK=16`, greedy, 16 generated tokens.  The Pi also
runs a Minecraft server, so use the ranges rather than individual decimals.

Every row is loaded from the `.mmcf` container named here; the legacy file of
the same name (without the suffix) is the packer input.  These are the
**2026-11-02** numbers, measured on an older build in a lower GPU-clock state
(`vcgencmd measure_clock core` read 300 MHz): the decode campaign, the fused-FFN
default and the two-position prefill all landed afterwards.  The last 32-token
measurement of the two MiniMind3 containers is 195.5 / 291.7 ms/token
(2026-11-05, plain short prompt, median of three interleaved runs, `SEQ=768`) --
a different protocol from this 16-token table, and neither column is "today".
Rows keep their date so the cost model stays auditable.

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

- **2026-11-13: the current Pi matrix.**  HEAD `71cb1e0`, 14 containers,
  protocol v1 (a discarded warm-up plus three timed runs).  q8 is 14-22% faster
  than the 2026-11-02 table; i16 is 41% slower and unresolved.  The section is
  at the bottom of this file.
- **2026-11-12: the deploy that made it possible**, and the board storage
  failure that delayed it by a day.
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

**The mixed-precision idea is dead on this driver.**  `engine/probes/texsize_probe.cpp`
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

The isolated probe (`engine/probes/matvec_probe.cpp`) measures it on identical weights
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
    -o /tmp/matvec_probe engine/probes/matvec_probe.cpp -lEGL -lGLESv2 -lm
/tmp/matvec_probe 1536 768 10        # rows, cols, iterations

g++ ... -o /tmp/texsize_probe engine/probes/texsize_probe.cpp -lEGL -lGLESv2 -lm
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

`engine/probes/matvec_probe.cpp` measures it in isolation (2048x768, two rows per
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

## LFM2.5-230M on the Pi Zero (2026-11-11)

The first hybrid model on the board, and the first 230M one.  ARMv6 build from
the vlab cross host (`Tag_CPU_arch: v6`, `Tag_FP_arch: VFPv2`), greedy,
`SEQ=256` (`KV_PACK=16` is the hybrid default), one run each unless noted.
`CmaFree` was 210 MB before the first run and 195-201 MB after it: the weights
do not land in CMA the way the 198M MoE's did (that one failed at q8 with
`FBO incomplete`), so the board has room for both containers.

| container | size | path | prefill | decode | text |
|---|---:|---|---:|---:|---|
| `lfm2.5.q4.dedup.mmcf` | 168.9 MB | `GPU_LAYER=1 CONV_GPU=1` | 3.2 s/token | **1133 ms/token** | `Paris, a historic city renowned` |
| `lfm2.5.q4.q6.mmcf` | 152.4 MB | `GPU_LAYER=1 CONV_GPU=1` | 2.2 s/token | 1371 ms/token | `Paris. It is the largest city in` |
| `lfm2.5.q4.q6.mmcf` | 152.4 MB | serial (CPU attention/conv) | — | 1538 ms/token | `Paris.` |
| `lfm2.5.q4.dedup.mmcf` chat | 168.9 MB | `GPU_LAYER=1 CONV_GPU=1` | ~6 s/token | — | `I'm a text` |

Two surprises worth keeping:

* **The q8 head is faster than the 6-bit one** (1133 against 1371 ms/token).
  The 16 MB the q6 container saves cost more per token in the unpacking kernel
  than the traffic they save, and the q6 head is 14% worse in PPL.  For this
  board the dedup q8 container is the one to ship.
* **Both fit.**  The 170 MB CMA figure from the MoE work did not reproduce
  here: a 168.9 MB container runs with 195 MB free afterwards.  Whether that is
  driver-dependent (textures backed outside CMA) or pumpkin's occupancy
  changed is not established; the numbers above are one session on the board.

### Context ceiling on the Pi: 2048 is reachable at 16-bit KV (2026-11-11)

`docs/long-context.md` says the binding limit is the KV texture height (2048 on
VC4).  For this model the limit was lower than that until the allocation was
trimmed: a hybrid model was creating a KV texture for **all 14 layers** while
only the 6 attention layers ever read one.  At SEQ 2048 the 8 dead entries cost
8 x 512 x 2048 x 4 = 33.5 MB of CMA, and the board failed with `FBO incomplete`
at SEQ 1280.

After `init_attention()` takes the layer types and skips them, the same
container runs at every size (dedup q8 head, `GPU_LAYER=1 CONV_GPU=1`,
`KV_PACK=16`, greedy, one 6-token prompt):

| SEQ | result | decode (position 5) |
|---:|---|---:|
| 1024 | runs | 1027 ms/token |
| 1536 | runs | 1040 ms/token |
| 2048 | **runs** | 981 ms/token |

These are shallow-depth numbers: a 2048-token prefill costs ~35 minutes at
~1 s a token and was not run, so the decode cost *at depth* (attention over
2048 rows) is unmeasured.

`KV_PACK=8` also fits 2048 (it halves the width), but on the GPU attention path
it costs a lot of quality: fused PPL 126.89 -> 194.07 on the corpus.  With the
trim there is no reason to use it for context.

### Where the fused path's PPL gap comes from (2026-11-11, Xe)

Same container (`lfm2.5.q4.dedup.mmcf`), same corpus, `SEQ=512`.  Two probes in
the CPU layer loop change exactly one thing each:

| path | PPL | delta |
|---|---:|---:|
| serial (CPU f32 rms, read-back adds) | 112.15 | — |
| + GPU rmsnorm (fixed-point partial sums) | **117.57** | +5.42 |
| + fused `wo`/`w2` residual adds (`matvec_add`) | 117.57 | **0.00** |
| fused (`GPU_LAYER=1 CONV_GPU=1`) | 126.89 | +9.32 |

So about a third of the gap is the **RMSNorm's statistics**: the shader sums
squares in fixed point, the CPU path in f32, and this model's residual grows a
"massive" channel (~25-34) at layer 5-6, where the norm's global scale is
dominated by that one channel.  The **residual-add order contributes nothing**
(fusing the add leaves PPL unchanged to four decimals), and **attention is
exact**: `GPU_LAYER=0 GPU_ATTN=1` reproduces the serial path's `x1` bit for bit
at every layer.

Per-layer `x1` dumps say the rest is a **step at layer 6** — 2-8 q16 units
through L0-L5, then 282 units (8 of 1024 channels) from L6 on, largest on the
massive channel (25.3 vs 24.2, a 4.4% relative change), with no clamping.  With
the GPU rms swapped in the step moves but does not go away (590), so the
remaining ~9 points are neither the attention nor the add order nor the final
norm; they sit in the fused loop's fixed-point handling of that channel and are
not attributed to a single stage yet.

Reproduce: `CPU_RMS_GPU=1`, `CPU_ADD_GPU=1` (docs/test-hooks.md).

## The QAD weights (2026-11-11, Xe)

Liquid publishes a quantization-aware-distillation checkpoint in
`LiquidAI/LFM2.5-230M-GGUF/qad/` (FP32 safetensors, sha `1a55fc50...`, the
provenance says the released QAD-Q4_0 GGUF is its byte-exact Q4_0 conversion).
Packed with the same flags as the base container (`--quant q4 --head-dedup
--seq 512`), it is the same 168.9 MB file format.

| weights | i16 PPL | q4 serial | q4 fused |
|---|---:|---:|---:|
| base LFM2.5-230M | **98.10** | **112.15** | 126.89 |
| QAD (qad/) | 109.56 | 117.08 | **114.53** |

Three things follow:

* QAD **trades full precision for robustness**: its i16 ceiling is 11.5 points
  worse (109.56 against 98.10), and its q4 penalty is much smaller (+6.9%
  against +14.3%).
* On the **serial** path the base weights still win (112.15 against 117.08).
* On the **fused** path QAD wins by **12.4 points** (114.53 against 126.89) and
  lands within ~2% of the base serial reference.  That is the path the Pi runs
  for speed (1133 against 1538 ms/token), so for the board the QAD container is
  now the right one: same speed, PPL 114.5 instead of 126.9.

The fused-vs-serial inversion for QAD (114.53 against 117.08) is inside what a
333-token corpus can resolve; do not read it as "fused is better".

Packing:

```bash
python3 engine/tools/mmcf_pack.py --in models/lfm2.5-230m/qad --quant q4 --seq 512 \
  --head-dedup --tok models/lfm2.5-230m/qad/lfm2.5.tok \
  --out models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --name lfm2.5-230m-qad --template chatml --chat \
  --chat-template models/lfm2.5-230m/qad/chat_template.jinja \
  --system "You are a helpful assistant trained by Liquid AI." \
  --bos 1 --eos 7 --stop 7 --add-bos 0 --add-eos 0 --verify
```

## The conv block is matvec-bound, not pass-bound (2026-11-11, Pi Zero + Xe)

Four changes to the hybrid conv block, all in the fused path
(`GPU_LAYER=1 CONV_GPU=1`):

| change | what it removes |
|---|---|
| three rotating state slots | 2 fullscreen copy draws per conv layer per token (`s0 <- s1`, `s1 <- gated`); the gate's output is written straight into the slot holding `x[pos-3]`, which is dead |
| `conv3_act` two channels per fragment | every input (state rows, gate, kernel row) is an I16_2 texel holding exactly the pair `(2q, 2q+1)`, so 512 fragments with 4 pair fetches replace 1024 fragments with 3 single-value fetches |
| `conv3_act` writes the q16 pair | 1024 24-bit texels and a 24-bit decode in the second gate; `mul_act` now takes the source layout per operand |
| literal width, `*0.00390625` | an `aw` uniform of the same value (a constant is not a uniform) and a division the driver lowers to `frcp + fmul` |

`conv3_act` and `mul_act` are also in `--dump-shaders` now: their text is the
conv numerics and no digest covered it before.

**Quality is unchanged.** Same container, same corpus, `SEQ=512`, Xe:

| container | serial | fused before | fused after |
|---|---:|---:|---:|
| base `lfm2.5.q4.dedup.mmcf` | 112.1544 | 126.8862 | **126.8862** |
| `base.i16.mmcf` | 98.0988 | — | — |

The fused number is identical to four decimals, and `regression.sh` is ACCEPT
(q8 270.3170, q4 330.5302 unchanged).  The q16 rounding of the conv result --
the one place the new path is *not* bit-exact by construction, because the 24-bit
store used to keep 1/256 of a q16 unit -- moves nothing measurable.

**Speed: not measurable on the board.**  Pi Zero, `SEQ=256 KV_PACK=16`,
`lfm2.5.qad.q4.mmcf` (the only container on the board), `--gen 24`, greedy,
`LAYER_PROFILE=1`, first token dropped (it carries the shader compile), two
interleaved rounds of each binary:

| stage | older binary r1 / r2 | this change r1 / r2 |
|---|---:|---:|
| `conv` (8 layers) | 205.3 / 172.8 | 200.0 / 184.9 |
| `attn` (6 layers) | 213.9 / 156.3 | 198.5 / 133.1 |
| `ffn` (14 layers) | 404.1 / 406.5 | 396.9 / 423.0 |
| `logits` | 231.3 / 173.9 | 175.1 / 174.0 |
| fastest token | **1115.2 / 1119.1** | 1119.6 / 1126.1 |

`attn` and `logits` were not touched and move by 14-58 ms, so the noise floor
here is ~±50 ms on a 1.1 s token; `conv` moves 5 ms one way and 12 the other.
The most stable statistic, the fastest token, is unchanged: 1115-1119 before,
1120-1126 after.  On Xe the same change is neutral too (conv 23.46 -> 23.53 of a
168 ms token).

So the estimate this work started from -- "pairing the fetches should save
80-100 ms/token on the Pi" -- was wrong, and the reason is worth keeping: the
conv block's ~190 ms is dominated by the three 1024-row **in_proj matvecs**
(3.15 MMAC per layer), not by the three elementwise passes.  At the engine's q4
rate those matvecs are most of it; the elementwise passes were a few ms per
layer and are now fewer.  Nothing on the elementwise side can win more than a
few ms until the conv weights themselves get cheaper, and a fused B/C/X draw
hits the 2048 hazard documented in `docs/engine-internals.md` section 5.  The
fetch reduction is a precondition for that, not a win on its own -- keep it for
being strictly less work at identical numerics, not for the clock.

Two things the measurements also say, which cost time to learn:

* **A single `decode avg` run cannot resolve 1% on this board.**  The first
  generated token pays the shader compile of every program it touches --
  measured 443 ms of `logits` in one run and 2964 ms in another with a new
  conv program -- and the SD card (load average ~5, 90 MB available) adds
  ±150 ms/token run to run.  Drop the first token, prefer `LAYER_PROFILE`'s
  per-stage lines (they are taken across a `glFinish`), and read the *fastest*
  token rather than the mean.
* **The 16 removed draws are not measurable on VC4.**  `DRAWCOUNT` shows 375 ->
  353 per token, and the time does not move.  The fused path already skips the
  per-pass finishes (`layer_fast_finish_`), so a draw on this board is ~10 us
  and 16 of them are noise.

The 16 removed draws are not measurable on VC4.**  `DRAWCOUNT` shows 375 ->
  353 per token, and the time does not move.  The fused path already skips the
  per-pass finishes (`layer_fast_finish_`), so a draw on this board is ~10 us
  and 16 of them are noise.

**"Three fetches from one sampler" is safe here, and that was worth checking.**
`probes/conv_probe` (new) drives the exact patterns: 1/2/3/5 fetches per
fragment from the 512x3 kernel texture, `.rg` and `.ba`, height 3 and 4, plus
the shipped `conv3_act`/`mul_act` against a CPU reference that mirrors the
shader's float32 order.  **Every case is byte-exact on both Xe and VC4.**  The
per-row scale plane's failure (`probes/wrow_probe`) is therefore shape-specific
-- a tall plane texture -- and not a rule about samplers, which is what the
critique of this block assumed.

Still open, found while measuring: an **i16** hybrid container on the fused path
is wrong (PPL 1.39e7 against 98.10 serial), and it was already wrong at
`f22bb76` -- the same number to four decimals before and after this work, so the
conv changes neither caused nor fixed it.  q4 containers, the path the board
runs, are exact.

## LFM2.5 answers badly because its activations do not fit the q16 grid (2026-11-11, Xe)

The model runs, the text is fluent, and it is wrong.  Measured against two
independent references on the same weights, the same prompt and -- after the
fix below -- the same token ids:

| reference | first token after `assistant\n` for "What is C. elegans?" |
|---|---|
| float32 numpy, from the same safetensors (`probe_logits.py`) | **544 `C`, logprob -0.042** |
| prebuilt llama.cpp b12283, official QAD-Q4_0 GGUF | **544 `C`, logprob -0.037** |
| this engine, i16 container, serial | 841 `ĠC`, logprob -0.215 |
| this engine, q4 container, serial | 841 `ĠC`, logprob -1.109 |

Perplexity on `docs/engine-internals.md` (7110 tokens, `SEQ=512`):

| implementation | PPL |
|---|---:|
| float32 numpy, same safetensors, same ids | **14.71** (on `data/ppl_test.txt`, 334 tokens) |
| llama.cpp b12283, QAD-Q4_0, 13 chunks | **44.20** |
| this engine, QAD q4, serial, 14 windows | 225.56 |

So a 230M model that is merely small in llama.cpp is unusable here, and the
gap is not the quantization (i16 is just as wrong as q4) nor the tokenizer
(prose tokenizes identically, 334/335 ids) nor the GPU (serial and fused agree
to a few PPL points).

The first wrong number is the first normalization.  `Session::embedding()`
quantizes every component of the embedding row onto the activation grid
(step 1/256, rail 128).  LFM2.5's embedding row peaks at **0.05** and is
typically 0.008: at that size the grid keeps 2-3 bits per component, and
components below 0.002 become exactly zero.  The operator norm then amplifies
the damage, because it divides by the vector's RMS:

| layer 0 `xb` (operator_norm output), position 0 | float32 numpy | engine |
|---|---:|---:|
| channels 0-5 | 0.2291 0.2390 -0.0277 -0.1201 0.1244 -0.1798 | 0.2344 0.2188 **0.0000** -0.1094 0.1133 -0.2227 |

8.8% relative error on the first activation, cos 0.996; it accumulates to
cos 0.92 by layer 12, which is enough to pick the wrong token.

`ACT_F32=1` removes every q16 store from the CPU layer loop as a probe.  It
fixes the first norm (`xb` then matches numpy to four digits) but only moves
the q4 container 117.08 -> 115.81 PPL, and makes the **i16** container *worse*
(98.10 -> 107.88) -- so the activation grid is the first error, not the only
one.  The remaining ones are unexplained; do not treat this section as a fix.

### What the comparison did fix

* **Chat templates never saw their special tokens.**  All three render sites
  passed `""` for `bos_token`/`eos_token`, so LFM2.5's template (which starts
  with `{{- bos_token -}}`) rendered no BOS at all and every chat prompt began
  at `<|im_start|>`.  Passing the container's token text is not enough on its
  own: the context is marked as caller input, so `"<|startoftext|>"` was
  tokenised as four text tokens (`<`, `|`, `start`, `oftext`).  The two strings
  are now marked as template text (`ctx.set_val`), and all six template cases
  -- plain, no system, tools, multi-turn, think-block, tool round-trip --
  render byte-identically to Jinja2 *and* tokenise to the same ids as
  llama.cpp's `/apply-template` + `/tokenize` (31/31, 16/16, 110/110, 52/52,
  140/140; the think case differs on purpose, llama.cpp preserves reasoning by
  default).
* **`--ppl` scored only the first window.**  It now walks every window
  (`windows=N` in the line).  Every pinned number is unchanged because those
  corpora are shorter than `SEQ` (333 < 512, 293 < 768) -- but a 7110-token
  file used to report `tokens=510` and now reports `tokens=7115 windows=14`.
* **`mmcf_dump.py --template`** prints the container's Jinja text, which is how
  the 4762-byte template was compared with the GGUF's `tokenizer.chat_template`
  (byte-identical) in the first place.

### The tokenizer is not LFM2's for numbers

`tokenizer.ggml.pre = 'lfm2'` makes llama.cpp use the **llama-3** pre-tokenizer
with `ignore_merges`: digits split as `\p{N}{1,3}`, which is why "6789" becomes
`678``9` there and `6``789` here.  Ordinary prose is identical (334 of 335 ids
on the PPL corpus), and 15 of 20 hand probes are identical; the five that
differ all contain digit runs of 4+ or a double space (`"a  b"`).  Since the
model is asked arithmetic, this matters, and it is not fixed yet.

Reproduce: `models/lfm2.5-230m/probe_logits.py --ids <ids>` (float32 logits
from the safetensors), `ACT_F32=1`, `CHAT_DUMP=1` (now also prints
`PROMPT_IDS`).

### The q16 grid is *not* the 6.7x (2026-11-11, Xe)

The previous section named the activation grid as the first error and left it
there.  A simulation settles how much of the gap it explains: run the float32
numpy reference with `_q()` at every store the engine makes (embedding, every
matvec result *and* input, the conv gate/cv/y, `att`, both residual adds, the
SwiGLU product) and it costs almost nothing:

| reference | PPL (`data/ppl_test.txt`, 334 tokens) |
|---|---:|
| numpy f32 | 14.71 |
| numpy f32 + the engine's q16 stores everywhere | **15.78** |
| this engine, QAD q4, serial | 117.08 |
| this engine, base i16, serial | 98.10 |

So the fixed-point format is worth 7% and the engine is 6.7x off: **the
remaining error is in the engine's maths, not in the format.**  That is the
opposite of what the section above implies, and it moves the search.

Instrument first (`DUMP_F32=1` prints the raw IEEE bits instead of q16 texels;
the q16 dump's 1/256 was coarser than the differences being hunted).  Then:

* **The head is exact.**  Applying numpy's final norm and tied head to the
  engine's *own* layer-13 output reproduces the engine's logits to cos
  0.999999.  Every logit difference comes from the hidden state.
* **The body drifts.**  On a 20-token prompt the engine's layer-13 output is
  cos 0.86 against numpy at the last position (52% relative), and layer 0's
  first position is already 68% off.
* **Layer 0's conv output is quantised away.**  Its second gate is
  `y = C * cv` with `cv ~ 1e-4` at the first position; `y` then goes through
  `gpu_.matvec()`, which uploads it as a q16 texture (step 1/256), so `y`
  becomes zero and the conv layer contributes nothing.  That is a real defect
  and `ACT_F32` does not fix it (that flag only stops the CPU-side stores).
* **Attention at the first position is 8.8% off**, although a single position
  makes the softmax exactly 1 and so `att` must equal `v` (measured against `v`
  recomputed from the engine's *own* `xb`, cos 0.9962).  `v` there is order
  0.1, i.e. 4% per component on the 1/256 grid -- consistent, and not
  explained by the grid alone.

The instrumented comparison is `refs/llamacpp/scripts/bisect_stages.py DUMP
--f32`, one line per (layer, position, stage) with cosine and relative error,
and `refs/llamacpp/scripts/np_ppl.py IDS --q16` for the simulation above.
`probe_logits.py` in the model store gives the float32 logits for any id list.

## Correction: every LFM2.5 PPL above is 5x too high, because `--ppl` sent no BOS (2026-11-11)

`--ppl` tokenised with `encode(text, false, false)`: no BOS, no dummy prefix.
No template is involved in a perplexity run, so nothing else supplied the
`<|startoftext|>` that LFM2.5's template starts with.  The model was being
scored on a prompt it never sees.  On the same 334-token stream:

| tokenisation | PPL (base i16, serial) |
|---|---:|
| without BOS (what every LFM2.5 number in this file used) | 98.10 |
| **with BOS** (what the model expects, and what llama.cpp does) | **15.97** |
| float32 numpy reference, without BOS | 85.44 |
| **float32 numpy reference, with BOS** | **14.71** |

So the engine was never 5x off the reference: **15.97 against 14.71 is 8.6%**,
and the two sides agree on the token stream and on the first generated token.

The corrected numbers, `SEQ=512`, `data/ppl_test.txt` (334 scored tokens):

| container | path | before (no BOS) | **after (BOS)** |
|---|---|---:|---:|
| base i16 | serial | 98.10 | **15.97** |
| base i16 | serial, `ACT_F32=1` | 107.88 | 15.20 |
| QAD q4 | serial | 117.08 | **18.73** |
| QAD q4 | serial, `ACT_F32=1` | 115.81 | 18.48 |
| QAD q4 | fused (`GPU_LAYER=1 CONV_GPU=1`) | 114.53 | **19.52** |

Three things follow, and they change conclusions this file drew before:

* **The fused path is 4% worse than serial, not 2% better.**  `19.52` against
  `18.73` is the physically expected direction; the old pair (`114.53` against
  `117.08`) inverted it because both were dominated by the missing BOS.
* **MiniMind3's pins do not move** (`270.3170` / `330.5302`): its container has
  no BOS id, so `encode(text, true, false)` is byte-identical to before.  The
  change only affects checkpoints that *have* a BOS.
* **The "quality is bad" investigation had two real engine bugs in it anyway**,
  found while chasing the phantom gap.  They are fixes, not noise:
  * the chat template never received `bos_token`/`eos_token` (previous commit),
    so chat prompts had no `<|startoftext|>` either;
  * `gpu_.matvec()` uploads its activation as a q16 texture, and LFM2.5's conv
    gate output at the start of a sequence is order 1e-4 -- a handful of grid
    steps, often literally zero, so the conv layer contributed nothing.
    `GPU::matvec_s()` picks a power-of-two scale from the input's own maximum,
    uploads `x * 2^k` and divides the result back.  With the serial path routed
    through it, layer-by-layer agreement with the float32 reference went from
    6-11% to **0.2-1.4% per layer**, and the first generated token for
    "What is C. elegans?" is now `544 "C"` at logprob -0.01, where the float32
    reference says -0.042 and llama.cpp says -0.037 (the engine previously
    answered `841 "ĠC"`).

What is still true from the earlier section: an **i16 container on the fused
path** is broken (PPL 1.4e7), and that is pre-existing at `f22bb76`.  The i16
*serial* path is the 15.97 above.

### The same fix on the board (2026-11-11, Pi Zero)

`bin/engine_mm3_pi.bos`, cross-built from `4771eac`, same container
(`lfm2.5.qad.q4.mmcf`), `SEQ=256 KV_PACK=16`, `GPU_LAYER=1 CONV_GPU=1`, greedy,
24 tokens.

What the two binaries answer, same prompts, same seed-free path:

| prompt | before (no BOS in the template) | after |
|---|---|---|
| What is C. elegans? | "a type of insect that belongs to the order Coleoptera" | **"a small, round, worm-like organism that is commonly used in research to study genetics"** |
| Write a haiku about rain. | "A gentle rain, a quiet moment." (one line, not a haiku) | **"Rain whispers softly,"** |
| 用一句话介绍你自己。 | **echoed the prompt**: "用一句话介绍你自己。" | **"我是一个AI助手，帮助您理解和交流。"** |
| Compute 12345\*6789. | "To compute 12345 * 6789, follow these steps:" | "12345 \* 6,789 = …" (both wrong; a 230M model) |
| List three uses of a paperclip. | "Three uses of a paperclip are:" | "Here are three common uses of a paperclip:" |

The echo of the Chinese prompt is the clearest symptom the missing BOS caused:
with no `<|startoftext|>` the model treats the first line as a continuation and
repeats it.  Three of five prompts change for the better, two are neutral.

Board PPL, `SEQ=256`, `data/ppl_test.txt` (334 scored tokens, two windows --
not comparable with the `SEQ=512` rows above):

| path | PPL |
|---|---:|
| fused (`GPU_LAYER=1 CONV_GPU=1`) | 45.52 |
| serial (with `matvec_s`) | **43.91** |

Serial is 3.7% better, which is the same direction as on Xe (`18.73` against
`19.52`); the old board numbers had it the other way round for the same reason
the absolute values were wrong.

The fused path gets the BOS fix but **not** `matvec_s`: that scales the
*serial* path's matvex inputs, while the fused path uploads through its own
I16_2 targets, which still have no scale.  Making the fused path's targets
carry one is the next step, and it is the difference between the board's
quality today and the serial path's.

### Artifacts deleted after these measurements (2026-11-11)

The containers a few numbers above were measured on are gone, to reclaim 1.6 GB:
`lfm2.5.q4.mmcf` (tiled head, 235 MB), `base.i16.mmcf` and `qad.i16.mmcf`
(563 MB each), and the two base GGUFs (Q4_0, Q4_K_M; the vendor *recipes* are
in `docs/lfm2.5-230m-assessment.zh-CN.md`).  The surviving containers are
`lfm2.5.q4.dedup.mmcf`, `lfm2.5.q4.q6.mmcf` and `qad/lfm2.5.qad.q4.mmcf`, plus
the QAD-Q4_0 GGUF that llama.cpp is compared against.  Rebuild what is missing
with

```bash
python3 engine/tools/mmcf_pack.py --in models/lfm2.5-230m --quant i16 --seq 512 \
  --tok models/lfm2.5-230m/lfm2.5.tok --out models/lfm2.5-230m/base.i16.mmcf \
  --name lfm2.5-base-i16 --template chatml --chat \
  --chat-template models/lfm2.5-230m/chat_template.jinja --verify
```

(the `q4` recipe for the base container is the same command with `--quant q4`,
`--head-dedup` dropped and a `q4` output name; the QAD one is quoted in the
assessment doc).

### Open: our PPL degrades window by window (found 2026-11-11, not fixed)

`--ppl` now scores every window, and on a 1134-token Markdown file
(`docs/engine-internals.md`, first 3600 bytes, `SEQ=512`) the per-window NLL is
not flat:

| window | our mean NLL | llama.cpp's chunk |
|---|---:|---:|
| 0 | 4.23 | 3.96 |
| 1 | 5.88 | 3.88 |
| 2 (111 tokens) | 7.41 | — |

Total: ours **197.67** (QAD q4) against llama.cpp's **48.28** on the same text
at the same context.  The first window agrees to 7%; every later one is worse,
and the degradation grows with the window index (4.23 -> 5.88 -> 7.41), which is
the shape of state carried across windows, not of harder text -- llama.cpp's
second chunk is *easier* than its first.

This does not affect any number measured on a short corpus (every pinned PPL,
and every engine-vs-numpy comparison in this file, is one window: `14.71` numpy
against `15.97` engine is window 0).  It does affect long-context perplexity,
and it is the next thing to chase: `--ppl` builds a fresh `Session` per window,
so the suspects are the state that lives in `GPU` rather than in `Session`
(the fused path's conv targets, the shared `Target` pool) and the rope/KV
boundary itself.  Reproduce with

```bash
head -c 3600 docs/engine-internals.md > /tmp/c3.txt
SEQ=512 GPU_LAYER=0 PPL_DUMP=/tmp/nll.txt ./engine_mm3 --ppl /tmp/c3.txt   # from engine/
```

(`refs/llamacpp/` is gitignored and local to this machine: a prebuilt
llama.cpp b12283 — CPU and Vulkan — the comparison scripts and a venv, so its
numbers and commands only run where that directory exists.  `refs/llamacpp/README.md`
records the two archives' official sha256 and how to re-create it.)

### Correction: the window-by-window "degradation" was a cold window mean (2026-11-12, Iris Xe)

The section above called it state carried across windows.  It is not: the engine
was right and the measurement was wrong.  `--ppl` built a fresh `Session` per
window and scored *every* token of it, so every window after the first was a
**cold** start -- 511 tokens cut out of the middle of a document with no context
in front of them.  On this text that costs about **1.8 nats/token**, and the
apparent growth with the window index was the third window being short (111
tokens, all of them cold prefix).  The float32 numpy reference says the same
thing about the *model* and nothing about the engine:

| window | cold: fed / scored | warm: fed / scored | engine q4, cold | f32 numpy, cold | engine q4, warm | f32 numpy, warm |
|---|---|---|---:|---:|---:|---:|
| 0 | 0-511 / 1-511 | 0-511 / 1-511 | 4.2304 | 3.7990 | 4.2304 | 3.7990 |
| 1 | 511-1022 / 512-1022 | 256-767 / 512-767 | 5.8826 | 5.6112 | 5.5400 | 5.1131 |
| 2 | 1022-1133 / 1023-1133 | 512-1023 / 768-1023 | 7.4052 | 7.3714 | 5.1153 | 4.9242 |
| 3 | -- | 768-1133 / 1024-1133 | -- | -- | 5.7050 | 5.5469 |
| **total** | 1133 tokens | 1133 tokens | **5.2866** (PPL 197.67) | **4.9663** (143.50) | **4.8694** (130.24) | **4.5199** (91.82) |

("fed" is the window's token range, "scored" the targets it accumulates; the
warm protocol of window 1 feeds the 256 tokens before the ones it scores.  The
ranges are half-open in the code but written inclusive here.)

The engine sits 0.32 nats above f32 when the windows are cold, 0.35 when they
are warm, and 0.33 on the whole file in one window (`SEQ=2048`: 3.8951 against
3.5645) -- the q4 container's cost, identical under all three protocols.  A
window-dependent defect would not be constant.

The llama.cpp number that started this was not a cold-window mean either.  On
the same file at `-c 512` it prints `[1]52.4995,[2]48.2827`: the overall 48.28
matches our **full-context** 49.16 (3.8951 nll, one window), not our cold-window
197.67.  Its chunked perplexity keeps the context of the earlier chunks, so it
is a full-context estimate; its own `--ppl-stride` is the option that says so
explicitly, and its chunk means at small `-c` can even fall *below* the f32
truth (first 255 tokens: 3.08 against 3.66), which no quantised model can.

There is nothing to fix in the session or the conv state.  The fix is the
metric: since this correction `--ppl` **warms each window up**.  A window feeds
`SEQ` tokens and scores only the last `SEQ/2` (`stride=` in the output;
`PPL_STRIDE=n` changes it, `PPL_STRIDE=0` restores the old cold windows), which
is the sliding-window protocol of HuggingFace's `evaluate` and llama.cpp's
`--ppl-stride`.  Corpora shorter than `SEQ` are unaffected -- every pinned
number is one window, and `PPL 18.7251 tokens=334 windows=1` is what the q4 QAD
container still prints on `engine/data/ppl_test.txt`.

Same file, three protocols, both engines (`SEQ=512` unless noted):

| protocol | engine q4 serial | engine q4 fused | f32 numpy | windows |
|---|---:|---:|---:|---:|
| cold, `PPL_STRIDE=0` | 197.6719 | 199.9021 | 143.4955 | 3 |
| warm, stride `SEQ/2` | 130.2420 | 127.6320 | 91.8232 | 4 |
| one window, `SEQ=2048` | 49.1586 | -- | 35.3204 | 1 |

Reproduce (the corpus is the file as it was at `f79dc6c`, so the numbers above
stay reachable after this document grows):

```bash
git show f79dc6c:docs/engine-internals.md | head -c 3600 > /tmp/c3.txt
SEQ=512 GPU_LAYER=0 PPL_DUMP=/tmp/nll.txt ./engine_mm3 --ppl /tmp/c3.txt   # warm
SEQ=512 GPU_LAYER=0 PPL_STRIDE=0 ./engine_mm3 --ppl /tmp/c3.txt            # cold
SEQ=2048 GPU_LAYER=0 ./engine_mm3 --ppl /tmp/c3.txt                        # one window
```

The per-window columns come from bucketing `PPL_DUMP` by the window start the
`stride=` line implies; the f32 columns from a numpy forward over the same
windows (`refs/llamacpp/scripts/np_windows.py` prototype, same protocol,
safetensors source, no container).

Every number in this file that spans more than one window is a cold-window sum
for the same reason, and stays reachable only with `PPL_STRIDE=0`: the board's
LFM2.5 rows at `SEQ=256` (`43.91` serial, `45.52` fused) are the same corpus
that Xe now scores `42.63` / `45.00` cold and `28.82` / `30.10` warm.  Single
window rows -- every MiniMind3 pin, the `SEQ=512` LFM2.5 rows, `14.71` against
`15.97` -- are untouched: one window has nothing to warm up on, and the loop for
it is the same code as before (`PPL 18.7251 tokens=334 windows=1` on the q4 QAD
container, before and after).

### Board re-measure under both PPL protocols, and the deploy that made it possible (2026-11-12, Pi Zero)

The correction above was Xe-only for a few hours because the board runs a
standalone binary.  Cross-built it on vlab from `ee7929b` (ARMv6/VFPv2, 1546980
bytes, md5 `aefc2e99dde64d61b22f4978bc039a52`), uploaded to
`/home/david/vc4-llm-runtime/bin/engine_mm3_pi`, previous binary kept as
`engine_mm3_pi.4771eac`, corpus copied to
`/home/david/vc4-llm-runtime/data/ppl_test.txt`.  Same container
(`lfm2.5.qad.q4.mmcf`), `SEQ=256 KV_PACK=16`, 334 scored tokens, two windows:

| Pi Zero, QAD q4 | cold (`PPL_STRIDE=0`) | warm (default, `stride=128`) |
|---|---:|---:|
| serial (`GPU_LAYER=0`) | 43.9130 | **28.6196** |
| fused (`GPU_LAYER=1 CONV_GPU=1`) | 45.5248 | 31.6824 |

Both cold values are the two numbers this file recorded on 2026-11-11 (`43.91`
serial, `45.52` fused), reproduced to four digits -- the refactored window loop
is a no-op for the old protocol, which is the check that matters for a change to
a measurement path.  Warm is what a long file now scores: the protocol alone is
worth 35% on this corpus and board.

The fused path's gap is protocol-dependent on the board: 3.7% behind serial
cold, **10.7%** behind warm (Xe: 5.5% and 4.4%).  So the board's older "serial
is 3.7% better" understated what the fused path costs there; the warm protocol
is the one to quote.

Deploy checks beyond PPL, same binary, fused `SEQ=256`, greedy, default system
prompt unless noted:

| prompt | answer |
|---|---|
| What is C. elegans? | `C. elegans is a small, round, worm-like organism that is commonly used in research to study genetics, physiology` (24-token limit) |
| Write a haiku about rain. | `Rain whispers softly,` / `a gentle sigh of the sky.` (two lines now; the 2026-11-11 run stopped at the first) |
| 用一句话介绍你自己。 | `我是一个AI助手，帮助您理解和交流。` |
| MiniMind3 q8 tripwire, `SEQ=768`, `--system "You are a helpful assistant."` | `我是MiniMind，一个由jingyaogong` -- the pinned Pi string, unchanged |

Nothing about decode speed moved: this is a measurement-path change, and the
board still runs ~1040 ms/token on the fused path.  The board's acceptance pin
is untouched as well -- MiniMind3 q8 at `SEQ=768` on `data/ppl_test.txt` prints
`PPL 271.6420 tokens=293 windows=1`, the value pinned in `AGENTS.local.md`
(one window, so the window protocol cannot move it).

## MiniMind3-MoE audit: llama.cpp oracle, the activation rail, and the fixes (2026-11-12, Xe)

An audit of the first MoE block (container v3, see
[`checkpoint-container.md`](checkpoint-container.md)) against two independent
implementations and against its own containers.  The short version: the block's
semantics are right and the containers are faithful; what remains is
quantization, one silent format clamp the packer did not check, and a set of
tooling gaps.

Oracles, all on the same 30 prompt ids (the container's own chat render of
system `You are a helpful assistant.` + `Hello`):

* **llama.cpp b12283**, the upstream CI build in `refs/llamacpp/bin/b12283-cpu`
  (used as shipped, **not** compiled here), on a pure-python F16 GGUF of the
  safetensors source: log probability max |d| **3.2e-05** over the full
  6400-token vocabulary, same argmax, same top-5, same greedy chain (9 tokens
  then EOS), PPL **370.9727**.
* **torch f32** (transformers `Qwen3MoeForCausalLM`): PPL 370.9732, same
  greedy chain.
* **numpy f32** (`engine/tools/ref_forward.py`): PPL 370.9727, 240/240 routes
  identical to the torch chain.

Every engine-vs-reference delta below is therefore the engine's arithmetic, not
a different model.  The engine, same ids, `GPU_LAYER=1 KV_PACK=16`:

| container | logits vs f32: max |d| / mean |d| / cos | greedy | PPL (293 tokens) |
|---|---|---:|---:|
| i16 | 0.775 / 0.140 / 0.99856 | 9 tokens + EOS, identical | 389.3532 (+5.0%) |
| q8 | 1.093 / 0.183 / 0.99751 | identical | 350.8860 (-5.4%) |
| q4 | 3.974 / 0.690 / 0.96484 | identical | 437.7821 (+18%) |

The i16/q8 PPL inversion is **bias**, not noise: per-token NLL against f32 is
+0.048 nats/token (i16) and -0.056 (q8) with the same scatter, and a float32
chain that applies *only* the documented q16 activation stores reproduces the
i16 end-to-end numbers (logits 0.830 / 0.144 / 0.99847, 9 route flips against
the engine's 4).  The q16 grid compresses the logit vector by about 0.5% (fit
slope 0.99471 in the simulation, 0.99349 in the engine); q8's weight noise
happens to push the other way on this corpus, which is all the "inversion" is.

Routing over 30 positions (240 layer-positions): i16 4, q8 3-5, q4 26-29 flips
against f32, every one at a f32 top-2 margin of 0.002-1.3.  The pure-grid
simulation itself flips 7/240, so the router is as faithful as the activation
format allows — it is not a MoE bug.  The router stays i16 in every container
(q4 included) and the expert map is the identity (12x12 cosine; the container
audit's numbers).

**The one hard defect: the FFN hidden rail.**  The expert SwiGLU hidden
(`silu(w1*xbf) * w3*xbf`) is stored through the q16 activation format and
clamps at +/-127.996.  On a chat prompt it reaches **158/161** (layer 3) and
**250/222** (layer 7) at the template's `<|im_start|>` positions, i.e. 49% of
the layer-7 value is cut; clipping the f32 hidden reproduces the engine's FFN
output at those positions to 0.02-0.6% (unclipped: 4-25%).  The dense 64M model
peaks at 88.8 and the plain-text PPL corpus at 15.1, so `--ppl` cannot see it.
The numeric fix (a per-tensor power-of-two scale on the hidden store, folded
into the w2 read-back) is **not** landed — it needs its own A/B and
re-baseline.  The packer now warns, and its calibration samples the tokenizer's
added tokens, which is what makes the chat-template positions visible at pack
time: see [`model-formats.md`](model-formats.md), "The packer checks the
residual and FFN-hidden rails".

Landed with the audit, all verified:

* packer: a MoE container always gets `split_w13 = 1` (an expert width <= 1024
  used to pack a file the loader then refused); the q4 record's `group` byte is
  the real group size; the skew report covers the expert matrices.
* engine: the CPU layer loop's RMSNorm uses the container's eps instead of a
  hardcoded 1e-5 (the CPU-vs-GPU xb difference halves on a model that states
  1e-6); `MODEL_BORROW=0` works for q4 (the copy path truncated the group
  planes) and keeps a copy of the tokenizer blob instead of pointing into the
  unmapped file; q4 planes are shape-checked at load.
* tools: `--verify` compares every record field with the source tensor (a
  swapped expert `aux` now fails instead of running the wrong expert);
  `mmcf_requant_q8.py` refuses a non-i16 source; `mmcf_dump.py` counts MoE
  expert tensors and states the default eps when key 11 is absent;
  `ref_stages.py` gained MoE support (its dense multi-token prefix bug is
  documented but left as it was).
* `regression.sh`: optional `MOE_ROUTES` baseline, so the discrete part of a
  MoE can be pinned per machine, not only the text.

If the llama.cpp comparison is repeated: the host's
`~/.config/llama.cpp/config.ini` sets `cache-type-k/v = q8_0` and
`flash-attn = on`, which moves log probabilities by up to 0.148 (four orders of
magnitude over the real agreement) unless `-ctk f32 -ctv f32 -fa off` is
passed, and `ignore_eos` fabricates a greedy divergence by banning EOS.  Also
`llama-perplexity -c 768` cannot run on a 294-token corpus (it wants at least
2*n_ctx tokens), so the PPL above was measured per position through
`/completion` with `n_probs=n_vocab`.


## The FFN hidden rail: per-layer 2^-k on the store (2026-11-12, Xe)

This supersedes the "numeric fix is not landed" note in the audit section
above; the fix landed the same day.  The pattern is the one `GPU::matvec_s()`
established for a matvec input: one exact power-of-two scale at the point where
a value crosses the fixed-point grid.

**How it works.**  The packer's float32 calibration (three vocabulary runs plus
one run per tokenizer added token) yields a per-layer hidden peak.  Key 15
carries one byte per layer: `k` is the smallest exponent with
`peak / 2^k <= 127.996`, capped at 5.  The serial FFN kernels write the SwiGLU
hidden `* 2^-k` (one multiply before the q16 encode; the emitted GLSL is
byte-identical when `k = 0`, so `gate/shader_hashes.sh` is untouched), and
`Model::apply_hidden_scale()` multiplies every w2 and w2 chunk by `2^k` after
`split_ffn()`.  The read side needs no shader change: `ws`/`ps` already carry
a per-matrix scale, and the q8 row-scale path multiplies the same unit.  The
two-position prefill kernels have no hidden scale, so `pair_ok()` refuses a
scaled container's prefill pair (the MoE block already refused it); scaled
dense models would fall back to the serial loop.

For MiniMind3-MoE the calibration gives `k = 0 0 0 1 0 0 0 1` (layers 3 and
7; store peak 249.9 -> 124.9).  Repacked from the same safetensors, `SEQ=768`,
30-token chat prompt, `GPU_LAYER=1 KV_PACK=16`:

| measurement | shipped (no key 15) | repacked (k=1 on L3/L7) |
|---|---:|---:|
| q8 `hidA` at l3 t0: max error vs float32 | 1.748 | **0.247** |
| q8 `hidA` at l7 t0: max error vs float32 | 7.668 | **0.349** |
| q8 `hidA` mean error, all 240 positions | 0.0864 | **0.0624** |
| i16 `hidA` at l7 t0: max error | 7.589 | **0.217** |
| q8 logits vs float32: mean / cos | 0.1834 / 0.997507 | **0.1764 / 0.997707** |
| i16 logits vs float32: mean / cos | 0.1405 / 0.998571 | **0.1261 / 0.998847** |
| routes, 304 lines (prompt + 8 decoded) | -- | identical |
| greedy text | `Hello! How can I assist you today` | identical |

PPL, same 293-token corpus.  That text never crosses the hidden rail (peak
15.1), so this table is the *resolution cost* of one bit on two layers, not the
clip:

| container | shipped | scaled |
|---|---:|---:|
| i16 | 389.3532 | 391.1756 (+0.47%) |
| q8 | 350.8860 | 352.5846 (+0.48%) |
| q4 | 437.7821 | **426.6095 (-2.55%)** |

+0.005 nats/token is well inside the per-token scatter (sd ~0.3-0.4 on 293
tokens); q4 moves the other way because its weight noise dominates.  The trade
is deliberate: the clip cut 49% of the layer-7 hidden at the template's
`<|im_start|>` positions on *every* chat prompt, and that is gone.  A model that
stays under the rail gets no key 15 and a byte-identical container: MiniMind3
dense peaks at 88.3 and the plain-text corpus at 15.1.

**What this does not fix.**  The scale is per layer, not per position; a
per-token dynamic scale (the way `matvec_s` works) would avoid the resolution
cost but needs a readback per layer per token, and that readback is exactly
what the MoE route already pays once per layer.  Containers already in the
store predate key 15 and still clamp until re-packed — `models/minimind3-moe/`
was not touched by this change.  Re-pack with the usual command; the packer
prints the chosen exponents next to the peaks.

## 2026-11-12 — deliberate shader re-baseline (hybrid q8 conv fix)

`matvec_q8_*_i16_2` (the I16_2 fast path of the non-pair q8 matvec) changed
text: 839 B -> 748 B, digest `3ca31cd46926e2fb` -> `470ba9d4bf6c94e7`.  It used
to encode its result as an unsigned q16 (adding 65536 for a negative sum) and
*then* add the q24 bias, so every negative output came back 256 too large.
The only user is the LFM2 conv block (dense q8 models never call the non-pair
path), which is why no existing PPL pin moved: laptop q8 PPL 270.3170 and q4
330.5302 are unchanged, `scripts/regression.sh` ACCEPTs, and the golden strings
reproduce.  The hybrid q8 container now translates (`This product is out of
stock.`) instead of emitting noise; `docs/lfm2.5-230m.md` gap 8 has the staging
evidence.

## 2026-11-12 — HEAD deployed; the re-calibration stopped at the board's storage (Pi Zero)

The board had been running a binary from `4ffb536` (the warm-window `--ppl`
deploy).  It was one code commit behind, and that commit matters: `6c2a0cd`
carries the hybrid fused-loop repairs, the MoE per-layer FFN-hidden store scale
(metadata key 15), the container-stated RMS eps and `MOE_ROUTES`.  A fresh
cross-build was made on vlab from `f8b3fb8` (docs-only on top of `6c2a0cd`),
ARMv6 / VFPv2, 1,560,176 B, md5 `a171cd94f778e3d35f2c6b634b6888dc`, and deployed
over the old one, which is kept as `engine_mm3_pi.aefc2e99` (`4ffb536`, md5
`aefc2e99dde64d61b22f4978bc039a52`).

What the new binary was checked on before the board became unusable:

* MiniMind3 q8, `SEQ=768`, `--chat`, pinned system prompt, greedy, `--gen 16`:
  `我是MiniMind，一个由jingyaogong` -- the Pi acceptance pin, unchanged.  So
  `6c2a0cd` does not move the board's tripwire.
* seven of the twenty containers in the runtime store were started and emitted
  at least one token, all `GPU_LAYER=1 KV_PACK=16`, greedy:
  `minimind3.q8` (768), `minimind3-moe.q4` (768), `lfm2.5.qad.q4` (256,
  `CONV_GPU=1`), `tinybrainbot.q4` (512), `picolm85.q4q24` (512),
  `void1.q8r` (512) and `vertex100.q4` (512).  The other thirteen were **not**
  exercised on this build: either feng container, the MiniMind2 trio, the three
  stories containers, `minimind3.q4`, `minimind3-moe.q8`, `vertex15`,
  `vertex35` and `void1.q4`.  That is a coverage check, not a claim that they
  run.
* **No usable timing.**  Every run from 20:37 on was concurrent with an
  inventory sweep reading ~1.5 GB from the same card.  The first MiniMind3 q8
  run (prefill 27.6 s, `decode_avg` 219.1 ms/token, `--gen 16`) is contended,
  and the repeats show the board rather than the model: two identical `--gen 8`
  runs at 21:05 gave prefill 23.1 vs 34.0 s and 267.1 vs 492.7 ms/token.  No
  row in this file may be quoted from them.

Why it stopped, recorded for the next person:

* The card is a white-label 32 GB SD (`SD32G`, manfid `0x009f`, oemid `0x5449`,
  date `08/2024`), 91% full.  Every boot prints `mmc0: Problem switching card
  into high-speed mode!`; `/sys/kernel/debug/mmc0/ios` reports 25 MHz **legacy**
  timing at 4-bit, and `dd` measures ~8.5 MB/s.
* Under load the block layer returned EIO at **eight distinct LBAs** spread over
  6.3-27.6 GB (`I/O error, dev mmcblk0, sector ...`), and ten pages of the
  512 MB `/var/swap` file could not be read back (`Read-error on swap-device`).
* With 437 MB of RAM and ~190 MB in swap, a failed swap-in delivered SIGBUS to
  `tailscaled` (each Go crash dump is tens of KB); journald wrote them to the
  same card at 45% CPU, load reached 20-27, and the 5-minute hardware watchdog
  reset the board three times in twenty minutes.  `tailscaled`'s binary
  verified clean (`dpkg -V`), and the seven failed LBAs that were re-read
  afterwards all read OK, so no permanent bad block is proven -- what is proven
  is that this storage cannot carry a benchmark.
* Therefore: do not run a Pi matrix on this board until the card is replaced (or
  swap is moved off it and the resident services are stopped), and do not run a
  benchmark while a tool sweeps the model store -- the failing reads coincided
  with exactly that.

## 2026-11-13 — the Pi campaign on HEAD: 14 containers, protocol v1 (Pi Zero)

Binary: HEAD `71cb1e0` (`6c2a0cd`'s code plus the fused-W13 guard found on
2026-11-13), cross-built on vlab, ARMv6 / VFPv2, 1,560,176 B, deployed over the
2026-11-12 build (kept as `engine_mm3_pi.2026-11-12`).  Board otherwise as it
ships: `pumpkin` resident, swap on the SD card, load 0.2-3.7 during the runs.

Protocol v1 was adopted after the first three models showed that a cold page
cache inflates the first run by up to 50%: before each model `drop_caches`, then
one **discarded warm-up** run (it faults the mmap'd embedding rows the timed
runs will touch), then **three timed runs, 20 s apart**, median reported.  All
runs are `GPU_LAYER=1 KV_PACK=16`, greedy, `--gen 16`; `SEQ` and prompt are per
row.  Rows marked † predate the warm-up rule and carry the noise it explains.

| model | container | SEQ | prompt | decode (median, range) | prefill |
|---|---|---:|---|---:|---:|
| MiniMind2 q8 † | `minimind2-small.q8.mmcf` | 512 | chat, `用一句话介绍你自己。` | **123.5** (123.2-123.7) | 7.3-7.9 s |
| stories15M q8 † | `stories15M.q8.mmcf` | 256 | `Once upon a time` | **116.2** (114.8-117.7) | 3.1-3.9 s |
| TinyChat15M q8 | `TinyChat15M.q8.mmcf` | 256 | `Once upon a time, there was a little girl named` | **115.4** (114.5-161.2) | 3.6-4.5 s |
| minimind2 q4 | `minimind2-small.q4.mmcf` | 512 | chat, `用一句话介绍你自己。` | **153.7** (152.1-154.8) | 8.9 s |
| stories42M q8 | `stories42M.q8.mmcf` | 1024 | `Once upon a time` | **160.2** (158.4-160.4) | 4.3-6.7 s |
| feng q8 † | `feng.q8.mmcf` | 512 | chat, `用一句话介绍你自己。` | **192.1** (189.2-194.3) | 8.2-14.8 s |
| feng q8r † | `feng.q8r.mmcf` | 512 | chat, `用一句话介绍你自己。` | **192.8** (189.3-197.9) | 8.0-16.9 s |
| vertex15 q4 | `vertex15.q4.mmcf` | 512 | chat, `What is the capital of France?` | **196.9** (193.4-211.4) | 8.6-8.7 s |
| vertex35 q4 | `vertex35.q4.mmcf` | 512 | chat, `What is the capital of France?` | **241.4** (224.0-265.1) | 8.9-9.1 s |
| MiniMind3 q4 | `minimind3.q4.mmcf` | 768 | chat, `用一句话介绍你自己。` | **304.4** (303.7-310.0) | 16.0-35.9 s |
| MiniMind3-MoE q4 | `minimind3-moe.q4.mmcf` | 768 | chat, `用一句话介绍你自己。` | **322.0** (319.3-345.3) | 18.2-22.5 s |
| MiniMind2 i16 † | `minimind2-small.i16.mmcf` | 512 | chat, `用一句话介绍你自己。` | **326.8** (322.7-372.3) †† | 11.8-19.9 s |
| vertex100 q4 | `vertex100.q4.mmcf` | 512 | chat, `Write a haiku about rain.` | **487.0** (487.0-633.3) | 18.6-23.1 s |
| void1 q4 | `void1.q4.mmcf` | 1024 | `Once upon a time,` + `TOK_NO_PREFIX=1` | **554.8** (533.8-578.8) | 14.0 s |

Two short-sample rows: `vertex15`/`vertex35` stop at EOS after 7 tokens and
`vertex100` after 15, so a single token weighs 7-14% of their mean.  `void1 q4`
uses a 259-entry byte tokenizer, so 16 tokens are about 16 characters.

What moved against the 2026-11-02 table, in one place:

* **q8 is roughly 12-22% faster**: MiniMind2 q8 123.5 against 140-190, stories15M q8
  116.2 against 148, stories42M q8 160.2 against 190, TinyChat15M 115.4 against
  139-171.  The INT8-only work of 2026-11-03..05 (row-paired qkv, packed
  residual, wrow planes, the fused FFN default) is what that table predates.
* **q4 moved the same way**: MiniMind3 q4 304.4 against 381 (+20% faster),
  MoE q4 322.0 against 341.8 (+6%), MiniMind2 q4 153.7 against 193-212.
* **i16 went the other way**: 326.8 against 226/232, and it is now explained
  and fixed — see the decode section below.  †† marks the row measured with the
  `floor` decode; the default decode gives **216.0**.
* **Decode is the reproducible number, prefill is not.**  Identical MiniMind3 q4
  runs gave 16.0 and 35.9 s prefill (load 0.9 versus 2.4) while decode stayed
  inside ±1%.  The board's SD card and the background load move prefill; quote
  prefill only with its run.
* **TinyChat15M is fixed, and the old caveat was the wrong diagnosis.**  It is a
  llama-2-chat-format 15M model; the container had no chat template, so the
  engine refused `--chat` and the plain prompts used for the probe were out of
  distribution.  `templates/llama2_chat.jinja` renders `[INST] ... [/INST]`, and
  the store's q8/i16 containers were repacked with `--chat-template`; `--chat`
  now answers in dialogue form (` I am feeling quite disgusted with the current
  situation in the world.` to "Hello, who are you?").  The timing in the table
  was taken on the old plain container; the weights are unchanged, so it applies
  to the repacked one.  Answers are still the
  quality a 15M model gives; "coherent" here means grammatical and in-format,
  not correct.

Not re-measured in this campaign, and why: MiniMind3 q8 (the only HEAD run was
the 2026-11-12 pilot, 219.1 ms/token, taken while the store was being scanned;
not usable), LFM2.5-230M q4 (only a `--gen 1` load check on HEAD), TinyBrainBot
q4, picolm85 q4q24 and void1 q8r (load checks only), and MiniMind3-MoE q8 (the
203.5 MB container cannot fit the CMA; `FBO incomplete`, as recorded on
2026-11-11).  Their last numbers are the ones already in this file, with their
own dates.

## 2026-11-13 — the q16 decode: the cheap spelling is the default, the exact one is a switch

Follow-up to the campaign above.  Its i16 row was 41% slower than the
2026-11-02 number while every q8/q4 row got faster, so the i16 container was
bisected on the board — first with the leftover binaries, then with two
builds:

| binary | i16 decode (ms/token) |
|---|---:|
| 2026-11-09 morning, `625c591` (before `6234328`) | 215.2 / 217.7 / 221.7 |
| 2026-11-09 evening, `6234328` (integer decode) | 323.5 / 335.2 / 379.5 |
| HEAD floor decode (`71cb1e0`) | 322.9 / 323.5 / 327.8 |

`6234328` is the cause.  It rounds each q16 byte with `floor(c*255+0.5)` so
VC4 stops reading the -32768 rail as +32768, and `floor` costs ~5 QIR there
(FTOI truncates, so Mesa adds a compare-and-fix) against 2 for FTOI+ITOF, and
it breaks the add/mul pairing (`docs/vc4-glsl-costs.md`).  q8/q4 have paired
kernels that amortise the decode; i16 does not (`pair_ok()` is INT8-only), and
that is why only i16 paid visibly.

Two cheaper spellings were compared on the board, both classifying the sign at
`32767.5` (the encoder clamps to 32767, so nothing positive can enter
[32767.5, 32768)):

| decode | i16 | board q8 PPL | board q4 PPL | LFM2 serial warm | TinyBrainBot vs Xe (5 prompts) |
|---|---:|---:|---:|---:|---:|
| `floor` (`6234328`) | 323.5 | 271.6420 | 326.8088 | 28.6196 | 0.99878 |
| `float(int())` (`7258faa`) | 288.8 | bit-identical to `floor` by construction | | | |
| **default: two multiplies** (`7f976d1`) | **216.0** | 271.2855 | 325.3751 | 29.1588 | 0.99831 |
| 2026-11-09 broken decode (for scale) | — | — | — | — | 0.7233 |

The default row is the pre-`6234328` expression.  It keeps the **sign** right
(TinyBrainBot stays at 0.998, not 0.72) but reconstructs *every* value with
the 8-bit texture sample's small VC4 error, which is the part `6234328` also
fixed.  That is the deliberate trade: 33% back on the board's i16 model for
about 0.0005 cosine against Xe, +1.9% on LFM2's warm PPL and a moved board
q8 pin.  `Q16_DECODE=exact` restores the `6234328` spelling and the older
board values (`i16 289.7`) when exactness matters; the switch is documented in
[`test-hooks.md`](test-hooks.md).

Xe is unchanged in every row (q8 PPL 270.3170, q4 330.5302, both golden
strings, TinyBrainBot logits bit-identical): a float32 texture sample is exact
there, so only VC4 sees the difference.  `shader_hashes.txt` does not move for
the default, because its emitted GLSL is the same text `7f976d1` had.

The campaign table above was measured with the `floor` binary, so its i16 row
(326.8) is superseded by 216.0; that is the one row whose time moves with this
change (q8 spot checks are inside the noise: 185.7 with `floor` against 184.1
with the default).  Board pins after this change: q8 string
`我是一个AI助手，没有自我意识，也没有身体，所以我无法` (`Q16_DECODE=exact`
prints `我是MiniMind，一个由jingyaogong`), q8 PPL 271.2855, q4 PPL 325.3751,
LFM2 serial warm 29.1588 (with `Q16_DECODE=exact`: 271.6420 / 326.8088 /
28.6196).

## 2026-11-13 — documentation audit: `CONV_GPU` was never a switch

Every LFM2.5 command in this log that carries `CONV_GPU=1` passed a variable the
engine has never read.  `getenv("CONV_GPU")` appears in no source file, in no
commit of any branch (`git log --all -S CONV_GPU` touches Markdown only) and in
no built binary (`strings engine/engine_mm3 .bench/engine_mm3_pi.71cb1e0 |
grep -c CONV_GPU` → 0, 0).  The row was written into `docs/test-hooks.md` by
`0b50fb0`, the same commit that added the GPU conv passes — whose call site is
the unconditional `if (m_.is_conv(l))` — so it was documented and unread from
birth, not removed later.

Consequence for the entries above: **none of their numbers is wrong.**  What
selects the fused path is `GPU_LAYER`, and every fused/serial pair in this file
differs in exactly that (`serial` rows carry `GPU_LAYER=0`).  Read
`GPU_LAYER=1 CONV_GPU=1` as `GPU_LAYER=1`.  The switch row is gone from
`test-hooks.md` and `lfm2.5-230m.md` now explains it; the commands above stay as
they were run, per the append-only rule.

Two older sections in this file still call the top table "the current matrix
above" (the 2026-11-01 and 2026-11-02 reports, written when it was the newest
build).  That table is history; the pointer at the top of this file and the
2026-11-13 campaign are the authority.

## 2026-11-13 — can q4 skip the nibble unpack?  Xe cannot answer it

The question was whether a q4 weight can reach the ALU without a nibble
decode.  Nothing in this file changes the answer measured on VC4 (2026-11-04/
05): exactness needs a floor, floor is the cost, the split-plane format was
built and lost 12%.  What this entry adds is that **the host GPU is the wrong
instrument for the question, and the whole-token protocol cannot be used for
it** — measured, not assumed, while the board was down.

`probes/q4mv_probe` now builds natively (new `probes/q4mv_probe` target; only
the `_pi` recipe existed) and takes `Q4MV_ITERS` for the pipelined loop.  Its
default 20 iterations is 2 ms on Xe, where a 1216-row gate draw is 0.045 ms,
so the arm-to-arm noise swamps the difference the probe exists to price.
2000 iterations, min of six runs, `minimind3.q4` shapes:

| arm | 1216-row gate draw, ms | vs default |
|---|---:|---:|
| default (ESSL 3.00 shift decode) | 0.044-0.046 | |
| `Q4_NOUNPACK=1` (whole unpack removed, wrong weights) | 0.038-0.047 | ~-16% |
| `Q4_SHIFT=0` (float decode) | 0.047-0.060 | +4-30% |
| `Q4_BFAST=1` (byte floor dropped, wrong weights) | 0.054-0.154 | unusable |

Three things follow.

1. **The unpack is ~1% of a Xe token.**  The gate draw is about 5% of the
   15.26 ms/token decode (15.26 = prefill-free, 32 greedy tokens, `SEQ=768`),
   and removing *all* of the unpack moves that draw by 16%.  A 1% effect is
   below the token-level noise, so no amount of token runs will price this on
   Xe.  The same A/B on one binary in one session gave 15.1 / 23.3 / 44.6
   ms/token for the *default* arm: that spread is the protocol failing, not the
   machine.
2. **`Q4_SHIFT` is a VC4 win only.**  On Xe the float decode is not slower than
   the shift decode (both ~0.045 ms).  The shift decode's 32 ms of gate on the
   Pi was a VC4 measurement and does not generalise.
3. **Wrong-weight arms are not stable everywhere.**  `Q4_BFAST` and
   `Q4_NOUNPACK` feed garbage into `matvec_i4_pair_swiglu`'s swiglu, whose
   result goes through `exp`/divide, so the arm's timing depends on the
   nonlinearity's input range: 0.054 to 0.154 ms for the same kernel.  The
   default arm is stable to 2% in the same session.  The 2026-11-04 VC4 numbers
   for these arms were stable across three interleaved runs each, so they stand
   as measured — but the probe is only valid on a driver where it was validated,
   and an arm that changes the arithmetic needs its own stability check before
   its number is quoted.

**A design note for when the board returns.**  The reverted `Q4_SPLIT`
(9c91a68) repacked the nibbles on the CPU into hi/lo RGBA4 planes before
upload.  That is not necessary: the same bytes bound as `GL_RGBA4` at **twice
the width** are already 4 weights per texel.  The packer writes even columns in
the low nibble and odd columns in the high one (`mmcf_hf.py:77`,
`q[:, 0::2] | (q[:, 1::2] << 4)`), and `GL_UNSIGNED_SHORT_4_4_4_4` puts R in the
**most** significant nibble — measured on Xe, not read off the spec: texels
`(R=t, G=15-t, B=2t, A=t+7)` come back with shader `.r` = the bits 12-15 field.
So texel `2q` is `(w[c+3], w[c+2], w[c+1], w[c])`, i.e. the four weights in
**reverse** column order.  That costs one `.wzyx` on the sample (or the same
swizzle on `xa`) and nothing else, so the format still costs two fetches per
eight weights, zero ALU, zero repack, and **every existing q4 container works
unchanged** — only the tap values change, from `n` to `n/15`, so the pins move
and the containers do not.

That is strictly cheaper than what 9c91a68 built (which also paid eight mads
re-folding `c*15`), and it is the arm worth pricing: `Q4_PLANE` on the Pi, not
another decode variant on Xe.  Expected from the 2026-11-04 split: one extra
fetch ≈ +13%, removing the floors ≈ -34%, net ≈ -21% of q4 decode, which is why
it is still worth a measurement.

Two VC4 constraints the format has to clear, both from the driver source:

* **Not renderable.**  `vc4_formats.c:94` has
  `FORMAT(B4G4R4A4_UNORM, NO, RGBA4444, SWIZ(Y, Z, W, X))`: the TMU has an
  `RGBA4444` texture type, but the render-target column is `NO` (vc4 renders to
  RGBA8888 and RGB565 only).  A `GL_RGBA4` *renderbuffer* is therefore
  unsupported — that is `probes/progbinary_probe.cpp:135`, a different question
  from sampling one.
* **Tiled only.**  `vc4_resource.c:get_resource_texture_format()` returns `~0`
  (unsupported) for any non-RGBA8888 format on a **linear** resource; 4444 is
  only sampleable from a T-tiled buffer.  `glTexImage2D` textures are tiled by
  default, which is why the 2026-11-04 split worked, but any path that forces a
  linear texture drops this format silently.

The third piece of evidence is the board's own probe output: the 2026-11-04
entry above records `c*15` as one ULP **below** `v` for v = 9..14, and an
incomplete texture would have returned `-128`, not `-1`.  That evidence is not
airtight — `q4fmt_probe` checks no GL error, and it asks for `GL_RGBA4` as an
*internalformat*, which is ES3-only (the ES2 spelling is `GL_RGBA` +
`GL_UNSIGNED_SHORT_4_4_4_4`), so on an ES2 context the upload may have been
refused rather than accepted.  If the format turns out to be refused on the
board, this design note is dead and the answer to the question above closes as
"no path at all".

```
make -C engine -B probes/q4mv_probe_pi CROSS_SYSROOT=<sysroot>
Q4MV_ITERS=2000 ./q4mv_probe_pi                      # default arm
Q4MV_ITERS=2000 Q4_NOUNPACK=1 ./q4mv_probe_pi        # ceiling, wrong weights
```

## Replacing a model without rebuilding the GL context (2026-11-14, Xe)

`/model PATH` in the chat REPL and `{"op":"load"}` on the worker replace the
container while keeping the EGL context and its compiled programs.  On the Pi
Zero the parts that survive are priced in "Where the first token goes" above:
4.23 s of EGL context and 1.89 s of program links per process, against 6.12 s to
read a 69 MB container and 2.99 s to upload it.  **Those board numbers are the
reason the command exists; they are not measurements of the command.**  What was
actually measured is Xe, greedy, `SEQ=512 GPU_LAYER=1 KV_PACK=16`, one discarded
warm-up request before each swap:

| quantity | Xe |
|---|---:|
| a fresh process, to its first answer | 561 / 605 / 634 ms |
| `/model` handler, minimind3.q8 -> minimind2-small.q8 | 37 / 47 / 64 ms |
| same, worker `{"op":"load"}` (minimind3.q8 -> minimind2-small.q8 -> minimind3.q4) | 29 / 53 ms |
| RSS across a 13-swap alternating chain | 112-180 MB, sawtooth (the peak is the container read) |

So on Xe a swap costs about a tenth of a start.  The board is the machine that
matters and it was down for this; a Pi number for the same command is still
missing, and so is `CmaFree` after a swap, which is what decides whether the
release is actually complete on VC4.

**Correctness, not just cost.**  A swap is only worth having if it is
invisible: every turn after a swap must be what a fresh process would print.
That is `engine/scripts/swap_matrix.py`, and it found three real defects before
it passed:

- the program cache was keyed by name only, so the second model ran the first
  model's attention kernels (text hash now validated on a cache hit);
- the scene teardown missed the lazily filled `target_for()` map and the
  second activation cache;
- `main` did more to a model than read it -- `split_ffn()` (two 1216-wide chunks
  against one 2432-wide pass), `apply_hidden_scale()` (metadata key 15),
  `apply_precision()`/`quantize_q4_weights()` -- and a swap repeated none of
  them, which moved logits by 1 ULP on the dense models (cosine 0.9998, enough
  to flip a minimind3.i16 answer) and by 5.97 at the max on the hybrid.

A less obvious one, and the reason the worker needed a fix of its own: the
rolling prefix cache belongs to the model that produced it.  After a `load` the
next request skipped prefill against a prefix the new model never computed and
answered with repeated syllables ("德德德德..."); the REPL already cleared its
`kv_tokens`, the worker did not.

With all of that, a quantisation change inside one process is bit-identical too
(q8 -> q4 -> i16), which is why the `/model` command no longer refuses one.

## 2026-11-14 — RGBA4 on VC4, measured: it works, and the 2026-11-04 fact was real

The format question was open because the evidence was indirect: a driver table
entry (`vc4_formats.c`), a probe that never checked `glGetError`, and an
`internalformat` spelling (`GL_RGBA4`) that is ES3-only while vc4 reports
**GLES 2.0**.  `engine/probes/rgba4444_probe.cpp` settles it on the board.  It is
standalone — EGL + GLES2 and nothing else, no `engine/src` object — so it builds
and runs while `gpu.cpp` is being split.

Pi Zero, `VC4 V3D 2.1`, `OpenGL ES 2.0 Mesa 26.2.2-1~bpo13+0~rpt1`, GLSL ES
1.0.16, `EXT_texture_integer` absent, `EXT_draw_buffers` **present**:

| test | result |
|---|---|
| B upload, ES2 spelling (`GL_RGBA` + `UNSIGNED_SHORT_4_4_4_4`) | accepted, no GL error, sampled `17n` at 16/16 texels |
| B upload, ES3 spelling (`internalformat GL_RGBA4`, what `q4fmt_probe` used) | also accepted — Mesa is lenient here, so the old probe was not silently black |
| C component order | `R` = bits 12-15, `G` = 8-11, `B` = 4-7, `A` = 0-3 — **identical on VC4 and Xe** |
| D exactness | `round(c*15) == n` at 16/16; the residual is **-1** (2^-20) for n = 9..14 and 0 elsewhere, exactly the 2026-11-04 claim |
| E the real shape: 1216 rows x 192 texels (466 944 B, one q4 matrix) | accepted; **144/144 sampled nibbles match the packed bytes** at rows 0, 1, 5, 608, 1214, 1215 |
| F `glRenderbufferStorage(GL_RGBA4)` | **accepted, FBO complete** |

Three conclusions, the first two of which correct what this file said hours
earlier:

1. **The 2026-11-13 "Not renderable" bullet is wrong.**  It was inferred from
   `vc4_formats.c`'s render-target column (`RT_NO` for `B4G4R4A4_UNORM`), and
   the board contradicts it: an `GL_RGBA4` renderbuffer is accepted and its FBO
   is complete.  Whatever Mesa does underneath, `progbinary_probe.cpp:135` is
   not the failure the inference predicted.  This is the case for measuring a
   capability instead of reading the driver table for it.
2. **The 2026-11-13 "Tiled only" warning is untested, not confirmed.**  It comes
   from `get_resource_texture_format()` returning `~0` for a non-RGBA8888
   *linear* resource.  `glTexImage2D` textures are tiled and the 1216-row
   matrix works, so the warning only binds a path that would pass
   `PIPE_BIND_LINEAR` — nothing in the engine does today.
3. **The design note survives.**  The same bytes bound as `GL_RGBA4` at twice
   the width are four weights per texel, in reverse column order (C above, on
   both machines), so a kernel needs one `.wzyx` and no repack.  What remains
   open is cost, not capability: two fetches per eight weights against the
   current eight floors per eight weights.

One cross-machine detail worth keeping: the residual in D is **0 on Xe and -1 on
VC4**.  A decode that multiplies by 15 and does not round is therefore exact on
the laptop and one ULP low on the board — the exact shape of a pin that moves
only when someone runs it on the target.

```
# On the board (it has gcc and the Mesa headers, so no cross build is needed):
g++ -O2 -std=c++17 -o rgba4444_probe rgba4444_probe.cpp -lEGL -lGLESv2
./rgba4444_probe
```

## 2026-11-14, later — the "renderable" row was a false positive: only pixels tell

An hour after the entry above, the same probe grew the two checks that separate
"the API accepted it" from "the hardware rendered to it", and the answer flips:

| on the Pi | result |
|---|---|
| `glRenderbufferStorage(GL_RGBA4)` | accepted, no GL error |
| `glCheckFramebufferStatus` | `GL_FRAMEBUFFER_COMPLETE` |
| `GL_RENDERBUFFER_INTERNAL_FORMAT` | `0x8056` = **GL_RGBA4** |
| clear to 0.1/0.5/0.9 and read it back | **26/128/230 = an 8-bit surface** |

On Xe the same probe reads 34/136/238 — 2/8/14 in fifteenths, a real 4-bit
surface.  On VC4 the clear is not a multiple of 1/15: Mesa's state tracker
substituted RGBA8 for the storage and **reported the requested format back**.
So `vc4_formats.c`'s `RT_NO` was right, the 2026-11-13 bullet *"Not renderable"*
was right, and conclusion 1 of the entry above — which called that bullet wrong
— is the wrong one.  The sampler results are unaffected: B/C/D/E still stand,
4444 sampling is exact at the real matrix shape, and the q4 design only ever
*samples* the texture.

Two things worth keeping:

* **A capability query is not a measurement.**  Acceptance, FBO completeness and
  the format query all said 4444; a 0.1/0.5/0.9 clear said 8-bit.  Same shape as
  the `CONV_GPU` lesson, one layer down: the layer that answers is the one that
  produces pixels, and `progbinary_probe.cpp:135`'s `GL_RGBA4` renderbuffer was
  never a 4444 surface either.
* It closes a design nobody had proposed yet: "unpack the nibbles once on the
  GPU into a 4444 texture" cannot work on VC4, because nothing can be rendered
  into one.  The unpack stays in the shader, or the format stays 8-bit.

## 2026-11-15 — the RGBA4 q4 gate is adopted (-5%), and the decode was never the cost

`Q4_FMT=rgba4` now binds the dense FFN gate's INT4 matrices as `GL_RGBA4` at
twice the width: four weights per texel, `n/15` straight out of the sampler,
15 folded into the group scale, no decode at all.  The bytes are the
container's, so nothing is re-packed.

**Where the q4 token actually goes** (`MATVEC_DUP`, Pi, 16 greedy tokens,
`minimind3.q4`, base 286 ms):

| pass | marginal cost | share |
|---|---:|---:|
| FFN gate (w1+w3+swiglu) | 124-163 | ~50% |
| w2 (both halves) | 65 | 23% |
| attention (all 7 passes) | 34-44 | 14% |
| fixed passes (embed, rms x3, logits, read-back) | 44-45 | 15% |
| wo | 15 | 5% |

The three matvecs are ~78% of the token.  Against q8 the same probes give
ffn 47-64 and w2 23-44, so the q4 gate is ~2.3x the q8 gate -- and `Q4_NOUNPACK`
says only ~32 ms of the whole token is the floor chain.  **The unpack was never
the cost; the gate kernel's shape was.**

**What the format is worth, measured on the board** (load 0.12, three
interleaved pairs, `Q4_FMT=0` restoring the packed view in the same binary):

| pair | rgba4 | packed |
|---|---:|---:|
| 1 | 270.6 | 286.2 |
| 2 | 268.4 | 288.0 |
| 3 | 277.3 | 288.3 |
| 4 | 276.5 | 286.4 |

~-4 to -6% of the token, and -8% on the gate's own marginal cost (134.5 ->
123.9).  Small, but it is outside the pair-to-pair spread (<1%), so it is a
real 5%.

Two things it is *not*:

* **Not the sampler.**  `probes/rgba4444_probe` section G times 192 fragments x
  96 texels: 8-bit 0.172 ms/draw against 4-bit 0.172 ms/draw on VC4 (+0%).  The
  4-bit format costs nothing per texel, so the remaining 2x against q8 is the
  kernel (two taps per eight weights, four P-plane fetches per group), not the
  texture.
* **Not bit-exact.**  `15*(n/15)` is not exactly `n`; the logits move by about
  1 ULP (Xe, `LOGITS_OUT` compared: first difference at byte 7).  The pinned
  strings did not move -- Xe q4 and the Pi q4 both still print
  `我是MiniMind，一个高效的小参数AI模型，专注于` -- but PPL does.

**Adoption.**  Default on for the dense FFN gate; `Q4_FMT=0` restores the
packed view for an A/B in one binary.  Two refusals, both measured: the hybrid
(LFM2.5 q4 dies on an incomplete FBO with the gate bound as texels, cause not
yet known) and the unchunked FFN path (`ffn_split` reads packed nibbles).  Both
keep the packed view and say so if the switch was asked for.  qkv, wo and w2
keep the packed view -- extending the format there is where the next 5% is, and
it is a separate kernel each.

## 2026-11-15 — the hybrid's "FBO incomplete" is an out-of-memory, and it was always one

The 4-bit gate refused to run on LFM2.5 q4 with a bare `FBO incomplete`.  With
the diagnostic added to `GPU::make_target` / `make_texture` (`Q4_FMT=force`
reproduces it) the first failure is upstream of any FBO:

    texture upload failed: RGBA8 256x1536 (0x505), 336 textures / 150732800 B
    texture upload: CmaFree 13400 kB of 262144 kB, MemFree 37496 kB

0x505 is `GL_OUT_OF_MEMORY`.  The hybrid's weights are **143.7 MiB of
textures** before the first token, on a 256 MiB CMA with 13 MiB left; the next
256x1536 RGBA8 texture needs 1.5 MiB and does not fit.  The old message was a
second-order symptom: `make_texture` never checked `glGetError`, so the failed
upload left a black texture, and the FBO built on it was incomplete.

Two consequences:

* **The 4-bit view does not cause this.**  A packed q4 matrix is `cols/2` bytes
  a row and the 4-bit view is `(cols/4) * 2` -- the same bytes.  The hybrid sits
  at the edge either way; the switch only decides which allocation dies first.
  The guard therefore keeps the hybrid on the packed view (which fits), and
  making the 4-bit view fit is a memory-diet problem, not a format problem.
* **Any `make_texture` failure used to be silent.**  Every upload now checks
  `glGetError` and reports its size, the texture count and the byte total, with
  `CmaFree`/`MemFree`; a failed FBO prints its size, the GL error, the driver's
  texture/renderbuffer/viewport limits and the same budget.

### Where the hybrid's 143.7 MiB goes

`lfm2.5-230m` q4, `dim 1024 hidden 2560 layers 14 vocab 64402`:

| | MiB |
|---|---:|
| q4 weights, 14 layers (qkv, wo, w1, w3, w2 or the conv triples) | ~88 |
| the q4 group planes (scale+min, 4 B per 32 weights) | ~19 |
| logits, q8, 64402 x 1024 | **63** |
| activations, KV, scratch targets | ~10 |

The logits matrix alone is 44% of the texture budget and 67M MACs per token --
it is the largest single object in the model and it is *sampled once per
token*.  The embedding table is not on the GPU at all (`Session::embedding`
looks the row up on the CPU), so there is nothing to reclaim there.

Headroom is the whole problem: 143.7 MiB of weights against a 256 MiB CMA that
also holds the session's FBOs and the KV cache.  The 4-bit view needs a few
MiB of slack to be safe there, and the cheapest place to find it is the group
planes: they carry a 16-bit scale and a 16-bit min per 32 weights, and halving
that frees ~9 MiB across the 14 layers.

### The logits pass on the hybrid: 155 ms of a 1000 ms token

`PASS_DUP=logits` / `logits_draw` on LFM2.5 q4 (Pi, 4 greedy tokens):

| | ms/token |
|---|---:|
| baseline | 999.6 |
| `PASS_DUP=logits_draw` (the draw only) | +97.1 |
| `PASS_DUP=logits` (draw + finish + read-back) | +154.5 |

So 155 ms, 15% of the token, split 97 draw / 57 read-back.  The draw is the
*fastest* part of this model per MAC (67M MACs in 97 ms = 690 MMAC/s, against
the q4 gate's 250), so the kernel is not the problem -- the matrix is simply
63 MiB (44% of the texture budget) and it is read in full once per token.

Two things follow, and only one of them is free:

* **The read-back is avoidable.**  `Session::step` pulls all 64k logits to the
  CPU every token (`logits_from` -> one `glReadPixels` of the whole grid plus a
  `glFinish`) and samples there, while `logits_candidates_from` already computes
  the top-2 on the GPU with `x_top2_fast_*`.  A greedy / temp-0 path that asks
  the GPU for the winning index would return a few bytes instead of 258 KiB and
  save most of those 57 ms.  It has to reproduce the greedy text exactly, so it
  is a byte-identity change, not a numeric one.
* **The footprint is not, without changing the model.**  Halving it means q4
  logits (a packer decision with a quality cost, and the docs' "logits stay q8"
  is deliberate); the other routes -- low-rank logits, a coarse-then-refine
  top-k -- are model transforms, not engine ones.

### The q4 gate's residual 1.52x is not in any single stage

Kernel level, Pi, 1216 rows, 1000-2000 pipelined draws (`q4mv_probe`):

| arm | ms/draw | reading |
|---|---:|---|
| q8 | 4.435 | the reference |
| 4-bit | 8.181 | |
| one matrix (`Q4_4X4_ONEW`) | 6.311 | -1.87 |
| one row (`Q4_4X4_ONEROW`) | 5.711 | -2.47 |
| no min term (`Q4_NOMIN`) | 6.738 | -1.44 |
| no group plane (`Q4_4X4_NOP`) | 6.846 | -1.34 |
| no output (`Q4_4X4_NOOUT`) | 0.798 | **not a price** |

The last row is the trap: discarding the fragment value lets the driver
eliminate the entire kernel (the output target is untouched, `nz 0`), so it
prices nothing.  It is kept in the tree as the control that makes that visible.

The four real arms remove half the matrices (-23%), half the rows (-30%), the
min term (-18%) and the group plane (-16%), and none of them is additive: two
halves of the work cost 6.3 and 5.7 against 8.2, so roughly 4-5 ms a draw is
fixed cost under the per-matrix-row work.  Removing every piece of q4 metadata
(min term *and* group plane) lands at ~6.7 against q8's 4.4, so **1.5x survives
the complete removal of everything that distinguishes a q4 kernel from a q8
one** -- same weight-texel count, same four-lane dots, fewer activation
fetches.  That is below the resolution of the arms available here, and the
honest state is that the q4/q8 gap is not explained.

### 2026-11-15, later — the 4-bit view on all four matvecs

`qkv` and `wo` join the gate and w2 (`matvec_i4_4x4`, `matvec_i4_add_4x4`), so
every q4 matrix in the dense model is bound as 4-bit texels with no decode.

Board, load 0.16, minimind3 q4, 16 greedy tokens, three interleaved pairs:

    all four 4-bit   252.9 / 252.9
    packed           286.1 / 305.3      -14.5%

against the packed kernel, and -5% on top of the gate+w2 step (266.2).  The q4
token is 252.9 against q8's ~190, i.e. **1.33x**, down from 1.45x.

**PPL moved out of tolerance**: q4 PPL 330.5302 -> 326.1134 (-1.34%).  Every
tap is now n/15 with the 15 folded into the group scale, and the four-term dot
sums in a different order, so a sub-percent move was expected -- 1.34% is past
the 1% the gate enforces, so `Q4_PPL_EXPECT` has to be re-baselined to 326.1134
on Xe (and the board's row re-measured) in the same commit that adopts it.  The
pinned strings did not move on either machine, and the suite is ACCEPT.

### The `dot()` builtin was costing 14% of the 4-bit gate

The q8 kernel had already recorded the answer in its own comments ("the generic
X4Q macro takes its two texel fetches as arguments and VC4 compiles that loop
measurably slower; dA/dB are exactly the decode and the four accumulators keep
the same summation order"): VC4 lowers `dot(a,b)` as four multiplies plus a
horizontal add tree per texel, while `w.x*a0+w.y*a1+w.z*a2+w.w*a3` accumulates
straight into the destination with no reduction.  Every 4-bit kernel here was
written with `dot`, so each eight-weight texel pair paid one reduction.

Rewriting the seven `dot(wt*,x*)` sites as eight-term scalar expressions:

    gate, 1216 rows   8.181 -> 7.011 ms/draw   (-14%, output hash unchanged)
    token             252.9 -> 247.2            (-5.7 ms, load 0.22)

The probe hash is the same (776b29b8), so the arithmetic is bit-identical: this
is a lowering change, not a numeric one, and PPL confirms it (326.1134 ->
326.1483, the 0.01% of a re-association at the last bit).

q4 is now 247.2 against q8's ~190: **1.30x**, from 1.45x at the start of this
goal.  What is left at kernel level is 7.01 against 4.42, and with the P plane
and the min term both removed it was ~6.7 before this change -- so roughly 2.5
ms a draw is still unattributed.

The general lesson, which the repo had already paid for once on the q8 side and
I did not apply: **do not write `dot` in a VC4 kernel.**  The scalar form with
the accumulator as the destination is the fast one.

### 2026-11-15 — the hybrid's OOM is a CMA race on this board, not the format

The earlier "143.7 MiB of textures against 13 MiB of CmaFree" reading was taken
while the board's other software held most of the CMA.  Measured across the
session with nothing of ours running:

    CmaTotal   262144 kB
    CmaFree   128156 kB   (after a reboot, before any engine run)
    CmaFree    67596 kB   (minutes later, server stack busy)
    CmaFree    35464 kB   (busy again)

`pumpkin` + `gate` + `frps` hold 130-230 MB of a 256 MB CMA depending on load,
so the engine's own 143.7 MiB for lfm2.5-230M q4 fits or does not fit depending
on what else is resident *at that moment*.  In the same session, with CmaFree at
67 MB, **both** the packed and the `Q4_FMT=force` 4-bit hybrid completed; with
CmaFree at 35 MB the 4-bit run came back empty again.  The 4-bit view is
size-neutral by construction (`cols/2` bytes a row either way), so it is neither
the cause nor the cure.

What that means for the guard: it stays.  The hybrid is refused the 4-bit view
not because the format is wrong there but because the board cannot reliably hold
the model plus everything else, and a run that half-loads is the worst outcome.
Verifying the hybrid on the 4-bit view needs a board with the server stack
stopped; until then the honest statement is "not verified", not "does not work".

### The ratio, measured in one session

Same binary (`engine_mm3_pi.mad`), same session, interleaved, 16 greedy tokens:

| pair | q8 | q4 (all four matvecs 4-bit) |
|---|---:|---:|
| 1 | 190.4 | 258.4 |
| 2 | 186.8 | 248.5 |
| 3 | 206.4 | 264.6 |

**1.30-1.36x** (best pair 186.8 / 248.5; the clean-board runs give 190 / 247.2
= 1.30x).  The spread is the board's load average, which was 5.97 during this
run against 0.2 for the clean ones -- every row here is inflated and the q4 row
more than the q8 row, so quote the clean numbers and treat this as the
confirmation that the ratio is ~1.3x, not as the number itself.

For the record, the goal this session set out from was 1.45x (276 against 190),
and the path was: nibble reversal at upload (268.5), w2 on the 4-bit view
(266), qkv/wo (252.9), scalar accumulate instead of `dot()` (247.2).

### 2026-11-15 — the 4-bit view on the hybrid, verified on Xe

The hybrid was refused the 4-bit weight view on the strength of a board failure
that turned out to be CMA starvation.  Xe has no such limit, so the question
could be asked directly.  With the refusal lifted, lfm2.5-230M q4 answers
**identically** to the packed view on both prompts tried:

    "用一句话介绍你自己。"           -> 我是一个聊天机器人，专注...
    "What is the capital of France?" -> The capital of France is Paris.

`swap_matrix.py` still passes with the hybrid in the chain (every swap turn
byte-identical to a fresh process) and `regression.sh` is ACCEPT with q4 PPL
326.1483 unchanged.  So the 4-bit view is the default for every q4 matrix, the
hybrid included; what a board with little CMA free does is now a legible
failure (`make_texture` reports the size, the texture budget and `CmaFree`)
instead of the old silent `FBO incomplete`.

### 2026-11-15 — what a 4444 texture actually costs, and why the view is opt-in

Two measurements, both wrong, and then one that holds.

`CmaFree` around a `glTexImage2D` is not a measurement on this driver.  The same
q4 texture shape read 2020, 2436, 4576, 3344 and 4544 kB in five consecutive
uploads, and an RGBA8 upload once *raised* CmaFree by 716 kB: CMA pages are
movable, so every allocation moves page cache in and out.  The "17.6 MB for an
8 MB surface" and "0.5 MB for the same surface" numbers in earlier drafts of
this section came from that noise and are withdrawn.

The driver's own ledger is `/sys/kernel/debug/dri/0/bo_stats`.  Measured there,
in the engine's own context, before any model was loaded (a temporary
`Q4_TEX_ALLOC` switch, since removed, that allocated eight 1024x1024 textures
and held still while `bo_stats` was read):

| upload | V3D BOs | per texture | vs paper |
|---|---:|---:|---:|
| `GL_RGBA` + `UNSIGNED_BYTE` | 32776 kB | 4.0 MB | 1.0x |
| `GL_RGBA` + `UNSIGNED_SHORT_4_4_4_4` | 36872 kB | 4.0 MB | 2.0x |
| `GL_RGBA4` (sized) + `UNSIGNED_SHORT_4_4_4_4` | 36872 kB | 4.0 MB | 2.0x |

So the driver stores four bytes a texel whichever spelling is used: the nibbles
are expanded to RGBA8 at upload (`v*17`, which samples exactly like n/15, which
is why the view looked correct and why the expansion went unnoticed).  vc4 does
list `B4G4R4A4_UNORM` in its sampler table (`vc4_formats.c`), and the sized
`GL_RGBA4` is not even legal in an ES2 context, where the internal and client
formats must match -- neither changes the outcome.

Consequences, both now in the code:

- **The 4-bit view costs twice the packed view in VRAM.**  Measured on the whole
  model: minimind3 q4 is 81948 kB of V3D BOs with the view and 50892 kB without,
  i.e. +31 MB on a ~24 MB q4 body.  It is a speed knob (-14% on a dense q4
  token: 247 against 276 ms), not a memory saving, and it is **opt-in**
  (`Q4_FMT=rgba4`) as of this commit.
- **LFM2.5 q4 never fit because of this.**  With the view on, its 61 MB of q4
  weights become 122 MB, which is why the hybrid stopped loading in 2026-11-15
  while the packed view still did.  That is the regression the board reported,
  and it was this, not the CMA.

### 2026-11-15 — the 4-bit surface exists, just not through glTexImage2D

`st_ChooseTextureFormat` adds `PIPE_BIND_RENDER_TARGET` for `GL_RGBA4` (a GL
texture may become a render target and Mesa cannot know), vc4 cannot render to
B4G4R4A4 (`vc4_formats.c`: `rt = RT_NO`), so `find_supported_format` skips both
two-byte candidates and takes `DEFAULT_RGBA_FORMATS`.  The retry that would drop
the render-target binding runs only when nothing matches, and RGBA8 always
matches, so no spelling and no mutability can reach it -- `glTexImage2D` with
`GL_RGBA4` or with a 4444 client type, and `glTexStorage2D` with `GL_RGBA4`,
all measured four bytes a texel against `bo_stats` (with that same, since
removed, allocation switch).

`probes/egl_image_probe` goes around that function: it allocates the buffer
itself and imports it.

    A GL   OES_EGL_image=yes    A EGL  image_base / dma_buf_import / dma_buf_modifiers = yes
    B drm  64x16 at 2 bytes a texel = 2048 B, dma-buf fd 7
    C import  ARGB4444 accepted
    C sample  texel0 = 1 1 1 1   texel1 = 2 2 2 2     (exact nibbles)

So a true 4-bit surface is reachable: `DRM_IOCTL_VC4_CREATE_BO` +
`VC4_MMAP_BO` + `PRIME_HANDLE_TO_FD` + `EGL_LINUX_DMA_BUF_EXT` (linear) +
`glEGLImageTargetTexture2DOES`.  Doing that in `make_q4_texture` would make the
4-bit view memory-neutral instead of 2x, and the weights would be written
straight into the mapped BO instead of uploaded through `glTexImage2D`.

### 2026-11-15 — the dma-buf import works, but not at the widths the engine needs

`probes/egl_image_probe` allocates the buffer itself and imports it as an
EGLImage, which does get around `st_ChooseTextureFormat` (the surface is two
bytes a texel, and the nibbles come back exactly).  The component order differs
from the upload path -- a word whose nibbles are n0..n3 samples as
(n2, n1, n0, n3) -- and `(v & 0xF0F0) | ((v & 0x000F) << 8) | ((v & 0x0F00) >> 8)`
maps it back to column order, verified at 64x16.

It does not survive a wider texture.  Same probe, same pattern, one process,
linear import, pitch = width * 2:

| shape | sampled |
|---|---|
| 64 x 16 | `0 1 2 3` (correct) |
| 80 x 16 | `0 0 0 0` |
| 96 x 16 | `0 0 0 0` |
| 120 x 16 | `0 0 0 0` |
| 128 x 16 | `0 0 0 0` |
| 160 x 16 | `0 0 0 0` |
| 192 x 16 | `0 0 0 0` |

A larger BO (16 KB of slack) changes nothing, and neither does `msync` before
the unmap, so it is not the size or the write-back.  The engine needs 192, 256
and 384 texels wide (`cols / 4` of 768, 1024, 1536), so the linear import is not
usable as it stands.  The next thing to try is the tiled modifier
(`DRM_FORMAT_MOD_BROADCOM_VC4_T_TILED`) with the utile swizzle done by hand,
rather than the linear one.

The engine-side path was written, measured (memory-neutral: 50508 kB of V3D BOs
against packed's 50888, where the glTexImage2D view costs 81948) and then
withdrawn, because it answers wrongly at the engine's widths.  What is left in
the tree is this probe and this section.

### 2026-11-15 — the 4-bit view is memory-neutral, and the hybrid fits

The view used to cost twice the packed layout because `glTexImage2D` cannot
produce a two-byte surface on this driver (previous section).  It does not need
to: the surface is allocated here and imported as a T-tiled dma-buf EGLImage
(`engine/src/gpu_dmabuf.cpp`), and the weights are written with a plain
`glTexSubImage2D` so the *driver* does the tiling.

Three things were needed and each cost a round:

- the modifier is queried (`eglQueryDmaBufModifiersEXT`); this board answers
  `0x0700000000000001`, not the value a reader would compute from the vendor id;
- the BO carries it (`DRM_IOCTL_VC4_SET_TILING`), or the import is `EGL_BAD_MATCH`;
- the plane pitch is present even for a tiled image -- `dri2_check_dma_buf_format`
  requires `DMABufPlanePitches` on every plane.

The surface also reports the word's nibbles reversed, the same order the upload
path sees as `.wzyx`, so the import is handed the `rev` buffer the upload path
already builds; `Q4_IMPORT_CHECK=1` is what showed it (word `0x727a` with nibbles
10 7 2 7 samples as 7 2 7 10).

Board, minimind3 q4, both arms alive at the same point, `bo_stats` + `TIMING`:

| view | V3D BOs | decode (16 tokens) |
|---|---:|---:|
| 4-bit, dma-buf import | **51524 kB** (133 BOs) | **224.8 ms** |
| packed | 50892 kB (191 BOs) | 292.8 ms |

+1.2% memory (it was +61% through `glTexImage2D`) and -23% time.  The golden
strings are unchanged on the board for q8 and q4, and the import path against the
packed one is **cosine 0.999553**, max abs 0.39, mean 0.075 on logits spanning
-11.4 to 14.5 -- i.e. the `15*(n/15)` reassociation the view always had, not a
layout error (a layout error reads 0.7-0.9).

**LFM2.5-230M q4 now loads on the board**, which it had not done since the view
became the default: 61 MB of q4 weights stay 61 MB instead of becoming 122.  Its
greedy string with the view is one token different from the packed one
(`我是一个AI的一个智能助手` against `我是一个一个简单而富有趣`), which is the
same reassociation amplified by a hybrid's conv path -- quote the packed string
for the hybrid, and the dense pins for the view.

### 2026-11-15 — outside vc4 the sized format is the one to ask for

The fallback upload asked for an *unsized* `GL_RGBA` with a 4444 client type,
which was right for vc4 and wrong everywhere else.  The difference is
renderability: `st_ChooseTextureFormat` adds `PIPE_BIND_RENDER_TARGET` because a
texture may become a render target, and then

- on vc4, `B4G4R4A4` is `RT_NO`, so `find_supported_format` skips it and lands on
  RGBA8 -- four bytes a texel, which is why the dma-buf import exists;
- on a driver where RGBA4 *is* renderable, the same table entry is chosen and a
  sized `GL_RGBA4` is genuinely two bytes a texel, so the unsized spelling was
  throwing half the memory away.

`probes/rgba4444_probe` section F on the Iris Xe says it plainly:

    F renderable  RGBA4 renderbuffer: accepted; FBO complete (0x8cd5); storage 0x8056 = GL_RGBA4
    F precision   clear 0.1/0.5/0.9 reads 34/136/238 -> 4-bit surface (a multiple of 1/15)

`make_q4_texture` now asks for the sized format whenever the context is ES 3.0
(the only place it is legal; ES2 requires `internalformat == format`), which is
exactly when the driver reports ESSL 3 -- the same test the q4 shift path uses.
The output is unchanged on both machines (the pins hold: the sampled values are
n/15 either way), so this is a memory change only, and it cannot be measured in
process RSS because the texture lives on the driver's side.

### 2026-11-15 — the conv block joins the 4-bit view

`apply_q4_fmt` marked qkv/wo/w13 and left the hybrid's conv matrices packed, so
the block that is 16% of a hybrid token kept the nibble decode.  The three
`in_proj` slices (`conv_in_b/c/x`) and `conv_out` are ordinary INT4 Mats and both
their kernels -- `matvec_to` and `matvec_add` -- already had a 4x4 variant, so
marking them was the whole change.

Xe, LAYER_PROFILE on lfm2.5 q4, same binary either way (`Q4_FMT=0` turns the view
off), 6 tokens:

| view | conv, ms a call | conv total |
|---|---:|---:|
| 4-bit | **1.768** | 325.3 ms |
| packed | 2.036 | 374.6 ms |

-13.2% on the stage, and the stage is ~16% of a token, so ~2% of the token.

Board, the whole hybrid, interleaved, load 2.5:

| | decode |
|---|---:|
| 4-bit (conv included) | **699 / 703 ms** a token |
| packed | 911 / 919 ms |

-23% for the hybrid as a whole.  The pins did not move: minimind3 q8 still prints
`我是一个AI助手，没有自我意识...`, q4 `我是MiniMind，一个高效的小参数AI模型...`,
and the hybrid keeps its 4-bit string `我是一个AI的一个智能助手` against packed's
`我是一个一个简单而富有趣` -- the conv change did not move it.

## 2026-11-15: q4 perplexity is not bit-reproducible (RGBA4444)

The `gpu.cpp` split (P2, same day) moved Xe's q4 PPL 330.5302 -> **326.1483**
(1.3%) while everything that exists to catch a real regression stayed put:

- q8 PPL **270.3170** to the last digit, before and after;
- the golden strings unchanged (`我是MiniMind，一个高效的小参数AI模型，专注于` for
  q4 on Xe);
- `scripts/swap_matrix.py` PASS, i.e. every q4 turn is byte-identical to a fresh
  process on the same container;
- a `LOGITS_OUT` diff of the same q4 prompt between the two builds is
  `max|d| = 0.0000` over all 6400 vocabulary rows.

So the model maths did not move; the q4 *decode/reconstruction* path did.  The q4
weight and KV textures are 4 bits per channel (RGBA4444), and reconstructing a
value from a 4-bit channel is not bit-exact: a reorganisation of the code is
enough to flip the last bits, and a corpus-level number is sensitive to that in a
way a greedy string is not.  The x86 driver's RGBA4 conversion rounds; VC4's
differs.

Consequence for the gate: q4's PPL pin is a *format* pin, not a numerics pin.
Re-baselined (Xe 326.1483) and expected to move by ~1% whenever the q4 path is
rearranged.  The q8 PPL, the string pins and the swap matrix are the checks that
have actually caught regressions, and none of them is affected.
## 2026-11-15, later — the GPU greedy argmax: byte-identical, and a net loss on VC4

The logits-pass entry above said the hybrid's 155 ms logits pass splits
97 draw / 57 read-back, and that the read-back was avoidable: ask the GPU
for the winning index instead of pulling 258 KiB to the CPU.  This entry
records building that, and the measurement that killed the premise.

**What was built.**  `Session::step_greedy` runs the forward pass and takes
the argmax where the logits are: the proven `top2_fast_packed` kernel keeps
the best (value, index) per 16-logit chunk, `top1_group` reduces 16 chunk
winners to one per group, `top1_final` reduces the groups to one winner per
tile, and the host reads `tiles * 4` bytes (168 B for the hybrid's 42 tiles)
and keeps the first maximum across tiles.  Strict `>` at every level is
`std::max_element`'s tie rule and the values travel as the stored q16
integers, never re-rounded, so the winner is **byte-identical**, not close.
The chat loops, the tool loop and the worker all take it; the plain path's
decode timer wraps it.  Eligibility: greedy turns without penalties, marker
masks, `LOGITS_OUT` or logprobs; the unpacked logits layout (i16 weights,
ACT8) keeps the read-back.  `GPU_ARGMAX=1` enables it (opt-in, see below);
`ARGMAX_DUMP=1` prints the chunk winners, the tile winners and the CPU
argmax of the same logits; `READBACK_TIME=1` prices the finish and the
transfer separately.  The three new scratch targets are in
`reset_targets()`; `swap_matrix.py` passes with the worker load op.

**The VC4 part, which cost five board rounds.**  Xe ran the first version
byte-identical; the board returned token 0 for every prompt.  ARGMAX_DUMP
localized it: the float maxima were all correct, but every index stuck at
its first-iteration value.  What was tried, in order, all with the same
symptom on the board and all byte-identical on Xe:

| form | board |
|---|---|
| single-branch `if(v>best){best=v;bi=idx;}`, int index | index sticks |
| index carried as float | index sticks |
| two-branch best1/best2/i1/i2 (the top2 form) | index sticks |
| stage 1 = the literal `top2_fast_packed` text | **exact** |
| 96-iteration `top1_reduce` over the exact chunk winners | index sticks |
| 16-iteration group loop with a runtime-bound `break` | index sticks |
| 16-iteration group loop, no break, sentinel clear | **exact** |

The pattern: **a loop the compiler does not unroll loses the conditional
index updates; the float max survives.**  The 16-iteration literal-bound
top2 loop is unrolled and exact; a 96-iteration loop, or a 16-iteration loop
whose `break` bound is a runtime value, stays rolled and loses them.  The
shipped reduction therefore keeps every loop at the unrollable length and
has no runtime break: the pass clears the chunk target to the sentinel texel
(-32768, index 65535) before stage 1, stage 1 draws only each tile's real
chunks (the TOPK_FAST viewport), and the trailing sentinels can never win —
-32768 is the smallest q16 value, and a tie keeps the earlier, real chunk.
Recorded in `docs/vc4-glsl-costs.md` §6; only a board run catches this class.

**Identity (both machines).**  Greedy text and TOKENS_OUT byte-identical
with the arm on and off: Xe q8/q4/i16 and LFM2.5 q4 (two prompts), Pi Zero
q8/q4 and LFM2.5-QAD q4 (two prompts); LOGITS_OUT dumps identical (the dump
disqualifies the fast path by design); the plain completion path too.  Xe:
regression.sh ACCEPT (q8 PPL 270.3170, q4 326.1483, both string pins, MoE
determinism), swap_matrix.py PASS, warncheck 0, server selftest 108/108,
shader_hashes additive only.

**Timing (Pi Zero, protocol v1: discarded warm-up, interleaved timed pairs
20 s apart, GPU_LAYER=1 KV_PACK=16, greedy).**  The hybrid rows use
`IGNORE_EOS=1 --gen 16` (the model stops at 7 tokens on its own; a 5 s
decode window was too short for the board's load swings).  The first
campaign ran right after a two-engine CMA incident and is noise; the clean
campaign is the second:

| lfm pair | fast | slow | load (fast/slow) |
|---|---:|---:|---|
| 1 | 686.8 | 678.8 | 2.33 / 2.07 |
| 2 | 706.2 | 677.1 | 2.04 / 1.73 |
| 3 | 694.7 | 677.9 | 1.47 / 1.93 |
| 4 | 699.4 | 697.6 | 1.32 / 1.54 |
| 5 | 681.8 | 738.1 | 2.20 / 1.72 |
| median | **694.7** | **678.8** | |

Paired deltas +8.0 / +29.0 / +16.7 / +1.8 / -56.4, median **+8.0 ms/token —
the reduction is slower**.  Dense (first campaign, --gen 16 natural EOS):
q8 fast 189.6 / 188.3 / 190.7 vs slow 205.1 / 188.9 / 187.5 (medians 189.6
vs 188.9), q4 fast 229.0 / 230.9 / 228.8 vs slow 236.3 / 227.0 / 225.5
(medians 229.0 vs 227.0): neutral to +2 ms.  Xe plain-path interleaved pairs
agree in sign: 31.99/29.15, 27.47/22.04, 21.35/19.23 (+2.8/+5.4/+2.1 ms).
The board's first fast token also pays ~6.4 s of reduction-shader compile
and QPU generation (one-time; the warm-up absorbs it).

**Why — READBACK_TIME, steady-state hybrid token:**

| path | stage | ms |
|---|---|---:|
| read-back | finish (waits the forward + logits draws) | ~156 |
| read-back | **readback: 258 KiB + 64402-float decode** | **6.0-7.2** |
| reduction | logits + finish | ~172 |
| reduction | stage 1: clear + 42 top2 draws + finish | 12.3-15.3 |
| reduction | group draw + finish | 0.9-1.5 |
| reduction | final draw + finish | 0.8-1.3 |
| reduction | readback: 168 B | 0.16-0.28 |

**The read-back is ~6 ms, not 57.**  The PASS_DUP figure above was a replay
artifact: the replay's finish waits on the replayed logits draw, and that
wait is unavoidable GPU work, not transfer — the same trap the
`LAYER_PROFILE` entry documents.  In the real flow the transfer plus the
CPU decode of 64402 floats costs 6-7 ms.  The chunk pass re-reads the whole
logits texture on-GPU (roughly the same fetches) and adds three pipeline
drains and 44 draws: ~14 ms, which cannot beat 6.  The dense read-back is
~1 ms (the 2026-11-03 PASS_DUP entry was right about that one), so the
reduction loses there too.

**Disposition.**  The full read-back stays the default everywhere;
`GPU_ARGMAX=1` keeps the reduction as a byte-identical measurement arm — it
may pay where `glReadPixels` is dearer than a texture re-read plus three
drains, which VC4 is not.  The correction to the entry above: the 57 ms is
not an avoidable read-back, and "the read-back is avoidable" was a hypothesis
that the end-to-end A/B falsified.  What the investigation bought instead is
the rolled-loop rule in `docs/vc4-glsl-costs.md` §6, the ARGMAX_DUMP /
READBACK_TIME instruments, and a corrected cost model: the hybrid's logits
pass is 97 ms of draw plus ~6 ms of read-back, and both are near their floor
without changing the model.

Board acceptance on the final default binary: q8 PPL **271.2855** to the
digit (the pin), q4 PPL **328.0042** (unpinned; the 4-bit view moved it from
the packed view's 325.3751, +0.8% — the format-pin behaviour the RGBA4 entry
predicts, measured here for the record), the q8/q4 string pins, the poem
behaviour check, hybrid determinism (two runs byte-identical), and the
default byte-identical to the pre-argmax binary on q8/q4/hybrid.
## 2026-11-16 — the q4 gate's unattributed 1.5x was the nested loop; the flat gate is byte-identical and -43%

The 2026-11-15 entry left ~2.5 ms/draw of the q4/q8 gate gap unattributed:
with the P plane and the min term both removed, the 4-bit gate still ran
~6.7 ms against q8's 4.4 at 1216 rows.  Static counting said the gap could
not be arithmetic — per 32 columns the metadata-free q4 gate issues the
same 48 fetches and 128 MACs as q8, but **zero** weight-decode FMADs (q8
pays 128 `c*255-128`) and cheaper coordinates.  Less work, yet slower: the
cost had to be the loop shape.

**The decomposition (`q4mv_probe`, Pi Zero, 1216 rows, 2000 pipelined
draws, ms/draw; new arms `Q4_4X4_FLAT*`).**

| arm | ms/draw | what it removes |
|---|---:|---|
| q8 reference | 4.45 | — |
| 4x4 nested, full metadata (default) | 6.55 | — |
| nested, `NOP`+`NOMIN` (no metadata) | 5.85 | P fetch + scale/min |
| `FLAT` — q8's single 4-wide loop, no metadata | 4.27 | the nesting |
| `FLAT8` — single 8-wide loop, no metadata | 2.90 | nesting + per-group reset |
| `FLAT8M` — flat 8-wide, metadata per iteration (96x) | 4.70 | — |
| `FLAT8G` — flat 8-wide, metadata per group via conditional (24x) | 3.70 | — |
| **`FLAT8B` — flat 8-wide, byte-identical per-group arithmetic** | **3.74** | — |

The nested group loop itself costs ~2.95 ms/draw (5.85 nested-no-metadata
vs 2.90 flat-no-metadata).  VC4 does not unroll the inner 4-iteration loop
inside the outer group loop, and the rolled nested structure schedules
badly; a single flat loop of 96 eight-weight iterations pipelines the
weight fetches properly.  The 8-wide form beats the 4-wide (2.90 vs 4.27):
fewer iterations, eight fetches sharing one coordinate pair.

**Byte identity.**  `FLAT8B` keeps the nested kernel's exact operation
order in the flat loop: `d1/d2/e1/e2/xsum` reset at the group start
(`q%4==0`), accumulate per iteration in the same tap order, flush
`s10+=sa1*d1+ma1*xsum` at the group end (`q%4==3`).  The conditional
updates are all **float**, which the VC4 rolled-loop rule preserves
(`docs/vc4-glsl-costs.md` §6 — it is conditional *int index* updates that
are lost).  Probe hash matches the nested kernel at every size on both
machines: VC4 304 `adee9498`, 608 `63e355a2`, 1216 `776b29b8`, 2048
`428b8e1f`; Xe 1216 `056eeba9`.  Engine greedy text + TOKENS_OUT identical
with `Q4_GATE_NESTED=1` on Xe (q4/q8) and the board (q4 and LFM2.5-QAD).

**Row sweep (FLAT8B vs nested, ms/draw):** 304 2.31→1.59, 608 3.89→2.42,
1216 6.55→3.74, 2048 10.14→5.72 — -31% to -44%, growing with rows.

**Token A/B (Pi Zero, protocol v1: discarded warm-up, three interleaved
pairs 20 s apart, `GPU_LAYER=1 KV_PACK=16`, greedy, `--gen 16`; flat
default vs `Q4_GATE_NESTED=1`).**

| model | flat (median) | nested (median) | delta |
|---|---:|---:|---|
| MiniMind3 q4 | **185.7** (185.5/187.9/185.7) | 226.7 (227.8/226.7/226.2) | **-41 ms, -18%** |
| LFM2.5-QAD q4 (`IGNORE_EOS`) | **566.3** (566.3/562.9/567.4) | 675.2 (680.8/675.2/672.5) | **-109 ms, -16%** |

q4 load 0.6-1.2, hybrid load 1.4-2.8.  **MiniMind3 q4 at 185.7 ms/token is
now faster than q8 (~190)** — the q4/q8 ratio flips from 1.18x to ~0.98x.
The gate is ~50% of the q4 token (16 gate draws/token), so the -2.8 ms/draw
kernel win lands as ~41 ms/token, on the prediction.

**Acceptance.**  Xe: regression.sh ACCEPT (q8 PPL 270.3170, q4 326.1483 —
both pins to the digit, both string pins, MoE determinism), swap_matrix.py
PASS, warncheck 0.  Board: q8 PPL 271.2855 (pin), q4 PPL 328.0042
(unchanged from the pre-gate 4-bit value), q4/hybrid flat==nested.  The
shader dump now covers the 4x4 gate/qkv/w2 (it had only the packed gate);
`shader_hashes.txt` gains those four entries and nothing else moves.
`check_q16_decode.sh` learned the one unsigned reconstruction — the q4
group scale `65280.0*15.0*ps` builds a positive constant and carries no
negative rail — which the newly-dumped 4x4 kernels exposed.

**Disposition.**  The flat 8-wide gate (`FLAT8B`) is the default;
`Q4_GATE_NESTED=1` restores the nested form for the A/B, and the legacy
pricing arms (`NOP`/`ONEW`/`ONEROW`/`NOOUT`/`NOMIN`) route to the nested
body they are defined on.  The qkv (`matvec_i4_4x4`) and w2/wo
(`matvec_i4_add_4x4`) 4-bit kernels are still nested — the same flat-loop
rewrite applies to them (qkv ~14%, w2 ~23% of the token), the obvious next
step.

## 2026-11-16, later — the flat rewrite does NOT extend to qkv/w2: the single-row kernels stay nested

The gate entry above called the qkv/w2 flat rewrite "the obvious next
step".  It was built, verified byte-identical, and measured — and it is a
net loss.  This entry corrects that expectation.

**What was built.**  `i4_qkv_flat` and `i4_w2_flat`: the same flat 8-wide
loop with the nested kernel's exact per-group order (d/xsum reset at
`q%4==0`, flush `s+=sc*d+mn*xsum` at `q%4==3`, conditional float updates).
The qkv keeps the inline dA/dB decode; the w2/wo keeps its generic `X4Q`
decode and residual add.  Byte-identity held everywhere it was checked:
Xe q4 and LFM2.5 q4 greedy text + TOKENS identical with the flat forms on
and off, and the board agreed (q4 and LFM2.5-QAD).

**The clean three-arm A/B (Pi Zero, MiniMind3 q4, discarded warm-up, four
interleaved rounds 15 s apart, `--gen 16` greedy, load 0.6-0.9; round 1
dropped as cold).**  Arms: *nested* = all three `Q4_*_NESTED` switches (the
then-defaults), *gateonly* = gate flat + qkv/w2 nested, *allflat* = all
three flat.

| round | nested | gateonly | allflat |
|---|---:|---:|---:|
| 2 | 227.3 | 185.2 | 197.0 |
| 3 | 225.8 | 184.6 | 197.5 |
| 4 | 228.1 | 185.0 | 197.8 |
| median | **227.3** | **185.0** | **197.5** |

**The qkv/w2 flat form costs ~12.5 ms/token** (197.5 allflat vs 185.0
gateonly): it erodes a third of the gate's 41 ms win.  The hybrid agrees
in sign and magnitude: gate-only flat measured 566 ms/token above, the
all-flat binary 622-632 (its conv matrices go through the same qkv/w2
kernels), against 674-701 all-nested.

**Why the gate won and these lost.**  The gate is a *pair* kernel: two
rows per fragment, eight weight fetches and four accumulators per inner
iteration — enough independent work per iteration that the flat loop's
pipelining pays and the nested loop's rolled scheduling hurts.  qkv/w2
are *single-row* kernels: two weight fetches, one accumulator per inner
iteration.  There the nested loop schedules fine, and the flat form's
per-iteration group conditional (`q-g*4==0` / `==3`), resets and flushes
are pure overhead.  The lesson is not "flat beats nested" but "the gate's
nested loop was scheduling badly because of its wide body"; the rewrite
is shape-specific.

**Disposition.**  qkv (`matvec_i4_4x4`) and w2/wo (`matvec_i4_add_4x4`)
keep the nested form as the default.  The flat forms stay as opt-in
measurement arms (`Q4_QKV_FLAT=1`, `Q4_W2_FLAT=1`), byte-identical, the
record of this negative result — the same treatment as `Q4_MV4`.  The
shipped q4 default is therefore **gate flat + qkv/w2 nested**: MiniMind3
q4 **185.0 ms/token** (-18% against the 227.3 all-nested baseline, faster
than q8's ~190), LFM2.5-QAD q4 ~566 (-16%).  `Q4_GATE_NESTED=1` alone
restores the full nested baseline for the A/B.  Acceptance re-run on this
disposition: Xe regression ACCEPT (q8 PPL 270.3170, q4 326.1483, both
pins), swap_matrix PASS; the shader hashes move only the gate entries.

## 2026-11-16 — the REPL batch: answers and decode unchanged (Pi Zero)

Cross-built on vlab from `4c198b3`, ARMv6/VFPv2, 1,784,640 B, sha256
`0fd1e4e08fd46cf99be1eec92b86fd3b8b5046298941b887e524a629525d1817`, deployed as
`bin/engine_mm3_pi` over the 2026-11-13 build, which is kept as
`engine_mm3_pi.2026-11-16-pre-qol`.  The batch is REPL-only: multi-line editing
with the terminal's paste frame, the `/params` knobs, `/stats`, the upload bar,
and `/model` refusing a bad container from its metadata *before* releasing the
current model (no capacity pre-check any more).

Interleaved A/B on the idle board (`loadavg 0.46`, 1.0 GHz, `throttled=0x0`,
36.3 C), minimind3 q8, `GPU_LAYER=1 KV_PACK=16`, `--gen 16`, greedy:

| binary | decode (ms/token) |
|---|---:|
| `8044002` (previous, `engine_mm3_pi.2026-11-16-pre-qol`) | 185.1 / 211.4 |
| `4c198b3` (this deploy) | 187.7 / 185.7 |

The container's own seq (256) was used; `SEQ=768` gave 189.4.  That is the
q8 ~190 the q4 disposition above already compares against, so the batch is
timing-neutral as well as text-neutral: the q8 acceptance pin still prints
`我是一个AI助手，没有自我意识，也没有身体，所以我无法` and all 127 shader digests
are identical.

Do not read the 115.4 ms/token row of the AGENTS.md campaign table as this
model: it is TinyChat15M q8.  The only earlier HEAD number for minimind3 q8 was
the 2026-11-12 pilot's 219.1 ms/token, taken while the store was being scanned;
today is faster than it.

## 2026-11-16 — RWKV-7 P4: layer-loop verification and the R-1 residual grid (Iris Xe)

Machine: laptop, Intel Iris Xe (ADL GT2), Mesa 26.2.5, `GPU_LAYER=1 KV_PACK=16`.
Container: `models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf` (g1d 0.1B, q8
per-row scale, embedded .tok + template, key 8 bos 0 / eos 261 / stop 261,
key 17 logit_shift 2, seq 512).  The P2 serial loop, no new shader.

**Layer loop faithful.**  `models/rwkv7/phase2/engine_on_container_weights.py`
now decodes the per-row container's part-3 scale planes (the head is 43 tiles
*plus* 43 rows×1 f32 planes; the old tile filter concatenated the planes and
died).  It runs the f32 oracle on the container's own quantised weights, so the
difference against the engine is the activation grid alone:

| engine arm | vs oracle-on-container-weights cos (q8 per-matrix) | (q8r per-row) |
|---|---:|---:|
| default (q16 residual) | 0.99604774 | 0.99808681 |
| `ACT_F32=1` (f32 residual) | 0.99742538 | **0.99893934** |

Prompt: the P2 e2e prompt (system "You are a helpful assistant.", user "Hello,
my name is", 23 ids), top-1 agrees in all four arms.  q8r + `ACT_F32` is the
≈0.999 the plan's P4 gate asks for; what is left is the matvec's q16 input /
I24 output grid (the oracle's matvec is f32), not the loop.  Removing the
residual grid moves cos by +0.00085, exactly the R-1 direction.

**R-1 on the gate corpus.**  `engine/data/ppl_test.txt` is 421 World-vocab
tokens; with the `--ppl` BOS (key 8 id 0) the stream is 422 ≤ SEQ=512, so one
warm window scores every token with full left context (`stride=256` printed,
none needed).  The oracle side reproduces the same stream
(`models/rwkv7/phase4/ppl_grid.py`: [0] + vocab.encode(text); the raw vocab
file does not carry the BOS marker as one entry, so the id is prepended
directly).  All engine arms SEQ=512, same corpus, same window:

| arm | PPL | nll/token |
|---|---:|---:|
| oracle f32 (same stream) | 8.1014 | 2.09203 |
| oracle resid-q16 simulation (R-1) | 8.4133 | 2.12981 |
| engine q8r, q16 residual (default) | 9.2919 | 2.2291 |
| engine q8r, q24 residual (key 12) | 9.2633 | 2.2261 |
| engine q8r, `ACT_F32=1` | 8.8201 | 2.1770 |
| engine q8 per-matrix, q16 | 12.0831 | 2.4918 |

The q24 arm ran on the binary rebuilt for `RWKV_DUMP`; the q16 arm re-run on
that binary is bit-identical (9.2919), so the dump hook is numerics-neutral.

**Decision: q16 stays the RWKV default; q24 is not a quality lever here.**
q24 is the *same* 1/256 step with 256× the range, and g1d's residual peaks at
37.4 (packer rail check; 3.4× under the 128 rail), so the range is unused.  The
measured q24−q16 gap (−0.0030 nats, −0.3% PPL) is tie-rounding, not resolution:
the q24 branch rounds with `roundf` (half away from zero) and the q16 branch
with `lrintf` (half to even), and the only off-grid input to `add_q16` is the
exact-f32 embedding row at layer 0's first residual add — every later add is
grid+grid.  The grid's real cost is the *step*: the oracle simulation prices it
at +0.0378 nats/token on this corpus, and the engine's `ACT_F32` arm recovers
−0.0521 nats (−5.1% PPL) against the q16 default.  If P6 wants that back on the
GPU, the lever is a finer residual encoding (per-layer scale or an f32 rail),
not key 12.  Key 12 remains the escape hatch for a model that *exceeds* the
rail, exactly as on the transformer path.  The q24 variant is packed and kept
for the record: `rwkv7-g1d-0.1b.q8r.q24.chat.mmcf` (same pack flags plus
`--res q24`, 382 tensors, verify clean).

Row-scale weights are worth −0.263 nats/token on this corpus (12.0831 →
9.2919, −23% PPL): the q8r container is the right shipping default, confirming
P2's choice with a corpus number instead of one prompt's cosine.

**Xe acceptance values for the RWKV pins** (`regression.sh` reads
`RWKV_EXPECT` / `RWKV_PPL_EXPECT` from the machine override; the gate prints
them when unset).  Prompt "用一句话介绍你自己。", system "You are a helpful
assistant.", `--gen 16 --greedy`, the whole multi-line reply:

```sh
RWKV_EXPECT='>
Okay, so I need to figure out how to write a short sentence'
RWKV_PPL_EXPECT='9.2919'   # SEQ=512 (RWKV_PPL_SEQ), data/ppl_test.txt, q8r container
```

The reply is English "thinking out loud" — g1d's own style, the template ends
at `Assistant:` and the model opens `>\nOkay, ...`.  It is a tripwire like the
dense strings, not a quality claim; the quality numbers are in
[evaluation.md](evaluation.md)'s 2026-11-16 section.

**Speed (context for P5/P6, not a campaign).**  The PPL run's 421 steps took
75.7 s wall including one container load ≈ **170 ms/token** on Xe: the P2
serial loop pays ~180 matvec read-backs per token.  The identity run (25
prefill + 16 decode) was 10.3 s.  P6's fused loop is the fix; Pi numbers are
P5's item.


### Addendum (same day, later): the nothink template and the corrected pins

The chat template changed after the section above was measured:
`templates/rwkv7-g1.jinja` gained an `open_thinking` variable and the q8r
container was repacked with it (18:38).  The default (`open_thinking` false)
ends the prompt with an empty think block whose closing tag is left one
character short — `Assistant: \n</think` without the final `>` — and the
model's first generated token supplies it: every chat reply opens `>`, then
answers directly instead of thinking out loud (the open `Assistant: <think`
of the P2/P3 rows let it think for paragraphs).  That leading `>` is also why
the multiple-choice battery cannot score RWKV in chat mode — see the protocol
corrections below.  (The official fla-hub G1 template closes the tag fully,
`</think>`; this repo's nothink branch leaves the `>` to the model.  Noted
for P3's template owner; the numbers here are measured on the shipped file.)  P4 repacked the other two containers to
match (`rwkv7-g1d-0.1b.q8.chat.mmcf`, `rwkv7-g1d-0.1b.q8r.q24.chat.mmcf`,
both `--verify` clean).  The repack is **weight-neutral**: the container-weight
oracle on the *old* P2 prompt ids reproduces cos 0.99808681 bit-for-bit against
the repacked file, and `--ppl` (raw text, no template) is unchanged at
**9.2919**.

The identity pin moves with the template (same prompt/system/gen 16, greedy,
deterministic across two runs).  The reply still opens `>` — g1d emits it even
after the closed empty think block — and then answers in Chinese:

```sh
RWKV_EXPECT='>
"我是一个AI，我的名字是‘智'
RWKV_PPL_EXPECT='9.2919'   # unchanged; SEQ=512, data/ppl_test.txt, q8r container
```

**Layer-loop verification across positions** (engine ↔ oracle-on-container-
weights, q8r; cos is position-dependent — the plan's §四 warns the g1d scoring
positions are not flat, so one prompt is not the whole picture):

| prompt (chat, new template) | default cos | ACT_F32=1 cos | top-1 |
|---|---:|---:|---|
| P2 e2e "Hello, my name is" (old 23-id render) | 0.99808681 | 0.99893934 | yes |
| same user text, nothink render (27 ids) | 0.99762827 | 0.99783206 | yes |
| "The capital of France is" | 0.99787658 | **0.99900299** | yes |
| "人工智能是" | 0.99759716 | 0.99847656 | yes |

Same pattern in every position: removing the residual grid lifts cos, the
remainder is the matvec's q16 input / I24 output grid, top-1 never moves.  The
≈0.999 the P4 gate asks for is reached with `ACT_F32` (0.99900 on the France
prompt, 0.9978–0.9990 across the four); the layer loop is faithful.

**Battery protocol corrections (user-directed, same day).**  Two changes to how
the quality batteries score RWKV, both in `engine/scripts/`:

- `bench_mc.py --completion`: the chat template puts a think marker at the
  answer position, so the first decoded token is `>`, not a letter — the
  chat-mode letter logits score noise (fmt ≈ 0; the invalid 100-item run is
  kept at `models/rwkv7/phase4/eval_mc_chat_invalid/`).  In completion mode
  (no `--chat`) the same prompt's first token is the letter itself (fmt 1.000
  on the probe).  The transformer models stay in chat mode.
- RWKV is scored with its **model card's recommended sampling**, not greedy:
  temp 1.0 / top_p 0.3 / repetition_penalty 1.2 (fla-hub/rwkv7-0.1B-g1 card;
  generation_config.json carries only bos/eos).  One copy in
  `evalkit.RWKV_SAMPLING`; `eval_ifeval.py --preset auto` (the new default)
  and `eval_translate.py --preset auto` pick it per container via metadata
  key 16, and keep the historical presets for everything else.

The corrected numbers are in [evaluation.md](evaluation.md)'s 2026-11-16
section; the raw dumps in `models/rwkv7/phase4/eval_*`.


### Addendum 2 (same day, final): the template truncation fix and the final pins

The nothink branch's first edit dropped the closing tag's `>`; the second
edit restored it but dropped the opening tag; the third (final) file renders
**byte-identical to the official fla-hub G1 nothink prompt** — verified by hex:
`417373697374616e743a203c7468696e6b3e0a3c2f7468696e6b3e` =
`Assistant: \n</think>` (the repo writes the newline as the Jinja
escape `\n`, the official file as a raw 0a; the render is the same).  The
switch stays `open_thinking` (the official `enable_thinking` is undefined in
the engine's Jinja context, so copying the official condition verbatim would
always take the open-thinking branch).  All three containers were repacked
from the final file (`--verify` clean): q8 197 tensors / q8r 382 / q8r.q24
382.  **Weight-neutral again**: the old P2 prompt's container-weight oracle
cos reproduces 0.99808681 bit-for-bit, and `--ppl` is still **9.2919**.

With the complete empty think block the model no longer opens `>`: the first
generated token is a newline, then the answer.  Final Xe identity pin (same
prompt/system, `--gen 16 --greedy`, deterministic across two runs, and the
container-default system renders identically):

```sh
RWKV_EXPECT='
"我是一个AI，我的名字是‘智能'
RWKV_PPL_EXPECT='9.2919'   # SEQ=512 (RWKV_PPL_SEQ), data/ppl_test.txt, q8r
```

(The pin string starts with a newline — the reply's first token.  The gate's
`rwkv_text` keeps it; `check_prefix` is a prefix match.)

The ifeval/translate rounds run before this fix (the truncated-template round
and the earlier greedy partial) are archived under
`models/rwkv7/phase4/eval_ifeval_truncated_tmpl/` and
`eval_ifeval_greedy_partial/` with READMEs; they are not the model's numbers.
The MC completion battery is template-independent (no `--chat`) and its
100-item run is kept.  The final ifeval (rwkv preset, 2 seeds — the 2026-11-12
table's protocol) and translate (rwkv preset) numbers are in
[evaluation.md](evaluation.md)'s 2026-11-16 section.


### Addendum 3 (same day, acceptance): the full gate on the final containers

`scripts/regression.sh` **ACCEPT** on the final (official-nothink) q8r
container, with the machine's pins exported: MiniMind3 q8/q4 golden strings
unchanged (`我是一个AI助手，我的存在是为了提供帮助和信息。` /
`我是MiniMind，一个高效的小参数AI模型，专注于`), poem behaviour OK,
q8 PPL **270.3170** (pin), q4 PPL **326.1483** (the re-baselined pin of the
2026-11-15 gpu.cpp split, documented above — the 330.5302 still standing in
AGENTS.local.md is the 2026-11-13 value and needs the override update),
ACT8=2 283.1356 / ACT8=1 273.6691 printed.  The new RWKV block: determinism
OK, default-system OK, prefix pin OK, behaviour OK, PPL **9.2919** (pin,
SEQ=512).  34 containers round-trip through mmcf_format (the three RWKV v4
files included).

`scripts/swap_matrix.py` **PASS**, including the new rwkv chain
(minimind3.q8 → rwkv7 q8r → rwkv7 q8 → rwkv7 q8r → minimind3.q8): every turn
byte-identical to a fresh process — the WKV state and both token shifts are
built on the way in, the per-row scale planes appear/disappear with the
family change, and nothing RWKV survives the way back out.  All five existing
chains, the slash-line check, the twice-run determinism and the worker load
op are unchanged and green.  `make -C engine check`: 108 passed / 0 failed.

## 2026-11-16 — RWKV-7 P6: the GPU fused layer loop (Xe, Iris Xe ADL GT2)

`RWKV_GPU=1` selects RWKV-7's third layer loop on textures (the elementwise
ops + the WKV7 recurrence); the default stays the CPU serial `forward_rwkv`.
The WKV state is **bit-cast f32** in RGBA8 ping-pong textures (native ESSL 3.00,
`floatBitsToUint`/`uintBitsToFloat` — the Q4_SHIFT ES3 upgrade generalized to
also trigger on `RWKV_GPU` and on the bit-cast builtins), so the recurrence is
exact f32 like the CPU reference; the norm statistics are f32 bit-cast too.  The
elementwise intermediates stay I24_1; only matvec I/O and the residual keep the
quantized grid.  Without ESSL 3.00 (or with `RWKV_WKV_CPU=1`) the recurrence
falls back to the host f32 state.

**Acceptance is PPL, not cross-path cosine/greedy** (AGENTS.md measurement
rules: golden strings are chaotic across GPUs/reassociations; a CPU-vs-GPU
stage delta is a lead, not a defect).  Container
`models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf`, `data/ppl_test.txt`,
SEQ=512 (stride 256, 421 tokens, 1 window):

| path | PPL | nll |
|---|---:|---:|
| CPU (default, the pin) | **9.2919** | 2.2291 |
| `RWKV_GPU=1` (GPU pin) | **9.1850** | 2.2176 |
| `RWKV_GPU=1 RWKV_WKV_CPU=1` | 9.1781 | 2.2168 |

The GPU PPL is the same magnitude as CPU and slightly lower; **9.1850 is the
GPU-path pin, separate from the CPU 9.2919 — the two are not comparable.**
The prefill logits cosine vs CPU is ~0.9988 and the greedy text diverges from
the CPU pin (CPU `\n"我是一个AI，我的名字是‘智能` vs GPU
`\n**用一句话介绍你自己**\n**1.` at --gen 16): this is the chaotic temp-0
sensitivity of a 0.1B model, **not** a quality defect — the PPL裁决 above is the
acceptance.  The GPU greedy string may be pinned as the GPU path's own tripwire;
it differing from the CPU string is chaos, not a bug.

**Decode speed** (bench protocol: discarded warm-up + 3 timed runs, median;
canonical chat prompt, --gen 48, loadavg ~1.2 on 16 cores):

| path | min ms/token | median ms/token |
|---|---:|---:|
| CPU | 114.5 | 119.1 |
| `RWKV_GPU=1` | 35.5 | 36.0 |

**3.3x** median speedup.  (A short --gen 32 sweep under different load read
1.8x; the steady-state --gen 48 number is the one to cite.)

**Determinism**: two `RWKV_GPU=1` runs of the same binary are byte-identical
(TOKENS_OUT and reply text).  Behaviour: non-empty, no collapse.

**Gate**: `make -C engine check` 108/0; `scripts/regression.sh` ACCEPT with
the CPU pins unchanged (q8/q4 strings, q8 PPL 270.3170, q4 PPL 326.1483, RWKV
CPU PPL 9.2919).  `shader_hashes.txt` re-baselined additions-only (the new
rwkv_* builders; no existing builder text moved).  Pi verification pending
(lead to arrange; the bit-cast kernels need the VC4 Mesa 300 es path, else the
CPU f32 fallback).

### Addendum (same day): VC4 texture-width portability fix

Pi Zero (VC4, GL_MAX_TEXTURE_SIZE=2048) hit `texture upload failed: RGBA8
4608x1 (0x501=GL_INVALID_VALUE)` on the GPU path: `init_rwkv` uploaded the
per-layer constant vectors as **1D** q24 textures, and `RC_LERP`=6*dim=4608
(and `RC_SCALE`=3*dim=2304) exceed 2048.  Xe's 16384 limit hid it; the CPU
path never uploads these vectors (reads f32 directly), so it never triggered.
Fix: the multi-row constants are now **2D dim x rows** textures (lerp 768x6,
scale 768x3, the norm affines 768x2), so the width is always dim=768<=2048;
the sampling shaders (`rwkv_lerp`'s L, `rwkv_l2_*`/`rwkv_keydelta`/`rwkv_rk_reduce`'s
SC, the norm affines' W) read texel (i, row) instead of a flat offset.  bias
stays split into dim-wide w0/a0/v0 rows; cm_lerp is one dim-wide row.  A
`dim > max_target_dim_` guard now fails loudly instead of a silent 0x501.
Value-identical on Xe (GPU PPL still 9.1850, greedy tripwire and determinism
unchanged); `shader_hashes` re-baseline touches only the rwkv_* builders.
Pi runtime re-test pending the lead's cross-deploy.


### Addendum 2 (2026-11-17): the VC4 numerical fix (Pi Zero, VC4 Mesa 26.2.2)

After the width fix the Pi GPU path ran but collapsed to garbage
("HealthchoicedefaultGrounddefault..."), and the hybrid (CPU f32 WKV state)
collapsed too — so the culprit was the elementwise chain, not the state.
Board-level probes (engine/probes/rwkv_probe[_pi], sections A–J: 2D NPOT
constant fetch, I24_1 render/fetch round trip, bit-cast round trip, q16 read,
matvec/matvec_add, render→fetch glFinish A/B) all PASS in isolation; the
per-op GPU-vs-CPU dump named the first divergence at layer 1, then the
sqrelu of the 1536-wide chunked cm hidden: its single 1536-fragment draw
exceeds VC4's kDrawCap=1024 and was silently dropped (target left at the
clear color, I24 -32768).  Two fixes, both gated to needs_finish() (VC4) so
Xe is byte-identical:

1. rwkv_draw1 splits into kDrawCap-wide strips on VC4 (the sqrelu, and any
   elementwise draw wider than the cap).  Xe drives the wide draw in one piece.
2. the wo/cm residual add uses the CPU's two-step order on VC4 — matvec to I24
   (rwkv_mv) then a q16 add (rwkv_res_add, new builder) — because the fused
   matvec_add drops its dot on VC4's I24_1 chain.  Xe keeps the fused
   matvec_add (its own rounding).

Result: Pi GPU greedy is now byte-identical to Xe GPU (the canonical --gen 16
reply, a leading newline then the echo prefix) — the garbage is gone.  Xe
unchanged: GPU PPL 9.1850, greedy tripwire, determinism, make check 108/0,
regression ACCEPT (CPU pins 270.3170/326.1483/9.2919).

Pi speed (measured, bench TIMING, --gen 16): GPU decode ~1138 ms/token vs
CPU ~969 ms/token — on VC4 the GPU path is draw-bound (~1879 draws/token at
~0.6 ms/draw) and slightly slower than the CPU path; the >=3x win is
Xe-specific (Xe ~36 vs ~119 ms/token, 3.3x, low per-draw overhead).  A Pi
speedup would need the elementwise ops fused into the matvec kernels (fewer
draws), which is future work.

Pi GPU PPL (SEQ=512, ppl_test.txt, 421 tokens, 1 window): **9.4464** (nll
2.2456).  Higher than Xe GPU's 9.1850 — the VC4 two-step residual add and the
strip-split sqrelu round differently from Xe's fused path, and VC4's q8 matvec
decode differs from Xe's.  The greedy reply is byte-identical to Xe GPU and
coherent (no collapse); PPL is the same magnitude as the CPU path.  Pi CPU PPL
re-measure in progress for the cross-path comparison.

Pi CPU PPL (same corpus/SEQ): **9.2990** (nll 2.2299) — essentially the Xe CPU
9.2919 (PPL is stable across GPUs even though greedy strings are chaotic).  Pi
GPU 9.4464 vs Pi CPU 9.2990: same magnitude, the VC4 two-step residual add and
strip-split sqrelu round slightly differently from the fused Xe path.  The GPU
path is numerically合格 on Pi (coherent greedy, PPL within ~0.15 of CPU); the
acceptance is PPL, not cross-path greedy byte-match (AGENTS.md).

## 2026-11-17 — RWKV-7 P7: launch granularity on both paths (Iris Xe ADL GT2)

P6 left a puzzle: the fused GPU path spends 1879 draws a token on the board and
36.0 ms/token on the laptop, while the serial path spends ~200 *host syncs* a
token and 119.1 ms.  P7 asked which of those two meters a token actually pays
for — on each path — and then moved the one that pays.

### The meters

`SYNCCOUNT=1` is new: every `glFinish` plus every `glReadPixels` on the
read-back path (`docs/test-hooks.md`).  `DRAWCOUNT=1` existed but printed a
process total and **only on the `--prompt` completion path**, so the chat runs
the timings come from never printed it.  Both counters are now decode-scoped:
the turn records its decode loop's delta, so the prompt's launches cannot move a
per-token figure.

### Before / after (bench protocol: discarded warm-up, interleaved A/B, one
binary per arm, 6 rounds; canonical chat prompt, `--gen 48`, `SEQ` from the
container)

| arm | ms/token median (min) | draws/token | syncs/token |
|---|---:|---:|---:|
| CPU path, before | 108.6 (105.4) | 233 | 167 |
| CPU path, **after A2/A3** | **70.0 (61.3)** | 233 | **73** |
| `RWKV_GPU=1`, before | 34.8 (33.7) | 1158 | 3 |
| `RWKV_GPU=1`, after | 35.8 (34.2) | **594** | 3 |

PPL unmoved on both pins: CPU **9.2919** (bit-identical, `SEQ=512`,
`data/ppl_test.txt`), GPU **9.1850**.

### A2/A3 on the serial path: sharing the read-back (delivered, 1.55x)

The serial layer loop crosses the bus once per matvec — 14 times a layer, and on
Xe a read-back costs ~0.6 ms.  The ops are independent, so P7 draws them into the
rows of one target and takes the block with a single `glReadPixels`
(`GPU::matvec_rows`, `Session::rwkv_mv_group`):

- **A2** r/k/v: three read-backs → one.
- **A3** the four low-rank branches: eight read-backs → two (four downs, then
  four ups).  tanh and sigmoid stay host-side and in the same order; only the
  batching is new.

Every matrix keeps its own draw, program, uniforms and 2^k input normalisation,
and the decode is `matvec_from`'s element by element, so the vectors are
**bit-identical** — the CPU PPL is the same number to four decimals, not "within
noise".  Syncs 167 → 73 (the floor for this structure: rkv + downs + ups + wo +
cm-key + cm-value per layer, plus the logits).  The fused path is still the
faster one on Xe (35.8 against 70.0); what changed is that the serial path's
cost is now the matvec work rather than the stalls.

What it does *not* change: the upload count.  The same 14 inputs still cross the
bus the other way, and on VC4 `upload_activations` finishes after each one, so
the board's serial path saves the read-backs only (94 of the 167).

### The WKV update's strips: 576 of the GPU path's draws were a VC4 rule

The recurrence's update pass is one fragment per state element (heads·hs·hs =
49152 here), so it is split at `kDrawCap` — *unconditionally*, which is why Xe
paid 48 draws a layer for a cap it does not have.  Gated on `needs_finish()`
like `rwkv_draw1`, the shader derives both indices from `gl_FragCoord`, so the
fragments are identical: **1158 → 594 draws/token, and the decode does not move
(31.1 against 31.5 ms/token over 4 interleaved rounds)**.

That is the measurement the rest of this section turns on: **on Xe a draw costs
nothing.**  Deleting 564 draws changed the token by less than the noise.

### A1 (six token-shift lerps in one draw): declined, with the numbers

One `768 × 6` draw is 4608 fragments.  `rwkv_draw1` splits at `kDrawCap`
*fragments*, so on VC4 that draw is 5 strips against the 6 draws it replaces —
**one draw a layer, 12 a token**, i.e. ~6 ms of the board's 1138 by P6's own
0.6 ms/draw division.  On Xe it would be 6 → 1 and worth the ~0 ms a draw costs
there.  The price is the whole banded-input plumbing: a `768 × 6` lerp target
plus a matvec variant that reads input *row* `b` for output band `b`, which is
the same machinery A2 would need on the GPU path — and that one pays nothing at
all (next paragraph).  Declined as a net-negative on the only machine it can be
verified on; if the Pi's draw cost is confirmed to be real, the same two pieces
would buy back ~6 ms and a *much* larger share of the WKV question below.

### A2 on the GPU path: no draw to save

`draw()` splits an m-row matvec at `kDrawCap` **rows**, so a merged
`3·dim`-row rkv draw is 2304 rows = 3 strips = exactly the three draws it would
replace.  Merging rows cannot reduce a draw count on either machine; only the
*consumer* side would gain (one FBO bind instead of three), and that is not a
draw.  The serial path does not care either: it buys the shared read-back, which
it now has.

### A3 on the GPU path: the ups cannot be merged for free

The four down projections (rank 64/64/32/128 = 288 rows) *could* be one draw —
288 rows is under the cap — for 3 draws a layer, 36 a token, ~18 ms on the board
at P6's 0.5 ms.  The four ups cannot: they are `dim × r` matrices with four
different inputs and four different ranks, so a single draw needs either a
zero-padded `128 × 3072` weight texture per layer (+1.8x the MACs, ~2 MB more
resident, a new upload path) or four weight samplers plus four row-scale
samplers (9 texture units).  At 1.8x the MACs on a fetch-bound kernel, and
against the board's measured 283 MMAC/s, that is a loss the 3 draws do not
cover.  Not attempted; the down merge is the coherent half and is left for
whoever measures the board first (see below).

### What the board should measure next

The Pi's GPU path is the one place where a draw really is expensive (P6: ~0.6 ms
of the 1138, from 1879 draws).  **576 of those 1879 — 31% — are the WKV update's
cap-sized strips**, so the question that decides the board's GPU path is what
`kDrawCap` actually limits: *fragments per draw* (as `rwkv_wkv` and
`rwkv_draw1` assume) or *width* (as `draw()`'s comment says: "a draw wider than
`kDrawCap`", with the 2048-wide scratch target as the reason).  If it is width,
the update is one `64 × 768` draw on VC4 and the board's GPU path loses a third
of its launches without touching a single number.  `probes/rwkv_probe` section F
(render then fetch, glFinish A/B) is the vehicle; it needs the board, not this
laptop.

Until that is answered, the serial path is the board's best bet and P7's own
change is path-independent: 94 fewer read-backs a token, and the same
`glFinish`-per-upload behaviour it already had.


## 2026-11-17, later — RWKV-7 P7: the channel mix, priced; two negative results (Iris Xe + Pi Zero)

P7 left the fused path's channel mix as one number: 42.5 ms a layer on the Pi
Zero, 43% of a 98 ms layer, against a 4.72M-MAC estimate of 16.7 ms — a 2.5x
gap.  This section prices the inside of that stage, then measures the two
levers the gap suggested on the board.  **Both are negative**: the region is
matvec throughput, not launches, and the two switches are kept as default-off
measurement arms rather than shipped defaults.

### The instrument: `RWKV_PROFILE`'s second fused-path line

`RWKV_PROFILE=1` now prints one more line per layer, breaking `cm` into the
passes the region consists of (docs/test-hooks.md): `pre` = `att_norm_2` + the
channel-mix lerp + the token-shift copy, `key_*`/`sq_*` = the key matvec and
its relu^2 per chunk of `split_rwkv`'s cut, `val_*`/`add_*` = the value matvec
and the q16 residual add per chunk.  The laps are the same `glFinish` deltas the
stage laps use, so the same caveat applies — the totals are inflated against a
real token and only the shares are to be read — and the sub-lap total is folded
back into the layer line's `cm`, so that field still means the whole region.

Xe, q8r 0.1B chat container, SEQ=512, `--gen 12`, layer 6 (steady state, two
tokens):

| cm line | total | pre | key | sq | val | add |
|---|---:|---:|---:|---:|---:|---:|
| token A (unchunked) | 4.05 | 1.28 | 0.75 | 0.51 | 1.52 | 0.00 |
| token B (unchunked) | 4.43 | 0.90 | 1.35 | 0.75 | 1.42 | 0.00 |

(`add` is 0 on the fused path because `matvec_add` carries the residual add
inside `val`; the two-step form is the board's.)  Two things already showed on
the laptop: `pre` is 20–30% of the region — the `att_norm_2` statistics pass
fetches all 768 elements per fragment — and the key matvec (2.36M MAC, 3
strips) is *not* slower than the value matvec (2.36M MAC, one draw), which is
what a MAC-bound region looks like and the first hint that the board's 16 draws
here were not the cost.
### The calibrated probe: `RWKV_SHIFT_PP=1`

Every layer draws `rwkv_copy` twice to store the normalised row into the
layer's shift texture.  The shift is a per-layer *state*, but the two roles do
not need one texture: `rwkv_cur_` already flips once a token for the WKV
state's ping-pong, so the norm can render this token's row into slot `par` and
the lerp read slot `par ^ 1`, which holds exactly the bytes the copy left
there.  `RWKV_SHIFT_PP=1` does that; with it off every index is slot 0, i.e.
the shipped draw sequence.

It is bit-identical, and that is the point — it is the one change on this path
whose only difference is launches.  Measured on Xe: PPL 9.1850 default and
9.1850 with it, 9.1781/9.1781 on the hybrid (`RWKV_WKV_CPU=1`), and
9.3712/9.3712 and 9.1726/9.1726 on the chunked arms; `TOKENS_OUT` identical
over 64 tokens across two turns.

### Where the draws are, and what they are worth

`DRAWCOUNT=1` on Xe (a count, not a timing, so the arithmetic transfers; only
`rwkv_draw1`'s strip split is driver-dependent):

| arm | draws/token | delta |
|---|---:|---:|
| default (copy) | 594 | — |
| `RWKV_SHIFT_PP=1` | 570 | **-24** = the two copies a layer, exactly |
| chunked (`RWKV_CHUNK_TEST=1`) | 630 | +36 = chunking's three extra passes a layer |
| chunked + `RWKV_CM_SPLIT=2048` | 618 | **-12** on Xe, -24 on VC4 (below) |
| chunked + `RWKV_CM_SPLIT=2048` + `RWKV_SHIFT_PP=1` | 594 | the default's count, on the chunked path |

`split_rwkv` cut the 3072-row channel-mix key at the even 1536 because one
weight texture may not exceed `GL_MAX_TEXTURE_SIZE` (2048 on VC4).  The cut is
now `RWKV_CM_SPLIT=<rows>`, and where it falls decides how many `kDrawCap`-sized
strips the key matvec and its relu^2 take: 1536+1536 is 2+2 strips of each,
2048+1024 is 2+1 — and `init_rwkv` now sizes each chunk's relu^2 scratch to that
chunk too, since one max width had made the narrow half's `rwkv_sqrelu` as
expensive as the wide one's.  Xe measures -12 a token, not -24, because it does
not split an elementwise draw at all (`rwkv_draw1` strips only under
`needs_finish()`); on the board the relu^2 half of the saving is real, and the
profile's strip counts moved 4 -> 3 exactly as predicted.

The cut also moves the numbers, by construction: the q16 residual add runs once
per chunk, so the cut point changes where the channel mix is rounded.  Xe with
`RWKV_CHUNK_TEST=1`, SEQ=512: 1536 -> 9.3712 (bit-identical to unset),
2048 -> 9.1726, 1024 -> 9.2531, 2040 -> 9.4001.  Adopting a cut would
re-baseline the board's GPU pin.
### The board's verdict: no gain from either switch (Pi Zero, VC4)

The instrument and both switches were cross-built and run on the board by the
lead (this laptop cannot).  Decode, interleaved, q8r 0.1B chat:

| arm | ms/token | what it removed |
|---|---:|---|
| default | 993.6 | — |
| `RWKV_CM_SPLIT=2048` | 1015.3 | 2 cm strips a layer (4 -> 3 per chunked pass) |
| `RWKV_SHIFT_PP=1`, round 1 | 990.3 -> 970.6 | 24 draws a token |
| `RWKV_SHIFT_PP=1`, round 2 | 965.1 -> 976.4 | the same 24 draws |

Neither moves the token: the cut is 21.7 ms the *wrong* way on one pair, and the
ping-pong's two rounds have opposite signs (-19.7 then +11.3).  Both are inside
the board's own spread for one binary (965–1015 across these runs, and P6's
1138 was measured under a different prompt and load), so the honest reading is
**no effect**, and the useful part is that the draw arithmetic was right:
removing 24 of ~1879 draws a token, with every byte of arithmetic identical,
changes nothing measurable — which is the same conclusion P7 reached on Xe from
the other direction (deleting 564 draws cost nothing).  **A draw on this board
is not what a token pays for.**

### Why: the fetch ratio, not slack

Then what does the 42.5 ms/layer buy?  cm's 4.72M MACs at 42.5 ms is ~111
MMAC/s, and the board's whole-token rate lands in the same band (~127–145
MMAC/s by the lead's board estimate over the layer's MACs), against the ~283
MMAC/s the dense paths see.  The two kernels' inner loops explain it: per four
multiply-adds, `matvec_q8`'s I24_1 single-row form issues **1 weight fetch + 4
input fetches**, its I16_2 inline form **1 + 2**, and the pair form feeds *two*
output rows from one input decode — 1.25 fetches per MAC against 0.5, a factor
of 2.5, which is the gap to within the noise of the numbers it is computed
from.  The channel mix is not paying for its draws or its elementwise passes;
it is paying for `I24_1` activations in a single-row kernel.

That also bounds what is left inside the stage, and none of it is 25%:

- **The relu^2 folded into the value matvec's input sampling** would remove the
  measured `sq_a + sq_b` — 4 of the region's 16 draws and two 1536-element
  write/read passes — at the price of the relu^2 paid once per input fetch
  (2.36M squarings a layer instead of 3072).  It needs a shader builder
  (`shaders.cpp`), and it is a few percent at best.
- **Merging the two q16 residual adds** is declined, not deferred: summing the
  two I24 deltas first, or the fused `matvec_add`, is one rounding instead of
  two, i.e. a different grid, and the fused form is exactly what P6 had to give
  up on VC4.
- **A pair kernel or packed activations for this path** is the only thing that
  addresses the 2.5x, and both are `shaders.cpp`: the `rwkv_*` elementwise
  builders hard-code `I24_1`, so neither can be reached from the driver.

Conclusion for the board: the fused RWKV path's cost is matvec MAC throughput,
not launches.  Until the kernels change shape, the Pi's GPU path stays where P6
left it (GPU ~1.1 s/token against the CPU's ~0.97 s), and the >=3x win stays
Xe-only.  The two switches stay in the tree as default-off measurement arms —
`RWKV_SHIFT_PP` is the cheapest known "remove 24 draws, change nothing" probe
on this path.

## 2026-11-17 — RWKV-7 P1: chunked prefill (batched projections, serial recurrence)

The serial prefill walks one token through all L layers, so every projection is
drawn and read back once per token: 14 matvecs a layer, ~170 host syncs a token.
Nothing in a projection depends on the previous token, so the block can be
evaluated one *layer* pass at a time for a batch of tokens, with the token shift,
the elementwise chain and the WKV7 recurrence left per token and in token order
(the recurrence is 1.1-2.0% of the MACs — docs/rwkv7-plan.zh-CN.md section 12).

`RWKV_PREFILL=1` selects it (default off; `RWKV_PREFILL_N` = batch size, default
16).  The kernel is `matvec_q8_batch`: `matvec_q8`'s text with the input row taken
from `gl_FragCoord.y`, and each input keeps its own 2^k normalisation — the rows
are grouped by k, one draw per group, with the matrix scale folded exactly as
`Session::rwkv_mv` folds it — so every output row is the serial path's, bit for
bit.  One read-back per matrix per batch instead of one per token.

### Before / after (Iris Xe ADL GT2, g1d 0.1B q8r chat, one binary, interleaved)

512-token prompt (512 exactly: `data/ppl_test.txt` twice, truncated), `--gen 1`.
Every run is a fresh process, because the recurrence state has to start at zero.

| arm | prefill (3 runs) | per prompt token |
|---|---:|---:|
| serial | 62.25 / 62.48 / 60.43 s | 121.6 ms |
| `RWKV_PREFILL=1` (n=16) | **6.81 / 6.81 / 6.81 s** | **13.3 ms** |

**9.1x** on this run.  An earlier interleaved run the same day, on a quieter
machine, gave 52.20 -> 6.69 s = 7.8x: the ratio moves with the laptop's load, the
batched arm does not.  A 445-token prompt: 46.87 -> 5.91 s (7.9x).

Decode is untouched — the loop is the same code and the state it starts from is
bit-identical: 125.3 / 126.1 / 122.7 ms per token serial against 127.3 / 129.8 /
118.0 batched (medians 125.3 vs 127.3, one batched run faster: this laptop's
ordering noise), and **233 draws / 73 syncs per decode token on both arms**.

### Numerics

Byte-identical, not 'within noise': the same `LOGITS_OUT` file and the same greedy
text with `RWKV_PREFILL=0/1`

- g1d 0.1B q8r: 100-, 445- and 512-token prompts, batch sizes 1/4/16/64;
- g1d 0.4B q8r (dim 1024, 4x the modules): 100-token prompt;
- `RWKV_PREFILL_CHECK=1` compares each batched row against `GPU::matvec` on the
  same input in the same run: zero differences at every batch size, both
  containers.

That switch is what found the two defects worth recording, because both looked
plausible in the source:

1. **The read-back must not apply 2^k.**  The draw's `ws` uniform already carries
   the input's 2^k (the serial path folds it into the Mat view), so the kernel's
   integer *is* the final q24 value.  Scaling it again made every row with k>0
   come back 2^k too large — the first smoke test ran 8.5x 'faster' with logits
   that differed in the second byte.
2. **The row coordinate belongs to the texture, not to the block.**  The staging
   texture grows monotonically (one texture per call would put 392 of them in the
   scene ledger for a 445-token prompt), so a 1-matrix channel-mix call after a
   4-matrix low-rank one has a texture taller than its own block.  `(row+0.5)/xh`
   is a normalized coordinate, so `xh` has to be the texture's height; passing the
   logical block height read row `(b+0.5)/bh*tin.h` instead — every row of every
   call after the first block-size change.  A short prompt (one block size) looked
   clean, which is why the 100-token self-check is the regression that switch
   exists for.

PPL is unchanged: 9.2919 (nll 2.2291, 421 tokens, SEQ=512) on both arms.  The
`--ppl` path runs token by token either way, so that number is the serial path's
on both sides — the batched path's evidence is the byte-identical logits above,
not a PPL delta.  `RWKV_GPU=1` keeps the fused loop; its prefill is not batched.

### What the board still has to answer

Two VC4 questions decide the Pi's number, and both are the board run of this
commit:

- **The draw cap.**  A batched draw is `width x batch_rows` fragments, and the P7
  section above leaves open whether `kDrawCap` limits *fragments* (`rwkv_draw1`
  and `rwkv_wkv` assume so) or *width* (`draw()`'s comment says so).  The default
  splits so that `width*batch` stays inside the cap, i.e. the board pays the
  serial path's draw count and wins only the read-backs; `RWKV_PREFILL_STRIP`
  tests the width reading.
- **Width.**  The staging block is `groups x n` texture rows (64 at n=16) and the
  widest input is `dim/2` texels, so 0.1B needs 384x64 — inside VC4's 2048.

### The board's first attempt failed numerically (3fd8b21, before the fix)

The Pi Zero ran the A/B on the first integrated commit (`3fd8b21`, 0.1B q8r,
SEQ=512, ~90-token prompt): prefill **90.69 s -> 39.26 s (2.3x)** and the logits
**wrong** — max|d| 46.96, mean|d| 16.76, all 65536 dimensions over one q16 LSB,
top-1 moving from 11 to 8321.  Serial-vs-batched determinism held, so it was a
deterministic row-mapping defect, not a race.

The cause is defect 2 above: on the board the block sizes also change between
calls (the four low-rank downs are 4 rows deep, the channel mix is 1), and the
first block sizes the staging texture at 4 rows, so the *downs* call was already
reading `(b+0.5)/4*64` instead of row `b`.  Xe reproduced it the moment the check
ran on a prompt long enough to have more than one block-size change (12
mismatching matvecs on the 0.1B), which is why every earlier Xe check — short
prompts, one block size — came out clean and the board found it first.

`0a8b271` fixes it (`xh` is the texture's height, `xw` is already its width) and
adds the one-time guard: when a session first batches, one real projection goes
through both paths, and a mismatch on that machine falls back to the serial
prefill with a line saying so.  On Xe the logits are byte-identical again at
n=1/4/16/64 on 100-, 445- and 512-token prompts and on the 0.4B container.

### The board on the fixed commit: 2.6x, and numerically close but not exact

Pi Zero, q8r chat container, ~90-token prompt, one binary, `0a8b271`:

| arm | prefill |
|---|---:|
| serial | 93.63 s |
| `RWKV_PREFILL=1` (n=16) | **36.05 s — 2.6x** |

`LOGITS_OUT` against the serial arm: **max|d| 0.3672, mean|d| 0.0661, 63067 of
65536 dimensions over one q16 LSB, top-1 and the whole top-5 order identical**
(11 / 11685 / 12307 / 10460 / 12605).  Cosine is ~0.9999.

**So the Pi is not byte-identical, where Xe is.**  The bet was that the kernel
text is the serial kernel's, and on Xe it is, byte for byte.  On VC4 the same
text reads its staging row through a *uniform* coordinate while the serial kernel
bakes a literal, and this tree already has the measurement that those two are not
the same arithmetic there (AGENTS.md: "a constant is not a uniform" cost
MiniMind3 PPL 270.3170 -> 269.9303 for exactly that reason).  A sample landing on
the far side of a texel boundary in a few rows is enough: it is a coordinate
rounding difference, not a different sum order, which is why the magnitude is
0.37 and not 47.

Consequences, recorded rather than smoothed over:

- the switch stays **default off**; it is a long-prompt opt-in, and on the board
  it buys 2.6x with a ~0.9999 cosine and an unchanged top-1;
- the one-time guard ("the batched kernel must prove itself") compares one real
  projection against `GPU::matvec` on the machine that is running and accepts a
  difference up to 1.0 -- the board's 0.37 passes, the broken row mapping's 47
  does not -- and says on stderr when it accepted a non-zero one;
- `RWKV_PREFILL_CHECK=1` is still the exact comparison, for a machine where a
  difference needs to be named;
- nothing here claims bit-identity across machines.  Xe's byte-identical result
  is Xe's, and the board's is `cos ~= 0.9999, top-1 unchanged`.

The project stopped at this measurement (RWKV-7 optimization ended: the 0.1B is
slower and weaker than LFM2.5-230M on the board).  The machinery stays as a
default-off arm; the 0.4B, the strip question and a VC4-exact kernel are open.

## 2026-11-17 — MNN-Pocket-0.3B: f32 activation and KV streams, the q4 pitch bug, and the attention grids' names

The question was whether `models/pocket-0.3b` (`taobao-mnn/MNN-Pocket-0.3B`, a
Qwen3 dense 0.3B: 28 layers, hidden 896, 14 q heads / **2 kv heads**, head_dim
64, FFN 2560, qk-norm, rope theta 1e6 rotate-half, tied head) can run in this
engine at all, and what it costs.  Machine: Iris Xe (ADL GT2, Mesa 26.2.5) and
Pi Zero (VC4).  Container: `pocket-0.3b.q4q24.dedup.q6.mmcf` (198,007,797 B,
q24 residual declared, q6 tied head from the BF16 lm_head).

**Fidelity is where the campaign was won or lost.**  The reference is
`engine/tools/ref_forward.py` in float32 (the model's residual peaks at 1975 —
15x the q16 rail — and its attention scores reach 54, so this is a format
question all the way down).  Cosine of the engine's logits against that
reference, prompt ids 576 6649 315 9454 374:

| arm | Xe | Pi |
|---|---:|---:|
| q8 container, q16 streams (ceiling) | 0.9733 | — |
| q4, q16 streams, q24 residual | 0.9415 | — |
| q4, **f32 activation streams**, KV q16 @512 | 0.9531 | 0.9532 |
| q4, f32 streams, **KV bf16** | 0.9522 | 0.9454 |
| q4, f32 streams, **KV f32** | **0.9552** | — |
| q4 + q16 streams + `ATT_SUM=log2` | see below | — |

Two readings of that table matter.  **The activation streams were the biggest
single term** (0.9415 -> 0.9552); the KV encoding is worth 0.002 in the best
case.  **bf16 is not the KV upgrade it looks like**: it keeps the same two
values a texel and gives a uniform 0.1% relative error, which loses to the fixed
grid at the head max and only wins 2% a decade below it, so it measured
neutral-to-worse than simply widening the grid (q16 at `KV_SCALE=512`).  The
only encoding that removes the last-bit disagreements between the two drivers is
f32, and it costs 2x the KV memory and 2x the KV fetches.

`ATT_DUMP` + `check_att.py` at layer 0 with the f32 streams: qrot 0.40%, KV
1.18%, score 0.71%, smax 0.53%, **ssum 2.91%**, attout 2.49%, growing to 13-19%
by layer 8.  The sum was the largest term *after* the streams, which is what
produced the `ATT_SUM` question below.

**`ATT_SCALE` is not the fix for a saturated score.**  The scores saturate the
texel (|score| > 32 at `ATT_SCALE=1024`) and yet 1024 gives the best cosine on
*both* drivers: 512 and 256 make the distribution worse than the clipping they
avoid, because the coarser step costs more than the clip saves.  Recorded in
`docs/test-hooks.md` so the next person does not widen it again.

**`ATT_SUM=linear|log2` is a trade with two signs.**  The sum's linear q16 grid
is a 1/32 step — 2% of a peaked softmax's sum — and the log2 domain
(`log2(sum)*2048`) is 0.03% for any sum.  The accurate one is *not* the better
model: MiniMind3 `SEQ=768` on Xe, q8 PPL **270.3170 (linear) against 274.4160
(log2)**, q4 **326.1483 against 323.8378**.  The two containers disagree in
sign, so the sum's rounding redistributes a sub-0.1% per-head gain rather than
improving anything, and the default stays the linear grid; `ATT_SUM=log2` is
the A/B, and the f32-stream numbers above were measured with the linear one.

**The q4 pitch bug is the reason the model "did not fit".**  The first Pi upload
died at 93 MB with `texture upload failed: RGBA4 224x1280 (0x505)`; the
dma-buf import was refused with `unsupported stride 448 instead of 512` because
`t_bytes()` truncated the row pitch to a whole number of tiles (w=224 -> 896 B ->
448 B "pitch").  The silent fallback was `glTexImage2D` RGBA8 at 2x the memory,
which is what turned a 189 MB model into a 222 MB one and made CMA the story.
Fixed with `t_pitch_texels(w) = (w+63)&~63`; the fallback now warns once.  The
load then reached 189 MB of 222 MB at CMA 320 and completed.

**What the board costs.**  Pi Zero, `KV_FMT=f32 RES_FMT=f32 ACT_FMT=f32`,
`SEQ=256`: load 80-100 s, prefill ~10 s for 5 tokens, decode **~2.4 s/token**;
CMA 320 MB is required (256 MB cannot hold it) and leaves the OS ~165 MB.  A
6-prompt greedy battery: Pi answers 3.5/6 (correct `Paris`, `2 + 2 = 4`,
`The sky is blue.`), Xe 4/6 — the difference is a 3-prompt artifact of temp-0
chaos, not a machine defect: Xe vs Pi per-token logit difference mean -0.41,
std 0.89, max 4.0, and top-16 overlap Xe∩Pi 10/16 against Xe∩ref 9/16 and
Pi∩ref 8/16.

**The QAT Q4 file is a different checkpoint.**  `taobao-mnn/MNN-Pocket-0.3B`'s
GGUF Q4 is QAT (`Q4_1`, block 32); converted with
`models/pocket-qat/gguf_q4_to_hf.py` (GGUF v3 -> HF safetensors, dequantized to
F16, `reshape(reversed(dims))` for 2-D, no transpose — the first attempt had the
matrices transposed and scored cosine 0.005 against 0.9955 once fixed).  Weight
cosine against the BF16 release is 0.9955 and its norm/embedding vectors are
exact, but its *logits* against the BF16 reference cosine only 0.9261, so it
needs its own reference before any of its numbers mean anything.  On Xe it
answers `84 * 3 = 252` and a coherent Chinese sentence where the BF16-derived q4
degenerates on the Pi; on the Pi it no longer degenerates.  PPL/cosines for it
are still open.

**Page-cache bound, measured neutral.**  `Model::drop_pages()` +
`POSIX_FADV_DONTNEED` after each texture upload was added to keep a 200 MB
container from holding its own page cache in the movable zone during the upload.
Measured on both machines: no change to the load time or the peak (the kernel
reclaims clean file pages first under CMA pressure).  It stays as a bound, not
as a speed claim.

## 2026-11-17 (addendum) — pocket on the Pi: the degeneration is in the residual path, not the rails

Handling report: on the board, `pocket-qat` **greedily** degenerates after two
or three tokens (`杭州是中国中国中国中国…`), where the same binary on Xe writes a
full sentence.  Measured, one run each:

| configuration (both machines) | Pi | Xe |
|---|---|---|
| `RES=f32 ACT=f32 KV=q16@512` | `杭州是中国中国中国…` | full sentence |
| `RES=i24 ACT=i16 KV=q16@512` (container defaults) | loops | fine |
| `RES=f32 ACT=f32 KV=f32` | loops | fine |
| `+ ATT_SUM=log2` (sum grid 2% -> 0.03%) | loops | fine |
| `+ ATT_SCORE_GRID=256` (scores +-32 -> +-128) | loops | fine |
| `Q4_FMT=0` (packed q4 decode instead of the imported RGBA4 view) | loops | fine |
| `+ FFN_HID=f32` (new, same date) | loops (`…党的第1155年` repeating) | fine |

`Q4_IMPORT_CHECK=1` on the board reports `word0=0x8ca6 nibbles 6 10 12 8 ->
sampled 8 12 10 6`: the imported q4 surface round-trips the integers exactly
(reversed order, which the shader compensates), so the texture path is not the
error.

**What the per-layer dump says.**  Same prompt, same formats, `--gen 0`,
`HIDDEN_DUMP=1 DUMP_F32=1 STAGE_DUMP=1`, Pi vs Xe, cosine per stage:

| stage | cosine | max\|x\| | max\|Pi - Xe\| |
|---|---:|---:|---:|
| layer 0 `xb` | **1.00000** | 0.71 | 0.0015 |
| layer 0 `att` / `x1` | 0.99986 / 0.99998 | 0.50 | 0.004 / 0.003 |
| mid, `x1` | 0.9998 | 1827 | 3.8 |
| next, `x1` | 0.9951 | **2315** | **418** |
| late, `x1` | 0.79 .. 0.59 | 5 .. 20 | 4 .. 10 |
| last groups' first stage | 0.719 | — | 8 |

Layer 0 is identical to five decimals, so no kernel is *wrong*; the divergence
appears in exactly the layers whose residual reaches ~2300 — an 18% relative
difference where the signal is that large, because the dot products cancel — and
then accumulates to cosine 0.7 in the last layers.  That is VC4's arithmetic
(summation order and the texture sample's rounding) against Xe's, amplified by
this model's activation magnitudes, not a format limit: `RES`, `ACT`, `KV` and
now `FFN_HID` are all f32 in the best arm and it still loops.

**Consequence.**  Greedy decoding of this container is not usable on the board;
sampling with a repetition penalty is (`--temp 0.7 --topp 0.9 --top-k 64
--repeat-penalty 1.1`).  The engine keeps `FFN_HID=f32` because it removes a real
8x rail violation (the container's hidden peaks at 1005); it is not the fix for
this and was measured not to be.

## 2026-11-17 (addendum 2) — VC4's fragment ALU loses precision in multi-accumulator sums, and the probe that says so

Written for the pocket-on-Pi degeneration (addendum above).  The per-layer dump
put the loss in the layers whose residual reaches 2300; the question was whether
a *kernel* is wrong or the *arithmetic* differs.  `probes/sumorder_probe`
answers it with one 1024-term dot product whose exact value is known in closed
form: terms in pairs (+M, -M*(1-2^-20)), so 512 pairs give M/2048 exactly and
the sum cancels two orders of magnitude away from its result.  Two machines,
same binary source:

| shape | Xe: gpu - host | Pi: gpu - host | same value on both? |
|---|---:|---:|---|
| one accumulator (`s += w*x`) | 0.000% | 0.000% | **yes** |
| tree over the vector (`pairwise`) | 0.000% | 0.000% | **yes** |
| two accumulators, 2-wide trees (`two-tree`) | 0.001% | 0.000% | **yes** |
| one accumulator, 8-wide tree per group (`tree8`) | 0.043% | 0.000% | **yes** |
| four accumulators, 8 lanes, 32-wide group (**the shipped matvec shape**) | 0.000% | **44%** | no |
| eight accumulators | 0.000% | **489%** | no |

The pattern is unambiguous: **a single accumulator or a tree is bit-identical to
Xe; several live f32 accumulators are not.**  VC4's V3D 2.1 fragment ALU is
fixed-point under Mesa, and a shape that keeps four or eight accumulators alive
loses the low bits of the result when the accumulator's magnitude is set by the
inputs rather than by the result — exactly what a layer with |x| = 2300 does to
a dot product whose output is O(1).  Xe computes those shapes in IEEE f32 and
gets 0.000%.

Two side findings, both recorded in the probe's header: VC4 returns 0 for a
dynamically indexed **uniform array** (`mags[int(y)]`) while Xe honours it — the
engine never indexes a uniform array, so it is a probe-side constraint; and the
magnitudes' table also prices the *orders themselves* (at M=2300 the shipped
four-accumulator shape is 50% off the exact value **on Xe too**, against 0.17%
for a tree — an order that cancels early is better arithmetic, independently of
the driver).

### The first application, and what it did not fix

`matvec_i4_pair` (the q4 gate, five live accumulators) now builds the group's
32 taps as a tree and adds once per group — commit `c9c7853`.  It moves exactly
one program's digest (`matvec_i4_pair_2432x768_i16_2`, both q4shift variants),
and MiniMind3 q4's PPL 326.1483 and pinned string are **bit-identical** before
and after, i.e. the change is invisible where the accumulators never carried a
large magnitude.

**It did not stop pocket's repetition on the board** (`杭州是中国中国中国…` again,
decode 1696 ms/token).  So the q4 gate was not the source.  The remaining
candidates, in the order the per-layer dump suggests:

1. `att` is the first stage to diverge at the large-magnitude layers, but its
   own accumulators (`att_stats`' sum, `att_v`'s four channel accumulators) hold
   O(1) magnitudes, where the probe shows no loss.  Its *inputs* are the qkv
   matvec's output, whose accumulator reaches ~10-100x the result.
2. **`exp()` precision**: `att_v` evaluates `exp(dec16(score)*SCSI - mx)` per
   position.  VC4's transcendental approximation is not Xe's, and a relative
   difference there is a *multiplicative* weight error on every value in the
   sum, applied 28 times per token.  This is the next thing to measure, and it
   is a two-second probe, not a model run.

### The portable exp costs 0.4-1.8% and does not fix the degeneration

`ATT_EXP=poly` (commit below) replaces the driver's `exp` in the attention
weights with the degree-6 minimax that `probes/sumorder_probe` measures at
1.5e-06 on VC4 against the driver's 4.6e-05.  Same binary, one switch, runs
interleaved on the board:

| model | `ATT_EXP=drv` (ms/token) | `ATT_EXP=poly` | delta |
|---|---:|---:|---:|
| MiniMind3 q4, SEQ=128, gen 24 | 180.6 (median of 3) | 181.3 | **+0.4%** |
| pocket qat, SEQ=128, gen 32 | 1694.5 | 1725.3 | **+1.8%** |

Not an explosive cost, and the reason is structural: at SEQ=128 `att_v` runs 64
iterations that are dominated by texture fetches, so ~12 extra ALU ops per exp
hide under the fetch latency -- and VC4's own `exp`/`exp2` is a multi-op
sequence there, not a single instruction (that is *why* it is 20x less accurate
than Xe's).

**It does not stop pocket's repetition**: both settings print
`杭州是中国中国中国…`.  So the transcendental is not the cause either.  With this
change and the `matvec_i4_pair` tree (above), every kernel in pocket's decode
path now has a single accumulator or a tree -- the shape the probe calls safe --
and the divergence at the 2300-magnitude layers is still there.  The next
measurement is therefore not another knob: it is a same-path stage dump on the
board at those layers, before and after the two changes, to see which *stage*
still differs rather than which *kernel* might.

## 2026-11-17 (correction) — VC4's ALU is fp32, not fixed point; the weak link is the SFU

The two sections above explain the board's arithmetic with "VC4's fragment ALU is
fixed-point under Mesa".  **That is wrong**, and the probe numbers do not need it:
per the *VideoCore IV 3D Architecture Reference Guide* §3, the QPU's two ALUs are
32-bit **floating point** (one add pipe, one multiply pipe duality) with integer
and 4x8-bit packed modes alongside, and fixed point appears only in the
interfaces around them (Z is 24-bit in the FEP payload, colour is unorm8 in the
pack path, rasteriser coordinates are 11-bit integers).  Mesa compiles even the
fixed-function fragment pipeline (blend, texture combiners) to QPU float shaders.

What the `sumorder_probe` numbers still say, restated without the wrong
mechanism: single-accumulator and tree sums reproduce Xe **bit for bit** at every
magnitude, and the multi-accumulator shapes do not (44% and 489% at magnitude
2300).  With an IEEE-fp32 ALU the only way to get that is a **different
summation order** -- which is the compiler's, not the ALU's -- so the actionable
conclusion is the same (write the tree, keep one accumulator) but the reason is
"the backend schedules this shape differently", not "the mantissa is fixed
point".

The genuine accuracy gap is the **SFU**: `exp2`/`log2` are the shared
lookup-table unit and Mesa applies **no** refinement to them, unlike `rcp`/`rsqrt`
which get a Newton-Raphson step (`vc4_program.c`: `ntq_rcp`, `ntq_rsq`; commits
`1fc124b80f`, `15eb4c59f6`, and `283d4d18e5` for 1/W).  Mesa's own
`broadcom-rpi3-fails.txt` still lists `fs-exp-*`, `exp2-*`, `log-*`, `log2-*`,
`pow-*` against piglit's 1e-5 strict tolerance while `*sqrt` and `*-op-div` have
left it -- the observable consequence of N-R versus no N-R.  This matches what
the probe measures on the board (exp 4.6e-05 against Xe's 2.2e-06) and it is
*compliant*: GLSL ES 1.00 §4.5 allows 2^-16 relative for `highp` and says
compute-heavy built-ins may return less.

Consequence for this engine: the SFU is touched far more widely than the
attention weights -- the SiLU is `a/(1+exp(-a))` and lives in **five** emitted
kernels (`shaders.cpp`), i.e. 71,680 `exp` calls per token for pocket's
28 x 2560 FFN, against `att_v`'s two per position-fragment.  So `ATT_EXP=poly`
above prices and fixes the smaller half.  The next change is to promote it to a
single `EXP=drv|poly` axis covering the SiLU kernels as well, on the same
evidence: 1.5e-06 on the board against the driver's 4.6e-05, for +0.4-1.8%
decode.

### `EXP=poly` across every exponential: +1.5% decode, and pocket still loops

The axis was promoted from the attention to every shader-side `exp` (five SiLU
sites, the RWKV sigmoid, the attention weights).  Board, same binary:

| | decode (ms/token) | text |
|---|---:|---|
| `EXP=drv` | 1608.7 | `杭州是中国中国中国…` |
| `EXP=poly` | 1632.1 | `杭州是中国中国共产党党的党的…` |

**+1.5%**, consistent with the part-measurement above, and the degeneration is
unchanged.  So of the four things the per-layer dump left open, three are now
measured and excluded (format rails, summation order, exponential precision) and
the fourth -- the residual path's own arithmetic at 2300 magnitudes -- has no
single site left to fix: `matvec_i4_add`, `matvec_i4` and (since `c9c7853`)
`matvec_i4_pair` all sum in the shape the probe calls driver-identical.

What that means for the container: **greedy decoding of pocket on the board is
not reproducible**, and no switch in this tree makes it so.  Sampling with a
repetition penalty is the usable mode (`--temp 0.7 --topp 0.9 --top-k 64
--repeat-penalty 1.1`), and the Pi numbers for quality should come from the eval
batteries at `--limit`, not from a greedy string.  On Xe the same container is
fine; the divergence is the board's arithmetic plus this model's activation
magnitudes, accumulated over 28 layers.

## 2026-11-18 — the fine-grained layer comparison: no clean layer, and the oracle passes on both machines

Everything above used a per-stage **cosine** on `HIDDEN_DUMP`, which said layer 0
was identical to five decimals and the loss appeared around layer 7.  That
reading was wrong for the purpose: on a residual of magnitude 2300 a cosine of
0.99999 hides differences that matter, and it averages over 896 values.  The
finer instrument is `ATT_DUMP` — the q16/q24 **integers** each of the five
attention passes consumed and produced, for one position, every layer — plus
`scripts/cmp_att.py` (new) to diff two such logs.

Configuration: `ATT_SUM=log2 ATT_DUMP=12`, `SEQ=128 KV=q16 KV_GRID=512 RES=f32
ACT=f32`, the same prompt and container on both machines.  `ATT_SUM=log2` because
that is the domain `check_att.py` oracles; the comparison itself would work in
either.

**Both machines are integer-consistent with the specification.**
`check_att.py --sum log2` reports `FAILURES: 0` for all 28 layers on Xe
(`sum dev 0.000`, `att_out dev 0`) and on the board (`scores dev<=1`,
`att_out dev=1`).  So no attention kernel is wrong on either driver -- the
divergence is in what feeds the block.

**And it is already there at layer 0.**  max |difference| per pass, Xe against
the board, with the number of differing elements of 896 (q) / 1536 (k) / 182
(scores over 14 heads x 13 positions) / 896 (att_out):

| layer | qrot | k | score | att_out |
|---|---:|---:|---:|---:|
| 0 | 75 / 281 | 368 / 944 | 215 / 168 | 1 / 42 |
| 1 | 116 / 728 | 116 / 1252 | 770 / 163 | 1 / 110 |
| 6 | 72 / 817 | 179 / 1426 | 470 / 177 | 4 / 249 |
| 7 | 181 / 862 | 233 / 1608 | 649 / 182 | **33** / 782 |
| 8 | 667 / 891 | 789 / 1659 | 2322 / 182 | 242 / 863 |
| 10 | 45 / 760 | 1938 / 1638 | 4480 / 179 | 1114 / 884 |
| 23 | 504 / 892 | 3635 / 1660 | **13808** / 182 | 556 / 887 |
| 27 | 442 / 887 | 3037 / 1659 | 6897 / 182 | 446 / 895 |

Scores are on the 1/1024 grid, so layer 0's 215 is **0.21 natural units** and
layer 23's 13808 is **13.5 nats** -- enough to reshape a softmax.  q and k (post
rope, grids 1/256 and 1/kv_scale) grow the same way, and `att_out` only follows
after layer 6.

The layer-0 row is the result that changes the picture: qrot differs by up to 75
LSB = **0.29** while the hidden state it was computed from agrees to five
decimals.  Rope is a rotation with a host-uploaded table, so it cannot amplify a
1e-5 input difference into 0.29 -- **the qkv matvec itself already differs on the
first layer**.  That is consistent with q4 weights decoding a handful of nibbles
differently on VC4 (each off-by-one nibble moves one product by 1/15 of an
activation), which is the many-texel round-trip check that has *not* been run:
`Q4_IMPORT_CHECK=1` only samples texel (0,0).  That check is now the top
suspect again, with the integer dump as its evidence rather than a hypothesis.

Tooling: `scripts/cmp_att.py` added (with a README row); `check_att.py` stays
log2-only -- the linear domain's sum packing (`SMSI = 256/sum_scale`) did not
match its formula in one pass, so the default configuration still has **no**
oracle and that gap is recorded here rather than papered over.

## 2026-11-18 (addendum) — what the layer-by-layer integers exclude, and a correction

The section above concluded from layer 0 that "the q4 weight decode is back on
top".  Four experiments since then, each a full `ATT_DUMP` on the board diffed by
`scripts/cmp_att.py` against the Xe log, exclude every candidate that conclusion
implied.  Layer 0's numbers, Xe against the board, `RES=i24 ACT=i16 KV=q16@512`,
`ATT_SUM=log2 ATT_DUMP=12`:

| board configuration | qrot | k | score | att_out |
|---|---:|---:|---:|---:|
| default (imported q4 view) | 92 / 540 | 382 / 943 | 367 / 182 | 2 / 99 |
| `Q4_FMT=0` (packed q4 decode) | 92 / 540 | 382 / 943 | 367 / 182 | 2 / 99 |
| `Q16_DECODE=exact` | 92 / 540 | 382 / 943 | 367 / 182 | 2 / 99 |

Identical to the last digit, so:

- **the q4 weight path is not it** -- and not the imported RGBA4 view
  specifically, which was what `Q4_IMPORT_CHECK`'s texel (0,0) sample left open;
- **the q16 activity decode is not it** at layer 0 (it does move layers 1-2
  slightly, which is the documented cheap-decode behaviour);
- the qkv matvec is already a tree, and `sumorder_probe` shows trees are
  bit-identical across the two drivers.

**Correction to the section above**: the inference "qrot differs by 0.29 while
its input agrees, therefore the weights differ" does not survive these numbers.
What is left is the input difference itself -- `HIDDEN_DUMP` measured layer 0's
`xb` at max |d| 0.0015 on |x| 0.71, i.e. ~0.2% -- amplified through a 896-term
dot product.  At layer 0, q's own magnitude is not the O(1) the cosine made it
look: pocket's |K| reaches 33.6 (`probe_residuals.py`), so a 0.36 LSB difference
is about 1% of q, which is the right order for a 0.2% input difference integrated
over 896 terms.

So the board's picture is now: **no wrong kernel, no wrong format, no wrong
decode -- a 0.2%-per-value difference at the input of layer 0 that compounds to
10 nats of score difference by layer 23.**  Two things follow for anyone
continuing: the remaining lever is the *input* streams' precision (the rmsnorm
output and what feeds it), not the attention block or the weights; and a
comparison at this resolution must run in the formats the dump decodes
(`i24`/`i16`) -- the earlier f32 comparison was invalid because `ATT_DUMP`'s
decoder assumes q16 and both logs were misread the same way, which is why
`check_att.py` reported 56 failures on a healthy Xe run.

## 2026-11-18 — the rms partial sums saturate, and that is what broke pocket on the board

Found by working backwards from the one row the layer-by-layer integers showed as
a *pure scale*: layer 0's normalised row differed by the constant 1.00212
(corr(delta,h) = +1.000).  The rms norm is `x / sqrt(mean(x^2) + eps)`, and a
scale on `x` cancels there, so the factor had to come from `inv_rms` or the norm
weight.  `RMS_DUMP` (new, `engine/src/gpu_upload.cpp`) prints the 32 lane sums as
the shader wrote them, and the answer was in the encoding rather than in either:
the partials travel in **24-bit fixed point with a fixed 1/4096 step, biased by
2^23** (`kEncs4096`), so the largest lane sum it can carry is 2^23/4096 = **2048**.

MNN-Pocket's residual grows through the depth, and so do its sums.  Measured on
Xe with `RMS_DUMP`, sum of squares over the row:

| layer | 24-bit store | f32 store | error |
|---|---:|---:|---:|
| 0 | 0.4587 | 0.4582 | +0.1% |
| 10 | 261.5 | 259.5 | +0.8% |
| 15 | 1.059e6 | 3.350e6 | **-68.4%** |
| 20 | 1.066e6 | 3.930e6 | **-72.9%** |
| 25 | 1.068e6 | 3.928e6 | **-72.8%** |

From layer ~15 the lane sums pin at 8.39M LSB and stop growing: the 24-bit store
**saturates**, so `inv_rms` is about 2x too large and the last fourteen layers are
normalised by the wrong scale -- silently, on every machine.  At layer 0 the same
encoding has the opposite problem (lane sums of 5..92 LSB, about seven bits),
which is where the 0.212% scale came from.

`RMS_STORE=f32` stores the same four bytes as a bit-cast float and has neither
problem.  Board, MNN-Pocket qat, greedy, `SEQ=128 KV=q16 KV_GRID=512`, `--gen 24`:

| configuration | text | decode |
|---|---|---:|
| `RES=i24 ACT=i16` (container defaults) | `杭州是中国中国中国…` | 1694 ms/token (32 tokens) |
| `RES=f32 ACT=f32` | `杭州是中国中国中国…` | 1787 ms/token |
| **`RMS_STORE=f32`, `RES=i24 ACT=i16`** | **`杭州，是中华人民共和国浙江省的省会，位于中国东南部…`** | **1168 ms/token** |
| `RMS_STORE=f32`, `RES=f32 ACT=f32` | the same sentence | 1680 ms/token |

The repetition is gone, and the cheapest correct configuration is *faster* than
the broken one because the i24 activation streams are cheaper than f32 and the
f32 sums cost nothing measurable.  Every earlier attempt in this investigation
(formats, KV grids, summation order, portable exp) was working around a layer
whose normalisation was saturating.

**Default stays i24.**  An exact store moves MiniMind3's numbers (q8 PPL 270.3170
-> 276.1542), because MiniMind3's sums saturate too from mid-depth and its pins
were baselined in that state; switching the default is a re-baseline campaign,
not a bug fix.  `RMS_STORE=f32` is the switch for a container whose rms exceeds
the 24-bit range, and Pocket is the first one measured.

Caveat recorded deliberately: the layer-0 scale of 1.00212 **persists** with the
f32 store (it measured 0.00271), so the layer-0 quantisation and the layer-15
saturation are two ends of one encoding's range problem, and the greedy fix is
the saturation -- not the layer-0 factor.  The numbers above are the ones that
decided it.

## 2026-11-18 (addendum) — where a pocket token goes, and why it is already near the scaling law

`LAYER_PROFILE` now prints each lap's **draw count** as well as its time (a pass
costs a fixed draw overhead on VC4 before any arithmetic, so "which phase" and
"how many draws" are the same question when attention is 45% of a block whose
FLOPs are tiny).  MNN-Pocket qat on the board, `RMS_STORE=f32`, `SEQ=128
KV=q16 KV_GRID=512`, `--gen 1`, one token:

| phase | draws/token | draws/layer | note |
|---|---:|---:|---|
| attention | 224 | 8 | qkv, qknorm, qrot, kv_rope, scores, stats, v |
| both rmsnorm passes | 112 | 4 | split + norm each (the reduce is fused on VC4) |
| FFN gate | 56 | 2 | two chunks |
| w2 | 56 | 2 | two chunks |
| wo | 28 | 1 | |
| **total** | **476** | **17** | (`DRAWCOUNT` in the decode loop reports 563/token including the embedding and logits laps) |

**It is draw-bound, not FLOP-bound.** At ~1.2-1.5 s a token that is 2.5-3 ms a
draw, and the repo's own cost model puts a VC4 draw plus its forced finish at
~1 ms before any work.  Attention is 8 draws a layer for a block that computes
14 heads x 128 positions x 64 -- trivial arithmetic.

**Per layer it already matches the parameter count**, which is the scaling law
the model implies rather than anything the engine is losing:

| model | layers | dim | FFN | vocab | ms/token | ms/layer |
|---|---:|---:|---:|---:|---:|---:|
| MiniMind3 q4 | 8 | 768 | 2432 | 6400 | 304 | 38 |
| pocket qat (i24) | 28 | 896 | 2560 | 65536 | 1235 | 44 |
| pocket qat (f32 rms) | 28 | 896 | 2560 | 65536 | 1493 | 53 |

44/38 = 1.16 per layer against a projection ratio of (896/768)^2 = 1.36 and an
FFN ratio of 1.28 -- the same order, with the difference absorbed by the fixed
per-pass cost.  The 65536-entry head is a one-off (embed 2.5 ms, logits read-back
a few ms) and does not scale with depth.

**What is left, honestly**: the only structural lever is fewer passes, and the
two candidates are one draw a layer each -- folding `qknorm` into `qrot` (both
read and write q; scale then rotate is exact) and the `wo` -> `rms_att` seam.
28 draws a token is ~2% of the token, and **this board cannot resolve 2%**: two
interleaved runs of the *same* configuration measured 1439 and 1191 ms
(+/-20%), `LAYER_FAST_FINISH=1` measured 1236 and 1183 against those baselines,
and `ATTN_DUP=1` did not even change the draw count.  A future attempt needs an
interleaved >=6-run protocol per arm or a draw-count proxy, not a single A/B.

The one speed number from this session that *is* solid, because it is a same-binary
interleaved 2x2: the correctness fix costs **+21%** (1235 -> 1493 ms/token,
`RMS_STORE=i24` -> `f32`).  Earlier single runs that suggested the opposite were
warm-up noise.

## 2026-11-18 (correction) — pocket runs at SEQ=512 on the board

An earlier note in this log said the board can only take `SEQ=128` for this
container.  That is wrong.  Measured on the board, MNN-Pocket qat with
`RMS_STORE=f32 RES=i24 ACT=i16 KV=q16 KV_GRID=512`:

```
SEQ=512, --temp 0.7 --topp 0.9 --top-k 40 --repeat-penalty 1.1 --gen 256
-> 256 tokens, 0.8 tok/s (1.25 s/token), prefill 27.55 s, ctx 279/512
```

A coherent 256-token answer, no truncation before the context ran out.  The
constraint that bit earlier was the combination, not `SEQ` alone: with
`RES=f32 ACT=f32` the session's targets are wider (and it is also 21% slower),
and `SEQ=768` -- the container's own declaration -- is still untested here.  The
5-minute watchdog warning in the same note is also moot: 27.55 s of prefill plus
256 tokens at 1.25 s is about 5.9 minutes of wall time and it completed.

## 2026-11-18 — portability: request ES 3.0, and stop relying on Mesa's leniency

Two things made the ESSL 3.00 paths **Mesa-only**, both fixed today.

**The context was requested as version 2.**  The engine reached ESSL 3.00 only
through `MESA_GLES_VERSION_OVERRIDE`/`MESA_GLSL_VERSION_OVERRIDE`, which vc4
needs because it does not advertise ES 3.0 itself -- but on a driver that is not
Mesa there is no such variable, so every f32 stream (and bf16, when it lands) was
unreachable there.  `init_egl` now asks for a config renderable as ES 3.0 and a
version-3 context, falling back to the ES 2.0 pair if either step fails.  An
ES 3.0 context still accepts ESSL 1.00 shaders -- backward compatibility is in
the specification -- and the kernels that do not opt into ES3 keep their text,
their digests and their numbers: q8 PPL 270.3170, the q8 pin, 158 program
digests, and the f32 rms path at 276.1542 all unchanged on Xe, with the board's
q8 pin `271.2855` and pocket's sentence reproduced on the cross-built binary.
The Mesa overrides stay: they make the *request* succeed on vc4, they are no
longer what makes ES3 possible.  Measured separately: the context version alone
costs nothing (ES2 1116/1188 ms a token against ES3 1217/1106, i.e. inside the
board's noise), so the +21% of `RMS_STORE=f32` is the decode, not the context.

**The shell used two constructs only Mesa accepts.**  `#define gl_FragColor
frag_color` defines a `gl_`-prefixed macro, which the GLSL specification
reserves; the identifier is now renamed in the text instead.  And every f32 decode
calls `texture2D`, which ESSL 3.00 *removed*, so the shell now adds
`#define texture2D texture` and the same for `textureCube` -- legal macros that
leave the builders' text untouched.  Neither is visible to the gate, and that is
expected: `SHADER_SRC` and `--dump-shaders` print the source *before* the shell,
so the digests cover the pre-shell text.  What they are is the difference between
"compiles on Mesa" and "compiles on anything that implements ESSL 3.00".

Still not portable, and deliberately so: the q4 dma-buf import
(`EGL_LINUX_DMA_BUF_EXT` + `glEGLImageTargetTexture2DOES` + the tiled pitch),
which has a functional fallback in `Q4_FMT=0` (measured on the board to produce
**bit-identical** numbers, 2026-11-18), and the `MESA_*` overrides themselves,
which are a no-op elsewhere.

## 2026-11-18 — `XFormat::BF16`, and the three streams that can use it

bfloat16 is the top half of an f32: float32's exponent, an 8-bit mantissa, two
bytes a value.  It is now a first-class `XFormat` (`BF16 = 5`, two values a
texel, round-to-nearest-even in both the GLSL and the host packer) rather than a
private codec inside the attention block, and all three streams that could use
it do:

| stream | axis | default | bf16 measured |
|---|---|---|---|
| KV cache | `KV=bf16` | q16/i24 | **270.3891**, bit-identical to the codec it replaced |
| residual | `RES=bf16` | container's q24 | pocket **58.9558** (i24 62.3863, f32 59.3594) |
| activation streams | `ACT=bf16` | i16 | pocket **61.3250** (i16 62.3863, f32 61.8216) |
| FFN hidden | `FFN_HID=bf16` | i16 | pocket **60.9346** (i16 62.3863, f32 59.0763) |

Where the gain actually comes from, decomposed on pocket at SEQ=768:

| configuration | PPL |
|---|---:|
| container defaults (q24 residual, q16 acts) | 62.3863 |
| `RMS_STORE=f32` alone | **51.9263** |
| `RES=bf16 ACT=bf16 FFN_HID=bf16` alone | 61.0463 |
| bf16 streams *and* f32 rms | **51.1838** |

So most of it is the rms saturation fix (-17% perplexity on this container),
and bf16 adds a further 1.4% relative on top while halving three stores.  bf16
is not magic: it is the format that happens to be wide enough for a residual of
2300, and it is only better than q24 because q24's *range* was already fine but
its companion rms sum was not.

MiniMind3, same corpus: `ACT=bf16` 275.4097 against i16's 270.3170 and f32's
272.2534; `FFN_HID=bf16` 275.1395.  That is the trade as designed -- bf16 has
0.4% *relative* precision, so it beats q16 where the magnitudes are large
(pocket's residual reaches 2300, its FFN hidden 1005) and loses to q16 where they
are small (MiniMind3's residual is ~20, where q16's absolute 1/256 is finer).
It is the rung between "q16 rails at ±128" and "f32 costs twice the store and
21% of the decode".

**Defaults do not move**: q8 PPL 270.3170, q4 326.1483 and the q8 golden string
are unchanged, and 23 programs' digests moved only because their text now
carries the bf16 helpers (re-baselined in the same commit, with that evidence).

Three things this cost, each found by a wrong measurement rather than by reading:

- **The pair paths pack two *positions*, not two values.**  `x_p2_*` is the
  prefill pair layout, so teaching it bf16 is not the same as teaching the
  per-position pair kernels; `RES=bf16` produced 1e80 until `matvec_q8_add_pair`,
  `matvec_i4_add_pair` and their wrappers took the residual's format *and* the
  q8 inline fast path was guarded on `rf == I16_2`.
- **The FFN pair path is q16 by construction.**  `ffn_fmt_` bf16 has to take the
  two-buffer path, exactly as f32 does: it measured 7148610 with the pair path
  and 60.93 without.
- **The q6 tied head refused bf16 loudly** (`q6 head needs I16_2 or F32
  activations`) instead of computing garbage -- the closed-set rule working as
  intended on a *kernel* rather than a switch.
