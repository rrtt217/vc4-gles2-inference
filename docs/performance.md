# Performance

Measurements are from the Pi Zero and the local Xe/GLES2 machine. Treat them
as regression baselines, not portable benchmarks.

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
