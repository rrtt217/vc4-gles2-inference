# Engine internals and VC4 notes

This document collects the VC4 quirks, the FFN split policy, the GPU
attention/KV path, persistent GPU layers, and the Top-K candidate path.

## VC4 findings (current)

The earlier failures were not one bug.  Three separate things mattered.

### 1. `gl_FragCoord` is integer on VC4

`engine/probes/vc4_probe.cpp` shows:

| renderer | first pixel |
|---|---|
| Iris Xe | `(0.5, 0.5)` |
| VC4 V3D 2.1 | `(0, 0)` |

VC4 returns integer window coordinates even without
`pixel_center_integer`.  Sampling code must use an explicit pixel center:

```glsl
float x = float(int(gl_FragCoord.x));
vec2 uv = vec2((x + 0.5) / width, 0.5);
```

The engine uses this form in all FBO-sampling shaders.

### 2. A single 1536-column matvec draw is unreliable

A matvec whose output width is exactly 1536 (the logits tile width and fused
W13 width) intermittently reads a stale or zero target.  A zero RGBA target
decodes as q16 `-32768`, which is why an entire FFN could look like
`h = -5e7` and why logits sometimes had the wrong top-1.

Fix in `draw()`: when `m.rows == 1536`, render the viewport in two 768-column
strips.  `gl_FragCoord.x` includes the viewport origin, so the shader still
addresses the global row; only the draw width changes.

Other widths (864, 4096, 11008, 6144) stay in one draw.  Splitting every
matvec was noticeably slower on 42M.

This one fix removed the VC4 nondeterminism:

- 15M q8, `--temp 0.001`, 5/5 runs: `and Ben went to the park`
- forced `TOPK=64` gives the same output as full logits
- 42M q8: `and Tom were playing`
- MiniMind CPU FFN and GPU FFN now produce the same text

### 3. FBO -> texture sampling needs an explicit finish

On VC4, `glReadPixels` is not enough after rendering to an FBO texture, and a
second pass that samples that texture needs a finish before it starts.  The
engine keeps `glFlush(); glFinish();` on VC4:

- after logits are rendered, before the Top-K candidate pass;
- after matvecs, before readback;
- after texture uploads that feed a draw.

Iris does not force these finishes.

### 4. Activation textures hold two int16 values per texel

Every texture passed as `X` to a matvec program contains **two** int16
activations per RGBA8 texel: `(r,g)` is element `2t`, `(b,a)` is `2t+1`, both
big-endian in the same way as before.  Texture width is therefore
`(cols + 1) / 2`, not `cols`.

Why: before this change each multiply-accumulate cost one activation fetch plus
a quarter of a weight fetch (weights are 4-per-texel), i.e. 1.25 texture samples
per MAC.  Packing the activations halves the activation traffic to 0.75 samples
per MAC.  The logits pass predicted 6.14M samples / ~185 Msamples/s = 32ms and
measured 33.4ms, which is what identified texture sampling as the bottleneck.

Producers of an `X` texture — **all four must use the packed layout**:

| producer | where |
|---|---|
| host upload | `upload_activations` (used by `upload_input`, `logits`) |
| RMSNorm output | `rms_norm_program` (writes pairs, width `att_dim_/2`) |
| SwiGLU output | `swiglu_2row_program`, `swiglu_program` (width `hidden/2`) |
| attention output | `v_program` (width `att_dim_/2`) |

Residual (`R`) textures, matvec outputs and the KV cache are **unchanged**: one
value per texel.  `rms_stats_program` and `rms_norm_program` still *read* their
input one value per texel, because their input is a residual, not an `X`.

Missing any one of the four producers silently degrades generation into
repeated single tokens (the model sees near-constant activations).  That is what
`engine/verify_packing.sh` guards: it compares the first 16 greedy tokens
against the pre-packing engine byte for byte.


### 5. Texture limits: 2048 per side, no half-float, so every weight is decoded

Probed with `engine/texsize_probe.cpp` on this board (Mesa 26.2.2, VC4 V3D 2.1):

| check | result |
|---|---|
| `GL_MAX_TEXTURE_SIZE` | 2048 |
| `glTexImage2D(192 x 2432)` | `GL_INVALID_VALUE` (0x501) |
| `OES_texture_half_float` / `OES_texture_float` | **not exposed** |

Two consequences:

- **Anything taller than 2048 rows must be tiled.** The FFN matrices are chunked
  (`ffn_split_chunked`, `m_.ffn_chunk`) and the logits pass draws per 1536-row
  tile; a single 2432-row weight texture silently fails to upload.
- **There is no free-precision path.** Weights can only live in RGBA8, so every
  weight is decoded in the fragment shader. An "I16 is much faster" note in this
  repo's older tables does not hold on this driver: the INT16 format is also
  packed as bytes and decoded (`int16_program` spends *more* ops per weight than
  int8). Any further speedup has to come from doing less ALU per weight, not
  from a wider texture format.

## FFN policy

- VC4, split W1/W3 models: **GPU FFN** (W1/W3 -> SwiGLU -> W2, only W2 read back).
- VC4, fused W13 models (15M): **CPU FFN** (the 1536-row W13 matvec is split
  into strips by `draw()`).
- Iris: GPU FFN for split models; fused 15M stays CPU.
- `CPU_FFN=1` forces the CPU FFN path.
- `GPU_FFN=1` forces the GPU FFN path.

Fused W13 GPU SwiGLU was measured on Pi 15M q8: ~139.4 ms/token vs ~138.3
ms/token for CPU FFN, with GPU prefill slightly slower.  It removes the W13
readback but the extra GPU pass/finish cancels the gain, so fused 15M stays CPU
by default.  `GPU_FFN=1` still forces it for experiments.

## GPU attention / KV cache (experimental)

> `GPU_LAYER=0` is **not** CPU-only: attention/FFN move to the CPU, but the
> matvecs and logits still run on the GPU.  The engine has no CPU matvec path,
> so the CPU-attention numbers below are mixed CPU/GPU measurements.


`GPU_ATTN=1` keeps RoPE, the KV cache, attention scores/softmax and the
weighted V sum on the GPU.  Only the attention output is fed to the existing
GPU `wo` matvec; the CPU still handles the residual and FFN.

Layout:

- `qkv_rope` / `kv_cache`: 24-bit (or packed q16/q8) KV, scale 16384;
- rotated q and scores: q16 pairs, scale 1024; scores pack two positions per
  texel;
- stats: one texel per head, `(max, sum)` packed as q16 with scales 1024 and
  32;
- probabilities: decoded inside `att_v`; the separate probability texture is
  retired after the q16 score layout;
- attention output: the active X format (q16 pair or `I8_4`), ready for `wo`.

The CPU KV cache is not allocated when `GPU_ATTN=1`, so for 42M the 33 MB CPU
cache is replaced by a 33 MB GPU cache instead of adding both.

Correctness on Iris Xe (I16/q8, `--temp 0.001`):

| model | top-1 | logits cosine | MAE |
|---|---|---:|---:|
| 15M | same | 0.99992 | 0.213 |
| 42M | same | 0.99999 | 0.0175 |
| MiniMind | same at prefill | 0.99815 | 0.186 |

15M/42M match CPU attention for 32-token greedy generations; MiniMind can
diverge after ~20 tokens at `temp=0.001` because of q16/24-bit rounding.

Performance is mixed on Iris Xe:

| case | CPU attention | `GPU_ATTN=1` |
|---|---:|---:|
| 15M, 32 tokens | ~11.8 ms/token | ~15.1 ms/token |
| 42M, 32 tokens | ~22.1 ms/token | ~22.5 ms/token |
| 42M, 141-token prefill | 4.01 s | **3.60 s** |
| 42M, decode after long prefill | 22.5 ms/token | **19.7 ms/token** |

So it only starts to win when the sequence is long enough to amortize the
extra passes; at short sequences the CPU path is still better.

Pi Zero, q8, same comparison:

| case | CPU attention | `GPU_ATTN=1` |
|---|---:|---:|
| 15M, short prompt | 140.8 ms/token | 166.7 ms/token |
| 15M, 71-token prefill | 9.08 s | 11.75 s |
| 42M, short prompt | 350.6 ms/token | 382.3 ms/token |
| 42M, 71-token prefill | 32.14 s | **29.63 s** |
| 42M, decode after 71-token prefill | 414–484 ms/token | **383–421 ms/token** |

The 42M long-context result is the useful one: GPU attention is ~8% faster in
prefill and roughly 10–20% faster in decode in the best runs; the decode number
is noisy.  At short context it is still slower.  The Pi ran the full
GPU-attention path without VC4 errors; CMA free was ~100 MB afterward.

MiniMind q8 with `GPU_ATTN=1` is coherent but can diverge from CPU attention
after roughly 10 tokens at `temp=0.001` because of q16/24-bit rounding.

## Persistent GPU layer (experimental)

`GPU_LAYER=1` keeps the hidden state on the GPU for every layer:

- attention, RoPE and KV cache stay on GPU;
- Q RoPE is applied inside the score shader, K/V RoPE is written directly to
  the KV cache; the separate `qkv_rope` pass is gone;
- `wo` + residual and `W2` + residual are fused into one matvec pass each;
- RMSNorm and FFN stay on GPU;
- the CPU only uploads the token embedding and reads the final logits.

This removes all per-layer readbacks: 15M/42M go from ~25 readbacks/token to
one final logits readback.  The two residual fusions remove two more passes per
layer.  RMSNorm sum-of-squares uses a 24-bit fixed-point value at scale 4096.

Iris Xe results:

| case | CPU layer | `GPU_LAYER=1` |
|---|---:|---:|
| 15M, 32 tokens | ~17.9 ms/token | **~7.4 ms/token** |
| 42M, 32 tokens | ~24.9 ms/token | **~17.6 ms/token** |
| 42M, 141-token prefill | 3.80 s | **3.26 s** |
| 42M, decode after long prefill | 31.4 ms/token | **21.2 ms/token** |

Pi Zero results:

| case | CPU layer | `GPU_LAYER=1` |
|---|---:|---:|
| 15M, short prompt | 142–143 ms/token | **~135–138 ms/token** |
| 42M, short prompt | ~332–350 ms/token | **~306–330 ms/token** |
| MiniMind q8, 12 tokens | 15.7 s wall | **12.1 s wall** |

On VC4 the per-pass FBO finishes were the dominant cost.  `GPU_LAYER` now skips
the finish after RMS/MAT/ATT passes and uses `glFlush` after SwiGLU; this keeps
the same output as the fully-finished path on 15M/42M long runs (128/64 tokens),
while cutting decode from ~191/435 ms to ~135/306 ms.  `LAYER_SAFE_FINISH=1`
restores the fully-finished path for debugging.  Fusing the two residual adds
helped Iris but not VC4 on its own.  On Iris `GPU_LAYER` is a significant win.
Iris I16 15M/42M greedy outputs match CPU for 32 tokens.  On Pi, 15M I16 matches
CPU through 64 tokens; 15M at 128 tokens and 42M q8 at 32 tokens can diverge
because of q16/24-bit rounding.  MiniMind2 q8 is coherent but can diverge after
roughly 10 tokens.  MiniMind3 q8 is exact to the float32 reference for 64
tokens on the spring prompt; the self-introduction prompt diverges at token 41
on VC4, while the CPU-attention path stays exact.

Profile (`LAYER_PROFILE=1`, with a `glFinish` at stage boundaries; absolute
values are inflated, use the shares):

| stage | 15M share | 42M share |
|---|---:|---:|
| attention (5 passes) | ~35% | ~18% |
| final logits readback | ~14% | ~8% |
| RMSNorm (2 passes/layer) | ~20% | ~45% |
| FFN | ~14% | ~11% |
| wo / W2 matvec | ~17% | ~12% |

The two remaining bottlenecks are the 5-pass attention chain and the single
128 KiB logits readback.  On 42M, the two RMSNorm passes become the largest
serialized stage.  `TOPK=64` does not win on 15M because the top-2 pass costs
more than the logits readback it removes.

Rejected experiments:

- MRT fused RoPE + KV write (`GL_EXT_draw_buffers`): compiled, but long output
  diverged (MAE ~0.38); reverted.
- Removing all finishes: 15M 146 ms, 42M 306 ms, but stale FBO made long output
  wrong; only the selective RMS/MAT/ATT shortcut is kept.
- CPU final RMSNorm, skipping hidden-upload finish: no measurable gain.
- `NF_FFN` (skip SwiGLU finish): slower on VC4.

`GPU_LAYER=1` implies GPU attention and GPU FFN, and `TOPK=64` still works on
top of it.

## Top-K

`TOPK=N` enables the exact candidate path:

- a second GPU pass emits per-16-token top-2 values/indices and an online
  logsumexp;
- the CPU reads only candidates, computes the global denominator, and falls
  back to full logits if the candidate mass is below `top_p`;
- `TOPK_FAST=1` is the older approximate candidate-renormalized path; it is
  not the correctness baseline.

After the 1536-column draw fix, Top-K is correct on VC4 too.  It is still not a
net speed win there, so **full logits remains the default**; use `TOPK=64`
when you want to test the candidate path.

Pi Zero, 15M q8:

| mode | decode |
|---|---:|
| full logits | ~137–151 ms/token |
| `TOPK=64` | ~146–147 ms/token |

Pi Zero, 42M q8:

| mode | decode |
|---|---:|
| full logits | ~346–355 ms/token |
| `TOPK=64` | ~350–428 ms/token |

Iris Xe, 15M I16: Top-K can be faster than full logits; measure on the target.
