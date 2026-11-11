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

> **Reproduced on the board, 2026-11-08.**  Lifting the draw cap
> (`QKV_PAIR=0 MATVEC_STRIP=1536`, i.e. the unpaired qkv drawn in one 1536-wide
> pass) prints *different* garbage every run - `..nnnnnnnn...`,
> `听今天今天...`, `佳译译译...`, `儿火火火...` - and `--ppl` goes
> **271.2855 -> 71669.1013**.
>
> **What is actually happening.**  `DRAW_CLEAR=1` (fill the target with
> 0x33/0x66/0x99/0xff before the draw, see `docs/test-hooks.md`) plus
> `QKV_DUMP=1` shows the target still holding the clear pattern after the draw:
> the draw was not executed, it did not merely read stale input.  With
> `PREFILL_PAIR=0` and one 1536 draw per layer, every layer's qkv comes back
> with **layer 0's** hash - the first full-width draw into the target lands and
> every later one is dropped.  Narrower draws (1535+1, 1024+512, 768+768) always
> land, so the whole target gets written.
>
> Not sync: `LAYER_SAFE_FINISH=1` (a real `glFinish` after the producing pass)
> changes nothing, and neither does drawing the qkv twice (`ATTN_DUP=qkv`).
> The command list does contain the 1536-wide clip window, so the call reaches
> the driver.
>
> The 2026-11-04 width sweep does **not** contradict this: `draw_probe` draws the
> wide matvec repeatedly into one target and compares after `glReadPixels`, and
> reports 1536 as byte-identical - its first draw lands, and it only needs one.
> The 2048 case is a different failure (5030/8192 bytes wrong in readback).
>
> **Where it sits in the driver.**  `vc4_draw_vbo()` (`vc4_draw.c`) has exactly
> two silent drops: an empty clip window (`vc4->clip_window_empty`, cached until
> `VC4_DIRTY_CLIP_WINDOW` is set again) and a failed shader variant
> (`vc4_update_compiled_shaders()`, whose warning is `debug_warn_once` - compiled
> out unless Mesa is built with `MESA_DEBUG`).  Two experiments narrow it:
>
> - forcing a clip-window recompute by drawing with a viewport one pixel wider
>   (the clip clamps back to the framebuffer, output identical) changes nothing -
>   so a stale empty clip window is not the whole story;
> - a failed variant would stay failed, and later full-width draws do land, so
>   the shader path does not fit either.
>
> What is left is below `vc4_draw_vbo`: the BCL the driver emits does contain the
> 1536-wide clip packet (`VC4_DEBUG=cl`), and the kernel builds the per-tile list
> from it.  Pinning that needs a Mesa built with `MESA_DEBUG` (so the drop warns)
> or a patched driver that counts the two early returns; neither is in this
> repository.
>
> The 2026-11-04 width sweep does **not** contradict any of this: `draw_probe`
> draws the wide matvec repeatedly into one target and compares after
> `glReadPixels`, and reports 1536 as byte-identical - its first draw lands, and
> it only needs one.  The 2048 case is a different failure (5030/8192 bytes wrong
> in readback).
>
> Both split points now use one rule: `draw()` and `matvec_add()` each cap a
> single draw at 1024 fragments, which is what keeps the qkv at 1024+512 and
> every other matvec narrower than its own target.  (The old 1536-only special
> case in `matvec_add()` was removed in `625c591`: it split 1536 into 768+768 and
> left every other width in one pass, and its loop walked rows instead of
> fragments, issuing a second strip entirely outside the paired target.)

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
`engine/scripts/verify_packing.sh` guards: it compares the first 16 greedy tokens
against the pre-packing engine byte for byte.


### 4b. The residual stream is q16, and that is a 128 rail

The residual is fixed point at 1/256, so it cannot hold a value beyond **±128**;
`add_q16` clamps there.  That is invisible for the models here (MiniMind3's
residual peaks around 40, see `docs/performance.md`), and it is the hard limit
for a model whose residual is larger: supra2-100M reaches **2300** at layer 4
and saturates, which reproduces exactly as a wrong argmax
(`models/README.md` has the measurement, including the clamped-simulation
proof).  A per-layer power-of-two residual scale is the fix that fits
(`GPU::set_residual_scale` exists but no caller does), and a single global
factor does not work: the early layers need ~1e-2 resolution while the late
ones need ~1e3 of range.

### 4c. Which GLSL ES builtins actually run

Measured with `engine/probes/packunpack_probe.cpp` on the Pi Zero (Mesa 26.2.2)
and on Iris Xe, ESSL 3.0 and 3.1, **with dynamic operands** — a literal is
folded on the CPU and proves nothing:

| builtin | Pi Zero | Iris Xe |
|---|---|---|
| shifts, `&`, `\|` on `highp uint` | ok | ok |
| `floatBitsToInt`, `intBitsToFloat`, `uintBitsToFloat` | ok | ok |
| `bitfieldExtract` | ok at 310 only, and only for 8/16-bit fields at uniform offsets | ok |
| `packUnorm2x16`, `unpackUnorm2x16` | **aborts the process** | ok |
| `packSnorm2x16`, `unpackSnorm2x16` | **aborts** | ok |
| `packHalf2x16`, `unpackHalf2x16` | **aborts** | ok |
| `roundEven` | **aborts** | ok |
| `floor` | ok | ok |

The abort is the VC4 backend admitting it has no lowering for the NIR op:
`MESA: error: unknown NIR ALU inst: 32 %2 = pack_unorm_2x16`, then `abort()`
from `vc4_program.c`'s `emit_alu()` default case.  Consequences:

- **half precision is not reachable** on VC4 — neither as a texture format
  (`OES_texture_half_float` is absent) nor as `packHalf2x16` (aborts).  A
  wider or floating-point residual would have to be assembled from float
  operations by hand.
- `roundEven` is likewise unavailable, which is worth knowing before rewriting a
  decode that wanted it.

Per-function instruction costs, the latency model and what Mesa lowers away are
in [`docs/vc4-glsl-costs.md`](vc4-glsl-costs.md), read from the same Mesa
version the board runs.

### 4d. Context length is a texture height

The KV cache is one target per layer, `(att_kv_width, seq)`, so `seq` is capped
by the smallest of the driver's texture, renderbuffer and viewport limits - 2048
on VC4, 16384 on Xe - and the engine clamps `SEQ` above it with a message.  The
score buffer `(seq/2, heads)` would allow 4096, so the KV cache is the binding
constraint.  Splitting the cache into segments is host-side for writes but needs
per-segment score targets and a reduce pass for reads (packed fixed point cannot
be added by the blender), and putting the sequence along the width instead gives
8-16k in one texture at the cost of changing every KV address.  Details,
alternatives and cost estimates: [`docs/long-context.md`](long-context.md).

### 5. Texture limits: 2048 per side, no half-float, so every weight is decoded

Probed with `engine/probes/texsize_probe.cpp` on this board (Mesa 26.2.2, VC4 V3D 2.1):

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

A third consequence: `SEQ` is clamped to
`min(GL_MAX_TEXTURE_SIZE, GL_MAX_RENDERBUFFER_SIZE, GL_MAX_VIEWPORT_DIMS)`
before the KV cache and attention targets are allocated.  On VC4 that limit is
2048; on the Iris Xe laptop it is 16384.  The engine prints a warning when it
clamps, instead of dying later with `FBO incomplete`.  A context longer than
the limit would need a tiled KV cache, which is not implemented.

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
serialized stage.  `--top-k 64 --top-k-candidates` does not win on 15M
because the top-2 pass costs more than the logits readback it removes.

Rejected experiments:

- MRT fused RoPE + KV write (`GL_EXT_draw_buffers`): compiled, but long output
  diverged (MAE ~0.38); reverted.
- Removing all finishes: 15M 146 ms, 42M 306 ms, but stale FBO made long output
  wrong; only the selective RMS/MAT/ATT shortcut is kept.
- CPU final RMSNorm, skipping hidden-upload finish: no measurable gain.
- `NF_FFN` (skip SwiGLU finish): slower on VC4.

`GPU_LAYER=1` implies GPU attention and GPU FFN, and
`--top-k 64 --top-k-candidates` still works on top of it.

### 4e. The activation grid is absolute, so a matvec input carries its own scale

`I16_2` stores q16: step 1/256, rail +-128.  That is an *absolute* resolution,
and a model whose activations live at 1e-4 -- LFM2.5's conv gate output at the
first position, its attention output at the second layer -- lands on a handful
of levels or on zero.  The upload is the boundary where this bites, because
every matvec reads its activation from a texture:

```cpp
vector<float> GPU::matvec_s(const Mat& m, const vector<float>& x);  // gpu.cpp
```

`matvec_s` takes the input's own maximum, picks the power of two that puts it
near 64 on the grid, uploads `x * 2^k` and multiplies the result back.  Because
`2^k` is exact and the hardware rounds on a lattice that scales with it,
`q16(x * 2^k) / 2^k` equals `q16(x)` whenever `x` is already well scaled -- MiniMind3's
perplexity is bit-identical with the serial path routed through it -- and only
differs where the unscaled grid would have rounded the value away.  LFM2.5's
layer-by-layer agreement with the float32 reference went from 6-11% to 0.2-1.4%,
and its first generated token for "What is C. elegans?" became the one the
reference predicts.

The same argument applies to the I16_2 *targets* the fused path writes
(`conv_mul`'s gate output, `conv3`'s result): they have no scale, so the fused
path still loses small activations.  The serial path is the reference today.

## Hybrid (conv) block

LFM2 alternates 8 short-convolution layers with 6 attention layers.  A conv
layer is a gated depthwise causal convolution:

```
xb   = x + attn(x)                 (residual already in xb)
b    = W_b * xb        B slice, 1024 rows
c    = W_c * xb        C slice, 1024 rows
g    = W_x * xb        X slice, 1024 rows
gate = b * g                       elementwise
cv   = sum_t K[:, t] * gate[pos - 2 + t]      depthwise, causal, k = 3
y    = c * cv                      elementwise
x    = x + W_out * y
```

`conv_in` is 3072 rows and the draw/target cap is 2048 (§5), so it is sliced
into three 1024-row views (`conv_in_b/c/x`) and drawn three times.  The two
gates, the convolution and the residual add are elementwise full-screen passes.

**State.** The last `k-1 = 2` gated rows are the conv's history.  Each conv
layer owns **three** rotating one-row targets: `conv_cur_` names the slot
holding `x[pos-1]`, the slot one step round holds `x[pos-2]`, and the third
holds `x[pos-3]`, which is dead.  The gate's output *is* `x[pos]`, so it is
written straight into the dead slot; the shift used to be two fullscreen copy
draws per layer per token and now costs nothing.  The three indices are
distinct mod 3, so no draw reads a target another one writes.

The prefix cache (v4) stores the two rows per conv layer in the order
`(0) = x[pos-2]`, `(1) = x[pos-1]`; a reload writes them into slots 2 and 0 and
resets the rotation, so the file format did not change.

**Widths.** `conv3_act` runs two channels per fragment: the two state rows, the
gate and the kernel row are all I16_2 texels holding exactly the pair
`(2q, 2q+1)`, so each texture is fetched once and the store is the q16 pair the
second gate wants (`x_st2q`).  That replaced 1024 fragments with three
single-value fetches each by 512 fragments with four pair fetches.

`mul_act` takes the source layout **per operand**: the first gate multiplies two
24-bit matvec outputs (`I24_1`, two fetches per texel), the second multiplies
`C*xb` (24-bit) by the q16 conv result (one fetch).  The 24-bit side is
`x_i24q`, i.e. real units, and the q16 side is raw/256, so both products are
real and the store is `x_st2`.

**Units.** Everything inside the block is raw q16.  `conv3_act` sums three raw
products and multiplies by 0.00390625 (an exact power-of-two scale, and a
multiply instead of the division the driver lowers to `frcp + fmul`) to stay
raw; `x_st2q` then clamps to the ±32768 rail.  The conv result therefore carries
the same rail as the residual, not the 256x range a 24-bit store would allow.

`probes/conv_probe` covers this block: section A drives the kernel fetch
patterns (1/2/3/5 fetches per fragment, `.rg`/`.ba`, height 3 and 4) and
compares **bytes**, sections B/C run the shipped `conv3_act`/`mul_act` from
`shaders.cpp` against a CPU reference that mirrors the shader's float32
expression order.  On both Xe and VC4 every case is exact, including three
fetches from one sampler at 512x3 -- the wrow plane's failure (§ below,
`probes/wrow_probe`) is shape-specific, not a general rule.

Known gap: an **i16** hybrid container on the fused path is wrong (PPL 1.39e7
against 98.10 serial).  It was already wrong at `f22bb76`, the serial path is
fine, and the q4 containers (the path the board runs) are exact; the cause is
not identified, so the fused path is only safe for q4/(q6 head) hybrids today.

## Top-K

`--top-k N --top-k-candidates` sources the k candidates from the candidate
pass instead of the full logits:

- a second GPU pass emits per-16-token top-2 values/indices and an online
  logsumexp;
- the CPU reads only candidates, computes the global denominator, and falls
  back to full logits if the candidate mass is below `top_p`;
- the candidate set is not guaranteed to contain the exact global top-k (one
  tile can contribute more than two of the k), so this is the approximate
  path.  Without `--top-k-candidates` the sampler reads the full logits and
  is exact;
- `TOPK_FAST=1` (with `--top-k-candidates`) is the older approximate
  candidate-renormalized path; it is not the correctness baseline.

After the 1536-column draw fix, the candidate path is correct on VC4 too.  It
is still not a net speed win there, so **full logits remains the default**;
use `--top-k 64 --top-k-candidates` when you want to test the candidate path.
The timings below were measured with the old `TOPK=64` switch, which selected
the same candidate pass.

Pi Zero, 15M q8:

| mode | decode |
|---|---:|
| full logits | ~137–151 ms/token |
| `TOPK=64` (now `--top-k 64 --top-k-candidates`) | ~146–147 ms/token |

Pi Zero, 42M q8:

| mode | decode |
|---|---:|
| full logits | ~346–355 ms/token |
| `TOPK=64` (now `--top-k 64 --top-k-candidates`) | ~350–428 ms/token |

Iris Xe, 15M I16: Top-K can be faster than full logits; measure on the target.
