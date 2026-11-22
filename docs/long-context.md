# Long context on the Pi Zero: where 2048 comes from and what it costs to pass it

The limit is the **height of the KV-cache texture**, not the model, and the
engine says so and clamps:

```
$ SEQ=4096 ./engine_mm3_pi ...
SEQ 4096 exceeds GL target limit 2048; clamping to 2048
```

## 1. Which texture runs out

`GPU::max_target_dim_` is the smallest of `GL_MAX_TEXTURE_SIZE`,
`GL_MAX_RENDERBUFFER_SIZE` and `GL_MAX_VIEWPORT_DIMS`, read at startup
(`GPU::GPU`, which takes the smallest of the three limits).  On VC4 that is **2048**; on Iris Xe it is 16384.
`TEX_LIMIT=N` lowers it and never raises it, so a machine with a larger limit
runs the board's geometry and the difference between the two runs is the layout.

Per session (`GPU::init_attention`):

| target | shape | limit it imposes |
|---|---|---|
| `kv_cache_[layer]` | `(att_kv_width, seq)` | **`seq <= 2048`** — the binding one |
| `att_scores_` | `(seq/2, heads)` | `seq <= 4096` |
| `rope_tex_` | `(2*half, seq)` | `seq <= 2048` — a plain texture, no FBO, same limit |
| `vs_targets_[layer]` | `(att_kv_heads, seq)` | `seq <= 2048` — `KV=q8` with `KV_VSTEP` only |
| `att_out` | `(heads, head_dim)` | none |

So the score buffer already allows 4096; the KV cache is what stops us at 2048,
and it stops us by its *height*, because the sequence is laid out along rows and
the KV vector along columns.  Two more targets share that height and have to
move with it, not after it: the rope table (read by every one of the five
attention passes at `y = pos`) and, on the 8-bit path, the per-head V step
plane.

**The limit is the hardware's, not the driver's.**  The VC4 texture
configuration field is 11 bits (`HEIGHT` bits 30:20, `WIDTH` bits 18:8 in
Table 16 of `literature/broadcom/VideoCoreIV-AG100-R.pdf`, with `0 = 2048`), so
2047 is the largest dimension the format can name, and the errata entry
`HW-2753` says a child image at width or height 2048 fails with no workaround.
`probes/texsize_probe` on the board reads exactly that edge:

```
GL_MAX_TEXTURE_SIZE = 2048
w=192 h=1024 -> err=0x0
w=192 h=2048 -> err=0x0
w=192 h=2432 -> err=0x501
```

Allocating a 2048-tall texture succeeds and *using* it is what fails, which is
why the engine's own rule is stricter than the queried limit in two places: the
draw strips cap at `kDrawCap` (1024) and the t3 body tiles at 1024 rows, both
with a probe behind them (`probes/qkv_rows_probe`, `probes/t3mv_probe`).

Widths for the models here (`att_kv_width` = `kv_dim` at `KV=q16`):

| model | kv_dim | positions per row if the sequence went along the width |
|---|---:|---|
| MiniMind3 | 384 | 5 (1920 of 2048 used) |
| Supra2-100M | 256 | 8 |
| Supra2-Medium | 512 | 4 |

**Hybrid models (2026-11-11).** LFM2 has conv layers that never read a KV
cache, but the session used to allocate one for every layer.  At SEQ 2048 those
8 dead entries were 33.5 MB of CMA and the board failed at SEQ 1280 with
`FBO incomplete`; `init_attention()` now takes the layer types and skips them,
which puts 2048 back within reach at `KV=q16` (measured 2026-11-13, see
`docs/benchmarks.md`).  `KV=q8` fits too but costs fused PPL 126.89 ->
194.07, so it is not needed for context any more.

## 2. Why "just split the rows" is not a one-liner

Splitting the cache into N textures of 2048 rows is host-side for the **write**
path: `kv_rope_write` already draws one row at `(0, pos, kv_width, 1)`, so
binding `segment[pos / 2048]` and using `y = pos % 2048` needs no shader change.

The **read** path is where it costs, for two reasons.

1. *Samplers cannot be indexed.* ESSL 1.00 requires `texture2D` on a uniform
   sampler, so a shader cannot pick "cache texture number k" from a computed
   position.  With N a compile-time constant that is fine — N explicit sampler
   uniforms, no arrays — but the kernel has to be written for a fixed N.
2. *The accumulations cannot go through the blender.* The score pass writes
   packed fixed point (`x_i24`-style, 4 bytes carrying one 24-bit value), and
   `glBlendFunc(GL_ONE, GL_ONE)` adds per 8-bit channel without carrying.  Two
   segments added that way are not the sum of two 24-bit numbers.

So Na = 2 segments needs: one score draw per segment into its own target, then a
sum pass over the two (a shader with two samplers), then the existing softmax;
and the same shape for the value pass.  That is the honest cost of "split the
rows": **N× the attention draws plus a reduce pass**, not a config flag.

The reduce is cheaper than it reads, because the two accumulators already add in
the shader: `att_score`'s `s0`/`s1` and `att_v`'s `a0..a3` are float sums, so
segment *k* writes its partial sum into the same grid and the pass that adds
them is one fetch an output, not a re-run of the inner loop.  What it costs is
what §5 measures: on this board a draw is ~0.4 ms of submission before any
fragment is shaded, and the split multiplies the number of draws per layer per
token by `N`.

## 3. The alternative: tile the sequence along the width

A texture is 2048×2048, so laying the sequence across columns instead of rows
gives `positions_per_row = 2048 / kv_texels_per_position` rows of positions:

| model | positions per row | capacity in one texture |
|---|---:|---:|
| Supra2-100M | 8 | 16384 |
| MiniMind3 | 5 | 10240 |
| Supra2-Medium | 4 | 8192 |

One texture, one draw, no reduce pass — but every KV consumer changes its
address arithmetic: the rope write, the score pass, the value pass, the prefix
cache dump and its loader.  This is the design that scales; it is also the one
that touches the attention path everywhere.

## 4. What the models can actually use

| model | `max_position_embeddings` | trained context |
|---|---:|---|
| Supra2-100M | 2048 | 1024 (card) |
| Supra2-Medium | 1024 | 1024 (card) |
| MiniMind3 | 32768 | 768 (repo notes) |

Two of the three models here were trained at 1024, so a container that accepts
4096 buys **room for a prompt**, not better answers: past the training length
the model degrades regardless of what the engine can address.

## 5. Cost at runtime, measured

Attention work per generated token is linear in the context, and on this board
the linear term is no longer the small one.  MiniMind3 q8, `GPU_LAYER=1
KV=q16`, greedy, `--gen 4..8`, one run each (`TIMING`):

| machine | context | decode |
|---|---|---:|
| Pi Zero | ~1 token | **192.0** ms/token |
| Pi Zero | ~1.4k tokens | **317.2** ms/token |
| Iris Xe | ~1 token | **9.4** ms/token |
| Iris Xe | ~1.2k tokens | **16.3** ms/token |
| Iris Xe | ~2.4k tokens | **23.9** ms/token |
| Iris Xe | ~4.8k tokens | **38.7** ms/token |

Measured 2026-11-21 on the board (Mesa 26.2.2, cross-built binary) and on Iris
Xe (Mesa 26.2.6), one run each, prompt text repeated to length, `--gen 1..8`.
Decode is the reproducible number and prefill is not (AGENTS.md), which is why
only decode is tabulated; the same campaign's prefill ran 11 s on the board at
1.5k tokens and 89 s on Xe at 4.8k.

`LAYER_PROFILE` puts the per-layer attention block at 5.2-5.6 ms of a ~9.4 ms Xe
token *with an empty context*, and the three model projections around it at
1.5-1.7 ms each: the block is five dependent draws whose cost is submission and
per-draw setup, not `pos`.  So the shape of the cost is a fixed ~0.6 ms a draw
plus a `pos`-proportional scan, and both terms grow with what a longer window
asks for.

Extrapolating the linear part is what the design has to answer: at 38.7 ms for
~4.8k on Xe the slope is ~6 ms a 1k, so 16k of context is ~110 ms/token against
9.4 ms today, and the Pi's slope (~0.18 ms a token of context, from 192 at 1
against 317 at 1.4k) reaches the same place much sooner.  **A larger window is
affordable as a prompt, not as a conversation**: prefill pays the scan once,
every decoded token pays it again.

**What a split costs on the board.**  Measured 2026-11-22 on the Pi Zero with
the shipped q8 container, the same 2048-position window at one slice and at two:

| | 1 slice | 2 slices | |
|---|---:|---:|---|
| PPL | 271.2855 | 273.5285 | +0.83% |
| decode | 196.4 ms/token | 291.2 ms/token | **+48%** |
| prefill | 12.06 s | 14.99 s | +24% |

The decode cost is the one that matters and it is not the estimate this page
used to carry.  The split does not add two draws a layer as a paper calculation
suggests: it doubles the attention block's draws (`att_score` and `att_v` go
from one each to two, `att_stats` from one to two passes a slice, plus the value
fold), and a board draw costs ~0.4-0.6 ms of submission before any fragment is
shaded.  On Xe the same comparison is free -- 9.4 ms either way -- because a
draw there costs a fraction of that.  So a split window is worth its price only
when the window is the reason for the run: a conversation that fits in 2048 is
faster unsplit, and `SEQ` defaults stay where they are.

The PPL column's difference is the board alone and is not explained.  The same
comparison on Xe reads the same number for both, and repeating it on the board
with `Q16_DECODE=exact` still differs, so it is not the q16 decode's
classification either.  What is left is the order of the partial sums: a
two-slice window sums two halves of the stats scan and two halves of the value
row, and on VC4's fixed-point grid that can land an LSB away from the single
draw.

The window is not the only thing that grows.  The KV cache is
`layers x kv_width x seq x 4` bytes -- 3.1 MB at SEQ 2048 for MiniMind3 q8, and
that is the figure that decides whether the board's 320 MB CMA has room for the
weights beside it, not the one that decides whether the texture can exist.

## 6. What is implemented

**Two slices.**  `SEQ` may be up to twice the GL target limit: the sequence is
cut into two textures of at most `lim` positions each, and the context is then
bounded by memory rather than by 2048.  The layout is one texture a slice for
the KV cache, the rope table, the 8-bit V step plane and the score grid; each
slice gets its own program (`SEGK` is a compile-time constant), because ESSL
1.00 cannot index a sampler and VC4 has eight texture units.

The passes combine as follows, and none of them needed a new arithmetic:

| pass | a slice | the join |
|---|---|---|
| qkv, rope write, rotated Q | its own texture, its own program | none -- one position is in one slice |
| score | writes its own `seg/2`-wide grid | none |
| stats | scans the global pair range it owns, writes `(max, sum)` for its slice | a host fold: max of maxima, then `sum_k * exp(max_k - max_global)` in double |
| value | writes its own output row | a GPU fold: one draw, `X1` of each slice summed in float and re-encoded with the format's own `XST` |

Two folds are needed because neither quantity is additive as stored: the
stats sums were exponentiated against different maxima, and the value row is a
packed fixed-point layout that the blender cannot add channel by channel.

**Any number of slices works.**  The value fold is the only pass whose cost
depends on the count: its program carries one sampler a slice, so when there are
more slices than texture units it runs in rounds, ping-ponging the running sum
through two scratch targets (a round cannot sample the texture it renders into).
An adding round needs a unit for that sum, so it folds one fewer slice than a
reading round -- binding the sum to unit `n` with `n == units` is off the end and
the driver reads unit 0, which counts the first slice twice.

Two rules the sweep below found, both of which silently produce a plausible
number when broken:

- **A slice holds an even number of positions.**  The score grid packs two
  positions a texel, so an odd slice length puts its last pair half in this
  slice and half in the next.  The engine rounds the length down.
- **A pair index normalises by `(SEG+1)/2`, not `SEG/2`.**  They differ by one
  texel for an odd length, and every fetch then lands a texel short.

**The acceptance.**  Measured 2026-11-22 on Iris Xe (Mesa 26.2.6), MiniMind3
q8, `GPU_LAYER=1 KV=q16`, `data/ppl_test.txt`:

| window | slices | PPL |
|---|---:|---:|
| SEQ 768 (the shipped default) | 1 | **270.3170** |
| SEQ 4096, `SEG_CAP=2048` | 2 | **270.3170** |
| SEQ 1024, `SEG_CAP=512` | 2 | **270.3170** |
| SEQ 512, `SEG_CAP=256` | 2 | 270.4452 |

Measured 2026-11-22 as well, the same window at every slice count from one to
sixteen (`SEG_CAP` 2048 down to 128, MiniMind3 q8, same corpus).  Slices that
divide the window evenly reproduce 270.3170 exactly; the others land within
-1.4% / +2.0% of it (266.71 at fourteen, 275.79 at fifteen), which is the float
summation order of a different number of partial sums, not a structural error.

The last row is the one that differs, by 0.05%: a two-slice window that is not
a multiple of the slice length takes a different float summation order, and the
stats sum is quantised on a 1/32 grid, so one LSB of it moves.  It is a
rounding difference and not a defect -- the same window at one slice is in the
exact-match row above -- but it is why the table quotes that row separately.

`TEX_LIMIT=N` lowers the GL target limit so a machine with a larger one runs
the board's geometry; `SEG_CAP=n` forces a smaller slice than the driver needs,
which is how the split is exercised on Xe at all.  Both are probes, and
`nseg == 1` emits the shipped GLSL byte for byte -- `gate/shader_hashes.sh`
holds 164 programs whose digests did not move.

## 7. What has to move in the same change

The clamp lives in one place (`App::prepare_model`, and again after a `/model`
swap), and it is applied to every model that is not RWKV-7 because the CPU
layer loop allocates its own `seq`-sized KV.  That is why the number a user
sees is the same with `GPU_LAYER=0`: the limit is not the GPU's to lift, it is
the session's.

Every seq-sized target has to move together: the KV cache, the rope table, the
V step plane and the score grid.  A change that raises the cache and not the
rope table fails at the next `make_texture` with a message that names the
texture; one that raises both and not the score grid fails on the draw that
samples past its own height.  `att_probs_` was the fifth: it was allocated
`seq` tall for a pass that has not run since the fused q16 layout retired it,
and it alone would have kept `SEQ` clamped.
