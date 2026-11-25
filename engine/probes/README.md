# Probes

Standalone programs that answer one question about the driver, the hardware or a
kernel, outside the engine.  They are not linked into `engine_mm3`, they are not
built by `make -C engine` (each is its own target), and the `_pi` variants are
cross-built for the board.

**Read this table before writing a new one.**  AGENTS.md records how a long
investigation once hand-rolled a probe that `engine/scripts/` already contained;
the same failure is easy to repeat here, because eight of these files carry their
own copy of the surfaceless-EGL setup.

**A per-draw time is two numbers.**  The board exposes no timer query --
`eglinfo -s` lists `GL_AMD_performance_monitor` and nothing with "timer" or
"disjoint" -- so a probe that prints ms/draw is printing QPU work plus the
submit/finish around it, and on VC4 the second term dominates the cheap draws.
`perfcnt.h` is the shared wrapper: `PERFCNT=1` adds the V3D counter columns
(QPU cycles, TMU quads, cache misses) next to the wall time, and unset leaves a
probe's output unchanged.  `draw_probe`, `matvec_probe`, `t3mv_probe` and
`rmsfuse_probe` are wired to it; the numbers are in
[`docs/benchmarks.md`](../../docs/benchmarks.md), "What a draw's time is made of".

| probe | answers | targets |
|---|---|---|
| `packunpack_probe` | which GLSL ES pack/unpack builtins does this driver actually run?  Every language feature the engine relies on, one draw each | `packunpack_probe_pi` |
| `egl_image_probe` | can a 4-bit texture reach the TMU *without* Mesa choosing the format?  Yes on this board: a 2048-byte dma-buf BO (64x16 at two bytes a texel) imports through `EGL_LINUX_DMA_BUF_EXT` + `glEGLImageTargetTexture2DOES` and samples back as exact nibbles (1 1 1 1 / 2 2 2 2).  That path bypasses `st_ChooseTextureFormat`, which otherwise adds `PIPE_BIND_RENDER_TARGET` for `GL_RGBA4` and lands on RGBA8 because vc4 cannot render to B4G4R4A4 | on the board (`g++ ... -I/usr/include/libdrm -lEGL -lGLESv2 -ldrm`) |
| `glsl_probe` | which GLSL ES dialect does the vc4 ES2 context really give us, and can the q4 nibble unpack use integer ops? | `glsl_probe_pi` |
| `q4fmt_probe` | what does the VC4 texture unit return for a small UNORM component?  The q4 split layout depends on the exact rounding | `q4fmt_probe_pi` |
| `rgba4444_probe` | does this driver give a real 4-bit-per-channel texture?  The q4 weight path is 4 bits per channel.  **Answered on VC4 (2026-11-14): yes** — exact `n/15`, same component order as Xe, 144/144 nibbles right on a 1216x192 matrix; see [`docs/benchmarks.md`](../../docs/benchmarks.md) | `rgba4444_probe`, `rgba4444_probe_pi` |
| `texsize_probe` | what texture height does this driver accept?  The engine allocates weight textures `rows` tall.  On the board it reproduces the hardware's 11-bit dimension field: 2048 allocates, 2432 is `GL_INVALID_VALUE` | `texsize_probe`, `texsize_probe_pi` |
| `matvec_probe` | how much of the engine's matvec time is int8 decode?  Synthetic weights, no model.  `PERFCNT=1` separates ALU (executing-valid) from TMU (text-quads) | `matvec_probe`, `matvec_probe_pi` |
| `q4mv_probe` | synthetic INT4 kernels at 64x64 and the fused FFN gate, with and without the unpack cost | `q4mv_probe`, `q4mv_probe_pi` |
| `wrow_probe` | does a per-row q8 scale plane reach the fragment shader?  Drives the engine's own kernels and names the rows that differ — it is how VC4's undefined two-fetch-from-one-sampler behaviour was found | `wrow_probe`, `wrow_probe_pi` |
| `conv_probe` | does the conv (hybrid) block's fetching work?  A: the kernel fetch patterns byte-for-byte; B/C: the shipped `conv3_act`/`mul_act` against a CPU reference | `conv_probe`, `conv_probe_pi` |
| `lnstats_probe` | how should the v5 LayerNorm's two statistics be encoded?  S = the shipped `(mean, 1/sd)` f32 pair (the engine's own split -> reduce -> apply); K = the `(mean, sd)·2^-k` q16 pair it replaced; B/C = mean in the residual format with `1/sd` as f32/bf16; D = `(mean, sd)` as bf16.  Every arm writes an f32 output, so the only difference measured is the statistics; the rows are a spread sweep plus the dec1 massive-activation row | host (needs an ES3 GL context) |
| `codec_probe` | what does each XFormat's encode and decode cost per value?  One decode kernel (one fragment a value, `X1`) and one encode (`XSTp`) per format, built from the engine's own codecs, both checked exactly (the pattern is k/16 in +-7.875, which every format represents: encode vs `x_pack` byte for byte, decode by an untimed decode->re-encode round trip).  `PERFCNT=1` adds the QPU's fragment-shading time.  **Answered on VC4 (2026-11-25)**, gpu ns/value, identical in two runs (wall ns is 2-4x and moves, it is submit+finish): decode i16_1 **4.83**, i24_1 6.08, f32 6.08, s24_1 6.91, bf16 13.16, i16_2 13.30, i8_4 18.30; encode i8_4 **2.83**, bf16 3.75, i16_2 4.03, f32 4.41, i16_1 4.69, i24_1 5.24, s24_1 6.63.  i16_2 decodes 2.8x a value more than i16_1 because one fragment a value fetches the texel per value and selects `.rg`/`.ba`, while the engine's pair kernels share one fetch (`x_i16pair`) -- that is a kernel decision, not the codec's floor.  Its first run also caught the I8_4 value decode dividing by 16 instead of 256 (16x; every hot kernel reads `X1Q`, so nothing had noticed) | `codec_probe`, `codec_probe_pi` |
| `rwkv_probe` | does the RWKV-7 GPU elementwise chain's texture traffic work on this driver?  A/A3: 2D NPOT q24 constant fetch (GLES2/ES3); B: I24_1 render/fetch round trip (GL_DITHER A/B); C: shipped sqrelu at 768/1024/1025/1536 wide; D: bit-cast f32 round trip; E: matvec I24_1 X4Q 4-fetch; F/I/J: render→fetch and render→matvec(_add) glFinish A/B; G: q16 read ES3 vs GLES2; H: matvec_add I24_1+q16.  Used to locate the VC4 kDrawCap sqrelu drop and the fused matvec_add dot drop (P6) | `rwkv_probe`, `rwkv_probe_pi` |
| `t3mv_probe` | is a 2-bit ternary weight cheaper than a q8 one, and by how much?  The engine's q8 matvec body against the same loop with 16 weights a texel, decoded three ways (ES2 float `mod`/`floor`, ES3 uint shift/mask, and a fetch-only arm with the unpack removed), each checked against a CPU dot product.  **Answered on both machines (2026-11-18)**: on VC4 the uint decode is 2.44 ns/MAC against q8's 2.98 (1.22x) while the float decode is 7.5 (2.5x *slower*), and the fetch-only arm is 0.66 -- so the matvec is ALU-bound and 4x tighter weights buy a quarter, not four times; on Iris Xe the 2-bit kernels are no faster than q8 at all.  Numbers and the model-level extrapolation: [`docs/benchmarks.md`](../../docs/benchmarks.md).  The `t3p` arm answers the follow-up -- what a texel's four bytes are *worth*: with the packing read as 2 rows x 8 columns a fragment consumes all four bytes for 16 MACs and 4 decodes instead of 4 MACs, and VC4 measures **1.21-1.25 ns/MAC against the shipped text's 9.73** (2026-11-19). | `t3mv_probe`, `t3mv_probe_pi` (the board builds it natively with `g++ -O2 ... -lEGL -lGLESv2`) |
| `t3dec_probe` | what does the T3 weight *decode* cost, isolated?  Holds the shipped matvec_t3 layout/fetch/accumulation fixed and varies only `dec(byte)->vec4`: `ship` (the shipped text's 8 redundant byte->uint conversions), `hoist` (uint once), `code` (2-bit code algebra), `lut` (vec4 dynamic index), `signbit`/`sb_unroll`/`sb_trunc` (bit-cast +-1.0f), `sb_noand`/`sb_mask` (micro-omissions), `int_acc` (integer-domain multiply-free dot), and the 2-row pair forms.  **Answered on VC4 (2026-11-20)**: signbit 3.02 ns/MAC vs ship 3.46 (-13%), pair 2.03; the integer dot is +13%/+27% SLOWER (float sign-magnitude negation is free, two's-complement is not), the vec4 LUT +22%, the micro-omissions free-or-slower -- the float sign-bit decode is the floor | `t3dec_probe`, `t3dec_probe_pi` |
| `t3lut_probe` | can the T3 dot skip per-weight decode entirely (T-MAC's way)?  A 256-entry-per-group activation LUT in a TEXTURE, indexed by the 8-bit weight code, against the signbit decode on the same row-major weights.  **Answered on VC4 (2026-11-20): no -- the LUT is 2.4x SLOWER** (5.63 vs 2.33 ns/MAC at 1024x1024).  The packed format's 1-fetch-per-16-weights is its advantage; the LUT adds 4 random-access LUT fetches per 16 weights (5x traffic) and VC4's TMU cannot sustain it.  On Xe the LUT is faster (0.39 vs 0.45) -- fast texture, big cache -- which is exactly why T-MAC is CPU-only and does not transfer.  The LUT is host-built so the draw timing is the matvec alone; a model would also pay a GPU build pass per layer | `t3lut_probe`, `t3lut_probe_pi` |
| `t3act_probe` | does the ACTIVATION format move the t3 matvec?  I8_4 (`ACT=i8`, 4 values/texel, byte decode, 2 fetches per 8 activations) vs I16_2 (default q16 pair, 2/texel, dec16, 4 fetches) on the same signbit 2-row pair kernel.  **Answered on VC4 (2026-11-20): I8_4 is ~20% faster** (2.03 vs 2.54 ns/MAC) -- halving the activation fetches beats the weight-decode savings.  But the I8_4 grid is +-7.94 / step 1/16 (`x_i8enc4` clamps v*16 to +-127): the rmsnorm-normalized matvec input mostly fits, the residual (BitCPM peak ~41) does NOT, so the gain needs i8 matvec INPUTS with a q16 residual -- not the current `ACT=i8` axis, which sets the residual too and is gated to all-q8 models (session.cpp:225).  A precision-changing path: needs a PPL re-baseline | `t3act_probe`, `t3act_probe_pi` |
| `rmsfuse_probe` | what does the rmsnorm NORMALIZE draw cost, and what would folding it into the consumer matvec save?  Prices the 32-lane stats split, the 1-fragment reduce, and the normalize draw (dim/2 fragments) fused 32-partial-fetch (norm32) vs pre-reduced 1-scalar (norm1).  **Answered on VC4 (2026-11-20, dim 1024): norm32 0.145 ms, norm1 0.034, stats 0.073, reduce 0.020.**  The fusion saves only ~0.4% model-wide: it removes the normalize draw (+finish ~0.525 ms/pass) but must add a reduce draw (+finish ~0.4 ms/pass) plus inline matvec normalize, net ~0.1 ms/pass x 48.  The rmsnorm overhead is the FINISH (serialization), not the normalize fragment work -- moving the work does not remove the pass.  Lever 1 is marginal; the non-decode overhead is structural per-pass serialization | `rmsfuse_probe`, `rmsfuse_probe_pi` |
| `logits_grid_probe` | does the logits grid stay in one place?  The vocabulary matrix is stored in container tiles | `logits_grid_probe` |
| `progbinary_probe` | can a linked program be serialised on vc4, and does reloading skip the compile?  Item 6 (first-token latency) needed the answer | `progbinary_probe_pi` |
| `load_probe` | how long does the 69 MB q8 checkpoint take to reach RAM on this board?  Buffered vs mmap | `load_probe_pi` |
| `draw_probe` | what does one draw cost on vc4 before any real work is in it, and how much of that is QPU?  `PERFCNT=1` adds the counters | `draw_probe_pi` |
| `att_batch_probe` | does batching a prefill's N queries over one KV fetch pay?  Prices the compute: q1 (the shipped decode shape) against qN (N dot products a fragment, one KV fetch), with fetch-only arms and a q16 round-trip sanity check.  **Answered on VC4 (2026-11-19)**: ns/MAC falls linearly with N (19.03 at 1 query to 1.16 at 16) while ms/draw stays flat -- but the qN arm folds N-1 rows out of its output, so it prices the compute, not a dispatchable kernel; `att_out_probe` is the follow-up | `att_batch_probe`, `att_batch_probe_pi` |
| `att_out_probe` | does that win survive *writing the output*?  A VC4 fragment writes one texel (two q16 scores), so N queries a KV pair cannot all come out of one fragment without MRT.  Prices the dispatchable shapes at engine fragment counts: q1, qN_serial (own query only, N passes in one draw), qN_recomp (all N queries computed, own row written, N x redundant MACs), qN_mrt (glDrawBuffers, one KV fetch).  Every row checked against a CPU dot product; run at `--heads 1` and `--heads 16` to separate the batching effect from grid occupancy | `att_out_probe`, `att_out_probe_pi` |
| `vc4_draw_drop_probe` | is a full-target-width viewport (`w == target_w`) what makes VC4 drop a draw?  Six widths x two target heights, each drawn full-width and one-pixel-narrower into a pre-filled target, plus a repeat test.  **Answered on VC4 (2026-11-21): no -- every arm lands, including 1536x1 drawn 1536 wide.**  The engine-side failure it was written for is not a width effect: `MATVEC_STRIP=1536` alone is fine, and only `QKV_PAIR=0 MATVEC_STRIP=1536` corrupts, so the trigger needs the unpaired qkv's 1280-row target as well | `vc4_draw_drop_probe`, `vc4_draw_drop_probe_pi` |
| `draw_cost_probe` | why does one engine draw cost 0.48-0.67 ms when a pipelined null draw is 0.008?  Changes one thing at a time on the same shader: the FBO -> texture dependency, the glFinish, and whether the draws are independent.  **Answered on VC4 (2026-11-21): the dependency is 5x (0.635 against 0.131 ms for identical work), the finish is a seventh of that, and the remaining ~0.47 ms appears in NO counter** -- `idle` and `waiting-varyings` were added to test that and both read 0, so it is outside the QPU's cycle accounting (a bin/render synchronisation).  A fusion A/B's wall delta is the evidence, not a counter delta | `draw_cost_probe`, `draw_cost_probe_pi` |
| `q16dec_probe` | what does the q16 activation decode cost, and can the sign fix be cheaper without changing a byte?  The engine's `u>=32768.0?u-65536.0:u` is 2 mul + add + compare + select, four of them per inner iteration against 8 MACs.  Prices the shipped text against an int sign-extend, a uint sign-extend and a branchless `step()`, all inside the real matvec loop and all bit-exact.  **Answered on VC4 (2026-11-21): the shipped text is already the cheapest -- `exec=` goes 1.99 -> 2.15 (branchless), 2.28 (uint), 2.30 (int) ms.**  Float select is nearly free on this ALU and integer negation is not, the same result `t3dec_probe` reached for the 2-bit sign decode | `q16dec_probe`, `q16dec_probe_pi` |
| `qkv_rows_probe` | how many output rows should one q8 matvec fragment amortise its activation decode over?  The engine's qkv pairs two rows and its FFN gate four, and the gate measures 1.79 ns/MAC against qkv's 2.87, so this prices a 4-row qkv kernel against the shipped 2-row one.  **Answered on VC4 (2026-11-21): two rows is already right, and the gate's advantage cannot be transplanted.**  Doing it properly needs two colour attachments and VC4 has no MRT (`GL_MAX_DRAW_BUFFERS` 1, `glDrawBuffers(2)` -> `GL_INVALID_OPERATION`); the dispatchable alternatives are +15% (grid runs both halves) or -2% inside noise (deepened body, but texture quads +50%).  The gate is fast because four weight fetches share one activation decode -- 16 MACs against qkv's 8 -- which is its maths, not a technique.  Every row is checked against a CPU dot product | `qkv_rows_probe`, `qkv_rows_probe_pi` |
| `sumorder_probe` | does the **summation order** change the answer on this driver?  A 1024-term dot product whose terms are 2^40 larger than its result (the shape MNN-Pocket's residual creates at magnitude 2300), through the four accumulation forms the engine's kernels use | `sumorder_probe`, `sumorder_probe_pi` |
| `ffn2_probe` | does 2-position packing make one weight fetch serve two positions, and does it stay byte-identical? | `ffn2_probe_pi` |
| `vc4_probe` | minimal FBO/texture smoke test: is the EGL/GLES path alive at all? | `vc4_probe` |
| `vc4_ffn_probe` | minimal split-FFN / Top-2 / q16 smoke test | `vc4_ffn_probe` |
| `vc4_topk_probe` | isolate Top-2: static texture against FBO-rendered texture, exact engine packing | `vc4_topk_probe` |

## The counter wrapper

`perfcnt.h` is a header, like `egl_surfaceless.h`, and for the same reason: a
probe is one translation unit and two of them link nothing but EGL and GLESv2.
It is not a probe and has no row above -- `check_scripts.py` maps four
directories and `probes/` is not one of them, so this table is the map by
convention, not by gate.

Three things about the counters are load-bearing, all of them measured on the
board (2026-11-21):

- **The result record is four words a counter**, `{group, counter, value, 0}`.
  Walking the buffer three words at a time reads the group id as a counter id
  and reports a plausible, wrong number -- the first version of the wrapper did
  exactly that and put `fragment-shading` at 0 for every arm while `idle` was
  fine.  A one-counter run and a many-counter run must agree; that A/B is the
  check.
- **The cycle counters are summed over 12 QPU-equivalents**, not 4.  With the
  wrong divisor a matvec reports 291% GPU time; with 12 it reports 94-97%, which
  is also what a wall-clock reading of the same arm says.
- **30 counters, 16 active at once.**  Pick the ones the question needs;
  `pick()` matches on a substring of the driver's own names and prints what it
  could not find rather than dropping it silently.

The clock is `vcgencmd get_config gpu_freq` (300 MHz here).  `clock MHz` and the
QPU count are compile-time defaults in the header; a probe on another board
should pass a measured value instead of trusting them.

## Where the answers are written down

Numbers go to [`docs/benchmarks.md`](../../docs/benchmarks.md) immediately after
a probe says something; driver capabilities go to
[`docs/vc4-glsl-costs.md`](../../docs/vc4-glsl-costs.md) and
[`docs/engine-internals.md`](../../docs/engine-internals.md).  A probe whose
answer changed a decision also appears in AGENTS.md's debugging table
("does a new sampler / texture fetch work on the target?"), which is where the
`wrow_probe` and `conv_probe` cases live.

## Build and run

```bash
make -C engine -B probes/<name>[_pi]                 # native
make -C engine -B probes/wrow_probe CROSS_SYSROOT=…  # the _pi variants need --sysroot
```

Native builds run on the laptop's GL driver; `_pi` builds are ARMv6 and run on the
board.  Only the `_pi` column is evidence about VC4 — a native probe run says
nothing about the board.

## Known duplication

Eight probes still carry their own surfaceless-EGL initialisation (50–60 lines
each, four of them copied from one commit).  `probes/egl_surfaceless.h` is the
extraction that was asked for when the ninth probe was added: include it, do not
copy the set-up again.  The eight older files can move over one at a time — a
probe's only contract is that its numbers stay the same.
