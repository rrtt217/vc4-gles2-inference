# Probes

Standalone programs that answer one question about the driver, the hardware or a
kernel, outside the engine.  They are not linked into `engine_mm3`, they are not
built by `make -C engine` (each is its own target), and the `_pi` variants are
cross-built for the board.

**Read this table before writing a new one.**  AGENTS.md records how a long
investigation once hand-rolled a probe that `engine/scripts/` already contained;
the same failure is easy to repeat here, because eight of these files carry their
own copy of the surfaceless-EGL setup.

| probe | answers | targets |
|---|---|---|
| `packunpack_probe` | which GLSL ES pack/unpack builtins does this driver actually run?  Every language feature the engine relies on, one draw each | `packunpack_probe_pi` |
| `egl_image_probe` | can a 4-bit texture reach the TMU *without* Mesa choosing the format?  Yes on this board: a 2048-byte dma-buf BO (64x16 at two bytes a texel) imports through `EGL_LINUX_DMA_BUF_EXT` + `glEGLImageTargetTexture2DOES` and samples back as exact nibbles (1 1 1 1 / 2 2 2 2).  That path bypasses `st_ChooseTextureFormat`, which otherwise adds `PIPE_BIND_RENDER_TARGET` for `GL_RGBA4` and lands on RGBA8 because vc4 cannot render to B4G4R4A4 | on the board (`g++ ... -I/usr/include/libdrm -lEGL -lGLESv2 -ldrm`) |
| `glsl_probe` | which GLSL ES dialect does the vc4 ES2 context really give us, and can the q4 nibble unpack use integer ops? | `glsl_probe_pi` |
| `q4fmt_probe` | what does the VC4 texture unit return for a small UNORM component?  The q4 split layout depends on the exact rounding | `q4fmt_probe_pi` |
| `rgba4444_probe` | does this driver give a real 4-bit-per-channel texture?  The q4 weight path is 4 bits per channel.  **Answered on VC4 (2026-11-14): yes** — exact `n/15`, same component order as Xe, 144/144 nibbles right on a 1216x192 matrix; see [`docs/benchmarks.md`](../../docs/benchmarks.md) | `rgba4444_probe`, `rgba4444_probe_pi` |
| `texsize_probe` | what texture height does this driver accept?  The engine allocates weight textures `rows` tall | `texsize_probe` |
| `matvec_probe` | how much of the engine's matvec time is int8 decode?  Synthetic weights, no model | `matvec_probe` |
| `q4mv_probe` | synthetic INT4 kernels at 64x64 and the fused FFN gate, with and without the unpack cost | `q4mv_probe`, `q4mv_probe_pi` |
| `wrow_probe` | does a per-row q8 scale plane reach the fragment shader?  Drives the engine's own kernels and names the rows that differ — it is how VC4's undefined two-fetch-from-one-sampler behaviour was found | `wrow_probe`, `wrow_probe_pi` |
| `conv_probe` | does the conv (hybrid) block's fetching work?  A: the kernel fetch patterns byte-for-byte; B/C: the shipped `conv3_act`/`mul_act` against a CPU reference | `conv_probe`, `conv_probe_pi` |
| `rwkv_probe` | does the RWKV-7 GPU elementwise chain's texture traffic work on this driver?  A/A3: 2D NPOT q24 constant fetch (GLES2/ES3); B: I24_1 render/fetch round trip (GL_DITHER A/B); C: shipped sqrelu at 768/1024/1025/1536 wide; D: bit-cast f32 round trip; E: matvec I24_1 X4Q 4-fetch; F/I/J: render→fetch and render→matvec(_add) glFinish A/B; G: q16 read ES3 vs GLES2; H: matvec_add I24_1+q16.  Used to locate the VC4 kDrawCap sqrelu drop and the fused matvec_add dot drop (P6) | `rwkv_probe`, `rwkv_probe_pi` |
| `t3mv_probe` | is a 2-bit ternary weight cheaper than a q8 one, and by how much?  The engine's q8 matvec body against the same loop with 16 weights a texel, decoded three ways (ES2 float `mod`/`floor`, ES3 uint shift/mask, and a fetch-only arm with the unpack removed), each checked against a CPU dot product.  **Answered on both machines (2026-11-18)**: on VC4 the uint decode is 2.44 ns/MAC against q8's 2.98 (1.22x) while the float decode is 7.5 (2.5x *slower*), and the fetch-only arm is 0.66 -- so the matvec is ALU-bound and 4x tighter weights buy a quarter, not four times; on Iris Xe the 2-bit kernels are no faster than q8 at all.  Numbers and the model-level extrapolation: [`docs/benchmarks.md`](../../docs/benchmarks.md).  The `t3p` arm answers the follow-up -- what a texel's four bytes are *worth*: with the packing read as 2 rows x 8 columns a fragment consumes all four bytes for 16 MACs and 4 decodes instead of 4 MACs, and VC4 measures **1.21-1.25 ns/MAC against the shipped text's 9.73** (2026-11-19). | `t3mv_probe`, `t3mv_probe_pi` (the board builds it natively with `g++ -O2 ... -lEGL -lGLESv2`) |
| `t3dec_probe` | what does the T3 weight *decode* cost, isolated?  Holds the shipped matvec_t3 layout/fetch/accumulation fixed and varies only `dec(byte)->vec4`: `ship` (the shipped text's 8 redundant byte->uint conversions), `hoist` (uint once), `code` (2-bit code algebra), `lut` (vec4 dynamic index), `signbit`/`sb_unroll`/`sb_trunc` (bit-cast +-1.0f), `sb_noand`/`sb_mask` (micro-omissions), `int_acc` (integer-domain multiply-free dot), and the 2-row pair forms.  **Answered on VC4 (2026-11-20)**: signbit 3.02 ns/MAC vs ship 3.46 (-13%), pair 2.03; the integer dot is +13%/+27% SLOWER (float sign-magnitude negation is free, two's-complement is not), the vec4 LUT +22%, the micro-omissions free-or-slower -- the float sign-bit decode is the floor | `t3dec_probe`, `t3dec_probe_pi` |
| `t3lut_probe` | can the T3 dot skip per-weight decode entirely (T-MAC's way)?  A 256-entry-per-group activation LUT in a TEXTURE, indexed by the 8-bit weight code, against the signbit decode on the same row-major weights.  **Answered on VC4 (2026-11-20): no -- the LUT is 2.4x SLOWER** (5.63 vs 2.33 ns/MAC at 1024x1024).  The packed format's 1-fetch-per-16-weights is its advantage; the LUT adds 4 random-access LUT fetches per 16 weights (5x traffic) and VC4's TMU cannot sustain it.  On Xe the LUT is faster (0.39 vs 0.45) -- fast texture, big cache -- which is exactly why T-MAC is CPU-only and does not transfer.  The LUT is host-built so the draw timing is the matvec alone; a model would also pay a GPU build pass per layer | `t3lut_probe`, `t3lut_probe_pi` |
| `t3act_probe` | does the ACTIVATION format move the t3 matvec?  I8_4 (`ACT=i8`, 4 values/texel, byte decode) vs I16_2 (default q16 pair, 2/texel, dec16) on the same signbit 2-row pair kernel -- halves the activation fetches.  Prices what enabling `ACT=i8` for t3 would buy (it is currently gated to all-q8 models, session.cpp), before the PPL cost | `t3act_probe`, `t3act_probe_pi` |
| `logits_grid_probe` | does the logits grid stay in one place?  The vocabulary matrix is stored in container tiles | `logits_grid_probe` |
| `progbinary_probe` | can a linked program be serialised on vc4, and does reloading skip the compile?  Item 6 (first-token latency) needed the answer | `progbinary_probe_pi` |
| `load_probe` | how long does the 69 MB q8 checkpoint take to reach RAM on this board?  Buffered vs mmap | `load_probe_pi` |
| `draw_probe` | what does one draw cost on vc4 before any real work is in it? | `draw_probe_pi` |
| `att_batch_probe` | does batching a prefill's N queries over one KV fetch pay?  Prices the compute: q1 (the shipped decode shape) against qN (N dot products a fragment, one KV fetch), with fetch-only arms and a q16 round-trip sanity check.  **Answered on VC4 (2026-11-19)**: ns/MAC falls linearly with N (19.03 at 1 query to 1.16 at 16) while ms/draw stays flat -- but the qN arm folds N-1 rows out of its output, so it prices the compute, not a dispatchable kernel; `att_out_probe` is the follow-up | `att_batch_probe`, `att_batch_probe_pi` |
| `att_out_probe` | does that win survive *writing the output*?  A VC4 fragment writes one texel (two q16 scores), so N queries a KV pair cannot all come out of one fragment without MRT.  Prices the dispatchable shapes at engine fragment counts: q1, qN_serial (own query only, N passes in one draw), qN_recomp (all N queries computed, own row written, N x redundant MACs), qN_mrt (glDrawBuffers, one KV fetch).  Every row checked against a CPU dot product; run at `--heads 1` and `--heads 16` to separate the batching effect from grid occupancy | `att_out_probe`, `att_out_probe_pi` |
| `sumorder_probe` | does the **summation order** change the answer on this driver?  A 1024-term dot product whose terms are 2^40 larger than its result (the shape MNN-Pocket's residual creates at magnitude 2300), through the four accumulation forms the engine's kernels use | `sumorder_probe`, `sumorder_probe_pi` |
| `ffn2_probe` | does 2-position packing make one weight fetch serve two positions, and does it stay byte-identical? | `ffn2_probe_pi` |
| `vc4_probe` | minimal FBO/texture smoke test: is the EGL/GLES path alive at all? | `vc4_probe` |
| `vc4_ffn_probe` | minimal split-FFN / Top-2 / q16 smoke test | `vc4_ffn_probe` |
| `vc4_topk_probe` | isolate Top-2: static texture against FBO-rendered texture, exact engine packing | `vc4_topk_probe` |

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
