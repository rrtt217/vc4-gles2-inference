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
| `glsl_probe` | which GLSL ES dialect does the vc4 ES2 context really give us, and can the q4 nibble unpack use integer ops? | `glsl_probe_pi` |
| `q4fmt_probe` | what does the VC4 texture unit return for a small UNORM component?  The q4 split layout depends on the exact rounding | `q4fmt_probe_pi` |
| `rgba4444_probe` | does this driver give a real 4-bit-per-channel texture?  The q4 weight path is 4 bits per channel.  **Answered on VC4 (2026-11-14): yes** — exact `n/15`, same component order as Xe, 144/144 nibbles right on a 1216x192 matrix; see [`docs/performance.md`](../../docs/performance.md) | `rgba4444_probe`, `rgba4444_probe_pi` |
| `texsize_probe` | what texture height does this driver accept?  The engine allocates weight textures `rows` tall | `texsize_probe` |
| `matvec_probe` | how much of the engine's matvec time is int8 decode?  Synthetic weights, no model | `matvec_probe` |
| `q4mv_probe` | synthetic INT4 kernels at 64x64 and the fused FFN gate, with and without the unpack cost | `q4mv_probe`, `q4mv_probe_pi` |
| `wrow_probe` | does a per-row q8 scale plane reach the fragment shader?  Drives the engine's own kernels and names the rows that differ — it is how VC4's undefined two-fetch-from-one-sampler behaviour was found | `wrow_probe`, `wrow_probe_pi` |
| `conv_probe` | does the conv (hybrid) block's fetching work?  A: the kernel fetch patterns byte-for-byte; B/C: the shipped `conv3_act`/`mul_act` against a CPU reference | `conv_probe`, `conv_probe_pi` |
| `logits_grid_probe` | does the logits grid stay in one place?  The vocabulary matrix is stored in container tiles | `logits_grid_probe` |
| `progbinary_probe` | can a linked program be serialised on vc4, and does reloading skip the compile?  Item 6 (first-token latency) needed the answer | `progbinary_probe_pi` |
| `load_probe` | how long does the 69 MB q8 checkpoint take to reach RAM on this board?  Buffered vs mmap | `load_probe_pi` |
| `draw_probe` | what does one draw cost on vc4 before any real work is in it? | `draw_probe_pi` |
| `ffn2_probe` | does 2-position packing make one weight fetch serve two positions, and does it stay byte-identical? | `ffn2_probe_pi` |
| `vc4_probe` | minimal FBO/texture smoke test: is the EGL/GLES path alive at all? | `vc4_probe` |
| `vc4_ffn_probe` | minimal split-FFN / Top-2 / q16 smoke test | `vc4_ffn_probe` |
| `vc4_topk_probe` | isolate Top-2: static texture against FBO-rendered texture, exact engine packing | `vc4_topk_probe` |

## Where the answers are written down

Numbers go to [`docs/performance.md`](../../docs/performance.md) immediately after
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
