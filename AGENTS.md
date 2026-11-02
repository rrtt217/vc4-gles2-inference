# AGENTS.md

Instructions for coding agents working in this repository.  Machine-specific
details live in local override files, not here:

- `AGENTS.local.md` — DSH local override (this laptop).
- `AGENTS.override.md` — Hermes/VLAB full override; it replaces this file.

Those files are local/personal and are gitignored.  Do not copy host names,
sysroots, deploy paths, or local-only benchmark numbers from them into this
file.  Target reference timings are allowed when explicitly labeled with the
target (for example Pi Zero).

## Mission

Maintain a practical GLES2/EGL inference stack for small language models on
Raspberry Pi Zero (VC4) and desktop GLES2. Preserve working behavior; do not
replace working paths with demos or cosmetic rewrites.

## Repository map

| Path | Contents |
|---|---|
| `engine/src/` | EGL/GLES2 engine: X-format layouts, shader builders, GPU, model, session, tokenizer, CLI |
| `engine/probes/` | standalone VC4 feature/throughput probes (not linked into the engine) |
| `engine/quantize/make_i16.py` | F16/llama2.c -> I16M1/M2 |
| `engine/quantize/make_q8_from_i16.py` | I16 -> Q8M1/M2 (splits fused W13) |
| `engine/quantize/make_q4.py` | F16 safetensors -> Q4M1 |
| `engine/quantize/make_q4_from_i16.py` | I16 -> Q4M1 |
| `engine/scripts/` | benchmark and regression helpers |
| `server/server.py` | OpenAI-compatible HTTP server and worker protocol |
| `models/` | central model store (gitignored) |
| `minimind3/` | MiniMind3 source config/tokenizer/reference; artifacts live in `models/` |
| `docs/` | deep dives; keep READMEs short and link here |

## Build

Build for the target and do not reuse a stale binary:

- Native GLES2: `make -C engine -B engine_mm3`
- Pi: cross-build with `make -C engine -B pi CROSS_SYSROOT=<sysroot>` on the
  cross host, or build on the Pi with the ARMv6 flags in the Makefile.

Do not use a short build timeout.  Machine-specific host/sysroot/deploy paths
belong in `AGENTS.local.md` (DSH) or `AGENTS.override.md` (Hermes/VLAB).

### Pi Zero reference (measured, not a target)

Measured on a Pi Zero, `GPU_LAYER=1 KV_PACK=16`, greedy, 32 generated tokens,
best of two runs; MiniMind3 uses `SEQ=768`.  Prefill is prompt-specific; the
full matrix and prefills are in [`docs/performance.md`](docs/performance.md).

| model | decode |
|---|---:|
| MiniMind3 q8 | 252 ms/token |
| MiniMind3 q4 | 388 ms/token |
| MiniMind2 q8 | 144 ms/token |
| MiniMind2 q4 | 187 ms/token |
| MiniMind2 i16 | 232 ms/token |
| stories15M q8 | 138 ms/token |
| stories42M q8 | 188 ms/token |
| TinyChat15M q8 | 137 ms/token |

A native ARMv6 build on a Pi Zero takes about 8–10 minutes; never run two
builds concurrently.  Prefer the VLAB cross-build (see the VLAB override).

## Canonical smoke tests

Run from `engine/` with the model store paths used by the local checkout.  The
system prompt is pinned on the command line: the compiled default changed in
`c1bb4f6` (Chinese persona -> neutral English), so a gate that omits `--system`
silently changes meaning and stops reproducing.

q8:

```bash
MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Offline q4 (do not set `Q4_WEIGHTS`):

```bash
MODEL=../models/minimind3/minimind3.q4 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Expected, byte for byte (Pi Zero and Xe agree):

- q8: `Assistant: 我是一个AI助手，我的存在是为了提供帮助和信息。`
- q4: `Assistant: 我是MiniMind，一个高效的小参数AI模型，专注于`

The same command **without** `--system` must print the same text; that is the
check that the compiled default has not moved.  If it differs, change
`kDefaultSystem`, this gate, `engine/scripts/regression.sh` and
`docs/performance.md` together, never one of them alone.

The older strings (`MiniMind是一个高效的小...` for q8,
`我是MiniMind，一个由jingyaogong...` for q4) were measured with the pre-`c1bb4f6`
Chinese default prompt.  They are still reproducible by passing that text to
`--system`; do not use them as the gate.  If a local q8 checkpoint is known to
differ, set `Q8_EXPECT` in the local override instead of weakening this gate.

## Invariants

- **Q4 draw** output is 24-bit; **Q4 residual-add** output is q16. Do not
  merge these encodings.
- **Q4 source**: quantize from F16/I16, not from q8. q8 -> q4 is a second
  quantization and loses quality.
- **KV cache**: Pi and laptop caches are not byte-identical. Build disk cache
  on the target machine.
- **CMA**: `CmaFree` is actual occupancy; reducing `cma`/`gpu_mem` does not
  create free CMA.
- **Prefix cache** header flags include pack mode; loading with another
  `KV_PACK` must be rejected.
- **FFN**: hidden 2432 is split into 1216-wide chunks because VC4 FBO width is
  2048. q4 slicing must copy block scale/min and repack nibbles.
- **CPU weights** are freed after upload by default; keep
  `KEEP_CPU_WEIGHTS=1` only for debugging.

## Documentation rules

- Root README = overview + quick start.
- `engine/README.md` = engine quick start; deep details go to `docs/`.
- Keep English and Chinese quick starts in sync:
  `README.md` / `README.zh-CN.md`,
  `engine/README.md` / `engine/README.zh-CN.md`.
- When changing a format, update `docs/model-formats.md`.
- When changing quantization, update `docs/quantization.md`.
- When changing performance numbers, update `docs/performance.md`.

## Source split policy

The engine is split into `engine/src/` modules; see `docs/engine-layout.md`.
Shader text belongs in `shaders.cpp` / `shaders_attn.cpp`, and activation
layout knowledge belongs in `xformat.h/.cpp`. Do not reintroduce per-shader
activation-layout strings. Run q8, q4 and `--ppl` comparisons after any GPU or
format change; a byte-identical greedy string is the minimum gate.

## Local overrides

- DSH reads `AGENTS.local.md` in addition to this file.
- Hermes/VLAB reads `AGENTS.override.md`, which **fully replaces** this file;
  keep that copy self-contained.
- Both files are gitignored and must not be treated as portable project rules.
