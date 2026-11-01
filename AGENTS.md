# AGENTS.md

Instructions for coding agents working in this repository.

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
| `engine/quantize/make_q8_from_i16.py` | I16 -> Q8M1/M2 |
| `engine/quantize/make_q4.py` | F16 safetensors -> Q4M1 |
| `engine/scripts/` | benchmark and regression helpers |
| `server/server.py` | OpenAI-compatible HTTP server and worker protocol |
| `models/` | central model store (gitignored) |
| `minimind3/` | MiniMind3 source config/tokenizer/reference; artifacts live in `models/` |
| `docs/` | deep dives; keep READMEs short and link here |

## Build

Local:

```bash
make -C engine -B engine_mm3 CXXFLAGS='-O2 -std=c++17 -Wall -Wextra'
```

Pi (build separately from the run; 8–10 minutes on a Zero; never run two builds):

```bash
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

Do not use a short build timeout; a stale binary produces misleading results.

## Canonical smoke tests

q8:

```bash
cd engine
MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Offline q4 (do not set `Q4_WEIGHTS`):

```bash
cd engine
MODEL=../models/minimind3/minimind3.q4 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Expected q4 starts with `我是MiniMind，一个由jingyaogong...`.
Expected q8 starts with `MiniMind是一个高效的小...`.

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
