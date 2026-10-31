# VC4 / GLES2 MiniMind3 inference

Practical, headless inference for small language models on a Raspberry Pi Zero
(BCM2835 / VC4 V3D 2.1) through GLES2/EGL. It is not a demo: it runs a
Qwen3-style 64M model, a native tool loop, an OpenAI-compatible HTTP server,
and a disk KV prefix cache.

[中文说明](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What works

- **MiniMind3 64M** — 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV heads,
  head_dim 96, vocab 6400, RoPE theta 1e6, per-head Q/K RMSNorm.
- **Formats** — legacy llama2.c, Q8M1, I16M1, Q8M2/I16M2, and offline **Q4M1**.
- **KV cache** — 24-bit baseline, packed q16 (default recommendation), and
  packed q8 (quality caveat).
- **Server** — Python stdlib OpenAI-compatible chat completions, SSE,
  standard tool calls, usage stats.
- **Tools** — built-in time and arithmetic tools; the model emits tool calls,
  the worker executes them and feeds back tool responses.
- **Prefix cache** — worker token/KV reuse plus an `MMKV1` disk cache.

## Quick start

### Laptop

```bash
cd engine
make engine_mm3

MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Pi Zero

Build on the Pi, separately from the run:

```bash
cd engine
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

Run:

```bash
cd engine
MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Offline q4

```bash
python3 engine/make_q4.py --input models/minimind3 --output models/minimind3/minimind3.q4 --seq 512

cd engine
MODEL=../models/minimind3/minimind3.q4 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

Offline Q4M1 is 45 MB vs 66 MB for q8. After the shader unroll it is close
to q8 locally (~16–23 vs ~14–20 ms/token) but saves weight memory. Use the F16 source; q8 -> q4
is a second quantization and degrades.

### OpenAI-compatible server

```bash
cd server
python3 server.py --seq 768 --kv-pack 16 \
  --prefix-cache prefix_tools_pack16_768.kv
```

See [server/README.md](server/README.md) for endpoints, tools and cache
details.

## Repository map

| Path | Contents |
|---|---|
| [engine/](engine/) | GLES2 engine, converters, probes |
| [models/](models/) | central model store (gitignored) |
| [minimind3/](minimind3/) | MiniMind3 source config, tokenizer source, reference |
| [server/](server/) | OpenAI-compatible Python server |
| [docs/](docs/) | deep dives: formats, internals, quantization, performance |

## Model sizes and speed

| Path | Size | Local decode |
|---|---:|---:|
| MiniMind3 q8 | 66 MB | ~17 ms/token |
| MiniMind3 offline q4 | 45 MB | ~16–23 ms/token |
| MiniMind3 i16 | 132 MB | reference/source |

On Pi Zero q8: ~520 ms/token decode, ~42.7 s for an 82-token prefill.
Numbers are machine-specific; see [performance.md](docs/performance.md).

## Documentation

- [Engine quick start](engine/README.md)
- [Model formats](docs/model-formats.md)
- [Quantization and Q4M1](docs/quantization.md)
- [VC4 / engine internals](docs/engine-internals.md)
- [Prefix cache](docs/prefix-cache.md)
- [Performance](docs/performance.md)
- [Test hooks](docs/test-hooks.md)
- [Source layout / split decision](docs/engine-layout.md)
- [Future optimization directions](docs/future-optimization.md)
- [Agent notes](AGENTS.md)

## Models and credits

Model weights and tokenizers are **not included**; put them under `models/` as described in [models/README.md](models/README.md).
The MiniMind3 architecture/tokenizer follows [jingyaogong/minimind](https://github.com/jingyaogong/minimind) (Apache-2.0); the base engine follows [karpathy/llama2.c](https://github.com/karpathy/llama2.c) (MIT).

## License

MIT. See [LICENSE](LICENSE). Model weights and third-party model assets are
not covered by this repository license; see the credits above.

## Known caveats

- Pi and laptop KV caches are **not byte-identical**; build the disk cache on
  the machine that will load it.
- `CmaFree` is real CMA occupancy. Reducing `gpu_mem` or `cma` does not create
  free CMA and can make pressure worse.
- Pi Zero with `SEQ=768` + packed q16 is tight (~2 MB `CmaFree` after load);
  test the exact model/cache combination.
- q8 KV is fluent but not quality-safe; q16 packed is the safe KV default.
- q4 is not bit-exact, and its arithmetic quality is prompt-dependent.
- The engine is a single translation unit today; see
  [engine-layout.md](docs/engine-layout.md) for the staged split plan.

