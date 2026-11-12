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
- **MiniMind2 Small** — i16/q8/q4; Qwen-style chat, weaker instruction
  following than MiniMind3.
- **LFM2.5-230M** — the hybrid model: 8 depthwise-conv layers interleaved with
  6 attention layers, 1024 dim, 64402 vocab, the vendor's own ChatML-shaped
  template with tools and `<think>`.  Fused decode ~33 ms/token on the laptop
  and ~1.0 s/token on a Pi Zero at `SEQ=256`; perplexity is within 9% of the
  float32 reference.  Page: [docs/lfm2.5-230m.md](docs/lfm2.5-230m.md).
- **One file format** — the self-describing **MMCF** container.  Weights carry
  their tokenizer, chat template, rope configuration and capability flags, so
  one `MODEL=` path is the whole configuration.  The legacy layouts
  (llama2.c fp32, Q8M1, I16M1, Q8M2/I16M2, Q4M1) are conversion *inputs* to
  `engine/tools/mmcf_pack.py`; the engine reads only `MMCF`.
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

MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
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
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Offline q4

```bash
# One step: the packer quantizes from safetensors, no intermediate file.
python3 engine/tools/mmcf_pack.py --in models/minimind3 --quant q4 --seq 512 \
  --tok models/minimind3/minimind3.tok --out models/minimind3/minimind3.q4.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja --verify

cd engine
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
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
| [engine/](engine/) | GLES2 engine, the container toolchain, converters, probes, scripts |
| [models/](models/) | central model store (gitignored) |
| [minimind3/](minimind3/) | MiniMind3 source config, tokenizer source, reference |
| [server/](server/) | OpenAI-compatible Python server |
| [docs/](docs/) | deep dives: container, formats, internals, quantization, performance |

## Model sizes and speed

Every row is an `.mmcf` container; the legacy file it was packed from stays
next to it in `models/` as a re-pack source.

| Model | Size | Pi decode | Notes |
|---|---:|---:|---|
| LFM2.5-230M q4 (QAD) | 169 MB | ~1040 ms/token | hybrid conv+attention; `GPU_LAYER=1 CONV_GPU=1`, `SEQ=256` |
| MiniMind3 q8 | 66 MB | ~238 ms/token | default Chinese chat |
| MiniMind3 q4 | 45 MB | ~381 ms/token | memory mode; lower quality |
| MiniMind2 i16 | 58 MB | ~226 ms/token | reference; weak instruction following |
| MiniMind2 q8 | 29 MB | ~140–190 ms/token | rope comes from the container |
| MiniMind2 q4 | 21 MB | ~193–212 ms/token | memory mode; identity answers unreliable |
| TinyChat15M q8 | 24 MB | ~139–171 ms/token | runs, but output is not coherent; format probe only |
| stories15M i16 | 49 MB | ~167 ms/token | TinyStories completion; coherent |
| stories42M q8 | 58 MB | ~190 ms/token | TinyStories completion; coherent |
| stories15M q8 | 24 MB | ~148 ms/token | q8 conversion produces gibberish; use i16 |
| stories260K | 1 MB | ~85 ms/token | floor probe; prefill 5.4 s |

Pi numbers are single-run and noisy; see [performance.md](docs/performance.md).
TinyChat15M needs a working tokenizer/prompt pairing before it can be called a
chat model.

## Documentation

- [Engine quick start](engine/README.md)
- [Checkpoint container (MMCF)](docs/checkpoint-container.md)
- [Model formats](docs/model-formats.md)
- [Quantization and Q4M1](docs/quantization.md)
- [VC4 / engine internals](docs/engine-internals.md)
- [Prefix cache](docs/prefix-cache.md)
- [Performance](docs/performance.md)
- [Test hooks](docs/test-hooks.md)
- [Evaluation: model quality](docs/evaluation.md) — the multiple-choice, instruction-following and translation batteries
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

- **SmolLM2-135M is not supported.**  The weights in `models/smollm2-*` are
  source material, not a validated checkpoint; do not read them as support.
- **The engine loads MMCF containers only.**  A `.bin`/`.q8`/`.q4`/`.i16` file
  is a packer input; `engine/tools/mmcf_pack.py` converts it.
- Pi and laptop KV caches are **not byte-identical**; build the disk cache on
  the machine that will load it.
- `CmaFree` is real CMA occupancy. Reducing `gpu_mem` or `cma` does not create
  free CMA and can make pressure worse.
- Pi Zero with `SEQ=768` + packed q16 is tight (~2 MB `CmaFree` after load);
  test the exact model/cache combination.
- q8 KV is fluent but not quality-safe; q16 packed is the safe KV default.
- q4 is not bit-exact, and its arithmetic quality is prompt-dependent.
- The engine is split into `engine/src/` modules; see
  [engine-layout.md](docs/engine-layout.md) for the X-format and regression
  rules.

