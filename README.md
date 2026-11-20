# VC4 / GLES2 MiniMind3 inference

Practical, headless inference for small language models on a Raspberry Pi Zero
(BCM2835 / VC4 V3D 2.1) through GLES2/EGL. It is not a demo: it runs a
Qwen3-style 64M model, a native tool loop, an OpenAI-compatible HTTP server,
and a disk KV prefix cache.

[中文说明](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Vibe coding declaration

This project is developed in a **vibe-coding** workflow: the engine code, the
container toolchain, the test batteries and most of `docs/` are written by AI
coding agents (DeepSeek Harness running a team of agents) under human
direction, review and measurement.  Two consequences are worth knowing before
you build on it:

- **The measurements are real; the prose is generated.**  Every number in
  `docs/benchmarks.md` and `docs/evaluation.md` comes from a command that
  exists in this tree, and `engine/scripts/regression.sh` /
  `engine/gate/shader_hashes.sh` pin the numerics.  An explanation is not
  evidence: run the gate.
- **Review before you rely on it.**  Agent-written changes are reviewed and
  gated here, but this is a research/edge-inference stack, not a certified
  product.  There is no warranty beyond the MIT licence; if you need a
  guarantee, read the diff, run the suite on your own target and measure your
  own workload.

Contributions follow the same rule: `AGENTS.md` is what keeps the generated
parts honest -- cite evidence for a numeric change, update the document that
owns the fact, and never re-baseline a pin without a reason.

## What works

- **MiniMind3 64M** — 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV heads,
  head_dim 96, vocab 6400, RoPE theta 1e6, per-head Q/K RMSNorm.
- **MiniMind2 Small** — i16/q8/q4; Qwen-style chat, weaker instruction
  following than MiniMind3.
- **LFM2.5-230M** — the hybrid model: 8 depthwise-conv layers interleaved with
  6 attention layers, 1024 dim, 64402 vocab, the vendor's own ChatML-shaped
  template with tools and `<think>`.  Fused decode ~33 ms/token on the laptop
  and ~570 ms/token on a Pi Zero at `SEQ=256`; perplexity is within 9% of the
  float32 reference.  Page: [docs/lfm2.5-230m.md](docs/lfm2.5-230m.md).
- **MiniMind3-MoE** — the mixture-of-experts checkpoint (198M-A64M, top-1,
  every expert resident): 131 MB at q4 is what fits a Pi Zero, q8 does not.
  Container v3; the route is read back per layer, per token.
- **RWKV-7 0.1B** — the recurrent one (container v4): no attention and no KV
  cache, a CPU and a GPU layer loop that are not comparable by string.
  Page: [docs/rwkv7.md](docs/rwkv7.md).
- **BitCPM-CANN-0.5B** — the ternary one: a 2-bit body (`--quant t3`, 167.5 MB
  for 0.5B weights, needs an ESSL 3.00 context) and a q8 body that is lossless
  because 168 of its 169 matrices are per-row `{0, ±s}`.
  Page: [docs/bitcpm.md](docs/bitcpm.md).
- **SmolLM2-135M** — Llama-shaped, and the model that needs `RMS_STORE=f32`
  (see the caveats below).  Page: [docs/smollm2-135m.md](docs/smollm2-135m.md).
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
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Pi Zero

Build on the Pi, separately from the run:

```bash
cd engine
make -B armv6
```

Run:

```bash
cd engine
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Offline q4

```bash
# One step: the packer quantizes from safetensors, no intermediate file.
python3 engine/tools/mmcf_pack.py --in models/minimind3 --quant q4 --seq 256 \
  --tok models/minimind3/minimind3.tok --out models/minimind3/minimind3.q4.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja --verify

cd engine
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

Offline Q4M1 is 45 MB against 66 MB for q8: it saves weight memory, and it is
not bit-exact or quality-neutral — [benchmarks.md](docs/benchmarks.md) pins its
PPL separately from q8's.  Use the F16 source; q8 -> q4 is a second quantization
and degrades.

### OpenAI-compatible server

```bash
# From the repository root: the cache builder runs the engine with this
# directory as its cwd, so its paths are repository-relative, and the server's
# worker inherits the caller's cwd.
python3 server/build_prefix_cache.py build --model models/minimind3/minimind3.q8.mmcf \
  --seq 768 --kv-pack 16 --out pfx_tools_pi768.kv        # writes pfx_tools_pi768.kv.json too
python3 server/server.py --seq 768 --kv-pack 16 --prefix-cache pfx_tools_pi768.kv
```

See [server/README.md](server/README.md) for endpoints, tools and cache
details.

## Repository map

| Path | Contents |
|---|---|
| [engine/](engine/) | GLES2 engine, the container toolchain, probes, gate, scripts |
| [models/](models/) | central model store (gitignored) |
| [minimind3/](minimind3/) | MiniMind3 source config, tokenizer source, reference |
| [server/](server/) | OpenAI-compatible Python server |
| [docs/](docs/) | deep dives: container, formats, internals, quantization, performance |

## Model sizes and speed

Every row is an `.mmcf` container; the legacy file it was packed from stays
next to it in `models/` as a re-pack source.

| Model | Size | Pi decode (latest per model) | Notes |
|---|---:|---:|---|
| LFM2.5-230M q4 (QAD) | 169 MB | ~566 ms/token | hybrid conv+attention, `SEQ=256`; the board's tightest fit |
| MiniMind3 q8 | 66 MB | ~190 ms/token | default Chinese chat |
| MiniMind3 q4 | 45 MB | ~186 ms/token | memory mode; lower quality, and the faster path since the flat q4 gate |
| MiniMind2 i16 | 58 MB | ~216 ms/token | reference; the default q16 decode (`Q16_DECODE=exact` gives ~327) |
| MiniMind2 q8 | 29 MB | ~124 ms/token | rope comes from the container |
| MiniMind2 q4 | 21 MB | ~154 ms/token | memory mode; identity answers unreliable |
| TinyChat15M q8 | 24 MB | ~115 ms/token | llama-2 chat format; the container carries its own `[INST]` template; 15M quality |
| stories15M i16 | 49 MB | not on the board | TinyStories completion; coherent |
| stories42M q8 | 58 MB | ~160 ms/token | TinyStories completion; coherent |
| stories15M q8 | 24 MB | ~116 ms/token | TinyStories completion; coherent |

The Pi column is the latest measured row per model: the 2026-11-13 campaign on
HEAD `71cb1e0` (`GPU_LAYER=1 KV=q16`, greedy, `--gen 16`, a discarded warm-up
plus three timed runs, median), plus the rows re-measured since (MiniMind3
q4/q8 and LFM2.5 on 2026-11-16).  Decode only.  Rows marked "not on the board"
keep their last number in [benchmarks.md](docs/benchmarks.md), which owns the
matrix — it also carries the containers this table does not list (MoE, RWKV-7,
BitCPM, SmolLM2), the protocol, the prefill ranges and the q16-decode switch.
Prefill moves with the board's SD card and background load; do not quote it as a
constant.

## Documentation

- [Engine quick start](engine/README.md)
- [Checkpoint container (MMCF)](docs/checkpoint-container.md)
- [Model formats](docs/model-formats.md)
- [Quantization and Q4M1](docs/quantization.md)
- [VC4 / engine internals](docs/engine-internals.md)
- [Prefix cache](docs/prefix-cache.md)
- [Performance](docs/benchmarks.md)
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

- **SmolLM2-135M needs `RMS_STORE=f32`.**  Its residual reaches 2.7e4 and the
  default 24-bit rms store pins at 2048, after which the reply comes back
  *empty*; `models/smollm2-135m/smollm2-135m.q8q24.mmcf` is the container.  The
  board and the laptop disagree on its logits and on its greedy answer, and the
  cause is still open — [docs/smollm2-135m.md](docs/smollm2-135m.md).
- **The engine loads MMCF containers only.**  A `.bin`/`.q8`/`.q4`/`.i16` file
  is a packer input; `engine/tools/mmcf_pack.py` converts it.
- Pi and laptop KV caches are **not byte-identical**; build the disk cache on
  the machine that will load it.
- `CmaTotal` is the hard CMA ceiling; `CmaFree` is **not** an occupancy figure
  and not a budget — it reports what a caller can *probably* still get, and its
  deltas are noise.  Reducing `cma`/`gpu_mem` does not create CMA.  Near the
  ceiling, attempt the load and let it fail; do not refuse it on arithmetic.
- Pi Zero with `SEQ=768` + packed q16 is tight (~2 MB `CmaFree` after load);
  test the exact model/cache combination.
- q8 KV is fluent but not quality-safe; q16 packed is the safe KV default.
- q4 is not bit-exact, and its arithmetic quality is prompt-dependent.
- The engine is split into `engine/src/` modules; see
  [engine-layout.md](docs/engine-layout.md) for the X-format and regression
  rules.

