# GLES2 inference engine

A small GLES2/EGL inference engine for llama2.c-family and Qwen3-style
checkpoints. It runs headless on Raspberry Pi Zero (VC4 V3D 2.1) and on
desktop GLES2 drivers.

This README is the quick start. Deep notes are in [../docs/](../docs/):

- [model formats](../docs/model-formats.md)
- [VC4 / engine internals](../docs/engine-internals.md)
- [quantization and Q4M1](../docs/quantization.md)
- [performance](../docs/performance.md)
- [test hooks](../docs/test-hooks.md)
- [source layout](../docs/engine-layout.md)

## Source layout

| Path | Purpose |
|---|---|
| `src/` | EGL/GLES2 engine, model, session, tokenizer, CLI |
| `src/xformat.*` | activation layout and GLSL X macros |
| `src/shaders*.cpp` | GLSL source builders |
| `probes/` | standalone VC4 probes, not linked into the engine |
| `quantize/` | F16/I16/q8/q4 converters |
| `scripts/` | benchmark and regression helpers |
| `Makefile` | local and Pi build flags |

## Build

Local:

```bash
cd engine
make engine_mm3
```

Pi Zero, build separately from the run (8–10 minutes; never run two builds):

```bash
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

Do not build under a short timeout: a stale binary makes results look
nondeterministic.

### Cross build on VLAB

VLAB has `arm-linux-gnueabihf-g++-12` and a Raspbian trixie rootfs at
`$HOME/raspi-root4`. Build the Pi binary there (about 25 seconds):

```bash
make -B pi CROSS_SYSROOT=$HOME/raspi-root4
```

The target uses `--sysroot` plus `-B` for the sysroot's own GCC startfiles;
the sysroot is a complete Raspbian root filesystem (ARMv6, VFPv2), not a
hand-picked set of libraries. The output is `engine_mm3_pi`; copy it to the Pi
only after checking `readelf -A` reports `Tag_CPU_arch: v6`.

## Run

Smallest example:

```bash
MODEL=$PWD/../models/llama2c/stories15M.i16 \
TOKENIZER=../models/llama2c/tokenizer.bin \
./engine --prompt "One day, Lily" --gen 8
```

MiniMind3 q8, GPU layers and packed KV:

```bash
MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "请写一段关于春天的短文。" --gen 64 --greedy
```

MiniMind3 offline q4 (45 MB, recommended q4 path):

```bash
MODEL=../models/minimind3/minimind3.q4 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

Do not set `Q4_WEIGHTS` when the model file is already Q4M1.

## MiniMind3 64M

64M Qwen3-style dense model: 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV
heads, head_dim 96, vocab 6400, RoPE theta 1e6, tied embedding, per-head
Q/K RMSNorm.

`--template minimind3` uses ChatML, an empty `<think>` block, and the
MiniMind system prompt. Add `TIMING=1` for prefill/decode timing.

Pi Zero q8, 82-token prefill + 64 generated tokens:

| item | value |
|---|---:|
| prefill | 42.7 s |
| decode | **520.9 ms/token** |
| CPU-attention path | 559.5 ms/token |
| q8 weights | 66 MB |

The FFN hidden width 2432 exceeds the VC4 2048-pixel FBO limit.
`Model::split_ffn()` splits W1/W3 rows and W2 columns into 1216-wide chunks;
the second W2 matvec adds into the first residual.

## Thinking and tools

```bash
# explicit thinking
... --template minimind3 --thinking --prompt "1+1等于几？" --gen 64

# tool definitions
... --template minimind3 --tools-file minimind3/tools.json \
  --prompt "现在几点了？" --gen 64 --greedy
```

Tool prompts are longer than 256 tokens; with tools configured the engine
raises the KV cache to 768 unless `SEQ` is set explicitly.

Built-in tools:

- `get_current_time` — IANA timezone;
- `calculate_math` — safe recursive-descent arithmetic (`+ - * / % ^`,
  parentheses, unary, Python-style `**`).

The native tool loop generates until `<|im_end|>`, parses all complete
`<tool_call>` blocks, executes them, appends one user turn with
`<tool_response>` blocks, and re-prefills. Up to 3 rounds.

Thinking mode often fails to close `</think>` within 64 tokens on the 64M
model; tool calls are much more reliable.

## Environment variables

| Variable | Meaning |
|---|---|
| `MODEL`, `TOKENIZER` | checkpoint and tokenizer paths |
| `SEQ` | KV cache length; tools auto-raise to 768 |
| `GPU_ATTN=1` | GPU RoPE + KV cache + attention |
| `GPU_LAYER=1` | hidden state, attention, RMSNorm, FFN on GPU |
| `KV_PACK=16` | packed q16 KV (two int16 per texel) |
| `KV_PACK=8` | packed q8 KV; quality is not default-safe |
| `KV_SCALE` | KV fixed-point scale |
| `Q4_WEIGHTS=1` | runtime q4 conversion (experiments only) |
| `KEEP_CPU_WEIGHTS=1` | keep CPU weights after upload (debug) |
| `TIMING=1` | print prefill/decode timing |
| `LAYER_PROFILE=1` | per-layer timing |
| `TOPK`, `TOPK_FAST` | candidate Top-K paths |
| `RES_PACK=1` | two q16 residual values/texel; neutral, opt-in |
| `ACT8=2` | FFN hidden state in I8_4, q8 weights only; opt-in |
| `ACT8=1` | all matvec inputs in I8_4, q8 weights only; opt-in |

See [test hooks](../docs/test-hooks.md) for the full list.

## Perplexity

Teacher-forced perplexity is the quality gate for lossy activation formats:

```bash
MODEL=../models/minimind3/minimind3.q8 TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --template minimind3 --ppl data/ppl_test.txt
```

Use the same corpus when comparing formats; `ACT8=1` and `ACT8=2` are not
default-safe on every corpus.

## Documentation map

| Document | Read it for |
|---|---|
| [model-formats.md](../docs/model-formats.md) | byte layouts |
| [quantization.md](../docs/quantization.md) | q4/q6, Q4M1, packed KV, quality |
| [engine-internals.md](../docs/engine-internals.md) | VC4 shader/FBO quirks |
| [performance.md](../docs/performance.md) | Pi/laptop numbers |
| [prefix-cache.md](../docs/prefix-cache.md) | worker and disk KV cache |
| [engine-layout.md](../docs/engine-layout.md) | whether to split the source |

## Server

The OpenAI-compatible Python server lives in [../server/](../server/):

```bash
python3 server.py --seq 768 --kv-pack 16 \
  --prefix-cache prefix_tools_pack16_768.kv
```

See [server/README.md](../server/README.md).

## Shutdown

`SIGINT`/`SIGTERM` set a flag instead of tearing down EGL inside the signal
handler. The current GPU call/layer/token finishes, then `glFinish()` and
normal EGL teardown run. In chat mode Ctrl+C stops the current turn and
returns to the prompt; Ctrl+C at the prompt exits.
