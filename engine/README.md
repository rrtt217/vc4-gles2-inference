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
MODEL=$PWD/../models/llama2c/stories15M.i16.mmcf \
./engine_mm3 --prompt "One day, Lily" --gen 8
```

MiniMind3 q8, GPU layers and packed KV:

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "请写一段关于春天的短文。" --gen 64 --greedy
```

MiniMind3 offline q4 (45 MB, recommended q4 path):

```bash
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

Do not set `Q4_WEIGHTS` when the model file is already Q4M1.

### MiniMind2 Small

MiniMind2 uses the Qwen chat template.  The legacy Q8M1 file does **not** carry
RoPE metadata, so q8 needs the environment override:

```bash
\
MODEL=../models/llama2c/minimind2-small.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

The I16 and Q4M1 files carry theta/half in their headers.  Q4 is a memory
mode and is noticeably weaker on identity questions:

```bash
MODEL=../models/llama2c/minimind2-small.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

### TinyChat15M

TinyChat15M.q8 loads and runs, but the repository tokenizer/prompt pairing
currently produces incoherent text.  Treat it as a format/probe model, not as
a chat model.

### TinyStories 15M/42M

These are base completion models, not chat models.  Use a plain prompt and
no `--chat`:

```bash
MODEL=../models/llama2c/stories42M.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 \
  --prompt "Once upon a time" --gen 64 --greedy
```

On the Pi, stories15M i16 is coherent at ~167 ms/token, the legacy
`stories15M.bin` at ~161 ms/token, and stories42M q8 at ~190 ms/token.
`stories15M.q8` currently produces gibberish in the repo's conversion; use the
`.bin` or i16 file.  stories42M i16 is too large for the current CMA budget.  The
260K legacy file (`stories260K.bin` + `tok512.bin`) is coherent after the
legacy-I16 fix at ~85 ms/token, but still has 5.4 s prefill; treat it as a floor
probe.

Legacy fp32 checkpoints now load as I16 by default.  Set `LEGACY_Q8=1` only when
the extra memory of I16 does not fit (for example a 42M `.bin`); small
TinyStories models lose coherence in that mode.

## MiniMind3 64M

64M Qwen3-style dense model: 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV
heads, head_dim 96, vocab 6400, RoPE theta 1e6, tied embedding, per-head
Q/K RMSNorm.

`` uses ChatML, an empty `<think>` block, and the same
short English default system prompt as the other chat templates
(`You are a helpful assistant.`). Override it with `--system`. Add `TIMING=1`
for prefill/decode timing.

Pi Zero, 16 generated tokens (2026-11-02 run; Pi also runs a Minecraft
server, so expect noise):

| model | prefill | decode |
|---|---:|---:|
| MiniMind3 q8 | 20.5 s | **238 ms/token** |
| MiniMind3 q4 | 33.0 s | **381 ms/token** |
| MiniMind2 i16 | 9.2 s | 226 ms/token |
| MiniMind2 q8 | 8.4 s | 140–190 ms/token |
| MiniMind2 q4 | 9.7 s | 193–212 ms/token |
| TinyChat15M q8 | 4.8 s | 139–171 ms/token (not coherent) |

The historical 520.9 ms/token and 42.7 s prefill figures were measured before
the q16 attention and q4 paired-residual work; use the table above.

The FFN hidden width 2432 exceeds the VC4 2048-pixel FBO limit.
`Model::split_ffn()` splits W1/W3 rows and W2 columns into 1216-wide chunks;
the second W2 matvec adds into the first residual.

## Thinking and tools

```bash
# explicit thinking
...  --thinking --prompt "1+1等于几？" --gen 64

# tool definitions
...  --tools-file minimind3/tools.json \
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
| `MODEL` | checkpoint path (one MMCF container carries the tokenizer, template and rope) |
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
| `RES_PACK=0` | disable the default packed residual stream (q8/q4) |
| `ACT8=2` | FFN hidden state in I8_4, q8 weights only; opt-in |
| `ACT8=1` | all matvec inputs in I8_4, q8 weights only; opt-in |

See [test hooks](../docs/test-hooks.md) for the full list.

## Perplexity

Teacher-forced perplexity is the quality gate for lossy activation formats:

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3  --ppl data/ppl_test.txt
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
| [engine-layout.md](../docs/engine-layout.md) | source modules, X formats, regression rule |

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
