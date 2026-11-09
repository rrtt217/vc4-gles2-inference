# GLES2 inference engine

A small GLES2/EGL inference engine for llama2.c-family, Llama-shaped and
Qwen3-shaped checkpoints. It runs headless on Raspberry Pi Zero (VC4 V3D 2.1)
and on desktop GLES2 drivers, and it reads exactly one file format: the
self-describing [MMCF container](../docs/checkpoint-container.md).

This README is the quick start. Deep notes are in [../docs/](../docs/):

- [checkpoint container](../docs/checkpoint-container.md)
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
| `tools/` | MMCF packer/inspector and numpy reference forwards (host python) |
| `third_party/jinja/` | vendored llama.cpp Jinja engine (MIT), renders the container's chat template |
| `scripts/` | regression gate, CPU-vs-GPU diff, attention/logits oracles, benchmarks |
| `probes/` | standalone VC4 probes, not linked into the engine |
| `Makefile` | local, Pi and probe build flags |

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

### Interactive chat

`--chat` without `--prompt` is a REPL: the model loads once and the prompt comes
back for the next line, which is what makes a 7-second Pi start worth waiting
for.

```bash
MODEL=../models/supra2-medium/medium.q8.q24.mmcf \
GPU_LAYER=1 KV_PACK=16 SEQ=512 ./engine_mm3 --chat --gen 64 --greedy
```

```
User: What is the capital of France?
Assistant: The capital of France is Bordeaux, ...
User: /help
```

| key | |
|---|---|
| Left / Right | move the cursor (UTF-8 aware, one character at a time) |
| Up / Down | previous lines, up to 32 of them |
| Home / End, Ctrl-A / Ctrl-E | start / end of line |
| Backspace, Delete, Ctrl-D | delete backwards, forwards, forwards |
| Ctrl-K / Ctrl-U / Ctrl-W | kill to end, kill to start, kill the word before |
| Ctrl-L | clear the screen and redraw |
| Ctrl-C | abandon the line, keep the session |
| Ctrl-D on an empty line | leave |

`/clear` forgets the conversation, `/exit` leaves, `/help` lists them.  When
stdin is a pipe (`echo "hi" | engine_mm3 --chat`) the editor is bypassed and the
old `getline` behaviour is used, so scripts and the worker protocol are
unaffected.

The REPL **keeps the conversation**: every turn re-renders the whole exchange
through the container's template, so the model sees the same prompt shape it was
trained on, system turn included, and `/clear` is the way to start over.  Only
the tokens that are new since the previous turn are prefilled - the KV cache is
reused up to the longest common prefix - so a long conversation costs the last
exchange, not all of it:

```
User: Hi.                                    prefill 0.372 s   (first turn, compiles)
User: My name is Alice.                      prefill 0.081 s
User: What is my name?  -> "My name is Alice." prefill 0.083 s
```

Before this, each turn carried only the current line *and dropped the system
turn*, which is why a recall question answered with an invented name.

The same chain covers `--tools`.  The tool loop keeps the conversation (the
assistant turn, its tool calls and the tool results are appended to the history
like the HTTP server does), and every model call reuses the KV rows before the
longest common token prefix.  A tool round-trip therefore prefills only the tool
response and the template markers; the system + tools block and the exchange
before it are already in the cache.  A second user turn after a tool round-trip
costs the new line, not the whole tools preamble:

```
What is the system status?   round 0  prefill 233   (first turn, compiles)
                             round 1  prefill 101   (the tool response)
What is the temperature?     round 0  prefill  24   (the new line)
                             round 1  prefill  99   (the tool response)
```

A token can carry only the first bytes of a multi-byte character.  The REPL
holds that tail until the character is complete, or flushes U+FFFD when
generation stops mid-character, so the re-rendered history is always valid
UTF-8.  `CHAT_CACHE=1` prints the `cached=` / `prefill=` split of each model
call.

### MiniMind2 Small

MiniMind2 uses the Qwen chat template (template kind 1 in the container).  The
source `Q8M1` file carries no RoPE metadata, so the packer needs `--theta
1000000 --rope-half` once; after that the container states it and no
environment override is involved:

```bash
MODEL=../models/llama2c/minimind2-small.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

Q4 is a memory mode and is noticeably weaker on identity questions:

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

On the Pi, stories15M i16 is coherent at ~167 ms/token and stories42M q8 at
~190 ms/token.  `stories15M.q8` produces gibberish in this repo's conversion;
use the i16 container.  stories42M i16 is too large for the current CMA budget.
The 260K model is a floor probe: ~85 ms/token, but 5.4 s prefill, because the
attention passes walk the full 512-position grid.

These come from llama2.c fp32 `.bin` sources, which the packer accepts
directly (`--in .../stories15M.bin --quant i16`).  There is no `LEGACY_Q8=1` path
anymore: the engine has no legacy reader.

## Converting a checkpoint

The engine loads only `.mmcf`.  A `.bin`, `.i16`, `.q8` or `.q4` file is an
input to the packer:

```bash
# One step from safetensors: the packer quantizes, so no .i16/.q8/.q4 file is
# written.  --quant picks the precision, --seq the KV length.
# --chat-template stores the model's own Jinja template (metadata key 9), so the
# prompt stops being a C++ build option; without it the container falls back to
# --template's kind.
python3 tools/mmcf_pack.py --in ../models/<model> --quant q4 --seq 512 \
  --tok ../models/<model>/<model>.tok \
  --out ../models/<model>/<model>.q4.mmcf \
  --name <model> --template chatml --chat \
  --chat-template ../models/<model>/tokenizer_config.json --verify

# what does the engine actually read from it?
python3 tools/mmcf_dump.py ../models/<model>/<model>.q4.mmcf --check
```

The packer prints the residual peak per layer and refuses a model that reaches
more than 4x the engine's q16 rail (128); `--allow-over-rail` overrides,
`--no-rail-check` skips the 2 s calibration.  See the residual-rail note in
[model-formats.md](../docs/model-formats.md).

The `--tok` blob comes from the model's own `tokenizer.json`:

```bash
python3 tools/make_tok.py --input ../models/<model> --output ../models/<model>/<model>.tok
```

It handles ByteLevel (`Ġ`) and SentencePiece (`▁`) vocabularies, and lays the
chat markers that sit *above* the BPE range onto their ids — a blob built from
`model["vocab"]` alone ends at the wrong id and the engine splits
`<|im_start|>` into characters.

A llama2.c `.bin` works the same way (`--in ../models/llama2c/stories15M.bin
--quant i16 --tok ... --template plain --chat`).  A legacy checkpoint is still
accepted, and then `--theta` and `--rope-half` are required for a `Q8M1`/`I16M1`
source, because that header carries no rope at all and the packer refuses to
guess.  Templates:
`plain` (0), `chatml`/`qwen` (1), `minimind3` (2), `tinybrainbot` (3).  The
tokenizer is embedded, and the special-token ids are derived from it.

## Chat templates

A container may carry the model's own `chat_template` (the HuggingFace Jinja
text, metadata key 9 — the same string a GGUF file holds).  When it does, that
template is the prompt builder; `--template`'s kind is only the fallback for
containers written before the key existed.  A template the engine cannot parse
is refused at load with the parser's message, never silently replaced.

```bash
python3 tools/mmcf_dump.py ../models/minimind3/minimind3.q8.mmcf | grep -i 'chat tmpl'
CHAT_DUMP=1 ./engine_mm3 --chat --prompt hi --gen 1 --greedy 2>&1 | grep PROMPT
```

`CHAT_DUMP=1` prints the rendered prompt one segment per line, tagged
`PROMPT[tmpl]` or `PROMPT[in]`.  The `[in]` segments are text that came from a
message, and the tokenizer matches special tokens only **outside** them: a user
message containing `<|im_end|>` is tokenised as ordinary text, so it cannot
forge a turn boundary.  That is a behaviour change from the kind-based path,
which built one string and matched specials everywhere.

Tools still take the legacy path (`run_tool_turn`), so `--tools` remains
MiniMind3-only until that loop is message-based; a template that renders `tools`
is already able to describe them, but the loop's continuation is still ChatML.

## MiniMind3 64M

64M Qwen3-style dense model: 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV
heads, head_dim 96, vocab 6400, RoPE theta 1e6, tied embedding, per-head
Q/K RMSNorm.

Its container carries template kind 2: ChatML, an empty `<think>` block, and
the same short English default system prompt as the other chat templates
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
| `MODEL` | checkpoint path.  One MMCF container carries the tokenizer, template, rope and special-token ids, so there is nothing else to set |
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
| `TOPK_FAST=1` | approximate candidate path; needs `--top-k-candidates` |
| `RES_PACK=0` | disable the default packed residual stream (q8/q4) |
| `ACT8=2` | FFN hidden state in I8_4, q8 weights only; opt-in |
| `ACT8=1` | all matvec inputs in I8_4, q8 weights only; opt-in |

See [test hooks](../docs/test-hooks.md) for the full list, including the dump
and probe switches (`STAGE_DUMP`, `RAW_DUMP`, `QKV_DUMP`, `ATT_DUMP`,
`ATTN_DUP`, `PASS_DUP`, `DRAWCOUNT`, ...).

Top-k is a sampler flag, not an environment switch: `--top-k N` keeps the N
most likely tokens before top-p (0 = off, the default), and
`--top-k-candidates` takes those k from the candidate pass instead of the full
logits.  The candidate path is approximate: it falls back to full logits when
the candidate mass is below `top_p` and does not guarantee the exact global
top-k.  Without `--top-k-candidates`, the CPU path selects with `nth_element`
and sorts only the k kept logits.

## Perplexity

Teacher-forced perplexity is the quality gate for lossy activation formats:

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3  --ppl data/ppl_test.txt
```

Use the same corpus when comparing formats; `ACT8=1` and `ACT8=2` are not
default-safe on every corpus.

## Diagnosing a wrong answer

The question decides the tool.  All of these live in this repository.

| Question | Tool |
|---|---|
| does the file say what I meant? | `tools/mmcf_dump.py <file>` |
| is the conversion faithful? | `tools/mmcf_pack.py --verify`, or `LOGITS_OUT=` + `scripts/cmp_logits.py` |
| is the source model sane? | `tools/ref_forward.py <hf-dir> "<prompt>"` |
| which stage differs CPU vs GPU? | `scripts/diff_paths.sh <file.mmcf>` + `scripts/dump_diff.awk` |
| is a row saturating? | `STAGE_DUMP=1` + `scripts/dump_stats.py` |
| is the attention block self-consistent? | `ATT_DUMP=<pos>` + `scripts/check_att.py` |
| is one pass the cost? | `ATTN_DUP`, `PASS_DUP`, `DRAWCOUNT`, `LAYER_PROFILE` |
| did numerics change? | `scripts/regression.sh` (PPL, not one string) |

`GPU_LAYER=0` is not a CPU reference for the matvecs: attention and FFN move to
the CPU, but every matvec still runs on the GPU.  A per-stage CPU-vs-GPU delta
therefore names a lead, not a defect — read the header of `scripts/diff_paths.sh`
before concluding anything from it.

## Documentation map

| Document | Read it for |
|---|---|
| [checkpoint-container.md](../docs/checkpoint-container.md) | the MMCF layout, metadata keys, capability bits |
| [model-formats.md](../docs/model-formats.md) | the legacy layouts the packer consumes |
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
