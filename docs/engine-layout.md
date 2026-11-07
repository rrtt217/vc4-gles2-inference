# Engine source layout

The engine is no longer one translation unit. The split is intentionally by
responsibility, not by line count:

| File | Responsibility |
|---|---|
| `src/common.h/.cpp` | shared types (`Mat`, `Target`), q16 helpers, signals, file helpers |
| `src/xformat.h/.cpp` | **the only activation-layout definition** (I16_1, I16_2, I8_4, and the P2 pair-of-positions target) and its GLSL macros |
| `src/shaders.h/.cpp` | matvec / FFN / RMS shader builders |
| `src/shaders_attn.cpp` | attention, KV and logits shader builders |
| `src/gpu.h/.cpp` | EGL context, textures, targets, program cache, GPU passes |
| `src/model.h/.cpp` | model metadata, the MMCF reader, q4/q8/i16 conversion, FFN split |
| `src/session.h/.cpp` | KV cache, RoPE, attention, transformer forward, prefix cache |
| `src/tokenizer.h/.cpp` | llama2.c tokenizer and byte-level BPE, embedded-blob loader |
| `src/sampling.h/.cpp` | sampling, penalties, top-p/min-p |
| `src/chattmpl.h/.cpp` | renders the container's Jinja chat template; returns the prompt split into template text and caller text.  Also parses the worker's message request (column 11) |
| `src/tools.h/.cpp` | tool JSON, restricted arithmetic, built-in handlers |
| `src/engine.cpp` | CLI, worker protocol, chat loop, capability/template handling |

The sibling directories are **not** linked into `engine_mm3`:

| Path | Runs on | Contents |
|---|---|---|
| `engine/third_party/jinja/` | engine | vendored llama.cpp Jinja engine (MIT); see its README for the three local edits |
| `engine/tools/` | host python | container packer/inspector, numpy reference forwards |
| `engine/scripts/` | host shell | regression gate, CPU-vs-GPU diff, attention/logits oracles, benchmarks |
| `engine/probes/` | target GLES2 | standalone VC4 feature and throughput probes |

`engine/tools/` and `engine/scripts/` are where a numerics question should start;
`AGENTS.md` maps a question to the tool.  `engine/probes/` predates them and an-
swers narrower questions (what does a draw cost, does the driver accept a 2432-
row texture, does `roundEven` work).

## X formats

Every activation texture is RGBA8. `XFormat` decides the byte layout and the
quantization scale:

| X format | values / texel | storage |
|---|---:|---|
| `I16_1` | 1 | q16 in `.rg` |
| `I16_2` | 2 | q16 pair: `.rg`, `.ba` |
| `I8_4` | 4 | signed int8 in RGBA, 1/16 scale |

Producers call `XST1/XST2/XST4`; consumers call `X1/X2/X4`. Hot matvec kernels
use the q16-domain variants `X1Q/X2Q/X4Q` and `R1Q` so the inner loop does not
pay a divide per activation. The residual stream has its own `R1/R2/R1Q`
macros and is never assumed to have the activation layout.

This is the fix for the previous ACT8 failures: adding a format changes
`xformat.cpp`, not seven shader strings. `ACT8=1` selects `I8_4` for all
matvec inputs, `ACT8=2` only for the FFN hidden state; both are opt-in.

The prompt is a separate matter from the activation layouts: `chattmpl.cpp`
renders the template and hands back `PromptPart`s that carry an `is_input`
flag, and `Tokenizer::encode_parts` matches special tokens only outside those
parts.  Template text may produce `<|im_end|>`; a user message may not.

The **P2** layout is a target layout, not an `XFormat`: in the two-position
prefill pair, one texel holds value *k* of position A in `.rg` and value *k* of
position B in `.ba`, so `x_per_texel()` does not apply and the width is
`x_pair2_texels(n) == n`.

## Build

```bash
make -C engine -B engine_mm3 CXXFLAGS='-O2 -std=c++17'
make -C engine -B pi CROSS_SYSROOT=<sysroot>       # Pi Zero, ARMv6
make -C engine -B probes/draw_probe_pi             # a probe
```

Probes have their own targets under `engine/probes/`; the Pi ones take the same
`CROSS_*` variables as `pi`.

## Regression rule

A change to GPU state, X formats or shader text must preserve:

1. q8 greedy output for the canonical prompt;
2. q4 greedy output for the canonical prompt;
3. `--ppl` on the fixed corpus, within the recorded tolerance.

Then `engine/scripts/regression.sh`, which is the gate — determinism, a second
prompt's behaviour, the per-machine pins, and PPL.  Only after those pass should
`ACT8` or performance numbers be compared.

A same-path A/B (one flag, one kernel) is what decides a numerics question; a
CPU-vs-GPU stage delta from `diff_paths.sh` names a lead, not a defect, because
the CPU path is f32 that round-trips q16 only at the dump.
