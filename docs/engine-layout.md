# Engine source layout

The engine is no longer one translation unit. The split is intentionally by
responsibility, not by line count:

| File | Responsibility |
|---|---|
| `src/common.h/.cpp` | shared types, q16 helpers, signals, file helpers |
| `src/xformat.h/.cpp` | **the only activation-layout definition** (I16_1, I16_2, I8_4) and its GLSL macros |
| `src/shaders.h/.cpp` | matvec / FFN / RMS shader builders |
| `src/shaders_attn.cpp` | attention, KV and logits shader builders |
| `src/gpu.h/.cpp` | EGL context, textures, targets, program cache, GPU passes |
| `src/model.h/.cpp` | model metadata, format readers, q4/q8/i16 conversion |
| `src/session.h/.cpp` | KV cache, RoPE, attention, transformer forward |
| `src/tokenizer.h/.cpp` | llama2.c tokenizer and byte-level BPE |
| `src/sampling.h/.cpp` | sampling, penalties, top-p/min-p |
| `src/tools.h/.cpp` | tool JSON, restricted arithmetic, built-in handlers |
| `src/engine.cpp` | CLI, worker protocol, chat loop |

`engine/probes/`, `engine/quantize/` and `engine/scripts/` are separate:
they are not linked into `engine_mm3`.

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

## Build

```bash
make -C engine -B engine_mm3 CXXFLAGS='-O2 -std=c++17'
```

The Pi build uses the ARMv6 flags in the Makefile/AGENTS.md. Probes have their
own targets under `engine/probes/`.

## Regression rule

A change to GPU state, X formats or shader text must preserve:

1. q8 greedy output for the canonical prompt;
2. q4 greedy output for the canonical prompt;
3. `--ppl` on the fixed corpus, within the recorded tolerance.

Only after those pass should `ACT8` or performance numbers be compared.
