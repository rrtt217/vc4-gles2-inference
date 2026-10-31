# Engine source layout and split decision

## Current state

`engine/engine.cpp` is one 3175-line translation unit (~148 KB), containing:

| Area | Approx. lines | Coupling |
|---|---:|---|
| EGL/GLES helpers, shaders, GPU passes | ~1500 | high: GL state, FBOs, shader cache |
| Tokenizer | ~120 | low |
| Model loaders / formats | ~430 | medium |
| Session / forward | ~380 | high: GPU + model |
| CLI, sampling, tools, worker | ~700 | low-medium |

It builds in one command on both x86 and Pi, and there is no unit-test
harness. That is why the source has stayed a single file.

## Should it be split?

**Yes, eventually; not as a cosmetic refactor right now.**

The file is large because the GPU code is genuinely stateful, not because it
is disorganized. A blind split of `GPU` across files would move GL state
ownership and shader-program caching across translation units, which is the
highest-risk change in this project. The q4 residual-output bug showed how
easy it is to break a format contract that is only visible at runtime.

The q4 format and FFN slicing are now implemented and tested; that is a good
point to freeze behavior before moving code.

## Recommended staged split

### Stage 1 — low risk

1. Move GLSL shader string builders to `engine/shaders.cpp` /
   `engine/shaders.h`, keeping the `GPU` class as the only caller.
2. Move tokenizer to `engine/tokenizer.h/.cpp` — pure CPU, easy to test.
3. Move tool JSON/arithmetic helpers to `engine/tools.h/.cpp` — pure CPU.
4. Keep `engine.cpp` as the CLI/session/GPU orchestration unit.

This reduces the file by ~700 lines without changing GPU behavior.

### Stage 2 — format layer

Move model format readers/writers to:

```text
engine/model.h        Mat, WeightType, Model metadata
engine/model_q8.cpp   Q8M1/Q8M2 + I16M1/I16M2 readers
engine/model_q4.cpp   Q4M1 reader
```

Add round-trip tests for the q4 header, block scale/min, and FFN slicing
before moving anything.

### Stage 3 — only if needed

Split `GPU` into:
- `gpu_context` (EGL, textures, targets, programs);
- `gpu_matvec` (q4/q8/q16 shaders);
- `gpu_attention` (RoPE, KV cache, scores, V).

Do this only with a VC4 regression suite (text outputs, logits, cache
equality), because GL state and finish points are subtle.

## What not to do

- Do not split `GPU` and `Session` into separate libraries just to reduce
  line count.
- Do not move shader strings into many files before adding a test that
  compiles all programs.
- Do not change the q4 add-output format (q16) while refactoring; the
  downstream RMSNorm/attention path depends on it.
- Do not create a header-only split that still includes one giant
  `engine.cpp`; that adds include churn without isolation.

## Decision

Keep the single translation unit for the current release. Do Stage 1 when
the next feature touches tokenizer/tools/shaders; do Stage 2 when a new model
format is added. A full `GPU` split is justified only after a regression
harness exists.
