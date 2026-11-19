# Documentation

Deep dives live here; the component READMEs carry quick starts.

## Start here

- [Project README](../README.md) — what this is, quick start, components.
- [Chinese overview](../README.zh-CN.md) — 中文项目说明。
- [Engine quick start](../engine/README.md) — build/run on laptop and Pi.
- [引擎快速开始（中文）](../engine/README.zh-CN.md)
- [Server quick start](../server/README.md) — OpenAI-compatible HTTP server.
- [服务器快速开始（中文）](../server/README.zh-CN.md)

## The container and its encoding

| Document | Contents |
|---|---|
| [checkpoint-container.md](checkpoint-container.md) | **the MMCF container**: header, metadata TLV, tensor table, loader rules, what it enforces; the t3 body and the v4 (RWKV-7) layout |
| [model-formats.md](model-formats.md) | the legacy layouts the packer consumes, the models this repo runs, and the `templates/` inventory |
| [quantization.md](quantization.md) | q4/q6 experiments, offline Q4M1, packed KV, quality caveats |

## Running the engine

| Document | Contents |
|---|---|
| [engine-layout.md](engine-layout.md) | source layout, X formats, regression rule |
| [engine-internals.md](engine-internals.md) | VC4 findings, FFN split, GPU attention, persistent layers, Top-K |
| [test-hooks.md](test-hooks.md) | environment switches: probes, dumps, A/B hooks — generated from `engine/src/switches.cpp` |
| [prefix-cache.md](prefix-cache.md) | in-memory and disk KV prefix cache, packed cache |
| [long-context.md](long-context.md) | why the Pi stops at 2048 tokens, and what passing it would cost |
| [deterministic-tools.md](deterministic-tools.md) | model-free tool routing, argument validation, cache interaction |

## Measured numbers and quality

| Document | Contents |
|---|---|
| [benchmarks.md](benchmarks.md) | **timings and model-quality numbers**, the protocol that produced them, and each row's conditions — a reference table, not a log |
| [evaluation.md](evaluation.md) | **model quality, measured**: the multiple-choice battery (letter logprob, calibrated, format compliance), the instruction-following battery and the translation battery (chrF) |

## Models

| Document | Contents |
|---|---|
| [lfm2.5-230m.md](lfm2.5-230m.md) | **LFM2.5-230M, the hybrid (conv + attention) model**: containers, run commands, quality numbers, what is still broken |
| [bitcpm.md](bitcpm.md) | **BitCPM-CANN-0.5B, the ternary model**: the container, its rope table and tokenizer prefix, `k = 1/scale_emb`, `RMS_STORE=f32`, the quality numbers, what is not done |
| [smollm2-135m.md](smollm2-135m.md) | **SmolLM2-135M, the model that needs `RMS_STORE=f32`**: why the rmsnorm accumulator was the limit, the per-switch isolation, the open Pi-vs-Xe divergence |
| [rwkv7.md](rwkv7.md) | **RWKV-7 (Goose), the recurrent one**: no KV cache and no attention, the v4 container's key 16/17, the third layer loop, the per-machine numbers, the chat-marker traps, and why the speed work stopped |

## Building for the Pi, and VC4 hardware

| Document | Contents |
|---|---|
| [cross-build.md](cross-build.md) | **how to build for the Pi Zero**: the ARMv6 toolchain and sysroot recipe, why a missing `-B` fails silently, how the build verifies the artifact |
| [vc4-glsl-costs.md](vc4-glsl-costs.md) | what each GLSL function costs on VC4, read from Mesa 26.2.2 — and the six that abort the process |
| [vc4-mesa-debug-build.md](vc4-mesa-debug-build.md) | cross-building Mesa 26.2.2 for the Pi on 107, the soname/version traps, what the instrumented driver showed |

## Decisions and records (dated, not maintained)

These answer "why did we choose that" for work that is finished or still in
progress.  They are written at a point in time and are not updated as the engine
moves; the code and `AGENTS.md` are the present state.

| Document | Contents |
|---|---|
| [translation-decision-model-fit.zh-CN.md](translation-decision-model-fit.zh-CN.md) | 20–150M 翻译/决策模型选型：Jev-like 决策、适配计划；**唯一记录"评估后决定不做"的测量**，所以留着 |
| [swap-drift.md](swap-drift.md) | `/model` has been wrong twice: what a swap owns, and the order of evidence that found it |
| [vc4-1536-repro.md](vc4-1536-repro.md) | the runbook for two open VC4 questions: is a single 1536 draw wrong, and what changing the vocabulary grid costs |
| [future-optimization.md](future-optimization.md) | prioritized future work, measured dead ends, explicit non-goals |

## Working on a numerics question?

`AGENTS.md` has the "Debugging protocol" table that maps a question to the tool
that answers it — `mmcf_dump.py`, `ref_forward.py`, `diff_paths.sh` /
`dump_diff.awk`, `check_att.py`, `cmp_logits.py`.

## Conventions

- Numbers are measured on the machine named in the document; they are not
  promises for another Pi or driver, and `benchmarks.md` says which comparisons
  are legal.
- The Pi and laptop do **not** produce byte-identical KV caches.  Build a disk
  cache on the machine that will load it.
- `CmaTotal` is the hard CMA ceiling; `CmaFree` is not an occupancy figure and
  not a budget — near the limit, attempt the load.  The invariant is in
  `AGENTS.md`.

How a page here is written — where it goes, its sections in order, its length,
and the files a new model has to touch — is in [`AGENTS.md`](AGENTS.md), which is
for agents rather than for readers.
