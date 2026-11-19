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
| [rwkv7-plan.zh-CN.md](rwkv7-plan.zh-CN.md) | **RWKV-7 实施计划与逐算子清单**（目标文档，进行中）：28 个算子 + 20 项配套的数学/形状/现状/归属/阶段、数值格式决策点、验收门 |
| [rwkv7-assessment.zh-CN.md](rwkv7-assessment.zh-CN.md) | RWKV-7（Goose）支持评估：算子清单、容器/分词器/状态差距、Pi 与 Xe 的性能内存估算、分阶段工作量 |
| [lfm2.5-230m-assessment.zh-CN.md](lfm2.5-230m-assessment.zh-CN.md) | LFM2.5-230M 的打包与可行性评估：架构事实、残差/激活幅值实测、官方 GGUF 配方、工作量与风险 |
| [bitcpm-assessment.zh-CN.md](bitcpm-assessment.zh-CN.md) | BitCPM-CANN-0.5B（MiniCPM4 的三值 QAT）支持评估：三值权重的实测结构、两个缺口、容器大小与 Pi 预算 |
| [translation-decision-model-fit.zh-CN.md](translation-decision-model-fit.zh-CN.md) | 20–150M 翻译/决策模型选型：Jev-like 决策、适配计划 |
| [swap-drift.md](swap-drift.md) | `/model` has been wrong twice: what a swap owns, and the order of evidence that found it |
| [vc4-1536-repro.md](vc4-1536-repro.md) | the runbook for two open VC4 questions: is a single 1536 draw wrong, and what changing the vocabulary grid costs |
| [future-optimization.md](future-optimization.md) | prioritized future work, measured dead ends, explicit non-goals |

## Working on a numerics question?

`AGENTS.md` has the "Debugging protocol" table that maps a question to the tool
that answers it — `mmcf_dump.py`, `ref_forward.py`, `diff_paths.sh` /
`dump_diff.awk`, `check_att.py`, `cmp_logits.py`.

## Conventions

- **This directory is flat on purpose.**  `gate/docs_check.sh` and
  `gate/check_docs_links.py` glob `docs/*.md`, so a subdirectory would silently
  fall out of the link and path checks.  Grouping lives in the sections above;
  if a split ever earns its cost, those two globs change in the same commit.
- Numbers are measured on the machine named in the document; they are not
  promises for every Pi or driver.
- `benchmarks.md` is a **reference table, not a log**: a new measurement
  replaces the row it supersedes and names its commit, and the record of how it
  moved is Git.  The 120 dated sections it used to carry are at
  `git show e7a36ec^:docs/performance.md`.  Do not append to it.
- The Pi and laptop do **not** produce byte-identical KV caches. Build a disk
  cache on the machine that will load it.
- `CmaTotal` is the hard CMA ceiling; `CmaFree` is not an occupancy figure and
  not a budget — near the limit, attempt the load.  See the CMA invariant in
  `AGENTS.md`.
