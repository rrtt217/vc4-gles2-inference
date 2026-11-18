# Documentation

Start here:

- [Project README](../README.md) — what this is, quick start, components.
- [Chinese overview](../README.zh-CN.md) — 中文项目说明。
- [Engine quick start](../engine/README.md) — build/run on laptop and Pi.
- [引擎快速开始（中文）](../engine/README.zh-CN.md)
- [Server quick start](../server/README.md) — OpenAI-compatible HTTP server.
- [服务器快速开始（中文）](../server/README.zh-CN.md)

Deep dives:

| Document | Contents |
|---|---|
| [model-formats.md](model-formats.md) | includes the `templates/` inventory: which template each shipped model carries |
| [checkpoint-container.md](checkpoint-container.md) | **the MMCF v2 container**: header, metadata TLV, tensor table, loader rules, what it enforces |
| [model-formats.md](model-formats.md) | the legacy layouts the packer consumes, and the models this repo runs |
| [engine-internals.md](engine-internals.md) | VC4 findings, FFN split, GPU attention, persistent layers, Top-K |
| [engine-layout.md](engine-layout.md) | source layout, X formats, regression rule |
| [cross-build.md](cross-build.md) | **how to build for the Pi Zero**: the ARMv6 toolchain and sysroot recipe, why a missing `-B` fails silently, and how the build verifies the artifact |
| [quantization.md](quantization.md) | q4/q6 experiments, offline Q4M1, packed KV, quality caveats |
| [benchmarks.md](benchmarks.md) | Pi/laptop timing tables, the chronological lab notebook, regression baselines |
| [swap-drift.md](swap-drift.md) | `/model` has been wrong twice: what a swap owns, and the order of evidence that found it |
| [long-context.md](long-context.md) | why the Pi stops at 2048 tokens, and what passing it would cost |
| [vc4-glsl-costs.md](vc4-glsl-costs.md) | what each GLSL function costs on VC4, read from Mesa 26.2.2 — and the six that abort the process |
| [test-hooks.md](test-hooks.md) | environment switches: probes, dumps, A/B hooks |
| [evaluation.md](evaluation.md) | **model quality, measured**: the multiple-choice battery (letter logprob, calibrated, format compliance), the instruction-following battery and the translation battery (chrF), with the 2026-11-12 numbers |
| [vc4-1536-repro.md](vc4-1536-repro.md) | the runbook for the two open VC4 questions: is a single 1536 draw wrong, and what changing the vocabulary grid costs |
| [vc4-mesa-debug-build.md](vc4-mesa-debug-build.md) | cross-building Mesa 26.2.2 for the Pi on 107, the soname/version traps, and what the instrumented vc4 driver showed |
| [prefix-cache.md](prefix-cache.md) | in-memory and disk KV prefix cache, packed cache |
| [deterministic-tools.md](deterministic-tools.md) | model-free tool routing, argument validation, cache interaction |
| [future-optimization.md](future-optimization.md) | prioritized future work, measured dead ends, explicit non-goals |
| [translation-decision-model-fit.zh-CN.md](translation-decision-model-fit.zh-CN.md) | 20–150M translation/decision model fit, Jev-like decisions, adaptation plan (中文) |
| [lfm2.5-230m.md](lfm2.5-230m.md) | **LFM2.5-230M, the hybrid (conv + attention) model**: containers, run commands, the quality numbers and what is still broken |
| [lfm2.5-230m-assessment.zh-CN.md](lfm2.5-230m-assessment.zh-CN.md) | LFM2.5-230M 的打包与可行性评估：架构事实、残差/激活幅值实测、官方 GGUF 配方、工作量与风险（历史记录；实现现状见 `lfm2.5-230m.md`） |
| [rwkv7-assessment.zh-CN.md](rwkv7-assessment.zh-CN.md) | RWKV-7（Goose）支持评估：算子清单、容器/分词器/状态差距、Pi 与 Xe 的性能内存估算、分阶段工作量与风险（评估文档） |
| [rwkv7-plan.zh-CN.md](rwkv7-plan.zh-CN.md) | **RWKV-7 实施计划与逐算子清单**（目标文档）：28 个算子 + 20 项配套（文本/状态/容器/验证）的数学/形状/现状/归属/阶段、数值格式决策点、验收门、任务板映射 |
| [bitcpm.md](bitcpm.md) | **BitCPM-CANN-0.5B, the ternary model**: the container, its two new metadata fields (rope table, tokenizer prefix), the three folded constants and why `k = 1/scale_emb`, `RMS_STORE=f32`, the quality numbers, and what is not done (the Pi) |
| [bitcpm-assessment.zh-CN.md](bitcpm-assessment.zh-CN.md) | BitCPM-CANN-0.5B（MiniCPM4 的三值 QAT）支持评估：逐行三值权重的实测结构、`longrope`/`scale_emb` 两个缺口、残差峰值与 lane 和、容器大小与 Pi 预算、分阶段工作量（评估文档） |
| [smollm2-135m.md](smollm2-135m.md) | **SmolLM2-135M, the model that needs `RMS_STORE=f32`**: the container, why the rmsnorm accumulator (not the residual) was the limit, the per-switch isolation, the cost on both machines, and the open Pi-vs-Xe divergence |

Working on a numerics question?  `AGENTS.md` has the "Debugging protocol"
table that maps a question to the tool that answers it — `mmcf_dump.py`,
`ref_forward.py`, `diff_paths.sh` / `dump_diff.awk`, `check_att.py`,
`cmp_logits.py`.

Conventions:

- Numbers are measured on the machine named in the section; they are not
  promises for every Pi or driver.
- `benchmarks.md` is chronological.  The **newest dated campaign near the
  bottom** is the current matrix (with a documentation addendum after it); the
  table near the top is the 2026-11-02 history, kept so the cost model stays
  auditable.
- The Pi and laptop do **not** produce byte-identical KV caches. Build a disk
  cache on the machine that will load it.
- `CmaFree` is actual CMA occupancy; reducing `gpu_mem` or `cma` does not
  create free CMA.
