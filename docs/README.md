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
| [quantization.md](quantization.md) | q4/q6 experiments, offline Q4M1, packed KV, quality caveats |
| [performance.md](performance.md) | Pi/laptop timing tables, the chronological lab notebook, regression baselines |
| [vc4-glsl-costs.md](vc4-glsl-costs.md) | what each GLSL function costs on VC4, read from Mesa 26.2.2 — and the six that abort the process |
| [test-hooks.md](test-hooks.md) | environment switches: probes, dumps, A/B hooks |
| [prefix-cache.md](prefix-cache.md) | in-memory and disk KV prefix cache, packed cache |
| [deterministic-tools.md](deterministic-tools.md) | model-free tool routing, argument validation, cache interaction |
| [future-optimization.md](future-optimization.md) | prioritized future work, measured dead ends, explicit non-goals |
| [translation-decision-model-fit.zh-CN.md](translation-decision-model-fit.zh-CN.md) | 20–150M translation/decision model fit, Jev-like decisions, adaptation plan (中文) |

Working on a numerics question?  `AGENTS.md` has the "Debugging protocol"
table that maps a question to the tool that answers it — `mmcf_dump.py`,
`ref_forward.py`, `diff_paths.sh` / `dump_diff.awk`, `check_att.py`,
`cmp_logits.py`.

Conventions:

- Numbers are measured on the machine named in the section; they are not
  promises for every Pi or driver.
- `performance.md` is chronological.  Its top matrix is current; later
  sections retain historical numbers so the cost model stays auditable.
- The Pi and laptop do **not** produce byte-identical KV caches. Build a disk
  cache on the machine that will load it.
- `CmaFree` is actual CMA occupancy; reducing `gpu_mem` or `cma` does not
  create free CMA.
