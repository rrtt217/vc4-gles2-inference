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
| [model-formats.md](model-formats.md) | legacy, Q8M1, I16M1, Q8M2/I16M2, Q4M1 |
| [engine-internals.md](engine-internals.md) | VC4 findings, FFN split, GPU attention, persistent layers, Top-K |
| [quantization.md](quantization.md) | q4/q6 experiments, offline Q4M1, packed KV, quality caveats |
| [performance.md](performance.md) | Pi/laptop timing tables and regression baselines |
| [test-hooks.md](test-hooks.md) | environment switches for A/B tests |
| [prefix-cache.md](prefix-cache.md) | in-memory and disk KV prefix cache, packed cache |
| [engine-layout.md](engine-layout.md) | source layout and whether to split `engine.cpp` |
| [future-optimization.md](future-optimization.md) | prioritized future work and explicit non-goals |

Conventions:

- Numbers are measured on the machine named in the section; they are not
  promises for every Pi or driver.
- The Pi and laptop do **not** produce byte-identical KV caches. Build a disk
  cache on the machine that will load it.
- `CmaFree` is actual CMA occupancy; reducing `gpu_mem` or `cma` does not
  create free CMA.
