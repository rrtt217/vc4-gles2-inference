# VC4 / GLES2 MiniMind3 推理

在树莓派 Zero（BCM2835 / VC4 V3D 2.1）上通过 GLES2/EGL 做实用的小模型
无头推理。它不是演示程序：可以跑 Qwen3 风格 64M 模型、原生工具调用、
OpenAI 兼容 HTTP 服务器和磁盘 KV prefix cache。

[English](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 目前可用

- **MiniMind3 64M** — 8 层、768 hidden、2432 FFN、8 Q / 4 KV heads、
  head_dim 96、vocab 6400、RoPE theta 1e6、per-head Q/K RMSNorm。
- **MiniMind2 Small** — i16/q8/q4；Qwen 风格 chat，指令跟随弱于 MiniMind3。
- **只有一种文件格式** — 自描述的 **MMCF** 容器。权重自带 tokenizer、
  chat 模板、rope 配置和能力位，所以一个 `MODEL=` 就是全部配置。
  legacy 布局（llama2.c fp32、Q8M1、I16M1、Q8M2/I16M2、Q4M1）只是
  `engine/tools/mmcf_pack.py` 的**转换输入**；引擎只读 `MMCF`。
- **KV cache** — 24-bit 基线、packed q16（推荐默认）、packed q8
  （质量有风险）。
- **服务器** — Python 标准库 OpenAI 兼容接口、SSE、标准 tool calls、
  usage 统计。
- **工具** — 内置时间/算术工具；模型输出 tool call，worker 执行后把
  tool response 喂回。
- **Prefix cache** — worker 内 token/KV 复用，以及 `MMKV1` 磁盘缓存。

## 快速开始

### 笔记本

```bash
cd engine
make engine_mm3

MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### Pi Zero

在 Pi 上单独构建，构建和运行分开：

```bash
cd engine
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

运行：

```bash
cd engine
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

### 离线 q4

```bash
# 一步完成：打包器直接从 safetensors 量化，不写中间文件
python3 engine/tools/mmcf_pack.py --in models/minimind3 --quant q4 --seq 512 \
  --tok models/minimind3/minimind3.tok --out models/minimind3/minimind3.q4.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja --verify

cd engine
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

离线 Q4M1 为 45 MB，q8 为 66 MB。shader unroll 后本地约
16–23 ms/token（q8 约 14–20 ms/token），但省权重显存。要从 F16 源量化；q8 -> q4 是二次
量化，质量会下降。

### OpenAI 兼容服务器

```bash
cd server
python3 server.py --seq 768 --kv-pack 16 \
  --prefix-cache prefix_tools_pack16_768.kv
```

接口、工具和缓存细节见 [server/README.zh-CN.md](server/README.zh-CN.md)。

## 仓库结构

| 路径 | 内容 |
|---|---|
| [engine/](engine/) | GLES2 引擎、容器工具链、转换器、探针、脚本 |
| [models/](models/) | 统一模型目录（gitignore） |
| [minimind3/](minimind3/) | MiniMind3 源配置、tokenizer 源和参考实现 |
| [server/](server/) | OpenAI 兼容 Python 服务器 |
| [docs/](docs/) | 深入文档：容器、格式、内部、量化、性能 |

## 模型大小和速度

每一行都是 `.mmcf` 容器；打包前的 legacy 文件留在 `models/` 里作为重新打包的源。

| 模型 | 大小 | Pi decode | 说明 |
|---|---:|---:|---|
| MiniMind3 q8 | 66 MB | ~238 ms/token | 默认中文对话 |
| MiniMind3 q4 | 45 MB | ~381 ms/token | 内存模式，质量略低 |
| MiniMind2 i16 | 58 MB | ~226 ms/token | 参考模型，指令跟随较弱 |
| MiniMind2 q8 | 29 MB | ~140–190 ms/token | rope 由容器自带 |
| MiniMind2 q4 | 21 MB | ~193–212 ms/token | 内存模式，身份类回答不可靠 |
| TinyChat15M q8 | 24 MB | ~139–171 ms/token | 能运行但输出不连贯，只作格式探针 |
| stories15M i16 | 49 MB | ~167 ms/token | TinyStories 补全，连贯 |
| stories42M q8 | 58 MB | ~190 ms/token | TinyStories 补全，连贯 |
| stories15M q8 | 24 MB | ~148 ms/token | q8 转换输出乱码，改用 i16 |
| stories260K | 1 MB | ~85 ms/token | 下限探针；prefill 5.4 s |

Pi 数字是单次运行，受后台负载影响；详见 [performance.md](docs/performance.md)。
TinyChat15M 目前还不能算可用的对话模型，需要先修 tokenizer/prompt 配对。

## 文档

- [引擎快速开始](engine/README.zh-CN.md)
- [检查点容器（MMCF）](docs/checkpoint-container.md)
- [模型格式](docs/model-formats.md)
- [量化与 Q4M1](docs/quantization.md)
- [VC4 / 引擎内部](docs/engine-internals.md)
- [Prefix cache](docs/prefix-cache.md)
- [性能](docs/performance.md)
- [测试开关](docs/test-hooks.md)
- [源码布局 / 拆分建议](docs/engine-layout.md)
- [未来优化方向](docs/future-optimization.md)
- [Agent 说明](AGENTS.md)

深入文档目前以英文为主；项目总览、引擎和服务器快速开始有中文版。

## 模型与致谢

模型权重和 tokenizer **不包含在仓库中**，请按 [models/README.md](models/README.md) 放到 `models/`。
MiniMind3 架构/tokenizer 参考 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）；基础引擎参考 [karpathy/llama2.c](https://github.com/karpathy/llama2.c)（MIT）。

## 许可证

MIT，见 [LICENSE](LICENSE)。模型权重和第三方模型资源不包含在本仓库许可证范围内，
其归属见上面的致谢。

## 已知限制

- **引擎只加载 MMCF 容器。** `.bin`/`.q8`/`.q4`/`.i16` 是打包工具输入，
  用 `engine/tools/mmcf_pack.py` 转换。
- Pi 和笔记本生成的 KV cache **不是逐字节相同**；磁盘缓存要在目标机器上生成。
- `CmaFree` 是真实 CMA 占用。调小 `gpu_mem` 或 `cma` 不会凭空增加 CMA，
  反而可能让压力更大。
- Pi Zero 上 `SEQ=768` + packed q16 很紧（加载后 `CmaFree` 约 2 MB）；
  要用实际模型/缓存组合验证。
- q8 KV 流畅但质量不安全；packed q16 是安全的 KV 默认。
- q4 不是 bit-exact，算术质量依赖 prompt。
- 引擎已拆成 `engine/src/` 多个模块；X 格式和回归规则见
  [engine-layout.md](docs/engine-layout.md)。

