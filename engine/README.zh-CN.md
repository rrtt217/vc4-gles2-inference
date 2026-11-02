# GLES2 推理引擎

面向 llama2.c 系列和 Qwen3 风格 checkpoint 的小型 GLES2/EGL 推理引擎。
可以在树莓派 Zero（VC4 V3D 2.1）上无头运行，也能在桌面 GLES2 驱动上运行。

这是快速开始文档。深入内容在 [../docs/](../docs/)：

- [模型格式](../docs/model-formats.md)
- [VC4 / 引擎内部](../docs/engine-internals.md)
- [量化与 Q4M1](../docs/quantization.md)
- [性能](../docs/performance.md)
- [测试开关](../docs/test-hooks.md)
- [源码布局与拆分建议](../docs/engine-layout.md)

## 源码布局

| 路径 | 用途 |
|---|---|
| `src/` | EGL/GLES2 引擎、模型、session、tokenizer、CLI |
| `src/xformat.*` | 激活纹理布局和 GLSL X 宏 |
| `src/shaders*.cpp` | GLSL 源码构建 |
| `probes/` | 独立 VC4 探针，不链接进引擎 |
| `quantize/` | F16/I16/q8/q4 转换脚本 |
| `scripts/` | 基准和回归脚本 |
| `Makefile` | 本地和 Pi 构建参数 |

## 构建

本地：

```bash
cd engine
make engine_mm3
```

Pi Zero（构建约 8–10 分钟，构建和运行分开；不要同时跑两个构建）：

```bash
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

不要在很短的 timeout 下构建；旧二进制会让结果看起来不稳定。

## 运行

最小例子：

```bash
MODEL=$PWD/../models/llama2c/stories15M.i16 \
TOKENIZER=../models/llama2c/tokenizer.bin \
./engine_mm3 --prompt "One day, Lily" --gen 8
```

MiniMind3 q8，GPU 层 + packed KV：

```bash
MODEL=../models/minimind3/minimind3.q8 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "请写一段关于春天的短文。" --gen 64 --greedy
```

MiniMind3 离线 q4（45 MB，推荐的 q4 路径）：

```bash
MODEL=../models/minimind3/minimind3.q4 \
TOKENIZER=../models/minimind3/minimind3.tok \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template minimind3 \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

模型文件本身是 Q4M1 时，不要设置 `Q4_WEIGHTS`。

### MiniMind2 Small

MiniMind2 使用 Qwen ChatML 模板。旧 Q8M1 文件头没有 RoPE 元数据，q8 必须显式补：

```bash
ROPE_THETA=1000000 ROPE_HALF=1 \
MODEL=../models/llama2c/minimind2-small.q8 \
TOKENIZER=../models/llama2c/minimind2-tokenizer.bin \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat --template qwen \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

I16/Q4M1 文件头自带 theta/half。q4 是内存模式，身份类回答明显更弱。

TinyChat15M.q8 能加载运行，但当前 tokenizer/prompt 配对下输出不连贯，
只能当格式探针，不要当对话模型使用。

### TinyStories 15M/42M

它们是 base completion 模型，不是 chat 模型；用普通 prompt，不要加 `--chat`：

```bash
MODEL=../models/llama2c/stories42M.q8 \
TOKENIZER=../models/llama2c/tokenizer.bin \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 \
  --prompt "Once upon a time" --gen 64 --greedy
```

Pi 上 stories15M i16 约 167 ms/token，legacy `stories15M.bin` 约 161 ms/token，
stories42M q8 约 190 ms/token，均连贯。`stories15M.q8` 当前转换输出乱码，
改用 `.bin` 或 i16；stories42M i16 超出当前 CMA 预算。260K legacy
（`stories260K.bin` + `tok512.bin`）在 legacy-I16 修复后约 85 ms/token，
但仍要 5.4 s prefill，只能当 floor 探针。

legacy fp32 默认按 I16 加载；只有内存不够时才用 `LEGACY_Q8=1`（例如 42M `.bin`）。
小 TinyStories 模型在 q8 模式下会失去连贯性。

## MiniMind3 64M

64M Qwen3 风格 dense 模型：8 层、768 hidden、2432 FFN、8 Q / 4 KV heads、
head_dim 96、vocab 6400、RoPE theta 1e6、tied embedding、per-head Q/K RMSNorm。

`--template minimind3` 使用 ChatML、空 `<think>` 块，并和其他 chat 模板共用
短英文默认系统提示（`You are a helpful assistant.`）。需要时用 `--system` 覆盖。
加 `TIMING=1` 打印 prefill/decode 时间。

Pi Zero，16 token 生成（2026-11-02，Pi 上还有 Minecraft server，数字有噪声）：

| 模型 | prefill | decode |
|---|---:|---:|
| MiniMind3 q8 | 20.5 s | **238 ms/token** |
| MiniMind3 q4 | 33.0 s | **381 ms/token** |
| MiniMind2 i16 | 9.2 s | 226 ms/token |
| MiniMind2 q8 | 8.4 s | 140–190 ms/token |
| MiniMind2 q4 | 9.7 s | 193–212 ms/token |
| TinyChat15M q8 | 4.8 s | 139–171 ms/token（不连贯） |

旧的 520.9 ms/token、42.7 s prefill 是 q16 attention 和 q4 paired residual 之前的数字。

FFN hidden 2432 超过 VC4 2048 像素 FBO 限制。
`Model::split_ffn()` 把 W1/W3 按行、W2 按列切成 1216 宽两块；
第二块 W2 matvec 加到第一块 residual 上。

## Thinking 和工具调用

```bash
# 显式 thinking
... --template minimind3 --thinking --prompt "1+1等于几？" --gen 64

# 工具定义
... --template minimind3 --tools-file minimind3/tools.json \
  --prompt "现在几点了？" --gen 64 --greedy
```

配置 tools 时，引擎会把 KV cache 自动提高到 768；显式 `SEQ` 优先。

内置工具：

- `get_current_time` — IANA 时区；
- `calculate_math` — 安全的递归下降算术解析（`+ - * / % ^`、括号、
  一元运算、Python 风格 `**`）。

工具循环：生成到 `<|im_end|>`，解析所有完整 `<tool_call>`，执行，
追加一个包含 `<tool_response>` 的 user turn，再 prefill；最多 3 轮。

64M 模型在 64 token 内经常不输出 `</think>`；工具调用更可靠。

## 环境变量

| 变量 | 含义 |
|---|---|
| `MODEL`, `TOKENIZER` | 模型和 tokenizer 路径 |
| `SEQ` | KV cache 长度；配置 tools 时自动到 768 |
| `GPU_ATTN=1` | GPU RoPE + KV cache + attention |
| `GPU_LAYER=1` | hidden、attention、RMSNorm、FFN 全在 GPU |
| `KV_PACK=16` | packed q16 KV（每个 texel 两个 int16） |
| `KV_PACK=8` | packed q8 KV；质量不适合默认 |
| `KV_SCALE` | KV fixed-point scale |
| `Q4_WEIGHTS=1` | 运行时 q4 转换（实验） |
| `KEEP_CPU_WEIGHTS=1` | 上传后保留 CPU 权重（调试） |
| `TIMING=1` | 打印 prefill/decode 时间 |
| `LAYER_PROFILE=1` | 逐层时间 |
| `TOPK`, `TOPK_FAST` | candidate Top-K 路径 |
| `RES_PACK=0` | 关闭默认的 packed residual（q8/q4） |
| `ACT8=1/2` | int8 activation 实验，需 q8 权重，默认关闭 |

完整列表见 [测试开关](../docs/test-hooks.md)。

## 文档地图

| 文档 | 内容 |
|---|---|
| [model-formats.md](../docs/model-formats.md) | 字节布局 |
| [quantization.md](../docs/quantization.md) | q4/q6、Q4M1、packed KV、质量 |
| [engine-internals.md](../docs/engine-internals.md) | VC4 shader/FBO 细节 |
| [performance.md](../docs/performance.md) | Pi/笔记本数据 |
| [prefix-cache.md](../docs/prefix-cache.md) | worker 和磁盘 KV cache |
| [engine-layout.md](../docs/engine-layout.md) | 源码模块、X 格式和回归规则 |

## 服务器

OpenAI 兼容服务器在 [../server/](../server/)：

```bash
python3 server.py --seq 768 --kv-pack 16 \
  --prefix-cache prefix_tools_pack16_768.kv
```

见 [server/README.zh-CN.md](../server/README.zh-CN.md)。

## 退出

`SIGINT`/`SIGTERM` 只设置标志，不在信号处理里销毁 EGL。
当前 GPU 调用/层/token 完成后，再 `glFinish()` 并正常销毁 EGL。
chat 模式下 Ctrl+C 停止当前轮并回到提示符；在提示符处 Ctrl+C 退出。
