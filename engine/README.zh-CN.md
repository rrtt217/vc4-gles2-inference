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
| `engine.cpp` | EGL/GLES2、加载器、tokenizer、session、CLI |
| `make_i16.py` | F16/llama2.c -> I16M1/M2 |
| `make_q8_from_i16.py` | I16 -> Q8M1/M2 |
| `make_q4.py` | F16 safetensors -> Q4M1（离线 q4） |
| `Makefile` | 本地和 Pi 构建参数 |
| `vc4_*.cpp` | 最小 VC4 探针 |

## 构建

本地：

```bash
cd engine
make engine_mm3
```

Pi Zero（构建约 3–4 分钟，构建和运行分开）：

```bash
make -B engine_mm3 CXXFLAGS='-O2 -march=armv6 -mfpu=vfp -mfloat-abi=hard -ffast-math -std=c++17'
```

不要在很短的 timeout 下构建；旧二进制会让结果看起来不稳定。

## 运行

最小例子：

```bash
MODEL=$PWD/../models/llama2c/stories15M.i16 \
TOKENIZER=../models/llama2c/tokenizer.bin \
./engine --prompt "One day, Lily" --gen 8
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

## MiniMind3 64M

64M Qwen3 风格 dense 模型：8 层、768 hidden、2432 FFN、8 Q / 4 KV heads、
head_dim 96、vocab 6400、RoPE theta 1e6、tied embedding、per-head Q/K RMSNorm。

`--template minimind3` 使用 ChatML、空 `<think>` 块和 MiniMind 系统提示。
加 `TIMING=1` 打印 prefill/decode 时间。

Pi Zero q8，82-token prefill + 64 token 生成：

| 项目 | 数值 |
|---|---:|
| prefill | 42.7 s |
| decode | **520.9 ms/token** |
| CPU-attention 路径 | 559.5 ms/token |
| q8 权重 | 66 MB |

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

完整列表见 [测试开关](../docs/test-hooks.md)。

## 文档地图

| 文档 | 内容 |
|---|---|
| [model-formats.md](../docs/model-formats.md) | 字节布局 |
| [quantization.md](../docs/quantization.md) | q4/q6、Q4M1、packed KV、质量 |
| [engine-internals.md](../docs/engine-internals.md) | VC4 shader/FBO 细节 |
| [performance.md](../docs/performance.md) | Pi/笔记本数据 |
| [prefix-cache.md](../docs/prefix-cache.md) | worker 和磁盘 KV cache |
| [engine-layout.md](../docs/engine-layout.md) | 源码是否该拆 |

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
