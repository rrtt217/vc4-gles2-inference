# GLES2 推理引擎

面向 llama2.c 系列、Llama 结构和 Qwen3 风格 checkpoint 的小型 GLES2/EGL 推理
引擎。可以在树莓派 Zero（VC4 V3D 2.1）上无头运行，也能在桌面 GLES2 驱动上
运行；它只读一种文件格式：自描述的
[MMCF 容器](../docs/checkpoint-container.md)。

这是快速开始文档。深入内容在 [../docs/](../docs/)：

- [检查点容器](../docs/checkpoint-container.md)
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
| `tools/` | MMCF 打包器/查看器和 numpy 参考前向（host python） |
| `quantize/` | F16/I16/q8/q4 转换脚本（legacy 布局，由打包器串起来） |
| `scripts/` | 回归门禁、CPU-vs-GPU 逐段对比、attention/logits oracle、基准、模型质量评测 |
| `probes/` | 独立 VC4 探针，不链接进引擎 |
| `Makefile` | 本地、Pi 和探针构建参数 |

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
MODEL=$PWD/../models/llama2c/stories15M.i16.mmcf \
./engine_mm3 --prompt "One day, Lily" --gen 8
```

MiniMind3 q8，GPU 层 + packed KV：

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "请写一段关于春天的短文。" --gen 64 --greedy
```

MiniMind3 离线 q4（45 MB，推荐的 q4 路径）：

```bash
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 64 --greedy
```

模型文件本身是 Q4M1 时，不要设置 `Q4_WEIGHTS`。

MiniMind3-MoE（4 experts、top-1、容器 v3）——q4 的 131 MB 是 Pi Zero 装得下的
上限，q8 的 204 MB 装不下（权重吃光 CMA，session 的 FBO 创建失败）。每个 token
只读一个专家，所以解码约等于同宽度 dense 模型，另加一次路由回读：

```bash
MODEL=../models/minimind3-moe/minimind3-moe.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

`MOE_DUMP=1` 打印每层的路由和 gate logits；`MOE_FIXED=<e>` 固定专家以测量回读
的开销。打包与格式见 [`../docs/checkpoint-container.md`](../docs/checkpoint-container.md)
的 “MoE” 一节。

### MiniMind2 Small

MiniMind2 使用 Qwen ChatML 模板（容器里的 template kind 1）。源 `Q8M1` 文件头
没有 RoPE 元数据，打包时一次性给出 `--theta 1000000 --rope-half` 即可；之后
容器自己声明 rope，运行时不需要任何额外环境变量：

```bash
MODEL=../models/llama2c/minimind2-small.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 24 --greedy
```

q4 是内存模式，身份类回答明显更弱。

TinyChat15M.q8 能加载运行，但当前 tokenizer/prompt 配对下输出不连贯，
只能当格式探针，不要当对话模型使用。

### TinyStories 15M/42M

它们是 base completion 模型，不是 chat 模型；用普通 prompt，不要加 `--chat`：

```bash
MODEL=../models/llama2c/stories42M.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 \
  --prompt "Once upon a time" --gen 64 --greedy
```

Pi 上 stories15M i16 约 167 ms/token、stories42M q8 约 190 ms/token，均连贯。
`stories15M.q8` 在本仓库的转换下输出乱码，改用 i16 容器；stories42M i16 超出
当前 CMA 预算。260K 是下限探针：约 85 ms/token，但 prefill 要 5.4 s，因为
attention 仍然走完整的 512 位置网格。

这些模型来自 llama2.c fp32 `.bin` 源，打包器可以直接接受
（`--in .../stories15M.bin --quant i16`）。`LEGACY_Q8=1` 已经不存在：
引擎没有 legacy 读取器。

## 打包一个 checkpoint

引擎只加载 `.mmcf`。`.bin`、`.i16`、`.q8`、`.q4` 都只是打包器的输入：

```bash
# 一步到位：打包器直接从 safetensors 量化，不写 .i16/.q8/.q4 中间文件。
# --quant 选精度，--seq 是 KV 长度；--chat-template 把模型自己的 Jinja 模板
# 写进容器（metadata key 9），prompt 不再是 C++ 的编译选项。
python3 tools/mmcf_pack.py --in ../models/<model> --quant q4 --seq 512 \
  --tok ../models/<model>/<model>.tok \
  --out ../models/<model>/<model>.q4.mmcf \
  --name <model> --template chatml --chat \
  --chat-template ../models/<model>/tokenizer_config.json --verify

# 引擎到底从这个文件里读到了什么？
python3 tools/mmcf_dump.py ../models/<model>/<model>.q4.mmcf --check
```

`Q8M1`/`I16M1` 源必须给 `--theta` 和 `--rope-half`，因为那种文件头完全没有
rope，打包器拒绝猜测。模板取值：`plain`(0)、`chatml`/`qwen`(1)、
`minimind3`(2)、`tinybrainbot`(3)。tokenizer 会被内嵌，特殊 token id 从词表推导。

`--tok` 由模型自己的 `tokenizer.json` 生成：

```bash
python3 tools/make_tok.py --input ../models/<model> --output ../models/<model>/<model>.tok
```

ByteLevel（`Ġ`）与 SentencePiece（`▁`）两种词表都支持，并把位于 BPE 范围**之上**
的 chat 标记铺到各自 id —— 只用 `model["vocab"]` 生成的 blob 会在错误的 id 上结束，
引擎就会把 `<|im_start|>` 拆成字符。

## MiniMind3 64M

64M Qwen3 风格 dense 模型：8 层、768 hidden、2432 FFN、8 Q / 4 KV heads、
head_dim 96、vocab 6400、RoPE theta 1e6、tied embedding、per-head Q/K RMSNorm。

它的容器带 template kind 2：ChatML、空 `<think>` 块，并和其他 chat 模板共用
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
...  --thinking --prompt "1+1等于几？" --gen 64

# 工具定义
...  --tools-file minimind3/tools.json \
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
| `MODEL` | 模型路径（一个 MMCF 容器自带 tokenizer、模板和 rope） |
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
| `TOPK_FAST=1` | 近似 candidate 路径；需配合 `--top-k-candidates` |
| `RES_PACK=0` | 关闭默认的 packed residual（q8/q4） |
| `ACT8=1/2` | int8 activation 实验，需 q8 权重，默认关闭 |

完整列表见 [测试开关](../docs/test-hooks.md)，包括 dump 和探针开关
（`STAGE_DUMP`、`RAW_DUMP`、`QKV_DUMP`、`ATT_DUMP`、`ATTN_DUP`、
`PASS_DUP`、`DRAWCOUNT` 等）。

Top-k 现在是采样参数而不是环境变量：`--top-k N` 在 top-p 之前只保留概率最高的
N 个 token（0 = 关闭，默认），`--top-k-candidates` 则让这 k 个候选走上表的
candidate 路径而不是全 logits。candidate 路径是近似的：候选质量不足 `top_p` 时
回退全 logits，也不保证精确的全局 top-k；不加这个开关时，CPU 路径用
`nth_element` 选出 k 个、只排序这 k 个。

## 困惑度（PPL）

teacher-forced 困惑度是量化格式的质量闸门：

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3  --ppl data/ppl_test.txt
```

比较格式时用同一个语料；`ACT8=1` 和 `ACT8=2` 并非在所有语料上都安全。

语料长于 `SEQ` 时分窗口打分：每个窗口喂 `SEQ` 个 token，只计最后 `SEQ/2`
个（输出里的 `stride=`；`PPL_STRIDE=n` 可改）。热身不是可选的：起点落在正文
中间的窗口是"冷"的，在真实文档上每个 token 贵约 1.8 nats —— 2026-11-11 那次
"逐窗口退化"就是这么把一个正确的引擎测成了坏的（见
[`docs/performance.md`](../docs/performance.md)）。`PPL_STRIDE=0` 复原旧的冷
窗口，两者不可比。`PPL_DUMP=path` 在两种协议下都会写出每个 token 的 NLL。

## 排查一个错误答案

问题决定工具，这些工具都在本仓库里：

| 问题 | 工具 |
|---|---|
| 文件内容是我以为的吗？ | `tools/mmcf_dump.py <file>` |
| 转换是否忠实？ | `tools/mmcf_pack.py --verify`，或 `LOGITS_OUT=` + `scripts/cmp_logits.py` |
| 源模型本身对吗？ | `tools/ref_forward.py <hf-dir> "<prompt>"` |
| CPU 和 GPU 在哪个 stage 不同？ | `scripts/diff_paths.sh <file.mmcf>` + `scripts/dump_diff.awk` |
| 某一行是不是饱和了？ | `STAGE_DUMP=1` + `scripts/dump_stats.py` |
| attention 块自身是否自洽？ | `ATT_DUMP=<pos>` + `scripts/check_att.py` |
| 是不是某一次 pass 的开销？ | `ATTN_DUP`、`PASS_DUP`、`DRAWCOUNT`、`LAYER_PROFILE` |
| 数值有没有被我改坏？ | `scripts/regression.sh`（看 PPL，不要只看一个字符串） |

`GPU_LAYER=0` 不是 matvec 的 CPU 参考：attention 和 FFN 走 CPU，但所有 matvec
仍在 GPU 上。所以逐段 CPU-vs-GPU 差异只是线索而不是缺陷，下结论前先读
`scripts/diff_paths.sh` 的文件头。

## 文档地图

| 文档 | 内容 |
|---|---|
| [checkpoint-container.md](../docs/checkpoint-container.md) | MMCF 布局、元数据键、能力位 |
| [model-formats.md](../docs/model-formats.md) | 打包器消费的 legacy 布局 |
| [quantization.md](../docs/quantization.md) | q4/q6、Q4M1、packed KV、质量 |
| [engine-internals.md](../docs/engine-internals.md) | VC4 shader/FBO 细节 |
| [performance.md](../docs/performance.md) | Pi/笔记本数据 |
| [prefix-cache.md](../docs/prefix-cache.md) | worker 和磁盘 KV cache |
| [engine-layout.md](../docs/engine-layout.md) | 源码模块、X 格式和回归规则 |
| [evaluation.md](../docs/evaluation.md) | 多选知识、指令遵循与翻译三项基准 |

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
