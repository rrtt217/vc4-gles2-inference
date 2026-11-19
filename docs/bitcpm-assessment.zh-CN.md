# BitCPM-CANN-0.5B 支持评估

状态：**评估文档；Phase A 已实现（2026-11-18）**。现状（怎么打包、怎么跑、数字、
哪些没做）在 [bitcpm.md](bitcpm.md)；本文保留"当时评估了什么、错了哪两处"，
第八节是 Phase A 的实现记录，两处被实测推翻的推断就写在里面。

本文回答"当前引擎能不能跑 `https://huggingface.co/openbmb/BitCPM-CANN-0.5B`，
不能的话要做什么"。

三类数字不可混读：

- **来源事实**：上游 `config.json`、`modeling_minicpm.py`、`tokenizer.json` 与模型卡，
  本地副本在 `models/bitcpm-0.5b/`（`config.json`、`modeling_minicpm.py`、`tokenizer.json`、
  `pytorch_model.bin`，2026-11-18 下载）；
- **实测**：本仓库代码的观察（file:line），加上在该 checkpoint 上跑的 numpy f32 前向
  （`models/bitcpm-0.5b/probe_forward.py` + `pth_reader.py`，无 torch，2026-11-18）；
- **估算**：带假设的外推，给区间，不作承诺。

## 结论

- **今天不支持，而且是在打包第一步被点名拒绝**，不是悄悄跑错：

  ```text
  MiniCPMForCausalLM: rope_scaling 'longrope' is not representable; the container
  carries a theta and a half/plain convention, nothing else.
  ```

  出处 `engine/tools/mmcf_hf.py:259`（`check_source`），复现命令见第七节。
  **被拒绝是这套流程里最好的一种失败**。真正危险的是下一个坑：`scale_emb`、
  `scale_depth`、`dim_model_base` 三个常数**没有任何检查**（`grep -n scale_emb
  engine/tools/*.py` 为空），所以只要把 rope 那一关放宽，packer 就会按"官方权重"
  静默打包，而模型的实际前向是 `12·E`、每层残差乘 `1.4/√24`、head 输入再除 4，
  答案一定是乱的。这条比 rope 重要：**先加检查，再加支持**。

- **架构几乎就是引擎现有的那一个块**：RMSNorm + silu SwiGLU + rotate-half RoPE +
  GQA（16 heads / 2 kv，GROUP=8）、无 bias、无 qk-norm、head 与 embedding tie。
  **rope 是唯一的一关**：把 `config.json` 的 `rope_scaling` 去掉重跑 packer，
  `check_source` 一声不响地放过，直接进到读张量（`KeyError: model.layers.1...`，因为
  stub 只有 layer 0）——也就是说 act / bias / qk_norm / head_dim / 张量名全都不拦。
  张量名就是 `model.layers.N.self_attn.q_proj.weight` 这一套，`load_minimind()` 直接映射。

- **它真正独特的地方是权重**：168 个矩阵（358.6M 参数，占 82.6%）是**逐行三值**
  `w ∈ {0, ±s_row}`（37.3% 是精确的 0），只有 embedding（75.2M）是 bf16。
  - 对 `q8` 是好消息：逐行对称量化把 `±s_row` 映到 `±127`，**重建误差实测为 0**
    （bit-exact，见 3.2）；
  - 对 `q4` 是坏消息：32-block 的 min/max 网格里 0 落在半个步长上，**每个 0 变成
    ±s/15**，等于把 37% 的精确 0 换成一层噪声。

- **内存决定 Pi 能不能跑，结论是"要新的三值格式"**：q8 434 MB、q4 337 MB（Pi 放不下；
  参考 SmolLM2 135M 的 164 MB 能跑、MiniMind3-MoE 的 q8 204 MB 失败）；
  **2-bit 三值 90.8 MB + q8 embedding 75.2 MB = 166 MB** 才回到 SmolLM2 的量级。

- **工作量**：**桌面正确可聊 5–7 人日**（Phase A，零新 shader）；**Pi 可跑再 7–12 人日**
  （Phase B：新的 2-bit 张量种类 + 新的 matvec kernel）；**Pi 的速度不乐观**（估算
  0.8–1.5 s/token，是 MiniMind3 q4 的 3–5 倍）。

- **建议**：如果目标是"Xe 上多一个 0.5B、质量明显高于现有 64–135M 的对话模型"，
  **值得做，Phase A 就能兑现**——模型卡报 11 项平均 51.98，而本地 f32 前向在中文
  chat 提示上直接答出"你好，我是MiniCPM系列模型，由面壁智能和OpenBMB开源社区开发"。
  如果目标是"Pi 上跑 0.5B"，Phase B 做完也只是"能跑"：同样的工日花在 Pi 的
  draw 数上收益更大。

## 一、模型是什么（来源事实）

`config.json`（`architectures: ["MiniCPMForCausalLM"]`，`_name_or_path: openbmb/MiniCPM4-0.5B`
——BitCPM 是 MiniCPM4 的三值 QAT 版本）：

| 项 | 值 | 与引擎的现状 |
|---|---|---|
| hidden / layers / FFN | 1024 / 24 / 4096 | FFN 4096 = 2×2048，正好是 VC4 的 chunk 上限（`engine/tools/mmcf_hf.py:1413` 判 `2*ffn_w > 2048`） |
| heads / kv heads / head_dim | 16 / 2 / 64 | GROUP=8；shader 用 `heads/kv_heads` 定义（`engine/src/shaders_attn.cpp:92`） |
| vocab | 73448（tie） | 48 个 1536 行的 logits tile（`kLogitsGrid`），略大于 SmolLM2 的 32 |
| 归一化 | RMSNorm，eps 1e-5 | 与编译默认一致（metadata key 11 可显式写） |
| 激活 | silu SwiGLU | 就是引擎的块 |
| rope | theta 10000 + `longrope` 32 项因子表 | **不可表示**（见 2.1） |
| scale_emb / scale_depth / dim_model_base | 12 / 1.4 / 256 | **没有检查、没有实现**（见 2.2） |
| bias / qk_norm / sliding window | 无 / 无 / 无 | 引擎的要求正好如此 |
| 参数 | 433.9M（矩阵 358.6M + embedding 75.2M） | 是引擎现有最大模型（LFM2.5-230M）的 1.9 倍 |
| 权重格式 | 伪量化 bf16，`pytorch_model.bin`（pickle，**无 safetensors**） | packer 只读 safetensors（见 2.5） |

**"CANN"只描述训练**：模型卡写明权重是 pseudo-quantized（fake quant）bf16，
"no special quantization libraries or custom kernels are required"，即推理与昇腾/CANN
无关。这一点很重要：它不是一个"要有 NPU 才能跑"的模型，只是一个把三值权重存成浮点的
checkpoint。

**长上下文**：32768 位置。`MiniCPMLongRoPE` 在 `max_position_embeddings ==
original_max_position_embeddings` 时算出 `scaling_factor = sqrt(1 + ln(1)/ln(32768)) = 1`
（`modeling_minicpm.py:157`），而本 checkpoint 的 short_factor 与 long_factor **完全相同**
——所以它退化成**一张固定的逐维频率因子表**，不是长度分支，也不需要"超过 32k 换表"的逻辑。

## 二、引擎今天缺什么

| 组件 | 引擎今天 | BitCPM 要的 | 性质 |
|---|---|---|---|
| RoPE | 一个 theta + half/plain（metadata key 2，`engine/src/model.cpp:663`） | theta 10000 + 32 项逐维因子 | **被拒绝**，需新 metadata 项 |
| 缩放常数 | 无 | `12·E`、`1.4/√24`、`head/4` | **静默忽略**，需 pack 期折叠 |
| 词表头 | `--head-dedup` 支持 tie | tie | 现成 |
| 三值权重 | `q8`/`q4`/`i16`/`bf16`/`f32` | 逐行三值 | q8 精确、q4 失真；2-bit 缺失 |
| 分词器 | 贪心最长匹配 + merge（`engine/src/tokenizer.cpp`） | SentencePiece BPE + byte fallback | **两处具体缺陷**，见 2.3 |
| 模板 | key 9 存 Jinja（`engine/tools/mmcf_format.py:200`） | 一段 ChatML 变体 | 5 行新文件 |
| 权重来源 | safetensors | pickle | 一次性转换 |
| 数值轴 | RES=q24、RMS_STORE=f32 都有 | 残差峰值 802、lane 和 1.12e6 | **现成**，见 3.1 |

### 2.1 RoPE：一张 32 项因子表（唯一被拒绝的一关）

上游算法（`modeling_minicpm.py:160-175`）：

```text
freq_i = p · (1/factor_i) · theta^(-2i/head_dim)      i = 0..31
cos/sin = cos(freq_i)·scaling_factor, sin(freq_i)·scaling_factor   (scaling_factor = 1)
```

引擎今天在 host 上就建好了这张表：`engine/src/session.cpp:193` 用
`powf(rope_theta, 2i/head_dim)` 算频率，`engine/src/gpu_attn.cpp:156` 把
cos/sin 量化成 24-bit 纹理（步长 1/4096）上传。**所以这一步不是 shader 工作，是
"把 32 个数带进容器、在那一行除一下"**：

- `engine/tools/mmcf_format.py`：新 metadata key（例如 18），payload = `head_dim/2` 个 f32；
- `engine/tools/mmcf_hf.py:128`（`rope_of`）：`rope_scaling.rope_type == 'longrope'` 时返回
  因子表，`check_source`（:259）改成只在"表长 ≠ head_dim/2"或"short ≠ long"时拒绝；
- `engine/src/model.cpp:663`：读 key 18；
- `engine/src/session.cpp:193`：`freq /= rope_factor[i]`（没有表时因子 1.0，旧容器不变）；
- `engine/tools/mmcf_dump.py`、`gate/test_mmcf_format.py`、`docs/checkpoint-container.md`
  跟一行。

规模：约 120–200 行，含 gate。精度上没有风险（cos/sin 已经是 q24/4096 的表）。

### 2.2 三个缩放常数：应该折进权重，引擎一行都不用改

上游前向（`modeling_minicpm.py:1070`、`:842`、`:849`、`:1253`）：

```text
x   = E[token] · 12
x   = x + block_out(x) · (1.4/√24)           # attention 与 FFN 各一次
logits = E · rms_final(x) / (1024/256)
```

**关键性质**：这个块的每一个子层输入都先过 RMSNorm，而且**没有 bias**。RMSNorm 对正
缩放严格不变（`rms(kx) = k·rms(x)`，eps 在 `mean(x²) ≫ eps` 时可忽略），因此整个网络对
残差流的**全局缩放是精确等变的**：`f(kx) = f(x)`。于是这三个常数可以在 pack 期一次性折进
权重，得到与上游**逐位等价（到 f32）**的等价模型：

| 常数 | 折到哪里 | 代价 |
|---|---|---|
| `scale_emb = 12` | embedding 的**逐行 scale 平面** ×12（`T_EMB` 已经每行一个 f32） | 0，精确 |
| `scale_depth/√layers = 1.4/√24` | `o_proj` 与 `down_proj` 的**逐行 scale** ×0.2858 | 0，精确（三值层仍是 {0,±1}，scale 吸收常数） |
| `hidden/dim_model_base = 4` | `rms_final` 的权重向量 ×1/4（f32 向量，精确） | 0 |

顺带一个更有用的推论：因为 k 是**自由**的，packer 可以选 `k = 128/peak` 让残差峰值正好
落在 q16 的 ±128 里——**这个模型不存在 Supra2 那种"单一全局因子跨不过早层分辨率与
末层幅值"的问题**，因为它的残差峰值只跨 47→802（17 倍），而 RMS_STORE=f32 本来也不需要
缩放（3.1）。

反向要求：`check_source` 必须在 `scale_emb/scale_depth/dim_model_base` 不是默认值、
又没有显式 `--fold-scale` 时**拒绝**，否则就是"打包成功、答案乱掉"。

### 2.3 分词器：两处具体缺陷（实测）

`make_tok.py` 能直接吃这个 `tokenizer.json`：

```text
wrote bitcpm.tok: vocab=73448 merges=57977 maxlen=48 added=8 (sentencepiece)
```

但把生成的 blob 按引擎的算法（`encode_into` + `bpe_merge`，`engine/src/tokenizer.cpp`）跑
`engine/eval/data/*.jsonl` 里的 124 条真实提示，与 HF 参考 tokenizer 逐 id 对比：

| 情况 | 与参考一致的提示数 |
|---|---|
| 现状（`TOK_SPACE_FOLD=1`） | 77/124 |
| 把 id 72181 改成 59320 之后 | **124/124** |

两处缺陷，都是具体的、小的：

1. **`▁` 与空格撞车**：这个 vocab 里 **同时**有 `▁`（id 59320）和一个字面空格
   （id 72181）。`make_tok.py` 把 `▁` 重写成空格，两条 entry 变成同一个字节串，引擎的
   `lookup_` 后写覆盖前写（`engine/src/tokenizer.cpp`，`lookup_[s] = i`）→ 每个词首的
   `▁` 都变成 72181。124 条提示里 47 条因此不同，**这是唯一的差异来源**。修法是在
   `make_tok.py` 里对"SentencePiece 的 `▁` 与字面空格"定一条优先级规则（保 `▁` 那条，
   或干脆只在没有 `▁` 时才让空格占位），并加一条 tokenizer 对比测试。
2. **byte fallback 从来没有生效**：这个 vocab 带 256 条 `<0xNN>`，`make_tok.py` 把它们
   写成 **6 个字符的字面文本**（实测 blob 里 256 条 `b"<0x41>"`，没有一条单字节高字节
   entry）。于是词表外的字符（生僻字、部分 emoji）走引擎的兜底
   `t.push_back((int)b + 3)`——**一个猜测出来的 id**。实测 `"𠀀 rare CJK"` 得到
   `243 163 131 131`，参考是 `1329 1249 1217 1217`。修法同 `make_tok_byte.py` 的思路：
   写 blob 时把 `<0xNN>` 落成裸字节，且**只在没有普通 piece 覆盖该字节时**才登记
   （否则 `?` 这种既有普通 piece 又有 `<0x3F>` 的字节会被抢走——这个坑我第一版就踩了）。

两处都是 10–30 行的改动，但都需要一个"引擎侧 encode 可被对拍"的入口（现在没有：
`--dump-tokens` 之类的模式不存在），否则修完仍然只能靠肉眼。

### 2.4 模板与特殊 token：几乎没有工作

- 模板：官方 `tokenizer_config.json` 给的就是 ChatML 变体（`<|im_start|>role\ncontent<|im_end|>\n`
  + 生成提示），差别只是它**没有 system 特判**，system 也是一条普通 message。`templates`
  目录下新增一个 jinja 文件（照 `templates/chatml.jinja` 去掉 system 分支）即可，key 9 存文本。
- `bos = 1`（`add_bos_token: True` → `--bos 1 --add-bos 1`），
  `eos = {2, 73440}`（`--eos 2 --stop 73440`，`<|im_end|>`），引擎的 key 8 正好有 eos+stop 两项。
- 默认 system prompt：官方 `use_default_system_prompt: False`，容器里留空。

### 2.5 权重来源：只有 pickle

仓库里只有 `pytorch_model.bin`（868 MB，torch zip + pickle）。packer 读 safetensors
（`engine/tools/mmcf_hf.py:700`）。两条路：

- 写一个 pth→safetensors 的转换器（本次评估已经写了读取器 `models/bitcpm-0.5b/pth_reader.py`，
  约 100 行，不需要 torch，bf16 是 fp32 的高 16 位、转换精确）；
- 或让 `mmcf_hf` 直接支持 pth（不推荐：把一条一次性路径塞进打包器）。

顺带：模型卡给的 GGUF 仓库（`openbmb/BitCPM-CANN-0.5B-gguf`）不能当输入，packer 不读 GGUF。

## 三、数值与内存（实测）

### 3.1 残差幅值：q24 + f32 就够，没有新的格式工作

本地 numpy f32 前向（`models/bitcpm-0.5b/probe_forward.py --measure`，一段 43 token 的英文
技术提示，全部位置参与；第七节的命令就是它）：

| 量 | 实测 | 引擎的轨 | 结论 |
|---|---|---|---|
| 残差峰值（逐层最大值） | 47 → 708（L2）→ 802（L22）→ 738（L23） | q16 ±128 / q24 ±32768 | **q16 会削顶，q24 安全** |
| 残差 rms | 1.3 → 10.8 | — | 与 MiniMind3 同量级 |
| rmsnorm 的 32-lane 平方和（最坏 lane） | 4067 → **1.12e6** | i24 的 2048 | **必须 `RMS_STORE=f32`**（SmolLM2 同款，已上线） |
| FFN hidden 峰值 | 22 → 84 | i16 32767 / q8 | 安全 |
| q/k 峰值（rope 之后） | 8.5 → 15.8 | kv q16 | 安全 |
| logits 幅值 | −15.4 … +12.9 | — | 无需 `--logit-shift` |

两个要注意的点：

- **峰值集中在少数位置**：L2 之后 p99 ≈ 690、中位数只有 15–95，最坏位置集中在 BOS
  与模板标记那几个 token 上。这是"massive activation"型的离群通道，不是我的前向算错——同一段
  提示用短 prompt 跑，L23 峰值只有 131。**压测要按长提示算，短 prompt 会低估 6 倍。**
- 残差峰值 802 距 SmolLM2 的 26884 差 33 倍，距 Supra2 的 2300 差 3 倍，所以**不需要**
  Supra2 那种逐层缩放，也不需要新格式；`RES=q24` + `RMS_STORE=f32` 两个已上线的开关覆盖。
  packer 的 rail 检查（`RESIDUAL_RAIL` / `RMS_LANE_RAIL`，`engine/tools/mmcf_hf.py`）会按
  校准流再报一次，那是打包时该看的数。

### 3.2 三值权重：q8 精确、q4 有系统性偏差

实测（`models/bitcpm-0.5b/probe_forward.py` 的同目录分析，对 `q_proj`/`down_proj` 抽样，
并与 `engine/tools/mmcf_hf.py:60` 的 `q8mat` / `:69` 的 `q4mat` 规则对齐）：

| 事实 | 数值 |
|---|---|
| 逐行三值的矩阵 | 168 个，358.6M 参数（82.6%） |
| 值域 | 每行 `{0, ±s_row}`，**行内只有一个非零幅值** |
| 零的比例 | 37.3% |
| 行间 scale 的跨度 | 最小 0.020，中位 0.049，最大 0.216（10.8 倍） |
| q8 重建 | **误差 0**（`±s_row → ±127`，`127/127 = 1`） |
| q4 重建 | 非零元素误差 0；**每个 0 变成 ±s/15**（最宽行 1.4e-2） |

含义：

- **q8 对这类权重是"无损"的**，这一点与高斯权重的经验相反（高斯权重 q8 是 1.076 B/参数，
  但会掉质量）；这里 q8 只是把 {−1,0,1} 原样存下来。
- **q4 的代价全部落在 0 上**：37% 的精确 0 变成 ±s/15 的噪声，相当于给每个零加了一个
  ~4% 行幅值的扰动。这对 PPL 有多大影响**必须实测**（本仓库的 q4 PPL 是格式 pin，见
  `docs/benchmarks.md`），不能假设"三值模型 q4 也没事"。
- 因此真正合适的表示是**新的 2-bit 三值种类**：0.25 B/参数、精确、并且 matvec 的纹理读取
  量是 q8 的 1/4、q4 的 1/2。这不是优化，它是这个模型在 Pi 上**能不能放下**的分界线（3.3）。

### 3.3 容器大小与 Pi 预算

按本仓库已上线容器的实测口径（`models/minimind3/minimind3.q8.mmcf` 1.076 B/参数、
`minimind3.q4.mmcf` 0.731 B/参数、q4 = 4 bit + 每 32 值 8 字节 scale/min、i16 = 2 B/参数）：

| 组合 | 大小 | Pi Zero（320 MB CMA，实测 ≤ ~180 MB 可用） | Xe |
|---|---:|---|---|
| q8 body + q8 embedding（tie，`--head-dedup`） | **434 MB** | ✗ | ✓ |
| q4 body + q8 embedding | **337 MB** | ✗ | ✓ |
| 2-bit body + q8 embedding | **166 MB** | 边界（SmolLM2 是 164 MB，能跑） | ✓ |
| 2-bit body + q6 embedding | **147 MB** | ✓ | ✓ |

再加上 tokenizer（1 MB）、KV（SEQ=2048 时 24 层 × 128 × 2048 × 2 B ≈ 12 MB）与 FBO。

**结论：Pi 上没有一条"现有格式"的路。** q4 的 337 MB 已经比 MiniMind3-MoE 失败的 204 MB
更大；唯一可行的组合要求 Phase B 的 2-bit 种类。这也解释了为什么"这个模型适合本引擎"
的说法只在**做了三值种类之后**才成立。

### 3.4 速度（估算）

- **Xe**：SmolLM2 135M（164 MB、30 层、576 宽、49152 词表）是 11.3 ms/token。本模型参数
  3.2 倍、FFN 宽 2.7 倍、24 层，**估 25–45 ms/token**（q8）；2-bit body 后权重流量降到
  1/3 左右，估 20–30 ms/token。
- **Pi Zero**：SmolLM2 135M 是 630 ms/token（454 draws/token）。本模型 24 层、FFN 4096（每层
  2 个 2048 chunk）、词表 73448（48 个 logits tile，48 次 draw）——**draw 数估 300–350/token，
  单次 draw 的权重流量是 SmolLM2 的 2–3 倍**，因此 **0.8–1.5 s/token**，prefill 同比例
  （SmolLM2 是 23 s 到首 token，这里估 60–100 s）。同一个数量级里最接近的锚点是
  LFM2.5-230M 的 1133 ms/token（169 MB q4）。

这些是外推，不是承诺；真正的数只能按 `docs/benchmarks.md` 的规程（丢弃 warm-up + 三次中位）
在板上测。

## 四、工作量

### Phase A — 桌面正确（5–7 人日，零新 shader）

| # | 工作 | 落点 | 人日 |
|---|---|---|---:|
| A1 | rope 因子表：新 metadata key（32 f32）、packer 读写、`session.cpp:193` 除因子、dump/gate/docs | `engine/tools/mmcf_format.py`、`engine/tools/mmcf_hf.py:128`/`:259`、`engine/src/model.cpp:663`、`engine/src/session.cpp:193`、`engine/tools/mmcf_dump.py`、`engine/gate/test_mmcf_format.py` | 1–1.5 |
| A2 | 三个缩放常数的折叠 + **拒绝**（未折叠时必须报错） | `engine/tools/mmcf_hf.py` 的 embedding/o_proj/down_proj/rms_final 写出处，`check_source` | 0.5–1 |
| A3 | 分词器：`▁`/空格优先级、`<0xNN>` 裸字节、一个可对拍入口（例如 `--dump-tokens`）+ 用 `engine/eval/data` 做回归 | `engine/tools/make_tok.py`、`engine/src/cli.cpp`、`engine/gate/` | 1–2 |
| A4 | 模板 + 特殊 token + 打包命令（`--chat --system "" --bos 1 --add-bos 1 --eos 2 --stop 73440 --head-dedup`） | `templates/`、`engine/tools/mmcf_pack.py` 命令行 | 0.25 |
| A5 | pth → safetensors（本次已有可复用的读取器） | `models/bitcpm-0.5b/pth_reader.py`（未跟踪）→ 一个正式转换脚本 | 0.25 |
| A6 | 验收：与 numpy f32 前向对 logits、PPL、`bench_mc`/`eval_ifeval` 基线、文档 | `engine/scripts/`、`engine/eval/`、`docs/benchmarks.md`（追加） | 1.5–2 |

验收门沿用现有规矩：`engine/scripts/regression.sh`（确定性 + 行为 + pin + PPL）、
`gate/shader_hashes.sh` 必须**不变**（Phase A 不动 shader）、`bench_mc.py` 的分数落在
模型卡 11 项平均 51.98 附近（三值模型本身有 10% 的损失）。

### Phase B — Pi 可跑（7–12 人日）

| # | 工作 | 落点 | 人日 |
|---|---|---|---:|
| B0 | **已做（2026-11-18）**：`engine/probes/t3mv_probe` 把 2-bit 的三种解包写法与 q8 主体一起定价，两机器都测；结论与模型级外推见 8.4 与 `docs/benchmarks.md` | `engine/probes/t3mv_probe` | 0.5（已完成） |
| B1 | 新的 2-bit 三值张量种类（容器 record 的 dtype、packer 检测"逐行三值"、q4 切片等价物、dump、`test_mmcf_format.py`、`checkpoint-container.md`） | `engine/tools/mmcf_format.py`、`engine/tools/mmcf_hf.py`、`engine/tools/mmcf_dump.py`、`engine/src/model.cpp` | 1.5–2 |
| B2 | **已做（2026-11-19）**：`matvec_t3` / `matvec_t3_add` 跑在 Xe 与 Pi Zero 上，两边同一段文本、同一个答案；板上 **1323 ms/token**、prefill 28 s，桌面 37.8 ms/token。做法、列主序 texel 布局与板上独有的约束见 `docs/bitcpm.md` 与 `docs/checkpoint-container.md` | `engine/src/shaders.cpp`、`engine/src/gpu_matvec.cpp`、`engine/src/gpu_upload.cpp` | 4–6（已完成） |
| B3 | 板上 bring-up：CMA 实测、SEQ、`scripts/swap_matrix.py`、Pi 与 Xe 的 PPL 对照 | `engine/scripts/`、`docs/benchmarks.md` | 2–4 |

B2 是唯一有真实技术风险的部分：`q6` 当年要拆成两个纹理是因为"VC4 一个 fragment 不能从
同一个 sampler 取两个 texel"（`engine/tools/mmcf_hf.py` 的 `q6mat` 注释），2-bit 的三值
解包要么走同样的拆平面，要么用整数运算在 shader 里展开——`docs/vc4-glsl-costs.md` 与
`probes/packunpack_probe` 是唯一能回答"代价多少"的工具。

## 五、风险

1. **分词器漂移**（高、已量化）：现状 47/124 条提示的 id 与参考不同，修好优先级后 0/124。
   修完必须用 `engine/eval/data` 做回归，否则"模型能跑但答得比 HF 差"会被误判成模型问题
   ——本仓库在 LFM2 上踩过同一个坑（数字串切分）。
2. **Pi 上的数值分歧**（中）：SmolLM2 在 Pi 上是 3.49 的 logits 偏差（Xe 1.079）、PPL
   14.29 对 7.72，且**加宽精度会更差**。这个模型有三值的 {0,±1} 结构、残差峰值 802，
   没有理由假设它在 VC4 上更听话。按 gate 的规矩：Pi 与 Xe 的 pin 分开、用 PPL 判、不要
   比字符串。
3. **q4 的质量**（中）：如果 Phase B 不做，q4 是唯一的"省内存"选项，而它把 37% 的精确 0
   变成 ±s/15。这一项只有 PPL 能判，不能靠猜。
4. **FFN 4096 没有余量**（低）：2×2048 正好等于 VC4 的 chunk 上限，任何"把 FFN 再切开"
   的想法都要改 `engine/src/gpu_ffn.cpp` 的 chunk 数，而不是改一个常数。
5. **0.5B 三值的绝对质量**（低–中）：模型卡自己说 0.5B 只保留 90.1%（11 项平均 51.98 对
   57.71），GSM8K 从 52.08 掉到 39.42。它比现有 64–135M 强，但不是"0.5B 的智能"。
6. **48 个 logits tile**（低）：Pi 上每 token 48 次 draw 是纯成本，词表没法切小（tie 的
   head 就是 embedding）。

## 六、如果要做，第一步是什么

顺序不能反：

1. **先加拒绝**（A2 的一半）：`check_source` 在 `scale_emb/scale_depth/dim_model_base`
   非默认且没有 `--fold-scale` 时报错。这一步**不依赖任何其他工作**，而且它防止的是
   "打包成功、答案乱掉"——比 rope 那一关更值得先做。
2. **再做 rope 因子表**（A1）：这是唯一被拒绝的一关，做完 packer 就能打包。
3. **同时修分词器**（A3）：两处都是 10–30 行，但需要先有一个对拍入口，否则无从验收。
4. 打包 q8（`--head-dedup`，434 MB，Xe）、跑 `regression.sh` 与 `bench_mc.py`，
   **用 `models/bitcpm-0.5b/probe_forward.py` 的 logits 做第二判据**（它已经在本机跑通，
   贪心输出是连贯中文，可以直接当 oracle）。
5. 只有当"Pi 上跑 0.5B"被确认为目标时，才开 Phase B（2-bit 种类 + kernel）。

## 七、如何复现这些数字

```bash
# 1. 打包（Phase A 之前这条命令会被拒绝，见正文第一节；现在是 doc/bitcpm.md 里那条）
python3 engine/tools/mmcf_pack.py --in models/bitcpm-0.5b --quant q8 --seq 1024 \
  --tok models/bitcpm-0.5b/bitcpm.tok --out models/bitcpm-0.5b/bitcpm-0.5b.q8.mmcf \
  --name bitcpm --template chatml --chat --system "" \
  --chat-template templates/minicpm4.jinja --head-dedup \
  --bos 1 --eos 2 --stop 73440 --add-bos 1 --verify
#   -> fold: ... | residual: peak 41.1 (rail 128.0) | rms store: 5.4e+04 -> RMS_STORE=f32
#   -> rope: theta=10000 half=True table=32 factors | verify: every tensor present verbatim

# 2. 分词器 blob 与两处缺陷
python3 engine/tools/make_tok.py --input models/bitcpm-0.5b \
  --output models/bitcpm-0.5b/bitcpm.tok
#   -> vocab=73448 merges=57977 maxlen=48 added=8 (sentencepiece)
#   注意 engine/src/tokenizer.cpp:25 的 `lookup_[s] = i`：b" " 的两条 entry 里后写的那条获胜

# 3. numpy f32 前向（贪心输出 + logits oracle）
cd models/bitcpm-0.5b && python3 probe_forward.py --prompt "用一句话介绍你自己。" --gen 20
#   -> greedy: '你好，我是MiniCPM系列模型，由面壁智能和OpenBMB开源社区开发'

# 4. 逐层残差 / FFN 峰值 / 最坏 32-lane 平方和（3.1 的表就是这个提示 43 个 token 的结果）
cd models/bitcpm-0.5b && python3 probe_forward.py --measure --prompt "The Transformer architecture replaces recurrence with self-attention, which lets every position see every other position in one step. Explain why the residual stream grows with depth."
```

`models/` 是 gitignore 的，所以 `probe_forward.py`、`config.json`、
`modeling_minicpm.py`、`tokenizer.json`、`bitcpm.tok` 都不入库（`pth_reader.py` 已经
移进 `engine/tools/`，那里是入库的）——要复现就先按上面的命令
下载（四个 HF 文件 + `pytorch_model.bin`，共约 880 MB）。`probe_forward.py --measure`
依赖 `numpy`；不带 `--measure` 时如果装了 `tokenizers`，它会用参考分词器当 oracle，
否则退回自带的近似实现。这也是本评估只有"数字"而没有"可跑的容器"的原因。

## 八、Phase A 记录（2026-11-18）

按第六节的顺序做完（"先加拒绝"这一步按用户要求跳过：中间状态不会被投放，所以
`check_source` 对 `scale_emb` 之类只做折叠、不拒绝）。现状文档见 [bitcpm.md](bitcpm.md)；
这一节只记**当时的判断被实测改掉的两处**，以及落点。

### 8.1 三处推断错了，实测改掉

1. **折叠的自由度用错了地方。** 第二节的推论是"k 自由，取 `k = 128/peak` 让残差落进
   q16"，并按"`scale_emb` 折进 embedding 的逐行 scale"实现。实测：cosine **0.885**、
   生成退化成"我是AI，我是AI，我是AI"。原因是 **tie**：head 和 embedding 是同一批行，
   把 12 折进行 scale 以后 head 的输入必须再除 12（0.25 → 0.0208），而引擎把 head 输入
   存在 q16 激活格式里（步长 1/256），0.0208 只剩 5 步 → logits 变成噪声。
   正解是 `k = 1/scale_emb`：表保持 checkpoint 原样，12 折进 layer 输出。
   量测：k=1 时 cosine 0.885，k=1/12 时 **0.9868**，而折叠后的 f32 前向与原模型
   **1.000000**（max|d| 0.0087）。顺带把残差峰值从 802 降到 41，`RES=q16` 就够。
2. **分词器还有第三个缺陷（当时没看见）：每个 text run 都要前缀。**
   参考分词器在**每个** added token 之后的文字段上跑一遍 normalizer，所以
   `<|im_start|>` 后面的 "user" 是 `▁user`（3060），引擎只在整段开头放一个标记
   → "user"（1836）。实测这条差异让同一个 chat 提示的 logits cosine 只有 **0.912**。
   修法不是改全局默认（byte-level 模型的两条路径产出相同 token，而它们的 pin 是历史
   基线），而是让**容器说话**：metadata key 10 的第三字节 `sp_prefix`，打包时由
   `tokenizer.json` 的 normalizer/decoder 判定（BitCPM 为 1，SmolLM2/MiniMind3 为 0）。
3. **`RMS_STORE=f32` 是必须项，不是建议。** 打包时的 lane 平方和 5.4e4（折叠后）对
   2048 的轨是 30 倍；没有它，答案退化成重复。已在 `docs/bitcpm.md` 里写成运行命令的
   一部分，而不是脚注。

### 8.2 落点（文件一览）

| 工作 | 落点 |
|---|---|
| rope 因子表（key 18） | `engine/tools/mmcf_format.py`（常量 + `pack_rope_factors`）、`engine/tools/mmcf_hf.py`（`rope_factors()`：只接受 short==long 且不拉伸的表）、`engine/tools/mmcf_pack.py`、`engine/src/model.cpp`（读 + 长度校验） |
| rope 生效 | `engine/src/session.cpp`（`freq /= factor[i]`，表本来就是 host 侧生成） |
| 三个常数的折叠 | `engine/tools/mmcf_hf.py`（`fold_scale_constants()`，`k = 1/scale_emb`） |
| 分词器前缀 | metadata key 10 第三字节、`engine/src/tokenizer.cpp`（`encode_into` 的 run 状态）、`engine/src/app.cpp`（`set_seg_prefix` 在 `/model` 会重跑的那一处） |
| 分词器 blob 两处 | `engine/tools/make_tok.py`（`<0xNN>` 裸字节、`dedupe()` 的优先级：真 piece > byte fallback，id 小者胜） |
| 只有 pickle 的源 | `engine/tools/pth_reader.py` + `load_minimind` 的回退（不写中间文件、不需要 torch） |
| 模板 | `templates/minicpm4.jinja`（官方模板逐字） |

### 8.3 验收（Iris Xe，2026-11-18）

- 与 f32 numpy 前向（`models/bitcpm-0.5b/probe_forward.py`）：logits KL **0.018** nats、
  top-20 命中 19/20；667 token 混合文本上 4 个位置的 nll 均值 3.107 对 3.024；
  长位置（500 token 上下文）两边的 nll 0.0001 对 0.0003 —— **rope 表在长距离上也对**。
- `engine/eval/bench_mc.py` 100 题：acc **0.690**、acc_cal 0.660、fmt 0.690，
  null 0.254（同表里 pocket qat 0.480、LFM2.5-230M 0.380、MiniMind3 q8 0.220）。
- `--ppl` SEQ=768：36.5246（nll 3.5980）。注意 `--ppl` 路径不带前导空格标记，这个模型
  在缺标记时第一个 token 要付 16 nats；它是**同模型回归 pin**，不跨分词器比较。
- 解码 58 ms/token、34 token 提示的 prefill 1.0–1.3 s（`TIMING=1`）。

### 8.4 Phase B 的第一步已经做了：2-bit matvec 的定价（2026-11-18）

`engine/probes/t3mv_probe` 把引擎的 q8 matvec 主体和"一个 texel 装 16 个三值权重"的
同一段循环放在一起量，三个写法（ES2 浮点 `mod`/`floor`、ES3 整数移位、以及把解包去掉的
取纹理对照），每个先与 CPU 点积对拍再计时（`docs/benchmarks.md`）：

| ns/MAC | q8 | ES2 浮点 | ES3 整数 | 只取纹理 |
|---|---:|---:|---:|---:|
| Pi Zero (VC4) 1024x1024 | 2.97 | 7.53 | **2.45** | 0.66 |
| Pi Zero 1024x4096 | 2.99 | 7.52 | 2.42 | 0.66 |
| Iris Xe 1024x1024 | 0.063 | 0.115 | 0.109 | 0.032 |

结论改了 Phase B 的账：

- **VC4 上 2-bit 只快 1.22 倍**（2.98 → 2.44），不是 4 倍：只取纹理是 q8 的 22%，
  说明这个 matvec 是 **ALU/发射受限，不是带宽受限**，权重压紧 4 倍只买到四分之一。
- **ES2 的浮点解包比 q8 还慢 2.5 倍**（7.53），所以三值种类**只能在 ESSL 3.00 上写**
  ——这也正是 Q4_SHIFT 当年走的同一条路。
- **Iris Xe 上 2-bit 完全不赚**（0.109 对 0.063）：它不缺带宽，多出来的 ALU 是纯成本。
  2-bit 是 VC4 专用特性。
- 模型级（434M MAC/token）：q8 1.29 s/token 对 2-bit 1.10 s，**买的是"放得下"
  （165 MB 对 434 MB），不是速度**。所以 B2 依然值得做，但收益预期要从"1.2 s → 0.7 s"
  改成"从不能跑到能跑，外加 1.2 倍"。

下一步若要继续 Phase B，先做 B1（容器）而不是 B2（kernel）：kernel 的写法已经被这个
probe 定下来了（ES3 整数解包、16 权重一 texel），B1 不依赖板上时间。

### 8.5 仍未做

- **q4 容器没打**：`q4mat` 的 32 值 min/max 把 0 放在半个步长上，每个 0 变成 `±s/15`
  —— 这是这个模型唯一"不无损"的格式，要用 PPL 判。
- **Pi Zero 只放得下 2-bit 容器**（q8 434 MB / q4 337 MB 对板上约 180 MB；t3 167.5 MB
  是唯一放得下的）。B2 落地后的实测是 **1323 ms/token**（`GPU_LAYER=1`，prefill 28.2 s），
  比当时的估算慢——估算是按 matvec 的 ns/MAC 外推的，没有算 24 层的 draw 开销与
  一次取 4 个权重的 fetch 形状。
- **SEQ > 1024 没测**；容器声明 1024。
