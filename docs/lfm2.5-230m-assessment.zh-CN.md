# LFM2.5-230M 支持评估

状态：**评估文档；packer 与 conv 块的串行路径已实现（2026-11-11），GPU 前向未完成**。
已落地：`load_lfm2` + key 14 + 张量 `0x17/0x18/0x19`；serial 路径可运行
（PPL 111.83，`The capital of France is` → `Paris`）；GPU conv 链已搭好但输出仍
与串行参考不一致（`CONV_GPU=1` 才启用）。详见 `docs/performance.md` 末尾。本文件回答"把 LFM2.5-230M 跑在本引擎上要做多少
工作、卡在哪里"。文中数字分三类，不可混读：

- **来源事实**：来自 HF 配置、参考实现源码或本仓库源码，已标注出处；
- **现状事实**：本仓库当前状态的直接观察（文件、行号）；
- **估算**：明确标注"估算"，在 Phase 0 实测前不作为承诺。

## 结论

- 工作量：**约 7–13 个工作日（1.5–2.5 周）**，桌面 Iris Xe 优先，Pi Zero 收尾。
- 真正的新代码集中在 **8 个 conv 层的 GPU/CPU 实现**（约 3–7 天）和 packer 的
  LFM2 名称映射（约 1–2 天）；容器格式、注意力、SwiGLU、量化管线、q24 残差、
  模板渲染全部可复用。
- 最大不确定项（按影响排序）：
  1. ~~残差 ±128 rail~~——**已实测（2026-11-11）：峰值 29.5，为 rail 的 0.23 倍，
     q16 足够，不需要 q24**，见第八节；
  2. **Pi Zero 的 CMA 预算**——实测良好条件下约 **170 MB** CMA，200 MB 量级的
     权重会挤掉 KV cache。按此预算：q4+去重+q8 head（167 MB 权重）已放不下；
     必须 **去重 + q6 head**（~151–155 MB 权重，见第十节）。这使 q6 head 从
     "可选优化"变成 Pi 的**必需项**；
  3. conv 门乘幅值——**已实测**：`B*x` ≤ 11.4、`C*conv` ≤ 17.7，相对 I16_2 的
     ±2048 有 >100 倍余量。

## 一、模型事实（已核实）

来源：[HF config.json](https://huggingface.co/LiquidAI/LFM2.5-230M)、
[transformers 5.2 参考实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py)、
[模型卡](https://huggingface.co/LiquidAI/LFM2.5-230M)。

| 项 | 值 | 对引擎的含义 |
|---|---|---|
| 层结构 | 14 层 = 8× conv + 6× full attention，按 `layer_types` 交错（conv,conv,attn,conv,attn,…） | 引擎只有"每层都带注意力"的循环，conv 层是全新算子 |
| hidden / FFN | 1024 / SwiGLU 2560（`feed_forward.w1/w3/w2`，无偏置） | SwiGLU 已有；2560 > 2048 走既有 `split_ffn`（两块 1280，`hidden≤4096` 规则内） |
| 注意力 | GQA 16q/8kv，head_dim 64，RoPE θ=1e6 非交错，**q/k per-head RMSNorm** | 引擎已有 `qk_norm` + `q_norm/k_norm` + GQA，注意力层几乎零改动 |
| conv 块 | `in_proj` 1024→3072 → 拆 B/C/x → `B*x` → 深度因果 conv1d(k=3, 无偏置, 状态 3) → `C*conv` → `out_proj` 1024→1024 | 需新 GPU 核、CPU 参考与状态管理 |
| Norm | `embedding_norm` 只作**最终归一化**（第 520 行，最后一次层输出后）；eps 1e-5 | 直接对应 `rms_final`；1e-5 恰为引擎字面量默认，数值中性；无新增 pass |
| 词表 | embedding 65536×1024，tie；tokenizer 实际 id 0..64401 | packer 已有 `vocab_limit` 截断路径（截到 64402 行，省 1134 行死重） |
| 精度 | bf16 权重，全部无 bias | 符合引擎"无 bias 张量"约束 |

### tokenizer 与模板

- tokenizer 为 `tokenizers` 格式 BPE：基础词表 64400 + added tokens，pre-tokenizer
  为 `[Split(GPT-2 正则), ByteLevel]`，decoder 为 ByteLevel——与 `make_tok.py`
  的 ByteLevel 路径同构。特殊 id：`<|startoftext|>`=1（BOS）、`<|im_end|>`=7（EOS/stop）、
  `<|pad|>`=0。added tokens 分布于词表中段（64011–64401）与尾部扩展，需校验
  blob 构建与编解码往返。
- 模板文件在 HF repo 内（`chat_template.jinja`），ChatML 风格，含 tools 与
  thinking 分支。引擎自带 Jinja 的兼容性已逐项核对：
  - `generation`/`endgeneration` 被显式忽略（`engine/third_party/jinja/parser.cpp:219`），
    对首轮与历史轮渲染都无副作用；
  - `namespace()`、`dict()`、宏、`join`（数组）、`default/tojson/trim/length`、
    `defined/string/mapping` 测试均存在；
  - 空数组、空串在布尔上下文中为 false，`{%- if tools -%}` 不会污染 system 提示；
  - **缺 `split`、`endswith` 字符串方法**——单轮对话不受影响；多轮中出现
    `think` 标签或 tool_calls 的历史消息会触发，需要给 vendored Jinja 补两个字符串
    方法（小改动，但属于 third_party 本地编辑，需在 `engine/third_party/jinja/README`
    记录为第四个本地编辑）。
- 模板自带 `{{- bos_token -}}`，因此容器必须 `--add-bos 0`，否则双 BOS。
- 模型卡建议的默认 system 提示为 "You are a helpful assistant trained by Liquid AI."，
  packer 的 `--system` 应按模型卡设置。

## 二、引擎现状：可复用清单

架构参数全部来自容器元数据（`Model` 的 dim/hidden/layers/heads/kv_heads/vocab/seq
均从 key 1 读取），没有硬编码 MiniMind 形状。可直接复用的部分：

| 能力 | 位置 | 说明 |
|---|---|---|
| MMCF v2 容器 | `docs/checkpoint-container.md` | 未知 tensor id 忽略、未知 metadata key 跳过——**加 conv 张量与层类型元数据不需要 v3** |
| q8/q4/i16 量化与上传 | `engine/tools/mmcf_hf.py` | 含 per-row q8 scale plane、两步量化顺序 |
| q24 残差 | key 12 + `Model::res_q24` + `x_i24enc` | rail 超限时的既有机制 |
| GQA + qk_norm 注意力 | `session.cpp` / `shaders_attn.cpp` | 与 LFM2 注意力同构 |
| 融合 SwiGLU FFN | `shaders.cpp` `swiglu()` | 含 q16 域门乘+clamp 的现成范式 |
| FFN 分块 | `Model::split_ffn` | 2560 → 1280+1280，规则内 |
| matvec 核 | `shaders.cpp` | in_proj（1024→3072）/out_proj（1024→1024）就是普通 matvec |
| tokenizer 嵌入 | `make_tok.py` | ByteLevel BPE + added tokens 落位 |
| logits 网格 | `logits_grid.h`（1536 行/tile） | 64402 行 → 42 个 tile，规则允许（ceil 全部成立） |
| 回归门 | `engine/scripts/regression.sh` | 增量改动不得移动 MiniMind3 引脚 |

## 三、差距清单（按组件）

### 1. packer（`mmcf_hf.py`，估算 1–2 天）

- `load_lfm2()`：名称映射（`model.layers.N.conv.{in_proj,conv,out_proj}`、
  `self_attn.{q,k,v,out}_proj`+`{q,k}_layernorm`、`feed_forward.{w1,w3,w2}`、
  `operator_norm`/`ffn_norm`/`embedding_norm`）；conv 层无 qkv，填占位。
- 新 tensor id：如 `0x40` in_proj、`0x41` conv 权重、`0x42` out_proj；新 metadata：
  14 层类型掩码 + "embedding 后 norm"标志 + conv kernel size（均为新 key，
  旧 reader 跳过；新 engine 独占，无回兼容问题）。
- `residual_peaks()` 加 LFM2 分支（numpy conv 递推），否则 rail 检查对 LFM2 不生效。
- 词表截断到 tokenizer 实际条目数（64402），并用 `--bos 1 --eos 7 --stop 7`
  显式指定特殊 id（名称表里没有 `<|startoftext|>`）。
- `check_source()` 今天会干净地拒绝 LFM2（报"not the Llama/Qwen layout"）——行为
  正确，只差 loader。

### 2. 引擎（估算 3–7 天，核心）

- conv 层 CPU 前向（f32 参考路径，约 200 行）。
- GPU 新核：
  - 门乘 `B*x`：q16 域逐元素，范式同融合 swiglu（已有先例）；
  - 深度因果 conv k=3：prefill 版 + decode 版（状态 2 个隐值/通道）；新 GLSL，
    必须按 `wrow_probe` 先例在 VC4 上验证采样行为；
  - in_proj/out_proj 复用 matvec 核（宽度 3072/1024 都是 4/32 倍数）。
- 所有前向路径按层类型分支（GPU 融合、GPU 串行、CPU、双位置 prefill pair）；
  conv 层可先走串行路径保正确性，再谈融合。
- conv 状态：8 层 × 3 × 1024 i16 ≈ 48 KB。
- prefix cache：头加 conv 状态段（增量式，含 pack 模式/KV_SCALE 兼容检查）。
- KV cache：只为 6 个注意力层分配（或保留 14 层分配、接受 ~12 MB 浪费）。

### 3. 验证与收尾（估算 2–4 天）

- PyTorch/transformers（≥5.2）参考 logits + PPL；可选 llama.cpp GGUF 交叉对照。
- 模板 render-diff（tools/thinking 路径需要先补 `split`/`endswith`）。
- tokenizer 编解码往返；两 prompt 行为检查（非空、不崩）。
- `regression.sh` + `shader_hashes.sh` 必须过；MiniMind3 q8/q4 引脚不得移动。
- Pi 交叉编译、部署、计时矩阵；文档更新（`model-formats.md` 新 tensor id、
  `checkpoint-container.md` 新 key、`test-hooks.md`、`performance.md`）。

## 四、工作量分解

| 阶段 | 内容 | 估算 |
|---|---|---|
| 0 | 参考环境 + 数值测量（残差峰值、门乘幅值、模板/分词验证） | 0.5–1 天 |
| 1 | packer：`load_lfm2` + 新 tensor id/metadata + rail 分支 | 1–2 天 |
| 2 | 引擎：conv 核（GPU+CPU）+ 层循环分支 + prefix cache + embedding norm | 3–7 天 |
| 3 | 验证、Pi 部署、文档 | 2–4 天 |
| 合计 | | **7–13 个工作日** |

## 五、风险清单

| # | 风险 | 证据 | 缓解 |
|---|---|---|---|
| 1 | ~~残差超 ±128 rail~~ **已排除** | 实测峰值 29.5（0.23× rail，chat 提示 25.3，随机文本 6.9），见第八节 | 默认 q16 路径即可；q24 不必启用 |
| 2 | Pi Zero 的 CMA 放不下 | 实测 ~170 MB CMA（良好条件），200 MB 量级挤掉 KV；q4+去重+q8 head 已 167 MB | 去重 + q6 head → ~151–155 MB；见第十节。桌面优先 |
| 3 | logits 开销：42 tile × 1536×1024 | `kLogitsGrid=1536`，64402 行；MiniMind3 仅 5 tile | 评估 `--top-k-candidates`；接受 Pi 上 decode 变慢 |
| 4 | Pi Zero decode 预计 ≥1 s/token | 线性外推（估算）：q8 ≈ 0.8–1.1 s，q4 ≈ 1.3–1.6 s（以 MiniMind3 252/388 ms 为基数） | 实测为准；这是模型规模决定的，不是缺陷 |
| 5 | ~~conv 门乘 q16 幅值~~ **已实测** | `B*x` ≤ 11.4、`C*conv` ≤ 17.7、conv 输出 ≤ 4.1；FFN gate/up ≤ 5.1、乘积 ≤ 26 | 全部在 I16_2（±2048）与 q16 内，无需缩放 |
| 6 | 新 GLSL 在 VC4 上的采样行为 | wrow 探针的先例（单 sampler 两次采样未定义） | 先写 probe 验证 conv 行移位读取 |
| 7 | 模板多轮/工具路径 | 缺 `split`/`endswith` | 补 vendored Jinja 字符串方法 + render-diff |
| 8 | tokenizer added-token 布局 | 中段 64011–64401 + 尾部扩展 | 编解码往返 + 与 HF 逐 token 对比 |

## 六、Phase 0 剩余动作（幅值测量已完成）

1. ~~下载权重、跑 f32 前向、量残差/门乘~~——**已完成**（第八节），结论：q16 够用。
2. 仍需：一个短语料的参考 PPL（作为引擎对照基准），需要 torch/transformers 环境
   （本机未安装）或改用 llama.cpp GGUF 交叉对照。
3. 用同一组 messages 对比 `apply_chat_template` 与引擎渲染（必要时补渲染测试开关），
   并做 tokenizer 往返。
4. 把 PPL 与模板/分词结果填回本文件。

## 七、明确不做（本次范围外）

- 32k 上下文（Pi 的实际上限见 `docs/long-context.md`；SEQ 由内存决定）。
- ~~LFM2.5 的 Pythonic 工具调用协议~~——已支持（2026-11-11）：`parse_tool_calls()`
  解析 `<|tool_call_start|>[f(a="x")]<|tool_call_end|>`，参数转 JSON 对象，接入既有
  工具循环；`get_weather`（未知工具）与 `calculate_math`（执行成功）两条路径均实测。
- `<think>` 通道已接线（`--thinking` 打开、闭合块拆到 `reasoning` 并跨轮渲染），
  但模型在三次探测里都没有输出思考块。
- 许可证：LFM1.0（`license_name: lfm1.0`），分发前需单独确认条款。

## 八、残差与激活幅值实测（2026-11-11）

工具：`models/lfm2.5-230m/probe_residuals.py`（numpy f32 前向，按 transformers
`modeling_lfm2.py` 实现：conv 层 / 注意力层分支、q/k per-head RMSNorm、rotate-half
RoPE θ=1e6、GQA 16/8、SwiGLU、最终 `embedding_norm`、tied logits）。conv 对齐、
rope 约定、GQA 映射三项结构自检与独立公式逐元素一致（diff = 0）。

样本：3 组连续词表 id（seq 8）、10 组随机 id（seq 8）、32 token 长上下文、
以及一个含 `<|startoftext|>`/`<|im_start|>`/`<|im_end|>` 的真实 chat 形态提示（24 id）。

| 场景 | 残差峰值 `max|x|` | 相对 q16 rail (128) |
|---|---:|---:|
| 随机文本 id（10 样本） | 6.9 | 0.05× |
| 连续文本 id | 6.6 | 0.05× |
| **真实 chat 形态提示** | **25.3**（layer 5 起 ~25 平台） | **0.20×** |
| 全特殊 token 序列（人工最坏） | 29.5 | 0.23× |
| 32 token 上下文 | 29.5（同最坏） | 0.23× |

其它激活幅值（同一前向，取各场景最大）：

| 量 | 峰值 | 引擎格式与上限 | 余量 |
|---|---:|---|---:|
| FFN gate `silu(w1 h)` / up `w3 h` | 5.1 | I16_2（±2048） | ~400× |
| FFN 乘积 `silu*up` | 26.0 | q16（±128）或 I16_2 | 4.9×（若 q16） |
| conv `in_proj` 输出 | 4.4 | I16_2 | ~460× |
| conv 门乘 `B*x` | 11.4 | I16_2 | ~180× |
| conv 输出（卷积后） | 4.1 | I16_2 | ~500× |
| conv 门乘 `C*conv` | 17.7 | I16_2 | ~115× |
| q/k/v | 17.0 | 注意力纹理 | — |
| 注意力分数跨度 | 36.4 | f32 内 softmax | — |
| logits | 25.2 | f32 | — |

**结论：q16 残差路径足够，`--res q24` 不需要。** 峰值 29.5 是 14 层里第 5 层出现的
"massive activation" 平台（chat 形态为 ~25），比 supra2-100M 的 2300（18× rail）和
SmolLM2-135M 的 25982（204× rail）低两个数量级。q16 步长 1/256 在 25 的量级上仍有
约 12 bit 有效位。

注意：峰值对**特殊 token 的存在敏感**（纯文本 id 只有 6.9，含 `<|im_start|>` 的
chat 提示升到 25）。所有 chat 提示都以 `<|startoftext|><|im_start|>` 开头，因此
验收应以 chat 形态的 25–30 为准，而不是纯文本的 7。

## 九、Q4 词表头（embedding + tied logits）实测（2026-11-11）

packer 目前把 tied head **存两份**：`T_EMB`（全部 64402 行，q8 per-row）和
`T_LOGITS`（同一矩阵切 1536 行 tile，q8 per-tile）。对 LFM2.5 这是 2×66 MB。

| 方案 | 文件大小 |
|---|---:|
| q8 现状（head 两份） | ~296 MB |
| q4 现状（head 仍 q8，两份） | ~234 MB |
| q4 head（两份都 q4） | ~185 MB |
| q8 + head 去重（logits 直接读 emb 行 + per-row 平面） | ~230 MB |
| q4 head + 去重 | ~144 MB |

工具：`models/lfm2.5-230m/probe_head_q4.py`（把 head 按各方案量化后跑完整 f32
前向，比较 logits 与 greedy 续写；两个实质提示："The capital of France is"、
"List three colors:"）。

| 变体 | logits max\|d\| | cos | greedy 文本是否与 f32 一致 |
|---|---:|---:|---|
| q8 per-row（去重形态） | 0.54 | 0.9994 | 提示 1 一致；提示 2 在近似并列处翻转 |
| q8 per-tile（现状 logits） | 0.53–0.57 | 0.9993 | 一致 |
| **q4 仅 logits（输出侧）** | **0.70–0.78** | **0.9985** | **一致（两提示）** |
| **q4 仅 embedding（输入侧）** | **7.4–7.7** | **0.87–0.90** | **不一致** |
| q4 两侧 | 7.5–7.9 | 0.87–0.90 | 不一致 |

**结论：q4 词表头不可取。** 输出侧（logits）q4 无害——logits 幅值 ~25，误差
<1；但输入侧（embedding 行）q4 的误差经 14 层放大到 ~7.5，cos 掉到 0.87，文本
改变。tied 矩阵无法只 q4 一侧，而"emb q8 + logits q4 两份"是 103 MB，比去重的
66 MB 更差，无收益。

**真正的杠杆是去重，不是 q4。** head 存一份（q8 per-row）省 66 MB（296→230 MB），
且 per-row 精度严格优于 per-tile（每行的 scale 不被同 tile 的其它行拖累）；代价是
引擎要支持"从 emb 行渲染 logits"（`matvec_q8_pair_logits` 的 `wrow` 变体已存在，
但需要把 `T_LOGITS` tile 换成 emb 行视图），并用 PPL 重新定基线——这是跨模型的
优化，不属于 LFM2.5 支持本身。

若仍想要 q4 head，需要换量化方案（例如 per-row 裁剪后对称 q4，或 block-8），
并按 `AGENTS.md` 的规则用 PPL 而不是单条文本判定；当前 block-32 min/scale 在
这个头上误差偏大，原因是行内存在 ~0.26 的离群值，32 元组里一个离群值把步长
抬高到 0.017。

### 官方 GGUF 的 q4 配方（2026-11-11 实测）

`LiquidAI/LFM2.5-230M-GGUF` 发布 Q4_0（149.1 MB, 5.19 bpw）、Q4_K_M（153.4 MB,
5.34 bpw）、QAD-Q4_0（149.1 MB）三个 4-bit 文件。用 `models/lfm2.5-230m/gguf_report.py`
解析张量表：

| 张量 | Q4_0 | Q4_K_M | QAD-Q4_0 |
|---|---|---|---|
| `token_embd.weight` | **Q6_K**(55.1 MB) | **Q6_K**(55.1 MB) | **Q6_K**(55.1 MB) |
| `output.weight` | 无（tied，只存一份） | 无 | 无 |
| 全部 attn/ffn/conv 权重 | Q4_0 | Q4_K，另有 2/6 层 `attn_v` 与 6/14 层 `ffn_down` 用 Q6_K | Q4_0 |
| norm / `shortconv.conv` | F32 | F32 | F32 |

三点结论：

1. **官方三份 4-bit 文件全部把词表头留在 Q6_K**（6.56 bpw，55 MB），包括用 QAD
   专门训过的版本——与本文第九节的实测一致：输入侧 4-bit 的代价不可接受，连
   quantization-aware distillation 也没把它压到 4-bit。
2. **tied head 只存一份**，`output.weight` 不存在。我们的 packer 存两份（+66 MB）
   是纯浪费，与质量无关。

### 十、Pi 的 CMA 预算（2026-11-11，实测 ~170 MB）

用户实测：良好条件下约 **170 MB CMA** 可用；200 MB 量级的权重会挤掉 KV cache。
按此重算（权重 + KV(seq768, 6 层, i16) + FBO ≈ 3 MB）：

| 配置 | 权重 | 含 KV/FBO | 对 170 MB |
|---|---:|---:|---|
| q4（int16 平面）+ q8 head | 166.8 | 179.3 | 放不下 |
| q4 + q6 head（block-32, 6.5bpw） | 154.8 | 167.2 | 余 3 MB，太薄 |
| q4 + q6 head（per-row, 6.0bpw） | 150.7 | 163.2 | 余 7 MB，可试 |
| q4（int8 平面）+ q6 head | 144.5–142.5 | 157–155 | 余 13–15 MB，稳 |
| q4（int8 平面）+ q8 head | 156.6 | 169.0 | 仍放不下 |

由此：

1. **q6 head 是 Pi 的唯一出路**：q8 head 66 MB 占掉预算的 40%，不缩就没有 KV 的
   空间。q6 必须是**真 6-bit 纹理**（文件 6-bit、加载时展开成 q8 纹理对 CMA 无效），
   即前文路径 B（3–5 天）。
2. **去重是前提**：不去重还要多 66 MB，任何 head 方案都救不回来。
3. 留余量：170 MB 是"良好条件"，建议目标 ≤155 MB 权重；q4 平面从 int16 降到
   int8（或 group 32→64）还能再省 10 MB，`KV_PACK` 与小 `SEQ` 也能再省 3–5 MB。
4. 替代方案（不开新 shader）：logits 留在 CPU、head 常驻 RAM——CMA 只需 ~111 MB，
   但按 Pi CPU 实测（100M 模型 490–504 ms/token）head 的 66M MAC 会加 ~330 ms/token，
   且 168 MB 常驻 RAM 对 226 MB 的板子风险更大。
3. **按层混合精度在官方配方里是常态**（Q4_K_M 的 `attn_v`/`ffn_down` 部分 Q6_K）。
   我们的容器 dtype 本来就是逐张量的，`apply_precision()` 已支持按层/按部件切
   q8，所以"Q4_K_M 式配方"在引擎侧今天就可表达，只差 packer 的一个选项。

尺寸对照：官方 Q4_K_M 153.4 MB vs 我们 q4+去重（head q8）168 MB；差 15 MB ≈
head q8(65.9) − q6(55.1) 加上 block-32 平面的 0.25 byte/param 开销（k-quant 约
0.06）。要追平需要 6-bit head（新格式 + 新核），不建议；168 MB 是现实目标。
llama.cpp 的 Q4_K 是 256 元素超块，本引擎的核是 32 元素组，直接采用 k-quant
等于换量化器 + 换 shader，超出本评估范围。


## 参考

- [HF config.json](https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/config.json)
- [transformers LFM2 建模源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py)
- [Liquid 文档：LFM2.5-230M](https://docs.liquid.ai/lfm/models/lfm25-230m)
- [Liquid 文档：Chat Template](https://docs.liquid.ai/lfm/key-concepts/chat-template)
- [模型卡 README](https://huggingface.co/LiquidAI/LFM2.5-230M)
