# RWKV-7（Goose）支持评估

状态：**评估文档；未实现（2026-11-12）**。本文回答"把 RWKV-7 跑在本引擎上要做多少工作、值不值得、
卡在哪里"。数字分三类，不可混读：

- **来源事实**：上游源码/规格，出处为本地 llama.cpp b12283 副本（`refs/llamacpp/src-b12283/**`，
  该版本已支持 RWKV-7）、HF 配置或 URL；
- **现状事实**：本仓库当前代码/文档的直接观察（file:line）；
- **估算**：带假设的外推，给区间，不作承诺。

## 结论

- **工作量：18–32 人日（约 4–6.5 周，1 名熟悉本仓库的工程师）到"桌面正确、Pi 能跑能对话"；
  再加 6–12 人日做 GPU 融合与性能收尾，合计 24–44 人日（约 5–9 周）。中位约 25 / 34 人日。**
  最快的可用里程碑是 **2–4 周（11–19 人日）**：CPU 串行路径跑通、logits 对上 numpy f32 参考，
  **一个新 shader 都不写**。
- 这不是"加一个块"，是"加一个架构"：新容器 arch（22–28 个张量 id + 1 个 metadata key）、
  新层循环（12 个新算子，全是逐元素/归约）、新状态（**没有 KV cache**，改成每层 64×64 递推状态）、
  新分词器（贪心最长匹配，不是 BPE）、新验证参考（numpy f32 全序列前向）。
  规模约 **1500–2200 行 C++/GLSL + 500–700 行 Python**，是 LFM2 conv 块（该评估给 7–13 人日）的 3–4 倍。
- **风险低、体力活多**：RWKV-7 没有新数学——WKV7 递推只占每 token MAC 的 1.1–2.0%，
  而且逐算子都有 llama.cpp 的 f32 kernel 与图可以逐位对拍。真正的成本是"每个算子都要对一遍"
  和"无 KV 的生命周期"（状态、prefix cache、prefill、pair 路径）。
- 三个最大不确定项，按影响排序：
  1. **Pi Zero 只有 0.1B 放得下**。q8 容器的 CMA 约 141–152 MB（embedding 不上纹理；
     本仓库实测 q8 是 1.08 B/参数，按这个口径取上界），余量约 13–25 MB；
     层 q4 + q8 head 是 107 MB 更稳，但 q4 body 更慢。
     Pi 上 decode **0.55–0.85 s/token（q8）/ 0.70–0.99 s（层 q4）**，比 MiniMind3（252 ms）
     和 LFM2.5-230M（1040 ms）都不占优。0.4B 最小形态的 CMA 仍要 241 MB，**无解**。
  2. **"长上下文"的卖点在 Pi 上兑现不了**：decode 确实与上下文长度无关（0 KV），
     但 prefill 逐 token 串行：0.1B 的 512 token 要 2.5–3.7 min（不优化 6–11 min），
     4k 要 20–35 min。真正兑现它的是桌面 Xe：4k prompt 约 70–95 s，之后每 token 30–37 ms。
  3. **0.1B 的质量**：模型卡只给 5T+ token 训练与 UncheatableEval 链接，没有可直接对比的
     MMLU 数字（未核实）。如果它明显弱于同尺寸的 MiniMind3/LFM2.5，工程投入换不回可用产品——
     这是唯一可能让整件事不成立的变量。
- **建议**：做，但**桌面优先、Pi 只做"能跑"验证**。如果目标只是"多支持一个模型家族"，
  不建议（同样工日放在别处收益更高）；如果目标是"长上下文 / 无 KV cache / 无注意力"的推理栈，
  RWKV-7 是这个引擎唯一现实的路：注意力的 2048 上下文上限是 **KV 纹理高度**
  （`docs/long-context.md:14-16`），RWKV 把这一项从结构上删掉了。

## 一、RWKV-7 与本引擎假设的差别

引擎今天只有一个 transformer 块：RMSNorm + SwiGLU + RoPE + GQA，`Mat` 无 bias
（`engine/src/common.h:24-73`），没有 LayerNorm、没有逐元素激活库（`engine/src/shaders.h`）。
RWKV-7 把这一块整体换掉：

| 组件 | 本引擎今天 | RWKV-7 | 差别性质 |
|---|---|---|---|
| 跨 token 状态 | KV cache（每层 `seq × kv_dim`，纹理） | **WKV 状态**（每层 `heads×64×64` f32）+ token shift（每层 `2·dim`） | 新状态，无 KV |
| 混合方式 | softmax 注意力 + RoPE | 递推 `S = S·diag(w) + v kᵀ − (S κ)(κ⊙a)ᵀ` | 无 softmax、无 RoPE |
| 归一化 | RMSNorm | **LayerNorm(mean+var, w+b)** 5 处 + 每 head 的 LayerNorm 1 处 | 新算子 + bias |
| FFN | SwiGLU（融合 w1/w3） | `relu(Wx)²`（squared-ReLU） | 新激活 |
| 位置 | RoPE | 无（token shift + 衰减承载） | 少一件事 |
| 词表头 | 可 tie、可 q6 | **untied**，两套 65536×dim | 内存 + MAC |
| 低秩支路 | 无 | w/a/v/g 四条（rank 32–256） | 小矩阵多 |
| 分词器 | BPE（byte-level / SentencePiece） | **贪心最长匹配**（无 merge） | 新算法 |

每层的精确算子序（来源事实：`refs/llamacpp/src-b12283/src/models/rwkv7-base.cpp:8-136`，
逐条与 HF 的 `modeling_rwkv7.py` 对应）：

1. `att_norm = LayerNorm(w,b)(x)`；
2. `x_prev = concat(token_shift, att_norm[:-1])`，六路 lerp 得 `xr/xw/xk/xv/xa(/xg)`；
3. `r = W_r xr`、`k = W_k xk`、`v = W_v xv`（各 dim×dim）；
4. `w = exp(-0.606531·sigmoid(W_w2 tanh(W_w1 xw) + w0))`；
5. `v += (v_first − v)·sigmoid(W_v2(W_v1 xv) + v0)`（第 0 层记录 `v_first`）；
6. `a = sigmoid(W_a2(W_a1 xa) + a0)`，`g = W_g2 sigmoid(W_g1 xg)`（有门时）；
7. `kk = l2norm_per_head(k·k_k)`，`k = k + (a−1)·k·k_a`；
8. `(y, S) = WKV7(r, w, k, v, −kk, kk⊙a, S)`：每个 head、每行 i，
   `sa = ⟨a, S[i]⟩`；`S[i] = S[i]·w + v[i]·k + sa·b`；`out[i] = ⟨S[i], r⟩`
   （来源事实：`.../ggml/src/ggml-et/et-kernels/src/rwkv_wkv7_f32.c:3-9`）；
9. `y = per_head_LayerNorm(y)·ln_w + ln_b`；`rk = Σ_head(k·r·r_k)`；`y += v·rk`；
10. `y = W_o y`；`x += y`；`ffn_norm = LayerNorm(attn_norm_2)(x)`；
    `x += W_cm_value relu(W_cm_key·lerp2(ffn_norm))²`；
11. 存 `token_shift = [att_norm, ffn_norm]`（每层 2·dim）。

状态尺寸（来源事实：`llama-hparams.cpp:236-240`，`n_embd_s = n_embd·64`）：0.1B 2.36 MB、
0.4B 6.29 MB、1.5B 12.58 MB（f32），**与上下文长度无关**；token shift 每层 2·dim。

## 二、模型事实

官方发布在 HF 上有两套：`BlinkDL/rwkv7-g1`（pth 原始权重）与 `fla-hub/rwkv7-*`（HF/safetensors
格式，带 `modeling_rwkv7.py`）。G1 系列含 0.1B/0.4B/1.5B/2.9B/7.2B（另有 13.3B），
World 系列含 0.1B/0.4B/1.5B/2.9B/7.2B。下表数值来自各 repo 的 `config.json`（来源事实；
0.1B 取 `fla-hub/rwkv7-0.1B-g1`，其余取 `-world` 与 `-1.5B-world`）：

| 尺寸 | 层 × 宽 | FFN | 低秩 decay/a/v/g | 参数 | MAC/token | 权重 f16 | 词表 | 许可 |
|---|---|---:|---|---:|---:|---:|---:|---|
| 0.1B | 12×768 | 3072 | 64/64/32/128 | 191M | 143.7M | 382 MB | 65536 | Apache-2.0 |
| 0.4B | 24×1024 | 4096 | 64/64/32/128 | 452M | 393M | 902 MB | 65536 | Apache-2.0 |
| 1.5B | 24×2048 | 8192 | 96/96/64/256 | 1528M | 1409M | 3.05 GB | 65536 | Apache-2.0 |

- `head_dim` 恒 64，`hidden_act = sqrelu`，`norm_bias = true`，`norm_eps = 1e-5`，
  `tie_word_embeddings = false`，`max_position_embeddings = 2048`。参数量由 safetensors 字节数
  反算（191/452/1528M），与 llama.cpp 的类型表（190M/450M/1.5B）一致。更大的 2.9B/7.2B/13.3B
  是 L32-D2560 / L32-D4096 / L61-D4096，不在本文讨论范围。
  **低秩 rank 必须读 config，不能用公式反推**：llama.cpp 转换脚本的兜底公式
  （`calc_lora_rank = round(d^e·m/32)·32`，`conversion/rwkv.py:148-171`）对 0.4B 的 gate 给出 160，
  而 config 里是 **128**；0.1B/1.5B/2.9B 则与公式一致。
- 0.1B 的 MAC 是 MiniMind3 的 2.25×，但字节数是 3.0×——一半参数在两张 65536×768 词表矩阵上
  （head 占 0.1B 的 34.9% MAC、35% 字节）。
- 词表/分词器：RWKV World vocab `rwkv_vocab_v20230424.txt`（65536 条，尾部用 `[PADi]` 填满），
  自定义 `RwkvTokenizer`（repo 里**没有 tokenizer.json**）。token 0 =
  `<|rwkv_tokenizer_end_of_text|>`，BOS=0；EOS 文本是 `"\n\n"`（token 261，即 EOT）。
  各 release 的 `config.json` 对 bos/eos 的写法不一致（0.1B-g1 写 0/0，0.4B-world 写 1/2），
  **容器必须显式打包特殊 id，不能信 config**。
- 模板：HF `tokenizer_config.json` 里的 `chat_template` 很短，只用到 `for`/`if`/`elif`/
  `is defined`/字符串拼接与 `{{bos_token}}`——vendored Jinja 全部支持（现状事实：
  `engine/third_party/jinja/value.cpp:498-532`、`runtime.cpp:124-129`）。模板自带 BOS 文本，
  与 LFM2 一样要 `add_bos = 0`。注意模型卡要求 **prompt 末尾不要留空格**（会乱分词）。
- 质量：模型卡只给"World v3.5、5T+ token"与 UncheatableEval 链接，**0.1B/0.4B 没有任何公开
  benchmark**（未核实）。能拿到的只有二手汇总（EdgeRock 的 README）：G1i/G1j 的 1.5B MMLU
  49.7/49.3、2.9B 63.6/64.7。0.1B 的 12 层/768 里一半参数是词表，**更像续写/闲聊模型**
  （估算），必须用本仓库的 MC/IFEval/translate battery 自己量——这比找二手 MMLU 可靠。

## 三、引擎现状：可以复用的部分

| 能力 | 位置 | 对 RWKV-7 的用途 |
|---|---|---|
| MMCF v2/v3 容器 | `docs/checkpoint-container.md` | 未知 key/张量被跳过，新 arch 可增量扩展 |
| q8/q4/i16/q6 量化与纹理上传 | `engine/src/model.cpp`、`gpu.cpp:93-195` | 大矩阵（r/k/v/o、channel-mix）直接复用 |
| per-row q8 scale 平面 | `shaders.cpp:272-307` | RWKV 的 lerp 输入行间差异大，用得上 |
| f32 一维向量张量（dtype 0） | `model.cpp:556-559`、`rms_att = layers×dim`（:687） | 13 个逐通道向量（lerp/k_k/k_a/r_k/LN w+b/低秩 bias）可直接存 |
| 融合 GPU 层循环（0 回读） | `session.cpp:575-677` | 目标形态：RWKV 也全程在纹理上 |
| 串行层循环（matvec 在 GPU、逐元素在 CPU） | `session.cpp:386-569` | **正确性里程碑**：0 个新 shader |
| 残差加 / logits tile / top-k | `matvec_add`、`logits_grid.h` | 不改 |
| conv 状态（LFM2，prefix cache v4） | `session.cpp:200-205,249-259`、`docs/prefix-cache.md:150-172` | token shift / WKV state 的先例 |
| 模板（Jinja）与 BOS/EOS 元数据 | `chattmpl.cpp`、key 9/10 | RWKV 模板可直接渲染 |
| 采样 / server / 确定性工具 / 评估 battery | `server/`、`engine/scripts/eval_*` | 不改 |

## 四、差距清单（按组件）

### 4.1 容器与 packer

- 现状：packer **按名字拒绝 RWKV**（`engine/tools/mmcf_hf.py:165`：`"the RWKV block (a
  linear-attention recurrence, no softmax)"`），并且任何 `.bias` 张量都会被拒
  （`mmcf_hf.py:224-230`）。RWKV 的大矩阵没有 bias，bias 只出现在 LayerNorm 与 `w0/v0/a0`
  （来源事实：`rwkv7.cpp:52-115`）——所以**不需要给 matvec 加 bias**，这不是阻塞点。
- 新张量：RWKV 每层约 25 个张量（`r/k/v/o`、w/a/v/g 的低秩对、`w0/v0/a0`、`lerp_fused`、
  `k_k/k_a/r_k`、两个 LayerNorm 的 w+b、`time_mix_ln` w+b、channel-mix 的 lerp 与两个矩阵）；
  其中 `W_r|W_k|W_v` 可拼进现有 `0x10`（引擎本来就把 qkv 当一行读）、`W_o`→`0x11`、
  `W_cm_key`→`0x12`、`W_cm_value`→`0x15`，剩下 **22 个新 kind**（现状事实：id 是
  `kind(8)|layer(6)|part(2)`，`model.cpp:610`，已用 kind 只有 `0x01-0x08/0x10-0x19/0x20/0x30`）。
- metadata：新增一个 key（建议 16）承载 `arch` 与 RWKV 超参（head_size、四个 rank、两个 eps），
  旧 reader 跳过未知 key（`docs/checkpoint-container.md:100-134`）。
- **建议升 v4**：结构上 v2 装得下，但 v4 能让旧引擎在版本检查处干净拒绝（`model.cpp:385`），
  而不是加载到一半在 `need_shape(rows=0)` 处 `exit(2)`（`shaders.cpp:96-102`）。
  代价约 40 行 + 三处文档。
- packer：可照抄 `load_lfm2()` 的结构（`mmcf_hf.py:402-535`）：新写 `load_rwkv7()`（约 220 行）、
  在 `tensors_from_hf` 加分支（约 120 行）、放行 `check_source`、新写 RWKV 的残差峰值校准
  （约 180 行，可先用 `--no-rail-check` 跳过）。q4 group-32 对 64 列的低秩矩阵形状可行，
  但建议低秩矩阵保持 q8（只占参数 1–4%，误差要进 sigmoid/exp）。

### 4.2 loader 与层循环

- `Model::load` 的三个 switch 都要加分支（主解析 `model.cpp:657-728`、per-row scale 归属
  `:736-757`、q4 归属 `:829-836`），否则 per-row 平面会被判成"不匹配的矩阵"。
- `Session` 新增一条 RWKV 路径：串行版可以直接在 `forward()` 里加（`session.cpp:386-569`），
  每层 13–14 次 `matvec_s` 回读；融合版要把 6 路向量画进同一张"向量银行"再回读一次
  （现有 `matvec_to_row` + `read_bytes(h>1)` 就能表达，`gpu.h:56`、`gpu.cpp:1226-1233`）。
- **两个一行级守卫**：`pair_ok()` 必须对 RWKV 返回 false（`session.cpp:691-735`，否则会调到
  `attention_block`）；prefix cache 的 `!kv_cache_.empty()` 闸（`session.cpp:212,273`）要么放宽、
  要么明确报"RWKV 不支持"。
- 形状：0.1B 的 FFN 3072 → 2×1536 安全；**0.4B 的 4096 → 2×2048 正好落在"2048 行的 draw
  会静默出错"那条线上**（`docs/performance.md:2757-2763`），需要 3 块，而 `split_ffn` 只有
  a/b 两块（`model.cpp:111-155`）；1.5B 的 8192 要 4 块，引擎结构不支持。

### 4.3 GPU 核

缺的 **12 项全是逐元素/归约**，没有一个是矩阵乘（现状事实：`engine/src/shaders.h` 的清单里
没有它们）：mean+sumsq 的 LayerNorm（现有只有 RMS，`shaders.cpp:1362-1404`）、per-head LayerNorm、
`tanh`（`engine/src` 里 0 处，ESSL 1.00 也没有，要用 `exp` 写）、独立 sigmoid/exp、
逐通道 lerp（`a*x+b`，`mul_act` 只有 `a*b`）、per-head l2norm（≈ `att_qnorm_scale` 乘 √64、
换 eps）、`k` 的 axpy 更新、per-head 标量 `rk` 的归约与广播、squared-ReLU、逐通道 bias 加。
WKV7 在 GPU 上要 3 个 pass/层（sa / 更新 / out），每个 fragment 内 64 长归约；
VC4 没有浮点纹理（`docs/engine-internals.md:188-197`），state 只能定点 + ping-pong。

### 4.4 状态与 prefix cache

- WKV state 在 CPU 就是 `vector<float>`（dim·64/层，0.1B 196 KiB/层，12 层 2.36 MB），
  先例是 `conv_state_`（`session.h:47`、`session.cpp:109-113`）；放 GPU 则要 2 份纹理轮转。
- prefix cache：现在的第一道闸是"必须有 KV cache"，RWKV 永远被拒。扩展成状态快照要：
  版本 5 + 新 flag bit（每层写 WKV state 与 token shift）、头部字段能表达"rope/KV_PACK 不适用"、
  加载后 `forget_state()`。conv 的 v4 是可直接照抄的模板。

### 4.5 分词器

- 现状：`.tok` blob 是 `(score, len, bytes)`，`Tokenizer::encode` 末尾**无条件**调
  `bpe_merge`（`tokenizer.cpp:148-149`），而 `bpe_merge` 在全零 score 的 65536 词表上会乱合并
  （`tokenizer.cpp:102-125` 从 `best=-1e30` 起步）——必须先加"RWKV 模式"旁路。
- 新算法：反转义每个词表条目（`\xNN`/`\t`/`\n`，**不是 llama2.c 的 `<0xNN>`**）→
  贪心最长匹配 → 命中即取。不一定要真 trie：用现有 `lookup_` 哈希表从 `min(maxlen, 剩余)`
  向下试即可（约 60 行）。新打包器 `make_tok_rwkv.py` 读 `rwkv_vocab_v20230424.txt`（约 60 行）。
- 特殊 id：BOS=0、EOT=261、无 dummy prefix（`add_prefix_space=false`）；decode 直接输出裸字节。

### 4.6 模板

RWKV 的模板（HF `tokenizer_config.json`）只需要 `for/if/elif/is defined/`、字符串拼接与
`bos_token`，vendored Jinja 都在。**但**只做了静态核对（未实跑），且
`docs/lfm2.5-230m-assessment.zh-CN.md:59-61,144` 里"缺 split/endswith"这句已经过期
（`engine/third_party/jinja/value.cpp:799-836` 已实现）——这是一处文档漂移，顺手修。

### 4.7 验证工具

`ref_forward.py` 是 llama 专用（张量名与 GQA/SwiGLU 写死，`ref_forward.py:60-114`），
RWKV 需要新的 numpy 全序列前向 + 逐阶段 dump + state 快照（约 180–260 行）；
`STAGE_DUMP` 的阶段名要新定义；`regression.sh` 的 pin 是 per-model 环境变量，需要一组 `RWKV_*`；
新增的 `RWKV_DUMP`/`STATE_DUMP` **必须同时写进 `docs/test-hooks.md`**，否则
`scripts/regression.sh` 第 0 步的 `docs_check.sh` 会失败（它比对 `getenv()` 清单与文档）。
新 GPU 核（WKV、groupnorm）在 VC4 上的采样行为要按 `probes/wrow_probe`/`conv_probe` 的方式先写 probe。

## 五、工作量分解

| 阶段 | 内容 | 人日 | 锚点 |
|---|---|---:|---|
| 0 | 参考实现 + 数值测量 | 2–4 | numpy 前向 180–260 行并与 llama.cpp 对拍；量残差/state 幅值；分词/模板往返 |
| 1 | 容器 + packer | 3–5 | 22 个新 kind + key 16 + v4；`load_rwkv7` ~220 + 分支 ~120 + rail ~180 |
| 2 | loader + CPU 层循环（12 算子 f32） | 6–10 | `model.cpp` 250–350 + `session.cpp` 300–420；LFM2 同类（loader+session+conv）约 540 行 / 3–7 天 |
| 3 | 分词器 + 模板 + CLI/server | 2–4 | 160–190 行 C++ + 60 行 Python + render-diff |
| 4 | 验证 + 质量基线 | 3–5 | ref 对拍、PPL、MC/IFEval/translate、regression pin、dump 钩子 |
| 5 | Pi 部署 + 实测 + 文档 | 2–4 | q8/q4 打包、交叉编译、上传 40–60 s、计时、CMA |
| **小计（可用版本）** | | **18–32** | ≈ 4–6.5 周 |
| 6 | GPU 逐元素核 + 融合层循环 + WKV7 GPU + pair prefill | 6–12 | 450–600 行 GLSL + 200 行胶水；WKV 150 GLSL + 250 胶水/probe |
| **合计** | | **24–44** | ≈ 5–9 周 |

校准依据（现状事实，`git show --numstat`）：LFM2.5 全套约 2100 行（引擎+packer+文档+Pi），
其评估给 7–13 人日；MoE 核心约 940 行；per-row q8 scale 那条数值调查约 1550 行。RWKV 的
引擎侧约 1500–2200 行、packer 侧 500–700 行，落在"再加一个模型家族"的量级，而不是"加一个块"。

**必须先做（正确性）**：阶段 0、1、2、状态容器与 token shift、验证钩子；外加两个守卫
（`pair_ok()` 拒绝 RWKV、prefix cache 明说支持/不支持）。
**可后置（按收益排序）**：GPU 逐元素核（把 14 次回读/层降到 1 次）、融合层循环、
WKV7 GPU（VC4 上性价比最低）、prefix cache 状态快照、q6 head。

## 六、性能与内存

成本模型（推导自本仓库实测）：Pi Zero 的 q8 decode `T ≈ 78 ms + 2.27 ms/MB` 容器
（5 个实测点拟合，R²=0.91）；**q4 是 ALU 受限**（有效 120–165 MB/s），
所以 q4 只省内存不提速。有效吞吐的实测跨度是 110–348 MMAC/s，不能按 MAC/s 线性外推。

| 模型 | MAC/token | Pi Zero decode | Xe decode |
|---|---:|---:|---:|
| MiniMind3 64M q8（实测锚点） | 63.9M | 252 ms | — |
| LFM2.5-230M q4（实测锚点） | 165.6M | 1040 ms | — |
| RWKV-7 0.1B | 143.7M | **0.55–0.85 s**（q8，CMA 141–152 MB）/ **0.70–0.99 s**（层 q4 + q8 head，CMA 107 MB） | **30–37 ms**（q8） |
| RWKV-7 0.4B | 393M | 放不下（最省形态 CMA 241 MB） | **70–85 ms**（q8） |
| RWKV-7 1.5B | 1409M | 放不下 | **160–200 ms**（q4） |

- **Pi 的判定**：0.1B q8 能跑但只剩约 13–25 MB CMA；层 q4 + q8 head 留 59 MB 但更慢。
  0.4B 的最小 4-bit 形态（无 plane 的理论下界 226 MB）已经超过预算，**没有任何精度能放下**。
- **logits 是 0.1B 的大项**：65536 词表 = 43 个 tile（`kLogitsGrid=1536`），占 34.9% 的 MAC，
  Pi 上约 110–150 ms/token。head 是 untied，`--head-dedup` 不适用。
- **与现成 GGUF 的量级交叉核对**（来源事实，第三方转换）：0.1B 的 Q8_0 是 202.5 MB、
  Q5_K_M 148.9 MB、Q4_K_M 138.8 MB；0.4B 的 Q4_K_M 已 303.3 MB。这和我们算的容器量级吻合
  （q8 191–206 MB、层 q4 119 MB / 157 MB），也独立说明 **0.4B 在 Pi 上无解**。
- **prefill**：递推**不必**废掉 `PREFILL_PAIR`——matvec/FFN 没有跨位置依赖，只有 WKV 与 token
  shift 逐位置；正确做法是 matvec 走 pair、WKV 在 pair 之后跑两遍。把整条 pair 路径拒掉才损失
  +15–26%（MoE 实测 +26%，`docs/performance.md:4542-4544`）。
  0.1B 在 Pi 上 512 token 的 prefill：优化后 2.5–3.7 min，串行 6–11 min（含 3–12 s 固定项）。
- **长上下文**：decode 与深度无关（0 KV、state 固定 2.36 MB）；Xe 上 4k prompt 70–95 s、
  之后每 token 30–37 ms，这是 RWKV 相对现有 transformer 的真正优势所在。
- **形状**：所有 dim/rank/词表都是 32 的倍数，q4 group-32 无一处需要 pad；纹理 2048 与
  FFN 2 块规则见 §4.2。

## 七、风险清单

| # | 风险 | 证据 | 缓解 |
|---|---|---|---|
| 1 | **WKV state 的定点精度**：state 是对 k⊗v 的衰减累加，动态范围未测；VC4 无浮点纹理 | `xformat.h:18-26`、`engine-internals.md:188-197` | Phase 0 用 numpy 量 state 幅值；先用 CPU f32 state 保正确性；GPU 用 I24_1 或 per-head scale |
| 2 | **Pi 上"无限上下文"兑现不了**：prefill 逐 token | 本评估 §6；`docs/performance.md:3530-3536` 的 prefill 固定项 | 桌面优先；Pi 只做 ≤512 token 的演示；文档写清 prefill 墙 |
| 3 | 0.4B 的 FFN 4096 = 2×2048 正好踩中 VC4 静默出错的行数 | `docs/performance.md:2757-2763`、`model.cpp:111-155` | 0.4B 只做桌面（Xe 上限 16384）；或先给 `split_ffn` 加 3 块 |
| 4 | Pi CMA 预算本身是 ±20 MB 的估计 | `docs/performance.md:4658-4661` 记过同一容器跑完还有 195–201 MB free | 以实测为准；先试 q8，再退层 q4 |
| 5 | q4 更慢，可能"省内存换速度"不划算 | MiniMind3 q4 388 vs q8 252 ms | Pi 上优先 q8 + 缩词表头（q6 head，需重测 Pi 上是否仍更慢） |
| 6 | 分词器无 merge 旁路会静默乱合并 | `tokenizer.cpp:102-125,148-149` | 先加显式模式；用 HF `RwkvTokenizer` 逐 id 对拍 |
| 7 | 模板未实跑（只静态核对） | §4.6 | Phase 0 用 `apply_chat_template` 做 render-diff |
| 8 | 0.1B 的质量未知 | 模型卡只有 UncheatableEval 链接 | Phase 0 用本仓库 MC/IFEval/translate battery 跑一遍，和 MiniMind3/LFM2.5 同表比较 |
| 9 | 文档漂移：LFM2 评估里"缺 split/endswith"已过期 | `value.cpp:799-836` | 顺手修正，避免下一个人重新踩 |
| 10 | 许可证：Apache-2.0，但训练数据含 Oscar/Pile 等，分发前仍需确认 | 模型卡 tags | 与现有模型同等级别处理 |

## 八、Phase 0 待测清单（把"未核实"变成事实）

1. 拉一个 0.1B 与 0.4B 的 HF 权重（已确认可访问：`fla-hub/rwkv7-0.1B-g1` /
   `-0.4B-world` / `-1.5B-world`），跑 numpy f32 全序列前向，与 llama.cpp 的同一 prompt
   logits 对拍（top-1 一致、cos > 0.999）——这一步同时产出参考实现。**f32 oracle 用官方
   `RWKV-v7/rwkv_v7_numpy.py`**（单 token 前向，公式最干净）；**不要**用
   `train_temp/rwkv7_train_simplified.py` 交叉验证：它的 decay 参数化是 `-softplus(...)-0.5`，
   与已发布权重的 `exp(-σ(z)/√e)` 不是同一个参数化，对不上不是 bug。
   llama.cpp 侧也有现成 GGUF（`HaloWang/rwkv-weights`、`shoumenchougou` 的 RWKV7-GXX-GGUF），
   不必自己转换；注意 llama.cpp 的 recurrent state 是 **f32**，它能验证公式，验证不了定点可行性。
2. 量残差峰值、`att_norm`/`ffn_norm` 幅值、WKV state 的 |max| 与分布，回答"q16 rail 够不够"
   （LFM2 的教训是 embedding 只有 0.05 量级时要 `ACT_F32=1`，`session.cpp:345-354`）。
3. 用 HF `RwkvTokenizer` 与 vocab 文件做逐 id 往返（含空行/空格/中文/emoji），确认
   `maxlen` 与特殊 id；确认 `add_prefix_space=false` 的效果。
4. `apply_chat_template` vs 引擎渲染的 render-diff（含 `enable_thinking` 分支）。
5. 质量基线：MC（中英）、IFEval、translate 三套跑 0.1B/0.4B，与 MiniMind3、LFM2.5 同表。
6. 在 Q4 上测一次 0.1B 的 PPL 与 greedy 文本，确认 q4 是否掉点（RWKV 没有 QAD 权重）。
7. 若要做长上下文：确认 Xe 上 4k prompt 的 prefill 实测与 state 数值稳定性。

## 九、明确不做（本评估范围外）

- 训练/微调/蒸馏；量化感知训练（QAD）权重。
- 1.5B/2.9B/7.2B 上 Pi；0.4B 上 Pi（内存不成立）。
- 并行/分块 WKV scan 的训练式前向、多序列 batch（引擎目前 batch=1）。
- 超过 2048 的 transformer 路径改造（那是 `docs/long-context.md` 的题目）。
- 用 RWKV 替换现有默认模型；默认仍是 MiniMind3／LFM2.5。

## 十、建议与取舍

- **路线 A（推荐）**：桌面优先。先做阶段 0–4，在 Xe 上把 0.1B/0.4B 跑对、跑出质量数字，
  再决定是否投入阶段 6 的 GPU 优化。桌面 0.1B 30–37 ms、0.4B 70–85 ms，已经能当"长上下文
  小模型"用（0 KV、4k prompt 70–95 s）。
- **路线 B（Pi 优先）**：只做 0.1B q8（CMA 141–152 MB）的"能跑"验证，接受 0.55–0.85 s/token 与
  512 token 的 prefill 墙。它不会替代 MiniMind3/LFM2.5，价值在"长上下文/无 KV"这条演示线。
- **路线 C（不做引擎）**：如果目标只是"在 Pi 上评估 RWKV-7 的质量"，llama.cpp 已经支持
  RWKV-7，而 Pi 上已经有一个交叉编译好的 `llama-cli-armv6`（见 `AGENTS.local.md`）：
  下一份 GGUF 就能跑，0.5–1 人日 + 文档。代价是放弃本引擎的 q4/q8 纹理管线、server 集成与
  确定性工具；速度上 GPU 路径可能更快，但没量过。**给决策者的这句话最重要：
  "想要 RWKV-7 的能力"和"想要 RWKV-7 在这个引擎里"是两个不同的问题，工日差 20–40 倍。**
- **顺序建议**：如果确定要做引擎侧，先做阶段 0（2–4 人日）——如果那一步测出 0.1B 质量不可用、
  或 state 需要 f32 纹理、或残差超 rail 太多，后面 20+ 人日的方案都要重排；
  Phase 0 是这次评估的止损点，也是最有信息量的 2–4 天。

## 十一、0.4B 实测（2026-11-17，P2 阶段 1）

本文 §二/§六 的 0.4B 行是估算。2026-11-17 拿到了官方 pth 并打完容器，实测如下
（细节、命令与 sha256 在 `models/rwkv7/p10-phase1-0.4b-report.md`）：

| 项 | 估算 | 实测 |
|---|---|---|
| 权重 | f16 902 MB | `rwkv7-g1d-0.4b-20260210-ctx8192.pth` 901,776,749 B，**全部 bf16**（不是 f16：尾数 8 位） |
| 参数 | 452M | **450.8M**（丢掉的 L0 value 低秩对）；emb/head 各 67.1M |
| 低秩 | 64/64/32/128 | 一致；gate=128 由 config 给出，不经过 llama.cpp 的兜底公式（它给 160） |
| q4 容器 | ~335 MB | **334,871,084 B（334.9 MB）**，913 张量 |
| q8 容器 | — | **453,364,524 B**（1.005 B/参数；0.1B 是 1.009） |
| 贴图 | q4 CMA ~266 MB | q4 减去留在 CPU 的 embedding 67.1 MB = **267.8 MB**（q8 是 386.3 MB） |
| 残差峰 | 未知 | **80.2**（rail 128，余量 1.6×；0.1B 是 37.12 / 3.4×） |
| sqrelu 峰 | 未知 | **76.3**（余量 1.68×；0.1B 91.95 / 1.39×） |
| 最终 LN 峰 | 0.1B 315.6 → key 17 | **304.19**，同样 k=2 |
| WKV state \|max\| | 未测 | **30.7**（层 2.5–30.7；0.1B 51.8） |

结论未变：Xe 上能加载能出话（三个容器都过加载检查），Pi 仍然放不下（q8 贴图 386 MB）。
0.4B 的残差余量只有 0.1B 的一半（1.6×），是 0.4B 上 rail 的第一处观察点；质量判据仍是 PPL。

## 参考

- 本地 llama.cpp b12283 副本：`refs/llamacpp/src-b12283/src/models/rwkv7.cpp`（超参/张量表）、
  `src/models/rwkv7-base.cpp`（算子序）、`ggml/src/ggml-et/et-kernels/src/rwkv_wkv7_f32.c`
  （WKV7 f32 参考）、`conversion/rwkv.py`（HF→GGUF 命名与缺省超参）、
  `src/llama-vocab.cpp:1353-1463`（RWKV 分词器）、`models/templates` 下的 world 模板
- [RWKV-7 论文 (arXiv 2503.14456)](https://arxiv.org/abs/2503.14456)
- [fla-hub/rwkv7-0.1B-g1](https://huggingface.co/fla-hub/rwkv7-0.1B-g1)、
  [fla-hub/rwkv7-0.4B-world](https://huggingface.co/fla-hub/rwkv7-0.4B-world)、
  [fla-hub/rwkv7-1.5B-world](https://huggingface.co/fla-hub/rwkv7-1.5B-world)（config.json、
  tokenizer_config.json、模型卡）
- [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1)（官方 pth 与模型卡）
- 本仓库相关文档：`docs/long-context.md`（2048 的来源）、`docs/checkpoint-container.md`、
  `docs/engine-internals.md`（VC4 限制）、`docs/prefix-cache.md`、
  `docs/lfm2.5-230m-assessment.zh-CN.md`（同量级的评估格式与工期锚点）
