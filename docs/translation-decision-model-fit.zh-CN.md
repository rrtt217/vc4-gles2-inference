# 20–150M 翻译 / 判断模型与 VC4-GLES2 引擎的适配评估

状态：研究笔记，2026-11-02。本文回答两个问题：

1. 20–150M 参数区间里，哪些翻译模型和“类 Jev”判断模型相对适合当前引擎？
2. 要把它们跑起来，还需要补哪些适配？

结论先行：

- **翻译**：同尺寸下最稳的 off-the-shelf 选择是 **OPUS-MT 77M（Marian，en↔zh）**。它不是 decoder-only，需要给引擎补 encoder + cross-attention + LayerNorm/bias/非门控 FFN + SentencePiece；参考实现约 4–6 周，优化再加 2–3 周。若不改架构，只能走 **MiniMind3 64M 蒸馏/微调**，零架构改动，但质量与延迟都要妥协。
- **判断**：固定标签、低延迟的场景，**3–25M 的 BERT/RoBERTa 编码器**最合适（UER L2/L4、BERT-tiny、MiniLM-L6、bge-small-zh）；0.2–2 s 可出结果。类 Jev 的**动态 criteria/options** 则要么用 64M decoder-only 做选项打分（Pi 上 5–15 s），要么用 cross-encoder 逐选项打分；**150 ms 级实时判断在 Pi Zero 上不现实**。
- **不建议**：100M+ 编码器在 Pi 上做在线判断；250k 词表模型（XLM-R/mDeBERTa/mT5/NLLB）；Prompt Guard 2 “86M”（实际 278M）；ModernBERT-base 在 Pi 上（桌面 GLES2 可以）。

---

## 1. 引擎现状：为什么不能“直接换模型”

当前引擎是 llama2.c 风格的 **decoder-only、逐 token、causal** 推理器，优势是 MiniMind3 已经跑通；限制也很明确。

| 约束 | 现状 | 对候选模型的影响 |
|---|---|---|
| 只有 decoder-only causal attention | `Session::attention` 只回看 `t <= pos`（[session.cpp:364](../engine/src/session.cpp)） | BERT/Marian encoder 的**双向** attention 需要新的“整段编码”调度，不能只改 mask |
| prefill 逐 token | `session.step()` 一次一个位置 | encoder 成本 = N × 单 token；没有 batch prefill |
| 归一化只有 RMSNorm | 无 LayerNorm、无 bias | Marian/BERT 全部线性层和 LayerNorm 都带 bias |
| FFN 只有 SwiGLU/融合 W13 | 没有非门控 FFN | Marian 是 `swish(W1x)`，BERT 是 `GELU(W1x)` |
| 位置编码只有 RoPE | 无正弦/可学习绝对位置 | Marian 用固定正弦，BERT 用 learned position |
| 只有 self-attention | 无 cross-attention | Marian decoder 的核心结构缺失 |
| `Mat` 没有 bias | 见 [common.h](../engine/src/common.h) | qkv/wo/FFN/LN 的 bias 都要补 |
| logits 按 1536 行分 tile | 65k 词表 = 43 个 tile | OPUS 每 token 有 33M MAC 花在 logits 上 |
| tied embedding 只支持 legacy | Q8 格式仍单独存 logits | OPUS tied 权重会被复制一份 33 MB |
| EOS 硬编码为 1/2 | [engine.cpp:262](../engine/src/engine.cpp)、`:380`、`:455`、`:550` | OPUS 的 EOS 是 0，而 token 2 是逗号，会**在逗号处停止** |
| 单模型 | 一个 `Model`/`Session` | 翻译 + 判断同时驻留需要多模型或共享 encoder |
| 纹理上限 2048 | `GL_MAX_TEXTURE_SIZE=2048` | Marian FFN 2048 行刚好；BERT 3072 行必须切块 |
| Q4 要求 `cols % 32 == 0` | [model.cpp:48](../engine/src/model.cpp) | Marian 512、BERT 768/3072 都满足 |
| tokenizer 只有 BPE | 无 SentencePiece unigram、无 WordPiece | OPUS/BERT 的 tokenizer 都要补 |

已经测过的基线（Pi Zero，q8，greedy，见 [performance.md](performance.md)）：

| 模型 | 规模 | 每 token | 说明 |
|---|---:|---:|---|
| MiniMind3 q8 | 64M | ~295 ms | GPU_LAYER=1，seq 768 |
| MiniMind3 prefill | 64M | ~333 ms/token | 73 token 24.3 s，说明 prefill ≈ decode |
| TinyChat15M q8 | 15M | ~135 ms | 32k 词表 logits 占比很大 |
| MiniMind3 CPU-attention | 64M | ~559 ms | 不用 GPU_LAYER 的代价约 1.9× |

下面的延迟估算都以“约 120–220M MAC/s 有效吞吐”折算；这是从上面三组实测反推的，不是对新模型的实测，误差按 ±50% 看。

---

## 2. 翻译模型：20–150M 的选项

### 2.1 候选总表

| 模型 | 参数 | 结构 | 词表 | 质量 | 引擎今天能跑？ | 主要工作 | Pi 估算（20 token 句子） |
|---|---:|---|---:|---|---|---|---|
| **Helsinki-NLP/opus-mt-en-zh**（Apache-2.0） | **77M**（44M 非嵌入 + 33M tied embedding） | Marian enc-dec，6+6 层，d=512，FFN 2048，8 heads | 65,001（源/目标各 32k SPM，联合词表） | Tatoeba en→zh BLEU 31.4 | 否 | encoder + cross-attn + LN/bias + SPM | 6–13 s |
| Helsinki-NLP/opus-mt-zh-en（CC-BY-4.0） | 77M | 同上 | 65,001 | 同量级 | 否 | 同上 | 6–13 s |
| Helsinki-NLP/opus-mt-tc-big-en-zh | ~230M | Marian big | 65k | 更高 | 否 | 同上 + 内存 | Pi 不现实 |
| M2M100 / NLLB-600M | 418M / 600M | enc-dec | 128k / 256k | 强 | 否 | 超出区间 | 不现实 |
| mT5-small | ~300M | enc-dec | 250k | 多语 | 否 | 超出区间 | 不现实 |
| **MiniMind3 64M 微调/蒸馏** | 64M | decoder-only Qwen3 | 6,400 | 取决于训练 | **是**（现有格式） | 数据 + 训练；英文源 tokenization 差 | 6–12 s |
| SmolLM2-135M 微调 | 135M | decoder-only Llama | 49,152 | 取决于训练 | 接近（要 converter） | 30 层，logits 重；训练 | 15–25 s |
| stories15M-llama2-enzh（仓库已有） | 21.9M | decoder-only llama2 | 55,296 | 玩具级，实测不能稳定翻译 | 是（legacy） | 无 | 3–6 s，但质量不可用 |

### 2.2 为什么 OPUS-MT 77M 是“甜点”

- **结构刚好卡在引擎的纹理限制内**：FFN 2048 行 = 2048 纹理高度上限；qkv 1536 行；d=512、8 heads、head_dim 64 都很规整。
- **词表效率高**：源/目标各 32k SPM，英文和中文都不会被拆得太碎；联合词表 65,001。
- **质量有公开基线**：Tatoeba en→zh BLEU 31.4、chrF 0.268（模型卡）。
- **同参数下比 decoder-only 更省**：
  - decoder-only 翻译：源 N + 目标 M 个位置，每个位置都过全部层 + 词表 logits。
  - encoder-decoder：encoder 只过 N 个位置（6 层、无 logits），decoder 过 M 个位置。
  - 以 20+20 token 为例：MiniMind3 约 `40 × 64M = 2.56B` MAC；OPUS 约 `20 × 19M + 20 × 42M = 1.22B` MAC（目标侧 logits 只算 32k）。约 **2×** 差距。
- **内存**：q8 tied 约 78 MB，q4 约 60 MB；如果只保留 32k 目标侧 logits，q8 约 60 MB、q4 约 43 MB。作为对比，MiniMind3 64M q8 是 66 MB，已经在本机跑通。

### 2.3 OPUS-MT 的“隐藏坑”

1. **tied embedding 不能按普通 Q8 转换**：Q8M1 会在文件里单独存一份 logits tile（65k×512 ≈ 33 MB），等于把 embedding 复制一遍。必须加 `tied_logits` 标志，让 converter 直接把 embedding 数据切成 logits tile 上传（不再单独存一份），或只导出目标侧 31,959 个 piece。否则 q8 文件会从 78 MB 涨到 111 MB。
2. **目标侧词表不是连续的 32k**：联合词表按频率混合，目标 SPM 的 31,959 个 piece 散布在 0–64,853 之间。若要做“只算目标 logits”，需要一次 row-gather/重排，不能简单切前 32k 行。
3. **EOS / decoder start 与 MiniMind 完全不同**：OPUS 用 `decoder_start_token_id=65000`（`<pad>`）、`eos_token_id=0`（`</s>`）、`pad=65000`；引擎硬编码的 1/2 会立刻停错。
4. **bias + LayerNorm + 非门控 FFN**：Marian 的每个线性层都有 bias，LayerNorm 有 gamma/beta，FFN 是 `swish(W1x+b1)` 再 `W2`；这些在当前 `Mat`/`Session` 里都没有。
5. **tokenizer 是 SentencePiece unigram，不是 BPE**：源/目标各一个 32k unigram 模型，还有 `>>cmn_Hans<<` 这种整词特殊 token（联合词表 ID 5）。现有 BPE merge 逻辑无法复现。
6. **post-LN**：Marian 是 `x = LN(x + sublayer(x))`，不是 pre-LN；顺序错了质量会明显掉。
7. **scale_embedding=true**：Marian 的 embedding 要乘 `sqrt(512)≈22.6`；不乘会让整个模型的数值尺度错掉。
8. **长度**：模型最大 512 个位置；源 + 目标要留余量，引擎 `seq` 建议 ≥ 512。

### 2.4 如果不想动引擎：MiniMind3 64M 路线

- 现有格式、现有 kernel、现有 tokenizer 都能直接用；只需要训练数据和 prompt 模板。
- 建议做 **OPUS-MT 蒸馏**：用 OPUS 生成大规模 en↔zh 平行句，再蒸到 MiniMind3 64M；比从零训练靠谱。
- 主要问题：
  - 6,400 词表偏中文；英文源会被拆成很多 token，源长度和 prefill 成本都上升。
  - 8 层 768 的容量做双向理解 + 生成，质量上限低于 6+6 的 encoder-decoder。
  - 延迟 6–12 s/句，和 OPUS 路线接近，但质量通常更低。
- 适合“先跑通产品、后换模型”的阶段，不适合作为最终质量方案。

---

## 3. 判断模型：类 Jev 的约束

Jev 的接口不是“生成一段话”，而是三种 typed question：

- **Choice**：从最多 255 个选项里选一个，返回 winner、每个选项的概率、confidence。
- **Score**：在 2–10 个有序等级上返回概率分布和期望值（可以落在两级之间）。
- **Noul**：返回 P(yes)。

它强调 state + instructions + criteria 的结构化输入，官方用例页宣称 150 ms 级延迟（服务端）。这带来两个本地化难点：

1. **动态 options/criteria**：选项描述是请求时给的，不是固定类别。固定 255 类分类头无法直接支持新选项。
2. **校准概率**：不仅要 argmax，还要可用的 confidence；需要温度缩放/校准集，不能只取 softmax。

因此“类 Jev”在本引擎上有三条实现路径：

| 路径 | 结构 | 动态选项 | Pi 延迟 | 适合 |
|---|---|---|---|---|
| A. 固定标签分类头 | 3–25M 编码器 + Linear(N) | 否（标签固定） | 0.2–2 s | 意图/语言/风险/yes-no，高频 |
| B. decoder-only 选项打分 | 64M 生成模型，读 state+options，算各选项 log-prob | 是（少量选项） | 5–15 s | 低频、复杂 criteria |
| C. cross-encoder 逐选项打分 | 4–22M 编码器，对 (state, option) 打分 | 是 | 2–8 s（≤5 选项，4–10M） | 中等频率、选项不多 |

**255 个动态选项在 Pi Zero 上不可行**：即使每次只跑 5 个 token，255 次前向也要几十秒。实际做法是“检索短名单 → 只对 5–10 个候选做 B/C”，或者把选项收敛成固定 taxonomy。

### 3.1 固定标签编码器候选

| 模型 | 参数 | 深度/宽度 | 词表 | 每 token MAC | Pi 估算（16 token） | 备注 |
|---|---:|---|---:|---:|---:|---|
| uer/chinese_roberta_L-2_H-128 | ~3M | 2×128 | 21k | 0.39M | 0.2–0.3 s | 中文最快 |
| prajjwal1/bert-tiny | 4.4M | 2×128 | 30.5k | 0.39M | 0.2–0.3 s | 英文最快 |
| **uer/chinese_roberta_L-4_H-256** | **~8.6M** | **4×256** | **21k** | **3.15M** | **0.5–0.8 s** | **中文甜点** |
| google/bert_uncased_L-4_H-256_A-4 | ~11M | 4×256 | 30.5k | 3.15M | 0.6–1.0 s | 英文 |
| sentence-transformers/all-MiniLM-L6-v2 | 22.7M | 6×384 | 30.5k | 10.6M | 1.3–2.0 s | 英文，质量更好 |
| BAAI/bge-small-zh-v1.5 | 24M | 4×512 | 21k | 12.6M | 1.5–2.2 s | 中文 |
| distilbert-base-uncased | 66M | 6×768 | 30.5k | 42.5M | 4–6 s | 在线判断偏慢 |
| bert-base-chinese | 102M | 12×768 | 21k | 85M | 8–12 s | 只适合离线/批处理 |
| microsoft/deberta-v3-xsmall | ~70M（含 49M embedding） | 12×384 + relative pos | 128k | 21M | 2–3 s + 49 MB 词表 | 相对位置实现复杂，不推荐 |
| answerdotai/ModernBERT-base | 149M | 22×768，8k ctx | 50k | 91M | Pi 7–13 s；桌面可以 | 长上下文/桌面优先 |
| meta-llama/Llama-Prompt-Guard-2-86M | **278M（实测）** | encoder | 大词表 | — | 不可行 | 名字里的 86M 不是总参数 |

关键点：

- **分类头不跑词表 logits**，所以 8–25M 的编码器在 Pi 上可以做到 0.5–2 s；如果错误地复用 30k 词表 logits，成本会翻倍。
- **深度比宽度更伤延迟**：UER L4/H256（8.6M）比 DistilBERT（66M）快 6–8 倍，质量在固定任务上通常够用。
- **多任务多 head**：一个 8.6M encoder 可以同时挂 intent/language/risk/yes-no 多个线性头，一次前向出全部结果；这是最省内存的“类 Jev”固定标签方案。
- **校准**：温度缩放 + 校准集；Score 用 `Σ p_i·i` 得期望；Noul 用 sigmoid；confidence 用 max prob 或 entropy。校准参数放 sidecar，由 Python server 应用，引擎只出 logits。

### 3.2 decoder-only 判断（动态选项/复杂 criteria）

- 模型：MiniMind3 64M（现成）、SmolLM2-135M（要 converter，30 层太慢）。
- 做法：把 state + question + options 拼成 prompt，取各选项首 token 的 logits，或对选项 token 序列做长度归一化 log-prob，再 softmax。
- Pi 成本：20 token 的 prompt 先 prefill ~6 s，每个选项再 1–2 个 token；5 个选项约 8–12 s。
- 适合：离线批处理、低频高风险判断、需要读 criteria 文本的场景；不适合每轮对话的实时路由。
- 若坚持 decoder-only 且要快：把模型压到 4–8M、输入压到 16 token、选项压到 4 个，可到 1–2 s，但准确率/校准会明显下降。

### 3.3 训练自定义模型时的更优解

如果你们能训练模型，**不必**为了判断任务去补 encoder：

- 训练一个 4–8 层、256–384 宽的 **decoder-only 分类器**，在输入末尾放一个 `<cls>` token，取它的 hidden state 接一个小分类头。
- 因果 attention 对分类没有害处：`<cls>` 在最后，能看到全部输入；不需要双向。
- 这样引擎只需要加一个 `head` 矩阵和 `--classify`，不需要 LayerNorm/GELU/双向调度，工作量从“周”降到“天”。
- 代价：没有 BERT 预训练权重，需要更多标注数据；但可以和 MiniMind 共用 tokenizer/格式/训练栈。

同理，自定义 decoder-only 翻译模型（蒸馏 OPUS）也不需要 encoder-decoder 支持，代价是质量上限。

---

## 4. 推荐组合

### 路线 A：质量优先，接受引擎改造（推荐给“翻译是核心功能”）

1. **翻译**：OPUS-MT 77M（en→zh 和 zh→en 各一个，或按需加载）。
2. **判断**：UER L4/H256（中文）或 BERT-tiny/L4-H256（英文）做固定标签；如果判断只针对源文本，可以把分类头挂在 Marian encoder 的 pooled 输出上，一次 encode 同时完成“判断 + 翻译”，省一个模型。
3. **引擎工作**：encoder 调度 + cross-attn + LayerNorm/bias/plain FFN + SPM/WordPiece（或用 token-ID 协议把 tokenizer 放 Python）+ 可配置 EOS/decoder start + tied logits。
4. **预期**：翻译 6–13 s/20-token 句（q8，未优化）；判断 0.5–2 s；内存 q4 约 60 MB（翻译）+ 9 MB（判断）。

### 路线 B：工期优先，自己训练（推荐给“先上线、后迭代”）

1. **翻译**：MiniMind3 64M 蒸馏 OPUS；现有格式/内核/CLI 全部复用。
2. **判断**：4–8M decoder-only 分类器 + `<cls>` 头；或直接用 MiniMind3 64M 做低频判断。
3. **引擎工作**：分类头 + `--classify` + 自定义 tokenizer（可选）+ 可配置 EOS；约 1–2 周。
4. **预期**：翻译 6–12 s，质量低于 OPUS；判断 0.5–2 s（小模型）或 6–10 s（64M）。

### 路线 C：只做判断，不做翻译

- 直接上 UER/BERT 小编码器（需要 encoder 支持，约 2–3 周），或 decoder-only 分类器（约 1 周）。
- 这是三条路线里最快能出产品的；如果翻译不是刚需，不建议为了判断去背 Marian 的包袱。

### 路线 D：产品化折中

- 80% 的判断用现有确定性工具/规则（仓库已有 10–30 ms 的 `/time`、`/math`、`/status` 路由）。
- 15% 用 3–9M 固定标签分类器。
- 5% 用 64M decoder-only 做“读 criteria 的难例”，异步/离线。
- 翻译独立走 OPUS 或蒸馏模型。
- 这样避免“每个请求都跑一个 64M 判断模型”，也避免 255 选项的动态打分。

---

## 5. 额外适配清单

### P0：正确性（不做就跑不起来）

1. **模型描述与格式**
   - 增加 `arch`（llama/marian-encdec/bert-encoder）、`norm`（rms/layer，pre/post）、`act`（swiglu/silu/gelu）、`pos`（rope/sinusoidal/learned）、`bias`、`tied_logits`、`src_vocab/tgt_vocab`、`bos/eos/pad/decoder_start`。
   - 新 magic 或 sidecar JSON；不要继续用环境变量拼配置。
2. **HF → 引擎 converter**
   - 支持 Marian/BERT 的 tensor 命名、qkv 融合、bias/LayerNorm、非门控 FFN、tied logits、目标词表子集。
   - Q8 目前是 per-matrix scale；Marian/BERT 建议先评估 q8，掉点明显再上 per-row q8 或 q4 per-group。
3. **tokenizer**
   - SentencePiece **unigram**（Viterbi、NFKC、`▁`、dummy prefix、byte fallback）；OPUS 的 `.spm` 是 unigram，不是 BPE。
   - WordPiece（BERT：basic tokenization、`##` 贪心最长匹配、`[CLS]/[SEP]`）。
   - 特殊 token：`>>cmn_Hans<<`（ID 5）要作为整词匹配；`</s>`/`<pad>`/`<unk>` 不能按普通文本拆。
4. **生成循环**
   - 可配置 `eos_ids`（OPUS 是 0，不是 1/2）、`decoder_start`（65000）、`bos`（Marian 源端不加 BOS，末尾加 `</s>`）。
   - 停止条件改为模型元数据驱动；顺手修掉 `next == 1 || next == 2` 的硬编码。
5. **encoder 调度（关键）**
   - 不要试图“把 causal mask 关掉”就完事：双向 attention 需要先算完整段的 K/V，再算所有位置的 Q/attention。
   - 可行且改动最小的做法是**逐层三段式**：
     1. 对 N 个位置算 QKV（存 Q/K/V）；
     2. 对 N 个位置做双向 attention（K/V 已齐）；
     3. 对 N 个位置做 FFN。
   - 每层仍是 N 次 matvec，不需要 batched prefill；只多一个 N×d 的 hidden 缓冲和 K/V 缓冲。这是“参考实现”能在一两周内落地的关键。
6. **cross-attention**
   - encoder 输出（最终 LN 后）存一份 `[N × d]`；每个 decoder 层再用自己的 `Wk_c/Wv_c` 投影，或编码后一次性预计算 6 层的 K/V（seq 512 时约 12.6 MB，实际按 N 分配）。
   - decoder 每步：self-attn（causal）→ cross-attn（Q 来自 decoder，K/V 来自 encoder，无 causal mask）→ FFN。
7. **逐元素算子**
   - LayerNorm（post-LN 和 pre-LN 两种）、SiLU/GELU（tanh 近似或 exact）、bias add、非门控 FFN。
   - 第一版全部放 CPU：matvec 仍走 GPU，读回后在 CPU 做 norm/act/bias；6 层模型的读回开销可接受，先保正确性。
8. **判断头**
   - `head` 矩阵 `[labels × dim]`；`--classify` 输出 logits/probs。
   - Score：N 个有序 label + 期望值；Noul：2 类或 1 个 logit + sigmoid。
   - 校准（温度、先验）放 server/sidecar，引擎只出原始 logits。

### P1：性能与内存

9. **tied logits / 目标词表 logits**
   - 把 embedding 数据直接切成 logits tile，省 33 MB；或只保留 31,959 个目标 piece，省 16 MB + 一半 logits MAC。
   - 这是 OPUS q8 能否在 Pi 上舒服跑起来的关键。
10. **GPU 端 LayerNorm/bias/GELU/cross-attn**
    - 第一版 CPU 逐元素操作会拉低 1.5–2×；等正确性稳定后再把 hot path 搬回 shader。
11. **batch prefill / 位置打包**
    - 引擎的 P0 backlog（[future-optimization.md](future-optimization.md)）对 encoder 同样有效；但这是大改，不建议和 Marian 支持同时做。
12. **beam search**
    - NMT 通常 beam 4 能涨 1–2 BLEU；Pi 上 beam 4 = 4× decode，不建议。最多 beam 2，或保持 greedy + 长度惩罚/重复惩罚。
13. **多模型 / 共享 encoder**
    - 两个 GPU 进程在 Pi 上会争 CMA（仓库有过 worker 被饿死的案例），优先一个进程内两个 `Model`，或共享 Marian encoder + 多 head。

### P2：评测与运维

14. **parity**：tokenizer 逐 ID 对比；logits cosine > 0.999、top-1 一致；再谈 BLEU。
15. **质量**：Tatoeba/OPUS dev 的 BLEU/chrF；判断任务 accuracy/F1/ECE；q8 vs q4 分开记录。
16. **性能**：prefill/decode、CmaFree、seq 16/32/64；桌面 GLES2 先跑通，再上 Pi。
17. **回归**：MiniMind3 q8/q4 的 canonical greedy 输出必须逐字节不变。

### 工作量粗估（1 名熟悉代码的工程师）

| 工作 | 参考实现 | 优化后 |
|---|---:|---:|
| 格式 + converter | 1–1.5 周 | 1.5–2 周 |
| tokenizer（C++ SPM/WordPiece） | 1–1.5 周 | 1.5–2 周 |
| encoder/decoder + cross-attn | 2–3 周 | 3–4 周 |
| 判断头 + 校准 + CLI/server | 0.5–1 周 | 1 周 |
| GPU 优化 | — | 2–3 周 |
| **合计** | **4.5–7 周** | **8–12 周** |

如果走“decoder-only 自定义模型”路线：分类头 + `--classify` 约 3–5 天；翻译只需 converter/tokenizer（若沿用 MiniMind 格式则零改动），主要成本在训练和评测。

---

## 6. 性能/内存估算

### 6.1 每 token MAC

| 模型 | 结构 | MAC/token | 备注 |
|---|---|---:|---|
| MiniMind3 64M | 8×768，FFN 2432 | 63.9M | 含 6400 词表 logits |
| OPUS decoder | 6×512，FFN 2048 | 58.5M / 41.5M | 全词表 / 仅目标 32k |
| OPUS encoder | 6×512，FFN 2048 | 18.9M | 无 logits |
| UER L2/H128 | 2×128 | 0.39M | 分类头另算，可忽略 |
| UER L4/H256 | 4×256 | 3.15M | 同上 |
| MiniLM-L6 | 6×384 | 10.6M | 同上 |
| bge-small-zh | 4×512 | 12.6M | 同上 |
| DistilBERT | 6×768 | 42.5M | 同上 |
| BERT-base | 12×768 | 85M | 同上 |
| ModernBERT-base | 22×768 | 91M | 同上 |

### 6.2 Pi 延迟（估算）

按 120M MAC/s（CPU-attention/读回路径）到 220M MAC/s（GPU_LAYER 优化路径）：

| 场景 | 每 token | 输入/输出 | 估算 |
|---|---:|---|---:|
| OPUS encoder | 90–160 ms | 20 token | 1.8–3.2 s |
| OPUS decoder（目标 32k） | 190–350 ms | 20 token | 3.8–7.0 s |
| OPUS decoder（全 65k） | 270–480 ms | 20 token | 5.4–9.6 s |
| **OPUS 整句** | — | 20+20 | **6–13 s** |
| UER L2/H128 | 10–20 ms | 16 token | 0.2–0.3 s |
| UER L4/H256 | 30–50 ms | 16 token | 0.5–0.8 s |
| MiniLM-L6 | 60–110 ms | 16 token | 1.0–1.8 s |
| bge-small-zh | 70–125 ms | 16 token | 1.1–2.0 s |
| DistilBERT | 250–400 ms | 16 token | 4–6 s |
| BERT-base | 500–750 ms | 16 token | 8–12 s |
| ModernBERT-base | 550–800 ms | 16 token | 7–13 s |
| MiniMind3 64M 判断 | 300–500 ms | 20 token prompt | 6–10 s |

再次强调：这些是按 MAC 折算的**估算**，不是新模型实测；真实值受层数、draw 次数、readback、logits tile 数影响。正确顺序是先在桌面 GLES2 用合成权重/真实权重测，再上 Pi。

### 6.3 内存

| 模型 | q8 | q4 | 说明 |
|---|---:|---:|---|
| OPUS 77M，naive（logits 复制） | ~110 MB | ~93 MB | 不建议 |
| OPUS 77M，tied logits | ~78 MB | ~60 MB | 推荐起点 |
| OPUS，tied + 目标 32k logits | ~60 MB | ~43 MB | 最省 |
| MiniMind3 64M | 66 MB | 45 MB | 已跑通 |
| UER L4/H256 | ~9 MB | — | 判断 |
| MiniLM-L6 | ~23 MB | — | 判断 |
| ModernBERT-base（分类） | ~150 MB | ~95 MB | Pi 太紧 |

KV：OPUS decoder self-attn 6 层 × 512 × 512 × 2 × 4 B ≈ 12.6 MB；cross-attn K/V 再 ~12.6 MB；encoder 临时 K/V 可复用。合计 ~25 MB，可接受。

---

## 7. 建议的验证顺序

1. **桌面 GLES2**：先用真实 OPUS/BERT 权重转 q8，跑 tokenizer/logits parity；确认 LayerNorm/bias/位置编码无符号/布局错误。
2. **桌面 synthetic**：用随机权重的 Marian/BERT 形状测 per-token 成本，校准上面的估算；不依赖真实权重质量。
3. **Pi 单模型**：先跑 OPUS q4 + 目标 logits，确认 CMA、EOS、长度；再跑判断小模型。
4. **Pi 双模型/共享 encoder**：确认不会争 CMA；若不行，判断模型走 CPU 或共享 encoder head。
5. **质量**：BLEU/chrF + 判断 accuracy/F1/ECE；q8/q4 分开。
6. **回归**：MiniMind3 q8/q4 canonical 输出不变。

---

## 8. 待确认的问题

1. “类 Jev”是**固定标签**（意图/路由/yes-no/风险）还是**请求时动态 options/criteria**？这决定走编码器还是 decoder-only。
2. 目标语言对：只 en↔zh，还是多语？多语会把词表/模型选择推向 mT5/NLLB，尺寸就不够了。
3. 延迟预算：翻译 10 s、判断 1 s 是否可接受？如果要求 150 ms，Pi Zero 上只能做规则/极小的固定分类器。
4. 是否能训练/蒸馏？能训练的话，decoder-only 自定义模型可以省掉 5–7 周引擎改造。
5. 部署形态：Pi 单机，还是桌面 GLES2 + Pi 混合？桌面可以上 ModernBERT/更大模型，Pi 只做小模型。
6. 是否需要 beam/长度控制？NMT 的 greedy 质量通常可接受，beam 在 Pi 上代价太高。

---

## 参考

- 引擎性能基线：[performance.md](performance.md)
- 引擎内部/VC4 限制：[engine-internals.md](engine-internals.md)
- 格式与 Q4：[model-formats.md](model-formats.md)、[quantization.md](quantization.md)
- 未来优化（batch prefill 等）：[future-optimization.md](future-optimization.md)
- OPUS-MT en-zh 模型卡：https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
- OPUS-MT zh-en 模型卡：https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
- Jev/TypeSafe 文档：https://docs.typesafe.ai/primitives/choice、https://docs.typesafe.ai/primitives/score
- Jev 用例与 150 ms 说法：https://docs.typesafe.ai/concepts/use-case-map
- ModernBERT：https://huggingface.co/answerdotai/ModernBERT-base
- MiniLM：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- UER Chinese RoBERTa：https://huggingface.co/uer/chinese_roberta_L-4_H-256
- bge-small-zh：https://huggingface.co/BAAI/bge-small-zh-v1.5
