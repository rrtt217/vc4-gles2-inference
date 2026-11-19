# RWKV-7 实施计划与算子清单

状态：**实施中（2026-11-16 立目标）**。本文是目标的控制文档：
[assessment 文档](rwkv7-assessment.zh-CN.md) 记录"值不值得、卡在哪里"（2026-11-12 的评估，
数字未变），本文记录"怎么做、做到哪、每个算子落在哪段代码"。
所有"现状"都核对过当前源码（HEAD `083ea5e`，2026-11-16）。

- 目标：在引擎里实现 RWKV-7（Goose）支持，先桌面 GLES2 正确，再 Pi Zero 跑 0.1B，
  最后做 GPU 逐元素核与 WKV7 融合。
- 规模：**18–32 人日到可用（阶段 0–5），24–44 人日含优化（阶段 6）**；
  依据是 assessment 文档第五节的分解，本文只把它拆到算子粒度。
- 不变式：现有模型的引脚（MiniMind3 q8/q4、LFM2.5、MoE、i16）**一个都不许动**；
  `scripts/regression.sh`、`shader_hashes.sh`、`swap_matrix.py`、
  `check_target_reset.sh`、`warncheck` 每一步都要绿。

## 一、总体形状

RWKV-7 不是 transformer 块：没有 qkv/wo 的语义、没有 KV cache、没有 softmax/RoPE，
取而代之是**每层一个 64×64 的递推状态**（每 head 一个）+ 每层两个 token-shift 向量。
引擎今天的两条层循环（`Session::forward` 串行 f32、`Session::forward_gpu_hidden` 全纹理）
都假设"一个 transformer 块"，所以新增的是**第三条层循环**，不是给旧循环加分支：

| 层循环 | 位置 | 现状 | RWKV-7 的用法 |
|---|---|---|---|
| 串行（CPU 逐元素，GPU matvec） | `engine/src/session.cpp:284` | 每层 8 次回读 | **阶段 2 的正确性路径** |
| 融合（全纹理，0 回读） | `engine/src/session.cpp:473` | 每层 0 回读 | 阶段 6 的目标形态 |
| 双位置 prefill（pair） | `engine/src/session.cpp:650` | 2 位置/层 | 只给 matvec/FFN；WKV 必须逐位置（O31） |

状态生命周期跟 `conv_state_`（`engine/src/session.h:68`）同构，但要重得多：
0.1B 的 WKV 状态 = 12 层 × 768×64 f32 = 2.36 MB，token shift = 12 × 2×768 × 4 B = 74 KB。

## 二、算子清单（逐算子）

记号：**d** = dim（768/1024/2048），**L** = 层数（12/24/24），**F** = FFN 宽（3072/4096/8192），
**H** = d/64 个 head，**r_w/r_a/r_v/r_g** = 低秩（64/64/32/128 起）。
"现状"列里 `—` 表示引擎里没有这个算子；括号是当前最接近的原语。
数值格式按引擎的三条轨：**q16**（残差/激活，±128，1/256）、**I24**（matvec 输出，1/256，±32768）、
**f32**（CPU 侧）。

### A. 词表与文本侧（阶段 3）

| # | 算子 | 数学 | 现状 | 归属 | 数值 |
|---|---|---|---|---|---|
| A1 | 贪心最长匹配分词 | 对词表建查找，从 `min(maxlen, n)` 向下试子串，命中即取；无 merge | —（`Tokenizer::encode` 末尾无条件 `bpe_merge`，`engine/src/tokenizer.cpp:151`） | `Tokenizer` 新 `encode_greedy`，`tokenizer.h:22` 旁 | 字节 |
| A2 | 词表反转义 | 词表条目是 `repr(bytes)` 文本（`\xff`/`\t`/`\n`），打包时还原成裸字节 | —（`.tok` blob 只有 `(score,len,bytes)`：`tokenizer.cpp:19-26`） | 新打包器 `make_tok_rwkv.py`（与 `make_tok_byte.py` 并列） | 字节 |
| A3 | decode 裸字节 | 直接输出条目字节；不识别 `<0xNN>` | 部分：`decode` 只特殊处理 `<0xNN>`（`tokenizer.cpp:183-188`） | 同一函数 + 容器 arch 分支 | 字节 |
| A4 | 特殊 id | BOS=0（`<\|rwkv_tokenizer_end_of_text\|>`/`<s>`）、EOT/stop=261（`"\n\n"`） | 已有：key 8（`model.cpp:553`） | 容器写入，引擎不改 | — |
| A5 | 模板 | `User: …\n\nAssistant:`，HF `chat_template` 文本 | 已有：key 9 + Jinja（`chattmpl.cpp`） | 容器写入，`add_bos=0` | — |

### B. 每层 time-mix（阶段 2 的 CPU 版 / 阶段 6 的 GPU 版）

| # | 算子 | 数学 | 形状 @0.1B | 现状 | 归属 | 数值 |
|---|---|---|---|---|---|---|
| O1 | LayerNorm（5 处） | `mean/var`，`(x-μ)/√(σ²+eps)`，仿射 w+b，eps=1e-5 | d | **—**（只有 RMS：`shaders.cpp:1362` 起 `rms_norm`，`session.cpp` 的 `rms_norm`） | 新 builder `layernorm` + `GPU::layernorm` + CPU `layernorm` | q16 入 / q16 出；μ、σ 在 f32 累加 |
| O2 | token shift（att） | `x_prev = [shift; norm[:-1]]` | 1×d | **—**（conv 状态是唯一先例：`conv_state_`） | `Session` 新状态 `rwkv_shift_` | q16 |
| O3 | 六路 lerp | `sx=x_prev-x`；`sx*lerp[k]+x`，k=r,w,k,v,a,g | 6×d | **—**（`mul_act` 只做 a*b，`shaders.cpp:78`） | 新 builder `lerp6`（或 CPU 循环） | q16 |
| O4 | `W_r` matvec | `r = W_r·xr` | d×d | 已有：`matvec_q8*`（`shaders.cpp:31` 起） | 复用 `GPU::matvec_to`（`gpu.h:72`） | 出 24 位 |
| O5 | `W_k` matvec | `k = W_k·xk` | d×d | 同上 | 同上 | 24 位 |
| O6 | `W_v` matvec | `v = W_v·xv` | d×d | 同上 | 同上 | 24 位 |
| O7 | decay 低秩支路 | `w = exp(-0.606531·σ(W_w2·tanh(W_w1·xw)+w0))` | d→r_w→d | 低秩 matvec 可复用；**tanh/sigmoid/exp 三件缺失**（`engine/src` 里 0 处 tanh） | 新 builder `lowrank_sigmoid`（w/a/v/g 四用）+ CPU 版 | `w` 不能用 q16：见 §四 W-1 |
| O8 | value residual | `v += (v_first−v)·σ(W_v2·(W_v1·xv)+v0)`，第 0 层记录 `v_first` | d→r_v→d | **—**（跨层状态，transformer 没有） | `Session` 新状态 `v_first_` + 同一 builder | 24 位/ q16 |
| O9 | ICLR 低秩 | `a = σ(W_a2·(W_a1·xa)+a0)` | d→r_a→d | 同 O7 | 同 O7 | q16（σ∈(0,1)） |
| O10 | gate 低秩（可选） | `g = W_g2·σ(W_g1·xg)` | d→r_g→d | 同 O7 | 同 O7 | q16 |
| O11 | per-head l2norm | `kk = (k·k_k)/max(‖·‖₂,1e-12)`，每 head 64 | d | 近邻：`att_qnorm_scale`（per-head RMS，`shaders_attn.cpp:89`） | 新 builder `l2norm_head`（或参数化 qnorm） | 见 §四 K-1 |
| O12 | key delta | `k = k + (a−1)·k·k_a` | d | **—** | 新 builder `axpy`（3 输入 1 输出） | 24 位 |
| O13 | **WKV7 状态更新** | 每个 head、每行 i：`sa=⟨−kk,S[i]⟩`；`S[i] = S[i]·w + v[i]·k + sa·(kk⊙a)` | L×H×64×64 | **—**（无任何递推原语） | 阶段 2：CPU f32 循环；阶段 6：`wkv7_state`/三 pass | 状态见 §四 S-1 |
| O14 | WKV7 输出 | `out[i] = ⟨S[i], r⟩` | d | **—** | 同上（阶段 6 与 O13 同 pass） | 24 位 |
| O15 | per-head groupnorm | 每 head 64 通道的 mean/var（eps=64e-5），再 `·ln_w+ln_b` | d | **—**（O1 的 per-head 变体） | 新 builder `groupnorm_head` | q16 入 / q16 出 |
| O16 | `r_k` bonus | `rk=Σ_head(k·r·r_k)`；`y += v·rk` | d | **—**（attention 的 per-head 归约是 score/stats，不通用） | 新 builder `head_reduce_add`（2 draw） | 标量 f32→q16 |
| O17 | gating | `y *= g` | d | 部分：`mul_act`（`shaders.cpp:78`） | 复用 `mul_act` | q16 |
| O18 | `W_o` matvec | `y = W_o·y` | d×d | 已有：`matvec_add`（`gpu.h:139`） | 复用 | 残差 q16 |
| O19 | 残差加 | `x += y` | d | 已有（`matvec_add` 的 residual） | 复用 | q16 |

### C. 每层 channel-mix（阶段 2 / 6）

| # | 算子 | 数学 | 形状 @0.1B | 现状 | 归属 | 数值 |
|---|---|---|---|---|---|---|
| O20 | LayerNorm（attn_norm_2） | 同 O1 | d | **—** | 同 O1（第二次实例） | q16 |
| O21 | token shift（ffn） | `x_prev = [shift; norm2[:-1]]` | 1×d | **—** | 同 O2（第二个槽） | q16 |
| O22 | channel-mix lerp | `xk = (x_prev−x)·lerp_k + x` | 1×d | **—**（`lerp6` 的单向量形态） | 同 O3 | q16 |
| O23 | `W_cm_key` matvec | `h = W_cm_key·xk` | d×F | 已有：`ffn_split`/`matvec_to`（`gpu.h:81`） | 复用（不需要 w1/w3 双矩阵） | 24 位 |
| O24 | squared-ReLU | `h = max(h,0)²` | F | **—**（SwiGLU 的 `swiglu` 是 `x·σ(x)`，`shaders.cpp:72`） | 新 builder `sqrelu` | 见 §四 Q-1 |
| O25 | `W_cm_value` matvec | `x += W_cm_value·h` | F×d | 已有：`matvec_add` | 复用 | 残差 q16 |

### D. 全局（阶段 2）

| # | 算子 | 数学 | 形状 @0.1B | 现状 | 归属 | 数值 |
|---|---|---|---|---|---|---|
| O26 | 最终 LayerNorm | O1 变体 | d | **—** | 同 O1 | **不能是 q16**：g1d 实测出峰 315.6 = rail 的 2.5× → 写 `x·2^-k` + head tile scale 折 `2^k`（key 17），或 I24_1 |
| O27 | logits matvec | `logits = head·x`，**untied**（两套 65536×768） | 65536×d | 已有：43 个 tile + `logits_from`（`gpu.h:170`） | 复用；packer 写两份 | q8（不上 q4）；**输入尺度见 O26** |
| O28 | argmax / 采样 | — | — | 已有：`step_greedy`（`session.cpp:739`） | 复用 | — |

### E. 状态与生命周期（阶段 2 必须，缓存部分可后置）

| # | 项 | 要求 | 现状 | 归属 |
|---|---|---|---|---|
| O29 | 状态清零 | 载入/换模型/重放 prompt 时 WKV 与 token shift 归零 | 先例：`zero_conv_state`（`session.h:23`） | `Session::zero_rwkv_state` |
| O30 | prefix cache | 存 WKV（L×H×64×64）+ token shift，版本 5 + flag；**没有 KV cache 也要能存** | 现在第一道闸是 `!kv_cache_.empty()`（`session.cpp` 的 save/load） | 复用 conv v4 的模式 |
| O31 | pair/prefill | matvec/FFN 可 pair，WKV 与 token shift 逐位置；最省事是 `pair_ok()` 直接拒绝 | `pair_ok()`（`session.cpp:604`）已有拒绝模板（hybrid/MoE） | 一行守卫 + 后续 pair 变体 |
| O32 | REPL/swap | `/model` 换容器后状态必须重建；`prepare_model()`/`release_scene()`/`reset_targets()` 都要认这个新 arch | 见 `docs/checkpoint-container.md` 的换模型一节 + `scripts/swap_matrix.py` | `Session::reset` + swap 链 |
| O33 | 未知 arch 拒绝 | 旧引擎读到 v4 必须干净拒绝；`Model::probe()` 不许崩 | `model.cpp` 的版本检查 + `probe()` | loader |

### F. 容器与 packer（阶段 1）

| # | 项 | 要求 | 现状 | 归属 |
|---|---|---|---|---|
| O34 | f32 逐通道向量 | 13 个/层的向量（lerp 6、k_k/k_a/r_k、LN w/b、ln_w/b、w0/v0/a0）按 `layers×d` 存 | 已有：dtype 0 + `rms_att` 就是 `layers×dim`（`model.cpp:687` 一带） | 新 kind，读法照抄 |
| O35 | LayerNorm bias | 每层 4 条 `d` 向量 + 最终 1 条 | **—**（引擎无 bias 概念；`Mat` 没有 bias 字段，`common.h:24`） | 新 kind（dtype 0，rows=1） |
| O36 | metadata key 16 | `arch, head_size, r_w, r_a, r_v, r_g, norm_eps`；容器升 v4 | key 1–15 已用（`model.cpp:528-602`）；v3 是 MoE | `mmcf_format.py` + `mmcf_pack.py` + `model.cpp` |
| O37 | 18 个新 kind | `W_r/W_k/W_v` 可拼进 `0x10`（按 qkv 读）、`W_o`→`0x11`、`W_cm_key`→`0x12`、`W_cm_value`→`0x15`，其余新开 | id 空间 `kind(8)|layer(6)|part(2)`（`model.cpp:715`），已用 `0x01-0x19/0x20/0x30` | 三个 switch 同步改（`:715`/`:802`/`:837`） |
| O38 | packer | `load_rwkv7()` + `check_source` 放行 + 残差/状态峰值检查 | packer 现在按名字拒绝（`mmcf_hf.py:165`） | `mmcf_hf.py` |

### G. 验证（贯穿，阶段 0 起步）

| # | 项 | 内容 | 现状 |
|---|---|---|---|
| O39 | numpy oracle | 全序列 f32 前向 + 逐阶段 dump + 状态快照；以官方 `rwkv_v7_numpy.py` 为准 | `ref_forward.py` 是 llama 专用（`ref_forward.py:60` 一带） |
| O40 | llama.cpp 对拍 | 同一 prompt 的 logits/top-1；本机 `refs/llamacpp/bin/b12283-cpu/llama-cli` 已能跑 RWKV7 | 已有，只需接进攻具 |
| O41 | dump 钩子 | `RWKV_DUMP`（逐层 att_norm/w/a/g/state 校验和）+ `STAGE_DUMP` 新阶段 | 新开关必须同时写 `docs/test-hooks.md`，否则 gate 第 0 步失败 |
| O42 | 质量基线 | `--ppl` + `bench_mc.py`/`eval_ifeval.py`/`eval_translate.py` | 已有，直接可用 |
| O43 | 引脚 | regression 的 `RWKV_*` 变量 + 文档 | `regression.sh` 已有 per-model pin 机制 |


## 三、阶段与工作量

| 阶段 | 内容 | 出口判据 | 人日 |
|---|---|---|---:|
| P0 | numpy oracle + **与官方 rwkv 包对拍** + 数值测量（w/state/kk/sqrelu/残差） + 分词/模板往返 | oracle 与官方包在**同一 g1d pth** 上 cos>0.9999、greedy 逐 token 一致；§四 每个决策点有数 | 2–4 |
| P1 | 容器 v4 + key 16 + 18 个新 kind + f32 向量/bias + packer `load_rwkv7` | `mmcf_dump.py` 能读、`test_mmcf_format.py` 往返、旧容器不受影响 | 3–5 |
| P2 | loader + 串行层循环（12 个新算子 CPU 版）+ 状态 + 两个守卫 | 桌面 GLES2 上 logits 对上 oracle；`regression.sh` 现有引脚不动 | 6–10 |
| P3 | 分词器（贪心最长匹配）+ 模板 + 特殊 token/stop | 分词与 HF `RwkvTokenizer` 逐 id 一致；chat 能出话 | 2–4 |
| P4 | 验证：dump 钩子、PPL、三套 battery、regression 引脚 | `RWKV_*` 引脚入 gate；质量表与 MiniMind3/LFM2.5 同表 | 3–5 |
| P5 | Pi 部署 + 计时 + 文档 | 板子上 0.1B 能对话、有 ms/token 与 CMA 数 | 2–4 |
| **小计** | 可用版本 | | **18–32** |
| P6 | GPU 逐元素核 + 融合层循环 + WKV7 GPU + pair 变体 | 与 P2 数值同；回读 14/层 → 1/层 | 6–12 |
| **合计** | 含优化 | | **24–44** |

## 四、数值格式决策点（P0 必须先量，否则 P2 会返工）

| 代号 | 对象 | 问题 | 候选 | 判据 |
|---|---|---|---|---|
| W-1 | decay `w` | `w∈(0.545,1]`；q16 的 1/256 绝对误差在这里是 4e-3，而状态每 token 乘一次 | (a) CPU f32；(b) GPU 存 `1−w` 用 I16_2 配 scale 65536（1.5e-5）；(c) I24_1 | 把 w 量化到候选后跑 PPL/logits 漂移 |
| S-1 | WKV 状态 | 每层 `H×64×64` 的 k⊗v 衰减累加，动态范围未测；VC4 无浮点纹理 | (a) CPU f32（P2 必须）；(b) I24_1 + 每层 scale；(c) I16_2 + 每 head scale | 单层 state `\|max\|`；量化后 logits |
| K-1 | `kk` | L2 归一化后分量约 1/8，q16 的 1/256 是 3% 相对误差，可能太粗 | (a) ×8 缩放后 I16_2；(b) I24_1；(c) f32 | 量化 kk 后的 logits/PPL |
| Q-1 | sqrelu 输出 | 平方把幅度抬一档，可能超 q16 rail | 复用 key 15 的 `2^-k` 隐藏存储 / I16_2 | `W_cm_key` 输出峰值 |
| R-1 | 残差 | RWKV 残差峰值未知（LFM2 是 25–30；SmolLM2 25982 是反例） | q16 默认；超 128 走 key 12 的 q24；或 key 15 的逐层缩放 | numpy 前向的逐层峰值 |
| A-1 | `a/v/g` | 都是 sigmoid 输出，量级 ∈(0,1)/(−1,1) | q16 | 量化后 logits（预期无影响，验一遍） |

**P0/P0b 实测结论（2026-11-16，official g1d pth，f32）**。三方一致：oracle vs 官方 pip 包
**cos 1.00000000 / maxabs 8e-5 / greedy 12/12**；同 checkpoint 的 Q5_K_M GGUF 逐通道向量与
safetensors **逐位相同**，llama.cpp **8/8 argmax**（旧 fla+Q4_0 是 6/8——那 2/8 是 4 bit 加另一代
finetune，不是公式）。逐层表见 `models/rwkv7/phase0-g1d-report.md`。

- **W-1 确认**：存 `1−w` 进 I16_2 + scale 65536（(a) 的 −5.2e-5 是噪声，(d) 才无损；(b) 与 (a) 同为
  16 位整数、成本相同 → 选 (b)）。
- **S-1 改严**：1/256 的 state 量化在 g1d 上 **dNLL +0.1543 nats/token（NLL 0.2357→0.3900，+65%）、
  dmax 3.20、greedy 6/8**，而打分位置不平凡（p1=0.706、间隔 1.22 nats）→ 代价是真的。**P2 必须 CPU
  f32 state**；GPU 必须 **per-head scale + ≥18 bit**（每 head 跨度 L0 278× / L11 161×）。`|S|max` 51.8
  < 128，rail 不是瓶颈，**步长与 per-head 动态范围**才是（I24 与 q16 逐位相同）。
- **R-1 改写**：残差峰 **37.12**（rail 余量 3.4×，装得下），但**步长**代价 **+0.1111 nats/token
  （NLL +47%）、dmax 4.14、greedy 6/8** → "残差按 q16 存"本身就是 0.1 nats/token 量级；
  若门的判据是 cos>0.9995，残差要走 q24（key 12）或逐层 scale。q16 仍作起点，由 P4 的 PPL 决定。
- **Q-1 部分改写**：量化代价仍可忽略（−2.9e-3、cos 0.9999989、greedy 8/8），但 **rail 余量只有 1.39×**
  （sqrelu 峰 **91.95**，是全模型最接近 rail 的张量）→ P1 的 rail check 必须报这一项，key 15 的
  `2^-k` 能力保留，别现在删。
- **K-1 确认**（+5.3e-4、cos 1.0、8/8；kk 分量 0.063–0.086，不是 1/8）；**A-1 确认**（+5.1e-3、dmax 0.496）。
- **新约束（已写进 O26）**：**最终 LayerNorm 输出 315.6 = q16 rail 的 2.5 倍**，而引擎今天把
  `xf_` 写成 q16（`session.cpp:586` 的 `rmsnorm(..., xf_)`）→ logits matvec 的输入会被 clamp 成 ±128。
  解法：写 `x·2^-k` 并把 `2^k` 折进每个 head tile 的 `scale`（新的 metadata key 17，手法同 key 15 的
  `apply_hidden_scale()`），或让 `xf_` 走 I24_1。**注意 q16 的 rail 是 ±128（ACT_SCALE=256、int16），
  不是 ±2048；"±2048" 那档是 I16_2 这个布局名容易引起的误会，不能用来装 315。**
- 量化扫的打分位置：fla 那份很平（间隔 0.111 nats），g1d 的不平（1.22 nats）——**读 dNLL 与 cos，
  不要读 greedy 的 6/8**。

## 五、验收门（每一步都要过，不是最后补）

1. `make -C engine check`：warncheck、docs_check、shader_hashes、check_q16_decode、
   check_target_reset、server selftest。
2. `engine/scripts/regression.sh`：MiniMind3 q8/q4、LFM2.5、MoE、i16 的引脚**一个都不许动**；
   RWKV 只新增自己的 `RWKV_*` 引脚。
3. `engine/gate/shader_hashes.sh`：P2 **不新增 builder**，文本应零变化；
   P6 每加一个 builder 都要在同一个 commit 里重基 `gate/shader_hashes.txt` 并写明原因。
4. `engine/scripts/swap_matrix.py`：新 arch 要么进链，要么在 `/model` 处明确拒绝（不许半加载）。
5. 新模型的自身门：numpy cos>0.999 且 top-1 一致；PPL；MC/IFEval/translate 三套；Pi 计时。

## 六、风险与止损点

| 风险 | 影响 | 止损 |
|---|---|---|
| 0.1B 质量不可用 | 整个目标的收益归零 | P0 就用 `bench_mc.py`/三套 battery 量；不合格就停在 P0 |
| W-1/S-1 精度不够 | GPU 路径要改编码甚至放弃定点 | P0 量化实验定编码；P2 先用 f32 CPU 状态，不阻塞正确性 |
| Pi 内存不够 | 只能 q4 或层 q4 | 评估已算过：q8 CMA 141–152 MB、层 q4+q8 head 107 MB；P5 实测 |
| prefill 串行 | Pi 上 512 token 要 2.5–3.7 min | 文档写真话；桌面优先；P6 再谈 pair |
| 0.4B 的 FFN 4096=2×2048 | VC4 的 2048 行静默出错 | 0.4B 只做桌面；要上 Pi 先给 `split_ffn` 加 3 块 |
| REPL/swap 状态重建 | 换模型后答案漂移 | `swap_matrix.py` 加 RWKV 链；P2 就接 `Session::reset` |

## 七、任务板映射

| 任务 | 阶段 | 写作用域 |
|---|---|---|
| T1 oracle 与数值测量 | P0（完成） | engine/tools/ 下新增 `ref_rwkv.py`、`models/rwkv7/` |
| T1b g1d 权重上重跑 measure/quant-sweep | P0b | `models/rwkv7/`（报告）、`engine/tools/ref_rwkv.py` |
| T2 容器与 packer | P1 | `engine/tools/mmcf_format.py`、`mmcf_pack.py`、`mmcf_hf.py`、`model.cpp` |
| T3 loader + 串行层循环 + 状态 | P2 | `engine/src/model.*`、`session.*`、`common.h` |
| T4 分词器 + 模板 | P3 | `engine/src/tokenizer.*`、engine/tools/ 下新增 `make_tok_rwkv.py` |
| T5 验证与基线 | P4 | `engine/scripts/`、`docs/test-hooks.md` |
| T6 Pi 部署与文档 | P5 | `docs/benchmarks.md`、`models/README.md` |
| T7 GPU 核与融合 | P6 | `engine/src/shaders.cpp`、`gpu_*.cpp`、`session.cpp` |

## 八、非目标（本次不做）

- 训练/微调/蒸馏、QAD 权重；
- 2.9B/7.2B/13.3B，以及 0.4B 上 Pi；
- 多序列 batch、训练式并行 chunked scan；
- 用 RWKV 替换现有默认模型（默认仍是 MiniMind3 / LFM2.5）；
- 超过 2048 的 transformer 路径改造（那是 `docs/long-context.md` 的题目）。
- DeepEmbed（rwkv7a，channel-mix 逐 token 双线性门，s_emb[vocab,1024]/层，0.1B≈2GB）与
  DEA（rwkv7b，二次注意力支路）：内存重、Pi 不可行，属桌面-only 的后续扩展，非本目标。

**翻译特化（2026-11-16 调研）**：0.1B 专用翻译模型（Alic-Li/RWKV_v7_G1_Translate）是
DE/DEA 变体（2.0/3.7GB），非标准 v7、Pi 不可行；标准 v7 翻译微调从 0.4B 起。廉价路线是
**state-tuning**（nekocross/RWKV7-Translate-State）：训练一个初始递归状态 .state（O(1)、~MB），
预载进 RwkvState 即特化，架构免费。公开 state 仅 2b9；可自训 0.1B 翻译 state（冻结权重、只优化
状态），是 Pi 友好的翻译特化首选。引擎需加 --rwkv-state 预载钩子（小改）。

## 九、当前状态（滚动更新）

| 阶段 | 状态 | 备注 |
|---|---|---|
| P0 | **完成 2026-11-16** | oracle 与官方 rwkv 包在同一 g1d pth 上 cos 1.00000000、maxabs 8e-5、greedy 12/12 一致；§四 五个决策点有数；分词/模板两个坑已记录（§十） |
| P0b | **完成 2026-11-16** | g1d 上重跑：S-1 改严（state 1/256 → +0.154 nats/token）、R-1 改写（残差步长 +0.111）、Q-1 部分改写（sqrelu rail 余量 1.39×）、**新发现最终 LN 315.6 = 2.5× rail → key 17**；三方一致（oracle/官方/llama.cpp 同 checkpoint 8/8 argmax） |
| P1 | **完成 2026-11-16** | `models/rwkv7/packed/rwkv7-g1d-0.1b.q8.mmcf`（191,844,864 B，196 张量，version 4，key 16 + key 17，`logit_shift=2`）；`test_mmcf_format.py` 31 容器往返、旧容器 summary 逐字节不变、`make check` 108/0；旧 HEAD 引擎对 v4 干净拒绝（`unsupported version 4`）；负测试（head 32 / v3 / v5 / r_g 96）都是 rc=1 不崩 |
| P2 | **完成 2026-11-16** | `forward_rwkv` 第三条层循环（CPU f32 状态 + GPU matvec），`split_rwkv` 切 rkv 三带，app 加 `set_rwkv`；桌面端到端跑通、确定性、minimind3 引脚不变、`make check` 108/0、无新 shader。详见 §十一 |
| P3 | **完成 2026-11-16** | 引擎侧 `tokenizer.*` 贪心模式（encode/encode_parts/decode 分支 rwkv_；96 组旧路径 md5 全等；全量 65536 往返 0 漂移；模板端到端 exact）。packer 侧：`engine/tools/make_tok_rwkv.py`、`models/rwkv7/g1d.tok`（897,740 B，65536 条，maxlen 128，id0 文本；贪心编码与 oracle **逐 id 一致**）、`templates/rwkv7-g1.jinja`（官方 G1 写法，生成提示 `Assistant: <think`）、以及内嵌两者的 `models/rwkv7/packed/rwkv7-g1d-0.1b.q8.chat.mmcf`（197 张量，192,743,132 B，caps=0x1，add_bos=0）；引擎侧 `tokenizer.*` 的贪心模式由 p3-tokenizer 在做 |
| P4 | **完成 2026-11-16** | 层循环忠实（engine↔容器权重 oracle cos≈0.999 ACT_F32）；R-1 q16 保持默认；三套 battery 最终口径入 docs/evaluation.md；RWKV_DUMP 登记；regression ACCEPT + swap_matrix rwkv 链 PASS；Xe 引脚入 AGENTS.local.md |
| P5 | **完成 2026-11-16** | Pi Zero 跑通 0.1B：**q8r 可加载且中文连贯**（"我是一名AI语言模型"），q4 亦可跑但中文退化（重复/回声）；SEQ=4096 不夹取（O(1) 状态）。此前段错误实为 apply_q4_fmt 越界读空 w3 + 分块时误传完整 3072 行 w1，均已修（62895ce/cb16490），**非 OOM/CMA**；q8r 实际能装下 | cm_key/value 已按 VC4 2048 行分块（split_rwkv(max_dim)，Xe 不触发、cos 不变）；ARMv6 二进制已部署。但 Pi 当前自由 CMA ~94MB（pumpkin 占用），0.1B q8r 需 ~141MB、q4（head q4）亦超自由 CMA，均上传失败段错误；需停 pumpkin 独占 CMA 再测 decode/prefill/长上下文。P6 已在 Xe 启动 |
| P6 | **Xe 交付 2026-11-16；Pi 数值修复 2026-11-17（速度待优化）** | `RWKV_GPU=1` 融合层循环（`gpu_rwkv.cpp` + 27 个 `rwkv_*` builder），WKV state 用 **bit-cast f32 RGBA8 双纹理 ping-pong**（`floatBitsToUint`/`uintBitsToFloat`，精确 f32，S-1 从根上解决）；ES3 上下文/编译路径由 Q4_SHIFT 一般化（`g_es3_ok`，`needs_es3` 也匹配 bit-cast）。**PPL 裁决**（q8r chat，data/ppl_test.txt，SEQ=512）：CPU 9.2919 / GPU **9.1850** / 混合(CPU f32 state) 9.1781 —— GPU 与 CPU 同量级且略优，数值合格。greedy 串 GPU≠CPU 是 0.1B temp0 混沌（仓库规则：跨路径让 PPL 裁决，不追 byte 一致），非退化。逐元素链保留 I24_1（其网格误差被 PPL 容忍）；无 ES3 回退 CPU f32（`RWKV_WKV_CPU=1`）。速度（bench 口径：丢弃预热 + 三次中位，--gen 48）CPU 119.1 / GPU 36.0 ms/token = **3.3×**；确定性两次运行 byte 一致、behaviour 非空不塌缩。GPU PPL 引脚 9.1850、速度、确定性已入 docs/benchmarks.md、docs/test-hooks.md 与 AGENTS.local.md GPU 引脚注释；`make check` 108/0、regression ACCEPT（CPU 引脚全不变）、shader_hashes 仅新增。**Pi VC4 数值修复（2026-11-17）**：宽度修复后 Pi GPU 仍塌缩成 garbage（混合路径亦塌缩 ⇒ 元凶是逐元素链非 state）。板级 probe（`probes/rwkv_probe[_pi]` A–J）逐项隔离皆 PASS，逐 op GPU-vs-CPU dump 定位到 **layer 1 的 1536 宽 chunked cm sqrelu**：单 draw 超 VC4 `kDrawCap`=1024 被静默丢弃（target 留在 clear 色 I24 −32768）。两处修复均 gate 到 `needs_finish()`（VC4），Xe 逐位不变：(1) `rwkv_draw1` 在 VC4 按 kDrawCap 切条；(2) wo/cm 残差加在 VC4 走 CPU 两步序（`rwkv_mv`→I24 + 新 builder `rwkv_res_add`→q16），因 fused `matvec_add` 在 VC4 I24_1 链上丢 dot；Xe 保留 fused。修后 **Pi GPU greedy 与 Xe GPU 逐位一致**（garbage 消失）；Xe GPU PPL 9.1850 不变。**Pi 速度**：GPU ~1138 / CPU ~969 ms/token —— VC4 上 GPU draw-bound（~1879 draws/token × ~0.6ms）略慢于 CPU，≥3× 仅在 Xe 成立；Pi 提速需把逐元素融进 matvec 核（减少 draw），属后续工作。逐次测量见 git（当前表在 docs/benchmarks.md） |
| P7 | **进行中 2026-11-17** | 优化路线立项（goal-4f0af8a7）：实测瓶颈是 launch/sync 粒度（Pi CPU ~200 sync/token、Pi GPU ~1879 draw/token；Xe 1.2/4.0 GMAC/s），非 MAC 受限。任务板 task-12..16：P0 算子打包/融合（A1 六路 lerp 合并、A2 r/k/v 合并、A3 四组低秩合并，p7-pack 进行中）→ P1 分块 prefill + 状态快照 → P2 Pi I16_2/head 专项与 Xe 0.4B。0.4B 已打包并测 PPL（q8r 6.3432 / q8 7.0384 / q4 7.7273；f32 oracle 5.7405；量化代价 +0.100/+0.204/+0.297 nats·token⁻¹；q4 334.9MB、贴图 267.8MB → Pi 仍不可行）。**task-12 结果（2026-11-17，Xe）**：A2/A3 在**串行路径**落地——r/k/v 三次回读合成一次、四组低秩 8 次回读合成 2 次（`GPU::matvec_rows` + `Session::rwkv_mv_group`，每矩阵仍各一次 draw/程序/2^k 归一，逐位相同）：syncs/token **167 → 73**、decode **108.6 → 70.0 ms/token（1.55×）**，PPL 9.2919 不动。GPU 路径的 WKV update 48 条 strip 本就是 VC4 规则（Xe 白付 576 draws）：draws/token **1158 → 594**、decode 不变（31.1 vs 31.5）⇒ **Xe 上 draw 几乎免费**。A1 因此不做：768×6 = 4608 fragment，VC4 切 5 条 vs 原 6 次（省 1 draw/层 = 12/token ≈ 6ms/1138），却要 banded-input 全套；A2 的 3·dim 行合并被 `draw()` 的 1024 行 strip 上限抵消（2304 行 = 3 条 = 原来 3 次）；A3 的四个 up 需零填充权重（1.8× MAC）或 9 个采样单元，均为负收益。**板上下一步**：Pi 的 1879 draws 里 576（31%）是 WKV update 的 strip，`kDrawCap` 到底是「每 draw fragment」还是「宽度」决定它是 1 draw 还是 48——`probes/rwkv_probe` F 段可直接验。逐次测量见 git（当前表在 docs/benchmarks.md） |

**P2 开工清单（P1 的交接，按代码位置）**：

1. 删掉 `engine/src/model.cpp:1414` 的 `is_rwkv()` 拒绝块（注释写着 "P2 deletes this block"）。
2. 层循环取张量的方式：`qkv[l]` = `[W_r; W_k; W_v]`（`3d×d`，用 `slice_rows` 切三片）、
   `wo[l]` = `W_o`、`w1[l]` = channel-mix key（`F×d`）、`w2[l]` = channel-mix value（`d×F`）、
   九个 `rwkv_*` f32 向量组（`rwkv_lerp/bias/scale/att_norm/att_norm_2/ln_x/cm_lerp/tok_norm/out_norm`）、
   四对低秩 `rwkv_{w,a,v,g}{1,2}`；L0 的 `rwkv_v1/v2` 为空（记 `v_first`），无门时 `rwkv_g1/g2` 为空。
3. head 输入按 key 17 写 `x·2^-logit_shift`（tile 的 scale 已经带 `2^k`，不要再乘）；
   `logit_shift=0` 是历史路径。
4. **cm_key 的 3072 行**：桌面 Xe 的纹理上限 16384，一条 matvec 可以；**Pi 的 2048 行限制属 P5/P6**
   （那时要按 `split_ffn` 的办法切 2×1536，但它是单矩阵不是 w1/w3 对，不能直接复用 dense 的 chunk 代码）。
5. `pair_ok()` 拒绝 RWKV；prefix cache 明确拒绝（或无 KV 的状态快照，留到 P6）。
6. 数值约束（§四）：**state 用 CPU f32**；残差先按 q16 起步（P4 的 PPL 决定 q24）；sqrelu 输出走
   `I16_2` + key 15 的能力保留。
7. **端到端文本**用 `models/rwkv7/packed/rwkv7-g1d-0.1b.q8.chat.mmcf`（内嵌 .tok 与模板，
   key 8 = bos 0 / eos 261 / stop 261，add_bos 0）；旧的 `rwkv7-g1d-0.1b.q8.mmcf` 没有
   tokenizer，只做纯数值测试。**唯一剩余接缝**：app.cpp 里 `tokenizer.set_rwkv(model.is_rwkv())`
   由 P2 加（P3 已提供 API 但主树暂无调用点）。


## 十、官方推理入口与权重来源（2026-11-16 核实）

**结论：官方实现是 pip 包 `rwkv`（0.8.32），用 `RWKV_V7_ON=1` 打开 v7；权重用 BlinkDL 的 pth。**

```python
import os
os.environ["RWKV_V7_ON"] = "1"     # 必须在 import 之前
os.environ["RWKV_CUDA_ON"] = "0"   # CPU
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
m = RWKV(model="/path/to/rwkv7-g1d-0.1b-20260129-ctx8192", strategy="cpu fp32")
p = PIPELINE(m, "rwkv_vocab_v20230424")   # 包内自带词表 + TRIE 分词器
ids = p.encode("User: Hello, my name is\n\nAssistant: <think")
logits, state = m.forward(ids, m.generate_zero_state())
```

- 环境：`uv` 装 Python 3.12 + **torch 2.15.0+cpu**（`--index-url https://download.pytorch.org/whl/cpu`，
  cuda 不可见）；装在仓库的 `.venv-rwkv/`（gitignored）。**系统 Python 3.14 没有 torch 轮子**，
  这就是用 3.12 的原因。
- 实测（0.1B g1d，fp32，4 线程）：加载 0.49 s（mmap）、prefill **25.7 ms/prompt-token**、
  decode **40.0 ms/token**；greedy 样例 `>\nOkay, the user wants to know how to use`。
- 备选：仓库自带的 `RWKV-v7/rwkv_v7_demo_rnn.py`（RNN 模式，官方脚本，需把 `map_location`
  改成 `cpu`、`USE_CUDA_KERNEL=False`）；Albatross 是官方的高性能引擎，本目标用不到。
- **权重来源**：`BlinkDL/rwkv7-g1` 的 `rwkv7-g1d-0.1b-20260129-ctx8192.pth`（382 MB）。
  `fla-hub/rwkv7-0.1B-g1` 的 safetensors **不是同一个 checkpoint**（实测 max|d|：
  `r_proj` 0.139、`head` 0.176、`emb` 0.121），且 fla 侧已停止跟进——**以 pth 为准**。
  转换：`models/rwkv7/official/pth_to_safetensors.py`（只在 P0/P1 用一次，torch 读 pth，
  写成 fla 命名的 safetensors + config.json + 词表），产物 `models/rwkv7/g1d-hf/`。
- **对拍（同一权重）**：numpy oracle vs 官方包 → **cos 1.00000000、maxabs 8e-5、rmse 2e-5、
  top-1/top-10 全对、12 个 greedy token 逐个相同**。所以 P2 的对拍目标就是官方包。
- 转换踩到的坑（P1 的 packer 会再遇到）：**L0 没有 v_lora**（pth 里的 `v0/v1/v2`
  官方加载器直接置空，不能当损坏）；**`pre_norm` 只在 L0**；形状约定按张量不同——
  fla 把六个 time-mix lerp 向量留成 `(1,1,d)`，其余向量压平，`ffn.x_k` 是平的一维，
  照抄这一套才能让 oracle 与官方对齐。
- **词表是 CRLF**（P0b 发现）：从 pip 包取到的 `rwkv_vocab_v20230424.txt` 是 1,159,262 B / CRLF，
  HF 原版是 1,093,733 B / LF。**packer 读词表必须同时容忍 LF 与 CRLF**，否则每行字节数带上回车、
  全表错一位。全量 65529 条往返 + 2422/2423（`b'\xbf\xbd'` vs 码点 U+0080）已由 P0b 验证通过。

## 十一、P2 验收与量化归因（2026-11-16，lead 独立复核）

容器三档（都内嵌 .tok + 模板 + key 8 bos0/eos261/stop261）：
`rwkv7-g1d-0.1b.q8.chat.mmcf`（每矩阵 scale）、`...q8r.chat.mmcf`（`--q8-row-scale`，382 张量）。

| 配置 | engine↔f32 oracle cos | greedy 前 8 | 说明 |
|---|---:|---:|---|
| q8 每矩阵 scale | 0.99608 | 8/8 | P2 报告，lead 复跑一致 |
| q8 逐行 scale | 0.99803 | 4/8 | cos 升；第 5 个是 oracle 近平局位，greedy 对舍入敏感 |
| q8 逐行 + ACT_F32=1 | 0.99892 | 4/8 | 残差 f32 再抬一点 |

归因（P2 的三方对照，lead 认可）：f32↔oracle-on-容器权重 ≈0.996（每矩阵）/≈0.9999（逐行，仿真），
engine↔oracle-on-容器权重 ≈0.996→ 差在**权重量化 + q16 网格**，**不是层循环**。
所以"cos>0.999"要靠 P4 的量化研究（逐行 scale 已做；残差 q24/key12、matvec 输入幂次缩放待定），
而不是改层循环。greedy 前缀数不是单调指标，读 cos。

P2 修的两个真实数值坑（都有数据）：xv 峰 225 越 q16 rail → 输入除 2^k、矩阵 scale 乘 2^k；
embedding 行峰 0.08 被 q16 网格放大 → RWKV 路径 embedding 走 exact_f32。

遗留（P4）：`engine_on_container_weights.py` 还不认逐行容器的 head part-3 平面（concat 报错），
需补；R-1 残差 q16→q24 用 PPL 定；RWKV_DUMP 接管 STAGE_DUMP 的 RWKV 行。

**SEQ 夹取对 RWKV 不成立（2026-11-16，用户问、lead 修）**：`SEQ … exceeds GL target
limit … clamping` 源于注意力 score/KV 纹理按 seq 增高；RWKV 无 KV/注意力
（`gpu_attn_=false`，状态 O(1)），不建任何 seq 尺寸 GL target，故不应被夹。旧代码在
app.cpp 无条件夹取，会把 RWKV 上下文错夹到 GL 上限（VC4=2048）。已修：app.cpp 对
`is_rwkv()` 跳过夹取；session.cpp 对 RWKV 不分配 seq*kv_dim KV 行与 seq*half RoPE 表。
验证：RWKV SEQ=32768 不再 clamp 且正常跑；minimind3 SEQ=32768 仍 clamp 16384；
`make check` 108/0。RWKV seq 现只受内存约束，Pi 长上下文留 P5 实测。

**思考开关与评测口径（2026-11-16，用户指出、lead 验证）**：模板已接 `open_thinking`：
默认（无 `--thinking`）→ `Assistant: <think>
</think>`（完整空 think 块，假思考直接答，与官方逐字一致）；`--thinking` →
`Assistant: <think`（真思考，数百~上千 token）。**bench_mc 的"首 token 字母 logit"对 RWKV
chat 模式无效**（首 token 是 '>' 闭合标签）；已验证 completion（无 --chat）首 token 即字母
（id 66='A'）。故 RWKV 的 MC 必须 `--completion`；ifeval/translate 用默认 nothink + 足够 --gen。

**评测采样用模型推荐值（用户要求）**：RWKV 推荐 temperature=1.0 / top_p=0.3 /
repetition_penalty=1.2（fla-hub 0.1B-g1 模型卡；generation_config.json 无采样字段）。
ifeval/translate 对 RWKV 用该 preset（多 seed），不用 greedy/引擎默认；bench_mc 为
--gen 0 logits，不受影响。P4 落实并写入 docs/evaluation.md。

## 附录 A：容器 id 与 metadata 提案（P1 的输入）

新 kind 全部落 `kind(8)|layer(6)|part(2)` 的空位（已用：`0x01-0x19`、`0x20`、`0x30`）：

| kind | 名字 | 形状 | dtype | 复用/新增 |
|---|---|---|---|---|
| `0x10` | `rkv`（`[W_r; W_k; W_v]` 三行拼接） | `3d × d` | q8/q4 | 复用 qkv 槽，按 `slice_rows` 切成三片 |
| `0x11` | `wo`（`W_o`） | `d × d` | q8/q4 | 复用 |
| `0x12` | `w1`（`W_cm_key`） | `F × d` | q8/q4 | 复用 |
| `0x15` | `w2`（`W_cm_value`） | `d × F` | q8/q4 | 复用 |
| `0x1A` | `time_mix_lerp` | `6 × d` | f32 | 新 |
| `0x1B` | `time_mix_bias`（w0,a0,v0） | `3 × d` | f32 | 新 |
| `0x1C` | `time_mix_scale`（k_k,k_a,r_k） | `3 × d` | f32 | 新 |
| `0x1D` | `att_norm`（w,b） | `2 × d` | f32 | 新 |
| `0x1E` | `att_norm_2`（w,b） | `2 × d` | f32 | 新 |
| `0x1F` | `time_mix_ln`（w,b） | `2 × d` | f32 | 新 |
| `0x21` | `channel_mix_lerp` | `1 × d` | f32 | 新 |
| `0x22` | `tok_norm`（w,b，全局） | `2 × d` | f32 | 新，layer 位 = 0 |
| `0x23` | `output_norm`（w,b，全局） | `2 × d` | f32 | 新，layer 位 = 0 |
| `0x24/0x25` | w 低秩 `w1(d×r_w)` / `w2(r_w×d)` | | q8 | 新 |
| `0x26/0x27` | v 低秩 `v1(d×r_v)` / `v2(r_v×d)` | | q8 | 新 |
| `0x28/0x29` | a 低秩 `a1(d×r_a)` / `a2(r_a×d)` | | q8 | 新 |
| `0x2A/0x2B` | g 低秩 `g1(d×r_g)` / `g2(r_g×d)` | | q8（无门时可以缺） | 新 |

metadata key 16（`<B f B B B B f>` 之类的小结构，具体布局在 P1 定稿并写进
`docs/checkpoint-container.md`）：`arch=1(rwkv7)`、`head_size=64`、
`r_w/r_a/r_v/r_g`、`norm_eps`；容器版本 4。

metadata key 17（**P0b 新增**，`<B>`）：**logits 输入尺度指数 k**。最终 LayerNorm 的出峰
（g1d：315.6）超 q16 rail 2.5×，所以引擎写 `x·2^-k`、每个 logits tile 的 `scale` 在 pack 时
乘 `2^k`（与 key 15 的 `apply_hidden_scale()` 同一手法）。packer 从 probe 前向的 final-LN
峰值算 `k = ceil(log2(max/127))`，dump 时要打印；没有这个 key 的容器 = `k=0`。
**bias 不进 `Mat`**：它们是 `2 × d` 的 f32 行，读进 `Model` 的 `vector<float>`，
和 `rms_att` 同构（`model.cpp` 的 dtype 0 路径），只有新 LayerNorm 核用得上。

## 附录 B：每层张量数（0.1B 校验）

每层：`rkv` 3d² + `wo` d² + `cm_key` dF + `cm_value` dF + 8 个低秩对 + 12 条 d 向量
（6 lerp + 3 bias + 3 scale）+ 6 条 d 向量（三个 norm 的 w/b）+ 1 条 lerp_k；
全局：embedding + head（各 65536×d）+ tok_norm/output_norm。
0.1B：d=768、F=3072、r=64/64/32/128 → 每层 4·d² + 2·dF + 2·d·(r_w+r_v+r_a+r_g) + 约 25d
≈ 2.36M + 4.72M + 0.39M ≈ 7.47M MAC，12 层 89.6M + head 50.3M ≈ 140M MAC/token
（与评估的 143.7M 一致，差在逐通道项）。

## 十二、优化路线（2026-11-17 立项）

**瓶颈画像（实测反推，不是算力）**

- Pi CPU 969 ms/token ≈ ~200 次 `rwkv_mv`（GPU matvec + 回读 sync）× ~3.6 ms/sync；Pi GPU 1138 ms/token ≈ ~1879 draws × ~0.6 ms/draw；Xe CPU 119 / GPU 36 ms 对应 143.7M MAC 只有 1.2/4.0 GMAC/s。⇒ **launch/sync-bound**。
- 所以优化优先级 = 减少算子数（融合/打包/批处理），不是换更快的数学。

**任务板（Team tasks）**

| 任务 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| task-12 | P0：A1 六路 lerp 合一次 draw、A2 r/k/v 合一次 matvec、A3 四组低秩 down/up 合并 | — | p7-pack 进行中 |
| task-13 | P1：分块 prefill（批 matmul，仅 WKV 递推串行） | task-12 | 待派 |
| task-14 | P1：状态快照 + 多轮/跨请求复用 | task-12 | 待派 |
| task-15 | P2：0.4B 打包与质量验证（Xe） | — | 完成 |
| task-16 | P2：Pi 专项（I16_2 逐元素 A/B、head 量化） | task-12 | 待派 |

**0.4B 实测（Xe，2026-11-17）**

- 容器：q4 334,871,084 B（估算 335MB，命中 0.04%）、q8 453,364,524 B、q8r 454,949,100 B；metadata v4 / 24×1024 / FFN 4096 / 16×64 / ranks 64/64/32/128 + gate 128 / key 17 k=2。
- PPL（SEQ=512，421 token）：q8r **6.3432**、q8 7.0384、q4 7.7273；f32 oracle **5.7405** → 量化代价 +0.60/+1.30/+1.99 PPL（+0.100/+0.204/+0.297 nats·token⁻¹）；0.1B q8r 对照 9.2919（oracle 8.1014，+0.137）。模型层面 0.4B 低 29% PPL / 0.344 nats。q8r < q8 < q4 单调（逐行 scale 在 0.4B 上值 ~10%）。
- 模板结论：`<user_input>` **不是模板问题**，是 q8（每矩阵 scale）的数值漂移；q8r/q4 与 oracle 前两 token 一致。现有 chatml/488B 模板对 0.4B 合适。
- 风险记档：0.4B 残差峰 **80.2**（rail 128，余量仅 1.6×；0.1B 是 3.4×）；sqrelu 76.3、最终 LN 304.19、|state|max 30.7。
- 结论：Xe 上 0.4B q8r 值得（PPL 6.34 vs 9.29，量化代价更低）；Pi 仍只能 0.1B（q4 贴图 267.8MB > 可用 CMA）。

**验证口径**：每步同机 A/B、PPL 裁决、draws/syncs 计数下降、`make -C engine check` 108/0 + `regression.sh` ACCEPT + `shader_hashes` 仅新/改 builder；跨机/跨路径只比 PPL。

**收尾结论（2026-11-17）**

- **Xe（桌面）**：P7 回读合并后 CPU 108.6 → 64–70 ms/token（约 1.6×），GPU ~36 ms；0.4B q8r PPL 6.3432（f32 oracle 5.7405），量化代价 +0.100 nats/token。GPU 融合 + bit-cast f32 state 保留。
- **Pi Zero（VC4）decode**：两条路径都在 ~0.97–1.04 s/token，**已到这套 q8 纹理核的地板**（~127–145 MMAC/s；dense 路径 ~283 MMAC/s，差距来自取指比 1.25 vs 0.5 fetch/MAC）。syncs 167→73、draws 1879→1855、cm 切点 1536+1536→2048+1024、token-shift ping-pong 等改动在板子上都 <2%（噪声内），**draw/sync 不是 Pi 的边际成本**。
- **Pi prefill**：P1 批量 prefill（默认关）93.63 → 36.05 s（2.6×），但 VC4 上 logits 不逐位（max|d|=0.367、top-1 不变）；Xe 上 62.25 → 6.81 s（9.1×，逐位）。
- **定位**：Pi 上 RWKV-7 0.1B q8r ~0.97 s/token vs LFM2.5-230M q4 1133 ms/token（参数少 2.3×），而质量全面更低（MC 0.240/0.310 vs 0.380/0.400；ifeval 0.25 vs 0.562/0.542；translate chrF 0.099 vs 0.328/0.368）。**RWKV 0.1B 在 Pi 上不具竞争力**；0.4B（453 MB）只在桌面 Xe 可用。决定：停止 RWKV 优化，保留已交付的容器/loader/CPU 层循环/分词器/模板、Xe GPU 融合与回归引脚。

