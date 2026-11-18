# BitCPM-CANN-0.5B

The first model in this repository whose **weights are ternary and exact**: 168 of
its 169 matrices are `{0, ±s_row}` (37.3% zeros), so the q8 body is a *lossless*
copy of them (`docs/bitcpm-assessment.zh-CN.md` has the measurements and the
work list that produced this support).  It is MiniCPM4-0.5B trained with
1.58-bit QAT on Ascend and released as `openbmb/BitCPM-CANN-0.5B`; the weights
are pseudo-quantized bf16, so nothing about running it needs CANN or an NPU.

| | |
|---|---|
| container | `models/bitcpm-0.5b/bitcpm-0.5b.q8.mmcf` (434 MB) |
| shape | dim 1024, 24 layers, 16 heads / 2 kv, FFN 4096, vocab 73448 (tied head), 434M params |
| template | MiniCPM4's own (ChatML variant, no system special case), embedded as key 9 |
| specials | bos 1, eos 2, stop 73440 (`<|im_end|>`), `add_bos 1`, `seg_prefix 1` |
| residuals | default `RES=q16` (peak 41), `RMS_STORE=f32` **required** (lane sums 5.4e4) |
| FFN hidden | peak 184 at layer 23, so key 15 stores that layer `*2^-1` |

## Run

```bash
cd engine
MODEL=../models/bitcpm-0.5b/bitcpm-0.5b.q8.mmcf RMS_STORE=f32 \
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --prompt "用一句话介绍你自己。" --gen 32 --greedy
```

`RMS_STORE=f32` is not optional: the residual's lane sums reach 5.4e4 against
the 24-bit store's 2048 and the answer degenerates without it (the same switch
SmolLM2 needs, for the same reason).  On Iris Xe this is 52.5 ms/token decode (median of three, 46.3 / 53.8 / 52.5) and
~1 s of prefill for a 34-token prompt (2026-11-18, `TIMING=1`, `KV=q16`).

## Pack

```bash
python3 engine/tools/mmcf_pack.py --in models/bitcpm-0.5b --quant q8 --seq 1024 \
  --tok models/bitcpm-0.5b/bitcpm.tok \
  --out models/bitcpm-0.5b/bitcpm-0.5b.q8.mmcf \
  --name bitcpm --template chatml --chat --system "" \
  --chat-template templates/minicpm4.jinja --head-dedup \
  --bos 1 --eos 2 --stop 73440 --add-bos 1
```

`--in` takes the directory as it comes from HuggingFace: the release ships only
`pytorch_model.bin`, and `mmcf_hf.load_minimind` reads that torch zip through
`engine/tools/pth_reader.py` when there is no `model.safetensors`, so no
conversion step and no torch install.

`--head-dedup` is right here: the head is tied, and after the fold (below) the
two readers want the *same* scale, so the vocabulary costs nothing.

## Two things the container states that no other container does

**Metadata key 18 — the rope table.**  The model's `rope_scaling` is
`longrope` with 32 per-dimension factors, i.e. 32 different bases on a 64-wide
head, which one theta cannot express.  In this checkpoint `short_factor ==
long_factor` and `max_position_embeddings == original_max_position_embeddings`,
so the table is *static* and the packer writes it (`freq_i = theta^(-2i/head_dim)
/ factor_i`).  A `longrope` whose two tables differ, or that stretches, is
refused by name: the container has room for one table.

**Metadata key 10's third byte — the prefix policy.**  The tokenizer is
SentencePiece (`▁` marks a word start) and the reference runs its normalizer on
every text run between added tokens, so `user` after `<|im_start|>` is
`▁user` (3060), not `user` (1836).  The engine's historical behaviour puts one
marker at the front of the whole prompt; with `seg_prefix` set it marks every
run.  Measured on this model: logits cosine 0.912 with the old behaviour,
1.000000 with the container saying what it wants (a byte-level vocabulary
produces the same tokens either way, which is why this is a container field and
not a new default).

## The three constants are folded at pack time

MiniCPM applies `scale_emb = 12` to the embedding, `scale_depth/sqrt(layers) =
1.4/sqrt(24)` to every residual increment, and divides the head's input by
`hidden/dim_model_base = 4`.  The block is RMSNorm-only and bias-free, so its
output is exactly invariant to a rescaling of the residual stream and all three
fold into weights the container already stores with an f32 scale each:

```text
embedding table   kept exactly as the checkpoint writes it   (k = 1/scale_emb)
o_proj / down_proj rows   x (scale_depth/sqrt(layers)) / scale_emb
rms_final         x (dim_model_base/hidden) / (k*scale_emb)  = x 0.25
```

`k` is free; `k = 1/scale_emb` is the choice that keeps the *head's* input at
1/4 (RMS 0.25, comfortable in the q16 activation store) instead of 1/48 (RMS
0.02, five steps of the store -- the logits come out as noise).  Measured
against a float32 forward of the same checkpoint: cosine 0.885 with `k = 1`,
0.9868 with `k = 1/scale_emb`, and the folded model against the unfolded one
1.000000 (`docs/benchmarks.md`, 2026-11-18).

The fold is also what makes `RES=q16` enough: the residual peak is 41 in these
units instead of 802.

## Quality and agreement with the float reference

Iris Xe, 2026-11-18, `RMS_STORE=f32`, greedy:

| | |
|---|---|
| multiple choice, 100 items (`engine/eval/bench_mc.py`) | **acc 0.690**, acc_cal 0.660, fmt 0.690, p_gold 0.637 (null 0.254) |
| the same battery on the **t3 container** | acc 0.680, acc_cal 0.670, fmt 0.670, p_gold 0.640 — the two containers agree |
| same battery, for scale | pocket qat 0.480, LFM2.5-230M 0.380, MiniMind3 q8 0.220 |
| `--ppl` on `engine/data/ppl_test.txt`, SEQ=768 | 36.0302 (nll 3.5844, 247 tokens) |
| logits vs the float32 numpy forward (`models/bitcpm-0.5b/probe_forward.py`) | KL 0.018 nats, top-20 19/20, entropy 1.546 vs 1.512 |
| per-token nll on a 667-token mixed text, 4 positions | mean 3.107 against 3.024 |

The `--ppl` path encodes without the leading space marker (it treats the file as
a continuation), and this model pays 16 nats on its first token when the marker
is missing; the median position is nll 2.86.  PPL is not comparable across
tokenizers anyway (a 73448-entry Chinese-first vocabulary against SmolLM2's
49152 byte-level one): the number here is a within-model regression pin, not a
ranking.

The chat answer is fluent at 32 tokens
(`你好，我是AI助手，我的名字是李华，我来自中国北京，目前是一名计算机科学家。`)
and it stops on `<|im_end|>` when the prompt is short enough to be answered in
one paragraph.

## The 2-bit ternary container

`--quant t3` writes the body at 2 bits a weight (dtype 7, 16 weights an RGBA8
texel, four a byte, bit 0 magnitude and bit 1 sign, plus the per-row scale plane
the q8 path already has).  It is only written for a matrix that *is* per-row
ternary: `t3mat()` certifies that, and it decodes what it wrote and refuses if
the round trip is not exact, so the format cannot quietly lose a bit.

```bash
python3 engine/tools/mmcf_pack.py --in models/bitcpm-0.5b --quant t3 --seq 1024 \
  --tok models/bitcpm-0.5b/bitcpm.tok \
  --out models/bitcpm-0.5b/bitcpm-0.5b.t3.mmcf \
  --name bitcpm --template chatml --chat --system "" \
  --chat-template templates/minicpm4.jinja --head-dedup \
  --bos 1 --eos 2 --stop 73440 --add-bos 1
```

`models/bitcpm-0.5b/bitcpm-0.5b.t3.mmcf` is 167.5 MB against the q8 container's
435.4 MB, and it is numerically equivalent or better (per-row scales vs the q8
path's one scale a matrix): the same chat answer, KL 0.0278 against the float32
forward where q8 measures 0.0288, and fmt 1.000 on a 10-item multiple-choice
slice.

**The GPU kernel does not exist yet** (Phase B2).  Until it does the loader
expands a t3 body into q8 levels on the host -- the exact dequantisation a
`--q8-row-scale` q8 body would have carried -- and prints

```text
t3 body: 120 matrices expanded to q8 on the host (85 MB on disk -> 342 MB in RAM)
```

That is the packer's acceptance path, not the shipping one: it costs 4x the
file's memory, which is precisely why the board cannot run this container yet.
Measured on the Pi Zero (2026-11-18, `docs/benchmarks.md`): the board parses the
file (`mmcf_dump.py`: 246 tensors, CRCs ok, `quant t3`) and reads it at 13 MB/s,
the engine that predates the kind refuses it by name (`unknown dtype 7`), and the
engine that reads it is OOM-killed at 280 MB of the 342 MB the expansion needs.
A t3 container is a *file* until the matvec exists.

## Not done

- **The Pi Zero cannot hold it**: q8 is 434 MB and q4 is 337 MB (the q4 body
  pays 8 bytes of scale/min per 32 values), against the ~180 MB the board has.
  The route is Phase B in `docs/bitcpm-assessment.zh-CN.md`: a 2-bit ternary
  kind (90.8 MB) plus a q8 embedding (75.2 MB), which is SmolLM2's footprint.
  The **pricing is now measured** (`engine/probes/t3mv_probe`, both machines,
  `docs/benchmarks.md` 2026-11-18): a 2-bit matvec costs 2.44 ns/MAC on VC4
  against q8's 2.98, i.e. **1.22x**, and the ES2 float decode is 2.5x *slower*
  than q8, so the kind is only worth writing on the ESSL 3.00 path.  Model-level
  that is 1.10 s/token of matvec against 1.29 -- 2-bit buys the *fit*, and a
  fifth off the decode, not four times.
- **A t3 kernel.**  The container kind is in (dtype 7); the matvec is B2.  What
  the probe priced: 2.44 ns/MAC on VC4 against q8's 2.98, so the kernel buys
  1.22x on top of the memory win, and the ES3 uint decode is the only viable
  form (the ES2 float decode is 2.5x *slower* than q8).
- **q4 is not packed.** `q4mat`'s 32-value min/max grid puts 0 half a step off,
  so every zero becomes `±s/15`; that is the one place this model's weights are
  not lossless.  Judge it with PPL before shipping a q4 container.
- **SEQ above 1024** is untested.  The rope table covers 32768 and the container
  claims 1024; the KV textures are the limit (`docs/long-context.md`).
- The **CPU layer loop** (`GPU_LAYER=0`) reproduces the GPU numbers here, which
  is unusual for the models in this tree (it degenerates on SmolLM2 and supra2)
  and worth keeping as a second acceptance path.
