# RWKV-7 (Goose)

The third architecture this engine runs, and the first one that is **not a
transformer block**: no qkv, no KV cache, no softmax, no RoPE.  Each layer keeps
one 64x64 recurrence matrix a head plus two token-shift vectors, so the state is
`O(1)` in context length — 2.36 MB at 0.1B, against a KV cache that grows with
every token.  That is why the engine grew a **third layer loop**
(`Session::forward_rwkv`) rather than a branch in the two it had; the block itself
is in [`engine-internals.md`](engine-internals.md) under "RWKV-7 (recurrent)
block", the container format in
[`checkpoint-container.md`](checkpoint-container.md) under "v4".

**Delivered, and not being optimised further** (2026-11-17).  It runs on Iris Xe
and on the Pi Zero, the CPU and the fused GPU loops agree to within PPL, and the
regression gate pins both.  It is also *not competitive*: on the board 0.1B costs
~0.97 s/token and scores below the smaller models beside it
([`evaluation.md`](evaluation.md)), so the speed work stopped where the
measurements said it should ("Known gaps" below).  The 0.4B is desktop-only.

| | |
|---|---|
| containers | `models/rwkv7/packed/` — 0.1B and 0.4B, `q8` / `q8r` / `q4`, chat and numeric |
| 0.1B shape | 12 layers, dim 768, FFN 3072, 12 heads of 64, low rank 64/64/32/128, 191M params, 143.7M MAC/token |
| 0.4B shape | 24 layers, dim 1024, FFN 4096, 16 heads of 64, the same ranks plus a gate at 128, 450.8M params, 393M MAC/token |
| state | `heads x 64 x 64` f32 a layer (2.36 MB at 0.1B, 6.29 MB at 0.4B) + `2 x dim` token shift; **constant in context length** |
| activations | LayerNorm, not RMSNorm: mean and variance, with a bias vector, five times a layer |
| FFN | squared-ReLU on one matrix (`W_cm_key`, `W_cm_value`), not a fused SwiGLU pair |
| vocab | 65536, **untied** head (two `65536 x dim` matrices, 34.9% of 0.1B's MAC) |
| tokenizer | greedy longest match over a 65536-entry list, **no BPE merge** — `encode_greedy`, mode selected by the container's arch |
| template | the vendor's short one: `User: …\n\nAssistant:` plus the `<think` markers |
| specials | bos 0, eos 261 (the text `"\n\n"`), stop 261, `add_bos 0` |
| metadata | key 16 (arch and hyper-parameters), key 17 (the head input's scale — see below) |

Two container fields exist because of this model and are documented where they
belong: **key 16** carries `arch, head_size, r_w, r_a, r_v, r_g, norm_eps` and a
gate flag, and **key 17** carries `logit_shift`, because the final LayerNorm's
output peaks at **315.6** — 2.5x the q16 rail — so the engine writes `x * 2^-k`
and the packer folds `2^k` into every logits tile's scale.

## Run

```bash
cd engine
# Xe, the pinned path: q8r chat container, fused GPU loop
MODEL=../models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
RWKV_GPU=1 GPU_LAYER=1 KV=q16 SEQ=512 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 32 --greedy

# the same container on the CPU layer loop — a SEPARATE pin, not comparable
MODEL=../models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
GPU_LAYER=0 KV=q16 SEQ=512 ./engine_mm3 --chat --prompt "用一句话介绍你自己。" --gen 32

# Pi Zero (runtime dir, no checkout)
MODEL=models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
RWKV_GPU=1 GPU_LAYER=1 KV=q16 SEQ=512 ./bin/engine_mm3_pi --chat \
  --system "You are a helpful assistant." --prompt "用一句话介绍你自己。" --gen 32 --greedy
```

* `RWKV_GPU=1` selects the fused loop (27 `rwkv_*` builders).  Without it the
  serial loop runs the same mathematics on the CPU with GPU matvecs, which is the
  reference path.  **Cross-path acceptance is PPL, never the greedy string**: the
  two loops differ by temp-0 chaos on a 0.1B, and `AGENTS.md` says so.
* The WKV state travels as **bit-cast f32 in RGBA8 ping-pong textures**
  (`floatBitsToUint` / `uintBitsToFloat`), which needs an ES3 context; without one
  the state stays on the CPU (`RWKV_WKV_CPU=1` forces that path).  This is the
  fix for the one measurement that would otherwise have forced a lossy encoding:
  1/256 integer state costs **+0.1543 nats/token** (NLL +65%) on this model.
* `SEQ` is not clamped for this architecture and there is no KV texture to clamp
  to: `SEQ=32768` runs, bounded only by memory.
* `pair_ok()` refuses an RWKV model, so a prompt always prefills through the
  serial loop; the batched prefill (default off) applies the matvecs to a block
  and keeps only the recurrence per position.

## Containers

| file | size | what |
|---|---:|---|
| `rwkv7-g1d-0.1b.q8.mmcf` | 191.8 MB | numeric only (no tokenizer, no template) — pure forward tests |
| `rwkv7-g1d-0.1b.q8.chat.mmcf` | 192.7 MB | one scale a matrix, tokenizer and template embedded |
| `rwkv7-g1d-0.1b.q8r.chat.mmcf` | 193.5 MB | `--q8-row-scale` — **the pinned container** |
| `rwkv7-g1d-0.1b.q8r.q24.chat.mmcf` | 193.5 MB | the same body with a q24 residual (`--res q24`) |
| `rwkv7-g1d-0.1b.q4.chat.mmcf` | 158.9 MB | 4-bit body — runs, but Chinese degenerates into repetition |
| `rwkv7-g1d-0.4b.q8r.chat.mmcf` | 454.9 MB | **the 0.4B to ship** |
| `rwkv7-g1d-0.4b.q8.chat.mmcf` | 453.4 MB | one scale a matrix: PPL 7.04 against q8r's 6.34 |
| `rwkv7-g1d-0.4b.q4.chat.mmcf` | 334.9 MB | 4-bit: collapses into tag repetition in generation |

**The row scale is worth its 0.2% of file size** at 0.4B: `rkv[12]` has a row
skew of 103, and one scale a matrix costs +0.70 PPL there (+0.11 at 0.1B).  All
four matrices of the block (rkv, wo, cm_key, cm_value) take q8, q4 and
`--q8-row-scale` without a new kernel.

`cm_key` is `3072 x 768` at 0.1B, above VC4's 2048-row draw limit, so
`split_rwkv` cuts the channel-mix the way `split_ffn` cuts a dense FFN (Xe does
not trigger it, and the cosine is unchanged).  The 0.4B's FFN is 4096 = 2x2048 —
exactly on the line where a VC4 draw is silently dropped — which is the other
reason 0.4B stays on the desktop.

## Quality

Perplexity, `engine/data/ppl_test.txt`, `SEQ=512`, the container's own tokenizer
and template (not comparable with the transformer rows: different tokenizer,
different `SEQ`):

| container | path | PPL | nll |
|---|---|---:|---:|
| 0.1B q8r | GPU fused (`RWKV_GPU=1`) | **9.1850** | 2.2176 |
| 0.1B q8r | CPU loop | **9.2919** | 2.2230 |
| 0.1B q8r | GPU, CPU f32 state (`RWKV_WKV_CPU=1`) | 9.1781 | — |
| 0.1B f32 oracle (numpy, same ids) | — | 8.1014 | 2.09 |
| 0.4B q8r | GPU fused | **6.3432** | 1.8474 |
| 0.4B q8 | GPU fused | 7.0384 | 1.9514 |
| 0.4B q4 | GPU fused | 7.7273 | 2.0448 |
| 0.4B f32 oracle | — | 5.7405 | 1.7476 |

So the engine sits 0.13 nats/token above a f32 forward of the same weights at
0.1B (0.10 at 0.4B through the row scale), and the GPU loop is *slightly* better
than the CPU one — inside the encoding's own noise, which is why both are pinned
separately.

The three batteries are in [`evaluation.md`](evaluation.md), which owns the
protocol: **multiple choice must run in completion mode** (`--completion`), because
the chat template's first token is the think marker's `>`, not a letter, and the
instruction-following and translation rows use the model card's sampling preset
(`--temp 1.0 --topp 0.3 --repeat-penalty 1.2`) instead of greedy.  The summary:

| 0.1B q8r | score | the models beside it |
|---|---:|---|
| multiple choice, 100 items, completion | 0.240 / 0.310 (2 seeds) | LFM2.5-230M 0.380 / 0.400 |
| instruction following, 24 cases | 0.250 | LFM2.5-230M 0.562 / 0.542 |
| translation, 30 sentences, chrF | 0.099 | LFM2.5-230M 0.328 / 0.368 |

On the board that is the whole verdict: **~0.97 s/token for a model that scores
below a 230M transformer which costs 566 ms a token** (2026-11-16 numbers,
`benchmarks.md`).  The 0.4B is the opposite story — PPL 6.34 against 9.29, and in
generation a qualitative jump: it answers 北京 where the 0.1B answers 上海/成都,
and it translates instead of repeating the source sentence — but it needs 453 MB
of texture, so it is Iris Xe only.

## Known gaps

| # | what | status |
|---|---|---|
| 1 | **the board's speed is the q8 texture kernel's floor**, not a leftover inefficiency: both loops sit at 0.97–1.04 s/token (~127–145 MMAC/s against the dense path's 283), and the sync/draw reductions (167→73 syncs, 1879→1855 draws, a different cm split, token-shift ping-pong) each moved the board by <2% | accepted; optimisation stopped 2026-11-17 |
| 2 | **prefill is the wall, not decode**: 512 tokens cost 36.0 s on the board (93.6 s without the batched prefill) and 6.8 s on Xe (62.3 s) — the recurrence is per position, so the batching only covers the matvecs.  The board's logits are not bit-identical under it (max\|d\| 0.367, top-1 unchanged) | batched prefill is default-off |
| 3 | **the chat markers are a weak point of the checkpoint**: the 0.4B invents `<translation_output>`, `<answer>` wrappers and, worst, echoes the user turn as `<user>…</user>`; the 0.1B q8 body answers `\n<user_input>\n…`.  Rendering and token ids were checked and are identical to the oracle, so this is the model, not the template | documented, not fixable here |
| 4 | **`--thinking` is not a switch this family honours the way the dense models do**: the 0.1B g1d was trained to open `<think>` itself, so the no-think template does not suppress it, and two translation prompts spent all 80 tokens inside the block | use `--thinking` deliberately and give it budget |
| 5 | **pair prefill is refused** (`pair_ok()`), so a hybrid-like shortcut is not available here | by design |
| 6 | **the prefix cache refuses an RWKV model** — it would need a state snapshot (version 5), not a KV copy | not implemented |
| 7 | **0.4B is desktop-only**: 453 MB of texture against a ~170 MB board budget | by size |

## Measuring it

```bash
cd engine
# PPL against the pinned container; the two loops are separate pins
MODEL=../models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
RWKV_GPU=1 GPU_LAYER=1 KV=q16 SEQ=512 ./engine_mm3 --ppl data/ppl_test.txt

# where a layer diverges: RWKV_DUMP (per layer attn/LN/w-a-g/state checksums)
MODEL=... RWKV_DUMP=/tmp/rwkv.txt RWKV_GPU=1 ./engine_mm3 --ppl data/ppl_test.txt

# the reference: the official pip package, not a re-implementation
pip install rwkv==0.8.32   # plus torch on CPU; RWKV_V7_ON=1 before the import
python3 models/rwkv7/official/pth_to_safetensors.py   # pth -> the fla naming
python3 engine/tools/ref_rwkv.py ...                  # the numpy f32 oracle
```

The oracle to trust is the official `rwkv` package with `RWKV_V7_ON=1` (or its
`RWKV-v7/rwkv_v7_demo_rnn.py`); on the same g1d pth it agrees with the numpy
forward at **cos 1.00000000, max abs 8e-5, 12/12 greedy tokens**, and llama.cpp's
Q5_K_M GGUF agrees 8/8 on argmax.  Do **not** use
`train_temp/rwkv7_train_simplified.py` as the reference: its decay is
parameterised `-softplus(...)-0.5`, which is not `exp(-sigma(z)/sqrt(e))` of the
published weights.  The vocab file matters too — the pip package ships it CRLF
(1,159,262 B) and HF ships it LF (1,093,733 B), so the packer tolerates both.

The regression pins (`RWKV_EXPECT`, `RWKV_PPL_EXPECT`, `RWKV_PPL_SEQ`,
`RWKV_GPU_EXPECT`) live in the machine's local override, and the GPU greedy
tripwire is a different string from the CPU one by design.
