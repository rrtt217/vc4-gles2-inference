# Model formats

Two layers of format live in this repository, and they are easy to confuse:

1. **What the engine loads — exactly one thing: an MMCF container.**  See
   [`checkpoint-container.md`](checkpoint-container.md).  `Model::load()`
   accepts `"MMCF"` and refuses everything else, naming the packer in the
   message.  Version 2 is dense, version 3 adds the mixture-of-experts block;
   one reader takes both.
2. **What the packer consumes — the legacy layouts.**  `Q8M1`, `Q8M2`,
   `I16M1`, `I16M2`, `Q4M1` and llama2.c fp32 are *conversion inputs*.  They
   are documented below because `engine/tools/mmcf_pack.py` reads them, not
   because the engine can run them.

The legacy readers were deleted in `fcc4095`; before that the engine carried
four of them plus per-format magics.  Do not reintroduce a reader — extend the
container and the packer.

## What the engine runs

| model | family | notes |
|---|---|---|
| MiniMind3 64M | Qwen3-style dense, tied embedding | 8 layers, 768 hidden, 2432 FFN, 8 Q / 4 KV heads, head_dim 96, vocab 6400, theta 1e6, per-head Q/K RMSNorm, ChatML + `<think>` |
| MiniMind3-MoE 198M-A64M | Qwen3-MoE (container v3) | the same attention and tokenizer, FFN replaced by 4 experts of width 2432 and a 4x768 router, top-1 with a normalised gate; only one expert is read per token, so decode is dense-speed and the cost is 3x the memory |
| MiniMind2 Small | Qwen-style | i16/q8/q4; weaker instruction following |
| Supra2-100M | Qwen3-shaped | see `models/supra2` |
| Feng30M | Llama-shaped | see `models/feng30m` |
| TinyBrainBot-100M-v5.1 | Llama-shaped, tied embedding | `<\|user\|>`/`<\|assistant\|>` markers, no system turn; **q8 is not usable, use q4 or i16** |
| TinyChat15M, stories15M/42M/260K | llama2.c TinyStories family | base completion models, not chat models |

Every one of them is converted to `.mmcf` before use; `models/` keeps the
legacy source next to the container so a re-pack is possible.

GQA is genuinely supported by the engine, not emulated: `GROUP = heads /
kv_heads` and the attention shader indexes `int kvh = h / GROUP`.  MiniMind3
(8 Q / 4 KV) exercises it.  Nothing else about a new model is generic — the
tokenizer, the rope layout, the template and the embedding grid are all
per-checkpoint, so a new architecture is a conversion investigation, not a
config change.

## Templates in the store

`templates/` holds the chat templates this repository ships, so a container can
be re-packed without going back to HuggingFace:

| file | used by | provenance |
|---|---|---|
| `minimind3/chat_template.jinja` | MiniMind3 | upstream MiniMind3, kept with its model source |
| `templates/chatml.jinja` | MiniMind2 / Qwen-family | written here to be byte-identical to the renderer it replaces |
| `templates/tinybrainbot.jinja` | TinyBrainBot-100M | verbatim from the model's `tokenizer_config.json` |

The packer also accepts a `tokenizer_config.json` directly
(`--chat-template models/<m>/tokenizer_config.json`), which is what most HF
checkpoints will want.

Two of the three are verified by token equality against the C++ renderer they
replace: MiniMind3 q8 and MiniMind2 q8 produce **identical `TOKENS_OUT`** through
the kind path and the template path.  TinyBrainBot is the exception, because its
hand-written kind-3 renderer was never the model's trained format — see below.

## Legacy layouts (packer input)

All are little-endian and use the same tensor order; only the header and the
weight encoding differ.

### I16M1

> The layouts below are **packer inputs only**.  `engine/quantize/*.py` (the
> scripts that wrote them) is gone: `mmcf_pack.py --in <hf dir> --quant i16|q8|q4`
> quantizes straight from safetensors, and `--in <legacy file>` still reads an
> old file produced elsewhere.  Nothing in this repository writes these formats
> any more.


```
u32 magic = 0x4931364D ("I16M")
i32 dim, hidden, layers, heads, kv_heads, vocab, seq
f32 rope_theta
i32 rope_half     // 0 = adjacent pairs, 1 = half-split
i32 split         // 1 = separate W1/W3, 0 = fused W13

i16 emb_q[vocab*dim]
f32 emb_scale[vocab]
f32 rms_att[layers*dim]
f32 rms_ffn[layers*dim]
f32 rms_final[dim]

per layer:
  mat qkv [dim+2*kv_dim, dim]
  mat wo  [dim, dim]
  if split: mat w1 [hidden, dim], mat w3 [hidden, dim]
  else:     mat w13 [2*hidden, dim]
  mat w2 [dim, hidden]

logits tiles: mat [<=1536, dim]

mat:
  i32 rows, i32 cols, f32 scale, i16 q[rows*cols]
```

Q8M1 uses the same layout with int8 data and no rope metadata.  MiniMind q8
needs theta 1e6 and rotate-half; I16M1 stores these fields, Q8M1 does not — the
packer therefore **refuses to guess** (see `checkpoint-container.md`).

The packer's q8 output always writes `split=1`, splitting a fused W13 into
W1/W3.  The split layout keeps separate q8 scales and takes the paired q16
W1/W3 path.  The runtime also supports fused q8 W13 through
`matvec_q8_pair_fused`, so old fused files remain convertible.

### I16M2 / Q8M2

MiniMind3 uses a Qwen3-style block: per-head `q_norm` and `k_norm` RMSNorm
before RoPE.  These formats keep the I16M1/Q8M1 layout but add:

```
u32 magic (I16N=0x4931364E, Q8M2=0x51384D32)
... same 7 ints, theta, half, split ...
i32 qk_norm          // 1 = q_norm/k_norm present
... embeddings, scales, rms_att, rms_ffn, rms_final ...
f32 q_norm[layers*head_dim]
f32 k_norm[layers*head_dim]
... same per-layer mats ...
```

Q8M2 stores theta/half itself; no environment override is needed.

### Q4M1

Q4M1 is written by `mmcf_pack.py --quant q4`, always from the float weights
(q8 is never a q4 source; I16 is an allowed one when a legacy file is the
input):

```text
u32 magic = 0x51344D31 ("Q4M1")
i32 dim, hidden, layers, heads, kv_heads, vocab, seq
f32 rope_theta
i32 rope_half, split, qk_norm

q8 embedding + f32 row scales
f32 rms_att, rms_ffn, rms_final
f32 q_norm/k_norm (optional)

per layer:
  q4 matrix qkv, wo, W1/W3 or W13, W2
  q8 logits tiles

q4 matrix:
  i32 rows, cols, groups, f32 param_scale
  u8 packed nibbles [rows * cols/2]
  i16 block_scale[rows * groups]
  i16 block_min[rows * groups]
```

groups = ceil(cols/32); scale/min are q16 values multiplied by
`param_scale`. Q4 matrices require `cols % 32 == 0`. Embeddings and logits
stay q8.

## Converting a model

```bash
# One command per precision, straight from safetensors.  --seq is the KV length
# recorded in the container; the rope configuration comes from config.json.
for q in i16 q8 q4; do
  python3 engine/tools/mmcf_pack.py --in models/<model> --quant $q --seq 512 \
    --tok models/<model>/<model>.tok \
    --out models/<model>/<model>.$q.mmcf \
    --name <model> --template chatml --chat \
    --chat-template models/<model>/tokenizer_config.json --verify
done

# confirm what the engine will read
python3 engine/tools/mmcf_dump.py models/<model>/<model>.q4.mmcf --check
```

A llama2.c fp32 `.bin` is the other accepted source: `--in models/llama2c/
stories15M.bin --quant i16 --tok .../tokenizer.bin --template plain --chat`.

### The packer checks the residual and FFN-hidden rails

The engine's residual stream is q16 at 1/256, so it clamps at **127.996**.
The FFN's SwiGLU hidden travels through the same activation format and clamps at
the same value.  A model over either rail still packs and still runs, but from
a truncated state — supra2-100M's residual reaches 2300 and it echoes the
prompt, and MiniMind3-MoE's expert hidden reaches **250** at the chat template's
`<|im_start|>` positions (a 49% cut of that hidden) while its residual peaks at
118.  Because the packer is the only cheap place to notice, `mmcf_pack.py` runs
a short float32 calibration before writing — three 8-token vocabulary runs plus
one run per special/added token (ids from `tokenizer.json`, so the template
positions a chat prompt actually uses are exercised; position 0 depends only on
its own embedding, which is what the added-token runs exploit) — about 2 s, and
prints both peaks:

```
residual: peak 118.7 at layer 6 (rail 128.0), per layer 1 21 30 110 111 118 119 95
ffn hidden: peak 249.9 at layer 7 (activation rail 128.0), per layer 4 51 35 163 7 32 10 250
```

and then, for either stream:

| peak | behaviour |
|---|---|
| residual < 96, hidden <= 127.996 | silent |
| residual 96 – 512 | warning, packs |
| hidden > 127.996 | warning with the cut depth, packs |
| either > 4x the rail (512) | **refuses**, exit 1, names the ratio and points at `models/README.md` |

`--allow-over-rail` packs anyway; `--no-rail-check` skips the calibration.  A
legacy source has no float weights to run, so the packer prints a note instead
of guessing.  A residual over the rail is a packing decision; a *hidden* over
the rail is handled by **key 15** — the packer writes the smallest per-layer
exponent `k` whose store fits, the serial FFN kernels write the hidden
`*2^-k`, and the engine folds `2^k` back into w2.  `--no-rail-check` writes no
key and keeps the old clamped behaviour.  See
[`checkpoint-container.md`](checkpoint-container.md), "The activation rail".

The arithmetic is the same as the converters that used to exist, which is why a
container packed from safetensors is **byte-identical** to one packed from the
legacy file it replaces — measured on minimind3 i16/q8/q4 and on stories15M.

The source type is detected from `--in`: a directory is an HF checkpoint
(`config.json` + `model.safetensors`, BF16/F16/F32 read from the header), a file
whose first word is a legacy magic is a legacy checkpoint, anything else is
assumed to be llama2.c.

### What the packer refuses

The engine has exactly one block: RMSNorm, silu SwiGLU, RoPE with the angle in
the shader, grouped-query attention, no bias tensors, no sliding window.  A
checkpoint outside that shape is refused at `--in`, before any tensor is read
(`check_source()` in `engine/tools/mmcf_hf.py`).  It used to be accepted and
packed, with the unsupported part silently dropped — the file looked fine and
the answers were wrong, which is the worst failure mode this tool has.

| the source is | the packer says |
|---|---|
| a GPT-2 / GPT-NeoX / OPT / Falcon / Mamba / RWKV / Mixtral block (`model_type`) | names the block and what it needs (LayerNorm, gelu, learned positions, a recurrence, routing); `docs/model-formats.md` is where the list lives |
| any checkpoint with `*.bias` tensors (Qwen2 with `attention_bias`, GPT-2, pythia) | prints up to three of them and stops — the matvec kernels add no bias |
| `hidden_act` other than silu/swish | names the value (`gelu_new`, `gelu`) |
| `rope_scaling` other than default | the container carries a theta and a half/plain convention, nothing else |
| `head_dim` that is not `hidden_size / num_attention_heads` | both numbers — the engine derives the head width |
| `tie_word_embeddings: false` with no `lm_head.weight` | refuses rather than use the embedding as the head |
| a mixture-of-experts block that is not the supported one: Mixtral, a shared expert, `num_experts_per_tok > 1`, `norm_topk_prob: false`, or sparse layers that are not every layer | names the condition.  The one MoE the engine runs is the Qwen3-MoE layout with top-1 and a normalised gate (`minimind-3-moe`: 4 experts, expert_hidden 2432); see `docs/checkpoint-container.md`, "MoE" |

What is still accepted but warned about: `sliding_window` (the attention is
dense, so keep `SEQ` inside the window).  What is accepted because the engine
really does support it: tied or untied heads, GQA, per-head q/k norms
(`--qk-norm` is implied by the tensors), a tokenizer with or without added
tokens.

`mmcf_requant_q8.py` is the shorter route when only q8 is wanted: it rebuilds
the q8 matrices inside a container from the same model's i16 container, so the
tokenizer, template and rope metadata are not re-derived.

### Conventions that used to live in shell history

| convention | where it lives now |
|---|---|
| `rope_theta` / rotate-half | metadata key 2.  A `Q8M1`/`I16M1` source has no rope at all, so the packer refuses to guess: `--theta` plus `--rope-half`, `--rope-plain` (llama2) or `--assume-llama2-rope` |
| BOS / EOS / stop ids | metadata key 8, derived from the embedded vocabulary.  llama2.c stores `"<s>"` as `"\n<s>\n"`, so matching is on the trimmed token |
| chat template | metadata key 9 (the Jinja text, verbatim); key 4 keeps the kind + default system + name as the fallback |
| BOS policy | metadata key 10 (`add_bos` / `add_eos`) |
| the LM head | the embedding when the checkpoint is tied; `lm_head.weight` when it ships one (2026-11-09).  A `tie_word_embeddings: false` config with no `lm_head.weight` is **refused**, not packed — before that the head silently became the embedding and the container ran but answered nonsense (Minueza-32M, pythia, `genz-translator`) |
| the tokenizer blob | `engine/tools/make_tok.py --input <dir> --output model.tok`: ByteLevel (`Ġ`) or SentencePiece (`▁`) detected from the vocabulary, added tokens laid onto their ids, merges split on the first space |
| tokenizer | the `.tok` blob is inside the container |

## TinyBrainBot-100M-v5.1-Instruct

A HF Llama-shaped model with theta 10000, rotate-half, no qk-norm, tied
embeddings and a SentencePiece BPE tokenizer (32000 entries, `▁` for space,
byte fallback).  Its special tokens are `<s>`=0, `</s>`=1, `<unk>`=3,
`<|user|>`=4, `<|assistant|>`=5, `<|system|>`=6, `<|end|>`=7.  The prompt has
no system turn and no BOS:

```
<|user|> {question}<|assistant|> {answer}<|end|>
```

Template kind 3 emits `"<|user|> " + q + "<|assistant|> "` and disables the
tokenizer's dummy prefix; the stop set is `{7}` (EOS 7), not `</s>`.

**q8 is not usable for this model, and the scale is not why.**  `Q8M1` stores
one scale per matrix (`q8mat` in `engine/tools/mmcf_hf.py` takes `max(|w|)`
over the whole matrix), which used to be blamed for the q8 build's degenerate output.
The measurement says otherwise: the file is a faithful `max|w|/127`
quantization (cosine 0.998-0.9999, per-row median error 2.14% against 1.10% for
the MiniMind3 q8 that works), and a re-quantization at a clipped p99.9 scale
(0.92%) still degenerates while this model's q4, at 7.83% weight error, answers
normally.  The break is in the q8 runtime path, not the weights; see
docs/performance.md, "q8 precision is not why TinyBrainBot degrades".  Use
**q4 or i16** for this model.

It also needs the CPU layer: its `GPU_LAYER=1` path is not correct on VC4
(docs/performance.md, item 12), so `--no-gpu-layer` is the deployable setting.

### With its real template (2026-11-07)

The hand-written renderer above was an approximation: `<|user|> {q}<|assistant|> `
with no `<|end|>` after the user turn.  With the model's own template
(`templates/tinybrainbot.jinja`: a leading space, `<|user|>\n{q}\n<|end|>\n`,
then `<|assistant|>\n`) and the container's stop id, the output changes
qualitatively.  q4 and i16, `GPU_LAYER=0`, greedy:

| prompt | q4, old renderer | q4, real template | i16, real template |
|---|---|---|---|
| `Hello!` | `<\|end\|>: '2+2 = 4''2`-class garbage | `Hi there! How can I help you today?` | `Hi! I'm doing well — how can I help?` |
| `Name three colors.` | — | `<\|user\|>` (degenerate) | `Red, Green, and Blue.` |
| `What is 2+2?` | mangled | `The answer is 2.` (wrong) | `2+2 is equal to 4.` |

So: **i16 is the usable precision for this model**, q4 answers fluently but is
unreliable, and the earlier "the repo picker is wrong" conclusion was partly a
renderer problem, not only a precision one.

The same run turned up an engine bug worth recording: the CLI stopped
generation on the literal token ids 1 and 2, which is MiniMind3's coincidence.
TinyBrainBot's EOS is 7, so it never stopped and emitted `<|end|>` until the
length limit.  The CLI now uses the container's EOS/stop ids everywhere, like
the worker already did.
