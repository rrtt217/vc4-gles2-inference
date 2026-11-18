# SmolLM2-135M

The model that named a limit the engine had backwards: what blocked it was not
the residual's range and not the FFN hidden, but the **rmsnorm's accumulator**.
It is the model whose pack reports a squared store overflowing by six orders of
magnitude, the first that needs `RMS_STORE=f32` to answer at all, and — unlike
LFM2.5 and BitCPM — the one whose numbers **do not agree between the two
machines**, for a reason that is still open.

This page is the operational record: what the model is, which container exists,
how to run it, what the numbers are on each machine, and what is still broken.
The chronological measurements are in
[`benchmarks.md`](benchmarks.md) (2026-11-18, Xe and Pi Zero).

## The model

| | |
|---|---|
| layers | 30, all attention (no hybrid, no MoE) |
| dim / ffn | 576 / 1536 (SwiGLU) |
| attention | 9 q heads, 3 kv heads, head_dim 64, GQA, RoPE θ=100000 |
| vocab | 49152, **byte-level BPE** (`<\|im_start\|>`=1, `<\|im_end\|>`=2) |
| template | ChatML |
| source | `HuggingFaceTB/SmolLM2-135M-Instruct` |

Two consequences that cost time and are worth stating up front:

- **The byte-level BPE chops Chinese into 1.76 bytes/token.**  The house PPL
  corpus is Chinese, so this model's PPL is only comparable against *itself*.  A
  per-token perplexity over a different tokenizer is not a quality comparison,
  and it says nothing about SmolLM2 against MiniMind3's 270 on the same file.
  Do not put the two numbers in the same column.
- **Its terminator is `<|im_end|>` (2) and its BOS is `<|im_start|>` (1).**  The
  packer's name lists put `eos` on `<|endoftext|>` (0).  Generation still stops
  because `stop` is `<|im_end|>`, but the metadata is wrong, so pack with
  `--bos 1 --eos 2`.  A Llama-with-ChatML model always needs that line until the
  packer prefers the source's own `eos_token`.

## The container

One: `models/smollm2-135m/smollm2-135m.q8q24.mmcf`, 164 MB, q8 body with a q24
residual (188 tensors).

```bash
# --in takes the HF directory as downloaded; the template is in
# tokenizer_config.json, so no --chat-template is needed.
python3 engine/tools/mmcf_pack.py --in models/smollm2-135m \
  --quant q8 --res q24 --seq 512 --template chatml \
  --name smollm2-135m --chat --bos 1 --eos 2 --verify
```

`--res q24` is not optional: the residual peaks at **26884.3** at layer 11,
which is 82% of the q24 rail and 210x the q16 one.  On q16 the model emits its
end token immediately.

`--bos 1 --eos 2` matches the source's own declaration (`<|im_start|>` /
`<|im_end|>`) and is what the shipped container records (`bos 1 eos 2 stop 2`).
The packer would otherwise derive `eos` as `<|endoftext|>` (0) from its name
lists.

## Running it

The one switch that matters is `RMS_STORE=f32`; without it the model answers with
its end token (Xe) or 24 tokens of raw bytes (Pi).

```bash
# laptop (Iris Xe)
MODEL=$PWD/models/smollm2-135m/smollm2-135m.q8q24.mmcf \
RMS_STORE=f32 GPU_LAYER=1 KV=q16 SEQ=512 ./engine/engine_mm3 --chat \
  --prompt "What is the capital of France?" --gen 24 --greedy
```

On the board the same container and the same switches cost 23 s to the first
token; see the cost table below.

## Why `RMS_STORE=f32`

It was not always stated that way.  The model was first recorded as having *two*
limits — a residual of 25982 at layer 11 and an FFN hidden of 3164 that "still
travels through an I16_2 texture" — and half of that was already stale, while
the other half named the wrong tensor.

**The FFN hidden was never the problem.**  The per-layer `2^-k` store (metadata
key 15) already absorbs it: the packer writes `k=5` for layers 11 and 28, and
the worst *stored* peak is **118.6** against the 128 rail.  The raw peak of
3268.2 never reaches a texture at that magnitude.

**The real limit is the rmsnorm's accumulator.**  It accumulates `sum(x^2)` per
lane into a **24-bit fixed-point** store that pins at 2048.  A lane holds
`dim/32 = 18` values, so at a residual peak of 26884 a lane's sum can reach
`18 * 26884^2 = 1.3e10` — 6.4e6 times the rail.  Once pinned, `inv_rms` is wrong
from layer 11 down, and the model answers with its end token (Xe) or with raw
bytes (Pi).

Nothing about the residual's *range* is involved: q24 holds 26884 at 82% of its
rail.  The same fact appears from three directions — MiniMind3 (~50), pocket
(~2300) and SmolLM2 (26884) all trip the packer's bound — and it is what makes
"the 2048 rail is a ~11 residual" a rule rather than an anecdote.

The packer prints the bound whenever `dim/32 * peak^2` exceeds 2048.

## Isolation, one switch at a time

Xe, `SEQ=512`, greedy, one prompt:

| configuration | answer | PPL |
|---|---|---:|
| container defaults (q24 residual, 24-bit rms) | empty | 4.4e55 |
| `FFN_HID=f32` alone | empty | — |
| `RES=f32` alone | `%` | — |
| **`RMS_STORE=f32` alone** | **`The capital of France is Paris. ...`** | **7.7157** |
| `RMS_STORE=f32 RES=f32 FFN_HID=f32` | correct | 7.6679 |
| `RMS_STORE=f32 RES=bf16 FFN_HID=bf16` | correct | 7.6906 |

So a q24 residual with q16 activation and FFN-hidden streams is within **0.62%**
of an all-f32 container, and the only stream that had to move is the one nobody
suspected.

Cross-check against the source weights: `ref_forward.py` on the same rendered
prompt (float32 numpy) gives argmax 504 `'The'`, top-5 `The / Paris / As / Bon /
I`, which is the first token the engine emits.  Greedy is deterministic across
two runs.

## Cost

**Xe** (`GPU_LAYER=1 KV=q16 RMS_STORE=f32`): **11.3 ms/token** (88 tok/s),
derived from two gen lengths (2.05 s at 32, 4.59 s at 256), **515 draws/token**,
startup+prefill ~1.7 s.

**Pi Zero**, same container, `SEQ=512`, one prompt (`TIMING=1`):

| phase | measured |
|---|---:|
| container read | 5.66 s |
| upload (182 textures; 16.89 s wall, 2.22 s of it texture time) | 16.89 s |
| session ready | 23.04 s |
| prefill, ~40-token prompt | 22.86 / 23.58 / 24.05 s |
| **decode** (median of three) | **630.72 ms/token** |
| draws | 454/token |

1.59 tok/s, against pocket's 0.78 at `SEQ=512`; the cost law (135M vs 361M)
predicts that ratio.  Prefill is ~590 ms/token, i.e. as expensive per token as
decode: a prefill pass is a decode pass over two positions, so a 40-token prompt
is 20 of them, and the attention pass's width grows with `SEQ` (a 720-token PPL
file at `SEQ=512` took 26 minutes).

164 MB in a 320 MB CMA, with 209 MB of `CmaFree` left afterwards — see the CMA
invariant in [`../AGENTS.md`](../AGENTS.md) for why that figure is not an
occupancy.

## Pi Zero does not agree with Xe, and it is open

(2026-11-18, same container on both machines.)  The board answers `17*4` as
**220** against Xe's **68**, twice, deterministically, and `EXP=poly` does not
change it.  So it is arithmetic, and the logits say how much:

| logits vs the float32 numpy reference | mean abs diff | cosine |
|---|---:|---:|
| Xe, same container, same prompt | **1.079** | 0.996120 |
| Pi Zero | **3.490** | 0.981316 |

Both keep argmax 504 `'The'`, but the Pi sits 3.2x further from f32.  PPL agrees
(2026-11-18): **14.2941 on the board against 7.7157 on Xe** (720 tokens,
`SEQ=512`), and on a 236-token slice at `SEQ=256`, 12.4945 against 8.7143.

**The format axes are excluded, and that is the surprising part** (2026-11-18):
on the board, widening the streams makes the model *worse*, the opposite of Xe.

| config (`RMS_STORE=f32` always) | Pi | Xe |
|---|---:|---:|
| q24 residual, q16 act, q16 FFN hid | **12.4945** | **8.7143** |
| `RES=f32 FFN_HID=f32` | 13.4776 | 8.5863 |
| `RES=bf16 FFN_HID=bf16` | 13.2558 | 8.5892 |

Also excluded: the q16 decode (`Q16_DECODE=exact` moves the Pi's logits 3.490 →
3.400) and the exponentials.  What is left is the shared fixed-point path on VC4
itself.

**One measured lead**: `MATVEC_STRIP` — the matvec draws' fragment budget, which
reassociates the dot product — is a **no-op on Xe** (mean |d| = 0.000) and moves
this model's Pi logits by **12.677**, flipping the argmax and dropping the cosine
to 0.021 — while MiniMind3 q8 on the same board with the same switch prints its
pin **byte for byte**.  So the reassociation is not broken (it is exact on a
residual of ~50); it loses the model when the residual is 26884.  That is the
shape of a *precision-at-magnitude* problem, not a wiring bug.

**The next experiment** is a per-layer `HIDDEN_DUMP` diff between the two
machines to find where the 3.5-logit divergence enters.

**Do not cite a Pi number as this model's number.**  The board is usable (630
ms/token, correct answers) but its answer is not Xe's, and the gap is 4x what
MiniMind3 shows (0.36%).

## Known gaps

- **The CPU layer loop degenerates** (`GPU_LAYER=0`) with `ACT_CPU=q16` and with
  `ACT_CPU=f32` alike.  That is the supra2 defect, not the format axes.
- **The Pi/Xe divergence above** — no cause yet; the format axes are excluded.
- **No `--ppl` pin for this model** anywhere.  The house gate does not cover it,
  because its PPL is only comparable against itself and each machine needs its
  own number.
- **The `eos` metadata is wrong** in the container unless `--bos 1 --eos 2` was
  passed at pack time; `stop` masks it during generation.

## Measuring it

```bash
# the isolation that named the limit (Xe or Pi; on the Pi, expect the numbers above)
RMS_STORE=f32 GPU_LAYER=1 KV=q16 SEQ=512 DRAWCOUNT=1 \
  ./engine/engine_mm3 --chat --prompt "What is the capital of France?" \
  --gen 24 --greedy

# the divergence: compare the same container on both machines against f32.
# ref_forward.py takes the HF directory positionally and an optional prompt.
python3 engine/tools/ref_forward.py models/smollm2-135m "What is the capital of France?" \
  --logits-out /tmp/smol.ref.bin
LOGITS_OUT=/tmp/smol.mmcf.bin RMS_STORE=f32 GPU_LAYER=1 KV=q16 \
  ./engine/engine_mm3 --chat --prompt "What is the capital of France?" --gen 1
python3 engine/scripts/cmp_logits.py /tmp/smol.ref.bin /tmp/smol.mmcf.bin

# per-layer divergence between the machines (the open experiment): dump the same
# prompt on each and diff the per-layer rows
HIDDEN_DUMP=1 RMS_STORE=f32 GPU_LAYER=1 KV=q16 \
  ./engine/engine_mm3 --chat --prompt "What is the capital of France?" \
  --gen 1 > /tmp/xe.hidden     # the board writes /tmp/pi.hidden with its own binary
```

The reference dump must be over the *same token ids* the engine consumed, so
render the prompt once and pass `--ids` if the two tokenizers could disagree.


