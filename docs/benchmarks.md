# Benchmarks

Decode timings, model quality numbers and the protocol that produced them, for
the two machines this engine targets. These are regression baselines for the
machine named, not portable benchmarks.

**This is a reference table, not a log.**  A measurement that supersedes another
replaces its row; the commit that produced it is named in the row or in the
section heading.  The history of how a number moved is Git's job —
`git log --follow docs/benchmarks.md`, and the pre-compression text (120 dated
sections) is at `git show e7a36ec^:docs/performance.md`.  Nothing is appended
here.

The Pi table is the **latest measurement per model**, from whatever binary
produced it; each row names its date.  A row is comparable to itself over time,
not necessarily to the row above it.

## Protocol

| | |
|---|---|
| Pi Zero | cross-built ARMv6/VFPv2, `GPU_LAYER=1 KV=q16`, greedy, `--gen 16`, `SEQ` and prompt per row |
| Iris Xe | in-tree `engine_mm3`, `GPU_LAYER=1 KV=q16`, greedy |
| warm-up (Pi) | `drop_caches`, then one **discarded** run (it faults the mmap'd rows the timed runs touch), then **three timed runs 20 s apart**, median reported |
| PPL | `data/ppl_test.txt`, `SEQ` as stated, `PPL_STRIDE` default `SEQ/2` (warm windows) |

Two rules that came from getting them wrong:

- **Decode is the reproducible number; prefill is not.**  Identical MiniMind3 q4
  runs gave 16.0 and 35.9 s prefill (background load 0.9 against 2.4) while
  decode stayed inside ±1%.  Quote prefill only together with its run.
- **Never compare across machines, `SEQ`, or tokenizers.**  PPL is comparable
  only within one model and one `SEQ`; a byte-level BPE and a 64402-entry
  SentencePiece vocabulary turn the same corpus into different token counts.

## Pi Zero decode

**The latest measurement per model, not one campaign.**  Each row names the date
it was taken and the binary generation behind it, so a row is comparable to
itself over time, not necessarily to the row above it.  Sorting is by speed and
is a rough guide only: rows come from different builds, and the engine has gained
10-20% on several models since the oldest of them.

| model | decode (ms/token) | SEQ | measured | conditions |
|---|---:|---:|---|---|
| TinyChat15M q8 | **115.4** | 256 | 2026-11-13 | plain-completion prompt; the container now also ships a `llama2_chat` template |
| stories15M q8 | **116.2** | 256 | 2026-11-13 | `Once upon a time` |
| MiniMind2 q8 | **123.5** | 512 | 2026-11-13 | chat prompt |
| minimind2 q4 | **153.7** | 512 | 2026-11-13 | chat prompt |
| stories42M q8 | **160.2** | 1024 | 2026-11-13 | `Once upon a time` |
| **MiniMind3 q4** | **185.7** | 768 | **2026-11-16** | flat 8-wide q4 gate, the default since; 226.7 with `Q4_GATE_NESTED=1` |
| MiniMind3 q8 | ~190 | 768 | 2026-11-16 | the same session's comparison; measured 188.6 on 2026-11-07 |
| feng q8 | **192.1** | 512 | 2026-11-13 | chat prompt |
| feng q8r | **192.8** | 512 | 2026-11-13 | chat prompt |
| vertex15 q4 | **196.9** | 512 | 2026-11-13 | stops at EOS after 7 tokens |
| MiniMind2 i16 | **216.0** | 512 | 2026-11-13 | cheap q16 decode; 326.8 with `Q16_DECODE=exact` |
| vertex35 q4 | **241.4** | 512 | 2026-11-13 | stops after 7 tokens |
| MiniMind3-MoE q4 | **322.0** | 768 | 2026-11-13 | 131 MB, top-1 routing |
| vertex100 q4 | **487.0** | 512 | 2026-11-13 | stops after 15 tokens |
| void1 q4 | **554.8** | 1024 | 2026-11-13 | 259-entry byte tokenizer, `TOK_NO_PREFIX=1` |
| LFM2.5-QAD q4 | **566.3** | — | 2026-11-16 | hybrid (conv + attention), `IGNORE_EOS` |
| SmolLM2-135M | **630.7** | 512 | 2026-11-18 | `RMS_STORE=f32` is required; 164 MB in a 320 MB CMA |
| RWKV-7 0.1B q8r | **1138** GPU / **969** CPU | 512 | 2026-11-17 | the GPU loop is draw-bound on VC4 and slightly slower than the CPU one; the 3.3x GPU win is Xe-only |
| pocket 0.3B | **1168** | — | 2026-11-18 | `RMS_STORE=f32`; the default i24 rms store answers `杭州是中国中国中国…` |
| BitCPM-CANN-0.5B t3 | **1323** | — | 2026-11-19 | 2-bit matvec, 167.5 MB file; 5.6x MiniMind3's parameters |

**MiniMind3 q4 is now faster than q8** (185.7 against ~190): the q4 gate is
about half the token's work, so the flat 8-wide kernel moved it past q8 rather
than leaving it 1.18x behind.  Any note that still quotes 304.4 for MiniMind3 q4
predates 2026-11-16.

The 2026-11-13 rows were taken under protocol v1 (a discarded warm-up after
`drop_caches`, then three timed runs 20 s apart, median, `GPU_LAYER=1 KV=q16`,
greedy, `--gen 16`) with a per-model prompt; the `SEQ` column is theirs.  Prefill
from that campaign ranged 3-36 s and is not carried here — it moves with the SD
card and the background load, so it is only meaningful next to its own run.

Not measured on the board since the engine moved on: TinyBrainBot q4, picolm85
q4q24 and void1 q8r (load checks only), and MiniMind3-MoE q8 (203.5 MB:
`FBO incomplete`).  Ask for a fresh row rather than quoting another machine's
number for these.

## Iris Xe decode

The 2026-11-02 Xe matrix is superseded (it predates MMCF and the paired q16
paths) and is no longer carried here; `git show e7a36ec^:docs/performance.md`
has it.  Current rows, each with the conditions that produced it:

| model | container | decode | conditions |
|---|---|---:|---|
| SmolLM2-135M | `smollm2-135m.q8q24.mmcf` | **11.3** ms/token | `RMS_STORE=f32`, `SEQ=512`, 88 tok/s derived from two gen lengths |
| BitCPM-CANN-0.5B q8 | `bitcpm-0.5b.q8.mmcf` | **52.5** ms/token | `RMS_STORE=f32`, `--gen 64`, three runs (46.3 / 53.8 / 52.5) |
| BitCPM-CANN-0.5B t3 | `bitcpm-0.5b.t3.mmcf` | **37.8** ms/token | `RMS_STORE=f32`, 2-bit matvec, prefill 0.75 s |

Draw counts explain much of the difference between the two machines: SmolLM2
is 515 draws/token on Xe against 454 on the board, and VC4 is draw-bound at
about 0.6 ms a draw.

## Per-model numbers that are state

Only the rows whose numbers are *required* to run the model, or that pin a
regression baseline.  Each points at the document that owns the explanation.

| model | must be set / known | PPL | read |
|---|---|---|---|
| MiniMind3 q8 | — | **270.3170** (SEQ=768) | `AGENTS.md` |
| MiniMind3 q4 | — | **326.1483** (SEQ=768; a *format* pin, expect ~1% drift when the q4 path is rearranged) | `AGENTS.md` |
| MiniMind2 i16 | — | 280.2156 | `docs/quantization.md` |
| SmolLM2-135M | `RMS_STORE=f32`; `--res q24` at pack time | **7.7157** Xe / **14.2941** Pi (SEQ=512) | `docs/smollm2-135m.md` |
| BitCPM q8 | `RMS_STORE=f32` | **35.5598** (SEQ=768, 247 tokens; 2026-11-19, reproduced from a fresh pack of the recipe) | `docs/bitcpm.md` |
| BitCPM t3 | `RMS_STORE=f32`; 2-bit matvec path | **38.1541** (SEQ=768, 2026-11-19); the same file expanded to q8 levels (`T3_EXPAND=1`) is **35.6132**, so the 2-bit kernel now costs 0.070 nats/token | `docs/bitcpm.md` |
| RWKV-7 0.1B q8r | CPU loop / `RWKV_GPU=1` are separate pins | **9.2919** CPU / **9.1850** GPU (SEQ=512) | `docs/rwkv7.md` |
| LFM2.5-230M q4 | serial warm path | 29.1588 (28.6196 with `Q16_DECODE=exact`) | `docs/lfm2.5-230m.md` |
| MoE (v3) | top-1, host routing | see the container's own rows | `AGENTS.md` |

**BitCPM's two bodies, same weights, same text** (Iris Xe, `RMS_STORE=f32`):

| | q8 container | t3 container |
|---|---:|---:|
| bytes | 435,368,297 | **167,520,233** |
| RAM to run it | 435 MB of texture | ~85 MB of texture (167.5 MB of file); the earlier host-expansion path wanted 342 MB and the board was OOM-killed at 280 |
| KL against the float32 forward | 0.0288 | 0.0278 |
| `bench_mc.py`, 100 items (2026-11-19) | acc 0.690 (fmt 0.690) | acc 0.710 (fmt 0.700) |
| IFEval, 24 cases x 3 seeds, greedy (2026-11-19) | pass **0.500** | pass 0.500; the failed-check lists coincide cell for cell |
| translation, 30 sentences, greedy (2026-11-19) | chrF **0.495** (0.511 zh→en / 0.479 en→zh) | 0.468 (0.521 / 0.415); t3 leaks the source script less (0.007 against 0.035) |

The t3 body is not a lossy step for this model: it carries a per-row scale where
the q8 path carries one scale a matrix, hence the slightly better KL and
`bench_mc`.  Its one repeatably worse cell is en→zh translation (0.415 against
0.479).  Its win is **memory, not speed** — on VC4 the 2-bit matvec costs
2.45 ns/MAC against q8's 2.97, a 1.2x, because the decode is ALU-bound and the
fetch is only 22% of the time; on Xe the uint decode is *slower* than q8
(0.109 against 0.063 at 1024x1024).  `probes/t3mv_probe` is the measurement.

## Model quality (the eval batteries)

Same-family comparisons only: these are absolute levels for one model, not a
ranking across tokenizers.

| model | `bench_mc` acc | notes |
|---|---:|---|
| BitCPM t3 | 0.710 | 2843 ms an item |
| BitCPM q8 | 0.690 | 4077 ms an item |
| pocket qat | 0.480 | |
| LFM2.5-230M | 0.380 | |
| MiniMind3 q8 | 0.220 | |

IFEval and translation for BitCPM are in the table above; `docs/evaluation.md`
owns the batteries and how to run them.

## Comparability

- **Across machines**: never for decode, strings or PPL.  The VLAB board and
  the laptop round the fused FFN differently, and SmolLM2-135M is the standing
  example of the same container disagreeing by 3.2x on logit distance.
- **Across `SEQ`**: never for PPL.  The window protocol moves with it.
- **Across tokenizers**: never for PPL as a quality statement.  SmolLM2's
  byte-level BPE turns the Chinese corpus into 1.76 bytes/token; BitCPM's
  73448-entry Chinese-first vocabulary pays 16 nats on its first token when the
  space marker is missing.
- **Across containers of the same model**: fine when the weights are the same
  and the body is lossless (BitCPM q8 against t3 above), otherwise state which
  body produced the number.

## Where the history is

The dated campaign sections that used to live here — 120 of them, from the
first Pi matrix of 2026-11-01 to the t3 bring-up of 2026-11-19 — were removed
when this file became a reference table.  They are not lost:

```bash
git log --follow --oneline docs/benchmarks.md     # the file's own history
git show e7a36ec^:docs/performance.md              # the pre-compression text
git log -S 'a phrase from the measurement'        # find the commit that made it
```

Per the documentation rule in `AGENTS.md`, a number that matters is either in a
row here, in the owning document as one dated line with its commit, or in a
commit message.  Nothing is appended to this file.
