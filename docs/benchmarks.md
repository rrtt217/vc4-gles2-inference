# Benchmarks

Decode timings, model quality numbers and the protocol that produced them, for
the two machines this engine targets. These are regression baselines for the
machine named, not portable benchmarks.

**This is a reference table, not a log.**  A measurement that supersedes another
replaces its row; the commit that produced it is named in the row or in the
section heading.  The history of how a number moved is Git's job —
`git log --follow docs/benchmarks.md`, and the pre-compression text (120 dated
sections) is at commit `e7a36ec` and earlier.  Nothing is appended here.

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

Campaign of 2026-11-13 on HEAD `71cb1e0`, the current reference.  † marks a row
that predates the warm-up rule and carries the noise it explains; †† a row
measured with the old `floor` q16 decode (the default gives 216.0).

| model | container | SEQ | prompt | decode (median, range) | prefill |
|---|---|---:|---|---:|---:|
| MiniMind2 q8 † | `minimind2-small.q8.mmcf` | 512 | chat, `用一句话介绍你自己。` | **123.5** (123.2-123.7) | 7.3-7.9 s |
| stories15M q8 † | `stories15M.q8.mmcf` | 256 | `Once upon a time` | **116.2** (114.8-117.7) | 3.1-3.9 s |
| TinyChat15M q8 | `TinyChat15M.q8.mmcf` | 256 | `Once upon a time, there was a little girl named` | **115.4** (114.5-161.2) | 3.6-4.5 s |
| minimind2 q4 | `minimind2-small.q4.mmcf` | 512 | chat, `用一句话介绍你自己。` | **153.7** (152.1-154.8) | 8.9 s |
| stories42M q8 | `stories42M.q8.mmcf` | 1024 | `Once upon a time` | **160.2** (158.4-160.4) | 4.3-6.7 s |
| feng q8 † | `feng.q8.mmcf` | 512 | chat, `用一句话介绍你自己。` | **192.1** (189.2-194.3) | 8.2-14.8 s |
| feng q8r † | `feng.q8r.mmcf` | 512 | chat, `用一句话介绍你自己。` | **192.8** (189.3-197.9) | 8.0-16.9 s |
| vertex15 q4 | `vertex15.q4.mmcf` | 512 | chat, `What is the capital of France?` | **196.9** (193.4-211.4) | 8.6-8.7 s |
| vertex35 q4 | `vertex35.q4.mmcf` | 512 | chat, `What is the capital of France?` | **241.4** (224.0-265.1) | 8.9-9.1 s |
| MiniMind3 q4 | `minimind3.q4.mmcf` | 768 | chat, `用一句话介绍你自己。` | **304.4** (303.7-310.0) | 16.0-35.9 s |
| MiniMind3-MoE q4 | `minimind3-moe.q4.mmcf` | 768 | chat, `用一句话介绍你自己。` | **322.0** (319.3-345.3) | 18.2-22.5 s |
| MiniMind2 i16 †† | `minimind2-small.i16.mmcf` | 512 | chat, `用一句话介绍你自己。` | **216.0** (326.8 with the `floor` decode) | 11.8-19.9 s |
| vertex100 q4 | `vertex100.q4.mmcf` | 512 | chat, `Write a haiku about rain.` | **487.0** (487.0-633.3) | 18.6-23.1 s |
| void1 q4 | `void1.q4.mmcf` | 1024 | `Once upon a time,` + `TOK_NO_PREFIX=1` | **554.8** (533.8-578.8) | 14.0 s |

Sample-size notes: `vertex15`/`vertex35` stop at EOS after 7 tokens and
`vertex100` after 15, so one token weighs 7-14% of their mean.  `void1 q4` uses
a 259-entry byte tokenizer, so 16 tokens are about 16 characters.

Not re-measured in this campaign, and why: MiniMind3 q8 (the only HEAD run was a
pilot taken while the store was being scanned), LFM2.5-230M q4 (load check only),
TinyBrainBot q4, picolm85 q4q24 and void1 q8r (load checks only), and
MiniMind3-MoE q8 (203.5 MB: `FBO incomplete`, recorded 2026-11-11).  Ask for a
fresh row rather than quoting an older machine's number for these.

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
| BitCPM q8 | `RMS_STORE=f32` | **36.0302** (SEQ=768) | `docs/bitcpm.md` |
| BitCPM t3 | `RMS_STORE=f32`; 2-bit matvec path | **35.6726** (SEQ=768, `PPL_STRIDE=384`) | `docs/bitcpm.md` |
| RWKV-7 0.1B q8r | CPU loop / `RWKV_GPU=1` are separate pins | **9.2919** CPU / **9.1850** GPU (SEQ=512) | `docs/rwkv7-plan.zh-CN.md` |
| LFM2.5-230M q4 | serial warm path | 29.1588 (28.6196 with `Q16_DECODE=exact`) | `docs/lfm2.5-230m.md` |
| MoE (v3) | top-1, host routing | see the container's own rows | `AGENTS.md` |

**BitCPM's two bodies, same weights, same text** (Iris Xe, `RMS_STORE=f32`):

| | q8 container | t3 container |
|---|---:|---:|
| bytes | 435,368,297 | **167,520,233** |
| RAM to run it | 435 MB of texture | ~85 MB of texture (167.5 MB of file); the earlier host-expansion path wanted 342 MB and the board was OOM-killed at 280 |
| KL against the float32 forward | 0.0288 | 0.0278 |
| `bench_mc.py`, 100 items | acc 0.690 (fmt 0.690) | acc 0.710 (fmt 0.760) |
| IFEval, 24 cases x 3 seeds, greedy | pass **0.500** | 0.417; the failed-check *lists* coincide except `json_keys` |
| translation, 30 sentences, greedy | chrF **0.495** | 0.479; t3 leaks the source language less (0.01 against 0.03) |

The t3 body is not a lossy step for this model: it carries a per-row scale where
the q8 path carries one scale a matrix, hence the slightly better KL and
`bench_mc`.  Its win is **memory, not speed** — on VC4 the 2-bit matvec costs
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
