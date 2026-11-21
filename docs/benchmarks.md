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
| MiniMind3-MoE q4 | **222.8** | 768 | 2026-11-21 | 131 MB, top-1 routing; median of three (222.8 / 216.9 / 227.2), was 322.0 on the 2026-11-13 build |
| vertex100 q4 | **487.0** | 512 | 2026-11-13 | stops after 15 tokens |
| void1 q4 | **554.8** | 1024 | 2026-11-13 | 259-entry byte tokenizer, `TOK_NO_PREFIX=1` |
| LFM2.5-QAD q4 | **566.3** | — | 2026-11-16 | hybrid (conv + attention), `IGNORE_EOS` |
| SmolLM2-135M | **630.7** | 512 | 2026-11-18 | `RMS_STORE=f32` is required; 164 MB in a 320 MB CMA |
| RWKV-7 0.1B q8r | **1138** GPU / **969** CPU | 512 | 2026-11-17 | the GPU loop is draw-bound on VC4 and slightly slower than the CPU one; the 3.3x GPU win is Xe-only |
| pocket 0.3B | **1168** | — | 2026-11-18 | `RMS_STORE=f32`; the default i24 rms store answers `杭州是中国中国中国…` |
| BitCPM-CANN-0.5B t3 | **1288** | — | 2026-11-20 | 2-bit matvec, 167.5 MB file; signbit weight decode (`6fee221`), median of 3, was 1323 |

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

## What a draw's time is made of (VC4 counters)

A probe's ms/draw is QPU work plus the submit/finish around it, and the second
term dominates the cheap draws.  The board has no timer query, but it does
expose `GL_AMD_performance_monitor` with 30 V3D counters, so the split is
measurable rather than arguable: `probes/perfcnt.h` (2026-11-21) is the shared
wrapper and `PERFCNT=1` turns it on for `draw_probe`, `matvec_probe`,
`t3mv_probe` and `rmsfuse_probe`.  Measured on the Pi Zero (Mesa 26.2.2, QPU
clock 300 MHz, counters summed over 12 QPU-equivalents).

**A null draw is almost all submission.**  `draw_probe` at 1216x1 fragments,
300 draws an arm:

| arm | wall ms/draw | QPU frag ms/draw | QPU idle ms/draw | GPU share |
|---|---:|---:|---:|---:|
| NULL 1x1 | 0.4194 | 0.0000 | 0.0066 | 0% |
| NULL 384x1 | 0.4142 | 0.0013 | 0.0134 | 0.3% |
| NULL 1216x1 | 0.4956 | 0.0042 | 0.0321 | 0.8% |

The 0.42 ms a draw that `draw_probe`'s header calls the per-draw overhead is
therefore ~0.4 ms of submit+finish and 4 us of fragment shading; that is why
"fewer draws" was never a lever, and it is the same conclusion the probe reached
from wall time alone, now with the driver's own numbers behind it.

**A matvec is 94-97% QPU**, 2432x768, 3 tiled draws a matvec, 10 matvecs:

| arm | wall ms/draw | QPU frag ms/draw | GPU share |
|---|---:|---:|---:|
| q8 act1/texel | 3.774 | 3.614 | 96% |
| q8 act2/texel | 3.435 | 3.339 | 97% |
| q8 fast decode | 2.102 | 1.968 | 94% |
| q8 paired rows | 2.737 | 2.641 | 97% |

The fast-decode arm moves 15.2M of 30.0M fragment cycles, so its win is the
decode, not the fetch.  The counters also give the waiting split the ns/MAC
numbers could not: `waiting-TMU` is 0.4% of fragment cycles on the act1/texel
arm and 7.7% on fast decode, which is a kernel that has traded a decode for a
stall.

**t3mv's conclusion is a ratio the counters name directly**: the 2-bit arms are
98-100% QPU, and `t3u_fetch` (no unpack) is 0.6746 ms against `t3u`'s 2.5184, so
the decode is ~73% of the arm's GPU time.  `t3t` (the engine's tiled text) is
100% GPU at 10.097 ms/draw -- there is no submission slack left to win there.

**The rmsnorm arm confirms its own header.**  `rmsfuse_probe --dim 1024`: the
1-fragment reduce is 13% GPU (0.0193 wall against 0.0025 QPU), `norm1` is 48%
and `norm32` is 87%.  A pass whose wall is mostly not fragment work is exactly
the "the overhead is the FINISH, not the normalize" claim, and it is the reason
folding normalize into the consumer matvec saved ~0.4%.

Two limits belong with these numbers: the counters are whole-GPU totals (a
region must not overlap another client's draws), and `PERFCNT` is off by
default, so an existing probe run reproduces its old output byte for byte.

## Finding a stage worth optimising

The counters exist to answer "which pass should the next hour go into", and the
answer is not the pass with the biggest ms.  `LAYER_PROFILE=1 PERFCNT=1` prints
a mark per stage with four numbers; the procedure is three steps and only the
first one uses the wall time.

**1. Sort by GPU milliseconds, not wall.**  A mark's wall includes the submit and
the finish the mark itself performs, and on VC4 those are per-draw constants.  The
`gpu=` column is fragment-shading cycles over `12 * 300 MHz`, so it is the part an
optimisation of the stage can reach.  MiniMind3 q8, 8 layers of 768, decode, the
same layer three ways:

| stage | wall ms | gpu ms | exec ms | ms/draw | quads | exec/gpu |
|---|---:|---:|---:|---:|---:|---:|
| `rms_att` | 2.79 | 0.172 | 0.067 | 0.086 | 28,976 | 39% |
| `attn` | 16.92 | 4.361 | 3.229 | 0.623 | 1,367,712 | 74% |
| `wo` | 2.46 | 1.501 | 1.206 | 1.501 | 602,896 | 80% |
| `rms_ffn` | 2.72 | 0.176 | 0.067 | 0.088 | 28,976 | 38% |
| `ffn` | 9.08 | 6.715 | 4.207 | 3.357 | 2,801,664 | 63% |
| `w2` | 6.50 | 4.779 | 3.815 | 2.390 | 1,908,256 | 80% |

By wall the order is `attn` (16.9) then `ffn` (9.1) then `w2` (6.5); by GPU time
it is `ffn` (6.7) then `w2` (4.8) then `attn` (4.4).  They disagree because
`attn` spends 12.6 of its 16.9 ms outside the QPU -- 7 draws a stage at ~1.8 ms
of launch each -- while `ffn`'s 9.1 ms contains 6.7 ms of shading.  The block is
17.7 ms of GPU against 40.5 ms of wall: **more than half of a decode token is
not arithmetic**, which is the single most useful thing these counters have said.

**2. Read `exec/gpu` to pick the lever.**  A stage near 80% executing-valid is
doing arithmetic: make the arithmetic cheaper (`t3dec_probe`'s signbit decode,
`matvec_probe`'s fast decode) or give it fewer instructions.  A stage near 40%
(`rms_att`) is not ALU-bound at all -- 61% of its fragment cycles are something
else, and it is only 0.17 ms, so leave it alone.  A stage whose `gpu=` is a small
fraction of its ms has a dispatch problem, not a compute problem, and belongs to
the fusion work, not to the kernel work.

**3. Never compare a `PERFCNT` wall with a `PERFCNT`-less wall.**  Reading one
monitor costs about 25 ms a stage on this board (`attn` went 16.9 -> 40.2 ms), so
the ms column of a counter run is only good for the `gpu=`/ms ratio.  A/B the
change itself with the counters off.

**The table's shape is a triage, not a verdict.**  `ffn`+`w2` are 11.5 ms of GPU
and 65% of the block; they are the matvec kernels, and their `exec/gpu` of 63-80%
says a cheaper decode there is worth having -- which is what `matvec_probe`'s fast
decode (2.08 ms against 3.82 ms, 94-97% GPU) already measured.  `attn`'s 4.4 ms of
GPU across 7 draws is a fusion problem that no kernel rewrite can reach, and the
12.6 ms it spends launching is the number `attn_out_probe` and the qN batching
work exist to attack.

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
| BitCPM t3 | `RMS_STORE=f32`; 2-bit matvec path | **35.6726** (SEQ=768, 247 tokens, 2026-11-20 after the gate-row fix `4c5b189`); the same file expanded to q8 levels (`T3_EXPAND=1`) is **35.6132**, so the 2-bit kernel costs 0.0015 nats/token (0.17%) | `docs/bitcpm.md` |
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
