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

**3. Never compare a `PERFCNT` wall with a `PERFCNT`-less wall.**  Both a
counter run and a `*_DUP` replay add synchronisation the production path does not
have, so the ms column of either is only good for the `gpu=`/ms ratio.  A/B the
change itself with the counters off.  Three consequences, each measured:

- a `DUP_PROFILE` replay's wall is two injected `glFinish`es plus the pass, so
  `attn stats` prices at 1.30 ms there while replaying it end to end is free
  (191 against 201 ms a token, 2026-11-21);
- the pass whose own `finish` drains the whole token's pipeline cannot be timed
  by wall at all.  The logits pass reads 18.5 ms of `finish` with the counters
  off and 0.04 ms with a counter region in front of it, because the region has
  already retired the pipeline; its real cost is 11.61 ms of QPU time
  (`LOGITS_PERFCNT`), see "The logits pass" below;
- a single reading is not a measurement.  One `attn` run read 40.2 ms a stage
  against 16.2-16.9 in every other run, and that outlier was briefly taken as
  the cost of reading a monitor.

**What a draw's cost is, and what the counters cannot see.**  An engine draw
costs 0.48-0.67 ms against `draw_probe`'s 0.008-0.019 ms pipelined null draw.
`probes/draw_cost_probe` isolates the cause (VC4, 1216 fragments, 8 draws a
chain, same shader and targets):

| arm | ms/draw | frag | scoreboard | varyings | idle |
|---|---:|---:|---:|---:|---:|
| dep + finish | 0.721 | 0.128 | 0.019 | 0.000 | 0.000 |
| dep, no finish | 0.635 | 0.127 | 0.019 | 0.000 | 0.000 |
| indep + finish | 0.677 | 0.127 | 0.019 | 0.000 | 0.000 |
| indep, no finish | 0.131 | 0.021 | 0.006 | 0.000 | 0.000 |

The FBO -> texture dependency is 5x for identical work; a `glFinish` is a
seventh of that; and the remaining ~0.47 ms appears in **no** counter -- `idle`
and `waiting-varyings` both read 0, which is why those two were added.  It is
outside the QPU's cycle accounting, consistent with a bin/render
synchronisation.  So a fusion change is judged by its wall delta, not by a
counter delta.

**The table's shape is a triage, not a verdict.**  Every matvec pass in the
table reads 1.8-2.5 ns/MAC with 62-86% of its `gpu=` in executing-valid
instructions, so the block is arithmetic-bound and the passes are
interchangeable in efficiency -- there is no outlier to attack:

| pass | ns/MAC | exec share |
|---|---:|---:|
| `ffn` | 1.82 | 62% |
| `logits` | 2.36 | 86% |
| `wo` | 2.49 | 80% |
| `w2` | 2.51 | 80% |

What that rules out is the whole family of optimisations aimed at anything but
the per-MAC instruction count: `probes/qkv_rows_probe` shows four rows a
fragment is 15% slower (VC4 has no MRT, so the alternatives are the grid running
both halves or a deeper body that costs 50% more texture quads), and halving the
activation fetches with `ACT=i8` leaves `gpu=` and `exec=` unchanged while the
end-to-end run gets 9% slower.  What remains is the ~0.5 ms a draw round trip
above: 135 draws a token pay it, no counter can see it, and it is judged by a
fusion A/B's wall delta or not at all.

## What limits the decode (VC4, measured against the architecture guide)

The QPU's shape decides this, and `literature/broadcom/VideoCoreIV-AG100-R.pdf`
states it: a **4-way SIMD** machine multiplexed to 16 lanes by running one
instruction for **four clock cycles** on four 4-lane vectors, dual-issue (one
add and one multiply a cycle).  At the board's 300 MHz that is

```
300 MHz / 4 clocks per instruction x 12 QPUs = 0.90 G instructions/s
0.90 G x 16 lanes x 2 ops                     = 28.8 GFLOP/s = 14.4 GMAC/s
```

and the measured decode is **0.377 GMAC/s**, i.e. **2.6% of peak**.  The gap is
not memory, and three separate measurements say so:

| what was measured | result |
|---|---|
| L2C hit rate over a matvec | **99.9%** (270,267 hit / 218 miss; 69 B a transaction, i.e. a 64 B line) |
| halving the weight bytes (q8 -> q4) | **2.9%** faster (190.3 -> 184.8 ms), not the 30% a bandwidth bound would give |
| TMU utilisation | **42%** of the guide's 1 G pixel/s, so the texture units are not saturated |

The kernel is **issue-bound**: 0.90 G instructions/s against roughly 2.2
instructions a MAC gives a 0.41 GMAC/s ceiling, and the measured 0.377 is 92% of
it.  That is why the levers aimed at bytes and draws all measured as nothing --
they do not change the instruction count.

**The one real difference between passes is MACs per decode.**  The FFN gate
feeds four weight fetches from one decoded activation pair (16 MACs an
iteration, 1.82 ns/MAC); a plain projection has two fetches to share the same
decode (8 MACs, 2.5-2.9 ns/MAC).  Widening a projection to four rows needs a
second output texel, i.e. MRT, and VC4 does not have it (below).

### The q16 decode is already minimal

Four `dA`/`dB` calls run per inner iteration and are over half its
instructions, so they were priced against every alternative that stays
bit-exact (`probes/q16dec_probe`, inside the real matvec loop):

| decode | `exec=` ms | vs shipped |
|---|---:|---:|
| **shipped** `u>=32768.0?u-65536.0:u` | **1.99** | -- |
| branchless `u-65536.0*step(32768.0,u)` | 2.15 | +8% |
| uint sign-extend | 2.28 | +15% |
| int sign-extend | 2.30 | +16% |

A float select is nearly free on this ALU and an integer sign-extend is not,
which is the same finding `probes/t3dec_probe` reached for the 2-bit sign
decode.  Do not rewrite it.

## The logits pass

The vocabulary matvec is the one pass whose wall time is worthless.  Greedy
decode reads its result back, so the `glFinish` before the read waits for the
whole token's pipeline, and a wall lap charges the pass for that drain: 18.5 ms,
against 11.61 ms of QPU time (`PERFCNT=1 LOGITS_PERFCNT=1`, MiniMind3 q8,
6400x768, 2026-11-21).

| quantity | value |
|---|---:|
| `gpu=` | 11.61 ms |
| `exec=` | 9.93 ms (86%) |
| `idle=` | 0.15 ms |
| quads | 4,915,200 |
| **ns/MAC** | **2.36** |

2.36 ns/MAC is inside the 1.8-2.5 range every other matvec sits in, so the
logits pass is not slow and there is nothing in it to fix.  The 2.36 ns/MAC and
the 2.36 ns/quad agreeing is a coincidence of the kernel shape, not a
derivation -- the `quads` counter's accounting does not match a per-`texture2D`
model (the FFN gate reports 3x its fetch count, this pass 2x), so quads is a
relative signal only.

Greedy also has a second path, `GPU_ARGMAX=1`, which reduces on the GPU instead
of reading the logits back.  It is a measured loss on this board (193.4 against
190.5 ms a token) and stays opt-in.

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
