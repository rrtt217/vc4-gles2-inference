# Prefill batching

How a prompt's positions are evaluated, and the rule a batching scheme
has to earn its complexity against.  `docs/benchmarks.md` owns the
numbers.

## What a position costs

Measured on the board with a 2-bit ternary body (`docs/bitcpm.md`), one
position, `LAYER_PROFILE` corrected for its one-sync-a-mark tax:

| | ms | draws |
|---|---:|---:|
| arithmetic (433.8 M MAC at the kernel's 1.23 ns/MAC) | 534 | -- |
| the whole position | 1535 | 456 |

The 456 draws are `ffn` 144, `attn` 144, `rms_att` 48, `rms_ffn` 48,
`w2` 48, `wo` 24 -- six, six, two, two, two and one a layer.  A VC4 draw
costs about 1 ms before its fragments do any work
(`docs/engine-internals.md`), so **two thirds of a position is launch
cost, not arithmetic**.

## The rule

**A scheme that shares only the matvecs can reach at most the 534 ms.**
The pair path did exactly that -- `w1/w3`, `w2` and the residual adds
shared between two positions, the 288 attention and rmsnorm draws per
position untouched -- and measured no wall-time win on the board
(2026-11-05, `ec356b4`: serial 6.49 s against pair 6.59 s on a 37-token
prompt).

So a scheme is worth its complexity only if it removes **draws**.  Three
ways, and they compose:

| lever | what it removes | helps |
|---|---|---|
| fuse the elementwise chain into the kernels around it (rmsnorm into the matvec that reads it, the add into the matvec that produces it) | passes per position | prefill and decode |
| N positions a pass | draws per position, but only for the batched phases | prefill |
| chunked or speculative evaluation | positions | out of scope here |

## The constraint that makes N>2 expensive

Attention is sequential in the positions: position b reads the KV rows of
every position before it.  A batched block therefore either

- runs attention per position and batches only the projections -- then
  the attention's 144 draws stay per-position, which is the pair path's
  ceiling; or
- writes the whole prompt's K/V first and runs attention as one batched
  pass over a full QK^T.  That is a second execution mode: its own KV
  staging, its own softmax layout, and its own correctness story.

The first is cheap and buys the smaller half.  The second is what moves
prefill, and it is a project.

## Acceptance

The pair path's rule is the one to keep, because it caught a real defect:
its first wiring rounded the matvec to q16 and added the residual in a
later pass, and **two of 768 q16 values moved by one LSB** -- enough to
change the greedy text.  A batched path is accepted when

- `LOGITS_OUT` is **byte-identical** between the batched and serial arms
  on the same prompt (`cmp -l`, not a cosine, not the generated text),
- `--ppl` matches the serial number exactly, and
- the golden string is unchanged.

A cosine of 0.99 is a failure.  A cross-GPU comparison is not an
acceptance at all (`AGENTS.md`, Measurement rules).

## Measuring before building

`LAYER_PROFILE` gives the per-phase split and `DRAWCOUNT` the launches;
both exist.  A scheme whose target phase is not the largest one is the
mistake this page exists to prevent.
