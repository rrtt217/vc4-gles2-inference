# Prefill batching

How a prompt's positions are evaluated, what batching can and cannot buy on
VC4, and the rule a scheme has to earn its complexity against.
`docs/benchmarks.md` owns the numbers.

## What a position costs

Measured on the board with a 2-bit ternary body (`docs/bitcpm.md`), one
position, `LAYER_PROFILE` corrected for its one-sync-a-mark tax:

| | ms | draws |
|---|---:|---:|
| arithmetic (433.8 M MAC at the kernel's 1.23 ns/MAC) | 534 | -- |
| the whole position | 1535 | 456 |

The 456 draws are `ffn` 144, `attn` 144, `rms_att` 48, `rms_ffn` 48,
`w2` 48, `wo` 24 -- six, six, two, two, two and one a layer.  Most of a
position is not arithmetic.  The per-draw overhead is shape-dependent, not a
universal constant: `probes/draw_probe` prices an empty draw at 0.12-0.19 ms,
but the attention passes at a prefill position carry 1.6-5.5 ms each
(`PROFATT`, t3 body, board: qkv 5.5, rope 2.2, score 2.0, qrot 1.7, stats 1.6,
v 2.3), because their grids are small or mostly causal-masked early-exits while
the draw's fixed cost stays.

## The rule

**A scheme that shares only the matvecs can reach at most the arithmetic
floor.**  The pair path did exactly that -- `w1/w3`, `w2` and the residual adds
shared between two positions, the attention and rmsnorm draws per position
untouched -- and measured no wall-time win on the board (2026-11-05,
`ec356b4`: serial 6.49 s against pair 6.59 s on a 37-token prompt).

So a scheme is worth its complexity only if it removes **draws** or makes the
fragments themselves cheaper.  Three levers, and they compose:

| lever | what it removes | helps |
|---|---|---|
| fuse the elementwise chain into the kernels around it | passes per position | prefill and decode |
| N positions a pass | draws per position, and KV fetches per fragment | prefill |
| chunked or speculative evaluation | positions | out of scope here |

## What N queries a KV fetch is worth on VC4

`probes/att_batch_probe` priced N dot products over one KV fetch with the
output folded away: ns/MAC fell linearly with N (19.03 at one query to 1.16 at
16).  That prices the compute, not a kernel -- a fragment writes ONE texel,
two q16 scores, so N queries against a KV pair (2N scores) cannot all come out
of one fragment.  `probes/att_out_probe` priced the dispatchable shapes at the
engine's grid (`seq/2 x heads`, 16 heads, seq 512, board):

| arm | ms for a full 8-query score set | ns/MAC |
|---|---:|---:|
| `q1` -- the shipped decode shape, 8 draws | 66.1 | 15.8 |
| `qN_serial` -- one draw, a fragment computes only its own query | 22.4 | 5.3 |
| `qN_recomp` -- one draw, a fragment computes all 8 queries, writes its own row | 56.9 | 13.6 |
| `qN_mrt` -- `glDrawBuffers` | unsupported: `GL_MAX_DRAW_BUFFERS` is 1 on VC4 | -- |

Two conclusions.  The chain-sharing form does not survive writing the output:
the redundant MACs cost more than the extra chains save.  But **fusing the N
per-position draws into one draw is a 2.9x win with identical per-fragment
arithmetic** -- the row layout is head-major (`row = h*N + b`), so adjacent
rows of a head read the same K slice and the 2x2 fragment quads coalesce their
KV fetches.  `q1` stays at ~16 ns/MAC even at 4096 fragments, so this is not a
small-grid occupancy artifact.

## The shipped design

`PREFILL_BATCH=n` (0 = off, the default; 2..16) runs the prefill in groups of
n positions, three phases a layer:

1. **stage**, per position: rmsnorm, the qkv draw, the KV write, the rotated Q
   rendered into row b of an n-row staging texture -- the serial draws, in the
   serial order.  The residual is untouched here, so the order across
   positions is free.
2. **score**, once per group: one draw, grid `seq/2 x heads*n`, row
   `h*n + b`, causal up to `pos0 + b`.
3. **finish**, per position: stats and v read the batched score rows through
   `row = h*n + qrow0`; wo and the FFN are the serial kernels, the residual
   ping-ponging per-position targets exactly as the serial path ping-pongs
   `hidden_a_`/`hidden_b_`.

Every kernel is the serial text with the row substituted (the
`matvec_q8_batch` trick), which is what makes the logits byte-identical; the
serial shader text is emitted verbatim when `qbatch == 1`, so the digests do
not move.  No weight kernel changes, so the path is **format agnostic**: t3,
q8, q4, fused-W13, every KV axis and the hid_shift FFN scale all take it.  It
refuses what has no dense attention block to fuse (RWKV, hybrid conv) or
routes per position (MoE).  `--ppl` groups each window's warm-up run, so the
perplexity gate exercises the path.

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

On Iris Xe (2026-11-20) the path is byte-identical to the serial loop at batch
2/3/4/5/8/16 on the BitCPM t3 and q8 containers and on MiniMind3 q8
(`PREFILL_PAIR=0` both arms), and the SEQ=256 PPL is equal to the digit
(35.6726, both arms).  The acceptance also caught a latent defect in the
serial t3 FFN -- the pair chunk's first call reallocated its two-row target
between the two row draws, dropping the gate row; fixed in `gpu_ffn.cpp`
(`ffn_t3_pair_chunk`), which moved the serial t3 PPL 41.3734 -> 35.6726.

## Measuring before building

`LAYER_PROFILE` gives the per-phase split and `DRAWCOUNT` the launches;
both exist.  A scheme whose target phase is not the largest one is the
mistake this page exists to prevent.