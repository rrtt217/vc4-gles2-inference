# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.

| | value |
|---|---:|
| decode (default path) | 229.2 ms/token |
| decode with FFN_FUSE=1 | 208.9 ms/token mean (206.3 / 213.1 / 207.3) |
| CPU | 114.5 ms/token (user 89.5 / sys 25.0) |
| draws | 188/token default, 164 with FFN_FUSE=1 (~0.18 ms/draw, measured) |
| gate | q8/q4 byte-identical, PPL 311.1636 (default path re-verified after the LSB bisect) |

Landed commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN merge plan),
70bb245 (fused FFN kernel), plus the FFN_DUMP bisect hook (FFN_DUMP=1)
and 1a7da7c (per-draw cost measurement: draw merging is not the lever).

## Next step (do this first)

**Pair the logits rows.**  `logits_from` is 5 tiles of 1536x768, one vocabulary
entry per fragment, ~20 ms/token, and every fragment decodes all 768 activations,
so two rows per fragment halves that decode.  Use **two accumulators** (the shape
of the FFN gate pair, which is bit-exact), not four: the four-accumulator shape is
what drifted 1 LSB under `FFN_FUSE`.  Output packs two logits per texel, so
`read_i32` and the Top-K candidate shader both need a packed reader.  Gates:
greedy text byte-identical and PPL 311.1636.

**Do not merge draws any more.**  Measured 2026-11-03: a draw costs ~0.18 ms.
The probe was bit-exact: `RMS_SPLIT=1` re-inserts 17 trivial draws/token with
identical arithmetic and costs +3.0 ms mean (see `docs/performance.md`,
Measured per-draw cost).  So 188 draws are ~34 ms of the 229 and the swiglu-pair
merge this file used to list (8 draws, ~1.5 ms) is cancelled before being
written.  The `FFN_FUSE` A/B of -20.4 ms is ~4 ms of draws and ~16 ms of real
work: one activation decode feeding two rows.

Where the time goes (`LAYER_PROFILE=1`, sync-inflated ratios): attn 36%,
ffn 27%, w2 17%, wo 8%, rms 12%, plus logits ~20 ms/token outside the loop.
After logits, the next probe is the attention chain of five dependent draws plus
finishes (~9 ms/layer beyond the 0.18 ms/draw submission cost).

Do not re-chase `FFN_FUSE` (1 LSB of q16 on 2 of ~28k values; accumulation
orders measured 309.2918 / 312.1366 / 313.5574) and do not retry merging the FFN
weight textures (2432 rows > the 2048 target limit).

## Rules for every run

1. One build at a time. The loop token (`~/.hermes/scripts/vc4_loop_guard.sh`)
   guarantees no overlap; release it with `rmdir ~/.hermes/locks/vc4-loop.lock.d`
   before you finish, even on failure.
2. Verify with `sh engine/scripts/regression.sh` on the Pi: gates must be
   byte-identical and PPL must match 311.1636.  Never widen a gate to pass.
3. Commit every landed step with its measured numbers; append to
   `docs/performance.md`.  If a direction fails, record what was tried and how
   much it measured, then move on.
4. Budget ~20 minutes per run: land one step, verify, commit, update this file,
   release the token.  Leave the tree clean and the repo building.
