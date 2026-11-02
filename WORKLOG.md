# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.

| | value |
|---|---:|
| decode (default path) | 229.2 ms/token |
| decode with FFN_FUSE=1 | 208.9 ms/token mean (206.3 / 213.1 / 207.3) |
| CPU | 114.5 ms/token (user 89.5 / sys 25.0) |
| draws | 188/token default, 164 with FFN_FUSE=1 |
| gate | q8/q4 byte-identical, PPL 311.1636 (default path verified this run) |

Landed commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN merge plan).

## Next step (do this first)

Make the fused FFN gate+swiglu kernel bit-exact so `FFN_FUSE` can become the
default: it is already **-20.4 ms/token** but moves PPL from 311.1636 (nll 5.7403)
to 312.1366 (nll 5.7434).  Greedy q8/q4 texts are unchanged, so the drift is small
and localised.

How to bisect it: add a debug env (e.g. `FFN_DUMP=1`) that prints, for layer 0,
the first 16 values of `ha`/`hb` from `ffn_pair_chunk + swiglu_pair` and from
`ffn_pair_swiglu_chunk`, then diff.  Two candidates, in order:

1. **Accumulation loop.** The fused kernel runs four accumulators in one column
   loop; the two-pass version runs two.  `-ffast-math` lets VC4 reassociate, so add
   the four sums as four explicit serial chains (they already are) or split the
   loops so each chain is textually identical to the two-pass one.
2. **The RGBA8 round-trip.** The two-pass path stores the gate pair as q16 through
an 8-bit texture before silu; the fused path uses the float directly.  If that is
the cause, quantise explicitly inside the fused shader (`q = floor(q/256.0)*256.0`
style re-encode) instead of relying on the texture.

Do NOT retry "merge the FFN weight textures" as written in
`docs/future-optimization.md`: w1/w3 are uploaded as `rows`-tall textures and a
merged 2432-row matrix needs a 2432 texture, over the board limit.  The fused
kernel gets the same draw reduction with two weight textures per matrix.

After that, the remaining verified-worthy candidates from the last run are: fold
`att_stats` into `att_v` (blocked by q16 packing), the qkv strip (driver), and
the compute term via `ACT8=2` (-6%).

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
