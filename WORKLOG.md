# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.

| | value |
|---|---:|
| decode (default path) | 229.2 ms/token |
| decode with FFN_FUSE=1 | 208.9 ms/token mean (206.3 / 213.1 / 207.3) |
| CPU | 114.5 ms/token (user 89.5 / sys 25.0) |
| draws | 188/token default, 164 with FFN_FUSE=1 |
| gate | q8/q4 byte-identical, PPL 311.1636 (default path re-verified after the LSB bisect) |

Landed commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN merge plan),
70bb245 (fused FFN kernel), plus the FFN_DUMP bisect hook (FFN_DUMP=1).

## Next step (do this first)

Fuse the **two swiglu passes** (chunk a and chunk b) into one draw.  This is the
one remaining draw merge that does not touch an accumulation chain, and the last
run showed that touching one is what breaks bit-exactness (see
`docs/performance.md`, "Bisecting the fused FFN drift").

How: build a `swiglu_pair_x2(hidden_a, hidden_b)` shader whose viewport is
`aw_a + aw_b` (`608 + 608 = 1216`) fragments.  Fragment `t` picks its source by
`t < aw_a`, reads `A` or `B` at local index `t` / `t - aw_a`, and runs the *same
`actv()` body and the same `x_st2` store as `swiglu_pair`.  Both pair buffers are
1216-wide, so two sampler uniforms and one `if` are enough (GLES2 cannot index
samplers, a branch between two reads is the only way).  Draws go 188 -> 160 per
token, i.e. about -17 ms at the measured ~0.6 ms/draw.

Verify it the hard way: `FFN_DUMP=1` layer-0 `ha`/`hb` must be byte-identical to
the two-pass path over a multi-token prompt, and PPL must be 311.1636.  The dump
is the fast check; do not trust PPL alone, its noise floor is +-2 (ACT8=1 gives
309.7812, ACT8=2 gives 314.8067 on the unchanged path).

If the branchy swiglu also drifts, stop merging draws through shader rewrites and
switch to the CPU side: the remaining 114.5 ms/token of CPU time (user 89.5 /
sys 25.0) is now a third of the total, so look at the sys side (FBO/texture churn
per layer) before another kernel rewrite.

Do not retry "merge the FFN weight textures" (`docs/future-optimization.md`,
superseded) and do not re-chase `FFN_FUSE`: its drift is 1 LSB of q16 on 2 of
~28k values, and the three accumulation orders measured 309.2918 / 312.1366 /
313.5574, so there is nothing to fix - only a shader text to preserve.

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
