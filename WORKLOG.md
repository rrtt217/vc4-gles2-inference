# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.
State at the start of the autonomous loop (2026-11-03):

| | value |
|---|---:|
| decode | 229.2 ms/token (start 245, baseline 237.3) |
| CPU | 114.5 ms/token (user 89.5 / sys 25.0) |
| draws | 188/token (baseline 204) |
| gate | q8/q4 byte-identical, PPL 311.1636 unchanged |

Commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused, 3->2 draws),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN chunk merge plan).

## Next step (do this first)

Merge the chunked FFN weight textures at load time — see
`docs/future-optimization.md`, section "Plan: merge the chunked FFN weight
textures".  w1a/w1b, w3a/w3b, w2a/w2b are one matrix split in two by the
converter; merging them in the loader before the CPU copy is freed is a repack,
not a format change, so values stay identical.  Expected gate 2->1, swiglu 2->1,
w2 2->1: 188 -> 164 draws, about -12 ms.

After that, re-measure and pick the next item from the remaining budget
(rmsnorm 4, qkv 2 driver-strip, rope/scores/stats/v/wo/qrot 6, ~15 outside the
layer loop) or move to the compute term (ACT8=2, -6% measured, behind an env
switch).

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
