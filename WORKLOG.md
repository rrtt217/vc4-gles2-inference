# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.

| | value |
|---|---:|
| decode (default path) | 227.1 ms/token median (225.0 min) after paired logits |
| decode with FFN_FUSE=1 | 208.9 ms/token mean (206.3 / 213.1 / 207.3) |
| CPU | 114.2 ms/token (user 90.1 / sys 24.1) |
| draws | 183/token default, 159 with FFN_FUSE=1 (~0.18 ms/draw, measured) |
| gate | q8/q4 byte-identical, PPL 311.1636 (re-verified after the profiling marks) |

Landed commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN merge plan),
70bb245 (fused FFN kernel), 1a7da7c (per-draw cost: 0.18 ms, draw merging is
not the lever), a40593a (paired logits rows: -6.8 ms, bit-exact,
`LOGITS_SPLIT=1` restores the old path), then 2cc1f7a (attention profile:
`PROFATT2` marks inside `att_scores`; the score-viewport shrink was bit-exact but
gained nothing and was reverted).

## Next step (do this first)

**The attention chain is bound by the number of dependent passes, not by
fragments.**  Profiled this run (`docs/performance.md`), decode only, per layer:
qkv 5.22 ms, score grid 3.22, qrot 2.87, stats 2.76, v 2.57, rope 0.88.  The
12-fragment `stats` draw costs as much as the 4608-fragment score grid, and
shrinking the score viewport to `(pos+2)/2` (bit-exact, gates green) changed
nothing - the grid is not where the time is.  Count the passes: ~11 per layer x
8 layers = ~88 dependent drains per token; at the ~1.5-2 ms these laps show that
is most of the 227 ms.

So the next kernel is a **pass merge, not a pair kernel**: fold `qrot` into the
qkv matvec so the chain loses a link - both consumers (`kv_rope_write` and the
score grid) already read `qkv_target_`, so produce the rotated q in the pass that
writes q.  Keep the q16 encode and exactly the rounding `x_i24encq` applies; if
the merged value differs by 1 LSB, stop and put the change behind a default-off
switch instead of shipping it.  Second candidate after that: fold the `stats`
max+sumexp reduction into `att_v` - same shape that drifted under `FFN_FUSE`, so
only with the identical accumulation order.

Do **not** shrink viewports for attention passes (measured, no gain), and do not
open with a fused prob+v.  `LAYER_PROFILE=1` now prints `PROFATT`
(qkv/rope/scores/stats/v) and `PROFATT2` (qrot/score); isolate decode by
differencing a `--gen 12` run against `--gen 4` so prefill cancels, and normalise
by `qkv`, which is position-independent.

## Where the time goes

`LAYER_PROFILE=1` (sync-inflated, use the ratios): attn 36%, ffn 27%, w2 17%,
wo 8%, rms 12%, and logits **~14 ms/token** outside the loop (measured by the
pairing: halving its fragment count returned 6.8 ms, so nearly all of the logits
pass was the activation decode, not weight fetch).  183 draws are ~33 ms of the
227; the remaining ~190 ms is shader work.

## Ruled out — do not retry

* **Draw merging.**  A draw costs ~0.18 ms, measured bit-exactly with
  `RMS_SPLIT=1` (17 extra trivial draws = +3.0 ms mean).  The old 0.5-1.2
  ms/draw model, and the ~94 ms of "draw overhead" it implied, were wrong.
* **`FFN_FUSE` bit-exactness.**  1 LSB of q16 on 2 of ~28k values, caused by the
  four-accumulator shape (accumulation orders 309.2918 / 312.1366 / 313.5574).
  Keep it behind the switch and bank the -20 ms only as an opt-in.
* **Merging the FFN weight textures.**  2432 rows exceeds the 2048 texture
  limit.  The fused kernel reaches the same draw reduction without a repack.

## Rules for every run

1. One build at a time. The loop token (`~/.hermes/scripts/vc4_loop_guard.sh`)
   guarantees no overlap; release it with `rmdir ~/.hermes/locks/vc4-loop.lock.d`
   before you finish, even on failure.
2. Verify with `sh engine/scripts/regression.sh` on the Pi: gates must be
   byte-identical and PPL must match 311.1636.  Never widen a gate to pass.
   Greedy gates do not exercise the candidate path, so also run one
   `TOPK=1 --temp 0.0001` (must equal the greedy text) whenever Top-K changes.
3. Commit every landed step with its measured numbers; append to
   `docs/performance.md`.  If a direction fails, record what was tried and how
   much it measured, then move on.
4. Budget ~20 minutes per run: land one step, verify, commit, update this file,
   release the token.  Leave the tree clean and the repo building.
