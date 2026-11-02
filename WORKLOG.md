# WORKLOG — vc4 decode loop

Goal: Pi Zero MiniMind3 q8 decode **<=150 ms/token**, byte-identical output.

| | value |
| decode (default path) | 226.4 ms/token median (224.8 min), A/B vs engine_base 228.7 |
| decode with FFN_FUSE=1 | 208.9 ms/token mean (206.3 / 213.1 / 207.3) |
| CPU | 114.2 ms/token (user 90.1 / sys 24.1) |
| gate | q8/q4 byte-identical, PPL 311.1636 (re-verified after the qknorm lap) |

Landed commits: a053d35 (G1 GL state caching), e7cc839 (rms reduce fused),
faa833a (results doc), 318ac15 (draw-bound finding), 84b9a88 (FFN merge plan),
70bb245 (fused FFN kernel), 1a7da7c (per-draw cost: 0.18 ms, draw merging is
not the lever), a40593a (paired logits rows: -6.8 ms, bit-exact,
`LOGITS_SPLIT=1` restores the old path), then 66e5f00 (attention profile:
`PROFATT2` marks inside `att_scores`; the score-viewport shrink was bit-exact but
gained nothing and was reverted), 96edebb (qk-norm is ON: q chain is four
passes; the qrot-into-matvec merge is structurally blocked and was reverted;
`PROFATT2 qknorm` lap added).

## Next step (do this first)

**MiniMind3 has qk-norm ON** (dim 768, 8 heads, head_dim 96, kv_dim 384,
`QKNORM=1`; verified with `ATTN_QFUSE_DEBUG=1`).  The previous entry said qk_norm
was off - it is not, and the scale draw runs on every layer.  `LAYER_PROFILE=1`
now prints `PROFATT2 qknorm` too: decode split per layer is scale 0.99 ms, qrot
1.56, score 2.20.  The q chain is qkv matvec -> qk-norm -> qrot -> score, i.e.
**four dependent passes, three drains**, and 8 ms/token goes to a pass whose whole
output is 8 scalars.

**Fold the qk-norm scale into `qrot`** (one draw instead of two draws + a drain).
`rq = 1/sqrt(sum q^2/HEADDIM + eps)` is per head and needed *before* the rotation,
so it cannot move into the matvec or the score pass - but every qrot fragment can
recompute it: store the raw q rows as q16 pairs (two rows per matvec fragment, the
same integers the 24-bit encoding holds, so bit-exact) and let each fragment read
its head 48 texels instead of doing a 96-serial-fetch scan in 8 fragments.
Expected ~8 ms/token if the redundant scan is cheaper than the drain it removes;
keep the summation order of `att_qnorm_scale` exactly (`for z in 0..HEADDIM`,
`dec24q` -> q16 decode of the same integer) and stop if any value moves 1 LSB.

**Do not retry "fold qrot into the qkv matvec"** - implemented this run and
reverted (`docs/performance.md`): with qk-norm on the scale pass needs the raw q
rows, and a fragment writes one texel, so the matvec cannot emit raw q and rotated
q at once.  It was inert by construction: gates green, A/B -1.7% at load 1.71 =
noise.  The kernel is not in the tree.

Do **not** shrink viewports for attention passes (measured, no gain), and do not
open with a fused prob+v.  Isolate decode by differencing a `--gen 12` run against
`--gen 4` so prefill cancels, and normalise by `qkv`, which is position-independent.
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
