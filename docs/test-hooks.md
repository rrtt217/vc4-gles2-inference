# Test hooks

Environment switches used for A/B tests, diagnostics and debugging.  They are
not production settings.

Five rules before using this list:

- **A switch that no longer exists is worse than a missing one.**  `--template`,
  `ROPE_THETA=` and `ROPE_HALF=` were removed with the legacy
  readers (`fcc4095`); the container carries all four.  Passing them now is
  silently ignored.
- **Prefer an in-binary switch to a second build.**  Two builds add enough
  variance to hide a 1 ms effect on the Pi.
- **The registry is the inventory.**  Every switch the engine reads has a row in
  `engine/src/switches.cpp`: canonical name, the spellings it replaced, type,
  the closed set of values, whether it is permanent or a probe, and the *axis*
  it moves.  `./engine_mm3 --dump-switches` prints that table with what this
  environment resolved to (no GPU, no model), and `gate/check_switches.py`
  fails the gate if the source reads a name with no row, if a row is read
  nowhere, if a retired name is read again, or if a row is missing from this
  document.
- **A switch names an axis, not a module.**  `ACT=i16|i8|f32` is the activation
  streams' precision whichever file reads it; `KV=i24|q8|q16|bf16|f32` is the
  KV cache's encoding; `RES=i16|i16_1|i24|f32` is the residual's.  The old
  spellings keep working — the registry rewrites them onto the axis value and
  says so on stderr once: `ACT8=1` -> `ACT=i8` + `FFN_HID=i8`, `ACT8=2` ->
  `FFN_HID=i8`, `ACT_FMT` -> `ACT`, `ACT_F32` -> `ACT_CPU`, `RES_FMT` -> `RES`,
  `RES_PACK=0` -> `RES=i16_1`, `KV_PACK=16` -> `KV=q16`, `KV_FMT=bf16` ->
  `KV=bf16`, `KV_SCALE` -> `KV_GRID`, `ATT_SCALE` -> `ATT_SCORE_GRID`,
  `ATT_SUM_SCALE` -> `ATT_SUM_GRID`.
- **A closed set is validated, not defaulted.**  `ACT=i9` stops the run and
  prints the list; it does not fall back to `i16`.  Validation happens once, for
  every set switch, before any kernel reads one — including switches nothing in
  this model's path reads, so a typo cannot hide until the next model.

## The inventory

The table below is rendered from the registry by
`engine/gate/gen_test_hooks.py --write` and verified by `--check` (step 0 of
the gate), so it cannot drift from the code.  `permanent` is part of the
documented configuration surface, `probe` is a diagnostic or a pricing arm
(candidates for retirement), `retired` is gone.

<!-- BEGIN generated: switches -- `engine/gate/gen_test_hooks.py --write`
     rewrites this block from `engine/src/switches.cpp`.  Do not edit it by
     hand: the registry is the inventory, and `--check` (step 0 of the gate)
     fails when this text and the registry disagree. -->

The rows below are the registry, one per switch, grouped by the axis they move.
`permanent` is part of the documented configuration surface; `probe` is a
diagnostic or a pricing arm; `retired` is gone (`--dump-switches` prints the
same table with what *this* environment resolved to).

138 switches: 28 permanent, 96 probe, 14 retired; 16 legacy spellings are translated.

### The activation streams

| Switch | Takes | What it moves |
|---|---|---|
| `ACT` <br><sub>`ACT_FMT`</sub> | `i16\|i8\|bf16\|f32` | the GPU activation streams' format: i16 q16 grid (default), i8, bf16, or bit-cast f32 |
| `ACT_CPU` <br><sub>`ACT_F32`</sub> | `q16\|f32` | the CPU layer loop's activation grid (ACT_F32 meant f32, whatever its value) |
| `FFN_HID` | `i16\|i8\|bf16\|f32` | the FFN hidden stream's format: q16 (default), i8, bf16, or f32 past the q16 rail |
| `Q16_DECODE` | value | the q16 decode: fast classifies the -32768 rail at 32767.5, exact is integer |
| `EXP` <br><sub>`ATT_EXP`</sub> | `drv\|poly` | every exponential the shaders emit: drv (default) or a portable polynomial; VC4's SFU exp is 20x Xe's error |
| `QKV_PAIR` | `=1` | two qkv rows per fragment (int8 only, default on); QKV_PAIR=0 disables |
| `RMS_STORE` | `i24\|f32` | the rms partial sums: 24-bit fixed point (default, 1/4096 step) or a bit-cast f32 |

### The attention block's fixed-point

| Switch | Takes | What it moves |
|---|---|---|
| `ATT_SCORE_GRID` <br><sub>`ATT_SCALE`</sub> | N | score/(max) grid: round(score*N), so +-32768/N range and 1/N step (1024) |
| `ATT_SUM` | `linear\|log2` | the sum's domain: linear grid (default) or log2(sum)*2048 — a trade, see the doc |
| `ATT_SUM_GRID` <br><sub>`ATT_SUM_SCALE`</sub> | N | the linear sum grid in the same (max,sum) texel (32 = step 1/32) |

### The KV cache

| Switch | Takes | What it moves |
|---|---|---|
| `KV` | `i24\|q8\|q16\|bf16\|f32` | the KV value encoding (i24 fixed 24-bit, q16/q8 packed, bf16, or f32) |
| `KV_GRID` <br><sub>`KV_SCALE`</sub> | N | the fixed-point KV scale (16384 for i24, 2048 for q16, 16 for q8) |
| `KV8_SIM` | `0..6` | price an 8-bit KV grid on the CPU attention path (1..6 = which grids) |
| `KV8_SIM_SCALE` | value | the global grid the KV8_SIM pricing arms use (default 16) |
| `KV8_STATS` | value | the per-head V statistics KV8_SIM needs, on the CPU path |
| `KV_VSTEP` | `=1` | q8 only: a per-head, per-position V grid (buys q4 6% PPL, costs q8 6%) |

### The residual stream

| Switch | Takes | What it moves |
|---|---|---|
| `RES` <br><sub>`RES_FMT`</sub> | `i16\|i16_1\|i24\|bf16\|f32` | residual format: i16 packed pairs (default), i16_1 one a texel, i24, bf16, or f32 |

### The FFN hidden stream and kernel shape

| Switch | Takes | What it moves |
|---|---|---|
| `FFN_FLUSH` | `=1` | flush policy after the FFN (FFN_FLUSH=0 keeps the default) |
| `FFN_FUSE` | `=1` | the fused gate+swiglu (default on); FFN_FUSE=0 restores the two-pass path |
| `FFN_PAIR` | `=1` | two output rows per FFN fragment (default on); FFN_PAIR=0 disables |
| `FFN_SW_INLINE` | value | FFN_SW_INLINE=0 restores the generic X4Q activation decode in the swiglu |
| `FFN_SYNC` | `=1` | sync after the FFN's draws |
| `FORCE_GLFINISH` | value | force the VC4 finish path on other drivers |
| `LAYER_SAFE_FINISH` | value | restore the fully-finished GPU layer path |
| `Q8AP_INLINE` | value | Q8AP_INLINE=0 restores the generic X4Q activation decode |
| `RMS_FUSED` | value | force the fused rmsnorm reduce |
| `RMS_SPLIT` | value | force the split rmsnorm reduce |

### Placement: which path runs

| Switch | Takes | What it moves |
|---|---|---|
| `GPU_ATTN` | `=1` | GPU attention with a CPU FFN |
| `GPU_LAYER` | `=1` | hidden state, attention, RMSNorm and FFN all on the GPU |
| `CONV_SKIP` | value | the LFM2 conv layers become identity copies |
| `CPU_FFN` | value | force the FFN onto the CPU (overrides GPU_LAYER's FFN) |
| `DRAW_CLEAR` | `=1` | clear the target before every draw (a correctness oracle, not a setting) |
| `GPU_FFN` | value | force the FFN onto the GPU |
| `MATVEC_STRIP` | value | override the fragment budget of the matvec draws |
| `NO_SHADER_SELFTEST` | value | skip the startup shader self-test |

### Cost attribution

| Switch | Takes | What it moves |
|---|---|---|
| `ATTN_DUP` | value | replay one attention pass; the end-to-end delta is its marginal cost |
| `DRAWCOUNT` | value | draws per decoded token |
| `LAYER_PROFILE` | `=1` | per-pass marks inside a layer |
| `PAIR_PROFILE` | value | the same for the two-position prefill pair |
| `READBACK_TIME` | value | the per-token logits finish/read-back time |
| `RWKV_CHUNK_TEST` | value | sweep the RWKV channel-mix cut (probe: it moves the numbers) |
| `RWKV_PROFILE` | `=1` | the RWKV paths' per-phase timing |
| `SHADER_TIME` | `=1` | STARTUP phase marks and one line per program |
| `SYNCCOUNT` | value | host <-> GPU synchronisations per decoded token |
| `TIMING` | value | prefill and decode timings to stderr |
| `UPLOAD_TIME` | value | upload wall time and the per-texture breakdown |

### Diagnostics: what the engine dumps

| Switch | Takes | What it moves |
|---|---|---|
| `PPL_STRIDE` | value | window stride for --ppl (0 restores the cold pre-2026-11-12 windows) |
| `ARGMAX_DUMP` | value | with the greedy fast path, print the chunk winners |
| `ATT_DUMP` | value | the score/softmax integers at one position, for check_att.py |
| `CHAT_CACHE` | value | the cache split of every model call (cached / prefill / prompt) |
| `CHAT_DUMP` | `=1` | the rendered prompt on stderr, segment by segment |
| `CONV_DEBUG` | value | the LFM2 conv block's stages for layer 0, next to the CPU reference |
| `CPU_ADD_GPU` | value | the CPU layer loop folds the residual add in the kernel |
| `CPU_RMS_GPU` | value | the CPU layer loop's RMSNorm runs the GPU shader (A/B for the PPL gap) |
| `DUMP_F32` | value | dump rows as raw IEEE bits instead of q16 texels |
| `FFN_DUMP` | value | the first 8 texels of layer 0's gate output |
| `HIDDEN_DUMP` | value | the same for the layer output on the GPU path (GPU_LAYER 0 vs 1) |
| `LOGITS_OUT` | value | float32 logits after prefill, for cmp_logits.py |
| `MATVEC_DUP` | value | replay one weight-matrix pass per layer (wo,ffn,w2 or all); dup_env() |
| `MMCF_DEBUG` | value | per-tensor detail while walking the container table |
| `MODEL_DUMP` | value | the parsed container metadata, right after load |
| `MOE_DUMP` | value | the route of every MoE layer/position |
| `MOE_FIXED` | value | pin every MoE layer to one expert and skip the router (a different model) |
| `PASS_DUP` | value | replay one fixed per-token pass (embed, rms_att, rms_ffn, logits); dup_env() |
| `PPL_DUMP` | value | one `position token nll` line per scored token |
| `QKV_DUMP` | value | hash every qkv matvec row; 2 also prints the first two |
| `RMS_DUMP` | value | prints the 32 rms partial sums as the shader wrote them (RMS_DUMP=<layer> for one layer) |
| `RWKV_DUMP` | `=1` | the RWKV block's per-layer peaks (state/residual/sqrelu rails) |
| `SHADER_SRC` | value | print the GLSL of every program whose key contains the substring |
| `STAGE_DUMP` | value | per-layer xb/att/x1/xbf/hid/hidA as q16 texels; diff_paths.sh reads it |
| `TOKENS_OUT` | value | prompt and generated token ids (int32) |

### The 4-bit weight kernels

| Switch | Takes | What it moves |
|---|---|---|
| `Q4_SHIFT` | `0\|1` | Q4_SHIFT=0 forces the float nibble decode instead of the ESSL 3.00 shift |
| `Q4_BFAST` | value | pricing arm: the fast block form of the q4 gate |
| `Q4_FMT` | `rgba4\|0` | bind an INT4 matrix as GL_RGBA4 (dma-buf); Q4_FMT=0 keeps the packed view |
| `Q4_GATE_NESTED` | value | restore the nested group-loop 4-bit FFN gate (pre-2026-11-16 default) |
| `Q4_HFAST` | value | pricing arm: the fast half form of the q4 gate |
| `Q4_IMPORT_CHECK` | value | after the first imported q4 surface, verify texel (0,0) end to end |
| `Q4_INLINE` | value | inline the q4 group decode into the kernel body |
| `Q4_MV4` | value | the four-output q4 matvec form |
| `Q4_NOMIN` | value | pricing arm: drop the q4 min term |
| `Q4_NOUNPACK` | value | pricing arm: constant scale/min instead of the group plane (wrong weights) |
| `Q4_QKV_FLAT` | value | the flat-loop form of the q4 qkv matvec |
| `Q4_SCOPE` | `all\|attn\|ffn` | which parts Q4_WEIGHTS converts |
| `Q4_SHIFT_NO_MESA` | value | skip the MESA ES 3.00 version override that Q4_SHIFT needs |
| `Q4_SHIFT_VERBOSE` | value | report the q4 decode choice at startup |
| `Q4_TEX_STATS` | value | per texture: dma-buf import or upload, and its logical size |
| `Q4_W2_FLAT` | value | the flat-loop form of the q4 w2/wo matvec-add |
| `Q4_WEIGHTS` | value | quantize to q4 at load (prefer an offline container: q8 -> q4 loses twice) |

### RWKV-7

| Switch | Takes | What it moves |
|---|---|---|
| `RWKV_CM_SPLIT` | n | the RWKV channel-mix cut point (moves the numbers: the add is per chunk) |
| `RWKV_GPU` | `=1` | RWKV's fused GPU layer loop (P6); the CPU serial loop is the reference |
| `RWKV_GROUP` | `0..4` | the serial RWKV path's matvec grouping (P7) |
| `RWKV_GROUP_CHECK` | `=1` | with the grouping on, recompute every grouped matvec and compare |
| `RWKV_PREFILL` | `=1` | RWKV chunked prefill (P1): one draw per weight over a batch of tokens |
| `RWKV_PREFILL_CHECK` | `=1` | compare every batched matvec row against the serial kernel, in the run |
| `RWKV_PREFILL_N` | n | batch size for RWKV_PREFILL |
| `RWKV_PREFILL_STRIP` | value | fragment budget of one batched prefill draw (default kDrawCap) |
| `RWKV_SHIFT_PP` | `=1` | with RWKV_GPU: ping-pong the two token-shift copies (removes 24 draws a token) |
| `RWKV_WKV_CPU` | `=1` | with RWKV_GPU: the elementwise chain on the GPU, the WKV state on the host |
| `RWKV_WKV_ONE` | `=1` | with RWKV_GPU: force the WKV update into one draw |

### Prefill batching

| Switch | Takes | What it moves |
|---|---|---|
| `PREFILL_PAIR` | value | the two-position prefill pair (on by default); PREFILL_PAIR=0 disables it |
| `PAIR_WROW` | value | keep the single-position prefill for a model with per-row scale planes |
| `PREFILL_REPEAT` | value | replay the prefill n times, for timing |

### Loading, memory and the container

| Switch | Takes | What it moves |
|---|---|---|
| `MODEL` | value | the container to run when --model is absent |
| `MODEL_BORROW` | value | MODEL_BORROW=0 copies the CPU weights instead of borrowing the mapping |
| `SEQ` | value | override the container's KV cache length |
| `TOKENIZER` | value | override the container's tokenizer blob; nothing warns, leave it unset |
| `KEEP_CPU_WEIGHTS` | value | keep the CPU weights after the upload (debugging only) |
| `Q8_LAYERS` | value | per-layer weight precision mix at load |
| `Q8_PARTS` | value | per-part weight precision mix at load |
| `UPLOAD_XOR` | value | UPLOAD_XOR=0 restores the byte-at-a-time int8 -> uint8 conversion |

### The tokenizer

| Switch | Takes | What it moves |
|---|---|---|
| `TOK_NO_PREFIX` | `=1` | do not force a leading space/prefix before the first token |
| `TOK_SPACE_FOLD` | `=1` | fold the space that SentencePiece prefixes onto the next piece |

### Sampling and generation

| Switch | Takes | What it moves |
|---|---|---|
| `FREQUENCY_PENALTY` | value | sampler: frequency penalty |
| `IGNORE_EOS` | value | keep generating past the stop token |
| `MIN_P` | value | sampler: min-p cut |
| `NO_REPEAT_NGRAM` | value | sampler: block an n-gram from repeating |
| `PRESENCE_PENALTY` | value | sampler: presence penalty |
| `REPETITION_PENALTY` | value | sampler: repetition penalty |
| `BLOCK_MARKERS` | `=1` | mask the <|...|> marker tokens while generating |
| `GPU_ARGMAX` | value | greedy: reduce the packed q16 logits on the GPU |
| `LOGITS_SPLIT` | value | unpaired logits rows (a kernel-shape arm of the logits matvec) |
| `TOPK_FAST` | value | the fast top-k selection path |

### The disk prefix cache

| Switch | Takes | What it moves |
|---|---|---|
| `PREFIX_CACHE_RESAVE` | value | write the loaded KV rows back through the save path, to diff the two |

### Read from the environment

| Switch | Takes | What it moves |
|---|---|---|
| `TZ` | value | the timezone the tool's date functions render in |

### retired

| Switch | Takes | What it moves |
|---|---|---|
| `ATTN_FUSE_PROB` | value | dead: there is no probability texture to read, att_v fuses the decode |
| `ATTN_HOIST` | value | dead: the hoisted attention path is mandatory, there is no other score kernel |
| `LEGACY_Q8` | value | dead: gone with the legacy q8 reader (the container states its precision) |
| `Q4_4X4_FLAT` | value | dead: the 4-wide flat gate arm; the flat 8-wide body is the default |
| `Q4_4X4_FLAT8` | value | dead: the flat 8-wide arm; it became the default and the arm was removed |
| `Q4_4X4_FLAT8B` | value | dead: the byte-identical candidate; it is the shipped gate now |
| `Q4_4X4_FLAT8G` | value | dead: pricing arm, metadata decoded once per group |
| `Q4_4X4_FLAT8M` | value | dead: pricing arm, metadata applied per iteration |
| `Q4_4X4_NOOUT` | value | dead: stage-removal arm; the driver deletes the kernel, so it measured 0.8 ms |
| `Q4_4X4_NOP` | value | dead: pricing arm, constant scale/min (84 ms of the gate's 159 on the Pi) |
| `Q4_4X4_ONEROW` | value | dead: stage-removal arm, one row of the 4-bit gate (-> 5.71 ms) |
| `Q4_4X4_ONEW` | value | dead: stage-removal arm, one matrix of the 4-bit gate (8.18 -> 6.31 ms) |
| `ROPE_HALF` | value | dead: the container carries the rope configuration |
| `ROPE_THETA` | value | dead: the container carries the rope configuration |

### Legacy spellings

These still work: the registry rewrites them onto the axis value and says so on stderr once.  New code and new notes use the axis name.

| Old | Means |
|---|---|
| `ACT_FMT` | `ACT=<same>` |
| `ACT8=0` | `ACT=i16` |
| `ACT8=1` | `ACT=i8` |
| `ACT_F32` | `ACT_CPU=f32` |
| `ATT_SCALE` | `ATT_SCORE_GRID=<same>` |
| `ATT_SUM_SCALE` | `ATT_SUM_GRID=<same>` |
| `ACT8=0` | `FFN_HID=i16` |
| `ACT8=1`, `ACT8=2` | `FFN_HID=i8` |
| `KV_FMT` | `KV=<same>` |
| `KV_PACK=0` | `KV=i24` |
| `KV_PACK=16` | `KV=q16` |
| `KV_PACK=8` | `KV=q8` |
| `KV_SCALE` | `KV_GRID=<same>` |
| `RES_FMT` | `RES=<same>` |
| `RES_PACK=0` | `RES=i16_1` |

<!-- END generated: switches -->


## Deep dives

The inventory above is generated from the registry; the sections below are the measurements and the decisions behind the rows that have one.  A switch with no deep dive is a switch whose note is the whole story.


### Diagnostics and cost attribution

These write the engine's real numbers out so an offline oracle can check them.
They are the reason `engine/tools/ref_*.py` and `engine/scripts/check_att.py`
exist.

`engine/scripts/dump_stats.py` turns a `STAGE_DUMP`/`HIDDEN_DUMP` file into
`max|x| / mean|x| / clamped-count` per row, which is how a saturating
accumulator is spotted.  `engine/scripts/diff_paths.sh` runs both paths and
compares them with `dump_diff.awk` — read that script's header before drawing a
conclusion from its output.

- `MATVEC_DUP=wo,ffn,w2|all` — the same for the weight-matrix passes, which the
  two probes above leave as one residual (they are ~78% of a q4 token).  `ffn`
  prices the whole split-chunked FFN (gate + swiglu), `w2` both column halves.
  The w2 halves cannot be replayed in place: `w2b` writes the texture `w2a`
  reads as its residual, so a replay after the fact doubles it — each replay
  goes to its own scratch target instead.

- `DRAWCOUNT=1` — print draws per decoded token.

- `SYNCCOUNT=1` — print host↔GPU synchronisations per decoded token: every
  `glFinish` plus every `glReadPixels` on the read-back path (the matvec
  target, the logits target, `read_bytes`).  Both counters are **decode only**
  — the turn records the delta around its decode loop, so the prompt's launches
  do not move a per-token figure — and they are the meters the RWKV work is
  judged by, because that path is launch/sync-bound rather than MAC-bound
  (`docs/rwkv7-plan.zh-CN.md` §九, P6).  One token on the CPU path is ~230
  draws / ~167 syncs; the fused GPU path is ~1150 draws / ~3 syncs (Xe).

- `LAYER_PROFILE=1` — per-pass marks.  The `glFinish` in the lap also retires
  the draw issued before it, so use the replay probes to price a pass; keep
  `LAYER_PROFILE` for finding *which* pass moved.

- `PAIR_PROFILE=1` — the same for the two-position prefill pair.

- `TIMING=1` — prefill and decode to stderr.  The line is printed on every way
  out of a turn, including the interrupted ones: Ctrl-C during prefill, Ctrl-C
  or SIGTERM during decode, and a context that filled up all used to return
  before the print, so the runs worth measuring were the ones with no numbers.
  An interrupted turn appends `cut=prefill|decode|context` after `gen=`, which
  makes a short `gen=` unambiguous.  Covers chat turns, tool turns and the
  plain `--prompt` path.

- `SHADER_TIME=1` — `STARTUP <ms>` phase marks and one line per fragment
  program link (`link <ms>`, source bytes, key, cumulative).  It also prints
  `rmsnorm eps: <v> (container-stated)` when the container carries key 11.

- `UPLOAD_TIME=1` — `UPLOAD <s> wall, <n> textures, texture time <ms>` after
  the weight upload (also implied by `SHADER_TIME=1`).
- `RWKV_CHUNK_TEST=1`：在桌面（GL 上限 16384）强制 RWKV channel-mix 按 2048 行分块，
  复现/验证 VC4 的分块路径（Pi 天然触发）。只影响 split_rwkv 的 max_dim。

- `RWKV_GPU=1`：启用 RWKV-7 GPU 融合层循环（P6，默认 CPU 串行）。

- `RWKV_GPU_DBG=1`：RWKV GPU 路径逐层诊断打印（P6）。

### The decode path's A/B switches

- `QKV_PAIR=0` — single-row qkv matvec; the paired rows are the default
  (200.7 vs 183.5 ms/token).

- `FFN_FUSE=0` — two-pass FFN gate+swiglu; the fused kernel is the default
  (228.7 vs 183.5 ms/token).

- `FFN_PAIR=0` — unpaired FFN rows; the paired q8/q4 path is the default (the
  paired shader covers q8 and q4, and `FFN_FUSE` only applies inside it).

- `LOGITS_SPLIT=1` — unpaired logits rows.

- `GPU_ARGMAX=1` — greedy sampling reduces the packed q16 logits to the
  winning token on the GPU (the proven `top2_fast_packed` chunk pass, then
  two small 16-bound reduce passes; one texel per tile read back) instead of
  reading the whole vocabulary back.  **Opt-in: measured a net loss on VC4**
  (the read-back is ~6 ms a token on the hybrid, the reduction ~14 ms; see
  `docs/performance.md`, 2026-11-15), kept as a byte-identical measurement
  arm.  The reduction compares the exact stored integers and keeps the first
  maximum at every level, which is `std::max_element`'s tie rule, so the
  winner is byte-identical to the read-back.  Only greedy turns without
  penalties, marker masks, `LOGITS_OUT` or logprobs take it; the unpacked
  logits layout (i16 weights, `ACT=i8`) always keeps the read-back.

- `ARGMAX_DUMP=1` — with the greedy fast path, print the chunk winners, the
  per-tile winners and the CPU argmax of the same logits, so a wrong winner
  can be located stage by stage.  Diagnostic only; it reads the vocabulary
  back, so it erases the saving it is debugging.

- `READBACK_TIME=1` — print the per-token logits finish/read-back timing:
  `finish` and `readback` separately on the full-read-back path, and one lap
  per reduction stage on the greedy fast path.  Diagnostic only.

- `RWKV_PROFILE=1` — the RWKV paths' per-phase timing, read-only (no numbers
  move: every hook is a cached switch around a clock read, and the fused path's
  laps are `glFinish` only when the switch is on).  Serial path prints one line
  per layer and a token total:
  `RWKV_PROFILE cpu layer=3 total=12.87 mv_up=0.14 mv_draw=0.14 mv_read=12.42 elem=0.17`,
  where the three `mv_*` terms are wall time around the matvec's GL steps (upload,
  draw, read-back) and `elem` is the layer's remaining host time (the lerps,
  norms, recurrence, sqrelu, the residual adds).  On a `needs_finish()` driver
  the upload finishes inside its own step, so `mv_draw` carries the matvec's GPU
  work there while `mv_read` is the transfer; on the laptop the read-back carries
  both, because the draws are still in flight when it runs.  Fused path
  (`RWKV_GPU=1`) prints one line per layer and a `layers` total per token:
  `RWKV_PROFILE gpu layer=3 total=14.80 ln=1.55 lerp=0.53 matvec=2.34 lowrank=2.67 wkv=1.43 gn=1.29 rk=0.74 gate=1.00 cm=3.24`
  (each share is a `glFinish` lap, so the total is inflated against a real
  token — same caveat as `LAYER_PROFILE`).  A second fused-path line splits the
  `cm` share (the channel mix, 43% of a Pi Zero layer) per chunk of
  `split_rwkv`'s cut:
  `RWKV_PROFILE cm layer=3 total=3.24 pre=0.31 key_a=0.44 sq_a=0.22 key_b=0.44 sq_b=0.22 val_a=0.55 val_b=0.55 add_a=0.05 add_b=0.05`,
  where `pre` is `att_norm_2` + the channel-mix lerp + the shift copy, `key_*`
  and `sq_*` are the key matvec and its relu^2 per chunk, and `val_*`/`add_*`
  are the value matvec and the q16 residual add per chunk (the VC4 two-step
  order; on the fused path the add is inside the matvec, so `add_*` reads 0
  and `val_*` carries it).  The unchunked path (Xe) prints the same four
  stages once as `key/sq/val/add`.  The sub-lap total is folded back into the
  layer line's `cm`, so that field still means the whole cm region.  A head line
  (`RWKV_PROFILE head total=...`) prices the logits pass on both paths.  The
  first token of a process includes the program links and is not steady state.

- `RWKV_WKV_ONE=1` — force the fused path's WKV update to **one** draw on a
  driver that would otherwise split it at `kDrawCap` (VC4: 48 strips a layer,
  576 draws a token = 31% of the board's draws).  Default off, and it exists to
  settle what `kDrawCap` limits: if it is *fragments per draw*, VC4 silently
  drops the 64 x heads*hs draw and the state comes back as the clear colour —
  the reply collapses, which is itself the answer; if it is *width*, the board
  saves 47 of its 48 update draws a layer with the numbers unmoved (Xe takes
  this arm by default, so it is bit-identical there).

- `RWKV_GROUP=0|1|2|3|4` — the serial RWKV path's matvec grouping (P7, A2/A3).
  `1` (the default everywhere except VC4) draws a group of independent matvecs
  into the rows of one target
  and reads the block back once; `0` is the pre-P7 path verbatim (one read-back
  per matrix), which makes it the reference arm for a same-binary A/B; `2` reads
  the block at the target's *full* width instead of the group's own, and `3`
  reads one row per matrix (no sync win) — the two read-back arms of the VC4
  bisect for the +0.0006 nats/token drift the board saw on 2026-11-17; `4`
  groups only runs of equal-row matrices, which is the safe arm if the
  unequal-width draws are what VC4 dislikes.  Same numbers on Xe in all five.
  **The default is `0` on VC4** (`needs_finish()`): the board measured 9.2990
  before P7 and 9.3047 after, so a machine whose read-back behaviour has not
  been bisected keeps the pre-P7 path and loses only the sync win, not the pin.
  Every other driver keeps `1`.

- `RWKV_GROUP_CHECK=1` — with the grouping on, recompute every grouped matvec
  with `rwkv_mv()` in the same process and print the first 12 members that
  differ (`member rows cols ndiff/n maxdiff first`).  It turns "the board drifts
  by 0.0006 nats" into "member 2 of the down group is off by one q16 step from
  index 40", which is what named the read-back rather than the arithmetic.
  Diagnostic only: the extra matvec and read-back erase the saving.

- `RMS_SPLIT=1`, `RMS_FUSED=1` — force the split or fused rmsnorm reduce.  VC4
  defaults to fused (it drops one draw per norm); other drivers default to
  split, where 32 extra fetches per fragment can cost more than the draw.

- `Q8AP_INLINE=0`, `FFN_SW_INLINE=0` — restore the generic `X4Q` activation
  decode in the w2/wo and FFN gate+swiglu kernels.

- `UPLOAD_XOR=0` — restore the byte-at-a-time int8 → uint8 weight conversion.
  The default converts four bytes at a time with the equivalent `^ 0x80`; the
  texture bytes are identical either way.

- `Q16_DECODE=exact` — the q16 activation decode.  The default is the fast
  float reconstruction (`c.x*65280 + c.y*255`) with the negative-rail sign
  classified at `32767.5`: two multiplies and an add per value, which is what
  the kernels cost before `6234328`.  It reconstructs every value with the
  texture sample's small VC4 error, so the board's PPL pins move with it
  (MiniMind3 q8 271.2855, q4 325.3751, LFM2 warm 29.1588; TinyBrainBot q4
  against Xe: cosine 0.9983, mean of five prompts — board, 2026-11-13,
  `docs/performance.md`).  `Q16_DECODE=exact`
  restores `6234328`'s integer decode (`float(int(c*255+0.5))`, bit-identical
  to floor for these positive arguments): MiniMind3 q8 271.6420 and the older
  pinned strings, at the cost of the board's i16 model (326.8 against 216.0
  ms/token).  Both spellings are the same arithmetic up to the reconstruction
  error; see `docs/vc4-glsl-costs.md` for why the fast one is cheaper.

- `RES=i16|i16_1|i24|f32` — the residual stream's format.  `i16` (default) is
  the two-q16-per-texel packed stream (q8 and q4 pack by default; an i16
  container does not, because its add kernels write one q16 a texel); `i16_1`
  is that stream unpacked (was `RES_PACK=0`); `i24` the 24-bit once-a-texel
  form; `f32` the bit-cast one, same memory as `i24`, needs ESSL 3.00.
  `RES` overrides a container's own q24 declaration (which is what
  `RES_FMT=i24_1` used to do, in the other direction).

- `PREFILL_PAIR=0` — disable the two-position prefill pair (the default since
  step 5 of the prefill-batching work).  `diff_paths.sh` sets it so both paths
  step every position through the same code.

- `PAIR_WROW=0` — keep the single-position prefill path for a model with
  per-row q8 scale planes.  The pair kernels have wrow variants and are on by
  default: byte-identical tokens, and warm prefill 7.0-7.2 s against 8.2-8.4 s
  on a Pi Zero with void.1 q8r.  The paired path compiles the P2 program set,
  so its *first* prefill is dearer (15.9 against 14.1 s); that is startup.  The
  switch exists so the A/B can run inside one binary, with `PREFILL_REPEAT`.

- `FFN_FLUSH=0`, `FFN_SYNC=1`, `LAYER_SAFE_FINISH=1`, `FORCE_GLFINISH=1` —
  finish/flush policy.  `LAYER_SAFE_FINISH=1` restores the fully-finished
  `GPU_LAYER` path; `FORCE_GLFINISH=1` forces the VC4 finish path on other
  drivers.

### Precision and the format axes

- `ACT=i16|i8|f32` — the GPU activation streams' format: the uploaded `h`, the
  attention block's output, `xb` and `xbf`.  `i16` (default) is the q16 pair
  grid at `ACT_SCALE=256`; `i8` puts all matvec inputs in `I8_4` (q8 weights
  only — was `ACT8=1`); `f32` bit-casts them into RGBA8, one value a texel (was
  `ACT_FMT=f32`).  The f32 form is the largest single fidelity term measured on
  MNN-Pocket: cosine 0.9415 with the q16 streams against 0.9552 with f32 ones
  (float32 numpy reference, same q4 container).

- `FFN_HID=i16|i8|f32` — the FFN hidden stream's format alone (the legacy
  `ACT8=2` set only this; the legacy `ACT8=1` set this *and* `ACT`).  `i8` needs
  q8 weights; **`f32` exists because the q16 grid rails at ±128** and a model
  whose hidden state exceeds it loses the tail of every FFN output — MNN-Pocket's
  peaks at 1005 (8x the rail, `models/pocket-0.3b/probe_residuals.py`), measured
  2026-11-17.  The f32 form is one value a texel, so it doubles the hidden
  target's width and forces the two-buffer FFN path: the paired gate kernels
  write (gate_r, up_r) as two q16 values in one texel, a layout f32 cannot
  express (`gpu_ffn.cpp` says so at the branch).

- `ACT_CPU=q16|f32` — the CPU layer loop's activation grid, a *placement*
  question next to the two above (was `ACT_F32=1`; that spelling meant "on" for
  any value, including `ACT_F32=0`, and the new one means `f32`).  The grid is
  fine for a residual of order 1 and fatal for one of order 0.01: LFM2.5's
  embedding row peaks at 0.05, so its first norm saw 2-3 significant bits per
  component and came out 8.8% off against the float32 numpy reference.  On the
  q4 container it is worth 1.3 PPL, on i16 it is *worse* than the grid — the
  activation grid is not the only error left in that model.

- `Q8_LAYERS=all|0,1,...`, `Q8_PARTS=qkv,wo,ffn,w2,logits,emb` — precision mix.

- `Q4_WEIGHTS=1` — runtime q4 conversion for experiments.  Prefer the offline
  Q4M1 container; `Q4_SCOPE=all|attn|ffn` only applies to runtime conversion.

- `Q4_SHIFT=0` — force the float nibble decode instead of the ESSL 3.00 shift
  decode (`Q4_SHIFT_NO_MESA`, `Q4_SHIFT_VERBOSE` are VC4-specific overrides).
  `gpu_q4_shift()` reports the choice at startup.

- `Q4_GATE_NESTED=1` — restore the nested group-loop form of the 4-bit FFN
  gate (the pre-2026-11-16 default) for an interleaved A/B.  The default is
  the flat 8-wide gate: byte-identical (probe hash matches at 304/608/1216/2048
  rows on Xe and VC4) and ~43% faster at the engine's 1216-row chunk
  (6.55 -> 3.74 ms/draw).  The nine `Q4_4X4_*` arms that priced that road are
  gone (see Retired below): the flat 8-wide body is the default, so what is left
  of them is the measurement, in `docs/performance.md`.

- `Q4_QKV_FLAT=1`, `Q4_W2_FLAT=1` — opt into the flat-loop form of the qkv
  (`matvec_i4_4x4`) and w2/wo (`matvec_i4_add_4x4`) 4-bit kernels.  **Default
  off: measured a net loss.**  The same rewrite that sped the gate up -43%
  made these single-row kernels slower (Pi Zero 2026-11-16, clean interleaved
  rounds: all-flat 197.5 ms/token vs gate-only-flat 185.0 vs all-nested 227.3
  — the qkv/w2 flat form costs ~12.5 ms/token).  The flat forms are
  byte-identical (same per-group operation order via conditional float
  updates; w2 keeps its generic `X4Q` decode) and stay as the measurement
  arms that produced the negative result.

- `Q4_FMT=rgba4` — bind every INT4 weight matrix (qkv, wo, w13 and, on the
  hybrid, the conv block's `in_proj` slices and `out_proj`) as a four-bit texel surface
  at twice the width: four weights per texel, sampled as n/15 with the 15 folded
  into the group scale, so the kernel decodes nothing.  **Default on.**  The
  surface is allocated here and imported as a T-tiled dma-buf EGLImage
  (`engine/src/gpu_dmabuf.cpp`), because `glTexImage2D` cannot produce a
  two-byte surface on this driver at all -- `st_ChooseTextureFormat` adds
  `PIPE_BIND_RENDER_TARGET` for `GL_RGBA4`, vc4 cannot render to B4G4R4A4, and
  `find_supported_format` then takes DEFAULT_RGBA_FORMATS, so the upload stores
  RGBA8 with the nibbles expanded.  Measured on the board: **51524 kB of V3D BOs
  against packed's 50892** (+1.2%, where the upload path cost 81948, +61%) and
  **224.8 against 292.8 ms** a token.  Where the import is unavailable (no vc4,
  no render node) it falls back to the upload, which is correct but costs the
  2x.  `Q4_FMT=0` keeps the packed view for an A/B.  Still refused on the
  unchunked FFN path (`ffn_split` reads packed nibbles).  Not bit-exact:
  `15*(n/15)` is not exactly `n`, so q4 PPL reads ~326 against the packed 330
  and the greedy strings have not moved on either machine.

- `Q4_IMPORT_CHECK=1` — after the first imported q4 surface, sample texel (0,0)
  through a 1x1 render and print the word handed in against the four components
  that came back.  This is what showed the imported surface reports the word's
  nibbles reversed, i.e. that the import must be given the same `rev` buffer the
  upload path uses.

- `Q4_TEX_STATS=1` — print, per texture, which path it took (dma-buf import or
  upload) and its logical size.  It deliberately reports no `CmaFree` delta: on
  this driver those are page-cache migration, not the texture (the same shape
  read 2020, 2436, 4576 and 4544 kB in four consecutive uploads, and one RGBA8
  upload *raised* CmaFree).  The measurement it used to attempt is now done with
  `/sys/kernel/debug/dri/0/bo_stats`, read around a whole process.

- `KV=i24|q8|q16|bf16|f32` — the KV cache's encoding, one value for one axis.
  `i24` (the default for a dense model) is one 24-bit fixed-point value a texel
  (width `2*kv_dim`); `q16` two int16 a texel (width `kv_dim`) and `q8` four
  int8 (width `kv_dim/2`); a hybrid model (LFM2) defaults to `q16`, because the
  24-bit layout is wrong for kv_dim 512 / head_dim 64 in the fused path.
  `bf16` and `f32` are the 2026-11-17 additions; see the two rows below.
  Measured on MiniMind3 q8 at `SEQ=768` (Xe), one value a run: `i24` 271.6942,
  `q8` 269.6962, `q16` 270.3170, `bf16` 270.3891, `f32` 268.3908 — five
  encodings, one 1.2% band, i.e. the KV encoding is worth about a percent of
  PPL on a q8 container and a great deal more on a model whose KV range does not
  fit the grid (MNN-Pocket).
  **`bf16` implies the packed layout** (two 16-bit halves a texel, width
  `kv_dim`): it is a pair codec, and leaving it on the one-value-a-texel layout
  wrote 24-bit fixed point and read bf16 back — PPL 71840.  That combination was
  reachable before the fold (`KV_FMT=bf16` without `KV_PACK=16`) and is not
  reachable now, which is the argument for one value per axis in one sentence.
  `KV_PACK=0|8|16` and `KV_FMT=bf16|f32` are the spellings this replaces, and
  they are what the server still sends (`KV_PACK=<bits>`), which is why the
  registry translates them.

- `KV_GRID=N` — the fixed-point KV scale (was `KV_SCALE`; default 16384 for
  `i24`, 2048 for `q16`, 16 for `q8`).

- `KV=bf16|f32` — the value encodings this axis gained.  Default is the fixed
  q16/q24 grid on `KVSCALE`.  `bf16` keeps two values a texel (same layout, same
  memory, same fetch count) with a uniform 0.1% relative error, which is the
  right knob for a model whose KV *range* does not fit the fixed grid; `f32`
  uses the one-value-a-texel 24-bit layout with no rounding at all.  Measured on
  MNN-Pocket (q4 + f32 activation streams, against a float32 numpy forward):
  f32 0.9552 cosine, q16 at KV_GRID=512 0.9531, bf16 0.9454 — the fixed grid
  wins on large values and bf16's uniform 0.1% loses to it, so bf16 is not a
  free upgrade; f32 is the only one that removes the rounding, and it buys
  0.002 cosine for 2x the KV memory and 2x the KV fetches.

- `ATT_SCORE_GRID=N` (was `ATT_SCALE`) — the attention score/(max) grid: the score texel holds
  `round(score*N)`, so the range is `+-32768/N` and the step `1/N`.  Default
  1024 (+-32, step 1/1024), which is what every model before 2026-11-17 was
  measured on.  Widening it for a model whose scores exceed the range is a
  trade, not a free win: MNN-Pocket's scores reach 54 under a chat prompt, and
  on both drivers `ATT_SCORE_GRID=256` (``+-128``, step 1/256) made the distribution
  *worse* than the clipping it avoids (Xe entropy 0.07 -> 0.33, Pi 0.16 -> 0.24
  with the f32 streams), because the coarser step costs more than the clip
  saves.  The fix for a score that really overflows is the f32 score stream,
  not a wider fixed grid.

- `ATT_SUM_GRID=N` (was `ATT_SUM_SCALE`) — the exp-sum grid in the same
  (max,sum) texel when the sum is linear (default 32, step 1/32).  Ignored with
  `ATT_SUM=log2`.

- `ATT_SUM=linear|log2` — the sum's *domain*.  `log2` stores
  `log2(sum)*2048` in the same q16 slot and costs one `exp2` in `att_v`; the
  relative step is then 0.03% for any sum from 1 to `SEQ`, where the linear
  grid's 1/32 is 2% of a peaked softmax's sum.  **The accurate one is not the
  better model, and the default is the linear grid** (2026-11-17, MiniMind3 at
  `SEQ=768` on Xe): q8 PPL 270.3170 (linear) against 274.4160 (log2) and q4
  326.1483 against 323.8378 — the two containers disagree in sign, so this is not a quality
  lever, it redistributes a sub-0.1% per-head gain.  `ATT_DUMP` +
  `check_att.py` reads either (the checker decodes `2**(ssum/2048)`).
  With `KV=q16` the packed layout is two int16 per RGBA8 texel (width 384) and
  `KV=q8` four int8 (width 192), instead of the `i24` layout's one 24-bit value
  (width 768) — that is the `KV_PACK=16|8` spelling folded into the axis.

- `KV8_SIM=1..6`, `KV8_SIM_SCALE=N`, `KV8_STATS=1` — price an 8-bit KV grid on
  the CPU attention path (`GPU_LAYER=0`): 1 global K+V, 2 per-head K+V, 3 global
  K, 4 global V, 5 per-head V, 6 per-head K.  Probe only, no GPU kernel.

- `SEQ=N` — override the checkpoint KV cache length; tools auto-bump to 768,
  worker mode auto-bumps to 1024.

### Placement

- `GPU_LAYER=1` — hidden state, attention, RMSNorm and FFN all on the GPU.  The
  fast path; not correct for every model on VC4 (see `docs/performance.md`).

- `GPU_ATTN=1` — GPU RoPE + KV cache + attention, CPU FFN.

- `CPU_FFN=1`, `GPU_FFN=1` — force FFN placement.

### Loading, tokenizer, sampling, prefix cache

- **`TOKENIZER=`** is the one trap in this file: it is *not* retired, and the
  engine loads that blob *instead of* the container's embedded tokenizer
  (`engine.cpp`: `if (model.tok_blob && !getenv("TOKENIZER"))`).  Nothing warns,
  the token ids then mean something else, and for a model whose stop token is
  vocabulary-dependent the answer comes back **empty**.  Leave it unset; if a
  run prints no `tokenizer: ... embedded` line, this variable is why.

- `MODEL_BORROW=0` — copy the weights out of the mapping instead of borrowing
  them.  Borrowing is the default: a 69 MB q8 checkpoint then costs no RSS and
  reloads in tens of milliseconds.

- `KEEP_CPU_WEIGHTS=1` — keep the CPU copies after texture upload (default:
  free them; q8 MiniMind3 RSS drops from ~119 MB to ~56 MB).
- `TOK_NO_PREFIX=1` — do not add it at all.  This is what matches HF's
  `tokenizers.encode()` on a raw string, and it is what a fixture test needs
  when the reference tokenizes the same text.

- `TOK_SPACE_FOLD=1` — add it as part of the text (`" hello"`) instead of
  emitting it as its own token, so the BPE can merge it into `▁hello`.  The
  default emits it separately, which is what the MiniMind2/3 gates were
  baselined on.
- `--seed N` — RNG seed for non-greedy sampling (default 1234).

- `--top-k N` — hard top-k truncation before top-p (0 = off).  The default is
  the exact CPU path: the k kept logits are selected with `nth_element` and
  sorted, the rest are dropped.

- `--top-k-candidates` — with `--top-k`, use the candidate pass (per-tile
  top-2 + logsumexp) instead of the full logits; it falls back to full logits
  when the candidate mass is below `top_p`.  This is the approximate path, not
  the correctness baseline.

- `TOPK_FAST=1` — with `--top-k-candidates`, use the older renormalized
  candidate path; also approximate.

- `IGNORE_EOS=1` — keep generating after the stop tokens, for 64-token math
  tests.
- `RES=i16|i16_1|i24|f32` — see the residual row above.  `f32` stores the
  residual as a bit-cast IEEE float, one value per RGBA8 texel — the 24-bit
  layout, so the same memory — and needs the ESSL 3.00 compiler `init_egl()`
  asks for when this is set.  A residual that exceeds the q16 rail (±128) is
  what a container's q24 flag answers; `f32` is the same answer without the
  fixed-point step.

- `ACT=i16|i8|f32` — the *activation streams'* format: the uploaded `h`, the
  attention block's output, `xb` and `xbf` (`i16` is the default q16 grid at
  `ACT_SCALE=256`, `f32` the same bit-cast one-value-a-texel encoding as
  `RES=f32`, also ESSL 3.00).  Defined in `xformat.cpp` (`x_f32`/`x_f32enc`)
  and threaded through every shader that carries one of those streams, so it is
  a format switch, not a placement one: `ACT_CPU` moves the CPU loop, `FFN_HID`
  changes the width of the FFN hidden stream.  Measured on MNN-Pocket
  (q4 container, f32 residual, against the float32 numpy forward): cosine
  0.9415 with the q16 streams, 0.9552 with the f32 ones — the largest single
  fidelity gain of the 2026-11-17 campaign, and the reason the qkv and attention
  kernels had to learn the format at all.

- `KV_FMT` also has an `f32` value; see the KV row in this section.

- `SHADER_SRC=<substr>` — prints the generated GLSL for program keys containing
  the substring.  The builder branches on the activation and residual formats,
  and reading the generated text is the only way to be sure which branch ran.

- `BLOCK_MARKERS=1` — mask the `<|...|>` marker tokens while generating.  Off
  by default: the engine cannot tell a marker from content (`<think>` is a
  special token for TinyBrainBot, and blocking it removes the scratchpad the
  model was trained to use), and it would not stop the failure it was written
  for — echoing `<|user|>` starts with a plain `<` and the rest is spelled out
  token by token.

## Set by the environment, not by the engine

- `MESA_GLES_VERSION_OVERRIDE=3.0|3.1|3.2` — the only knob that raises the
  shader language on VC4 (it must be set before `eglInitialize`; a client-3
  context does not exist there).  It is also a **ceiling**: pinning it to 3.0
  makes `#version 310 es` fail on Xe too.  `MESA_GLSL_VERSION_OVERRIDE` is
  inert on both boards — measured, see `docs/engine-internals.md` §4c.

## Removed

The generated table above lists every retired switch (the `retired` axis) with
what it did.  Three things about the removals are still worth a sentence:

- `--template`, `ROPE_THETA=` and `ROPE_HALF=` were removed with the legacy
  readers (`fcc4095`): the container carries the chat template, the rope
  configuration, the BOS policy and the special ids.  Passing them is silently
  ignored, which is worse than an error.
- The nine `Q4_4X4_*` arms were deleted on 2026-11-17 (194 lines of builder):
  they were the road to the flat 8-wide gate, which is the default, and each one
  was a second kernel text to keep verified for a conclusion already in the
  code.  `shader_hashes.txt` did not move when they went, which is what makes
  the deletion safe to believe.  What they measured stays in
  `docs/performance.md`: `NOOUT` is not a price (0.8 ms against 8.2, the driver
  deletes the kernel), one matrix 8.18 -> 6.31 ms, one row 5.71, and the group
  plane 84 of the gate's 159 ms on the Pi.
- `--kv-pack` and `--seq` are **server** flags, never engine flags: the server
  turns them into `KV` / `SEQ` for the worker it spawns.
