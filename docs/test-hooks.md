# Test hooks

Environment switches used for A/B tests, diagnostics and debugging.  They are
not production settings.

Three rules before using this list:

- **A switch that no longer exists is worse than a missing one.**  `--template`,
  `ROPE_THETA=` and `ROPE_HALF=` were removed with the legacy
  readers (`fcc4095`); the container carries all four.  Passing them now is
  silently ignored.
- **Prefer an in-binary switch to a second build.**  Two builds add enough
  variance to hide a 1 ms effect on the Pi.
- **The source is the ground truth.**  The full inventory, including ad-hoc
  debug flags that are not listed here, is:

  ```bash
  # both spellings: a direct getenv() and one of the env_* helpers in common.h
  grep -rhoE '(getenv|env_set|env_flag|env_int|env_double)\(\s*"[A-Z_0-9]+"' engine/src \
    | sed -E 's/.*"([A-Z_0-9]+)"/\1/' | sort -u    # 93 today
  ```

## Diagnostics: dumps

These write the engine's real numbers out so an offline oracle can check them.
They are the reason `engine/tools/ref_*.py` and `engine/scripts/check_att.py`
exist.

| Switch | What it dumps |
|---|---|
| `STAGE_DUMP=1` | per-layer `xb` (rms_att out), `att`, `x1` (after the `wo` residual), `xbf` (rms_ffn out), `hid` (FFN output), `hidA` (layer out) as hex q16 texels, one line per row: `DUMP <name> w=<texels> h=<hash> <texels>` |
| `HIDDEN_DUMP=1` | the same for the layer output on the GPU path, so `GPU_LAYER=0` and `GPU_LAYER=1` runs are diffable |
| `CHAT_DUMP=1` | the rendered prompt on stderr, one line per segment: `PROMPT[in] / PROMPT[tmpl] <bytes> B  <text>`, with newlines escaped.  `[in]` segments came from a message and are the ones the tokenizer must not turn into special tokens; `[tmpl]` is template text.  Before it escaped, a segment's own newlines split it across lines and the tags looked scattered - it read as if the switch did nothing |
| `CHAT_CACHE=1` | the cache split of every model call on stderr: `CHAT_CACHE turn cached=<n> prefill=<n> prompt=<n>` in the chat REPL, `CHAT_CACHE round=<r> ...` in the tool loop (one line per round).  `cached` is the longest common token prefix with the previous call (or the loaded disk prefix on the first), so `prefill` is what the GPU actually ran |
| `QKV_DUMP=1` | hashes every layer/position of the `qkv` matvec output; `2` also prints the first two rows.  Layout-agnostic, so `QKV_PAIR=0` and `QKV_PAIR=1` runs compare exactly |
| `ATT_DUMP=<pos>` | the exact integers the score/softmax path consumed and produced, for every layer at one position.  Feed the log to `engine/scripts/check_att.py` |
| `FFN_DUMP=1` | the first 8 texels of the layer-0 gate output, for bisecting a 1-LSB drift |
| `CPU_RMS_GPU=1` | the CPU layer loop computes its RMSNorm with the GPU shader (fixed-point partial sums) instead of the f32 CPU mean: the A/B probe for the fused path's PPL gap |
| `CPU_ADD_GPU=1` | the CPU layer loop folds the `wo`/`w2` residual add in the kernel (`matvec_add`) instead of a read-back plus `add_q16`: the other half of that A/B |
| `ACT_F32=1` | the CPU layer loop stores activations in f32 instead of on the q16 grid (1/256).  The grid is fine for a residual of order 1 and fatal for one of order 0.01: LFM2.5's embedding row peaks at 0.05, so its first norm saw 2-3 significant bits per component and came out 8.8% off (measured against the float32 numpy reference).  Off by default; on the q4 container it is worth 1.3 PPL, on i16 it is *worse* than the grid -- i.e. the activation grid is not the only error left in this model |
| `PPL_DUMP=path` | `--ppl` writes one `position token nll` line per scored token.  A mean is not enough to see what a gap is made of: a few catastrophic positions and a uniform 2% error raise PPL the same way |
| `PPL_STRIDE=n` | window stride for `--ppl` on a file longer than `SEQ`: each window feeds `SEQ` tokens and scores only the last `n`, so every scored token has at least `SEQ-n` tokens of context (default `SEQ/2`, printed as `stride=`).  `0` restores the pre-2026-11-12 cold windows -- stride `SEQ-1`, no warm-up, ~1.8 nats/token worse -- which is the only way to reproduce a long-file PPL measured before that date |
| `DUMP_F32=1` | `STAGE_DUMP`/`HIDDEN_DUMP` rows print as raw IEEE bits instead of q16 texels.  The q16 form is right for comparing two engine paths, but its 1/256 step is coarser than the early-layer differences in LFM2.5 (activations of order 0.01), so an engine-vs-numpy bisect needs this |
| `PROMPT_IDS` (with `CHAT_DUMP=1`) | the ids the model will actually see, as `PROMPT_IDS <n>: <id> ...`.  A template that renders to the same bytes can still tokenise differently, and the ids are what the model reads |
| `CONV_SKIP=1` | LFM2 hybrid (`GPU_LAYER=1`): the conv layers become identity copies, to separate a conv-block bug from the rest of the GPU path |
| `CONV_DEBUG=1` | prints the first values of the LFM2 conv block's stages (`xb`, `in_proj`, gates, conv output) for layer 0, next to the CPU reference; on the fused path (`GPU_LAYER=1`) it also prints the GPU `gated`/`cv`/`y` stages and the residual the `matvec_add` wrote, so the two paths can be compared without a dump file |
| `LOGITS_OUT=path` | float32 logits after prefill.  `engine/scripts/cmp_logits.py` compares two of them per tile |
| `TOKENS_OUT=path` | prompt + generated token ids (int32).  A tool turn rewrites it after each model round, so a watcher sees every round |
| `MODEL_DUMP=1` | the parsed model metadata, right after load |
| `MMCF_DEBUG=1` | per-tensor detail while walking the container table |
| `MOE_DUMP=1` | the route of every MoE layer/position: `MOE l=<layer> route=<expert> gate=<logits>`.  The gate values are the q16 read-back, which is what the argmax saw — compare them with `ref_forward.py --routes` rather than judging a route by the text |
| `MOE_FIXED=<e>` | pin every MoE layer to expert `e` and skip the router matvec and its read-back.  Prices the routing: on Xe q8 29.3 ms/token against 10.5 pinned, i.e. the eight per-token syncs are most of the difference.  Output is a different model, so it is a measurement switch only |
| `RWKV_DUMP=1` | the RWKV-7 block's per-layer peaks on stderr, one line per layer per token: `RWKV l=<layer> state_max=<|S|max> resid_max=<|x|max> sqrelu_max=<h²max> w_max=<max w> a_max=<|a|max> [g_max=<|g|max>]`, plus `RWKV final_ln_max=<peak>` once per step before the key-17 `*2^-k` store.  These are the rails the format decisions watch (S-1 state, R-1 residual, Q-1 sqrelu, key 17 final LN); compare them with `engine/tools/ref_rwkv.py --measure`'s per-layer table, which computes the same numbers in f32.  P2 parked them under `STAGE_DUMP`; since P4 they have their own switch and `STAGE_DUMP` no longer prints them |
| `RWKV_GPU=1` | RWKV-7's fused GPU layer loop (P6): the elementwise ops (LayerNorm, six-way lerp, per-head L2/group norm, key delta, r_k bonus, squared-ReLU, the decay/ICLR/gate low-rank tanh/sigmoid/exp chains) and the WKV7 recurrence run on textures; the matvec family is reused and only the final residual crosses back once per token.  Default off — the CPU serial loop (`forward_rwkv`) stays the reference.  The WKV state is bit-cast f32 in RGBA8 ping-pong textures (native ESSL 3.00, exact recurrence); the norm stats are f32 too.  Requests the MESA ES3 override like `Q4_SHIFT`.  On Xe the g1d 0.1B q8r chat container decodes ~36 ms/token against ~119 CPU (3.3x; bench protocol: warmup discarded + 3-run median).  Acceptance is PPL, not cross-path cosine or greedy (AGENTS.md measurement rules): GPU PPL 9.1850 at SEQ=512 against the CPU pin 9.2919 — same magnitude, slightly lower; the two are separate pins and are not comparable.  The greedy text differs from the CPU pin (chaotic temp-0 sensitivity of a 0.1B model, not a quality defect); the GPU path is deterministic (two runs byte-identical).  See docs/performance.md (2026-11-16, P6) |
| `RWKV_WKV_CPU=1` | with `RWKV_GPU=1`, keep the elementwise chain on the GPU but run the WKV7 recurrence on the host f32 state (the hybrid fallback).  Also forced automatically when the driver offers no ESSL 3.00, since the bit-cast state kernels need it.  An A/B for the recurrence's contribution; reads the six WKV inputs back once per layer, so it is slower than the full GPU path |

`engine/scripts/dump_stats.py` turns a `STAGE_DUMP`/`HIDDEN_DUMP` file into
`max|x| / mean|x| / clamped-count` per row, which is how a saturating
accumulator is spotted.  `engine/scripts/diff_paths.sh` runs both paths and
compares them with `dump_diff.awk` — read that script's header before drawing a
conclusion from its output.

## Probes: cost attribution

All default off, and the replay probes are bit-exact, so they can be enabled
without moving the gate:

- `ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v|all` — replay one attention pass
  per layer; the end-to-end delta is that pass's marginal cost.
- `PASS_DUP=embed,rms_att,rms_ffn,rms_final,logits,logits_draw|all` — the same
  for the fixed per-token passes (`logits_draw` also skips the read-back).
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

## A/B switches for the landed decode work

Each defaults to the fast path and exists so a change can be A/B-ed inside one
binary.  They are measurement hooks, not production settings.

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
  logits layout (i16 weights, `ACT8`) always keeps the read-back.
- `ARGMAX_DUMP=1` — with the greedy fast path, print the chunk winners, the
  per-tile winners and the CPU argmax of the same logits, so a wrong winner
  can be located stage by stage.  Diagnostic only; it reads the vocabulary
  back, so it erases the saving it is debugging.
- `READBACK_TIME=1` — print the per-token logits finish/read-back timing:
  `finish` and `readback` separately on the full-read-back path, and one lap
  per reduction stage on the greedy fast path.  Diagnostic only.
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
  against Xe: cosine 0.9983, mean of five prompts).  `Q16_DECODE=exact`
  restores `6234328`'s integer decode (`float(int(c*255+0.5))`, bit-identical
  to floor for these positive arguments): MiniMind3 q8 271.6420 and the older
  pinned strings, at the cost of the board's i16 model (326.8 against 216.0
  ms/token).  Both spellings are the same arithmetic up to the reconstruction
  error; see `docs/vc4-glsl-costs.md` for why the fast one is cheaper.
- `RES_PACK=0` — disable the default two-q16-per-texel residual stream (q8 and
  q4 use packed residual by default).
- `PREFILL_PAIR=0` — disable the two-position prefill pair (the default since
  step 5 of the prefill-batching work).  `diff_paths.sh` sets it so both paths
  step every position through the same code.
- `PAIR_WROW=0` — keep the single-position prefill path for a model with
  per-row q8 scale planes.  The pair kernels have wrow variants and are on by
  default: byte-identical tokens, and warm prefill 7.0-7.2 s against 8.2-8.4 s
  on a Pi Zero with void.1 q8r.  The paired path compiles the P2 program set,
  so its *first* prefill is dearer (15.9 against 14.1 s); that is startup.  The
  switch exists so the A/B can run inside one binary, with `PREFILL_REPEAT`.
- `PREFILL_REPEAT=n` — replay the prefill `n` times, for timing.
- `FFN_FLUSH=0`, `FFN_SYNC=1`, `LAYER_SAFE_FINISH=1`, `FORCE_GLFINISH=1` —
  finish/flush policy.  `LAYER_SAFE_FINISH=1` restores the fully-finished
  `GPU_LAYER` path; `FORCE_GLFINISH=1` forces the VC4 finish path on other
  drivers.

## Placement: which path runs

- `GPU_LAYER=1` — hidden state, attention, RMSNorm and FFN all on the GPU.  The
  fast path; not correct for every model on VC4 (see `docs/performance.md`).
- `GPU_ATTN=1` — GPU RoPE + KV cache + attention, CPU FFN.
- `CPU_FFN=1`, `GPU_FFN=1` — force FFN placement.

> `GPU_LAYER=0` is **not** CPU-only: attention/FFN move to the CPU, but the
> matvecs and logits still run on the GPU.  The engine has no CPU matvec path,
> so "CPU attention" numbers are mixed CPU/GPU measurements.

## Precision and quality

- `ACT8=1` — all matvec inputs in `I8_4` (q8 weights only, opt-in).
- `ACT8=2` — FFN hidden state in `I8_4` (q8 weights only, opt-in).
- `Q8_LAYERS=all|0,1,...`, `Q8_PARTS=qkv,wo,ffn,w2,logits,emb` — precision mix.
- `Q4_WEIGHTS=1` — runtime q4 conversion for experiments.  Prefer the offline
  Q4M1 container; `Q4_SCOPE=all|attn|ffn` only applies to runtime conversion.
- `Q4_SHIFT=0` — force the float nibble decode instead of the ESSL 3.00 shift
  decode (`Q4_SHIFT_NO_MESA`, `Q4_SHIFT_VERBOSE` are VC4-specific overrides).
  `gpu_q4_shift()` reports the choice at startup.
- `Q4_4X4_ONEW=1`, `Q4_4X4_ONEROW=1`, `Q4_4X4_NOOUT=1` — stage-removal arms for
  the 4-bit gate: one matrix, one row, and the swiglu/output.  **`NOOUT` is not
  a price**: discarding the fragment value lets the driver eliminate the whole
  kernel, so it measures 0.8 ms against 8.2 and means nothing.  The other two
  are real: 8.18 -> 6.31 (one matrix) and 5.71 (one row), i.e. sub-linear, so a
  fixed ~4-5 ms a draw sits under the per-matrix-row work.
- `Q4_4X4_NOP=1` — pricing arm for the 4-bit gate above, the `Q4_NOUNPACK`
  methodology: it drops the four P fetches and the scale/min decode per group
  (constant scale), with wrong weights on purpose.  It measured 84 ms of the
  gate's 159 on the Pi, which is why the group plane is the next target.
- `Q4_GATE_NESTED=1` — restore the nested group-loop form of the 4-bit FFN
  gate (the pre-2026-11-16 default) for an interleaved A/B.  The default is
  the flat 8-wide gate: byte-identical (probe hash matches at 304/608/1216/2048
  rows on Xe and VC4) and ~43% faster at the engine's 1216-row chunk
  (6.55 -> 3.74 ms/draw).  The legacy pricing arms below route to the nested
  body too.
- `Q4_QKV_FLAT=1`, `Q4_W2_FLAT=1` — opt into the flat-loop form of the qkv
  (`matvec_i4_4x4`) and w2/wo (`matvec_i4_add_4x4`) 4-bit kernels.  **Default
  off: measured a net loss.**  The same rewrite that sped the gate up -43%
  made these single-row kernels slower (Pi Zero 2026-11-16, clean interleaved
  rounds: all-flat 197.5 ms/token vs gate-only-flat 185.0 vs all-nested 227.3
  — the qkv/w2 flat form costs ~12.5 ms/token).  The flat forms are
  byte-identical (same per-group operation order via conditional float
  updates; w2 keeps its generic `X4Q` decode) and stay as the measurement
  arms that produced the negative result.
- `Q4_4X4_FLAT=1`, `Q4_4X4_FLAT8=1`, `Q4_4X4_FLAT8M=1`, `Q4_4X4_FLAT8G=1`,
  `Q4_4X4_FLAT8B=1` — structure arms for the residual gap that survives
  `NOP`+`Q4_NOMIN` (all q4 metadata gone, yet ~1.5x the q8 gate).  `FLAT`/`FLAT8`
  use the NOP constant scale and no min, wrong weights on purpose: `FLAT` is
  q8's single 4-wide loop with 4-bit weights (no decode), `FLAT8` is the current
  8-wide inner body flattened into one loop.  `FLAT8M` fetches/applies the
  per-group scale/min per iteration (96x); `FLAT8G` fetches/decodes once per
  group via a conditional (24x), applies per iteration (both reassociate the
  min, so pricing arms).  `FLAT8B` is the byte-identical candidate: flat 8-wide
  loop, per-group d1/xsum reset at q%4==0 and flush `s10+=sa1*d1+ma1*xsum` at
  q%4==3, the nested kernel's exact operation order (`q4mv_probe`).
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
- `KV_SCALE=N` — KV fixed-point scale (default 16384; q16 uses 2048, q8 uses 16).
- `KV_PACK=16|8` — packed KV layout: two int16 per RGBA8 texel (width 384) or
  four int8 (width 192), instead of one 24-bit value (width 768).  A hybrid
  model (LFM2) defaults to 16: the 24-bit layout is wrong for kv_dim 512 /
  head_dim 64 in the fused path, and 16 is the canonical Pi/Xe setting.  An
  explicit `KV_PACK=0` still selects the 24-bit layout.
- `KV8_SIM=1..6`, `KV8_SIM_SCALE=N`, `KV8_STATS=1` — price an 8-bit KV grid on
  the CPU attention path (`GPU_LAYER=0`): 1 global K+V, 2 per-head K+V, 3 global
  K, 4 global V, 5 per-head V, 6 per-head K.  Probe only, no GPU kernel.
- `SEQ=N` — override the checkpoint KV cache length; tools auto-bump to 768,
  worker mode auto-bumps to 1024.

## Loading and memory

- `MODEL` — the container path.  Everything else (tokenizer, template, rope,
  specials) comes from the file.
- `MODEL_BORROW=0` — copy the weights out of the mapping instead of borrowing
  them.  Borrowing is the default: a 69 MB q8 checkpoint then costs no RSS and
  reloads in tens of milliseconds.
- `KEEP_CPU_WEIGHTS=1` — keep the CPU copies after texture upload (default:
  free them; q8 MiniMind3 RSS drops from ~119 MB to ~56 MB).

## Tokenizer

The container normally decides how text becomes ids: the blob is inside the
file, and the template supplies whatever prefix the model was trained with.
Two switches exist for the cases where it does not, both of them about the
SentencePiece dummy prefix (a leading `▁` that a Llama-shaped tokenizer adds
before the first word):

- `TOK_NO_PREFIX=1` — do not add it at all.  This is what matches HF's
  `tokenizers.encode()` on a raw string, and it is what a fixture test needs
  when the reference tokenizes the same text.
- `TOK_SPACE_FOLD=1` — add it as part of the text (`" hello"`) instead of
  emitting it as its own token, so the BPE can merge it into `▁hello`.  The
  default emits it separately, which is what the MiniMind2/3 gates were
  baselined on.

Measured on a Llama-shaped fixture (`tiny-random-LlamaForCausalLM`), `"hello"`:

| setting | ids |
|---|---|
| default | `[22172, 28401]` — prefix emitted as its own token |
| `TOK_SPACE_FOLD=1` | `[22172, 28401]` — no merge, same output |
| `TOK_NO_PREFIX=1` | `[12199, 4456]` — no prefix at all |
| HF `tokenizers` | `[22172]` — prefix merged into the word |

So neither switch reproduces HF for that model yet; see `engine/src/tokenizer.cpp`
and the note in `docs/performance.md` (2026-11-09).  `make_tok.py --input` builds
the blob these switches act on.

## Generation

- `--greedy` — deterministic argmax sampling.
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
- `--system TEXT` — override the default system prompt.  The compiled default
  is `You are a helpful assistant.`; the checkpoint may state its own.
- `--thinking`, `--open-thinking` — prefill `<think>` instead of the empty
  think block.
- `--tools JSON`, `--tools-file PATH` — inject tool definitions.
- `--worker` — long-lived JSONL generation worker used by the OpenAI server.
- `IGNORE_EOS=1` — keep generating after the stop tokens, for 64-token math
  tests.

## Prefix cache

- `--build-prefix-cache PROMPT_FILE CACHE_FILE` — prefill a fixed prefix and
  dump the per-layer KV textures.
- `--prefix-cache CACHE_FILE` — load a validated KV prefix at worker startup.
  The header binds the activation format, `KV_PACK` and `KV_SCALE`, so a cache
  built with different flags is rejected.
- `PREFIX_CACHE_RESAVE=path` — CLI only, with `--prefix-cache`: write the
  just-loaded KV rows back out through the normal save path, so the load can be
  compared byte for byte with the built file (`cmp`).

- `RES_FMT=i16_1|i24_1` — overrides the container's residual width, so a
  format can be A/B'd on one file.
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

## Retired

- `ATTN_HOIST=0`, `ATTN_FUSE_PROB=0` — the hoisted/fused attention path became
  mandatory with the q16 layout, and the code they selected was removed: the
  variables now do nothing at all (there is no unhoisted `att_score` and no
  probability texture to read).
- `LEGACY_Q8=1`, `ROPE_THETA=`, `ROPE_HALF=`, `--template` — gone with the
  legacy readers; the container states all of it.
- **`TOKENIZER=` is *not* retired**, and it is the one to be careful with: the
  engine loads that blob *instead of* the container's embedded tokenizer
  (`engine.cpp`: `if (model.tok_blob && !getenv("TOKENIZER"))`).  Nothing warns,
  the token ids then mean something else, and for a model whose stop token is
  vocabulary-dependent the answer comes back **empty**.  Leave it unset; if a
  run prints no `tokenizer: ... embedded` line, this variable is why.
- `--kv-pack` and `--seq` are **server** flags, never engine flags: the server
  turns them into `KV_PACK` / `SEQ` for the worker it spawns.

## Ad-hoc debug

Investigation leftovers that this document does not define.  None is part of a
contract: read the call site before relying on one, and never build a product
path on one.

| Switch | What it does |
|---|---|
| `Q4_INLINE`, `Q4_MV4`, `Q4_NOUNPACK`, `Q4_BFAST`, `Q4_HFAST`, `Q4_NOMIN` | select q4 kernel variants (inline decode, vec4, no unpack, fast/slow nibble paths, drop the min term).  Landed variants are the default; these exist to A/B them inside one binary.  Every q4 kernel unpacks through `q4_taps()`, so `Q4_NOUNPACK`/`Q4_BFAST`/`Q4_HFAST` and the ESSL 3.00 shift decode reach all of them (`scripts/shader_hashes.sh` shows `q4shift0/` and `q4shift1/` entries); `Q4_MV4` and `Q4_NOMIN` are scoped to the paired swiglu gate. |
| `KV_VSTEP` | 8-bit KV per-head V step (step 9b/9c experiment) |
| `MATVEC_STRIP` | debug the 1536-column strip split in `draw()` |
| `NO_SHADER_SELFTEST` | skip the link-time program self-test |
| `DRAW_CLEAR` | fill a draw target with 0x33/0x66/0x99/0xff before the draw, so a readback can tell "the draw did not execute" from "the draw saw stale input" (used to pin down the 1536-draw drop, `docs/engine-internals.md` §2) |
| `TZ` | timezone for the built-in `get_current_time` tool |

`REPETITION_PENALTY`, `FREQUENCY_PENALTY`, `PRESENCE_PENALTY`,
`NO_REPEAT_NGRAM` and `MIN_P` mirror the CLI flags of the same name; the flags
win, and the worker protocol sets them per request.

- `RWKV_CHUNK_TEST=1`：在桌面（GL 上限 16384）强制 RWKV channel-mix 按 2048 行分块，
  复现/验证 VC4 的分块路径（Pi 天然触发）。只影响 split_rwkv 的 max_dim。

- `RWKV_GPU=1`：启用 RWKV-7 GPU 融合层循环（P6，默认 CPU 串行）。
- `RWKV_GPU_DBG=1`：RWKV GPU 路径逐层诊断打印（P6）。
