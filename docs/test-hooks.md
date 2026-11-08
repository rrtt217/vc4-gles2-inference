# Test hooks

Environment switches used for A/B tests, diagnostics and debugging.  They are
not production settings.

Three rules before using this list:

- **A switch that no longer exists is worse than a missing one.**  `--template`,
  `TOKENIZER=`, `ROPE_THETA=` and `ROPE_HALF=` were removed with the legacy
  readers (`fcc4095`); the container carries all four.  Passing them now is
  silently ignored.
- **Prefer an in-binary switch to a second build.**  Two builds add enough
  variance to hide a 1 ms effect on the Pi.
- **The source is the ground truth.**  The full inventory, including ad-hoc
  debug flags that are not listed here, is:

  ```bash
  grep -rho 'getenv("[A-Z_0-9]*")' engine/src | sort -u    # 74 today
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
| `QKV_DUMP=1` | hashes every layer/position of the `qkv` matvec output; `2` also prints the first two rows.  Layout-agnostic, so `QKV_PAIR=0` and `QKV_PAIR=1` runs compare exactly |
| `ATT_DUMP=<pos>` | the exact integers the score/softmax path consumed and produced, for every layer at one position.  Feed the log to `engine/scripts/check_att.py` |
| `FFN_DUMP=1` | the first 8 texels of the layer-0 gate output, for bisecting a 1-LSB drift |
| `LOGITS_OUT=path` | float32 logits after prefill.  `engine/scripts/cmp_logits.py` compares two of them per tile |
| `TOKENS_OUT=path` | prompt + generated token ids (int32) |
| `MODEL_DUMP=1` | the parsed model metadata, right after load |
| `MMCF_DEBUG=1` | per-tensor detail while walking the container table |

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
- `DRAWCOUNT=1` — print draws per generated token.
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
- `LOGITS_SPLIT=1` — unpaired logits rows.
- `RMS_SPLIT=1`, `RMS_FUSED=1` — force the split or fused rmsnorm reduce.  VC4
  defaults to fused (it drops one draw per norm); other drivers default to
  split, where 32 extra fetches per fragment can cost more than the draw.
- `Q8AP_INLINE=0`, `FFN_SW_INLINE=0` — restore the generic `X4Q` activation
  decode in the w2/wo and FFN gate+swiglu kernels.
- `UPLOAD_XOR=0` — restore the byte-at-a-time int8 → uint8 weight conversion.
  The default converts four bytes at a time with the equivalent `^ 0x80`; the
  texture bytes are identical either way.
- `RES_PACK=0` — disable the default two-q16-per-texel residual stream (q8 and
  q4 use packed residual by default).
- `PREFILL_PAIR=0` — disable the two-position prefill pair (the default since
  step 5 of the prefill-batching work).  `diff_paths.sh` sets it so both paths
  step every position through the same code.
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
- `TOPK=N` — exact Top-K candidate path (0/absent = full logits).
- `TOPK_FAST=1` — approximate candidate path; not the correctness baseline.

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
- `KV_SCALE=N` — KV fixed-point scale (default 16384; q16 uses 2048, q8 uses 16).
- `KV_PACK=16|8` — packed KV layout: two int16 per RGBA8 texel (width 384) or
  four int8 (width 192), instead of one 24-bit value (width 768).
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

## Generation

- `--greedy` — deterministic argmax sampling.
- `--seed N` — RNG seed for non-greedy sampling (default 1234).
- `--system TEXT` — override the default system prompt.  The compiled default
  is `You are a helpful assistant.`; the checkpoint may state its own.
- `--thinking`, `--open-thinking` — prefill `<think>` instead of the empty
  think block.
- `--tools JSON`, `--tools-file PATH` — inject tool definitions.
- `--worker` — long-lived JSONL generation worker used by the OpenAI server.
- `IGNORE_EOS=1` — keep generating after the stop tokens, for 64-token math
  tests.
- `TOPK`, `TOPK_FAST` — see placement above.

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

- `ATTN_HOIST=0`, `ATTN_FUSE_PROB=0` — the hoisted/fused attention path is
  mandatory since the q16 layout; the variables are ignored.
- `LEGACY_Q8=1`, `ROPE_THETA=`, `ROPE_HALF=`, `TOKENIZER=`, `--template` — gone
  with the legacy readers; the container states all of it.
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
| `TZ` | timezone for the built-in `get_current_time` tool |

`REPETITION_PENALTY`, `FREQUENCY_PENALTY`, `PRESENCE_PENALTY`,
`NO_REPEAT_NGRAM` and `MIN_P` mirror the CLI flags of the same name; the flags
win, and the worker protocol sets them per request.
