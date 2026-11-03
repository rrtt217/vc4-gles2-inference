# Test hooks

Environment switches used for A/B tests and debugging. They are not all
production settings.

## Decode-campaign switches (2026-11-03)

Each of these defaults to the fast path and exists so a change can be A/B-ed
inside one binary (two builds add enough variance to hide a 1 ms effect).  They
are measurement hooks, not production settings.

- `QKV_PAIR=0` - single-row qkv matvec; the paired rows are the default
  (200.7 vs 183.5 ms/token).
- `FFN_FUSE=0` - two-pass FFN gate+swiglu; the fused kernel is the default
  (228.7 vs 183.5 ms/token).
- `LOGITS_SPLIT=1` - unpaired logits rows (two vocabulary entries per fragment
  otherwise).
- `RMS_SPLIT=1`, `RMS_FUSED=1` - force the split or fused rmsnorm reduce.  VC4
  defaults to fused (it drops one draw per norm); other drivers default to
  split, where 32 extra fetches per fragment can cost more than the draw.
- `Q8AP_INLINE=0`, `FFN_SW_INLINE=0` - restore the generic `X4Q` activation
  decode in the w2/wo and FFN gate+swiglu kernels.

Probes.  All default off, and all are bit-exact replays, so they can be enabled
without moving the gate:

- `ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v|all` - replay one attention pass
  per layer; the end-to-end delta is that pass's marginal cost.
- `PASS_DUP=embed,rms_att,rms_ffn,rms_final,logits,logits_draw|all` - the same
  for the fixed per-token passes (`logits_draw` also skips the read-back).
- `DRAWCOUNT=1` - print draws per generated token.
- `FFN_DUMP=1` - dump the first 8 texels of the layer-0 gate output, for bisecting
  a 1-LSB drift.
- `LAYER_PROFILE=1` - per-pass marks.  The `glFinish` in the lap also retires the
  draw issued before it, so use the replay probes above to price a pass; keep
  `LAYER_PROFILE` for finding *which* pass moved.

## Test hooks

- `TOPK=N` — exact Top-K candidate path (0/absent = full logits).
- `TOPK_FAST=1` — approximate candidate path.
- `CPU_FFN=1`, `GPU_FFN=1` — force FFN placement.
- `GPU_ATTN=1` — GPU RoPE + KV cache + attention (experimental).
- `GPU_LAYER=1` — hidden state, attention, RMSNorm and FFN all on GPU.
- `LAYER_SAFE_FINISH=1` — disable the GPU_LAYER selective-finish shortcut.
- `FFN_SYNC=1` — finish after FFN passes.
- `FFN_PAIR=0` — disable the paired gate matvec.
- `RES_PACK=0` — disable the default two-q16-per-texel residual stream;
  q8 and q4 use packed residual by default.
- `ROPE_THETA=1000000 ROPE_HALF=1` — required for legacy Q8M1
  MiniMind2-small.q8; those fields are not in the old Q8 header.
- `LEGACY_Q8=1` — quantize a legacy fp32 `.bin` checkpoint to q8 instead of
  the default I16.  Use only when I16 does not fit; small TinyStories models
  (15M/260K) lose coherence in q8.
- `ACT8=2` — FFN hidden state in I8_4 (q8 weights only, opt-in).
- `ACT8=1` — all matvec inputs in I8_4 (q8 weights only, opt-in).
- `FORCE_GLFINISH=1` — force the VC4 finish path on other drivers.
- `Q8_LAYERS=all|0,1,...`, `Q8_PARTS=qkv,wo,ffn,w2,logits,emb` — precision mix.
- `LOGITS_OUT=path` — write float32 logits after prefill.
- `TOKENS_OUT=path` — write prompt + generated token ids (int32).
- `TIMING=1` — print chat prefill and decode timing to stderr.
- `IGNORE_EOS=1` — keep generating after `<|im_start|>`/`<|im_end|>` for
  64-token math tests.
- `--greedy` — deterministic argmax sampling.
- `--seed N` — RNG seed for non-greedy sampling (default 1234).
- `--template minimind3` — MiniMind3 ChatML/thinking prompt.
- `--template qwen` — Qwen-style ChatML for MiniMind2/TinyChat-style files.
- `--system TEXT` — override the default neutral system prompt
  (`You are a helpful assistant.`); the same prompt is used by the server.
- `ATTN_HOIST=0`, `ATTN_FUSE_PROB=0` — retired after the q16 attention
  layout; the hoisted/fused path is mandatory.  The variables are ignored.
- `--thinking`, `--open-thinking` — prefill `<think>` instead of the empty
  think block.
- `--tools JSON`, `--tools-file PATH` — inject MiniMind3 tool definitions.
- `SEQ=N` — override the checkpoint KV cache length; tools auto-bump to 768,
  worker mode auto-bumps to 1024.
- `--worker` — long-lived JSONL generation worker used by the OpenAI server.
- `--build-prefix-cache PROMPT_FILE CACHE_FILE` — prefill a fixed prefix and
  dump the per-layer KV textures.
- `--prefix-cache CACHE_FILE` — load a validated KV prefix at worker startup.
  The cache header binds the activation format, so an `ACT8` cache is not
  accepted by a non-ACT8 process (and vice versa).
- `--ppl FILE` — teacher-forced perplexity over FILE; use the same corpus for
  format comparisons.
- `KV_SCALE=N` — KV fixed-point scale (default 16384; q16 uses 2048,
  q8 uses 16).
- `KV_PACK=16|8` — packed KV layout: two int16 per RGBA8 texel (width 384)
  or four int8 (width 192), instead of one 24-bit value (width 768).
- `KEEP_CPU_WEIGHTS=1` — keep the CPU copies after texture upload for debugging
  (default: free them; q8 MiniMind3 RSS drops from ~119 MB to ~56 MB).
- `Q4_WEIGHTS=1` — runtime q4 conversion for experiments.  Prefer the
  offline Q4M1 checkpoint (`minimind3.q4`), which quantizes directly from F16.
  `Q4_SCOPE=all|attn|ffn` only applies to runtime conversion.
