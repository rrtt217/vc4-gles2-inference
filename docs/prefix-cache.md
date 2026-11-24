# Prefix KV cache

The worker keeps the token and KV prefix of the previous request.  On a new
request it finds the longest common token prefix, reuses the KV entries before
that point, and pre-fills only the suffix.  A fully cached prompt costs one
token to re-seat the final logits.
The CLI `--prefix-cache` path clamps the skip the same way; before step 10 it
could skip every token and sample from an empty logits vector (docs/benchmarks.md).

The chat REPL is a third chain, in memory only.  It keeps the tokens of the
previous turn, re-renders the whole exchange through the container's template,
and reuses the KV rows before the longest common token prefix.  That includes
the assistant reply, which re-tokenizes to the tokens that produced it, and it
survives `--tools`: the tool loop appends the assistant turn, its tool calls and
the tool results to the same history, so a round that follows a tool response
prefills only the response and the template markers - the system + tools
preamble and the earlier exchange stay in the cache.  Nothing is written to
disk and no `MMKV1` header is involved; `/clear` drops the chain.  `CHAT_CACHE=1`
prints the `cached=` / `prefill=` split of every call.

`start` and `done` events include:

```json
{"cached_tokens":79,"prefill_tokens":1}
```

The HTTP response exposes them as:

```json
"usage": {
  "prompt_tokens": 79,
  "completion_tokens": 8,
  "total_tokens": 87,
  "prompt_tokens_details": {"cached_tokens": 79},
  "prefill_tokens": 1
}
```

Local q8, same 79-token chat prompt:

| request | cached | prefill | wall |
|---|---:|---:|---:|
| first | 0 | 79 | 1064 ms |
| repeat | 79 | 1 | **119 ms** |

Tool round-trip (system + tools + call + tool response):

| request | cached | prefill | wall |
|---|---:|---:|---:|
| tool call | 53 | 393 | 5.8 s |
| after tool response | **446** | 75 | **1.4 s** |

The cache is one prefix chain, not a tree.  A changed system prompt or tools
block invalidates from the first differing token.  There is no eviction; the
worker keeps only the latest chain.

### Disk prefix cache

The worker can load a saved prefix at startup and keep it as the initial chain.
Build it with the same model, precision, device, and `SEQ`:

```bash
# The engine renders the header with the container's own template, prefills it,
# writes the KV and writes a `<cache>.json` describing it (2026-11-14; the Python
# builder used to render the header itself, hard-coded to ChatML).
cd server
python3 build_prefix_cache.py build \
  --model ../models/minimind3/minimind3.q8.mmcf --seq 768 --kv-pack 16 \
  --system "You are a helpful assistant." \
  --tools system_status get_current_time \
  --out pfx_tools_pi768.kv

python3 server.py --seq 768 --prefix-cache pfx_tools_pi768.kv
```

`--prefix-cache` may also name a **directory**: the engine loads the entry whose
header matches the model it is holding, so several models share one directory and
a model swap picks the right one.  A `.kv` without its `.json` is refused at
server startup -- a stale cache does not fail, it silently prefills everything
again, which is the 83 s against 14.4 s case above.

The system+tools prefix is 417 tokens, 9.8 MB (8 layers × 768 KV columns ×
417 positions × 4 bytes).  Pi Zero measurements:

| scenario | cached | prefill | wall |
|---|---:|---:|---:|
| cold tool call | 0 | 447 | 277.2 s |
| same request, in-memory cache | 447 | 1 | 26.5 s |
| disk cache, tool call | 417 | 30 | **50.7 s** |
| disk cache, tool response round-trip | 447 | 71 | **57.5 s** |
| build cache | — | 417 | 258 s |

Loading the 9.8 MB file is much cheaper than rebuilding the prefix.  The disk
cache saves the 417 system/tools tokens; the user message and generated tokens
still have to be prefilled/generated.

`SEQ=768` is required for the full tool round-trip at Pi (a 518-token prompt).
At 768 CmaFree was only ~2 MB after load, so this is tight; `SEQ=512` is safer
for a single tool call but cannot hold the tool response round-trip.  Generate
the cache on the same Pi/model; copying the model changes its mtime and
invalidates the cache.



### Packed KV

The layout is chosen by `KV` in the engine's environment; the server's
`--kv-pack` flag only sets that variable for the worker it spawns.

```text
KV=i24    24-bit, one value per RGBA8 texel, width 768
KV=q16   two int16 per texel, width 384, scale 2048
KV=q8    four int8 per texel, width 192, scale 16
```

Local Xe validation:

| mode | cache file (417 tok) | output |
|---|---:|---|
| 24-bit | 9.8 MB | baseline |
| packed q16 | **4.9 MB** | exact to unpacked q16 |
| packed q8 | **2.5 MB** | exact to unpacked q8 |

Packed q16 preserved 64-token output on the spring and self-introduction
prompts.  Packed q8 is numerically exact to uniform q8, but **uniform q8 is not
acceptable as a default**:

- self-introduction top-1 changes;
- spring generation drifts into a winter scene ("银装") in a spring essay;
- first-step logits vs 24-bit: cosine ~0.9975, MAE ~0.18, max ~1.09.

q16 first-step logits: cosine ~0.9998, MAE ~0.05, max ~0.30.  Use q16 if
reducing KV memory; q8 needs per-head/block scales or better calibration.

The `MMKV1` header (version 2) binds the model metadata, the `KV` value and its
codec (`i24`/`q16`/`q8`/`bf16`/`f32`), `KV_GRID`, `qk_norm`, `rope_half`, and
the activation format (`ACT`).  The codec has its own flag bits because bf16
and q16 share the width, the bits and the default scale; without them a cache
built under one was accepted under the other and decoded on the wrong grid.  A
cache built with `ACT=i8` is rejected by a q16-stream process and vice versa,
and a cache written at a different `KV_GRID` is rejected with a
`PREFIX_CACHE KV_GRID mismatch` line; rebuild on the target machine after
changing any of those flags.  Header version 1 files (written before
2026-11-04, when the scale was not recorded) fail the version check, so an
existing cache has to be rebuilt once.  The 2026-11-02 switch to
the neutral English system prompt also invalidates old tool/chat caches;
`build_prefix_cache.py verify` and the sidecar `system_sha` catch that.

Build a packed disk cache with the same packing mode:

```bash
python3 build_prefix_cache.py build \
  --model ../models/minimind3/minimind3.q8.mmcf --seq 768 --kv-pack 16 \
  --out pfx_tools_pi768.kv

python3 server.py --seq 768 --kv-pack 16 --prefix-cache pfx_tools_pi768.kv
```

The cache header records the packing mode and the `KV_GRID` grid; loading
with a different `--kv-pack` or `KV_GRID` is rejected.

## Version 4: the hybrid conv history (2026-11-11)

A hybrid model (LFM2) keeps `k-1` gated rows per conv layer; without them a
reloaded cache resumes from a state that never existed.  Version 4 adds them
after the KV rows (and the optional V-step plane): two rows per conv layer,
`x_texels(dim)` texels each, written in layer order.  Flag bit 32 marks the
file, and a reader refuses a version 4 file when the flag and the model
disagree.

Two rules come with it, both measured:

* a cache whose token prefix is **not fully reused** must have its conv state
  zeroed before the prefill (`zero_conv_state()`), or the first two tokens see
  a history from another token stream.  The single-shot path and the chat REPL
  both do this; the CLI cache build (`--build-prefix-cache`) bumps `SEQ` to
  1024 unless `SEQ` is set, so build and load must agree on `SEQ` or the header
  is rejected.
* `load_prefix_cache` ends with `forget_state()`: its uploads bypass the
  engine's texture cache, and a stale binding showed up as the next draw
  sampling the last uploaded row.

Round-trip is byte-identical (`PREFIX_CACHE_RESAVE` and `cmp`), and a loaded
cache with a non-matching prefix produces the same text as no cache.
