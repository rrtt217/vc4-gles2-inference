# Prefix KV cache

The worker keeps the token and KV prefix of the previous request.  On a new
request it finds the longest common token prefix, reuses the KV entries before
that point, and pre-fills only the suffix.  A fully cached prompt costs one
token to re-seat the final logits.

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
cd engine
SEQ=768 GPU_LAYER=1 ./engine_mm3 \
  --build-prefix-cache ../server/prefix_tools.txt \
  ../server/prefix_tools_pi768.kv

cd ../server
python3 server.py --seq 768 \
  --prefix-cache prefix_tools_pi768.kv
```

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

Two packed layouts are available:

```text
--kv-pack 0   24-bit, one value per RGBA8 texel, width 768
--kv-pack 16  two int16 per texel, width 384, scale 2048
--kv-pack 8   four int8 per texel, width 192, scale 16
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

The `MMKV1` header binds the model metadata, `KV_PACK` mode, `qk_norm`,
`rope_half`, and the activation format (`ACT8`).  A cache built with
`ACT8=1` is rejected by a non-ACT8 process and vice versa; rebuild on the
target machine after changing any of those flags.

Build a packed disk cache with the same packing mode:

```bash
KV_PACK=16 SEQ=768 GPU_LAYER=1 ./engine_mm3 \
  --build-prefix-cache prefix_tools.txt prefix_tools_pack16_768.kv

python3 server.py --seq 768 --kv-pack 16 \
  --prefix-cache prefix_tools_pack16_768.kv
```

The cache header records the packing mode; loading with a different
`--kv-pack` is rejected.
