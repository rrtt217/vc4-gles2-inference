# Checkpoint container v2 (proposal)

Status: **implemented**.  `engine/tools/mmcf_pack.py` converts q8, q4 and i16
checkpoints, and `Model::load_mmcf()` loads them.  All three print the same
greedy text and the same PPL as their legacy source on Xe:

| model | legacy | `.mmcf` |
|---|---:|---:|
| minimind3 q8  | 270.3170 | **270.3170** |
| minimind3 q4  | 330.5302 | **330.5302** |
| minimind3 i16 | 280.2156 | **280.2156** |

With only `MODEL=` set, the container supplies the tokenizer (81,371 B
embedded), the template name and the default system prompt:

```
$ MODEL=/tmp/minimind3.mmcf ./engine_mm3 --chat --prompt "用一句话介绍你自己。" --gen 16 --greedy
template: minimind3 (from the checkpoint)
tokenizer: 81371 B embedded in the checkpoint
Assistant: 我是一个AI助手，我的存在是为了提供帮助和信息。
```  Today the engine reads four
formats, each with its own magic and its own positional header
(`MAGIC_Q8`/`MAGIC_Q8_V2`, `MAGIC_I16`/`MAGIC_I16_V2`, `MAGIC_Q4`, and the
legacy loop), and the tokenizer, chat template and inference flags live
outside the file in environment variables and command-line flags.  That is why
a checkpoint can be loaded with the wrong tokenizer, why `rope_half` had to be
passed by hand for MiniMind2, and why adding one flag (the `+128` pre-bias)
would need a fifth magic.

## Goals

1. **One container for every precision.**  i16, q8 and q4 differ by a dtype
   enum and a group size, not by a magic.
2. **Self-describing.**  Tokenizer, chat template, inference parameters and
   capability flags travel with the weights.
3. **mmap-friendly.**  Every tensor has an explicit offset and is 16-byte
   aligned, so `Mat::ext*` can point straight into the mapping (ARMv6 has no
   unaligned `int16` loads worth relying on).
4. **Extensible without a new magic.**  Version, section sizes and a
   skip-unknown rule.
5. **Integrity on demand.**  A per-tensor CRC in the table, verified only when
   asked (`--verify` or fused into the upload pass), not on every load.
6. **A pre-biased q8 dtype** so the runtime `+128` conversion (69 MB of
   per-byte or XOR work and a 69 MB temporary buffer) disappears.

## Layout

```
+---------------------------------------------------------------+ 0
| header, 64 B, little endian                                   |
+---------------------------------------------------------------+
| metadata: one JSON document, NUL padded to 4-byte alignment   |
+---------------------------------------------------------------+
| tensor table: n_tensors * 40 B                                |
+---------------------------------------------------------------+
| data: each tensor 16-byte aligned, in the layout the texture  |
|       wants (rows * cols/4 * 4 B for q8, etc.)                |
+---------------------------------------------------------------+
```

### Header (fixed, 64 B)

| field | type | notes |
|---|---|---|
| `magic` | `u32` | `"MMCF"` |
| `version` | `u16` | 2 |
| `header_size` | `u16` | 64, so a future header can grow |
| `n_tensors` | `u32` | |
| `flags` | `u32` | bit 0 chat, bit 1 tools, bit 2 vision, bit 3 thinking |
| `meta_offset` / `meta_size` | `u32` / `u32` | JSON block |
| `data_offset` | `u64` | tensor data base |
| `file_size` | `u64` | must equal `stat().st_size`; catches truncation |
| `crc32_header` | `u32` | of the first 60 bytes |
| reserved | `u8[8]` | zero |

### Metadata (JSON, ~1-3 KB)

```json
{
  "name": "minimind3",
  "arch": "qwen3-dense",
  "model": {
    "dim": 768, "layers": 8, "heads": 8, "kv_heads": 4, "head_dim": 96,
    "hidden": 2432, "vocab": 6400, "seq": 768,
    "rope": {"theta": 1000000.0, "half": true},
    "qk_norm": false, "tied_embeddings": true
  },
  "quant": {"scheme": "q8", "group": 0, "pre_biased": true},
  "tokenizer": {
    "kind": "bytelevel-bpe",
    "blob": "tokenizer",          // tensor name, stored verbatim
    "special": {"eos": 0, "pad": 0, "im_start": 6400, "im_end": 6401}
  },
  "template": {
    "kind": "chatml",
    "default_system": "You are a helpful assistant.",
    "roles": {"system": "<|im_start|>system\n", "user": "<|im_start|>user\n",
              "assistant": "<|im_start|>assistant\n"},
    "stop": ["<|im_end|>", "<|endoftext|>"]
  },
  "capabilities": {"chat": true, "tools": false, "vision": false}
}
```

JSON is deliberate: the block is small, human-readable in `strings`, and a
missing key is a defined fallback rather than a silent misparse.  The old
positional `cfg[7]` cannot express "absent".

### Tensor table (n_tensors x 40 B)

| field | type | notes |
|---|---|---|
| `name` | `u16` | index into a name table in the metadata |
| `dtype` | `u8` | 0 F32, 1 I16, 2 Q8_U8 (pre-biased), 3 Q8_I8, 4 Q4, 5 Q4_PARAM |
| `group` | `u8` | q4 group size (32), 0 otherwise |
| `flags` | `u16` | reserved |
| `rows` / `cols` | `u32` / `u32` | |
| `offset` / `bytes` | `u64` / `u64` | relative to `data_offset` |
| `crc32` | `u32` | of the tensor bytes |

`dtype` replaces the magic: q8 pre-biased and q8 signed are two values of one
enum, so the upload path picks "copy straight from the map" or "flip the sign
bit" per tensor without a format branch.

## Loader rules

- `version` unknown -> refuse with the version number in the message.
- `file_size != st_size` -> refuse (truncated download).
- Unknown `dtype` -> refuse; unknown *tensor names* -> ignore (forward compat).
- Missing `tokenizer` -> fall back to `TOKENIZER=`, warn once.
- Missing `template` -> fall back to `--template`, warn once.
- `capabilities.chat == false` and `--chat` -> refuse with a clear message.
- CRC verified when `--verify` is passed, or fused into the upload pass when
  `UPLOAD_VERIFY=1`; never on the default path (a full pass costs 0.1-0.3 s
  warm and would force the 6 s SD read on a cold start).

## What this buys, in project terms

| problem seen in this repo | fixed by |
|---|---|
| wrong tokenizer silently paired with a checkpoint | tokenizer inside the file |
| `ROPE_HALF=1` typed by hand for MiniMind2 | rope config in metadata |
| compiled default system prompt drifting from the gate | `default_system` in the template |
| `+128` upload conversion, 69 MB temp buffer | `Q8_U8` dtype |
| a flag bump needing a new magic + loader | `dtype`/`flags`/version |
| stale `bin/ref/*.base` measured against the wrong build | `name`/`arch`/checksum in metadata |

## Migration

1. `tools/convert_container --in model.q8 --tokenizer model.tok --template
   minimind3 --out model.mmcf` writes v2; the engine gains a v2 reader next to
   the existing ones.
2. Acceptance: the converted file must produce **byte-identical** weights, the
   same greedy text and the same PPL as the source.  A conversion that changes
   one nibble is a bug, not a re-baseline.
3. Only after every checkpoint in `models/` is converted do the old readers go
   away.

## Open questions

- Tensor **names**: a string table (richer, +bytes) or fixed ids (`qkv.3`,
  `w2.7`)?  Fixed ids keep the table at 40 B and the loader simple.
- Tokenizer blob: store the existing `.tok` bytes verbatim (simplest, no
  re-encoding risk) or normalise to a JSON vocab (portable, bigger)?
- One file per precision (current `models/` layout) or one file with several
  precisions of the same model (saves the tokenizer/template duplication, but
  every load pays for the whole file's page cache)?

## As built (packer)

`engine/tools/mmcf_pack.py --in model.q8 --tok model.tok --out model.mmcf
--name minimind3 --template minimind3 --system "You are a helpful assistant."
--chat --verify` converts a legacy q8 checkpoint (q4/i16 raise "not
implemented yet") and checks itself: the file-size field, every tensor CRC and
a byte-for-byte search for each tensor body.  A real run:

```
wrote /tmp/minimind3.mmcf: 53 tensors, 68980491 bytes (source 68897356)
verify: 53 tensors, caps=0x1 meta=64+154 data@2344 size=68980491 crc_bad=0
verify: every tensor present verbatim
```

The 83 KB of overhead is the header (64), the TLV block (154), the table
(53 * 40), alignment padding and the tokenizer blob.

### Header, exactly

| offset | size | field |
|---|---|---|
| 0 | 4 | `"MMCF"` |
| 4 | 2 | version = 2 |
| 6 | 2 | header_size = 64 |
| 8 | 4 | n_tensors |
| 12 | 4 | capability flags (1 chat, 2 tools, 4 vision) |
| 16 | 4 | meta_offset = 64 |
| 20 | 4 | meta_size |
| 24 | 8 | data_offset (16-byte aligned) |
| 32 | 8 | file_size (checked against `st_size`) |
| 40 | 4 | crc32 of bytes 0..39 (reserved, not yet computed) |
| 44 | 20 | reserved |

### Metadata TLV

Each record is `<HHI` (key, type, length) followed by the payload:

| key | payload |
|---|---|
| 1 | `<7i` dim, hidden, layers, heads, kv_heads, vocab, seq |
| 2 | `<fBBH` rope theta, rope_half, qk_norm, pad |
| 3 | `<BBBB` scheme (0 i16, 1 q8, 2 q4), group, pre_biased, split |
| 4 | `u8 kind`, `default system\0`, `template name\0` |
| 5 | `<I` capability flags (mirror of the header) |
| 6 | `name\0` |
| 7 | `<i` tokenizer blob length |

### Tensor record, 40 B

`<HBBBBIIQQIIH`: id, dtype, flags, group, pad, rows, cols, offset, bytes,
crc32, pad, pad.  `offset` is relative to `data_offset` and every body starts
16-byte aligned.

Tensor ids: `1` embedding, `2` embedding scale, `3` rms_att, `4` rms_ffn,
`5` rms_final, `6` q_norm, `7` k_norm, `0x10` qkv, `0x11` wo, `0x12` w1,
`0x13` w3, `0x14` w13, `0x15` w2, `0x20` logits tile, `0x30` tokenizer blob.
Per-layer tensors carry the layer in the high byte (`id | layer << 8`).

dtypes: `0` f32, `1` i16, `2` q8u8 (the packer's output for q8: biased, so the
upload is a copy), `3` q8i8, `4` q4, `5` blob.

### Reader checklist (all done)

1. `Model::load()` recognises `"MMCF"` before the legacy magics and calls a new
   `load_mmcf`.
2. Parse the header, reject `version != 2` and `file_size != st_size`, walk the
   TLV block into the existing `Model` fields (dim, layers, rope, qk_norm, ...).
3. Build one `Mat` per tensor: `q8u8` points `ext8` at `map_ + data_offset +
   offset` (no copy, no bias); `f32`/`i16` are copied into vectors as today.
4. `Tokenizer::load_memory(blob, len)` for the embedded blob; keep
   `TOKENIZER=` as the fallback.
5. Chat template and `default_system` from key 4 replace the compiled default
   and the `--template` flag, with the flag still winning.
6. Acceptance: the converted file must print the same greedy text and the same
   PPL as the source on both machines, and `UPLOAD_TIME=1` must show the q8
   conversion gone.

## Implementation notes (what actually broke)

- The per-matrix scale (the `ws` uniform) is a field of the tensor record.  The
  first draft dropped it and every dot product ran with scale 0.
- A `"<HBBBBIIQQIIf"` record is **42 bytes**, not 40 or 48; every later offset
  shifted by 6 bytes per tensor.  Records are now built by explicit
  concatenation and padded to 48.
- `slice_rows`/`slice_cols` did not copy the new `prebiased` flag, so the split
  FFN weights (hidden 2432 splits every layer) were biased a second time at
  upload and the model answered garbage.
- q4 needs three tensors per matrix (nibbles, group scale, group min) sharing
  one id; `m.groups` must be the per-row group count (`scl->cols`), not the
  plane size, or `slice_cols` walks the scale plane out of bounds.
- The layer vectors are sized by the file's `split` flag, so indexing
  `layers_w13` on a split file is UB - index each vector by its own size.
- ESSL 1.00 has no implicit int->float conversion: `int(x) * 1.5` does not
  compile in a shader, it needs `float(int(x)) * 1.5`.

## What the reader enforces

- **version** unknown -> refuse with the number printed.
- **file_size != st_size** -> refuse (truncated download).
- **header crc32** (bytes 0..39, stored at 40) -> refuse on mismatch; a zero
  field means "not written" and is accepted for hand-made files.
- **unknown dtype** -> refuse, naming the tensor id and the dtype.  Guessing
  here would turn a future fp8 file into silent q8 arithmetic.
- **capabilities** -> a file that does not advertise `chat` refuses `--chat`
  (exit 2); a file that does advertise it enters chat mode on its own when the
  caller passes neither `--chat` nor a generation length.
- **template kind** -> 0 plain, 1 chatml/qwen, 2 minimind3; an unknown kind
  falls back to the name string, and `--template` always wins.
- tokenizer blob, when present and `TOKENIZER=` is unset.

`engine/tools/mmcf_dump.py` prints all of it from the file alone
(`--tensors` adds one line per tensor, `--check` verifies every CRC), so
"self-describing" is checkable:

```
$ python3 engine/tools/mmcf_dump.py minimind3.q4.mmcf --check
version     2   header 64 B   tensors 133
sections    meta 64+156   data 6608   size field 46866347 (ok)
header crc  0x535a1c69 ok
caps        0x1 (chat)
model       dim 768 hidden 2432 layers 8 heads 8 kv_heads 4 vocab 6400 seq 256
rope        theta 1e+06 half True qk_norm True
quant       q4 group 32 pre_biased True split_w13 True
template    kind 2 (minimind3) name 'minimind3'
system      'You are a helpful assistant.'
tokenizer   81371 B embedded
crc check   all 133 tensors ok
```

## Extensibility, honestly

Supported without touching the layout: new metadata keys (an unknown key is
skipped), new optional tensors (an unknown tensor id is ignored), new dtypes
(a file that uses one is refused, not misread), per-tensor precision (the dtype
is per tensor, so a mixed q8/q4 file is expressible today), extra header fields
(`header_size` can grow).

Not supported without a v3: a tensor dimension beyond `kind(8) | layer(6) |
part(2)` - MoE experts and multi-adapter files need either an `expert` field or
a string name table; structured (JSON) metadata, if human editing matters more
than a dependency-free reader; and per-tensor compression, which would break
the "borrow straight from the mapping" property that makes the loader free.
