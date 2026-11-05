# Checkpoint container v2 (proposal)

Status: design draft, nothing implemented.  Today the engine reads four
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
