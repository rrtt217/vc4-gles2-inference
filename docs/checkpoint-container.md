# Checkpoint container v2 (MMCF)

Status: **implemented, and it is the only format**.  `engine/tools/mmcf_pack.py`
converts q8, q4 and i16 checkpoints, `Model::load_mmcf()` loads them, and since
`fcc4095` the engine has no other reader — `Model::load()` refuses anything that
is not `"MMCF"` and names the packer in the message.  All three precisions print
the same greedy text and the same PPL as their legacy source on Xe:

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
```

Read "Layout", "Header", "Metadata" and "Tensor record" as the specification.
"Goals" and "Why it exists" are the history behind it, and the sections after
"Loader rules" keep the migration and the implementation notes — the mistakes
that cost time and are worth not repeating.

## Why it exists (the problem it replaced)

Before the container the engine read four formats, each with its own magic and
its own positional header (`MAGIC_Q8`/`MAGIC_Q8_V2`, `MAGIC_I16`/`MAGIC_I16_V2`,
`MAGIC_Q4`, and the legacy loop), and the tokenizer, chat template and inference
flags lived outside the file in environment variables and command-line flags.
That is why a checkpoint could be loaded with the wrong tokenizer, why
`rope_half` had to be passed by hand for MiniMind2, and why adding one flag (the
`+128` pre-bias) would have needed a fifth magic.

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
5. **Integrity on demand.**  A per-tensor CRC in the table, checked by the
   packer (`--verify`) and the inspector (`--check`), not on every load.
6. **A pre-biased q8 dtype** so the runtime `+128` conversion (69 MB of
   per-byte or XOR work and a 69 MB temporary buffer) disappears.

## Layout

```
+---------------------------------------------------------------+ 0
| header, 64 B, little endian                                   |
+---------------------------------------------------------------+
| metadata: TLV records, 4-byte aligned                         |
+---------------------------------------------------------------+
| tensor table: n_tensors * 48 B                                |
+---------------------------------------------------------------+
| data: each tensor 16-byte aligned, in the layout the texture  |
|       wants (rows * cols/4 * 4 B for q8, etc.)                |
+---------------------------------------------------------------+
```

(The proposal said 40 B per record.  The record as built is 48 B — it carries a
per-tensor `scale`.  See "Tensor record" below.)

### Header (fixed, 64 B)

| offset | size | field | notes |
|---:|---:|---|---|
| 0 | 4 | `magic` | `"MMCF"` |
| 4 | 2 | `version` | 2 (dense) or 3 (mixture of experts, "MoE" below) |
| 6 | 2 | `header_size` | 64, so a future header can grow |
| 8 | 4 | `n_tensors` | |
| 12 | 4 | `capability flags` | bit 0 chat, bit 1 tools, bit 2 vision, bit 3 reserved |
| 16 | 4 | `meta_offset` | 64 |
| 20 | 4 | `meta_size` | |
| 24 | 8 | `data_offset` | 16-byte aligned |
| 32 | 8 | `file_size` | must equal `stat().st_size`; catches truncation |
| 40 | 4 | `crc32` | of bytes 0..39.  A zero field means "not written" and is accepted for hand-made files |
| 44 | 20 | reserved | zero |

Bit 3 is reserved.  The packer never sets it and the reader reads the caps word
only for bit 0 (chat).  The inspector used to print it as "thinking", which was
never implemented either; do not attach meaning to a bit without adding both the
writer and the reader.

### Metadata (TLV, ~150-250 B)

Each record is `<HHI>` (key, type, length) followed by the payload; `type` is 1
(bytes/struct) and an unknown key is skipped.

| key | payload |
|---|---|
| 1 | `<7i` dim, hidden, layers, heads, kv_heads, vocab, seq |
| 2 | `<fBBH` rope theta, rope_half, qk_norm, pad |
| 3 | `<BBBB` scheme (0 i16, 1 q8, 2 q4), group, pre_biased, split |
| 4 | `u8 kind`, `default system\0`, `template name\0` |
| 5 | `<I` capability flags (mirror of the header) |
| 6 | `name\0` |
| 7 | `<i` tokenizer blob length |
| 8 | `<6i` bos, eos, stop, pad, unk, im_start (all `-1` = absent) |
| 9 | `u8 n`, then n × (`name\0`, `u32 len`, **Jinja template text**).  The default variant is stored with an empty name and read as `default` |
| 10 | `<BB` add_bos, add_eos (`255` = not stated; `-1` in memory) |
| 12 | `<B` residual width: absent or 0 = q16 (the historical grid), 1 = q24 |
| 11 | `<f` `rms_norm_eps`.  Absent = the engine's historical 1e-5, compiled as a literal.  A file that states one switches the norm shader to a uniform, which is **not** numerically neutral on this driver (MiniMind3 PPL 270.3170 -> 269.9303 with the same value), so the default path stays literal |
| 14 | one byte per layer: `0` attention, `1` the LFM2 conv block.  Absent = every layer is attention.  A file with conv tensors and no key 14 is refused, not guessed |
| 13 | `<4i` mixture of experts: `n_experts`, `top_k`, `expert_hidden`, flags.  Absent = dense.  Flags bit 0 = `norm_topk_prob`.  Present means the container is version 3 and every layer is sparse (see "MoE") |

Key 9 is the model's own `chat_template` from HuggingFace, stored **verbatim**:
the same text a GGUF file carries in `tokenizer.chat_template`.  A reader that
does not know the key skips it and falls back to key 4's kind, which is why
adding it did not need a version bump.

Template kinds: 0 plain, 1 chatml/qwen, 2 minimind3, 3 tinybrainbot.  The kind
is authoritative; the name string is kept for humans and as the fallback for a
kind an older engine does not know.

The original proposal used one JSON document here.  The TLV block replaced it:
same "absent is expressible" property, no parser, and 154 bytes for MiniMind3.
The full block is printed by `engine/tools/mmcf_dump.py` from the file alone.

### Tensor record (48 B in v2, 64 B in v3)

Built by explicit concatenation, not a struct format — an earlier
`"<HBBBBIIQQIIf"` layout is **42 bytes**, not 40 or 48, and shifted every later
offset by 6 bytes per tensor.  Version 3 appends `aux` (the expert index) inside
v2's tail padding and pads the record to 64 bytes, so every field above keeps
its offset and a v2 file still parses with `aux = 0`:

| offset | size | field |
|---:|---:|---|
| 0 | 2 | id (`kind(8) | layer(6) | part(2)`) |
| 2 | 1 | dtype |
| 3 | 1 | flags |
| 4 | 1 | group (q4 group size, else 0) |
| 5 | 1 | pad |
| 6 | 4 | rows |
| 10 | 4 | cols |
| 14 | 8 | offset (relative to `data_offset`, 16-byte aligned) |
| 22 | 8 | bytes |
| 30 | 4 | crc32 of the tensor body |
| 34 | 4 | `f32` scale (the per-matrix `ws`; the first draft dropped it and every dot product ran with scale 0) |
| 38 | 2 | **v3:** `aux` — the expert index; 0 in a dense tensor.  v2: pad |
| 40 | 24 | pad to 64 |

The record size is not a header field: the table starts at the first 16-byte
boundary after the metadata, so `rec_size = (data_offset - table_start) /
n_tensors` is 48 or 64 and a reader that knows the version can validate it.
64 is a multiple of 16, so appending 16 bytes per record keeps `data_offset`
16-aligned without a second pad.

Tensor ids: `1` embedding, `2` embedding scale, `3` rms_att, `4` rms_ffn,
`5` rms_final, `6` q_norm, `7` k_norm, `8` embedding_i16, `0x10` qkv, `0x11`
wo, `0x12` w1, `0x13` w3, `0x14` w13, `0x15` w2, `0x16` router (MoE),
`0x17` conv in_proj, `0x18` conv depthwise kernel (f32, dim x k), `0x19` conv
out_proj (the LFM2 hybrid block; see "Hybrid layers"), `0x20`
logits tile, `0x30` tokenizer blob.  Per-layer tensors carry the layer in the
high byte (`id | layer << 8`).  q4 needs three tensors per matrix (nibbles,
group scale, group min) sharing one id.

The top two id bits are a plane index.  `1` and `2` are the q4 group scale and
group min; **`3` is a per-row q8 scale plane**: `f32`, `rows x 1`, one value per
output row, the row's scale *divided by the matrix unit* the record's own
`scale` field carries (`f32` there is the largest row scale).  The engine
uploads it as one 24-bit value per row and every q8 kernel multiplies the row's
dot product by it instead of by a single `ws`
(`wrow_scale()` in `engine/src/shaders.cpp`).  A container either has a plane
for every q8 matrix or for none: a matrix that keeps its per-matrix scale while
the file says otherwise is silently wrong, which is why the reader refuses a
plane whose `rows` or byte count does not match its matrix.

Why it exists: void.1's weight rows span 500:1 within one matrix, so a single
scale leaves 58-66% of layer 0's FFN weights at zero.  Per row the median
relative error falls 2.36% -> 0.79% and the worst matrix 16.6% -> 1.9%; on the
engine the model goes from unusable (logits 14.4 off f32, misspelled bytes) to
i16 quality (max 1.1 off, top-1 the same, PPL 3.80 vs i16's 3.78).

### Hybrid layers (LFM2)

A layer typed `1` in key 14 has no qkv/wo.  It runs `0x17` (in_proj,
`3*dim x dim`), a depthwise causal convolution `0x18` (`dim x k` floats, k =
the block's `conv_L_cache`) and `0x19` (out_proj, `dim x dim`).  The gating
splits the in_proj output: rows `0:dim` (B) times rows `2*dim:3*dim` (x), then
the conv output times rows `dim:2*dim` (C).  The session keeps `k-1` gated rows
per conv layer as state.  `q_norm`/`k_norm` still carry one row per layer; a
conv layer's row is unused and packed as ones.  Embeddings, logits tiles and
the FFN are unchanged, so the dense path runs it unmodified once the layer loop
knows the type.  A reader that predates key 14 sees attention on a conv layer
and refuses on the missing qkv rather than answering wrongly.

The vocabulary (`0x20`) may be **absent** when the head is tied
(`mmcf_pack.py --head-dedup`): the embedding rows already carry one scale each,
so the engine builds the tiles as views into the embedding bytes with that
plane (`matvec_q8_pair_logits` with `wrow`).  A container with neither tiles
nor a q8 embedding is refused.

The vocabulary (`0x20`) is stored as several tiles of at most **1536 rows**
(`LOGITS_GRID` in `engine/tools/mmcf_grid.py`, `kLogitsGrid` in
`engine/src/logits_grid.h`), and the last tile is short: MiniMind3's 6400 rows
are 4 x 1536 + 256.  The engine renders one tile per target row and maps
`token = tile * 1536 + index`, so the stride is a property of the file, not of
the engine build; `Model::load` refuses a file whose tiles do not fit the grid
rather than decoding with shifted token ids.  The rule is: the tile count must
be `ceil(vocab / 1536)`, every tile except the last must be exactly 1536 rows,
and the last 1..1536.  The count alone is not enough - a 1300-row stride gives
the same five tiles and every row fits, but shifts token ids by 236, and a run
with such a file printed the correct greedy text *and* the correct `--top-k 8 --top-k-candidates`
sample because every chosen token happened to be in the first tile.  `engine/probes/logits_grid_probe`
holds the packer, the host and the four Top-K kernels to the same number.

Where 1536 came from is not recorded: it arrived with the first commit, already
baked into the legacy layout by converters that were deleted without a note.  It
is three quarters of VC4's 2048 texture limit, 2 x 768 (MiniMind3's hidden size
and the fused W13 width of the llama2.c models the engine was first brought up
on), and MiniMind3's qkv width - which of those drove the choice is unknown.
6400 = 4 x 1536 + 256 is a side effect, not a derivation: 1600 would have
divided the vocabulary into four even tiles.

The stride is a convention, not an algorithm requirement: repacking MiniMind3 q8
at 1024 rows (7 tiles instead of 5) gave the same greedy text and PPL 270.5287
against 270.3170 - the 0.08% is the per-tile q8 scale, since each tile
quantises its own rows.  Changing it therefore costs a repack of every model
and two more logits draws per token, and the unpacked logits target is
`LOGITS_GRID` texels wide, so it must stay within the driver's texture/FBO
limit (2048 on VC4).

dtypes: `0` f32, `1` i16, `2` q8u8 (biased, so the upload is a copy), `3` q8i8,
`4` q4, `5` blob.  `dtype` replaces the magic: q8 pre-biased and q8 signed are
two values of one enum, so the upload path picks "copy straight from the map" or
"flip the sign bit" per tensor without a format branch.

## MoE (version 3)

A version-3 container carries a mixture-of-experts block instead of the dense
FFN.  Version 2 files are byte-identical to what they were and the engine reads
both; the packer writes v3 only when it writes expert tensors, so every existing
container is untouched.

Metadata key 13 is the switch.  When it is present:

- every layer is sparse.  `mlp_only_layers` and `decoder_sparse_step != 1` are
  refused at pack time: a container is all-dense or all-MoE.
- each layer has a router tensor (`0x16`, layer in the high byte, `aux = 0`,
  `rows = n_experts`, `cols = dim`).  It stays i16 in every container, q4
  included: the router is 4x768 values and a route flip is not a rounding
  error, it is a different FFN.
- each layer has `n_experts` w1/w3/w2 triples with the dense FFN's kinds and
  the expert index in the record's `aux`.  `split_w13` must be 1; a fused W13
  is not a MoE tensor.  The q4 group scale/min planes (id parts 1 and 2) and
  the per-row q8 scale plane (part 3) carry the same `aux`, or four experts
  would resolve to one matrix.
- `expert_hidden` is the expert FFN width and may differ from key 1's `hidden`
  (the config's dense width, unused when key 13 is present).  Every expert's
  w1/w3 is `expert_hidden x dim`, w2 is `dim x expert_hidden`.
- `top_k` must be 1 and bit 0 of the flags must be set (`norm_topk_prob`).
  With top-1 and a normalised gate the selected expert's output is used with
  weight exactly 1.0, so the runtime needs neither a softmax nor a weighted
  combine.  Both the packer and the loader refuse the rest.

The runtime for this block is the dense one.  The router is an
`n_experts x dim` matvec whose logits are read back (one sync per layer per
token: a discrete choice on a GLES2 pipeline has to come back to the host), and
the selected expert's w1/w3/w2 go through the same `ffn_split` /
`ffn_split_chunked` kernels as a dense layer.  The FFN chunks are per expert
(`expert_hidden > 2048` splits it in two), so a 2432-wide expert costs the same
per token as a 2432-wide dense FFN: only one expert's weights are read.

The exception is prefill: the two-position pair path runs one FFN for two
positions, and the two positions route independently (to different experts in
general).  `pair_ok()` therefore refuses MoE, and prefill falls back to the
serial loop — the difference measured in `session.cpp` (7.0-7.2 s paired
against 8.2-8.4 s serial on a 17-byte prompt), not a factor of two.

Consequence for the model store: MiniMind3-MoE is 198M parameters (397 MB of
F16, ~204 MB as q8, ~138 MB as q4) against the dense 64M model's 66 MB at q8.
Per token it reads one expert, so decode is dense-speed; what MoE costs on this
stack is *memory*, and the engine keeps every expert resident.  A board whose
CMA cannot hold all of them needs a smaller expert count or width — not a
different kernel.  `docs/performance.md` has the measurements.

## Loader rules

- `version` unknown -> refuse with the version number in the message.
- `file_size != st_size` -> refuse (truncated download).
- header `crc32` mismatch -> refuse.
- Unknown `dtype` -> refuse, naming the tensor id and the dtype.  Guessing here
  would turn a future fp8 file into silent q8 arithmetic.
- Unknown *tensor id* -> ignore (forward compat).
- MoE, in a v3 file: `top_k != 1`, a gate without `norm_topk_prob`, a fused
  W13, an incomplete or wrongly shaped expert, an expert index outside
  `n_experts`, or a router tensor in a container without key 13 -> refuse with
  the layer/expert in the message.  A file that loads must be runnable; the
  alternative is an answer computed from three of four experts.
- Missing tokenizer blob -> fall back to `TOKENIZER=`, warn once.
- Missing template -> the compiled default; there is no `--template` flag.
- `capabilities & 1 == 0` and `--chat` -> refuse with a clear message (exit 2).
  A file that *does* advertise chat enters chat mode on its own when the caller
  passes neither `--chat` nor a generation length.
- Per-tensor CRC is verified by `mmcf_pack.py --verify` and by
  `mmcf_dump.py --check`, never on the load path: a full pass costs 0.1-0.3 s
  warm and would force the 6 s SD read on a cold start.

## What this buys, in project terms

| problem seen in this repo | fixed by |
|---|---|
| wrong tokenizer silently paired with a checkpoint | tokenizer inside the file |
| `ROPE_HALF=1` typed by hand for MiniMind2 | rope config in metadata |
| compiled default system prompt drifting from the gate | `default_system` in the template |
| `+128` upload conversion, 69 MB temp buffer | `Q8_U8` dtype |
| a flag bump needing a new magic + loader | `dtype`/`flags`/version |
| stale `bin/ref/*.base` measured against the wrong build | `name`/`arch`/checksum in metadata |

## Migration (as it happened)

1. `mmcf_pack.py --in model.q8 --tok model.tok --template minimind3
   --out model.mmcf` writes v2; the engine gained a v2 reader next to the
   existing ones.
2. Acceptance: the converted file must produce **byte-identical** weights, the
   same greedy text and the same PPL as the source.  A conversion that changes
   one nibble is a bug, not a re-baseline.  All 15 checkpoints in `models/` were
   converted and verified this way, on Xe and on the Pi.
3. Only after every checkpoint in `models/` was converted did the old readers go
   away — `fcc4095`, which also deleted `--template`.

## Open questions (answered, kept for the record)

- Tensor **names**: fixed ids won.  The table is 48 B either way, but the loader
  stays a switch instead of a string lookup.
- Tokenizer blob: the existing `.tok` bytes verbatim.  No re-encoding risk, and
  the packer derives the special-token ids from it.
- One file per precision: kept.  The tokenizer/template duplication is ~80 KB
  against a 45-326 MB file.

## As built (packer)

`engine/tools/mmcf_pack.py` converts every legacy layout this repo produces —
`Q8M1`, `Q8M2`, `Q4M1`, `I16M1`, `I16M2` and llama2.c fp32 — and checks itself:
the header CRC, the file-size field, every tensor CRC and a byte-for-byte search
for each tensor body.  A real run:

```
wrote /tmp/minimind3.mmcf: 53 tensors, 68980491 bytes (source 68897356)
verify: 53 tensors, caps=0x1 meta=64+154 data@2344 size=68980491 crc_bad=0
verify: every tensor present verbatim
```

The 83 KB of overhead is the header (64), the TLV block (154), the table
(53 * 48), alignment padding and the tokenizer blob.

### Reader checklist (all done)

1. `Model::load()` recognises `"MMCF"` before the legacy magics and calls a new
   `load_mmcf`.
2. Parse the header, reject `version != 2` and `file_size != st_size`, walk the
   TLV block into the existing `Model` fields (dim, layers, rope, qk_norm, ...).
3. Build one `Mat` per tensor: `q8u8` points `ext8` at `map_ + data_offset +
   offset` (no copy, no bias); `f32`/`i16` are borrowed the same way.
4. `Tokenizer::load_memory(blob, len)` for the embedded blob; keep
   `TOKENIZER=` as the fallback.
5. Chat template and `default_system` from key 4 replace the compiled default.
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

## What the reader derives from the file

Beyond the refuse-rules above, three things that used to be guesses are now
stated by the file:

- **special token ids**: BOS/EOS/stop come from the file (key 8).  A legacy file
  keeps the historical guess (1 = BOS, 2 = EOS); the container derives them from
  the embedded vocabulary (`<s>`, `<|endoftext|>`, `</s>`, `<|im_end|>`; matching
  is on the trimmed token, because llama2.c stores `<s>` as `"\n<s>\n"`).  This
  is not cosmetic: MiniMind3's id 0 is `<|endoftext|>` and id 1 is
  `<|im_start|>`, so the old rule stopped on `<|im_start|>` and never on
  `<|endoftext|>`, and with a chatml template it would also prepend a duplicated
  `<|im_start|>` as BOS.  tinybrainbot is where it shows: the legacy build stops
  on its first token and prints an empty answer, the container keeps going.
- **rope**: the container's theta/half are authoritative.  Because a
  `Q8M1`/`I16M1` header carries no rope at all, the packer **refuses to guess**:
  it wants `--theta` plus `--rope-half`, or `--rope-plain` for a llama2 model,
  or an explicit `--assume-llama2-rope`.
- **template kind**: 0 plain, 1 chatml/qwen, 2 minimind3, 3 tinybrainbot.  An
  unknown kind falls back to the name string; there is no `--template` flag to
  override it with.  Since key 9 exists, the kind is the *fallback* for
  containers written before it: a file that carries a template renders that
  template, and refuses to load if the engine cannot parse it.
- **BOS policy** comes from key 10 when present.  Without it the reader keeps
  the historical rule, which derived "no BOS" from the template kind — that
  coupling is exactly what key 10 removes.

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
