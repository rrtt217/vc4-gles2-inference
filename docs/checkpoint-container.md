# Checkpoint container v2 (MMCF)

Status: **implemented, and it is the only format**.  `engine/tools/mmcf_pack.py`
converts q8, q4 and i16 checkpoints, `Model::load_mmcf()` loads them, and since
`fcc4095` the engine has no other reader — `Model::load()` refuses anything that
is not `"MMCF"` and names the packer in the message.  All three precisions print
the same greedy text and the same PPL as their legacy source (measured
2026-11-14 on Xe, `docs/benchmarks.md`):

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
   packer (`--verify`) and the inspector (`--check`), not on every load.  The
   packer's `--verify` also re-reads every record's id/dtype/shape/group/
   scale/`aux` and compares it with the tensor it just wrote: a CRC covers the
   body only, so without that a container with two experts' `aux` swapped
   passes inspection and runs the wrong expert (the 2026-11-12 MoE audit's
   mutant test).  `--check` is standalone and still CRC-only.
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
| 4 | 2 | `version` | 2 (dense), 3 (mixture of experts, "MoE" below) or 4 (RWKV-7, see "v4" at the end).  The reader refuses anything else |
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
| 10 | `<BB`, then optionally a third byte: add_bos, add_eos (`255` = not stated; `-1` in memory), `sp_prefix` (1 = the tokenizer prefixes every text run with its space marker, see below) |
| 12 | `<B` residual width: absent or 0 = q16 (the historical grid), 1 = q24 |
| 11 | `<f` `rms_norm_eps`.  Absent = the engine's historical 1e-5, compiled as a literal.  A file that states one switches the norm shader to a uniform, which is **not** numerically neutral on this driver (MiniMind3 PPL 270.3170 -> 269.9303 with the same value), so the default path stays literal |
| 14 | one byte per layer: `0` attention, `1` the LFM2 conv block.  Absent = every layer is attention.  A file with conv tensors and no key 14 is refused, not guessed |
| 13 | `<4i` mixture of experts: `n_experts`, `top_k`, `expert_hidden`, flags.  Absent = dense.  Flags bit 0 = `norm_topk_prob`.  Present means the container is version 3 and every layer is sparse (see "MoE") |
| 18 | one f32 per rope pair (`head_dim/2` of them), a `longrope` model's static per-dimension frequency table: `freq_i = theta^(-2i/head_dim) / factor_i`.  Absent = every factor is 1, which is every container packed before 2026-11-18 |
| 15 | one byte per layer: the FFN hidden store scale exponent `k`.  Layer `k` means the SwiGLU hidden is written `*2^-k` (the activation rail becomes `127.996 * 2^k`) and w2's read-back multiplies by `2^k` (folded into its scale after load).  Absent = every layer 0, which is every container packed before 2026-11-12.  The loader refuses a length that is not `layers` or a value over 5 |

### Metadata keys 10 (third byte) and 18: the tokenizer's marker, the rope table

Two facts that used to be implicit in the packer and are now stated in the file.

**Key 18 — a rope table.**  MiniCPM4 (and BitCPM-CANN, its ternary QAT release)
divides rope pair `i`'s frequency by `factor[i]`: 32 different bases on a 64-wide
head, which one theta cannot express.  The payload is `head_dim/2` little-endian
f32s; the engine divides the angle it builds in `Session` and validates the
length against `head_dim/2` (a table of the wrong length is refused, because the
angles would be wrong *silently*).  Only a *static* table is representable: the
packer accepts `rope_type: longrope` when the short and long tables agree and
`max_position_embeddings == original_max_position_embeddings` (the model then
computes `scaling_factor = 1`), and refuses the rest by name.

**Key 10's third byte — the prefix policy.**  The reference tokenizers run their
normalizer on *every* text run between added tokens, so each one starts with the
space marker (`\u2581user`), while the engine's historical behaviour puts one
marker at the front of the whole prompt.  On a SentencePiece vocabulary the two
produce different ids -- "user" 1836 vs "\u2581user" 3060 on BitCPM, logits
cosine 0.912 -- so the container states which one it wants instead of the engine
guessing from the template.  A two-byte payload keeps the historical behaviour,
which is why the byte was appended rather than put in a key of its own.

Neither key changes the tensor table, so both are version-2 additions that an
older reader skips.

### The t3 body: 2-bit ternary

`mmcf_pack.py --quant t3` writes a body whose matrices are 2 bits a weight.  It
is only written for a matrix that **is** per-row ternary -- `t3mat()` certifies
every row is `{0, +s, -s}`, decodes what it wrote and refuses if the round trip
is not exact to float32 -- so the format is lossless by construction: the same
claim `q8mat` makes on such a matrix, at a quarter of the bytes.

The runtime has **two** readers:

- **The 2-bit kernel** (`matvec_t3` and its pair and add forms), ESSL 3.00 only:
  the ES2 float decode is 2.5x *slower* than q8, so a driver without integer
  shifts has no fast form for this kind.  The body uploads as a 2D tile of at
  most `kDrawCap` rows -- VC4's `GL_MAX_TEXTURE_SIZE` is 2048 and the FFN gate is
  4096 rows -- and the four bytes of a texel are two rows by two column groups.
  `docs/bitcpm.md` has the measurements and the two kernel rules that came out of
  them (one texture fetch a fragment, `ws` carries the missing 127).
- `T3_EXPAND=1` expands the body into q8 levels at load time (`Model::read_tables`,
  `model.cpp`, `as_t3`): +-127 for a nonzero weight, with the part-3 per-row plane
  attached, i.e. the exact dequantisation a q8 body would carry with
  `--q8-row-scale`.  The load prints

  ```text
  t3 body: 120 matrices expanded to q8 on the host (85 MB on disk -> 342 MB in RAM)
  because T3_EXPAND is set
  ```

  That is the packer's acceptance route and the oracle the 2-bit kernels are
  diffed against; it forfeits the memory win (4x the file) and the board was
  OOM-killed at 280 MB.

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

The `group` byte of a q4 body is **32** in files written since 2026-11-12 and
**0** in every file written before that (the writer never set it; the engine has
always used 32 from metadata key 3 and never read the record's copy).  Both
parse; a reader should take the group size from key 3.

Dtypes: `0` f32, `1` i16, `2` q8u8 (biased by 128), `3` q8i8 (signed), `4` q4
(nibbles, plus a scale plane as part 1 and a min plane as part 2), `5` blob
(the tokenizer), `6` q6 (two planes, `--head-q6`) and `7` **t3**: a 2-bit
ternary body, 16 weights an RGBA8 texel, four a byte, two bits a weight -- bit 0
magnitude, bit 1 sign, so a weight is `mag * (1 - 2*sign)` -- written only for a
matrix that *is* per-row ternary, with a per-row scale plane as part 3 exactly
like `--q8-row-scale`.

The body is a texel grid of `ceil(cols/8) x ceil(rows/2)`: one texel carries
**two rows by two column groups**, so a kernel that computes two output rows a
fragment uses all four bytes it fetched.  Texel `(c, t)` holds row `2t` columns
`8c..8c+3` in `b0`, row `2t` columns `8c+4..8c+7` in `b1`, and the same two for
row `2t+1` in `b2`, `b3`.  The row-major reading -- four column groups of one
row -- was the first layout; a kernel could use one of the four bytes per fetch,
and using all four needed four activation fetches from one sampler, which VC4
returns undefined data for once the texture is tall (`probes/wrow_probe`).  The
column-major reading is 7.8-8.1x cheaper per MAC on the board
(`probes/t3mv_probe`, the `t3p` arm, 2026-11-19).

`cols` pads to 8 and `rows` to 2 with zeros, which decode to zero weights.  The
grid is half as tall as the matrix, which is why no shape of this model reaches
VC4's 2048-texel texture limit.

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
against 270.3170 (2026-11-08, Xe; `docs/vc4-1536-repro.md`) - the 0.08% is the
per-tile q8 scale, since each tile quantises its own rows.  Changing it therefore costs a repack of every model
and two more logits draws per token, and the unpacked logits target is
`LOGITS_GRID` texels wide, so it must stay within the driver's texture/FBO
limit (2048 on VC4).

The Python side of this layout lives in `engine/tools/mmcf_format.py` (one copy
for `mmcf_pack.py`, `mmcf_dump.py` and `mmcf_requant_q8.py`), and
`engine/gate/test_mmcf_format.py` round-trips every container in `models/`
through it and checks its constants against `engine/src/model.cpp`.

dtypes: `0` f32, `1` i16, `2` q8u8 (biased, so the upload is a copy), `3` q8i8,
`4` q4, `5` blob, `6` q6 (the 6-bit embedding body `mmcf_hf.py --head-q6` writes:
two planes per row, `dim/2` bytes of 4 bits then `dim/4` bytes of 2, with a
per-row scale in the `embedding_scale` tensor that follows it; `model.cpp` reads
it as `INT6`, and `mmcf_dump.py` prints it as `q6`).  `dtype` replaces the magic: q8 pre-biased and q8 signed are
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
against 8.2-8.4 s serial on a 17-byte prompt), not a factor of two.  It refuses
a **hybrid** model for the same structural reason: the pair loop implements a
dense transformer block and calls `attention_block()` for every layer, while a
conv layer carries no qkv (the LFM2-350M q8 pack aborted there, 2026-11-12).

### The activation rail

The FFN's SwiGLU hidden is written through the q16 activation format (step
1/256, rail **+/-127.996**), so a hidden value over the rail is clamped —
silently, because the shader clamps to the int16 range.  MiniMind3-MoE is the
first model here that trips it: on a chat prompt the expert hidden reaches
**158/161 (layer 3) and 250/222 (layer 7)** at the template's `<|im_start|>`
positions, i.e. 49% of the layer-7 value is cut, while the dense model peaks at
88.8 and plain text (the `--ppl` corpus) at 15.1.  The four positions are part
of every chat prompt (the template prefix), so the error is systematic, small
in the logits it was measured against, and invisible without a calibration.

`mmcf_pack.py` tracks both rails in its float32 calibration (residual and FFN
hidden, sampled over vocabulary runs *and* the tokenizer's added tokens) and
writes **key 15**: one byte per layer, the smallest `k` whose store fits
(`peak / 2^k <= 127.996`, capped at 5).  The serial FFN kernels write the hidden
`*2^-k` (one exact multiply before the q16 encode; the emitted GLSL is
byte-identical when `k = 0`), and `Model::apply_hidden_scale()` folds `2^k` into
every w2 (and its chunks) after `split_ffn()` — free on the read side, because
`ws`/`ps` already carry a per-matrix scale.  This is the same device
`GPU::matvec_s` uses for a matvec input far from the grid.

The two-position prefill kernels have no hidden scale, so `pair_ok()` refuses a
scaled container's prefill pair (the MoE block already refused it).  The cost is
one bit of hidden resolution on the scaled layers: measured on MiniMind3-MoE,
the clip is gone (`hidA` at layer 7 position 0: 7.59 -> 0.22 against float32)
while PPL moves +0.5% on i16/q8 (0.005 nats/token, inside the per-token
scatter) and -2.5% on q4.  A container packed with `--no-rail-check` gets no
key 15 and keeps the old clamped behaviour.

Consequence for the model store: MiniMind3-MoE is 198M parameters (397 MB of
F16, ~204 MB as q8, ~138 MB as q4) against the dense 64M model's 66 MB at q8.
Per token it reads one expert, so decode is dense-speed; what MoE costs on this
stack is *memory*, and the engine keeps every expert resident.  A board whose
CMA cannot hold all of them needs a smaller expert count or width — not a
different kernel.  `docs/benchmarks.md` has the measurements.

## Loader rules

- `version` unknown -> refuse with the version number in the message.
- `file_size != st_size` -> refuse (truncated download).
- header `crc32` mismatch -> refuse.
- Unknown `dtype` -> refuse, naming the tensor id and the dtype.  Guessing here
  would turn a future fp8 file into silent q8 arithmetic.
- Unknown *tensor id* -> ignore (forward compat).
- a q4 body whose scale/min planes do not have the body's shape (`rows`), a
  matching `cols` and `2 bytes x rows x groups` of payload -> refuse, naming
  the tensor and expert.  The copy path (`MODEL_BORROW=0`) used to copy one
  row's worth of planes and read the rest out of bounds.
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

## v4: RWKV-7 (version 4)

Added 2026-11-16 with the model's first container; the model page is
[`rwkv7.md`](rwkv7.md).  An
RWKV-7 container is **version 4**: v3's 64-byte record, one new metadata key
(16) and 17 new tensor ids.  Dense containers stay version 2 and MoE
containers stay version 3, so every container in `models/` is byte-identical
to what it was; a v4 file is the only thing this section describes.

### Why a version at all, and why the record did not grow

A v2 or v3 file can already *carry* the numbers: an unknown key is skipped and
an unknown tensor id is ignored.  That is exactly the trap.  A reader that
skipped key 16 would load the tensors it knows, find no `rms_att`, and a
reader that skipped only some ids would run a transformer block over an RWKV
checkpoint.  The version number turns that into one line at the header
(`mmcf: unsupported version 4`, which is what the pre-v4 engine prints) instead
of a half-loaded model.  `Model::probe()` reads the header and the metadata
only, so the REPL's `/model` refuses a v4 container before it releases the
scene it is running.

The record stays 64 bytes because nothing v4 adds is per-tensor: the
architecture's hyper-parameters are model-wide (key 16) and every new kind's
shape is already in the record's `rows`/`cols`.  A longer record would be 16
zero bytes per tensor, and `rec_size(4) == 64` keeps one reader for all three
layouts (`MMCF_REC_V3` covers v3 and v4).

### Metadata key 16 (16 B)

`<BBBBfHHHH>`: arch, head_size, flags, pad, norm_eps, r_w, r_a, r_v, r_g.

| offset | size | field | notes |
|---:|---:|---|---|
| 0 | 1 | `arch` | `1` = RWKV-7.  A v4 file without this key is refused, and so is the key in any other version |
| 1 | 1 | `head_size` | the WKV head width (64 in every released model).  Must equal key 1's `dim / heads`, which is how the reader checks that the two agree |
| 2 | 1 | `flags` | bit 0 = the time-mix has the gate branch (`0x2A`/`0x2B` present).  The rest is reserved and 0 |
| 3 | 1 | pad | 0 |
| 4 | 4 | `norm_eps` (f32) | the LayerNorm epsilon.  The per-head group norm keeps the architecture's **64e-5** and is not a field: a model that wanted another one would be a different block |
| 8 | 2 | `r_w` | decay low rank (0.1B: 64) |
| 10 | 2 | `r_a` | ICLR low rank (64) |
| 12 | 2 | `r_v` | value-residual low rank (32) |
| 14 | 2 | `r_g` | gate low rank (128, or 0 for a model without the gate) |

The four ranks are stored rather than derived because they are not derivable:
llama.cpp's fallback formula `round(d^e * m / 32) * 32` gives **160** for
0.4B's gate where its `config.json` says **128**, and the packer refuses a
source that does not state them instead of guessing.

### The 17 new tensor ids

All of them sit in the empty slots of `kind(8) | layer(6) | part(2)`, and the
names are llama.cpp's GGUF names so a reader can put the two side by side.
The HF column is the source name (fla-hub naming, which the official `.pth`
conversion reproduces: `models/rwkv7/official/pth_to_safetensors.py`).

| id | name | rows x cols | dtype | HF tensor |
|---|---|---|---|---|
| `0x1A` | `time_mix_lerp` | `layers x 6*dim` | f32 | `attn.x_{r,w,k,v,a,g}` in that order |
| `0x1B` | `time_mix_bias` | `layers x 3*dim` | f32 | `attn.{w,a,v}_lora.lora.2.bias` (w0, a0, v0) |
| `0x1C` | `time_mix_scale` | `layers x 3*dim` | f32 | `attn.k_k`, `attn.k_a`, `attn.r_k` (`r_k` is `heads x 64`, flattened) |
| `0x1D` | `att_norm` | `layers x 2*dim` | f32 | `attn_norm.{weight,bias}`: the LayerNorm before the time mix |
| `0x1E` | `att_norm_2` | `layers x 2*dim` | f32 | `ffn_norm.{weight,bias}`: before the channel mix |
| `0x1F` | `time_mix_ln` | `layers x 2*dim` | f32 | `attn.g_norm.{weight,bias}`: the per-head group norm |
| `0x21` | `channel_mix_lerp` | `layers x dim` | f32 | `ffn.x_k` |
| `0x22` | `tok_norm` | `1 x 2*dim` | f32 | `model.layers.0.pre_norm.{weight,bias}`; absent when the source has none |
| `0x23` | `output_norm` | `1 x 2*dim` | f32 | `model.norm.{weight,bias}` |
| `0x24`/`0x25` | `time_mix_w1`/`w2` | `r_w x dim` / `dim x r_w` | q8/q4 | `attn.w_lora.lora.{0,2}.weight` |
| `0x26`/`0x27` | `time_mix_v1`/`v2` | `r_v x dim` / `dim x r_v` | q8/q4 | `attn.v_lora.lora.{0,2}.weight` |
| `0x28`/`0x29` | `time_mix_a1`/`a2` | `r_a x dim` / `dim x r_a` | q8/q4 | `attn.a_lora.lora.{0,2}.weight` |
| `0x2A`/`0x2B` | `time_mix_g1`/`g2` | `r_g x dim` / `dim x r_g` | q8/q4 | `attn.g_lora.lora.{0,2}.weight` |

Three rules that are not visible from the table:

- **Rows are output rows**, like every other `Mat` the engine reads: the down
  projection is `rank x dim` and the up projection `dim x rank`, not the other
  way round.  The vector groups keep the HF order *inside* a row, so
  `time_mix_lerp`'s row is literally `x_r | x_w | x_k | x_v | x_a | x_g`.
- **Layer 0 has no value-residual pair.**  It records `v_first` instead of
  mixing it in, so `0x26`/`0x27` are absent there and present on every other
  layer; the reader allows exactly that, and a gated model must have
  `0x2A`/`0x2B` on every layer or on none (key 16's flag says which).
- **`split_w13` must be 1.**  The channel mix is an ordinary key/value pair
  and there is no fused W13 to flag, so a v4 file that says `split_w13 = 0`
  is refused.

### What v4 reuses

| id | RWKV-7 use | shape |
|---|---|---|
| `0x10` | `[W_r; W_k; W_v]` stacked in one row block | `3*dim x dim` |
| `0x11` | `W_o` | `dim x dim` |
| `0x12` | `W_cm_key` | `hidden x dim` |
| `0x15` | `W_cm_value` | `dim x hidden` |

Reusing them is not cosmetic: the q4 group planes, the per-row q8 scale plane
(`part` 1/2/3) and the upload path all key on the id, so the four largest
matrices of an RWKV model get q8, q4 and `--q8-row-scale` without a second
code path.  `Model` puts them in `qkv`, `wo`, `w1` and `w2` for the same
reason (`engine/src/model.h`).

### Metadata key 17: the head input's scale (added 2026-11-16, P0b)

One byte, `k`.  The vector that feeds the logits matvec is the final
LayerNorm's output, and the engine writes it through the q16 activation format
like any other vector: grid 1/256, rail **+-127.996** (`ACT_SCALE = 256` on an
int16).  RWKV-7 0.1B peaks at **315.6** there, 2.5x the rail, so a container
that stored it unscaled would clamp the very input of its own head.

The container states the fix instead of changing a kernel: the head input is
written `x * 2^-k` and **every logits tile's scale already carries `2^k`**,
so `(W/2^k-wscaled) dot (x * 2^-k)` comes out in the model's own units.  The
packer computes `k = ceil(log2(final_ln_peak / 127.996))` from its float32
calibration and folds `2^k` into the `scale` field of the `0x20` records - the
bodies are bit-identical to the same pack without the key, only the scale moves
(measured: tile 0 at 0.00501353 with `k = 2` against 0.00125338 without).

Absent (or 0) means the input is written unscaled, which is every container
packed before this key existed, so the read path is bit-identical for them.  A
v4 file without key 17 is legal: it is what `--no-rail-check` produces, and it
clamps at run time.

The packer also prints the analytic worst case for that input, from the weights
alone: a LayerNorm output is bounded channel-wise by `sqrt(dim)*|w_i| + |b_i|`
(the squared normalised values sum to `dim`), i.e. 668.3 on 0.1B, which would
want `k = 3`.  It is a bound that needs one channel to carry all the variance,
the measured peak is what sets `k`, and the printed margin is how far a real
prompt would have to go before the store clips.

### Reader rules for v4

- version 4 without key 16, key 16 in another version, an unknown `arch`, a
  `head_size` that disagrees with `dim / heads`, a non-positive rank, key 14
  (layer types) or key 13 (MoE) in a v4 file: refused by name.
- The nine f32 vector groups are shape-checked against `layers`, `dim` and
  the group count before anything reads them, and the low-rank pairs against
  key 16's ranks.  Without that the four reused ids would load as a
  transformer block with an RWKV head width.
- `rms_att`/`rms_ffn`/`rms_final` are transformer tensors and are not
  required in a v4 file (`att_norm` and friends replace them); an embedding
  and the logits tiles still are.
- Metadata key 17 (the head input's scale) is read into `Model::logit_shift`
  and a value over 5 is refused; the reader does nothing else with it, because
  the *2^k already sits in the logits tiles' scales.
- Metadata key 11 (`rms_norm_eps`) is **not** written for a v4 container:
  key 16 carries the LayerNorm epsilon, and key 11 keeps its own meaning
  (the RMSNorm shader's uniform, which is not numerically neutral).
- The tokenizer blob (`0x30`) and the chat template (key 9) are ordinary
  keys/tensors; a v4 file may have neither (the RWKV tokenizer and template
  are P3 of the plan) and the engine then falls back as it always did.
- **P1 boundary**: this build reads a v4 container and refuses to run it
  (`mmcf: ... no layer loop for the block yet`).  The block's layer loop, the
  WKV state and the two guards (`pair_ok()`, the prefix cache) are P2;
  deleting that one refusal is part of adding them, and the `Model` fields the
  reader fills (`rwkv`, `rwkv_*`) are what the loop consumes.

### How the mapping was verified

`mmcf_pack.py --verify` compares the container against its source tensor by
tensor, which proves the container is faithful but not that the *names* were
understood.  The mapping was checked against llama.cpp's own conversion of the
same official weights (`refs/llamacpp/src-b12283/conversion/rwkv.py` and
`gguf-py/gguf/tensor_mapping.py`): every f32 tensor of
`models/rwkv7/rwkv7-g1d-0.1b-Q5_K_M.gguf` matches its mapped HF tensor byte
for byte - the five norm pairs, all six `time_mix_lerp` slots in order, `k_k`,
`k_a`, `r_k`, the channel-mix lerp, `w0`/`a0` - and layer 0's missing
`v_lora` matches llama.cpp's dummy `v0/v1/v2` (which its conversion drops
because the official loader never uses them).

### The 0.1B container (`models/rwkv7/packed/rwkv7-g1d-0.1b.q8.mmcf`)

196 tensors, 191,844,864 B, version 4, ranks 64/64/32/128, gate present, `seq`
512 (nominal: RWKV has no KV cache).  The packer's float32 calibration (three
runs of 8 vocabulary ids plus one chat-shaped prompt encoded with the
`rwkv_vocab_v20230424.txt` that ships next to the weights) reports:

| stream | peak | engine rail | consequence |
|---|---:|---:|---|
| residual | 37.4 (layer 11) | 128.0 | q16 fits, no key 12 |
| channel-mix hidden (`relu(x)^2`, the sqrelu stream) | 91.3 (layer 11) | 128.0 | 1.40x margin, no key 15 |
| final LayerNorm output (the head's input) | 312.5 | 128.0 | **2.44x**: key 17 = 2 |
| WKV state, per-layer max abs | 51.8 (layer 11) | not a rail | the number S-1 of the plan needs |

The calibration stream is the model's own vocabulary (a chat-shaped prompt,
`RWKV_RAIL_TEXT` in `engine/tools/mmcf_hf.py`): three runs of vocabulary ids
find 58 instead of 91 there, because the peaks live on real text.  These are
pack-time facts P2 starts from, not a model-quality claim: the recurrence is
never renormalised, so its magnitude is what decides whether the Pi can hold
the state in fixed point at all.
