# Long context on the Pi Zero: where 2048 comes from and what it costs to pass it

Measured 2026-11-07 on the board (VC4, Mesa 26.2.2).  The short version: the
limit is the **height of the KV-cache texture**, not the model, and the engine
says so and clamps:

```
$ SEQ=4096 ./engine_mm3_pi ...
SEQ 4096 exceeds GL target limit 2048; clamping to 2048
```

## 1. Which texture runs out

`GPU::max_target_dim_` is the smallest of `GL_MAX_TEXTURE_SIZE`,
`GL_MAX_RENDERBUFFER_SIZE` and `GL_MAX_VIEWPORT_DIMS`, read at startup
(`gpu.cpp:70`).  On VC4 that is **2048**; on Iris Xe it is 16384.

Per session (`GPU::set_model`, `gpu.cpp:623-665`):

| target | shape | limit it imposes |
|---|---|---|
| `kv_cache_[layer]` | `(att_kv_width, seq)` | **`seq <= 2048`** — the binding one |
| `att_scores_` | `(seq/2, heads)` | `seq <= 4096` |
| `att_out` | `(heads, head_dim)` | none |

So the score buffer already allows 4096; the KV cache is what stops us at 2048,
and it stops us by its *height*, because the sequence is laid out along rows and
the KV vector along columns.

Widths for the models here (`att_kv_width` = `kv_dim` at `KV_PACK=16`):

| model | kv_dim | positions per row if the sequence went along the width |
|---|---:|---|
| MiniMind3 | 384 | 5 (1920 of 2048 used) |
| Supra2-100M | 256 | 8 |
| Supra2-Medium | 512 | 4 |

**Hybrid models (2026-11-11).** LFM2 has conv layers that never read a KV
cache, but the session used to allocate one for every layer.  At SEQ 2048 those
8 dead entries were 33.5 MB of CMA and the board failed at SEQ 1280 with
`FBO incomplete`; `init_attention()` now takes the layer types and skips them,
which puts 2048 back within reach at `KV_PACK=16` (measured, see
`docs/performance.md`).  `KV_PACK=8` fits too but costs fused PPL 126.89 ->
194.07, so it is not needed for context any more.

## 2. Why "just split the rows" is not a one-liner

Splitting the cache into N textures of 2048 rows is host-side for the **write**
path: `kv_rope_write` already draws one row at `(0, pos, kv_width, 1)`, so
binding `segment[pos / 2048]` and using `y = pos % 2048` needs no shader change.

The **read** path is where it costs, for two reasons.

1. *Samplers cannot be indexed.* ESSL 1.00 requires `texture2D` on a uniform
   sampler, so a shader cannot pick "cache texture number k" from a computed
   position.  With N a compile-time constant that is fine — N explicit sampler
   uniforms, no arrays — but the kernel has to be written for a fixed N.
2. *The accumulations cannot go through the blender.* The score pass writes
   packed fixed point (`x_i24`-style, 4 bytes carrying one 24-bit value), and
   `glBlendFunc(GL_ONE, GL_ONE)` adds per 8-bit channel without carrying.  Two
   segments added that way are not the sum of two 24-bit numbers.

So Na = 2 segments needs: one score draw per segment into its own target, then a
sum pass over the two (a shader with two samplers), then the existing softmax;
and the same shape for the value pass.  That is the honest cost of "split the
rows": **N× the attention draws plus a reduce pass**, not a config flag.

## 3. The alternative: tile the sequence along the width

A texture is 2048×2048, so laying the sequence across columns instead of rows
gives `positions_per_row = 2048 / kv_texels_per_position` rows of positions:

| model | positions per row | capacity in one texture |
|---|---:|---:|
| Supra2-100M | 8 | 16384 |
| MiniMind3 | 5 | 10240 |
| Supra2-Medium | 4 | 8192 |

One texture, one draw, no reduce pass — but every KV consumer changes its
address arithmetic: the rope write, the score pass, the value pass, the prefix
cache dump and its loader.  This is the design that scales; it is also the one
that touches the attention path everywhere.

## 4. What the models can actually use

| model | `max_position_embeddings` | trained context |
|---|---:|---|
| Supra2-100M | 2048 | 1024 (card) |
| Supra2-Medium | 1024 | 1024 (card) |
| MiniMind3 | 32768 | 768 (repo notes) |

Two of the three models here were trained at 1024, so a container that accepts
4096 buys **room for a prompt**, not better answers: past the training length
the model degrades regardless of what the engine can address.

## 5. Cost at runtime

Attention work per generated token is linear in the context, so the decode cost
of doubling the window is roughly the attention share of a token:

| context | attention share of one token (Supra2-Medium, 7 layers) |
|---:|---:|
| 1024 | ~10% |
| 2048 | ~18% |
| 4096 | ~30% |

(estimate from the per-layer shape: `heads × head_dim × context` against the
3.1M-MAC weight stack; the engine's own `LAYER_PROFILE` can measure it.)

So 4096 is affordable — the model's own limits arrive first.

## 6. Recommendation

1. **Do nothing until a container needs it.** Everything here runs at 1024 and
   the engine's `SEQ` defaults are 256-768.
2. When a long prompt is the goal, implement **N-segment row splitting for
   N = 2** first (4096): it reuses every existing kernel, adds one reduce pass,
   and needs no address-arithmetic change anywhere.  The cost is a compile-time
   `N` and two extra draws per layer per token.
3. Only if 8k+ is really needed, do the width tiling — it is the design that
   scales, and it is also the one that rewrites the attention addressing.

Nothing in this file is implemented yet; it is the map, not the road.
