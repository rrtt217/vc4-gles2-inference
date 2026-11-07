# Models

Central store for all model artifacts used by this repository.  The files
are large and are intentionally **not tracked by Git**; only this README is
tracked.

**The engine loads `.mmcf` only**, and the store keeps only what is
needed to build a container: a *source* (safetensors, a llama2.c `.bin`, a
`.gguf` reference, or one legacy checkpoint that has no other source) next to
the container(s) built from it.  Derived `.i16` / `.q8` / `.q4` checkpoints are
deleted once their container exists — they are byte-for-byte inside it, and
`mmcf_pack.py --quant` rebuilds them from the source in one step.

```text
models/
  minimind3/                     Qwen3-style 64M, the default chat model
    model.safetensors            source (F16)
    minimind-64M-3-F16.gguf      llama.cpp reference export
    config.json
    minimind3.{i16,q8,q4}.mmcf   what the engine loads
    minimind3.tok                tokenizer source, embedded by the packer

  llama2c/                       llama2.c TinyStories family + MiniMind2 Small
    stories15M.bin               source
    stories42M.bin               source
    TinyChat15M.bin              source; output is not coherent (format probe)
    stories260K.bin + tok512.bin 5-layer floor probe, 512-token vocabulary
    minimind2-small.i16          kept: the only local source for this model
    minimind2-small.{i16,q4,q8}.mmcf
    minimind2-tokenizer.bin      tokenizer source
    tokenizer.bin                the 32k llama2.c vocabulary
    stories15M-llama2-enzh.bin + tokenizer-llama2enzh.bin   experiment, no container

  tinybrainbot/                  TinyBrainBot-100M-v5.1-Instruct, template kind 3
    model.safetensors, config.json, llama/ (upstream reference code)
    convert_tinybrainbot.py, make_tok.py, ref_numpy.py
    tinybrainbot.tok
    tinybrainbot.{i16,q4}.mmcf   q8 is degenerate: use i16 (q4 is unreliable)

  supra2/                        Supra2-100M-Instruct (Qwen3-shaped): converts and
                                 tokenizes exactly, but the engine does not reproduce
                                 the float reference on it -- see the note below
  feng30m/                       feng-30m (Qwen3-shaped, 11 layers): works

  survey_under_110m.csv          candidate survey
```

`smollm2-135m/` and `smollm2-135m-instruct/` were **deleted**: SmolLM2 is
declared unsupported (see the note at the end of this file).

Rebuild MiniMind3 artifacts from the repository root:

```bash
# I16 -> Q8 -> Q4 -> .mmcf, plus the tokenizer
minimind3/convert.sh
```

It packs all three precisions from safetensors in one step each, so there is
nothing left over:

```bash
minimind3/convert.sh
```

Notes:

- `minimind2-small.q8` is a legacy Q8M1 header with **no rope fields**; packing
  it requires `--theta 1000000 --rope-half`, because the packer refuses to
  guess.  The container then states the rope, so nothing is passed at run time.
- `minimind2-small` is the one model whose local source is an I16 checkpoint
  rather than safetensors, so it is packed with `--in minimind2-small.i16`;
  q8 is never a source for q4.
- q8 is not usable for `tinybrainbot` (the break is in the q8 runtime path, not
  the weights); use q4 or i16.
- **SmolLM2-135M is not supported.**  The directories hold source weights and
  containers that were built outside this line; nothing here validates them,
  and the containers in them are not a claim that the model runs.  Its
  embedding needs a grid this engine does not implement, and its `GPU_LAYER=1`
  path is not correct on VC4.  Deleting or ignoring them is fine.
- `stories260K.bin` needs `tok512.bin` (not the 32k `tokenizer.bin`) when a
  legacy tool needs a tokenizer; the container embeds it.
- The llama2.c `.bin` sources are packed directly (`--quant i16`); small
  TinyStories models lose coherence in q8.

Model files, tokenizers, KV caches and logs are ignored by `.gitignore`.
The HuggingFace `tokenizer.json` source for MiniMind3 is also not tracked;
place it in `minimind3/tokenizer.json` if you need to rebuild `minimind3.tok`.

## supra2 status (2026-11-07)

Converted from `model.safetensors` with `templates/supra2.jinja` (the model's
own template, fetched from HuggingFace).  **Not supported**: it loads and
tokenizes exactly, but the engine's forward pass does not reproduce an
independent float32 forward over the same weights.

### Config audit

`config.json` (all of it), `generation_config.json` and
`tokenizer_config.json`, and what this stack does with each field:

| field | value | engine |
|---|---|---|
| `architectures` | `Qwen3ForCausalLM` | dense Qwen3 block, supported |
| `hidden_size` / `intermediate_size` | 768 / 2048 | `split_ffn()` returns early at `hidden <= 2048`, so no chunking — fine, but this is the first model at exactly the limit |
| `num_attention_heads` / `num_key_value_heads` | 12 / 4 | `GROUP = 3`, baked into the attention shader |
| `head_dim` | 64 | equals `dim/heads`, so the derived value agrees |
| `layer_types` / `sliding_window` / `use_sliding_window` | all full / null / false | no sliding window to model |
| `attention_bias` | false | no bias tensors |
| `hidden_act` | silu | SwiGLU gate |
| `tie_word_embeddings` | true | `wcls` = embedding |
| `vocab_size` | 32768 | 22 logits tiles, last one 512 rows |
| `rms_norm_eps` | 1e-06 | now metadata key 11 (`--rms-eps`), defaulting to the historical 1e-5; measured below to make **no** difference |
| `rope_parameters` | `{theta: 10000, rope_type: default}` | transformers 5 form: `rope_of()` used to read only top-level `rope_theta` and silently produced 10000/10000 — now handled |
| `bos_token_id` / `eos_token_id` / `pad` | 0 / 2 / 1 | container: bos -1 (chat templates add no BOS), eos 2, stop 6 (`<|im_end|>`), pad 1 |
| `generation_config` | eos 2, no sampling knobs | matches |
| `tokenizer_config.extra_special_tokens` | 24 markers | all present in the blob; `<|reserved_*|>` are unused |

Verified separately: the weights are the **Instruct** checkpoint (byte ranges at
offset 0 and 200 MB match `SupraLabs/Supra2-100M-Instruct` and differ from the
base repo), the tokenizer blob is id-exact against `tokenizer.json`
(32768 entries, ids 0-39 and the prompt's ids identical), and the template
renders identically under real Jinja2.

### What the engine does instead

Same 27 prompt ids, `GPU_LAYER=0`, greedy.  Engine logits vs the float32
reference: **cosine 0.977**, max |diff| 9.6.

```
reference top-5   I      You    Welcome   Please   My
engine    top-5   Introdu  Welcome  I     You    immer
```

`Introdu` is the user's own word — the model picks up the echo, which is what
"it repeats the question" looked like from the outside.

Ruled out by measurement, each one a single change:

| hypothesis | result |
|---|---|
| base instead of instruct weights | byte-range match says instruct |
| tokenizer / prompt mismatch | engine prompt ids == HF ids, 27/27 |
| template rendering | identical under Jinja2 |
| q16 activation grid | q16-simulated reference still says `I` |
| i16 weight quantization | i16-simulated reference still says `I` |
| FFN placement / chunking | `CPU_FFN=1`, `GPU_FFN=1` unchanged; `split_ffn()` no-ops at 2048 |
| prefill pair path | `PREFILL_PAIR=0` unchanged |
| GQA mapping `h % kv` | reference then says `s`, not `Introdu` |
| interleaved rope | reference then says `\n`, not `Introdu` |
| no qk-norm | reference then says `st`, not `Introdu` |
| `eps 1e-6` | reference still says `I` |
| `eps 1e-6` **in the engine** (key 11, repacked) | engine reports `rmsnorm eps: 1e-06 (container-stated)` and its logits are **bit-identical** to the 1e-5 run: the sum of squares is computed in fixed point, so a 1e-5-vs-1e-6 epsilon is below its resolution |
| the epsilon as a *uniform* instead of a literal | moves MiniMind3's PPL 270.3170 -> 269.9303 with the same numeric value, so the shader keeps the literal unless the container states an epsilon |

So the divergence is inside the engine's maths for this checkpoint shape, and
none of the config fields above explains it.

### Stage-level measurement: the q16 residual rails at layer 4 (2026-11-07)

`engine/tools/ref_stages.py` prints the same per-stage `DUMP` lines as
`STAGE_DUMP=1` from a float32 forward, so the engine can be compared against a
reference instead of against its own other path.  Prompt "hi" → token 22727,
`GPU_LAYER=0`, one position, 60 lines each, names aligned:

| layer | xb | att | x1 | xbf | hidA |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9998 | 0.9998 | 0.9998 | 0.9996 | 0.9998 |
| 1 | 0.9997 | 0.9998 | 0.9998 | 0.9997 | 0.9999 |
| 2 | 0.9998 | 0.9995 | 0.9999 | 0.9997 | 0.9999 |
| 3 | 0.9998 | 0.9997 | 0.9998 | 0.9998 | 0.99997 |
| 4 | 0.99996 | 0.9999 | 0.99997 | 0.99997 | **0.9197** |
| 5 | 0.669 | 0.593 | 0.918 | 0.837 | 0.919 |
| 11 | 0.398 | 0.265 | 0.761 | 0.880 | **-0.167** |

(cosine, engine vs float32).  The first stage that departs is **layer 4's
`hidA`, the residual add after the FFN**, and everything after it degrades.

**Cause: the model's residual stream does not fit the engine's q16 grid.**
Measured on the reference, max|x| per layer is

```
layer   0    1    2    3     4     5     6     7     8     9    10    11
max|x|  11   16   14   38  2314  2315  2315  2314  2313  2313  2309   174
```

and it is prompt-independent: tokens 22727, 6398 and 24932 all jump at layer 4
(2261-2314) and drop again at layer 11.  The engine's residual is fixed point at
1/256, so its rail is 32767/256 = **128**; the model needs **2300**, eighteen
times the rail, and saturates from layer 4 on.

Proved by construction rather than by inspection: running the *same* float32
forward with the engine's clamp applied to each residual add reproduces the
engine's wrong answer exactly,

```
f32 reference (no clamp)            -> I        You    Welcome
same model + q16 clamp, no scaling  -> Introdu  I      Welcome     <- the engine
+ per-layer residual scale 2^5      -> Welcome  I      You
```

`Introdu` is the user's own word, which is what "it repeats the question"
looked like from outside.  The clamped simulation puts it first; the f32
reference does not.

**What a fix needs.**  A single global factor does not work: it has to buy 18x
of range for the late layers without spending the early layers' resolution
(their values are ~0.05), and one q16 grid cannot span both.  A *per-layer*
power-of-two scale does work — the row above shows 2^5 from layer 4 on removes
the echo — and the engine already has the arithmetic for it
(`GPU::set_residual_scale`, `res_in_`/`res_out_`, `rescale_residual`), but
nothing at this commit calls it, and the CPU path's `add_q16` clamps
unconditionally.  Until that is wired, supra2 is out of range on the default
path and stays unsupported.
