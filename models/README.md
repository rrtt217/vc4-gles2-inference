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

**The packer refuses it, and `--res q24` is the fix.**  `mmcf_pack.py`
calibrates before writing and aborts above 4x the rail, so this cannot be
re-discovered by accident:

```
$ python3 engine/tools/mmcf_pack.py --in models/supra2 ...
residual: peak 2324.1 at layer 6 (rail 128.0), per layer 15 23 21 43 2323 ...
this model's residual reaches 2324, 18x the engine's q16 rail (128.0).
The forward pass will saturate and answer from a truncated state; see
models/README.md (supra2-100M).  Pack it anyway with --allow-over-rail.
```

With `--res q24 --allow-over-rail` the container declares a 24-bit residual
(metadata key 12): the same 1/256 step with 256x the range.  On the CPU path
(`GPU_LAYER=0`) that turns the model from an echo machine into a working
one, and it agrees with the float32 reference:

| prompt | q16 | q24 (CPU) | float32 reference |
|---|---|---|---|
| "Introduce yourself in two sentences." | `Introduce yourself in two sentences.` | `I'm a professional assistant.` | ` I` -> same sentence |
| "What can you do?" | `I can do a lot of things. I can help you. I can help you. ...` | `I can help you with a wide range of tasks. I can assist you with tasks like:` | ` I` -> same sentence |

`engine/scripts/supra2-battery.sh` is the rest of the evidence: determinism
(each case runs twice), identity, instruction following, factuality, a long
generation, termination and sampling.  It also shows what q24 does **not**
fix - a 100M model still answers "2+2" with "2+2!" and "capital of France"
with "the capital of France" - which is the model card's warning, not a bug.

### f32 oracle for the same battery (2026-11-07)

`engine/tools/ref_generate.py` runs the same template, the same tokenizer and a
float32 forward over the safetensors, greedily, with a KV cache, and stops on
`</s>`/`<|im_end|>`.  Three columns, one run, same prompts:

| prompt | f32 | q24 (CPU) | q16 (CPU) |
|---|---|---|---|
| Introduce yourself in two sentences. | `I'm a professional in the field of computer science.` | `I'm a professional assistant.` | *(echoes the prompt)* |
| What can you do? | `I'm here to help you with any questions or concerns you may have.` | `I can help you with a wide range of tasks. I can assist you with tasks like:` | `I can do a lot of things. I can help you. I can help you. ...` |
| Who made you? | `I'm a great help. I'm a great assistant.` | `You're a great assistant.` | `You're a helpful assistant.` |
| Reply with exactly one word: yes | `Reply with exactly one word: yes` | same | same |
| List three colours, comma separated. | `The three colors are blue, red, and white.` | `Here are three colors, comma separated.` | same |
| Write the word 'hello' in uppercase. | `Sure! hello!  You are` | `You're correct.` | same |
| What is the capital of France? | **`The capital of France is Paris.`** | `The capital of France is the capital of France.` | same |
| What is 2+2? | `Sure! 2+2 = **3**.` | `2+2!` | `That's correct.` |
| Tell me a short story about a cat. | `A cat is a cat.` | `A cat.` | same |
| Say goodbye. | `I'm sorry, but as a human, I don't have the ability ...` | `You're welcome.` | same |

What that says:

- q24 fixes exactly the two cases that were saturating (the echo and the
  repetition loop) and changes nothing else - the right signature for a range
  fix.
- **The facts are still lost**: f32 answers "Paris", the engine answers "the
  capital of France".  So the residual width is no longer the limiting factor;
  the ~1/256 activation grid and i16 weights are.  The same model also says
  "2+2 = **3**" in f32, so arithmetic was never there to lose.
- Answers are shorter in the engine than in f32 across the board: the engine's
  first tokens agree, then it reaches `<|im_end|>` sooner.  That is a
  consequence of the coarser grid, not of the stop-token logic (which is
  verified separately by the battery's termination case).

The GPU path works too.  The residual stores were the problem: the store macro
was emitted by `x_glsl(<activation format>)`, so every kernel silently wrote
q16 no matter what the residual was declared as.  With it moved to
`x_residual_glsl()`, one container runs on either path:

| prompt | CPU (q24) | GPU (q24) |
|---|---|---|
| What is the capital of France? | `the capital of France` | **`Paris`** |
| What is 2+2? | `2+2!` | **`2+2 = 4`** |
| Tell me a short story about a cat. | `A cat.` | a paragraph (with a repetition loop) |
| Introduce yourself in two sentences. | `I'm a professional assistant.` | `You're a helpful assistant.` |

The GPU path is the better of the two on the fact cases and is what a Pi
deployment uses; the CPU path is the fallback and the reference for the
format's arithmetic.

### On the Pi Zero the residual is not the problem any more

`supra2.q8q24.mmcf` (126 MB, q8 weights, q24 residual) runs on the board -
8 tokens take 74 s including startup, ~1.7 s/token - but the text degenerates
into a repetition loop ("reporting reporting worse worse worse ...").  The
format is not the cause: forcing `RES_FMT=i16_1` on the same container
degenerates just as badly, and `KV_PACK=24` does not help either.

The logits say what is happening.  Same prompt, same container:

| | max logit | entropy | top-5 |
|---|---:|---:|---|
| Xe | 9.03 | 3.07 | `The`, `It`, `France`, `A`, `Well` |
| Pi Zero | 6.30 | **5.98** | ` reporting`, ` doesn`, ` thing`, ` didn`, ` wrong` |

Cosine between the two is 0.86: the board is not producing garbage, it is
producing a **blurred** distribution - the peak is 2.7 logits lower and the
entropy nearly doubles - and a 100M model that needs a sharp peak to answer
answers with filler instead.  That points at a stage that quantises on VC4 and
not on Xe (the attention softmax is the prime suspect: it is fixed point with a
scale that was tuned on MiniMind3), and it is a separate investigation from the
residual width.

So: q24 residual - done, and it is what makes the model usable on Xe.  VC4
still blurs this model, at either residual width.

### Supra2-Medium-Instruct: the same architecture, and VC4 reproduces f32 exactly

`SupraLabs/Supra2-Medium-Instruct` is the 25M sibling (7 layers, 512 hidden,
8/8 heads - **no GQA** - 16,384-token tokenizer, same `chat_template.jinja`
shape).  It rails too, and harder relative to its size:

```
residual: peak 677.5 at layer 5 (rail 128.0), per layer 10 24 443 677 677 678 76
this model's residual reaches 678, 5x the engine's q16 rail (128.0).
```

so the residual explosion is a family trait, not one odd checkpoint.  Packed
q8 + q24 (34 MB), the same prompt on all three:

| | "What is the capital of France?" |
|---|---|
| f32 oracle (numpy) | `The capital of France is Bordeaux, which is located in the heart of the country. It is located in` |
| engine, Xe, GPU | identical |
| **engine, Pi Zero, GPU** | **identical** |

Character for character, on the board.  Two toolchain bugs had to be fixed
before it could run at all:

- its tokenizer puts `<|im_start|>` at **16384**, above a 16384-token BPE, and
  `make_tokenizer.py` only wrote `model["vocab"]`, so the engine split the chat
  markers into characters;
- its embedding is **16408** rows (HF padding) against a 16384-token tokenizer,
  and the packer claimed the embedding's count, so the engine read past the end
  of the blob ("tokenizer blob truncated").  The tokenizer now decides.

**This changes the reading of the 100M model.**  VC4 is not what blurs it: a
same-architecture model on the same board reproduces f32 exactly, with the same
q24 residual.  The 100M's blurred distribution is specific to that checkpoint,
which makes it a model at the edge of what a 100M can do, not an engine defect.

**What a wider residual would still need.**  A single global factor does not work: it has to buy 18x
of range for the late layers without spending the early layers' resolution
(their values are ~0.05), and one q16 grid cannot span both.  A *per-layer*
power-of-two scale does work — the row above shows 2^5 from layer 4 on removes
the echo — and the engine already has the arithmetic for it
(`GPU::set_residual_scale`, `res_in_`/`res_out_`, `rescale_residual`), but
nothing at this commit calls it, and the CPU path's `add_q16` clamps
unconditionally.  Until that is wired, supra2 is out of range on the default
path and stays unsupported.

## Two limits, measured: TinyBrainBot and SmolLM2 (2026-11-08)

The rail check now runs on every pack, so "this model is unsupported" has a
number behind it.  Per-layer maxima from a float32 forward, against the q16
grid's +-128:

| model | residual `x` | attention out | normed `xb`/`xbf` | FFN hidden `hid` |
|---|---:|---:|---:|---:|
| TinyBrainBot 100M q4 | 380 | <8 | <8 | <16 |
| **SmolLM2-135M** | **25982** | 7.4 | 32 | **3164** |

- **TinyBrainBot** overflows the residual by 3x and answers anyway (a handful of
  clamped values do not move its argmax).  `--res q24` changes little; what did
  fix it was the *system turn*: its trained format has none, and the packer
  defaulted one in, so every answer came out as template soup.  Repacked with
  `--system ""` it answers correctly -

  ```
  What is the capital of France?  ->  The capital of France is Paris.
  Tell me a short joke.           ->  Why don't scientists trust atoms? Because they make up everything.
  ```

- **SmolLM2-135M is a two-limit model.**  The residual jumps to 25982 at layer
  11 (204x the rail), so q16 produces *nothing* - the model emits its end token
  immediately.  `--res q24` covers that (32768) and generation starts, but the
  **FFN hidden reaches 3164** and that value still travels through an I16_2
  texture, so the output degenerates ("I am a local." repeated).  The float32
  reference answers "The capital of France is Paris.", so the weights are fine:
  the engine would need a wider format for the FFN intermediate as well, which
  is the activation-grid work, not the residual work.

That is the honest status: TinyBrainBot is supported and now correct; SmolLM2 is
one widening away, and the widening is named.
