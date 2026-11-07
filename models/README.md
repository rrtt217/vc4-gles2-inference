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
own template, fetched from HuggingFace — it is plain ChatML, **no** `<think>`
block; opening an empty think block makes this model stop on its first token).

What is verified:

- the weights in `models/supra2` **are** the Instruct checkpoint: a
  byte-for-byte range comparison against both HF repos matches
  `SupraLabs/Supra2-100M-Instruct` at offsets 0 and 200 MB, and differs from
  `SupraLabs/Supra2-100M` (the base repo) in both;
- the tokenizer is exact: the engine's prompt ids and `tokenizers.encode()` on
  the same rendered text are **identical, 27/27**;
- the model is English-only and 100M, so it is weak by design.

What is not: with those exact 27 ids the engine's first generated token is a
byte-fallback token, while an independent float32 forward pass over the same
safetensors predicts `'I'` (`I'm a professional assistant.`).  Feeding the
engine one extra leading space produces that sensible answer, which points at a
position/prefix-dependent discrepancy rather than broken weights.  Do not call
this model supported until that is resolved; the next step is a stage-by-stage
engine-vs-numpy diff (the `STAGE_DUMP` route) on this checkpoint alone.
