# Models

Central store for all model artifacts used by this repository.  The files
are large and are intentionally **not tracked by Git**; only this README is
tracked.

**The engine loads `.mmcf` only.**  Every model here therefore exists twice:
the legacy file the packer consumes (`.i16`, `.q8`, `.q4`, `.bin`), and the
container that the engine runs.  Keep both — the legacy file is the re-pack
source, and a re-pack is how a numerics change is replayed.

Layout:

```text
models/
  minimind3/                     Qwen3-style 64M, the default chat model
    config.json                  source config used by the converters
    model.safetensors            original F16 source
    minimind-64M-3-F16.gguf      alternate source export (llama.cpp reference)
    minimind3.i16 / .q8 / .q4    legacy layouts (packer input)
    minimind3.{i16,q8,q4}.mmcf   what the engine loads
    minimind3.tok                engine tokenizer (embedded into the container)

  llama2c/                       llama2.c TinyStories family + MiniMind2 Small
    stories15M.bin / .i16 / .q8  (.q8 is gibberish: use i16)
    stories42M.bin / .i16 / .q8
    stories260K.bin + tok512.bin 5-layer floor probe, needs the 512-token vocab
    TinyChat15M.bin / .i16 / .q8 format probe; output is not coherent
    minimind2-small.i16 / .q8 / .q4
    tokenizer.bin                the 32k llama2.c vocabulary
    tokenizer-llama2enzh.bin     for stories15M-llama2-enzh.bin

  tinybrainbot/                  TinyBrainBot-100M-v5.1-Instruct, template kind 3
    model.safetensors, config.json, llama/ (upstream reference code)
    convert_tinybrainbot.py, make_tok.py, ref_numpy.py
    (q4 and i16 are the usable precisions; q8 degenerates — see
     docs/model-formats.md)

  smollm2-135m/                  SmolLM2-135M base — NOT SUPPORTED, source
  smollm2-135m-instruct/         weights only (see the note below)
    model.safetensors, tokenizer.json, vocab.json, merges.txt
    <name>.i16 / .q8 / .q4 + .mmcf, <name>.tok

  supra2/                        Supra2-100M-Instruct (Qwen3-shaped)
  feng30m/                       feng-30m (Llama-shaped), see README.txt

  survey_under_110m.csv          candidate survey: params, arch, downloads,
                                 engine_supported, note
```

Rebuild MiniMind3 artifacts from the repository root:

```bash
# I16 -> Q8 -> Q4 -> .mmcf, plus the tokenizer
minimind3/convert.sh
```

`convert.sh` stops at the legacy layouts; pack them afterwards, otherwise the
artifacts it produces are not loadable by the engine:

```bash
for p in i16 q8 q4; do
  python3 engine/tools/mmcf_pack.py --in models/minimind3/minimind3.$p \
    --tok models/minimind3/minimind3.tok \
    --out models/minimind3/minimind3.$p.mmcf \
    --name minimind3 --template minimind3 --chat --verify
done
```

Notes:

- `minimind2-small.q8` is a legacy Q8M1 header with **no rope fields**; packing
  it requires `--theta 1000000 --rope-half`, because the packer refuses to
  guess.  The container then states the rope, so nothing is passed at run time.
- `minimind2-small.q4` is generated directly from I16 with
  `engine/quantize/make_q4_from_i16.py`; q8 is never a source for q4.
- q8 is not usable for `tinybrainbot` (the break is in the q8 runtime path, not
  the weights); use q4 or i16.
- **SmolLM2-135M is not supported.**  The directories hold source weights and
  containers that were built outside this line; nothing here validates them,
  and the containers in them are not a claim that the model runs.  Its
  embedding needs a grid this engine does not implement, and its `GPU_LAYER=1`
  path is not correct on VC4.  Deleting or ignoring them is fine.
- `stories260K.bin` needs `tok512.bin` (not the 32k `tokenizer.bin`) when a
  legacy tool needs a tokenizer; the container embeds it.
- The llama2.c sources default to I16 (`make_i16.py --kind llama2c`).  Small
  TinyStories models lose coherence in q8.

Model files, tokenizers, KV caches and logs are ignored by `.gitignore`.
The HuggingFace `tokenizer.json` source for MiniMind3 is also not tracked;
place it in `minimind3/tokenizer.json` if you need to rebuild `minimind3.tok`.
