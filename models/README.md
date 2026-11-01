# Models

Central store for all model artifacts used by this repository.  The files
are large and are intentionally **not tracked by Git**; only this README is
tracked.

Layout:

```text
models/
  minimind3/
    model.safetensors          # original F16 source
    config.json                # model config used by converters
    minimind-64M-3-F16.gguf    # alternate source export
    minimind3.i16              # I16M2 engine checkpoint
    minimind3.q8               # Q8M2 engine checkpoint
    minimind3.q4               # offline Q4M1 engine checkpoint
    minimind3.tok              # engine tokenizer
  llama2c/
    stories15M.bin/.q8/.i16
    stories42M.bin/.q8/.i16
    TinyChat15M.bin/.q8/.i16
    minimind2-small.q8/.i16
    stories15M-llama2-enzh.bin
    tokenizer.bin
    tokenizer-llama2enzh.bin
    stories260K.bin
```

Rebuild MiniMind3 artifacts from the repository root:

```bash
# I16 -> Q8 -> Q4 and tokenizer
minimind3/convert.sh

# or just Q4 from F16
python3 engine/quantize/make_q4.py --input models/minimind3 \
  --output models/minimind3/minimind3.q4 --seq 512
```

Model files, tokenizers, KV caches and logs are ignored by `.gitignore`.
The HuggingFace `tokenizer.json` source is also not tracked; place it in
`minimind3/tokenizer.json` if you need to rebuild `minimind3.tok`.
