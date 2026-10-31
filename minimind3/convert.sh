#!/bin/sh
set -eu
cd "$(dirname "$0")"
MODELS=../models/minimind3
if [ ! -f tokenizer.json ]; then
  echo "tokenizer.json is not tracked; place the MiniMind3 tokenizer source in minimind3/ first" >&2
  exit 1
fi
python3 ../engine/make_i16.py --kind minimind --input "$MODELS" --output "$MODELS/minimind3.i16" --seq 256
python3 ../engine/make_q8_from_i16.py "$MODELS/minimind3.i16" "$MODELS/minimind3.q8"
python3 ../engine/make_q4.py --input "$MODELS" --output "$MODELS/minimind3.q4" --seq 512
python3 make_tokenizer.py --input tokenizer.json --output "$MODELS/minimind3.tok"
