#!/bin/sh
# Rebuild every MiniMind3 artifact from the F16 safetensors source, ending with
# the MMCF containers the engine actually loads.
#
#   i16   <- safetensors   (make_i16.py, --seq 256)
#   q8    <- i16           (make_q8_from_i16.py; q8 is terminal, never a q4 source)
#   q4    <- safetensors   (make_q4.py, --seq 512; quantize from F16, not q8)
#   tok   <- tokenizer.json
#   .mmcf <- the three above (mmcf_pack.py --verify, byte-for-byte self check)
set -eu
cd "$(dirname "$0")"
MODELS=../models/minimind3
PACK=../engine/tools/mmcf_pack.py
if [ ! -f tokenizer.json ]; then
  echo "tokenizer.json is not tracked; place the MiniMind3 tokenizer source in minimind3/ first" >&2
  exit 1
fi
python3 ../engine/quantize/make_i16.py --kind minimind --input "$MODELS" --output "$MODELS/minimind3.i16" --seq 256
python3 ../engine/quantize/make_q8_from_i16.py "$MODELS/minimind3.i16" "$MODELS/minimind3.q8"
python3 ../engine/quantize/make_q4.py --input "$MODELS" --output "$MODELS/minimind3.q4" --seq 512
python3 make_tokenizer.py --input tokenizer.json --output "$MODELS/minimind3.tok"

# The engine reads only MMCF, so without this step the artifacts above are not
# loadable.  --verify re-reads the file and compares every tensor with the
# source bytes: a conversion that changes one nibble is a bug, not a re-baseline.
for p in i16 q8 q4; do
  python3 "$PACK" --in "$MODELS/minimind3.$p" --tok "$MODELS/minimind3.tok" \
    --out "$MODELS/minimind3.$p.mmcf" --name minimind3 --template minimind3 \
    --system "You are a helpful assistant." --chat --verify
done
