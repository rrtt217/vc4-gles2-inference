#!/bin/sh
# Build every MiniMind3 artifact from the F16 safetensors source.
#
# The packer quantizes directly from safetensors, so there is no intermediate
# .i16/.q8/.q4 file any more: the old engine/quantize converters wrote legacy
# layouts the engine cannot read, and are gone.
#
#   q4  <- safetensors, block-32 min/scale, embeddings and logits q8   (seq 512)
#   i16 <- safetensors, per-matrix int16                              (seq 256)
#   q8  <- int16 first, then q8, which is why it is not simply q8-of-float
#   tok <- tokenizer.json
set -eu
cd "$(dirname "$0")"
MODELS=../models/minimind3
PACK=../engine/tools/mmcf_pack.py
if [ ! -f tokenizer.json ]; then
  echo "tokenizer.json is not tracked; place the MiniMind3 tokenizer source in minimind3/ first" >&2
  exit 1
fi
python3 make_tokenizer.py --input tokenizer.json --output "$MODELS/minimind3.tok"

pack() {  # $1 quant, $2 seq
  python3 "$PACK" --in "$MODELS" --quant "$1" --seq "$2" \
    --tok "$MODELS/minimind3.tok" --out "$MODELS/minimind3.$1.mmcf" \
    --name minimind3 --template minimind3 \
    --system "You are a helpful assistant." --chat \
    --chat-template "$(pwd)/chat_template.jinja" --add-bos 0 --verify
}
pack q4 512
pack i16 256
pack q8 256
