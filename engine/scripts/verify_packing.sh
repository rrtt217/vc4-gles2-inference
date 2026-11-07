#!/bin/sh
# Correctness + speed check for the packed-activation layout.
#
# The reference is a binary built from the pre-packing tree (or any known
# good engine binary).  Set REF to point at it; otherwise the script only
# reports the new binary's output and timing.
# Both engines must produce the same token text byte for byte; a packing bug
# shows up as repeated single tokens, which still looks "fluent" at a glance.
#
# The model files are MMCF containers, so the template, tokenizer and rope come
# from the file; passing --template here would be silently ignored.
set -eu
cd "$(dirname "$0")/.."
export GPU_LAYER=1 KV_PACK=16 TIMING=1
REF="${REF:-/tmp/engine_base}"
PROMPT="用一句话介绍你自己。"

text_of() {
    MODEL="$1" "$2" --chat --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
        | sed -n 's/^Assistant: //p' | head -1
}

timing_of() {
    MODEL="$1" "$2" --chat --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
        | grep '^TIMING' | tail -1
}

for model in ../models/minimind3/minimind3.q8.mmcf ../models/minimind3/minimind3.q4.mmcf; do
    new=$(text_of "$model" ./engine_mm3)
    printf '%s\n  new: %s\n' "$model" "$new"
    if [ -x "$REF" ]; then
        base=$(text_of "$model" "$REF")
        printf '  ref: %s\n' "$base"
        [ "$new" = "$base" ] && echo "  MATCH" || { echo "  MISMATCH"; exit 1; }
    else
        echo "  (no $REF; rebuild the baseline to compare)"
    fi
    printf '  %s\n' "$(timing_of "$model" ./engine_mm3)"
    [ -x "$REF" ] && printf '  ref %s\n' "$(timing_of "$model" "$REF")"
done
