#!/usr/bin/env bash
# Long-context decode cost: does KV packing matter once the cache is deep?
# Usage: bash longctx.sh [repeats]
set -u
cd "$(dirname "$0")/.." || exit 1
REPEATS="${1:-45}"
export GPU_LAYER=1 TIMING=1
export SEQ="${SEQN:-768}"
MODEL=../models/minimind3/minimind3.q8.mmcf
FILLER=$(python3 -c "print('树莓派是一块很小的电脑板。' * $REPEATS + '只回答两个字：收到')")

for pack in 16 8 0; do
  out=$(MODEL=$MODEL KV_PACK=$pack timeout 1200 ./engine_mm3 --chat --template minimind3 \
        --prompt "$FILLER" --gen 8 --greedy 2>&1)
  line=$(printf '%s\n' "$out" | grep -E "^TIMING" | tail -1)
  printf 'kv_pack=%-2s %s\n' "$pack" "${line:-FAILED}"
done
