#!/bin/sh
# Interleaved A/B of two engine binaries on the same prompt.
# Usage: sh bench_ab.sh [rounds]
# Alternating the runs keeps thermal drift and background load from favouring
# whichever binary happened to run first.
set -eu
cd "$(dirname "$0")/.."
export TOKENIZER=../models/minimind3/minimind3.tok
export GPU_LAYER=1 KV_PACK=16 TIMING=1
export MODEL=../models/minimind3/minimind3.q8
REF="${REF:-/tmp/engine_base}"
NEW="${NEW:-./engine_mm3}"
ROUNDS="${1:-3}"
PROMPT="用一句话介绍你自己。"

one() {
    "$1" --chat --template minimind3 --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
        | grep '^TIMING' | tail -1 \
        | sed -e 's/.*prefill=\([0-9.]*\)s decode_avg=\([0-9.]*\)ms.*/\1 \2/'
}

i=0
while [ "$i" -lt "$ROUNDS" ]; do
    i=$((i + 1))
    set -- $(one "$REF");  printf 'ref  %s s  %s ms/token\n' "$1" "$2"
    set -- $(one "$NEW");  printf 'new  %s s  %s ms/token\n' "$1" "$2"
done
