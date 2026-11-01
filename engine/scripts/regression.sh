#!/bin/sh
# Canonical regression: q8/q4 greedy text + fixed-corpus perplexity.
# Usage: scripts/regression.sh [engine_binary]
set -eu
cd "$(dirname "$0")/.."
BIN="${1:-${BIN:-./engine_mm3}}"
TOK="${TOKENIZER:-../models/minimind3/minimind3.tok}"
Q8="${Q8:-../models/minimind3/minimind3.q8}"
Q4="${Q4:-../models/minimind3/minimind3.q4}"
CORPUS="${PPL_FILE:-data/ppl_test.txt}"
PROMPT="用一句话介绍你自己。"

if [ ! -x "$BIN" ]; then echo "missing engine: $BIN" >&2; exit 1; fi
if [ ! -f "$CORPUS" ]; then echo "missing PPL corpus: $CORPUS" >&2; exit 1; fi

text_of() {
    MODEL="$1" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 \
        "$BIN" --chat --template minimind3 --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
        | sed -n 's/^Assistant: //p' | head -1
}

q8=$(text_of "$Q8")
case "$q8" in
    MiniMind*) echo "q8 OK: $q8" ;;
    *) echo "q8 FAIL: $q8" >&2; exit 1 ;;
esac

q4=$(text_of "$Q4")
case "$q4" in
    我是MiniMind*) echo "q4 OK: $q4" ;;
    *) echo "q4 FAIL: $q4" >&2; exit 1 ;;
esac

ppl() {
    MODEL="$Q8" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 ACT8="$1" \
        "$BIN" --template minimind3 --ppl "$CORPUS" 2>&1 | sed -n 's/^PPL /PPL /p'
}
echo "PPL baseline: $(ppl 0)"
echo "PPL ACT8=2:   $(ppl 2)"
echo "PPL ACT8=1:   $(ppl 1)"
