#!/bin/sh
# Canonical regression: q8/q4 greedy text + fixed-corpus perplexity.
# Usage: scripts/regression.sh [engine_binary]
#
# The system prompt is pinned on the command line.  The compiled default
# changed in c1bb4f6 (Chinese persona -> neutral English), so a gate that
# omits --system silently changes meaning.  The script also checks that the
# run without --system prints the same text, i.e. that the compiled default has
# not moved; if it has, update kDefaultSystem, AGENTS.md, this script and
# docs/performance.md together.
#
# Env overrides:
#   SYS        system prompt to pin (default: You are a helpful assistant.)
#   Q8_EXPECT  expected q8 prefix (default: 我是一个AI助手)
#   Q4_EXPECT  expected q4 prefix (default: 我是MiniMind)
set -eu
cd "$(dirname "$0")/.."
BIN="${1:-${BIN:-./engine_mm3}}"
TOK="${TOKENIZER:-../models/minimind3/minimind3.tok}"
Q8="${Q8:-../models/minimind3/minimind3.q8}"
Q4="${Q4:-../models/minimind3/minimind3.q4}"
CORPUS="${PPL_FILE:-data/ppl_test.txt}"
PROMPT="用一句话介绍你自己。"
SYS="${SYS:-You are a helpful assistant.}"
Q8_EXPECT="${Q8_EXPECT:-我是一个AI助手}"
Q4_EXPECT="${Q4_EXPECT:-我是MiniMind}"

if [ ! -x "$BIN" ]; then echo "missing engine: $BIN" >&2; exit 1; fi
if [ ! -f "$CORPUS" ]; then echo "missing PPL corpus: $CORPUS" >&2; exit 1; fi

# $1 = model, $2 = "" (compiled default) or the text to pin.
text_of() {
    if [ -n "$2" ]; then
        MODEL="$1" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --template minimind3 --system "$2" \
            --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
            | sed -n 's/^Assistant: //p' | head -1
    else
        MODEL="$1" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --template minimind3 \
            --prompt "$PROMPT" --gen 16 --greedy 2>&1 \
            | sed -n 's/^Assistant: //p' | head -1
    fi
}

check() {
    tag="$1" want="$2" got="$3"
    case "$got" in
        "$want"*) echo "$tag OK: $got" ;;
        *) echo "$tag FAIL: got '$got', want prefix '$want'" >&2
           echo "  (system prompt: $SYS)" >&2
           exit 1 ;;
    esac
}

q8=$(text_of "$Q8" "$SYS")
check q8 "$Q8_EXPECT" "$q8"
q4=$(text_of "$Q4" "$SYS")
check q4 "$Q4_EXPECT" "$q4"

# Compiled default must agree with the pinned prompt.
q8_def=$(text_of "$Q8" "")
if [ "$q8_def" != "$q8" ]; then
    echo "system prompt drift: --system '$SYS' gives '$q8'" >&2
    echo "  but the compiled default gives '$q8_def'" >&2
    echo "  update kDefaultSystem, AGENTS.md, this script and docs/performance.md" >&2
    exit 1
fi
echo "default system prompt OK: same text without --system"

ppl() {
    MODEL="$Q8" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 ACT8="$1" \
        "$BIN" --template minimind3 --ppl "$CORPUS" 2>&1 | sed -n 's/^PPL /PPL /p'
}
echo "PPL baseline: $(ppl 0)"
echo "PPL ACT8=2:   $(ppl 2)"
echo "PPL ACT8=1:   $(ppl 1)"
