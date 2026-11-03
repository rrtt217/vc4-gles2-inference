#!/bin/sh
# Acceptance suite for the GPU engine: determinism, behaviour, golden strings
# (as a tripwire) and perplexity.
#
# Usage:
#   scripts/regression.sh [engine_binary]
#   scripts/regression.sh --rebaseline [engine_binary]
#
# Why not a single golden string: the greedy text is chaotic.  A deliberate
# reassociation (the fused FFN, for one) moves it on VC4 while the same tree
# still prints the old string on Xe, and a 64M model at temp 0.8 gives a
# different sample for the same weights.  So the suite checks in this order:
#
#   1. determinism   - the same binary, run twice, prints the same text.  A
#                      failure here is a real bug (uninitialised state, a race,
#                      a stale cache), never "noise".
#   2. behaviour     - a second prompt must produce a plausible answer: non
#                      empty, long enough, no collapse into one repeated chunk.
#   3. golden        - the identity answer is compared against Q8_EXPECT /
#                      Q4_EXPECT when those are set.  They are pinned per
#                      machine (Pi Zero and Xe differ); an empty value only
#                      prints the string.  A deliberate numeric change
#                      re-baselines them in the same commit, with PPL and the
#                      behaviour checks as evidence - see AGENTS.md.
#   4. ppl           - printed for baseline/ACT8=2/ACT8=1; compared against
#                      PPL_EXPECT (default 1% tolerance) when set.
#
# Env:
#   SYS          system prompt to pin (default: You are a helpful assistant.)
#   Q8_EXPECT    expected q8 prefix; empty = print only
#   Q4_EXPECT    expected q4 prefix; empty = print only
#   PPL_EXPECT   expected baseline PPL; empty = print only
#   PPL_TOL      relative tolerance for PPL_EXPECT (default 0.01)
#   PPL_FILE     corpus (default data/ppl_test.txt)
#   NO_PPL=1     skip the perplexity pass (it is the slow part)
set -eu
cd "$(dirname "$0")/.."

REBASE=0
if [ "${1:-}" = "--rebaseline" ]; then REBASE=1; shift; fi

BIN="${1:-${BIN:-./engine_mm3}}"
TOK="${TOKENIZER:-../models/minimind3/minimind3.tok}"
Q8="${Q8:-../models/minimind3/minimind3.q8}"
Q4="${Q4:-../models/minimind3/minimind3.q4}"
CORPUS="${PPL_FILE:-data/ppl_test.txt}"
SYS="${SYS:-You are a helpful assistant.}"
Q8_EXPECT="${Q8_EXPECT:-}"
Q4_EXPECT="${Q4_EXPECT:-}"
PPL_EXPECT="${PPL_EXPECT:-}"
PPL_TOL="${PPL_TOL:-0.01}"
PROMPT="用一句话介绍你自己。"
PROMPT2="写一首关于春天的短诗。"

[ -x "$BIN" ] || { echo "missing engine: $BIN" >&2; exit 1; }
if [ "${NO_PPL:-0}" != 1 ]; then
    [ -f "$CORPUS" ] || { echo "missing PPL corpus: $CORPUS" >&2; exit 1; }
fi

# $1 model, $2 prompt, $3 gen, $4 system ("" = compiled default)
answer() {
    if [ -n "$4" ]; then
        MODEL="$1" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --template minimind3 --system "$4" \
            --prompt "$2" --gen "$3" --greedy 2>&1
    else
        MODEL="$1" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --template minimind3 \
            --prompt "$2" --gen "$3" --greedy 2>&1
    fi
}

text() { answer "$@" | sed -n 's/^Assistant: //p' | head -1; }

check_prefix() {
    tag="$1" want="$2" got="$3"
    if [ -z "$want" ]; then
        echo "$tag (unpinned): $got"
        return 0
    fi
    case "$got" in
        "$want"*) echo "$tag OK: $got" ;;
        *) echo "$tag FAIL: got '$got', want prefix '$want'" >&2
           echo "  (system prompt: $SYS)" >&2
           exit 1 ;;
    esac
}

# Behaviour: non-empty, at least 6 characters, no 4-char window repeated three
# times in a row, and a sentence-ending mark or a hard stop.
behaviour() {
    tag="$1" got="$2"
    python3 - "$tag" "$got" <<'PY'
import sys
tag, s = sys.argv[1], sys.argv[2]
bad = []
if len(s) < 6:
    bad.append("len<6")
for i in range(len(s) - 11):
    w = s[i:i+4]
    if s[i:i+12] == w * 3:
        bad.append("repeated %r" % w)
        break
if s and not s.rstrip().endswith(("。", "！", "？", "…", "，", "、", "”", "’")):
    bad.append("no terminator")
if bad:
    print("%s FAIL: %s | %r" % (tag, ", ".join(bad), s), file=sys.stderr)
    sys.exit(1)
print("%s OK: %s" % (tag, s))
PY
}

q8=$(text "$Q8" "$PROMPT" 16 "$SYS")
q8b=$(text "$Q8" "$PROMPT" 16 "$SYS")
if [ "$q8" != "$q8b" ]; then
    echo "determinism FAIL: two runs of the same binary differ" >&2
    echo "  run 1: $q8" >&2
    echo "  run 2: $q8b" >&2
    exit 1
fi
echo "determinism OK"

q8_def=$(text "$Q8" "$PROMPT" 16 "")
[ "$q8_def" = "$q8" ] || {
    echo "system prompt drift: --system '$SYS' gives '$q8'" >&2
    echo "  but the compiled default gives '$q8_def'" >&2
    echo "  update kDefaultSystem, AGENTS.md, this script and docs/performance.md" >&2
    exit 1
}
echo "default system prompt OK"

check_prefix q8 "$Q8_EXPECT" "$q8"
q4=$(text "$Q4" "$PROMPT" 16 "$SYS")
check_prefix q4 "$Q4_EXPECT" "$q4"
poem=$(text "$Q8" "$PROMPT2" 48 "$SYS")
behaviour poem "$poem"

ppl() {
    MODEL="$Q8" TOKENIZER="$TOK" GPU_LAYER=1 KV_PACK=16 ACT8="$1" \
        "$BIN" --template minimind3 --ppl "$CORPUS" 2>&1 \
        | sed -n 's/^PPL \([0-9.]*\).*/\1/p' | head -1
}

p0=""
if [ "${NO_PPL:-0}" != 1 ]; then
    p0=$(ppl 0); p2=$(ppl 2); p1=$(ppl 1)
    echo "PPL baseline: $p0"
    echo "PPL ACT8=2:   $p2"
    echo "PPL ACT8=1:   $p1"
    if [ -n "$PPL_EXPECT" ]; then
        python3 - "$PPL_EXPECT" "$p0" "$PPL_TOL" <<'PY'
import sys
want, got, tol = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if abs(got - want) / want > tol:
    print("PPL FAIL: %.4f vs expected %.4f (>%.1f%%)" % (got, want, tol * 100), file=sys.stderr)
    sys.exit(1)
print("PPL OK: %.4f (expected %.4f)" % (got, want))
PY
    fi
fi

if [ "$REBASE" = 1 ]; then
    cat <<EOF

# --- copy into this machine's override (the values are machine specific) ---
Q8_EXPECT='$q8'
Q4_EXPECT='$q4'
PPL_EXPECT='${p0:-}'
EOF
fi

echo "ACCEPT"
