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
#   0. docs          - the documents still describe this tree: every switch the
#                      source reads is in docs/test-hooks.md, every in-tree path
#                      a document names exists, and no document calls a live
#                      switch dead.  scripts/docs_check.sh, no GPU needed; a
#                      drift here is a real defect (a stale doc sends the next
#                      reader at the wrong thing), so it fails the gate.
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
#   4. ppl           - printed for the q8 baseline/ACT8=2/ACT8=1 and for the
#                      q4 baseline; compared against PPL_EXPECT /
#                      Q4_PPL_EXPECT (default 1% tolerance) when set.
#
# Env:
#   SYS          system prompt to pin (default: You are a helpful assistant.)
#   Q8_EXPECT    expected q8 prefix; empty = print only
#   Q4_EXPECT    expected q4 prefix; empty = print only
#   PPL_EXPECT   expected baseline q8 PPL; empty = print only
#   Q4_PPL_EXPECT expected q4 PPL (same SEQ); empty = print only
#   MOE          a MoE container (v3) to check as well; empty = skip.  It gets
#                the determinism and behaviour checks and, when MOE_EXPECT is
#                set, the same prefix pin.  No PPL by default: the q4 MoE runs
#                at ~340 ms/token on a Pi Zero, so it is opt-in via MOE_PPL=1.
#   MOE_EXPECT   expected MoE prefix; empty = print only
#   MOE_PPL      compute the MoE container's PPL as well (=1; slow on the Pi)
#   MOE_PPL_EXPECT expected MoE PPL; empty = print only
#   PPL_TOL      relative tolerance for both (default 0.01)
#   PPL_FILE     corpus (default data/ppl_test.txt)
#   PPL_SEQ      SEQ used for the ppl runs (default 768; values are not
#                comparable across SEQ)
#   NO_PPL=1     skip the perplexity pass (it is the slow part)
set -eu
cd "$(dirname "$0")/.."

REBASE=0
if [ "${1:-}" = "--rebaseline" ]; then REBASE=1; shift; fi

BIN="${1:-${BIN:-./engine_mm3}}"
TOK="${TOKENIZER:-}"   # optional: the container embeds its tokenizer
if [ -n "$TOK" ]; then export TOKENIZER="$TOK"; fi
Q8="${Q8:-../models/minimind3/minimind3.q8.mmcf}"
Q4="${Q4:-../models/minimind3/minimind3.q4.mmcf}"
CORPUS="${PPL_FILE:-data/ppl_test.txt}"
SYS="${SYS:-You are a helpful assistant.}"
Q8_EXPECT="${Q8_EXPECT:-}"
Q4_EXPECT="${Q4_EXPECT:-}"
PPL_EXPECT="${PPL_EXPECT:-}"
Q4_PPL_EXPECT="${Q4_PPL_EXPECT:-}"
PPL_TOL="${PPL_TOL:-0.01}"
PPL_SEQ="${PPL_SEQ:-768}"
MOE="${MOE:-}"
MOE_EXPECT="${MOE_EXPECT:-}"
MOE_PPL="${MOE_PPL:-0}"
MOE_PPL_EXPECT="${MOE_PPL_EXPECT:-}"
PROMPT="用一句话介绍你自己。"
PROMPT2="写一首关于春天的短诗。"

[ -x "$BIN" ] || { echo "missing engine: $BIN" >&2; exit 1; }
if [ "${NO_PPL:-0}" != 1 ]; then
    [ -f "$CORPUS" ] || { echo "missing PPL corpus: $CORPUS" >&2; exit 1; }
fi

# $1 model, $2 prompt, $3 gen, $4 system ("" = compiled default)
answer() {
    if [ -n "$4" ]; then
        MODEL="$1" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --system "$4" \
            --prompt "$2" --gen "$3" --greedy 2>&1
    else
        MODEL="$1" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat \
            --prompt "$2" --gen "$3" --greedy 2>&1
    fi
}

# The engine appends its "[stopped at the --gen limit ...]" note to the same
# line, so a run that does not emit EOS would otherwise fail the terminator
# check; strip it before anything judges the text.
text() { answer "$@" | sed -n 's/^Assistant: //p' | head -1 | sed 's/\[stopped.*//'; }

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

# Documentation first: it costs a fraction of a second, needs no model, and a
# command that no longer runs is a bug in the document (AGENTS.md).
sh scripts/docs_check.sh || {
    echo "docs FAIL: run scripts/docs_check.sh for the list" >&2
    exit 1
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

# Mixture of experts: the route is a host-side argmax over a read-back, so the
# thing to gate is that it is *deterministic* and that the answer is a sentence.
# A route flip changes the whole text, which is why this is not compared with
# the dense string.
if [ -n "$MOE" ]; then
    m1=$(text "$MOE" "$PROMPT" 24 "$SYS")
    m2=$(text "$MOE" "$PROMPT" 24 "$SYS")
    if [ "$m1" != "$m2" ]; then
        echo "moe determinism FAIL: two runs of the same binary differ" >&2
        echo "  run 1: $m1" >&2
        echo "  run 2: $m2" >&2
        exit 1
    fi
    echo "moe determinism OK"
    check_prefix moe "$MOE_EXPECT" "$m1"
    behaviour moe "$m1"
else
    echo "moe: skipped (set MOE=<container> to check a v3 model)"
fi

# SEQ is pinned: the ppl path chunks the corpus per window, so the token count
# and the value both move with SEQ and are not comparable across settings.
# $1 model, $2 ACT8 mode.  ACT8 only rewrites the all-INT8 activation feed,
# so the q4 model is measured once, with ACT8=0.
ppl() {
    MODEL="$1" GPU_LAYER=1 KV_PACK=16 SEQ="$PPL_SEQ" ACT8="$2" \
        "$BIN" --ppl "$CORPUS" 2>&1 \
        | sed -n 's/^PPL \([0-9.]*\).*/\1/p' | head -1
}

check_ppl() {
    tag="$1" want="$2" got="$3"
    if [ -z "$want" ]; then echo "$tag (unpinned): $got"; return 0; fi
    python3 - "$tag" "$want" "$got" "$PPL_TOL" <<'PY'
import sys
tag, want, got, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if abs(got - want) / want > tol:
    print("%s FAIL: %.4f vs expected %.4f (>%.1f%%)" % (tag, got, want, tol * 100), file=sys.stderr)
    sys.exit(1)
print("%s OK: %.4f (expected %.4f)" % (tag, got, want))
PY
}

p0=""
q4p=""
if [ "${NO_PPL:-0}" != 1 ]; then
    p0=$(ppl "$Q8" 0); p2=$(ppl "$Q8" 2); p1=$(ppl "$Q8" 1)
    q4p=$(ppl "$Q4" 0)
    echo "PPL baseline: $p0"
    echo "PPL ACT8=2:   $p2"
    echo "PPL ACT8=1:   $p1"
    echo "PPL q4:       $q4p"
    check_ppl PPL "$PPL_EXPECT" "$p0"
    check_ppl "PPL q4" "$Q4_PPL_EXPECT" "$q4p"
    if [ -n "$MOE" ] && [ "$MOE_PPL" = 1 ]; then
        moep=$(ppl "$MOE" 0)
        echo "PPL moe:      $moep"
        check_ppl "PPL moe" "$MOE_PPL_EXPECT" "$moep"
    fi
fi

if [ "$REBASE" = 1 ]; then
    cat <<EOF

# --- copy into this machine's override (the values are machine specific) ---
Q8_EXPECT='$q8'
Q4_EXPECT='$q4'
PPL_EXPECT='${p0:-}'
Q4_PPL_EXPECT='${q4p:-}'
MOE_EXPECT='${m1:-}'
EOF
fi

echo "ACCEPT"
