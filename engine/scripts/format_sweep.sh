#!/bin/bash
# Sweep the format axes and require one answer.
#
#   scripts/format_sweep.sh <container> "<expected assistant text>"
#
# A kernel that hardcodes a format instead of following the axis is invisible
# until the axis moves: the BiCrossGate block's `bicross_act` stored a hand-picked
# q16 while `ACT` said bf16, and `mul2` wrote the activation stream where the
# FFN_HID stream belongs, so both answered garbage while the control model was
# fine.  Neither showed up in a single default run.
#
# What this asserts: every value of every stream axis, alone and in pairs,
# produces the *same* assistant text.  A format axis is a precision knob, not a
# behaviour knob, so one distinct answer over N runs is the pass condition -- and
# the number of distinct answers is the diagnostic when it fails.
#
# A refusal counts as a difference: an axis value a block cannot support must be
# refused by name at load (`exit 2` with a message), which the engine's conv and
# FFN_HID paths do, and a silent wrong answer is the bug this exists to catch.
set -u
MODEL=${1:-}
EXPECT=${2:-}
if [ -z "$MODEL" ] || [ -z "$EXPECT" ]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
# Absolute before any cd: the container is named relative to the caller's cwd,
# which on the board is the runtime directory, not the checkout.
case "$MODEL" in /*) ;; *) MODEL=$PWD/$MODEL ;; esac
# `ENGINE` from the environment is how the sweep runs on the board: the Pi has no
# checkout, only a runtime directory, and a kernel that VC4's register allocator
# refuses (the q8 row-pair qkv did) is a failure the laptop cannot show.
ENGINE=${ENGINE:-}
if [ -z "$ENGINE" ]; then
    cd "$(dirname "$0")/../.." || exit 2
    ENGINE=engine/engine_mm3
else
    cd "$(dirname "$ENGINE")" || exit 2
    ENGINE=./$(basename "$ENGINE")
fi
[ -x "$ENGINE" ] || { echo "format_sweep: no $ENGINE (build it first)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "format_sweep: no $MODEL" >&2; exit 2; }

WORK=$(mktemp -d) || exit 2
trap 'rm -rf "$WORK"' EXIT

# The axes and their closed sets, as `--dump-switches` prints them.
ACT="i16 bf16 f32"          # i8 is refused for a q4 qkv, so it is not a format here
RES="i16 i24 f32"
FFN="i16 bf16 f32"
PROJ="i24 f32"
KV="q16 bf16"

run() {   # run VAR=VAL... -> the assistant line, or a marker
    local line
    line=$(timeout 180 env GPU_LAYER=1 "$@" "$ENGINE" --model "$MODEL" --chat \
             --prompt "你好，你是谁？" --gen 24 --greedy 2>&1 \
           | grep -a "^Assistant" | sed 's/\[stopped.*//' | head -1)
    printf '%s' "${line:-<no answer>}"
}

n=0
: > "$WORK/out"
add() {   # add TAG VAR=VAL...
    local tag=$1; shift
    local got; got=$(run "$@")
    printf '%s\t%s\n' "$got" "$tag" >> "$WORK/out"
    n=$((n + 1))
}

# One list of KEY=VAL tokens: a bare value is an env command name, which is how
# the first version of this script ran `i16` and reported 30 missing answers.
one=""
for a in $ACT; do one="$one ACT=$a"; done
for a in $RES; do one="$one RES=$a"; done
for a in $FFN; do one="$one FFN_HID=$a"; done
for a in $PROJ; do one="$one PROJ=$a"; done
for a in $KV; do one="$one KV=$a"; done
for t in $one; do add "$t" "$t"; done
# Pairs are where the two mistakes above lived: one axis moved, the other's
# hardcoded format stayed behind.
for a in $ACT; do
    for b in $RES; do add "ACT=$a RES=$b" "ACT=$a" "RES=$b"; done
    for b in $FFN; do add "ACT=$a FFN_HID=$b" "ACT=$a" "FFN_HID=$b"; done
    for b in $PROJ; do add "ACT=$a PROJ=$b" "ACT=$a" "PROJ=$b"; done
    for b in $KV; do add "ACT=$a KV=$b" "ACT=$a" "KV=$b"; done
done

want=$(printf 'Assistant: %s' "$EXPECT")
distinct=$(cut -f1 "$WORK/out" | sort -u | wc -l | tr -d ' ')
bad=$(grep -vcF "$want" "$WORK/out" || true)

echo "format sweep: $n runs, $distinct distinct answer(s), $bad not the expected text"
if [ "$distinct" -ne 1 ] || [ "$bad" -ne 0 ]; then
    echo "FAIL  an axis moved the answer; the format is not the precision knob it claims to be" >&2
    cut -f1 "$WORK/out" | sort | uniq -c | sort -rn | sed 's/^/  /' >&2
    echo "  --- which runs:" >&2
    grep -vF "$want" "$WORK/out" | cut -f2 | sed 's/^/  /' >&2
    exit 1
fi
echo "  $want  (every axis value, alone and paired)"
exit 0