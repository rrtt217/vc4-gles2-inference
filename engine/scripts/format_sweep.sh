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
#
# The names carry an AX_ prefix on purpose.  They used to be ACT/RES/PROJ/KV, the
# switch names themselves, and a caller who exported one -- `KV=q16 script.sh`,
# which is the natural thing to do because the container wants it -- silently
# re-exported the *list* to every child the moment this line ran: the engine then
# read `switch KV=q16 bf16` and refused 35 of 43 runs while the four KV rows
# passed, because those override it on the command line.  A local name that
# shadows an exported one is not local.
AX_ACT="i16 bf16 f32"       # i8 is refused for a q4 qkv, so it is not a format here
AX_RES="i16 i24 f32"
AX_FFN="i16 bf16 f32"
AX_PROJ="i24 f32"
AX_KV="q16 bf16"

# A container can *require* a switch on a host -- MiniYuanYuan's q8 refuses to
# load on VC4 without QKV_PAIR=0 -- and that switch is not one of the axes, so it
# belongs to every run rather than to one.  EXTRA holds KEY=VAL tokens; inheriting
# them from the caller's environment instead is how the first board sweep ran 35
# of 43 axes into `program 'xkv' did not draw` while its four KV rows passed.
: "${EXTRA:=}"
run() {   # run VAR=VAL... -> the assistant line, or a marker that says why
    local raw line why
    raw=$(timeout 180 env GPU_LAYER=1 $EXTRA "$@" "$ENGINE" --model "$MODEL" --chat \
            --prompt "你好，你是谁？" --gen 24 --greedy 2>&1)
    line=$(printf '%s\n' "$raw" | grep -a "^Assistant" | sed 's/\[stopped.*//' | head -1)
    if [ -n "$line" ]; then printf '%s' "$line"; return; fi
    # "no answer" alone cost an hour on the board: the failures were the load, not
    # the axis, and the reason was on stderr that this line used to throw away.
    # Anything that is not a chat banner is the reason, last line first.
    why=$(printf '%s\n' "$raw" | grep -avE '^(template|chat template|tokenizer|specials|alias|formats|PREFIX_CACHE|checkpoint|model|rope|quant|name|caps|system|rms|residual|ffn block|seq|tokens|tokenizer:)' \
          | grep -av '^$' | tail -1)
    printf '%s (%s)' "<no answer>" "${why:0:160}"
}

n=0
: > "$WORK/out"
add() {   # add TAG VAR=VAL...
    local tag=$1; shift
    local got; got=$(run "$@")
    printf '%s\t%s\n' "$got" "$tag" >> "$WORK/out"
    n=$((n + 1))
    # Progress on stderr, one line a run.  A 43-run sweep on the board is ten
    # minutes of silence otherwise, and a silent long run is indistinguishable
    # from a hung one -- the summary still goes to stdout alone.
    printf '  [%2d] %s\n' "$n" "$tag" >&2
}

# One list of KEY=VAL tokens: a bare value is an env command name, which is how
# the first version of this script ran `i16` and reported 30 missing answers.
one=""
for a in $AX_ACT; do one="$one ACT=$a"; done
for a in $AX_RES; do one="$one RES=$a"; done
for a in $AX_FFN; do one="$one FFN_HID=$a"; done
for a in $AX_PROJ; do one="$one PROJ=$a"; done
for a in $AX_KV; do one="$one KV=$a"; done
for t in $one; do add "$t" "$t"; done
# Pairs are where the two mistakes above lived: one axis moved, the other's
# hardcoded format stayed behind.
for a in $AX_ACT; do
    for b in $AX_RES; do add "ACT=$a RES=$b" "ACT=$a" "RES=$b"; done
    for b in $AX_FFN; do add "ACT=$a FFN_HID=$b" "ACT=$a" "FFN_HID=$b"; done
    for b in $AX_PROJ; do add "ACT=$a PROJ=$b" "ACT=$a" "PROJ=$b"; done
    for b in $AX_KV; do add "ACT=$a KV=$b" "ACT=$a" "KV=$b"; done
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