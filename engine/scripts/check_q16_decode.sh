#!/bin/sh
# One rule, one place: every q16 decode in the emitted GLSL.
#
# The q16 decode is `u = hi*65280 + lo*255`, then the negative rail is folded back
# (`u >= T` subtracts 65536).  The rule lives in xformat.cpp, but its *text* is
# copied into every kernel that reads q16 by hand: the x_i16 family, the q4
# kernels' dA/dB, the KV decoders, the pair readers.  Ten spellings across three
# files, and two thresholds.
#
# The two thresholds are both correct and that is why this is a check and not a
# refactor: a decoded u is always a multiple of 255 (the bytes are 1/255 steps),
# so no value can land in [32767.5, 32768).  Changing the text would change the
# numerics by the letter of the rule in AGENTS.md ("the emitted text *is* the
# numerics"), so the text stays and this gate keeps the *rule* single: a decode
# site must carry one of the two known thresholds and the fold, and a new spelling
# fails here instead of on the board as a sign error.
#
# Usage: scripts/check_q16_decode.sh [dump-file]   (no dump = run --dump-shaders=raw)
set -u
cd "$(dirname "$0")/.." || exit 1

# $1 is the engine binary the gate is testing (or a saved dump file, which is how
# the negative tests are run).  On the board the binary is the runtime one, so a
# bare `./engine_mm3` is not good enough -- that was a board-only failure.
arg=${1:-}
if [ -n "$arg" ] && [ -x "$arg" ]; then
    dump=$("$arg" --dump-shaders=raw 2>/dev/null) || {
        echo "q16 decode: $arg --dump-shaders=raw failed" >&2; exit 1; }
elif [ -n "$arg" ] && [ -f "$arg" ]; then
    dump=$(cat "$arg")
else
    dump=$(./engine_mm3 --dump-shaders=raw 2>/dev/null) || {
        echo "q16 decode: cannot dump shaders" >&2; exit 1; }
fi

sites=$(printf '%s\n' "$dump" | grep -c '65280\.0')
[ "$sites" -gt 0 ] || { echo "FAIL  no q16 decode found in the dump"; exit 1; }

# Per line: as many rail folds as decodes.  Counting per *line* rather than per
# file is what makes the pair/quad decoders work (two values, two folds on one
# line); comparing the counts is what catches a decode whose fold was dropped.
# The q4 group scale is the one UNSIGNED reconstruction: `65280.0*15.0*ps` builds
# a positive scale constant (the 15 folds the n/15 texel into the group scale),
# so it carries no negative rail and needs no fold.  Strip that spelling before
# counting, or the 4x4 kernels (whose scale is unsigned by design) fail here.
report=$(printf '%s\n' "$dump" | awk '
  { line = $0
    gsub(/65280\.0\*15\.0/, "Q4USCALE", line)
    d = gsub(/65280\.0/, "&", line)
    f1c = gsub(/>=32767\.5/, "&", line); f2c = gsub(/>=32768\.0/, "&", line)
    f = f1c + f2c
    sites += d; f1 += f1c; f2 += f2c
    if (d > f) { bad = bad "  " d " decode(s), " f " fold(s): " substr($0, 1, 110) "\n" } }
  END { printf "%d %d %d", sites, f1, f2
        if (bad != "") { printf "\n%s", bad > "/dev/stderr"; exit 1 } }')
[ $? = 0 ] || { echo "FAIL  a q16 decode lost its rail fold" >&2; exit 1; }
sites=$(printf '%s\n' "$report" | cut -d' ' -f1)
t1=$(printf '%s\n' "$report" | cut -d' ' -f2)
t2=$(printf '%s\n' "$report" | cut -d' ' -f3)

odd=$(printf '%s\n' "$dump" | grep -o '>=[0-9][0-9]*\.[0-9]' | sort -u |
      grep -vE '^>=32767\.5$|^>=32768\.0$|^>=128\.0$' | head -3)
if [ -n "$odd" ]; then
    echo "FAIL  a decode uses a threshold that is not one of the documented ones:" >&2
    printf '%s\n' "$odd" | sed 's/^/      /' >&2
    exit 1
fi

echo "q16 decode: ok ($sites sites; ${t1}x 32767.5, ${t2}x 32768.0 -- equivalent at 255 steps)"
