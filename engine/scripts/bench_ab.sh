#!/bin/sh
# Interleaved A/B of two engine binaries on the same prompt, with a summary.
#
#   sh scripts/bench_ab.sh [rounds]
#   REF=./engine_mm3 NEW=./engine_mm3_pi sh scripts/bench_ab.sh 5
#
# Alternating the runs keeps thermal drift and background load from favouring
# whichever binary ran first.  The summary prints the minimum and the median per
# binary, because this board is noisy: it also runs a Minecraft server most of the
# time, and under load a single run can swing by 50%.  Minimum is the most stable
# statistic; the median tells you whether the run was clean.
#
# If the load average is above 1.5 the numbers are still comparable (both sides
# see the same interference) but they are not absolute.
set -eu
. "$(dirname "$0")/lib.sh"
measure_env
export_model

REF=${REF:-/tmp/engine_base}
NEW=${NEW:-./engine_mm3}
ROUNDS=${1:-3}
GEN=${GEN:-16}
PROMPT=${PROMPT:-用一句话介绍你自己。}

require_bin "$REF"
require_bin "$NEW"
load_guard

one() {
    "$1" --chat --prompt "$PROMPT" --gen "$GEN" --greedy 2>&1 \
      | timing_line | timing_fields
}

REF_OUT=$(mktemp)
NEW_OUT=$(mktemp)
trap 'rm -f "$REF_OUT" "$NEW_OUT"' EXIT

i=0
while [ "$i" -lt "$ROUNDS" ]; do
    i=$((i + 1))
    set -- $(one "$REF"); printf '%s %s\n' "$1" "$2" >> "$REF_OUT"
    printf 'ref  %8s s  %8s ms/token\n' "$1" "$2"
    set -- $(one "$NEW"); printf '%s %s\n' "$1" "$2" >> "$NEW_OUT"
    printf 'new  %8s s  %8s ms/token\n' "$1" "$2"
done

REF_SUM=$(stats_min_median ref "$REF_OUT"); REF_MIN=$(printf '%s\n' "$REF_SUM" | tail -1)
NEW_SUM=$(stats_min_median new "$NEW_OUT"); NEW_MIN=$(printf '%s\n' "$NEW_SUM" | tail -1)
printf '%s\n%s\n' "$(printf '%s\n' "$REF_SUM" | head -1)" "$(printf '%s\n' "$NEW_SUM" | head -1)"

awk -v a="$REF_MIN" -v b="$NEW_MIN" 'BEGIN {
    if (a > 0) printf "new vs ref: %+.1f%% decode (%.2f -> %.2f ms)\n", (b / a - 1) * 100, a, b
}'
