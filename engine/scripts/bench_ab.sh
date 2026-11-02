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
cd "$(dirname "$0")/.."

TOKENIZER=${TOKENIZER:-../models/minimind3/minimind3.tok}
MODEL=${MODEL:-../models/minimind3/minimind3.q8}
export TOKENIZER MODEL GPU_LAYER=${GPU_LAYER:-1} KV_PACK=${KV_PACK:-16} TIMING=1

REF=${REF:-/tmp/engine_base}
NEW=${NEW:-./engine_mm3}
ROUNDS=${1:-3}
GEN=${GEN:-16}
PROMPT=${PROMPT:-用一句话介绍你自己。}

[ -x "$REF" ] || { echo "missing REF binary: $REF" >&2; exit 2; }
[ -x "$NEW" ] || { echo "missing NEW binary: $NEW" >&2; exit 2; }

LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN { if (l > 1.5) printf "warning: load average %s — expect noisy numbers\n", l }' >&2

one() {
    "$1" --chat --template minimind3 --prompt "$PROMPT" --gen "$GEN" --greedy 2>&1 \
        | grep '^TIMING' | tail -1 \
        | sed -e 's/.*prefill=\([0-9.]*\)s decode_avg=\([0-9.]*\)ms.*/\1 \2/'
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

summary() {
    sort -k2 -n "$2" | awk -v tag="$1" '
        { buf[NR] = $2; pre[NR] = $1; sum += $2 }
        END {
            n = NR
            med = (n % 2) ? buf[(n + 1) / 2] : (buf[n / 2] + buf[n / 2 + 1]) / 2
            printf "%-4s min %8.2f  median %8.2f  prefill(min) %6.2f s\n", tag, buf[1], med, pre[1]
            printf "%s\n", buf[1]
        }'
}

REF_SUM=$(summary ref "$REF_OUT"); REF_MIN=$(printf '%s\n' "$REF_SUM" | tail -1)
NEW_SUM=$(summary new "$NEW_OUT"); NEW_MIN=$(printf '%s\n' "$NEW_SUM" | tail -1)
printf '%s\n%s\n' "$(printf '%s\n' "$REF_SUM" | head -1)" "$(printf '%s\n' "$NEW_SUM" | head -1)"

awk -v a="$REF_MIN" -v b="$NEW_MIN" 'BEGIN {
    if (a > 0) printf "new vs ref: %+.1f%% decode (%.2f -> %.2f ms)\n", (b / a - 1) * 100, a, b
}'
