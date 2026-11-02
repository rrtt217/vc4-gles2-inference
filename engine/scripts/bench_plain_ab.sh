#!/bin/bash
# A/B two engine binaries in plain (non-chat) mode: decode ms/token and CPU/token.
#
#   sh scripts/bench_plain_ab.sh <binA> <binB> [gen] [reps]
#
# Plain mode is the canonical performance path (`--template ... --prompt ... --gen N`),
# so it does not touch --system or the chat template; its timer output is the
# `decode avg X ms/token` line, not the chat-mode `TIMING ...` line.
#
# CPU/token is a difference between a short and a long run, which cancels model
# load and prefill.  It is split into user and sys because they point at different
# fixes: user is the GL command stream built in-process, sys is the kernel side of
# every ioctl.  Runs alternate A/B so thermal drift and background load hit both.
set -eu
cd "$(dirname "$0")/.."

# Works from engine/ (../models) and from the runtime root (models).
if [ -d ../models/minimind3 ]; then _m=../models/minimind3; else _m=models/minimind3; fi
MODEL=${MODEL:-$_m/minimind3.q8}
export MODEL TOKENIZER=${TOKENIZER:-$_m/minimind3.tok}
export GPU_LAYER=${GPU_LAYER:-1} KV_PACK=${KV_PACK:-16} SEQ=${SEQ:-768}

A=${1:?usage: bench_plain_ab.sh binA binB [gen] [reps]}
B=${2:?}
GEN=${GEN:-32}
REPS=${REPS:-3}
LONG=${LONG_GEN:-128}
PROMPT=${PROMPT:-Once upon a time, there was a little}

decode_of() { # decode_of <bin> <gen>
    _d=$("$1" --template minimind3 --prompt "$PROMPT" --gen "$2" --greedy 2>&1 |
        sed -n 's/^decode avg \([0-9.]*\) ms\/token$/\1/p' | tail -1)
    if [ -z "$_d" ]; then
        echo "decode_of: no timing line from $1 (gen $2) — model path or run failed" >&2
        exit 2
    fi
    printf '%s' "$_d"
}

rusage_of() { # rusage_of <bin> <gen> -> "user sys"
    _f=$(mktemp)
    { TIMEFORMAT='%U %S'; time "$1" --template minimind3 --prompt "$PROMPT" \
        --gen "$2" --greedy >/dev/null ; } 2>"$_f"
    awk '{print $1, $2}' "$_f"
    rm -f "$_f"
}

summary() { # summary <label> <decodes...>
    _l=$1; shift
    printf '%s\n' "$@" | sort -n | awk -v tag="$_l" '
        { v[NR] = $1 }
        END {
            n = NR
            med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
            printf "%-4s decode min %7.1f  median %7.1f ms/token\n", tag, v[1], med
            printf "%s %s\n", v[1], med
        }'
}

LOAD=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD" 'BEGIN { if (l > 1.5) printf "warning: load %s — expect noise\n", l }' >&2

aD=""; bD=""
i=0
while [ "$i" -lt "$REPS" ]; do
    i=$((i + 1))
    aD="$aD $(decode_of "$A" "$GEN")"
    bD="$bD $(decode_of "$B" "$GEN")"
done

# shellcheck disable=SC2086
A_SUM=$(summary A $aD); A_MIN=$(printf '%s\n' "$A_SUM" | tail -1 | cut -d' ' -f1)
# shellcheck disable=SC2086
B_SUM=$(summary B $bD); B_MIN=$(printf '%s\n' "$B_SUM" | tail -1 | cut -d' ' -f1)
printf '%s\n%s\n' "$(printf '%s\n' "$A_SUM" | head -1)" "$(printf '%s\n' "$B_SUM" | head -1)"

cpu_line() { # cpu_line <label> <bin> <genmin> <genmax>
    _s=$(rusage_of "$2" "$3"); _l=$(rusage_of "$2" "$4")
    _su=$(printf '%s' "$_s" | cut -d' ' -f1); _ss=$(printf '%s' "$_s" | cut -d' ' -f2)
    _lu=$(printf '%s' "$_l" | cut -d' ' -f1); _ls=$(printf '%s' "$_l" | cut -d' ' -f2)
    awk -v tag="$1" -v a="$_su" -v b="$_ss" -v c="$_lu" -v d="$_ls" -v n=$(( $4 - $3 )) 'BEGIN {
        printf "%-4s cpu %6.1f ms/token (user %6.1f  sys %5.1f)\n", tag,
               ((c+d)-(a+b))*1000/n, (c-a)*1000/n, (d-b)*1000/n
    }'
}

cpu_line A "$A" "$GEN" "$LONG"
cpu_line B "$B" "$GEN" "$LONG"

awk -v a="$A_MIN" -v b="$B_MIN" 'BEGIN {
    if (a > 0) printf "B vs A: %+.1f%% decode\n", (b / a - 1) * 100
}'
