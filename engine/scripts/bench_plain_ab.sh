#!/bin/bash
# A/B two engine binaries in plain (non-chat) mode: decode ms/token and CPU/token.
#
#   sh scripts/bench_plain_ab.sh <binA> <binB> [gen] [reps]
#
# Plain mode is the canonical performance path (`--prompt ... --gen N`),
# so it does not touch --system or the chat template; its timer output is the
# `decode avg X ms/token` line, not the chat-mode `TIMING ...` line.
#
# CPU/token is a difference between a short and a long run, which cancels model
# load and prefill.  It is split into user and sys because they point at different
# fixes: user is the GL command stream built in-process, sys is the kernel side of
# every ioctl.  Runs alternate A/B so thermal drift and background load hit both.
set -eu
. "$(dirname "$0")/lib.sh"
measure_env
export_model
export SEQ=${SEQ:-768}

A=${1:?usage: bench_plain_ab.sh binA binB [gen] [reps]}
B=${2:?}
GEN=${GEN:-32}
REPS=${REPS:-3}
LONG=${LONG_GEN:-128}
PROMPT=${PROMPT:-Once upon a time, there was a little}

decode_of() { # decode_of <bin> <gen>
      _d=$("$1" --prompt "$PROMPT" --gen "$2" --greedy 2>&1 | plain_decode)
    if [ -z "$_d" ]; then
        echo "decode_of: no timing line from $1 (gen $2) — model path or run failed" >&2
        exit 2
    fi
    printf '%s' "$_d"
}

rusage_of() { # rusage_of <bin> <gen> -> "user sys"
    # The engine's own stderr lands in the same file as `time`'s report unless it is
    # discarded explicitly, and then awk reads "template: minimind3" as the numbers:
    # that is why CPU/token printed 0.0 for as long as this script existed.  Only
    # the "0.123 0.045" line is kept.
    _f=$(mktemp)
    { TIMEFORMAT='%U %S'; time "$1" --prompt "$PROMPT" \
        --gen "$2" --greedy >/dev/null 2>/dev/null ; } 2>"$_f"
    awk 'NF == 2 && $1 ~ /^[0-9.]+$/ && $2 ~ /^[0-9.]+$/ {print $1, $2}' "$_f" | tail -1
    rm -f "$_f"
}

load_guard

aD=""; bD=""
i=0
while [ "$i" -lt "$REPS" ]; do
    i=$((i + 1))
    aD="$aD $(decode_of "$A" "$GEN")"
    bD="$bD $(decode_of "$B" "$GEN")"
done

# shellcheck disable=SC2086
A_SUM=$(printf "%s\n" $aD | min_median_list A); A_MIN=$(printf '%s\n' "$A_SUM" | tail -1 | cut -d' ' -f1)
# shellcheck disable=SC2086
B_SUM=$(printf "%s\n" $bD | min_median_list B); B_MIN=$(printf '%s\n' "$B_SUM" | tail -1 | cut -d' ' -f1)
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
