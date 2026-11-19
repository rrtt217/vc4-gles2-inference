#!/bin/sh
# The facts the measurement scripts kept re-deriving (P2-6 of the 2026-11-13 audit).
#
# Five scripts -- bench.sh, bench_ab.sh, bench_plain_ab.sh, longctx.sh,
# verify_packing.sh -- each worked out the tree root, the default model, the
# GPU_LAYER/KV_PACK/TIMING environment, the load-average guard, the "last TIMING
# line" parse and a min/median summary.  They agree today; they are also six facts
# in five copies, which is the shape that drifts silently (the dtype table had
# already drifted when its check was written).
#
# Source it, do not run it:
#
#     . "$(dirname "$0")/lib.sh"
#
# Every value is overridable from the environment, and every function keeps the
# behaviour the scripts had, because these numbers are cited in docs/benchmarks.md
# and an accidental change of protocol would invalidate every row in it.

# --- where we are -------------------------------------------------------
# $0 is the *caller* (scripts/*.sh), which is the documented way to run them:
# `bash engine/scripts/bench.sh` or `sh scripts/regression.sh`.
ENGINE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$ENGINE/.." && pwd)
cd "$ENGINE" || exit 2

# --- which binary, which model -----------------------------------------
BIN=${BIN:-./engine_mm3}
MODEL=${MODEL:-../models/minimind3/minimind3.q8.mmcf}
Q8=${Q8:-../models/minimind3/minimind3.q8.mmcf}
Q4=${Q4:-../models/minimind3/minimind3.q4.mmcf}

# The runtime directory on the board has models/ where the checkout has ../models;
# bench_plain_ab.sh learned that the hard way, so the fallback lives here now.
if [ ! -d ../models/minimind3 ] && [ -d models/minimind3 ]; then
    MODEL=${MODEL:-models/minimind3/minimind3.q8.mmcf}
    Q8=${Q8:-models/minimind3/minimind3.q8.mmcf}
    Q4=${Q4:-models/minimind3/minimind3.q4.mmcf}
fi

# --- the environment a measurement needs -------------------------------
# GPU_LAYER=1 and TIMING=1 are the protocol every number in docs/benchmarks.md
# was measured under; KV_PACK=16 is the default the gate pins.
require_bin() {
    [ -x "$1" ] || { echo "missing engine binary: $1" >&2; exit 2; }
}

measure_env() {
    export GPU_LAYER="${GPU_LAYER:-1}"
    export TIMING="${TIMING:-1}"
    export KV_PACK="${KV_PACK:-16}"
}

# `TOKENIZER` must not be exported when empty: the engine treats a *set* variable
# as an override and then fails with `cannot open tokenizer: ` (AGENTS.md documents
# this trap).  bench_ab.sh and bench_plain_ab.sh both exported it empty, so a plain
# `sh scripts/bench_ab.sh` could not run at all -- found while putting them on this
# library on 2026-11-14.
export_model() {
    export MODEL
    if [ -n "${TOKENIZER:-}" ]; then export TOKENIZER; else unset TOKENIZER; fi
    return 0
}

# --- small helpers the five scripts each carried -----------------------
cma_free() { awk '/^CmaFree:/{print $2}' /proc/meminfo; }

# The last TIMING line of a run, or empty.  The engine prints one per run.
timing_line() { grep -E '^TIMING' | tail -1; }

# "prefill=<s> decode_avg=<ms>" -> "<s> <ms>"; nothing else on the line matters.
timing_fields() {
    sed -e 's/.*prefill=\([0-9.]*\)s decode_avg=\([0-9.]*\)ms.*/\1 \2/'
}

# Plain (non-chat) mode prints a different line: "decode avg X ms/token".  That is
# the second timing format the audit counted, and the canonical performance path.
plain_decode() {
    sed -n 's/^decode avg \([0-9.]*\) ms\/token$/\1/p' | tail -1
}

# Warn (never block) when the machine is busy: a single run can swing 50% under
# load, which is why bench_ab.sh prints min and median instead of one number.
load_guard() {
    awk -v l="$(awk '{print $1}' /proc/loadavg)" 'BEGIN {
        if (l > 1.5) printf "warning: load average %s - expect noisy numbers\n", l
    }' >&2
}

# min/median of a bare list of numbers (one per line), for the plain-mode A/B.
min_median_list() {
    sort -n | awk -v tag="$1" '
        { v[NR] = $1 }
        END {
            n = NR
            if (!n) { printf "%-4s no runs\n", tag; exit 1 }
            med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
            printf "%-4s decode min %7.1f  median %7.1f ms/token\n", tag, v[1], med
            printf "%s %s\n", v[1], med
        }'
}

# min/median of the *second* column of a "prefill decode" file, plus the prefill
# that belongs to the minimum.  Prints the summary line, then the minimum on its
# own line so callers can compare two of them.
stats_min_median() {
    tag=$1; file=$2
    sort -k2 -n "$file" | awk -v tag="$tag" '
        { buf[NR] = $2; pre[NR] = $1 }
        END {
            n = NR
            if (!n) { printf "%-4s no runs\n", tag; exit 1 }
            med = (n % 2) ? buf[(n + 1) / 2] : (buf[n / 2] + buf[n / 2 + 1]) / 2
            printf "%-4s min %8.2f  median %8.2f  prefill(min) %6.2f s\n", tag, buf[1], med, pre[1]
            printf "%s\n", buf[1]
        }'
}
