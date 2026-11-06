#!/bin/sh
# CPU-vs-GPU differential dump (item 12/13 in WORKLOG.md).
#
#   scripts/diff_paths.sh <model.mmcf> [prompt] [gen]
#
# Runs one container twice, GPU_LAYER=0 and GPU_LAYER=1, with STAGE_DUMP=1
# HIDDEN_DUMP=1 and PREFILL_PAIR=0 so both paths step every position through the
# same code, then compares the per-stage q16 rows with dump_diff.awk.  Stages, in
# order and per layer: xb (rms_att out), att (attention block out), x1 (after the
# wo residual), xbf (rms_ffn out), hidA (layer out).
#
# The CPU path is a float32 reference that round-trips q16 only at the dump, so a
# few LSB of difference are expected.  Measured caveat: on MiniMind3 q8, a model
# that answers correctly, the per-layer cosine between the paths falls to 0.46 by
# layer 7 - the residual stream saturates and the two numerics part company.  A
# large delta therefore names a lead, not a defect; a same-path A/B (one flag,
# one kernel) is what decides.
set -eu
cd "$(dirname "$0")/.."
BIN="${BIN:-./engine_mm3_pi}"
MODEL="${1:?usage: diff_paths.sh <model.mmcf> [prompt] [gen]}"
PROMPT="${2:-你}"
GEN="${3:-0}"
TOL="${TOL:-8}"
TMP="${TMPDIR:-/tmp}"
CPU="$TMP/diff_paths_cpu.txt"
GPU="$TMP/diff_paths_gpu.txt"
[ -x "$BIN" ] || { echo "missing engine: $BIN" >&2; exit 1; }
run() {
    if [ -n "${SEQ:-}" ]; then
        env MODEL="$MODEL" GPU_LAYER="$1" KV_PACK="${KV_PACK:-16}" SEQ="$SEQ" \
            PREFILL_PAIR=0 STAGE_DUMP=1 HIDDEN_DUMP=1 \
            "$BIN" --prompt "$PROMPT" --gen "$GEN" --greedy 2>&1 | grep "^DUMP" > "$2" || true
    else
        env MODEL="$MODEL" GPU_LAYER="$1" KV_PACK="${KV_PACK:-16}" \
            PREFILL_PAIR=0 STAGE_DUMP=1 HIDDEN_DUMP=1 \
            "$BIN" --prompt "$PROMPT" --gen "$GEN" --greedy 2>&1 | grep "^DUMP" > "$2" || true
    fi
}
run 0 "$CPU"
run 1 "$GPU"
echo "lines: cpu $(wc -l < "$CPU") - gpu $(wc -l < "$GPU")  (tolerance $TOL LSB)"
echo "dumps: $CPU $GPU"
awk -v tol="$TOL" -f "$(dirname "$0")/dump_diff.awk" "$CPU" "$GPU"
