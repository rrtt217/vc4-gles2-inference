#!/usr/bin/env bash
# Benchmark sweep for the MiniMind3 engine on the Pi.
# Usage: bash bench.sh [prompt] [gen]
# Prints one line per config: tag, prefill seconds, decode ms/token, CmaFree delta.
set -u
cd "$(dirname "$0")" || exit 1
PROMPT="${1:-用一句话介绍你自己。}"
GEN="${2:-16}"
export TOKENIZER=../models/minimind3/minimind3.tok
export GPU_LAYER=1
export TIMING=1
Q8=../models/minimind3/minimind3.q8
Q4=../models/minimind3/minimind3.q4

cma_free() { awk '/^CmaFree:/{print $2}' /proc/meminfo; }

run() {
  local tag="$1" model="$2"; shift 2
  local before after out line
  before=$(cma_free)
  out=$(env MODEL="$model" "$@" timeout 300 ./engine_mm3 --chat --template minimind3 \
        --prompt "$PROMPT" --gen "$GEN" --greedy 2>&1)
  after=$(cma_free)
  line=$(printf '%s\n' "$out" | grep -E "^TIMING" | tail -1)
  local sample
  sample=$(printf '%s\n' "$out" | sed -n 's/^Assistant: //p' | head -1 | cut -c1-40)
  if [ -z "$line" ]; then
    printf '%-22s FAILED  %s\n' "$tag" "$(printf '%s' "$out" | tail -2 | tr '\n' ' ' | cut -c1-90)"
    return
  fi
  printf '%-22s %s  cma_free=%sKB  sample=%s\n' "$tag" "$line" "$after" "$sample"
}

run "q8 kv0"              "$Q8" KV_PACK=0
run "q8 kv8"              "$Q8" KV_PACK=8
run "q8 kv16"             "$Q8" KV_PACK=16
run "q8 kv16 gpu_ffn=0"   "$Q8" KV_PACK=16 GPU_FFN=0
run "q8 kv16 gpu_ffn=1"   "$Q8" KV_PACK=16 GPU_FFN=1
run "q8 kv16 cpu_ffn"     "$Q8" KV_PACK=16 CPU_FFN=1
run "q4 kv16"             "$Q4" KV_PACK=16
run "q4 kv8"              "$Q4" KV_PACK=8
