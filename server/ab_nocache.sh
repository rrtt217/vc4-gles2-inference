#!/usr/bin/env bash
# Cache A/B: same request against a cold server (no prefix cache).
# Usage: bash ab_nocache.sh <port> <outfile>
set -u
PORT="${1:-8013}"
OUT="${2:-/tmp/ab_nocache.txt}"
B="http://127.0.0.1:$PORT"

for i in $(seq 1 60); do
  if curl -s -m 3 "$B/health" > /dev/null 2>&1; then break; fi
  sleep 2
done

BODY=$(cat <<'JSON'
{"messages":[{"role":"user","content":"现在几点了？"}],
 "tools":[{"type":"function","function":{"name":"system_status","description":"Report hostname, uptime, load average, free memory, CPU temperature and GPU CMA free memory of this Raspberry Pi.","parameters":{"type":"object","properties":{},"required":[]}}}],
 "max_tokens":16,"temperature":0}
JSON
)

{
  echo "port=$PORT"
  curl -s -m 500 -w "\nwall=%{time_total}\n" -X POST "$B/v1/chat/completions" \
       -H "Content-Type: application/json" -d "$BODY"
} > "$OUT" 2>&1
echo "done -> $OUT"
