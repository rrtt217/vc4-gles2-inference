#!/bin/sh
# Recommended Pi Zero launch line: KV packing on, prefix cache if present,
# one warm-up generation so shader setup is not paid by the first user request.
#
#   sh server/run-pi.sh                # port 8014, q8 model
#   PORT=8000 sh server/run-pi.sh      # other port
#   ONLY_TOOL=system_status sh server/run-pi.sh
set -eu
cd "$(dirname "$0")"

PORT="${PORT:-8014}"
SEQ="${SEQ:-768}"
KV_PACK="${KV_PACK:-16}"
CACHE="${CACHE:-pfx_sys_status_768.kv}"
MODEL="${MODEL:-../models/minimind3/minimind3.q8.mmcf}"

STATS="${STATS:-stats.jsonl}"
set -- --port "$PORT" --seq "$SEQ" --kv-pack "$KV_PACK" --warmup --model "$MODEL" --stats-file "$STATS"
if [ -f "$CACHE" ]; then
    set -- "$@" --prefix-cache "$CACHE"
else
    echo "note: $CACHE missing; build it with:" >&2
    echo "  python3 server/build_prefix_cache.py build --tools system_status --out server/$CACHE" >&2
fi
if [ -n "${ONLY_TOOL:-}" ]; then
    set -- "$@" --only-tool "$ONLY_TOOL"
fi

echo "starting on port $PORT (seq=$SEQ kv_pack=$KV_PACK cache=$CACHE stats=$STATS)" >&2
exec python3 server.py "$@"
