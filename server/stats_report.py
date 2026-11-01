#!/usr/bin/env python3
"""Summarise the per-request stats written by `server.py --stats-file`.

    python3 server/stats_report.py [path] [--hours N] [--by-model]

Reports cache-hit rate, prefill/decode split and wall-time percentiles.  The
point is to see whether the prefix cache is actually being hit in the wild — a
miss costs a full prefill, which on this board is tens of seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

DEFAULT = Path(__file__).with_name("stats.jsonl")


def pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DEFAULT))
    ap.add_argument("--hours", type=float, default=0.0, help="only the last N hours")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no stats at {path}; start the server with --stats-file {path}")
        return 1
    cutoff = time.time() - args.hours * 3600 if args.hours else 0.0

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ts", 0) < cutoff:
            continue
        rows.append(rec)

    if not rows:
        print("no requests in range")
        return 0

    ok = [r for r in rows if r.get("status") == 200]
    errs = [r for r in rows if r.get("status") != 200]
    walls = [r.get("wall_s", 0.0) for r in ok]
    prompt = [r.get("usage", {}).get("prompt_tokens", 0) or 0 for r in ok]
    prefill = [r.get("usage", {}).get("prefill_tokens", 0) or 0 for r in ok]
    cached = [max(0, p - f) for p, f in zip(prompt, prefill)]
    completion = [r.get("usage", {}).get("completion_tokens", 0) or 0 for r in ok]

    summary = {
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(errs),
        "wall_s_median": round(median(walls), 2) if walls else 0.0,
        "wall_s_p95": round(pct(walls, 95), 2),
        "prompt_tokens_median": median(prompt) if prompt else 0,
        "cached_tokens_median": median(cached) if cached else 0,
        "prefill_tokens_median": median(prefill) if prefill else 0,
        "completion_tokens_median": median(completion) if completion else 0,
        "cache_hit_rate": round(
            sum(c for c in cached) / sum(prompt), 3) if sum(prompt) else 0.0,
        "decode_tok_per_s": round(
            sum(completion) / max(1e-6, sum(walls) - 0.0), 2) if completion else 0.0,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"requests {summary['requests']}  ok {summary['ok']}  errors {summary['errors']}")
    print(f"wall time     median {summary['wall_s_median']}s   p95 {summary['wall_s_p95']}s")
    print(f"prompt tokens median {summary['prompt_tokens_median']:.0f}  "
          f"cached {summary['cached_tokens_median']:.0f}  "
          f"prefilled {summary['prefill_tokens_median']:.0f}")
    print(f"cache hit rate {summary['cache_hit_rate'] * 100:.1f}% of prompt tokens")
    print(f"throughput    {summary['decode_tok_per_s']:.2f} completion tokens/s (wall clock)")
    if errs:
        codes = {}
        for r in errs:
            codes[r.get("status")] = codes.get(r.get("status"), 0) + 1
        print("errors by status:", dict(sorted(codes.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
