#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the MiniMind3 GLES2 worker.

No third-party dependencies.  The C++ engine owns the GL context and KV cache;
this process owns HTTP/JSON/SSE and (later) tool execution.

Protocol to engine --worker, one request at a time:
  stdin : base64(prompt) \t max_tokens \t temperature \t top_p \t seed
  stdout: JSONL events: ready/start/token/done/error

The worker renders chat: this process sends the message list and never a prompt
string, and the engine renders it with the container's own Jinja template.  A
container packed without one cannot serve /v1/chat/completions (see the 400 in
the handler).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The server's own modules; `server.py` is the entry point and `main()` the only
# thing left in it (P2-2 of the 2026-11-13 audit split the rest out).
import localtools as local_tools  # noqa: E402  (deterministic tools, same directory)
from chat_render import DEFAULT_SYSTEM, build_messages  # noqa: E402,F401  (build_messages: selftest)
from engine_bridge import (WorkerError, ContextFull, EngineWorker,  # noqa: E402,F401
                           LOGPROBS_KEY)
from http_api import Handler  # noqa: E402  (the OpenAI routes)
# `selftest.py` imports these from here (`from server import StopFilter, ...`)
# because it tests them through the server package; keep the names visible.
from sampling import StopFilter, apply_stop_text  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible MiniMind3 server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--engine", default=str(repo / "engine/engine_mm3"))
    ap.add_argument("--model", action="append", default=None, metavar="MMCF",
                    help="Container to serve; repeat for more than one.  Each extra "
                         "model is advertised in /v1/models and loaded on demand, "
                         "because one container is resident at a time (the engine "
                         "keeps its GL context across the switch).")
    ap.add_argument("--tokenizer", default="",
                    help="Legacy .tok (empty = the container blob)")
    ap.add_argument("--model-id", action="append", default=None, metavar="ID",
                    help="Id for the matching --model, in order; defaults to the "
                         "container file name for every model after the first.")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--no-gpu-layer", action="store_true")
    ap.add_argument("--worker-log", default=str(Path(__file__).resolve().parent / "worker.log"))
    ap.add_argument("--prefix-cache", default=None)
    ap.add_argument("--kv-pack", type=int, choices=[0, 8, 16], default=0,
                    help="0=24-bit KV, 16=packed q16, 8=packed q8")
    ap.add_argument("--no-auto-restart", action="store_true",
                    help="Do not rebuild the engine worker after it exits")
    ap.add_argument("--max-restarts", type=int, default=5,
                    help="Give up after this many automatic worker restarts")
    ap.add_argument("--stats-file", default=None,
                    help="Append one JSON line per chat request (timings, tokens, "
                         "cache hits) to this file. Off by default.")
    ap.add_argument("--warmup", action="store_true",
                    help="Run one 1-token generation at startup with the cached "
                         "prefix text, so shader compilation does not land on the "
                         "first real request (~13 s on the Pi).")
    ap.add_argument("--only-tool", default=None,
                    help="Reduce the tool list in the prompt to this one tool name. "
                         "A 64M model chooses more reliably when there is no choice.")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.worker_log)), exist_ok=True)
    default_model = str(repo / "models/minimind3/minimind3.q8.mmcf")
    # `--model ID=PATH` is the spelling the eval batteries use, so accept it here
    # too; a plain path keeps its id from --model-id or from the file name.
    raw = list(a.model) if a.model else [default_model]
    inline_ids, paths = [], []
    for item in raw:
        if "=" in item and "/" not in item.split("=", 1)[0]:
            mid, path = item.split("=", 1)
            inline_ids.append(mid)
            paths.append(path)
        else:
            inline_ids.append(None)
            paths.append(item)
    ids = list(a.model_id) if a.model_id else []
    # An explicit ID=PATH wins for its position; --model-id fills the gaps in
    # order for the plain paths.
    given = [x for x in inline_ids if x]
    merged, gi = [], 0
    for i, x in enumerate(inline_ids):
        if x:
            merged.append(x)
        else:
            merged.append(ids[gi] if gi < len(ids) else "")
            gi += 1
    ids = [x for x in merged if x]
    if not ids and paths == [default_model]:
        ids = ["minimind3-64m"]          # keeps the historical default id
    while len(ids) < len(paths):
        ids.append(Path(paths[len(ids)]).name.replace(".mmcf", ""))
    models = [{"id": i, "path": p} for i, p in zip(ids, paths)]
    worker = EngineWorker(a.engine, models[0]["path"], a.tokenizer, a.seq,
                          gpu_layer=not a.no_gpu_layer, log_path=a.worker_log,
                          prefix_cache=a.prefix_cache, kv_pack=a.kv_pack,
                          auto_restart=not a.no_auto_restart, max_restarts=a.max_restarts)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.model_id = models[0]["id"]
    httpd.models = models
    httpd.active_path = models[0]["path"]
    httpd.active_id = models[0]["id"]
    httpd.model_lock = threading.Lock()
    httpd.worker = worker
    httpd.only_tool = a.only_tool
    httpd.seq = int(a.seq)
    httpd.stats_file = a.stats_file
    httpd.stats_lock = threading.Lock()
    httpd.inflight = 0
    httpd.durations = []

    if a.warmup:
        # The first generation after a process start pays shader compilation and
        # pipeline setup (~13 s on the Pi).  Do it here, with the prefix text the
        # cache was built from, so the KV chain still starts at the cached header.
        text = ""
        if a.prefix_cache:
            prompt_path = Path(str(a.prefix_cache) + ".prompt.txt")
            if prompt_path.exists():
                text = prompt_path.read_text(encoding="utf-8")
        if not text:
            # The warmup prompt has to use the same template as a real request,
            # or the KV chain starts from text the checkpoint never saw.
            if getattr(worker, "template", "chatml") == "tinybrainbot":
                text = "<|user|> Hello<|assistant|> "
            else:
                text = "<|im_start|>system\n" + DEFAULT_SYSTEM + "<|im_end|>\n"
        t0 = time.time()
        try:
            worker.generate(text, 1, 0.0, 0.9, 1234)
            print(f"warmup done in {time.time() - t0:.1f}s "
                  f"({len(text)} chars, prefix_cache_tokens={worker.prefix_cache_tokens})", flush=True)
        except WorkerError as e:
            print(f"warmup failed: {e}", flush=True)
    served = ", ".join(f"{m['id']} ({m['path']})" for m in models)
    print(f"listening on http://{a.host}:{a.port}  models=[{served}]"
          + ("" if len(models) == 1 else "  (one resident at a time, loaded on demand)"),
          flush=True)

    def stop(signum, frame):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        httpd.serve_forever()
    finally:
        worker.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
