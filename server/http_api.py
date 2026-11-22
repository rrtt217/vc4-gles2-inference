"""The HTTP surface: the OpenAI routes, the SSE twins and the diagnostics.

One `BaseHTTPRequestHandler` subclass, exactly as before the split -- `/v1/models`,
`/v1/chat/completions`, `/v1/completions`, `/props`, `/health`, `/v1/math`,
`/v1/tool`, the SSE writers, the access log and the error shapes.  It reaches all
of its state through `self.server` (`worker`, `model_id`, `stats_lock`), which is
what makes it movable: the server object is built in `server.py`'s `main()`.

Split out of `server.py` (2026-11-14, P2-2 of the 2026-11-13 audit).
"""

from __future__ import annotations

import functools
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import localtools as local_tools

from chat_render import (DEFAULT_SYSTEM, build_messages, parse_tool_calls,
                         prune_messages, request_chars)
from engine_bridge import LOGPROBS_KEY, ContextFull, WorkerError
from sampling import StopFilter, apply_stop_text, sampling_penalties, stop_strings, response_format
from tool_calls import build_tool_completion, deterministic_arguments, tool_choice_name

def _gauged(method):
    """Count POSTs in flight and remember how long recent ones took.

    `est_wait_ms` for a queued request is what people actually want when the
    engine is busy: the worker is single-flight, so N requests ahead of you cost
    N recent requests.  A real queue depth would need the engine's own lock, and
    this number is honest about being an estimate based on the recent past.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        srv = self.server
        lock = getattr(srv, "stats_lock", None)
        if lock is None:
            lock = srv.stats_lock = threading.Lock()
        if not hasattr(srv, "durations"):
            srv.durations = []
        with lock:
            srv.inflight = int(getattr(srv, "inflight", 0)) + 1
        t0 = time.time()
        try:
            return method(self, *args, **kwargs)
        finally:
            with lock:
                srv.inflight = max(0, int(getattr(srv, "inflight", 0)) - 1)
                srv.durations.append((time.time() - t0) * 1000.0)
                if len(srv.durations) > 64:
                    del srv.durations[:-64]
    return wrapper


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MiniMindOpenAI/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _access_log(self, cid: str, model: str, finish: str, usage: dict, stream: bool, **extra):
        """One line per finished request with its id.

        The base handler's line has only the path and the status, so a support
        question ("my request abc123 returned what?") had no way to find the
        request.  The id, the model, the finish reason and the token counts go
        here, and into the stats file when --stats-file is set.
        """
        rec = getattr(self, "_rec", None) or {}
        wall = time.time() - rec.get("ts", time.time())
        print(f"[req] {cid} model={model} stream={int(bool(stream))} finish={finish} "
              f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
              f"wall={wall:.2f}s", flush=True)
        self._stats(cid=cid, model=model, stream=bool(stream), finish=finish,
                    wall_s=round(wall, 3), usage=usage, **extra)

    def _stats(self, **fields):
        """Append one JSON line per request when --stats-file is set."""
        path = getattr(self.server, "stats_file", None)
        if not path:
            return
        rec = getattr(self, "_rec", None)
        if rec is None:
            rec = {"ts": time.time()}
        rec.update(fields)
        try:
            with self.server.stats_lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _select_model(self, name: str) -> None:
        """Map a request's `model` onto a configured container, loading it if the
        engine is holding a different one.

        One model is resident at a time: a Pi Zero's CMA fits one, and the whole
        point of the engine's load op is that the swap is cheaper than a restart.
        The cost is real (a container read plus an upload), so it happens on
        demand and is logged, not silently per request.
        """
        srv = self.server
        models = getattr(srv, "models", None)
        if not models:
            return                                  # single-model server
        chosen = None
        for m in models:
            if name in (m["id"], m["path"], Path(m["path"]).name.replace(".mmcf", "")):
                chosen = m
                break
        if chosen is None:
            raise ValueError("unknown model %r; this server serves %s"
                             % (name, ", ".join(m["id"] for m in models)))
        if chosen["path"] == getattr(srv, "active_path", None):
            return
        with getattr(srv, "model_lock", threading.Lock()):
            if chosen["path"] == getattr(srv, "active_path", None):
                return
            ms = srv.worker.load(chosen["path"])
            srv.active_path = chosen["path"]
            srv.active_id = chosen["id"]
            print(f"[model] {chosen['id']} resident, load {ms:.0f} ms", flush=True)

    def _est_wait_ms(self) -> int:
        """Rough wait for a request queued behind the ones already running.

        The worker handles one request at a time, so N requests in flight ahead of
        you cost N recent requests.  An estimate from the recent past, and it is
        named as one rather than dressed up as a measurement.
        """
        import statistics
        srv = self.server
        ahead = max(0, int(getattr(srv, "inflight", 0)) - 1)
        recent = list(getattr(srv, "durations", []))[-8:]
        if not ahead:
            return 0            # nothing queued in front of this request
        if not recent:
            return -1           # queued, but no completed request to extrapolate
        return int(statistics.median(recent) * ahead)

    def _last_request_ms(self) -> int:
        recent = list(getattr(self.server, "durations", []))[-1:]
        return int(recent[0]) if recent else 0

    def _worker_error(self, e, status: int = 500):
        """A 500 that carries the engine's stderr tail: the line that says why."""
        msg = str(e)
        try:
            tail = self.server.worker.stderr_tail()
        except Exception:
            tail = ""
        if tail:
            msg += "\n--- engine stderr tail ---\n" + tail
        self._error(msg, status, "server_error")

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message, status=400, typ="invalid_request_error"):
        self._stats(status=status, error=str(message)[:200])
        self._send_json({"error": {"message": message, "type": typ, "param": None, "code": None}}, status)

    def _handle_tool_post(self, path: str):
        """POST /v1/math (or /math), POST /v1/tool (or /tool) — never touch the model."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception as e:
            self._error(str(e), 400)
            return
        if path in {"/v1/math", "/math"}:
            expr = body.get("expression", body.get("expr", body.get("prompt", "")))
            name, args = "calculate_math", {"expression": expr}
        else:
            name = body.get("name") or body.get("tool") or ""
            args = body.get("arguments", body.get("args", {}))
        payload = local_tools.run_tool(name, args)
        payload["arguments"] = args if isinstance(args, dict) else {}
        self._send_json(payload, 200 if payload.get("ok") else 400)

    def _send_tool_stream(self, result: dict, include_usage: bool = False):
        """SSE twin of build_tool_completion: one content delta, then the call."""
        cid, created, model = result["id"], result["created"], result["model"]
        choice = result["choices"][0]
        msg = choice["message"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def chunk(obj):
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(("%X\r\n" % len(data)).encode("ascii") + data + b"\r\n")
            self.wfile.flush()

        def frame(delta, finish=None):
            chunk({"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        frame({"role": "assistant"})
        frame({"content": msg.get("content") or ""})
        calls = msg.get("tool_calls") or []
        if calls:
            frame({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
        frame({}, choice.get("finish_reason") or "stop")
        if include_usage:
            chunk({"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model, "choices": [], "usage": result.get("usage")})
        done = b"data: [DONE]\n\n"
        self.wfile.write(("%X\r\n" % len(done)).encode("ascii") + done + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        if path == "/health":
            worker = self.server.worker
            alive = bool(getattr(worker, "alive", True))
            self._send_json({
                "status": "ok" if alive else "degraded",
                "worker_alive": alive,
                "restarts": int(getattr(worker, "restarts", 0)),
                "prefix_cache_tokens": worker.prefix_cache_tokens,
                "template": getattr(worker, "template", "chatml"),
                "models": [m["id"] for m in (getattr(self.server, "models", None) or [])],
                "resident": getattr(self.server, "active_id", None),
                # Busy means the worker is single-flight; what a client wants to
                # know is the queue in front of it, not a percentage.
                "busy": int(getattr(self.server, "inflight", 0)) > 0,
                "inflight": int(getattr(self.server, "inflight", 0)),
                "est_wait_ms": self._est_wait_ms(),
                "last_request_ms": self._last_request_ms(),
                # What the engine last said about itself; the CMA number is what
                # decides whether another model fits on a Pi Zero.
                "engine": dict(getattr(worker, "engine_stats", {}) or {}),
            }, 200 if alive else 503)
        elif path == "/v1/models":
            now = int(time.time())
            models = getattr(self.server, "models", None) or [
                {"id": self.server.model_id}]
            self._send_json({
                "object": "list",
                "data": [{
                    "id": m["id"],
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                } for m in models],
            })
        elif path in {"/status", "/time", "/tools"}:
            # Deterministic routes: no model, no KV cache, sub-millisecond.
            if path == "/tools":
                self._send_json({
                    "object": "list",
                    "data": list(local_tools.TOOL_SPECS.values()),
                })
                return
            name = "system_status" if path == "/status" else "get_current_time"
            payload = local_tools.run_tool(name, {})
            payload["arguments"] = {}
            self._send_json(payload, 200 if payload.get("ok") else 500)
        elif path == "/math":
            from urllib.parse import parse_qs
            expr = (parse_qs(query).get("expr") or parse_qs(query).get("expression") or [""])[0]
            payload = local_tools.run_tool("calculate_math", {"expression": expr})
            payload["arguments"] = {"expression": expr}
            self._send_json(payload, 200 if payload.get("ok") else 400)
        else:
            self._error("not found", 404, "not_found")

    @_gauged
    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"/v1/math", "/math", "/v1/tool", "/tool"}:
            self._handle_tool_post(path)
            return
        if path != "/v1/chat/completions":
            self._error("not found", 404, "not_found")
            return
        self._rec = {"ts": time.time()}
        # The renderer follows the checkpoint (the worker reports its template
        # in the ready event), not a server flag and not a file name.
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            only_tool = getattr(self.server, "only_tool", None)
            if only_tool and isinstance(body.get("tools"), list) and body["tools"]:
                # "Keep one tool in the prompt whenever possible": a 64M model
                # picks better when the choice is not a choice.
                kept = [t for t in body["tools"]
                        if isinstance(t, dict)
                        and (t.get("function") or {}).get("name") == only_tool]
                body["tools"] = kept or body["tools"]
            # Which container serves this request: a multi-model server loads it
            # here, before anything reads the resident template.
            model = str(body.get("model") or getattr(self.server, "model_id", ""))
            self._select_model(model)
            # The engine is the only renderer.  A container packed without
            # --chat-template has nothing to render with, and guessing a format
            # here (this process used to fall back to ChatML) put markers in
            # front of the model that its training never saw.
            if not getattr(self.server.worker, "chat_template", False):
                raise ValueError(
                    "this container carries no chat template (metadata key 9), so "
                    "/v1/chat/completions has nothing to render it with; repack it "
                    "with mmcf_pack.py --chat-template <file>.jinja")
            req = build_messages(body)
            prompt = ""
            max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 64)) or 64)
            if max_tokens < 1:
                max_tokens = 1
            max_tokens = min(max_tokens, 256)
            temperature = float(body.get("temperature", 0.8) or 0.0)
            top_p = float(body.get("top_p", 0.9) or 0.9)
            top_k = int(body.get("top_k", 0) or 0)
            if top_k < 0:
                top_k = 0
            seed = int(body.get("seed", 1234) or 1234)
            stream = bool(body.get("stream", False))
            if int(body.get("n", 1) or 1) != 1:
                raise ValueError("only n=1 is supported")
            stops = stop_strings(body)
            penalties = sampling_penalties(body)
            resp_format = response_format(body)
            want_logprobs = bool(body.get("logprobs", False))
            if int(body.get("top_logprobs", 0) or 0) > 0:
                raise ValueError("top_logprobs is not implemented: logprobs=true returns "
                                 "the chosen token's logprob only")
            if want_logprobs and stream:
                raise ValueError("logprobs with stream=true is not implemented yet")
        except Exception as e:
            self._error(str(e), 400)
            return

        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        # --- deterministic tool routing (docs/future-optimization.md, P0-2) ---
        # A named tool that exists locally runs without the model: no prefill, no
        # KV cache, no chance of the 64M model inventing tool names or arguments.
        # `tool_choice: "required"` with exactly one local tool in the list is the
        # same situation -- there is nothing to choose, so do not ask.
        mode, tool_name = tool_choice_name(body)
        if not tool_name and mode == "required":
            offered = body.get("tools")
            if isinstance(offered, list) and len(offered) == 1:
                fn = (offered[0] or {}).get("function") if isinstance(offered[0], dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    tool_name = fn["name"]
        if tool_name:
            try:
                resolved = local_tools.resolve_tool_name(tool_name)
            except local_tools.ToolError:
                resolved = None
            if resolved is not None:
                args = deterministic_arguments(body, resolved)
                payload = local_tools.run_tool(resolved, args)
                payload["arguments"] = args
                result = build_tool_completion(cid, created, model, payload)
                if not stream:
                    self._send_json(result)
                    return
                self._send_tool_stream(
                    result,
                    include_usage=bool((body.get("stream_options") or {}).get("include_usage")))
                return

        # Streaming cannot retry once headers are out, so trim up front using a
        # conservative character estimate (~3 chars/token for mixed CJK + markup).
        seq = int(getattr(self.server, "seq", 0) or 0)
        if seq > 0:
            for _ in range(8):
                if request_chars(req) // 3 + max_tokens <= seq:
                    break
                trimmed = prune_messages(body.get("messages") or [])
                if trimmed is None:
                    break
                body["messages"] = trimmed
                req = build_messages(body)

        if not stream:
            # A long chat must not fail outright: if the engine says the prompt
            # does not fit, forget the oldest turns and try again.
            for _attempt in range(4):
                try:
                    text, finish, usage = self.server.worker.generate(
                        prompt, max_tokens, temperature, top_p, seed, penalties=penalties,
                        messages=req, top_k=top_k, request_id=cid,
                        logprobs=want_logprobs, response_format=resp_format)
                    break
                except ContextFull as e:
                    trimmed = prune_messages(body.get("messages") or [])
                    if trimmed is None:
                        self._error(f"prompt exceeds the context window: {e}", 400,
                                    "context_length_exceeded")
                        return
                    body["messages"] = trimmed
                    req = build_messages(body)
                except WorkerError as e:
                    self._worker_error(e)
                    return
            else:
                self._error("prompt still exceeds the context window after trimming history",
                            400, "context_length_exceeded")
                return
            if stops:
                text, stopped = apply_stop_text(text, stops)
                if stopped:
                    finish = "stop"
            rows = usage.pop(LOGPROBS_KEY, None)
            calls, content = parse_tool_calls(text)
            message = {"role": "assistant", "content": content or (None if calls else "")}
            if calls:
                message["tool_calls"] = calls
                finish = "tool_calls"
            self._access_log(cid, model, finish, usage, stream=False, status=200, tools=len(body.get("tools") or []))
            self._send_json({
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [dict({
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }, **({"logprobs": {"content": [
                    {"token": r.get("text", ""), "logprob": float(r.get("lp", 0.0))}
                    for r in rows if isinstance(r, dict)
                ]}} if (want_logprobs and rows is not None) else {}))],
                "usage": usage,
            })
            return

        # SSE / HTTP chunked.  Keep one worker request locked until done.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def chunk(obj):
            data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
            self.wfile.write(("%X\r\n" % len(data)).encode("ascii") + data + b"\r\n")
            self.wfile.flush()

        def send_delta(obj, finish=None):
            chunk({
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": obj,
                    "finish_reason": finish,
                }],
            })

        def delta(content, finish=None):
            send_delta({"content": content} if content is not None else {}, finish)

        try:
            delta("")
            if body.get("tools"):
                # Tool-call deltas are assembled only after the native
                # <tool_call> block is complete; do not stream partial JSON.
                text, finish, usage = self.server.worker.generate(
                    prompt, max_tokens, temperature, top_p, seed, penalties=penalties,
                    messages=req, top_k=top_k, request_id=cid, response_format=resp_format)
                if stops:
                    text, stopped = apply_stop_text(text, stops)
                    if stopped:
                        finish = "stop"
                calls, content = parse_tool_calls(text)
                if calls:
                    if content:
                        delta(content)
                    for i, c in enumerate(calls):
                        send_delta({"tool_calls": [dict(c, index=i)]})
                    finish = "tool_calls"
                else:
                    delta(content or "")
            else:
                stop_filter = StopFilter(stops, delta)
                text, finish, usage = self.server.worker.generate(
                    prompt, max_tokens, temperature, top_p, seed,
                    on_token=stop_filter.feed, penalties=penalties,
                    messages=req, top_k=top_k, request_id=cid, response_format=resp_format)
                stop_filter.finish()
                if stop_filter.hit:
                    finish = "stop"
            delta(None, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [], "usage": usage,
                })
            done = b"data: [DONE]\n\n"
            self.wfile.write(("%X\r\n" % len(done)).encode("ascii") + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            self._access_log(cid, model, finish, usage, stream=True, status=200, tools=len(body.get("tools") or []))
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream: tell the engine to stop rather than
            # generate tokens nobody will read.  Best effort -- the request may
            # have finished already, in which case the cancel is an idle ack.
            try:
                ok = self.server.worker.cancel(cid)
                print(f"[cancel] {cid} sent={int(bool(ok))} (client disconnected)", flush=True)
            except Exception as e:
                print(f"[cancel] {cid} failed: {e}", flush=True)
        except WorkerError:
            # Headers are already out; close the stream.
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass


