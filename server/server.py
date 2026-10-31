#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the MiniMind3 GLES2 worker.

No third-party dependencies.  The C++ engine owns the GL context and KV cache;
this process owns HTTP/JSON/SSE and (later) tool execution.

Protocol to engine --worker, one request at a time:
  stdin : base64(prompt) \t max_tokens \t temperature \t top_p \t seed
  stdout: JSONL events: ready/start/token/done/error
"""
from __future__ import annotations

import argparse
from pathlib import Path
import base64
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_SYSTEM = (
    "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。"
    "请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。"
)


class WorkerError(RuntimeError):
    pass


class EngineWorker:
    def __init__(self, engine: str, model: str, tokenizer: str, seq: int,
                 gpu_layer: bool = True, log_path: str = "worker.log",
                 prefix_cache: str | None = None, kv_pack: int = 0):
        env = os.environ.copy()
        env["MODEL"] = model
        env["TOKENIZER"] = tokenizer
        env["SEQ"] = str(seq)
        env["GPU_LAYER"] = "1" if gpu_layer else "0"
        if kv_pack:
            env["KV_PACK"] = str(kv_pack)
        self.log = open(log_path, "ab", buffering=0)
        cmd = [engine, "--worker"]
        if prefix_cache:
            cmd += ["--prefix-cache", prefix_cache]
        self.prefix_cache_tokens = 0
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self.lock = threading.Lock()
        self.model = model
        if not self._wait_ready(180.0):
            self.close()
            raise WorkerError("engine worker did not become ready")

    def _wait_ready(self, timeout: float) -> bool:
        assert self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not r:
                if self.proc.poll() is not None:
                    return False
                continue
            line = self.proc.stdout.readline()
            if not line:
                return False
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "ready":
                self.prefix_cache_tokens = int(ev.get("prefix_cache_tokens", 0))
                return True
            if ev.get("event") == "error":
                raise WorkerError(ev.get("message", "worker startup error"))
        return False

    def generate(self, prompt: str, max_tokens: int, temperature: float,
                 top_p: float, seed: int, on_token=None):
        """Return (text, finish_reason, usage).  on_token(text) is optional."""
        if self.proc.poll() is not None:
            raise WorkerError("engine worker exited")
        req = "\t".join([
            base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
            str(max(0, int(max_tokens))),
            f"{float(temperature):.6g}",
            f"{float(top_p):.6g}",
            str(int(seed)),
        ]) + "\n"
        with self.lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
            text = ""
            finish = "length"
            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0
            prefill_tokens = 0
            callback_error = None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise WorkerError("engine worker closed stdout")
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as e:
                    raise WorkerError(f"bad worker event: {line!r}") from e
                kind = ev.get("event")
                if kind == "start":
                    prompt_tokens = int(ev.get("prompt_tokens", 0))
                    cached_tokens = int(ev.get("cached_tokens", 0))
                    prefill_tokens = int(ev.get("prefill_tokens", prompt_tokens))
                elif kind == "token":
                    piece = ev.get("text", "")
                    if piece:
                        text += piece
                        if on_token is not None and callback_error is None:
                            try:
                                on_token(piece)
                            except Exception as e:
                                # Keep draining the worker so the next request
                                # does not read this request's remaining events.
                                callback_error = e
                                on_token = None
                elif kind == "error":
                    raise WorkerError(ev.get("message", "worker error"))
                elif kind == "done":
                    text = ev.get("text", text)
                    finish = ev.get("finish_reason", "length")
                    prompt_tokens = int(ev.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(ev.get("completion_tokens", completion_tokens))
                    cached_tokens = int(ev.get("cached_tokens", cached_tokens))
                    prefill_tokens = int(ev.get("prefill_tokens", prefill_tokens))
                    if callback_error is not None:
                        raise callback_error
                    return text, finish, {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                        "prefill_tokens": prefill_tokens,
                    }

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        try:
            self.log.close()
        except Exception:
            pass


def content_text(msg) -> str:
    c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
        return "".join(out)
    return str(c)


def render_tools(tools) -> str:
    if not tools:
        return ""
    lines = []
    for t in tools:
        lines.append(json.dumps(t, ensure_ascii=False))
    return (
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        + "\n".join(lines) +
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    )


def parse_tool_calls(text: str):
    calls = []
    spans = []
    pos = 0
    while True:
        a = text.find("<tool_call>", pos)
        if a < 0:
            break
        b = text.find("</tool_call>", a)
        if b < 0:
            break
        body = text[a + len("<tool_call>"):b].strip()
        try:
            obj = json.loads(body)
        except Exception:
            pos = b + len("</tool_call>")
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(name, str) and name:
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
            spans.append((a, b + len("</tool_call>")))
        pos = b + len("</tool_call>")
    content = text
    for a, b in reversed(spans):
        content = content[:a] + content[b:]
    return calls, content.strip()


def build_prompt(body: dict) -> tuple[str, bool]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    tools = body.get("tools") or []
    system = DEFAULT_SYSTEM
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system = content_text(msg) or system
            break
    text = "<|im_start|>system\n" + system + render_tools(tools) + "<|im_end|>\n"
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            text += "<|im_start|>user\n" + content_text(msg) + "<|im_end|>\n"
        elif role == "assistant":
            text += "<|im_start|>assistant\n"
            if msg.get("reasoning_content"):
                text += "<think>\n" + str(msg.get("reasoning_content")) + "\n</think>\n\n"
            else:
                text += "<think>\n\n</think>\n\n"
            text += content_text(msg)
            calls = msg.get("tool_calls") or []
            for c in calls:
                fn = c.get("function", {}) if isinstance(c, dict) else {}
                args = fn.get("arguments", "{}")
                try:
                    args_obj = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    args_obj = {}
                text += "\n<tool_call>\n" + json.dumps(
                    {"name": fn.get("name", ""), "arguments": args_obj},
                    ensure_ascii=False,
                ) + "\n</tool_call>"
            text += "<|im_end|>\n"
        elif role == "tool":
            text += "<|im_start|>user\n<tool_response>\n" + content_text(msg) + "\n</tool_response><|im_end|>\n"
        else:
            text += "<|im_start|>user\n" + content_text(msg) + "<|im_end|>\n"
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = bool(kwargs.get("open_thinking", body.get("thinking", False)))
    text += "<|im_start|>assistant\n"
    text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return text, thinking


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MiniMindOpenAI/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message, status=400, typ="invalid_request_error"):
        self._send_json({"error": {"message": message, "type": typ, "param": None, "code": None}}, status)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send_json({"status": "ok", "prefix_cache_tokens": self.server.worker.prefix_cache_tokens})
        elif self.path.rstrip("/") == "/v1/models":
            now = int(time.time())
            self._send_json({
                "object": "list",
                "data": [{
                    "id": self.server.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }],
            })
        else:
            self._error("not found", 404, "not_found")

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._error("not found", 404, "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            prompt, thinking = build_prompt(body)
            max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 64)) or 64)
            if max_tokens < 1:
                max_tokens = 1
            max_tokens = min(max_tokens, 256)
            temperature = float(body.get("temperature", 0.8) or 0.0)
            top_p = float(body.get("top_p", 0.9) or 0.9)
            seed = int(body.get("seed", 1234) or 1234)
            stream = bool(body.get("stream", False))
            model = str(body.get("model") or self.server.model_id)
            if int(body.get("n", 1) or 1) != 1:
                raise ValueError("only n=1 is supported")
        except Exception as e:
            self._error(str(e), 400)
            return

        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        if not stream:
            try:
                text, finish, usage = self.server.worker.generate(
                    prompt, max_tokens, temperature, top_p, seed)
            except WorkerError as e:
                self._error(str(e), 500, "server_error")
                return
            calls, content = parse_tool_calls(text)
            message = {"role": "assistant", "content": content or (None if calls else "")}
            if calls:
                message["tool_calls"] = calls
                finish = "tool_calls"
            self._send_json({
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }],
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
                    prompt, max_tokens, temperature, top_p, seed)
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
                text, finish, usage = self.server.worker.generate(
                    prompt, max_tokens, temperature, top_p, seed,
                    on_token=lambda s: delta(s))
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
        except (BrokenPipeError, ConnectionResetError):
            pass
        except WorkerError:
            # Headers are already out; close the stream.
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible MiniMind3 server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--engine", default=str(repo / "engine/engine_mm3"))
    ap.add_argument("--model", default=str(repo / "models/minimind3/minimind3.q8"))
    ap.add_argument("--tokenizer", default=str(repo / "models/minimind3/minimind3.tok"))
    ap.add_argument("--model-id", default="minimind3-64m")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--no-gpu-layer", action="store_true")
    ap.add_argument("--worker-log", default=str(Path(__file__).resolve().parent / "worker.log"))
    ap.add_argument("--prefix-cache", default=None)
    ap.add_argument("--kv-pack", type=int, choices=[0, 8, 16], default=0,
                    help="0=24-bit KV, 16=packed q16, 8=packed q8")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.worker_log)), exist_ok=True)
    worker = EngineWorker(a.engine, a.model, a.tokenizer, a.seq,
                          gpu_layer=not a.no_gpu_layer, log_path=a.worker_log,
                          prefix_cache=a.prefix_cache, kv_pack=a.kv_pack)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.model_id = a.model_id
    httpd.worker = worker
    print(f"listening on http://{a.host}:{a.port}  model={a.model_id}", flush=True)

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
