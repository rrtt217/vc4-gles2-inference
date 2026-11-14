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
from pathlib import Path
import base64
import functools
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import localtools as local_tools  # noqa: E402  (deterministic tools, same directory)

DEFAULT_SYSTEM = "You are a helpful assistant."
# Carries the engine's per-token logprobs from generate() to the handler; removed
# from `usage` before it is sent, because a client must not see it.
LOGPROBS_KEY = "_logprobs"


class WorkerError(RuntimeError):
    pass


class ContextFull(WorkerError):
    """The prompt plus max_tokens does not fit the engine's context window."""


class EngineWorker:
    def __init__(self, engine: str, model: str, tokenizer: str, seq: int,
                 gpu_layer: bool = True, log_path: str = "worker.log",
                 prefix_cache: str | None = None, kv_pack: int = 0,
                 auto_restart: bool = True, max_restarts: int = 5):
        # Kept so a dead worker can be rebuilt without restarting the HTTP server.
        self._init_args = dict(engine=engine, model=model, tokenizer=tokenizer, seq=seq,
                               gpu_layer=gpu_layer, log_path=log_path,
                               prefix_cache=prefix_cache, kv_pack=kv_pack,
                               auto_restart=auto_restart, max_restarts=max_restarts)
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restarts = 0
        self._open(engine, model, tokenizer, seq, gpu_layer, log_path, prefix_cache, kv_pack)

    def _open(self, engine, model, tokenizer, seq, gpu_layer, log_path, prefix_cache, kv_pack):
        env = os.environ.copy()
        env["MODEL"] = model
        if tokenizer:
            # Optional: a legacy .tok beside the model.  A container carries its
            # own tokenizer blob, and setting this to a path that does not exist
            # makes the engine load that file instead of the embedded one.
            env["TOKENIZER"] = tokenizer
        env["SEQ"] = str(seq)
        env["GPU_LAYER"] = "1" if gpu_layer else "0"
        if kv_pack:
            env["KV_PACK"] = str(kv_pack)
        self._log_path = log_path
        self.log = open(log_path, "ab", buffering=0)
        # The container names its chat template (kind 3 TinyBrainBot, 2
        # MiniMind3, 1 chatml, 0 plain) and the worker reports the effective one
        # in its ready event; /health prints it, so a wrong template shows up
        # there instead of in the prompt.
        self.template = "chatml"   # pre-ready default
        # True when the container carries its own Jinja template (MMCF key 9).
        # The engine is the only renderer, so this decides whether
        # /v1/chat/completions can be served at all.
        self.chat_template = False
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
        # `generate` holds `lock` for the whole request, so a cancel from another
        # thread must not take it: this one guards the write only.
        self._write_lock = threading.Lock()
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
                # The checkpoint names its template; the server renders the
                # prompt, so this is the only place that choice is made.
                self.template = str(ev.get("template") or "chatml")
                self.chat_template = bool(ev.get("chat_template"))
                self._note_engine(ev)
                return True
            if ev.get("event") == "error":
                raise WorkerError(ev.get("message", "worker startup error"))
        return False

    def _note_engine(self, ev: dict) -> None:
        """Keep the last numbers the engine reported about itself.

        Only the engine can read the board's CMA, and asking it mid-request would
        queue behind a generation, so the values it sends with `ready`/`loaded`
        are cached here and `/health` reports those.
        """
        stats = dict(getattr(self, "engine_stats", {}) or {})
        for key in ("cma_free_kb", "mem_available_kb", "programs", "seq"):
            value = ev.get(key)
            if value is not None:          # a `done` event carries no seq
                stats[key] = value
        self.engine_stats = stats

    def stderr_tail(self, lines: int = 8) -> str:
        """The engine's last log lines: what a 500 needs to be actionable.

        The worker writes its diagnostics to a file, and "engine worker exited"
        on its own has cost real time more than once -- the line that says *why*
        (a CMA allocation failure, a GL error, an OOM) is in there.
        """
        try:
            with open(self._log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                text = f.read().decode("utf-8", "replace")
            tail = [l for l in text.strip().splitlines() if l.strip()]
            return "\n".join(tail[-lines:])
        except Exception:
            return ""

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def restart(self) -> bool:
        """Rebuild a crashed worker.  False when restarts are disabled/exhausted."""
        if not self.auto_restart or self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        try:
            self.close()
        except Exception:
            pass
        a = self._init_args
        self._open(a["engine"], a["model"], a["tokenizer"], a["seq"], a["gpu_layer"],
                   a["log_path"], a["prefix_cache"], a["kv_pack"])
        return True

    def ensure_alive(self) -> None:
        if self.alive:
            return
        if not self.restart():
            raise WorkerError("engine worker exited (auto-restart disabled or exhausted)")

    def load(self, model: str, timeout: float = 900.0) -> float:
        """Make `model` resident and return how long it took, in milliseconds.

        The engine keeps its EGL context and its compiled programs, so this is a
        container read plus an upload instead of a process start: 4.23 s of
        context and 1.89 s of program links are the part that survives (docs/
        performance.md, "Replacing a model without rebuilding the GL context").
        The template and the chat-template flag belong to the container, so they
        are replaced with it -- a server that kept the old ones would render the
        new model's prompt with the old model's markers.
        """
        with self.lock:
            self.ensure_alive()
            with self._write_lock:
                self.proc.stdin.write(json.dumps({"op": "load", "model": model}) + "\n")
                self.proc.stdin.flush()
            deadline = time.time() + timeout
            while time.time() < deadline:
                r, _, _ = select.select([self.proc.stdout], [], [], 1.0)
                if not r:
                    if self.proc.poll() is not None:
                        raise WorkerError("engine worker exited during load")
                    continue
                line = self.proc.stdout.readline()
                if not line:
                    raise WorkerError("engine worker closed its stream during load")
                ev = json.loads(line)
                kind = ev.get("event")
                if kind == "loaded":
                    self.model = model
                    self.template = ev.get("template", self.template)
                    self.chat_template = bool(ev.get("chat_template", self.chat_template))
                    self.prefix_cache_tokens = 0   # it belonged to the old model
                    self._note_engine(ev)
                    return float(ev.get("load_ms", 0.0))
                if kind == "error":
                    raise WorkerError(str(ev.get("message", "load failed")))
            raise WorkerError("timed out waiting for the engine to load " + model)

    def cancel(self, request_id: str = "") -> bool:
        """Ask the engine to stop the request in flight (best effort).

        The engine peeks at its stdin between tokens, so this takes effect within
        a few tokens rather than at the end of the generation.  False when there
        is no process to talk to.
        """
        try:
            with self._write_lock:
                if self.proc.poll() is not None or self.proc.stdin is None:
                    return False
                self.proc.stdin.write(json.dumps({"op": "cancel", "id": request_id}) + "\n")
                self.proc.stdin.flush()
            return True
        except Exception:
            return False

    def generate(self, prompt, max_tokens: int, temperature: float,
                 top_p: float, seed: int, on_token=None, penalties: dict | None = None,
                 messages: dict | None = None, top_k: int = 0, request_id: str = "",
                 logprobs: bool = False):
        """Return (text, finish_reason, usage).  on_token(text) is optional.

        ``prompt`` is either the rendered prompt string, or ``""`` when
        ``messages`` carries the request (worker column 11): then the engine
        renders it with the container's chat template.

        ``penalties`` carries the optional worker columns 6..10 and ``top_k``
        column 12; any field left out keeps the engine's process-level default.
        """
        p = penalties or {}
        def field(key, default=""):
            value = p.get(key)
            if value is None:
                return default
            return f"{float(value):.6g}" if isinstance(value, float) or isinstance(value, int) else str(value)
        col11 = ""
        if messages is not None:
            payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            col11 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        req = "\t".join([
            "" if messages is not None else base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
            str(max(0, int(max_tokens))),
            f"{float(temperature):.6g}",
            f"{float(top_p):.6g}",
            str(int(seed)),
            field("repetition_penalty"),
            field("frequency_penalty"),
            field("presence_penalty"),
            str(int(p.get("no_repeat_ngram"))) if p.get("no_repeat_ngram") else "",
            field("min_p"),
            col11,
            str(int(top_k)) if top_k and top_k > 0 else "",
            request_id,
            "1" if logprobs else "",
        ]) + "\n"
        with self.lock:
            # A crashed worker used to surface as "engine worker exited" on every
            # later request until someone restarted the server; rebuild it instead.
            self.ensure_alive()
            assert self.proc.stdin is not None and self.proc.stdout is not None
            with self._write_lock:
                self.proc.stdin.write(req)
                self.proc.stdin.flush()
            text = ""
            finish = "length"
            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0
            prefill_tokens = 0
            callback_error = None
            logprob_rows = None
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
                    message = ev.get("message", "worker error")
                    if "context full" in str(message).lower():
                        raise ContextFull(message)
                    raise WorkerError(message)
                elif kind == "done":
                    if isinstance(ev.get("logprobs"), list):
                        logprob_rows = ev["logprobs"]
                    text = ev.get("text", text)
                    finish = ev.get("finish_reason", "length")
                    prompt_tokens = int(ev.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(ev.get("completion_tokens", completion_tokens))
                    cached_tokens = int(ev.get("cached_tokens", cached_tokens))
                    # the engine reports what only it can see (the board's CMA, how
                    # many programs are compiled); /health shows the latest
                    self._note_engine(ev)
                    prefill_tokens = int(ev.get("prefill_tokens", prefill_tokens))
                    if callback_error is not None:
                        raise callback_error
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                        "prefill_tokens": prefill_tokens,
                    }
                    if logprob_rows is not None:
                        # Internal: the handler turns these into the OpenAI
                        # `logprobs` object and removes the key before sending.
                        usage[LOGPROBS_KEY] = logprob_rows
                    return text, finish, usage

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


def canonical_tools(tools):
    """Put the local tool pack in a stable order and shape.

    The prefix cache matches on exact tokens, so a client that paraphrases a
    description, reorders the tools or omits a field would otherwise miss it and
    pay a full prefill (measured: 83 s vs 6 s).  Tools whose name matches the
    server's pack are replaced by the canonical definition; anything unknown is
    passed through untouched.
    """
    if not tools:
        return tools
    known = {name: spec for name, spec in local_tools.TOOL_SPECS.items()}
    out = []
    for t in tools:
        name = None
        if isinstance(t, dict):
            fn = t.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
            name = name or t.get("name")
        if name in known:
            out.append({"type": "function", "function": known[name]})
        else:
            out.append(t)
    order = {name: i for i, name in enumerate(local_tools.TOOL_SPECS)}
    def key(t):
        fn = t.get("function") if isinstance(t, dict) else None
        return order.get((fn or {}).get("name"), len(order))
    return sorted(out, key=key)
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


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}
    return None


def _thinking_value(value):
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str):
            return kind.strip().lower() in {"enabled", "on", "true"}
        enabled = value.get("enabled")
        if isinstance(enabled, bool):
            return enabled
        effort = value.get("effort")
        return _thinking_value(effort) if effort is not None else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"none", "off", "minimal", "disabled"}:
            return False
        if text in {"low", "medium", "high", "max", "xhigh"}:
            return True
    return _as_bool(value)


def thinking_from_request(body: dict) -> bool:
    """Read thinking controls used by common OpenAI-compatible clients.

    Supported forms:
      chat_template_kwargs.open_thinking / enable_thinking
      thinking: bool or {"type": "enabled"|"disabled"}
      enable_thinking: bool
      reasoning_effort: none|minimal|low|medium|high|max
      reasoning: bool or {"enabled": bool, "effort": ...}
    Absent means off; there is no content-based heuristic.
    """
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        for key in ("open_thinking", "enable_thinking", "thinking"):
            if key in kwargs:
                value = _thinking_value(kwargs[key])
                if value is not None:
                    return value
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        for key in ("enable_thinking", "thinking"):
            if key in extra:
                value = _thinking_value(extra[key])
                if value is not None:
                    return value
        extra_kwargs = extra.get("chat_template_kwargs")
        if isinstance(extra_kwargs, dict):
            for key in ("open_thinking", "enable_thinking", "thinking"):
                if key in extra_kwargs:
                    value = _thinking_value(extra_kwargs[key])
                    if value is not None:
                        return value
    for key in ("thinking", "enable_thinking", "reasoning"):
        if key in body:
            value = _thinking_value(body[key])
            if value is not None:
                return value
    if "reasoning_effort" in body:
        value = _thinking_value(body["reasoning_effort"])
        if value is not None:
            return value
    return False


_CHATBOX_META_MARKERS = (
    "\n\nAdditional info for this conversation:",
    "\n## Runtime\nCurrent model:",
)
_GENERIC_SYSTEMS = {
    "you are a helpful assistant.",
    "you are a helpful assistant",
    "你是一个乐于助人的助手。",
    "你是一个乐于助人的助手",
}


def normalize_system_prompt(system: str) -> str:
    """Remove Chatbox runtime metadata from the system prompt.

    Chatbox appends model/date metadata and a system-reminder contract to the
    user's prompt.  That is useful for large models, but it dilutes MiniMind's
    short identity prompt.  Strip the known metadata and fall back to the
    MiniMind prompt unless the client supplied its own MiniMind prompt.
    """
    text = system or ""
    had_metadata = False
    for marker in _CHATBOX_META_MARKERS:
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
            had_metadata = True
    text = text.strip()
    if not text or text.lower() in _GENERIC_SYSTEMS:
        return DEFAULT_SYSTEM
    if had_metadata and "minimind" not in text.lower():
        return DEFAULT_SYSTEM
    return text


# One warning per unknown template name, not one per request.
def build_messages(body: dict) -> dict:
    """Turn an OpenAI body into the worker's message payload.

    Used only when the container carries its own chat template: then the engine
    renders it and this process never has to know what the prompt looks like.
    The system turn keeps the server's normalisation (Chatbox metadata stripped,
    the neutral default filled in), because that is a policy decision, not a
    property of the model.
    """
    src = body.get("messages")
    if not isinstance(src, list) or not src:
        raise ValueError("messages must be a non-empty array")
    sys_text = None
    out = []
    for msg in src:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("every message needs a role")
        if role == "system" and sys_text is None:
            sys_text = normalize_system_prompt(content_text(msg))
            continue
        kept = {"role": role, "content": content_text(msg)}
        for key in ("tool_calls", "tool_call_id", "name"):
            if msg.get(key) not in (None, "", []):
                kept[key] = msg[key]
        out.append(kept)
    if sys_text is None:
        sys_text = normalize_system_prompt("")
    out.insert(0, {"role": "system", "content": sys_text})
    req = {"messages": out, "open_thinking": bool(thinking_from_request(body))}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        req["tools"] = canonical_tools(tools)
    return req


def request_chars(req) -> int:
    """Rough prompt size for the context-window estimate (~3 chars per token)."""
    if isinstance(req, str):
        return len(req)
    return sum(len(str(m.get("content") or "")) for m in req.get("messages", []))
def prune_messages(messages):
    """Drop the oldest non-system message, or shorten the last one.

    Returns a new list, or None when there is nothing left to trim.  Used when
    the engine reports that the prompt does not fit: a long chat should degrade
    (forget the oldest turns) instead of failing the request.
    """
    if not isinstance(messages, list):
        return None
    out = [m for m in messages]
    keep_system = [m for m in out if isinstance(m, dict) and m.get("role") == "system"]
    rest = [m for m in out if not (isinstance(m, dict) and m.get("role") == "system")]
    if len(rest) > 1:
        return keep_system + rest[1:]
    if len(rest) == 1:
        msg = dict(rest[0])
        text = content_text(msg)
        if len(text) < 200:
            return None
        msg["content"] = text[: len(text) // 2] + "\n[earlier text trimmed to fit the context window]"
        return keep_system + [msg]
    return None


def tool_choice_name(body: dict):
    """Resolve request.tool_choice to (mode, tool_name).

    mode is one of "auto" (default), "none", "required", "named".
    Accepts the OpenAI forms:
      "none" | "auto" | "required"
      {"type": "function", "function": {"name": "calculate_math"}}
      "calculate_math"            (shortcut several local clients use)
    """
    raw = body.get("tool_choice")
    if raw is None:
        return "auto", None
    if isinstance(raw, str):
        text = raw.strip()
        low = text.lower()
        if low in {"none", "auto", "required"}:
            return low, None
        if text:
            return "named", text
        return "auto", None
    if isinstance(raw, dict):
        if raw.get("type") in {"none", "auto", "required"} and "function" not in raw:
            return str(raw["type"]), None
        fn = raw.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"].strip():
            return "named", fn["name"].strip()
    return "auto", None


def deterministic_arguments(body: dict, tool_name: str) -> dict:
    """Arguments for a directly-routed tool.

    Order: request.tool_arguments → tool name in the last user turn → empty.
    For calculate_math the user text is the expression, which is what the
    /math route does too; anything else falls back to {} so the tool reports a
    clean argument error instead of guessing.
    """
    explicit = body.get("tool_arguments")
    if isinstance(explicit, dict):
        return explicit
    messages = body.get("messages")
    text = ""
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = content_text(msg).strip()
                break
    if tool_name == "calculate_math" and text:
        prefix = "/math"
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
        return {"expression": text}
    return {}


def build_tool_completion(cid: str, created: int, model: str, payload: dict) -> dict:
    """OpenAI-shaped chat completion for a tool that ran without the model.

    The JSON result goes into `content` so a plain client sees an answer, and
    the same call is also reported in `tool_calls` so an agentic client can
    continue the normal tool loop.
    """
    name = payload.get("tool") or ""
    arguments = json.dumps(payload.get("arguments", {}), ensure_ascii=False)
    text = local_tools.tool_result_text(payload)
    message = {
        "role": "assistant",
        "content": text,
        "tool_calls": [{
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if payload.get("ok") else "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "tool_result": payload,
    }


class StopFilter:
    """Apply OpenAI ``stop`` sequences to a token stream.

    A stop string can straddle token boundaries, so the tail that could still
    become a stop sequence is held back instead of emitted.
    """

    def __init__(self, stops: list[str], emit):
        self.stops = [s for s in stops if s]
        self.emit = emit
        self.buf = ""
        self.hit = False
        self.held = 0

    def feed(self, piece: str) -> None:
        if self.hit or not piece:
            return
        self.buf += piece
        for s in self.stops:
            index = self.buf.find(s)
            if index >= 0:
                if index:
                    self.emit(self.buf[:index])
                self.buf = ""
                self.hit = True
                return
        if not self.stops:
            self.emit(self.buf)
            self.buf = ""
            return
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        if len(self.buf) > keep:
            cut = len(self.buf) - keep
            self.emit(self.buf[:cut])
            self.buf = self.buf[cut:]
        self.held = len(self.buf)

    def finish(self) -> None:
        if not self.hit and self.buf:
            self.emit(self.buf)
        self.buf = ""


def sampling_penalties(body: dict) -> dict:
    """Optional sampling knobs, forwarded to the worker.

    repetition_penalty is not an OpenAI field but every local runtime has one;
    frequency/presence follow the OpenAI ranges.  Values are clamped here and
    again in the engine.
    """
    out = {}
    def num(key, lo, hi, cast=float):
        if key not in body:
            return None
        try:
            value = cast(body[key])
        except (TypeError, ValueError):
            return None
        return max(lo, min(hi, value))

    rep = num("repetition_penalty", 0.5, 2.0)
    if rep is not None:
        out["repetition_penalty"] = rep
    freq = num("frequency_penalty", -2.0, 2.0)
    if freq is not None:
        out["frequency_penalty"] = freq
    pres = num("presence_penalty", -2.0, 2.0)
    if pres is not None:
        out["presence_penalty"] = pres
    ngram = num("no_repeat_ngram", 0, 8, int)
    if ngram:
        out["no_repeat_ngram"] = ngram
    minp = num("min_p", 0.0, 1.0)
    if minp is not None:
        out["min_p"] = minp
    return out


def stop_strings(body: dict) -> list[str]:
    raw = body.get("stop")
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    out = []
    for item in items:
        if isinstance(item, str) and item and len(item) <= 64:
            out.append(item)
    return out[:4]


def apply_stop_text(text: str, stops: list[str]) -> tuple[str, bool]:
    """Truncate at the earliest stop occurrence.  Returns (text, stopped)."""
    cut = None
    for s in stops:
        index = text.find(s)
        if index >= 0 and (cut is None or index < cut):
            cut = index
    if cut is None:
        return text, False
    return text[:cut], True


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
                        logprobs=want_logprobs)
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
                    messages=req, top_k=top_k, request_id=cid)
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
                    messages=req, top_k=top_k, request_id=cid)
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
