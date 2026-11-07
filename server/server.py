#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the MiniMind3 GLES2 worker.

No third-party dependencies.  The C++ engine owns the GL context and KV cache;
this process owns HTTP/JSON/SSE and (later) tool execution.

Protocol to engine --worker, one request at a time:
  stdin : base64(prompt) \t max_tokens \t temperature \t top_p \t seed
  stdout: JSONL events: ready/start/token/done/error

The worker does not render chat: the server builds the prompt text, and it
picks the renderer from the template the checkpoint reports in the ready event
(chatml / minimind3 keep the renderer below; tinybrainbot has its own).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import localtools as local_tools  # noqa: E402  (deterministic tools, same directory)

DEFAULT_SYSTEM = "You are a helpful assistant."


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
        self.log = open(log_path, "ab", buffering=0)
        # The container names its chat template (kind 3 TinyBrainBot, 2
        # MiniMind3, 1 chatml, 0 plain) and the worker reports the effective one
        # in its ready event.  The server still renders the text - the worker
        # protocol carries a prompt string - but it picks the renderer from that
        # report, never from a flag or from the file name.
        self.template = "chatml"   # pre-ready default: the historical renderer
        # True when the container carries its own Jinja template (MMCF key 9).
        # Then the *engine* renders the prompt and this process only ships
        # messages; otherwise the renderers below still run here.
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
                return True
            if ev.get("event") == "error":
                raise WorkerError(ev.get("message", "worker startup error"))
        return False

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

    def generate(self, prompt, max_tokens: int, temperature: float,
                 top_p: float, seed: int, on_token=None, penalties: dict | None = None,
                 messages: dict | None = None):
        """Return (text, finish_reason, usage).  on_token(text) is optional.

        ``prompt`` is either the rendered prompt string, or ``""`` when
        ``messages`` carries the request (worker column 11): then the engine
        renders it with the container's chat template.

        ``penalties`` carries the optional worker columns 6..10; any field left
        out keeps the engine's process-level default.
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
        ]) + "\n"
        with self.lock:
            # A crashed worker used to surface as "engine worker exited" on every
            # later request until someone restarted the server; rebuild it instead.
            self.ensure_alive()
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
                    message = ev.get("message", "worker error")
                    if "context full" in str(message).lower():
                        raise ContextFull(message)
                    raise WorkerError(message)
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


def render_tools(tools) -> str:
    if not tools:
        return ""
    lines = []
    for t in canonical_tools(tools):
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
_TEMPLATE_WARNED: set[str] = set()


def build_prompt_tinybrainbot(body: dict) -> tuple[str, bool]:
    """TinyBrainBot own prompt: ``<|user|> ...<|assistant|> ``, no system turn.

    The model was trained on ``<|user|> {q}<|assistant|> {a}<|end|>`` with no
    BOS and no system turn (docs/model-formats.md), so the chatml text this
    server used to build for every container was a prompt it had never seen.
    Assistant turns already in the history are closed with ``<|end|>``, the
    token the model emits itself; a tool message has no representation in this
    template and arrives as a plain user turn.  There is no ``<think>`` block.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    text = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue        # no system turn here; the engine CLI drops it too
        if role == "assistant":
            text += "<|assistant|> " + content_text(msg) + "<|end|>"
        else:
            text += "<|user|> " + content_text(msg)
    text += "<|assistant|> "
    return text, False


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


def build_prompt(body: dict, template: str = "chatml") -> tuple[str, bool]:
    """Render the chat for the template the checkpoint reported.

    Only tinybrainbot has its own renderer; the chatml/minimind3 path below is
    unchanged, so a MiniMind3 prompt stays byte-identical to what it was.
    """
    if template not in ("chatml", "minimind3", "tinybrainbot"):
        # A container that says "plain" (or names a template this build does not
        # know) has no renderer here; keep the historical text and say so once.
        if template not in _TEMPLATE_WARNED:
            _TEMPLATE_WARNED.add(template)
            print(f"note: no renderer for template {template!r}; using chatml", file=sys.stderr)
    if template == "tinybrainbot":
        return build_prompt_tinybrainbot(body)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    tools = body.get("tools") or []
    system = DEFAULT_SYSTEM
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system = normalize_system_prompt(content_text(msg)) or system
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
    thinking = thinking_from_request(body)
    text += "<|im_start|>assistant\n"
    text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return text, thinking


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MiniMindOpenAI/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

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
            }, 200 if alive else 503)
        elif path == "/v1/models":
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
        template = getattr(self.server.worker, "template", "chatml")
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
            # A container that carries its template is rendered by the engine:
            # this process sends messages and never a prompt string.
            use_tmpl = bool(getattr(self.server.worker, "chat_template", False))
            if use_tmpl:
                req = build_messages(body)
                thinking = bool(req["open_thinking"])
                prompt = ""
            else:
                prompt, thinking = build_prompt(body, template)
                req = prompt
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
            stops = stop_strings(body)
            penalties = sampling_penalties(body)
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
                if use_tmpl:
                    req = build_messages(body)
                else:
                    prompt, _thinking = build_prompt(body, template)
                    req = prompt

        if not stream:
            # A long chat must not fail outright: if the engine says the prompt
            # does not fit, forget the oldest turns and try again.
            for _attempt in range(4):
                try:
                    text, finish, usage = self.server.worker.generate(
                        prompt, max_tokens, temperature, top_p, seed, penalties=penalties,
                        messages=req if use_tmpl else None)
                    break
                except ContextFull as e:
                    trimmed = prune_messages(body.get("messages") or [])
                    if trimmed is None:
                        self._error(f"prompt exceeds the context window: {e}", 400,
                                    "context_length_exceeded")
                        return
                    body["messages"] = trimmed
                    if use_tmpl:
                        req = build_messages(body)
                    else:
                        prompt, _thinking = build_prompt(body, template)
                        req = prompt
                except WorkerError as e:
                    self._error(str(e), 500, "server_error")
                    return
            else:
                self._error("prompt still exceeds the context window after trimming history",
                            400, "context_length_exceeded")
                return
            if stops:
                text, stopped = apply_stop_text(text, stops)
                if stopped:
                    finish = "stop"
            calls, content = parse_tool_calls(text)
            message = {"role": "assistant", "content": content or (None if calls else "")}
            if calls:
                message["tool_calls"] = calls
                finish = "tool_calls"
            self._stats(wall_s=round(time.time() - getattr(self, "_rec", {"ts": time.time()})["ts"], 3),
                        usage=usage, finish=finish, stream=False, status=200,
                        tools=len(body.get("tools") or []))
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
                    prompt, max_tokens, temperature, top_p, seed, penalties=penalties,
                    messages=req if use_tmpl else None)
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
                    messages=req if use_tmpl else None)
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
            self._stats(wall_s=round(time.time() - getattr(self, "_rec", {"ts": time.time()})["ts"], 3),
                        usage=usage, finish=finish, stream=True, status=200,
                        tools=len(body.get("tools") or []))
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
    ap.add_argument("--model", default=str(repo / "models/minimind3/minimind3.q8.mmcf"))
    ap.add_argument("--tokenizer", default="",
                    help="Legacy .tok (empty = the container blob)")
    ap.add_argument("--model-id", default="minimind3-64m")
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
    worker = EngineWorker(a.engine, a.model, a.tokenizer, a.seq,
                          gpu_layer=not a.no_gpu_layer, log_path=a.worker_log,
                          prefix_cache=a.prefix_cache, kv_pack=a.kv_pack,
                          auto_restart=not a.no_auto_restart, max_restarts=a.max_restarts)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.model_id = a.model_id
    httpd.worker = worker
    httpd.only_tool = a.only_tool
    httpd.seq = int(a.seq)
    httpd.stats_file = a.stats_file
    httpd.stats_lock = threading.Lock()

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
