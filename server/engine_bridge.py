"""The engine worker: one C++ process, one request at a time.

The engine owns the GL context and the KV cache; this module owns the pipes to it
-- the 14-column tab-separated request line, the JSONL event stream, the control
lines (`{"op": ...}`), cancel, restart and the stderr tail a 500 shows.

Split out of `server.py` (2026-11-14, P2-2 of the 2026-11-13 audit) so the HTTP
layer, the chat renderer and this bridge can be read (and reviewed) separately.
The protocol itself is unchanged and is documented in `server/README.md`; the
engine side is `engine/src/engine.cpp`'s worker branch.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import select
import subprocess
import threading
import time
from pathlib import Path

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
        if prefix_cache:
            # A cache is machine- and model-specific, and a stale one does not
            # fail: it misses, and the request pays a full prefill (83 s against
            # 14.4 s on the Pi Zero for the 2026-11-02 tool header).  That is the
            # silent slowdown this refuses to start with.  The engine writes a
            # `<cache>.json` describing itself; without one there is nothing to
            # compare the model, SEQ and packing against.
            pc = Path(prefix_cache)
            if not pc.exists():
                raise WorkerError("--prefix-cache %s does not exist" % pc)
            cands = sorted(pc.glob("*.kv")) if pc.is_dir() else [pc]
            if not cands:
                raise WorkerError("--prefix-cache %s contains no *.kv" % pc)
            missing = [c for c in cands if not Path(str(c) + ".json").exists()]
            if missing:
                raise WorkerError(
                    "%s has no sidecar; rebuild it with "
                    "`python3 server/build_prefix_cache.py build --model ... --out ...`. "
                    "The engine writes the .json next to every cache it builds, and a cache "
                    "without one cannot be checked against the model, the SEQ or the packing "
                    "-- a stale cache is a silent slowdown, not an error." % missing[0])
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
        benchmarks.md, "Replacing a model without rebuilding the GL context").
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
                 logprobs: bool = False, response_format: dict | None = None):
        """Return (text, finish_reason, usage).  on_token(text) is optional.

        ``prompt`` is either the rendered prompt string, or ``""`` when
        ``messages`` carries the request (worker column 11): then the engine
        renders it with the container's chat template.

        ``penalties`` carries the optional worker columns 6..10 and ``top_k``
        column 12; any field left out keeps the engine's process-level default.
        ``response_format`` (column 15) is the OpenAI constraint object, e.g.
        ``{"type": "json_object"}``; None leaves generation unconstrained.
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
        col15 = ""
        if response_format is not None:
            payload = json.dumps(response_format, ensure_ascii=False, separators=(",", ":"))
            col15 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
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
            col15,
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
