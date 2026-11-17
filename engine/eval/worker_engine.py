"""A generate-only client for the engine's worker protocol.

The eval batteries used to start one engine process per prompt.  On a Pi Zero a
start is ~15 s (container read, EGL context, 13 program links) on top of the
generation, so a 100-item battery paid 25 minutes for the process launches alone;
in one worker the container is read once and the programs compile once.

`chat()` mirrors the interface the batteries already use (text, error, ms), so
`--reuse-worker` in eval_ifeval.py and eval_translate.py is a two-line switch.

**Experimental (2026-11-14), and the `--reuse-worker` flag says so.**  The texts
it produces are *not* yet interchangeable with the per-process path: on a
translation item whose prompt contains a newline, the CLI's `--greedy` and a
worker request with `temperature=0` answered differently from the same rendered
prompt (`今天天气很好，阳光明媚。` against `今天天气很好，我打算去公园散步。`),
and inside the CLI `--greedy` and `--temp 0` are not equivalent either (the
latter degenerates into `场景场景…`).  The prompts were confirmed byte-identical
with `CHAT_DUMP=1`, so the divergence is in the sampling/decoding path, not in
rendering; it is unfound.  Use the flag to measure *time*, not to score.
Prompts are sent as *messages* and rendered by the container's own template, which
is what the server and the REPL do -- a second renderer here would drift.

The control lines are the same ones the server uses: `{"op":"load"}` to replace
the container, `{"op":"stats"}` for the numbers only the engine can see, and
`{"op":"cancel"}` is available but unused here.
"""

import base64
import json
import os
import select
import subprocess
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# The batteries run the engine with cwd=ROOT (the repository root), because their
# `--model` paths are written relative to it.  The worker must use the same base or
# the same command line means two different files.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WorkerEngine:
    def __init__(self, binary, model, env, system, sampling):
        self.binary = binary
        self.model = model
        self.system = system
        self.sampling = dict(sampling)
        # A relative binary is resolved against the caller's directory first, then
        # against ROOT: the batteries pass an absolute default, but a hand-written
        # command may pass "./engine/engine_mm3" from either place.
        if not os.path.isabs(binary) and not os.path.exists(os.path.join(os.getcwd(), binary)):
            candidate = os.path.join(ROOT, binary)
            if os.path.exists(candidate):
                binary = candidate
        e = dict(os.environ)
        e.update(env or {})
        e["MODEL"] = model
        e.setdefault("SEQ", "512")
        self.proc = subprocess.Popen([binary, "--worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, env=e, cwd=ROOT)
        ready = self._next_event(timeout=300)
        if not ready or ready.get("event") != "ready":
            raise SystemExit("worker did not become ready: %r" % (ready,))
        self.template = ready.get("template", "chatml")

    # --- plumbing ---------------------------------------------------------
    def _next_event(self, timeout=900.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not r:
                if self.proc.poll() is not None:
                    return None
                continue
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    def _send(self, line):
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _sampling_columns(self, seed):
        s = self.sampling
        if s.get("greedy"):
            return "0", "1", 0
        temp = float(s.get("temp", 0.8))
        top_p = float(s.get("top-p", 0.9))
        top_k = int(s.get("top-k") or 0)
        return "%.6g" % temp, "%.6g" % top_p, top_k

    def _penalties(self):
        s = self.sampling
        def num(key):
            v = s.get(key)
            return "" if v is None else "%.6g" % float(v)
        return [num("repeat-penalty"), "", "", "", ""]

    # --- the interface the batteries use ----------------------------------
    def chat(self, prompt, gen, seed):
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        payload = base64.b64encode(json.dumps({"messages": messages}, ensure_ascii=False,
                                              separators=(",", ":")).encode("utf-8")).decode("ascii")
        temp, top_p, top_k = self._sampling_columns(seed)
        cols = ["", str(int(gen)), temp, top_p, str(int(seed))] + self._penalties()
        cols += [payload, str(top_k) if top_k else "", "", ""]
        t0 = time.time()
        self._send("\t".join(cols) + "\n")
        text = ""
        while True:
            ev = self._next_event()
            if ev is None:
                return None, "worker exited", (time.time() - t0) * 1000.0
            kind = ev.get("event")
            if kind == "token":
                text += ev.get("text", "")
            elif kind == "error":
                return None, str(ev.get("message", "error")), (time.time() - t0) * 1000.0
            elif kind == "done":
                text = ev.get("text", text)
                return text, "", (time.time() - t0) * 1000.0

    def stats(self):
        self._send(json.dumps({"op": "stats"}) + "\n")
        return self._next_event(timeout=60) or {}

    def load(self, model):
        self._send(json.dumps({"op": "load", "model": model}) + "\n")
        ev = self._next_event(timeout=900) or {}
        if ev.get("event") == "loaded":
            self.model = model
            self.template = ev.get("template", self.template)
        return ev

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
