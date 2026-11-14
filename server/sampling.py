"""Sampling controls: the stop filter and the per-request penalties.

`StopFilter` is the one piece of the server that holds state across tokens: a stop
string can straddle a token boundary, so it keeps a tail and pushes back the last
few characters until it knows whether a stop matches.

Split out of `server.py` (2026-11-14, P2-2).  Pure functions of the request plus
that one class; no engine state, no HTTP.
"""

from __future__ import annotations

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


