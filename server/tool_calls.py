"""Deterministic tool calls: choose a tool, fill its arguments, answer the client.

The engine is a 64M model: asking it to emit a JSON tool call works only by
accident.  The server therefore picks the tool deterministically from the request
(one named tool, or "call it"), fills the arguments from the request text where it
can, and renders an OpenAI-shaped tool_calls response.  `--tools` and
`localtools.py` define what exists.

Split out of `server.py` (2026-11-14, P2-2).
"""

from __future__ import annotations

import json
import time
import uuid


import localtools as local_tools  # the name the moved code uses

from chat_render import content_text

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


