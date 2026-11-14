"""Rendering a request into what the engine sees.

The engine renders the chat template itself (`--worker` takes the message list,
never a prompt string), so this module does the parts the *protocol* needs: pull
the text out of OpenAI's content shapes, normalise and canonicalise the tool list
(byte-stable, or every request misses the prefix cache), translate the thinking
switch, apply the system prompt, and prune a conversation that would overflow the
context.

Split out of `server.py` (2026-11-14, P2-2).  Pure functions of the request body.
"""

from __future__ import annotations

import json
import uuid

import localtools as local_tools  # the canonical tool list is filtered by it

# The compiled default is the container's; this string is what the server sends
# when a request has no system message at all.
DEFAULT_SYSTEM = "You are a helpful assistant."

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


