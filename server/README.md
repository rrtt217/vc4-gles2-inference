# OpenAI-compatible MiniMind3 server

This is a small, dependency-free OpenAI-style HTTP front end around the
MiniMind3 GLES2 engine.  The engine is a long-lived worker; the Python process
owns HTTP, JSON, SSE, and tool-call schema.

[中文说明](README.zh-CN.md)

```text
OpenAI client / curl
        |  HTTP + SSE
server.py (stdlib only)
        |  base64 + tab-separated request, JSONL events
engine_mm3 --worker (one GL context, one model)
```

## Start

```bash
cd server
python3 server.py --host 127.0.0.1 --port 8000
```

Defaults:

- engine: `../engine/engine_mm3`
- model: `../models/minimind3/minimind3.q8`
- tokenizer: `../models/minimind3/minimind3.tok`
- model id: `minimind3-64m`
- cache: 1024 tokens
- KV packing: off (24-bit); use `--kv-pack 16` or `8`
- worker log: `worker.log`

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Non-stream:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimind3-64m",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 32,
    "temperature": 0.8,
    "top_p": 0.9,
    "seed": 2
  }'
```

Streaming:

```bash
curl -sS -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimind3-64m",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 32,
    "stream": true
  }'
```

Thinking:

```json
{
  "messages": [{"role": "user", "content": "1+1等于几？"}],
  "chat_template_kwargs": {"open_thinking": true}
}
```

## Tool calls

The server uses the standard OpenAI flow: it returns `tool_calls`, the client
executes the tool, then sends a `role: "tool"` message back.

Request with tools:

```json
{
  "model": "minimind3-64m",
  "messages": [{"role": "user", "content": "帮我算一下 256 乘以 37"}],
  "tools": [{"type": "function", "function": {
    "name": "calculate_math",
    "description": "Evaluate arithmetic",
    "parameters": {"type": "object", "properties": {
      "expression": {"type": "string"}
    }, "required": ["expression"]}
  }}],
  "max_tokens": 64,
  "temperature": 0
}
```

Response:

```json
{
  "finish_reason": "tool_calls",
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "call_...",
      "type": "function",
      "function": {
        "name": "calculate_math",
        "arguments": "{\"expression\": \"256 * 37\"}"
      }
    }]
  }
}
```

Then send the tool result:

```json
{
  "messages": [
    {"role": "user", "content": "帮我算一下 256 乘以 37"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "id": "call_...", "type": "function",
      "function": {"name": "calculate_math", "arguments": "{\"expression\": \"256 * 37\"}"}
    }]},
    {"role": "tool", "tool_call_id": "call_...", "content": "{\"result\":\"9472\"}"}
  ]
}
```

Streaming tool calls are buffered until the native `<tool_call>` block is
complete, then emitted as one `tool_calls` delta.  This is intentional: partial
JSON arguments are not useful to clients.

## Prefix KV cache

The worker token/KV prefix reuse, disk `MMKV1` format, and packed KV modes
are documented in [../docs/prefix-cache.md](../docs/prefix-cache.md).

## Python overhead

The Python layer only does HTTP parsing, ChatML construction, JSON/SSE framing,
and pipe I/O.  Measured on this laptop with a fully cached 79-token prompt and
8 generated tokens:

| path | wall |
|---|---:|
| direct worker | 118.0 ms |
| HTTP server | 119 ms |

So the Python/HTTP overhead is **about 1 ms** for a cached request, and about
19 ms for the first cold request.  It is not the bottleneck.  On Pi Zero, the
same tens-of-ms absolute overhead is negligible next to ~500 ms/token decode
and 40 s prefill; the real costs are model prefill, attention, and KV-cache
size.

## Limitations

- One request at a time; the Pi has one GL context and one KV cache.
- No auth/TLS; bind to localhost or put it behind a reverse proxy.
- Prefix cache is a single chain; branching conversations fall back to the
  common prefix and may re-prefill most of the prompt.
- `n > 1`, `presence_penalty`, `frequency_penalty`, and stop strings are not
  implemented.
- Built-in tool *execution* is not server-side in this version; the standard
  client-side `tool_calls` flow is implemented.  The C++ CLI has built-in
  handlers for local testing.
- The worker runs with `GPU_LAYER=1` by default; use `--no-gpu-layer` for the
  CPU-attention path.
- This is not a multi-tenant server.  Queueing is via a Python lock; a second
  request waits for the first.

## Worker protocol

Request line:

```text
base64(prompt) \t max_tokens \t temperature \t top_p \t seed
```

Events:

```json
{"event":"ready"}
{"event":"start","prompt_tokens":80}
{"event":"token","text":"春"}
{"event":"done","text":"...","finish_reason":"stop","prompt_tokens":80,"completion_tokens":64}
{"event":"error","message":"..."}
```

`temperature <= 0` means greedy.  The worker is responsible for UTF-8
boundaries, so every `token.text` is valid UTF-8 even when the BPE token is a
partial byte sequence.
