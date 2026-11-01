# Deterministic tools: why the model is not asked

Status: implemented 2026-11-01 in `server/localtools.py`, `server/server.py`.

## The problem, measured

MiniMind3 64M with three tools in the prompt produced these failures on the Pi:

| asked | model produced |
|---|---|
| system status | `get_status`, `get_status_task` (names that do not exist) |
| `256*37` | `<tool_call>{"name":"calculate_math","arguments":{"expression":"32 + 45 * 3"}}` |
| system status | `get_current_time` |

Each of those costs a full generation: 299 prompt tokens means 185 s of prefill
on a cold cache, plus ~0.5 s per generated token.  The result is then wrong.

## The shape of the fix

Three deterministic tools live in the server process.  They are ordinary Python,
no model, no GPU:

| tool | arguments | result |
|---|---|---|
| `system_status` | — | host, machine, uptime, load[3], mem total/available, swap free, CPU temp, CmaFree/CmaTotal |
| `get_current_time` | `timezone` (optional) | ISO 8601 + epoch; `UTC`/`GMT`/`Z` resolved explicitly |
| `calculate_math` | `expression` | exact arithmetic |
| `network_info` | — | non-loopback IPv4 interfaces (ioctl, no shell), default gateway from `/proc/net/route`, DNS from `/etc/resolv.conf` |
| `disk_usage` | `path` (optional) | `statvfs` totals in MB; rejects relative paths and `..` |

### Math is not `eval`

`calculate_math` parses with `ast.parse(mode="eval")` and walks the tree with a
visitor that only understands `Expression`, `Constant` (numbers), `Name`
(`pi`, `e`, `tau`), `BinOp`, `UnaryOp` and `Call` to a whitelisted function.
Anything else raises `ToolError` before evaluation:

```
__import__('os').system('ls')   -> unsupported syntax: Call to non-Name
open('/etc/passwd').read()      -> unknown function: open
(1).__class__                   -> unsupported syntax: Attribute
2**9999                         -> exponent is too large
1/0                             -> division by zero
```

Limits: 200 characters, 120 AST nodes, exponent ≤ 64, |result| ≤ 1e15.  Floats
are rounded to 10 decimals and returned as ints when integral, so `2**10` prints
`1024`, not `1024.0`.

## Routing

`tool_choice` is resolved to `(mode, name)`:

- `"calculate_math"` or `{"type":"function","function":{"name":"calculate_math"}}` → named
- `"required"` + exactly one tool offered + that tool is local → the same thing
  (there is nothing to choose, so asking is pure latency)
- `"auto"`, `"none"`, `"required"` with several tools → unchanged model path

A named local tool short-circuits before the worker is touched.  The response is
a normal chat completion so existing clients keep working:

```json
{
  "choices": [{"message": {
      "role": "assistant",
      "content": "{\"ok\":true,\"tool\":\"calculate_math\",\"result\":{\"expression\":\"256*37\",\"result\":9472}}",
      "tool_calls": [{"id": "call_…", "type": "function",
                      "function": {"name": "calculate_math", "arguments": "{\"expression\": \"256*37\"}"}}],
      "finish_reason": "tool_calls"}}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "tool_result": {"ok": true, "tool": "calculate_math", "result": {"expression": "256*37", "result": 9472}}
}
```

`content` carries the JSON so a plain HTTP client still sees an answer;
`tool_calls` keeps agentic clients on the standard loop; `tool_result` is the
same payload for debugging.

Arguments come from `tool_arguments` when the caller sends it, otherwise from
the last user message (`calculate_math` treats the message as the expression and
strips a leading `/math`).  Missing or invalid arguments return
`{"ok": false, "error": …}` with HTTP 400 — the tool is never executed on a
guess.

Plain HTTP equivalents exist for scripts that do not want a chat envelope:

```
GET  /status      GET /time      GET /math?expr=256*37      GET /tools
POST /v1/math     POST /v1/tool
```

## Measured on the Pi Zero

| path | wall |
|---|---:|
| `GET /status` | 0.013 s |
| `GET /time` | 0.013 s |
| `GET /math?expr=256*37` | 0.015 s |
| `POST /v1/chat/completions` + `tool_choice: calculate_math` | 0.02 s |
| same question through the model, header cached | 25.0 s |
| same question through the model, cold | 184.9 s |

So the deterministic path is 1,200× faster than the cached model path and
9,000× faster than the cold one, and it cannot return a wrong tool name.

## Interaction with the prefix cache

A named-tool request never builds a prompt, so it also never invalidates the
cached KV chain.  Mixing deterministic routes and model turns on one server is
therefore safe: the KV chain only advances on real generations.

## What is deliberately not here

- No shell, no Python execution, no arbitrary file access, no GPIO writes.
- No side-effecting tools yet (`set_timer`, `network_info` need a policy first).
- The model path is unchanged when the client does not name a tool, so existing
  behaviour and the `tool_calls` contract are preserved.
