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

On the Pi the short form is `sh server/run-pi.sh`, which adds `--kv-pack 16`,
`--seq 768`, `--warmup` and the prefix cache when one exists (override with
`PORT`, `SEQ`, `KV_PACK`, `CACHE`, `MODEL`, `ONLY_TOOL`).

Defaults:

- engine: `../engine/engine_mm3`
- model: `../models/minimind3/minimind3.q8.mmcf`
- tokenizer: the blob inside the container (`--tokenizer` only for a legacy `.tok`)
- model id: `minimind3-64m`
- cache: 1024 tokens
- KV packing: off (24-bit); use `--kv-pack 16` or `8`
- worker log: `worker.log`

## Chat template

Two cases, decided by the worker's `ready` event — never by a flag or a file
name.

**The container carries its own Jinja template** (MMCF metadata key 9).
`ready` then has `"chat_template": true`, and this process stops rendering
prompts entirely: it sends the message list and the engine renders it with the
model's own template, exactly as `llama.cpp` does with
`tokenizer.chat_template`.  That is the only way a template this build has never
seen can work, because only the engine has the Jinja engine.

**It does not.**  `ready` reports the template *kind* (3 TinyBrainBot, 2
MiniMind3, 1 chatml, 0 plain) and the renderers in this file still build the
prompt, which is what every container written before key 9 needs.  `GET
/health` prints the name under `template`.

Either way the system turn is this process's policy: Chatbox's runtime metadata
is stripped and the neutral default is filled in when the client sends none.

> The server's KV layout is not the CLI's. `server.py` leaves `KV_PACK` unset
> (24-bit) unless you pass `--kv-pack 16`; the CLI gates use `KV_PACK=16`.  On
> MiniMind3 q8 the packed and unpacked grids give different first tokens
> (`我是一个AI助手，…` vs `我是MiniMind，…`), so pass `--kv-pack 16` when you
> want the server to reproduce the CLI string.

| template | prompt |
|---|---|
| `minimind3` / `chatml` | `<|im_start|>system ...` plus a `<think>` block, exactly as before |
| `tinybrainbot` | `<|user|> {q}<|assistant|> `, no system turn, no `<think>`; history turns end with `<|end|>` |
| `plain` / unknown | falls back to chatml and prints one warning |

TinyBrainBot also needs `--no-gpu-layer`: its full texture-resident GPU layer
path is not correct on VC4 yet (docs/performance.md, item 12), so the CPU path
is the deployable one.

## Endpoints

```text
GET  /health
GET  /v1/models
GET  /status            deterministic, no model
GET  /time              deterministic, no model
GET  /tools             list the deterministic tools
GET  /math?expr=...     deterministic arithmetic
POST /v1/math           {"expression": "..."}
POST /v1/tool           {"name": "...", "arguments": {...}}
POST /v1/chat/completions
```

The `/status`, `/time`, `/math` and `/tool` routes never touch the GL worker:
they answer in 10–30 ms on a Pi Zero, versus ~25 s for the same question asked
through the model.

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
    "top_k": 50,
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

Thinking is request-driven.  The server accepts the common client spellings:

```json
{"thinking": true}
{"thinking": {"type": "enabled"}}
{"reasoning_effort": "low"}
{"reasoning": {"enabled": true}}
{"enable_thinking": true}
{"chat_template_kwargs": {"enable_thinking": true}}
{"extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}
```

`thinking: false`, `{"type": "disabled"}`, `reasoning_effort: "none"`,
and an absent flag all mean **off**.  There is no content-based heuristic.

Chatbox appends `Additional info` / `## Runtime` metadata to the system prompt. The server strips that metadata and uses the short default `You are a helpful assistant.`; a custom prompt containing `MiniMind` is preserved.

## Sampling controls

The request body is forwarded per request, not fixed at worker start:

| field | range | notes |
|---|---|---|
| `temperature` | 0 = greedy | |
| `top_p` | (0, 1] | |
| `top_k` | >= 0, 0 disables | hard truncation before `top_p`; MiniMind3 recommends 50 |
| `seed` | integer | |
| `repetition_penalty` | 0.5–2.0 | not OpenAI, but every local runtime has it |
| `frequency_penalty` | -2.0–2.0 | OpenAI range |
| `presence_penalty` | -2.0–2.0 | OpenAI range |
| `no_repeat_ngram` | 0–8 | 0 disables |
| `min_p` | 0.0–1.0 | 0 disables |

They travel as optional tab-separated worker columns 6–10, and `top_k` is
column 12 (column 11 is the base64 request JSON); a client (or an old server)
that sends five columns still works, and values are clamped in both the server
and the engine.  Prefer greedy (`temperature: 0`) for tool JSON — the
penalties are for prose.

## Request stats

`python3 server.py --stats-file stats.jsonl` appends one JSON line per chat
request: wall time, prompt/cached/prefilled/completion tokens, finish reason,
stream flag, tool count and HTTP status.  `run-pi.sh` turns it on by default
(`STATS=` to override).

```sh
python3 server/stats_report.py server/stats.jsonl       # human summary
python3 server/stats_report.py server/stats.jsonl --json --hours 24
```

It answers the question that matters on this board — **is the prefix cache
actually being hit?**  The report prints the cached share of prompt tokens next
to wall-time percentiles.  First three requests on the Pi: median wall 8.3 s,
p95 12.5 s, cache hit rate 93.3% of prompt tokens.

## Tool definitions are canonicalised

The prefix cache matches on exact tokens, so the tool block in the prompt has to
be byte-stable.  Before rendering, `canonical_tools()` rewrites any tool whose
name matches the server's pack to the server's own definition and sorts the
known ones into pack order; unknown tools pass through untouched.

Measured on the Pi, one `system_status` tool plus a short user turn:

| client sends | prompt | prefill tokens | wall |
|---|---:|---:|---:|
| server's definition (before this change) | 295 | 242 | 83.1 s |
| server's definition (after) | 295 | 18 | 14.4 s |
| paraphrased definition (after) | 295 | 18 | 1.0 s* |

\* the second run reuses the KV the first one left in the engine.

So a client that paraphrases the description (or omits a field, or reorders the
tools) no longer pays a full prefill for it — that was the difference between
83 s and a cache hit.

**Keep the header small.** MiniMind3-64M was trained at 256 positions; the
one-tool header is already 277 tokens.  Building the cache with all five local
tools produces an 845-token header, which the engine rejects for a 768-token
window ("cache prompt empty or too long").  Send zero or one tool if you care
about latency and quality.

## Context overflow

A chat that outgrows the window degrades instead of failing:

1. The server estimates the prompt at ~3 characters per token and, when
   `prompt + max_tokens` would exceed `--seq`, drops the oldest non-system
   message before sending anything (streaming cannot retry after the headers are
   out, so this happens up front).
2. If the engine still answers `[context full]`, the non-streaming path trims
   and retries up to four times; a single over-long message is cut in half with
   an explicit `[earlier text trimmed to fit the context window]` marker.
3. Only when nothing is left to trim does the request fail, with HTTP **400** and
   `code: context_length_exceeded`.

Measured on the Pi: a 2880-character single message is trimmed and answered
(prefill dominates: ~190 s), and a two-turn chat whose first turn is that long
answers in ~10 s after the first turn is dropped.

`max_tokens` larger than the window is clamped by the engine rather than
rejected.

## Stop sequences

```json
{"messages": [{"role": "user", "content": "..."}], "stop": ["\n\n", "END"]}
```

Up to four strings, 64 characters each.  A hit truncates the text and sets
`finish_reason: "stop"`.  Streaming holds back the last `len(stop)-1` characters
so a stop sequence split across tokens is not leaked to the client.

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
same tens-of-ms absolute overhead is negligible next to ~315 ms/token decode
and 40 s prefill; the real costs are model prefill, attention, and KV-cache
size.

### Health and worker recovery

```json
{"status": "ok", "worker_alive": true, "restarts": 1, "prefix_cache_tokens": 277}
```

`GET /health` returns **503** with `status: degraded`, `worker_alive: false` as
soon as the engine process is gone.  The next `/v1/chat/completions` then
rebuilds the worker in place (model reload + prefix cache, ~30 s on the Pi) and
serves the request instead of failing forever.  `--no-auto-restart` disables
that, `--max-restarts N` (default 5) bounds it.

Measured: killing the worker, `GET /health` flipped to 503 immediately, and the
next chat request came back with a normal completion after the restart
(`restarts: 1`, cache reloaded).

## Deterministic tools (no model in the loop)

A 64M model cannot reliably pick between three tools and mangles their
arguments (measured: `system_status` requested as `get_status` /
`get_status_task`, `256*37` turned into `32 + 45 * 3`).  So every deterministic
tool can be invoked without asking the model at all.

Local tools live in `server/localtools.py`:

| tool | arguments | returns |
|---|---|---|
| `system_status` | — | host, machine, uptime, load, memory, CPU temp, **CmaFree** |
| `get_current_time` | `timezone` (optional) | ISO 8601 + epoch, UTC or local |
| `calculate_math` | `expression` | exact arithmetic (`+ - * / // % **`, sqrt/sin/cos/tan/log/exp/floor/ceil/round/min/max, pi, e) |
| `network_info` | — | non-loopback IPv4 interfaces, default gateway, DNS servers |
| `disk_usage` | `path` (optional, default `/`) | total/used/free MB and percent for that filesystem |

The math evaluator walks a restricted AST: no names, no attribute access, no
calls outside the whitelist, no exponent above 64, no expressions over 200
characters.  `__import__('os')`, `(1).__class__`, `lambda: 1` and friends are
rejected before anything is evaluated.

### Routing by `tool_choice`

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "256*37"}],
  "tool_choice": "calculate_math"
}'
```

Accepted forms: `"calculate_math"`, `{"type":"function","function":{"name":"calculate_math"}}`,
or the OpenAI strings `"auto"` / `"none"` / `"required"`.  When the name resolves
to a local tool the request never reaches the worker:

- `message.content` carries the JSON result, so a plain client still gets an answer;
- `message.tool_calls` reports the same call, so an agentic client can continue
  its normal loop;
- the top-level `tool_result` field carries the same payload for debugging;
- `usage` is zero, `finish_reason` is `tool_calls`.

Arguments come from `tool_arguments` when present, otherwise from the last user
message (for `calculate_math` the message text is the expression; a leading
`/math` is stripped).  Anything malformed returns
`{"ok": false, "error": "..."}` with HTTP 400 instead of being executed.

`--only-tool NAME` prunes the tool list rendered into the prompt to a single
tool.  The model still chooses (badly), but with one option the choice stops
being a coin flip.

## Self-test, evaluation, prefix cache

```bash
python3 server/selftest.py            # 54 assertions, no engine needed
python3 server/eval_suite.py --base http://127.0.0.1:8000 --out eval.json
python3 server/build_prefix_cache.py build  --tools system_status --out server/pfx.kv
python3 server/build_prefix_cache.py verify --cache server/pfx.kv
```

`selftest.py` substitutes a stub worker, so it checks routing, validation and
SSE framing in ~1 s without a GPU.  `eval_suite.py` runs a fixed case set
(identity, arithmetic, tool routing, multi-turn, long prompt, streaming,
malformed body) against a live server and writes JSON for diffing.  The
prefix-cache builder writes a sidecar `.json` (engine hash, model hash, seq,
KV packing, prompt hash, tool list) and `verify` refuses to load a cache whose
sidecar no longer matches.

## Limitations

- One request at a time; the Pi has one GL context and one KV cache.
- No auth/TLS; bind to localhost or put it behind a reverse proxy.
- Prefix cache is a single chain; branching conversations fall back to the
  common prefix and may re-prefill most of the prompt.
- `n > 1` is not implemented.  `stop` is implemented (string or list, up to
  4 × 64 chars); in streaming mode the tail that could still become a stop
  sequence is held back instead of emitted.  Sampling penalties are supported
  per request (see above).
- Built-in tool *execution* is limited to the three deterministic tools in
  `localtools.py`, and only when the client names one via `tool_choice`; the
  general case is still the client-side `tool_calls` flow.
- The worker runs with `GPU_LAYER=1` by default; use `--no-gpu-layer` for the
  CPU-attention path.
- This is not a multi-tenant server.  Queueing is via a Python lock; a second
  request waits for the first.
- **Only one GPU client may run at a time.** The worker holds the model and the
  KV cache in CMA; starting a second `engine_mm3` (a benchmark, another server)
  can starve it and kill the worker.  Stop the server before benchmarking.
  Symptom: `/health` still answers, `/v1/chat/completions` returns
  `{"error": {"message": "engine worker exited"}}`.

## Worker protocol

Request line:

```text
base64(prompt) \t max_tokens \t temperature \t top_p \t seed
               \t [repetition_penalty \t frequency_penalty \t presence_penalty
                  \t no_repeat_ngram \t min_p]
               \t [base64(request JSON) \t [top_k]]
```

The first five columns are required; columns 6-10 and 12 are optional and may be
empty (`\t\t`), which keeps an old server speaking to a new engine.  The engine resets
`g_sampling` to its process defaults before applying them, so a penalty on one
request never leaks into the next.

Column 11 is the message-based request: base64 of

```json
{"messages": [{"role": "user", "content": "..."}],
 "tools": [...], "open_thinking": false}
```

When it is present column 1 is empty and the engine renders the prompt from the
container's chat template.  A container without one rejects the request with an
`error` event rather than guessing a template, so a server that has not read
`chat_template` from `ready` cannot silently send the wrong shape.

Events:

```json
{"event":"ready","chat_template":true,"template":"minimind3"}
{"event":"start","prompt_tokens":80}
{"event":"token","text":"春"}
{"event":"done","text":"...","finish_reason":"stop","prompt_tokens":80,"completion_tokens":64}
{"event":"error","message":"..."}
```

`temperature <= 0` means greedy.  The worker is responsible for UTF-8
boundaries, so every `token.text` is valid UTF-8 even when the BPE token is a
partial byte sequence.
