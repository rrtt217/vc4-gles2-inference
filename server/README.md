# OpenAI-compatible MiniMind3 server

This is a small, dependency-free OpenAI-style HTTP front end around the
MiniMind3 GLES2 engine.  The engine is a long-lived worker; the Python process
owns HTTP, JSON, SSE, and tool-call schema.

[中文说明](README.zh-CN.md)

```text
OpenAI client / curl
        |  HTTP + SSE
http_api.py  <- server.py builds the server and calls main()
        |  base64 + tab-separated request, JSONL events
engine_mm3 --worker (one GL context, one model)
```

Modules (2026-11-14; `server.py` was one 1504-line file before):

| File | Holds |
|---|---|
| `server.py` | the entry point: CLI, model list, the server object, `main()` |
| `http_api.py` | the routes and the SSE writers (`Handler`) |
| `engine_bridge.py` | the worker: request line, JSONL events, control lines, cancel, restart |
| `chat_render.py` | messages/tools/thinking/system-prompt rendering and context pruning |
| `tool_calls.py` | deterministic tool choice, arguments and the OpenAI tool_calls reply |
| `sampling.py` | `StopFilter` and the per-request penalties |
| `localtools.py` | the deterministic tools themselves |

Everything is still stdlib only.

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

The engine is the only renderer: it renders from the container's own Jinja
template (MMCF metadata key 9), exactly as `llama.cpp` does with
`tokenizer.chat_template`.  This process sends the message list and never a
prompt string.  `ready` reports `"chat_template": true|false`; when it is false
the request is refused with a 400 naming the packer flag, because the
alternative — guessing a format here — is what used to put ChatML markers in
front of a model that had never seen them.  Every chat container packed from a
source with a `chat_template` carries key 9; the `stories*` completion models have
no chat turn to render, and a chat container packed without the key -- e.g.
`smollm2-135m.q8q24.mmcf` -- is refused with a 400 until it is repacked.  Repack
with `mmcf_pack.py --chat-template <file>.jinja`.  `GET
/health` prints the template name under `template`.

The system turn is still this process's policy: Chatbox's runtime metadata
is stripped and the neutral default is filled in when the client sends none.

> The server's KV layout is not the CLI's. `server.py` leaves `KV_PACK` unset
> (24-bit) unless you pass `--kv-pack 16`; the CLI gates use `KV=q16`.  On
> MiniMind3 q8 the packed and unpacked grids give different first tokens
> (`我是一个AI助手，…` vs `我是MiniMind，…`), so pass `--kv-pack 16` when you
> want the server to reproduce the CLI string.

| template | prompt |
|---|---|
| `minimind3` / `chatml` | `<|im_start|>system ...` plus a `<think>` block, exactly as before |
| `tinybrainbot` | `<|user|> {q}<|assistant|> `, no system turn, no `<think>`; history turns end with `<|end|>` |
| `plain` / unknown | whatever the container's own template renders; a container with no template is refused with 400 |

TinyBrainBot also needs `--no-gpu-layer`: its full texture-resident GPU layer
path is not correct on VC4 yet, so the CPU path is the deployable one.

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

### Health, queue and cancellation

`GET /health` answers what a client needs before it sends work:

| field | meaning |
|---|---|
| `status`, `worker_alive`, `restarts` | process-level liveness (503 when the engine is gone) |
| `busy`, `inflight` | requests being processed now; the engine is single-flight, so this is a queue |
| `est_wait_ms` | median of the last eight request durations times the requests ahead; `-1` when something is queued but nothing has finished yet to extrapolate from |
| `last_request_ms` | the previous request's wall time |
| `engine` | the last numbers the engine reported about itself: `cma_free_kb` (a headroom signal, **not** a budget: near the ceiling, attempt the load), `mem_available_kb`, `programs` compiled, `seq` |
| `models`, `resident` | every configured model and the one loaded now |

Every finished request logs one line with its id, so a client's `chatcmpl-...`
can be found in the server output:

```text
[req] chatcmpl-78b0... model=mm3-q8 stream=0 finish=length prompt=31 completion=4 wall=0.72s
```

If the client hangs up mid-stream the server sends `{"op":"cancel"}` to the
engine and logs `[cancel] <id> sent=1 (client disconnected)`; the engine stops
within a few tokens instead of generating the rest for nobody.  It is best
effort: the disconnect is only noticed when the next chunk write fails, so a
short remaining generation may finish first.  A dead worker's 500 carries the
engine's last log lines as `--- engine stderr tail ---`, which is where the CMA
or GL failure that killed it is.

### Several models on one server

`--model` may be repeated; each container is advertised by `GET /v1/models` and
`/health` lists them with the one that is resident:

```bash
python3 server.py --model ../models/minimind3/minimind3.q8.mmcf --model-id mm3 \
                  --model ../models/llama2c/minimind2-small.q8.mmcf --model-id mm2small
# listening ... models=[mm3 (...), mm2small (...)]  (one resident at a time, loaded on demand)
```

`--model-id` is positional against the `--model` list; a model without one takes
its container file name.  A request's `model` field selects the container, and
**exactly one is resident at a time**: the Pi Zero's CMA fits one, and the engine
keeps its EGL context and compiled programs across the switch, so a switch costs
a container read plus an upload (6.12 s + 2.99 s on the board) instead of a
process start (which would also pay 4.23 s of context and 1.89 s of program
links -- `engine/README.md`).  The switch happens on the first request that
names a different model and is logged as `[model] <id> resident, load <ms> ms`;
the resident model is not reloaded per request.  The container's chat template
travels with it, so `/health`'s `template` and the chat-template refusal follow
the model that is currently loaded.  An unknown name is a 400 that names what
the server serves.

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

**Keep the header small.** MiniMind3's trained context is 768
(`docs/long-context.md`), and 256 is the container's packed `seq`; the one-tool
header is already 234 tokens (measured with `build --tools system_status`).
Building the cache with all five local tools produces an 802-token header, which
the engine rejects for a 768-token window ("cache prompt empty or too long").
Send zero or one tool if you care about latency and quality.

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
   `type: context_length_exceeded` (`code` stays null).

Measured on the Pi: a 2880-character single message is trimmed and answered
(prefill dominates: ~190 s), and a two-turn chat whose first turn is that long
answers in ~10 s after the first turn is dropped.

`max_tokens` larger than the window is capped at 256 here and then clamped to the
remaining window by the engine; it is never rejected.

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
are documented in [../docs/prefix-cache.md](../docs/prefix-cache.md).  The path
may be a **directory**: the engine then loads the entry whose header matches the
model it is holding, so several models share one cache directory and a model
switch reloads the right one.  The **engine builds the cache** too
(`--build-prefix-cache - OUT` renders the header with the container's own template)
and writes the `<cache>.json` sidecar; `build_prefix_cache.py` is only the trigger
and the checker around it, because rendering the header in Python was a second
renderer hard-coded to ChatML.  The server **refuses to start** with a `.kv` that
has no sidecar (or a `--prefix-cache` path that does not exist): a stale cache
does not fail, it silently prefills everything again.

## Python overhead

The Python layer only does HTTP parsing, message and tool-list normalisation,
JSON/SSE framing, and pipe I/O.  Measured on this laptop with a fully cached 79-token prompt and
8 generated tokens:

| path | wall |
|---|---:|
| direct worker | 118.0 ms |
| HTTP server | 119 ms |

So the Python/HTTP overhead is **about 1 ms** for a cached request, and about
19 ms for the first cold request.  It is not the bottleneck.  On Pi Zero, the
same tens-of-ms absolute overhead is negligible next to ~190 ms/token decode
(`docs/benchmarks.md`) and a prefill measured in seconds; the real costs are
model prefill, attention, and KV-cache size.

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

A 64M model cannot reliably pick between five tools and mangles their
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
`{"ok": false, "error": "..."}` (HTTP 200, `finish_reason: "stop"`) instead of
being executed; only `POST /v1/tool` answers 400.

`--only-tool NAME` prunes the forwarded tool list to a single
tool.  The model still chooses (badly), but with one option the choice stops
being a coin flip.

## Self-test, evaluation, prefix cache

```bash
python3 server/selftest.py            # routing, SSE and sampling; no engine needed
python3 server/eval_suite.py --base http://127.0.0.1:8000 --out eval.json
python3 server/build_prefix_cache.py build  --tools system_status --out server/pfx.kv
python3 server/build_prefix_cache.py verify --tools system_status --cache server/pfx.kv
```

`selftest.py` substitutes a stub worker, so it checks routing, validation and
SSE framing in ~1 s without a GPU.  `eval_suite.py` runs a fixed case set
(identity, arithmetic, tool routing, multi-turn, long prompt, streaming,
malformed body) against a live server and writes JSON for diffing.  The
engine writes a sidecar `.json` next to the cache (container size and mtime, the
shape, `seq`, KV packing, prompt and tool hashes, token count); the builder only
triggers the build and then `verify` refuses a cache whose sidecar no longer
matches.

## Limitations

- One request at a time; the Pi has one GL context and one KV cache.
- No auth/TLS; bind to localhost or put it behind a reverse proxy.
- Prefix cache is a single chain; branching conversations fall back to the
  common prefix and may re-prefill most of the prompt.
- `n > 1` is not implemented.  `stop` is implemented (string or list, up to
  4 × 64 chars); in streaming mode the tail that could still become a stop
  sequence is held back instead of emitted.  Sampling penalties are supported
  per request (see above).
- Built-in tool *execution* is limited to the five deterministic tools in
  `localtools.py` (`system_status`, `get_current_time`, `calculate_math`,
  `network_info`, `disk_usage`), and only when the client names one via
  `tool_choice`; the general case is still the client-side `tool_calls` flow.
- The worker runs with `GPU_LAYER=1` by default; use `--no-gpu-layer` for the
  CPU-attention path.
- This is not a multi-tenant server.  Queueing is via a Python lock; a second
  request waits for the first.
- **Only one GPU client may run at a time.** The worker holds the model and the
  KV cache in CMA; starting a second `engine_mm3` (a benchmark, another server)
  can starve it and kill the worker.  Stop the server before benchmarking.
  Symptom: `/health` still answers, `/v1/chat/completions` returns
  `{"error": {"message": "engine worker exited"}}`.

## Logprobs

`logprobs: true` returns the OpenAI `logprobs.content` array: one entry per
generated token, `{token, logprob}`, where `logprob` is measured in the
distribution the sampler used (temperature, top-k and top-p applied, then
renormalised over the kept set).  The engine computes it from the same logits it
sampled from -- asking for it does not change the text, only the work — and the
candidate read-back path (`--top-k-candidates`) is bypassed for such a request,
because it does not carry the full vocabulary.

Not implemented yet, and refused rather than silently ignored: `top_logprobs`
(the alternatives — the engine would have to decode each candidate id back to
text) and `logprobs` with `stream: true`.

## Structured output (response_format)

`response_format: {"type": "json_object"}` guarantees the reply is valid JSON.
The engine runs a byte-level JSON automaton over the vocabulary at every decode
step and masks every token the grammar cannot continue with, so temperature,
top-k and top-p all act on the legal set only.  EOS is released only once a
complete top-level value has been parsed, which is why a constrained reply that
fits ends with `finish_reason: "stop"`.  The mask is a host-side edit at the same
point as the sampling penalties, so it forces the full-logits read-back path and
works on every architecture the engine runs (dense, MoE, hybrid, RWKV).

```json
{"messages": [{"role": "user", "content": "Give me a JSON object with name and age."}],
 "response_format": {"type": "json_object"}, "temperature": 0}
```

Two things to know:

* It guarantees **syntax, not semantics**.  A small model forced into JSON still
  chooses the content; on a weak model that can be a valid-but-unhelpful value
  (a string or a number where you wanted an object).  Prompt for the shape you
  want as well.  If `max_tokens` cuts the reply mid-structure the result is a
  valid JSON *prefix*, not a complete document — give it room to finish.
* `{"type": "text"}` (or absent) is unconstrained; any other type is a 400.  The
  engine is the source of truth for what it can constrain.

### json_schema

`response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema":
{...}}}` constrains the reply to a JSON Schema.  The engine compiles the schema to
a grammar and masks the vocabulary to it at every step, so the reply matches the
schema's shape and types, not just "some JSON".

```json
{"messages": [{"role": "user", "content": "Extract the person."}],
 "response_format": {"type": "json_schema", "json_schema": {"name": "person",
   "schema": {"type": "object",
     "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
     "required": ["name", "age"], "additionalProperties": false}}},
 "temperature": 0}
```

Supported subset: `type` (object/array/string/integer/number/boolean/null),
`properties` + `required` + `additionalProperties: false`, `items`, `enum`, and
nested schemas.  Objects are **closed** (`additionalProperties: false`, the OpenAI
structured-output contract); required keys are emitted in a fixed order, optional
keys in any order.

Accepted but **not enforced**: annotation keywords (`title`/`description`/
`default`/`examples`) and the bounds (`minimum`/`maximum`/`minLength`/`maxLength`/
`pattern`/`format`/`minItems`/`maxItems`).  Refused by name (a clear error, not a
wrong approximation): `$ref`, `anyOf`, `oneOf`, `allOf`, `not`, tuple `items`,
multiple `type`s, and `additionalProperties` other than `false`.  As with
`json_object`, the guarantee is shape and syntax -- a small model still chooses
the content, and `max_tokens` must leave room to finish.

## Worker protocol

Request line:

```text
base64(prompt) \t max_tokens \t temperature \t top_p \t seed
               \t [repetition_penalty \t frequency_penalty \t presence_penalty
                  \t no_repeat_ngram \t min_p]
               \t [base64(request JSON) \t [top_k \t [request id \t [logprobs \t [response_format]]]]]
```

A line that starts with `{` is a **control line**, not a request: a request's
first column is base64 of the prompt and base64 has no `{`, so the two cannot be
confused.  Three operations exist:

| op | reply | what it is for |
|---|---|---|
| `{"op":"load","model":PATH}` | `loaded` | replace the container, keeping the EGL context and its compiled programs |
| `{"op":"cancel","id":ID}` | `cancelled` | stop the request in flight (the engine peeks at stdin between tokens, so this lands within a few tokens); an idle cancel is acknowledged, not an error |
| `{"op":"stats"}` | `stats` | what only the engine can see: `cma_free_kb`, `mem_available_kb`, `programs` (compiled), `seq`, `kv_pack`, `template`, `uptime_ms` |

The first one replaces the model while keeping the EGL
context and its compiled programs (on a Pi Zero that is 4.23 s of context and
1.89 s of program links, against 6.12 s to read a container and 2.99 s to
upload it -- `engine/README.md`):

```json
{"op": "load", "model": "../models/minimind3/minimind3.q4.mmcf"}
```

The `load` reply is one event, `{"event":"loaded","model":...,"load_ms":...}`, or an
`error` event; a container that cannot be read ends the process, because leaving
a worker with no model would turn one clear failure into a stream of confusing
ones.  The rolling prefix cache is dropped with the model, for the same reason
the REPL drops its `kv_tokens`: a prefix the new model never computed is not a
prefix.  `engine/scripts/swap_matrix.py` checks that two loads inside one worker
answer exactly like two fresh workers, including a quantisation change.

The first five columns are required; columns 6-10 and 12-15 are optional and may be
empty (`\t\t`), which keeps an old server speaking to a new engine.  The engine resets
`g_sampling` to its process defaults before applying them, so a penalty on one
request never leaks into the next.

Column 11 is the message-based request: base64 of

```json
{"messages": [{"role": "user", "content": "..."}],
 "tools": [...], "open_thinking": false}
```

When it is present column 1 is empty and the engine renders the prompt from the
container's chat template — this is the only shape the server sends now.  A
container without one is refused by the server with a 400 before the engine is
touched (and the engine would reject it anyway, with an `error` event, rather
than guessing a template).

Column 15 is the constrained-decoding spec: base64 of the OpenAI
`response_format` object, e.g. `{"type":"json_object"}`.  When present, the
engine masks the vocabulary to the JSON grammar at every step (see
[Structured output](#structured-output-response_format)).  An unsupported type
is an `error` event, not a silent no-op.  Empty leaves generation unconstrained.

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
