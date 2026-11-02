# Future optimization directions

This is a working backlog, not a promise list. It records what is already
implemented, what is worth doing next, and what should not be attempted on the
current Pi Zero / VC4 target.

Last updated: 2026-11-02.

## Current baseline

| Area | Current state |
|---|---|
| Chat models | MiniMind3 64M q8/q4; MiniMind2 Small i16/q8/q4; TinyChat15M is a format probe only |
| Default Pi context | q4: 1024; q8: 768 |
| KV cache | packed q16 (KV_PACK=16) |
| q8 decode (Pi) | ~238 ms/token, prefill ~20.5 s for a 16-token chat run |
| q4 decode (Pi) | ~381 ms/token; paired gate + packed residual enabled by default |
| Tools | get_current_time, calculate_math, system_status |
| Tool cache | single-tool q8 cache, 252 tokens, built on the Pi |
| Direct chat cache | --chat --prefix-cache skips the cached prefix |
| Sampling | temperature, top-p, greedy, repetition/frequency/presence, no-repeat n-gram, min-p |
| Server | OpenAI-compatible server; per-request tool/sampling control is partial |
| Deterministic tools | `/status` `/time` `/math` `/v1/tool` + `tool_choice` routing: 10–30 ms, no model |
| Prefix cache automation | `server/build_prefix_cache.py build|verify`, sidecar metadata, refuses stale caches |
| Eval harness | `server/eval_suite.py` (fixed cases, JSON output) + `server/selftest.py` (54 assertions, no GPU) |
| Thinking | format works, but MiniMind3 q4/q8 does not reliably close think tags |
| Effective throughput | q8 on Pi: ~238 ms per decode token (2026-11-02), i.e. ~4.2 tok/s; older 517 ms figure is historical |

## P0: highest-value work

### 1. Batch prompt prefill

The engine currently advances one token at a time even for prompt prefill.
That remains the largest structural item; no batching implementation has
landed.  The numbers below are the original 2026-11-01 baseline and are kept
for the cost model.

**Measured on MiniMind3 q8 (2026-11-01):**

- prefill 37.5 s for 79 prompt tokens = **474 ms/token**
- decode **517 ms/token**

So one prefilled position costs the same as one decoded position.  The matvec
shaders are per-output-fragment loops over the weight columns: rendering *B*
positions means *B*× the fragments, so batching alone removes only per-draw
overhead (the attention block is 15.2 ms of a ~70 ms layer, and most of that is
the QKV matvec, not the fixed cost).

The win therefore has to come from **packing positions into the RGBA channels**
of the activation texture: one weight fetch would then feed four positions
(~4× on the matvec, which is 61% of a token) at the cost of rewriting every
activation layout, the rmsnorm/attention shaders and the KV write path.

Work, in order:

- Measure the per-draw overhead with a null shader before committing to batching.
- Prototype 2-position batching in the FFN path only (w1/w3 + w2) and compare
  greedy tokens against the serial path; any token mismatch means the packing
  is wrong, not "close enough".
- Only then consider 4-position RGBA packing, which is the version that can pay
  off.
- Keep causal masking for transformer attention.
- Start with no-tool chat; add tool prompts only after the basic path is
  correct.

Expected result: several-fold prefill improvement, no change to decode
quality. This is more valuable than another small prompt tweak.

### 2. Deterministic tool routing

**Done (2026-11-01).** `server/localtools.py` implements `system_status`,
`get_current_time` and `calculate_math` with a restricted-AST evaluator;
`server.py` routes `tool_choice: <name>` (string or OpenAI object form) straight
to the tool, returning the JSON in `content`, the call in `tool_calls` and the
payload in `tool_result`, with zero model tokens.  `/status`, `/time`, `/math`
and `/v1/tool` do the same over plain HTTP.  `--only-tool NAME` prunes the
prompt to a single tool for the cases that still go through the model.

Measured: 10–30 ms per call on the Pi versus 25–185 s through the model, and
the model's observed mistakes (get_status / get_status_task / `32 + 45 * 3`)
disappear because it is no longer asked.

Remaining work: expose the same routing in `run.sh`, and add `set_timer` /
`network_info` once their side effects are decided.


### 2b. Where the remaining matvec time goes

Current per-phase numbers live in `performance.md` ("where the remaining matvec
time goes"); after activation packing the split is w1+w3 33%, w2 22%, attention
20%, wo 8.5%, norms 9%, logits 6%.  Prefill costs ~470 ms/token and decode ~480
ms/token — the same per token, which rules out draw-call overhead as the driver
and points at the per-weight work inside the fragment shaders.

Scalar ops per weight in the q8 path (per weight texel = 4 weights): fetch 1,
int8 decode `floor(c*255+0.5)-128` 16, activation decode (two int16 from one
texel) ~32, MACs 4.  That is ~13 ops per multiply-accumulate, and at 12–16
lanes the engine lands at ~0.25 GFLOPS, i.e. roughly half of what the op count
predicts — the rest is texture-fetch latency.

Ranked ideas, none of them started:

1. **Fold the -128 bias out of the decode.** `Σ(w-128)·x = Σw·x - 128·Σx`, and
   `Σx` is shared by every output row, so it can be computed once per fragment.
   Roughly 3 of 13 ops per weight disappear; no numeric change if the final
   rounding is kept.
2. **int8 activations (4 per texel).** Implemented behind `ACT8` (X8_4) and
   measured with the `--ppl` harness.  The quality trade is corpus-dependent,
   so it stays opt-in; see the X-format section below.
3. **Two output rows per fragment.** Reuses the activation decode across rows,
   but GLES2 has a single colour attachment, so it needs either two passes or
   interleaved output — a real refactor.

### 3. Sampling controls

**Over the wire (2026-11-01).** The worker protocol now carries optional columns
6–10 (`repetition_penalty`, `frequency_penalty`, `presence_penalty`,
`no_repeat_ngram`, `min_p`); the server clamps them and forwards per request
instead of pinning them at worker start.  Five-column clients still work.
`stop` (string or list) is implemented in both non-streaming and streaming
modes, with the streaming tail held back so a stop string split across tokens
never leaks.

Implemented:

- --repeat-penalty
- --frequency-penalty
- --presence-penalty
- --no-repeat-ngram
- --min-p

Next:

- Add the same fields to the worker protocol and server request body.
- Add a maximum thinking length and a think-closing parser/stop policy.
- Never apply repetition penalties blindly to tool JSON; use greedy for tools.
- Add regression cases for JSON, Chinese text, and long generation.

## P1: practical improvements

### 4. Prefix-cache automation

**Done (2026-11-01).** `server/build_prefix_cache.py build --tools ... --out ...`
writes the chat header to a temp file (the engine takes a file, not inline
text), builds the cache on the target machine, and writes a sidecar `.json`
with engine hash, model hash, size, seq, KV packing, prompt hash and tool list.
`verify` re-derives the header, compares every field, and refuses to load a
cache whose metadata no longer matches; it then proves the engine really loads
it (`PREFIX_CACHE loaded N tokens`).

Measured on the Pi: a 299-token tool request went from 184.9 s (cold, no cache)
to 25.0 s with a 277-token header cache.

Remaining work: a wrapper that rebuilds the cache when the tool list or system
prompt changes (the sidecar already contains everything needed), and a second
cache per tool set (`--only-tool`) once the server runs that mode by default.

### 5. Evaluation harness

**Started (2026-11-01).** `server/eval_suite.py` runs a fixed case set against a
live server and writes JSON: deterministic routes, named-tool routing,
identity, natural-language arithmetic, multi-turn, long prompt, streaming and
malformed-body handling.  `server/selftest.py` covers the same routing logic
with a stub worker (54 assertions, no GPU, ~1 s), which is what makes it safe
to change the server without a Pi in the loop.

Still to add: q4 vs q8, tool-name/argument accuracy over several prompts,
`CmaFree` capture, and storing a baseline file to diff against.

### 6. Pi tool pack

Keep the set small and deterministic:

- system_status (done);
- set_timer;
- gpio_read (read-only);
- network_info;
- fixed-directory note read/append.

Do not add arbitrary shell, Python execution, arbitrary file paths, or GPIO
writes. Every tool result should be short JSON.

### 6b. Server resilience

**Done (2026-11-01).** `/health` reports `worker_alive` / `restarts` and returns
503 when the engine is gone instead of a cheerful "ok"; a dead worker is rebuilt
in place on the next request (`--no-auto-restart`, `--max-restarts N`).  This
was a real failure mode: after another GPU client starved CMA, the worker died
and every later request returned `engine worker exited` until the whole server
was restarted by hand.

### 7. Server ergonomics

The OpenAI-compatible server is not the primary interface right now. If it is
kept:

- expose per-request sampling and tool choice;
- keep one model per process;
- log the request body (redacted) for prompt debugging;
- return predictable errors for context overflow;
- do not pretend to support the full OpenAI surface.

A thin local HTTP API may be more useful than full compatibility.

## P2: model and platform experiments

### SmolLM2-135M-Instruct / MobileLLM-125M

Worth testing on the laptop first:

- Llama-style, GQA, English instruct;
- q4 weights may fit in roughly 80–100 MB;
- likely better English instruction following than TinyChat15M;
- unlikely to provide Chinese or tools.

Only port after a laptop quality/speed check. Do not assume the existing
llama2.c engine handles every Llama variant without conversion tests.

### MiniMind2-Small

Keep as an secondary experiment, not the primary chat model:

- i16 math is reasonable; Pi decode is ~226 ms/token;
- q8 is faster (~140–190 ms/token) but requires
  `ROPE_THETA=1000000 ROPE_HALF=1` because Q8M1 lacks RoPE metadata;
- q4 saves memory but identity answers are unreliable (“一台电脑”);
- identity and instruction following are weak; multi-question prompts often
  answer only the last question;
- thinking is not reliable;
- 1024 context is the practical limit in current tests.

### RWKV7-G1d-0.1B

Not a practical Pi Zero target now:

- Q4_K_M GGUF is about 145 MB, FP16 about 386 MB;
- 65536 vocabulary makes logits/embeddings expensive;
- RWKV is recurrent, so prefill is serial rather than batchable;
- a new GLES2 backend would be required.

Revisit only for Pi 5, phone, or laptop.

## Non-goals

- q8 with 2048 context on Pi Zero;
- q4 as a reliable tool-calling model;
- full OpenAI API compatibility before deterministic tools work;
- adding many tools to the prompt;
- i16 on Pi as a production path;
- treating repetition penalties as a fix for factual errors;
- porting RWKV/Mamba to VC4 before prefill batching and tool routing.

## Benchmark protocol

Use the same model, tokenizer, prompt, SEQ, KV_PACK, seed, and generation
length. Report:

    model / quantization
    SEQ / KV_PACK
    prefill seconds
    decode ms/token
    CmaFree after load
    output sample
    tool name / arguments / result

Do not compare a laptop run with a Pi run as if they were the same machine.


### Attention is now pass-count bound, not ALU bound (2026-11-01, after the hoist)

The block is eight draws per layer: qkv matvec, qk-norm factor, rotated q, KV
rope write, scores, softmax stats, probabilities, p·V.  Per-call profile after
the hoist (pos ~73, profiled so the numbers include the sync):

| pass | ms/layer | what is left in it |
|---|---:|---|
| qkv matvec | ~5.3 | 1536x768 GEMV, already at the measured matvec rate |
| scores | ~4.0 | one q fetch + one KV fetch per element |
| att_v | ~2.7 | one p decode + one KV decode per (dim, t) |
| KV rope write | ~2.5 | one draw, no loop: almost all of it is the draw |
| probabilities | ~1.7 | one exp per (head, t): the draw, not the math |
| normalise / stats | ~1.8 | 8 serial fragments, one draw |

**Correction (measured later): draws are pipelined**, so the per-pass 1.3 ms in
that table came from `LAYER_PROFILE`'s `glFinish()`, not from the draws.  With
`GPU_LAYER=1` the real `att_finish()` is a no-op and only total GPU work matters —
merging the probs pass into `att_v` turned out to be exactly neutral.  The items
below are therefore not worth doing:

- probabilities as q16 instead of 24-bit: saves ~0.3 ms/layer in att_v (~1%).
- parallelising the softmax stats: the draw it adds costs what the shorter
  serial chain saves (~0-2%).
- fusing qk-norm + rotation into one pass: the per-fragment 96-iteration loop
  that replaces the extra draw costs more than the draw (net negative).

To move the needle further you have to remove passes or change the KV layout so
scores and p·V run as ordinary matvecs (K contiguous per head, V^T contiguous
per head).  That is a redesign of the KV cache, not a tuning change.


### What is left, with estimates (measured model: only GPU ALU work counts)

**1. Pair the logits rows (~2%).** The vocabulary pass runs as 5 tiles of
1536x768, one row per fragment, ~20 ms/token.  Pairing rows halves the activation
decode exactly like the FFN gate.  It is self-contained on the write side but not
on the read side: `read_i32` and the Top-K candidate shader both sample the logits
texture one value per texel, so both need a packed reader.  Do it when there is
time to re-run the full eval, not in a hurry.

**2. Prefill batching (the big one, ~2-3x on prefill only).** Prefill currently
costs the same per token as decode (22.5 s for 73 tokens ~ 308 ms each) because
every token walks the whole layer stack on its own.  Running 4 tokens through one
weight fetch would amortise the weight decode (2 of the ~8 ops per weight) across
4 rows of activations, and the gate pairing already shows that sharing the decode
works.  The blocker is attention: 4 queries need 4 score rows, 4 softmax
normalisations and a KV write of 4 positions — a real redesign of
`attention_block`, not a tweak.

**3. KV layout for a matvec-shaped attention.** Storing K contiguous per head and
V transposed per head would let `scores` and `p*V` reuse the existing matvec
kernel, but the fetch and decode counts per element are already at 2 and 2 after
the hoist, so the win is in code reuse, not in throughput.

**4. int8 activations (quality risk, ~1.6x on every matvec).** Implemented as
`X8_4` and exposed by `ACT8=1` (all matvec inputs) / `ACT8=2` (FFN hidden only).
The `--ppl` harness exists; greedy text can stay identical while PPL moves, so
both must be checked.  Keep it opt-in unless a corpus-wide quality run says
otherwise.

## Plan: merge the chunked FFN weight textures (scope expanded)

Decode is 229 ms with 188 draws at ~0.5 ms per draw, so ~94 ms is draw overhead
and ~135 ms is texture-fetch plus MAC work (decode is flat across SEQ, so none
of that is sequence-dependent).  Reaching 150 ms needs both terms down.

The FFN chunks are the cheapest draw reduction left.  `w1a/w1b`, `w3a/w3b` and
`w2a/w2b` are separate Mats purely because the converter wrote them in two
pieces; the weights themselves are one matrix.  Merging them at load time (in
the model loader, from the same float data, before the CPU copy is freed) is a
repack, not a format change, so every value stays identical:

| pass | now | merged |
|---|---:|---:|
| FFN gate pair | 2 | **1** (pair kernel emits 2 rows/fragment, so 2432 rows = 1216 fragments, under the 2048 viewport) |
| swiglu | 2 | **1** (2432 values = 1216 texels) |
| w2 + residual | 2 | **1** (rows 768, cols 2432) |

That is 6 draws per layer down to 3: 188 -> 164 draws/token, about -12 ms.  It
does not reach 150 on its own; the remaining draw budget after it is rmsnorm 4,
qkv 2 (driver strip), rope/scores/stats/v/wo/qrot 6, plus 15 outside the layer
loop.

The compute term is dominated by reading the whole 66 MB q8 weight set once per
token, so the other lever is activation precision: ACT8=2 (-6%) and the paired
row kernels stay available behind their env switches and are worth re-measuring
once the draw count is lower.
