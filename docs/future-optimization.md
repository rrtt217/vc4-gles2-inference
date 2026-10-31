# Future optimization directions

This is a working backlog, not a promise list. It records what is already
implemented, what is worth doing next, and what should not be attempted on the
current Pi Zero / VC4 target.

Last updated: 2026-11-01.

## Current baseline

| Area | Current state |
|---|---|
| Chat model | MiniMind3 64M q4/q8, ChatML template |
| Default Pi context | q4: 1024; q8: 768 |
| KV cache | packed q16 (KV_PACK=16) |
| Tools | get_current_time, calculate_math, system_status |
| Tool cache | single-tool q8 cache, 252 tokens, built on the Pi |
| Direct chat cache | --chat --prefix-cache skips the cached prefix |
| Sampling | temperature, top-p, greedy, repetition/frequency/presence, no-repeat n-gram, min-p |
| Server | OpenAI-compatible server; per-request tool/sampling control is partial |
| Thinking | format works, but MiniMind3 q4/q8 does not reliably close think tags |
| Effective throughput | MiniMind2-Small i16 on Pi: about 0.12 GFLOPS, about 0.5 percent of the common 24 GFLOPS VC4 peak |

## P0: highest-value work

### 1. Batch prompt prefill

The engine currently advances one token at a time even for prompt prefill.
That is the largest avoidable cost: a 31-token MiniMind2 prompt took 12.6 s,
while decode was about 0.44 s/token.

Work:

- Add a matmul/tile path for multiple prompt positions.
- Keep causal masking for transformer attention.
- Measure prefill and decode separately; do not trade decode latency for a
  cosmetic prefill gain.
- Start with no-tool chat; add tool prompts only after the basic path is
  correct.

Expected result: several-fold prefill improvement, no change to decode
quality. This is more valuable than another small prompt tweak.

### 2. Deterministic tool routing

The 64M model cannot reliably choose among three tools. In local tests:

- system_status was sometimes called as get_status or get_status_task;
- calculate_math changed 256*37 into 32 + 45 * 3;
- the full tool list selected get_current_time for a system-status request.

Work:

- Add explicit routes such as /status, /time, /math EXPR.
- Add a tool_choice or --tool NAME path that does not ask the model to pick.
- For deterministic tools, return the raw JSON result and optionally skip the
  second generation.
- Keep one tool in the prompt whenever possible.
- Validate JSON arguments before execution; reject or retry malformed calls.

This is the most practical way to make tools useful on this hardware.

### 3. Sampling controls

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

Current cache loading and direct CLI use are implemented, but cache creation is
still manual and error-prone.

Work:

- Add a script that builds a cache from model + seq + KV + tools and verifies
  PREFIX_CACHE loaded N tokens.
- Put model source size/mtime, seq, KV packing, prompt hash, and tool hash in
  the cache metadata (already partly present).
- Refuse to use a cache built for another model or tool set.
- Build caches on the target machine; laptop and Pi caches are not
  interchangeable.
- Keep a single-tool cache for system_status and build the full-tool cache
  only when needed.

### 5. Evaluation harness

Single-prompt impressions are not enough. Add a small fixed suite:

- identity and language;
- simple arithmetic;
- tool selection and arguments;
- multi-turn context;
- long prompt / context boundary;
- q4 vs q8;
- q8 with and without tool cache;
- sampling settings A/B.

Record:

- tool name accuracy;
- argument accuracy;
- final answer accuracy;
- prefill/decode time;
- CmaFree;
- malformed JSON / missing think close.

### 6. Pi tool pack

Keep the set small and deterministic:

- system_status (done);
- set_timer;
- gpio_read (read-only);
- network_info;
- fixed-directory note read/append.

Do not add arbitrary shell, Python execution, arbitrary file paths, or GPIO
writes. Every tool result should be short JSON.

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

Keep as an experiment only:

- i16 math is reasonable;
- identity and instruction following are weak;
- q8 says it is Alibaba Cloud;
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
