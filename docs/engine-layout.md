# Engine source layout

The engine is no longer one translation unit. The split is intentionally by
responsibility, not by line count:

| File | Responsibility |
|---|---|
| `src/common.h/.cpp` | shared types (`Mat`, `Target`), q16 helpers, signals, file helpers |
| `src/xformat.h/.cpp` | **the only activation-layout definition** (I16_1, I16_2, I8_4, and the P2 pair-of-positions target) and its GLSL macros |
| `src/weight_format.h` | **the only weight-format definition**: one row per `WeightType` with its scale modes, slice alignment, accepted activation formats and the pipeline stages (`WStage`) it has a kernel for, plus the container dtype mapping |
| `src/logits_grid.h` | **the only vocabulary-grid definition**: tile rows, the packed half-width, the matvec strip threshold |
| `src/shaders.h` | the builder API: pure `(rows, cols, formats…) -> std::string` functions; no GL, no weight type |
| `src/shaders_internal.h` + `src/shaders_common.cpp` | the prelude every builder emits, and the decode text more than one format shares |
| `src/shaders_q8.cpp`, `_i16`, `_q4`, `_q6`, `_t3` | one weight format's builders and its private decode (the weight-format boundary) |
| `src/shaders_block.cpp` | row-elementwise block passes: RMSNorm/LayerNorm, SwiGLU/swish/squared ReLU, bias add, the hybrid conv, the P2 layout split |
| `src/shaders_rwkv.cpp` | RWKV-7's private f32 bit-cast codec, gates, norms and WKV recurrence |
| `src/shaders_attn.cpp` | attention (self and cross), KV and logits shader builders |
| `src/dump_shaders.h/.cpp` | the `--dump-shaders` matrix: every builder's emitted GLSL, digested |
| `src/glsl_io.h/.cpp` | runtime GLSL text: `GLSL_DUMP` seeds `<dir>/<key>.glsl`, `GLSL_OVERRIDE` compiles a named program's file (the kernel experiment without a rebuild) |
| `src/gpu.cpp` | the GL object's own state: the constructor (which reads the three target limits), the capability queries (`gpu_attn_enabled`, `gpu_layer_enabled`, `rms_fused`, `needs_finish`, `att_cfg`) and the qkv probes.  142 lines; it was 2674 before the 2026-11-14 split |
| `src/gpu_gl.cpp` | EGL/GLES2: `init_egl` (surfaceless context, the MESA version overrides for the q4 kernels), `init_gl`, the bind cache accessors and the read-backs |
| `src/gpu_state.cpp` | textures/targets factories, `attach_dump_fbo`, `release_scene`/`reset_targets`, the scene ledgers (the ones `gate/check_target_reset.sh` keeps complete) |
| `src/gpu_prog.cpp` | the program cache: `x_prog` (key **and** text hash), `link_program`, `es3ify`, the `Q4_SHIFT`/`Q4_INLINE` probes |
| `src/gpu_select.cpp` | **the `x_*` selectors**: builder text to a cached program, and the key policy (causal, rope_none, qbatch, the formats and the row-scale flag all change the text) |
| `src/gpu_matvec.cpp` | the projection drivers: `matvec_to`/`from`/`add`/`bias`, the read-back `matvec`/`matvec_s`, and `draw` |
| `src/gpu_logits.cpp` | the vocabulary projection: tiles to logits, candidates, the greedy argmax and their read-back |
| `src/gpu_p2.cpp` | the two-position prefill (P2): the gate/w2 matvecs and the pair split |
| `src/gpu_attn.cpp` | the attention block: qkv draws, RoPE, scores, softmax, KV writes and their probes |
| `src/gpu_ffn.cpp` | FFN: split/fused SwiGLU, the 2-row packing, the pair variants |
| `src/gpu_row.cpp` | the row-pass drivers: LayerNorm, bias add, swish and RMSNorm (one fullscreen draw a row) |
| `src/gpu_conv.cpp` | the hybrid (LFM2.5) conv block |
| `src/gpu_rwkv.cpp` | RWKV-7's private GPU loop: the bit-cast state, gates and WKV recurrence |
| `src/gpu_rwkv_prefill.cpp` | RWKV-7's chunked prefill: one draw for a batch of projections plus the read-back |
| `src/gpu_upload.cpp` | staging: host vectors and Mats into textures, and the container's bias/LayerNorm rows |
| `src/gpu_diag.cpp` | the dump path (`STAGE_DUMP`, `ATT_DUMP`, the row hashes) |
| `src/gpu_dmabuf.cpp` | the dma-buf import that makes the q4 4-bit view cost +1.2% instead of +61% |
| `src/gpu_internal.h` | the seam between them: bind cache, draw cap, scene ledger, measurement hooks, shared program-layer symbols |
| `src/gpu.h` | EGL context, textures, targets, program cache (keyed by name **and** shader text -- the name alone collides across models), GPU passes |
| `src/model.h/.cpp` | model metadata, the MMCF reader, q4/q8/i16 conversion, FFN split |
| `src/session.h/.cpp` | KV cache, RoPE, attention, transformer forward, prefix cache |
| `src/tokenizer.h/.cpp` | llama2.c tokenizer and byte-level BPE, embedded-blob loader |
| `src/sampling.h/.cpp` | sampling, penalties, top-p/min-p |
| `src/grammar.h/.cpp` | constrained decoding: a byte-level JSON pushdown automaton that masks the vocabulary above the logits boundary (the worker's `response_format` column); architecture-agnostic, no GPU or numerics change |
| `src/gbnf.h/.cpp` | a generic deterministic (LL(1)) grammar matcher and its token-mask wrapper, sharing the `Constraint` contract with the JSON automaton; the json_schema tier runs on it |
| `src/gbnf_full.h/.cpp` | full GBNF: a text parser and llama.cpp's general multi-stack NFA matcher over Unicode code points (partial UTF-8 across tokens is legal while it can still fit), for arbitrary user grammars via the `grammar` response_format type; ambiguous alternatives, repetition, left-recursion guard |
| `src/schema.h/.cpp` | JSON Schema -> grammar compiler (type/properties/required/items/enum, closed objects); trie-factors string sets so the matcher stays deterministic |
| `src/chattmpl.h/.cpp` | renders the container's Jinja chat template; returns the prompt split into template text and caller text.  Also parses the worker's message request (column 11) |
| `src/tools.h/.cpp` | tool JSON, restricted arithmetic, built-in handlers |
| `src/app.h/.cpp` | the live engine state (container, GL context, tokenizer, chat template, system prompt) and the four steps that make a loaded container runnable: `prepare_model`, `setup_from_model`, `upload_weights`, `load_prefix_cache_any`.  Startup and `/model` share them, which is why they are one object.  `Options` is held **by reference**: `setup_from_model` decides the template flags and `main` reads them back |
| `src/cli.h/.cpp` | `Options` and the one argument parser (the usage text lives here too) |
| `src/ppl.cpp` | `--ppl`: one sliding window at a time, BOS included, the gate's number |
| `src/prefix_build.cpp` | `--build-prefix-cache`: render the header, prefill, save the KV plus the sidecar |
| `src/chat.h/.cpp` | the chat path: `run_turn`/`run_tool_turn` (the template, the tool rounds, the marker mask, the prefix-cache reuse), `repl()` (the `--chat` REPL with `/model`) and `complete()` (one prefill, one decode loop).  `main()` is 135 lines of wiring around these |
| `src/worker.h/.cpp` | the `--worker` protocol: request line, JSONL events, control ops (`load`/`cancel`/`stats`), its own stdin reader (a cancel must be seen mid-request) and `proc_meminfo_kb` for the diagnostics |
| `src/chat_format.h/.cpp` | strip a tool-call block, split a `<think>` block, render tool calls for the template |
| `src/lineedit.h/.cpp` + `src/lineedit_layout.h/.cpp` | the REPL's line editor: raw-mode key decoding (UTF-8 and column aware, never byte counts), history, a buffer that may contain newlines, and the terminal's paste frame (`ESC[?2004h` answered by `ESC[200~ ... ESC[201~`).  The layout arithmetic -- rows, cursor position, vertical moves -- is pure in `lineedit_layout`, so `tests/test_lineedit.cpp` checks it (and the editor itself, through a pty) in `make check` |
| `src/engine.cpp` | the entry point: worker protocol, chat loop, capability/template handling, and the wiring around the modules above |

The sibling directories are **not** linked into `engine_mm3`:

| Path | Runs on | Contents |
|---|---|---|
| `engine/third_party/jinja/` | engine | vendored llama.cpp Jinja engine (MIT); see its README for the three local edits |
| `engine/tools/` | host python | container packer/inspector, numpy reference forwards |
| `engine/gate/` | host shell/python | the pass/fail checks that need no GPU: documents, the switch registry, shader digests, the q16 rail, the draw cap, the target ledger, the container layout |
| `engine/scripts/` | host shell/python | needs a GPU and/or a model: the acceptance gate, the CPU-vs-GPU diff, the attention/logits oracles, the benchmarks, `swap_matrix.py` |
| `engine/eval/` | host python | model quality: the MC/ifeval/translate batteries and their cases |
| `engine/tests/` | host C++ | compiled tests, built by the Makefile |
| `engine/probes/` | target GLES2 | standalone VC4 feature and throughput probes |

`engine/tools/` and `engine/scripts/` are where a numerics question should start;
`AGENTS.md` maps a question to the tool.  `engine/probes/` predates them and an-
swers narrower questions (what does a draw cost, does the driver accept a 2432-
row texture, does `roundEven` work).

## X formats

Every activation texture is RGBA8. `XFormat` decides the byte layout and the
quantization scale:

| X format | values / texel | storage |
|---|---:|---|
| `I16_1` | 1 | q16 in `.rg` |
| `I16_2` | 2 | q16 pair: `.rg`, `.ba` |
| `I8_4` | 4 | signed int8 in RGBA, 1/16 scale |

Producers call `XST1/XST2/XST4`; consumers call `X1/X2/X4`. Hot matvec kernels
use the q16-domain variants `X1Q/X2Q/X4Q` and `R1Q` so the inner loop does not
pay a divide per activation. The residual stream has its own `R1/R2/R1Q`
macros and is never assumed to have the activation layout.

This is the fix for the previous 8-bit-activation failures: adding a format
changes `xformat.cpp`, not seven shader strings. `ACT=i8` selects `I8_4` for all
matvec inputs, `FFN_HID=i8` only for the FFN hidden state; both are opt-in.

The prompt is a separate matter from the activation layouts: `chattmpl.cpp`
renders the template and hands back `PromptPart`s that carry an `is_input`
flag, and `Tokenizer::encode_parts` matches special tokens only outside those
parts.  Template text may produce `<|im_end|>`; a user message may not.

The **P2** layout is a target layout, not an `XFormat`: in the two-position
prefill pair, one texel holds value *k* of position A in `.rg` and value *k* of
position B in `.ba`, so `x_per_texel()` does not apply and the width is
`x_pair2_texels(n) == n`.

The scheme around that layout is format-neutral except for the weight half:
`rms_stats2`/`rms_norm_pair2` build the FFN input and `p2_split` hands the layer
residual back, so only the gate and w2 kernels belong to a weight format
(`WStage::P2`/`P2Add`, in `shaders_q8.cpp`, `shaders_q4.cpp` and
`shaders_t3.cpp`).  The pair
normaliser's divisor must be the serial normaliser's: on a fused-reduce backend
(VC4) it sums the 32 partials in the normalising fragment, on a split-reduce one
it reads the two totals `rms_stats_reduce2` wrote, because the serial path
quantises the total through `rms_stats_reduce`.

## Shader text

The builders are pure functions — `(rows, cols, XFormat) -> std::string` — and
the text they emit *is* the numerics. Wrapping a decode in a GLSL helper cost
VC4 177 against 283 MMAC/s and moved PPL with the same arithmetic
(`docs/benchmarks.md`, "The refactor's 12% regression"), so the emitted bytes
are the contract, not the builder's source. Two tools read them:

| Question | Tool |
|---|---|
| what did the program that just ran look like? | `SHADER_SRC=<substr> ./engine_mm3 ...` |
| what does *every* builder emit? | `./engine_mm3 --dump-shaders[=raw]` |
| did a refactor move any of it? | `gate/shader_hashes.sh` |

`--dump-shaders` needs no model and no GL context: it calls every builder over a
fixed matrix (q8/q4/i16 shapes, all four X formats, the attention layouts) and
prints one digest per program; `=raw` prints the GLSL. The one driver-dependent
choice in the builders, the q4 shift/and decode, is forced both ways, so
`q4shift0/…` and `q4shift1/…` appear only where the two texts differ and the
dump is machine independent.

`gate/shader_hashes.sh` compares that dump with `gate/shader_hashes.txt`
(`--rebaseline` rewrites it, `--show NAME` prints one program). A refactor that
claims to be text-neutral must leave the golden file untouched. The hot kernels
repeat the decode inline on purpose, so the repetition is generated from the
shared `k*` constants in `shaders_common.cpp` or the format TU that owns them
rather than copied: one definition, same bytes.

Which weight kernels exist is not part of that text: `src/weight_format.h` answers
`wf_can(type, stage)`, and the GPU adapter is the only code that includes both it
and `shaders.h`.  A stage the table grants and the adapter lacks a case for
stops the run; a case with no stage bit only leaves a path unused.  Neither
direction can fall through to another format's decoder.

JSON documents — the `--worker` event stream, tool arguments and results, and
the chat-template context — are built with `common_json` from
`third_party/jinja/json.h`, never by string concatenation.  It keeps object keys
in insertion order (so the worker stream stays byte-stable), escapes for you,
and is a pimpl: the vendored `nlohmann` stays confined to `json.cpp`, so no
other translation unit pays its compile cost.  The only `{"…"}` left in the
sources is the tool-call example inside the CLI's system prompt, which is text
for the model rather than a document.

The vocabulary grid is the one constant that is not a C++ detail: the packer
splits the logits matrix at the same stride, the host uses it as the read-back
stride and the token formula, and the four Top-K kernels divide by it.  It lives
in `src/logits_grid.h` and `probes/logits_grid_probe` checks all three languages
(`make -C engine probes/logits_grid_probe`, no GL needed).  Those kernels only
run when `--top-k-candidates` is set, so neither greedy output nor PPL would notice a
mismatch - measured, see the header's comment.

Builders also check their arguments where the text is made: the q8 column loop
steps by four and the q4 loop by 32, so a width that is neither is a shader that
reads past its activation texture and still compiles — it is now an abort that
names the builder.  `att_defs()` does the same for the attention layout
(`heads % kv_heads`, even `head_dim`, a known `KVPACK`).  A uniform that is set
but not declared is reported once with the program name: `-1` from
`glGetUniformLocation` used to be a silent no-op, which is a wrong number rather
than an error.

## Build

```bash
make -C engine -B engine_mm3 CXXFLAGS='-O2 -std=c++17'
make -C engine -B armv6                            # Pi Zero, built on the board
make -C engine -B pi CROSS_SYSROOT=<sysroot>       # Pi Zero, ARMv6, cross
make -C engine -B probes/draw_probe_pi CROSS_SYSROOT=<sysroot>   # a probe
```

The cross target discovers the compiler and the sysroot's gcc runtime, and fails
unless the linked binary is ARMv6/VFPv2 — see `docs/cross-build.md`; a `-B`
directory that does not exist is silently ignored by gcc and produces a binary
that faults on the ARM1176.

Probes have their own targets under `engine/probes/`; the Pi ones take the same
`CROSS_*` variables as `pi`.

## Regression rule

A change to GPU state, X formats or shader text must preserve:

1. q8 greedy output for the canonical prompt;
2. q4 greedy output for the canonical prompt;
3. `--ppl` on the fixed corpus, within the recorded tolerance.

Then `engine/scripts/regression.sh`, which is the gate — determinism, a second
prompt's behaviour, the per-machine pins, and PPL.  Only after those pass should
`ACT=i8` or performance numbers be compared.

When the change touches a builder at all, run `engine/gate/shader_hashes.sh`
first: it is seconds, it needs no GPU, and it says whether the emitted text moved
at all.  PPL can hide a one-texel change; a digest cannot.

A same-path A/B (one flag, one kernel) is what decides a numerics question; a
CPU-vs-GPU stage delta from `diff_paths.sh` names a lead, not a defect, because
the CPU path is f32 that round-trips q16 only at the dump.
