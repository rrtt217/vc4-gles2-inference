# AGENTS.md

Instructions for coding agents working in this repository.  Machine-specific
details live in local override files, not here:

- `AGENTS.local.md` — DSH local override (this laptop).
- `AGENTS.override.md` — Hermes/VLAB full override; it replaces this file.

Those files are local/personal and are gitignored.  Do not copy host names,
sysroots, deploy paths, or local-only benchmark numbers from them into this
file.  Target reference timings are allowed when explicitly labeled with the
target (for example Pi Zero).

## Mission

Maintain a practical GLES2/EGL inference stack for small language models on
Raspberry Pi Zero (VC4) and desktop GLES2. Preserve working behavior; do not
replace working paths with demos or cosmetic rewrites.

## Repository map

| Path | Contents |
|---|---|
| `engine/src/` | EGL/GLES2 engine: X-format layouts, shader builders, GPU, model, session, tokenizer, CLI |
| `engine/tools/` | container packer, container inspector, numpy reference forward passes |
| `engine/third_party/jinja/` | vendored llama.cpp Jinja engine (MIT) — renders the container's chat template |
| `templates/` | the chat templates this repository ships, for `mmcf_pack.py --chat-template` |
| `engine/scripts/` | the regression gate **and** the diagnostic tools |
| `engine/probes/` | standalone VC4 feature/throughput probes (not linked into the engine) |
| `server/` | OpenAI-compatible HTTP server, deterministic tools, self-test, eval suite |
| `models/` | central model store (gitignored), one `.mmcf` per precision |
| `minimind3/` | MiniMind3 source config/tokenizer/reference; artifacts live in `models/` |
| `refs/` | untracked upstream reference material for MiniMind3 (gitignored) |
| `docs/` | deep dives; keep READMEs short and link here |

**SmolLM2-135M is not supported.**  Measured (2026-11-08): its residual reaches
25982 at layer 11 (204x the q16 rail) *and* its FFN hidden reaches 3164, which
still travels through an I16_2 texture.  `--res q24` fixes the first and the
output then degenerates on the second.  The float32 reference is correct, so
this is a format limit, not a weight problem; `models/README.md` has the table.

Engine modules are described in [`docs/engine-layout.md`](docs/engine-layout.md);
the file format in [`docs/model-formats.md`](docs/model-formats.md) and
[`docs/checkpoint-container.md`](docs/checkpoint-container.md).

### `engine/tools/` — the container toolchain

| Tool | Use it when |
|---|---|
| `mmcf_pack.py` | converting an HF directory (safetensors + config) or a legacy checkpoint (Q8M1/Q8M2, Q4M1, I16M1/I16M2) into an MMCF container |
| `mmcf_dump.py` | answering "what does the engine actually read from this file" — the whole metadata block, from the file alone (`--check` verifies every CRC) |
| `mmcf_requant_q8.py` | rebuilding the q8 matrices of a container from its i16 sibling (`--clip P` for a percentile scale) |
| `ref_forward.py` | a numpy forward pass over an HF safetensors source: separates "bad conversion / bad tokenizer / engine bug" |
| `make_tok.py` | building the `.tok` blob from a HF `tokenizer.json` — ByteLevel or SentencePiece, with added tokens laid onto their ids (a blob built from `model["vocab"]` alone ends at the wrong id and the engine splits `<\|im_start\|>` into characters) |

### `engine/scripts/` — the gate and the diagnostics

| Script | Answers |
|---|---|
| `regression.sh` | the acceptance gate: docs drift, determinism, behaviour, pinned strings, PPL |
| `shader_hashes.sh` | did a builder change move the emitted GLSL: digests of every builder's text (`--dump-shaders`), no GPU or model needed |
| `diff_paths.sh` + `dump_diff.awk` | per-stage CPU-vs-GPU comparison, from `STAGE_DUMP=1 HIDDEN_DUMP=1 PREFILL_PAIR=0` |
| `dump_stats.py` | per-row `max/mean/clamp-count` of a `DUMP` file (is a row saturating?) |
| `check_att.py` | offline oracle for `ATT_DUMP`: score, softmax stats and `att_out` reproduced from the integers the shaders consumed |
| `cmp_logits.py` | compares two `LOGITS_OUT` dumps tile by tile |
| `verify_packing.sh` | byte-for-byte check of the packed-activation layout against a known-good binary |
| `bench.sh`, `bench_ab.sh`, `bench_plain_ab.sh`, `longctx.sh` | timings; the `ab` ones interleave two binaries to cancel drift |
| `docs_check.sh` | documentation drift: every switch the source reads is in `docs/test-hooks.md`, every in-tree path a document names exists, and nothing calls a live switch dead.  Runs without a model or a GPU |

These scripts existed since `14754be` but were missing from this map, which is
how a long investigation ended up hand-rolling probes they already contain.
When a numerics question comes up, read them first.

## Build

Build for the target and do not reuse a stale binary:

- Native GLES2: `make -C engine -B engine_mm3`
- Pi: cross-build with `make -C engine -B pi CROSS_SYSROOT=<sysroot>` on the
  cross host, or build on the Pi with the ARMv6 flags in the Makefile.
- Probes: `make -C engine -B probes/<name>[_pi]` — see the targets in
  `engine/Makefile`.

Do not use a short build timeout.  Machine-specific host/sysroot/deploy paths
belong in `AGENTS.local.md` (DSH) or `AGENTS.override.md` (Hermes/VLAB).

### Pi Zero reference (measured, not a target)

Measured on a Pi Zero, `GPU_LAYER=1 KV_PACK=16`, greedy, 32 generated tokens,
best of two runs; MiniMind3 uses `SEQ=768`.  Prefill is prompt-specific; the
full matrix and prefills are in [`docs/performance.md`](docs/performance.md),
and that file is the authority — this table is a pointer, not a re-baseline.

| model | decode |
|---|---:|
| MiniMind3 q8 | 252 ms/token |
| MiniMind3 q4 | 388 ms/token |
| MiniMind2 q8 | 144 ms/token |
| MiniMind2 q4 | 187 ms/token |
| MiniMind2 i16 | 232 ms/token |
| stories15M q8 | 138 ms/token |
| stories42M q8 | 188 ms/token |
| TinyChat15M q8 | 137 ms/token |

A native ARMv6 build on a Pi Zero takes about 8–10 minutes; never run two
builds concurrently.  Prefer the VLAB cross-build (see the VLAB override).

## The container is the only format

Since `fcc4095` the engine reads **only** MMCF v2.  `Model::load()` refuses
anything else and names the packer.  A `.q8` / `.q4` / `.i16` / `.bin` file in
`models/` is an *input to the packer*, never a model the engine can run:

```bash
# Straight from the source: the packer quantizes, so no intermediate .i16/.q8/
# .q4 file is written.  --quant is the precision, --seq the KV length.
python3 engine/tools/mmcf_pack.py --in models/minimind3 \
  --quant q8 --seq 256 \
  --tok models/minimind3/minimind3.tok \
  --out models/minimind3/minimind3.q8.mmcf \
  --name minimind3 --template minimind3 --chat \
  --chat-template minimind3/chat_template.jinja --verify
```

A legacy checkpoint (`Q8M1/Q8M2`, `Q4M1`, `I16M1/I16M2`) or a llama2.c `.bin`
is still accepted by `--in`; `--quant` then has no effect and the source's own
precision is kept.  The old `engine/quantize/*.py` converters are gone: they
only produced formats the engine cannot read.

The container carries the tokenizer, the chat template **text** (metadata
key 9, the same Jinja string a GGUF file holds in `tokenizer.chat_template`),
the default system prompt, the rope configuration, the BOS policy and the
special-token ids, so `ROPE_THETA=`, `ROPE_HALF=` and the engine's `--template`
are **gone**; setting them is silently ignored, which is worse than an error, so
a command copied from an old note still "works" and quietly tests something
else.

`TOKENIZER=` is the exception and the worse trap: it still works and it
**overrides** the embedded blob (`engine.cpp` only uses the container's
tokenizer when the variable is unset).  Nothing warns; the ids stop meaning what
the model was trained on and the reply comes back empty.  Leave it unset, and if
a run prints no `tokenizer: ... embedded in the checkpoint` line, that is why.
`docs/test-hooks.md` has the full list, including the three switches that used
to be undocumented (`FFN_PAIR`, `TOK_NO_PREFIX`, `TOK_SPACE_FOLD`).  Check a file with
`python3 engine/tools/mmcf_dump.py <file>` before blaming the engine.

## Canonical smoke tests

Run from `engine/`.  The system prompt is pinned on the command line: the
compiled default changed in `c1bb4f6` (Chinese persona → neutral English), so a
gate that omits `--system` silently changes meaning and stops reproducing.

q8:

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Offline q4 (do not set `Q4_WEIGHTS`):

```bash
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV_PACK=16 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Expected text.  The value is **machine specific** — VC4 and Xe round the fused
FFN differently — so never compare a Pi run with an Xe run:

- Pi Zero q8: `Assistant: 我是MiniMind，一个由jingyaogong`
- Pi Zero q4: `Assistant: 我是MiniMind，一个高效的小参数AI模型，专注于`
- Xe / desktop GLES2 q8: `Assistant: 我是一个AI助手，我的存在是为了提供帮助和信息。`

The strings are re-baselined whenever a deliberate numeric change moves them,
and each re-baseline cites its evidence.  The Pi q8 string last moved on
2026-11-09 with the integer q16 decode (`6234328`): the old decode rebuilt every
value from an 8-bit texture sample that VC4 does not return exactly, so the
whole board moved slightly, not only at the rail — the same change fixed
TinyBrainBot q4's residual sign flip (Pi-vs-Xe cosine 0.7233 -> 0.9971).  Before
that the Pi printed `我是一个AI助手，没有自我意识，也没有身体，所以我无法`
(2026-11-03, when the fused FFN became the default); `docs/performance.md` keeps
both, and the old binary is still runnable as
`bin/engine_mm3_pi.predecode` on the board.

**The string is a tripwire, not the acceptance criterion.**
`engine/scripts/regression.sh` is the gate: determinism (same binary twice),
a second prompt's behaviour (non-empty, no collapse), the pinned strings when
the machine sets them, and perplexity.  A deliberate numeric change
re-baselines `Q8_EXPECT` / `Q4_EXPECT` / `PPL_EXPECT` in the same commit and
cites the evidence — PPL on a corpus large enough to matter, the behaviour
checks, and where possible a cross-implementation check (llama.cpp on the same
prompt).  `scripts/regression.sh --rebaseline` prints the new values; keep them
in the machine's override, not here.

The same command **without** `--system` must print the same text; that is the
check that the compiled default has not moved.  If it differs, change
`kDefaultSystem`, this gate, `engine/scripts/regression.sh` and
`docs/performance.md` together, never one of them alone.

The older strings (`MiniMind是一个高效的小...` for q8,
`我是MiniMind，一个由jingyaogong...` for q4) were measured with the pre-`c1bb4f6`
Chinese default prompt.  They are still reproducible by passing that text to
`--system`; do not use them as the gate.  If a local q8 checkpoint is known to
differ, set `Q8_EXPECT` in the local override instead of weakening this gate.

## Debugging protocol

Pick the tool that answers the question before writing a new probe:

| Question | Tool |
|---|---|
| does this file say what I think it says? | `mmcf_dump.py` |
| are the tokenizer / template / rope the ones I meant? | `mmcf_dump.py`, then the engine's startup lines |
| is the conversion faithful? | `mmcf_pack.py --verify`, `LOGITS_OUT` + `cmp_logits.py` |
| is the source model sane? | `ref_forward.py` (safetensors) |
| which stage differs between CPU and GPU? | `diff_paths.sh` + `dump_diff.awk` |
| what does this GLSL function cost, or does it work at all? | `docs/vc4-glsl-costs.md`, then `probes/packunpack_probe` on the target |
| is a row saturating / clamping? | `dump_stats.py` on a `STAGE_DUMP` file |
| is one pass expensive? | `ATTN_DUP`, `PASS_DUP`, `DRAWCOUNT`, `LAYER_PROFILE` |
| is the attention block internally consistent? | `ATT_DUMP` + `check_att.py` |
| did my change alter numerics? | `regression.sh`, and PPL rather than one string |
| did my change move the emitted GLSL? | `scripts/shader_hashes.sh` (no GPU needed); `--dump-shaders` / `--dump-shaders=raw` to look |

`docs/test-hooks.md` lists the environment switches; `engine/src/*.cpp` is the
ground truth, and the inventory is one command:

```bash
grep -rho 'getenv("[A-Z_0-9]*")' engine/src | sort -u
```

## Measurement rules

These were paid for; do not relearn them.

- **Two builds add enough variance to hide a 1 ms effect.**  A/B inside one
  binary through an environment switch (`docs/test-hooks.md`), and interleave
  the runs (`bench_ab.sh`).
- **A CPU-vs-GPU stage delta is a lead, not a defect.**  The CPU path is f32
  that round-trips q16 only at the dump; `diff_paths.sh` says so in its own
  header, and on a *correct* model the per-layer cosine still falls to 0.46 by
  layer 7.  Only a **same-path** A/B (one flag, one kernel) decides.
- **A constant is not a uniform.** Replacing a GLSL literal with a uniform of
  the same value moved MiniMind3's PPL 270.3170 -> 269.9303 on Xe: the driver
  folds the literal differently.  Keep the literal unless the value really has
  to travel (metadata key 11 is the one place this was allowed to happen).
- **Verify a revert by hash, not by reading the diff.**  Keeping the shader
  uniforms and setting them to the old constant *looked* equivalent and
  reproduced the regressed numbers exactly; only the literal form restored the
  pin.  Compare `STAGE_DUMP` hashes and PPL against the parent commit.
- **A golden string is chaotic across GPUs and across reassociations.**  Pin it
  per machine, re-baseline deliberately, and let PPL decide.
- **Never compare a Pi run with an Xe run**, and never compare PPL across
  different `SEQ` values.
- **Never run the Pi suite while another GPU client holds the model** (the
  server, a benchmark); CMA is single-client on this board.

## Invariants

- **Container only**: the engine loads MMCF v2 and nothing else (see above).
- **Q4 draw** output is 24-bit; **Q4 residual-add** output is q16. Do not
  merge these encodings.
- **Q4 source**: quantize from F16/I16, not from q8. q8 → q4 is a second
  quantization and loses quality.
- **KV cache**: Pi and laptop caches are not byte-identical. Build disk cache
  on the target machine.
- **CMA**: `CmaFree` is actual occupancy; reducing `cma`/`gpu_mem` does not
  create free CMA.
- **Prefix cache** header flags include pack mode and `KV_SCALE`; loading with
  another `KV_PACK` or scale must be rejected.
- **FFN**: hidden 2432 is split into 1216-wide chunks because VC4 FBO width is
  2048. q4 slicing must copy block scale/min and repack nibbles.
- **The residual stream rails at ±128.**  It is q16 fixed point, so a model
  whose residual exceeds 128 cannot be represented on the default path
  (supra2-100M reaches 2300 at layer 4).  A per-layer power-of-two scale is the
  fix that fits; a single global factor cannot span the early layers'
  resolution and the late layers' range.
- **CPU weights** are borrowed from the checkpoint mapping by default
  (`MODEL_BORROW=0` restores the copy); keep `KEEP_CPU_WEIGHTS=1` only for
  debugging.

## Documentation rules

Root README = overview + quick start.  `engine/README.md` = engine quick start;
deep details go to `docs/`.  Keep English and Chinese quick starts in sync:
`README.md` / `README.zh-CN.md`, `engine/README.md` / `engine/README.zh-CN.md`,
`server/README.md` / `server/README.zh-CN.md`.

Update the document that owns the fact:

| Changing | Update |
|---|---|
| the container layout, metadata keys, capability bits | `docs/checkpoint-container.md`, `mmcf_dump.py`, `docs/model-formats.md` |
| a checkpoint format or the loader | `docs/model-formats.md` |
| a converter or the q4/q8 numerics | `docs/quantization.md` |
| a timing or a regression baseline | `docs/performance.md` |
| an environment switch | `docs/test-hooks.md` |
| a command a human will copy | the README for that component **and** this file |
| a GPU or format change | `docs/engine-internals.md`, `docs/engine-layout.md` |

A command in a document that no longer runs is a bug in the document, not a
detail.  `engine/scripts/docs_check.sh` is the mechanical half of that rule; it
is step 0 of `scripts/regression.sh`, and it runs in under a second: it compares
the `getenv()` inventory with
`docs/test-hooks.md`, checks that every `engine/tools`, `engine/scripts`,
`minimind3`, `server` and `templates` path a document tells you to run still
exists, and refuses to let a document call a switch dead while the source still
reads it.  Run it before committing a documentation change.

When a flag is removed, grep the whole tree for it: `--template` survived in
five scripts and two READMEs after the engine dropped it, and `docs_check.sh`
only catches the paths it knows about.

`docs/performance.md` is append-only history: new measurements go at the
bottom with the date and the machine, and the current state lives in its top
matrix.  Do not rewrite old entries; correct them with a new section that says
what was wrong.

## Source split policy

The engine is split into `engine/src/` modules; see `docs/engine-layout.md`.
Shader text belongs in `shaders.cpp` / `shaders_attn.cpp`, and activation
layout knowledge belongs in `xformat.h/.cpp`. Do not reintroduce per-shader
activation-layout strings. Run q8, q4 and `--ppl` comparisons after any GPU or
format change, and run `engine/scripts/regression.sh` (determinism, behaviour,
pinned strings, PPL).  A shader refactor also runs
`engine/scripts/shader_hashes.sh`: the emitted text is the numerics, so a
text-neutral change must leave `scripts/shader_hashes.txt` untouched.  A single
golden string is a tripwire, not the gate: it is chaotic across GPUs and across
reassociations, so pin it per machine and re-baseline it deliberately when the
numbers move on purpose.

## Local overrides

- DSH reads `AGENTS.local.md` in addition to this file.
- Hermes/VLAB reads `AGENTS.override.md`, which **fully replaces** this file;
  keep that copy self-contained.
- Both files are gitignored and must not be treated as portable project rules.
