# AGENTS.md

## Scope and precedence

This file is for agents changing this repository.  When two instructions
disagree, the higher one wins:

1. **The user's current, direct instruction.**  It outranks everything below:
   if the user says stop, stop.
2. **Platform policy** — the permission preset, approvals, and the sandbox.
   This file cannot grant access they refuse, and "keep going" is never a
   reason to work around an approval prompt.
3. **`AGENTS.override.md`** — the Hermes/VLAB copy.  It **replaces this file
   entirely** rather than adding to it, so on those hosts this file is not read.
   It exists on the VLAB host, not in this checkout; absent means "not that
   host", not "no override applies".
4. **`AGENTS.local.md`** — the DSH machine override.  It is read *in addition*
   to this file and owns host and machine facts: paths, sysroots, SSH aliases,
   pins, and local benchmark numbers.
5. **This file** — the portable rules and the component map.
6. **`README.md`, `docs/`, and the component READMEs** — the authoritative
   description of each subject; "Documentation rules" below says which one owns
   which fact.

Both override files are local and gitignored, so they are not portable project
rules in either direction: do not copy a host name, sysroot, deploy path or
local-only benchmark number from them into this file, and do not assume a
machine's pins or paths from this file.

**This file does not own facts.**  It says which component exists, what it is
for, and what a change to it affects.  The number, the field layout and the
exact check belong to the document named in "Documentation rules"; when this
file and that document disagree, that document is right and this file is the
bug.

**What "now" means in a sentence.**  A statement containing a date, "used to",
or a four-decimal measurement is a record of what was true then, not a rule
now.  Such a sentence earns its place only by carrying the commit that
established it; the record itself is Git, not a file in this tree.

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
| `engine/gate/` | the pass/fail checks that need no GPU: documentation, the switch registry, the shader digests, the q16 rail, the draw cap, the target ledger, the container layout.  `gate/README.md` is the map |
| `engine/scripts/` | oracles, measurement and acceptance, which need a GPU and/or a model: `regression.sh`, the benchmarks, `swap_matrix.py`, `check_att.py`.  `scripts/README.md` is the map |
| `engine/eval/` | model quality: `bench_mc.py`, `eval_ifeval.py`, `eval_translate.py`, `evalkit.py` and their `data/`.  `eval/README.md` is the map |
| `engine/tests/` | compiled tests: `test_lineedit.cpp`, `test_tokenizer.cpp`, `test_xformat.cpp` (`make -C engine test`) |
| `engine/probes/` | standalone VC4 feature/throughput probes (not linked into the engine); `README.md` is the map: one row per probe = the question it answers, its `_pi` target and where the answer is written down |
| `server/` | OpenAI-compatible HTTP server, deterministic tools, self-test, eval suite.  `server.py` is the entry point; `http_api.py` (routes/SSE), `engine_bridge.py` (the worker protocol), `chat_render.py`, `tool_calls.py` and `sampling.py` are the parts it was split into on 2026-11-14, all stdlib only |
| `models/` | central model store (gitignored), one `.mmcf` per precision |
| `minimind3/` | MiniMind3 source config, chat template, reference forward pass; artifacts live in `models/`.  `tokenizer.json` is **not** tracked (`.gitignore`): fetch it from the HF release into this directory before rebuilding a `.tok` |
| `refs/` | untracked upstream reference material for MiniMind3 (gitignored) |
| `docs/` | deep dives; keep READMEs short and link here |

Engine modules are described in [`docs/engine-layout.md`](docs/engine-layout.md);
the file format in [`docs/model-formats.md`](docs/model-formats.md) and
[`docs/checkpoint-container.md`](docs/checkpoint-container.md).

### `engine/tools/` — the container toolchain

| Tool | Use it when |
|---|---|
| `mmcf_pack.py` | converting an HF directory (safetensors + config) or a legacy checkpoint (Q8M1/Q8M2, Q4M1, I16M1/I16M2) into an MMCF container |
| `mmcf_format.py` | the layout in one place: magic, header, record format, TLV, CRC, the dtype/kind/template tables.  Imported by the three tools above rather than copied into each; `gate/test_mmcf_format.py` round-trips every container in `models/` through it and compares its constants with `engine/src/model.cpp` |
| `mmcf_dump.py` | answering "what does the engine actually read from this file" — the whole metadata block, from the file alone (`--check` verifies every CRC) |
| `store_audit.py` | the model store before a cleanup: every container with its version, size and tracked references, every source payload, every derived checkpoint (`--check` verifies CRCs) |
| `mmcf_requant_q8.py` | rebuilding the q8 matrices of a container from its i16 sibling (`--clip P` for a percentile scale) |
| `ref_forward.py` | a numpy forward pass over an HF safetensors source: separates "bad conversion / bad tokenizer / engine bug" |
| `make_tok.py` | building the `.tok` blob from a HF `tokenizer.json` — ByteLevel or SentencePiece, with added tokens laid onto their ids (a blob built from `model["vocab"]` alone ends at the wrong id and the engine splits `<\|im_start\|>` into characters), byte fallback written as raw bytes, and one id per byte string (`dedupe()`: a real piece beats a `<0xNN>` fallback, the lower id wins a tie) |
| `pth_reader.py` | reading a torch zip (`pytorch_model.bin`) without torch, for a release that ships no safetensors (the BitCPM family and anything older); `mmcf_hf.load_minimind` falls back to it, so `--in <dir>` works on the directory as downloaded |
| `make_tok_byte.py` | the same blob from a *pure byte* `byte_vocab.json` (259 entries, no merges — `void.1`): `<0xNN>` becomes the raw byte, not six characters. `make_tok.py` writes the text `<0x41>` for that vocab, which still gives every id a home, so nothing fails loudly |

The numpy half of the toolchain has a project environment: `uv run
engine/tools/mmcf_pack.py ...` resolves `pyproject.toml` + `uv.lock` and creates
`.venv/`; `uv sync` builds it without running anything.  `mmcf_format.py`,
`mmcf_dump.py` and `make_tok.py` are stdlib and run either way, and the gates
and the server stay stdlib-only: they must keep working with the system
`python3`, because CI and the board do not set up this environment.

### The four non-engine directories — split by what a file *needs*

| Directory | Rule | The map |
|---|---|---|
| `engine/gate/` | pass/fail checks that need **no GPU and no model store** (the source, the generated GLSL, the documents, a container file) | `gate/README.md` |
| `engine/scripts/` | oracles, measurement and acceptance: **a GPU and/or a model** | `scripts/README.md` |
| `engine/eval/` | **model quality** (accuracy, instruction-following, translation): the engine plus a container | `eval/README.md` |
| `engine/tests/` | **compiled** C++ tests, built by the Makefile | `tests/README.md` |

Each README's table is the map: one row per file with what it answers, what it
needs and what calls it, and `gate/check_scripts.py` fails the gate if a file has
no row or a row has no file.  That check exists because the failure mode is real:
AGENTS.md records a long investigation that hand-rolled a probe `engine/scripts/`
already contained, and before the split one directory held the gate, the
oracles, the benchmarks, the eval batteries, a C++ test and a doc generator.

**Read the map before writing a new script.**  A pass/fail check that needs no
GPU goes in `gate/` (`README.md` is the map, `AGENTS.md` the contract a new
check keeps); something that prints numbers for a human to read goes in
`scripts/`; anything that scores a model goes in `eval/`.

## Build

Build for the target and do not reuse a stale binary:

- Native GLES2: `make -C engine -B engine_mm3`
- Pi Zero, on the board: `make -C engine -B armv6` — the same flag set the cross
  target uses, with no sysroot.
- Pi Zero, cross: `make -C engine -B pi CROSS_SYSROOT=<sysroot>` on the cross
  host.  There is no default sysroot, and the recipe refuses to finish unless the
  artifact reports `Tag_CPU_arch: v6` / `Tag_FP_arch: VFPv2`: gcc silently
  ignores a `-B` directory that does not exist, then links the host's ARMv7
  start files and leaves a binary that faults on the ARM1176.  The sysroot and
  the compiler are prepared per `docs/cross-build.md` (the root filesystem must
  be Raspbian; Debian's armhf is ARMv7).
- Probes: `make -C engine -B probes/<name>[_pi]` — see the targets in
  `engine/Makefile`.
- A native ARMv6 build on a Pi Zero takes about 8–10 minutes; never run two
  builds concurrently.  Prefer the VLAB cross-build (see the VLAB override).
- Timings: [`docs/benchmarks.md`](docs/benchmarks.md) owns the matrix, the
  protocol and each row's conditions; this file does not repeat them.  The Pi
  rows are the **latest measurement per model**, each dated, not one campaign —
  a row is comparable to itself over time, not necessarily to its neighbour.
- Before pushing: `make -C engine check` — the no-GPU half in one command.  It
  builds and runs the compiled tests (`test_lineedit`, `test_tokenizer`,
  `test_xformat`), then `warncheck`, `docs_check.sh` (its sub-checks: the
  generated inventory, the switch registry, the doc style, the paths, the "gone"
  names, the four-directory maps and the links/index), `shader_hashes.sh`,
  `check_q16_decode.sh`, `check_draw_cap.sh`, `check_target_reset.sh`,
  `test_mmcf_format.py` and `server/selftest.py`.  It takes about a minute, and
  ~55 of those seconds are `warncheck`'s forced rebuild; the checks and the
  three test builds are the rest.  `-Wall -Wextra` is on by default; `warncheck`
  fails the target if
  `src/` produces any warning, and the eight warnings the vendored Jinja headers
  produce are filtered, not silenced.  `make -C engine test` is the same tests
  without the rebuild (a few seconds), for iterating on a unit.
- CI: `.github/workflows/gate.yml` runs that same `check` target on every push,
  so there is no second list to drift (the workflow listed the steps separately
  and had missed `check_q16_decode.sh` until 2026-11-17), plus the eval case
  file's `--selftest`, shell and Python syntax, and the tool CLIs.
  `test_mmcf_format.py` checks only the C++ constants there because `models/` is
  gitignored — it prints SKIP for the round trip when the store is absent.  The
  model-dependent half (golden strings, PPL,
  `swap_matrix.py`) needs the containers and a GL driver, so it stays local —
  run `scripts/regression.sh` before pushing anything that touches numerics.

Do not use a short build timeout.  Machine-specific host/sysroot/deploy paths
belong in `AGENTS.local.md` (DSH) or `AGENTS.override.md` (Hermes/VLAB).

## Git

`main` is the integration branch and published history is append-only.  The
canonical remote is protected (no force push); a mirror receives the same
`main`; a build checkout is updated by a push when it is reachable, or by
`git pull --ff-only` on the host.  Fetch before pushing; a rejected push is
merged or rebased locally, never forced.  Force-push only a branch nobody has
pulled.  The remotes' names and paths are machine facts: `AGENTS.local.md`.

A commit is the record (see "Where numbers and history go"), so its message is
part of the change:

- subject: `<area>: <what changed>`, one line, English, no period.  `docs(PAGE):`
  is the scoped form; the area is the subject, not a closed list.  There is no
  50/72 rule here -- long is fine when the change needs it.
- body: why, what it cost or broke, and the evidence that it works -- the gate
  result, the measurement, the re-baseline.  Wrap at ~80 columns, and stand
  alone: `git log -S` is how the next reader arrives.
- one coherent change per commit; squash WIP and fixup on your own branch
  before pushing.

Tracked: source, documents, tests.  Not model containers, binaries, build
outputs, caches, or local overrides -- `.gitignore` owns the patterns, and a new
artifact type gets a commented pattern there, not a commit.  Nothing secret:
assume the mirror is public.

Work that needs the hardware gate -- numerics, shaders, the loader, a container
layout -- or that spans more than one commit, starts on a branch and lands only
after `scripts/regression.sh` on the machine that owns the pin.  A single change
whose evidence is `make check` may go straight to `main`; that is the normal
case here.  Rebase onto `main` and fast-forward, or use `--no-ff` when each
commit is itself a measured step; delete the branch after the merge.  A pull
request is optional: use it when the host's CI is the cheapest way to get the
gate run, not as ceremony.

The pre-push gate is the one in Build; `git status` clean before pushing.
`git branch` is work in flight, not an archive.

### Switching models without restarting the engine

`--chat` without `--prompt` is a REPL, and `/model PATH` replaces the container
while keeping the GL context and its compiled programs (the Pi Zero pays 4.23 s
of context and 1.89 s of program links per process, against 6.12 s of container
read and 2.99 s of upload, so this is the one expensive part that survives).
What it replaces: weights, tokenizer, chat template, session targets, the
conversation.  What it refuses: a container that does not advertise chat support
when `--chat` is in effect (same policy as startup), and it refuses *before*
releasing anything — `Model::probe()` reads the header and the metadata only, so
a typo, a non-container or a completion model costs the session nothing.

Three rules, all of them paid for (the evidence is in
[`docs/swap-drift.md`](docs/swap-drift.md)):

- **The swap repeats every post-load step.**  `prepare_model()` (FFN split,
  hidden scale, quantisation) runs before the upload, for a swap and for
  startup, because the step slices Mats.  Skipping it still answers, with the
  whole-width FFN and no hidden scale.
- **A program key is not a program text.**  `x_prog(key, src)` caches by kernel
  name, so the cache validates the shader *hash* on every hit
  (`prog_key_hash_`); otherwise the second model runs the first model's kernels.
- **Everything the GPU owns is released *and* reset**, and
  `gate/check_target_reset.sh` fails the gate if a `Target` member is added
  without a line in `reset_targets()`.

**There is no capacity pre-check, deliberately.**  `CmaFree` is not the budget a
new GL scene gets — it is often less than what the driver can find — and a
desktop kernel has no CMA at all, so a size heuristic refused swaps that would
have worked.  A container that does not fit fails in the upload, which is where
the truth is.  The acceptance is `scripts/swap_matrix.py` (three template kinds,
both precisions, a round trip, an i16 and a hybrid chain; needs the GPU).

## The container is the only format

Since `fcc4095` the engine reads **only** the MMCF container family: v2 (dense),
v3 (mixture of experts) and v4 (RWKV-7).  `Model::load()` refuses any other
version and names the packer.  Because the version is dispatched on, a v2 reader
cannot half-load a v4 checkpoint.  A `.q8` / `.q4` / `.i16` / `.bin` file in
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

A mixture-of-experts checkpoint (`Qwen3MoeForCausalLM`: top-1, normalised
gate, no shared expert, every layer sparse) packs the same way and produces
**version 3**: 64-byte records with an `aux` expert index, metadata key 13, and
`0x16` router tensors.  Every other MoE shape is refused at pack time by name
(Mixtral, top-k>1, an unnormalised gate, a shared expert, mixed sparse/dense
layers).  The v2 reader is unchanged, so every existing container still loads:

```bash
python3 engine/tools/mmcf_pack.py --in models/minimind3-moe \
  --quant q4 --seq 768 --tok models/minimind3-moe/minimind3-moe.tok \
  --out models/minimind3-moe/minimind3-moe.q4.mmcf \
  --name minimind3-moe --template minimind3 --chat \
  --chat-template models/minimind3-moe/chat_template.jinja --rms-eps 1e-6 --verify
```

What it costs: the route is a 4x768 matvec whose logits are read back, so a
token pays 8 host syncs (~29 ms on a Pi Zero, ~19 ms on Xe); the weight traffic
is the dense model's, because only one expert is read.  The Pi Zero runs the
198M-A64M model at q4 (131 MB, 322.0 ms/token in the 2026-11-13 campaign; a
2026-11-11 24-token run gave 342).  At q8 it was **tried and failed** (204 MB:
the weights take the CMA and the session's FBOs fail) -- that is an observed
outcome on that board and configuration, not a size bound to apply elsewhere.
`docs/benchmarks.md` has the table; `MOE_DUMP=1` prints the routes and
`MOE_FIXED=<e>` prices them.

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
container supplies a default, so a run that omits `--system` silently tests that
prompt instead of the one the gate names, and stops reproducing.

q8:

```bash
MODEL=../models/minimind3/minimind3.q8.mmcf \
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Offline q4 (do not set `Q4_WEIGHTS`):

```bash
MODEL=../models/minimind3/minimind3.q4.mmcf \
GPU_LAYER=1 KV=q16 ./engine_mm3 --chat \
  --system "You are a helpful assistant." \
  --prompt "用一句话介绍你自己。" --gen 16 --greedy
```

Expected text.  The value is **machine specific** — VC4 and Xe round the fused
FFN differently — so never compare a Pi run with an Xe run:

- Pi Zero q8: `Assistant: 我是一个AI助手，没有自我意识，也没有身体，所以我无法`
  (`Q16_DECODE=exact` prints the older `我是MiniMind，一个由jingyaogong`)
- Pi Zero q4: `Assistant: 我是MiniMind，一个高效的小参数AI模型，专注于`
- Xe / desktop GLES2 q8: `Assistant: 我是一个AI助手，我的存在是为了提供帮助和信息。`

The strings are re-baselined whenever a deliberate numeric change moves them,
and each re-baseline cites its evidence.  The Pi q8 string last moved on
2026-11-09 with the integer q16 decode (`6234328`): the old decode rebuilt every
value from an 8-bit texture sample that VC4 does not return exactly, so the
whole board moved slightly, not only at the rail — the same change fixed
TinyBrainBot q4's residual sign flip (Pi-vs-Xe cosine 0.7233 -> 0.9971).  Before
that the Pi printed `我是一个AI助手，没有自我意识，也没有身体，所以我无法`
(2026-11-03, when the fused FFN became the default); it moved again on
2026-11-13, when the cheap q16 decode became the default (see the decode
section of `docs/benchmarks.md`).  `docs/benchmarks.md` keeps both.  A
pre-decode binary was kept on the board as `bin/engine_mm3_pi.predecode` when
the decode changed (2026-11-13); it is not in this repository, so treat it as a
board-local artifact and rebuild rather than count on it.

**The string is a tripwire, not the acceptance criterion.**
`engine/scripts/regression.sh` is the gate: determinism (same binary twice),
a second prompt's behaviour (non-empty, no collapse), the pinned strings when
the machine sets them, and perplexity.  A deliberate numeric change
re-baselines `Q8_EXPECT` / `Q4_EXPECT` / `PPL_EXPECT` in the same commit and
cites the evidence — PPL on a corpus large enough to matter, the behaviour
checks, and where possible a cross-implementation check (llama.cpp on the same
prompt).

**A gate with nothing to compare is not a gate.**  `regression.sh` fails when a
pin is unset — `q8 FAIL: no pin.  export Q8_EXPECT=...` — and `ALLOW_UNPINNED=1`
is the only way past it, because a run that prints `ACCEPT` having compared
nothing is counted as coverage it does not have (2026-11-17; it did exactly that
before).  `scripts/regression.sh --rebaseline` blanks the pins, measures, and
prints the block to paste into the machine's override: it no longer compares
against the values it is about to replace, which used to make it exit at the
first mismatch and never print anything.

The same command **without** `--system` must print the same text; that is the
check that the container's default system prompt (metadata key 4, alongside the
u8 template kind and the template name; key 6 is the model name) and the packer
that wrote it agree with the gate.  There is no compiled default any more --
`kDefaultSystem` was dead code and is gone since 2026-11-14; the engine uses the
container's prompt and `--system` overrides it.  If it differs, change the
container, this gate, `engine/scripts/regression.sh` and `docs/benchmarks.md`
together, never one of them alone.

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
| does a new sampler / texture fetch work on the target? | `probes/wrow_probe` (and `_pi`): drives the engine's own q8 kernels against a CPU reference and names the rows that differ.  It is how the per-row scale plane's two fetches from one sampler were found to be undefined on VC4 |
| does the conv (hybrid) block's fetching work? | `probes/conv_probe` (and `_pi`): section A drives the kernel fetch patterns byte-for-byte, B/C run the shipped `conv3_act`/`mul_act` against a CPU reference.  It is how "three fetches from one sampler" was cleared on VC4 -- the kernel texture is 512x3, and every case is exact |
| did my change alter numerics? | `regression.sh`, and PPL rather than one string |
| did my change move the emitted GLSL? | `gate/shader_hashes.sh` (no GPU needed); `--dump-shaders` / `--dump-shaders=raw` to look |

`docs/test-hooks.md` lists the environment switches and
[`engine/src/switches.cpp`](engine/src/switches.cpp) is the inventory: one row
per switch with its canonical name, the spellings it replaced, its type, its
closed set of values, whether it is permanent or a probe, and the axis it moves.
`./engine_mm3 --dump-switches` prints that table with what the environment
resolved to (no GPU, no model), and `gate/check_switches.py` (step 0 of the
gate) fails if the source reads a name with no row, a row is read nowhere, a
retired name is read again, or a row is missing from the document.

A switch names an **axis**, not a module: `ACT=i16|i8|f32`, `KV=i24|q8|q16|bf16|f32`,
`RES=i16|i16_1|i24|bf16|f32`, `FFN_HID=i16|i8|bf16|f32`, `ACT_CPU=q16|f32`.  The spellings they
replaced (`ACT_FMT`, `ACT8`, `ACT_F32`, `KV_PACK`, `KV_FMT`, `KV_SCALE`,
`RES_FMT`, `RES_PACK`, `ATT_SCALE`, `ATT_SUM_SCALE`) still work — the registry
rewrites them and says so on stderr once — but new code and new documents use the
axis name.  A closed set is validated at startup: a typo stops the run instead of
falling back to a default.

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
- **Decode is the reproducible number; prefill is not.**  Identical MiniMind3 q4
  runs gave 16.0 and 35.9 s prefill (background load 0.9 against 2.4) while
  decode stayed inside ±1%, because the board's SD card and the resident
  `pumpkin` move prefill.  Quote prefill only with its run, and never compare it
  across sessions.
- **Never compare a Pi run with an Xe run**, and never compare PPL across
  different `SEQ` values.
- **A window that starts mid-text is cold, and a cold window is not a
  perplexity.**  Cut 512 tokens out of the middle of a document and LFM2.5 pays
  about 1.8 nats/token more than the same tokens with their context (float32
  numpy: 3.80 with the document in front against 5.61 without), so a sum of
  cold windows measures the resets, not the model.  `--ppl` warms each window up
  since 2026-11-12 (`PPL_STRIDE=n`, default `SEQ/2`); `PPL_STRIDE=0` restores the
  old cold windows and the two numbers are not comparable.
- **Never run the Pi suite while another GPU client holds the model** (the
  server, a benchmark); CMA is single-client on this board.

## Invariants

- **MoE is top-1 only, and every expert is resident.**  A v3 container carries
  one router plus `n_experts` w1/w3/w2 triples per layer; the runtime routes on
  the host (one read-back per layer per token) and runs the dense FFN kernels on
  the chosen expert, with weight 1.0 because the gate is normalised.  A
  top-k>1 or shared-expert block is refused, never emulated, and the pair
  prefill path is refused because two positions route independently.

- **Container only**: the engine loads the MMCF family (v2, v3, v4) and nothing
  else (see above).
- **Q4 draw and q4 residual-add do not share a layout.**  A matvec writes its
  output in the *activation* format (`act_fmt_`, the `ACT` axis) and an
  add-into-residual writes in the *residual* format (`res_fmt_`, the `RES`
  axis), so the two are different textures whenever those axes differ.  On the
  shipped q4 containers that is 24-bit (I24_1) for the draw and the packed q16
  pair (I16_2) for the add.  Do not merge the two encodings, and do not assume
  one axis's format for the other.
- **Q4 source**: quantize from F16/I16, not from q8. q8 → q4 is a second
  quantization and loses quality.
- **KV cache**: Pi and laptop caches are not byte-identical. Build disk cache
  on the target machine.
- **CMA**: `CmaTotal` is the hard ceiling; `CmaFree` is **not** an occupancy
  figure.  It reports how much a caller can *probably still get*, it moves with
  page-cache migration rather than with the allocation just made, and on this
  driver the deltas are noise: `Q4_TEX_STATS` stopped reporting a `CmaFree`
  delta because four consecutive uploads of the same shape read 2020, 2436,
  4576 and 4544 kB, and an RGBA8 upload once *raised* it.  So
  `CmaFree` is neither "free space" nor a budget for a new scene, and refusing a
  load on its arithmetic is the mistake that a size heuristic made before swaps
  lost their pre-check.  The board's usable figure is empirical (the notes use a
  320 MB `CMA` configuration, against a 256 MB one that cannot hold the same
  model), and near the ceiling the answer comes from **attempting the load**,
  which is where the failure is real -- not from asserting that it will not fit.
  Reducing `cma`/`gpu_mem` does not create CMA.
- **Prefix cache** header flags include the KV axis and its grid; loading with
  another `KV` value or `KV_GRID` must be rejected.
- **FFN**: the hidden width is split in **two** chunks and no more
  (`split_ffn`), because the kernels carry two chunk slots; a hidden wider than
  `2 * chunk_limit` is refused by name rather than split again.  The limit is
  the GL target width (2048 on VC4, so 4096 there; passed in for Xe, 4608 for
  LFM2-350M), and the chunk is `(hidden + 1) / 2` -- 1216 for MiniMind3's 2432.
  A MoE expert hidden has its own 4096 ceiling.  q4 slicing must copy block
  scale/min and repack nibbles.
- **The rms partial sums saturate at 2048.**  They travel in 24-bit fixed point
  with a fixed 1/4096 step biased by 2^23, so a lane sum above 2048 pins and
  `inv_rms` comes out wrong for every deeper layer -- silently, on both drivers.
  MNN-Pocket reaches sum-of-squares 3.9e6 by layer 15 (24-bit store reads
  1.07e6, 73% low); `RMS_STORE=f32` stores the same four bytes as a bit-cast
  float and fixes it.  Default stays i24 because MiniMind3 was baselined in the
  saturated state.
  **A residual above ~11 saturates it by itself.**  The sum is a squared
  quantity: a lane holds `dim/32` values, so its sum is at most
  `dim/32 * peak^2`, and on a 512-wide model 2048 is reached at a peak of 11.
  SmolLM2's 26884 is 6e6 times over, and the reply comes back *empty* -- not
  truncated, empty.  The packer prints the bound at pack time; `RMS_STORE=f32`
  is the fix (2026-11-18).
- **The residual stream rails at ±128.**  It is q16 fixed point, so a model
  whose residual exceeds 128 cannot be represented on the default path
  (supra2-100M reaches 2300 at layer 4).  A per-layer power-of-two scale is the
  fix that fits; a single global factor cannot span the early layers'
  resolution and the late layers' range.
- **CPU weights** are borrowed from the checkpoint mapping by default
  (`MODEL_BORROW=0` restores the copy); keep `KEEP_CPU_WEIGHTS=1` only for
  debugging.

## Model-specific constraints

Four models carry constraints the dense path does not, and a change that moves
them has to be checked against the named document rather than against MiniMind3.
Each document owns the measurements; this section keeps only what a change must
not break.

| Model | What it forces | Read |
|---|---|---|
| LFM2.5-230M | the only hybrid (conv + attention): metadata key 14, the conv tensors `0x17`-`0x19`, the fused `GPU_LAYER=1` conv path, and a template that supplies its own `<\|startoftext\|>`.  Still broken: an i16 container on the fused path, a prompt longer than `SEQ`, digit-run tokenisation | [`docs/lfm2.5-230m.md`](docs/lfm2.5-230m.md) |
| SmolLM2-135M | `RMS_STORE=f32` is **required**: the residual reaches 2.7e4 and a lane sum of `x^2` pins the 24-bit store at 2048, after which the reply comes back empty | `models/README.md`, "Two limits, measured" |
| BitCPM-CANN-0.5B | the one ternary model: 168 of 169 matrices are per-row `{0, ±s}`, so its q8 body is lossless while `q4mat` is not.  Also the only container with a rope table (metadata key 18), the only one needing the tokenizer's per-run space marker (metadata key 10's third byte), and the only one whose 2-bit body (dtype 7, `--quant t3`) fits the board at all.  `RMS_STORE=f32` is required | [`docs/bitcpm.md`](docs/bitcpm.md) |
| RWKV-7 | the one recurrent model: a v4 container (key 16, key 17), no KV cache, no RoPE, `SEQ` deliberately unclamped, and **a separate PPL pin per layer loop** (CPU against `RWKV_GPU=1`) because the two are not comparable by string.  `pair_ok()` and the prefix cache both refuse it.  Its speed work is **closed**: the floor is the q8 kernel itself, so re-open it only with a measurement that contradicts that page | [`docs/rwkv7.md`](docs/rwkv7.md) |

The BitCPM pack-time constants are load-bearing: `k = 1/scale_emb` is the
choice that keeps the tied head in range, and `k = 1` destroys it.

**A Pi answer is not the Xe answer for SmolLM2-135M.**  The two machines
disagree on logits and on the greedy answer, the format axes are excluded as the
cause (widening every stream to f32 makes the board worse), and the cause is
open.  `MATVEC_STRIP` is the one measured lead; the per-switch isolation table is
in `models/README.md` and both machines are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Documentation rules

Root README = overview + quick start.  `engine/README.md` = engine quick start;
deep details go to `docs/`.  Keep English and Chinese quick starts in sync:
`README.md` / `README.zh-CN.md`, `engine/README.md` / `engine/README.zh-CN.md`,
`server/README.md` / `server/README.zh-CN.md`.

**The READMEs are for humans; `AGENTS.md` files are for agents.**  Before
writing or rewriting a page in `docs/`, read [`docs/AGENTS.md`](docs/AGENTS.md):
it carries what this section does not — where a page goes, its sections in
order, its length, and the cross-file bookkeeping a new model needs.

Update the document that owns the fact:

| Changing | Update |
|---|---|
| the container layout, metadata keys, capability bits | `docs/checkpoint-container.md`, `mmcf_dump.py`, `docs/model-formats.md` |
| a checkpoint format or the loader | `docs/model-formats.md` |
| a converter or the q4/q8 numerics | `docs/quantization.md` |
| a timing or a regression baseline | `docs/benchmarks.md` |
| an environment switch | its row in `engine/src/switches.cpp` **and** `docs/test-hooks.md` |
| a command a human will copy | the README for that component **and** this file |
| a GPU or format change | `docs/engine-internals.md`, `docs/engine-layout.md` |

**A document records the present, or it records the past — never both.**  Two
tiers, and a sentence belongs to exactly one of them:

| Tier | Answers | Lives in | May contain |
|---|---|---|---|
| state | what is true now | the READMEs, `docs/*.md`, `--help`, `--dump-switches` | commands in the canonical spelling, no dates, no "used to" |
| evidence | why it is this way, and not the other way | one paragraph next to the thing it protects | one dated line and a commit reference (`git log -S` finds it) |

**There is no history tier.**  Git is the record: what was tried, measured and
abandoned lives in the commit that did it, and a document that needs to say why
points at that commit instead of retelling it.  Nothing in this tree is
append-only, and no file is a log — a measure-and-append document grows without
bound and buries the present under its own past.

The test for a sentence: delete it — does someone make a wrong decision
tomorrow?  No → it is history, and it belongs in a commit message.  Yes → keep
it as one dated line that names the commit which established it.
`engine/gate/check_docs_style.py` is the mechanical half: a **command** in a
state document uses the axis spelling (`KV=q16`, not `KV_PACK=16` — the alias
still works, but the document is the teaching copy), and a **four-decimal
number** in a state document sits under a dated heading or in a dated paragraph,
because PPL is re-baselined deliberately and the current pins are in the
machine's override, not in prose.  It is step 0 of the gate.

A command in a document that no longer runs is a bug in the document, not a
detail.  `engine/gate/docs_check.sh` is the mechanical half of that rule; it
is step 0 of `scripts/regression.sh`, and it runs in under a second: it compares
the switch registry (`engine/src/switches.cpp`) with the source and with
`docs/test-hooks.md`, checks that every in-tree path a document tells you to run
still exists (`engine/tools`, `engine/scripts`, `engine/probes`, `engine/src`,
`engine/*.cpp`, `minimind3`, `server`, `templates`), and refuses to let a document
call a switch dead while the source still reads it.  The three directions that
need slug matching, relative resolution or git live in
`gate/check_docs_links.py`, called from the same step: every relative link and
`#anchor` resolves, every `docs/README.md` entry exists and is tracked, and every
switch `test-hooks.md` documents is read by `engine/src` -- that last one is the
direction that would have caught `CONV_GPU`.  Run it before committing a
documentation change.

When a flag is removed, grep the whole tree for it: `--template` survived in
five scripts and two READMEs after the engine dropped it, and `docs_check.sh`
only catches the paths it knows about.

### Where numbers and history go

Five kinds of content were once mixed in one dated file.  They have different
homes, and the rule is by kind, not by recency:

| content | home | what it looks like |
|---|---|---|
| a decode/PPL number that will be compared later | a row in `docs/benchmarks.md` | replaced when superseded, the commit named next to it |
| a load-bearing constant or format (rms 2048 rail, `k = 1/scale_emb`) | the owning document's invariant **and** a comment at the definition in `engine/src` | one sentence, no history |
| why a change was made, what it cost, what it broke | the commit message | as long as it needs to be |
| an approach that was tried and rejected | the commit message; a document may also keep a bounded "rejected" list **per topic**, never per date | one line and the commit |
| the sequence of a long investigation | commit messages, and `git log -S` / `git log --follow -- <path>` | not in a file |

**A value that will be compared over time must be machine-readable.**  A prose
table has to be re-typed to answer "did this move?", so a number that matters
belongs in a form a script can read.  `engine/scripts/bench.sh` still prints only
human-readable lines; until it grows a machine-readable mode, treat a new
campaign as replacing the table row and record the raw runs in the commit
message, not as another dated section.

**Never reintroduce a dated log.**  A file that is appended to becomes an
archive, and the archive buries the present state it was meant to serve.  If a
fact is worth keeping, it is worth putting where a reader will look for it: the
table, the invariant, or the commit.

`docs/benchmarks.md` currently holds the distilled reference table plus the
removed record's pointer (`git show e7a36ec^:docs/performance.md` has the 120
dated sections verbatim).  Do not grow it back.

## Source split policy

The engine is split into `engine/src/` modules; see `docs/engine-layout.md`.
Shader text belongs in `shaders.cpp` / `shaders_attn.cpp`, and activation
layout knowledge belongs in `xformat.h/.cpp`.  Environment switches are read
through `env_set` / `env_flag` / `env_int` / `env_double` (`common.h`), not by
calling `getenv` twice in one expression (`getenv(X) ? atoi(getenv(X)) : d` is
both slow in a per-layer loop and the shape where the two reads can disagree);
`env_flag` means "set and not 0", which is the convention `docs/test-hooks.md`
documents.  Every one of them has a row in
`engine/src/switches.cpp` — that table is the inventory, the place a rename is
declared and the place a probe is given an owner — and a switch read through the
registry's own `sw_enum`/`sw_str` accessors is seen by the same checks. Do not reintroduce per-shader
activation-layout strings.  JSON (the worker event stream, tool arguments and
results, the chat-template context) is built with `common_json` from
`third_party/jinja/json.h` and never by concatenation: it keeps key insertion
order, which the worker protocol's byte-compatibility relies on, and its pimpl
keeps `nlohmann` in `json.cpp`.

Run q8, q4 and `--ppl` comparisons after any GPU or
format change, and run `engine/scripts/regression.sh` (determinism, behaviour,
pinned strings, PPL).  A shader refactor also runs
`engine/gate/shader_hashes.sh`: the emitted text is the numerics, so a
text-neutral change must leave `gate/shader_hashes.txt` untouched.  A single
golden string is a tripwire, not the gate: it is chaotic across GPUs and across
reassociations, so pin it per machine and re-baseline it deliberately when the
numbers move on purpose.
