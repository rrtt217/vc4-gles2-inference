# LFM2.5-230M

The only **hybrid** model this engine runs: 8 depthwise-conv layers interleaved
with 6 full-attention layers.  It is why the engine has a conv block, why
`--ppl` had to learn what a window is, and why the activation format grew a
per-tensor scale.  This page is the operational record — what the model is,
which containers exist, how to run it, what the numbers are, and what is still
broken.  The packing/feasibility study (residual amplitudes, quant recipes,
vendor GGUF analysis, work estimate) is
[`lfm2.5-230m-assessment.zh-CN.md`](lfm2.5-230m-assessment.zh-CN.md); the
chronological measurements are in [`performance.md`](performance.md).

## The model

| | |
|---|---|
| layers | 14: `conv conv attn conv attn conv attn conv attn conv attn conv attn conv` (metadata key 14) |
| dim / ffn | 1024 / 2560 (SwiGLU, split into two 1280-wide chunks) |
| attention | 16 q heads, 8 kv heads, head_dim 64, GQA, per-head q/k RMSNorm, RoPE θ=1e6 rotate-half |
| conv | depthwise, causal, k=3, two rows of state per layer |
| vocab | 64402 (byte-level, `<\|startoftext\|>`=1, `<\|im_end\|>`=7), tied head |
| template | vendor ChatML variant, 4762 B, starts with `{{- bos_token -}}` |

A conv layer is `gated = (W_b x) * (W_x x)`, then a 3-tap causal depthwise
convolution over `gated`, then `y = (W_c x) * cv`, then `W_out y`; the three
`W_*` slices come from one 3072-row `in_proj`.  The state and the dataflow are
in [`engine-internals.md`](engine-internals.md) under "Hybrid (conv) block".

## Containers

| file | size | what |
|---|---:|---|
| `qad/lfm2.5.qad.q4.mmcf` | 169 MB | QAD weights, q4, `--head-dedup`, `SEQ 512` — **the container the board runs** |
| `lfm2.5.q4.dedup.mmcf` | 169 MB | base weights, same flags |
| `lfm2.5.q4.q6.mmcf` | 152 MB | base weights, q4 + 6-bit head (`--head-q6`) |
| `LFM2.5-230M-QAD-Q4_0.gguf` | 149 MB | the vendor file; the reference llama.cpp is compared against |

Sources kept beside them: `model.safetensors` (base, 459 MB),
`qad/model.safetensors` (918 MB), `lfm2.5.tok`, `tokenizer.json`,
`chat_template.jinja`, `config.json`.  The tiled-head q4 container and the two
i16 reference containers were deleted on 2026-11-11 to reclaim 1.6 GB; rebuild
them with

```bash
# base, q4, tiled head (no --head-dedup)
python3 engine/tools/mmcf_pack.py --in models/lfm2.5-230m --quant q4 --seq 512 \
  --tok models/lfm2.5-230m/lfm2.5.tok --out models/lfm2.5-230m/lfm2.5.q4.mmcf \
  --name lfm2.5-230m --template chatml --chat \
  --chat-template models/lfm2.5-230m/chat_template.jinja \
  --system "You are a helpful assistant trained by Liquid AI." \
  --bos 1 --eos 7 --stop 7 --add-bos 0 --add-eos 0 --verify
# the i16 reference (15.97 PPL below) is the same line with --quant i16
```

The QAD container's recipe is quoted in the assessment doc; it differs only in
`--in models/lfm2.5-230m/qad` and the name.

## Running it

The container states `add_bos 0` on purpose: the **template** carries
`<|startoftext|>`, so the tokenizer must not add a second one.  It also carries
the vendor's default system prompt, so `--system` is optional; pin it when the
run is a comparison.

```bash
cd engine
# laptop: fused (GPU conv + GPU attention + texture residual), ~33 ms/token
MODEL=../models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf SEQ=512 \
GPU_LAYER=1 CONV_GPU=1 ./engine_mm3 --chat \
  --system "You are a helpful assistant trained by Liquid AI." \
  --prompt "What is the capital of France?" --gen 32 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05   # see the sampling note below

# laptop: serial (CPU attention/conv, GPU matvecs) — the reference path
MODEL=../models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf SEQ=512 GPU_LAYER=0 \
  ./engine_mm3 --prompt "The capital of France is" --gen 24 --greedy

# Pi Zero (runtime dir, no checkout): SEQ=256 fits the CMA
MODEL=models/lfm2.5-230m/lfm2.5.qad.q4.mmcf SEQ=256 KV_PACK=16 \
GPU_LAYER=1 CONV_GPU=1 ./bin/engine_mm3_pi --chat \
  --system "You are a helpful assistant trained by Liquid AI." \
  --prompt "What is the capital of France?" --gen 32 --greedy
```

* `CONV_GPU=1` is what puts the conv block on the GPU; without it the serial
  path runs the conv on the CPU.  The serial path is the reference (see the gap
  below), the fused path is the fast one.
* `KV_PACK=16` is the hybrid default.  The 24-bit KV layout is **wrong for this
  shape** (kv_dim 512, head_dim 64) and the fused path answers garbage with it.
* Sampling: the model card and the probe agree on low temperature.  `--temp 0.1
  --top-k 50 --repeat-penalty 1.05` answered the factual prompts correctly where
  `--temp 0.7` drifted; the raw log of that sweep is
  `models/lfm2.5-230m/sampling.tsv`.

## Quality

`data/ppl_test.txt`, `SEQ=512`, 334 scored tokens, **with the BOS** (see
"history" below — every LFM2.5 number published before 2026-11-11 was measured
without it and is 5× too high):

| path | PPL | note |
|---|---:|---|
| float32 numpy, same safetensors, same ids | **14.71** | the reference |
| base i16, serial | 15.97 | container deleted; rebuild command above |
| QAD q4, serial | **18.73** | |
| base q4 dedup, serial | 19.87 | |
| base q4 + 6-bit head, serial | 19.07 | a 333-token corpus cannot resolve a 4% gap — the head choice is open, the old "q6 costs 14%" was a BOS artefact |
| QAD q4, fused | **19.52** | 4% worse than serial, as physics expects |

So the engine is within ~9% of the float32 reference on its own token stream,
and the fused path costs another 4%.

The sharpest single check is the first token after `assistant\n` for
"What is C. elegans?", because the reference and llama.cpp agree on it:

| | token | logprob |
|---|---|---|
| float32 numpy | 544 `C` | -0.042 |
| llama.cpp b12283, QAD-Q4_0 | 544 `C` | -0.037 |
| this engine, QAD q4, fused | 544 `C` | -0.02 |

Board (Pi Zero, `SEQ=256`, QAD q4, 24 tokens, greedy):

| | answer |
|---|---|
| What is C. elegans? | "a small, round, worm-like organism that is commonly used in research to study genetics" |
| Write a haiku about rain. | "Rain whispers softly," |
| 用一句话介绍你自己。 | "我是一个AI助手，帮助您理解和交流。" |
| 17\*23 with the `calculate_math` tool | **"Answer: 391"** |

Board PPL at `SEQ=256`, same corpus (two windows, not comparable with the
`SEQ=512` rows above).  The two rows this page used to carry -- fused **45.52**,
serial **43.91** -- are *cold-window* sums, the only protocol the board binary
had; both protocols now exist, and the cold one still prints them on the board
bit for bit (`43.9130` and `45.5248`), so the new build did not move that
measurement:

| Pi Zero, QAD q4, `SEQ=256`, 334 scored tokens | cold (`PPL_STRIDE=0`) | warm (default, `stride=128`) |
|---|---:|---:|
| serial (`GPU_LAYER=0`) | 43.9130 | **28.6196** |
| fused (`GPU_LAYER=1 CONV_GPU=1`) | 45.5248 | 31.6824 |
| Xe, same corpus, serial | 42.63 | 28.82 |
| Xe, same corpus, fused | 45.00 | 30.10 |

Two things to read there.  The protocol alone moves the number ~35% (43.91 ->
28.62 on the board), which is the correction recorded in
[`performance.md`](performance.md) -- compare warm with warm or cold with cold,
never one with the other.  And the fused path's gap is protocol-dependent on the
board: 3.7% behind serial cold, 10.7% behind warm (Xe: 5.5% / 4.4%), so the
board's older "3.7% behind" understated what the fused path costs on this
machine.

### History: what was wrong before 2026-11-11

Two engine bugs, both found by chasing a phantom 5× gap against llama.cpp, and
one bad measurement:

1. **Chat prompts had no BOS.**  All three render sites passed `""` for
   `bos_token`, and the template is *supposed* to supply `<|startoftext|>`; on
   the board the model then echoed the Chinese prompt verbatim and called
   C. elegans a beetle.  Fixed in `d802cb8`/`c92b895`; the strings also had to
   be marked as template text so they tokenise to the special id rather than to
   `<`, `|`, `start`, `oftext`.
2. **`--ppl` sent no BOS.**  No template is involved in a perplexity run, so
   nothing supplied it: 98.10 without against 15.97 with (float32 reference
   85.44 against 14.71).  Every LFM2.5 PPL in this repository older than
   2026-11-11 is therefore 5× too high, and one conclusion inverted with it
   (the fused path is now correctly *worse* than serial).
3. **A cold window is not a perplexity.**  `--ppl` scored every token of every
   window from a fresh `Session`, so each window after the first had no context
   and the total looked like state carried across windows (NLL 4.23 / 5.88 /
   7.41 against llama.cpp's 3.96 / 3.88).  The float32 reference settles it:
   5.61 for that text cold against 3.80 with the document in front, i.e. the
   model, and the engine tracks f32 by the same 0.3 nats (its q4 container)
   under cold, warm and single-window protocols alike.  `--ppl` now feeds `SEQ`
   tokens per window and scores only the last `SEQ/2`.

On the way, `GPU::matvec_s()` was added: a matvec input gets a power-of-two
scale from its own maximum before it is uploaded, because this model's conv
gate output at the start of a sequence is order 1e-4 and the q16 grid
(step 1/256) sent it to zero.  Layer-by-layer agreement with the float32
reference went from 6–11% to 0.2–1.4%; MiniMind3's PPL is bit-identical with the
change, because for well-scaled activations it is a no-op.

## The chat template

The container carries the vendor template byte for byte (4762 B) — compare it
with the GGUF's copy with

```bash
# the container's copy
python3 engine/tools/mmcf_dump.py models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --template 2>/dev/null > /tmp/tmpl_container.jinja
# the vendor's copy, out of the GGUF key block
python3 - > /tmp/tmpl_gguf.jinja <<'EOF'
import struct, sys
b = open('models/lfm2.5-230m/LFM2.5-230M-QAD-Q4_0.gguf', 'rb').read(); o = 0
def take(n):
    global o
    v = b[o:o+n]; o += n; return v
def u32(): return struct.unpack('<I', take(4))[0]
def u64(): return struct.unpack('<Q', take(8))[0]
def s(): n = u64(); return take(n).decode()
def val(t):
    if t in (0, 1, 7): return take(1)[0]
    if t == 2: return struct.unpack('<H', take(2))[0]
    if t == 3: return struct.unpack('<h', take(2))[0]
    if t == 4: return struct.unpack('<I', take(4))[0]
    if t == 5: return struct.unpack('<i', take(4))[0]
    if t == 6: return struct.unpack('<f', take(4))[0]
    if t == 8: return s()
    if t == 9:
        at, n = u32(), u64(); return [val(at) for _ in range(n)]
    if t == 10: return u64()
    if t == 11: return struct.unpack('<q', take(8))[0]
    if t == 12: return struct.unpack('<d', take(8))[0]
    raise SystemExit('value type %d' % t)
take(4); u32(); nt = u64(); nkv = u64()
for _ in range(nkv):
    k = s(); v = val(u32())
    if k == 'tokenizer.chat_template': sys.stdout.write(v)   # no trailing newline
EOF
cmp /tmp/tmpl_container.jinja /tmp/tmpl_gguf.jinja && echo identical
```

What was verified on 2026-11-11:

* **renders** — six cases (plain, no system, tools, multi-turn, a `<think>`
  block in history, a tool round-trip) render **byte-identically** to real
  Jinja2, once the `{% generation %}` tag (a llama.cpp extension upstream Jinja2
  lacks) is registered as a no-op and a missing key is allowed to be nil (the
  vendor template does `message.thinking or …` on maps that may not have it).
* **token ids** — the same six cases produce the **same ids** as llama.cpp's
  `/apply-template` + `/tokenize`: 31/31, 16/16, 110/110, 52/52, 140/140.  The
  `<think>` case differs by design: llama.cpp enables `preserve_thinking` by
  default (it detects the variable), this engine keeps the template's own
  default (`false`) unless a message carries reasoning or `--thinking` is
  given.
* `tojson` spacing matches: llama.cpp's filter uses Python's `", "` / `": "`
  separators, and so does the vendored copy.

The tool form the model was trained on is the template's: tools are appended to
the **system** message as `List of tools: [<json>, …]`, and the model answers
`<|tool_call_start|>[name(arg="value"), …]<|tool_call_end|>`.  Both the parsing
and the object-shaped `arguments` that the template's `func_args.items()` needs
are described in [`deterministic-tools.md`](deterministic-tools.md).

## Known gaps

| # | what | status |
|---|---|---|
| 1 | **prompt longer than `SEQ` corrupts the heap** — prints `[context full]` then aborts at exit (`corrupted size vs. prev_size`); pre-existing at `d802cb8`, most likely the pair prefill writing rows `pos, pos+1` | not fixed; treat `SEQ` as a hard prompt limit |
| 2 | **i16 container + fused path is wrong**: PPL 1.4e7 against 15.97 on the serial path; pre-existing at `f22bb76` | not fixed; the i16 container is serial-only |
| 3 | **tokenizer splits digit runs differently** from llama.cpp (`6789` → `6`+`789` here, `678`+`9` there, because `tokenizer.ggml.pre = lfm2` means the llama-3 pre-tokenizer with `\p{N}{1,3}`); prose is identical (334/335 ids on the PPL corpus), 5 of 20 probe strings differ | not fixed; matters for arithmetic prompts |
| 4 | **the fused path's I16_2 targets have no per-tensor scale** (`matvec_s` is serial-only), so the board's fused quality is 3.7% behind its own serial path | next step |
| 5 | ~~`--ppl` degrades window by window~~ — **was our metric, not the engine**: a window that starts mid-text is cold and costs ~1.8 nats/token (f32 3.80 against 5.61 on the same text), and llama.cpp's chunked number is a full-context estimate.  `--ppl` now warms each window up (`PPL_STRIDE=0` for the old behaviour) | fixed 2026-11-12 in `--ppl`; correction and numbers in `performance.md` |
| 6 | **`--thinking` is plumbed but unobserved** — the model did not emit `<think>` in three probes; an unclosed `<think>` is kept as text | plumbed only |

## Measuring it

```bash
cd engine
# the number that decides: PPL with the BOS; `windows=1` when the corpus is shorter than SEQ
MODEL=../models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf SEQ=512 GPU_LAYER=0 \
  ./engine_mm3 --ppl data/ppl_test.txt          # "PPL 18.7251 ... windows=1"
# per-token NLL, for "is the gap a few catastrophes or a uniform offset"
MODEL=... SEQ=512 GPU_LAYER=0 PPL_DUMP=/tmp/nll.txt ./engine_mm3 --ppl ...
# a file longer than SEQ is scored window by window: `stride=` (default SEQ/2)
# says how many tokens of each window are scored after the warm-up prefix.
# PPL_STRIDE=0 restores the pre-2026-11-12 cold windows; never compare the two.
MODEL=... SEQ=512 GPU_LAYER=0 ./engine_mm3 --ppl /tmp/long.md   # "PPL 130.2420 ... windows=4 span=512 stride=256"

# the float32 reference, from the same safetensors
python3 models/lfm2.5-230m/probe_logits.py --ids "1 6 24131 708 ..." --top 8
# residual amplitudes / whether the q16 rail is enough
python3 models/lfm2.5-230m/probe_residuals.py

# what llama.cpp does with the same weights -- only if the local reference
# build is present (refs/ is gitignored: prebuilt llama.cpp b12283 + the
# comparison scripts, see refs/llamacpp/README.md)
refs/llamacpp/bin/b12283-cpu/llama-perplexity \
  -m models/lfm2.5-230m/LFM2.5-230M-QAD-Q4_0.gguf -f docs/engine-internals.md -c 512 -t 8
```

For a byte-level bisect of *where* the engine diverges from the reference:
`STAGE_DUMP=1 HIDDEN_DUMP=1 DUMP_F32=1` (plus `ACT_F32=1` to take the fixed-point
stores out of the picture) gives one row per layer, position and stage, which is
how the conv-gate problem was localised; on the GPU side the fused path has
`CONV_DEBUG=1` instead.  [`test-hooks.md`](test-hooks.md) lists every switch, and
`refs/llamacpp/scripts/bisect_stages.py` (local, gitignored) is the reader for
that dump.
