# MiniYuanYuan: the BiCrossGate FFN

`deng-cn/MiniYuanYuan` is a 12-layer, `d=384`, 8302-word Chinese chat model whose
FFN is not a SwiGLU.  It is the first model on this engine with a block the dense
path does not have, so it is also the first that needs metadata key 23 and tensor
kinds `0x3D`/`0x3E`.  It runs at i16 and q4, and at q4 it is 16.9 MB.

| | |
|---|---|
| checkpoint | a torch zip, `{"model": state_dict, "cfg": {...}}`, no `config.json` |
| shape | 12 layers, `d=384`, 12 Q / 2 KV heads, `head_dim` 32, FFN `i=1152` |
| FFN | BiCrossGate, shared by layer pairs: 6 groups for 12 layers |
| vocab | 8302, tied head, `<space>`/`<newline>` as named entries |
| tokenizer | flat longest match, no merges (key 10's fifth byte) |
| template | `<bos><user>…<sep><assistant>`, no default system turn |
| rope | theta 1e6, rotate-half; `rms_norm_eps` 1e-6 |
| metadata | key 11 (eps), 12 (q24 residual), 19 (f32 rms store), 23 (the block) |

The FFN is

```text
q1, q2 = q(x).chunk(2)          # q: 384 -> 768
a = gelu(q1 * x)                # elementwise against the block's own input
c = silu(q2 * x)
y = down( gate(concat(a, c)) * up(x) )   # gate: 768 -> 1152, up: 384 -> 1152
```

`up` and `down` keep the `w1`/`w2` slots; `q` and `gate` are the two that need
kinds.  The two activations are key 23's two bytes, in that order.

## Containers

| file | size | precision | runs |
|---|---:|---|---|
| `miniyuanyuan.i16.mmcf` | 46.0 MB | i16 | laptop, and the reference for the two quantised ones |
| `miniyuanyuan.q8.mmcf` | 23.1 MB | q8 | laptop; **the one to prefer** |
| `miniyuanyuan.q4.mmcf` | 16.9 MB | q4 | laptop and board; the smallest |

**Prefer q8, then q4.**  On the board q8 is the *fastest* of the three
(266 ms/token against q4's 279 and i16's 342 -- see Quality) and its PPL is
within 0.7% of q4's, so the 6 MB q4 saves cost a little speed rather than buying
any.  q4 stays the answer when the file size is the constraint: it is half of q8
and 37% of i16.

```bash
python3 engine/tools/mmcf_pack.py --in <release>/weights/m12l_d384_ep2_fp16.pt \
  --quant q4 --seq 512 --tok models/miniyuanyuan/miniyuanyuan.tok \
  --out models/miniyuanyuan/miniyuanyuan.q4.mmcf \
  --name miniyuanyuan --template qwen --system "" --chat \
  --chat-template templates/miniyuanyuan.jinja --tok-greedy
```

`--tok-greedy` is required: the vocabulary has no merges, and the BPE pass on a
scoreless blob merges by insertion order and silently rewrites correct ids.  The
`.tok` comes from `make_tok_flat.py` over the release's `corpus/vocab_v2.json`.
`--system ""` is required too: the model has no system turn, and a default one
shifts every prompt by 30 tokens.

An i16 container needs `--quant i16` and is otherwise identical.

## Run

```bash
GPU_LAYER=1 ./engine_mm3 --model ../models/miniyuanyuan/miniyuanyuan.q4.mmcf \
  --chat --prompt "你好，你是谁？" --gen 32 --greedy
```

`GPU_LAYER=1` is the fused path; the BiCross block's three kernels only exist
there.  Without it the layer loop runs on the host and `CPU_FFN=1` selects the
block's own CPU oracle (`Session::bicross_ffn`).

On the Pi Zero the same command runs with `KV=q16` and, for the q8 container
only, `QKV_PAIR=0` (see Quality).  The rms partial sums need no switch: the
container states metadata key 19, the container wins over `RMS_STORE`, and the
`formats:` line names the result `rms f32 (container key 19)`.

`BICROSS_DUMP=1` prints the block's five stages in the format each is stored in
(`bicq`/`bicac`/`bicg`/`bicup`/`bichid`).  It is the only way to bisect the
block: the three kernels feed each other, so a wrong intermediate shows up as a
bad PPL two stages later.

## Quality

Measured 2026-11-28 on both machines, one protocol: `GPU_LAYER=1`, `KV=q16`, corpus
`engine/data/ppl_test.txt` (md5 `dc0c8b7c`), one 512-token window
(`SEQ=512 PPL_STRIDE=256`), which is 378 predicted tokens for this tokenizer.

| container | Xe PPL | Pi Zero PPL | size |
|---|---:|---:|---:|
| i16 | 70.0440 | 70.1238 | 46.0 MB |
| q8 | 69.6895 | 69.8373 | 23.1 MB |
| q4 | 69.4882 | 69.3326 | 16.9 MB |
| float32 reference (`ref_forward_pth.py`) | 81.4485 | — | — |

The three precisions produce the same greedy text,
`我是沅沅，来自湘西龙舟之乡沅陵县。<sep>`, on both machines.  The board tracks
Xe to 0.1-0.2% on every container, so VC4's own compiler limits change the speed
(see below) but not the arithmetic.

The engine sits *below* the float32 reference because its per-layer activation
quantisation is not the same function as the reference's float pass; the
comparison that matters is logits against the oracle, which agree at
**corr 0.99963 with the same argmax** on the ids `2,5,606,4,6`.

### Speed on the board

Measured 2026-11-28 on the Pi Zero, ARMv6/VFPv2: `TIMING=1`, `--gen 48 --greedy`,
median of three, warm (one discarded run).

| container | decode | prefill | note |
|---|---:|---:|---|
| **q8** | **252.2 ms/token** | 6.0 s | `QKV_PAIR=0` on VC4; the gate folded into the up matvec |
| q8, `BICROSS_FUSE=0` | 266.1 ms/token | 6.2 s | the two-pass form, for the A/B |
| q4 | 279.5 ms/token | 7.89 s | two passes: the fused kernel is q8-only |
| i16 | 342.3 ms/token | 6.85 s | two passes, and the widest stream |

q8 is faster *and* smaller than i16 by more than the file size suggests, and it
beats q4 despite carrying 37% more weights: q4's eight-wide gate is not enough to
pay for the byte unpacking here, which is the opposite of MiniMind3 on the same
board.  That is a per-model fact, so measure it per model.

`QKV_PAIR=0` is required for q8 on VC4 and is the only switch this model needs
beyond `GPU_LAYER=1`: without it the load dies in `program 'xkv' did not draw`,
vc4's register allocator refusing the row-pair int8 qkv kernel.  The workaround is
numerically neutral -- Xe PPL 69.6895 with pairing and 69.6895 without (measured
2026-11-28).

**Why a 19.7M model is slower here than the 64M MiniMind3.**  Measured 2026-11-28.
The unit is the pass -- a matvec or an elementwise kernel, one state change --
not the parameter count and not the draw:

| model | layers | dim/hidden | passes/layer | draws/token | decode |
|---|---:|---|---:|---:|---:|
| MiniMind3 q4 | 8 | 768/2432 | 19 | 143 | 185.7 |
| feng30m q8 | 11 | 448/896 | — | 178 | — |
| MiniYuanYuan q4 | 12 | 384/1152 | 22 | 248 | 279.5 |

A pass costs ~1.1 ms whatever it carries, so a quarter of MiniMind3's FLOPs in
one layer costs the same 23 ms: 12 layers here against 8 is the whole gap, and
8 layers of this model would decode in ~187 ms -- MiniMind3's 185.7.  The draw
count is a *shape* symptom, not the bill; the next section is the experiment that
settles it.

`LAYER_PROFILE` puts the block's cost where it belongs.  Per layer, draws:

| pass | MiniMind3 q4 | MiniYuanYuan q4 |
|---|---:|---:|
| `rms_att` / `rms_ffn` | 3 / 3 | 3 / 3 |
| `attn` (with the qkv matvec) | 8 | 7 |
| `wo` | 1 | 1 |
| `ffn` (w13, or the block's `q`/`gate`/`up`/`act`/`mul2`) | 2 | 7 |
| `w2` / `down` | 2 | 1 |
| **per layer** | **19** | **22** |

The FFN is 8 draws against SwiGLU's 4 -- four matvecs and three elementwise
passes instead of two matvecs and one fused pair -- while this model's
12Q/2KV attention saves the draw MiniMind3 spends on KV.  Net +3 a layer, 36 a
token, 14%.

That is the block as designed, not a fallback: `bicross_act` and `mul2` are
separate kernels because the chain has three operands, and SwiGLU's single
`swiglu_pair` has two.

### What a pass costs (and why the draw count is not the currency)

Measured 2026-11-28 on the board, in that order, because the first answer was
wrong:

| experiment | result |
|---|---|
| `MATVEC_STRIP=1152` (hidden 1152 is just over `kDrawCap` 1024, so `gate`/`up` split) | 248 -> 224 draws a token, PPL identical (69.3326 / 69.8373), decode 277.4 and 279.0 against 279.5: **no time saved** |
| `SYNCCOUNT=1` | **1** host<->GPU synchronisation a token, not one a pass |
| decode/12 layers against MiniMind3's decode/8 | 23.3 ms against 23.2 -- equal, at a quarter of the FLOPs |

So the unit is the pass: 22 a layer here, 19 there, at roughly 1.1 ms each, and a
draw inside an existing pass (the tail strip) is free.  `engine/probes/rmsfuse_probe`
reached the same conclusion from the other end on 2026-11-20 -- moving an
rmsnorm's work into its consumer does not remove the pass and buys 0.4%.

The block's own overhead is therefore **one pass a layer** (14% of its draws, but
4-5% of the time), and `matvec_q8_ffn_mulgate` removes it: the gate's projection
store is read inside the up matvec, so `mul2` stops existing.  Measured
2026-11-28 on the board, q8, median of three: **266.1 -> 252.2 ms/token (-5.2%)**
and PPL 69.8373 either way.  Draws fall 248 -> 224; the 24 and not 12 is because
the fused kernel draws the whole 576-fragment plane in one call where the matvec
loop had split at `kDrawCap` and `mul2` was a pass.

`BICROSS_FUSE=0` restores the two passes for the A/B.  The kernel exists for q8
weights with an I16_2 activation and an I16_2 hidden -- the same three conditions
`ffn_pair_swiglu_chunk` asks for -- so q4 and i16 keep the two passes and the
next ~5% is the `bicross_act` pass folded into the `gate` matvec, which needs the
same kernel family for those two weight types.

The six rmsnorm passes a layer are 27% of the token and are shared with every
dense model here; `rmsfuse_probe` priced folding them at 0.4%, because moving the
work does not remove the pass.

## Known gaps

| gap | what it is | why it is accepted |
|---|---|---|
| the MoE variant | `m12l_d384_moe_ep3_fp16.pt` is a different release: 6 routed experts, **top-2**, a shared expert, and routers on layers 6–9 only | the v3 reader refuses all three by name (top-1 only, no shared expert, no mixed sparse/dense).  Porting it is a container-format change, not a kernel one |
| the 4Q arena models | `model_1126.py`'s 4 parallel Q sets with no `W_K`/`W_V` and a per-head gate | no container form for a projection the attention does not have, and no published weights |
| q8 on VC4 | the row-pair int8 qkv kernel is refused by vc4's register allocator (`program 'xkv' did not draw`) | `QKV_PAIR=0` answers the same text for the same PPL; the paired form is a speed arm on Xe, not a numeric one |
| the fused FFN kernel is q8-only | `matvec_q8_ffn_mulgate` asks for q8 weights, an I16_2 activation and an I16_2 hidden -- the same three conditions `ffn_pair_swiglu_chunk` asks for | q4 and i16 keep the two-pass `mul2` and so pay the ~5% the fusion saves; the same kernel family for those weight types is the next step, not a redesign |

## Measuring it

```bash
# the oracle: same checkpoint, BiCrossGate in numpy
python3 engine/tools/ref_forward_pth.py <release>/weights/m12l_d384_ep2_fp16.pt \
  --vocab <release>/corpus/vocab_v2.json --ids 2,5,606,4,6 \
  --logits-out /tmp/ref.bin

# the engine's logits for the same ids
GPU_LAYER=1 LOGITS_OUT=/tmp/eng.bin ./engine_mm3 \
  --model ../models/miniyuanyuan/miniyuanyuan.q4.mmcf \
  --chat --prompt "你" --gen 1

python3 engine/scripts/cmp_logits.py /tmp/ref.bin /tmp/eng.bin
```

The block follows the format axes -- `ACT`, `RES`, `FFN_HID`, `PROJ` and `KV` all
produce the same answer, alone and paired -- and `engine/scripts/format_sweep.sh`
is what proves it:

```bash
bash engine/scripts/format_sweep.sh ../models/miniyuanyuan/miniyuanyuan.q4.mmcf \
  "我是沅沅，来自湘西龙舟之乡沅陵县。<sep>"
```

On the board the same script runs from the runtime directory:

```bash
ENGINE=bin/engine_mm3_pi.myy EXTRA='QKV_PAIR=0' KV=q16 bash tools/format_sweep.sh \
  models/miniyuanyuan/miniyuanyuan.q8.mmcf "我是沅沅，来自湘西龙舟之乡沅陵县。"
```

`EXTRA` carries the switch the container *requires* on that host and which is not
an axis -- `QKV_PAIR=0` here.  That is a *different* claim than the laptop's: VC4
answers "does this kernel exist on this compiler at all", which is how the q8 qkv
pair above was found, while Xe answers "does this kernel follow the axis".
Inheriting the required switch from the caller's environment instead of naming it
ran 35 of 43 axes into `<no answer>` and left the four KV rows passing, because
the failure was the load, not the axis.

Run it after touching any kernel that reads or writes two streams.  Two formats
were hardcoded in this block at first (`bicross_act` stored q16 whatever the
stream was, `mul2` wrote the activation stream where the FFN_HID stream belongs)
and both were invisible on the default plan, where every axis carries the same
value.

The ids are the check: a template that renders `<bos>你` instead of
`<bos><user>你<sep><assistant>` produces a different sequence and every
comparison after that is against a different prompt.  `TOKENS_OUT` dumps what the
engine actually fed.