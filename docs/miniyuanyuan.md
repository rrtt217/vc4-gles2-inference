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
| `miniyuanyuan.i16.mmcf` | 46.0 MB | i16 | laptop, and the reference for the q4 |
| `miniyuanyuan.q4.mmcf` | 16.9 MB | q4 | laptop; the one the board wants |

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

`BICROSS_DUMP=1` prints the block's five stages in the format each is stored in
(`bicq`/`bicac`/`bicg`/`bicup`/`bichid`).  It is the only way to bisect the
block: the three kernels feed each other, so a wrong intermediate shows up as a
bad PPL two stages later.

## Quality

Measured 2026-11-28 on Intel Iris Xe, `GPU_LAYER=1`, corpus
`engine/data/ppl_test.txt` (377 tokens), one 512-token window:

| container | PPL | nll | size |
|---|---:|---:|---:|
| i16 | 70.0440 | 4.2491 | 46.0 MB |
| q4 | 69.4882 | 4.2412 | 16.9 MB |
| float32 reference (`ref_forward_pth.py`) | 81.4485 | 4.4000 | — |

Both precisions produce the same greedy text,
`我是沅沅，来自湘西龙舟之乡沅陵县。<sep>`.

The engine sits *below* the float32 reference because its per-layer activation
quantisation is not the same function as the reference's float pass; the
comparison that matters is logits against the oracle, which agree at
**corr 0.99963 with the same argmax** on the ids `2,5,606,4,6`.

## Known gaps

| gap | what it is | why it is accepted |
|---|---|---|
| the MoE variant | `m12l_d384_moe_ep3_fp16.pt` is a different release: 6 routed experts, **top-2**, a shared expert, and routers on layers 6–9 only | the v3 reader refuses all three by name (top-1 only, no shared expert, no mixed sparse/dense).  Porting it is a container-format change, not a kernel one |
| the 4Q arena models | `model_1126.py`'s 4 parallel Q sets with no `W_K`/`W_V` and a per-head gate | no container form for a projection the attention does not have, and no published weights |
| q8 | not packed yet | q4 fits the board and measures better here; q8's reason to exist is other models |
| Pi Zero | not measured | the block is fine numerically but the board still needs `RMS_STORE=f32` (key 19) and a cross build |

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

Run it after touching any kernel that reads or writes two streams.  Two formats
were hardcoded in this block at first (`bicross_act` stored q16 whatever the
stream was, `mul2` wrote the activation stream where the FFN_HID stream belongs)
and both were invisible on the default plan, where every axis carries the same
value.

The ids are the check: a template that renders `<bos>你` instead of
`<bos><user>你<sep><assistant>` produces a different sequence and every
comparison after that is against a different prompt.  `TOKENS_OUT` dumps what the
engine actually fed.