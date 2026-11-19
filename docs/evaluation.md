# Evaluation: model quality, measured instead of judged

Quality claims in this tree are numbers or they are leads.  A chat comparison on
2026-11-12 ("the compact model writes better Chinese poems, the hybrid one
terminates") produced only leads: a different sampler, prompt or seed flips them.
These three batteries turn the same questions into numbers, and every model the
engine runs is scored on them, one row each.

| Battery | Script | Answers |
|---|---|---|
| Multiple-choice knowledge/reasoning | [`engine/eval/bench_mc.py`](../engine/eval/bench_mc.py) | does the model pick the right option, and can it even answer in the asked format? |
| Instruction following | [`engine/eval/eval_ifeval.py`](../engine/eval/eval_ifeval.py) | does the reply obey the constraint (JSON, word count, exact prefix/suffix, language)? |
| Translation | [`engine/eval/eval_translate.py`](../engine/eval/eval_translate.py) | does it translate the sentence, or just talk about translating it? |

The data lives next to the tools, so a re-run needs no download:

| File | Contents |
|---|---|
| [`engine/eval/data/mc_zh.jsonl`](../engine/eval/data/mc_zh.jsonl) | 50 Chinese 4-option items (math 10, chinese 8, history 8, geography 7, science 9, common_sense 5, logic 3; easy 15 / medium 25 / hard 10) |
| [`engine/eval/data/mc_en.jsonl`](../engine/eval/data/mc_en.jsonl) | 50 English 4-option items (science 14, common_sense 12, math 9, language 6, geography 5, history 4; same difficulty split) |
| [`engine/eval/data/ifeval.jsonl`](../engine/eval/data/ifeval.jsonl) | 24 instruction-following cases (zh 10, en 14), each with machine checks and a good/bad example |
| [`engine/eval/data/translate.jsonl`](../engine/eval/data/translate.jsonl) | 15 sentence pairs (daily/idiom/news/science/technical) run in **both** directions; each direction's reference is the other direction's source |

## The models in the tables

Every row below is one of these containers, on the same machine: laptop, Intel
Iris Xe (ADL GT2), Mesa 26.2.5, `GPU_LAYER=1 KV=q16`.

| Model | Container | What it is |
|---|---|---|
| MiniMind3 dense q8 (64M) | `models/minimind3/minimind3.q8.mmcf` | the default chat model |
| MiniMind3-MoE q8 (198M-A64M) | `models/minimind3-moe/minimind3-moe.q8.mmcf` | top-1 routing, 8 host syncs a token |
| LFM2.5-230M QAD q4 | `models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf` | hybrid conv + attention, the vendor's QAD q4 |
| RWKV-7 0.1B q8r | `models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf` | recurrent, v4 container, World vocab, G1 nothink template |
| MNN-Pocket-0.3B **qat** | `models/pocket-qat/pocket-qat.q4q24.dedup.q6.mmcf` | the GGUF `Q4_1` QAT upload, dequantized to F16 and repacked |
| MNN-Pocket-0.3B **bf16** | `models/pocket-0.3b/pocket-0.3b.q4q24.dedup.q6.mmcf` | the BF16 release, quantized here |
| BitCPM-CANN-0.5B **q8** | `models/bitcpm-0.5b/bitcpm-0.5b.q8.mmcf` | ternary MiniCPM4-0.5B (168 of 169 matrices per-row `{0, ±s}`), 434 MB, `RMS_STORE=f32` |
| BitCPM-CANN-0.5B **t3** | `models/bitcpm-0.5b/bitcpm-0.5b.t3.mmcf` | the same weights at 2 bits (167.5 MB), its own matvec path |

The two pocket containers are the same shape and size (198,007,797 B each) and
**different checkpoints**, not two packagings of one: their weights are 0.9955
cosine apart and their logits against the BF16 reference 0.9261, which is why
both are scored (measured 2026-11-17, sha256 `b164eee8…` and `d2a97700…`; the
engine's own fidelity against a float32 forward is in
[benchmarks.md](benchmarks.md)).

## Protocol

Two things are common to every row and are not repeated per table: one engine
run per item with the prompt prefilled, and the reply scored exactly as it comes
out -- no fence stripping, no case folding.

**Per-model overrides**, and there are only four:

| Model | Multiple choice | Instruction following | Translation |
|---|---|---|---|
| MiniMind3 pair | chat template, letter logits | `--preset minimind`, 2 seeds | greedy |
| LFM2.5-230M | chat template | `--preset lfm`, 2 seeds | greedy |
| RWKV-7 0.1B | **completion mode** (`--completion`) | `--preset rwkv` (card), 2 seeds | `--preset rwkv`, sampled |
| MNN-Pocket | chat template | `--preset greedy`, 3 seeds | greedy |
| BitCPM-CANN-0.5B | chat template | `--preset greedy`, 3 seeds | greedy |

* **RWKV-7's multiple choice must run in completion mode.**  Its chat template
  puts a think marker at the answer position, so the first decoded token is the
  marker, not a letter: a 100-item chat-mode run measured `fmt` ≈ 0 and `acc` at
  the null (archived at `models/rwkv7/phase4/eval_mc_chat_invalid/`).  In
  completion mode the first token's logits are the letter itself.
* **RWKV-7 is sampled at the model card's recommendation** -- `--temp 1.0
  --topp 0.3 --repeat-penalty 1.2` -- not greedy and not the engine default.
  `eval_ifeval.py --preset auto` (the default) and `eval_translate.py --preset
  auto` pick it for any container that declares RWKV-7 in metadata key 16 and
  keep the historical presets for everything else; `evalkit.RWKV_SAMPLING` holds
  the one copy.
* **The seed counts differ** (2 for the four older rows, 3 for pocket, i.e. 48
  against 72 attempts): the *rates* compare, the failure *counts* do not.  The
  per-case failure lists below are quoted with their attempt count for that
  reason.
* **BitCPM needs `RMS_STORE=f32`** on all three batteries: its residual lane sums
  reach 5.4e4 against the 24-bit store's 2048, and without the switch the reply
  degenerates instead of failing loudly (`docs/bitcpm.md`).
* **Pocket is measured on a fixed grid**, `SEQ=512 KV_GRID=512 RES=f32 ACT=f32`,
  the best configuration from its fidelity campaign -- a different format
  configuration from the other rows, chosen so the number measures the model
  rather than its encoding.

### The multiple-choice battery

`--gen 0` (no decode) and `LOGITS_OUT` carries the logits of the last prompt
token, the position where the answer starts.  The item is rendered as

    <question>
    A. <option 0>
    B. <option 1>
    C. <option 2>
    D. <option 3>
    只回答选项字母。
    答案：

(English items get a `Question: ` prefix; the instruction is Chinese for both
languages -- see "the English instruction problem" under *What these numbers
mean*.)

The four candidate ids are **not hard-coded**: the tool reads them out of the
model's own tokenizer by encoding the prompt with and without a trailing letter
and looking for the one id inserted; it tries the full-width colon first and a
newline second, and refuses the model if a letter is not one token (a merged
`":A"` token cannot be compared with a standalone `"A"`).  The MiniMind pair
gives 68-71, the LFM 542-545; `--out <dir>` keeps them in
`<model>.letters.json`.

Score of option L is the softmax over the four letter logits.  Three columns
matter:

* **acc** -- the argmax is the correct option.  Chance is 0.25.
* **acc_cal** -- the argmax after subtracting the letter logits of the same
  suffix with no question (`只回答选项字母。 答案：`), i.e. contextual
  calibration.  A model with a letter prior ("always answer A") scores 0.25
  here where `acc` can be much higher; a model that knows the answer scores
  the same or better.
* **fmt** -- the full-vocabulary argmax *is* one of A-D, i.e. the model was
  going to answer a letter at all.  When `fmt` is low, `acc` measures
  instruction following, not knowledge -- read both.

### The instruction-following battery

Each case is a prompt plus a list of checks; a case passes when every check
passes.  Cases are sampled `--seeds` times with the model's own recommended
sampling preset:

| Preset | Settings |
|---|---|
| `minimind` | `--temp 0.85 --top-k 50 --topp 0.85` |
| `lfm` | `--temp 0.1 --top-k 50 --repeat-penalty 1.05` |
| `rwkv` | `--temp 1.0 --topp 0.3 --repeat-penalty 1.2` |
| `greedy` | `--greedy` |

Supported checks (all must pass): `regex`, `absent_regex`, `starts_with`,
`ends_with`, `min_words`, `max_words`, `min_chars`, `max_chars`,
`min_lines`, `max_lines`, `is_upper`, `contains`, `contains_any`,
`contains_none`, `json_valid`, `json_keys`, `numbered_list`, `bullets`,
`cjk_ratio_min`, `ascii_ratio_min`.  "Words" are whitespace tokens with each
CJK character counting once, so a Chinese character budget is expressible.

Two numbers: **pass rate** over all attempts, and **hard pass** = the fraction
of cases that passed on every seed.  A case whose good answer needs luck shows
up as a gap between them.

`--selftest` runs no model: it checks the case file itself -- every
`good_example` must pass and every `bad_example` must fail -- so a wrong regex
cannot silently score a model down.

### The translation battery

This is the one capability the chat comparison got wrong in both directions: one
sampled reply to one sentence said the MiniMind pair "can translate" and the MoE
"cannot", and nothing separated "translated the sentence" from "finished a
sentence that started like a translation".

15 sentences, each run in **both** directions (30 items: 12 daily, 3 idiom,
1 science, 1 technical, 2 news -- the same sentence is the reference for the
other direction).  The instruction is Chinese in both directions ("把下面这句话
翻译成英文/中文，只输出译文。" + the source), for the same reason the
multiple-choice battery uses a Chinese answer instruction.

The score is **chrF** (character n-gram F-score, n = 1..6, beta = 2, precision
and recall averaged over n before combining, whitespace removed -- sacrebleu's
default recipe).  It needs no tokenizer, so a Chinese and an English reference
are scored the same way; a model that copies the source scores 0.0 because the
scripts differ, and a model that paraphrases gets partial credit instead of
zero.  Alongside it the dump records:

* **exact** -- output equals the reference after whitespace folding (a strict
  number, useful for "is it ever exactly right").
* **len** -- output length over reference length (a collapse is ~0.2, a decoder
  that will not stop is >2).
* **tgt / leak** -- fraction of the output's non-space characters in the target
  / source script.  `leak` ~1.0 means "repeated the source"; a low `tgt` with
  a high chrF means the model answered in the wrong language for part of the
  string.

## Results

The short version, before the tables:

| Battery | Best | Then |
|---|---|---|
| Multiple choice, 100 items | **BitCPM t3 0.710** (q8 0.690) | pocket qat 0.480, LFM2.5 0.380 |
| Instruction following, 24 cases | **pocket qat 0.792** | LFM2.5 0.562, BitCPM q8 0.500 |
| Translation, 30 items (chrF) | **BitCPM q8 0.495** (t3 0.479) | pocket bf16 0.383, LFM2.5 0.328 |

No model wins all three, and the two leaders are two different architectures at
two different sizes: a 0.5B ternary MiniCPM4 that knows things and translates but
only half-follows an instruction, and a 0.36B Qwen3 dense model that follows
instructions best and is beaten by 0.1-0.2 on the other two.  Every row below is
one container, all on the same machine; the translation table adds a
sampled row where that was measured, because for three of these models greedy
and sampled decoding disagree.  The dates in the captions are the run dates: the
four transformer/RWKV rows were measured 2026-11-12 and 2026-11-16, pocket's
2026-11-17, and BitCPM's 2026-11-19 (both containers, after the t3 kernel work
moved it -- the 2026-11-18 readings are in `docs/bitcpm.md`).

### Multiple choice

100 items (50 zh + 50 en), one deterministic run each, 2026-11-12 (chat mode),
2026-11-16 (RWKV, completion mode), 2026-11-17 (pocket) and 2026-11-19 (BitCPM,
both containers, re-measured after the t3 kernel work moved it).  "Null" is the
accuracy the same prediction letters would score against shuffled gold labels
(200 permutations, fixed seed): an accuracy at the null means the letter choice
carries no information about the question.

| Model | mode | acc | acc_cal | fmt | p_gold | null acc | null acc_cal | zh | en |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| MiniMind3 dense q8 (64M) | chat | 0.220 | 0.320 | 0.880 | 0.255 | 0.253 ± 0.032 | 0.260 ± 0.033 | 0.16 | 0.28 |
| MiniMind3-MoE q8 (198M-A64M) | chat | 0.250 | 0.250 | 0.620 | 0.254 | 0.251 ± 0.017 | 0.244 ± 0.035 | 0.24 | 0.26 |
| LFM2.5-230M QAD q4 | chat | 0.380 | 0.400 | 0.880 | 0.404 | 0.250 ± 0.038 | 0.254 ± 0.033 | 0.28 | 0.48 |
| RWKV-7 0.1B q8r | **completion** | 0.240 | 0.310 | 0.740 | 0.262 | 0.251 ± 0.040 | 0.255 ± 0.039 | 0.16 | 0.32 |
| MNN-Pocket qat | chat | **0.480** | **0.480** | 0.710 | 0.479 | 0.255 ± 0.037 | — | 0.46 | 0.50 |
| MNN-Pocket bf16 | chat | 0.450 | 0.400 | 0.740 | 0.443 | 0.254 ± 0.034 | — | 0.44 | 0.46 |
| BitCPM-CANN q8 | chat | 0.690 | 0.660 | 0.690 | 0.637 | 0.254 ± 0.040 | — | 0.62 | 0.78 |
| BitCPM-CANN t3 | chat | **0.710** | 0.700 | 0.700 | 0.653 | 0.253 ± 0.040 | — | 0.64 | 0.78 |

Chance is 0.25.  Reading the table:

* **BitCPM has the strongest signal in the table by a wide margin**: 0.690
  against a null of 0.254 ± 0.040 is 10.9 sigma (t3: 0.710 against 0.253 ± 0.040,
  11.4), and unlike every other row its predicted letters are flat (A 32, B 24,
  C 19, D 25), so there is no positional prior to remove and `acc_cal` lands on
  `acc`.  It is a 0.5B model with 168 ternary matrices against a 0.36B pocket --
  and the ternary body is not what limits it: t3 is 2 points *ahead* of q8 in
  raw accuracy on the same weights.
* **Pocket is next**: 0.480 against a null of 0.255 ± 0.037 is 6.0 sigma.  Its
  prior (A 57, B 1, C 20, D 22) costs it nothing either, so `acc_cal` does not
  move.  LFM2.5 is next: 0.380 against 0.250 ± 0.038 is 3.4 sigma, 4.4 on
  `acc_cal`.
* **The QAT container is the better checkpoint**, 3 points raw and 8 calibrated
  over bf16 (0.480 against 0.400 once the prior is removed).  That settles what
  the packaging campaign could not: the QAT upload is not a differently-produced
  file, it is the better model.
* **MiniMind3 dense sits at the null**: 0.220 against 0.253 ± 0.032.  It picks
  "C" 65 times out of 100 (A 30, B 4, D 1), so the letter is a positional
  prior, not a decision.  `acc_cal` 0.320 is 1.8 sigma over its own null -- a
  lead, not a result.
* **The MoE answers "A" 94 times out of 100**, is at the null exactly, and only
  puts a letter first 62% of the time (`fmt`): one fifth of the items it starts
  the reply with something else, which the 8-host-sync route does not explain --
  this is a model property.  It is also the slowest of the three (2.7 s/item
  against 0.9 s).
* **RWKV-7 is at the null raw** (0.240 against 0.251 ± 0.040, `acc_cal` 1.4
  sigma -- the dense MiniMind's kind of lead) but **answers in the asked
  format** (`fmt` 0.740), and the split is language-shaped: zh 0.92, en 0.56.
  Its knowledge is English-centric (en 0.32 against zh 0.16), the opposite
  weighting of the Chinese-tuned MiniMind pair.
* **Pocket is balanced across languages** (0.46 / 0.50), the only row that is;
  BitCPM is English-heavy (0.62 / 0.78), LFM2.5 0.28 / 0.48, MiniMind3 0.16 /
  0.28.  By category pocket is science
  0.652 (n=23), common_sense 0.647 (17), geography 0.500 (12), history 0.500
  (12), language 0.500 (6), **math 0.263 (19)**, chinese 0.250 (8), logic 0.000
  (3, too small to read).  Math is *identical* in both pocket containers, so the
  arithmetic weakness is the model and not the quantisation -- and it is the same
  weakness the board shows in a hand probe (`84 * 3 = ` gives `54.` on the Pi and
  `252.` on the Xe: two independent paths, one wrong answer).
* **`fmt` is a ceiling, not a footnote.**  Pocket's 0.710 means 29 items had a
  non-letter full-vocabulary argmax and are scored wrong, so 0.48 is a floor
  (0.48/0.71 among the formatted items); BitCPM's 0.690 is the same story
  (0.69 is 0.69/0.69 among the formatted ones, i.e. its floor and its rate
  coincide).  LFM2.5 follows the letter format far more often (0.88) and is less
  accurate: its weak half is knowledge, pocket's and BitCPM's is instruction
  following.
* **Difficulty orders BitCPM and LFM, and nothing else.**  BitCPM q8 goes
  0.80 / 0.68 / 0.55 on easy/medium/hard (30/50/20 items) and t3 0.80 / 0.70 /
  0.60; LFM2.5 goes 0.43 / 0.40 / 0.25 the same way.  Neither MiniMind improves
  on "easy" items, and RWKV-7's is inverted (0.17 / 0.22 / 0.40) -- at 100 items
  that is noise, and it is recorded here as noise.

Per-item probabilities are in `<out>/<model>.items.jsonl`, and the per-model JSON
summary carries the per-category, per-language and per-difficulty splits.

### Instruction following

24 cases, 2 or 3 seeds each (48 or 72 attempts, see the protocol table), scored
as it comes out of the engine: 2026-11-12 (MiniMind pair, LFM, 2 seeds),
2026-11-16 (RWKV, 2 seeds), 2026-11-17 (pocket, 3 seeds).

| Model | preset | seeds | pass | hard pass | zh | en | most-failed checks |
|---|---|---:|---:|---:|---:|---:|---|
| MiniMind3 dense q8 (64M) | minimind | 2 | 0.188 | 0.167 | 0.25 | 0.14 | max_words 8, json_valid+json_keys 6+6, bullets 4, starts_with 4, ends_with 4, max_chars 4 |
| MiniMind3-MoE q8 (198M-A64M) | minimind | 2 | 0.188 | 0.167 | 0.25 | 0.14 | identical pattern to the dense model (json 6+6, max_words 8, bullet/prefix/suffix/char 4 each) |
| LFM2.5-230M QAD q4 | lfm | 2 | 0.562 | 0.542 | 0.45 | 0.64 | json_valid+json_keys 6+6, max_words 6, max_chars 4, bullets 2, absent_regex 2 |
| RWKV-7 0.1B q8r | rwkv | 2 | 0.250 | 0.208 | **0.40** | 0.14 | max_words 8, json_valid+json_keys 6+6, ends_with 4, max_chars 4, max_lines 3, six checks at 2, bullets/numbered_list 1 each |
| MNN-Pocket qat | greedy | 3 | **0.792** | **0.792** | 0.700 | 0.857 | bullets 3, cjk_ratio_min 3, max_words 3, min_words 3, regex 3 |
| MNN-Pocket bf16 | greedy | 3 | 0.708 | 0.708 | 0.600 | 0.786 | max_words 9, bullets 3, contains_any 3, max_chars 3, min_words 3, regex 3 |
| BitCPM-CANN q8 | greedy | 3 | 0.500 | 0.500 | 0.500 | 0.500 | json_valid+json_keys 6+6, bullets 6, max_words 9, max_chars 6, max_lines/absent_regex/contains_none/regex 3 each |
| BitCPM-CANN t3 | greedy | 3 | 0.500 | 0.500 | 0.500 | 0.500 | identical to the q8 row, cell for cell |

The per-case pattern is more informative than the totals:

* **JSON is the universal failure -- except for pocket.**  Six of the eight
  rows (everything except pocket) lose `json_valid` + `json_keys` on all 6
  attempts: they wrap the object in a fenced block or answer with prose.  Pocket's failure list has no `json_*` at
  all, and `hard pass == pass` in both of its containers: an attempt either
  satisfies the instruction completely or not at all.
* **LFM2.5 passes exactly the cases a chat user would notice**: everything
  except the three JSON cases and the one-word/character-budget ones, where it
  answers with a full sentence ("天空是蓝色的。" for "用一个词回答").  ALL CAPS,
  exact prefix/suffix, numbered lists, bullets, two-line answers and the
  date-arithmetic regex pass on both seeds.
* **The MiniMind pair passes only the "write something long" cases** -- if-07,
  if-08, if-13, if-15, the length/language checks any verbose answer passes -- and
  every case with an exact format fails.  Their per-case patterns are
  *identical*, which is what an instruction battery looks like when neither model
  has learned the meta-skill: the failure mode is the tuning, not the parameter
  count (the MoE has 3x the parameters and scores the same).
* **RWKV-7 sits between the two** (0.250 against 0.188 / 0.562).  Its hard-pass
  cases are the MiniMind pattern plus if-21 (a steps answer); if-04 and if-16
  pass on one seed only, which is the 0.250 against 0.208 gap.
* **BitCPM sits between the MiniMind pair and LFM2.5, and is the one row whose
  two languages agree**: 0.500 in Chinese and 0.500 in English, identically in
  the q8 and the t3 container (36 of 72 attempts, `hard pass` the same 0.500).
  Its failure profile is the MiniMind pair's -- length and JSON -- not LFM2.5's.
* **Which language a model follows instructions in better is not its knowledge
  language**: the MiniMind pair (zh 0.25 against en 0.14) and RWKV-7 (0.40 against
  0.14) do better in Chinese, though RWKV-7's *knowledge* is English-centric;
  LFM2.5 (0.45 against 0.64) and pocket (0.700 against 0.857) do better in
  English; BitCPM is equal.  The instruction-format axis and the knowledge axis
  are independent, and only pocket is strong on both.
* **The spread is the product**: 0.19 (MiniMind pair), 0.25 (RWKV), 0.50
  (BitCPM), 0.56 (LFM), 0.71-0.79 (pocket) on the same 24 cases.

Two more things about reading a pass rate:

* **`gen` matters**: a case whose reply is cut at the token limit fails its
  checks for a reason the model cannot control.  Each case's `gen` was set from
  the expected answer length, and the per-attempt dump keeps the raw reply, so a
  suspected truncation can be checked before blaming the model.
* **Passing by rambling is possible** for the length/language checks
  (`min_words`, `cjk_ratio_min`): a model that answers a different question at
  length still passes.  This is what makes an instruction battery a *format*
  test, not a quality test -- read it next to the multiple-choice table, not
  instead of it.

### Translation

30 items (15 sentences x 2 directions), 2026-11-12 and 2026-11-17, chrF against
the reference.  "tgt/leak" are the target/source script fractions; see the method
section.  `chrF ≥ 0.5` / `≥ 0.3` (`—` where the run did not report the
distribution) are the usable / half-right counts, and `copies source` is a count
out of 30 except for pocket, whose dump reports a fraction.

| Model | decode | chrF | zh→en | en→zh | ≥ 0.5 | ≥ 0.3 | len |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniMind3 dense q8 (64M) | greedy | 0.130 | 0.203 | 0.058 | 2/30 | 2/30 | 3.0 |
| MiniMind3 dense q8, sampled | minimind | 0.085 | — | 0.017 | — | — | — |
| MiniMind3-MoE q8 (198M-A64M) | greedy | 0.133 | 0.172 | 0.094 | 1/30 | 2/30 | 2.1 |
| MiniMind3-MoE q8, sampled | minimind | 0.130 | — | — | — | — | — |
| LFM2.5-230M QAD q4 | greedy | 0.328 | 0.368 | 0.287 | 8/30 | 13/30 | 1.0 |
| LFM2.5-230M QAD q4, sampled | lfm | 0.335 | — | — | — | — | — |
| RWKV-7 0.1B q8r | rwkv | 0.099 | 0.139 | 0.059 | 0/30 | 4/30 | 3.43 |
| MNN-Pocket qat | greedy | 0.366 | 0.504 | 0.228 | — | — | 1.12 |
| MNN-Pocket bf16 | greedy | 0.383 | 0.480 | 0.286 | — | — | 1.12 |
| BitCPM-CANN q8 | greedy | **0.495** | 0.511 | 0.479 | — | — | 1.11 |
| BitCPM-CANN t3 | greedy | 0.468 | **0.521** | 0.415 | — | — | 1.08 |

Reading it:

* **BitCPM translates best, then pocket, then LFM2.5** -- 0.495 / 0.468 for the
  two BitCPM containers, 0.383 for pocket's bf16 file, 0.328 for LFM2.5 (8 of its
  30 outputs usable, 13 half-right, mean length 1.0x the reference, and it almost
  never talks about translating: 2/30).  LFM's translation is stable across
  decoding (0.328 greedy, 0.335 sampled), where the dense MiniMind's is not
  (0.130 -> 0.085).
* **The ternary body costs the en→zh direction, not the total**: q8 is balanced
  (0.511 zh→en / 0.479 en→zh) while t3 keeps the zh→en side (0.521) and drops
  en→zh to 0.415, with exact 3/30 against 2/30 and the same near-zero source
  leakage (0.035 / 0.007 of characters).  That is the shape of a rounding
  difference, not a broken container -- and it is the one battery where the
  167 MB file is measurably behind its 435 MB sibling.
* **A real MT fine-tune on the same engine is a different league**: the official
  LFM2-350M-ENJP-MT (EN<->JA) packed into a q4 container scores 0.559 on a
  9-sentence probe against 0.419 for the untuned 230M base, and 0.675 for the
  vendor GGUF through llama.cpp.  Build steps and packing traps in
  [lfm2.5-230m.md](lfm2.5-230m.md#a-translation-fine-tune-of-the-same-architecture-lfm2-350m-enjp-mt).
* **A human paraphrase scores ~0.65, not 1.0, on this corpus** ("The weather is
  lovely today." against "The weather is nice today." gives 0.651).  So 0.33
  means "half the content is wrong", not "almost right" -- and BitCPM's 0.495 is
  the best this corpus has produced outside the 350M MT fine-tune (0.559):
  short simple sentences
  come out well (0.71, 0.76, 0.84 are real), numbers do not ("ten million" ->
  "a million", "twenty minutes late" -> "two hours late"), idioms and complex
  clauses do not.  0.13 means "the sentence is not there".
* **The MiniMind pair does not translate; it describes the request.**  The dense
  model emits "Here is the translation of the sentence \"…\" into English:" in
  half the cases (15/30, 12 of them zh→en), and for en→zh it repeats the English
  source: 9 of its 15 en→zh outputs are more than half source script, and that
  direction scores 0.058 (0.017 sampled), the worst cell in the battery.  The
  MoE mixes both languages inside a sentence ("请关闭 window; it is very cold
  outside."), loops, and preambles in 12/30.
* **RWKV-7 is in the MiniMind pair's league, not the LFM's**: 0 of 30 usable, 4
  half-right, 18 of 30 more than half source script, mean length 3.4x -- it
  rambles.  en→zh is its worst cell (0.059, 10/15 copies): asked for Chinese it
  repeats the English source, sometimes behind a "Translation:" preamble.  Its
  best cell is en-zh-08 (0.448): "Learning a new language requires time and
  patience." expanded into a paragraph of Chinese advice.  zh-en-01 has the right
  sentence ("Today is a great day.") quote-wrapped and repeated twice, which chrF
  prices at 0.191 -- the right sentence in the wrong behaviour, which is what the
  0.099 mean says.
* **Pocket's shape is consistent with its other batteries**: it reads Chinese
  well (zh→en 0.50 against LFM's 0.37) and writes it worse (en→zh 0.23 against
  0.29).  Chinese *input* is its strength, Chinese *output* its weakness -- the
  same direction as its MC (zh 0.46) and ifeval (zh 0.700) rows, which are
  comprehension-heavy.  The bf16 container edges ahead here (0.383 against
  0.366), the one battery where it does, and it gets there by writing longer (its
  top ifeval failure is `max_words 9`), not by translating better.
* **The chat comparison's "the dense model translates" was one sentence**: 今天
  天气很好 is one of the 2/30 this battery scores above 0.5, and the other 28 were
  never in that test.  Same lesson as the multiple-choice battery: one sampled
  reply per prompt decides nothing.

The raw per-attempt replies (ifeval), per-item probabilities (multiple choice)
and hypotheses (translation) are in each run's `--out` directory next to the
summaries.

## What these numbers mean, and what they do not

* **100 items is a coarse instrument.**  With 100 questions the standard error
  of an accuracy near 0.25 is about 0.043, so differences under ~0.09 between
  two models are not resolved.  Read the per-category columns as leads and the
  totals as ±0.05.
* **Chance is 0.25, and the letter prior is real.**  The 64M model answers "C"
  65 times out of 100, the MoE "A" 94 times, pocket "A" 57 times; against
  balanced gold those constants alone score ~0.25.  `acc_cal` subtracts the
  letter prior and the null column says what the same letters would score against
  shuffled gold, so "0.22" or "0.25" is not read as "the model tried".
* **The English instruction problem.**  With "Answer with the option letter
  only." the MiniMind pair replies with prose to English items ("Based on...",
  "The question is...") under greedy decoding, for every English phrasing
  tried; the same question behind the Chinese instruction gets a letter from
  all three transformer models.  The battery therefore uses the Chinese
  instruction for both languages, and `fmt` reports compliance anyway.  This is
  a property of the models, not of the harness: a per-language instruction would
  have measured it instead of knowledge.
* **Instruction following is a separate axis.**  A model can know the answer
  and still fail a case because it added "Here are three bullets:" above the
  bullets or wrapped JSON in a code fence.  The ifeval pass rate is the number
  a chat user feels first, and it does not correlate with the MC accuracy: the
  best MC row is also the best ifeval row, but the MiniMind pair and RWKV-7
  order differently on the two.
* **This is not a leaderboard.**  The items are our own, deliberately easy
  (they are meant to be answerable by a bright child, which is roughly the
  ceiling of a 64M model).  They do not compare with published C-Eval / ARC
  numbers, which use thousands of items and full-size models.
* **Quantization is part of the model.**  LFM2.5 is measured at QAD q4 because
  that is the container that works best for it, the MiniMind pair at q8, pocket
  at q4 body + q6 head on a f32 stream grid, RWKV-7 at q8 with per-row scales.  A
  cross-precision comparison is a different experiment (`bench_mc.py` takes any
  container, so run both -- the two pocket rows are exactly that experiment).

## Reproducing every row

The three batteries, one command each for the 2026-11-12 baseline rows:

```bash
python3 engine/eval/bench_mc.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
  --model minimind3-moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
  --model lfm2.5-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --data engine/eval/data/mc_zh.jsonl \
  --data engine/eval/data/mc_en.jsonl \
  --out /tmp/eval_mc --resume

python3 engine/eval/eval_ifeval.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf --preset minimind --seeds 3 \
  --data engine/eval/data/ifeval.jsonl --out /tmp/eval_ifeval

python3 engine/eval/eval_translate.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
  --model minimind3-moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
  --model lfm2.5-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --data engine/eval/data/translate.jsonl --out /tmp/eval_translate
```

RWKV-7, where two of the three need a flag (`--preset auto`, the default, picks
the rwkv preset from metadata key 16 by itself; the explicit flag pins it):

```bash
python3 engine/eval/bench_mc.py \
  --model rwkv7-q8r=models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
  --data engine/eval/data/mc_zh.jsonl --data engine/eval/data/mc_en.jsonl \
  --out /tmp/eval_mc --completion --resume

python3 engine/eval/eval_ifeval.py \
  --model rwkv7-q8r=models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
  --preset rwkv --seeds 2 \
  --data engine/eval/data/ifeval.jsonl --out /tmp/eval_ifeval

python3 engine/eval/eval_translate.py \
  --model rwkv7-q8r=models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf \
  --preset rwkv --data engine/eval/data/translate.jsonl --out /tmp/eval_translate
```

MNN-Pocket, on its fixed grid and with absolute paths (the batteries spawn a
tokenizer probe whose cwd is not `engine/`):

```bash
cd engine
M=$(readlink -f ../models/pocket-qat/pocket-qat.q4q24.dedup.q6.mmcf)
E='--env GPU_LAYER=1 --env SEQ=512 --env KV=q16 --env KV_GRID=512 --env RES=f32 --env ACT=f32'
python3 eval/bench_mc.py      --model qat=$M --data eval/data/mc_en.jsonl \
    --data eval/data/mc_zh.jsonl $E --out /tmp/pocket_mc
python3 eval/eval_ifeval.py   --model qat=$M --preset greedy $E --out /tmp/pocket_if
python3 eval/eval_translate.py --model qat=$M --preset greedy $E --out /tmp/pocket_tr
```

`--resume` appends to `<out>/<model>.items.jsonl` and skips items already
scored, which is what a battery cut off by a tool timeout needs; every item's
four probabilities are kept there, so a surprising number can be re-checked
without the GPU.  One item costs 0.9 s on the dense model, 2.6 s on the LFM and
2.7 s on the MoE, so the full 100-item, three-model battery is ~10 minutes on the
laptop GPU.  The RWKV-7 runs of 2026-11-16 are archived at
`models/rwkv7/phase4/{eval_mc,eval_ifeval,eval_translate}/`, and the superseded
rounds next to them (chat-mode MC, greedy ifeval, truncated-template ifeval)
carry READMEs.

## Adding items

```json
{"id":"zh-051","lang":"zh","category":"math","difficulty":"medium",
 "question":"...","options":["...","...","...","..."],"answer":1,
 "why":"<=40 char justification"}
```

Rules that the tool relies on: exactly four distinct options; `answer` is the
0-based index; questions are self-contained and unambiguous; no "all of the
above"; no current events.  Keep the correct index roughly balanced across
A-D, or a letter-prior model scores above chance for free.

For instruction cases the schema is
`{"id","lang","prompt","gen","checks":[...],"good_example","bad_example"}`;
after editing, `--selftest` must report zero mismatches.
