# Evaluation: model quality, measured instead of judged

The chat comparison of 2026-11-12 produced opinions ("the compact model is
better at Chinese poems, the hybrid one terminates").  Those are useful as
leads and useless as regressions: a different sampler, prompt or seed flips
them.  This page documents the three batteries that turn the same questions into
numbers, and the numbers themselves.

| Battery | Script | Answers |
|---|---|---|
| Multiple-choice knowledge/reasoning | [`engine/scripts/bench_mc.py`](../engine/scripts/bench_mc.py) | does the model pick the right option, and can it even answer in the asked format? |
| Instruction following | [`engine/scripts/eval_ifeval.py`](../engine/scripts/eval_ifeval.py) | does the reply obey the constraint (JSON, word count, exact prefix/suffix, language)? |
| Translation | [`engine/scripts/eval_translate.py`](../engine/scripts/eval_translate.py) | does it translate the sentence, or just talk about translating it? |

The data lives next to the tools, so a re-run needs no download:

| File | Contents |
|---|---|
| [`engine/scripts/eval/mc_zh.jsonl`](../engine/scripts/eval/mc_zh.jsonl) | 50 Chinese 4-option items (math 10, chinese 8, history 8, geography 7, science 9, common_sense 5, logic 3; easy 15 / medium 25 / hard 10) |
| [`engine/scripts/eval/mc_en.jsonl`](../engine/scripts/eval/mc_en.jsonl) | 50 English 4-option items (science 14, common_sense 12, math 9, language 6, geography 5, history 4; same difficulty split) |
| [`engine/scripts/eval/ifeval.jsonl`](../engine/scripts/eval/ifeval.jsonl) | 24 instruction-following cases (zh 10, en 14), each with machine checks and a good/bad example |
| [`engine/scripts/eval/translate.jsonl`](../engine/scripts/eval/translate.jsonl) | 15 sentence pairs (daily/idiom/news/science/technical) run in **both** directions; each direction's reference is the other direction's source |

## The multiple-choice battery

### Method

One engine run per item: the prompt is prefilled, `--gen 0` (no decode), and
`LOGITS_OUT` carries the logits of the last prompt token -- the position where
the answer starts.  The item is rendered as

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
`":A"` token cannot be compared with a standalone `"A"`).  Run
`--out <dir>` keeps them in `<model>.letters.json`; the MiniMind pair gives
68-71, the LFM 542-545.

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

### Run

```bash
python3 engine/scripts/bench_mc.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
  --model minimind3-moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
  --model lfm2.5-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --data engine/scripts/eval/mc_zh.jsonl \
  --data engine/scripts/eval/mc_en.jsonl \
  --out /tmp/eval_mc --resume
```

One item is 0.9 s on the dense model, 2.6 s on the LFM and 2.7 s on the MoE
(measured, the same machine as the results below), so the full 100-item,
three-model battery is ~10 minutes on the laptop GPU.  `--resume`
appends to `<out>/<model>.items.jsonl` and skips items already scored, which is
what a battery cut off by a tool timeout needs.  Every item's four probabilities
are kept in that file: a surprising number can be re-checked without the GPU.

## The instruction-following battery

### Method

Each case is a prompt plus a list of checks; a case passes when every check
passes.  Cases are sampled `--seeds` times (default 3) with the model's own
recommended sampling preset:

| Preset | Settings |
|---|---|
| `minimind` | `--temp 0.85 --top-k 50 --topp 0.85` |
| `lfm` | `--temp 0.1 --top-k 50 --repeat-penalty 1.05` |
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

### Run

```bash
python3 engine/scripts/eval_ifeval.py --data engine/scripts/eval/ifeval.jsonl --selftest

python3 engine/scripts/eval_ifeval.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf --preset minimind --seeds 3 \
  --data engine/scripts/eval/ifeval.jsonl --out /tmp/eval_ifeval
```


## The translation battery

### Method

Translation was the one capability the chat comparison got wrong in both
directions: one sampled reply to one sentence said the MiniMind pair "can
translate" and the MoE "cannot", and nothing in that test separated "translated
the sentence" from "finished a sentence that started like a translation".  This
battery uses a fixed bilingual corpus and a reference-based metric.

15 sentences, each run in **both** directions (30 items: 12 daily, 3 idiom,
1 science, 1 technical, 2 news -- the same sentence is the reference for the
other direction).  Prompts are greedy: translation is a deterministic task and
the sampler noise of a 64M model swamps the signal.  The instruction is Chinese
in both directions ("把下面这句话翻译成英文/中文，只输出译文。" + the source),
for the same reason the multiple-choice battery uses a Chinese answer
instruction.

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

### Run

```bash
python3 engine/scripts/eval_translate.py \
  --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
  --model minimind3-moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
  --model lfm2.5-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
  --data engine/scripts/eval/translate.jsonl --out /tmp/eval_translate
```

30 items per model, under a second per item per model on the laptop GPU; the
battery is small on purpose -- it is long enough to see a trend and short enough
that it can sit in the same run as the other two.

## Results

Machine: laptop, Intel Iris Xe (ADL GT2), Mesa 26.2.5, `GPU_LAYER=1
KV_PACK=16`, 2026-11-12.  Containers: `minimind3.q8.mmcf`,
`minimind3-moe.q8.mmcf`, `lfm2.5.qad.q4.mmcf` (QAD, the vendor's
quantization-aware-distilled q4).  System prompt: `You are a helpful
assistant.`

### Multiple choice

100 items (50 zh + 50 en), one deterministic run each.  "Null" is the accuracy
the same prediction letters would score against shuffled gold labels (200
permutations, fixed seed): an accuracy at the null means the letter choice
carries no information about the question.

| Model | acc | acc_cal | fmt | p_gold | null (acc / acc_cal) | zh | en |
|---|---:|---:|---:|---:|---|---:|---:|
| MiniMind3 dense q8 (64M) | 0.220 | 0.320 | 0.880 | 0.255 | 0.253 ± 0.032 / 0.260 ± 0.033 | 0.16 | 0.28 |
| MiniMind3-MoE q8 (198M-A64M) | 0.250 | 0.250 | 0.620 | 0.254 | 0.251 ± 0.017 / 0.244 ± 0.035 | 0.24 | 0.26 |
| LFM2.5-230M QAD q4 | **0.380** | **0.400** | 0.880 | 0.404 | 0.250 ± 0.038 / 0.254 ± 0.033 | 0.28 | **0.48** |

Chance is 0.25.  Reading the table:

* **The LFM is the only model with signal**: 0.380 against a null of
  0.250 ± 0.038 is 3.4 sigma, and its calibrated 0.400 against 0.254 ± 0.033 is
  4.4 sigma -- the signal survives both the letter prior and the label
  shuffling.  Its accuracy is common sense (15/17 on that
  category) and science (11/23), with **arithmetic at 1/19** -- the same split
  the chat comparison showed from the other direction (it terminates and
  formats well, it cannot compute).
* **MiniMind3 dense sits at the null**: 0.220 against 0.253 ± 0.032.  It picks
  "C" 65 times out of 100 (A 30, B 4, D 1), so the letter is a positional
  prior, not a decision.  `acc_cal` (0.320) removes the prior and comes out
  at 1.8 sigma over its own null (0.260 ± 0.033) -- a lead, not a result.
* **The MoE answers "A" 94 times out of 100**, is at the null exactly (0.250
  against 0.251 ± 0.017), and only puts a letter first 62% of the time
  (`fmt`): one fifth of the items it starts the reply with something else,
  which the 8-host-sync route does not explain -- this is a model property.
  It is also the slowest of the three (2.7 s/item against 0.9 s).
* **Difficulty does not order the models.**  Neither MiniMind improves on
  "easy" items; the LFM's easy/medium/hard split is 0.43 / 0.40 / 0.25, which
  is the one monotone ordering in the three (the MiniMind pair is flat, i.e.
  noise).

Per-item probabilities are in `<out>/<model>.items.jsonl`; the per-model JSON
summary carries the per-category, per-language and per-difficulty splits.

### Instruction following

24 cases, 2 samples each (the battery was cut from 3 to 2 seeds to fit the
machine time; a case that needs luck still shows up as a gap between pass rate
and hard pass).  Presets: `minimind` for the MiniMind pair, `lfm` for the LFM
(see the table above).  The reply is scored exactly as it comes out of the
engine, with no fence stripping and no case folding:

| Model | pass | hard pass | zh | en | most-failed checks (count over 48 attempts) |
|---|---:|---:|---:|---:|---|
| MiniMind3 dense q8 (64M) | 0.188 | 0.167 | 0.25 | 0.14 | max_words 8, json_valid+json_keys 6+6, bullets 4, starts_with 4, ends_with 4, max_chars 4 |
| MiniMind3-MoE q8 (198M-A64M) | 0.188 | 0.167 | 0.25 | 0.14 | identical pattern to the dense model (json 6+6, max_words 8, bullet/prefix/suffix/char 4 each) |
| LFM2.5-230M QAD q4 | **0.562** | **0.542** | 0.45 | **0.64** | json_valid+json_keys 6+6, max_words 6, max_chars 4, bullets 2, absent_regex 2 |

The per-case pattern is more informative than the totals:

* **The LFM passes exactly the cases a chat user would notice**: JSON is the one
  family it loses (all three JSON cases, 6 of 6 attempts, because it wraps the
  object in \`\`\`json fences even when the prompt says not to), plus the
  one-word/character-budget cases where it answers with a full sentence
  ("天空是蓝色的。" for "用一个词回答").  Everything else -- ALL CAPS, exact
  prefix/suffix, numbered lists, bullets, two-line answers, the date-arithmetic
  regex -- passes on both seeds.
* **The MiniMind pair passes only the "write something long" cases**: if-07,
  if-08, if-13 and if-15 are the length/language checks, which any verbose
  answer passes; every case with an exact format fails.  Their per-case
  patterns are *identical*, which is what an instruction-following battery
  looks like when neither model has learned the instruction meta-skill: the
  failure mode is the tuning, not the parameter count (the MoE has 3x the
  parameters and scores the same).
* **The gap is the product**: 0.19 vs 0.56 on the same 24 cases, in the same
  direction as the chat comparison (termination, formatting, English), and in
  the same direction as the multiple-choice table (LFM 0.38, MiniMind at
  chance).  The two batteries disagree on *which* skill a model lacks -- the
  LFM can follow instructions and cannot compute, the MiniMind pair does
  neither.

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

30 items (15 sentences x 2 directions), greedy, chrF against the reference.
"tgt/leak" are the target/source script fractions; see the method section.

| Model | chrF (greedy) | zh→en | en→zh | chrF ≥ 0.5 | chrF ≥ 0.3 | copies source | meta-preamble | len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniMind3 dense q8 (64M) | 0.130 | 0.203 | 0.058 | 2/30 | 2/30 | 9/30 | 15/30 | 3.0 |
| MiniMind3-MoE q8 (198M-A64M) | 0.133 | 0.172 | 0.094 | 1/30 | 2/30 | 7/30 | 12/30 | 2.1 |
| LFM2.5-230M QAD q4 | **0.328** | **0.368** | **0.287** | **8/30** | **13/30** | 3/30 | 2/30 | 1.0 |

Sampling does not rescue the MiniMind pair: with each model's own recommended
preset (MiniMind temp 0.85 / top-p 0.85 / top-k 50, LFM QAD temp 0.1 / top-k 50
/ repetition 1.05, seed 42) the numbers are dense **0.085**, MoE **0.130**, LFM
**0.335**.  The LFM's translation is stable across greedy and sampled decoding;
the dense model's is not (0.130 -> 0.085, and its en→zh cell drops to 0.017).

Reading it:

* **Only the LFM translates.**  8 of 30 outputs are usable (chrF >= 0.5) and
  13 of 30 are half-right (>= 0.3); its mean length is 1.0x the reference and
  it almost never talks about translating (2/30 preambles).
* **The MiniMind pair does not translate; it describes the request.**  Dense
  emits "Here is the translation of the sentence \"…\" into English:" in
  half the cases (15/30, 12 of them zh→en), and for en→zh it repeats the English
  source instead of translating it: 9 of its 15 en→zh outputs are more than
  half source script, the direction scores 0.058 (0.017 sampled) -- the worst
  cell in the battery.  The MoE mixes
  the two languages inside one sentence ("请关闭 window; it is very cold
  outside."), loops, and preambles in 12/30.
* **A human paraphrase scores ~0.65, not 1.0, on this corpus** (measured:
  "The weather is lovely today." against "The weather is nice today." gives
  0.651).  So LFM's 0.33 means "half the content is wrong", not "almost right":
  it translates short simple sentences well (0.71, 0.76, 0.84 are real:
  "Learning a new language requires time and patience." / "Scientists
  discovered that this plant can thrive in deserts."), and it fails numbers
  ("ten million" -> "a million", "twenty minutes late" -> "two hours late"),
  idioms, and complex clause structure.  0.13 means "the sentence is not there".
* **A real MT fine-tune on the same engine is a different league**: the
  official LFM2-350M-ENJP-MT (EN<->JA) packed into a q4 container scores 0.559
  on a 9-sentence probe against 0.419 for the untuned 230M base -- and 0.675 for
  the vendor GGUF through llama.cpp.  The build steps, the two packing traps and
  the numbers are in [lfm2.5-230m.md](lfm2.5-230m.md#a-translation-fine-tune-of-the-same-architecture-lfm2-350m-enjp-mt).
* **The chat comparison's "the dense model translates" was one sentence.**  That
  sentence ("今天天气很好") is one of the 2/30 this battery scores above 0.5;
  the other 28 were never in that test.  This is the same lesson as the
  multiple-choice battery: one sampled reply per prompt decides nothing.

The raw per-attempt replies (ifeval), per-item probabilities (multiple choice)
and hypotheses (translation) are in each run's `--out` directory next to the
summaries; the tables above were produced with `--seeds 2` and the presets
stated in the method sections.

## What these numbers mean, and what they do not

* **100 items is a coarse instrument.**  With 100 questions the standard error
  of an accuracy near 0.25 is about 0.043, so differences under ~0.09 between
  two models are not resolved.  Read the per-category columns as leads and the
  totals as ±0.05.
* **Chance is 0.25, and the letter prior is real.**  The 64M model answers "C"
  65 times out of 100 and the MoE answers "A" 94 times; against balanced gold
  those constants alone score ~0.25.  `acc_cal` subtracts the letter prior and
  the null column says what the same letters would score against shuffled gold,
  so "0.22" or "0.25" is not read as "the model tried".
* **The English instruction problem.**  With "Answer with the option letter
  only." the MiniMind pair replies with prose to English items ("Based on...",
  "The question is...") under greedy decoding, for every English phrasing
  tried; the same question behind the Chinese instruction gets a letter from
  all three models.  The battery therefore uses the Chinese instruction for
  both languages, and `fmt` reports compliance anyway.  This is a property of
  the models, not of the harness: a per-language instruction would have
  measured it instead of knowledge.
* **Instruction following is a separate axis.**  A model can know the answer
  and still fail a case because it added "Here are three bullets:" above the
  bullets or wrapped JSON in a code fence.  The ifeval pass rate is the number
  a chat user feels first, and it does not correlate with the MC accuracy.
* **This is not a leaderboard.**  The items are our own, deliberately easy
  (they are meant to be answerable by a bright child, which is roughly the
  ceiling of a 64M model).  They do not compare with published C-Eval / ARC
  numbers, which use thousands of items and full-size models.
* **Quantization is part of the model.**  The LFM is measured at QAD q4 because
  that is the container that works best for it; the MiniMind pair at q8.  A
  cross-precision comparison is a different experiment (`bench_mc.py` takes
  any container, so run both).

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
