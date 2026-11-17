# `engine/eval/` — model quality

Accuracy, instruction-following and translation for a model, not a pass/fail on
the engine.  These need the engine binary and a container; they are run by hand
when a model or a quantisation changes, and their numbers go to
`docs/evaluation.md`.

`evalkit.py` is the shared half (one engine process per call, `--chat`/`--gen 0`
with `LOGITS_OUT`, the sampling table, the case loader); a battery that carries
its own copy of that is the bug this file exists to prevent.

| File | Answers | Needs | Wired into |
|---|---|---|---|
| `evalkit.py` | the shared client and case loader | engine + container | imported by the three batteries |
| `bench_mc.py` | letter-logit accuracy, calibrated accuracy and format compliance | engine + container | manual |
| `eval_ifeval.py` | does the reply obey the instruction (`--selftest` checks the case file) | engine + container | manual, `--selftest` in CI |
| `eval_translate.py` | does it translate, or talk about translating (chrF, script leakage) | engine + container | manual |
| `worker_engine.py` | one engine process for a whole battery (a Pi Zero start is ~15 s) | engine + container | imported by the batteries |
| `data/ifeval.jsonl`, `data/mc_*.jsonl`, `data/translate.jsonl` | the cases | — | read by the batteries |
