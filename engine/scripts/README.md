# `engine/scripts/` — oracles, measurement and acceptance

Everything here **needs a GPU, a model, or a dump made by one**.  The pass/fail
checks that can run without them are in `../gate/`; the model-quality batteries
are in `../eval/`.

Two conventions this directory has paid for: an `*_ab.sh` script interleaves two
binaries so drift cancels (two builds hide a 1 ms effect), and every measurement
script sources `lib.sh` rather than carrying its own copy of the model defaults,
the load guard and the statistics.

| File | Answers | Needs | Wired into |
|---|---|---|---|
| `regression.sh` | the model gate: determinism, default prompt, pinned strings, behaviour, PPL, RWKV | GPU + containers + the machine's pins | manual, before pushing numerics |
| `lib.sh` | the shared half: tree root, model defaults, `measure_env`, `load_guard`, timing fields, min/median | — | sourced by every `bench*`, `longctx.sh` |
| `bench.sh` | decode/prefill timing for one container | GPU + model | manual |
| `bench_ab.sh` | the same, two binaries interleaved | GPU + models | manual |
| `bench_plain_ab.sh` | the same for a plain (non-chat) prompt | GPU + models | manual |
| `longctx.sh` | how the timing and the quality move with `SEQ` | GPU + model | manual |
| `supra2-battery.sh` | a smoke battery for the supra2 CPU path | model | manual |
| `format_sweep.sh` | every value of every stream axis (`ACT`, `RES`, `FFN_HID`, `PROJ`, `KV`), alone and paired with `ACT`, produces the *same* assistant text -- a format hardcoded inside a kernel is invisible until the axis moves.  `ENGINE=<binary>` runs it from a runtime directory with no checkout, which is how the board is swept: on VC4 the same script also answers "does this kernel exist on this compiler", a question the laptop cannot ask | GPU + a chat container with a pinned answer (~43 runs) | manual, after touching a kernel that reads or writes two streams |
| `swap_matrix.py` | `/model` swaps are byte-identical to a fresh process | GPU + models (~3 min) | manual, before touching the loader |
| `verify_packing.sh` | the packed-activation layout against a known-good binary | a reference binary | manual |
| `diff_paths.sh` | which stage differs between the CPU and GPU paths | GPU + model | manual, with `dump_diff.awk` |
| `dump_diff.awk` | the per-stage comparison itself | `STAGE_DUMP` files | called by `diff_paths.sh` |
| `dump_stats.py` | per-row max/mean/clamp count of a dump (is a row saturating) | a `DUMP` file | manual |
| `check_att.py` | the attention block's score/softmax/output recomputed from the integers it consumed | an `ATT_DUMP` log | manual, the offline oracle |
| `cmp_att.py` | where do two machines stop agreeing, at the finest resolution the engine prints?  Compares two `ATT_DUMP` logs element by element, layer by layer, per attention pass; `check_att.py` is the one-machine oracle, this is the cross-machine diff |
| `cmp_logits.py` | two `LOGITS_OUT` dumps, tile by tile | two dumps | manual |
