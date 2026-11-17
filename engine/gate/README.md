# `engine/gate/` — the checks that need no GPU

One file per check, plus the two libraries they share and one golden data file.
**A check belongs here if it can run on a machine with no GPU and no model
store**: it reads the source, the generated GLSL, the documents, or a container
file.  If it needs a GPU or a model it belongs in `../scripts/`, if it scores a
model it belongs in `../eval/`.

`docs_check.sh` is the umbrella for the documentation ones and is step 0 of
`scripts/regression.sh`; the rest are called by `make -C engine check` and by CI.
This table is checked against the directory by `check_scripts.py` — a file with
no row, or a row with no file, fails the gate.

| File | Answers | Needs | Wired into |
|---|---|---|---|
| `docs_check.sh` | all of the documentation rules below, in one command | — | `make check`, CI, regression step 0 |
| `check_switches.py` | every switch the source reads has a registry row, every row is read, no retired row is read, every row is documented | — | `docs_check.sh` |
| `gen_test_hooks.py` | is `docs/test-hooks.md`'s inventory the registry's rendering (`--check`) | — | `docs_check.sh` |
| `switch_registry.py` | the parser for `src/switches.cpp` that the two above import | — | (library) |
| `check_docs_style.py` | canonical switch spellings in commands; four-decimal numbers carry a date | — | `docs_check.sh` |
| `check_docs_links.py` | relative links and anchors resolve; the docs index is tracked; a documented switch is read | git (for the tracked check) | `docs_check.sh` |
| `shader_hashes.sh` | did a builder change move the emitted GLSL (`--dump-shaders` digests) | the built binary | `make check`, CI, regression step 0b |
| `shader_hashes.txt` | the golden digests `shader_hashes.sh` compares against | — | (data) |
| `check_q16_decode.sh` | every q16 decode in the generated GLSL folds the negative rail with one of two spellings | the built binary | `make check`, CI, regression |
| `check_draw_cap.sh` | no bare `1024` in the strip loops of `src/gpu*.cpp` | — | `make check`, CI |
| `check_target_reset.sh` | every scratch `Target` member is cleared by `reset_targets()` | — | `make check`, CI, regression |
| `test_mmcf_format.py` | every container round-trips through `tools/mmcf_format.py`, and its constants match `model.cpp` | a model store for the round trip (CI has none: constants only) | `make check`, CI |
| `check_scripts.py` | this table and the three below match their directories | — | `docs_check.sh` |
