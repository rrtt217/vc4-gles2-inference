# `engine/tests/` — compiled tests

C++ tests that the Makefile builds and runs.  They are here because they are
*code*, not scripts: they link the engine's own translation units, so they need
a compiler and nothing else.  `make -C engine test` builds and runs them in a
few seconds; `make -C engine check` runs the same target first, so a broken unit
fails before the minute-long `warncheck` build.

`checks.h` is the shared assertion harness, not a test framework: a test is a
program with a `main()` that calls `CHECK*` and returns non-zero on failure.
That is enough for exact-value tests (an id list, a packed byte string), and it
keeps the build a single `g++` call with no third-party source or warning
suppression.

| File | Answers | Needs | Wired into |
|---|---|---|---|
| `test_lineedit.cpp` | the line editor's cursor arithmetic, its paste sanitiser, and the editor driven through a pty | a compiler (`make -C engine test`) — no GPU, no model, no tty | `make test`, `make check`, CI |
| `test_tokenizer.cpp` | the `.tok` blob parser, BPE rank, special-token atomicity, the user/template split in `encode_parts`, decode and reload | a compiler — a synthetic blob, no model | `make test`, `make check`, CI |
| `test_xformat.cpp` | the CPU half of every X format: texel counts and `x_pack`'s bytes for q16, i8, i24, f32 and bf16 | a compiler — no GPU, no model | `make test`, `make check`, CI |
| `test_grammar.cpp` | the JSON constraint's byte automaton: valid/invalid/prefix JSON, tokens spanning construct boundaries, EOS-only-when-done, and 400 random valid values plus every prefix | a compiler — a synthetic byte vocabulary, no model | `make test`, `make check`, CI |
| `test_schema.cpp` | the generic LL(1) matcher (reproduces the JSON corpus) and the schema compiler: typed/closed objects, required, enum, nested arrays, shared-prefix number enums, refused combinators, 400 random valid JSON | a compiler — synthetic byte vocabulary, no model | `make test`, `make check`, CI |
| `test_gbnf_full.cpp` | full GBNF: the general multi-stack matcher on ambiguous/shared-prefix grammars, repetition `* + ? {n,m}`, inverse ranges, groups, arithmetic + JSON grammars, Unicode code-point ranges and multi-byte tokens split across two tokens, and parser refusals (left recursion, undefined rule) | a compiler — synthetic byte vocabulary, no model | `make test`, `make check`, CI |
| `test_constraint.cpp` | the CLI/REPL constraint surface: the `/constrain` spec parser, the choice shortcut's escaped GBNF text (quotes, backslashes, spaces, CJK), each tier installed end to end (choice, json_object, a GBNF file, a JSON Schema file), the per-reply reset, and that a failed build keeps the previous constraint | a compiler — a synthetic byte vocabulary, no model | `make test`, `make check`, CI |
| `test_weight_format.cpp` | the weight-format contract: row order, stage caps, slice alignment, accepted `XFormat`s and `mat_bytes`' arithmetic | a compiler — header-only, no GPU, no model | `make test`, `make check`, CI |
| `checks.h` | the shared assertions (`CHECK`, `CHECK_LL`, `CHECK_STR`, `CHECK_INTS`, `CHECK_BYTES`) | — | included by `test_tokenizer.cpp`, `test_xformat.cpp`, `test_grammar.cpp`, `test_schema.cpp`, `test_gbnf_full.cpp`, `test_constraint.cpp` and `test_weight_format.cpp` |
