# `engine/tests/` — compiled tests

C++ tests that the Makefile builds and runs.  They are here because they are
*code*, not scripts: they link the engine's own translation units, so they need
a compiler and nothing else.

| File | Answers | Needs | Wired into |
|---|---|---|---|
| `test_lineedit.cpp` | the line editor's cursor arithmetic, its paste sanitiser, and the editor driven through a pty | a compiler (`make -C engine test_lineedit`) — no GPU, no model, no tty | `make check`, CI |
