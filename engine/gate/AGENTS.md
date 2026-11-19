# Writing a gate in `engine/gate/`

Rules for agents adding or changing a check here.  The map is
[README.md](README.md): one row per check, what it needs and what calls it.
This file is the part the table cannot hold -- the contract every gate keeps,
the language to write it in, and the negative run that proves it is a gate.

Like [docs/AGENTS.md](../../docs/AGENTS.md), this is a rule file, not a
document: the docs index has no row for it, and `docset.py` scans it for links
and style like any other page.  The general rules -- the two tiers, which
document owns which fact -- are in [AGENTS.md](../../AGENTS.md).

## The contract

`make check`, CI and a reader get two signals from a gate: the exit code and the
one-line verdict.  Keep both.

1. **One question per file.**  If the header needs an "and", it is two gates.
2. **Exit 0 or 1; 2 for a usage error.**  0 = the rule holds, 1 = it drifted or
   the gate could not run, 2 = the command line is wrong (a gate with modes,
   like `shader_hashes.sh`).  A gate that checked nothing never exits 0.
3. **The verdict is one line.**  Success prints `<name>: ok (<counts>)`; the
   count is what makes a check that inspected nothing visible.  Failure prints
   `FAIL  ...` to stderr with the file and line and what to change.  New gates
   use the two-space `FAIL  ` form (`check_target_reset.sh` predates it).
4. **Find the tree from the script**, not the caller: `cd "$(dirname "$0")/.."`
   in sh, `Path(__file__).resolve()` in Python, so it runs from the repository
   root and from `engine/`.
5. **No GPU and no model store required; no network.**  The Needs column names
   the one extra input a gate may take -- the built binary, `git`, or a model
   store -- and no undocumented environment variable decides what is checked.
6. **Deterministic.**  No timestamps, and no `sort`/`comm` on names: locale
   collation changes the answer.  `check_target_reset.sh`'s `has()` is the
   membership test to copy.
7. **Fail closed, and say when you skip.**  A missing input is a failure unless
   it is a supported absence (there is no model store), in which case print
   `SKIP` and keep checking what can be checked (`test_mmcf_format.py` still
   compares the C++ constants with no containers).
8. **Wire it in.**  A row in the table above (`check_scripts.py` fails without
   it) and a call from `make -C engine check` -- or from `docs_check.sh` if it
   belongs to the documentation family.  CI's gate step is `make check`, so a
   check with a row and no caller never runs on a push.

## Which language

The input decides, not preference:

| the gate... | write it in | copy |
|---|---|---|
| runs a command and compares or diffs its text | sh | `shader_hashes.sh`, `check_q16_decode.sh` |
| greps one rule out of source text | sh | `check_draw_cap.sh` |
| parses a table, registry, Markdown or binary header | Python 3, stdlib | `check_switches.py`, `check_scripts.py` |
| must reuse an existing parser | that parser's language | `gen_test_hooks.py` -> `switch_registry.py` |
| aggregates other gates | sh | `docs_check.sh` |

Shell for "run and compare", Python for "parse and decide".  When a gate does
both, put the judgment on one side and call the other: `check_docs_links.py`
runs `git`, and `shader_hashes.sh` runs the binary and diffs its output.

## The negative run

A gate is not known to be a gate until it has failed once on purpose.  Make the
condition false -- edit the source line, point it at a saved dump, delete a row
-- run it, and confirm both the exit code and the message.  `check_q16_decode.sh`
takes a dump file for exactly this; `shader_hashes.sh --show` and `--rebaseline`
are the same idea for the golden file.

## Before you commit

- Run the negative test; a check that has never been red is a hope.
- Add the row and the caller in the same commit.  The row is enforced by
  `check_scripts.py`; the call is not enforced yet, so verify it by hand.
- Run `make -C engine check` in full -- it is the list CI runs.
