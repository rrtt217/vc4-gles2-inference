# Writing a page in `docs/`

Rules for agents changing this directory.  What applies to every file in the
tree — the two tiers, the "delete it and ask whether someone makes a wrong
decision tomorrow" test, which document owns which fact — is in
[`../AGENTS.md`](../AGENTS.md) under "Documentation rules".  This file is the
part that is specific to `docs/`, and the part no gate can check.

The root `README.md` and [`README.md`](README.md) here are for **human**
readers: what this is, how to start, where to look.  Do not turn either of them
into a writer's guide; agent instructions live in `AGENTS.md` files.

## One page per subject

| the subject | where it goes |
|---|---|
| a model the engine runs | `docs/<model>.md` — one page, carrying everything about that model: what it is, its containers, how to run it, its numbers, its gaps |
| a mechanism (the conv block, the prefix cache, the top-k path) | a section of [`engine-internals.md`](engine-internals.md) or a page of its own, whichever the reader hits first |
| a number that will be re-measured | a row in [`benchmarks.md`](benchmarks.md) |
| a quality score | a row in [`evaluation.md`](evaluation.md) |
| an investigation, a plan, an assessment | **no page.**  The commit message; the conclusions move into the page above that owns them |

A plan is scaffolding and is deleted when its last task closes — after its
operator notes move into the mechanism page and its numbers into the table.  An
assessment is absorbed the same way, and survives only when it is the sole
measurement of an option that was rejected.  Both happened on 2026-11-19
(`2393b5b`): four pages went away and nothing was lost, because each had a
home to go to.

## The sections, in this order

Copy a page instead of inventing a shape.  A model page is:

1. one paragraph — what the model is and what it costs, before the first table;
2. a key/value table — container, shape, state, vocab, tokenizer, template,
   specials, metadata keys: whatever a reader needs to say "that is what it is";
3. `## Containers` — the files, their sizes, which one runs where, and the
   pack command that produces them;
4. `## Run` — the commands, with the switches that are **required** spelled out
   and the ones that are optional marked as such;
5. `## Quality` — the numbers with the protocol that produced them;
6. `## Known gaps` — a table of what is broken, what was accepted and why;
7. `## Measuring it` — how to reproduce the numbers above.

[`bitcpm.md`](bitcpm.md) is the short example (266 lines).
[`lfm2.5-230m.md`](lfm2.5-230m.md) is the long one (421: two architectures and
a fine-tune).  [`smollm2-135m.md`](smollm2-135m.md) is a page whose subject is
mostly one open question, and [`rwkv7.md`](rwkv7.md) is a page for an
architecture that was delivered and then deliberately closed.

## What a paragraph may be

Three kinds, and a paragraph is exactly one of them:

- **state** — present tense, what is true now, no date, no "used to";
- **evidence** — why it is this way: one dated line and the commit that
  established it, sitting next to the invariant or the gap it protects;
- **a number** — with the machine and the protocol it was measured under.

Nothing else.  A narrative of what happened, in order, belongs in a commit
message, which is also what `git log -S` finds later.

## Shape

- **A page that grows with every campaign, or that says the same thing twice,
  is a bug even when every line is true.**  `benchmarks.md` reached 8296 lines
  of dated sections and `evaluation.md` 708 lines holding four copies of one
  table; they are 179 and 538 now (`f733573`, `7c14966`).  Those sizes are the
  symptom, not the rule: the question is whether a reader has to diff two
  sections to learn which number is current.
- **A heading is a subject, never a date.**  `## 2026-11-16 — RWKV-7` is the
  shape that grows without bound, and it is how both of those happened.
- **Tables over prose** once there are more than two items.  The same table
  twice, in two sections, is the failure above.
- **One row per container** in a results table, never one table per run.
- **This directory is flat on purpose.**  `gate/docs_check.sh` and
  `gate/check_docs_links.py` glob `docs/*.md`, so a subdirectory would silently
  fall out of the link and path checks.  Grouping lives in the sections of
  [`README.md`](README.md); if a split ever earns its cost, those two globs
  change in the same commit.

## Cross-file bookkeeping

Adding a model is not a page, it is six files.  Miss one and the page is a dead
end:

| file | what goes there |
|---|---|
| `docs/<model>.md` | the page |
| [`README.md`](README.md) | its row, in the Models section, not appended at the bottom |
| [`benchmarks.md`](benchmarks.md) | a decode row with machine and date, and its PPL if it pins one |
| [`evaluation.md`](evaluation.md) | its row in each battery it was scored on |
| `../AGENTS.md` | a row in "Model-specific constraints" if a change can break it |
| `engine/scripts/regression.sh` and the machine's override | its pins, if it is in the gate |

## Naming

`docs/<subject>.md`: lower case, hyphens, no date, no `v2`.  `*.zh-CN.md` is
for the README pairs only (see `../AGENTS.md`); a deep dive is English, so that
the next agent can read the whole tree.  `AGENTS.md` in this directory is a
rule file, not a document: the index has no row for it, and it is held to the
same style check as `../AGENTS.md`.

## Before you commit

```bash
bash engine/gate/docs_check.sh     # step 0 of make -C engine check
```

It checks that every link and `#anchor` resolves, that every `README.md` entry
exists and is tracked, that every in-tree path a document tells you to run
exists, and that the switch registry agrees with the documents.  It cannot see
the three failures that actually happened here: a page that repeats another
page, a section that is a date instead of a subject, and a paragraph that only
history would need.  Read your own diff for those, and run the delete test on
every paragraph that carries a date.