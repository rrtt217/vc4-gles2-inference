#!/usr/bin/env python3
"""Documentation style: state records the present, history records the past.

Two rules, both of them mechanical, both of them about the same mistake -- a
reference document that carries history instead of state:

  **R1 -- commands use the canonical switch spelling.**  `KV_PACK=16` still
  works (the registry translates it, see `engine/src/switches.cpp`), but a
  command in a document is the *teaching* copy: a reader who learns it from
  there writes the superseded name.  So inside a fenced code block, or on a line
  that is a shell command, only the axis spelling is allowed.  The legacy table
  in `docs/test-hooks.md` is generated from the registry and is exempt, and so
  are the frozen pre-rule documents (the dated reports and the accumulated
  `benchmarks.md` entries), because rewriting those would edit the record.

  **R2 -- a measurement in a state document carries its date.**  A four-decimal
  number in a state document is a measurement, and a measurement is a fact about
  a moment, not about now: PPL is re-baselined deliberately (`regression.sh
  --rebaseline`) and the value that is true today is in the machine's pin file,
  not in a document.  So the number has to sit under a dated heading or in a
  dated paragraph, which makes it read as what it is.  The pre-rule sections of
  `docs/benchmarks.md` are exempt: they are dated by construction and frozen;
  the reference table at its top is a *state* document and is checked.

Run from anywhere: `engine/gate/check_docs_style.py [-v]`.  Exit 1 on drift.
Called by `gate/docs_check.sh`, so it is step 0 of the gate.
"""
import os
import re
import sys

import docset

ROOT = docset.ROOT

# The document set is derived, not listed: `docset.py` is the one place that
# decides what is scanned (tracked + untracked, minus the vendored pages), and it
# carries the short exemption list for the frozen record pages.  The hand lists
# this replaces had already left the four `engine/*/README.md` files outside the
# check.

LEGACY = re.compile(r'\b(KV_PACK|KV_FMT|KV_SCALE|ACT8|ACT_FMT|ACT_F32|RES_PACK|RES_FMT|'
                    r'ATT_SCALE|ATT_SUM_SCALE)=[A-Za-z0-9_]')
MEASURE = re.compile(r'\b\d{2,4}\.\d{4}\b')
DATE = re.compile(r'20\d\d-\d\d(-\d\d)?')

# A line that reads as a shell command even outside a fence.
COMMAND = re.compile(r'^\s*(?:[A-Z][A-Z_0-9]*=\S+\s+)*'
                     r'(?:MODEL=|GPU_LAYER=|\./engine_mm3|engine_mm3|sh scripts/|bash scripts/|'
                     r'python3 engine/|make -C engine)')

# A line that talks about the rename itself may quote the old spelling.
EXPLAIN = ("was ", "used to", "spelling", "alias", "legacy", "renamed", "removed",
           "retired", "->", "\u2192", "no longer", "previous", "before the fold",
           "the old ")

# The generated inventory carries the legacy spellings on purpose.
GEN_BEGIN = "<!-- BEGIN generated: switches"
GEN_END = "<!-- END generated: switches"


def scan(path):
    """(r1 hits, r2 hits) for one file, as [(line, text)].

    Both rules are decided per *paragraph*, because that is the unit a date or a
    "this is the old spelling" belongs to: a dated paragraph is a record, and a
    record may quote an old name and an old number.
    """
    r1, r2 = [], []
    in_fence = False
    in_gen = False
    heading = ""
    para = []          # [(lineno, text, in_fence)]
    caption_dated = False   # the paragraph above a table dates it
    lines = open(path, encoding="utf-8").read().split("\n")

    def flush():
        nonlocal caption_dated
        if not para:
            return
        text = " ".join(t for _, t, _ in para)
        is_table = para[0][1].lstrip().startswith("|")
        own = bool(DATE.search(text))
        dated = bool(DATE.search(heading) or own or (is_table and caption_dated))
        if not is_table:
            caption_dated = own
        explain = any(w in text.lower() for w in EXPLAIN)
        for lineno, line, fence in para:
            for m in LEGACY.finditer(line):
                if fence or COMMAND.match(line) or not (dated or explain):
                    r1.append((lineno, line.strip()[:110]))
            if dated:
                continue
            for m in MEASURE.finditer(line):
                r2.append((lineno, line.strip()[:110]))
        para.clear()

    for i, line in enumerate(lines, 1):
        if line.startswith(GEN_BEGIN):
            in_gen = True
        if in_gen:
            if line.startswith(GEN_END) or line.endswith("switches -->"):
                in_gen = False
            continue
        if line.startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if line.startswith("#"):
            flush()
            heading = line
            continue
        if not line.strip():
            flush()
            continue
        para.append((i, line, in_fence))
    flush()
    return r1, r2


def main():
    verbose = "-v" in sys.argv
    fail = 0
    for rel in docset.style():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("FAIL  %s: listed as a state document but missing" % rel)
            fail = 1
            continue
        r1, r2 = scan(p)
        for line, text in r1:
            print("FAIL  %s:%d uses a legacy switch spelling in a command\n      %s"
                  % (rel, line, text))
            fail = 1
        for line, text in r2:
            print("FAIL  %s:%d has an undated measurement (no 20YY-MM-DD in its "
                  "heading or paragraph)\n      %s" % (rel, line, text))
            fail = 1
    for rel in docset.history():
        if not os.path.exists(os.path.join(ROOT, rel)):
            print("FAIL  %s: listed as history but missing" % rel)
            fail = 1
    if verbose:
        print("  %d state documents, %d history documents exempt"
              % (len(docset.style()), len(docset.history())))
    if fail:
        print("\nsee the header of engine/gate/check_docs_style.py: a state "
              "document records the present; the record is Git.")
        return 1
    print("doc style: ok (%d state documents: canonical commands, dated "
          "measurements)" % len(docset.style()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
