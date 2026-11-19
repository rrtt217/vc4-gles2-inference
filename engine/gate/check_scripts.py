#!/usr/bin/env python3
"""The script directories against their READMEs.

Four directories hold code that is not the engine, split by what it *needs*:

  engine/gate/     pass/fail, no GPU and no model store
  engine/scripts/  oracles, measurement and acceptance: a GPU and/or a model
  engine/eval/     model quality: the engine plus a container
  engine/tests/    compiled C++ tests

Each has a README whose table is the map: one row per file, with what it answers,
what it needs and what calls it.  This checks both directions -- a file with no
row, and a row with no file -- because the failure mode is a script nobody can
find (AGENTS.md records a long investigation that hand-rolled a probe which
`engine/scripts/` already contained).

A row may name a path inside the directory (`data/mc_zh.jsonl`) and may name a
*set* of files in one cell (`data/a.jsonl`, `b.jsonl`); both are resolved against
the directory.

usage: engine/gate/check_scripts.py [-v]      (no GPU, no model, no build)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
DIRS = ["gate", "scripts", "eval", "tests"]
ROW = re.compile(r'^\|\s*`([^`]+)`')
# README.md is the map; AGENTS.md is the rule file the map points to (the
# docs/AGENTS.md precedent: a rule is not a check and has no row).
SKIP = {"README.md", "AGENTS.md", "__pycache__", ".DS_Store"}


def files_in(d):
    root = os.path.join(ENGINE, d)
    if not os.path.isdir(root):
        return None
    return sorted(f for f in os.listdir(root)
                  if not f.startswith((".", "__")) and f not in SKIP
                  and os.path.isfile(os.path.join(root, f)))


def rows_in(d):
    path = os.path.join(ENGINE, d, "README.md")
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path, encoding="utf-8"):
        m = ROW.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def main():
    verbose = "-v" in sys.argv
    fail = 0
    total = 0
    for d in DIRS:
        files, rows = files_in(d), rows_in(d)
        if files is None:
            print("FAIL  engine/%s/ is missing" % d)
            fail = 1
            continue
        if rows is None:
            print("FAIL  engine/%s/README.md is missing (it is the map)" % d)
            fail = 1
            continue
        total += len(files)
        # a row may be `dir/file.jsonl` or a list of names in one cell
        named = set()
        for r in rows:
            for part in re.split(r'[,\s]+', r):
                part = part.strip().strip("`")
                if part:
                    named.add(part.split("/")[-1])
        for f in files:
            if f not in named:
                print("FAIL  engine/%s/%s has no row in %s/README.md" % (d, f, d))
                fail = 1
        for r in rows:
            for part in re.split(r'[,\s]+', r):
                part = part.strip().strip("`")
                if not part:
                    continue
                p = os.path.join(ENGINE, d, part)
                if not os.path.exists(p):
                    print("FAIL  engine/%s/README.md has a row for %s, which does "
                          "not exist" % (d, part))
                    fail = 1
        if verbose:
            print("  engine/%-8s %2d files, %2d rows" % (d + "/", len(files), len(rows)))
    if fail:
        print("\nsee engine/*/README.md: the table is the map of the directory.")
        return 1
    print("scripts: ok (%d files in %d directories, each with a row)" % (total, len(DIRS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
