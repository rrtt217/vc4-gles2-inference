#!/usr/bin/env python3
"""The switch registry against the source and against docs/test-hooks.md.

Four directions, all of them mechanical:

  src -> registry   every switch engine/src reads has a row
  registry -> src   every non-retired row is read somewhere (a row nobody reads
                    is a lie that outlives the code it described)
  registry rules    a retired row must not be read, a rename must point at a
                    canonical row and keep a value that axis accepts, a closed
                    set must be spelled a|b|c or lo..hi
  registry -> docs  every row's name is mentioned in docs/test-hooks.md

The registry and the reader set are parsed by gate/switch_registry.py, which
is also what renders the document (gate/gen_test_hooks.py), so there is one
parser and one definition of "the source reads this switch".

usage: engine/gate/check_switches.py [-v]
"""
import sys

import switch_registry as reg

DOC = reg.ROOT + "/docs/test-hooks.md"


def main():
    verbose = "-v" in sys.argv
    rows, renames = reg.load()
    by_name = {r.name: r for r in rows}
    reads = reg.reads()
    doc = open(DOC, encoding="utf-8").read()
    fail = 0

    def bad(msg):
        nonlocal fail
        fail = 1
        print("FAIL  " + msg)

    # --- the rules inside the registry ---------------------------------------
    for r in rows:
        if r.values and not reg_re_fullmatch(r.values):
            bad("registry row %s: values '%s' is not a|b|c or lo..hi" % (r.name, r.values))
        if r.status == "Retired" and reads.get(r.name):
            bad("registry says %s is retired, src still reads it (%s)"
                % (r.name, reads[r.name][0]))
        if not reads.get(r.name) and r.status != "retired":
            bad("registry row %s is read nowhere in src (a stale row)" % r.name)
        if r.name not in doc:
            bad("registry row %s is not mentioned in docs/test-hooks.md" % r.name)
    for legacy, value, target, becomes in renames:
        t = by_name.get(target)
        if not t:
            bad("rename %s -> %s, which is not a registry row" % (legacy, target))
            continue
        if t.status == "Retired":
            bad("rename %s -> %s, which is retired" % (legacy, target))
        if not becomes and value and t.values and value not in t.values.split("|"):
            bad("rename %s=%s keeps a value the target axis does not accept (%s)"
                % (legacy, value, t.values))

    # --- src -> registry -----------------------------------------------------
    known = set(by_name) | {l for l, _, _, _ in renames}
    for name, where in sorted(reads.items()):
        if name not in known:
            bad("src reads %s (%s), the registry has no row for it"
                % (name, where[0]))

    if verbose:
        print("  %d registry rows, %d renames, %d switches read by src"
              % (len(rows), len(renames), len(reads)))
        hint = reg.suspicious(known)
        if hint:
            print("  not a switch (a label or a write), seen as an ALL-CAPS first "
                  "argument: %s" % ", ".join(sorted(hint)))

    if fail:
        print("\nsee engine/src/switches.cpp: the registry is the inventory, and "
              "docs/test-hooks.md is rendered from it.")
        return 1
    print("switches: ok (%d rows, %d renames, %d reads)" % (len(rows), len(renames), len(reads)))
    return 0


def reg_re_fullmatch(values):
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_.<>+-]+(\|[A-Za-z0-9_.<>+-]+)*", values)
                or re.fullmatch(r"\d+\.\.\d+", values))


if __name__ == "__main__":
    sys.exit(main())
