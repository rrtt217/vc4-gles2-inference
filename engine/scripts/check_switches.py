#!/usr/bin/env python3
"""The switch registry against the source and against docs/test-hooks.md.

Four directions, all of them mechanical:

  src -> registry   every switch engine/src reads has a row (found: 2026-11-17,
                    126 switches and no inventory outside a 484-line table)
  registry -> src   every non-retired row is read somewhere (a row nobody reads
                    is a lie that outlives the code it described)
  registry rules    a retired row must not be read, an alias must point at a
                    canonical row, a closed set must be spelled a|b|c
  registry -> docs  every row's name appears in docs/test-hooks.md

The registry is parsed from engine/src/switches.cpp rather than from
`--dump-switches`, so this runs before a build (docs_check.sh is step 0 of the
gate and must not need a compiler or a GPU).

usage: engine/scripts/check_switches.py [-v]
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
SRC = os.path.join(ENGINE, "src")
DOC = os.path.join(os.path.dirname(ENGINE), "docs", "test-hooks.md")

ROW = re.compile(
    r'\{"([A-Z][A-Z_0-9]+)",\s*"([^"]*)",\s*SwKind::(\w+),\s*"([^"]*)",\s*'
    r'SwStatus::(\w+),\s*SwAxis::(\w+),')
RENAME = re.compile(
    r'\{"([A-Z][A-Z_0-9]+)",\s*"([^"]*)",\s*"([A-Z][A-Z_0-9]+)",\s*"([^"]*)"\}')
# Both spellings read a switch: getenv() directly, the env_* helpers in
# common.h, and the registry accessors themselves (a switch read through
# sw_enum() is read just as much as one read through env_flag()).
READ = re.compile(r'(?:getenv|env_set|env_flag|env_int|env_double|sw_isset|sw_flag|'
                  r'sw_int|sw_num|sw_str|sw_enum)\(\s*"([A-Z_0-9]+)"')


def main():
    verbose = "-v" in sys.argv
    rows, renames = {}, []
    text = open(os.path.join(SRC, "switches.cpp"), encoding="utf-8").read()
    # The table is one array of {..} rows; the rename array's rows have four
    # string fields and no SwKind, so the two regexes do not overlap.
    for m in ROW.finditer(text):
        name, aliases, kind, values, status, axis = m.groups()
        rows[name] = dict(aliases=[a for a in aliases.split(",") if a], kind=kind,
                          values=values, status=status, axis=axis)
    for m in RENAME.finditer(text):
        renames.append(m.groups())

    reads = {}
    for dirpath, _, files in os.walk(SRC):
        for f in sorted(files):
            if not f.endswith((".cpp", ".h")):
                continue
            p = os.path.join(dirpath, f)
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                for m in READ.finditer(line):
                    reads.setdefault(m.group(1), []).append(
                        "%s:%d" % (os.path.relpath(p, ENGINE), i))

    doc = open(DOC, encoding="utf-8").read()
    fail = 0

    def bad(msg):
        nonlocal fail
        fail = 1
        print("FAIL  " + msg)

    # --- the rules inside the registry ---------------------------------------
    for name, r in rows.items():
        if r["values"] and not re.fullmatch(r"[A-Za-z0-9_.<>+-]+(\|[A-Za-z0-9_.<>+-]+)*",
                                            r["values"]):
            bad("registry row %s: values '%s' is not a|b|c" % (name, r["values"]))
        if r["status"] == "Retired" and reads.get(name):
            bad("registry says %s is retired, src still reads it (%s)"
                % (name, reads[name][0]))
        if name in reads and r["status"] == "Retired":
            pass
        if not reads.get(name) and r["status"] != "Retired":
            bad("registry row %s is read nowhere in src (a stale row)" % name)
        if name not in doc:
            bad("registry row %s is not mentioned in docs/test-hooks.md" % name)
    for legacy, value, target, becomes in renames:
        if target not in rows:
            bad("rename %s -> %s, which is not a registry row" % (legacy, target))
        if rows.get(target, {}).get("status") == "Retired":
            bad("rename %s -> %s, which is retired" % (legacy, target))
        if not becomes and value and target in rows:
            if rows[target]["values"] and value not in rows[target]["values"].split("|"):
                bad("rename %s=%s keeps a value the target axis does not accept (%s)"
                    % (legacy, value, rows[target]["values"]))

    # --- src -> registry -----------------------------------------------------
    known = set(rows) | {l for l, _, _, _ in renames}
    for name, where in sorted(reads.items()):
        if name in known:
            continue
        bad("src reads %s (%s), the registry has no row for it"
            % (name, where[0]))

    # --- the aliases a row declares ------------------------------------------
    for name, r in rows.items():
        for alias in r["aliases"]:
            if alias in rows:
                bad("registry row %s declares alias %s, which is itself a row"
                    % (name, alias))
            if alias not in doc:
                pass  # an alias is documented next to its row, not as a row

    if verbose:
        print("  %d registry rows, %d renames, %d switches read by src"
              % (len(rows), len(renames), len(reads)))
    if fail:
        print("\nsee engine/src/switches.cpp: the registry is the inventory, and "
              "docs/test-hooks.md is its prose.")
        return 1
    print("switches: ok (%d rows, %d renames, %d reads)" % (len(rows), len(renames), len(reads)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
