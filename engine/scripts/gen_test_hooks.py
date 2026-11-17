#!/usr/bin/env python3
"""Render the switch registry into docs/test-hooks.md.

The document has one generated region, between the two marker comments:

    <!-- BEGIN generated: switches ... -->
    ...
    <!-- END generated: switches -->

Everything between them is produced by this script from engine/src/switches.cpp;
everything outside is prose a person owns (the rules, the deep dives with the
measured numbers, the environment-only variables).  So the inventory cannot
drift from the code — `--check` (called by docs_check.sh) fails when it does —
and the prose is free to be prose.

usage:
    engine/scripts/gen_test_hooks.py --write     rewrite the region
    engine/scripts/gen_test_hooks.py --check     exit 1 if it is stale
"""
import sys

import switch_registry as reg

DOC = reg.ROOT + "/docs/test-hooks.md"

BEGIN = "<!-- BEGIN generated: switches"
END = "<!-- END generated: switches"

HEAD = """<!-- BEGIN generated: switches -- `engine/scripts/gen_test_hooks.py --write`
     rewrites this block from `engine/src/switches.cpp`.  Do not edit it by
     hand: the registry is the inventory, and `--check` (step 0 of the gate)
     fails when this text and the registry disagree. -->

The rows below are the registry, one per switch, grouped by the axis they move.
`permanent` is part of the documented configuration surface; `probe` is a
diagnostic or a pricing arm; `retired` is gone (`--dump-switches` prints the
same table with what *this* environment resolved to).
"""


def render():
    rows, renames = reg.load()
    axes = reg.by_axis(rows)
    out = [HEAD]
    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    out.append("%d switches: %d permanent, %d probe, %d retired; %d legacy "
               "spellings are translated.\n" % (
                   len(rows), counts.get("permanent", 0), counts.get("probe", 0),
                   counts.get("retired", 0), len(renames)))
    for axis in reg.AXES:
        group = axes.get(axis) or []
        if not group:
            continue
        out.append("### %s\n" % reg.AXIS_TITLES.get(axis, axis))
        out.append("| Switch | Takes | What it moves |")
        out.append("|---|---|---|")
        for r in group:
            out.append(r.md())
        out.append("")
    if renames:
        out.append("### Legacy spellings\n")
        out.append("These still work: the registry rewrites them onto the axis "
                   "value and says so on stderr once.  New code and new notes "
                   "use the axis name.\n")
        out.append("| Old | Means |")
        out.append("|---|---|")
        by_legacy = {}
        for legacy, value, target, becomes in renames:
            cell = "`%s=%s`" % (target, becomes) if becomes else "`%s=<same>`" % target
            if value:
                cell = "`%s=%s`" % (target, becomes or value)
            by_legacy.setdefault(cell, []).append(
                "`%s`" % (legacy + ("=" + value if value else "")))
        for cell in sorted(by_legacy):
            out.append("| %s | %s |" % (", ".join(sorted(by_legacy[cell])), cell))
        out.append("")
    out.append("<!-- END generated: switches -->")
    return "\n".join(out)


def main():
    text = open(DOC, encoding="utf-8").read()
    i = text.find(BEGIN)
    j = text.find(END)
    if i < 0 or j < 0:
        print("docs/test-hooks.md: the generated-region markers are missing\n"
              "  expected '%s ...' and '%s ...'" % (BEGIN, END), file=sys.stderr)
        return 2
    j = text.index("-->", j) + 3
    body = render()
    new = text[:i] + body + text[j:]
    if "--check" in sys.argv:
        if new == text:
            print("test-hooks: ok (generated from the registry, %d switches)"
                  % len(reg.load()[0]))
            return 0
        print("test-hooks: FAIL -- docs/test-hooks.md's inventory is stale.\n"
              "  run engine/scripts/gen_test_hooks.py --write and commit the diff",
              file=sys.stderr)
        return 1
    if "--write" in sys.argv:
        with open(DOC, "w", encoding="utf-8") as f:
            f.write(new)
        print("test-hooks: wrote the registry block (%d switches)" % len(reg.load()[0]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
