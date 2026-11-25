#!/bin/sh
# Documentation drift check.
#
# The rules in AGENTS.md say a command in a document that no longer runs is a
# bug in the document.  This looks for the mechanical cases, which are the ones
# that actually happened:
#
#   1. an environment switch the source reads but docs/test-hooks.md never
#      mentions (found: FFN_PAIR, TOK_NO_PREFIX, TOK_SPACE_FOLD);
#   2. a file a document tells you to run, which no longer exists (found:
#      minimind3/make_tokenizer.py after it was folded into tools/make_tok.py);
#   3. a switch a document calls "gone" while the source still reads it
#      (found: TOKENIZER=, which is worse than gone - it *overrides* the
#      container's tokenizer and the answer comes out empty);
#   4. a removed engine switch still used as a command;
#   5. the three directions that need more than grep (links and their heading
#      anchors, the docs/README.md index against git, and doc->source: every
#      switch test-hooks.md documents must be read by engine/src).  Those live in
#      gate/check_docs_links.py, which is where CONV_GPU would have been caught.
#
# Usage:  gate/docs_check.sh            (from engine/, or anywhere)
#         gate/docs_check.sh -v         also list what was checked
#
# Exit status is 1 if anything drifted.  Run it before committing a document
# change, and after adding or removing an environment switch.
set -u
cd "$(dirname "$0")/.." || exit 1          # engine/
ROOT=$(cd .. && pwd)
VERBOSE=${1:-}
fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; fail=1; }

# The document set is derived, not listed here: docset.py enumerates every
# Markdown file git would commit (tracked + untracked, minus the vendored
# pages), so a new page is scanned the moment it exists.  A hand list in this
# script is how the four engine/*/README.md files stayed outside every check.
DOCS=$(python3 gate/docset.py --all) || exit 1

# 1. the switch registry against the source and against test-hooks.md, and the
# document's generated inventory against the registry (the inventory is
# rendered, so "the doc is stale" is a mechanical verdict, not a review)
if ! python3 gate/gen_test_hooks.py --check; then fail=1; fi

# Every switch the engine reads has a row in src/switches.cpp (the inventory),
# the row has to be read somewhere, a retired row must not be read, and the row
# has to be documented.  This replaced a grep loop in 2026-11-17: the grep could
# only ask "does the doc mention this name", never "is this name a whole switch".
if ! python3 gate/check_switches.py $VERBOSE; then fail=1; fi
# The codec layouts live in xformat; a shader that types one is drift.
if ! python3 gate/check_codecs.py $VERBOSE; then fail=1; fi
n=$(grep -cE '^ *\{"[A-Z][A-Z_0-9]+",.*SwKind::' src/switches.cpp)

# 2. files a document tells you to run must exist
# Both spellings a document uses: `engine/gate/x.sh` from the repository root,
# and `gate/x.sh` from engine/ (which is where the commands are run from).  A
# bare prefix is resolved against engine/ as a second try, so a doc that says
# `scripts/bench.sh` is checked as well as one that says `engine/scripts/...`.
# Extensions only: a *runtime output* a document names (server/stats.jsonl) is
# not an in-tree path and must not be checked.
# Bare prefixes are limited to the three directories nothing else in this tree
# could mean: `tools/`, `src/` and `scripts/` also name host-side helpers and
# upstream llama.cpp files in these documents, and a path check that fires on a
# legitimate mention is a check people learn to ignore.
refs=$(grep -rhoE "(engine/(tools|scripts|probes|src|gate|eval|tests)|engine|minimind3|server|templates|gate|eval|tests)/[A-Za-z0-9_.-]+\.(py|sh|jinja|cpp|h|awk|txt)" \
        $DOCS 2>/dev/null | sort -u)
[ "$VERBOSE" = "-v" ] && say "  $(printf '%s\n' "$refs" | wc -l) in-tree paths referenced"
for f in $refs; do
    if [ ! -e "$ROOT/$f" ] && [ ! -e "$ROOT/engine/$f" ]; then
        bad "a document references $f, which does not exist"
    fi
done

# 3. a switch called "gone" that the source still reads
for v in TOKENIZER ROPE_THETA ROPE_HALF LEGACY_Q8; do
    if grep -rq "getenv(\"$v\")" src; then
        if grep -n "$v" $DOCS 2>/dev/null | grep -qiE "gone|removed|no longer|ignored"; then
            bad "$v is called gone/ignored in a document but src still reads it"
        fi
    fi
done

# 4. engine switches that no longer exist, still used as a command
for flag in --template; do
    # only an *engine* command; the packer legitimately has --template
    hits=$(grep -rn -- "$flag" $DOCS 2>/dev/null | grep "engine_mm3" | \
           grep -viE "gone|removed|no longer|there is no")
    if [ -n "$hits" ]; then
        say "$hits" | head -3 | while read -r l; do bad "engine $flag used as a command: $l"; done
        fail=1
    fi
done

# 4b. the style rules: canonical spellings in commands, dated measurements in a
# state document (see the header of gate/check_docs_style.py)
if ! python3 gate/check_docs_style.py $VERBOSE; then fail=1; fi

# 4c. the script directories against their READMEs: a file with no row, or a row
# with no file (the four directories are split by what each file *needs*)
if ! python3 gate/check_scripts.py $VERBOSE; then fail=1; fi

# 5. links, the index and the doc->source direction
if ! python3 gate/check_docs_links.py; then fail=1; fi

if [ "$fail" = 0 ]; then
    say "docs: ok ($n switches, $(printf '%s\n' "$refs" | wc -l) paths)"
else
    say ""
    say "see AGENTS.md, Documentation rules: update the document that owns the fact."
fi
exit "$fail"
