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
#      scripts/check_docs_links.py, which is where CONV_GPU would have been caught.
#
# Usage:  scripts/docs_check.sh            (from engine/, or anywhere)
#         scripts/docs_check.sh -v         also list what was checked
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

DOCS="$ROOT/AGENTS.md $ROOT/README.md $ROOT/README.zh-CN.md \
$ROOT/engine/README.md $ROOT/engine/README.zh-CN.md \
$ROOT/server/README.md $ROOT/server/README.zh-CN.md \
$ROOT/models/README.md $ROOT/engine/probes/README.md $(ls "$ROOT"/docs/*.md 2>/dev/null)"

# 1. every switch the engine reads is described in test-hooks.md
# Both spellings read an environment switch: `getenv("X")` directly, and the
# `env_*` helpers in common.h (2026-11-14 moved 21 double reads onto them).  Keying
# the inventory on `getenv` alone would have made every converted switch invisible
# here -- the check has to follow the *fact*, not the syntax it was written with.
grep -rhoE '(getenv|env_set|env_flag|env_int|env_double)\(\s*"[A-Z_0-9]+"' src \
    | sed -E 's/.*"([A-Z_0-9]+)"/\1/' | sort -u > /tmp/docs_check_env.$$
n=$(wc -l < /tmp/docs_check_env.$$)
[ "$VERBOSE" = "-v" ] && say "  $n environment switches in src/"
while read -r v; do
    grep -q "$v" "$ROOT/docs/test-hooks.md" || bad "src reads $v, docs/test-hooks.md does not mention it"
done < /tmp/docs_check_env.$$
rm -f /tmp/docs_check_env.$$

# 2. files a document tells you to run must exist
refs=$(grep -rhoE "(engine/(tools|scripts|probes|src)|engine|minimind3|server|templates)/[A-Za-z0-9_.-]+\.(py|sh|jinja|cpp|h)" \
        $DOCS 2>/dev/null | sort -u)
[ "$VERBOSE" = "-v" ] && say "  $(printf '%s\n' "$refs" | wc -l) in-tree paths referenced"
for f in $refs; do
    [ -e "$ROOT/$f" ] || bad "a document references $f, which does not exist"
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

# 5. links, the index and the doc->source direction
if ! python3 scripts/check_docs_links.py; then fail=1; fi

if [ "$fail" = 0 ]; then
    say "docs: ok ($n switches, $(printf '%s\n' "$refs" | wc -l) paths)"
else
    say ""
    say "see AGENTS.md, Documentation rules: update the document that owns the fact."
fi
exit "$fail"
