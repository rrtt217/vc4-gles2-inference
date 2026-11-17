#!/bin/sh
# One name for the draw limit, and no bare literal for it.
#
# VC4 processes one fragment per cycle and the driver's scratch target is 2048
# wide, so a draw covers at most kDrawCap (1024) fragments; wider strips are split
# by the loops in `draw()` and `matvec_add()`.  Until 2026-11-14 that 1024 was a
# bare literal in seven places (five loops plus two environment defaults), with no
# name and no check -- and a wrong width fails *silently*: the read-back returns
# the previous contents of the scratch target, which looks like a plausible
# number, so nothing downstream can notice.
#
# The rule is `kDrawCap` in gpu_internal.h and not a literal, in the files that
# split draws -- `src/gpu*.cpp`, which is where `draw()` and `matvec_add()` live.
# The scope is deliberate and it is stated in the summary line: a bare 1024
# elsewhere in src/ is legitimate (SEQ's 1024-token default, a sampling range, a
# shader-builder argument, a doc string -- four of them today), and an
# allowlist for those would be worse than a scoped rule.  What the scope costs:
# a new strip loop in a file that is not named gpu* is not checked.  Comments are
# ignored: several legitimately quote 1024 to explain the limit.
#
# Usage: scripts/check_draw_cap.sh        (no GPU, no model)
set -u
cd "$(dirname "$0")/.." || exit 1

bad=0
for f in src/gpu*.cpp; do
    # drop // comments, then look for 1024
    # A bare 1024 that is a *scaling unit* is legitimate and common (`/ 1024` for
    # kB, `* 1024` for kB->B); the draw cap is an additive bound (`+= 1024`,
    # `min(1024,`, a default).  Flag the additive forms only (2026-11-15: the
    # upload accounting's `/ 1024` tripped the first version).
    hits=$(sed 's|//.*||' "$f" | grep -nE '(^|[^0-9_.])1024([^0-9]|$)' \
           | grep -vE '([*/]|>>)[[:space:]]*1024' | head -3)
    if [ -n "$hits" ]; then
        echo "FAIL  $f uses a bare 1024; use kDrawCap (src/gpu_internal.h):" >&2
        printf '%s\n' "$hits" | sed 's/^/      /' >&2
        bad=1
    fi
done
grep -q 'constexpr int kDrawCap = 1024;' src/gpu_internal.h || {
    echo "FAIL  kDrawCap is missing from src/gpu_internal.h" >&2
    exit 1
}
[ "$bad" = 0 ] || exit 1
echo "draw cap: ok (kDrawCap = 1024, no bare literal in the src/gpu*.cpp draw loops)"
