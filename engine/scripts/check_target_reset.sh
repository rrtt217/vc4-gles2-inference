#!/bin/sh
# Every scratch Target in gpu.h must be cleared by GPU::reset_targets().
#
# release_scene() deletes the textures and FBOs, and the create sites in gpu.cpp
# are guarded with "if (t.tex == 0 || t.w != ...)".  A Target that still carries
# a deleted name is therefore *reused*, which is silent garbage rather than a
# visible error.  Adding a Target member without listing it in reset_targets()
# is a two-line mistake; this check costs milliseconds, needs no GPU or model,
# and does not sort - collation order would make the comparison locale-dependent.
set -u
cd "$(dirname "$0")/.." || exit 1

members=$(sed -n 's/^[[:space:]]*Target \([a-z_0-9_][a-z_0-9_, ]*\);$/\1/p' src/gpu.h |
          tr ',' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$')
reset=$(sed -n '/^void GPU::reset_targets/,/^    }$/p' src/gpu.cpp |
        sed -n 's/^[[:space:]]*\([a-z_0-9_]*\) = Target();$/\1/p')
containers=$(sed -n 's/^[[:space:]]*\(std::\)\{0,1\}vector<Target> \([a-z_0-9_]*\);.*$/\2/p' src/gpu.h
            sed -n 's/^[[:space:]]*[a-z_:]*map<[^>]*Target[^>]*> \([a-z_0-9_]*\);.*$/\1/p' src/gpu.h)
cleared=$(sed -n '/^void GPU::reset_targets/,/^    }$/p' src/gpu.cpp |
          sed -n 's/^[[:space:]]*\([a-z_0-9_]*\)\.clear();$/\1/p')

# membership test without sort/comm: collation differs between locales
has() { printf '%s\n' "$2" | grep -qxF "$1"; }

fail=0
for m in $members; do
    has "$m" "$reset" || { echo "FAIL: Target $m is not cleared by reset_targets()"; fail=1; }
done
for r in $reset; do
    has "$r" "$members" || { echo "FAIL: reset_targets() clears '$r', which is not a Target member"; fail=1; }
done
for c in $containers; do
    has "$c" "$cleared" || { echo "FAIL: Target container $c is not cleared by reset_targets()"; fail=1; }
done

if [ "$fail" = 0 ]; then
    echo "target reset: ok ($(printf '%s\n' "$members" | grep -c .) targets, $(printf '%s\n' "$containers" | grep -c .) containers)"
fi
exit "$fail"
