#!/usr/bin/env python3
"""The documentation checks that need more than grep (P1-4 of the 2026-11-13 audit).

`docs_check.sh` owns the src->doc direction: every switch the source reads is in
`docs/test-hooks.md`, every in-tree path a document names exists, no document
calls a live switch gone.  This file is the other three directions, because they
need slug matching, relative resolution or git:

  1. **links**: every relative `[text](path#frag)` a document writes resolves, and
     the `#frag` names a real heading in the target.  Today there are no dead
     links, which is exactly when it is cheap to lock that in.
  2. **index**: every file `docs/README.md` lists exists, and -- when this is a git
     checkout -- is tracked.  That is the check that would have caught the audit's
     P0-3 (an index entry pointing at an untracked file).
  3. **doc->source**: every switch `docs/test-hooks.md` *documents* is read by
     `engine/src`.  This is the direction that would have caught `CONV_GPU`, which
     six documents taught and no code ever read.

Runs in milliseconds and needs no GPU, no model and no network.  Exit 1 on drift.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOCS = [ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "README.zh-CN.md",
        ROOT / "engine/README.md", ROOT / "engine/README.zh-CN.md",
        ROOT / "server/README.md", ROOT / "server/README.zh-CN.md",
        ROOT / "models/README.md", ROOT / "engine/probes/README.md"] + \
       sorted((ROOT / "docs").glob("*.md"))

SKIP_SCHEMES = ("http://", "https://", "mailto:", "ftp://")

# Names that look like a switch but are not read by the engine: a Makefile
# variable, a gate variable, a shell variable in an example.  Every entry is a
# promise that it really is not an engine switch, so keep it short and true.
NOT_ENGINE_SWITCHES = {
    "CROSS_SYSROOT", "CROSS_CXX", "CXXFLAGS", "LDFLAGS", "ROOT", "REPO",
    "Q8_EXPECT", "Q4_EXPECT", "PPL_EXPECT", "Q4_PPL_EXPECT", "MOE_EXPECT",
    "MOE_ROUTES", "MOE_PPL_EXPECT", "PPL_TOL", "PPL_FILE", "PPL_SEQ", "NO_PPL",
    "SYS", "Q8", "Q4", "MOE", "MOE_PPL", "BIN", "DEBUG", "BASH", "HOME",
}


def fail(msg):
    print("FAIL  " + msg)
    return 1


def headings(path):
    """Heading slugs the way GitHub derives them, near enough for our own docs."""
    out = set()
    inside = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if inside:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        text = m.group(2).strip()
        text = re.sub(r"\s*\{#[-a-z0-9_]+\}\s*$", "", text)      # kramdown ids
        text = re.sub(r"`|\*|_|\[|\]\([^)]*\)", "", text)
        text = text.strip().lower()
        text = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", text)
        out.add(text.replace(" ", "-"))
    return out


LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def check_links():
    bad = 0
    for doc in DOCS:
        if not doc.exists():
            continue
        for target in LINK.findall(doc.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(SKIP_SCHEMES) or target.startswith("#"):
                continue
            path_part, _, frag = target.partition("#")
            if not path_part:
                continue
            dest = (doc.parent / path_part).resolve()
            if not dest.exists():
                bad += fail("%s links to %s, which does not exist"
                            % (doc.relative_to(ROOT), target))
                continue
            if frag and dest.suffix == ".md" and frag not in headings(dest):
                bad += fail("%s links to %s, but %s has no heading #%s"
                            % (doc.relative_to(ROOT), target,
                               dest.relative_to(ROOT), frag))
    return bad


def check_index():
    index = ROOT / "docs/README.md"
    if not index.exists():
        return 0
    bad = 0
    listed = set()
    for target in LINK.findall(index.read_text(encoding="utf-8")):
        if target.startswith(SKIP_SCHEMES) or "#" in target or not target.endswith(".md"):
            continue
        listed.add(target)
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-dir"],
                         capture_output=True, text=True)
    have_git = git.returncode == 0
    for target in sorted(listed):
        path = (index.parent / target).resolve()
        if not path.exists():
            bad += fail("docs/README.md links to %s, which does not exist" % target)
            continue
        if have_git and ROOT in path.parents:
            rel = str(path.relative_to(ROOT))
            tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                                     capture_output=True, text=True)
            if tracked.returncode != 0:
                bad += fail("docs/README.md links to %s, which is not tracked by git" % target)
    return bad


# A row whose first cell is `NAME=...`: that is how test-hooks.md lists a switch.
TEST_HOOK = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]{2,})=", re.M)

# Environment variables read by something that is not this engine (the driver, the
# linker, Mesa).  They are legitimately documented here.
EXTERNAL_SWITCH = re.compile(r"^(MESA_|EGL_|LIBGL_|LD_|VK_|GALLIUM_|AMD_|INTEL_)")


def check_doc_to_source():
    hooks = ROOT / "docs/test-hooks.md"
    if not hooks.exists():
        return 0
    src = " ".join(p.read_text(encoding="utf-8", errors="replace")
                   for p in (ROOT / "engine/src").glob("*.cpp"))
    read = set(re.findall(r'getenv\("([A-Z_0-9]+)"\)', src))
    bad = 0
    for name in sorted(set(TEST_HOOK.findall(hooks.read_text(encoding="utf-8")))):
        if name in NOT_ENGINE_SWITCHES or name in read or EXTERNAL_SWITCH.match(name):
            continue
        bad += fail("docs/test-hooks.md documents %s, but no engine/src file reads it" % name)
    return bad


def main() -> int:
    bad = 0
    bad += check_links()
    bad += check_index()
    bad += check_doc_to_source()
    if bad:
        return 1
    print("docs links: ok (%d documents, index and the doc->source direction)"
          % len([d for d in DOCS if d.exists()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
