#!/usr/bin/env python3
"""The Markdown files the gates scan, derived rather than listed.

Three checks used to carry their own list of documents: `check_docs_style.py`
had `STATE`/`HISTORY`, `check_docs_links.py` had `DOCS`, and `docs_check.sh`
had a shell variable.  The lists had already drifted -- the four
`engine/*/README.md` files and this directory's `AGENTS.md` were in none of
them, so a broken link or a stale command in those pages was unchecked.  A list
that must be updated by hand is a gate that can pass without looking.

The set is every Markdown file git would commit: tracked plus untracked, minus
ignored.  So a page is scanned the moment it is written, before it is staged.
The vendored Jinja pages are the one exclusion: they are upstream text, and
their links name upstream test files that are not in this tree.  Without git
the same walk runs over the tree and skips the ignored directories by name.

usage: engine/gate/docset.py [--all|--style|--history]
  --all      every scanned document (links, in-tree paths)
  --style    the state documents (see check_docs_style.py)
  --history  the frozen record pages, exempt from the style rules
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Vendored upstream documentation: its links point outside this repository.
EXCLUDE_PREFIX = ("engine/third_party/",)

# "Decisions and records" pages: dated by construction, not maintained as
# state.  docs/AGENTS.md says where a page belongs; this is the exemption list
# the style check needs, and it is deliberately short.
HISTORY = (
    "docs/benchmarks.md",
    "docs/swap-drift.md",
    "docs/vc4-1536-repro.md",
    "docs/translation-decision-model-fit.zh-CN.md",
)

WALK_SKIP_DIRS = {".git", ".bench", "refs", "node_modules", "__pycache__", ".venv"}
WALK_SKIP_FILES = {"AGENTS.local.md", "AGENTS.override.md"}


def _git_files():
    """(rel paths, or None when this is not a git work tree)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z", "*.md"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in done.stdout.decode("utf-8", "replace").split("\0") if p]


def _walk():
    out = []
    for p in ROOT.rglob("*.md"):
        rel = p.relative_to(ROOT).as_posix()
        parts = rel.split("/")
        # Hidden trees are caches (.uv-cache, .bench) and metadata (.git); none
        # of them is a document this repository owns.
        if any(d.startswith(".") or d in WALK_SKIP_DIRS for d in parts[:-1]):
            continue
        # models/ is a gitignored store with one tracked README; a stale copy of
        # the source tree lives under it and must not become a scanned document.
        if parts[0] == "models" and rel != "models/README.md":
            continue
        if rel in WALK_SKIP_FILES:
            continue
        out.append(rel)
    return out


def docs():
    """Every scanned document, as repository-relative paths, sorted."""
    rel = _git_files()
    if rel is None:
        rel = _walk()
    return sorted(r for r in rel if not r.startswith(EXCLUDE_PREFIX))


def paths():
    """`docs()` as absolute paths, for the checks that open them."""
    return [ROOT / r for r in docs()]


def style():
    """The state documents: everything except the frozen record pages."""
    return [r for r in docs() if r not in HISTORY]


def history():
    """The frozen record pages, in the order the exemption list states."""
    return list(HISTORY)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    if mode == "--all":
        rel = docs()
    elif mode == "--style":
        rel = style()
    elif mode == "--history":
        rel = history()
    else:
        print("usage: engine/gate/docset.py [--all|--style|--history]",
              file=sys.stderr)
        return 2
    for r in rel:
        print(ROOT / r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
