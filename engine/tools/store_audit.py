#!/usr/bin/env python3
"""What the model store holds, which containers the tree names, and what has no reference.

The store is a workbench, not an archive: `models/README.md` owns the rule for
what is kept and when a source or a container is deleted.  This tool is the
inventory that rule asks for -- it never deletes anything.

For every `*.mmcf` it prints the container version, its size, and whether a
tracked file names it (the file name, the stem, or at least the model).  A
container with no file-name reference is a *candidate* for the review the
policy describes, not a verdict: `stories42M.i16` is named as "stories42M i16"
in `engine/README.md`, and a brace row like `tinybrainbot.{i16,q4}.mmcf` names
two files without naming either.

For every large source payload (`*.safetensors`, `*.pth`, `*.gguf`, a llama2.c
`*.bin`) and every derived checkpoint (`*.i16`, `*.q8`, `*.q4`), it prints the
size and the containers in the same directory.

`--check` also verifies every container's CRCs with `mmcf_dump.py --check`.
It needs no GPU, no model and no numpy.

usage: engine/tools/store_audit.py [--check] [--json]
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "models"
MMCF = "engine/tools/mmcf_dump.py"
MAGIC = b"MMCF"
SOURCE_EXT = {".safetensors", ".pth", ".gguf"}
DERIVED_EXT = {".i16", ".q8", ".q4"}
BIN_MIN = 10 * 2**20          # a llama2.c weight; tokenizer .bin files are small


def containers():
    out = []
    for p in STORE.rglob("*.mmcf"):
        with open(p, "rb") as f:
            h = f.read(40)
        magic, version, _, n_tensors, _, _, _, _, size = struct.unpack("<4sHHIIIIQQ", h)
        out.append({
            "path": p,
            "rel": p.relative_to(ROOT).as_posix(),
            "model": p.relative_to(STORE).parts[0],
            "bytes": p.stat().st_size,
            "version": version,
            "tensors": n_tensors,
            "size_ok": size == p.stat().st_size,
            "magic_ok": magic == MAGIC,
        })
    return sorted(out, key=lambda r: r["rel"])


def sources():
    out = []
    for p in STORE.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in SOURCE_EXT or (ext == ".bin" and p.stat().st_size >= BIN_MIN):
            out.append({
                "path": p,
                "rel": p.relative_to(ROOT).as_posix(),
                "model": p.relative_to(STORE).parts[0],
                "bytes": p.stat().st_size,
                "ext": ext,
            })
    return sorted(out, key=lambda r: r["rel"])


def derived():
    out = []
    for p in STORE.rglob("*"):
        if p.is_file() and p.suffix.lower() in DERIVED_EXT:
            out.append({
                "path": p,
                "rel": p.relative_to(ROOT).as_posix(),
                "model": p.relative_to(STORE).parts[0],
                "bytes": p.stat().st_size,
                "ext": p.suffix.lower(),
            })
    return sorted(out, key=lambda r: r["rel"])


def refs(needle):
    """Tracked files that name `needle` (a literal string)."""
    done = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-F", "-l", "--", needle],
        capture_output=True, text=True,
    )
    out = [x for x in done.stdout.splitlines() if x]
    local = ROOT / "AGENTS.local.md"
    if local.exists() and needle in local.read_text(encoding="utf-8", errors="replace"):
        out.append("AGENTS.local.md")
    return out


def verify(path):
    done = subprocess.run(
        [sys.executable, str(ROOT / MMCF), "--check", str(path)],
        capture_output=True, text=True,
    )
    return done.returncode == 0, (done.stdout + done.stderr).strip().splitlines()[-1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify every container's CRCs")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cs = containers()
    ss = sources()
    ds = derived()
    rows = []
    for c in cs:
        exact = refs(c["path"].name)
        stem = refs(c["path"].stem)
        model = refs(c["model"])
        rec = dict(c, exact=len(exact), stem=len(stem), model=len(model),
                   exact_files=exact, path=c["rel"])
        if args.check:
            ok, tail = verify(c["path"])
            rec["crc_ok"] = ok
            rec["crc_tail"] = tail
        rows.append(rec)

    if args.json:
        print(json.dumps({"containers": rows, "sources": ss, "derived": ds}, indent=1))
        return 0

    print(f"containers: {len(cs)}  ({sum(c['bytes'] for c in cs) / 2**20:.1f} MB)")
    print(f"{'MB':>8} {'ver':>3} {'file':>4} {'stem':>4} {'model':>5}  container")
    for r in rows:
        flag = "" if r["magic_ok"] and r["size_ok"] else "  BAD HEADER"
        print(f"{r['bytes']/2**20:8.1f} {r['version']:3} {r['exact']:4} {r['stem']:4} "
              f"{r['model']:5}  {r['rel']}{flag}")
        if args.check:
            print(f"         crc {'ok' if r.get('crc_ok') else 'FAIL'}: {r['rel']}")

    print(f"\nsources: {len(ss)}  ({sum(s['bytes'] for s in ss) / 2**20:.1f} MB)")
    for s in ss:
        print(f"{s['bytes']/2**20:8.1f}  {s['rel']}")

    print(f"\nderived checkpoints: {len(ds)}  ({sum(d['bytes'] for d in ds) / 2**20:.1f} MB)")
    for d in ds:
        print(f"{d['bytes']/2**20:8.1f}  {d['rel']}")

    missing = [r for r in rows if not (r["magic_ok"] and r["size_ok"])]
    if args.check:
        bad = [r for r in rows if not r.get("crc_ok")]
        if missing or bad:
            print(f"\nFAIL  {len(missing)} header(s), {len(bad)} CRC(s)", file=sys.stderr)
            return 1
        print("\nall containers: header and CRCs ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
