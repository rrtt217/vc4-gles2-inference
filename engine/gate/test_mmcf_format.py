#!/usr/bin/env python3
"""The MMCF layout as the Python tools see it: round trips, and agreement with C++.

`engine/tools/mmcf_format.py` is now the only copy of the layout in the toolchain
(before 2026-11-14 three tools each carried their own magic, record size, `struct`
formats, TLV helper and dtype table -- and the dtype table had already drifted,
which is how P1-3 found q6 missing from one copy).  This proves the module two
ways:

  1. **round trip**: for every container under `models/`, the 64-byte header and
     the whole tensor-record table are parsed and rebuilt byte for byte.  A field
     the tools disagree about (dtype, the v2 reserved bytes, v3's `aux`) shows up
     here instead of as "the dump tool prints something odd for one model".
  2. **cross-language**: the magic, the header size and both record sizes match
     what `engine/src/model.cpp` reads.  Nothing else checks that the packer and
     the engine agree on the layout, and a packer that writes 64-byte records
     where the engine reads 48 produces a container nobody can load.

Reads headers and record tables only -- no weights, no GPU, no model.
"""

import os
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "engine/tools"))

import mmcf_format as fmt  # noqa: E402


def containers():
    """Every container in the store; `MMCF_MODELS` overrides it (the board's gate
    tree has no models/, the runtime has them)."""
    base = os.environ.get("MMCF_MODELS")
    root = Path(base).expanduser().resolve() if base else (ROOT / "models")
    return sorted(root.rglob("*.mmcf"))


def cpp_constants():
    """The layout as `engine/src/model.cpp` reads it (names checked 2026-11-14)."""
    src = (ROOT / "engine/src/model.cpp").read_text(encoding="utf-8", errors="replace")
    out = {}
    for name in ("MMCF_REC_V2", "MMCF_REC_V3"):
        m = re.search(r"\b%s\s*=\s*(\d+)" % name, src)
        out[name] = int(m.group(1)) if m else None
    # `if (magic != 0x46434D4DU) ... // "MMCF"`
    m = re.search(r"!=\s*(0x[0-9A-Fa-f]{8})U?[^\n]*\"MMCF\"", src)
    out["magic"] = int(m.group(1), 16) if m else None
    # `if (rd16(h + 6) != 64)` -- the header-size field the reader insists on
    m = re.search(r"rd16\(h \+ 6\) != (\d+)", src)
    out["header_size"] = int(m.group(1)) if m else None
    return out


def check_container(path):
    """(ok, message) for one container."""
    rel = path.relative_to(ROOT) if ROOT in path.parents else path.name
    with path.open("rb") as f:
        head = f.read(fmt.HEADER_SIZE)
        try:
            h = fmt.parse_header(head)
        except ValueError as e:
            return False, "%s: %s" % (rel, e)
        # Only the header and the record table are read: a 200 MB MoE container
        # must not be slurped to check 64-byte records.
        f.seek(h.data_off - h.n_tensors * h.rec_size)
        table = f.read(h.n_tensors * h.rec_size)
    if h.pack() != head[:h.header_size]:
        return False, "%s: the header does not survive a parse/rebuild" % rel
    broken = 0
    for i in range(h.n_tensors):
        rec = table[i * h.rec_size:][:h.rec_size]
        if len(rec) < h.rec_size or fmt.pack_rec(*fmt.rec_unpack(rec), size=h.rec_size) != rec:
            broken += 1
    if broken:
        return False, "%s: %d of %d records do not survive a parse/rebuild" % (
            rel, broken, h.n_tensors)
    return True, "ok    %-52s v%d  %3d records" % (rel, h.version, h.n_tensors)


def check_alias(path):
    """(ok, message) for a container whose records share one blob.

    Aliasing is legal (`off` is per record and the reader only bounds-checks it)
    and a block with an FFN shared by layer pairs depends on it, but nothing else
    in the gate would notice a file that aliases *wrongly*: every record still
    carries the right id, shape and scale, and each CRC still matches its own
    bytes.  Two invariants are checkable from the table alone:

    - every offset is 16-byte aligned (the same rule a non-aliased record keeps),
      so a shared blob lands where the reader's copy path expects it;
    - two records with the same offset and length decode to the same kind of
      body, which catches an embedding aliased onto a matrix (both are bytes, so
      a CRC check cannot tell).
    """
    with path.open("rb") as f:
        head = f.read(fmt.HEADER_SIZE)
        h = fmt.parse_header(head)
        f.seek(h.data_off - h.n_tensors * h.rec_size)
        table = f.read(h.n_tensors * h.rec_size)
    by_off = {}
    for i in range(h.n_tensors):
        tid, dt, _flags, _group, _rows, _cols, off, blen, _crc, _scale, _aux = \
            fmt.rec_unpack(table[i * h.rec_size:][:h.rec_size])
        if off % 16:
            return False, "%s: record %d's offset %d is not 16-byte aligned" % (
                path.name, i, off)
        by_off.setdefault((off, blen), []).append((tid, dt))
    shared = {k: v for k, v in by_off.items() if len(v) > 1}
    for (_off, _blen), recs in shared.items():
        kinds = {tid & 0xFF for tid, _dt in recs}
        dts = {dt for _tid, dt in recs}
        if len(dts) > 1:
            return False, "%s: one blob is read as dtypes %s" % (path.name, sorted(dts))
        # A per-layer kind may only share with the same per-layer kind (a shared
        # FFN group at w1[l0]/w1[l1] is the intended case); anything else means
        # two different tensors were collapsed into one.
        per_layer = {k for k in kinds if k in fmt.PER_LAYER_KINDS}
        if per_layer and kinds - per_layer:
            return False, "%s: a per-layer matrix shares its blob with kind(s) %s" % (
                path.name, sorted(kinds - per_layer))
    return True, "  alias %-52s %d shared blob(s)" % (path.name, len(shared))


def check_cpp():
    """(ok, messages) comparing the module's constants with model.cpp's."""
    cpp = cpp_constants()
    missing = [k for k, v in cpp.items() if v is None]
    if missing:
        return False, ["could not read %s out of engine/src/model.cpp" % ", ".join(missing)]
    checks = (("the magic", cpp["magic"], struct.unpack("<I", fmt.MAGIC)[0]),
              ("MMCF_REC_V2", cpp["MMCF_REC_V2"], fmt.REC_SIZE_V2),
              ("MMCF_REC_V3", cpp["MMCF_REC_V3"], fmt.REC_SIZE_V3),
              ("the header size", cpp["header_size"], fmt.HEADER_SIZE))
    bad = ["model.cpp's %s is %s, mmcf_format.py's is %s" % (n, got, want)
           for n, got, want in checks if got != want]
    return (not bad), bad


def main() -> int:
    bad = 0
    files = containers()
    if not files:
        print("SKIP  no containers (set MMCF_MODELS to a store; the C++ constants are "
              "still checked below)")
    for path in files:
        ok, msg = check_container(path)
        print(("ok    " if ok else "FAIL  ") + msg if not ok else "  " + msg)
        bad += 0 if ok else 1
        if ok:
            ok, msg = check_alias(path)
            if not ok:
                print("FAIL  " + msg)
                bad += 1

    ok, msgs = check_cpp()
    for m in msgs:
        print("FAIL  " + m)
    bad += 0 if ok else 1

    if bad:
        return 1
    print("mmcf format: ok (%d containers round-tripped, C++ constants agree)"
          % len(files) if files else "mmcf format: ok (no store here; C++ constants agree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
