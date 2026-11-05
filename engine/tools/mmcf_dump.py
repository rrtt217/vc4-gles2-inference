#!/usr/bin/env python3
"""Print an MMCF v2 container in human-readable form.

    mmcf_dump.py model.mmcf            # summary
    mmcf_dump.py model.mmcf --tensors  # one line per tensor
    mmcf_dump.py model.mmcf --check    # also verify every tensor CRC

The point is to make "the container is self-describing" a checkable fact rather
than a claim: everything the engine reads is printed here from the file alone.
"""

import argparse
import struct
import sys
import zlib

KINDS = {
    1: "embedding", 2: "embedding_scale", 3: "rms_att", 4: "rms_ffn",
    5: "rms_final", 6: "q_norm", 7: "k_norm", 8: "embedding_i16",
    0x10: "qkv", 0x11: "wo", 0x12: "w1", 0x13: "w3", 0x14: "w13", 0x15: "w2",
    0x20: "logits_tile", 0x30: "tokenizer_blob",
}
DTYPES = {0: "f32", 1: "i16", 2: "q8u8", 3: "q8i8", 4: "q4", 5: "blob"}
SCHEMES = {0: "i16", 1: "q8", 2: "q4"}
TEMPLATE_KINDS = {0: "plain", 1: "chatml/qwen", 2: "minimind3", 3: "tinybrainbot"}
CAPS = [(1, "chat"), (2, "tools"), (4, "vision"), (8, "thinking")]


def rd(raw, off, fmt):
    return struct.unpack_from(fmt, raw, off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    raw = open(args.path, "rb").read()
    if raw[:4] != b"MMCF":
        raise SystemExit("not an MMCF container")
    ver, hsz, n, caps, mo, ms, doff, fsize = rd(raw, 4, "<HHIIIIQQ")
    hcrc, = rd(raw, 40, "<I")
    print(f"file        {args.path}  {len(raw)} B")
    print(f"version     {ver}   header {hsz} B   tensors {n}")
    print(f"sections    meta {mo}+{ms}   data {doff}   size field {fsize} "
          f"({'ok' if fsize == len(raw) else 'MISMATCH'})")
    if hcrc:
        ok = (zlib.crc32(raw[:40]) & 0xFFFFFFFF) == hcrc
        print(f"header crc  0x{hcrc:08x} {'ok' if ok else 'MISMATCH'}")
    else:
        print("header crc  absent")
    names = [n for b, n in CAPS if caps & b]
    print(f"caps        0x{caps:x} ({', '.join(names) if names else 'none'})")

    # metadata
    p = mo
    tok_bytes = 0
    while p + 8 <= mo + ms:
        key, typ, ln = rd(raw, p, "<HHI")
        payload = raw[p + 8:p + 8 + ln]
        if key == 1 and ln >= 28:
            dim, hidden, layers, heads, kv, vocab, seq = struct.unpack_from("<7i", payload)
            print(f"model       dim {dim} hidden {hidden} layers {layers} heads {heads} "
                  f"kv_heads {kv} vocab {vocab} seq {seq}")
        elif key == 2 and ln >= 8:
            theta, half, qk, _ = struct.unpack_from("<fBBH", payload)
            print(f"rope        theta {theta:g} half {bool(half)} qk_norm {bool(qk)}")
        elif key == 3 and ln >= 4:
            scheme, group, preb, split = struct.unpack_from("<BBBB", payload)
            print(f"quant       {SCHEMES.get(scheme, scheme)} group {group} "
                  f"pre_biased {bool(preb)} split_w13 {bool(split)}")
        elif key == 4 and ln >= 3:
            kind = payload[0]
            parts = payload[1:].split(b"\0")
            system = parts[0].decode("utf-8", "replace") if parts else ""
            tname = parts[1].decode("utf-8", "replace") if len(parts) > 1 else ""
            print(f"template    kind {kind} ({TEMPLATE_KINDS.get(kind, 'unknown')}) name '{tname}'")
            print(f"system      {system!r}")
        elif key == 5 and ln >= 4:
            pass  # mirror of the header caps, already printed
        elif key == 6:
            print(f"name        {payload.rstrip(bytes([0])).decode('utf-8', 'replace')}")
        elif key == 7 and ln >= 4:
            tok_bytes, = struct.unpack_from("<i", payload)
            print(f"tokenizer   {tok_bytes} B embedded")
        p += 8 + ln

    # tensors
    table = doff - n * 48
    rows = []
    for i in range(n):
        r = raw[table + i * 48: table + (i + 1) * 48]
        tid, dt, flags, group, _pad = struct.unpack_from("<HBBBB", r, 0)
        rws, cls = struct.unpack_from("<II", r, 6)
        off, blen = struct.unpack_from("<QQ", r, 14)
        crc, = struct.unpack_from("<I", r, 30)
        scale, = struct.unpack_from("<f", r, 34)
        rows.append((tid, dt, rws, cls, off, blen, crc, scale))
    total = sum(t[5] for t in rows)
    print(f"tensor data {total} B in {n} tensors, {len(raw) - total - table} B of headers/padding")
    if args.tensors:
        for tid, dt, rws, cls, off, blen, crc, scale in rows:
            kind = tid & 0xFF
            layer = (tid >> 8) & 0x3F
            part = tid >> 14
            label = KINDS.get(kind, f"0x{kind:02x}")
            if kind in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15):
                label += f".{layer}"
            elif kind == 0x20:
                label += f".{layer}"
            if part:
                label += f".part{part}"
            print(f"  {label:22s} {DTYPES.get(dt, dt):5s} {rws:5d} x {cls:5d} "
                  f"off {off:10d} len {blen:9d} scale {scale:.6g} crc 0x{crc:08x}")
    if args.check:
        bad = 0
        for tid, dt, rws, cls, off, blen, crc, scale in rows:
            body = raw[doff + off: doff + off + blen]
            if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
                bad += 1
                print(f"  CRC MISMATCH tensor id 0x{tid:04x}")
        print(f"crc check   {'all ' + str(n) + ' tensors ok' if not bad else str(bad) + ' bad'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
