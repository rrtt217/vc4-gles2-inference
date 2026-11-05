#!/usr/bin/env python3
"""Per-row stats for DUMP/HIDDEN_DUMP lines (q16 pair texels, hex).

prints: index name max|x| n_clamped mean|x| for each DUMP line, in file order.
"""
import sys


def q16(b0, b1):
    v = (b0 << 8) | b1
    return v - 65536 if v >= 32768 else v


i = 0
for line in open(sys.argv[1], errors="replace"):
    if "DUMP " not in line:
        continue
    f = line.split()
    k = f.index("DUMP")
    name = f[k + 1]
    hexs = f[k + 3:]
    vals = []
    for hx in hexs:
        if len(hx) != 8:
            continue
        b = bytes.fromhex(hx)
        vals.append(q16(b[0], b[1]))
        vals.append(q16(b[2], b[3]))
    if not vals:
        continue
    mx = max(abs(v) for v in vals)
    ncl = sum(1 for v in vals if abs(v) >= 32767)
    mean = sum(abs(v) for v in vals) / len(vals)
    print("%4d %-5s max=%6d clamp=%3d mean=%7.1f" % (i, name, mx, ncl, mean))
    i += 1
