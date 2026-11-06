#!/usr/bin/env python3
"""Compare two LOGITS_OUT dumps (float32, vocabulary order).

Per-tile max deviation tells whether a difference is uniform rounding noise or
confined to one logits tile (the VC4 logits pass draws 1536-row tiles and the
last tile of a 32000-row vocabulary has only 1280 rows).

Usage: cmp_logits.py a.bin b.bin
"""
import struct
import sys


def load(path):
    b = open(path, "rb").read()
    return list(struct.unpack("<%df" % (len(b) // 4), b))


def top(v, k):
    idx = sorted(range(len(v)), key=lambda i: -v[i])[:k]
    return [(i, round(v[i], 2)) for i in idx]


a = load(sys.argv[1])
b = load(sys.argv[2])
n = min(len(a), len(b))
if len(a) != len(b):
    print("LENGTH MISMATCH %d %d" % (len(a), len(b)))
d = [abs(a[i] - b[i]) for i in range(n)]
print("n=%d max|d|=%.4f mean|d|=%.5f over_1LSB=%d" % (
    n, max(d), sum(d) / n, sum(1 for x in d if x > 1.0 / 256.0)))
print("top a:", top(a, 5))
print("top b:", top(b, 5))
tile = 1536
i = 0
while i < n:
    j = min(i + tile, n)
    m = max(d[i:j])
    print("tile %2d rows %5d-%5d max|d|=%.4f" % (i // tile, i, j - 1, m))
    i = j
