#!/usr/bin/env python3
"""Compare two ATT_DUMP logs element by element, layer by layer.

`check_att.py` answers "is this machine's attention block consistent with its own
specification" (integers in, integers out, one machine).  This answers the other
question: **where do two machines stop agreeing**, at the finest resolution the
engine can print -- the q16/q24 integers each pass consumed and produced, for one
position and every layer.

    ATT_SUM=log2 ATT_DUMP=12 ./engine_mm3 ... 2>xe.log
    ATT_SUM=log2 ATT_DUMP=12 ./engine_mm3_pi ... 2>pi.log   # same binary source
    python3 scripts/cmp_att.py xe.log pi.log

ATT_SUM=log2 is the configuration `check_att.py` oracles; for this comparison any
fixed configuration works, and log2 has the advantage that both logs are also
individually checkable.

Columns: max |difference| and the number of differing elements, per pass.  qrot is
post-rope q (grid 1/256), kv is post-rope K (grid 1/kv_scale), score is the
1/1024 grid, attout the q16 activation grid (1/256).  A run where layer 0 already
differs means no single pass is at fault -- the inputs to the block do.

Why it exists: on 2026-11-17 a per-stage cosine dump said layer 0 was identical to
five decimals and the loss appeared around layer 7; the integers say qrot already
differs by 0.29 at layer 0.  A cosine on a 2300-magnitude residual hides that,
which is the whole reason to compare integers.
"""
import sys
def load(path):
    d = {}
    for line in open(path, errors='replace'):
        p = line.split()
        if not p or p[0] != 'ATTD': continue
        tag = p[1]; layer = int(p[2]); pos = int(p[3])
        if tag == 'meta': continue
        vals = [int(x) for x in p[4:]]
        d.setdefault((tag, layer), []).append(vals)
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
def flat(v):
    out = []
    for row in v: out.extend(row)
    return out
print("%-5s %16s %16s %14s %14s" % ("layer", "qrot max|d|/n", "k max|d|/n", "score max|d|", "attout max|d|/n"))
worst = {}
for layer in sorted({k[1] for k in a}):
    row = []
    for tag in ('qrot', 'kv', 'score', 'attout'):
        va = flat(a.get((tag, layer), [])); vb = flat(b.get((tag, layer), []))
        if not va or len(va) != len(vb):
            row.append("shape"); continue
        dif = [abs(x - y) for x, y in zip(va, vb)]
        n = sum(1 for x in dif if x)
        row.append("%d/%d" % (max(dif), n))
        worst[tag] = max(worst.get(tag, 0), max(dif))
    print("%-5d %16s %16s %14s %14s" % (layer, row[0], row[1], row[2], row[3]))
print("worst over all layers:", worst)
