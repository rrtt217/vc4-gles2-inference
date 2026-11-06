#!/usr/bin/env python3
"""Offline oracle for the ATT_DUMP probe in engine/src/gpu.cpp.

The attention block on the VC4 path is five dependent passes (qrot, kv_rope,
score, stats, v).  ATT_DUMP prints the exact integers each one consumed and
produced for one layer and position, so this script checks the whole block
without a second engine path:

  score[t]   == round( (sum_i q_i*k_i) * invsqrt(head_dim) * 1024 )
  stats max  == max_t score[t]                (exact, same 1/1024 grid)
  stats sum  == round( 32 * sum_t exp((score[t]-max)/1024) )
  att_out[i] == round( 256 * sum_t p_t * v_t[i] ),  p_t = exp(..)/sum

q and att_out are q16 activations (int/256); K and V are on the KV grid
(int/KVSCALE); scores and the stats max on 1/1024; the stats sum on 1/32.
All grids and shapes come from the ATTD meta line.

Usage: check_att.py <log>
"""
import math
import sys


def parse(path):
    qrot = {}
    kv = {}
    kvv = {}
    score = {}
    smax = {}
    ssum = {}
    attout = {}
    meta = None
    for line in open(path, errors="replace"):
        i = line.find("ATTD ")
        if i < 0:
            continue
        f = line[i:].split()
        tag = f[1]
        if tag == "meta":
            meta = [float(x) for x in f[2:]]
            continue
        nums = [int(x) for x in f[2:]]
        if tag == "qrot":
            qrot[(nums[0], nums[1])] = nums[3:]
        elif tag == "kv":
            kv[(nums[0], nums[2])] = nums[4:]
        elif tag == "kvv":
            kvv[(nums[0], nums[2])] = nums[4:]
        elif tag == "score":
            score[(nums[0], nums[1], nums[2])] = nums[3:]
        elif tag == "statsmax":
            smax[(nums[0], nums[1])] = nums[3:]
        elif tag == "statssum":
            ssum[(nums[0], nums[1])] = nums[3:]
        elif tag == "attout":
            attout[(nums[0], nums[1])] = nums[3:]
    return meta, qrot, kv, kvv, score, smax, ssum, attout


def main():
    meta, qrot, kv, kvv, score, smax, ssum, attout = parse(sys.argv[1])
    if not meta:
        print("no ATTD meta line: was ATT_DUMP set?")
        return 1
    heads = int(meta[0])
    kv_heads = int(meta[1])
    head_dim = int(meta[2])
    dim = int(meta[3])
    kv_dim = int(meta[4])
    kvscale = meta[5]
    group = heads // kv_heads
    inv = 1.0 / math.sqrt(head_dim)
    fails = 0
    for layer in sorted(set(k[0] for k in score)):
        for pos in sorted(set(k[1] for k in score if k[0] == layer)):
            q = qrot[(layer, pos)]
            out = attout.get((layer, pos), [])
            mx_got = smax[(layer, pos)]
            sum_got = ssum[(layer, pos)]
            sdev = 0
            sbad = 0
            mb = 0
            sb = 0.0
            odev = 0
            for h in range(heads):
                got = score[(layer, pos, h)]
                kh = h // group
                ref = []
                for t, g in enumerate(got):
                    k = kv[(layer, t)]
                    s = 0.0
                    base = kh * head_dim
                    for i in range(head_dim):
                        s += (q[h * head_dim + i] / 256.0) * (k[base + i] / kvscale)
                    v = int(math.floor(s * inv * 1024.0 + 0.5))
                    v = max(-32768, min(32767, v))
                    ref.append(v)
                    if abs(v - g) > 1:
                        sbad += 1
                        if sbad <= 4:
                            print("  score L%d p%d h%d t%d got %d want %d" % (layer, pos, h, t, g, v))
                    sdev = max(sdev, abs(v - g))
                mx = max(got)
                if mx_got[h] != mx:
                    mb += 1
                    print("  statsmax L%d p%d h%d got %d want %d" % (layer, pos, h, mx_got[h], mx))
                st = sum(math.exp((v - mx) / 1024.0) for v in got)
                sb = max(sb, abs(sum_got[h] / 32.0 - st))
                if abs(sum_got[h] - st * 32.0) > 2.0:
                    print("  statssum L%d p%d h%d got %d want %.2f" % (layer, pos, h, sum_got[h], st * 32.0))
                sm = sum_got[h] / 32.0
                mxv = mx_got[h] / 1024.0
                pv = [math.exp(v / 1024.0 - mxv) / sm for v in got]
                for i in range(head_dim):
                    acc = 0.0
                    ii = kh * head_dim + i
                    for t in range(pos + 1):
                        acc += pv[t] * (kvv[(layer, t)][ii] / kvscale)
                    w = int(math.floor(acc * 256.0 + 0.5))
                    if out:
                        odev = max(odev, abs(w - out[h * head_dim + i]))
            bad = sbad + mb + (1 if sb > 0.5 else 0) + (1 if odev > 2 else 0)
            fails += bad
            print("L%-2d p%-2d scores dev<=%d bad=%d | stats max bad=%d sum dev=%.3f | att_out dev=%d %s"
                  % (layer, pos, sdev, sbad, mb, sb, odev, "FAIL" if bad else "ok"))
    print("FAILURES: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
