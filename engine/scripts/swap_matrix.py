#!/usr/bin/env python3
"""Acceptance for `/model`: swap containers in one process and compare.

Every turn must be byte-identical to what a *fresh* process prints for the same
container -- that is the whole contract of the swap, and a single golden string
is not enough to see it, because a wrong program or a stale target changes the
answer of some models and not others.

What it covers, in one run:

  q8 chain   minimind3 -> feng -> TinyChat15M -> minimind3-moe -> minimind2-small
             -> back to minimind3   (three template kinds, 15M to 198M-A64M MoE,
             and a round trip, with one turn generated before every swap)
  q4 chain   minimind3 -> minimind2-small -> feng -> back
  i16 chain  minimind3 -> minimind2-small -> TinyChat15M -> back
  hybrid     lfm2.5 dedup -> lfm2.5 q6 -> back   (conv path)
  mixed      minimind3.q8 -> .q4 -> minimind2.q4 -> feng.q8 -> lfm2.5.q4 -> back
  slash line a line that starts with "/" but is not a command is still a prompt
  worker     the protocol's {"op":"load"} line: two loads inside one worker,
             answers byte-identical to fresh workers, and a bad op is an error
  determinism the same chain twice prints the same bytes

Needs the GPU and the containers under ../models; takes about a minute on Xe.
Not part of regression.sh, which has to run without a GPU.

The chains cover q8, q4, i16 and the hybrid (conv) container, because each of
them broke in a different way: the dense q8/q4 chains hid a 1-ULP drift (the
second load ran the whole-width FFN where the first ran two chunks) and the
hybrid one lost metadata key 15 entirely.  A chain that only exercises dense
containers would pass with the bug present, which is exactly what happened
before prepare_model() was shared with the swap.

    python3 scripts/swap_matrix.py            # the chains above
    python3 scripts/swap_matrix.py --gen 8    # shorter generations
"""
import argparse, base64, json, os, subprocess, sys

SYS = "You are a helpful assistant."
GEN = 16
Q = "用一句话介绍你自己。"
ENV = dict(os.environ, SEQ="512", GPU_LAYER="1", KV_PACK="16")
BIN = "./engine_mm3"

GEN = 16
def run(lines, model, gen=None):
    gen = GEN if gen is None else gen
    e = dict(ENV, MODEL=model)
    p = subprocess.run([BIN, "--chat", "--system", SYS, "--gen", str(gen), "--greedy"],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, env=e, timeout=900)
    return p.stdout, p.stderr, p.returncode

def answers(out):
    """Every `Assistant:` payload, in order, as one string per turn."""
    res, cur = [], None
    for line in out.splitlines():
        if line.startswith("User: [model]") or line.startswith("User: [conversation"):
            continue   # a command echo, not a turn
        if line.startswith("User: "):
            if cur is not None: res.append(cur)
            cur = ""
            rest = line[len("User: "):]
            if rest.startswith("Assistant: "):
                cur = rest[len("Assistant: "):]
        elif cur is not None:
            cur += "\n" + line
    if cur is not None: res.append(cur)
    res = [a.strip() for a in res]
    while res and not res[-1]: res.pop()   # the trailing "User: " prompt at EOF
    return res

def fresh(model):
    out, err, rc = run([Q, "/exit"], model)
    a = answers(out)
    assert rc == 0 and a, f"fresh {model}: rc={rc} answers={a} err={err[-300:]}"
    return a[0]

def chain(models):
    lines, want = [], []
    for i, m in enumerate(models):
        if i: lines.append("/model " + m)
        lines.append(Q)
    lines.append("/exit")
    out, err, rc = run(lines, models[0])
    got = answers(out)
    assert rc == 0, f"chain rc={rc} err={err[-400:]}"
    return got, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--systems", default="You are a helpful assistant.")
    args = ap.parse_args()
    global SYS, GEN
    SYS, GEN = args.systems, args.gen
    q8 = ["../models/minimind3/minimind3.q8.mmcf",
          "../models/feng30m/feng.q8.mmcf",
          "../models/llama2c/TinyChat15M.q8.mmcf",
          "../models/minimind3-moe/minimind3-moe.q8.mmcf",
          "../models/llama2c/minimind2-small.q8.mmcf",
          "../models/minimind3/minimind3.q8.mmcf"]
    q4 = ["../models/minimind3/minimind3.q4.mmcf",
          "../models/llama2c/minimind2-small.q4.mmcf",
          "../models/feng30m/feng.q4.mmcf",
          "../models/minimind3/minimind3.q4.mmcf"]
    # i16 and the hybrid are here because they were *not* passable until
    # prepare_model() was shared with the swap: without split_ffn() the second
    # load ran the whole-width FFN where the first ran two chunks, and without
    # apply_hidden_scale() the hybrid lost metadata key 15.  Both were 1-ULP
    # invisible on the dense q8/q4 chains and changed answers here.
    i16 = ["../models/minimind3/minimind3.i16.mmcf",
           "../models/llama2c/minimind2-small.i16.mmcf",
           "../models/llama2c/TinyChat15M.i16.mmcf",
           "../models/minimind3/minimind3.i16.mmcf"]
    hybrid = ["../models/lfm2.5-230m/lfm2.5.q4.dedup.mmcf",
              "../models/lfm2.5-230m/lfm2.5.q4.q6.mmcf",
              "../models/lfm2.5-230m/lfm2.5.q4.dedup.mmcf"]
    # Quantisation changes inside one process: this is the case that answered
    # with repetitive garbage before prepare_model() was shared (the format plan
    # and the hidden scale belong to the model, and the second load inherited
    # neither).  It walks q8 -> q4 -> q4 -> q8 -> hybrid q4 -> q8.
    mixed = ["../models/minimind3/minimind3.q8.mmcf",
             "../models/minimind3/minimind3.q4.mmcf",
             "../models/llama2c/minimind2-small.q4.mmcf",
             "../models/feng30m/feng.q8.mmcf",
             "../models/lfm2.5-230m/lfm2.5.q4.dedup.mmcf",
             "../models/minimind3/minimind3.q8.mmcf"]
    fail = 0
    for tag, seq in (("q8", q8), ("q4", q4), ("i16", i16), ("hybrid", hybrid), ("mixed", mixed)):
        print(f"== {tag} chain: {len(seq)-1} swaps ==")
        ref = {}
        for m in seq:
            if m not in ref:
                ref[m] = fresh(m)
                print(f"  fresh {os.path.basename(m):26s} {ref[m][:32]}")
        got, err = chain(seq)
        if len(got) != len(seq):
            print(f"  FAIL: {len(got)} answers for {len(seq)} turns"); fail += 1; continue
        for i, (m, g) in enumerate(zip(seq, got)):
            ok = g == ref[m]
            print(f"  {'ok  ' if ok else 'FAIL'} turn {i+1} {os.path.basename(m):26s} {g[:32]}")
            fail += 0 if ok else 1
        if "RELEASE: gl error" in err: print("  FAIL: GL error after release"); fail += 1
    # a line that starts with / but is not a command must still be a prompt
    print("== a slash line that is not a command ==")
    slash = "/etc/hostname 是什么意思？"
    e = dict(ENV, MODEL=q8[0])
    p = subprocess.run([BIN, "--chat", "--system", SYS, "--gen", "8", "--greedy"],
                       input=slash + "\n/exit\n", capture_output=True, text=True, env=e, timeout=600)
    a = answers(p.stdout)
    print(f"  {'ok  ' if a else 'FAIL'} answered as a prompt: {a[0][:40] if a else '(none)'}")
    fail += 0 if a else 1
    # determinism: the same chain twice
    print("== determinism (chain run twice) ==")
    g1, _ = chain(q8); g2, _ = chain(q8)
    print(f"  {'ok  ' if g1 == g2 else 'FAIL'} identical")
    fail += 0 if g1 == g2 else 1
    print("== worker protocol load op ==")
    def wrun(lines, model):
        e = dict(ENV, MODEL=model)
        out = subprocess.run([BIN, "--worker"], input="".join(lines), capture_output=True,
                             text=True, env=e, timeout=900).stdout
        return out
    def wreq(msg, n=16, seed=42):
        doc = {"messages": [{"role": "user", "content": msg}]}
        b = base64.b64encode(json.dumps(doc, ensure_ascii=False,
                                        separators=(",", ":")).encode()).decode()
        return "\t%d\t0\t1\t%d\t\t\t\t\t\t%s\t\n" % (n, seed, b)
    def wsegs(out):
        segs, cur = [], []
        for line in out.splitlines():
            ev = json.loads(line)
            if ev.get("event") == "token": cur.append(ev.get("text", ""))
            elif ev.get("event") == "done": segs.append("".join(cur)); cur = []
        return segs
    wq = "用一句话介绍你自己。"
    wa, wb, wc = (q8[0], q8[4], q4[0])          # minimind3.q8, minimind2-small.q8, minimind3.q4
    fa = wsegs(wrun([wreq(wq)], wa))[0]
    fb = wsegs(wrun([wreq(wq)], wb))[0]
    fc = wsegs(wrun([wreq(wq)], wc))[0]
    mixed = wrun([wreq(wq),
                  json.dumps({"op": "load", "model": wb}) + "\n", wreq(wq),
                  json.dumps({"op": "load", "model": wc}) + "\n", wreq(wq)], wa)
    got = wsegs(mixed)
    ok = got == [fa, fb, fc]
    print(f"  {'ok  ' if ok else 'FAIL'} load op: {len(got)} answers across 2 loads "
          f"(q8 -> q8 -> q4), byte-identical to fresh workers")
    fail += 0 if ok else 1
    # a bad op must be an error event, not a crash
    bad_out = wrun(['{"op":"nope"}\n'], wa)
    errs = [json.loads(l) for l in bad_out.splitlines() if '"error"' in l]
    ok2 = bool(errs) and "unknown op" in errs[0].get("message", "")
    print(f"  {'ok  ' if ok2 else 'FAIL'} unknown op reported as an error event")
    fail += 0 if ok2 else 1

    print("RESULT:", "PASS" if fail == 0 else f"{fail} FAILURES")
    if fail:
        print("Each line named FAIL above prints the post-swap answer next to the\n"
              "fresh one; a wrong program cache or a stale target is the usual cause\n"
              "(see AGENTS.md, 'Switching models without restarting the engine').")
    return 0 if fail == 0 else 1

sys.exit(main())
