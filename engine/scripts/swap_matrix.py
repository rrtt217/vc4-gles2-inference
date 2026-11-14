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
  slash line a line that starts with "/" but is not a command is still a prompt
  determinism the same chain twice prints the same bytes

Needs the GPU and the containers under ../models; takes about a minute on Xe.
Not part of regression.sh, which has to run without a GPU.

Scope: the chains are dense q8 and q4 containers on purpose.  Swapping the
*same* container already moves the numbers by about one unit in the last q16
place (logits cosine 0.9998 against a fresh process), which flips the greedy
text on minimind3.i16 and, much more strongly, on both lfm2.5 containers
(cosine 0.96 -- the conv path reads across positions).  Those three are therefore
not in the chains until that drift is found; a chain that includes them fails
here for a reason that is real and already known, and adding them now would only
hide it.  See AGENTS.md, "Known open issue" in the swap section.

    python3 scripts/swap_matrix.py            # the chains above
    python3 scripts/swap_matrix.py --gen 8    # shorter generations
"""
import argparse, os, subprocess, sys

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
    fail = 0
    for tag, seq in (("q8", q8), ("q4", q4)):
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
    print("RESULT:", "PASS" if fail == 0 else f"{fail} FAILURES")
    if fail:
        print("Each line named FAIL above prints the post-swap answer next to the\n"
              "fresh one; a wrong program cache or a stale target is the usual cause\n"
              "(see AGENTS.md, 'Switching models without restarting the engine').")
    return 0 if fail == 0 else 1

sys.exit(main())
