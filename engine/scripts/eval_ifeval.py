#!/usr/bin/env python3
"""Instruction-following checks for the tiny models: does the reply obey?

Why this exists
---------------
"the reply looked good" is not a measurement.  The chat comparison of
2026-11-12 found that the MiniMind models answer questions passably but ignore
formatting instructions (JSON, word limits, exact endings), which is the failure
a chat user hits first.  This runs a case list, samples it a few times, and
scores each case against regexes and counts, so the pass rate is a number a
re-run reproduces.

A case is one JSON object per line:

    {"id": "json-1", "lang": "zh", "prompt": "Give me a JSON object with ...",
     "gen": 160,
     "checks": [{"kind": "json_valid"}, {"kind": "json_keys", "keys": ["a"]}],
     "good_example": "{\"a\": 1}",
     "bad_example": "a = 1"}

The *_example fields are optional; --selftest uses them to check the case file
itself (every good_example must pass, every bad_example must fail), which is
what stops a wrong regex from silently scoring a model down.

Checks (all must pass): regex, absent_regex, starts_with, ends_with, min_words,
max_words, min_chars, max_chars, min_lines, max_lines, is_upper, contains,
contains_any, contains_none, json_valid, json_keys, numbered_list, bullets,
cjk_ratio_min, ascii_ratio_min.

Usage
-----
    python3 engine/scripts/eval_ifeval.py \
        --model minimind3-q8=models/minimind3/minimind3.q8.mmcf --preset minimind \
        --data engine/scripts/eval/ifeval.jsonl --seeds 3 --out /tmp/eval_ifeval

    python3 engine/scripts/eval_ifeval.py --data engine/scripts/eval/ifeval.jsonl --selftest
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NL = chr(10)
WORD_RE = re.compile(r"([A-Za-z0-9']+|[\u4e00-\u9fff])")

# The two models' own recommended sampling settings, not one shared greedy run:
# a 64M model needs a little entropy to produce anything, the QAD LFM is
# distilled for near-greedy decoding.
PRESETS = {
    "minimind": {"temp": 0.85, "top-k": 50, "top-p": 0.85},
    "lfm": {"temp": 0.1, "top-k": 50, "repeat-penalty": 1.05},
    "greedy": {"greedy": 1},
}


def words(text):
    return WORD_RE.findall(text)


def lines(text):
    return [ln for ln in text.splitlines() if ln.strip()]


def _json(text):
    try:
        return json.loads(text.strip())
    except ValueError:
        return None


def check_one(chk, text):
    kind = chk["kind"]
    body = text.strip()
    if kind == "regex":
        flags = 0
        for f in chk.get("flags", []):
            flags |= {"i": re.I, "m": re.M, "s": re.S}[f]
        return re.search(chk["pattern"], body, flags) is not None, ""
    if kind == "absent_regex":
        return re.search(chk["pattern"], body) is None, ""
    if kind == "starts_with":
        return text.lstrip().startswith(chk["text"]), ""
    if kind == "ends_with":
        return text.rstrip().endswith(chk["text"]), ""
    if kind == "min_words":
        return len(words(body)) >= chk["n"], "%d words" % len(words(body))
    if kind == "max_words":
        return len(words(body)) <= chk["n"], "%d words" % len(words(body))
    if kind == "min_chars":
        return len(body) >= chk["n"], "%d chars" % len(body)
    if kind == "max_chars":
        return len(body) <= chk["n"], "%d chars" % len(body)
    if kind == "min_lines":
        return len(lines(body)) >= chk["n"], "%d lines" % len(lines(body))
    if kind == "max_lines":
        return len(lines(body)) <= chk["n"], "%d lines" % len(lines(body))
    if kind == "is_upper":
        letters = [c for c in body if c.isalpha()]
        return len(letters) >= 3 and all(c.isupper() for c in letters), ""
    if kind == "contains":
        return chk["text"] in body, ""
    if kind == "contains_any":
        return any(t in body for t in chk["texts"]), ""
    if kind == "contains_none":
        hit = [t for t in chk["texts"] if t in body]
        return not hit, ("found %r" % (hit[:3],)) if hit else ""
    if kind == "json_valid":
        obj = _json(body)
        return obj is not None, "" if obj is not None else "not JSON"
    if kind == "json_keys":
        obj = _json(body)
        if not isinstance(obj, dict):
            return False, "not a JSON object"
        missing = [k for k in chk["keys"] if k not in obj]
        return not missing, ("missing %r" % (missing,)) if missing else ""
    if kind == "numbered_list":
        n = sum(1 for ln in lines(body)
                if re.match(r"^\s*\d+[.)]\s+\S", ln))
        return n >= chk["n"], "%d numbered lines" % n
    if kind == "bullets":
        n = sum(1 for ln in lines(body)
                if re.match(r"^\s*[-*\u2022]\s+\S", ln))
        return n >= chk["n"], "%d bullets" % n
    if kind == "cjk_ratio_min":
        chars = [c for c in body if not c.isspace()]
        if not chars:
            return False, "empty"
        r = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff") / len(chars)
        return r >= chk["ratio"], "cjk %.2f" % r
    if kind == "ascii_ratio_min":
        chars = [c for c in body if not c.isspace()]
        if not chars:
            return False, "empty"
        r = sum(1 for c in chars if ord(c) < 128) / len(chars)
        return r >= chk["ratio"], "ascii %.2f" % r
    raise SystemExit("unknown check kind %r (see the module docstring)" % kind)


def run_checks(checks, text):
    bad = []
    for chk in checks:
        try:
            ok, detail = check_one(chk, text)
        except SystemExit:
            raise
        except Exception as e:  # a malformed case must not kill the battery
            ok, detail = False, "check error: %s" % e
        if not ok:
            bad.append({"kind": chk["kind"], "detail": detail})
    return bad


class Engine:
    def __init__(self, binary, model, env, system, sampling):
        self.binary = binary
        self.model = model
        self.env = dict(env)
        self.system = system
        self.sampling = dict(sampling)

    def chat(self, prompt, gen, seed):
        env = dict(os.environ)
        env.update(self.env)
        cmd = [self.binary, "--chat", "--system", self.system,
               "--model", self.model, "--prompt", prompt, "--gen", str(gen),
               "--seed", str(seed)]
        for k, v in sorted(self.sampling.items()):
            if k == "greedy":
                cmd.append("--greedy")
            elif k == "temp":
                cmd += ["--temp", str(v)]
            elif k == "top-k":
                cmd += ["--top-k", str(v)]
            elif k == "top-p":
                cmd += ["--topp", str(v)]
            elif k == "repeat-penalty":
                cmd += ["--repeat-penalty", str(v)]
            else:
                raise SystemExit("unknown sampling knob %r" % k)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=env, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            return None, str(e), (time.time() - t0) * 1000.0
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        if "Assistant: " not in out:
            return None, "no Assistant line; rc %s: %s" % (p.returncode,
                                                          err[-300:]), 0.0
        out = out.split("Assistant: ", 1)[1]
        # the engine prints one newline of its own after the reply
        return out.rstrip(chr(10)), err, (time.time() - t0) * 1000.0


def load_cases(path, limit=None):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def selftest(cases):
    bad = 0
    for c in cases:
        for key, want_pass in (("good_example", True), ("bad_example", False)):
            if key not in c:
                continue
            fails = run_checks(c["checks"], c[key])
            got = not fails
            if got != want_pass:
                bad += 1
                print("FAIL %-8s %-12s expected %s, got %s: %s" %
                      (c["id"], key, "pass" if want_pass else "fail",
                       "pass" if got else "fail", fails[:2]))
    n_good = sum(1 for c in cases if "good_example" in c)
    print("selftest: %d cases, %d with examples, %d mismatches" %
          (len(cases), n_good, bad))
    return 1 if bad else 0


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    passed = sum(1 for r in rows if r["pass"])
    by_kind = {}
    for r in rows:
        for b in r["bad"]:
            by_kind[b["kind"]] = by_kind.get(b["kind"], 0) + 1
    by_case = {}
    for r in rows:
        d = by_case.setdefault(r["id"], [0, 0])
        d[0] += 1
        d[1] += 1 if r["pass"] else 0
    s = {
        "attempts": n,
        "pass": passed,
        "pass_rate": passed / n,
        "cases": len(by_case),
        "hard_pass": sum(1 for v in by_case.values() if v[1] == v[0]) / len(by_case),
        "by_kind": by_kind,
    }
    for lang in ("zh", "en"):
        sel = [r for r in rows if r.get("lang") == lang]
        if sel:
            s.setdefault("by_lang", {})[lang] = sum(1 for r in sel if r["pass"]) / len(sel)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/tmp/eval_ifeval")
    ap.add_argument("--preset", default="minimind", choices=sorted(PRESETS))
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gen", type=int, default=0, help="override every case's gen")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--engine", default=os.path.join(ROOT, "engine", "engine_mm3"))
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cases = load_cases(args.data, args.limit or None)
    if not cases:
        raise SystemExit("no cases in %s" % args.data)
    if args.selftest:
        return selftest(cases)
    if not args.model:
        raise SystemExit("--model NAME=PATH is required (or use --selftest)")

    sampling = dict(PRESETS[args.preset])
    for k, v in (("temp", args.temp), ("top-k", args.top_k),
                 ("top-p", args.top_p), ("repeat-penalty", args.repeat_penalty)):
        if v is not None:
            sampling[k] = v
    env = {"GPU_LAYER": "1", "KV_PACK": "16"}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    os.makedirs(args.out, exist_ok=True)
    print("engine   %s" % args.engine)
    print("cases    %d from %s" % (len(cases), args.data))
    print("sampling %s  seeds %d" % (sampling, args.seeds))
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            raise SystemExit("no such model: %s" % path)
        print("%s== %s (%s)" % (NL, name, path), flush=True)
        eng = Engine(args.engine, path, env, args.system, sampling)
        out_path = os.path.join(args.out, name + ".attempts.jsonl")
        done = set()
        rows = []
        if args.resume and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    # keep the attempts so a resumed summary still covers the
                    # whole battery instead of only the new runs
                    rows.append(r)
                    done.add((r["id"], r["seed"]))
        fh = open(out_path, "a" if done else "w", encoding="utf-8")
        for case in cases:
            gen = args.gen or int(case.get("gen", 160))
            for seed in range(args.seeds):
                if (case["id"], seed) in done:
                    continue
                text, err, ms = eng.chat(case["prompt"], gen, seed)
                if text is None:
                    row = {"id": case["id"], "seed": seed, "error": err,
                           "pass": False, "bad": [{"kind": "engine", "detail": err}]}
                else:
                    bad = run_checks(case["checks"], text)
                    row = {"id": case["id"], "seed": seed, "text": text,
                           "pass": not bad, "bad": bad,
                           "ms": round(ms, 1), "lang": case.get("lang", "en")}
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + NL)
                fh.flush()
                if not args.quiet:
                    ok = sum(1 for r in rows if r["pass"])
                    print("  [%3d] %-10s seed %d  %s  pass %.2f %5.0f ms" %
                          (len(rows), case["id"], seed,
                           "OK" if row["pass"] else "FAIL",
                           ok / len(rows), row.get("ms", 0)), flush=True)
        fh.close()
        s = summarize(rows)
        with open(os.path.join(args.out, name + ".summary.json"), "w") as fh2:
            json.dump(s, fh2, ensure_ascii=False, indent=1)
        print("  cases %d  attempts %d  pass %.3f  hard-pass %.3f" %
              (s["cases"], s["attempts"], s["pass_rate"], s["hard_pass"]))
        if s["by_kind"]:
            print("  failed checks: %s" % "  ".join(
                "%s %d" % (k, v) for k, v in sorted(s["by_kind"].items())))
        if s.get("by_lang"):
            print("  by lang: %s" % "  ".join("%s %.3f" % (k, v)
                                              for k, v in sorted(s["by_lang"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
