#!/usr/bin/env python3
"""Multiple-choice accuracy for the tiny models, scored from LOGITS_OUT.

Why this exists
---------------
The 64M-230M models cannot be judged by reading one greedy reply: the reply is
chaotic, the sampler dominates it, and 80 tokens of one sample say nothing about
whether the model *knows* an answer.  A multiple-choice question scored by the
probability of the option letter is deterministic (no sampler), cheap (one
engine run per item: prefill plus --gen 0) and gives a number a human can
re-derive: the logits of "A"/"B"/"C"/"D" at the answer position.

The letter ids are not hard-coded.  They are read out of the model's own
tokenizer with the engine's TOKENS_OUT: encode "<question>...答案：" and
"<question>...答案：A", the first id after the common prefix is the token the
model would emit for "A" at that position.  A tokenizer that splits a letter
into more than one token is refused rather than scored wrongly.

Scoring
-------
* acc      -- argmax over the four letter logits (the headline number).
* acc_cal  -- the same after subtracting the logits the same prompt suffix
              produces with *no* question ("答案：" alone): the classic
              domain-contextual calibration, which removes the letter prior.
              A model that always answers "A" scores 0.25 here, not 0.95.
* fmt      -- fraction of items whose full-vocabulary argmax *is* one of the
              four letters, i.e. whether the model is even answering in the
              asked format (a low value means the accuracy is measuring the
              model's instruction following, not its knowledge).
* p_gold   -- mean probability the model puts on the correct letter.

Usage
-----
    python3 engine/scripts/bench_mc.py \
        --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
        --model moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
        --model lfm-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
        --data engine/scripts/eval/mc_zh.jsonl \
        --data engine/scripts/eval/mc_en.jsonl \
        --out /tmp/bench_mc

The per-item dump (<out>/<model>.items.jsonl) keeps the raw probabilities so a
surprising number can be re-checked without re-running the GPU.  --resume
appends to it and skips items already there, which is what a long battery
interrupted by the harness timeout needs.
"""
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LETTERS = "ABCD"
NL = chr(10)

# The rendering is a contract with the dataset files: an item carries a "lang"
# and the renderer follows from it.  The answer *instruction* is the Chinese one
# in both languages, and that is a measurement, not an oversight: with
# "Answer with the option letter only." the MiniMind pair answers English items
# with prose ("Based on...", "The question is...") under greedy decode, while
# the same question behind the Chinese instruction gets a letter from all three
# models (2026-11-12; three English phrasings tried, none worked).  A
# per-language instruction would have measured instruction-language preference
# instead of knowledge; the report's "fmt" column still shows how often the
# reply's first token is a letter.
SUFFIX = NL + "只回答选项字母。" + NL + "答案："
EN_PREFIX = "Question: "


def render(item):
    body = NL.join("%s. %s" % (L, o) for L, o in zip(LETTERS, item["options"]))
    if item.get("lang", "en") == "zh":
        return item["question"] + NL + body + SUFFIX
    return EN_PREFIX + item["question"] + NL + body + SUFFIX


def content_free():
    """The same suffix with no question, for the calibration baseline."""
    return SUFFIX.lstrip(chr(10))


def read_floats(path):
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError:
        return None
    if not b or len(b) % 4:
        return None
    return list(struct.unpack("<%df" % (len(b) // 4), b))


def read_ints(path):
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError:
        return None
    if not b or len(b) % 4:
        return None
    return list(struct.unpack("<%di" % (len(b) // 4), b))


def argmax(xs):
    best, bi = -1e30, 0
    for i, v in enumerate(xs):
        if v > best:
            best, bi = v, i
    return bi


def softmax(vals):
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [x / s for x in e]


class Engine:
    """One prefill per question, no decode: LOGITS_OUT is the whole answer."""

    def __init__(self, binary, model, env, system):
        self.binary = binary
        self.model = model
        self.env = dict(env)
        self.system = system

    def run(self, prompt, want_logits=True):
        t0 = time.time()
        tag = "%d_%.6f" % (os.getpid(), t0)
        lg = "/tmp/bench_mc_%s.lg" % tag
        tk = "/tmp/bench_mc_%s.tk" % tag
        env = dict(os.environ)
        env.update(self.env)
        if want_logits:
            env["LOGITS_OUT"] = lg
        env["TOKENS_OUT"] = tk
        cmd = [self.binary, "--chat", "--system", self.system,
               "--model", self.model, "--prompt", prompt, "--gen", "0"]
        try:
            p = subprocess.run(cmd, env=env, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rc, err = p.returncode, p.stderr.decode("utf-8", "replace")
        except OSError as e:
            return None, None, str(e), (time.time() - t0) * 1000.0, 127
        logits = read_floats(lg) if want_logits else None
        ids = read_ints(tk)
        for q in (lg, tk):
            if os.path.exists(q):
                os.remove(q)
        return logits, ids, err, (time.time() - t0) * 1000.0, rc


PROBE_ITEM = {"question": "1+1\u7b49\u4e8e\u51e0\uff1f",
              "options": ["1", "2", "3", "4"], "lang": "zh", "answer": 1}
# A separator that does not merge into the letter under byte-level BPE.  The
# full-width colon is the zh answer suffix; a newline is the safe fallback (it
# is also what the chat template puts in front of the reply).
PROBE_SEPARATORS = ["\uff1a", chr(10)]
# Reported as <model>.letters.json, one id per letter; the four ids are all the
# scorer needs, and they do not depend on the question or on the language.
def probe_letter_ids(engine, base_item=None):
    """The id each option letter has *at the answer position*.

    The engine renders "<user text><template><assistant marker>" and the letter
    is appended inside the user text: the tokens differ from the base prompt
    exactly in the trailing letter, so the first differing id is that letter.
    A tokenizer that merges the separator with the letter makes the splice
    visible (it is not one id), so another separator is tried before failing.
    """
    item = base_item or PROBE_ITEM
    base_prompt = render(item)
    last_err = ""
    for sep in PROBE_SEPARATORS:
        base = base_prompt + sep
        _, base_ids, err, _, rc = engine.run(base, want_logits=False)
        if not base_ids:
            last_err = "tokenizer probe failed (rc %s): %s" % (rc, err[-300:])
            continue
        ids, ok = {}, True
        for L in LETTERS:
            _, tok, err, _, rc = engine.run(base + L, want_logits=False)
            if not tok:
                last_err = "tokenizer probe failed for %r (rc %s): %s" % (
                    L, rc, err[-300:])
                ok = False
                break
            n = 0
            while n < min(len(base_ids), len(tok)) and base_ids[n] == tok[n]:
                n += 1
            # The letter lands *inside* the user text, so it is inserted before
            # the template's remaining markers: the appended list must be the
            # base list with exactly one id spliced in at n.
            if not (len(tok) == len(base_ids) + 1 and tok[n + 1:] == base_ids[n:]):
                last_err = ("letter %r is not one token after separator %r "
                            "(common prefix %d, base %d ids, appended %d ids)"
                            % (L, sep, n, len(base_ids), len(tok)))
                ok = False
                break
            if tok[n] in ids.values():
                last_err = "letters A-D map to a duplicated id %d" % tok[n]
                ok = False
                break
            ids[L] = tok[n]
        if ok:
            return ids
    raise SystemExit("cannot find the option-letter token ids: %s" % last_err)


def load_items(paths, limit=None):
    items = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                items.append(json.loads(line))
    return items[:limit] if limit else items


def load_done(path):
    done = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "error" not in r:
                    done[r["id"]] = r
    return done


def score_item(engine, item, letter_ids, base):
    logits, _, err, ms, rc = engine.run(render(item))
    if not logits:
        return {"id": item["id"], "error": "rc %s: %s" % (rc, err.strip()[-200:])}
    lv = [logits[letter_ids[L]] for L in LETTERS]
    p = softmax(lv)
    gold = int(item["answer"])
    pred = argmax(lv)
    row = {
        "id": item["id"],
        "lang": item.get("lang", "en"),
        "category": item.get("category", "?"),
        "difficulty": item.get("difficulty", "?"),
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "p": [round(x, 6) for x in p],
        "p_gold": round(p[gold], 6),
        "letter_logits": [round(x, 4) for x in lv],
        "top1_letter": argmax(logits) in letter_ids.values(),
        "ms": round(ms, 1),
    }
    if base is not None:
        cal = [lv[i] - base[i] for i in range(4)]
        row["pred_cal"] = argmax(cal)
        row["correct_cal"] = row["pred_cal"] == gold
    return row


def null_accuracy(preds, golds, trials=200, seed=1234):
    """Accuracy of the same answer letters against shuffled gold labels.

    A model with a strong letter prior scores whatever the prior does against
    balanced gold, so "0.21" only means something next to the null: the accuracy
    this prediction distribution would get if the questions carried no
    information.  Deterministic (fixed seed), no GPU.
    """
    rnd = random.Random(seed)
    golds = list(golds)
    accs = []
    for _ in range(trials):
        rnd.shuffle(golds)
        accs.append(sum(1 for p, g in zip(preds, golds) if p == g) / len(preds))
    m = sum(accs) / len(accs)
    sd = (sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5
    return m, sd


def summarize(name, rows):
    good = [r for r in rows if "error" not in r]
    n = len(good)
    s = {"model": name, "n": n, "errors": len(rows) - n}
    if not n:
        return s
    s["acc"] = sum(1 for r in good if r["correct"]) / n
    s["null_acc"], s["null_sd"] = null_accuracy(
        [r["pred"] for r in good], [r["gold"] for r in good])
    hist = {L: 0 for L in LETTERS}
    for r in good:
        hist[LETTERS[r["pred"]]] += 1
    s["pred_hist"] = hist
    s["fmt"] = sum(1 for r in good if r["top1_letter"]) / n
    s["p_gold"] = sum(r["p_gold"] for r in good) / n
    s["ms"] = sum(r["ms"] for r in good) / n
    if "correct_cal" in good[0]:
        s["acc_cal"] = sum(1 for r in good if r["correct_cal"]) / n
    by = {}
    for key in ("lang", "category", "difficulty"):
        d = {}
        for r in good:
            a = d.setdefault(r[key], [0, 0])
            a[0] += 1
            a[1] += 1 if r["correct"] else 0
        by[key] = {k: (v[1] / v[0], v[0]) for k, v in sorted(d.items())}
    s["by"] = by
    return s


def pct(x):
    return "%.3f" % x if x is not None else "  -  "


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--data", action="append", required=True,
                    metavar="FILE.jsonl", help="repeatable")
    ap.add_argument("--out", default="/tmp/bench_mc")
    ap.add_argument("--limit", type=int, default=0, help="score at most N items in total")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--engine", default=os.path.join(ROOT, "engine", "engine_mm3"))
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    env = {"GPU_LAYER": "1", "KV_PACK": "16"}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    models = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit("--model wants NAME=PATH, got %r" % spec)
        models.append((name, path))
    items = load_items(args.data, args.limit or None)
    if not items:
        raise SystemExit("no items")
    os.makedirs(args.out, exist_ok=True)
    print("engine  %s" % args.engine)
    print("items   %d from %d file(s)" % (len(items), len(args.data)))
    summaries = []
    for name, path in models:
        if not os.path.exists(path):
            raise SystemExit("no such model: %s" % path)
        print("%s== %s (%s)" % (NL, name, path), flush=True)
        eng = Engine(args.engine, path, env, args.system)
        cache = os.path.join(args.out, name + ".letters.json")
        letter_ids = None
        if os.path.exists(cache):
            with open(cache) as f:
                letter_ids = json.load(f)
        if not letter_ids:
            letter_ids = probe_letter_ids(eng)
            print("  letters %s" % letter_ids, flush=True)
            with open(cache, "w") as f:
                json.dump(letter_ids, f)
        base = None
        if not args.no_calibrate:
            lg, _, err, _, rc = eng.run(content_free())
            if not lg:
                raise SystemExit("calibration baseline failed (rc %s): %s"
                                 % (rc, err[-400:]))
            base = [lg[letter_ids[L]] for L in LETTERS]
        out_path = os.path.join(args.out, name + ".items.jsonl")
        done = load_done(out_path) if args.resume else {}
        rows = list(done.values())
        f = open(out_path, "a" if done else "w", encoding="utf-8")
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            row = score_item(eng, it, letter_ids, base)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + NL)
            f.flush()
            if not args.quiet:
                ok = sum(1 for r in rows if r.get("correct"))
                print("  [%3d/%3d] %-7s %-11s %6s  acc %.3f  %5.1f ms" %
                      (len(rows), len(items), it["id"], it.get("category", "?"),
                       "OK" if row.get("correct") else ("ERR" if "error" in row else "-"),
                       ok / len(rows), row.get("ms", 0)), flush=True)
        f.close()
        s = summarize(name, rows)
        with open(os.path.join(args.out, name + ".summary.json"), "w") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1)
        summaries.append(s)
        for key in ("lang", "category", "difficulty"):
            parts = ["%s %.3f(%d)" % (k, v[0], v[1]) for k, v in s["by"][key].items()]
            print("  %-10s %s" % (key, "  ".join(parts)))
    print("%s%-18s %4s %6s %8s %6s %7s %-14s %7s" %
          (NL, "model", "n", "acc", "acc_cal", "fmt", "p_gold", "null(acc+/-sd)", "ms"))
    for s in summaries:
        null = ("%.3f+/-%.3f" % (s["null_acc"], s["null_sd"])) if "null_acc" in s else "  -  "
        print("%-18s %4d %6s %8s %6s %7s %-14s %7.0f" %
              (s["model"], s["n"], pct(s.get("acc")), pct(s.get("acc_cal")),
               pct(s.get("fmt")), pct(s.get("p_gold")), null, s.get("ms") or 0))
        if "pred_hist" in s:
            print("  predicted letters: %s" % "  ".join(
                "%s %d" % (k, v) for k, v in sorted(s["pred_hist"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
