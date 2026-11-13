#!/usr/bin/env python3
"""Translation quality for the tiny models, with chrF instead of impressions.

Why this exists
---------------
The chat comparison of 2026-11-12 judged translation by reading one sampled
reply to "把'今天天气很好'翻译成英文".  One sentence, one seed, no reference:
that cannot separate "the model translates" from "the model finished the
sentence".  This battery runs a fixed bilingual corpus in both directions,
greedy (translation is a deterministic task, and the earlier session learned
that sampling noise on a 64M model swamps the signal), and scores each output
against a reference with chrF, the character n-gram F-score that is
tokenizer-free and therefore comparable between a Chinese and an English
reference.

A case is one JSON object per line:

    {"id": "zh-en-01", "src_lang": "zh", "tgt_lang": "en",
     "category": "daily", "source": "今天天气很好。",
     "reference": "The weather is nice today."}

Both directions use the same Chinese instruction (\u201c...\u53ea\u8f93\u51fa\u8bd1\u6587\u3002\u201d) for the
reason the multiple-choice battery documents: the MiniMind pair ignores English
formatting instructions, and a per-language instruction would measure that
preference rather than translation.

Metrics
-------
* chrF -- character n-gram F-score, n = 1..6, beta = 2 (recall-weighted), mean
  over sentences; whitespace removed, as sacrebleu's default chrF does.  A
  reference-free copy of the source scores ~0.3-0.5 here, so read the number
  next to the "leak" column rather than alone.
* exact -- the output equals the reference after whitespace folding.
* len -- mean output length over reference length (collapse shows as ~0.2, a
  stuck decoder as >2).
* tgt / leak -- fraction of the output's non-space characters that are in the
  *target* / *source* script (CJK for zh, ASCII letters for en).  A model that
  repeats the source instead of translating shows leak ~1.0.
* empty -- fraction of outputs with no content at all.

Usage
-----
    python3 engine/scripts/eval_translate.py \
        --model minimind3-q8=models/minimind3/minimind3.q8.mmcf \
        --model minimind3-moe-q8=models/minimind3-moe/minimind3-moe.q8.mmcf \
        --model lfm2.5-qad-q4=models/lfm2.5-230m/qad/lfm2.5.qad.q4.mmcf \
        --data engine/scripts/eval/translate.jsonl --out /tmp/eval_translate

The per-item dump keeps every hypothesis next to its reference, so a chrF that
looks wrong can be read rather than argued about.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NL = chr(10)
CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def render(item):
    lang = "英文" if item["tgt_lang"] == "en" else "中文"
    return ("把下面这句话翻译成" + lang + "，只输出译文。" + NL + item["source"])


def norm(text):
    return re.sub(r"\s+", "", text)


def chrf(hyp, ref, max_n=6, beta=2.0):
    """chrF: averaged char n-gram precision/recall, then F_beta.

    sacrebleu computes the precision and recall for each n, averages them over
    n, and only then combines them into one F score; doing it per n and
    averaging F instead is a different (slightly higher) number.
    """
    hyp, ref = norm(hyp), norm(ref)
    if not hyp or not ref:
        return 0.0
    prec_sum = rec_sum = 0.0
    for n in range(1, max_n + 1):
        h = [hyp[i:i + n] for i in range(max(0, len(hyp) - n + 1))]
        r = [ref[i:i + n] for i in range(max(0, len(ref) - n + 1))]
        if not h or not r:
            continue
        hr = {}
        for g in r:
            hr[g] = hr.get(g, 0) + 1
        match = 0
        for g in h:
            if hr.get(g, 0) > 0:
                hr[g] -= 1
                match += 1
        prec_sum += match / len(h)
        rec_sum += match / len(r)
    prec = prec_sum / max_n
    rec = rec_sum / max_n
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def scripts(text, lang):
    """(target-script chars, source-script chars) over the non-space chars."""
    body = norm(text)
    if not body:
        return 0, 0, 0
    if lang == "zh":
        tgt = len(CJK_RE.findall(body))
        src = sum(1 for c in body if c.isascii() and c.isalpha())
    else:
        tgt = sum(1 for c in body if c.isascii() and c.isalpha())
        src = len(CJK_RE.findall(body))
    return tgt, src, len(body)


def score(item, hyp):
    ref = item["reference"]
    tgt, src, n = scripts(hyp, item["tgt_lang"])
    return {
        "id": item["id"],
        "src_lang": item["src_lang"],
        "tgt_lang": item["tgt_lang"],
        "category": item.get("category", "?"),
        "hyp": hyp,
        "ref": ref,
        "chrf": round(chrf(hyp, ref), 4),
        "exact": norm(hyp) == norm(ref),
        "len_ratio": round(len(norm(hyp)) / max(1, len(norm(ref))), 3),
        "tgt_ratio": round(tgt / n, 3) if n else 0.0,
        "leak_ratio": round(src / n, 3) if n else 0.0,
        "empty": not hyp.strip(),
    }


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
               "--model", self.model, "--prompt", prompt,
               "--gen", str(gen), "--seed", str(seed)]
        if self.sampling.get("greedy"):
            cmd.append("--greedy")
        else:
            for k, v in sorted(self.sampling.items()):
                cmd += [{"temp": "--temp", "top-k": "--top-k",
                        "top-p": "--topp", "repeat-penalty": "--repeat-penalty"}[k],
                       str(v)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=env, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            return None, str(e), 0.0
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        if "Assistant: " not in out:
            return None, "no Assistant line; rc %s: %s" % (p.returncode,
                                                          err[-300:]), 0.0
        return out.split("Assistant: ", 1)[1].rstrip(chr(10)), err, (time.time() - t0) * 1000.0


def load_items(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            items.append(json.loads(line))
    return items[:limit] if limit else items


def summarize(name, rows):
    good = [r for r in rows if "error" not in r]
    s = {"model": name, "n": len(good), "errors": len(rows) - len(good)}
    if not good:
        return s
    s["chrf"] = sum(r["chrf"] for r in good) / len(good)
    s["exact"] = sum(1 for r in good if r["exact"]) / len(good)
    s["len"] = sum(r["len_ratio"] for r in good) / len(good)
    s["tgt"] = sum(r["tgt_ratio"] for r in good) / len(good)
    s["leak"] = sum(r["leak_ratio"] for r in good) / len(good)
    s["empty"] = sum(1 for r in good if r["empty"]) / len(good)
    s["ms"] = sum(r.get("ms", 0) for r in good) / len(good)
    by = {}
    for key in ("src_lang", "category"):
        d = {}
        for r in good:
            d.setdefault(r[key], []).append(r["chrf"])
        by[key] = {k: (sum(v) / len(v), len(v)) for k, v in sorted(d.items())}
    s["by"] = by
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--data", default=os.path.join(ROOT, "engine", "scripts",
                                                   "eval", "translate.jsonl"))
    ap.add_argument("--out", default="/tmp/eval_translate")
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--engine", default=os.path.join(ROOT, "engine", "engine_mm3"))
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--temp", type=float, default=None,
                    help="sample instead of greedy (sensitivity run)")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    sampling = {"greedy": 1}
    if args.temp is not None:
        sampling = {"temp": args.temp}
        for k, v in (("top-k", args.top_k), ("top-p", args.top_p),
                     ("repeat-penalty", args.repeat_penalty)):
            if v is not None:
                sampling[k] = v

    items = load_items(args.data, args.limit or None)
    if not items:
        raise SystemExit("no items in %s" % args.data)
    env = {"GPU_LAYER": "1", "KV_PACK": "16"}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    os.makedirs(args.out, exist_ok=True)
    print("engine  %s" % args.engine)
    print("items   %d from %s" % (len(items), args.data))
    summaries = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path or not os.path.exists(path):
            raise SystemExit("no such model: %r" % spec)
        print("%s== %s (%s)" % (NL, name, path), flush=True)
        eng = Engine(args.engine, path, env, args.system, sampling)
        out_path = os.path.join(args.out, name + ".items.jsonl")
        done = {}
        if args.resume and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if "error" not in r:
                        done[r["id"]] = r
        fh = open(out_path, "a" if done else "w", encoding="utf-8")
        rows = list(done.values())
        for it in items:
            if it["id"] in done:
                continue
            hyp, err, ms = eng.chat(render(it), args.gen, args.seed)
            if hyp is None:
                row = {"id": it["id"], "error": err}
            else:
                row = score(it, hyp)
                row["ms"] = round(ms, 1)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + NL)
            fh.flush()
            if not args.quiet:
                print("  [%2d/%2d] %-9s chrf %.3f %s" %
                      (len(rows), len(items), it["id"], row.get("chrf", 0),
                       "OK" if row.get("chrf", 0) > 0 else "  "), flush=True)
        fh.close()
        s = summarize(name, rows)
        with open(os.path.join(args.out, name + ".summary.json"), "w") as fh2:
            json.dump(s, fh2, ensure_ascii=False, indent=1)
        summaries.append(s)
        print("  chrF %.3f  exact %.3f  len %.2f  tgt %.2f  leak %.2f  %.0f ms" %
              (s["chrf"], s["exact"], s["len"], s["tgt"], s["leak"], s["ms"]))
        print("  by direction: %s" % "  ".join(
            "%s %.3f(%d)" % (k, v[0], v[1]) for k, v in s["by"]["src_lang"].items()))
    print("%s%-18s %4s %6s %6s %6s %6s %6s %7s" %
          (NL, "model", "n", "chrF", "exact", "len", "tgt", "leak", "ms"))
    for s in summaries:
        print("%-18s %4d %6.3f %6.3f %6.2f %6.2f %6.2f %7.0f" %
              (s["model"], s["n"], s["chrf"], s["exact"], s["len"],
               s["tgt"], s["leak"], s["ms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
