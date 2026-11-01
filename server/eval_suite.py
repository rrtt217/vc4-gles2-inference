#!/usr/bin/env python3
"""Fixed evaluation suite for the MiniMind3 GLES2 server (backlog P1-5).

Single-prompt impressions are not evidence.  This runs a fixed set of cases,
prints a table, and writes JSON so two runs can be diffed.

    python3 server/eval_suite.py --base http://127.0.0.1:8011 --out eval.json

Marks are deliberately strict: a case passes only when the assertion holds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

RESULTS = []


def http(base, method, path, body=None, timeout=240):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            dt = time.time() - t0
            try:
                return r.status, json.loads(raw), dt, raw
            except json.JSONDecodeError:
                return r.status, raw, dt, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        dt = time.time() - t0
        try:
            return e.code, json.loads(raw), dt, raw
        except json.JSONDecodeError:
            return e.code, raw, dt, raw
    except Exception as e:  # noqa: BLE001 - network errors are results too
        return 0, {"error": str(e)}, time.time() - t0, ""


def cma_free_kb():
    """CmaFree is the real GPU-memory signal on this board; absent elsewhere."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("CmaFree:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def record(name, ok, detail="", seconds=None, extra=None):
    RESULTS.append({"case": name, "ok": bool(ok), "detail": str(detail)[:400],
                    "seconds": round(seconds, 2) if seconds is not None else None,
                    "extra": extra or {}})
    mark = "PASS" if ok else "FAIL"
    t = f"{seconds:6.2f}s " if seconds is not None else "        "
    print(f"  {mark}  {t}{name}  {'' if ok else '-> ' + str(detail)[:160]}")


def text_of(resp):
    try:
        return resp["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8011")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-model", action="store_true", help="only run deterministic cases")
    ap.add_argument("--model", default="minimind3-64m")
    ap.add_argument("--baseline", default="", help="previous eval JSON to diff against")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    cma_start = cma_free_kb()

    print(f"== liveness  {base}")
    code, body, dt, _ = http(base, "GET", "/health", timeout=15)
    record("health", code == 200 and isinstance(body, dict) and body.get("status") == "ok", body, dt)

    print("== deterministic routes (no model)")
    code, body, dt, _ = http(base, "GET", "/status", timeout=15)
    record("/status returns host", code == 200 and body.get("ok") and body["result"].get("host"), body, dt)
    code, body, dt, _ = http(base, "GET", "/time", timeout=15)
    record("/time returns iso", code == 200 and re.match(r"\d{4}-\d\d-\d\dT", str(body.get("result", {}).get("iso", ""))), body, dt)
    code, body, dt, _ = http(base, "GET", "/math?expr=256*37", timeout=15)
    record("/math 256*37 == 9472", code == 200 and body.get("result", {}).get("result") == 9472, body, dt)
    code, body, dt, _ = http(base, "GET", "/math?expr=1/0", timeout=15)
    record("/math rejects 1/0", code == 400 and body.get("ok") is False, body, dt)
    code, body, dt, _ = http(base, "POST", "/v1/tool",
                             {"name": "get_status_task", "arguments": {}}, timeout=15)
    record("/v1/tool alias -> system_status", code == 200 and body.get("tool") == "system_status", body, dt)
    code, body, dt, _ = http(base, "POST", "/v1/tool", {"name": "network_info"}, timeout=15)
    record("network_info has interfaces", code == 200 and "interfaces" in body.get("result", {}), body, dt)
    code, body, dt, _ = http(base, "POST", "/v1/tool", {"name": "df", "arguments": {}}, timeout=15)
    record("disk_usage alias -> usage", code == 200 and body["result"]["total_mb"] > 0, body, dt)

    print("== deterministic chat routing")
    tool = {"type": "function", "function": {
        "name": "calculate_math",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}}
    body_req = {"model": a.model, "messages": [{"role": "user", "content": "256*37"}],
                "tools": [tool], "tool_choice": "calculate_math"}
    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", body_req)
    tr = resp.get("tool_result", {}) if isinstance(resp, dict) else {}
    record("named tool_choice -> 9472", code == 200 and tr.get("result", {}).get("result") == 9472,
           resp, dt, {"no_model": True})

    if a.skip_model:
        return finish(a, cma_start)

    print("== model path")
    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": "你叫什么名字？"}],
        "max_tokens": 40, "temperature": 0})
    txt = text_of(resp)
    record("identity mentions MiniMind", "minimind" in txt.lower(), txt, dt, {"text": txt})

    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": "用中文回答：1+1等于多少？"}],
        "max_tokens": 16, "temperature": 0})
    txt = text_of(resp)
    record("natural-language arithmetic contains 2", "2" in txt, txt, dt, {"text": txt})

    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", {
        "model": a.model,
        "messages": [{"role": "user", "content": "我在做一个树莓派项目。"},
                     {"role": "assistant", "content": "听起来不错。"},
                     {"role": "user", "content": "我该注意什么？"}],
        "max_tokens": 16, "temperature": 0})
    txt = text_of(resp)
    record("multi-turn produces text", len(txt.strip()) > 0, txt, dt, {"text": txt})

    filler = "树莓派是一块很小的电脑板。" * 20
    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", {
        "model": a.model,
        "messages": [{"role": "user", "content": filler + "\n只回答两个字：收到"}],
        "max_tokens": 8, "temperature": 0})
    txt = text_of(resp)
    usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
    record("long prompt survives", code == 200 and len(txt.strip()) > 0, txt, dt,
           {"prompt_tokens": usage.get("prompt_tokens")})

    code, resp, dt, _ = http(base, "POST", "/v1/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": "说一句话"}],
        "max_tokens": 8, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}})
    raw = resp if isinstance(resp, str) else json.dumps(resp)
    record("streaming has [DONE] + usage", "data: [DONE]" in raw and '"usage"' in raw,
           raw[-200:], dt)

    print("== validation")
    req = urllib.request.Request(base + "/v1/chat/completions", data=b"{not json",
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
        record("malformed body rejected", False, "accepted")
    except urllib.error.HTTPError as e:
        record("malformed body rejected", e.code == 400, f"http={e.code}")

    return finish(a, cma_start)


def finish(a, cma_start=None):
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["ok"])
    print(f"\n{passed}/{total} passed")
    slow = [r for r in RESULTS if r["seconds"]]
    if slow:
        print("slowest:", ", ".join(f"{r['case']} {r['seconds']}s"
                                    for r in sorted(slow, key=lambda r: -r["seconds"])[:3]))
    cma_end = cma_free_kb()
    if cma_start is not None and cma_end is not None:
        print(f"CmaFree: {cma_start} KB before -> {cma_end} KB after")
    if a.baseline:
        try:
            prev = json.load(open(a.baseline, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"baseline unreadable ({exc}); skipping diff")
        else:
            old = {c["case"]: c for c in prev.get("cases", [])}
            newly_failed = [r["case"] for r in RESULTS if not r["ok"] and old.get(r["case"], {}).get("ok")]
            fixed = [r["case"] for r in RESULTS if r["ok"] and old.get(r["case"]) and not old[r["case"]]["ok"]]
            slower = []
            for r in RESULTS:
                o = old.get(r["case"])
                if o and r["seconds"] and o.get("seconds"):
                    # Sub-second timings are dominated by scheduling noise.
                    if r["seconds"] < 1.0 and o["seconds"] < 1.0:
                        continue
                    ratio = r["seconds"] / o["seconds"]
                    if ratio > 1.3:
                        slower.append((r["case"], o["seconds"], r["seconds"], ratio))
            print(f"diff vs {a.baseline}: {len(newly_failed)} newly failing, {len(fixed)} fixed, "
                  f"{len(slower)} slower by >30%")
            for case in newly_failed:
                print("  REGRESSED:", case)
            for case in fixed:
                print("  FIXED:", case)
            for case, before, after, ratio in slower:
                print(f"  SLOWER: {case} {before}s -> {after}s ({ratio:.2f}x)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"base": a.base, "passed": passed, "total": total,
                       "cma_free_kb_start": cma_start, "cma_free_kb_end": cma_end,
                       "cases": RESULTS},
                      f, ensure_ascii=False, indent=1)
        print("wrote", a.out)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
