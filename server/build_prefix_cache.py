#!/usr/bin/env python3
"""Build and verify a prefix cache without touching the model weights (P1-4).

The cache covers only the chat *header* (system prompt + tool definitions), so
every conversation on this server hits it.  Nothing here needs a GPU on the
machine you run it from -- the engine does, so run it on the Pi.

    # build (header only, one tool)
    python3 server/build_prefix_cache.py build --out server/pfx_system_tools.kv \
        --tools system_status

    # verify that the server would really load it
    python3 server/build_prefix_cache.py verify --cache server/pfx_system_tools.kv

The sidecar `<cache>.json` records everything a stale cache could silently get
wrong: model identity, seq, KV packing, the exact prompt text and its token
count.  `verify` refuses a mismatch instead of loading garbage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import server as srv  # noqa: E402
import localtools  # noqa: E402


def sha256_file(path: Path, limit: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()[:16]


def header_text(system: str, tool_names: list[str]) -> str:
    """Exactly the prefix the server puts in front of every conversation."""
    tools = []
    for name in tool_names:
        spec = localtools.TOOL_SPECS.get(name)
        if spec is None:
            raise SystemExit(f"unknown tool {name!r}; have: {', '.join(localtools.TOOL_SPECS)}")
        tools.append({"type": "function", "function": spec})
    return "<|im_start|>system\n" + system + srv.render_tools(tools) + "<|im_end|>\n"


def cmd_build(a) -> int:
    engine = Path(a.engine)
    model = Path(a.model)
    tok = Path(a.tokenizer) if a.tokenizer else None
    paths = [engine, model] + ([tok] if tok else [])
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    text = header_text(a.system or srv.DEFAULT_SYSTEM, a.tools)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; pass --force to rebuild")

    env = dict(os.environ)
    env.update({
        "MODEL": str(model),
        "SEQ": str(a.seq),
        "GPU_LAYER": "1",
    })
    if tok:
        env["TOKENIZER"] = str(tok)
    if a.kv_pack:
        env["KV_PACK"] = str(a.kv_pack)
    # The engine takes a *file* for the cache prompt, not inline text.
    prompt_file = Path(str(out) + ".prompt.txt")
    prompt_file.write_text(text, encoding="utf-8")
    cmd = [str(engine), "--build-prefix-cache", str(prompt_file), str(out)]
    print("building:", " ".join(cmd[:1]), f"<{len(text)} chars>", out)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
    if r.returncode != 0 or not out.exists():
        sys.stderr.write(r.stderr[-1500:])
        return 1
    # The engine prints "CACHE_SAVED <path> tokens=N" on the build path and
    # "PREFIX_CACHE loaded N tokens" on load; this used to match only the
    # second one, so every sidecar recorded n_tokens=None.
    loaded = re.search(r"CACHE_SAVED .* tokens=(\d+)", r.stdout + r.stderr) or \
             re.search(r"PREFIX_CACHE (?:loaded|saved) (\d+)", r.stdout + r.stderr)
    n_tokens = int(loaded.group(1)) if loaded else None

    meta = {
        "cache": out.name,
        "created_at": srv.time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine_sha": sha256_file(engine),
        "model": model.name,
        "model_sha": sha256_file(model),
        "model_bytes": model.stat().st_size,
        # empty when the container carries its own tokenizer blob
        "tokenizer_sha": sha256_file(tok) if tok else "",
        "seq": a.seq,
        "kv_pack": a.kv_pack,
        "tools": a.tools,
        "system_sha": hashlib.sha256((a.system or srv.DEFAULT_SYSTEM).encode()).hexdigest()[:16],
        "prompt_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "prompt_chars": len(text),
        "n_tokens": n_tokens,
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"ok  {out}  {out.stat().st_size} bytes  tokens={n_tokens}")
    print("meta:", str(out) + ".json")
    return 0


def cmd_verify(a) -> int:
    cache = Path(a.cache)
    meta_path = Path(str(cache) + ".json")
    if not cache.exists():
        print("missing cache", cache)
        return 1
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        problems = []
        engine = Path(a.engine)
        model = Path(a.model)
        if meta.get("engine_sha") and engine.exists() and sha256_file(engine) != meta["engine_sha"]:
            problems.append("engine binary changed since the cache was built")
        if meta.get("model_sha") and model.exists() and sha256_file(model) != meta["model_sha"]:
            problems.append("model weights changed since the cache was built")
        if a.seq != meta.get("seq"):
            problems.append(f"seq {a.seq} != cached {meta.get('seq')}")
        if a.kv_pack != meta.get("kv_pack"):
            problems.append(f"kv_pack {a.kv_pack} != cached {meta.get('kv_pack')}")
        text = header_text(a.system or srv.DEFAULT_SYSTEM, a.tools or meta.get("tools", []))
        if hashlib.sha256(text.encode()).hexdigest()[:16] != meta.get("prompt_sha"):
            problems.append("prompt header differs from the cached one (system prompt or tools changed)")
        if problems:
            print("REFUSE:", "; ".join(problems))
            return 2
        print("meta ok:", {k: meta[k] for k in ("n_tokens", "seq", "kv_pack", "tools") if k in meta})

    env = dict(os.environ)
    env.update({"MODEL": a.model, "SEQ": str(a.seq), "GPU_LAYER": "1"})
    # The container embeds its tokenizer.  Only set TOKENIZER when the caller
    # names one: an empty value is not "unset", and the engine would try to load
    # the empty path instead of the embedded blob.
    if a.tokenizer:
        env["TOKENIZER"] = a.tokenizer
    if a.kv_pack:
        env["KV_PACK"] = str(a.kv_pack)
    # The template, tokenizer and rope come from the container; --template no
    # longer exists as an engine flag.
    cmd = [a.engine, "--chat", "--prompt", "hi", "--gen", "1",
           "--greedy", "--prefix-cache", str(cache)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout)
    m = re.search(r"PREFIX_CACHE loaded (\d+) tokens", r.stderr)
    if not m:
        sys.stderr.write(r.stderr[-1200:])
        print("FAIL: engine did not load the cache")
        return 1
    n = int(m.group(1))
    print(f"engine loaded {n} tokens from {cache.name}")
    return 0 if n > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "verify"])
    ap.add_argument("--engine", default=str(REPO / "engine/engine_mm3"))
    ap.add_argument("--model", default=str(REPO / "models/minimind3/minimind3.q8.mmcf"))
    ap.add_argument("--tokenizer", default="",
                    help="legacy .tok path; empty = use the blob inside the container")
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--kv-pack", type=int, default=16, choices=[0, 8, 16])
    ap.add_argument("--system", default="")
    ap.add_argument("--tools", nargs="*", default=[], help="tool names to embed in the header")
    ap.add_argument("--out", default=str(HERE / "pfx_system_tools.kv"))
    ap.add_argument("--cache", default=str(HERE / "pfx_system_tools.kv"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    return cmd_build(a) if a.action == "build" else cmd_verify(a)


if __name__ == "__main__":
    sys.exit(main())
