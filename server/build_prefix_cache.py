#!/usr/bin/env python3
"""Trigger and check a prefix cache.  The engine builds it.

The cache is a chat *header* -- the system prompt plus the tool definitions --
turned into KV, so every conversation begins from it instead of prefilling those
tokens again (417 tokens to 30 on the measurements in docs/prefix-cache.md).

Who does what, and why it moved here
------------------------------------
The engine owns the file.  `--build-prefix-cache - OUT` renders the header with
the container's own template, prefills it, writes the KV and writes
`<cache>.json` describing itself: container size and mtime, the full shape, the
packing, the scale, the token count, a fingerprint of the tokens, and 64-bit
hashes of the system prompt and the tool list it was built from.

This script only decides *when* to build, names the file, and compares the
sidecar with the current settings before a server loads it.  It used to render
the header itself -- in Python, hard-coded to ChatML -- which is wrong for every
container whose template is not ChatML, and which went dead the day
`render_tools()` was deleted with "the engine is the only renderer".  Rendering
belongs where the template is.

    # build on the machine that will serve it (the KV bytes are machine specific)
    python3 server/build_prefix_cache.py build \
        --model models/minimind3/minimind3.q8.mmcf --seq 768 --kv-pack 16 \
        --system "You are a helpful assistant." \
        --tools system_status get_current_time \
        --out server/pfx_system_tools_pi768.kv

    # before a server run: does this file still describe these settings?
    python3 server/build_prefix_cache.py verify \
        --cache server/pfx_system_tools_pi768.kv --seq 768 --kv-pack 16

`verify` refuses a cache whose sidecar disagrees with the model on disk (size or
mtime), the SEQ, the packing, or the system/tool inputs, and then proves the
engine really loads it (`PREFIX_CACHE loaded N tokens`).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import localtools  # noqa: E402  (the tool specs are the input to the header)


def fnv64(text: str) -> str:
    """The engine's sidecar hash: 64-bit FNV-1a, hex.  Not cryptographic."""
    h = 1469598103934665603
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def tools_text(names):
    """The tools array exactly as the server would send it (same bytes hashed)."""
    if not names:
        return ""
    specs = []
    for name in names:
        spec = localtools.TOOL_SPECS.get(name)
        if spec is None:
            raise SystemExit("unknown tool %r; have: %s"
                             % (name, ", ".join(sorted(localtools.TOOL_SPECS))))
        specs.append({"type": "function", "function": spec})
    return json.dumps(specs, ensure_ascii=False, separators=(",", ":"))


def sidecar(cache: Path) -> dict:
    path = Path(str(cache) + ".json")
    if not path.exists():
        raise SystemExit("no sidecar next to %s: rebuild it with this script "
                         "(the engine writes one)" % cache)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit("sidecar %s is not readable: %s" % (path, e))


def run_engine(a, extra_args, env_extra, stdin=""):
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [str(a.engine)] + extra_args
    return subprocess.run(cmd, env=env, cwd=str(REPO), input=stdin,
                          capture_output=True, text=True, timeout=a.timeout)


def cmd_build(a) -> int:
    engine, model = Path(a.engine), Path(a.model)
    for p in (engine, model):
        if not p.exists():
            raise SystemExit("missing: %s" % p)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit("%s exists; pass --force to rebuild" % out)

    # `-` means "render the header yourself, with the container's template".
    header_env = {"MODEL": str(model), "SEQ": str(a.seq), "KV_PACK": str(a.kv_pack),
                  "GPU_LAYER": "1"}
    args = ["--build-prefix-cache", "-", str(out), "--system", a.system]
    text = tools_text(a.tools)
    tmp = None
    try:
        if text:
            fd, tmp = tempfile.mkstemp(prefix="pfx_tools_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            args += ["--tools-file", tmp]
        r = run_engine(a, args, header_env)
    finally:
        if tmp:
            os.unlink(tmp)
    saved = [l for l in r.stdout.splitlines() if l.startswith("CACHE_SAVED")]
    header = [l for l in r.stderr.splitlines() if l.startswith("CACHE_HEADER")]
    if not saved:
        sys.stderr.write(r.stdout + r.stderr)
        return 1
    for line in header + saved:
        print(line)
    meta = sidecar(out)
    print("sidecar  %s: template=%s tokens=%s seq=%s kv=%s"
          % (out.name + ".json", meta.get("template"), meta.get("n_tokens"),
             meta.get("seq"), meta.get("kv_bits")))
    print("system_fnv64=%s tools_fnv64=%s tokens_fnv64=%s"
          % (meta.get("system_fnv64"), meta.get("tools_fnv64"),
             meta.get("tokens_fnv64")))
    return 0


def cmd_verify(a) -> int:
    cache = Path(a.cache)
    if not cache.exists():
        raise SystemExit("missing: %s" % cache)
    meta = sidecar(cache)
    model = Path(a.model)
    problems = []

    def want(name, got, expect):
        if got != expect:
            problems.append("%s: cache %r, settings %r" % (name, got, expect))

    # Only what the engine cannot check at load time is compared here: the inputs
    # the header was built from.  Everything else -- the container's size and
    # mtime, the shape, SEQ, the packing, the scale -- is in the KV header and the
    # engine refuses a file whose header disagrees, so repeating that policy in
    # Python would be a second copy of it (and the effective SEQ is not even the
    # requested one: a tool list raises it to at least 768).
    want("system_fnv64", meta.get("system_fnv64"), fnv64(a.system))
    want("tools_fnv64", meta.get("tools_fnv64"), fnv64(tools_text(a.tools)))
    if model.exists():
        st = model.stat()
        want("source_size", meta.get("source_size"), st.st_size)
        want("source_mtime", meta.get("source_mtime"), int(st.st_mtime))

    for p in problems:
        print("MISMATCH %s" % p)
    if problems:
        print("REFUSE  %s does not describe these settings; rebuild it" % cache)
        return 1

    # The header checked out; now prove the engine loads it.  A one-token request
    # is enough: the cache is read before `ready`.
    line = "\t1\t0\t1\t1\t\t\t\t\t\t\n"
    env = {"MODEL": str(a.model), "SEQ": str(a.seq), "KV_PACK": str(a.kv_pack),
           "GPU_LAYER": "1"}
    r = run_engine(a, ["--worker", "--prefix-cache", str(cache)], env, stdin=line)
    loaded = [l for l in r.stderr.splitlines() if l.startswith("PREFIX_CACHE")]
    if not any("tokens" in l for l in loaded):
        print("REFUSE  the engine did not load it: %s" % (loaded or "(no line)"))
        return 1
    print("OK      %s (%s tokens, template %s)"
          % (cache, meta.get("n_tokens"), meta.get("template")))
    for l in loaded:
        print("        " + l)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "verify"])
    ap.add_argument("--engine", default=str(REPO / "engine/engine_mm3"))
    ap.add_argument("--model", default=str(REPO / "models/minimind3/minimind3.q8.mmcf"))
    ap.add_argument("--tokenizer", default="", help="unused; the container carries it")
    ap.add_argument("--seq", type=int, default=768)
    ap.add_argument("--kv-pack", type=int, default=16, choices=[0, 8, 16])
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--tools", nargs="*", default=[], help="tool names in the header")
    ap.add_argument("--out", default=str(HERE / "pfx_system_tools.kv"))
    ap.add_argument("--cache", default=str(HERE / "pfx_system_tools.kv"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    return cmd_build(a) if a.action == "build" else cmd_verify(a)


if __name__ == "__main__":
    sys.exit(main())
