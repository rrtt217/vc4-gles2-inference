"""The eval batteries' shared half: one engine client, one JSONL loader.

`bench_mc.py`, `eval_ifeval.py` and `eval_translate.py` each carried a `class
Engine` that spawns `engine_mm3`, builds the `--chat --system --model --prompt
--gen --seed` command line, maps the sampling knobs to flags and parses the reply
out of stdout, plus two copies of the same JSONL loader and two logits/token-id
readers.  Three copies of a command line is how a flag rename silently disables
one battery, so this module is the only copy (P2-6 of the 2026-11-13 audit).

The three callers keep their scoring code, their CLI and their output files; they
only stop re-deriving the facts.

    from evalkit import Engine, load_jsonl

    eng = Engine(binary, model, env, system="You are a helpful assistant.")
    text, err, ms = eng.chat(prompt, gen=64, seed=1)      # sampled or greedy

`--gen 0` (one prefill, no decode) with `LOGITS_OUT`/`TOKENS_OUT` is `run()`, which
is what the multiple-choice battery needs: the answer is the logit row, not text.
"""

import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sampling dict keys -> engine flags.  One table: `eval_ifeval.py` used to raise on
# an unknown knob while `eval_translate.py` raised KeyError, and a third spelling
# would have disagreed with both.
SAMPLING_FLAGS = {
    "temp": "--temp",
    "top-k": "--top-k",
    "top-p": "--topp",
    "repeat-penalty": "--repeat-penalty",
}


def sampling_args(sampling):
    """The flag list for a sampling dict; `{"greedy": True}` is `--greedy`."""
    if not sampling:
        return []
    if sampling.get("greedy"):
        return ["--greedy"]
    out = []
    for key in sorted(sampling):
        if key not in SAMPLING_FLAGS:
            raise SystemExit("unknown sampling knob %r (known: %s)"
                             % (key, ", ".join(sorted(SAMPLING_FLAGS))))
        out += [SAMPLING_FLAGS[key], str(sampling[key])]
    return out


# The RWKV-7 0.1B model card's recommended sampling (fla-hub/rwkv7-0.1B-g1:
# temperature 1.0, top_p 0.3, repetition_penalty 1.2, do_sample).  The repo's
# generation_config.json carries only bos/eos, so the card is the source.  One
# copy: eval_ifeval's PRESETS["rwkv"] and eval_translate's --preset rwkv both
# read it, and the batteries' RWKV rows are scored with it, not greedy.
RWKV_SAMPLING = {"temp": 1.0, "top-p": 0.3, "repeat-penalty": 1.2}


def is_rwkv_container(path):
    """True when the MMCF file declares the RWKV-7 block (metadata key 16).

    Header and metadata TLVs only, never the tensors -- the same cheap probe
    the REPL's /model does.  The version alone is not the answer (a future
    block could share v4); key 16's first payload byte is the arch (1=rwkv7,
    engine/tools/mmcf_format.py RWKV_META_FMT).  A non-container or a read
    error is False: the batteries then keep their historical defaults.
    """
    import struct
    try:
        with open(path, "rb") as f:
            head = f.read(40)
            if len(head) < 40 or head[:4] != b"MMCF":
                return False
            ver, hsz, n, caps, mo, ms, doff, fsz = struct.unpack("<HHIIIIQQ", head[4:40])
            if ver < 4 or ms <= 0:
                return False
            f.seek(mo)
            meta = f.read(ms)
    except OSError:
        return False
    i = 0
    while i + 8 <= len(meta):
        key, typ, ln = struct.unpack("<HHI", meta[i:i + 8])
        if key == 16:
            return ln >= 1 and meta[i + 8] == 1
        i += 8 + ln
    return False


def read_floats(path):
    """A `LOGITS_OUT` dump: float32 values, little-endian."""
    with open(path, "rb") as f:
        raw = f.read()
    import struct
    return list(struct.unpack("<%df" % (len(raw) // 4), raw[:len(raw) // 4 * 4]))


def read_ints(path):
    """A `TOKENS_OUT` dump: int32 ids, little-endian."""
    with open(path, "rb") as f:
        raw = f.read()
    import struct
    return list(struct.unpack("<%di" % (len(raw) // 4), raw[:len(raw) // 4 * 4]))


def load_jsonl(path, limit=None):
    """The case files: one JSON object per line, blank lines and `//` ignored."""
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


class Engine:
    """One `engine_mm3` process per call: the batteries' unit of work."""

    def __init__(self, binary, model, env=None, system=None, sampling=None):
        self.binary = binary
        self.model = model
        self.env = dict(env or {})
        self.system = system
        self.sampling = dict(sampling or {})

    def _env(self, extra=None):
        env = dict(os.environ)
        env.update(self.env)
        if extra:
            env.update(extra)
        return env

    def _base_cmd(self, prompt, gen, seed):
        cmd = [self.binary, "--model", self.model, "--prompt", prompt,
               "--gen", str(gen)]
        if self.system is not None:
            cmd += ["--chat", "--system", self.system]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        return cmd + sampling_args(self.sampling)

    def chat(self, prompt, gen, seed=None):
        """(reply text, engine stderr, wall ms); text is None when there is none."""
        cmd = self._base_cmd(prompt, gen, seed)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=self._env(), cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            return None, str(e), (time.time() - t0) * 1000.0
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        if "Assistant: " not in out:
            return None, "no Assistant line; rc %s: %s" % (p.returncode, err[-300:]), \
                (time.time() - t0) * 1000.0
        # the engine prints one newline of its own after the reply
        return out.split("Assistant: ", 1)[1].rstrip("\n"), err, (time.time() - t0) * 1000.0

    def run(self, prompt, want_logits=True):
        """One prefill (`--gen 0`): (logits, ids, err, ms, rc).

        The multiple-choice battery scores the logit row at the answer position, so
        there is nothing to decode; the two dumps are written to temp files because
        the engine writes them, not stdout.
        """
        t0 = time.time()
        tag = "%d_%.6f" % (os.getpid(), t0)
        lg = "/tmp/evalkit_%s.lg" % tag
        tk = "/tmp/evalkit_%s.tk" % tag
        extra = {"TOKENS_OUT": tk}
        if want_logits:
            extra["LOGITS_OUT"] = lg
        cmd = self._base_cmd(prompt, 0, None)
        try:
            p = subprocess.run(cmd, env=self._env(extra), cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rc, err = p.returncode, p.stderr.decode("utf-8", "replace")
        except OSError as e:
            return None, None, str(e), (time.time() - t0) * 1000.0, 127
        logits = read_floats(lg) if want_logits and os.path.exists(lg) else None
        ids = read_ints(tk) if os.path.exists(tk) else None
        for q in (lg, tk):
            if os.path.exists(q):
                os.remove(q)
        return logits, ids, err, (time.time() - t0) * 1000.0, rc
