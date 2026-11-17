#!/usr/bin/env python3
"""The switch registry, parsed once, for the tools that check or render it.

`engine/src/switches.cpp` is the inventory (see the comment at the top of it and
`docs/test-hooks.md` for the rules).  Nothing here is authoritative: this module
only reads that file, so the gate can compare it with the source and the
documentation, and `gen_test_hooks.py` can render it into the document.

Parsed from the C++ rather than from `--dump-switches` on purpose: the gate
(scripts/docs_check.sh) is step 0 and must run without a compiler, a GPU or a
model.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
ROOT = os.path.dirname(ENGINE)
SRC = os.path.join(ENGINE, "src")
SWITCHES_CPP = os.path.join(SRC, "switches.cpp")

ROW = re.compile(
    r'\{"([A-Z][A-Z_0-9]+)",\s*"([^"]*)",\s*SwKind::(\w+),\s*"([^"]*)",\s*'
    r'SwStatus::(\w+),\s*SwAxis::(\w+),\s*"((?:[^"\\]|\\.)*)"\}')
RENAME = re.compile(
    r'\{"([A-Z][A-Z_0-9]+)",\s*"([^"]*)",\s*"([A-Z][A-Z_0-9]+)",\s*"([^"]*)"\}')

# The order axes are rendered in: the ones a person reaches for first, then the
# campaign axes, then the bookkeeping.
AXES = ["act", "att", "kv", "res", "ffn", "place", "probe", "dump", "q4", "rwkv",
        "prefill", "load", "tok", "gen", "prefix", "env", "retired"]

AXIS_TITLES = {
    "act": "The activation streams",
    "att": "The attention block's fixed-point",
    "kv": "The KV cache",
    "res": "The residual stream",
    "ffn": "The FFN hidden stream and kernel shape",
    "place": "Placement: which path runs",
    "probe": "Cost attribution",
    "dump": "Diagnostics: what the engine dumps",
    "q4": "The 4-bit weight kernels",
    "rwkv": "RWKV-7",
    "prefill": "Prefill batching",
    "load": "Loading, memory and the container",
    "tok": "The tokenizer",
    "gen": "Sampling and generation",
    "prefix": "The disk prefix cache",
    "env": "Read from the environment",
}

# A row whose note references one of these markers has a deep dive in the
# document, and the generator links it.  The convention is one word, lowercase.
DEEP_DIVE_HEADING = "Deep dives"


class Row:
    __slots__ = ("name", "aliases", "kind", "values", "status", "axis", "note")

    def __init__(self, name, aliases, kind, values, status, axis, note):
        # the C++ spells them SwStatus::Probe / SwAxis::Kv; the document and the
        # generators speak lowercase, so the conversion happens once, here.
        self.name, self.kind, self.values = name, kind.lower(), values
        self.status, self.axis, self.note = status.lower(), axis.lower(), note
        self.aliases = [a for a in aliases.split(",") if a]

    @property
    def arg(self):
        """What the switch takes, for a table cell."""
        if self.kind == "flag":
            return "`=1`"
        if self.values:
            # a markdown table: the pipes inside the value set have to be escaped
            return "`" + self.values.replace("|", "\\|") + "`"
        if self.kind == "int":
            return "n"
        if self.kind == "num":
            return "N"
        return "value"

    def md(self):
        """One table row: name (+aliases) | takes | what it does."""
        names = "`" + self.name + "`"
        if self.aliases:
            names += " <br><sub>" + ", ".join("`" + a + "`" for a in self.aliases) + "</sub>"
        return "| %s | %s | %s |" % (names, self.arg, self.note)


def load():
    """(rows in registry order, renames) from engine/src/switches.cpp."""
    text = open(SWITCHES_CPP, encoding="utf-8").read()
    rows = [Row(*m.groups()) for m in ROW.finditer(text)]
    renames = [m.groups() for m in RENAME.finditer(text)]
    return rows, renames


def reads():
    """Every switch name the engine's source reads, with one location each.

    Both spellings count: getenv() directly, the env_* helpers in common.h, and
    the registry's own accessors."""
    # Two ways a switch name reaches the environment: the known readers, and a
    # local helper that takes the name as its first argument (`dup_env("PASS_DUP",
    # name)` in session.cpp is how two switches stayed out of every inventory
    # until 2026-11-17).  The second pattern is deliberately wider than it needs
    # to be: a false positive here is a two-second read, a false negative is a
    # switch nobody knows about.
    # `env_zero` and `dup_env` were added on 2026-11-17: five switches
    # (FFN_FUSE, FFN_PAIR, QKV_PAIR, MATVEC_DUP, PASS_DUP) were read through
    # them and were invisible to every inventory this project had, including
    # the grep in AGENTS.md and the doc check that replaced it.
    pat = re.compile(r'(?:getenv|env_set|env_zero|env_flag|env_int|env_double|'
                     r'sw_isset|sw_flag|sw_int|sw_num|sw_str|sw_enum|dup_env)'
                     r'\(\s*"([A-Z_0-9]+)"')
    # Anything else that takes an ALL-CAPS literal as its first argument is
    # *reported* by suspicious(), not failed: `layer_finish("NF_ATT")` and
    # `renderer_.find("VC4")` are labels, and `setenv("MESA_...")` is a write.
    found = {}
    for dirpath, _, files in os.walk(SRC):
        for f in sorted(files):
            if not f.endswith((".cpp", ".h")):
                continue
            p = os.path.join(dirpath, f)
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                code = line.split("//")[0]
                for m in pat.finditer(code):
                    found.setdefault(m.group(1), []).append(
                        "%s:%d" % (os.path.relpath(p, ENGINE), i))
    return found


def suspicious(known):
    """ALL-CAPS literals passed to some call we do not recognise as a reader.

    A hint for the next local helper, not a verdict: see the comment in reads().
    """
    pat = re.compile(r'\b[a-z_][a-z_0-9]*\(\s*"([A-Z][A-Z_0-9]{2,})"')
    out = {}
    for dirpath, _, files in os.walk(SRC):
        for f in sorted(files):
            if not f.endswith((".cpp", ".h")):
                continue
            p = os.path.join(dirpath, f)
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                code = line.split("//")[0]
                for m in pat.finditer(code):
                    if m.group(1) in known:
                        continue
                    out.setdefault(m.group(1), "%s:%d" % (os.path.relpath(p, ENGINE), i))
    return out


def by_axis(rows):
    out = {a: [] for a in AXES}
    for r in rows:
        out.setdefault(r.axis, []).append(r)
    for a in out:
        out[a].sort(key=lambda r: (r.status != "permanent", r.name))
    return out
