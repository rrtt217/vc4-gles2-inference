#!/usr/bin/env python3
"""No shader re-types a codec layout; the body lives in xformat.

The engine's fixed-point layouts (q16 two bytes, the 24-bit offset-binary, the
8-bit offset byte, the q16 pair) have one implementation each, emitted by
`x_scaled_glsl()` / `x_glsl()` / `x_codec_glsl()`.  The attention used to keep
its own copies, which is how its q16 decode sat on the *pre-6234328* rail test
(`>= 32768`) for months while xformat's used `32767.5`: on VC4 the (128,0)
pair reconstructs as 32767.99x, so the copy read the negative rail as +127.99
and the two disagreed by a full 256 q16 units.  A copy that is not *seen*
drifts.

The check is textual on purpose: it is the text that is the codec.  Every
occurrence of a layout fingerprint in `src/shaders_*.cpp` has to be either
absent or named in the registry below with its owner.  A new copy does not
fail silently; it fails here, with the line, until it moves to xformat or its
entry says why it is a different layout.

The registry entries are *weight* layouts (the packer's q4/q6 scale planes)
and one non-codec bit trick.  They read the format the packer wrote, so their
owner is docs/quantization.md, not an XFormat stream; changing one is a
packer change first.

    python3 gate/check_codecs.py [--verbose]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# A line-level registry: (file suffix, regex on the line, owner/reason).
# Keep this list short; it is the debt list of codec bodies outside xformat.
REGISTRY = [
    # Weight layouts: the packer writes them (mmcf_pack / mmcf_format), and
    # the kernels that read them are the weight format's own.  Their owner is
    # docs/quantization.md, not an XFormat stream.
    ("src/shaders_common.cpp", r"65280\.0\+c\.g\*255",
     "the q4/q6 weight scale plane (a q16 pair from the packer)"),
    ("src/shaders_q4.cpp", r"65280\.0\*15\.0",
     "the same weight scale plane with the q4 15-level factor"),
    # Non-stream bit tricks: the T3 sign-magnitude weight decode builds +-1.0
    # from sign bits, and the exp polynomial builds 2^k from its exponent.
    ("src/shaders_common.cpp", r"uintBitsToFloat\(uint\(int\(k\)\+127\)",
     "the exp poly's exponent-table trick, not a codec"),
    ("src/shaders_t3.cpp", r"uintBitsToFloat\(0x3F800000u",
     "the T3 weight decode's sign-magnitude bit-cast, not a stream"),
    # Stream layouts not yet moved: these are the next slice, and they are
    # named here so the debt is visible rather than silent.
    ("src/shaders_rwkv.cpp", r"bcWr|uintBitsToFloat\(u\)|>>24|<<24",
     "the RWKV state's bit-cast f32/bf16 (docs/rwkv7.md owns the state)"),
]

# What a re-typed body looks like.  These are layout fingerprints, not common
# numbers: a 32768 in a *grid* (KVSCALE, SCS) is not a hit because it is not
# spelled as a rail test.
FINGERPRINTS = [
    (r"65280", "the q16 pair byte reconstruction"),
    (r"8388608", "the 24-bit offset"),
    (r">= *32768(?:\.0)?", "the pre-6234328 q16 rail test"),
    (r">= *32767(?:\.5)?", "a q16 rail test (use x_q16raw_at)"),
    (r"floatBitsToUint\(", "an f32 encode"),
    (r"uintBitsToFloat\(", "an f32 decode"),
    (r"<< *24|>> *24", "byte assembly"),
    (r"int\([A-Za-z_][\w.]*\.r \* *255\.0 *\+ *0\.5\)", "a byte reconstruction"),
]


def main():
    verbose = "--verbose" in sys.argv[1:]
    files = sorted(f for f in os.listdir(SRC)
                   if f.startswith("shaders_") and f.endswith(".cpp"))
    bad = []
    for fn in files:
        path = os.path.join(SRC, fn)
        rel = "src/" + fn
        with open(path) as f:
            lines = f.readlines()
        for no, line in enumerate(lines, 1):
            if line.lstrip().startswith("//"):
                continue
            for pat, what in FINGERPRINTS:
                if not re.search(pat, line):
                    continue
                allowed = any(rel == rfile and re.search(rpat, line)
                              for rfile, rpat, _why in REGISTRY)
                if allowed:
                    if verbose:
                        print("  registered %s:%d (%s)" % (rel, no, what))
                else:
                    bad.append((rel, no, what, line.strip()))
    if bad:
        print("codecs: %d re-typed layout line(s) outside xformat" % len(bad))
        for rel, no, what, line in bad:
            print("  %s:%d  %s" % (rel, no, what))
            print("    %s" % line[:150])
        print("  the layouts live in src/xformat.cpp (x_scaled_glsl, x_glsl, "
              "x_codec_glsl);\n  if this is a different layout, add it to "
              "gate/check_codecs.py's registry with its owner")
        return 1
    print("codecs: ok (%d shader TU(s), %d registered line(s))"
          % (len(files), len(REGISTRY)))
    return 0


if __name__ == "__main__":
    sys.exit(main())