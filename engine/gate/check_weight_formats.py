#!/usr/bin/env python3
"""The weight-format contract against the enum and the packer's dtype table.

One question: does every WeightType have a kWeightFormats row, and does
wf_from_dtype cover exactly the packer's matrix dtypes (f32 and blob are not
matrix bodies)?

The compiled test tests/test_weight_format.cpp owns the table's internal
invariants (caps, slice alignment, byte arithmetic); this gate is the cross-file
half: common.h's enum, weight_format.h's rows and wire cases, and
tools/mmcf_format.py's DTYPES.  A format added to one and not the others is the
drift this catches -- the loader would refuse a body the packer writes, or a row
would never be reachable.

usage: engine/gate/check_weight_formats.py [-v]     (no GPU, no model)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
SRC = os.path.join(ENGINE, "src")
sys.path.insert(0, os.path.join(ENGINE, "tools"))

import mmcf_format as fmt  # noqa: E402  (the tool module, stdlib-only)

# A dtype that is not a Mat body: 0 is a norm/embedding vector, 5 the tokenizer.
NON_MATRIX_DTYPES = {fmt.DT_F32, fmt.DT_BLOB}
# The container's name for a matrix body -> the WeightType that reads it.  q8u8
# and q8i8 are one kernel family; the difference is whether the +128 bias is in
# the bytes (the prebiased flag wf_from_dtype returns).
DTYPE_FAMILY = {
    "i16": "INT16",
    "q8u8": "INT8",
    "q8i8": "INT8",
    "q4": "INT4",
    "q6": "INT6",
    "t3": "T3",
}
# ... and the name its kWeightFormats row uses.
DTYPE_ROW = {
    "i16": "i16",
    "q8u8": "q8",
    "q8i8": "q8",
    "q4": "q4",
    "q6": "q6",
    "t3": "t3",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def enum_members(text, enum):
    m = re.search(r"enum class %s\s*(?::\s*\w+\s*)?\{(.*?)\}" % enum, text, re.S)
    if not m:
        return None
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def main():
    verbose = "-v" in sys.argv
    common = read(os.path.join(SRC, "common.h"))
    wf = read(os.path.join(SRC, "weight_format.h"))
    fail = 0

    def bad(msg):
        nonlocal fail
        fail = 1
        print("FAIL  " + msg)

    types = enum_members(common, "WeightType")
    stages = enum_members(wf, "WStage")
    if not types:
        bad("common.h: no WeightType enum")
        types = []
    if not stages:
        bad("weight_format.h: no WStage enum")

    rows = re.findall(r'\{\s*WeightType::(\w+)\s*,\s*"([^"]+)"', wf)
    row_type = [r[0] for r in rows]
    row_name = [r[1] for r in rows]
    if len(rows) != len(types):
        bad("weight_format.h: %d kWeightFormats rows for %d WeightType members"
            % (len(rows), len(types)))
    for i, t in enumerate(types):
        if i >= len(row_type):
            bad("WeightType::%s has no kWeightFormats row" % t)
        elif row_type[i] != t:
            bad("kWeightFormats[%d] is %s, enum order says %s" % (i, row_type[i], t))
    if len(set(row_type)) != len(row_type):
        bad("kWeightFormats names a WeightType twice")
    if len(set(row_name)) != len(row_name):
        bad("kWeightFormats has two rows with the same name")
    type_to_name = dict(zip(row_type, row_name))

    # The split gave every format one TU: the row and the file must agree, or
    # the table claims a format whose kernels live somewhere else (or nowhere).
    for name in row_name:
        path = os.path.join(ENGINE, "src", "shaders_%s.cpp" % name)
        if not os.path.isfile(path):
            bad("WeightFormat %r has no src/shaders_%s.cpp" % (name, name))

    # The wire mapping: one case per matrix dtype, no case for f32/blob, and the
    # type each case names is the row the packer's name belongs to.
    cases = {}
    for d, t, pb in re.findall(
            r'case\s+(\d+):\s*\*out\s*=\s*WeightType::(\w+);\s*'
            r'\*prebiased\s*=\s*(true|false);', wf):
        cases[int(d)] = (t, pb == "true")
    matrix_dtypes = set(fmt.DTYPES) - NON_MATRIX_DTYPES
    if set(cases) != matrix_dtypes:
        bad("wf_from_dtype cases %s, the packer's matrix dtypes are %s"
            % (sorted(cases), sorted(matrix_dtypes)))
    for d in sorted(matrix_dtypes):
        if d not in cases:
            continue
        name = fmt.DTYPES[d]
        want = DTYPE_FAMILY.get(name)
        if want is None:
            bad("mmcf_format dtype %d is %r, which has no family rule here" % (d, name))
        elif cases[d][0] != want:
            bad("wf_from_dtype(%d) is %s; %r is read by %s"
                % (d, cases[d][0], name, want))
        elif type_to_name.get(want) != DTYPE_ROW[name]:
            bad("row %s is named %r, the container calls that body %r"
                % (want, type_to_name.get(want), DTYPE_ROW[name]))

    # q8u8 is the biased spelling and q8i8 the signed one; a swap here would
    # make the upload add 128 twice or not at all.
    for d, biased in ((fmt.DT_Q8U8, True), (fmt.DT_Q8I8, False)):
        if d in cases and cases[d][1] != biased:
            bad("wf_from_dtype(%d) prebiased=%s, want %s"
                % (d, cases[d][1], biased))

    if verbose:
        print("WeightType: %s" % ", ".join(types))
        print("rows: %s" % ", ".join("%s=%s" % (t, n) for t, n in zip(row_type, row_name)))
        print("wire: %s" % ", ".join("%d->%s" % (d, cases[d][0]) for d in sorted(cases)))

    if fail:
        return 1
    print("weight formats: ok (%d types, %d matrix dtypes)"
          % (len(types), len(matrix_dtypes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())