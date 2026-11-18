#!/bin/sh
# Acceptance suite for the GPU engine: determinism, behaviour, golden strings
# (as a tripwire) and perplexity.
#
# Usage:
#   scripts/regression.sh [engine_binary]
#   scripts/regression.sh --rebaseline [engine_binary]
#
# Why not a single golden string: the greedy text is chaotic.  A deliberate
# reassociation (the fused FFN, for one) moves it on VC4 while the same tree
# still prints the old string on Xe, and a 64M model at temp 0.8 gives a
# different sample for the same weights.  So the suite checks in this order:
#
#   0. docs          - the documents still describe this tree: every switch the
#                      source reads is in docs/test-hooks.md, every in-tree path
#                      a document names exists, and no document calls a live
#                      switch dead.  gate/docs_check.sh, no GPU needed; a
#                      drift here is a real defect (a stale doc sends the next
#                      reader at the wrong thing), so it fails the gate.
#   0b. shaders      - the emitted GLSL *is* the numerics, so a builder change
#                      that moves it must move gate/shader_hashes.txt in the
#                      same commit.  --dump-shaders returns before any EGL call,
#                      so this needs no GPU and no model (~0.03 s).
#   0c. target reset - every scratch Target in gpu.h is cleared by
#                      GPU::reset_targets().  A Target that survives a scene
#                      release is *reused* with a deleted name (silent garbage),
#                      so the ledger has to stay complete.  No GPU, no model.
#   0e. draw cap    - the 1024-fragment draw limit is kDrawCap, not a bare number
#                      in gpu*.cpp; a wrong width fails silently (no GPU, no model)
#   0f. layout       - every container under models/ round-trips through
#                      tools/mmcf_format.py and the C++ constants agree (no GPU)
#
#   0d. q16 decode   - every q16 decode in the emitted GLSL folds the negative
#                      rail with one of the two documented thresholds (32767.5
#                      and 32768.0, equivalent because a decoded value is always
#                      a multiple of 255).  Ten spellings share the rule; this
#                      catches a new one or a dropped fold.  No GPU, no model.
#   1. determinism   - the same binary, run twice, prints the same text.  A
#                      failure here is a real bug (uninitialised state, a race,
#                      a stale cache), never "noise".
#   2. behaviour     - a second prompt must produce a plausible answer: non
#                      empty, long enough, no collapse into one repeated chunk.
#   3. golden        - the identity answer is compared against Q8_EXPECT /
#                      Q4_EXPECT when those are set.  They are pinned per
#                      machine (Pi Zero and Xe differ); an empty value only
#                      prints the string.  A deliberate numeric change
#                      re-baselines them in the same commit, with PPL and the
#                      behaviour checks as evidence - see AGENTS.md.
#   3b. renderer     - a software rasteriser (llvmpipe in a sandbox without
#                      /dev/dri) is not a machine the pins apply to.  The engine
#                      says so on stderr and the gate stops: a green ACCEPT from a
#                      software run would be a lie (2026-11-15).
#
#   4. ppl           - the q8 baseline and the q4 baseline are compared against
#                      PPL_EXPECT / Q4_PPL_EXPECT (default 1% tolerance) when
#                      set.  The ACT8=2/ACT8=1 runs are a diagnostic A/B, not
#                      part of the gate: they only run with ACT_PPL_DIAG=1.
#
#   RWKV_GPU selects a different layer loop and so changes the answers.  This
#   script holds no RWKV pins: RWKV_EXPECT / RWKV_PPL_EXPECT come from the
#   environment, so whoever exports them decides which loop they describe.  The
#   run does not refuse RWKV_GPU; it records the loop it used in the manifest so
#   a result is not misread as covering the other one.
#
# Env:
#   SYS          system prompt to pin (default: You are a helpful assistant.)
#   Q8_EXPECT    expected q8 prefix; empty = print only
#   Q4_EXPECT    expected q4 prefix; empty = print only
#   PPL_EXPECT   expected baseline q8 PPL; empty = print only
#   Q4_PPL_EXPECT expected q4 PPL (same SEQ); empty = print only
#   MOE          a MoE container (v3) to check as well; empty = skip.  It gets
#                the determinism and behaviour checks and, when MOE_EXPECT is
#                set, the same prefix pin.  No PPL by default: the q4 MoE runs
#                at ~340 ms/token on a Pi Zero, so it is opt-in via MOE_PPL=1.
#   MOE_EXPECT   expected MoE prefix; empty = print only
#   MOE_ROUTES   file with the expected MOE_DUMP route lines (gate values
#                stripped) for the same prompt; empty = print only.  Routes
#                are a discrete decision and can move with the driver/GPU, so
#                this is pinned per machine like the text strings.  Generate
#                it from a known-good tree, see the MoE block below.
#   MOE_PPL      compute the MoE container's PPL as well (=1; slow on the Pi)
#   MOE_PPL_EXPECT expected MoE PPL; empty = print only
#   RWKV         an RWKV-7 container (v4) to check as well; defaults to the g1d
#                0.1B q8r chat container when it exists, RWKV= (empty) or a
#                missing file skips it.  It gets determinism, the default-system
#                check, the greedy prefix pin (RWKV_EXPECT) and PPL.  The reply
#                is multi-line (g1d thinks out loud), so it is captured whole,
#                not head -1 like the dense strings.
#   RWKV_EXPECT  expected RWKV prefix; empty = print only
#   RWKV_PPL_EXPECT expected RWKV PPL; empty = print only
#   RWKV_PPL_SEQ SEQ for the RWKV PPL run (default 512, the container's own;
#                the dense PPL_SEQ does not apply -- PPL is not comparable
#                across SEQ)
#   PPL_TOL      relative tolerance for both (default 0.01)
#   PPL_FILE     corpus (default data/ppl_test.txt)
#   PPL_SEQ      SEQ used for the ppl runs (default 768; values are not
#                comparable across SEQ)
#   NO_PPL=1     skip the perplexity pass (it is the slow part)
#   ACT_PPL_DIAG=1  ALSO run the ACT8=2/ACT8=1 perplexity A/B.  Printed only,
#                never compared: it is a diagnostic, not an acceptance check.
set -eu
cd "$(dirname "$0")/.."

REBASE=0
if [ "${1:-}" = "--rebaseline" ]; then REBASE=1; shift; fi
# A gate that passes with nothing to compare is worse than no gate: it is
# counted as coverage (2026-11-17, an outside review of this design).  So an
# unset pin is a failure, and the only way past it is to say so out loud.
ALLOW_UNPINNED="${ALLOW_UNPINNED:-0}"

BIN="${1:-${BIN:-./engine_mm3}}"
TOK="${TOKENIZER:-}"   # optional: the container embeds its tokenizer
if [ -n "$TOK" ]; then export TOKENIZER="$TOK"; fi
# RWKV_GPU picks a different layer loop, so it changes the answers RWKV_EXPECT /
# RWKV_PPL_EXPECT are compared with.  Those pins are not in this script (it only
# reads the environment); whoever sets them owns which path they describe.  So do
# not refuse the run -- record which loop it was, and let the pins decide.  A
# mismatch under RWKV_GPU is a pin problem, not evidence about the GPU path.
RWKV_LOOP="cpu"
if [ "${RWKV_GPU:-0}" != 0 ] && [ "${RWKV_GPU:-0}" != "" ]; then
    RWKV_LOOP="gpu"
    echo "note: RWKV_GPU is set; the rwkv checks below compare against the pins" >&2
    echo "      you exported, which must be the GPU loop's own values." >&2
fi
# The coverage manifest is printed at the end; record every skip as it happens
# so a green ACCEPT cannot be read as "every supported path was compared".
COVERED=""
SKIPPED=""
covered() { COVERED="$COVERED $1"; }
# Keep each skip as a line, not as word-split text: the reason contains spaces.
skipped() {
    SKIPPED="$SKIPPED$1($2)
"
    echo "$1: skipped ($2)"
}
Q8="${Q8:-../models/minimind3/minimind3.q8.mmcf}"
Q4="${Q4:-../models/minimind3/minimind3.q4.mmcf}"
CORPUS="${PPL_FILE:-data/ppl_test.txt}"
SYS="${SYS:-You are a helpful assistant.}"
Q8_EXPECT="${Q8_EXPECT:-}"
Q4_EXPECT="${Q4_EXPECT:-}"
PPL_EXPECT="${PPL_EXPECT:-}"
Q4_PPL_EXPECT="${Q4_PPL_EXPECT:-}"
PPL_TOL="${PPL_TOL:-0.01}"
PPL_SEQ="${PPL_SEQ:-768}"
MOE="${MOE:-}"
MOE_EXPECT="${MOE_EXPECT:-}"
MOE_ROUTES="${MOE_ROUTES:-}"
MOE_PPL="${MOE_PPL:-0}"
MOE_PPL_EXPECT="${MOE_PPL_EXPECT:-}"
# ${RWKV-...} without the colon: an explicit RWKV= is a skip, not "use the
# default".  The file check below turns a missing container into a skip too.
RWKV="${RWKV-../models/rwkv7/packed/rwkv7-g1d-0.1b.q8r.chat.mmcf}"
RWKV_EXPECT="${RWKV_EXPECT:-}"
RWKV_PPL_EXPECT="${RWKV_PPL_EXPECT:-}"
RWKV_PPL_SEQ="${RWKV_PPL_SEQ:-512}"
# --rebaseline replaces these pins, so it must not be measured against them:
# before this, a deliberate numeric change made `--rebaseline` exit at the first
# comparison and never print the values it exists to produce.
if [ "$REBASE" = 1 ]; then
    Q8_EXPECT=""; Q4_EXPECT=""; PPL_EXPECT=""; Q4_PPL_EXPECT=""
    MOE_EXPECT=""; MOE_PPL_EXPECT=""; RWKV_EXPECT=""; RWKV_PPL_EXPECT=""
fi
PROMPT="用一句话介绍你自己。"
PROMPT2="写一首关于春天的短诗。"

[ -x "$BIN" ] || { echo "missing engine: $BIN" >&2; exit 1; }
# A missing container used to come back as an empty string and then compare as
# "unpinned", which is a pass.  Say it once, here, with the path.
for c in "$Q8" "$Q4"; do
    [ -f "$c" ] || { echo "missing container: $c  (Q8=/Q4= select them)" >&2; exit 1; }
done
if [ "${NO_PPL:-0}" != 1 ]; then
    [ -f "$CORPUS" ] || { echo "missing PPL corpus: $CORPUS" >&2; exit 1; }
fi

# The engine's exit status is part of the measurement: before this the answer
# came through `engine | sed | head`, so a crash that printed no "Assistant:"
# line was an empty string and the pipeline still exited 0.  A run is kept in a
# file and its status checked before anything judges the text (fail closed).
# $1 the file to capture into, $2 model, $3 prompt, $4 gen, $5 system ("" = the
# container's default prompt)
answer_to() {
    dst="$1" model="$2" prompt="$3" gen="$4" system="$5"
    if [ -n "$system" ]; then
        MODEL="$model" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat --system "$system" \
            --prompt "$prompt" --gen "$gen" --greedy > "$dst" 2>&1
    else
        MODEL="$model" GPU_LAYER=1 KV_PACK=16 \
            "$BIN" --chat \
            --prompt "$prompt" --gen "$gen" --greedy > "$dst" 2>&1
    fi
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "engine FAILED (rc=$rc): model=$model gen=$gen system=${system:-<container default>}" >&2
        sed 's/^/  /' "$dst" >&2
        return 1
    fi
    return 0
}

# The engine appends its "[stopped at the --gen limit ...]" note to the same
# line, so a run that does not emit EOS would otherwise fail the terminator
# check; strip it before anything judges the text.
text() {
    f=$(mktemp "${TMPDIR:-/tmp}/regression.XXXXXX") || return 1
    if ! answer_to "$f" "$@"; then rm -f "$f"; return 1; fi
    sed -n 's/^Assistant: //p' "$f" | head -1 | sed 's/\[stopped.*//'
    rm -f "$f"
}

# The RWKV reply spans lines (g1d answers with a leading marker line and then
# prose, in either template mode), so keep everything from the Assistant marker
# to the end, strip the marker on the first line and the gen-limit note wherever
# it landed.
rwkv_text() {
    f=$(mktemp "${TMPDIR:-/tmp}/regression.XXXXXX") || return 1
    if ! answer_to "$f" "$@"; then rm -f "$f"; return 1; fi
    sed -n '/^Assistant: /,$p' "$f" | sed '1s/^Assistant: //' | sed 's/\[stopped.*//'
    rm -f "$f"
}

# A software rasteriser invalidates every pin below it, so stop before them.
# The stderr of this probe is the evidence for the verdict, so it is captured
# rather than grepped out of a pipeline: a pipeline here also hid the engine's
# own failure (a broken binary simply did not match the string).
probe=$(mktemp "${TMPDIR:-/tmp}/regression.XXXXXX") || exit 1
MODEL="$Q8" GPU_LAYER=1 KV_PACK=16 "$BIN" --prompt x --gen 1 >/dev/null 2>"$probe" || true
if grep -q "software rasteriser" "$probe"; then
    rm -f "$probe"
    echo "FAIL  the renderer is a software rasteriser (no /dev/dri?); pins and PPL" >&2
    echo "      do not apply to it, so this run cannot accept the tree.  Use the board" >&2
    echo "      or a machine with a GPU, or the no-GPU half: make -C engine check." >&2
    exit 1
fi
# The probe's exit status is otherwise not a gate input: a binary that cannot
# run this at all fails on the determinism text below, with a better message.
rm -f "$probe"

# $1 tag, $2 expected prefix, $3 got, $4 the environment variable that holds it
check_prefix() {
    tag="$1" want="$2" got="$3" var="${4:-<pin>}"
    if [ -z "$want" ]; then
        if [ "$REBASE" = 1 ]; then echo "$tag (rebaseline): $got"; return 0; fi
        if [ "$ALLOW_UNPINNED" = 1 ]; then echo "$tag (unpinned, allowed): $got"; return 0; fi
        echo "$tag FAIL: no pin.  export $var='...' (the value is per machine; see" >&2
        echo "  AGENTS.local.md), or pass ALLOW_UNPINNED=1 to run without one" >&2
        exit 1
    fi
    case "$got" in
        "$want"*) echo "$tag OK: $got" ;;
        *) echo "$tag FAIL: got '$got', want prefix '$want'" >&2
           echo "  (system prompt: $SYS)" >&2
           exit 1 ;;
    esac
}

# Behaviour: non-empty, at least 6 characters, no 4-char window repeated three
# times in a row, and a sentence-ending mark or a hard stop.
behaviour() {
    tag="$1" got="$2"
    python3 - "$tag" "$got" <<'PY'
import sys
tag, s = sys.argv[1], sys.argv[2]
bad = []
if len(s) < 6:
    bad.append("len<6")
for i in range(len(s) - 11):
    w = s[i:i+4]
    if s[i:i+12] == w * 3:
        bad.append("repeated %r" % w)
        break
if s and not s.rstrip().endswith(("。", "！", "？", "…", "，", "、", "”", "’")):
    bad.append("no terminator")
if bad:
    print("%s FAIL: %s | %r" % (tag, ", ".join(bad), s), file=sys.stderr)
    sys.exit(1)
print("%s OK: %s" % (tag, s))
PY
}

# Documentation first: it costs a fraction of a second, needs no model, and a
# command that no longer runs is a bug in the document (AGENTS.md).
sh gate/docs_check.sh || {
    echo "docs FAIL: run gate/docs_check.sh for the list" >&2
    exit 1
}

# Shader text is the numerics: a builder change that moves the emitted GLSL has
# to move the golden deliberately, and a text-neutral refactor must leave it
# untouched (AGENTS.md, "Source split policy").  Pass the same binary the rest
# of the gate uses - the text does not depend on the target, so a Pi run is
# checked against the same golden.  Three outcomes, and they must not be
# confused: 0 passed, 1 the text really moved, anything else the check could
# not run here (a cross-built binary on the laptop exits 255, a build without
# --dump-shaders fails earlier) - still a gate failure, but not a numerics
# signal, so say which one happened.
shader_rc=0
sh gate/shader_hashes.sh "$BIN" || shader_rc=$?
case $shader_rc in
    0) ;;
    1) echo "shader FAIL: the emitted GLSL moved; run gate/shader_hashes.sh $BIN" >&2
       exit 1 ;;
    *) echo "shader gate could not run: gate/shader_hashes.sh $BIN exited $shader_rc" >&2
       echo "  (binary not runnable on this host, or built without --dump-shaders)" >&2
       exit 1 ;;
esac

# A Target that outlives a scene release is reused with a deleted name: the
# create sites in gpu.cpp are guarded on "tex == 0", and nothing else notices.
sh gate/check_q16_decode.sh "$BIN" || {
    echo "q16 decode FAIL: see gate/check_q16_decode.sh $BIN" >&2
    exit 1
}

sh gate/check_target_reset.sh || {
    echo "target reset FAIL: run gate/check_target_reset.sh for the list" >&2
    exit 1
}

# The draw cap and the container layout are two more checks that cost nothing and
# were only in `make check`/CI: the acceptance gate is the one that runs on the
# hardware, so it should not be the one with fewer checks (2026-11-15).
sh gate/check_draw_cap.sh || {
    echo "draw cap FAIL: a gpu*.cpp uses a bare 1024 instead of kDrawCap" >&2
    exit 1
}

MMCF_MODELS=${MMCF_MODELS:-../models} python3 gate/test_mmcf_format.py || {
    echo "container layout FAIL: see gate/test_mmcf_format.py (MMCF_MODELS=$MMCF_MODELS)" >&2
    exit 1
}

q8=$(text "$Q8" "$PROMPT" 16 "$SYS")
q8b=$(text "$Q8" "$PROMPT" 16 "$SYS")
if [ "$q8" != "$q8b" ]; then
    echo "determinism FAIL: two runs of the same binary differ" >&2
    echo "  run 1: $q8" >&2
    echo "  run 2: $q8b" >&2
    exit 1
fi
echo "determinism OK"

q8_def=$(text "$Q8" "$PROMPT" 16 "")
[ "$q8_def" = "$q8" ] || {
    echo "system prompt drift: --system '$SYS' gives '$q8'" >&2
    echo "  but the compiled default gives '$q8_def'" >&2
    echo "  update kDefaultSystem, AGENTS.md, this script and docs/benchmarks.md" >&2
    exit 1
}
echo "default system prompt OK"

check_prefix q8 "$Q8_EXPECT" "$q8" Q8_EXPECT
covered "text-q8"
q4=$(text "$Q4" "$PROMPT" 16 "$SYS")
check_prefix q4 "$Q4_EXPECT" "$q4" Q4_EXPECT
covered "text-q4"
poem=$(text "$Q8" "$PROMPT2" 48 "$SYS")
behaviour poem "$poem"
covered "behaviour"

# Mixture of experts: the route is a host-side argmax over a read-back, so the
# thing to gate is that it is *deterministic* and that the answer is a sentence.
# A route flip changes the whole text, which is why this is not compared with
# the dense string.
m1=""
if [ -n "$MOE" ]; then
    m1=$(text "$MOE" "$PROMPT" 24 "$SYS")
    m2=$(text "$MOE" "$PROMPT" 24 "$SYS")
    if [ "$m1" != "$m2" ]; then
        echo "moe determinism FAIL: two runs of the same binary differ" >&2
        echo "  run 1: $m1" >&2
        echo "  run 2: $m2" >&2
        exit 1
    fi
    echo "moe determinism OK"
    check_prefix moe "$MOE_EXPECT" "$m1" MOE_EXPECT
    behaviour moe "$m1"
    # The route sequence is the discrete part of a MoE: a flipped expert is a
    # different FFN, so a text pin alone can pass on a model whose routing has
    # moved near a tie.  MOE_ROUTES is a file of the MOE_DUMP lines with the
    # q16 gate read-back stripped (the route itself is machine independent in
    # principle, but a driver's rounding can move a tie, so pin per machine):
    #
    #   MODEL=$MOE GPU_LAYER=1 KV_PACK=16 MOE_DUMP=1 \
    #     $BIN --chat --system "$SYS" --prompt "$PROMPT" --gen 8 --greedy \
    #     2>&1 | sed -n 's/^MOE //p' | sed 's/ gate=.*//' > moe_routes.txt
    #
    # then set MOE_ROUTES=moe_routes.txt.
    if [ -n "$MOE_ROUTES" ]; then
        # An empty baseline used to compare equal to an empty capture, so a
        # route dump that failed (or that produced nothing) printed OK.  Require
        # a baseline with lines and a capture with lines, and check the status.
        [ -s "$MOE_ROUTES" ] || {
            echo "moe routes FAIL: the baseline $MOE_ROUTES is empty; regenerate it" >&2
            exit 1
        }
        routes_out=$(mktemp "${TMPDIR:-/tmp}/moe_routes.XXXXXX") || exit 1
        MODEL="$MOE" GPU_LAYER=1 KV_PACK=16 MOE_DUMP=1 "$BIN" --chat --system "$SYS" \
            --prompt "$PROMPT" --gen 8 --greedy > "$routes_out" 2>&1 || {
            echo "moe routes FAIL: the route dump exited nonzero" >&2
            sed 's/^/  /' "$routes_out" >&2
            rm -f "$routes_out"
            exit 1
        }
        sed -n 's/^MOE //p' "$routes_out" | sed 's/ gate=.*//' > "$routes_out.routes"
        rm -f "$routes_out"
        if [ ! -s "$routes_out.routes" ]; then
            echo "moe routes FAIL: the run printed no MOE route lines (MOE_DUMP=1)" >&2
            rm -f "$routes_out.routes"
            exit 1
        fi
        if diff -q "$MOE_ROUTES" "$routes_out.routes" >/dev/null 2>&1; then
            echo "moe routes OK ($(wc -l < "$routes_out.routes") lines)"
            covered "moe-routes"
        else
            echo "moe routes FAIL: the per-layer route sequence moved" >&2
            diff "$MOE_ROUTES" "$routes_out.routes" | head -10 >&2
            rm -f "$routes_out.routes"
            exit 1
        fi
        rm -f "$routes_out.routes"
    else
        skipped "moe-routes" "MOE_ROUTES unset"
    fi
    covered "moe-text"
else
    skipped "moe" "MOE unset"
fi

# RWKV-7: the third layer loop (docs/rwkv7-plan.zh-CN.md, P2/P4).  Same contract
# as the dense pins -- determinism first, then the per-machine prefix, then PPL
# -- but the reply is multi-line and cut at the gen limit mid-sentence, so the
# dense behaviour check (Chinese sentence terminator) does not apply: non-empty
# and no collapse is the bar.  The default-system check is the same invariant as the dense one: the
# container's key-6 prompt and the gate's --system must render the same answer.
r1=""
if [ -n "$RWKV" ] && [ -f "$RWKV" ]; then
    r1=$(rwkv_text "$RWKV" "$PROMPT" 16 "$SYS")
    r2=$(rwkv_text "$RWKV" "$PROMPT" 16 "$SYS")
    if [ "$r1" != "$r2" ]; then
        echo "rwkv determinism FAIL: two runs of the same binary differ" >&2
        echo "  run 1: $r1" >&2
        echo "  run 2: $r2" >&2
        exit 1
    fi
    echo "rwkv determinism OK"
    rdef=$(rwkv_text "$RWKV" "$PROMPT" 16 "")
    [ "$rdef" = "$r1" ] || {
        echo "rwkv system prompt drift: --system '$SYS' gives '$r1'" >&2
        echo "  but the container default gives '$rdef'" >&2
        echo "  update the container, this gate and docs/benchmarks.md together" >&2
        exit 1
    }
    echo "rwkv default system prompt OK"
    check_prefix rwkv "$RWKV_EXPECT" "$r1" RWKV_EXPECT
    python3 - "$r1" <<'PY'
import sys
s = sys.argv[1]
bad = []
if len(s) < 6:
    bad.append("len<6")
for i in range(len(s) - 11):
    w = s[i:i+4]
    if s[i:i+12] == w * 3:
        bad.append("repeated %r" % w)
        break
if bad:
    print("rwkv behaviour FAIL: %s | %r" % (", ".join(bad), s), file=sys.stderr)
    sys.exit(1)
print("rwkv behaviour OK")
PY
    covered "rwkv-$RWKV_LOOP-text"
elif [ -n "$RWKV" ]; then
    skipped "rwkv-$RWKV_LOOP" "no container at $RWKV"
    RWKV=""
else
    skipped "rwkv-$RWKV_LOOP" "RWKV= empty"
fi

# SEQ is pinned: the ppl path chunks the corpus per window, so the token count
# and the value both move with SEQ and are not comparable across settings.
# $1 model, $2 SEQUENCE.  The engine's status is checked before the number is
# read: `engine | sed` used to turn a crash into an empty string that the
# comparison below then reported as a very confusing ValueError.
ppl() {
    f=$(mktemp "${TMPDIR:-/tmp}/regression.XXXXXX") || return 1
    MODEL="$1" GPU_LAYER=1 KV_PACK=16 SEQ="$2" \
        "$BIN" --ppl "$CORPUS" > "$f" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ppl FAILED (rc=$rc): model=$1 SEQ=$2" >&2
        sed 's/^/  /' "$f" >&2
        rm -f "$f"
        return 1
    fi
    sed -n 's/^PPL \([0-9.]*\).*/\1/p' "$f" | head -1
    rm -f "$f"
}

check_ppl() {
    tag="$1" want="$2" got="$3" var="${4:-<pin>}"
    if [ -z "$want" ]; then
        if [ "$REBASE" = 1 ]; then echo "$tag (rebaseline): $got"; return 0; fi
        if [ "$ALLOW_UNPINNED" = 1 ]; then echo "$tag (unpinned, allowed): $got"; return 0; fi
        echo "$tag FAIL: no pin.  export $var='...' (the value is per machine; see" >&2
        echo "  AGENTS.local.md), or pass ALLOW_UNPINNED=1 to run without one" >&2
        exit 1
    fi
    python3 - "$tag" "$want" "$got" "$PPL_TOL" <<'PY'
import sys
tag, want, got, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if abs(got - want) / want > tol:
    print("%s FAIL: %.4f vs expected %.4f (>%.1f%%)" % (tag, got, want, tol * 100), file=sys.stderr)
    sys.exit(1)
print("%s OK: %.4f (expected %.4f)" % (tag, got, want))
PY
}

p0=""
q4p=""
if [ "${NO_PPL:-0}" != 1 ]; then
    p0=$(ppl "$Q8" "$PPL_SEQ")
    q4p=$(ppl "$Q4" "$PPL_SEQ")
    echo "PPL baseline: $p0"
    echo "PPL q4:       $q4p"
    check_ppl PPL "$PPL_EXPECT" "$p0" PPL_EXPECT
    covered "ppl-q8"
    check_ppl "PPL q4" "$Q4_PPL_EXPECT" "$q4p" Q4_PPL_EXPECT
    covered "ppl-q4"
    if [ -n "$MOE" ] && [ "$MOE_PPL" = 1 ]; then
        moep=$(ppl "$MOE" "$PPL_SEQ")
        echo "PPL moe:      $moep"
        check_ppl "PPL moe" "$MOE_PPL_EXPECT" "$moep" MOE_PPL_EXPECT
        covered "ppl-moe"
    fi
    if [ -n "$RWKV" ]; then
        # The RWKV container's own SEQ, not PPL_SEQ: the window protocol moves
        # with SEQ and the two models' numbers are not comparable anyway.
        rwkvp=$(ppl "$RWKV" "$RWKV_PPL_SEQ")
        echo "PPL rwkv:     $rwkvp"
        check_ppl "PPL rwkv" "$RWKV_PPL_EXPECT" "$rwkvp" RWKV_PPL_EXPECT
        covered "ppl-rwkv-$RWKV_LOOP"
    fi
else
    skipped ppl NO_PPL=1
fi

# ACT8=1/2 are a diagnostic A/B, not acceptance: the two runs are printed and
# never compared, so they cannot fail the gate and are not counted as coverage.
# They are off by default because a broken arm used to be silently ignored.
if [ "${ACT_PPL_DIAG:-0}" = 1 ] && [ "${NO_PPL:-0}" != 1 ]; then
    echo "PPL ACT8=2:   $(ppl "$Q8" 2)" || true
    echo "PPL ACT8=1:   $(ppl "$Q8" 1)" || true
    echo "  (ACT8 diagnostics only: not compared, not covered)"
fi

if [ "$REBASE" = 1 ]; then
    cat <<EOF

# --- copy into this machine's override (the values are machine specific) ---
Q8_EXPECT='$q8'
Q4_EXPECT='$q4'
PPL_EXPECT='${p0:-}'
Q4_PPL_EXPECT='${q4p:-}'
MOE_EXPECT='${m1:-}'
RWKV_EXPECT='${r1:-}'
RWKV_PPL_EXPECT='${rwkvp:-}'
EOF
fi

# What this run actually compared, so a green ACCEPT is not read as "every
# supported path passed".  Skips are listed with their reason.
echo "covered: $COVERED"
if [ -n "$SKIPPED" ]; then
    # Already one item per line; quote so the shell does not split the reasons.
    printf 'skipped: %s' "$SKIPPED"
else
    echo "skipped: -"
fi
# Fixed-scope statements: these are not gated here, whatever the flags say.
echo "rwkv loop:  $RWKV_LOOP (the RWKV pins are whoever exported them)"
echo "not covered: ACT8 arms (diagnostic only)"
echo "ACCEPT"
