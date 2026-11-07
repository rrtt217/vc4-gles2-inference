#!/bin/sh
# supra2-100M-Instruct smoke battery, CPU path.
#
# English only (the model card is explicit), 1024-token training context, and a
# 100M model, so judge it on "coherent and on topic", not on facts.  Every case
# runs twice to catch nondeterminism.
#
#   MODEL=../models/supra2/supra2.q24.mmcf sh supra2-battery.sh
#
# SEQ defaults to 512 to cover the longest prompt here.  LAYER=1 runs the same
# battery on the GPU path, which is what a Pi deployment uses.
set -u
BIN="${BIN:-./engine_mm3}"
MODEL="${MODEL:?set MODEL to a supra2 container}"
SYS="You are a helpful assistant."
GEN="${GEN:-24}"

run() { # prompt, gen, extra...
    p="$1"; shift
    MODEL="$MODEL" GPU_LAYER="${LAYER:-0}" KV_PACK=16 SEQ=512 "$BIN" --chat --system "$SYS" \
        --prompt "$p" --gen "$1" --greedy 2>/dev/null | sed -n 's/^Assistant: //p' | tr '\n' ' '
}

case_n() { # label, prompt, gen
    printf '%-26s | ' "$1"
    a=$(run "$2" "$3"); b=$(run "$2" "$3")
    printf '%s' "$a"
    [ "$a" = "$b" ] || printf '   [NONDETERMINISTIC]'
    echo
}

echo "model: $MODEL   greedy, gen $GEN, GPU_LAYER=${LAYER:-0} (LAYER=1 for the GPU path)"
echo
echo "--- identity / chat ---"
case_n "two sentences about you" "Introduce yourself in two sentences." 24
case_n "what can you do" "What can you do?" 24
case_n "who made you" "Who made you?" 24
echo
echo "--- instruction following ---"
case_n "one word: yes" "Reply with exactly one word: yes" 8
case_n "three colours" "List three colours, comma separated." 16
case_n "uppercase" "Write the word 'hello' in uppercase." 8
echo
echo "--- factuality (100M: weak by design) ---"
case_n "capital of France" "What is the capital of France?" 16
case_n "two plus two" "What is 2+2?" 12
echo
echo "--- long generation (loops / collapse) ---"
case_n "story, 48 tokens" "Tell me a short story about a cat." 48
echo
echo "--- termination: does it stop on its own? ---"
printf '%-26s | ' "gen 64, count chars"
out=$(run "Say goodbye." 64)
echo "$out (${#out} chars)"
echo
echo "--- sampling (temp 0.8) ---"
printf '%-26s | ' "temp 0.8 seed 1"
MODEL="$MODEL" GPU_LAYER="${LAYER:-0}" KV_PACK=16 SEQ=512 "$BIN" --chat --system "$SYS" \
    --prompt "Tell me a joke." --gen 24 --temp 0.8 --seed 1 2>/dev/null |
    sed -n 's/^Assistant: //p' | tr '\n' ' '; echo
printf '%-26s | ' "temp 0.8 seed 2"
MODEL="$MODEL" GPU_LAYER="${LAYER:-0}" KV_PACK=16 SEQ=512 "$BIN" --chat --system "$SYS" \
    --prompt "Tell me a joke." --gen 24 --temp 0.8 --seed 2 2>/dev/null |
    sed -n 's/^Assistant: //p' | tr '\n' ' '; echo
