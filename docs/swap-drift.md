# The `/model` drift, and how it was found

`/model PATH` is the REPL command that replaces the container while keeping the
GL context and its compiled programs.  It has been wrong twice, and both times
the *symptom* was a model that answered slightly differently after a swap than
from a fresh process.  This page is the record of the second one: what the
invariant is, why it exists, and the order in which the evidence appeared,
because that order is how the next one will be found.

The rule it produced is in [AGENTS.md](../AGENTS.md) ("Switching models without
restarting the engine") and the acceptance is
[`engine/scripts/swap_matrix.py`](../engine/scripts/swap_matrix.py); this page is
the reasoning behind both, dated 2026-11-14.

## What a swap owns

Back then `main` did more to a model than read it: `split_ffn()` decided the FFN
kernel width (MiniMind3 runs two 1216-wide chunks, not one 2432-wide pass),
`apply_hidden_scale()` applied metadata key 15, and `apply_precision()` /
`quantize_q4_weights()` rewrote Mats.  A swap that skipped those steps still
answered — with the *whole-width* FFN and no hidden scale.  The numbers: logits
moved by one unit in the last q16 place on the dense models (cosine 0.9998, which
was enough to flip a greedy answer on `minimind3.i16`) and by a lot more on the
hybrid (cosine 0.96, both LFM2.5 containers).  They now live in one
`prepare_model()` that startup and the swap both call, before the upload; the
step slices Mats, so it has to run first.

## How it was ruled out, in the order that worked

Two *fresh* processes were bit-identical, so it was the reload.  The emitted
GLSL was identical across reloads (26 program keys, no key with two texts), so it
was not the program cache.  Zeroing the new targets and the staging textures
changed nothing.  The first divergence was layer 0's FFN output — which pointed
at the shape of the kernel rather than at its inputs.

`SHADER_SRC=` prints the GLSL of every program as it is used, and the two
requests' key sets differed: `xq8p2a_768_1216` against `xq8p2a_768_2432`.  That
is what named the missing `split_ffn()`.  A stage dump shows *where* two paths
diverge; `SHADER_SRC=` shows *which kernel* they ran.

## The other one: a program key is not a program text

`GPU::x_prog(key, src)` caches by key, and the builders name programs by kernel,
not by shape: the same key yields different GLSL for another model (head width,
qk-norm presence, activation format).  Without the hash check on every cache hit
the second model silently ran the first model's attention kernels.  The cache
therefore validates the shader hash (`prog_key_hash_`).

## Why the matrix has an i16 chain and a hybrid chain

`swap_matrix.py` needs both because a dense-only matrix passed with the bug
present: `minimind3.i16` flipped a greedy answer on the last q16 place, and the
hybrids moved by cosine 0.96.  A single golden string would not have caught
either — some models answer correctly anyway, with a wrong program or a stale
target.

## The rule that came out of it

Everything the GPU owns must be released **and reset**, not left to the context
teardown: `release_scene()` deletes the textures/FBOs, `reset_targets()` zeroes
the scratch targets (including the lazily filled `target_for()` map), the
per-layer handle vectors and the activation caches are cleared, and
[`check_target_reset.sh`](../engine/gate/check_target_reset.sh) fails the gate
if a `Target` member is added without being cleared.
