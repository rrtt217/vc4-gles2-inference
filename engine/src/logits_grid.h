#pragma once

// The vocabulary grid.
//
// The logits matrix is stored in container tiles of kLogitsGrid rows: the
// packer splits it that way (tools/mmcf_grid.py, LOGITS_GRID), the logits pass
// renders one tile per target row at this width, the read-back uses it as the
// row stride, and the Top-K kernels divide the vocabulary index by it.  A
// token id is
//
//     token = tile * kLogitsGrid + index_within_tile
//
// so all of those have to agree.  Nothing in the toolchain checks that they
// do: a mismatch is not a compile error, not a link error and not a GL error,
// it is a decode with shifted tokens.  Measured before this constant existed:
// changing only the shader divisor left greedy output and PPL (270.3170)
// bit-identical, because the gate never runs the Top-K path, and changed
// TOPK=8 sampling from "我是由Jingyaogong创建的..." to "the manuscript is the
// speaker...".  engine/probes/logits_grid_probe holds the three languages to
// this value.
//
// 1536 is a VC4 texture-width choice (the FBO limit is 2048, and draw() still
// splits a 1536-row matvec because a single pass of that width is unreliable),
// not a model property: a smaller vocabulary simply ends with a short tile
// (MiniMind3: 4 x 1536 + 256 = 6400, which is a side effect, not a derivation -
// 1600 would have split 6400 into four even tiles).
//
// Provenance: the number arrived with the first import (571b099), already baked
// into the legacy checkpoint layout by converters that were later deleted
// without a note; no commit in this repository derives it.  What it happens to
// match: three quarters of VC4's 2048 limit; 2 x 768, the hidden size of
// MiniMind3 and the fused W13 width of the llama2.c models the engine was first
// brought up on; and MiniMind3's qkv width (768 + 2 x 384).  Which of those was
// the reason is not recorded.  It is also not load bearing: repacking
// MiniMind3 q8 at 1024 (7 tiles) gave the same greedy text and PPL 270.5287
// against 270.3170, the 0.08% being the per-tile q8 scale, at the cost of two
// more logits draws per token and a repack of every model.
constexpr int kLogitsGrid = 1536;
// The paired vocabulary kernel writes two logits per texel.
constexpr int kLogitsGridPacked = kLogitsGrid / 2;

