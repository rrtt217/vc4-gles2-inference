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
// (MiniMind3: 4 x 1536 + 256 = 6400).  It is also not load bearing: repacking
// MiniMind3 q8 at 1024 (7 tiles) gave the same greedy text and PPL 270.5287
// against 270.3170, the 0.08% being the per-tile q8 scale, at the cost of two
// more logits draws per token and a repack of every model.
constexpr int kLogitsGrid = 1536;
// The paired vocabulary kernel writes two logits per texel.
constexpr int kLogitsGridPacked = kLogitsGrid / 2;

// draw() splits a matvec into column strips when the matrix has this many rows
// (a single pass of that width is unreliable on VC4).  It is the qkv width for
// MiniMind3 and therefore equals kLogitsGrid, but it is a separate quantity:
// another model's qkv width changes this one and not the vocabulary grid.
constexpr int kMatvecStripRows = 1536;
