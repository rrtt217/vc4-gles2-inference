"""The vocabulary grid, shared by the packer and the HF loader.

The logits matrix is stored in container tiles of LOGITS_GRID rows, and the
engine maps token = tile * LOGITS_GRID + index.  The value must equal
kLogitsGrid in engine/src/logits_grid.h; engine/probes/logits_grid_probe checks
that this file, the header and the emitted GLSL all agree.
"""

LOGITS_GRID = 1536
