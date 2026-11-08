#pragma once

// Shader-text regression harness.  Every builder in shaders.cpp /
// shaders_attn.cpp is called with a fixed matrix of arguments and the
// resulting GLSL is printed, either as one digest per builder or verbatim.
//
// The point is that the emitted *text* is the thing the driver compiles and
// the thing a measurement pins.  "Semantically identical" GLSL is not
// bit-identical arithmetic on VC4 (docs/performance.md), so a refactor of the
// builders has to be shown to emit the same bytes, not argued to.
//
//   docs: docs/engine-layout.md
//   gate: scripts/shader_hashes.sh
//
// It needs no model and no GL context: builders are pure string functions, and
// the q4 shift decode (normally a driver query) is forced instead.
int dump_shaders(bool raw);
