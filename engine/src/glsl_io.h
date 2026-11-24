#pragma once
#include <string>

// Runtime GLSL text: the two switches that keep a kernel experiment out of the
// build loop (docs/test-hooks.md, "GLSL without a rebuild").
//
//   GLSL_DUMP=<dir>               write every program this run builds to
//                                 <dir>/<key>.glsl -- the seed, as one directory
//   GLSL_OVERRIDE=<k>=<file>,...  compile <file> for the program key <k> instead
//                                 of the builder's text (one or more programs)
//
// The key is the one x_prog caches by (`xq8ps_1216_768_x16_2`, `xkv`, ...), so a
// seeded file is readable and the command line names exactly which programs move.
// Unset means the builder's text, byte for byte: the default path is unchanged,
// and gate/shader_hashes.sh keeps hashing the builders, not this.
//
// The returned text is what compiles.  An override is a different text, so x_prog
// hashes it as a different program: two keys can build one text (xq8pl and xqkvp
// do), and an override for one of them must not reprogram the other.
//
// The text of a key is not a property of the key alone -- a model swap rebuilds
// the same key with another head width, so the dump numbers a second text for one
// key (`<key>.2.glsl`) rather than overwriting the first, and the exit report
// names an override whose key never matched at all.
std::string glsl_text(const std::string& key, const std::string& src);
