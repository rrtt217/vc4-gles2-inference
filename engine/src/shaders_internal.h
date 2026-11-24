#pragma once

// The seam between the shader translation units: the prelude every builder
// emits, and the decode text more than one weight format shares.  This is not
// the public builder API -- that is shaders.h -- and shaders_attn.cpp does not
// include it: it consumes exp_poly_source() through shaders.h only.
//
// The emitted text is the numerics, so a helper lives here exactly when two or
// more TUs emit it: one definition, same bytes.  A helper one format owns stays
// in that format's TU (q8's enc16, q4's nibble taps, t3's bit decode, block's
// rms codec, RWKV's bit-cast codec).
#include "xformat.h"

#include <string>

std::string head(XFormat xf);
void need_shape(const char* who, int rows, int cols, int unit);
void need_dim(const char* who, int n);

// EXP=drv|poly: the exponential the builders emit.  Functions, not strings,
// because the answer depends on the switch registry and reading it at static
// init time is too early.
std::string EXP(const char* arg);
std::string poly_src();
std::string qz_sl();

// Metadata key 15: the FFN hidden store scale (2^-hs), emitted only when set.
std::string hid_scale_def(int hs);
std::string hid_scale_mul(int hs);

// The q16 pair decode shared by the q8 and q4 kernels.
extern const std::string kDec16A;
extern const std::string kDec16B;

// Per-row weight scale (tensor id part 3).  Used by q8, q6 and t3; the pair of
// rows in one texel is not an optimisation (probes/wrow_probe).
std::string wrow_scale(int rows, const char* smp = "WS", const char* fn = "wsr",
                       const char* unit = "ws", bool hi = false);