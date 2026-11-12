#pragma once

// All GLSL source construction lives here.  The GPU object only caches and
// binds programs; shader text is not mixed with GL state changes.
#include "xformat.h"

#include <string>

struct AttnConfig {
    int dim = 0;
    int kv_dim = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int seq = 0;
    bool rope_half = false;
    bool qk_norm = false;
    bool qkv_pair = false;  // qkv matvec writes two q16 rows per texel
    int kv_bits = 0;
    float kv_scale = 16384.0f;
    int kv_width = 0;
    // 8-bit KV only: V is quantised on a per-head, per-position grid derived
    // from that head max (K keeps the global KVSCALE grid: it is qk-normed
    // before rope, so its range is already about unit RMS).
    bool kv_vstep = false;
};

std::string att_defs(const AttnConfig& c);
std::string att_helpers();

std::string matvec_q8(int rows, int cols, XFormat xf, bool wrow = false);
std::string matvec_q8_pair(int rows, int cols, XFormat xf, bool wrow = false);
std::string matvec_q8_pair_swiglu(int rows, int cols, XFormat xf, bool inl = true, bool wrow = false, int hs = 0);
std::string matvec_q8_gate2(int rows, int cols);
std::string matvec_q8_p2(int rows, int cols, bool wrow = false);
std::string rms_stats2(int dim, XFormat rf);
std::string rms_norm_pair2(int dim, XFormat rf, bool use_eps_uniform);
std::string matvec_q8_gate2_p2(int rows, int cols, bool wrow = false);
std::string matvec_q8_p2_add_p2(int rows, int cols, bool wrow = false);
std::string matvec_q8_p2_add(int rows, int cols, XFormat rf, bool wrow = false);
std::string p2_split(int rows, XFormat rf);
std::string matvec_q8_pair_fused(int rows, int cols, XFormat xf, bool wrow = false);
std::string matvec_q8_pair_logits(int rows, int cols, XFormat xf, bool wrow = false);
std::string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf, bool wrow = false);
std::string matvec_q8_add_pair(int rows, int cols, XFormat xf, bool inl = true, bool wrow = false);
std::string matvec_i16(int rows, int cols, XFormat xf);
std::string matvec_i16_add(int rows, int cols, XFormat xf, XFormat rf);
std::string matvec_i4(int rows, int cols, XFormat xf, bool inl = true);
// Signed 6-bit head: a 4-bit low plane and a 2-bit high plane in two textures
// (one sampler cannot be fetched twice in a fragment on VC4), one 24-bit value
// per fragment, per-row scale plane when `wrow`.
std::string matvec_i6(int rows, int cols, XFormat xf, bool wrow);
std::string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf);
std::string matvec_i4_pair(int rows, int cols, XFormat xf);
std::string matvec_i4_pair_swiglu(int rows, int cols, XFormat xf, bool inl = true, int hs = 0);
std::string matvec_i4_add_pair(int rows, int cols, XFormat xf, bool inl = true);

// `hs` is the container's per-layer FFN-hidden scale exponent (metadata key
// 15): the hidden is written as value * 2^-hs so it fits the q16 activation
// rail, and w2's read-back multiplies by 2^hs.  hs = 0 emits exactly the text
// the builder produced before the key existed, so every existing container's
// programs are byte-identical.
std::string swiglu_pair(int hidden, XFormat out, int hs = 0);
std::string swiglu_2row(int hidden, XFormat out, int hs = 0);
std::string swiglu(int hidden, bool fused, XFormat out, int hs = 0);
// LFM2 conv block: elementwise gate, 3-tap depthwise convolution, and a copy
// (the state shift).  All three work in raw q16 units on I16_2 textures.
// One elementwise multiply of two conv-block activations.  A and B each
// carry their own source layout: two matvec outputs are 24-bit, the
// convolution result is q16, and the decoder follows the operand.
std::string mul_act(int dim, XFormat xf, XFormat af, XFormat bf);
std::string conv3_act(int dim, XFormat xf);
std::string copy_act(int dim, XFormat xf);

std::string rms_stats_split(int dim, XFormat residual);
std::string rms_stats_reduce();
std::string rms_norm(int dim, XFormat residual, XFormat out, bool fused_reduce,
                     bool use_eps_uniform);

std::string att_kv_rope(const AttnConfig& c);
std::string att_vscale(const AttnConfig& c);
std::string att_qnorm_scale(const AttnConfig& c);
std::string att_qrot(const AttnConfig& c);
std::string att_score(const AttnConfig& c);
std::string att_stats(const AttnConfig& c);
std::string att_prob(const AttnConfig& c);
std::string att_v(const AttnConfig& c, XFormat out);

std::string top2_fast(int rows, int chunk);
std::string top2(int rows, int chunk);
std::string top2_fast_packed(int rows, int chunk);
std::string top2_packed(int rows, int chunk);
