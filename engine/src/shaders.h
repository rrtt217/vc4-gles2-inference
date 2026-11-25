#pragma once

// All GLSL source construction lives here.  The GPU object only caches and
// binds programs; shader text is not mixed with GL state changes.
#include "xformat.h"

#include <string>

// The attention block's grids, in one place: `AttnConfig`'s field defaults and
// the environment defaults in gpu_attn.cpp have to agree, and a bare 1024 in a
// gpu*.cpp is what check_draw_cap.sh refuses (kDrawCap has the same value and
// the two are unrelated -- that is exactly the confusion the check prevents).
constexpr float kAttnScoreScale = 1024.0f;   // score grid +-32, step 1/1024
constexpr float kAttnSumScale = 32.0f;       // (max,sum) texel, sum step 1/32
constexpr float kAttnSumLogScale = 2048.0f;  // sum stored as log2(sum)*this

// The post-norm block's LayerNorm rows and its (mean, invstd) texel: one q16
// pair a texel.  The producers (upload_v5_rows, ln_stats_split) and the reader
// (ln_apply_q16) share this one name -- it is a layout, not an axis, because
// both numbers are O(1) after a normalisation.  Decoding it with another
// stream's format (the residual's, as this used to) is right only by accident.
constexpr XFormat kLnPairFmt = XFormat::I16_2;

// The XFormat RMS_STORE selects for the rms partial sums: S24_1 (default) or
// F32.  One accessor so the host dump decodes with the shader's format.
XFormat rms_store_format();

// The KV cache's layout, built once from the `KV` value.  A codec is an
// encoding *and* the number of values a texel holds: i24 and f32 one, q16 and
// bf16 two, q8 four.  `vpt` is the constant every writer and reader needs --
// the cross-KV writer assumed two and lost half of every i24 row because no
// name carried it.  `texels` is the [K | V] row width, `scale` the grid the
// fixed codec uses, and `vstep` the 8-bit per-head V grid.
enum class KvCodec : uint8_t { Fixed = 0, Bf16 = 1, F32 = 2 };

struct KvLayout {
    KvCodec codec = KvCodec::Fixed;
    int bits = 0;      // 0 = i24/f32, 16 = q16/bf16, 8 = q8
    int vpt = 1;       // values a texel
    int texels = 0;    // [K | V] row width
    float scale = 16384.0f;
    bool vstep = false;
};

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
    // Every key is visible: a bidirectional pass (the encoder) or a
    // cross-attention.  Emitting nothing when it is true is what keeps every
    // decoder-only program's text byte-identical (gate/shader_hashes.sh).
    bool causal = true;
    // No rotation at all (v5): the block's positions are a table added to the
    // embedding, so the K write and the Q stage must not rotate either.
    bool rope_none = false;
    // The KV cache's layout, built once from `KV` in init_attention.  The
    // per-head V grid is part of it: 8-bit KV only, derived from the head max,
    // with K on the global KVSCALE grid (qk-normed, so already unit RMS).
    KvLayout kv;
    // ATT_EXP=poly: use the portable exp (x_exp) instead of the driver's.
    // VC4's exp/exp2 are the same fixed-point sequence at 4.6e-05 worst
    // relative error against Xe's 2.2e-06; the polynomial is 1.5e-06 on
    // both (probes/sumorder_probe).  Off by default: the pins and the
    // shader digests belong to the driver's exp.
    bool exp_poly = false;
    // The score and (max,sum) grids.  q16 with these scales: the score texel
    // holds round(score*score_scale), so the representable score range is
    // +-32768/score_scale (1024 -> +-32, 256 -> +-128) and the step is
    // 1/score_scale.  The stats max rides the same grid, the sum its own
    // (32 -> step 1/32).  Default 1024 is what every model before 2026-11-17
    // was measured on; ATT_SCALE overrides for a model whose scores exceed it.
    float score_scale = kAttnScoreScale;
    float sum_scale = kAttnSumScale;
    // The sum's domain.  `sum_log` stores log2(sum)*sum_log_scale in the same
    // q16 slot: 1/2048 per octave is 0.03% relative for any sum, where the
    // linear grid on `sum_scale` is 2% at sum ~ 1.5.  Default is the linear
    // grid, because the more accurate one is not the better model: MiniMind3
    // q8 PPL 270.3170 (linear) against 274.4160 (log2) at SEQ=768, and q4
    // 326.1483 against 323.8378 -- the two containers disagree in sign, so the
    // sum's rounding is not a quality lever here, it is a redistribution of a
    // sub-0.1% per-head gain.  ATT_SUM=log2 selects it.
    bool sum_log = false;
    float sum_log_scale = kAttnSumLogScale;
    // Prefill query batching (PREFILL_BATCH).  1 = the serial kernels, text
    // unchanged.  N > 1: the score grid is seq/2 x (heads*N), one query row a
    // head-major block (row = h*N + b), the rotated Q texture is N rows tall,
    // and `pos` carries the group's first position -- a fragment's causal bound
    // is pos + b.  stats/v keep one draw a position but address the batched
    // score texture through row = h*N + qrow0 (qrow0 = b, a uniform).  Every
    // fragment's arithmetic is the serial kernel's; only the row it reads and
    // writes moves, which is what keeps the logits byte-identical.
    int qbatch = 1;
    // Long context: the sequence is split into `nseg` textures of at most
    // `seg` positions each (docs/long-context.md).  `nseg == 1` is the shipped
    // layout and every builder emits its text unchanged -- the segment
    // arithmetic below is all guarded on `nseg > 1`, because a state document's
    // claim ("NSEG=1 is the old layout") has to be true of the emitted GLSL and
    // not only of the results.  With `nseg > 1` a texture holds `nseg * seg`
    // positions laid out k-major: position t lives in texture `t / seg` at
    // x = `(t % seg) * stride`, where `stride` is the KV vector's texel count
    // in the KV passes and 1 in the score passes (whose own x is the position
    // pair).  The segment is a compile-time constant, so the sampler is chosen
    // by a chain of `if (s == k)` and no array indexing is needed (ESSL 1.00
    // has no indexed sampler).
    // Each segment gets its own program (`segk`), so a kernel binds exactly one
    // sampler per plane and scans only its own slice of the sequence -- no
    // sampler chain (ESSL 1.00 cannot index one, and VC4 has eight units).
    int nseg = 1;
    int seg = 0;
    int segk = 0;
    // The value kernel adds into its destination instead of overwriting it (a
    // segmented session accumulates one slice at a time).
    bool accumulate = false;
};

// The GLSL that turns `t` (a sequence position, or a pair index for the score
// grid) into a fetch: the sampler for texture number `t / seg`, and the
// normalized coordinate along the sequence axis inside that texture.
// `nseg == 1` returns the caller's own two strings, so a builder that uses this
// emits byte-identical GLSL in the shipped layout.
struct SegRef {
    std::string sampler;   // the sampler expression to fetch from
    std::string y;         // the normalized coordinate along the sequence axis
};
// `single` is the builder's own normalized-x text for the shipped layout; it is
// returned unchanged when nseg == 1, which is what keeps the emitted GLSL
// byte-identical (gate/shader_hashes.sh).
SegRef seg_ref(const AttnConfig& c, const std::string& base, const std::string& t,
               const std::string& single_tex, const std::string& single_y,
               const std::string& div = "SEG");
// The rope table's y: `ropeY` in the shipped layout (the builders' own text),
// the segmented expression otherwise.  One helper because nine fetches in three
// kernels read it and they must agree.
std::string seg_rope_y(const AttnConfig& c);
// The sampler for a one-texel-a-position plane (the rope table, the V step plane).
std::string seg_seq_tex(const AttnConfig& c, const std::string& base);
// `uniform sampler2D a, b;` for the segment chain (nseg == 1: `a`).
std::string seg_uniforms(const AttnConfig& c, const std::string& base);

std::string att_defs(const AttnConfig& c);
// A projection's row into one half of a KV row (cross-attention, v5).
std::string att_cross_kv(const AttnConfig& c, bool v_half, XFormat xf, bool bias);
std::string att_helpers(const AttnConfig& c);

std::string matvec_q8(int rows, int cols, XFormat xf, bool wrow = false);
// The 2-bit ternary body (dtype 7, --quant t3).  ESSL 3.00: the text carries a
// `//T3` marker that link_program() keys on, exactly like the q4 nibble split's
// "&15".  Sixteen weights a texel, four a byte, bit 0 magnitude and bit 1 sign.
std::string matvec_t3(int rows, int cols, XFormat xf, bool wrow, int t3_th, int t3_tiles);
std::string matvec_t3_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                          bool wrow, int t3_th, int t3_tiles);
std::string matvec_t3_pair(int rows, int cols, XFormat xf, bool wrow, int t3_th, int t3_tiles);
std::string matvec_t3_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                               bool wrow, int t3_th, int t3_tiles);
// One draw for a whole batch of matvec inputs (RWKV-7 chunked prefill, P1):
// matvec_q8's text with the input row taken from gl_FragCoord.y, so fragment
// (row r, batch b) is matvec_q8's result for row r of input b.
std::string matvec_q8_batch(int rows, int cols, XFormat xf, bool wrow = false);
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
std::string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                          bool wrow = false);
std::string matvec_q8_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                               bool inl = true, bool wrow = false, bool bias = false);
std::string matvec_i16(int rows, int cols, XFormat xf);
std::string matvec_i16_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of);
std::string matvec_i4(int rows, int cols, XFormat xf, bool inl = true);
// Signed 6-bit head: a 4-bit low plane and a 2-bit high plane in two textures
// (one sampler cannot be fetched twice in a fragment on VC4), one 24-bit value
// per fragment, per-row scale plane when `wrow`.
std::string matvec_q6(int rows, int cols, XFormat xf, bool wrow);
std::string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf, XFormat of);
std::string matvec_i4_4x4(int rows, int cols, XFormat xf, bool inl = true);
std::string matvec_i4_add_4x4(int rows, int cols, XFormat xf, XFormat rf, XFormat of);
std::string matvec_i4_pair(int rows, int cols, XFormat xf);
std::string matvec_i4_pair_swiglu(int rows, int cols, XFormat xf, bool inl = true, int hs = 0);
// Q4_FMT=rgba4: the same gate with the weights bound as 4-bit texels, four per
// texel and zero decode.  A separate builder so the packed text (and its
// shader hash) cannot move.
std::string matvec_i4_pair_swiglu_4x4(int rows, int cols, XFormat xf, bool inl = true, int hs = 0);
std::string matvec_i4_add_pair(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                               bool inl = true);
std::string matvec_i4_add_pair_4x4(int rows, int cols, XFormat xf, XFormat rf, XFormat of,
                                   bool inl = true);

// `hs` is the container's per-layer FFN-hidden scale exponent (metadata key
// 15): the hidden is written as value * 2^-hs so it fits the q16 activation
// rail, and w2's read-back multiplies by 2^hs.  hs = 0 emits exactly the text
// the builder produced before the key existed, so every existing container's
// programs are byte-identical.
std::string swiglu_pair(int hidden, XFormat out, int hs = 0);
std::string swiglu_2row(int hidden, XFormat out, int hs = 0);
// the FFN gate when the weights are 2-bit: `A` is two rows of q16 pairs
std::string swiglu_pair2row(int hidden, XFormat out, int hs = 0);
// EXP=drv|poly: one switch for every exponential the shaders emit (the SiLU in
// the FFN kernels, the sigmoid in the RWKV gate kernels, and the attention
// weights).  See shaders_common.cpp for the evidence.
bool exp_poly_enabled();
std::string exp_poly_source();

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
// The multi-segment value fold: sum `nseg` slices of the same target layout
// into one row, in the format's own arithmetic.
std::string att_v_fold_shader(const AttnConfig& c, XFormat out, int nseg,
                              bool accumulate = false);

std::string top2_fast(int rows, int chunk);
std::string top2(int rows, int chunk);
std::string top2_fast_packed(int rows, int chunk);
std::string top2_packed(int rows, int chunk);
// Greedy argmax over the packed logits target: stage 1 reuses top2_fast_packed;
// these reduce the chunk winners to one winner per tile in two 16-bound
// loops (the VC4-proven length).  Byte-identical to read-back + max_element.
std::string top1_group(int n);
std::string top1_final(int groups);

// ---- RWKV-7 (Goose) GPU layer loop (docs/rwkv7.md) -----------
// The elementwise ops and the WKV7 recurrence of the third layer loop.  All
// activations are I24_1 (one value per texel, 1/256 step) except the residual
// stream, which stays q16 (I16_1) to match the CPU add_q16 grid bit-for-bit
// under the +-128 rail.  Constant per-channel vectors (lerp / bias / scale /
// norm affine) are uploaded scaled by 64 so their 1/256 texel step is 1/16384
// in model units; every reader multiplies by the literal 0.015625.
//
// Two tensors need a finer step than 1/256 and carry an explicit power-of-two
// store scale instead (the plan's W-1 / K-1 / S-1 decisions):
//   decay  stores (1-w)*256   -> w step 1/65536 (W-1: 1.5e-5)
//   kk     stores kk*16       -> kk step 1/4096 (K-1)
//   state  stores S*2^e_head  -> per-head exponent e_head, >=18 effective bits
//                               (S-1: the q16/I24 1/256 grid costs +0.154 nats).

std::string rwkv_ln_stats(int dim, XFormat rf);
std::string rwkv_ln_norm(int dim, XFormat rf);

// Layer 0's input is the token-normed embedding (~0.013 on g1d), far below the
// q16 residual step (1/256 = 30% of it).  The CPU keeps it f32 until the first
// residual add; the GPU stores it *256 in I24_1 and these read it back /256.
std::string rwkv_ln_stats_p(int dim);
std::string rwkv_ln_norm_p(int dim);
std::string rwkv_first_add(int dim);
std::string rwkv_res_add(int dim);

std::string rwkv_gn_stats(int dim, int hs);
std::string rwkv_gn_norm(int dim, int hs);
std::string rwkv_lerp(int dim, int which, int nrows);
std::string rwkv_l2_stats(int dim, int hs);
std::string rwkv_l2_norm(int dim, int hs);
std::string rwkv_keydelta(int dim);
std::string rwkv_tanh(int dim);
std::string rwkv_sigmoid(int dim, bool has_bias, int oscale);
std::string rwkv_decay(int dim);
std::string rwkv_vres(int dim);
std::string rwkv_mul(int dim, int adiv);
std::string rwkv_rk_reduce(int dim, int hs);
std::string rwkv_rk_add(int dim, int hs);
// WKV7 recurrence: three native-ESSL-3.00 passes over a per-head 64x64 state
// held as bit-cast f32 in RGBA8 ping-pong textures (exact recurrence; f32 carries
// its own exponent, so no per-head scale).  sa dots old state with kk; update
// applies S*w + v*k + sa*(kk*a); out dots new state with r.
std::string rwkv_wkv_sa(int heads, int hs);
std::string rwkv_wkv_update(int heads, int hs);
std::string rwkv_wkv_out(int heads, int hs);

// ---- post-norm block row passes --------------------------------------------
// Generic and XFORMAT-clean: LayerNorm (mean removed) in two draws, x += b, and
// the non-gated FFN's swish.  RWKV's rwkv_ln_* above are a different
// implementation with its own bit-cast stats codec; these are the ones a
// post-norm transformer block calls.
std::string ln_stats_split(int dim, XFormat rf);
std::string ln_stats_reduce(int dim);
std::string ln_apply_q16(int dim, XFormat rf, XFormat out);
std::string bias_add_q16(int dim, XFormat xf, XFormat f);
std::string act_swish(int dim, XFormat f);
std::string act_swish_biased(int dim, XFormat xf, XFormat f);
std::string act_sqrelu(int dim);
