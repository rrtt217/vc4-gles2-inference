// Emits the GLSL of every shader builder for a fixed matrix of arguments.
// See dump_shaders.h for why this exists; gate/shader_hashes.sh is the gate.
#include "dump_shaders.h"

#include "gpu.h"
#include "shaders.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using std::string;
using std::to_string;

namespace {

struct Case {
    string name;
    std::function<string()> build;
};

// FNV-1a/64.  Fixed by the algorithm, not by the standard library, so a digest
// means the same thing on the laptop, the Pi and after a toolchain upgrade.
uint64_t digest(const string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

AttnConfig attn_base() {
    // MiniMind3/Qwen3-64M: dim 768, 8 q heads, 4 kv heads, head_dim 96.
    AttnConfig c;
    c.dim = 768;
    c.kv_dim = 384;
    c.heads = 8;
    c.kv_heads = 4;
    c.head_dim = 96;
    c.seq = 768;
    c.rope_half = false;
    c.qk_norm = true;
    c.qkv_pair = true;
    c.kv = KvLayout{KvCodec::Fixed, 16, 2, 192, 16384.0f, false};
    return c;
}

void add(vector<Case>& v, const string& name, std::function<string()> f) {
    v.push_back({name, std::move(f)});
}

void add_attn(vector<Case>& v, const string& tag, const AttnConfig& c) {
    add(v, tag + "/att_defs", [c] { return att_defs(c); });
    add(v, tag + "/att_kv_rope", [c] { return att_kv_rope(c); });
    add(v, tag + "/att_vscale", [c] { return att_vscale(c); });
    add(v, tag + "/att_qnorm_scale", [c] { return att_qnorm_scale(c); });
    add(v, tag + "/att_qrot", [c] { return att_qrot(c); });
    add(v, tag + "/att_score", [c] { return att_score(c); });
    add(v, tag + "/att_stats", [c] { return att_stats(c); });
    add(v, tag + "/att_prob", [c] { return att_prob(c); });
    add(v, tag + "/att_v_i16_2", [c] { return att_v(c, XFormat::I16_2); });
    add(v, tag + "/att_v_i8_4", [c] { return att_v(c, XFormat::I8_4); });
}

vector<Case> cases() {
    vector<Case> v;
    const XFormat I16_1 = XFormat::I16_1, I16_2 = XFormat::I16_2, I8_4 = XFormat::I8_4,
                  I24_1 = XFormat::I24_1;

    // --- q8 matvec family: the shapes the engine actually asks for
    // (qkv 1536, logits 6400, FFN 2432, hidden 768; cols = dim 768).
    add(v, "matvec_q8_1536x768_i16_2", [] { return matvec_q8(1536, 768, I16_2, I24_1); });
    add(v, "matvec_q8_1536x768_i16_2_projf32",
        [] { return matvec_q8(1536, 768, I16_2, XFormat::F32); });
    add(v, "matvec_q8_6400x768_i16_2", [] { return matvec_q8(6400, 768, I16_2, I24_1); });
    add(v, "matvec_q8_2432x768_i8_4", [] { return matvec_q8(2432, 768, I8_4, I24_1); });
    add(v, "matvec_q8_2432x768_i16_1", [] { return matvec_q8(2432, 768, I16_1, I24_1); });
    // The chunked prefill's batched kernel (P1, task-13): matvec_q8's text with
    // the input row from gl_FragCoord.y.  Its own line in the golden file, so a
    // change to either kernel's arithmetic is visible in both.
    add(v, "matvec_q8_batch_768x768_i16_2",
        [] { return matvec_q8_batch(768, 768, I16_2, I24_1); });
    // No per-row-scale (wrow) entry: its unsigned scale decode carries no rail
    // fold and check_q16_decode.sh counts decodes against folds per line, which
    // is also why matvec_q8's own wrow form is not registered here.  The wrow
    // text is wrow_scale()'s, shared verbatim with that kernel.
    add(v, "matvec_q8_batch_64x768_i16_2",
        [] { return matvec_q8_batch(64, 768, I16_2, I24_1); });
    add(v, "matvec_q8_batch_768x3072_i16_2",
        [] { return matvec_q8_batch(768, 3072, I16_2, I24_1); });
    add(v, "matvec_q8_batch_768x768_i24_1",
        [] { return matvec_q8_batch(768, 768, XFormat::I24_1, I24_1); });
    add(v, "matvec_q8_pair_2432x768_i16_2", [] { return matvec_q8_pair(2432, 768, I16_2); });
    add(v, "matvec_q8_pair_2432x768_i8_4", [] { return matvec_q8_pair(2432, 768, I8_4); });
    add(v, "matvec_q8_pair_logits_6400x768_i16_2",
        [] { return matvec_q8_pair_logits(6400, 768, I16_2); });
    add(v, "matvec_q8_pair_swiglu_2432x768_i16_2_inl",
        [] { return matvec_q8_pair_swiglu(2432, 768, I16_2, true); });
    add(v, "matvec_q8_pair_swiglu_2432x768_i16_2_gen",
        [] { return matvec_q8_pair_swiglu(2432, 768, I16_2, false); });
    add(v, "matvec_q8_gate2_2432x768", [] { return matvec_q8_gate2(2432, 768); });
    add(v, "matvec_q8_p2_768x768", [] { return matvec_q8_p2(768, 768); });
    add(v, "matvec_q8_gate2_p2_2432x768", [] { return matvec_q8_gate2_p2(2432, 768); });
    add(v, "matvec_q8_p2_add_p2_768x768", [] { return matvec_q8_p2_add_p2(768, 768); });
    add(v, "matvec_q8_p2_add_768x768_i16_1",
        [] { return matvec_q8_p2_add(768, 768, I16_1); });
    add(v, "p2_split_768_i16_1", [] { return p2_split(768, I16_1); });
    add(v, "p2_split_768_i16_2", [] { return p2_split(768, I16_2); });
    add(v, "matvec_q8_pair_fused_2432x768_i16_2",
        [] { return matvec_q8_pair_fused(2432, 768, I16_2); });
    add(v, "matvec_q8_add_768x768_i16_2_i16_1",
        [] { return matvec_q8_add(768, 768, I16_2, I16_1, I16_1); });
    add(v, "matvec_q8_add_768x768_i8_4_i16_1",
        [] { return matvec_q8_add(768, 768, I8_4, I16_1, I16_1); });
    add(v, "matvec_q8_add_pair_768x768_i16_2_inl",
        [] { return matvec_q8_add_pair(768, 768, I16_2, I16_2, I16_2, true); });
    add(v, "matvec_q8_add_pair_768x768_i16_2_gen",
        [] { return matvec_q8_add_pair(768, 768, I16_2, I16_2, I16_2, false); });

    // --- 2-bit ternary weights (dtype 7).  Two shapes: the FFN gate's 4096 rows
    // tile into four, the qkv's 1280 into two, and the residual add takes both
    // res layouts the engine can hand it.
    add(v, "matvec_t3_1280x1024_i16_2_t1024x2",
        [] { return matvec_t3(1280, 1024, I16_2, I24_1, true, 1024, 2); });
    add(v, "matvec_t3_4096x1024_i16_2_t1024x4",
        [] { return matvec_t3(4096, 1024, I16_2, I24_1, true, 1024, 4); });
    add(v, "matvec_t3_add_1024x1024_i16_2_i16_1",
        [] { return matvec_t3_add(1024, 1024, I16_2, I16_1, I16_1, true, 1024, 1); });
    add(v, "matvec_t3_pair_1280x1024_i16_2_t640x1",
        [] { return matvec_t3_pair(1280, 1024, I16_2, true, 640, 1); });
    add(v, "swiglu_pair2row_4096_i16_2",
        [] { return swiglu_pair2row(4096, I16_2, 0); });
    add(v, "matvec_t3_add_pair_1024x1024_i16_2_i16_2",
        [] { return matvec_t3_add_pair(1024, 1024, I16_2, I16_2, I16_2, true, 512, 1); });
    add(v, "matvec_t3_gate2_p2_1280x1024_t640x1",
        [] { return matvec_t3_gate2_p2(1280, 1024, true, 640, 1); });
    add(v, "matvec_t3_p2_add_1024x1024_i16_2_t512x1",
        [] { return matvec_t3_p2_add(1024, 1024, I16_2, true, 512, 1); });
    add(v, "matvec_t3_p2_add_p2_1024x1024_t512x1",
        [] { return matvec_t3_p2_add_p2(1024, 1024, true, 512, 1); });

    // --- i16 weights (legacy precision, still loadable)
    add(v, "matvec_i16_6400x768_i16_2", [] { return matvec_i16(6400, 768, I16_2, I24_1); });
    add(v, "matvec_i16_add_1536x768_i16_2_i16_1",
        [] { return matvec_i16_add(1536, 768, I16_2, I16_1, I16_1); });

    // --- q4
    add(v, "matvec_i4_6400x768_i16_2_inl", [] { return matvec_i4(6400, 768, I16_2, I24_1, true); });
    add(v, "matvec_i4_6400x768_i16_2_gen", [] { return matvec_i4(6400, 768, I16_2, I24_1, false); });
    add(v, "matvec_i4_add_768x768_i16_2_i16_1",
        [] { return matvec_i4_add(768, 768, I16_2, I16_1, I16_1); });
    add(v, "matvec_i4_pair_2432x768_i16_2",
        [] { return matvec_i4_pair(2432, 768, I16_2); });
    add(v, "matvec_i4_pair_swiglu_2432x768_i16_2_inl",
        [] { return matvec_i4_pair_swiglu(2432, 768, I16_2, true); });
    add(v, "matvec_i4_pair_swiglu_2432x768_i16_2_gen",
        [] { return matvec_i4_pair_swiglu(2432, 768, I16_2, false); });
    add(v, "matvec_i4_add_pair_768x768_i16_2_inl",
        [] { return matvec_i4_add_pair(768, 768, I16_2, I16_2, I16_2, true); });
    add(v, "matvec_i4_add_pair_768x768_i16_2_gen",
        [] { return matvec_i4_add_pair(768, 768, I16_2, I16_2, I16_2, false); });
    // The q4 two-position family (4-bit view only), the same shapes as q8's.
    add(v, "matvec_i4_gate2_p2_2432x768",
        [] { return matvec_i4_gate2_p2_4x4(2432, 768); });
    add(v, "matvec_i4_p2_add_768x768_i16_1",
        [] { return matvec_i4_p2_add_4x4(768, 768, I16_1); });
    add(v, "matvec_i4_p2_add_p2_768x768",
        [] { return matvec_i4_p2_add_p2_4x4(768, 768); });
    // The 4-bit (Q4_FMT=rgba4) kernels, the default q4 path.  The gate is the
    // flat 8-wide form (byte-identical to the nested, ~43% faster on VC4).
    add(v, "matvec_i4_pair_swiglu_4x4_1216x768_i16_2",
        [] { return matvec_i4_pair_swiglu_4x4(1216, 768, I16_2, true, 0); });
    add(v, "matvec_i4_pair_swiglu_4x4_2432x768_i16_2",
        [] { return matvec_i4_pair_swiglu_4x4(2432, 768, I16_2, true, 0); });
    add(v, "matvec_i4_4x4_1536x768_i16_2",
        [] { return matvec_i4_4x4(1536, 768, I16_2, I24_1, true); });
    add(v, "matvec_i4_add_4x4_768x1216_i16_2_i16_1",
        [] { return matvec_i4_add_4x4(768, 1216, I16_2, I16_1, I16_1); });

    // --- elementwise / norm / logits
    add(v, "swiglu_pair_2432_i16_2", [] { return swiglu_pair(2432, I16_2); });
    add(v, "swiglu_pair_2432_i8_4", [] { return swiglu_pair(2432, I8_4); });
    add(v, "swiglu_2row_2432_i16_2", [] { return swiglu_2row(2432, I16_2); });
    // The two formats with a codec of their own: bf16 and f32 had **no** entry
    // here, so `gate/shader_hashes.sh` could not see the day one of them
    // changed (`x_f32r` was a hand copy and nothing compared it).  One program
    // each, at the shape the FFN's hidden uses.
    {
        // Marian (v5): 512 wide, 8 heads of 64, no rotation, no qk-norm, and the
        // KV axis at q16 -- the shape `encdec_encode` builds it in.  Built from
        // `attn_base()` so every field `need_attn` checks is filled.
        AttnConfig c = attn_base();
        c.dim = 512; c.kv_dim = 512; c.heads = 8; c.kv_heads = 8; c.head_dim = 64;
        c.seq = 64; c.rope_none = true; c.qk_norm = false; c.kv.bits = 16; c.kv.vpt = 2;
        c.kv.texels = 512;
        add(v, "att_cross_kv_512", [c] { return att_cross_kv(c, false, XFormat::I24_1, false); });
    }
    add(v, "act_swish_biased_2048_f24_i16", []
        { return act_swish_biased(2048, XFormat::I24_1, I16_2); });
    add(v, "swiglu_2row_2432_bf16", [] { return swiglu_2row(2432, XFormat::BF16); });
    add(v, "swiglu_2row_2432_f32", [] { return swiglu_2row(2432, XFormat::F32); });
    add(v, "swiglu_2432_fused_i16_2", [] { return swiglu(2432, true, I16_2); });
    add(v, "swiglu_2432_split_i8_4", [] { return swiglu(2432, false, I8_4); });
    add(v, "rms_stats_split_768_i16_1", [] { return rms_stats_split(768, I16_1); });
    add(v, "rms_stats_split_768_i8_4", [] { return rms_stats_split(768, I8_4); });
    add(v, "rms_stats_reduce", [] { return rms_stats_reduce(); });
    add(v, "rms_norm_768_i16_1_i8_4",
        [] { return rms_norm(768, I16_1, I8_4, false, false); });
    add(v, "rms_norm_768_i16_2_i16_2_fused_eps",
        [] { return rms_norm(768, I16_2, I16_2, true, true); });
    add(v, "rms_stats2_768_i16_1", [] { return rms_stats2(768, I16_1); });
    add(v, "rms_stats_reduce2_32", [] { return rms_stats_reduce2(32); });
    add(v, "rms_norm_pair2_768_i16_1_lit",
        [] { return rms_norm_pair2(768, I16_1, false, true); });
    add(v, "rms_norm_pair2_768_i16_1_eps",
        [] { return rms_norm_pair2(768, I16_1, true, true); });
    // The split-reduce backends quantise the total through S24_1 before the
    // normalise (rms_stats_reduce's round trip); these two are that text.
    add(v, "rms_norm_pair2_768_i16_1_lit_q",
        [] { return rms_norm_pair2(768, I16_1, false, false); });
    add(v, "rms_norm_pair2_768_i16_1_eps_q",
        [] { return rms_norm_pair2(768, I16_1, true, false); });
    add(v, "top2_fast_6400_256", [] { return top2_fast(6400, 256, I24_1); });
    add(v, "top2_fast_6400_256_projf32", [] { return top2_fast(6400, 256, XFormat::F32); });
    add(v, "top2_6400_256", [] { return top2(6400, 256, I24_1); });
    add(v, "top2_6400_256_projf32", [] { return top2(6400, 256, XFormat::F32); });
    add(v, "top2_fast_packed_6400_256", [] { return top2_fast_packed(6400, 256); });
    add(v, "top2_packed_6400_256", [] { return top2_packed(6400, 256); });
    add(v, "top1_group_96", [] { return top1_group(96); });
    add(v, "top1_final_6", [] { return top1_final(6); });

    // --- the hybrid (conv) block.  Its three elementwise passes are the only
    // shaders whose text decides the conv numerics, and they were not in this
    // dump: a change to conv3_act's tap order or to the gate's operand layouts
    // moved the model without moving a digest.
    add(v, "conv3_1024", [] { return conv3_act(1024, XFormat::I16_2); });
    add(v, "mul_act_1024_24_24",
        [] { return mul_act(1024, XFormat::I16_2, XFormat::I24_1, XFormat::I24_1); });
    add(v, "mul_act_1024_24_16",
        [] { return mul_act(1024, XFormat::I16_2, XFormat::I24_1, XFormat::I16_2); });
    add(v, "conv_copy_1024", [] { return copy_act(1024, XFormat::I16_2); });
    // The BiCrossGate FFN (metadata key 23).  Registered because the block's
    // kernels are where a format gets hardcoded by accident: `bicross_act` stored
    // a hand-picked q16 while the axis said bf16, and `mul2` wrote the activation
    // stream where the FFN_HID stream belongs.  Neither would have shown up in a
    // digest that did not cover them.
    add(v, "bicross_384_gelu_silu_i16", [] { return bicross_act(384, XFormat::I24_1,
                                                                XFormat::I16_2,
                                                                XFormat::I16_2, 0x102); });
    add(v, "bicross_384_gelu_silu_bf16", [] { return bicross_act(384, XFormat::I24_1,
                                                                 XFormat::BF16,
                                                                 XFormat::BF16, 0x102); });
    add(v, "mul2_1152_i24_i24_to_i16", [] { return mul2(1152, XFormat::I16_2,
                                                        XFormat::I24_1, XFormat::I24_1); });
    add(v, "mul2_1152_i24_i24_to_f32", [] { return mul2(1152, XFormat::F32,
                                                        XFormat::I24_1, XFormat::I24_1); });
    // The fused replacement, with and without a per-row weight scale: this is
    // the kernel BICROSS_FUSE=0 exists to compare against, and the reason it is
    // digested is that a fused kernel that drifts is a wrong FFN two stages
    // later with nothing in between to see.
    add(v, "q8_mulgate_1152x384_i16_i24", [] { return matvec_q8_ffn_mulgate(1152, 384,
                                                                          XFormat::I16_2,
                                                                          XFormat::I24_1); });
    add(v, "q8_mulgate_1152x384_i16_i24_wr", [] { return matvec_q8_ffn_mulgate(1152, 384,
                                                                             XFormat::I16_2,
                                                                             XFormat::I24_1,
                                                                             true); });

    // --- attention, one tag per layout the container can ask for
    AttnConfig c = attn_base();
    add(v, "att_helpers", [c] { return att_helpers(c); });

    c = attn_base();
    c.kv.bits = 16; c.kv.vpt = 2;
    add_attn(v, "attn_q8_kv16_qkvpair", c);

    c = attn_base();
    c.qkv_pair = false;
    add_attn(v, "attn_q8_kv16_unpaired", c);

    c = attn_base();
    c.kv.bits = 8; c.kv.vpt = 4;
    c.kv.texels = 96;
    c.kv.vstep = true;
    add_attn(v, "attn_q4_kv8_vstep", c);

    c = attn_base();
    c.kv.bits = 0; c.kv.vpt = 1;
    c.kv.texels = 384;
    add_attn(v, "attn_kv24", c);

    // Long context: one segmented program per segment kind, so a change to the
    // SEGK/SEG arithmetic is a digest and not a model difference.
    c = attn_base();
    c.kv.bits = 16; c.kv.vpt = 2;
    c.nseg = 2;
    c.seg = 1024;
    c.segk = 1;
    add_attn(v, "attn_q8_kv16_seg2", c);

    // The value fold, for two and three slices: its text is where a slice-count
    // bug shows up, and it is the only kernel whose sampler count depends on
    // nseg.
    add(v, "att_v_fold_2", [] { AttnConfig f = attn_base(); return att_v_fold_shader(f, XFormat::I16_2, 2); });
    add(v, "att_v_fold_add2", [] { AttnConfig f = attn_base(); return att_v_fold_shader(f, XFormat::I16_2, 2, true); });
    add(v, "att_v_fold_3", [] { AttnConfig f = attn_base(); return att_v_fold_shader(f, XFormat::I16_2, 3); });

    c = attn_base();
    c.qk_norm = false;
    c.rope_half = true;
    add_attn(v, "attn_q8_kv16_qkn0_ropehalf", c);

    c = attn_base();
    c.qkv_pair = false;
    c.kv.bits = 8; c.kv.vpt = 4;
    c.kv.texels = 96;
    c.kv.vstep = false;
    add_attn(v, "attn_kv8_nostep_unpaired", c);

    // --- RWKV-7 (P6): the elementwise ops and the WKV7 recurrence.  dim 768,
    // head_size 64, 12 heads (the 0.1B g1d shape).  The residual is q16 (I16_1).
    add(v, "ln_stats_768_i16_1", [] { return rwkv_ln_stats(768, I16_1); });
    add(v, "ln_stats_p_768", [] { return rwkv_ln_stats_p(768); });
    add(v, "ln_norm_p_768", [] { return rwkv_ln_norm_p(768); });
    add(v, "rwkv_first_add_768", [] { return rwkv_first_add(768); });
    add(v, "rwkv_res_add_768", [] { return rwkv_res_add(768); });
    add(v, "ln_norm_768_i16_1", [] { return rwkv_ln_norm(768, I16_1); });
    add(v, "rwkv_gn_stats_768_64", [] { return rwkv_gn_stats(768, 64); });
    add(v, "rwkv_gn_norm_768_64", [] { return rwkv_gn_norm(768, 64); });
    add(v, "rwkv_lerp_768_0", [] { return rwkv_lerp(768, 0, 6); });
    add(v, "rwkv_lerp_768_3", [] { return rwkv_lerp(768, 3, 6); });
    add(v, "rwkv_lerp_768_cm", [] { return rwkv_lerp(768, 0, 1); });
    add(v, "rwkv_l2_stats_768_64", [] { return rwkv_l2_stats(768, 64); });
    add(v, "rwkv_l2_norm_768_64", [] { return rwkv_l2_norm(768, 64); });
    add(v, "rwkv_keydelta_768", [] { return rwkv_keydelta(768); });
    add(v, "rwkv_tanh_768", [] { return rwkv_tanh(768); });
    add(v, "rwkv_sigmoid_768_bias16", [] { return rwkv_sigmoid(768, true, 16); });
    add(v, "rwkv_sigmoid_768_nobias1", [] { return rwkv_sigmoid(768, false, 1); });
    add(v, "rwkv_decay_768", [] { return rwkv_decay(768); });
    add(v, "rwkv_vres_768", [] { return rwkv_vres(768); });
    add(v, "act_sqrelu_768", [] { return act_sqrelu(768); });
    // post-norm block row passes.  512 is Marian's width; the kernels are
    // dim-parameterised, so one case each is what the digest covers.
    add(v, "ln_split_768_i16_1", [] { return ln_stats_split(768, I16_1); });
    add(v, "ln_reduce_768", [] { return ln_stats_reduce(768); });
    add(v, "ln_apply_q16_512", [] { return ln_apply_q16(512, XFormat::I16_1, XFormat::I24_1); });
    add(v, "bias_add_q16_512", [] { return bias_add_q16(512, XFormat::I24_1, I16_1); });
    add(v, "act_swish_i24_512", [] { return act_swish(512, XFormat::I24_1); });
    add(v, "rwkv_mul_768", [] { return rwkv_mul(768, 1); });
    add(v, "rwkv_mul_768_s256", [] { return rwkv_mul(768, 256); });
    add(v, "rwkv_rk_reduce_768_64", [] { return rwkv_rk_reduce(768, 64); });
    add(v, "rwkv_rk_add_768_64", [] { return rwkv_rk_add(768, 64); });
    add(v, "rwkv_wkv_sa_12_64", [] { return rwkv_wkv_sa(12, 64); });
    add(v, "rwkv_wkv_update_12_64", [] { return rwkv_wkv_update(12, 64); });
    add(v, "rwkv_wkv_out_12_64", [] { return rwkv_wkv_out(12, 64); });

    return v;
}

}  // namespace

int dump_shaders(bool raw) {
    vector<Case> cs = cases();
    // The q4 nibble decode has two forms: the ESSL 3.00 shift/and split and the
    // floor chain.  Which one ships is a driver query, so a builder's text is
    // not a property of the source alone.  Force both and keep the digest set
    // unambiguous: a name without a prefix means the text is the same either
    // way.
    gpu_force_q4_shift(false);
    vector<string> plain(cs.size());
    for (size_t i = 0; i < cs.size(); ++i) plain[i] = cs[i].build();

    gpu_force_q4_shift(true);
    vector<string> shifted(cs.size());
    for (size_t i = 0; i < cs.size(); ++i) shifted[i] = cs[i].build();
    gpu_force_q4_shift(false);

    vector<std::pair<string, const string*>> out;
    out.reserve(cs.size());
    for (size_t i = 0; i < cs.size(); ++i) {
        if (shifted[i] == plain[i]) {
            out.push_back({cs[i].name, &plain[i]});
        } else {
            out.push_back({"q4shift0/" + cs[i].name, &plain[i]});
            out.push_back({"q4shift1/" + cs[i].name, &shifted[i]});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<string, const string*>& a,
                 const std::pair<string, const string*>& b) { return a.first < b.first; });

    size_t bytes = 0;
    for (const auto& e : out) bytes += e.second->size();

    if (raw) {
        for (const auto& e : out) {
            printf("==== %s ====\n%s\n", e.first.c_str(), e.second->c_str());
        }
    } else {
        for (const auto& e : out) {
            printf("%016llx %7zu %s\n", (unsigned long long)digest(*e.second),
                   e.second->size(), e.first.c_str());
        }
    }
    fprintf(stderr, "# shader dump: %zu programs, %zu bytes of GLSL\n", out.size(), bytes);
    return 0;
}
