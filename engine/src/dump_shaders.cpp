// Emits the GLSL of every shader builder for a fixed matrix of arguments.
// See dump_shaders.h for why this exists; scripts/shader_hashes.sh is the gate.
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
    c.kv_bits = 16;
    c.kv_scale = 16384.0f;
    c.kv_width = 192;
    c.kv_vstep = false;
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
    const XFormat I16_1 = XFormat::I16_1, I16_2 = XFormat::I16_2, I8_4 = XFormat::I8_4;

    // --- q8 matvec family: the shapes the engine actually asks for
    // (qkv 1536, logits 6400, FFN 2432, hidden 768; cols = dim 768).
    add(v, "matvec_q8_1536x768_i16_2", [] { return matvec_q8(1536, 768, I16_2); });
    add(v, "matvec_q8_6400x768_i16_2", [] { return matvec_q8(6400, 768, I16_2); });
    add(v, "matvec_q8_2432x768_i8_4", [] { return matvec_q8(2432, 768, I8_4); });
    add(v, "matvec_q8_2432x768_i16_1", [] { return matvec_q8(2432, 768, I16_1); });
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
        [] { return matvec_q8_add(768, 768, I16_2, I16_1); });
    add(v, "matvec_q8_add_768x768_i8_4_i16_1",
        [] { return matvec_q8_add(768, 768, I8_4, I16_1); });
    add(v, "matvec_q8_add_pair_768x768_i16_2_inl",
        [] { return matvec_q8_add_pair(768, 768, I16_2, true); });
    add(v, "matvec_q8_add_pair_768x768_i16_2_gen",
        [] { return matvec_q8_add_pair(768, 768, I16_2, false); });

    // --- i16 weights (legacy precision, still loadable)
    add(v, "matvec_i16_6400x768_i16_2", [] { return matvec_i16(6400, 768, I16_2); });
    add(v, "matvec_i16_add_1536x768_i16_2_i16_1",
        [] { return matvec_i16_add(1536, 768, I16_2, I16_1); });

    // --- q4
    add(v, "matvec_i4_6400x768_i16_2_inl", [] { return matvec_i4(6400, 768, I16_2, true); });
    add(v, "matvec_i4_6400x768_i16_2_gen", [] { return matvec_i4(6400, 768, I16_2, false); });
    add(v, "matvec_i4_add_768x768_i16_2_i16_1",
        [] { return matvec_i4_add(768, 768, I16_2, I16_1); });
    add(v, "matvec_i4_pair_2432x768_i16_2",
        [] { return matvec_i4_pair(2432, 768, I16_2); });
    add(v, "matvec_i4_pair_swiglu_2432x768_i16_2_inl",
        [] { return matvec_i4_pair_swiglu(2432, 768, I16_2, true); });
    add(v, "matvec_i4_pair_swiglu_2432x768_i16_2_gen",
        [] { return matvec_i4_pair_swiglu(2432, 768, I16_2, false); });
    add(v, "matvec_i4_add_pair_768x768_i16_2_inl",
        [] { return matvec_i4_add_pair(768, 768, I16_2, true); });
    add(v, "matvec_i4_add_pair_768x768_i16_2_gen",
        [] { return matvec_i4_add_pair(768, 768, I16_2, false); });

    // --- elementwise / norm / logits
    add(v, "swiglu_pair_2432_i16_2", [] { return swiglu_pair(2432, I16_2); });
    add(v, "swiglu_pair_2432_i8_4", [] { return swiglu_pair(2432, I8_4); });
    add(v, "swiglu_2row_2432_i16_2", [] { return swiglu_2row(2432, I16_2); });
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
    add(v, "rms_norm_pair2_768_i16_1_lit",
        [] { return rms_norm_pair2(768, I16_1, false); });
    add(v, "rms_norm_pair2_768_i16_1_eps",
        [] { return rms_norm_pair2(768, I16_1, true); });
    add(v, "top2_fast_6400_256", [] { return top2_fast(6400, 256); });
    add(v, "top2_6400_256", [] { return top2(6400, 256); });
    add(v, "top2_fast_packed_6400_256", [] { return top2_fast_packed(6400, 256); });
    add(v, "top2_packed_6400_256", [] { return top2_packed(6400, 256); });

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

    // --- attention, one tag per layout the container can ask for
    add(v, "att_helpers", [] { return att_helpers(); });

    AttnConfig c = attn_base();
    c.kv_bits = 16;
    add_attn(v, "attn_q8_kv16_qkvpair", c);

    c = attn_base();
    c.qkv_pair = false;
    add_attn(v, "attn_q8_kv16_unpaired", c);

    c = attn_base();
    c.kv_bits = 8;
    c.kv_width = 96;
    c.kv_vstep = true;
    add_attn(v, "attn_q4_kv8_vstep", c);

    c = attn_base();
    c.kv_bits = 0;
    c.kv_width = 384;
    add_attn(v, "attn_kv24", c);

    c = attn_base();
    c.qk_norm = false;
    c.rope_half = true;
    add_attn(v, "attn_q8_kv16_qkn0_ropehalf", c);

    c = attn_base();
    c.qkv_pair = false;
    c.kv_bits = 8;
    c.kv_width = 96;
    c.kv_vstep = false;
    add_attn(v, "attn_kv8_nostep_unpaired", c);

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
