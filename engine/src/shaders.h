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
    int kv_bits = 0;
    float kv_scale = 16384.0f;
    int kv_width = 0;
};

std::string att_defs(const AttnConfig& c);
std::string att_helpers();

std::string matvec_q8(int rows, int cols, XFormat xf);
std::string matvec_q8_pair(int rows, int cols, XFormat xf);
std::string matvec_q8_add(int rows, int cols, XFormat xf, XFormat rf);
std::string matvec_q8_add_pair(int rows, int cols, XFormat xf);
std::string matvec_i16(int rows, int cols, XFormat xf);
std::string matvec_i16_add(int rows, int cols, XFormat xf, XFormat rf);
std::string matvec_i4(int rows, int cols, XFormat xf);
std::string matvec_i4_add(int rows, int cols, XFormat xf, XFormat rf);
std::string matvec_i4_pair(int rows, int cols, XFormat xf);
std::string matvec_i4_add_pair(int rows, int cols, XFormat xf);

std::string swiglu_pair(int hidden, XFormat out);
std::string swiglu_2row(int hidden, XFormat out);
std::string swiglu(int hidden, bool fused, XFormat out);

std::string rms_stats_split(int dim, XFormat residual);
std::string rms_stats_reduce();
std::string rms_norm(int dim, XFormat residual, XFormat out);

std::string att_kv_rope(const AttnConfig& c);
std::string att_qnorm_scale(const AttnConfig& c);
std::string att_qrot(const AttnConfig& c);
std::string att_score(const AttnConfig& c, bool hoist);
std::string att_stats(const AttnConfig& c);
std::string att_prob(const AttnConfig& c);
std::string att_v(const AttnConfig& c, XFormat out);

std::string sum_logits();
std::string top2_fast(int rows, int chunk);
std::string top2(int rows, int chunk);
