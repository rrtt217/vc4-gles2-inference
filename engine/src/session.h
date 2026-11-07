#pragma once

#include "common.h"
#include "model.h"
#include "gpu.h"
#include <string>
#include <vector>

class Session {
public:
    Session(Model& model, GPU& gpu);

    bool save_prefix_cache(const string& path, const vector<int>& toks);

    bool load_prefix_cache(const string& path, vector<int>& toks);

    vector<float> embedding(int token);

    vector<float> forward(int token, int pos);

    GLuint forward_gpu_hidden(int token, int pos, bool want_logits);

    vector<float> step(int token, int pos, bool want_logits);

    // Prefill batching (WORKLOG 7c): run one layer pass for two consecutive
    // positions.  Returns B logits when want_logits; a pair never contains the
    // last prompt token, whose logits are the only ones the caller needs.
    vector<float> step_pair(int tok_a, int pos_a, int tok_b, int pos_b, bool want_logits);

    bool pair_ok() const;

    CandidateResult step_candidates(int token, int pos, int topk, float temp, float top_p);

private:
    Model& m_;
    GPU& gpu_;
    bool gpu_attn_ = false;
    bool gpu_layer_ = false;
    vector<vector<float>> kv_cache_k_, kv_cache_v_;
    vector<float> rope_cos_, rope_sin_;

    vector<float> rms_norm(const vector<float>& x, const float* w);

    vector<float> head_rms_norm(const vector<float>& v, int n_heads, const float* w);

    vector<float> rope(const vector<float>& v, int n_heads, int pos);

    vector<float> attention(const vector<float>& q, int layer, int pos);

    vector<float> add_q16(const vector<float>& a, const vector<float>& b);

    vector<float> swiglu(const vector<float>& h, int n);

    vector<float> swiglu(const vector<float>& h);
};
