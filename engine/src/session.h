#pragma once

#include "common.h"
#include <cstdint>
#include "model.h"
#include "gpu.h"
#include <string>
#include <vector>

class Session {
public:
    Session(Model& model, GPU& gpu);

    // Rebuild this session for the model the same object now holds.  Used by the
    // REPL's /model: the GPU context and its compiled programs stay, the scene
    // (textures, FBOs, scratch targets) is released by GPU::release_scene()
    // before this runs, so every target is recreated from the new dimensions.
    void reset();

    // A cache whose token prefix is not fully reused must not leave its conv
    // history behind: the history is a running state, and replaying the prompt
    // from token 0 with a history from another token stream is wrong.
    void zero_conv_state();

    // system_fnv / tools_fnv are hashes of the inputs this cache was built from
    // (0 when the cache came from a prompt file).  They go into the sidecar so a
    // rebuild trigger compares hashes instead of re-rendering the header.
    bool save_prefix_cache(const string& path, const vector<int>& toks,
                           uint64_t system_fnv = 0, uint64_t tools_fnv = 0);

    bool load_prefix_cache(const string& path, vector<int>& toks);

    // exact_f32: keep the row off the q16 activation grid.  An RWKV container's
    // embedding peaks at 0.08, where the grid's 1/256 step is 5% of the row --
    // and the block's first op is a LayerNorm, which divides by the row's
    // deviation and amplifies exactly that.  The transformer CPU path has the
    // same trap behind ACT_F32; the fused path never sees it because its
    // texture is q16 by construction.
    vector<float> embedding(int token, bool exact_f32 = false);

    vector<float> forward(int token, int pos);

    GLuint forward_gpu_hidden(int token, int pos, bool want_logits);

    vector<float> step(int token, int pos, bool want_logits);

    // Greedy fast path: run one position and return the winning token id
    // without reading the vocabulary back.  When the GPU reduction is not
    // available (CPU path, unpacked logits, GPU_ARGMAX=0) it falls back to
    // step() + argmax_token, so the answer is the same either way.  Returns
    // -1 when g_stop interrupted the forward pass.  The caller must still
    // apply its own eligibility rules (penalties, marker masks, LOGITS_OUT,
    // logprobs) -- this only guarantees the argmax itself is unchanged.
    int step_greedy(int token, int pos);
    bool greedy_fast_ok() const;

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
    // The conv block's gated history: (k-1) rows of `dim` per conv layer, the
    // oldest first.  Decode reads it and then shifts in the new row.
    vector<vector<float>> conv_state_;
    vector<float> rope_cos_, rope_sin_;

    // ---- RWKV-7 (P2, docs/rwkv7-plan.zh-CN.md) ----------------------------
    // An RWKV layer has no KV cache: its whole history is one head_size x
    // head_size recurrence per head plus the two token-shift vectors, which is
    // why the state is a fixed-size block rather than [seq x dim].  0.1B:
    // 12 * 12 * 64 * 64 floats = 2.36 MB.  f32 on purpose -- the plan's S-1
    // measures the q16 grid on this state at +0.1543 nats/token and greedy
    // 6/8, so it is the one tensor P2 must not quantise.
    struct RwkvState {
        vector<float> wkv;        // [layers][heads][head_size][head_size]
        vector<float> shift_att;  // [layers][dim]: previous token's att_norm out
        vector<float> shift_ffn;  // [layers][dim]: previous token's ffn_norm out
        vector<float> v_first;    // [dim]: layer 0's value, mixed in from layer 1
        bool have_v_first = false;
    };
    RwkvState rwkv_state_;

    // One token through the block, returning the residual stream after the last
    // layer.  The four big matvecs and the logits matvec go through the GPU;
    // every elementwise op and the WKV recurrence run here in f32 (P2 is the
    // correctness path -- P6 is the fused/GPU one).
    vector<float> forward_rwkv(int token);
    // P6: the fused GPU layer loop (RWKV_GPU=1).  Elementwise ops and the WKV7
    // recurrence run on textures; only the final residual crosses back per token.
    vector<float> forward_gpu_rwkv(int token);
    bool rwkv_gpu_ = false;
    bool rwkv_wkv_cpu_ = false;   // RWKV_WKV_CPU=1: hybrid (GPU elementwise + CPU f32 WKV)
    // The output LayerNorm, the metadata key 17 head-input scale and the logits
    // matvec: one place, so step() and step_candidates() cannot disagree.
    vector<float> rwkv_head_input(const vector<float>& x);
    vector<float> step_rwkv(int token, bool want_logits);
    // A matvec through the GPU, with the input normalised by a power of two so
    // it cannot cross the q16 rail (the token-shift lerps reach 225).  The
    // compensating factor goes into the matrix's read-back scale, *not* into
    // the result -- GPU::matvec_s does it the other way round, which multiplies
    // the output before the kernel's own clamp and would break this block's
    // 90-magnitude outputs.  See the comment on the definition.
    vector<float> rwkv_mv(const Mat& m, const vector<float>& x);
    // LayerNorm with weight and bias, f32 statistics (O1): the mean and the
    // biased variance over the whole row, eps from key 16.
    static vector<float> rwkv_layernorm(const vector<float>& x, const float* wb, float eps);
    void zero_rwkv_state();

    vector<float> rms_norm(const vector<float>& x, const float* w);
    // A/B probe: the GPU rmsnorm (fixed-point partial sums) instead of the
    // CPU f32 mean, so the only changed variable is the norm's statistics.
    vector<float> rms_gpu(const vector<float>& x, int layer, bool ffn);
    vector<float> rms_final_gpu(const vector<float>& x);
    // A/B probe: read a residual-layout target back as floats.
    vector<float> read_hidden(const Target& t);

    vector<float> head_rms_norm(const vector<float>& v, int n_heads, const float* w);

    vector<float> rope(const vector<float>& v, int n_heads, int pos);

    vector<float> attention(const vector<float>& q, int layer, int pos);

    vector<float> add_q16(const vector<float>& a, const vector<float>& b);

    vector<float> swiglu(const vector<float>& h, int n);

    vector<float> swiglu(const vector<float>& h);
};
