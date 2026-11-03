#pragma once

#include "common.h"
#include "xformat.h"
#include "shaders.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <string>
#include <unordered_map>
#include <vector>

struct Candidate { float value; int token; };
struct CandidateResult {
    vector<Candidate> cand;
    vector<float> full;
    bool complete = false;
    float logsumexp = 0.0f;
    float max_logit = 0.0f;
    float denom = 1.0f;
    bool strict = false;
};

class GPU {
    friend class Session;
public:
    long draws() const { return draws_; }
    GPU();
    ~GPU();

    void upload(Mat& m);

    vector<float> matvec(const Mat& m, const vector<float>& x);

    GLuint upload_input(const vector<float>& x);

    void matvec_to(const Mat& m, GLuint x, Target& out);

    void matvec_to_row(const Mat& m, GLuint x, Target& out, int row);

    vector<float> matvec_from(const Mat& m, GLuint x);

    GLuint ffn_split(const Mat& w1, const Mat& w3, GLuint x, int hidden);
    GLuint ffn_fused(const Mat& w13, GLuint x, int hidden);

    // Paired gate matvec + swiglu for one FFN chunk.
    GLuint ffn_pair_chunk(const Mat& w1, const Mat& w3, GLuint x, int hidden, Target& pair_buf, Target& out);
    GLuint ffn_pair_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out);

    void ffn_split_chunked(const Mat& w1a, const Mat& w3a, const Mat& w1b, const Mat& w3b,
                           GLuint x, int n1, int n2, GLuint& ha, GLuint& hb);

    GLuint swiglu_2row(Target& src, int hidden);

    GLuint swiglu_2row(Target& src, int hidden, Target& out);

    GLuint swiglu_split(Target& a, Target& b);

    GLuint swiglu_fused(Target& src, int hidden);

    vector<float> logits(const vector<Mat>& tiles, const vector<float>& x);

    CandidateResult logits_candidates(const vector<Mat>& tiles, const vector<float>& x,
                                      int topk, float temp, float top_p);

    CandidateResult logits_candidates_from(const vector<Mat>& tiles, GLuint xt,
                                           int topk, float temp, float top_p);

    bool gpu_attn_enabled() const;

    bool gpu_layer_enabled() const;

    void init_attention(int dim, int kv_dim, int heads, int kv_heads, int head_dim,
                        int seq, bool rope_half, bool qk_norm, int layers,
                        const vector<float>& qn, const vector<float>& kn,
                        const vector<float>& rcos, const vector<float>& rsin, bool qkv_pair);

    GLuint attention_block(const Mat& qkv, GLuint x, int layer, int pos);

    void init_persistent(int dim, int layers, const vector<float>& rms_att,
                        const vector<float>& rms_ffn, const vector<float>& rms_final);

    GLuint upload_hidden(const vector<float>& x);

    GLuint rmsnorm(GLuint x, GLuint w, Target& out);

    void matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out, XFormat xf);

    vector<float> logits_from(const vector<Mat>& tiles, GLuint x);

    vector<uint8_t> read_bytes(Target& t, int w, int h);

    void sync();
    void dump_target(const Target& t, const char* name, int n);
    bool needs_finish() const;
    bool rms_fused() const;
    int max_target_dim() const { return max_target_dim_; }

private:
    AttnConfig att_cfg() const;
    GLuint x_prog(const string& key, const string& src);
    GLuint x_q8(int rows, int cols);
    GLuint x_q8_pair(int rows, int cols);
    GLuint x_q8_pair_swiglu(int rows, int cols);
    GLuint x_q8_pair_fused(int rows, int cols);
    GLuint x_i4_pair(int rows, int cols);
    GLuint x_q8_add(int rows, int cols, XFormat xf);
    GLuint x_q8_add_pair(int rows, int cols, XFormat xf);
    GLuint x_i4_add_pair(int rows, int cols, XFormat xf);
    GLuint x_i16(int rows, int cols);
    GLuint x_i16_add(int rows, int cols, XFormat xf);
    GLuint x_i4(int rows, int cols);
    GLuint x_i4_add(int rows, int cols, XFormat xf);
    GLuint x_swiglu_pair(int hidden);
    GLuint x_swiglu_2row(int hidden);
    GLuint x_swiglu(int hidden, bool fused);
    GLuint x_rms_split();
    GLuint x_rms_reduce();
    GLuint x_rms_norm();
    GLuint x_v();
    GLuint x_kv_rope();
    GLuint x_qnorm();
    GLuint x_qrot();
    GLuint x_score(bool hoist);
    GLuint x_stats();
    GLuint x_prob();
    GLuint x_sum();
    GLuint x_top2_fast(int rows, int chunk);
    GLuint x_top2(int rows, int chunk);
    GLuint x_q8_pair_logits(int rows, int cols);
    GLuint x_top2_fast_packed(int rows, int chunk);
    GLuint x_top2_packed(int rows, int chunk);
    bool logits_pair() const;
    bool logits_pair_ok(const vector<Mat>& tiles) const;
    void logits_pass(const vector<Mat>& tiles, GLuint x);
    vector<float> read_logits_full(const vector<Mat>& tiles);

    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    unordered_map<string, GLuint> programs_;
    struct ULocRec { const char* name; GLint loc; };
    unordered_map<GLuint, vector<ULocRec>> uloc_cache_;
    GLuint uloc_prog_ = 0;
    vector<ULocRec>* uloc_tab_ = nullptr;
    GLint uloc(GLuint prog, const char* name);
    void draw_tri();
    long draws_ = 0;
    GLint aloc(GLuint prog, const char* name);
    void bind_tex(int unit, GLuint tex);
    void bind_fbo(GLuint fbo);
    void use_program(GLuint prog);
    void set_sampler(GLuint prog, const char* name, int unit);
    void setup_attrib(GLuint prog);
    bool attrib_ready_ = false;
    GLint attrib_p_ = -1;
    unordered_map<GLuint, vector<ULocRec>> aloc_cache_;
    bool logits_packed_ = false;
    GLuint aloc_prog_ = 0;
    vector<ULocRec>* aloc_tab_ = nullptr;
    struct SampRec { const char* name; int unit; };
    unordered_map<GLuint, vector<SampRec>> sampler_set_;
    unordered_map<int, GLuint> input_textures_;
    unordered_map<int, Target> targets_;
    Target logits_target_;
    Target candidate_target_;
    Target sum_target_;
    Target ffn_a_, ffn_b_, ffn_h_, ffn_ab_;
    Target ffn_ab_a_, ffn_ab_b_, ffn_h_a_, ffn_h_b_;
    Target qkv_target_, att_scores_, att_stats_, att_probs_, att_out_, qrot_target_, qscale_target_;
    vector<Target> kv_cache_;
    Target hidden_a_, hidden_b_, xb_, xbf_, xf_, rms_stats_, rms_partial_, ffn_pair_a_, ffn_pair_b_;
    bool layer_fast_finish_ = false;
    bool res_pack_ = false;   // residual stream holds two int16 per texel
    XFormat act_fmt_ = XFormat::I16_2;  // matvec input (qkv/wo/gate/logits)
    XFormat ffn_fmt_ = XFormat::I16_2;  // FFN hidden state (w2 input)
    XFormat res_fmt_ = XFormat::I16_1;  // residual stream layout
    GLuint hidden_in_ = 0;
    vector<GLuint> rms_att_tex_, rms_ffn_tex_;
    GLuint rms_final_tex_ = 0;
    GLuint rope_tex_ = 0;
    int att_dim_ = 0, att_kv_dim_ = 0, att_heads_ = 0, att_kv_heads_ = 0;
    int att_head_dim_ = 0, att_seq_ = 0;
    bool att_rope_half_ = false;
    bool att_qk_norm_ = false;
    bool att_qkv_pair_ = false;  // qkv target holds two q16 per texel
    bool att_kv_packed_ = false;
    int att_kv_bits_ = 0;
    float att_kv_scale_ = 16384.0f;
    int att_kv_width_ = 0;
    int max_target_dim_ = 0;
    vector<GLuint> q_norm_tex_, k_norm_tex_;
    GLuint vbo_ = 0;
    string renderer_;

    void init_egl();

    void init_gl();

    static GLuint compile_shader(GLenum type, const string& src);

    static GLuint link_program(const string& fs);

    static GLuint make_texture(int w, int h, const void* data);

    static GLuint make_q24_texture(const vector<float>& x, float scale);

    static GLuint make_q16_texture(const vector<float>& x);
    static GLuint make_q16_pair_texture(const vector<float>& x);

    Target make_target(int w, int h);

    Target& target_for(int rows);

    GLuint upload_activations(const vector<float>& x);





    // q4: one RGBA8 texel holds 8 nibbles; load it once and unroll all 8.


    // w1 row r and w3 row r in one fragment, sharing the activation decode.


    // SwiGLU over the paired gate: (gate, up) already share a texel.




    // Two residual rows per fragment: rows 2t/2t+1 share one activation decode
    // and one residual fetch, and are written back as two int16 in one texel.





    // q4 + residual: same 8-nibble unroll, but output must stay q16.
















    void layer_finish(const char* skip_env = nullptr);

    void att_finish();



    // qk-norm factor per head: was recomputed inside every score fragment.


    // Rotated + normed q, one fragment per (head, dim); position-only work.










    void kv_rope_write(int layer, int pos);

    void att_scores(int layer, int pos);

    void att_stats(int pos);

    void att_probs(int pos);

    void att_v(int layer, int pos);

    // Parallel sum of squares: the serial version is one fragment looping DIM
    // times, so every iteration pays full texture latency (~3.5 ms per call).
    // Splitting it over RMS_PARTIALS fragments cuts the serial chain by 32x.
    static const int RMS_PARTIALS = 32;









    void draw(const Mat& m, GLuint x, Target& t, int row = 0);

    vector<int32_t> read_i32(Target& t, int w, int h);
};
