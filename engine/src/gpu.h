#pragma once

#include "common.h"
#include "xformat.h"
#include "shaders.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <string>
#include <unordered_map>
#include <vector>


// Wall clock since main() started; a static default keeps the probe usable
// before the engine sets it (it then only measures from first use).
void gpu_set_run_t0(Clock::time_point t);
// True when the context offers ESSL 3.00 and the q4 shift decode is enabled.
bool gpu_q4_shift();
// True when the context offers ESSL 3.00 (the RWKV bit-cast f32 state needs it).
bool gpu_es3_ok();
// Dump-only override: --dump-shaders has no context to query, and the q4
// builders emit different text for the two decodes, so it forces both.
void gpu_force_q4_shift(bool on);
double run_time_ms();
int gpu_upload_textures();
size_t gpu_scene_bytes();   // textures this scene has uploaded so far
double gpu_upload_texture_ms();

class Model;

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
    // Host<->GPU synchronisations: every glFinish plus every glReadPixels on the
    // hot path (SYNCCOUNT prints them; docs/test-hooks.md).  A draw is cheap to
    // issue and a sync is not, so the two are counted separately.
    long syncs() const { return syncs_; }
    GPU();
    ~GPU();

    void upload(Mat& m);

    // Everything this context created that is *scene* rather than *program*:
    // weight textures, per-layer tables, scratch/FBO targets, the upload cache.
    // Shader programs are the expensive part (docs/performance.md, "Where the
    // first token goes": 13 links cost 1.89 s on the board) and they survive
    // release_scene(), which is what makes replacing a model in one process
    // worth anything at all.  Handles are recorded as they are created, so a
    // swap cannot silently leave a texture behind (that would be a CMA leak on
    // the Pi, where one model barely fits).
    void release_scene();
    // Back to the constructor's format plan.  A new Session runs this first, so
    // a swap cannot inherit the previous model's ACT8/RES_PACK choices.
    void reset_format_plan();
    // Zeroes every scratch Target (textures and FBOs are in the ledger).  Kept
    // next to release_scene() because the two always run together, and
    // gate/check_target_reset.sh keeps the list complete.
    void reset_targets();

    vector<float> matvec(const Mat& m, const vector<float>& x);
    // Same, with the activation scaled by 2^k so a small input keeps its
    // resolution on the q16 grid (see the comment in gpu.cpp).
    vector<float> matvec_s(const Mat& m, const vector<float>& x);

    GLuint upload_input(const vector<float>& x);
    // Prefill batch: position A then position B in one texture (see upload_input2).
    GLuint upload_input2(const vector<float>& a, const vector<float>& b);

    void matvec_to(const Mat& m, GLuint x, Target& out);

    void matvec_to_row(const Mat& m, GLuint x, Target& out, int row);

    vector<float> matvec_from(const Mat& m, GLuint x);

    // `hs` = the layer's FFN-hidden store scale exponent (metadata key 15):
    // the SwiGLU output is written *2^-hs and w2's scale was already folded by
    // 2^hs in Model::apply_hidden_scale().  hs = 0 is the historical path.
    GLuint ffn_split(const Mat& w1, const Mat& w3, GLuint x, int hidden, int hs = 0);
    GLuint ffn_fused(const Mat& w13, GLuint x, int hidden, int hs = 0);

    // Paired gate matvec + swiglu for one FFN chunk.
    GLuint ffn_pair_chunk(const Mat& w1, const Mat& w3, GLuint x, int hidden, Target& pair_buf, Target& out, int hs = 0);
    GLuint ffn_pair_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out, int hs = 0);
    // Two-position variants: one weight fetch feeds two positions (item 7b).
    GLuint ffn_pair_swiglu_chunk2(const Mat& w1, const Mat& w3, GLuint x2, int rows, Target& out);
    void matvec_p2(const Mat& m, GLuint xp2, Target& out);
    // Prefill pair (WORKLOG 7c): pack two positions activations into P2, run
    // w2 for both with the residual add, and split the result back out.
    GLuint rmsnorm_pair2(GLuint xa, GLuint xb, GLuint w, Target& out);
    GLuint ffn_pair_swiglu_chunk2_p2(const Mat& w1, const Mat& w3, GLuint xp2, int rows, Target& out);
    void matvec_p2_add_p2(const Mat& m, GLuint xp2, GLuint rp2, Target& out);
    void matvec_p2_add(const Mat& m, GLuint xp2, GLuint ra, GLuint rb, Target& out);
    void p2_split(GLuint xp2, Target& out, int which);
    GLuint ffn_pair_i4_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out, int hs = 0);

    void ffn_split_chunked(const Mat& w1a, const Mat& w3a, const Mat& w1b, const Mat& w3b,
                           GLuint x, int n1, int n2, GLuint& ha, GLuint& hb, int hs = 0);

    GLuint swiglu_2row(Target& src, int hidden, int hs = 0);

    GLuint swiglu_2row(Target& src, int hidden, Target& out, int hs = 0);
    // The 2-bit FFN gate: `ab` holds the gate in row 0 and the up projection in
    // row 1, each as q16 pairs (one texel = rows 2t, 2t+1).
    GLuint swiglu_pair2row(Target& src, int hidden, Target& out, int hs = 0);
    GLuint ffn_t3_pair_chunk(const Mat& w1, const Mat& w3, GLuint x, int hidden,
                             Target& ab, Target& out, int hs = 0);
    void matvec_pair_to_row(const Mat& m, GLuint x, Target& out, int row);

    GLuint swiglu_split(Target& a, Target& b);

    GLuint swiglu_fused(Target& src, int hidden, int hs = 0);

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
                        const vector<float>& rcos, const vector<float>& rsin, bool qkv_pair,
                        bool hybrid = false, const std::vector<uint8_t>* layer_type = nullptr);

    GLuint attention_block(const Mat& qkv, GLuint x, int layer, int pos);

    // Prefill query batching (PREFILL_BATCH, docs/prefill-batching.md): the
    // attention-internal passes of N positions fuse into one score draw while
    // the residual chain stays serial.  att_qkv_draw + kv_rope_write +
    // att_qrot_stage run per position (phase 1), att_scores_batch once per
    // group (phase 2), att_stats_row / att_v_row per position against the
    // batched score texture (phase 3).  Every kernel is the serial text with
    // the row substituted, so the logits are byte-identical.
    void att_qkv_draw(const Mat& qkv, GLuint x);
    // Phase 1 of a group, one position: qkv draw, V grid, KV write, rotated Q
    // staged into row b.  The serial attention_block's draws in the same order.
    void att_stage(const Mat& qkv, GLuint x, int layer, int pos, int b, int n);
    void att_qrot_stage(int layer, int pos, int b, int n);
    void att_scores_batch(int layer, int pos0, int n);
    void att_stats_row(int pos, int b, int n);
    void att_v_row(int layer, int pos, int b, int n);
    // N hidden upload slots and the per-position residual targets the group
    // loop ping-pongs through (grp_tmp_ is the shared wo output / x1 scratch).
    GLuint upload_hidden_grp(const vector<float>& x, int b, int n);
    Target& grp_h(int b, int n);
    Target& grp_tmp();
    Target& grp_w2a();

    void init_persistent(int dim, int layers, const vector<float>& rms_att,
                        const vector<float>& rms_ffn, const vector<float>& rms_final);

    GLuint upload_hidden(const vector<float>& x);
    // Second hidden upload slot, so two positions can be in flight at once.
    GLuint upload_hidden_b(const vector<float>& x);

    GLuint rmsnorm(GLuint x, GLuint w, Target& out);

    void matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out, XFormat xf);

    // LFM2 conv block.  The three in_proj row slices are drawn into their own
    // 1024-wide targets (a 3072-wide draw would exceed VC4's 2048), the gates
    // and the 3-tap convolution are elementwise passes, and the k-1 rows of
    // state live in three rotating targets per conv layer: the newest row is
    // written straight into the slot that held x[pos-k+1], which is dead, so a
    // token costs no state copies at all (it used to be two fullscreen draws
    // per conv layer).  conv_cur_ names the slot holding x[pos-1].
    void init_conv(int layers, int dim);
    Target conv_t_b_, conv_t_c_, conv_t_x_, conv_cv_, conv_y_;
    std::vector<Target> conv_state_;        // [layer*3 + slot]
    std::vector<uint8_t> conv_cur_;         // newest slot per layer
    std::vector<GLuint> conv_k_tex_;
    int conv_texels_ = 0;
    Target& conv_slot(int layer, int slot) { return conv_state_[(size_t)layer * 3 + slot]; }
    GLuint conv_mul(GLuint a, XFormat af, GLuint b, XFormat bf, Target& out);
    GLuint conv3(GLuint s0, GLuint s1, GLuint g, GLuint ktex, Target& out);
    // Per-channel depthwise kernel as q16: width dim/2, height k, row t holds
    // tap t for channels 2q and 2q+1.
    void set_conv_kernel(int layer, const float* w, int dim, int k);
    // The prefix cache carries the conv history: one row per state texture.
    int conv_state_row_bytes() const { return conv_texels_ * 4; }
    bool read_conv_state(int layer, int which, vector<uint8_t>& out);
    bool write_conv_state(int layer, int which, const vector<uint8_t>& in);
    void clear_conv_state();
    GLuint conv_copy(GLuint src, Target& out);
    GLuint conv_block(const Mat& in_b, const Mat& in_c, const Mat& in_x,
                      const Mat& out_proj, GLuint xb, GLuint residual,
                      int layer, Target& out);

    // RWKV-7 (Goose) fused GPU layer loop (docs/rwkv7.md).
    // init_rwkv uploads the per-channel constant vectors (*64 into I24_1) and
    // allocates the per-head WKV state ping-pong, the token-shift rows and the
    // scratch targets; it also switches the format plan to I24_1 activations /
    // q16 residual.  rwkv_block runs one whole layer on textures and returns the
    // new residual texel row; the only host read-back is the final residual once
    // per token.  RWKV_GPU=1 selects it (Session::forward_gpu_rwkv).
    void init_rwkv(const Model& m);
    GLuint rwkv_block(int layer, const Model& m, GLuint residual, float* cpu_wkv_state = nullptr);
    // One host sync for a group of matvecs: each is drawn into its own row of one
    // scratch target and the block is read back with a single glReadPixels.  The
    // draws, their programs and their uniforms are unchanged, so the result is
    // bit-identical to calling matvec() once per matrix -- the CPU path's cost is
    // the read-back, and it paid one per matvec (docs/rwkv7.md).
    // rows[i].x is uploaded to its own staging texture (slot i), so several
    // inputs of the same length cannot overwrite each other.
    struct MvRow { const Mat* m; const vector<float>* x; };
    // `read_mode` is the VC4 bisect (RWKV_GROUP): 0 = read the block at the
    // group's own width, 1 = read it at the target's full width, 2 = one read
    // per matrix row (no sync win; isolates the multi-row read).
    vector<vector<float>> matvec_rows(const vector<MvRow>& rows, int slot0 = 0, int read_mode = 0);
    // P1 (task-13): a *batched* matvec -- one draw per matrix for a whole group
    // of inputs, instead of one draw per input.  xs holds N input rows; they are
    // uploaded as N rows of one staging texture and the kernel takes its input
    // row from gl_FragCoord.y (matvec_q8_batch), so fragment (row r, input b)
    // is matvec_q8's result for row r of input b.  Each input keeps its own 2^k
    // normalisation: rows are grouped by k, each group is one draw with the
    // matrix scale folded exactly as rwkv_mv() folds it, so every output row is
    // bit-identical to rwkv_mv() on that input.  One read-back per call.
    struct MvBatch { const Mat* m; const vector<const vector<float>*>* xs; };
    vector<vector<vector<float>>> matvec_batch_group(const vector<MvBatch>& groups);
    // The batch's staging texture (in_w x groups*N) and its read-back block
    // (max rows x groups*N); both grow monotonically, like rwkv_mvrows_.
    Target rwkv_batch_in_;
    Target rwkv_batch_out_;
    GLuint rwkv_upload_slot(const vector<float>& x, int slot);
    // RWKV_PROFILE (docs/test-hooks.md): where a serial-path token's time goes.
    // Wall time around the three GL steps of a matvec, filled only when the
    // switch is on.  On a needs_finish() driver the upload finishes inside its
    // own step, so the read-back is what carries the GPU work.
    struct RwkvProf { double upload_ms = 0, draw_ms = 0, read_ms = 0; };
    RwkvProf rwkv_prof;
    bool rwkv_prof_on() const;   // RWKV_PROFILE, read once
    // The fused loop's per-stage totals, printed once per token (RWKV_PROFILE).
    void rwkv_prof_fused_summary();
    void rwkv_flip() { rwkv_cur_ ^= 1; }
    void clear_rwkv_state();
    // Read the final q16 residual row back as floats (once per token).
    std::vector<float> rwkv_read_residual(GLuint tex);

    vector<float> logits_from(const vector<Mat>& tiles, GLuint x);

    // Greedy fast path: draw the logits, reduce to the winning token on the
    // GPU and read back one texel per tile instead of the whole vocabulary.
    // Only the packed q16 logits layout is supported (logits_argmax_ok);
    // the reduction is byte-identical to read_logits_full + max_element.
    // GPU_ARGMAX=0 refuses it so the two paths can be A/B-ed in one binary.
    bool logits_argmax_ok(const vector<Mat>& tiles) const;
    int logits_argmax_from(const vector<Mat>& tiles, GLuint x);

    // PASS_DUP=logits_draw probe: re-run the logits draw only, without the
    // finish/read-back, so the draw and the sync can be priced separately.
    void logits_pass_only(const vector<Mat>& tiles, GLuint x);

    vector<uint8_t> read_bytes(Target& t, int w, int h);

    void sync();
    // Drop the bind caches.  Any code that touches glBindFramebuffer /
    // glBindTexture directly (prefix-cache save/load, debug dumps) must call
    // this, or a later bind_tex()/bind_fbo() will skip a needed bind.
    void forget_state();
    void dump_target(const Target& t, const char* name, int n);
    // CPU path: the same bytes and the same printer, so HIDDEN_DUMP under
    // GPU_LAYER=0 is diffable against GPU_LAYER=1 (see dump_row in gpu.cpp).
    void dump_cpu_row(const vector<float>& v, XFormat f, const char* name, int n);
    // ATT_DUMP=<pos> probe: the exact integers the score/softmax path consumed
    // and produced, for every layer at one position.  qrot and the K row are in
    // their q16 grids (value = int/256 and int/KVSCALE), scores in the 1/1024
    // units att_score writes, stats raw as att_stats wrote them.  Offline that
    // is checkable with a dot product and a softmax, no second engine path.
    void dump_attention(int layer, int pos);
    // Decoded per-row q16 integers of the qkv target, hashed.  Layout-agnostic
    // (24-bit row texels and packed q16 pairs hash alike), so QKV_PAIR=0 and
    // QKV_PAIR=1 runs can be compared exactly for every layer and position.
    std::vector<int> qkv_values(int rows);
    unsigned long long qkv_hash(int rows);
    bool needs_finish() const;
    bool rms_fused() const;
    int max_target_dim() const { return max_target_dim_; }
    // How many programs this context has compiled.  A model swap keeps them, so
    // this is the quantity that makes the second start cheaper than the first.
    size_t program_count() const { return programs_.size(); }
    // RMSNorm epsilon: the container's `rms_norm_eps` when it states one, else
    // the historical 1e-5 the shaders used to hardcode.
    void set_rms_eps(float e, bool stated) { rms_eps_ = e > 0.0f ? e : 1e-5f; rms_eps_stated_ = stated; }
    float rms_eps() const { return rms_eps_; }

private:
    AttnConfig att_cfg() const;
    GLuint x_prog(const string& key, const string& src);
    GLuint x_q8(int rows, int cols, bool wrow = false);
    GLuint x_q8_batch(int rows, int cols, bool wrow);
    GLuint x_q6(int rows, int cols, bool wrow = false);
    GLuint x_q8_pair(int rows, int cols, bool wrow = false);
    GLuint x_q8_pair_swiglu(int rows, int cols, bool wrow = false, int hs = 0);
    GLuint x_q8_gate2(int rows, int cols);
    GLuint x_q8_p2(int rows, int cols, bool wrow = false);
    GLuint x_rms_stats2();
    GLuint x_rms_norm_pair2(bool use_eps_uniform);
    GLuint x_q8_gate2_p2(int rows, int cols, bool wrow = false);
    GLuint x_q8_p2_add_p2(int rows, int cols, bool wrow = false);
    GLuint x_q8_p2_add(int rows, int cols, bool wrow = false);
    GLuint x_p2_split(int rows);
    GLuint x_q8_pair_fused(int rows, int cols, bool wrow = false);
    GLuint x_i4_pair(int rows, int cols);
    GLuint x_i4_pair_swiglu(int rows, int cols, int hs = 0, bool fmt4 = false);
    GLuint x_q8_add(int rows, int cols, XFormat xf, bool wrow = false);
    GLuint x_t3(const Mat& m, bool wrow = false);
    GLuint x_t3_add(const Mat& m, XFormat rf, bool wrow = false);
    GLuint x_t3_pair(const Mat& m, bool wrow = false);
    GLuint x_t3_add_pair(const Mat& m, XFormat rf, bool wrow = false);
    GLuint x_q8_add_pair(int rows, int cols, XFormat xf, bool wrow = false);
    GLuint x_i4_add_pair(int rows, int cols, XFormat xf, bool fmt4 = false);
    GLuint x_i16(int rows, int cols);
    GLuint x_i16_add(int rows, int cols, XFormat xf);
    GLuint x_i4(int rows, int cols, bool fmt4 = false);
    GLuint x_i4_add(int rows, int cols, XFormat xf, bool fmt4 = false);
    GLuint x_swiglu_pair(int hidden, int hs = 0);
    GLuint x_swiglu_2row(int hidden, int hs = 0);
    GLuint x_swiglu(int hidden, bool fused, int hs = 0);
    GLuint x_rms_split();
    GLuint x_rms_reduce();
    GLuint x_rms_norm(bool use_eps_uniform);
    GLuint x_v();
    GLuint x_vscale();
    GLuint x_kv_rope();
    GLuint x_qnorm();
    GLuint x_qrot();
    GLuint x_score();
    GLuint x_stats();
    // qbatch variants: the score grid is seq/2 x heads*n; stats/v address the
    // batched score rows through the qrow0 uniform (n baked into the text).
    GLuint x_score_batch(int n);
    GLuint x_stats_row(int n);
    GLuint x_v_row(int n);
    GLuint x_prob();
    GLuint x_top2_fast(int rows, int chunk);
    GLuint x_top2(int rows, int chunk);
    GLuint x_q8_pair_logits(int rows, int cols, bool wrow = false);
    GLuint x_top2_fast_packed(int rows, int chunk);
    GLuint x_top2_packed(int rows, int chunk);
    GLuint x_top1_group(int n);
    GLuint x_top1_final(int groups);
    bool logits_pair() const;
    bool logits_pair_ok(const vector<Mat>& tiles) const;
    void logits_pass(const vector<Mat>& tiles, GLuint x);
    vector<float> read_logits_full(const vector<Mat>& tiles);

    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    unordered_map<string, GLuint> programs_;
    // Which shader text each key in programs_ was built from.  A key alone is
    // not enough: the same key produces different GLSL for another model or
    // format plan, and the entry used to be returned without looking at the
    // text at all (found by swapping a model: the second model ran the first
    // model's attention shaders).
    unordered_map<string, size_t> prog_key_hash_;
    unordered_map<size_t, GLuint> text_programs_;  // program text hash -> program
    struct ULocRec { const char* name; GLint loc; };
    unordered_map<GLuint, vector<ULocRec>> uloc_cache_;
    GLuint uloc_prog_ = 0;
    vector<ULocRec>* uloc_tab_ = nullptr;
    GLint uloc(GLuint prog, const char* name);
    void draw_tri();
    long draws_ = 0;
    long syncs_ = 0;
    void note_sync() { ++syncs_; }
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
    unordered_map<int, GLuint> input_textures2_;  // pair-of-positions activation
    unordered_map<int, Target> targets_;
    Target logits_target_;
    Target candidate_target_;
    // Greedy argmax reduction: chunk winners, group winners, tile winners.
    Target argmax_chunk_, argmax_group_, argmax_tile_;
    Target sum_target_;
    Target ffn_a_, ffn_b_, ffn_h_, ffn_ab_;
    Target ffn_ab_a_, ffn_ab_b_, ffn_h_a_, ffn_h_b_;
    Target qkv_target_, att_scores_, att_stats_, att_probs_, att_out_, qrot_target_, qscale_target_;
    // Prefill query batching: the staged rotated Q (dim/2 x n) and the batched
    // score texture (seq/2 x heads*n), plus the group's residual store.
    Target qrot_batch_, scores_batch_;
    vector<Target> grp_h_;
    Target grp_tmp_, grp_w2a_;
    vector<GLuint> grp_in_;
    // 8-bit KV: one per-head V step texel per position, one plane per layer.
    vector<Target> vs_targets_;
    vector<Target> kv_cache_;
    Target hidden_a_, hidden_b_, xb_, xbf_, xf_, rms_stats_, rms_partial_, rms_partial2_, ffn_pair_a_, ffn_pair_b_;
    // Prefill pair (7c): two hidden streams, their post-attention scratch and
    // the FFN chunk-0 intermediate.  pp_a_ is also the layer output slot, so
    // the three roles rotate inside one layer and never feedback.  pp_a3_/pp_b3_
    // have been unused since 7d, where the chunk-1 residual stays in P2.
    Target pp_a_, pp_a2_, pp_a3_, pp_b_, pp_b2_, pp_b3_;
    Target pp_in2_, pp_g_, pp_w2_, pp_w2b_, xb2_, xbf2_;
    bool layer_fast_finish_ = false;
    bool res_pack_ = false;   // residual stream holds two int16 per texel
    XFormat act_fmt_ = XFormat::I16_2;  // matvec input (qkv/wo/gate/logits)
    XFormat ffn_fmt_ = XFormat::I16_2;  // FFN hidden state (w2 input)
    XFormat res_fmt_ = XFormat::I16_1;  // residual stream layout
    GLuint hidden_in_ = 0;
    GLuint hidden_in_b_ = 0;
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
    float rms_eps_ = 1e-5f;
    // A container that states an epsilon makes the shader read a uniform; one
    // that does not keeps the literal 1e-5, because turning a constant into a
    // uniform is *not* a no-op on this driver (measured: PPL 270.3170 ->
    // 269.9303 with eps numerically identical).
    bool rms_eps_stated_ = false;
    float att_kv_scale_ = 16384.0f;
    int att_kv_fmt_ = 0;                 // KV_FMT: 0 fixed grid, 1 bf16
    float att_score_scale_ = 1024.0f;   // ATT_SCALE: score/(max) grid
    float att_sum_scale_ = 32.0f;       // ATT_SUM_SCALE: exp-sum grid (linear)
    bool att_sum_log_ = false;          // ATT_SUM=log2: sum stored as log2
    int att_kv_width_ = 0;
    bool att_kv_vstep_ = false;   // 8-bit KV: per-head V grid
    int max_target_dim_ = 0;
    vector<GLuint> q_norm_tex_, k_norm_tex_;
    GLuint vbo_ = 0;
    Target probe_;
    string renderer_;

    // ---- RWKV-7 GPU scene (P6) -------------------------------------------
    // Scratch targets indexed by RwkvTmp; per-layer WKV state / scale ping-pong
    // and token-shift rows; the uploaded constant vectors.  All released by
    // release_scene()/reset_targets() like the conv scene.
    enum RwkvTmp { RT_XN, RT_XN2, RT_XR, RT_XW, RT_XK, RT_XV, RT_XA, RT_XG,
        RT_R, RT_K, RT_KD, RT_V, RT_V2, RT_TW, RT_TW2, RT_TA, RT_TV, RT_TG, RT_TG2, RT_U,
        RT_DECAY, RT_A, RT_G, RT_KK, RT_Y, RT_Y2, RT_CMH, RT_CMH2, RT_CMH3, RT_CMH4,
        RT_LNSTAT, RT_GNSTAT, RT_L2STAT, RT_RK, RT_SA,
        RT_RES, RT_RES2, RT_COUNT };
    std::vector<Target> rwkv_tmp_;
    std::vector<Target> rwkv_state_;   // [layer*2 + buf], hs x (heads*hs), bit-cast f32
    std::vector<Target> rwkv_shift_;   // [layer*2 + (0 att, 1 ffn)], dim x 1
    std::vector<GLuint> rwkv_const_;   // per-layer constant textures (RwkvConst)
    Target rwkv_vfirst_;
    // layer-0 input, *256 I24_1 (precise; the q16 grid would round it coarsely)
    Target rwkv_x0_;
    // The grouped matvecs' read-back block (matvec_rows) and their staging
    // textures, one slot per input so equal lengths cannot collide.
    Target rwkv_mvrows_;
    vector<GLuint> rwkv_in_tex_;
    vector<int> rwkv_in_w_;            // each staging texture's width, in texels
    int rwkv_cur_ = 0;                 // state/scale ping-pong parity, flips per token
    int rwkv_dim_ = 0, rwkv_layers_ = 0, rwkv_heads_ = 0, rwkv_hs_ = 0;
    bool rwkv_have_vfirst_ = false;
    float rwkv_eps_ = 1e-5f;
    enum RwkvConst { RC_LERP, RC_W0, RC_A0, RC_V0, RC_SCALE, RC_ATTNorm,
        RC_ATTNorm2, RC_LNX, RC_CMLERP, RC_COUNT };
    GLuint rwkv_const(int layer, RwkvConst c) const {
        return rwkv_const_[(size_t)layer * RC_COUNT + c];
    }
    Target& rwkv_tmp(RwkvTmp i) { return rwkv_tmp_[i]; }
    // Granular passes (gpu_rwkv.cpp); each writes a pre-sized scratch target.
    GLuint rwkv_mv(const Mat& m, GLuint x, Target& out);
    GLuint rwkv_ln(GLuint x, GLuint wb, Target& out);
    GLuint rwkv_ln_p(GLuint x, GLuint wb, Target& out);
    GLuint rwkv_first_add(GLuint x0, GLuint delta, Target& out);
    GLuint rwkv_res_add(GLuint resid, GLuint delta, Target& out);
    GLuint rwkv_upload_x0(const std::vector<float>& x0);
    GLuint rwkv_gn(GLuint y, GLuint wb, Target& out);
    GLuint rwkv_lerp(GLuint xn, GLuint sh, GLuint lp, int which, int nrows, Target& out);
    GLuint rwkv_l2(GLuint k, GLuint sc, Target& out);
    GLuint rwkv_keydelta(GLuint k, GLuint a, GLuint sc, Target& out);
    GLuint rwkv_tanh(GLuint x, Target& out);
    GLuint rwkv_sigmoid(GLuint x, GLuint bias, Target& out, int oscale);
    GLuint rwkv_decay(GLuint u, GLuint w0, Target& out);
    GLuint rwkv_vres(GLuint v, GLuint vf, GLuint u, GLuint v0, Target& out);
    GLuint rwkv_sqrelu(GLuint x, Target& out);
    GLuint rwkv_mul(GLuint a, GLuint b, Target& out, int adiv);
    GLuint rwkv_rk(GLuint k, GLuint r, GLuint sc, GLuint v, GLuint y, Target& out);
    GLuint rwkv_wkv(int layer, GLuint r, GLuint k, GLuint v, GLuint a, GLuint kk,
                    GLuint decay, Target& out);
    void rwkv_copy(GLuint src, Target& dst);
    void rwkv_draw1(GLuint prog, int w, int h = 1);


    void init_egl();

    void init_gl();

    static GLuint compile_shader(GLenum type, const string& src);
    static bool shader_time_on();
    static double shader_time_ms();

    static GLuint link_program(const string& fs);
    void check_program_runs(GLuint p, const string& key);

    static GLuint make_texture(int w, int h, const void* data);
    // INT4 matrices as 4-bit texels (Q4_FMT=rgba4): same bytes, twice the
    // width, four weights per texel.  Measured on the board 2026-11-14:
    // accepted, exact, and the sample lands in reverse column order (GL packs
    // R into the most significant nibble).
    static GLuint make_q4_texture(int w, int h, const void* data);

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

    void kv_vscale(int layer, int pos);

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
    // One row of it, starting at `y` (the group bisect's per-row arm).
    vector<int32_t> read_i32_rows(Target& t, int w, int y, int h);
};
