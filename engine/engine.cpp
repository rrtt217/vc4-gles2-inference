// Clean GLES2 inference engine for llama2.c, Q8M1 and I16M1 checkpoints.
// Targets OpenGL ES 2.0 (Raspberry Pi VC4) and desktop GLES2 implementations.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef GL_COLOR_ATTACHMENT1
#define GL_COLOR_ATTACHMENT1 0x8CE1
#endif

using namespace std;
using Clock = chrono::steady_clock;
static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return chrono::duration<double, milli>(b - a).count();
}

// Ctrl+C must not tear down an active VC4 context from a signal handler.
// Set a flag, let the current GPU call/token finish, then exit normally.
static volatile sig_atomic_t g_stop = 0;
static void on_stop_signal(int) { g_stop = 1; }
static void install_stop_handlers() {
    struct sigaction sa {};
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // no SA_RESTART: interrupt getline()
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

static constexpr float ACT_SCALE = 256.0f; // q16 activation scale
static constexpr uint32_t MAGIC_Q8 = 0x51384D31;   // Q8M1
static constexpr uint32_t MAGIC_I16 = 0x4931364D;  // I16M1
static constexpr uint32_t MAGIC_Q8_V2 = 0x51384D32;  // Q8M2: theta/half/qk_norm
static constexpr uint32_t MAGIC_I16_V2 = 0x4931364E; // I16N: theta/half/qk_norm
static constexpr uint32_t MAGIC_Q4 = 0x51344D31;    // Q4M1: q4 linear + q8 embed/logits

static inline int16_t quantize_act(float x) {
    long v = lrintf(x * ACT_SCALE);
    if (v < -32768) v = -32768;
    if (v > 32767) v = 32767;
    return (int16_t)v;
}
static inline float dequantize_act(float q) { return q / ACT_SCALE; }
static inline float quantize_act_f(float x) { return (float)quantize_act(x) / ACT_SCALE; }

enum class WeightType { INT8, INT16, INT4 };

struct Mat {
    int rows = 0, cols = 0;
    WeightType type = WeightType::INT8;
    vector<int8_t> q8;
    vector<int16_t> q16;
    vector<uint8_t> q4;       // packed nibbles, low nibble first
    vector<int16_t> q4_scale, q4_min; // per row/group, quantized
    int group = 32, groups = 0;
    float q4_param_scale = 1.0f;
    float scale = 1.0f;
    GLuint tex = 0, q4_param_tex = 0;
};

struct Target {
    GLuint tex = 0, fbo = 0;
    int w = 0, h = 0;
};

class Session;

// -----------------------------------------------------------------------------
// GPU: EGL setup, weight upload, matvec and logits passes.
// -----------------------------------------------------------------------------
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
    GPU() { init_egl(); init_gl(); }
    ~GPU() {
        if (dpy_ == EGL_NO_DISPLAY) return;
        glFinish();
        eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (ctx_ != EGL_NO_CONTEXT) {
            eglDestroyContext(dpy_, ctx_);
            ctx_ = EGL_NO_CONTEXT;
        }
        eglTerminate(dpy_);
        dpy_ = EGL_NO_DISPLAY;
    }

    void upload(Mat& m) {
        if (m.type == WeightType::INT8) {
            vector<uint8_t> data((size_t)m.rows * m.cols);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (uint8_t)((int)m.q8[i] + 128);
            m.tex = make_texture(m.cols / 4, m.rows, data.data());
        } else if (m.type == WeightType::INT16) {
            const int tex_w = (m.cols + 1) / 2;
            vector<uint8_t> data((size_t)m.rows * tex_w * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int c = 0; c < m.cols; ++c) {
                    uint16_t u = (uint16_t)m.q16[(size_t)r * m.cols + c];
                    int o = ((size_t)r * tex_w + c / 2) * 4;
                    if ((c & 1) == 0) { data[o] = u & 255; data[o + 1] = u >> 8; }
                    else               { data[o + 2] = u & 255; data[o + 3] = u >> 8; }
                }
            }
            m.tex = make_texture(tex_w, m.rows, data.data());
        } else {
            m.tex = make_texture((m.cols + 7) / 8, m.rows, m.q4.data());
            vector<uint8_t> pd((size_t)m.rows * m.groups * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int g = 0; g < m.groups; ++g) {
                    uint16_t sc = (uint16_t)m.q4_scale[(size_t)r * m.groups + g];
                    uint16_t mn = (uint16_t)m.q4_min[(size_t)r * m.groups + g];
                    size_t o = ((size_t)r * m.groups + g) * 4;
                    pd[o + 0] = sc >> 8; pd[o + 1] = sc & 255;
                    pd[o + 2] = mn >> 8; pd[o + 3] = mn & 255;
                }
            }
            m.q4_param_tex = make_texture(m.groups, m.rows, pd.data());
        }
        if (!getenv("KEEP_CPU_WEIGHTS")) {
            vector<int8_t>().swap(m.q8);
            vector<int16_t>().swap(m.q16);
            if (m.type == WeightType::INT4) {
                vector<uint8_t>().swap(m.q4);
                vector<int16_t>().swap(m.q4_scale);
                vector<int16_t>().swap(m.q4_min);
            }
        }
    }

    vector<float> matvec(const Mat& m, const vector<float>& x) {
        Target& t = target_for(m.rows);
        GLuint xt = upload_activations(x);
        draw(m, xt, t);
        if (needs_finish()) { glFlush(); glFinish(); }
        vector<int32_t> raw = read_i32(t, m.rows, 1);
        vector<float> out(m.rows);
        for (int i = 0; i < m.rows; ++i) {
            int v = raw[i];
            if (v >= 32768) v -= 65536;
            out[i] = (float)v / ACT_SCALE;
        }
        return out;
    }

    GLuint upload_input(const vector<float>& x) { return upload_activations(x); }

    void matvec_to(const Mat& m, GLuint x, Target& out) {
        matvec_to_row(m, x, out, 0);
    }

    void matvec_to_row(const Mat& m, GLuint x, Target& out, int row) {
        if (out.tex == 0 || out.w != m.rows || out.h <= row) out = make_target(m.rows, row + 1);
        draw(m, x, out, row);
        layer_finish("NF_MAT");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
    }

    vector<float> matvec_from(const Mat& m, GLuint x) {
        Target& t = target_for(m.rows);
        draw(m, x, t);
        vector<int32_t> raw = read_i32(t, m.rows, 1);
        vector<float> out(m.rows);
        for (int i = 0; i < m.rows; ++i) {
            int v = raw[i]; if (v >= 32768) v -= 65536;
            out[i] = (float)v / ACT_SCALE;
        }
        return out;
    }

    GLuint ffn_split(const Mat& w1, const Mat& w3, GLuint x, int hidden) {
        if (ffn_ab_.tex == 0 || ffn_ab_.w != hidden || ffn_ab_.h != 2) ffn_ab_ = make_target(hidden, 2);
        matvec_to_row(w1, x, ffn_ab_, 0);
        matvec_to_row(w3, x, ffn_ab_, 1);
        return swiglu_2row(ffn_ab_, hidden);
    }
    GLuint ffn_fused(const Mat& w13, GLuint x, int hidden) {
        matvec_to(w13, x, ffn_a_);
        return swiglu_fused(ffn_a_, hidden);
    }

    void ffn_split_chunked(const Mat& w1a, const Mat& w3a, const Mat& w1b, const Mat& w3b,
                           GLuint x, int n1, int n2, GLuint& ha, GLuint& hb) {
        if (ffn_ab_a_.tex == 0 || ffn_ab_a_.w != n1) ffn_ab_a_ = make_target(n1, 2);
        if (ffn_ab_b_.tex == 0 || ffn_ab_b_.w != n2) ffn_ab_b_ = make_target(n2, 2);
        matvec_to_row(w1a, x, ffn_ab_a_, 0);
        matvec_to_row(w3a, x, ffn_ab_a_, 1);
        ha = swiglu_2row(ffn_ab_a_, n1, ffn_h_a_);
        matvec_to_row(w1b, x, ffn_ab_b_, 0);
        matvec_to_row(w3b, x, ffn_ab_b_, 1);
        hb = swiglu_2row(ffn_ab_b_, n2, ffn_h_b_);
    }

    GLuint swiglu_2row(Target& src, int hidden) {
        return swiglu_2row(src, hidden, ffn_h_);
    }

    GLuint swiglu_2row(Target& src, int hidden, Target& out) {
        if (out.tex == 0 || out.w != hidden) out = make_target(hidden, 1);
        GLuint p = swiglu_2row_program(hidden);
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, hidden, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, src.tex); glUniform1i(glGetUniformLocation(p, "A"), 0);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc); glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return out.tex;
    }

    GLuint swiglu_split(Target& a, Target& b) {
        if (ffn_h_.tex == 0 || ffn_h_.w != a.w) ffn_h_ = make_target(a.w, 1);
        GLuint p = swiglu_program(a.w, false);
        glBindFramebuffer(GL_FRAMEBUFFER, ffn_h_.fbo);
        glViewport(0, 0, ffn_h_.w, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, a.tex); glUniform1i(glGetUniformLocation(p, "A"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, b.tex); glUniform1i(glGetUniformLocation(p, "B"), 1);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc); glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return ffn_h_.tex;
    }

    GLuint swiglu_fused(Target& src, int hidden) {
        if (ffn_h_.tex == 0 || ffn_h_.w != hidden) ffn_h_ = make_target(hidden, 1);
        GLuint p = swiglu_program(hidden, true);
        glBindFramebuffer(GL_FRAMEBUFFER, ffn_h_.fbo);
        glViewport(0, 0, hidden, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, src.tex); glUniform1i(glGetUniformLocation(p, "A"), 0);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc); glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return ffn_h_.tex;
    }

    vector<float> logits(const vector<Mat>& tiles, const vector<float>& x) {
        const int rows = (int)tiles.size();
        if (logits_target_.tex == 0 || logits_target_.h != rows) logits_target_ = make_target(1536, rows);
        GLuint xt = upload_activations(x);
        for (int i = 0; i < rows; ++i) draw(tiles[i], xt, logits_target_, i);
        if (needs_finish()) { glFlush(); glFinish(); }
        vector<int32_t> raw = read_i32(logits_target_, 1536, rows);
        vector<float> out;
        out.reserve((size_t)rows * 1536);
        for (int i = 0; i < rows; ++i) {
            const Mat& m = tiles[i];
            for (int j = 0; j < m.rows; ++j) {
                int v = raw[(size_t)i * 1536 + j];
                if (v >= 32768) v -= 65536;
                out.push_back((float)v / ACT_SCALE);
            }
        }
        return out;
    }

    CandidateResult logits_candidates(const vector<Mat>& tiles, const vector<float>& x,
                                      int topk, float temp, float top_p) {
        return logits_candidates_from(tiles, upload_activations(x), topk, temp, top_p);
    }

    CandidateResult logits_candidates_from(const vector<Mat>& tiles, GLuint xt,
                                           int topk, float temp, float top_p) {
        const int rows = (int)tiles.size();
        if (logits_target_.tex == 0 || logits_target_.h != rows) logits_target_ = make_target(1536, rows);
        for (int i = 0; i < rows; ++i) draw(tiles[i], xt, logits_target_, i);
        // The candidate pass samples logits_target_ as a texture.  On VC4 the
        // tile pipeline does not order that sample after the FBO renders, so
        // finish the logits pass before starting Top-K.
        if (needs_finish()) { glFlush(); glFinish(); }

        if (getenv("TOPK_FAST")) {
            const int chunk = 16;
            int max_chunks = 0;
            for (auto& t : tiles) max_chunks = max(max_chunks, (t.rows + chunk - 1) / chunk);
            const int out_w = max_chunks * 2;
            if (candidate_target_.tex == 0 || candidate_target_.w != out_w || candidate_target_.h != rows)
                candidate_target_ = make_target(out_w, rows);
            for (int i = 0; i < rows; ++i) {
                int chunks = (tiles[i].rows + chunk - 1) / chunk;
                GLuint p = top2_fast_program(tiles[i].rows, chunk);
                glBindFramebuffer(GL_FRAMEBUFFER, candidate_target_.fbo);
                glViewport(0, i, chunks * 2, 1);
                glUseProgram(p);
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, logits_target_.tex);
                glUniform1i(glGetUniformLocation(p, "L"), 0);
                glUniform1f(glGetUniformLocation(p, "tr"), (float)i);
                glUniform1f(glGetUniformLocation(p, "nr"), (float)rows);
                GLuint loc = glGetAttribLocation(p, "p");
                glEnableVertexAttribArray(loc); glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            if (needs_finish()) { glFlush(); glFinish(); }
            vector<uint8_t> raw = read_bytes(candidate_target_, out_w, rows);
            vector<Candidate> cand;
            for (int r = 0; r < rows; ++r) {
                int M = tiles[r].rows;
                int chunks = (M + chunk - 1) / chunk;
                for (int c = 0; c < chunks * 2; ++c) {
                    int o = ((size_t)r * out_w + c) * 4;
                    int value = (int)raw[o] * 256 + raw[o + 1];
                    if (value >= 32768) value -= 65536;
                    int idx = (int)raw[o + 2] * 256 + raw[o + 3];
                    if (idx < M) cand.push_back({(float)value / ACT_SCALE, r * 1536 + idx});
                }
            }
            sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
            if ((int)cand.size() > topk) cand.resize(topk);
            CandidateResult res; res.cand = cand; res.complete = true; res.strict = false;
            return res;
        }

        const int chunk = 16;
        int max_chunks = 0;
        for (auto& t : tiles) max_chunks = max(max_chunks, (t.rows + chunk - 1) / chunk);
        const int out_w = max_chunks * 3;
        if (candidate_target_.tex == 0 || candidate_target_.w != out_w || candidate_target_.h != rows)
            candidate_target_ = make_target(out_w, rows);
        for (int i = 0; i < rows; ++i) {
            int chunks = (tiles[i].rows + chunk - 1) / chunk;
            GLuint p = top2_program(tiles[i].rows, chunk);
            glBindFramebuffer(GL_FRAMEBUFFER, candidate_target_.fbo);
            glViewport(0, i, chunks * 3, 1);
            glUseProgram(p);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, logits_target_.tex);
            glUniform1i(glGetUniformLocation(p, "L"), 0);
            glUniform1f(glGetUniformLocation(p, "tr"), (float)i);
            glUniform1f(glGetUniformLocation(p, "nr"), (float)rows);
            glUniform1f(glGetUniformLocation(p, "invt"), 1.0f / max(0.0001f, temp));
            GLuint loc = glGetAttribLocation(p, "p");
            glEnableVertexAttribArray(loc); glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        vector<uint8_t> raw = read_bytes(candidate_target_, out_w, rows);
        vector<Candidate> cand;
        vector<vector<float>> cmax(rows), csum(rows);
        float gmax = -1e30f;
        for (int r = 0; r < rows; ++r) {
            int M = tiles[r].rows;
            int chunks = (M + chunk - 1) / chunk;
            cmax[r].assign(chunks, -1e30f);
            csum[r].assign(chunks, 0.0f);
            for (int c = 0; c < chunks; ++c) {
                for (int rank = 0; rank < 3; ++rank) {
                    int o = ((size_t)r * out_w + c * 3 + rank) * 4;
                    if (rank < 2) {
                        int value = (int)raw[o] * 256 + raw[o + 1];
                        if (value >= 32768) value -= 65536;
                        int idx = (int)raw[o + 2] * 256 + raw[o + 3];
                        if (idx < M) {
                            float v = (float)value / ACT_SCALE;
                            cand.push_back({v, r * 1536 + idx});
                            cmax[r][c] = max(cmax[r][c], v);
                            gmax = max(gmax, v);
                        }
                    } else {
                        int value = (int)raw[o] * 256 + raw[o + 1];
                        if (value >= 32768) value -= 65536;
                        int sumq = (int)raw[o + 2] * 256 + raw[o + 3];
                        float mx = (float)value / ACT_SCALE;
                        cmax[r][c] = max(cmax[r][c], mx);
                        csum[r][c] = (float)sumq / 2048.0f;
                        gmax = max(gmax, mx);
                    }
                }
            }
        }
        sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
        float denom = 0;
        for (int r = 0; r < rows; ++r)
            for (size_t c = 0; c < cmax[r].size(); ++c)
                denom += csum[r][c] * expf((cmax[r][c] - gmax) / max(0.0001f, temp));

        if ((int)cand.size() > topk) cand.resize(topk);
        float cum = 0;
        for (auto& c : cand) cum += expf((c.value - gmax) / max(0.0001f, temp)) / denom;

        CandidateResult res;
        res.cand = cand; res.logsumexp = logf(denom) + gmax / max(0.0001f, temp); res.denom = denom; res.strict = true;
        res.max_logit = gmax;
        if (cum + 1e-6f >= top_p) {
            res.complete = true;
            return res;
        }
        res.cand.clear();
        vector<int32_t> full_raw = read_i32(logits_target_, 1536, rows);
        res.full.reserve((size_t)rows * 1536);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < tiles[i].rows; ++j) {
                int v = full_raw[(size_t)i * 1536 + j];
                if (v >= 32768) v -= 65536;
                res.full.push_back((float)v / ACT_SCALE);
            }
        }
        return res;
    }

    bool gpu_attn_enabled() const {
        return getenv("GPU_ATTN") && atoi(getenv("GPU_ATTN")) != 0;
    }

    bool gpu_layer_enabled() const {
        return getenv("GPU_LAYER") && atoi(getenv("GPU_LAYER")) != 0;
    }

    void init_attention(int dim, int kv_dim, int heads, int kv_heads, int head_dim,
                        int seq, bool rope_half, bool qk_norm, int layers,
                        const vector<float>& qn, const vector<float>& kn,
                        const vector<float>& rcos, const vector<float>& rsin) {
        att_dim_ = dim; att_kv_dim_ = kv_dim; att_heads_ = heads;
        att_kv_heads_ = kv_heads; att_head_dim_ = head_dim; att_seq_ = seq;
        att_rope_half_ = rope_half; att_qk_norm_ = qk_norm;
        att_kv_bits_ = getenv("KV_PACK") ? atoi(getenv("KV_PACK")) : 0;
        if (att_kv_bits_ == 1) att_kv_bits_ = 16;
        if (att_kv_bits_ != 0 && att_kv_bits_ != 8 && att_kv_bits_ != 16) att_kv_bits_ = 0;
        att_kv_packed_ = att_kv_bits_ != 0;
        att_kv_scale_ = getenv("KV_SCALE") ? atof(getenv("KV_SCALE")) :
                        (att_kv_bits_ == 16 ? 2048.0f : att_kv_bits_ == 8 ? 16.0f : 16384.0f);
        if (att_kv_scale_ < 1.0f) att_kv_scale_ = 1.0f;
        att_kv_width_ = att_kv_bits_ == 16 ? kv_dim : att_kv_bits_ == 8 ? kv_dim / 2 : 2 * kv_dim;
        if (att_kv_bits_ != 0 &&
            ((kv_dim % 2) != 0 || (att_kv_bits_ == 8 && ((kv_dim % 4) != 0 || (head_dim % 4) != 0)))) {
            cerr << "KV_PACK width not divisible; falling back to 24-bit\n";
            att_kv_bits_ = 0; att_kv_packed_ = false;
            att_kv_scale_ = 16384.0f; att_kv_width_ = 2 * kv_dim;
        }
        q_norm_tex_.clear(); k_norm_tex_.clear();
        if (qk_norm) {
            for (int l = 0; l < layers; ++l) {
                vector<float> q(qn.begin() + (size_t)l * head_dim, qn.begin() + (size_t)(l + 1) * head_dim);
                vector<float> k(kn.begin() + (size_t)l * head_dim, kn.begin() + (size_t)(l + 1) * head_dim);
                q_norm_tex_.push_back(make_q16_texture(q));
                k_norm_tex_.push_back(make_q16_texture(k));
            }
        }
        const int half = head_dim / 2;
        const int qkv_rows = dim + 2 * kv_dim;
        qkv_target_ = make_target(qkv_rows, 1);
        att_scores_ = make_target(seq, heads);
        att_stats_ = make_target(2 * heads, 1);
        att_probs_ = make_target(seq, heads);
        att_out_ = make_target(dim, 1);
        kv_cache_.clear();
        kv_cache_.reserve(layers);
        for (int l = 0; l < layers; ++l) kv_cache_.push_back(make_target(att_kv_width_, seq));
        const int rw = 2 * half;
        vector<uint8_t> data((size_t)seq * rw * 4);
        for (int p = 0; p < seq; ++p) {
            for (int i = 0; i < half; ++i) {
                int c = (int)lrintf(rcos[(size_t)p * half + i] * 4096.0f) + 8388608;
                int s = (int)lrintf(rsin[(size_t)p * half + i] * 4096.0f) + 8388608;
                for (int k = 0; k < 2; ++k) {
                    int u = k == 0 ? c : s;
                    size_t o = ((size_t)p * rw + 2 * i + k) * 4;
                    data[o + 0] = 0; data[o + 1] = (u >> 16) & 255;
                    data[o + 2] = (u >> 8) & 255; data[o + 3] = u & 255;
                }
            }
        }
        rope_tex_ = make_texture(rw, seq, data.data());
    }

    GLuint attention_block(const Mat& qkv, GLuint x, int layer, int pos) {
        if (qkv_target_.tex == 0 || kv_cache_.empty()) return 0;
        matvec_to(qkv, x, qkv_target_);
        att_finish();
        kv_rope_write(layer, pos);
        att_scores(layer, pos);
        att_stats(pos);
        att_probs(pos);
        att_v(layer, pos);
        return att_out_.tex;
    }

    void init_persistent(int dim, int layers, const vector<float>& rms_att,
                        const vector<float>& rms_ffn, const vector<float>& rms_final) {
        layer_fast_finish_ = true;
        hidden_a_ = make_target(dim, 1);
        hidden_b_ = make_target(dim, 1);
        xb_ = make_target(dim, 1);
        xbf_ = make_target(dim, 1);
        xf_ = make_target(dim, 1);
        rms_stats_ = make_target(1, 1);
        hidden_in_ = make_texture(dim, 1, nullptr);
        rms_att_tex_.clear(); rms_ffn_tex_.clear();
        for (int l = 0; l < layers; ++l) {
            vector<float> a(rms_att.begin() + (size_t)l * dim, rms_att.begin() + (size_t)(l + 1) * dim);
            vector<float> f(rms_ffn.begin() + (size_t)l * dim, rms_ffn.begin() + (size_t)(l + 1) * dim);
            rms_att_tex_.push_back(make_q16_texture(a));
            rms_ffn_tex_.push_back(make_q16_texture(f));
        }
        rms_final_tex_ = make_q16_texture(rms_final);
    }

    GLuint upload_hidden(const vector<float>& x) {
        vector<uint8_t> data((size_t)x.size() * 4);
        for (size_t i = 0; i < x.size(); ++i) {
            uint16_t u = (uint16_t)quantize_act(x[i]);
            data[i * 4 + 0] = u >> 8; data[i * 4 + 1] = u & 255;
            data[i * 4 + 2] = 0; data[i * 4 + 3] = 255;
        }
        glBindTexture(GL_TEXTURE_2D, hidden_in_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (int)x.size(), 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return hidden_in_;
    }

    GLuint rmsnorm(GLuint x, GLuint w, Target& out) {
        if (out.tex == 0 || out.w != att_dim_) out = make_target(att_dim_, 1);
        if (rms_stats_.tex == 0) rms_stats_ = make_target(1, 1);
        GLuint ps = rms_stats_program();
        glBindFramebuffer(GL_FRAMEBUFFER, rms_stats_.fbo);
        glViewport(0, 0, 1, 1);
        glUseProgram(ps);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(ps, "X"), 0);
        GLuint loc = glGetAttribLocation(ps, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        layer_finish("NF_RMS");

        GLuint pn = rms_norm_program();
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, att_dim_, 1);
        glUseProgram(pn);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(pn, "X"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, w);
        glUniform1i(glGetUniformLocation(pn, "W"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, rms_stats_.tex);
        glUniform1i(glGetUniformLocation(pn, "S"), 2);
        loc = glGetAttribLocation(pn, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        layer_finish("NF_RMS");
        return out.tex;
    }

    void matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out) {
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        GLuint p = (m.type == WeightType::INT8) ? int8_add_program(m.rows, m.cols)
                  : (m.type == WeightType::INT4) ? int4_add_program(m.rows, m.cols)
                                                 : int16_add_program(m.rows, m.cols);
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m.tex);
        glUniform1i(glGetUniformLocation(p, "W"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(p, "X"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, residual);
        glUniform1i(glGetUniformLocation(p, "R"), 2);
        if (m.type == WeightType::INT4) {
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, residual);
            glUniform1i(glGetUniformLocation(p, "R"), 3);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, m.q4_param_tex);
            glUniform1i(glGetUniformLocation(p, "P"), 2);
            glUniform1f(glGetUniformLocation(p, "ps"), m.q4_param_scale);
        }
        glUniform1f(glGetUniformLocation(p, "ws"), m.scale);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        const int strip = (m.rows == 1536) ? 768 : m.rows;
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
            int w = min(strip, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        layer_finish("NF_MAT");
    }

    vector<float> logits_from(const vector<Mat>& tiles, GLuint x) {
        const int rows = (int)tiles.size();
        if (logits_target_.tex == 0 || logits_target_.h != rows) logits_target_ = make_target(1536, rows);
        for (int i = 0; i < rows; ++i) draw(tiles[i], x, logits_target_, i);
        if (needs_finish()) { glFlush(); glFinish(); }
        vector<int32_t> raw = read_i32(logits_target_, 1536, rows);
        vector<float> out;
        out.reserve((size_t)rows * 1536);
        for (int i = 0; i < rows; ++i) {
            const Mat& m = tiles[i];
            for (int j = 0; j < m.rows; ++j) {
                int v = raw[(size_t)i * 1536 + j];
                if (v >= 32768) v -= 65536;
                out.push_back((float)v / ACT_SCALE);
            }
        }
        return out;
    }

    vector<uint8_t> read_bytes(Target& t, int w, int h) {
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        return b;
    }

    void sync() { glFinish(); }
    bool needs_finish() const {
        if (getenv("FORCE_GLFINISH")) return true;
        return renderer_.find("VC4") != string::npos || renderer_.find("VideoCore") != string::npos;
    }

private:
    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    unordered_map<string, GLuint> programs_;
    unordered_map<int, GLuint> input_textures_;
    unordered_map<int, Target> targets_;
    Target logits_target_;
    Target candidate_target_;
    Target sum_target_;
    Target ffn_a_, ffn_b_, ffn_h_, ffn_ab_;
    Target ffn_ab_a_, ffn_ab_b_, ffn_h_a_, ffn_h_b_;
    Target qkv_target_, att_scores_, att_stats_, att_probs_, att_out_;
    vector<Target> kv_cache_;
    Target hidden_a_, hidden_b_, xb_, xbf_, xf_, rms_stats_;
    bool layer_fast_finish_ = false;
    GLuint hidden_in_ = 0;
    vector<GLuint> rms_att_tex_, rms_ffn_tex_;
    GLuint rms_final_tex_ = 0;
    GLuint rope_tex_ = 0;
    int att_dim_ = 0, att_kv_dim_ = 0, att_heads_ = 0, att_kv_heads_ = 0;
    int att_head_dim_ = 0, att_seq_ = 0;
    bool att_rope_half_ = false;
    bool att_qk_norm_ = false;
    bool att_kv_packed_ = false;
    int att_kv_bits_ = 0;
    float att_kv_scale_ = 16384.0f;
    int att_kv_width_ = 0;
    vector<GLuint> q_norm_tex_, k_norm_tex_;
    GLuint vbo_ = 0;
    string renderer_;

    void init_egl() {
        auto get_platform_display = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        dpy_ = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        EGLint major = 0, minor = 0;
        if (!eglInitialize(dpy_, &major, &minor)) { cerr << "eglInitialize failed\n"; exit(1); }
        eglBindAPI(EGL_OPENGL_ES_API);
        EGLint attrs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE
        };
        EGLConfig cfg; EGLint n = 0;
        eglChooseConfig(dpy_, attrs, &cfg, 1, &n);
        EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
        EGLContext ctx = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, ctx_attrs);
        if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            cerr << "EGL context creation failed\n"; exit(1);
        }
        ctx_ = ctx;
        const char* r = (const char*)glGetString(GL_RENDERER);
        renderer_ = r ? r : "";
    }

    void init_gl() {
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    }

    static GLuint compile_shader(GLenum type, const string& src) {
        GLuint s = glCreateShader(type);
        const char* p = src.c_str();
        glShaderSource(s, 1, &p, nullptr);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[8192]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            cerr << "shader compile failed:\n" << log << "\n"; exit(1);
        }
        return s;
    }

    static GLuint link_program(const string& fs) {
        const char* vs = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }";
        GLuint p = glCreateProgram();
        glAttachShader(p, compile_shader(GL_VERTEX_SHADER, vs));
        glAttachShader(p, compile_shader(GL_FRAGMENT_SHADER, fs));
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[8192]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
            cerr << "program link failed:\n" << log << "\n"; exit(1);
        }
        return p;
    }

    static GLuint make_texture(int w, int h, const void* data) {
        GLuint t = 0; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFlush(); glFinish();
        return t;
    }

    static GLuint make_q24_texture(const vector<float>& x, float scale) {
        vector<uint8_t> data((size_t)x.size() * 4);
        for (size_t i = 0; i < x.size(); ++i) {
            int64_t q = (int64_t)llrintf(x[i] * scale) + 8388608;
            if (q < 0) q = 0;
            if (q > 16777215) q = 16777215;
            size_t o = i * 4;
            data[o + 0] = 0;
            data[o + 1] = (uint8_t)(q >> 16);
            data[o + 2] = (uint8_t)(q >> 8);
            data[o + 3] = (uint8_t)q;
        }
        return make_texture((int)x.size(), 1, data.data());
    }

    static GLuint make_q16_texture(const vector<float>& x) {
        vector<uint8_t> data((size_t)x.size() * 4);
        for (size_t i = 0; i < x.size(); ++i) {
            uint16_t u = (uint16_t)quantize_act(x[i]);
            data[i * 4 + 0] = u >> 8; data[i * 4 + 1] = u & 255;
            data[i * 4 + 2] = 0; data[i * 4 + 3] = 255;
        }
        return make_texture((int)x.size(), 1, data.data());
    }

    static Target make_target(int w, int h) {
        Target t; t.w = w; t.h = h;
        t.tex = make_texture(w, h, nullptr);
        glGenFramebuffers(1, &t.fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            cerr << "FBO incomplete\n"; exit(1);
        }
        return t;
    }

    Target& target_for(int rows) {
        auto it = targets_.find(rows);
        if (it != targets_.end()) return it->second;
        Target t = make_target(rows, 1);
        targets_[rows] = t;
        return targets_[rows];
    }

    GLuint upload_activations(const vector<float>& x) {
        const int n = (int)x.size();
        vector<uint8_t> data((size_t)n * 4);
        for (int i = 0; i < n; ++i) {
            uint16_t u = (uint16_t)quantize_act(x[i]);
            data[i * 4 + 0] = (u >> 8) & 255;
            data[i * 4 + 1] = u & 255;
            data[i * 4 + 2] = 0;
            data[i * 4 + 3] = 255;
        }
        auto it = input_textures_.find(n);
        if (it == input_textures_.end()) {
            GLuint t = make_texture(n, 1, data.data());
            input_textures_[n] = t;
            return t;
        }
        glBindTexture(GL_TEXTURE_2D, it->second);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, n, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return it->second;
    }

    GLuint int8_program(int rows, int cols) {
        string key = "i8_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,X;uniform float ws;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
            "return u>=32768.0?u-65536.0:u;}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
            "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
            "(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "vec4 w=floor(c*255.0+0.5)-128.0;float k=float(4*q);"
            "s+=w.x*dx(k)+w.y*dx(k+1.0)+w.z*dx(k+2.0)+w.w*dx(k+3.0);}"
            "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;"
            "v+=8388608.0;"
            "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
            "float b=floor(v/256.0)-r0*65536.0-g*256.0;float a=v-r0*16777216.0-g*65536.0-b*256.0;"
            "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint int16_program(int rows, int cols) {
        string key = "i16_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        const int tex_w = (cols + 1) / 2;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,X;uniform float ws;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
            "return u>=32768.0?u-65536.0:u;}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int q=0;q<" + to_string(tex_w) + ";q++){"
            "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tex_w) + "),"
            "(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float u0=floor(c.g*255.0+0.5)*256.0+floor(c.r*255.0+0.5);"
            "if(u0>=32768.0)u0-=65536.0;"
            "float u1=floor(c.a*255.0+0.5)*256.0+floor(c.b*255.0+0.5);"
            "if(u1>=32768.0)u1-=65536.0;"
            "float k=float(2*q);s+=u0*dx(k);if(2*q+1<" + to_string(cols) + ")s+=u1*dx(k+1.0);}"
            "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;"
            "v+=8388608.0;"
            "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
            "float b=floor(v/256.0)-r0*65536.0-g*256.0;float a=v-r0*16777216.0-g*65536.0-b*256.0;"
            "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    // q4: one RGBA8 texel holds 8 nibbles; load it once and unroll all 8.
    GLuint int4_program(int rows, int cols) {
        string key = "i4_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        const int texw = cols / 8, groups = (cols + 31) / 32;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,P,X;uniform float ps;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);return u>=32768.0?u-65536.0:u;}"
            "float d16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u;}"
            "vec4 enc(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;v+=8388608.0;float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;float b=floor(v/256.0)-r0*65536.0-g*256.0;float a=v-r0*16777216.0-g*65536.0-b*256.0;return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int g=0;g<" + to_string(groups) + ";g++){"
            "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float sc=d16(p.rg)*ps;float mn=d16(p.ba)*ps;"
            "for(int t=0;t<4;t++){int c=g*32+t*8;"
            "vec4 w=texture2D(W,vec2((float(g*4+t)+0.5)/float(" + to_string(texw) + "),(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float b0=floor(w.r*255.0+0.5);float b1=floor(w.g*255.0+0.5);float b2=floor(w.b*255.0+0.5);float b3=floor(w.a*255.0+0.5);"
            "float q0=b0-16.0*floor(b0/16.0);float q1=floor(b0/16.0);float q2=b1-16.0*floor(b1/16.0);float q3=floor(b1/16.0);"
            "float q4=b2-16.0*floor(b2/16.0);float q5=floor(b2/16.0);float q6=b3-16.0*floor(b3/16.0);float q7=floor(b3/16.0);"
            "s+=(q0*sc+mn)*dx(float(c+0))+(q1*sc+mn)*dx(float(c+1))+(q2*sc+mn)*dx(float(c+2))+(q3*sc+mn)*dx(float(c+3))"
            "+(q4*sc+mn)*dx(float(c+4))+(q5*sc+mn)*dx(float(c+5))+(q6*sc+mn)*dx(float(c+6))+(q7*sc+mn)*dx(float(c+7));}}"
            "gl_FragColor=enc(s);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint int8_add_program(int rows, int cols) {
        string key = "i8add_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,X,R;uniform float ws;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
            "return u>=32768.0?u-65536.0:u;}"
            "float dr(vec4 c){float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int q=0;q<" + to_string(cols / 4) + ";q++){"
            "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(cols / 4) + "),"
            "(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "vec4 w=floor(c*255.0+0.5)-128.0;float k=float(4*q);"
            "s+=w.x*dx(k)+w.y*dx(k+1.0)+w.z*dx(k+2.0)+w.w*dx(k+3.0);}"
            "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);"
            "vec4 rc=texture2D(R,vec2((float(r)+0.5)/float(" + to_string(rows) + "),0.5));"
            "float oq=v+dr(rc)*256.0;"
            "float q=clamp(floor(oq+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
            "float hi=floor(q/256.0),lo=q-hi*256.0;gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint int16_add_program(int rows, int cols) {
        string key = "i16add_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        const int tex_w = (cols + 1) / 2;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,X,R;uniform float ws;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);"
            "return u>=32768.0?u-65536.0:u;}"
            "float dr(vec4 c){float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int q=0;q<" + to_string(tex_w) + ";q++){"
            "vec4 c=texture2D(W,vec2((float(q)+0.5)/float(" + to_string(tex_w) + "),"
            "(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float u0=floor(c.g*255.0+0.5)*256.0+floor(c.r*255.0+0.5);"
            "if(u0>=32768.0)u0-=65536.0;"
            "float u1=floor(c.a*255.0+0.5)*256.0+floor(c.b*255.0+0.5);"
            "if(u1>=32768.0)u1-=65536.0;"
            "float k=float(2*q);s+=u0*dx(k);if(2*q+1<" + to_string(cols) + ")s+=u1*dx(k+1.0);}"
            "float v=clamp(floor(s*ws+0.5),-32768.0,32767.0);"
            "vec4 rc=texture2D(R,vec2((float(r)+0.5)/float(" + to_string(rows) + "),0.5));"
            "float oq=v+dr(rc)*256.0;"
            "float q=clamp(floor(oq+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
            "float hi=floor(q/256.0),lo=q-hi*256.0;gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }


    // q4 + residual: same 8-nibble unroll, but output must stay q16.
    GLuint int4_add_program(int rows, int cols) {
        string key = "i4add_" + to_string(rows) + "_" + to_string(cols);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        const int texw = cols / 8, groups = (cols + 31) / 32;
        string fs =
            "precision highp float;"
            "uniform sampler2D W,P,X,R;uniform float ps;"
            "float dx(float i){vec4 c=texture2D(X,vec2((i+0.5)/float(" + to_string(cols) + "),0.5));"
            "float u=floor(c.r*255.0+0.5)*256.0+floor(c.g*255.0+0.5);return u>=32768.0?u-65536.0:u;}"
            "float d16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u;}"
            "vec4 enc(float v){v=clamp(floor(v+0.5),-32768.0,32767.0);if(v<0.0)v+=65536.0;float hi=floor(v/256.0),lo=v-hi*256.0;return vec4(hi/255.0,lo/255.0,0.0,1.0);}"
            "void main(){int r=int(gl_FragCoord.x);float s=0.0;"
            "for(int g=0;g<" + to_string(groups) + ";g++){"
            "vec4 p=texture2D(P,vec2((float(g)+0.5)/float(" + to_string(groups) + "),(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float sc=d16(p.rg)*ps;float mn=d16(p.ba)*ps;"
            "for(int t=0;t<4;t++){int c=g*32+t*8;"
            "vec4 w=texture2D(W,vec2((float(g*4+t)+0.5)/float(" + to_string(texw) + "),(float(r)+0.5)/float(" + to_string(rows) + ")));"
            "float b0=floor(w.r*255.0+0.5);float b1=floor(w.g*255.0+0.5);float b2=floor(w.b*255.0+0.5);float b3=floor(w.a*255.0+0.5);"
            "float q0=b0-16.0*floor(b0/16.0);float q1=floor(b0/16.0);float q2=b1-16.0*floor(b1/16.0);float q3=floor(b1/16.0);"
            "float q4=b2-16.0*floor(b2/16.0);float q5=floor(b2/16.0);float q6=b3-16.0*floor(b3/16.0);float q7=floor(b3/16.0);"
            "s+=(q0*sc+mn)*dx(float(c+0))+(q1*sc+mn)*dx(float(c+1))+(q2*sc+mn)*dx(float(c+2))+(q3*sc+mn)*dx(float(c+3))"
            "+(q4*sc+mn)*dx(float(c+4))+(q5*sc+mn)*dx(float(c+5))+(q6*sc+mn)*dx(float(c+6))+(q7*sc+mn)*dx(float(c+7));}}"
            "vec4 rc=texture2D(R,vec2((float(r)+0.5)/float(" + to_string(rows) + "),0.5));"
            "gl_FragColor=enc(s+d16(rc.rg));}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint sum_program(int M) {
        string key = "sum_" + to_string(M);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;"
            "uniform sampler2D L;uniform float tr,nr,mrows,invt;"
            "float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){float mx=-1e30;for(int i=0;i<1536;i++){if(float(i)>=mrows)break;"
            "vec4 c=texture2D(L,vec2((float(i)+0.5)/1536.0,(tr+0.5)/nr));mx=max(mx,dec(c));}"
            "float sum=0.0;for(int i=0;i<1536;i++){if(float(i)>=mrows)break;"
            "vec4 c=texture2D(L,vec2((float(i)+0.5)/1536.0,(tr+0.5)/nr));sum+=exp((dec(c)-mx)*invt);}"
            "float v=floor(sum*4096.0+0.5)+8388608.0;"
            "float r0=floor(v/16777216.0);float g=floor(v/65536.0)-r0*256.0;"
            "float b=floor(v/256.0)-r0*65536.0-g*256.0;float a=v-r0*16777216.0-g*65536.0-b*256.0;"
            "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint top2_fast_program(int M, int chunk) {
        string key = "top2f_" + to_string(M) + "_" + to_string(chunk);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;"
            "uniform sampler2D L;uniform float tr,nr;"
            "float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){int x=int(gl_FragCoord.x);int ch=x/2;int rank=x-ch*2;int start=ch*" + to_string(chunk) + ";"
            "float best1=-1e30,best2=-1e30;int i1=0,i2=0;"
            "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(M) + ")break;"
            "vec4 c=texture2D(L,vec2((float(idx)+0.5)/1536.0,(tr+0.5)/nr));float v=dec(c);"
            "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}}"
            "float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
            "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
            "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
            "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint top2_program(int M, int chunk) {
        string key = "top2_" + to_string(M) + "_" + to_string(chunk);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;"
            "uniform sampler2D L;uniform float tr,nr,invt;"
            "float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){int x=int(gl_FragCoord.x);int ch=x/3;int rank=x-ch*3;int start=ch*" + to_string(chunk) + ";"
            "float best1=-1e30,best2=-1e30,lm=-1e30,ls=0.0;int i1=0,i2=0;"
            "for(int k=0;k<" + to_string(chunk) + ";k++){int idx=start+k;if(idx>=" + to_string(M) + ")break;"
            "vec4 c=texture2D(L,vec2((float(idx)+0.5)/1536.0,(tr+0.5)/nr));float v=dec(c);"
            "if(v>best1){best2=best1;i2=i1;best1=v;i1=idx;}else if(v>best2){best2=v;i2=idx;}"
            "if(v>lm){ls=ls*exp((lm-v)*invt)+1.0;lm=v;}else{ls+=exp((v-lm)*invt);}}"
            "if(rank<2){float bv=(rank==0?best1:best2)*256.0;int bi=rank==0?i1:i2;"
            "if(bv<0.0)bv+=65536.0;float vh=floor(bv/256.0),vl=bv-vh*256.0;"
            "float ih=floor(float(bi)/256.0),il=float(bi)-ih*256.0;"
            "gl_FragColor=vec4(vh/255.0,vl/255.0,ih/255.0,il/255.0);return;}"
            "float sum=ls;float mv=lm*256.0;if(mv<0.0)mv+=65536.0;float mh=floor(mv/256.0),ml=mv-mh*256.0;"
            "float sv=floor(sum*2048.0+0.5);float sh=floor(sv/256.0),sl=sv-sh*256.0;"
            "gl_FragColor=vec4(mh/255.0,ml/255.0,sh/255.0,sl/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint swiglu_2row_program(int hidden) {
        string key = "swiglu_2row_" + to_string(hidden);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs =
            "precision highp float;uniform sampler2D A;"
            "float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){float x=float(int(gl_FragCoord.x));"
            "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + to_string(hidden) + "),0.5/2.0));"
            "vec4 cb=texture2D(A,vec2((x+0.5)/float(" + to_string(hidden) + "),1.5/2.0));"
            "float a=dec(ca),b=dec(cb);float y=(a/(1.0+exp(-a)))*b;"
            "float q=clamp(floor(y*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
            "float hi=floor(q/256.0),lo=q-hi*256.0;gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
        GLuint p = link_program(fs); programs_[key] = p; return p;
    }

    GLuint swiglu_program(int hidden, bool fused) {
        string key = string("swiglu_") + (fused ? "fused_" : "split_") + to_string(hidden);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        auto dec = []() {
            return string("float dec(vec4 c){float u=floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;if(u>=32768.0)u-=65536.0;return u/256.0;}");
        };
        string body;
        if (fused) {
            body = "uniform sampler2D A;"
                   "void main(){float x=float(int(gl_FragCoord.x));"
                   "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + to_string(2*hidden) + "),0.5));"
                   "vec4 cb=texture2D(A,vec2((x+0.5+float(" + to_string(hidden) + "))/float(" + to_string(2*hidden) + "),0.5));"
                   "float a=dec(ca),b=dec(cb);"
                   "float y=(a/(1.0+exp(-a)))*b;"
                   "float q=clamp(floor(y*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
                   "float hi=floor(q/256.0),lo=q-hi*256.0;"
                   "gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
        } else {
            body = "uniform sampler2D A,B;"
                   "void main(){float x=float(int(gl_FragCoord.x));"
                   "vec4 ca=texture2D(A,vec2((x+0.5)/float(" + to_string(hidden) + "),0.5));"
                   "vec4 cb=texture2D(B,vec2((x+0.5)/float(" + to_string(hidden) + "),0.5));"
                   "float a=dec(ca),b=dec(cb);"
                   "float y=(a/(1.0+exp(-a)))*b;"
                   "float q=clamp(floor(y*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;"
                   "float hi=floor(q/256.0),lo=q-hi*256.0;"
                   "gl_FragColor=vec4(hi/255.0,lo/255.0,0.0,1.0);}";
        }
        string fs = "precision highp float;" + dec() + body;
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    string att_defs() const {
        return string("#define DIM ") + to_string(att_dim_) + "\n" +
               "#define KVDIM " + to_string(att_kv_dim_) + "\n" +
               "#define QKVDIM " + to_string(att_dim_ + 2 * att_kv_dim_) + "\n" +
               "#define HEADS " + to_string(att_heads_) + "\n" +
               "#define KVHEADS " + to_string(att_kv_heads_) + "\n" +
               "#define GROUP " + to_string(att_heads_ / att_kv_heads_) + "\n" +
               "#define HEADDIM " + to_string(att_head_dim_) + "\n" +
               "#define HALF " + to_string(att_head_dim_ / 2) + "\n" +
               "#define SEQ " + to_string(att_seq_) + "\n" +
               "#define ROPEHALF " + string(att_rope_half_ ? "1" : "0") + "\n" +
               "#define QKNORM " + string(att_qk_norm_ ? "1" : "0") + "\n" +
               "#define QKEPS 0.000001\n" +
               "#define KVSCALE " + to_string(att_kv_scale_) + "\n" +
               "#define KVPACK " + to_string(att_kv_bits_) + "\n" +
               "#define KVWIDTH " + to_string(att_kv_width_) + "\n";
    }

    static string att_helpers() {
        return
            "vec4 q16(float v){float q=clamp(floor(v*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;float hi=floor(q/256.0),lo=q-hi*256.0;return vec4(hi/255.0,lo/255.0,0.0,1.0);}\n"
            "float dec24raw(vec4 c){return floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;}\n"
            "float dec24q(vec4 c){float u=dec24raw(c);if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
            "vec4 enc24s(float v,float s){float q=floor(v*s+0.5)+8388608.0;float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;return vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}\n"
            "float dec24s(vec4 c,float s){return dec24raw(c)/s;}\n"
            "float dec16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}\n"
            "vec4 encKV(float v0,float v1){if(KVPACK==16){float q0=clamp(floor(v0*KVSCALE+0.5),-32768.0,32767.0);if(q0<0.0)q0+=65536.0;float q1=clamp(floor(v1*KVSCALE+0.5),-32768.0,32767.0);if(q1<0.0)q1+=65536.0;float h0=floor(q0/256.0),l0=q0-h0*256.0,h1=floor(q1/256.0),l1=q1-h1*256.0;return vec4(h0/255.0,l0/255.0,h1/255.0,l1/255.0);}return enc24s(v0,KVSCALE);}\n"
            "vec4 encKV8(float v0,float v1,float v2,float v3){float q0=clamp(floor(v0*KVSCALE+0.5),-128.0,127.0);if(q0<0.0)q0+=256.0;float q1=clamp(floor(v1*KVSCALE+0.5),-128.0,127.0);if(q1<0.0)q1+=256.0;float q2=clamp(floor(v2*KVSCALE+0.5),-128.0,127.0);if(q2<0.0)q2+=256.0;float q3=clamp(floor(v3*KVSCALE+0.5),-128.0,127.0);if(q3<0.0)q3+=256.0;return vec4(q0/255.0,q1/255.0,q2/255.0,q3/255.0);}\n"
            "float decKV8(vec4 c,int idx){float q=(idx==0)?c.r:(idx==1)?c.g:(idx==2)?c.b:c.a;float u=floor(q*255.0+0.5);if(u>=128.0)u-=256.0;return u/KVSCALE;}\n"
            "float decKV(vec4 c,int odd){if(KVPACK==16){float hi=(odd==1)?c.b:c.r;float lo=(odd==1)?c.a:c.g;float u=floor(hi*255.0+0.5)*256.0+floor(lo*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/KVSCALE;}if(KVPACK==8)return decKV8(c,odd);return dec24s(c,KVSCALE);}\n";
    }

    void layer_finish(const char* skip_env = nullptr) {
        bool skip = skip_env && getenv(skip_env);
        if (skip_env && layer_fast_finish_ && !getenv("LAYER_SAFE_FINISH")) {
            string s(skip_env);
            if (s == "NF_RMS" || s == "NF_MAT" || s == "NF_ATT") skip = true;
        }
        if (skip) return;
        if (needs_finish()) { glFlush(); glFinish(); }
    }

    void att_finish() {
        // VC4 needs an explicit finish between FBO -> texture passes.
        layer_finish("NF_ATT");
    }

    GLuint kv_rope_write_program() {
        string key = "att_kvrope_" + to_string(att_dim_) + "_" + to_string(att_kv_dim_) + "_" +
                     to_string(att_head_dim_) + "_" + to_string(att_seq_) + "_" +
                     (att_rope_half_ ? "1" : "0") + "_" + (att_qk_norm_ ? "1" : "0") +
                     "_" + (att_kv_packed_ ? "1" : "0") + "_" + to_string(att_kv_scale_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n" + att_defs() + att_helpers() +
            "uniform sampler2D QKV,RT,KN;uniform float pos;"
            "void main(){int x=int(gl_FragCoord.x);"
            "if(KVPACK==8){"
            "if(x<KVDIM/4){"
            "int base=4*x;int h=base/HEADDIM;int li0=base-h*HEADDIM;"
            "float rk=1.0;"
            "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
            "float zk=dec24q(texture2D(QKV,vec2((float(DIM+h*HEADDIM+z)+0.5)/float(QKVDIM),0.5)));ss+=zk*zk;}"
            "rk=1.0/sqrt(ss/float(HEADDIM)+QKEPS);}"
            "float o0=0.0,o1=0.0,o2=0.0,o3=0.0;"
            "for(int k=0;k<4;k++){int ii=base+k;int li=li0+k;int j,ti;int even=li-(li/2)*2;"
            "float vv=dec24q(texture2D(QKV,vec2((float(DIM+ii)+0.5)/float(QKVDIM),0.5)));"
            "if(QKNORM==1)vv=vv*rk*dec16(texture2D(KN,vec2((float(li)+0.5)/float(HEADDIM),0.5)).rg);"
            "if(ROPEHALF==1){if(li<HALF){j=li+HALF;ti=li;}else{j=li-HALF;ti=li-HALF;}}"
            "else{j=(even==0)?li+1:li-1;ti=li/2;}"
            "int pc=DIM+h*HEADDIM+j;"
            "float bb=dec24q(texture2D(QKV,vec2((float(pc)+0.5)/float(QKVDIM),0.5)));"
            "if(QKNORM==1)bb=bb*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
            "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "float c=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
            "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
            "float oo=first?vv*c-bb*sn:bb*sn+vv*c;if(k==0)o0=oo;else if(k==1)o1=oo;else if(k==2)o2=oo;else o3=oo;}"
            "gl_FragColor=encKV8(o0,o1,o2,o3);return;}"
            "int vb=4*(x-KVDIM/4);"
            "float v0=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb)+0.5)/float(QKVDIM),0.5)));"
            "float v1=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb+1)+0.5)/float(QKVDIM),0.5)));"
            "float v2=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb+2)+0.5)/float(QKVDIM),0.5)));"
            "float v3=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb+3)+0.5)/float(QKVDIM),0.5)));"
            "gl_FragColor=encKV8(v0,v1,v2,v3);return;}"
            "if(KVPACK==16){"
            "if(x<KVDIM/2){"
            "int i0=2*x;int h=i0/HEADDIM;int li0=i0-h*HEADDIM;"
            "float rk=1.0;"
            "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
            "float zk=dec24q(texture2D(QKV,vec2((float(DIM+h*HEADDIM+z)+0.5)/float(QKVDIM),0.5)));ss+=zk*zk;}"
            "rk=1.0/sqrt(ss/float(HEADDIM)+QKEPS);}"
            "float o0=0.0,o1=0.0;"
            "for(int k=0;k<2;k++){int ii=i0+k;int li=li0+k;int j,ti;int even=li-(li/2)*2;"
            "float vv=dec24q(texture2D(QKV,vec2((float(DIM+ii)+0.5)/float(QKVDIM),0.5)));"
            "if(QKNORM==1)vv=vv*rk*dec16(texture2D(KN,vec2((float(li)+0.5)/float(HEADDIM),0.5)).rg);"
            "if(ROPEHALF==1){if(li<HALF){j=li+HALF;ti=li;}else{j=li-HALF;ti=li-HALF;}}"
            "else{j=(even==0)?li+1:li-1;ti=li/2;}"
            "int pc=DIM+h*HEADDIM+j;"
            "float bb=dec24q(texture2D(QKV,vec2((float(pc)+0.5)/float(QKVDIM),0.5)));"
            "if(QKNORM==1)bb=bb*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
            "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "float c=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
            "bool first=(ROPEHALF==1)?(li<HALF):(even==0);"
            "float oo=first?vv*c-bb*sn:bb*sn+vv*c;if(k==0)o0=oo;else o1=oo;}"
            "gl_FragColor=encKV(o0,o1);return;}"
            "int vb=2*(x-KVDIM/2);"
            "float v0=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb)+0.5)/float(QKVDIM),0.5)));"
            "float v1=dec24q(texture2D(QKV,vec2((float(DIM+KVDIM+vb+1)+0.5)/float(QKVDIM),0.5)));"
            "gl_FragColor=encKV(v0,v1);return;}"
            "float v=dec24q(texture2D(QKV,vec2((float(DIM+x)+0.5)/float(QKVDIM),0.5)));"
            "if(x>=KVDIM){gl_FragColor=enc24s(v,KVSCALE);return;}"
            "int h=x/HEADDIM;int i=x-h*HEADDIM;int j,ti;int even=i-(i/2)*2;"
            "float rk=1.0;"
            "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
            "float zk=dec24q(texture2D(QKV,vec2((float(DIM+h*HEADDIM+z)+0.5)/float(QKVDIM),0.5)));ss+=zk*zk;}"
            "rk=1.0/sqrt(ss/float(HEADDIM)+QKEPS);}"
            "float vn=v;if(QKNORM==1)vn=v*rk*dec16(texture2D(KN,vec2((float(i)+0.5)/float(HEADDIM),0.5)).rg);"
            "if(ROPEHALF==1){if(i<HALF){j=i+HALF;ti=i;}else{j=i-HALF;ti=i-HALF;}}"
            "else{j=(even==0)?i+1:i-1;ti=i/2;}"
            "int pc=DIM+h*HEADDIM+j;"
            "float b=dec24q(texture2D(QKV,vec2((float(pc)+0.5)/float(QKVDIM),0.5)));"
            "float bn=b;if(QKNORM==1)bn=b*rk*dec16(texture2D(KN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);"
            "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "float c=dec24s(rc,4096.0),s=dec24s(rs,4096.0);"
            "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
            "float o=first?vn*c-bn*s:bn*s+vn*c;gl_FragColor=enc24s(o,KVSCALE);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint score_program() {
        string key = "att_score_" + to_string(att_dim_) + "_" + to_string(att_kv_dim_) + "_" +
                     to_string(att_heads_) + "_" + to_string(att_head_dim_) + "_" +
                     to_string(att_seq_) + "_" + (att_qk_norm_ ? "1" : "0") +
                     "_" + (att_kv_packed_ ? "1" : "0") + "_" + to_string(att_kv_scale_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n" + att_defs() + att_helpers() +
            "uniform sampler2D R,KV,RT,QN;uniform float pos,invsqrt;"
            "void main(){int t=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
            "if(float(t)>pos){gl_FragColor=enc24s(0.0,1024.0);return;}int kvh=h/GROUP;float s=0.0;"
            "float rq=1.0;"
            "if(QKNORM==1){float ss=0.0;for(int z=0;z<HEADDIM;z++){"
            "float zq=dec24q(texture2D(R,vec2((float(h*HEADDIM+z)+0.5)/float(QKVDIM),0.5)));ss+=zq*zq;}"
            "rq=1.0/sqrt(ss/float(HEADDIM)+QKEPS);}"
            "for(int i=0;i<HEADDIM;i++){"
            "float qv=dec24q(texture2D(R,vec2((float(h*HEADDIM+i)+0.5)/float(QKVDIM),0.5)));"
            "int j,ti;int even=i-(i/2)*2;"
            "if(ROPEHALF==1){if(i<HALF){j=i+HALF;ti=i;}else{j=i-HALF;ti=i-HALF;}}"
            "else{j=(even==0)?i+1:i-1;ti=i/2;}"
            "float b=dec24q(texture2D(R,vec2((float(h*HEADDIM+j)+0.5)/float(QKVDIM),0.5)));"
            "if(QKNORM==1){qv=qv*rq*dec16(texture2D(QN,vec2((float(i)+0.5)/float(HEADDIM),0.5)).rg);"
            "b=b*rq*dec16(texture2D(QN,vec2((float(j)+0.5)/float(HEADDIM),0.5)).rg);}"
            "vec4 rc=texture2D(RT,vec2((float(2*ti)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "vec4 rs=texture2D(RT,vec2((float(2*ti+1)+0.5)/float(2*HALF),(pos+0.5)/float(SEQ)));"
            "float c=dec24s(rc,4096.0),sn=dec24s(rs,4096.0);"
            "bool first=(ROPEHALF==1)?(i<HALF):(even==0);"
            "float q=first?qv*c-b*sn:b*sn+qv*c;"
            "int ki=kvh*HEADDIM+i;int tx=ki;int odd=0;"
            "if(KVPACK==16){tx=ki/2;odd=ki-tx*2;}else if(KVPACK==8){tx=ki/4;odd=ki-tx*4;}"
            "float k=decKV(texture2D(KV,vec2((float(tx)+0.5)/float(KVWIDTH),(float(t)+0.5)/float(SEQ))),odd);"
            "s+=q*k;}"
            "gl_FragColor=enc24s(s*invsqrt,1024.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint stats_program() {
        string key = "att_stats_" + to_string(att_heads_) + "_" + to_string(att_seq_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n" + att_defs() + att_helpers() +
            "uniform sampler2D S;uniform float pos;"
            "void main(){int x=int(gl_FragCoord.x);int h=x/2;int kind=x-h*2;float mx=-1e30;"
            "for(int t=0;t<SEQ;t++){if(float(t)>pos)break;"
            "mx=max(mx,dec24s(texture2D(S,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),1024.0));}"
            "float sum=0.0;"
            "for(int t=0;t<SEQ;t++){if(float(t)>pos)break;"
            "sum+=exp(dec24s(texture2D(S,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),1024.0)-mx);}"
            "gl_FragColor=enc24s(kind==0?mx:sum,1024.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint prob_program() {
        string key = "att_prob_" + to_string(att_heads_) + "_" + to_string(att_seq_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n" + att_defs() + att_helpers() +
            "uniform sampler2D S,ST;uniform float pos;"
            "void main(){int t=int(gl_FragCoord.x);int h=int(gl_FragCoord.y);"
            "vec4 sm=texture2D(ST,vec2((float(2*h)+0.5)/float(2*HEADS),0.5));"
            "vec4 ss=texture2D(ST,vec2((float(2*h+1)+0.5)/float(2*HEADS),0.5));"
            "float mx=dec24s(sm,1024.0);float sum=dec24s(ss,1024.0);float p=0.0;"
            "if(float(t)<=pos && sum>0.0)p=exp(dec24s(texture2D(S,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),1024.0)-mx)/sum;"
            "gl_FragColor=enc24s(p,65536.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint v_program() {
        string key = "att_v_" + to_string(att_dim_) + "_" + to_string(att_kv_dim_) + "_" +
                     to_string(att_heads_) + "_" + to_string(att_head_dim_) + "_" + to_string(att_seq_) +
                     "_" + (att_kv_packed_ ? "1" : "0") + "_" + to_string(att_kv_scale_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n" + att_defs() + att_helpers() +
            "uniform sampler2D P,KV;uniform float pos;"
            "void main(){int x=int(gl_FragCoord.x);int h=x/HEADDIM;int i=x-h*HEADDIM;int kvh=h/GROUP;float acc=0.0;"
            "for(int t=0;t<SEQ;t++){if(float(t)>pos)break;"
            "float p=dec24s(texture2D(P,vec2((float(t)+0.5)/float(SEQ),(float(h)+0.5)/float(HEADS))),65536.0);"
            "int vi=kvh*HEADDIM+i;int tx=KVDIM+vi;int odd=0;"
            "if(KVPACK==16){tx=KVDIM/2+vi/2;odd=vi-(vi/2)*2;}else if(KVPACK==8){tx=KVDIM/4+vi/4;odd=vi-(vi/4)*4;}"
            "float v=decKV(texture2D(KV,vec2((float(tx)+0.5)/float(KVWIDTH),(float(t)+0.5)/float(SEQ))),odd);"
            "acc+=p*v;}"
            "gl_FragColor=q16(acc);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    void kv_rope_write(int layer, int pos) {
        GLuint p = kv_rope_write_program();
        glBindFramebuffer(GL_FRAMEBUFFER, kv_cache_[layer].fbo);
        glViewport(0, pos, att_kv_width_, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qkv_target_.tex);
        glUniform1i(glGetUniformLocation(p, "QKV"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, rope_tex_);
        glUniform1i(glGetUniformLocation(p, "RT"), 1);
        if (att_qk_norm_) {
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, k_norm_tex_[layer]);
            glUniform1i(glGetUniformLocation(p, "KN"), 2);
        }
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

    void att_scores(int layer, int pos) {
        GLuint p = score_program();
        glBindFramebuffer(GL_FRAMEBUFFER, att_scores_.fbo);
        glViewport(0, 0, att_seq_, att_heads_);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qkv_target_.tex);
        glUniform1i(glGetUniformLocation(p, "R"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, kv_cache_[layer].tex);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, rope_tex_);
        glUniform1i(glGetUniformLocation(p, "RT"), 2);
        if (att_qk_norm_) {
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, q_norm_tex_[layer]);
            glUniform1i(glGetUniformLocation(p, "QN"), 3);
        }
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        glUniform1f(glGetUniformLocation(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

    void att_stats(int pos) {
        GLuint p = stats_program();
        glBindFramebuffer(GL_FRAMEBUFFER, att_stats_.fbo);
        glViewport(0, 0, 2 * att_heads_, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, att_scores_.tex);
        glUniform1i(glGetUniformLocation(p, "S"), 0);
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

    void att_probs(int pos) {
        GLuint p = prob_program();
        glBindFramebuffer(GL_FRAMEBUFFER, att_probs_.fbo);
        glViewport(0, 0, att_seq_, att_heads_);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, att_scores_.tex);
        glUniform1i(glGetUniformLocation(p, "S"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, att_stats_.tex);
        glUniform1i(glGetUniformLocation(p, "ST"), 1);
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

    void att_v(int layer, int pos) {
        GLuint p = v_program();
        glBindFramebuffer(GL_FRAMEBUFFER, att_out_.fbo);
        glViewport(0, 0, att_dim_, 1);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, att_probs_.tex);
        glUniform1i(glGetUniformLocation(p, "P"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, kv_cache_[layer].tex);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

    GLuint rms_stats_program() {
        string key = "rms_stats_" + to_string(att_dim_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n#define DIM " + to_string(att_dim_) + "\n"
            "uniform sampler2D X;"
            "float dec16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "void main(){float s=0.0;for(int i=0;i<DIM;i++){float v=dec16(texture2D(X,vec2((float(i)+0.5)/float(DIM),0.5)).rg);s+=v*v;}"
            "float q=floor(s*4096.0+0.5)+8388608.0;"
            "float r0=floor(q/16777216.0),g=floor(q/65536.0)-r0*256.0,b=floor(q/256.0)-r0*65536.0-g*256.0,a=q-r0*16777216.0-g*65536.0-b*256.0;"
            "gl_FragColor=vec4(r0/255.0,g/255.0,b/255.0,a/255.0);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    GLuint rms_norm_program() {
        string key = "rms_norm_" + to_string(att_dim_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        string fs = "precision highp float;\n#define DIM " + to_string(att_dim_) + "\n"
            "uniform sampler2D X,W,S;"
            "float dec16(vec2 c){float u=floor(c.x*255.0+0.5)*256.0+floor(c.y*255.0+0.5);if(u>=32768.0)u-=65536.0;return u/256.0;}"
            "float dec24raw(vec4 c){return floor(c.r*255.0+0.5)*16777216.0+floor(c.g*255.0+0.5)*65536.0+floor(c.b*255.0+0.5)*256.0+floor(c.a*255.0+0.5)-8388608.0;}"
            "vec4 q16(float v){float q=clamp(floor(v*256.0+0.5),-32768.0,32767.0);if(q<0.0)q+=65536.0;float hi=floor(q/256.0),lo=q-hi*256.0;return vec4(hi/255.0,lo/255.0,0.0,1.0);}"
            "void main(){int i=int(gl_FragCoord.x);"
            "float s=dec24raw(texture2D(S,vec2(0.5,0.5)))/4096.0;"
            "float inv=1.0/sqrt(s/float(DIM)+1e-5);"
            "float v=dec16(texture2D(X,vec2((float(i)+0.5)/float(DIM),0.5)).rg);"
            "float w=dec16(texture2D(W,vec2((float(i)+0.5)/float(DIM),0.5)).rg);"
            "gl_FragColor=q16(v*inv*w);}";
        GLuint p = link_program(fs);
        programs_[key] = p;
        return p;
    }

    void draw(const Mat& m, GLuint x, Target& t, int row = 0) {
        GLuint p = (m.type == WeightType::INT8) ? int8_program(m.rows, m.cols)
                  : (m.type == WeightType::INT4) ? int4_program(m.rows, m.cols)
                                                 : int16_program(m.rows, m.cols);
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m.tex);
        glUniform1i(glGetUniformLocation(p, "W"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(p, "X"), 1);
        if (m.type == WeightType::INT4) {
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, m.q4_param_tex);
            glUniform1i(glGetUniformLocation(p, "P"), 2);
            glUniform1f(glGetUniformLocation(p, "ps"), m.q4_param_scale);
        }
        glUniform1f(glGetUniformLocation(p, "ws"), m.scale);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        // VC4 is unreliable for a single 1536-column pass (the logits tile
        // and fused-W13 width).  Split only that width; other widths are fine
        // and keeping them in one draw avoids per-draw overhead.
        const int strip = (m.rows == 1536) ? 768 : m.rows;
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
            int w = min(strip, m.rows - x0);
            glViewport(x0, row, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
    }

    vector<int32_t> read_i32(Target& t, int w, int h) {
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        vector<int32_t> out((size_t)w * h);
        for (int i = 0; i < w * h; ++i) {
            int j = i * 4;
            int64_t v = (int64_t)b[j] * 16777216 + (int64_t)b[j + 1] * 65536 +
                        (int64_t)b[j + 2] * 256 + b[j + 3] - 8388608;
            out[i] = (int32_t)v;
        }
        return out;
    }
};

// -----------------------------------------------------------------------------
// Tokenizer: llama2.c tokenizer.bin, supports ByteLevel/BPE and special tokens.
// -----------------------------------------------------------------------------
class Tokenizer {
public:
    bool load(const string& path, int n) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open tokenizer: " << path << "\n"; return false; }
        int maxlen = 0; f.read((char*)&maxlen, 4);
        for (int i = 0; i < n; ++i) {
            float score = 0; int len = 0;
            f.read((char*)&score, 4); f.read((char*)&len, 4);
            string s(len, '\0'); f.read(&s[0], len);
            vocab_.push_back(s); scores_.push_back(score); lookup_[s] = i;
        }
        for (int i = 0; i < (int)vocab_.size(); ++i) {
            const string& s = vocab_[i];
            // Chat/tool/thinking markers are often added tokens rather than
            // BPE merges (e.g. <think>).  Treat any non-hex <...> token as
            // atomic so the byte-level BPE cannot split it.
            if (s.size() > 2 && s.front() == '<' && s.back() == '>' && s.compare(0, 3, "<0x") != 0)
                specials_.push_back({s, i});
        }
        sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });
        return true;
    }
    void set_qwen(bool q) { qwen_ = q; }
    void set_bytelevel(bool b) { bytelevel_ = b; }

    vector<int> encode(const string& text, bool bos = true, bool dummy_prefix = true) {
        vector<int> t;
        if (bos) t.push_back(1);
        if (dummy_prefix && !text.empty()) {
            auto it = lookup_.find(" ");
            if (it != lookup_.end()) t.push_back(it->second);
        }
        for (size_t i = 0; i < text.size();) {
            bool matched = false;
            for (auto& sp : specials_) {
                if (text.compare(i, sp.first.size(), sp.first) == 0) {
                    t.push_back(sp.second); i += sp.first.size(); matched = true; break;
                }
            }
            if (matched) continue;
            unsigned char c = text[i];
            if (bytelevel_) {
                string bs(1, (char)c);
                auto bt = lookup_.find(bs);
                t.push_back(bt != lookup_.end() ? bt->second : (int)c + 3);
                ++i;
                continue;
            }
            int len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > text.size()) len = 1;
            string s = text.substr(i, len); i += len;
            auto it = lookup_.find(s);
            if (it != lookup_.end()) t.push_back(it->second);
            else {
                for (unsigned char b : s) {
                    string bs(1, (char)b);
                    auto bt = lookup_.find(bs);
                    t.push_back(bt != lookup_.end() ? bt->second : (int)b + 3);
                }
            }
        }
        while (true) {
            float best = -1e30f; int bid = -1, ba = -1, bb = -1;
            for (size_t i = 0; i + 1 < t.size(); ++i) {
                string s = vocab_[t[i]] + vocab_[t[i + 1]];
                auto it = lookup_.find(s);
                if (it != lookup_.end() && scores_[it->second] > best) {
                    best = scores_[it->second]; bid = it->second;
                    ba = t[i]; bb = t[i + 1];
                }
            }
            if (bid < 0) break;
            // Merge every non-overlapping occurrence of the winning pair, as
            // the byte-level BPE reference does.
            for (size_t i = 0; i + 1 < t.size(); ) {
                if (t[i] == ba && t[i + 1] == bb) {
                    t[i] = bid;
                    t.erase(t.begin() + i + 1);
                } else {
                    ++i;
                }
            }
        }
        return t;
    }

    string decode(const vector<int>& toks, int prev = 0) {
        string out;
        for (int id : toks) {
            string p = vocab_[id];
            if (!qwen_ && prev == 1 && !p.empty() && p[0] == ' ') p = p.substr(1);
            if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
                unsigned v = 0;
                if (sscanf(p.c_str(), "<0x%02X>", &v) == 1) {
                    out.push_back((char)v); prev = id; continue;
                }
            }
            out += p; prev = id;
        }
        return out;
    }

private:
    vector<string> vocab_;
    vector<float> scores_;
    unordered_map<string, int> lookup_;
    vector<pair<string, int>> specials_;
    bool qwen_ = false;
    bool bytelevel_ = false;
};

// -----------------------------------------------------------------------------
// Model: llama2.c fp32, Q8M1 (int8) and I16M1 (int16).
// -----------------------------------------------------------------------------
class Model {
public:
    int dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
    int vocab = 0, seq = 0, kv_dim = 0, head_dim = 0;
    float rope_theta = 10000.0f;
    bool rope_half = false;
    bool shared_embeddings = true;
    bool qk_norm = false;
    uint64_t source_size = 0, source_mtime = 0;

    vector<float> emb_f32, rms_att, rms_ffn, rms_final;
    vector<float> q_norm, k_norm; // [layers * head_dim], only if qk_norm
    vector<int8_t> emb_q8;
    vector<int16_t> emb_i16;
    vector<float> emb_scale;
    bool has_q8_emb = false, has_i16_emb = false;

    vector<Mat> qkv, wo, w13, w1, w3, w2, logits_tiles;
    vector<Mat> w1a, w1b, w3a, w3b, w2a, w2b;
    bool ffn_chunked = false;
    int ffn_chunk = 0; // first hidden chunk; second is hidden - ffn_chunk

    bool load(const string& path) {
        struct stat st {};
        if (stat(path.c_str(), &st) == 0) {
            source_size = (uint64_t)st.st_size;
            source_mtime = (uint64_t)st.st_mtime;
        }
        ifstream magic_file(path, ios::binary);
        uint32_t magic = 0; magic_file.read((char*)&magic, 4);
        if (magic == MAGIC_Q8 || magic == MAGIC_Q8_V2) return load_q8(path);
        if (magic == MAGIC_I16 || magic == MAGIC_I16_V2) return load_i16(path);
        if (magic == MAGIC_Q4) return load_q4(path);
        return load_legacy(path);
    }

    void split_ffn() {
        if (hidden <= 2048) return;
        if (hidden > 4096) { cerr << "hidden too large for two FFN chunks\n"; exit(1); }
        ffn_chunked = true;
        ffn_chunk = (hidden + 1) / 2;
        for (int l = 0; l < layers; ++l) {
            w1a.push_back(slice_rows(w1[l], 0, ffn_chunk));
            w1b.push_back(slice_rows(w1[l], ffn_chunk, hidden));
            w3a.push_back(slice_rows(w3[l], 0, ffn_chunk));
            w3b.push_back(slice_rows(w3[l], ffn_chunk, hidden));
            w2a.push_back(slice_cols(w2[l], 0, ffn_chunk));
            w2b.push_back(slice_cols(w2[l], ffn_chunk, hidden));
        }
        w1.clear(); w3.clear(); w2.clear();
    }

    static bool quantize_q4(Mat& m) {
        if (m.type == WeightType::INT4) return true;
        if ((m.cols % 32) != 0) return false;
        const int group = 32;
        const int groups = (m.cols + group - 1) / group;
        vector<uint8_t> q4((size_t)m.rows * (m.cols / 2), 0);
        vector<float> fs((size_t)m.rows * groups), fm((size_t)m.rows * groups);
        float pmax = 0;
        for (int r = 0; r < m.rows; ++r) {
            for (int g = 0; g < groups; ++g) {
                int c0 = g * group, c1 = min(m.cols, c0 + group);
                float lo = 1e30f, hi = -1e30f;
                for (int c = c0; c < c1; ++c) {
                    float w = (m.type == WeightType::INT8)
                        ? (float)m.q8[(size_t)r * m.cols + c] * m.scale
                        : (float)m.q16[(size_t)r * m.cols + c] * m.scale;
                    lo = min(lo, w); hi = max(hi, w);
                }
                float sc = (hi - lo) / 15.0f;
                if (sc <= 0.0f) sc = 1.0f;
                fs[(size_t)r * groups + g] = sc;
                fm[(size_t)r * groups + g] = lo;
                pmax = max(pmax, max(fabsf(sc), fabsf(lo)));
                for (int c = c0; c < c1; ++c) {
                    float w = (m.type == WeightType::INT8)
                        ? (float)m.q8[(size_t)r * m.cols + c] * m.scale
                        : (float)m.q16[(size_t)r * m.cols + c] * m.scale;
                    int q = (int)lrintf((w - lo) / sc);
                    q = max(0, min(15, q));
                    size_t bi = (size_t)r * (m.cols / 2) + c / 2;
                    if ((c & 1) == 0) q4[bi] |= (uint8_t)q;
                    else q4[bi] |= (uint8_t)(q << 4);
                }
            }
        }
        float ps = pmax > 0 ? pmax / 32767.0f : 1.0f;
        m.q4_scale.resize((size_t)m.rows * groups);
        m.q4_min.resize((size_t)m.rows * groups);
        for (size_t i = 0; i < fs.size(); ++i) {
            m.q4_scale[i] = (int16_t)max(-32767L, min(32767L, lrintf(fs[i] / ps)));
            m.q4_min[i] = (int16_t)max(-32767L, min(32767L, lrintf(fm[i] / ps)));
        }
        m.q4.swap(q4);
        m.group = group; m.groups = groups; m.q4_param_scale = ps;
        m.type = WeightType::INT4;
        vector<int8_t>().swap(m.q8);
        vector<int16_t>().swap(m.q16);
        return true;
    }

    bool quantize_q4_weights(const string& scope) {
        bool attn = scope.empty() || scope == "all" || scope == "attn";
        bool ffn = scope.empty() || scope == "all" || scope == "ffn";
        bool ok = true;
        for (int l = 0; l < layers; ++l) {
            if (attn) { ok &= quantize_q4(qkv[l]); ok &= quantize_q4(wo[l]); }
            if (ffn) {
                if (!w1.empty()) { ok &= quantize_q4(w1[l]); ok &= quantize_q4(w3[l]); }
                else if (!w13.empty()) ok &= quantize_q4(w13[l]);
                if (!w2.empty()) ok &= quantize_q4(w2[l]);
                if (!w1a.empty()) {
                    ok &= quantize_q4(w1a[l]); ok &= quantize_q4(w3a[l]);
                    ok &= quantize_q4(w1b[l]); ok &= quantize_q4(w3b[l]);
                    ok &= quantize_q4(w2a[l]); ok &= quantize_q4(w2b[l]);
                }
            }
        }
        if (!ok) cerr << "Q4 dimension not divisible by 32; some tensors kept at original precision\n";
        return ok;
    }

    void apply_precision(const string& layers_spec, const string& parts_spec) {
        auto has_part = [&](const char* name) {
            if (parts_spec.empty() || parts_spec == "all") return true;
            string needle(name);
            size_t pos = 0;
            while (pos < parts_spec.size()) {
                size_t comma = parts_spec.find(',', pos);
                string item = parts_spec.substr(pos, comma == string::npos ? string::npos : comma - pos);
                if (item == needle) return true;
                if (comma == string::npos) break;
                pos = comma + 1;
            }
            return false;
        };
        auto has_layer = [&](int layer) {
            if (layers_spec.empty() || layers_spec == "all") return true;
            size_t pos = 0;
            while (pos < layers_spec.size()) {
                size_t comma = layers_spec.find(',', pos);
                string item = layers_spec.substr(pos, comma == string::npos ? string::npos : comma - pos);
                if (atoi(item.c_str()) == layer) return true;
                if (comma == string::npos) break;
                pos = comma + 1;
            }
            return false;
        };
        if (has_part("emb") && has_i16_emb) {
            emb_q8.resize(emb_i16.size());
            for (int row = 0; row < vocab; ++row) {
                float mx = 0;
                for (int i = 0; i < dim; ++i) mx = max(mx, fabsf((float)emb_i16[(size_t)row * dim + i] * emb_scale[row]));
                if (mx == 0) mx = 1;
                float s = mx / 127.0f;
                for (int i = 0; i < dim; ++i) {
                    long q = lrintf(((float)emb_i16[(size_t)row * dim + i] * emb_scale[row]) / s);
                    emb_q8[(size_t)row * dim + i] = (int8_t)max(-127L, min(127L, q));
                }
                emb_scale[row] = s;
            }
            emb_i16.clear(); has_i16_emb = false; has_q8_emb = true;
        }
        auto convert = [&](Mat& m) {
            if (m.type == WeightType::INT8) return;
            float mx = 0;
            for (int i = 0; i < m.rows * m.cols; ++i) mx = max(mx, fabsf((float)m.q16[i] * m.scale));
            if (mx == 0) mx = 1;
            float s = mx / 127.0f;
            m.q8.resize((size_t)m.rows * m.cols);
            for (int i = 0; i < m.rows * m.cols; ++i) {
                long q = lrintf(((float)m.q16[i] * m.scale) / s);
                m.q8[i] = (int8_t)max(-127L, min(127L, q));
            }
            m.q16.clear(); m.type = WeightType::INT8; m.scale = s;
        };
        for (int l = 0; l < layers; ++l) {
            if (!has_layer(l)) continue;
            if (has_part("qkv")) convert(qkv[l]);
            if (has_part("wo")) convert(wo[l]);
            if (has_part("ffn")) { if (!w1.empty()) { convert(w1[l]); convert(w3[l]); } else convert(w13[l]); }
            if (has_part("w2")) convert(w2[l]);
        }
        if (has_part("logits")) for (auto& m : logits_tiles) convert(m);
    }

private:
    static void read_exact(ifstream& f, void* p, size_t n) {
        f.read((char*)p, n);
        if (!f) { cerr << "truncated checkpoint\n"; exit(1); }
    }

    Mat read_mat(ifstream& f, bool int16) {
        Mat m;
        int32_t rows = 0, cols = 0; float scale = 0;
        read_exact(f, &rows, 4); read_exact(f, &cols, 4); read_exact(f, &scale, 4);
        m.rows = rows; m.cols = cols; m.scale = scale;
        m.type = int16 ? WeightType::INT16 : WeightType::INT8;
        if (int16) {
            m.q16.resize((size_t)rows * cols);
            read_exact(f, m.q16.data(), m.q16.size() * 2);
        } else {
            m.q8.resize((size_t)rows * cols);
            read_exact(f, m.q8.data(), m.q8.size());
        }
        return m;
    }

    bool load_legacy(const string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { cerr << "cannot open model: " << path << "\n"; return false; }
        struct stat st; fstat(fd, &st);
        uint8_t* base = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) { cerr << "mmap failed\n"; return false; }
        int32_t cfg[7]; memcpy(cfg, base, 28);
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; int vs = cfg[5]; seq = cfg[6];
        vocab = abs(vs); shared_embeddings = vs > 0;
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        const float* p = (const float*)(base + 28);
        auto take = [&](size_t n) { const float* r = p; p += n; return r; };
        const float* ep = take((size_t)vocab * dim);
        emb_f32.assign(ep, ep + (size_t)vocab * dim);
        const float* ra = take((size_t)layers * dim); rms_att.assign(ra, ra + (size_t)layers * dim);
        const float* wq = take((size_t)layers * dim * dim);
        const float* wk = take((size_t)layers * kv_dim * dim);
        const float* wv = take((size_t)layers * kv_dim * dim);
        const float* wo_p = take((size_t)layers * dim * dim);
        const float* rf = take((size_t)layers * dim); rms_ffn.assign(rf, rf + (size_t)layers * dim);
        const float* w1_p = take((size_t)layers * hidden * dim);
        const float* w2_p = take((size_t)layers * dim * hidden);
        const float* w3_p = take((size_t)layers * hidden * dim);
        const float* rfin = take(dim); rms_final.assign(rfin, rfin + dim);
        p += (size_t)seq * head_dim;
        const float* wcls = shared_embeddings ? ep : p;

        for (int l = 0; l < layers; ++l) {
            vector<float> fused((size_t)(dim + 2 * kv_dim) * dim);
            memcpy(fused.data(), wq + (size_t)l * dim * dim, (size_t)dim * dim * 4);
            memcpy(fused.data() + (size_t)dim * dim, wk + (size_t)l * kv_dim * dim, (size_t)kv_dim * dim * 4);
            memcpy(fused.data() + (size_t)(dim + kv_dim) * dim, wv + (size_t)l * kv_dim * dim, (size_t)kv_dim * dim * 4);
            qkv.push_back(quantize(fused.data(), dim + 2 * kv_dim, dim, WeightType::INT8));
            wo.push_back(quantize(wo_p + (size_t)l * dim * dim, dim, dim, WeightType::INT8));
            if (2 * hidden <= 2048) {
                vector<float> fused13((size_t)2 * hidden * dim);
                memcpy(fused13.data(), w1_p + (size_t)l * hidden * dim, (size_t)hidden * dim * 4);
                memcpy(fused13.data() + (size_t)hidden * dim, w3_p + (size_t)l * hidden * dim, (size_t)hidden * dim * 4);
                w13.push_back(quantize(fused13.data(), 2 * hidden, dim, WeightType::INT8));
            } else {
                w1.push_back(quantize(w1_p + (size_t)l * hidden * dim, hidden, dim, WeightType::INT8));
                w3.push_back(quantize(w3_p + (size_t)l * hidden * dim, hidden, dim, WeightType::INT8));
            }
            w2.push_back(quantize(w2_p + (size_t)l * dim * hidden, dim, hidden, WeightType::INT8));
        }
        for (int st = 0; st < vocab; st += 1536)
            logits_tiles.push_back(quantize(wcls + (size_t)st * dim, min(1536, vocab - st), dim, WeightType::INT8));
        munmap(base, st.st_size);
        return true;
    }

    bool load_q4(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open q4 model\n"; return false; }
        uint32_t magic = 0; int32_t cfg[7]; float theta = 0; int32_t half = 0, split = 0, qk = 0;
        read_exact(f, &magic, 4); read_exact(f, cfg, 28); read_exact(f, &theta, 4);
        read_exact(f, &half, 4); read_exact(f, &split, 4); read_exact(f, &qk, 4);
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        rope_theta = theta; rope_half = half != 0; qk_norm = qk != 0;
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        has_q8_emb = true;
        emb_q8.resize((size_t)vocab * dim); read_exact(f, emb_q8.data(), emb_q8.size());
        emb_scale.resize(vocab); read_exact(f, emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(f, rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(f, rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(f, rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(f, q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(f, k_norm.data(), k_norm.size() * 4);
        }
        auto read_q4_mat = [&]() {
            int32_t rows = 0, cols = 0, groups = 0; float ps = 1.0f;
            read_exact(f, &rows, 4); read_exact(f, &cols, 4);
            read_exact(f, &groups, 4); read_exact(f, &ps, 4);
            Mat m; m.rows = rows; m.cols = cols; m.type = WeightType::INT4;
            m.group = 32; m.groups = groups; m.q4_param_scale = ps;
            m.q4.resize((size_t)rows * (cols / 2));
            read_exact(f, m.q4.data(), m.q4.size());
            m.q4_scale.resize((size_t)rows * groups);
            read_exact(f, m.q4_scale.data(), m.q4_scale.size() * 2);
            m.q4_min.resize((size_t)rows * groups);
            read_exact(f, m.q4_min.data(), m.q4_min.size() * 2);
            return m;
        };
        for (int l = 0; l < layers; ++l) {
            qkv.push_back(read_q4_mat());
            wo.push_back(read_q4_mat());
            if (split) { w1.push_back(read_q4_mat()); w3.push_back(read_q4_mat()); }
            else w13.push_back(read_q4_mat());
            w2.push_back(read_q4_mat());
        }
        int nt = (vocab + 1535) / 1536;
        for (int i = 0; i < nt; ++i) {
            int32_t rows = 0, cols = 0; float sc = 1.0f;
            read_exact(f, &rows, 4); read_exact(f, &cols, 4); read_exact(f, &sc, 4);
            Mat m; m.rows = rows; m.cols = cols; m.scale = sc; m.type = WeightType::INT8;
            m.q8.resize((size_t)rows * cols);
            read_exact(f, m.q8.data(), m.q8.size());
            logits_tiles.push_back(m);
        }
        return true;
    }

    bool load_q8(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open q8 model\n"; return false; }
        uint32_t magic = 0; int32_t cfg[7]; int32_t split = 0;
        read_exact(f, &magic, 4); read_exact(f, cfg, 28);
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        if (magic == MAGIC_Q8_V2) {
            int32_t half = 0, qk = 0;
            read_exact(f, &rope_theta, 4); read_exact(f, &half, 4);
            read_exact(f, &split, 4); read_exact(f, &qk, 4);
            rope_half = half != 0; qk_norm = qk != 0;
        } else {
            read_exact(f, &split, 4);
            if (getenv("ROPE_THETA")) rope_theta = atof(getenv("ROPE_THETA"));
            rope_half = getenv("ROPE_HALF") != nullptr;
        }
        has_q8_emb = true;
        emb_q8.resize((size_t)vocab * dim);
        read_exact(f, emb_q8.data(), emb_q8.size());
        emb_scale.resize(vocab); read_exact(f, emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(f, rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(f, rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(f, rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(f, q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(f, k_norm.data(), k_norm.size() * 4);
        }
        for (int l = 0; l < layers; ++l) {
            qkv.push_back(read_mat(f, false));
            wo.push_back(read_mat(f, false));
            if (split) { w1.push_back(read_mat(f, false)); w3.push_back(read_mat(f, false)); }
            else w13.push_back(read_mat(f, false));
            w2.push_back(read_mat(f, false));
        }
        int nt = (vocab + 1535) / 1536;
        for (int i = 0; i < nt; ++i) logits_tiles.push_back(read_mat(f, false));
        return true;
    }

    bool load_i16(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "cannot open i16 model\n"; return false; }
        uint32_t magic = 0; int32_t cfg[7]; float theta = 0; int32_t half = 0, split = 0;
        read_exact(f, &magic, 4); read_exact(f, cfg, 28); read_exact(f, &theta, 4);
        read_exact(f, &half, 4); read_exact(f, &split, 4);
        if (magic == MAGIC_I16_V2) {
            int32_t qk = 0; read_exact(f, &qk, 4); qk_norm = qk != 0;
        }
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        rope_theta = theta; rope_half = half != 0;
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        has_i16_emb = true;
        emb_i16.resize((size_t)vocab * dim);
        read_exact(f, emb_i16.data(), emb_i16.size() * 2);
        emb_scale.resize(vocab); read_exact(f, emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(f, rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(f, rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(f, rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(f, q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(f, k_norm.data(), k_norm.size() * 4);
        }
        for (int l = 0; l < layers; ++l) {
            qkv.push_back(read_mat(f, true));
            wo.push_back(read_mat(f, true));
            if (split) { w1.push_back(read_mat(f, true)); w3.push_back(read_mat(f, true)); }
            else w13.push_back(read_mat(f, true));
            w2.push_back(read_mat(f, true));
        }
        int nt = (vocab + 1535) / 1536;
        for (int i = 0; i < nt; ++i) logits_tiles.push_back(read_mat(f, true));
        return true;
    }

    static Mat slice_rows(const Mat& src, int r0, int r1) {
        Mat m; m.rows = r1 - r0; m.cols = src.cols; m.type = src.type; m.scale = src.scale;
        if (src.type == WeightType::INT8) {
            m.q8.resize((size_t)m.rows * m.cols);
            memcpy(m.q8.data(), src.q8.data() + (size_t)r0 * m.cols, m.q8.size());
        } else if (src.type == WeightType::INT16) {
            m.q16.resize((size_t)m.rows * m.cols);
            memcpy(m.q16.data(), src.q16.data() + (size_t)r0 * m.cols, m.q16.size() * 2);
        } else {
            m.group = src.group; m.groups = src.groups; m.q4_param_scale = src.q4_param_scale;
            m.q4.resize((size_t)m.rows * (m.cols / 2));
            memcpy(m.q4.data(), src.q4.data() + (size_t)r0 * (m.cols / 2), m.q4.size());
            m.q4_scale.resize((size_t)m.rows * m.groups);
            memcpy(m.q4_scale.data(), src.q4_scale.data() + (size_t)r0 * m.groups, m.q4_scale.size() * 2);
            m.q4_min.resize((size_t)m.rows * m.groups);
            memcpy(m.q4_min.data(), src.q4_min.data() + (size_t)r0 * m.groups, m.q4_min.size() * 2);
        }
        return m;
    }

    static Mat slice_cols(const Mat& src, int c0, int c1) {
        Mat m; m.rows = src.rows; m.cols = c1 - c0; m.type = src.type; m.scale = src.scale;
        if (src.type == WeightType::INT8) {
            m.q8.resize((size_t)m.rows * m.cols);
            for (int r = 0; r < m.rows; ++r)
                memcpy(m.q8.data() + (size_t)r * m.cols, src.q8.data() + (size_t)r * src.cols + c0, m.cols);
        } else if (src.type == WeightType::INT16) {
            m.q16.resize((size_t)m.rows * m.cols);
            for (int r = 0; r < m.rows; ++r)
                memcpy(m.q16.data() + (size_t)r * m.cols, src.q16.data() + (size_t)r * src.cols + c0, m.cols * 2);
        } else {
            if ((c0 % src.group) != 0 || (m.cols % src.group) != 0)
                throw runtime_error("q4 column slice must be group aligned");
            m.group = src.group; m.groups = m.cols / src.group; m.q4_param_scale = src.q4_param_scale;
            const int g0 = c0 / src.group;
            m.q4.resize((size_t)m.rows * (m.cols / 2), 0);
            for (int r = 0; r < m.rows; ++r) {
                for (int c = 0; c < m.cols; c += 2) {
                    int s0 = c0 + c, s1 = s0 + 1;
                    int b0 = src.q4[(size_t)r * (src.cols / 2) + s0 / 2];
                    int b1 = src.q4[(size_t)r * (src.cols / 2) + s1 / 2];
                    int q0 = (s0 & 1) ? (b0 >> 4) : (b0 & 15);
                    int q1 = (s1 & 1) ? (b1 >> 4) : (b1 & 15);
                    m.q4[(size_t)r * (m.cols / 2) + c / 2] = (uint8_t)(q0 | (q1 << 4));
                }
            }
            m.q4_scale.resize((size_t)m.rows * m.groups);
            m.q4_min.resize((size_t)m.rows * m.groups);
            for (int r = 0; r < m.rows; ++r) {
                memcpy(m.q4_scale.data() + (size_t)r * m.groups,
                       src.q4_scale.data() + (size_t)r * src.groups + g0, m.groups * 2);
                memcpy(m.q4_min.data() + (size_t)r * m.groups,
                       src.q4_min.data() + (size_t)r * src.groups + g0, m.groups * 2);
            }
        }
        return m;
    }

    static Mat quantize(const float* src, int rows, int cols, WeightType type) {
        Mat m; m.rows = rows; m.cols = cols; m.type = type;
        float mx = 0;
        for (int i = 0; i < rows * cols; ++i) mx = max(mx, fabsf(src[i]));
        if (mx == 0) mx = 1;
        if (type == WeightType::INT16) {
            m.scale = mx / 32767.0f; m.q16.resize((size_t)rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                long q = lrintf(src[i] / m.scale);
                m.q16[i] = (int16_t)max(-32767L, min(32767L, q));
            }
        } else {
            m.scale = mx / 127.0f; m.q8.resize((size_t)rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                long q = lrintf(src[i] / m.scale);
                m.q8[i] = (int8_t)max(-127L, min(127L, q));
            }
        }
        return m;
    }
};

// -----------------------------------------------------------------------------
// Session: KV cache, RoPE, attention and transformer forward pass.
// -----------------------------------------------------------------------------
class Session {
public:
    Session(Model& model, GPU& gpu) : m_(model), gpu_(gpu) {
        gpu_layer_ = gpu_.gpu_layer_enabled();
        gpu_attn_ = gpu_layer_ || gpu_.gpu_attn_enabled();
        if (!gpu_attn_) {
            kv_cache_k_.resize(m_.layers, vector<float>((size_t)m_.seq * m_.kv_dim, 0));
            kv_cache_v_.resize(m_.layers, vector<float>((size_t)m_.seq * m_.kv_dim, 0));
        }
        const int half = m_.head_dim / 2;
        rope_cos_.resize((size_t)m_.seq * half);
        rope_sin_.resize((size_t)m_.seq * half);
        for (int p = 0; p < m_.seq; ++p) {
            for (int i = 0; i < half; ++i) {
                float freq = 1.0f / powf(m_.rope_theta, (2.0f * i) / m_.head_dim);
                float a = p * freq;
                rope_cos_[(size_t)p * half + i] = cosf(a);
                rope_sin_[(size_t)p * half + i] = sinf(a);
            }
        }
        if (gpu_attn_)
            gpu_.init_attention(m_.dim, m_.kv_dim, m_.heads, m_.kv_heads, m_.head_dim,
                                m_.seq, m_.rope_half, m_.qk_norm, m_.layers,
                                m_.q_norm, m_.k_norm, rope_cos_, rope_sin_);
        if (gpu_layer_)
            gpu_.init_persistent(m_.dim, m_.layers, m_.rms_att, m_.rms_ffn, m_.rms_final);
    }

    bool save_prefix_cache(const string& path, const vector<int>& toks) {
        if (!gpu_attn_ || gpu_.kv_cache_.empty() || toks.empty() || (int)toks.size() > m_.seq) return false;
        ofstream f(path, ios::binary);
        if (!f) return false;
        const char magic[8] = {'M','M','K','V','1','\0','\0','\0'};
        f.write(magic, 8);
        auto u32 = [&](uint32_t v) { f.write((char*)&v, 4); };
        auto u64 = [&](uint64_t v) { f.write((char*)&v, 8); };
        u32(1);
        u64(m_.source_size); u64(m_.source_mtime);
        u32(m_.dim); u32(m_.hidden); u32(m_.layers); u32(m_.heads);
        u32(m_.kv_heads); u32(m_.vocab); u32(m_.seq); u32(m_.head_dim);
        f.write((char*)&m_.rope_theta, 4);
        u32((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
            (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u));
        u32((uint32_t)toks.size());
        u32((uint32_t)gpu_.att_kv_width_);
        for (int t : toks) u32((uint32_t)t);
        gpu_.sync();
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        const int width = gpu_.att_kv_width_, n = (int)toks.size();
        vector<uint8_t> data((size_t)width * n * 4);
        for (int l = 0; l < m_.layers; ++l) {
            glBindFramebuffer(GL_FRAMEBUFFER, gpu_.kv_cache_[l].fbo);
            glReadPixels(0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            f.write((char*)data.data(), data.size());
        }
        return (bool)f;
    }

    bool load_prefix_cache(const string& path, vector<int>& toks) {
        if (!gpu_attn_ || gpu_.kv_cache_.empty()) return false;
        ifstream f(path, ios::binary);
        if (!f) return false;
        char magic[8] {};
        f.read(magic, 8);
        if (!f || memcmp(magic, "MMKV1", 5) != 0) return false;
        auto u32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
        auto u64 = [&](uint64_t& v) { f.read((char*)&v, 8); };
        uint32_t version = 0, dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
        uint32_t vocab = 0, seq = 0, head_dim = 0, flags = 0, n = 0, width = 0;
        uint64_t size = 0, mtime = 0; float theta = 0;
        u32(version); u64(size); u64(mtime);
        u32(dim); u32(hidden); u32(layers); u32(heads); u32(kv_heads);
        u32(vocab); u32(seq); u32(head_dim);
        f.read((char*)&theta, 4);
        u32(flags); u32(n); u32(width);
        if (!f || version != 1 || dim != (uint32_t)m_.dim || hidden != (uint32_t)m_.hidden ||
            layers != (uint32_t)m_.layers || heads != (uint32_t)m_.heads ||
            kv_heads != (uint32_t)m_.kv_heads || vocab != (uint32_t)m_.vocab ||
            seq != (uint32_t)m_.seq || head_dim != (uint32_t)m_.head_dim ||
            size != m_.source_size || mtime != m_.source_mtime ||
            theta != m_.rope_theta || width != (uint32_t)gpu_.att_kv_width_ ||
            n == 0 || n > (uint32_t)m_.seq ||
            flags != ((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
                      (gpu_.att_kv_bits_ == 16 ? 4u : gpu_.att_kv_bits_ == 8 ? 8u : 0u))) return false;
        vector<int> tmp(n);
        for (uint32_t i = 0; i < n; ++i) { uint32_t v = 0; u32(v); tmp[i] = (int)v; }
        if (!f) return false;
        vector<uint8_t> data((size_t)width * n * 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        for (int l = 0; l < m_.layers; ++l) {
            f.read((char*)data.data(), data.size());
            if (!f) return false;
            glBindTexture(GL_TEXTURE_2D, gpu_.kv_cache_[l].tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        }
        gpu_.sync();
        toks = std::move(tmp);
        return true;
    }

    vector<float> embedding(int token) {
        vector<float> x(m_.dim);
        if (m_.has_i16_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f((float)m_.emb_i16[(size_t)token * m_.dim + i] * s);
        } else if (m_.has_q8_emb) {
            float s = m_.emb_scale[token];
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f((float)m_.emb_q8[(size_t)token * m_.dim + i] * s);
        } else {
            for (int i = 0; i < m_.dim; ++i)
                x[i] = quantize_act_f(m_.emb_f32[(size_t)token * m_.dim + i]);
        }
        return x;
    }

    vector<float> forward(int token, int pos) {
        vector<float> x = embedding(token);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return x;
            vector<float> xb = rms_norm(x, m_.rms_att.data() + (size_t)l * m_.dim);
            vector<float> wo;
            if (gpu_attn_) {
                GLuint xtex = gpu_.upload_input(xb);
                GLuint atex = gpu_.attention_block(m_.qkv[l], xtex, l, pos);
                wo = gpu_.matvec_from(m_.wo[l], atex);
            } else {
                vector<float> qkv = gpu_.matvec(m_.qkv[l], xb);
                vector<float> q(qkv.begin(), qkv.begin() + m_.dim);
                vector<float> k(qkv.begin() + m_.dim, qkv.begin() + m_.dim + m_.kv_dim);
                vector<float> v(qkv.begin() + m_.dim + m_.kv_dim, qkv.begin() + m_.dim + 2 * m_.kv_dim);
                if (m_.qk_norm) {
                    q = head_rms_norm(q, m_.heads, m_.q_norm.data() + (size_t)l * m_.head_dim);
                    k = head_rms_norm(k, m_.kv_heads, m_.k_norm.data() + (size_t)l * m_.head_dim);
                }
                q = rope(q, m_.heads, pos);
                k = rope(k, m_.kv_heads, pos);
                memcpy(kv_cache_k_[l].data() + (size_t)pos * m_.kv_dim, k.data(), m_.kv_dim * 4);
                memcpy(kv_cache_v_[l].data() + (size_t)pos * m_.kv_dim, v.data(), m_.kv_dim * 4);
                vector<float> att = attention(q, l, pos);
                vector<float> att_q16(att.size());
                for (size_t i = 0; i < att.size(); ++i) att_q16[i] = quantize_act_f(att[i]);
                wo = gpu_.matvec(m_.wo[l], att_q16);
            }
            vector<float> x1 = add_q16(x, wo);
            vector<float> xbf = rms_norm(x1, m_.rms_ffn.data() + (size_t)l * m_.dim);
            vector<float> o2;
            bool gpu_ffn = !m_.w1.empty();
            if (getenv("CPU_FFN")) gpu_ffn = false;
            else if (getenv("GPU_FFN")) gpu_ffn = atoi(getenv("GPU_FFN")) != 0;
            if (!gpu_ffn) {
                vector<float> h;
                if (m_.ffn_chunked) {
                    vector<float> h1 = gpu_.matvec(m_.w1a[l], xbf);
                    vector<float> h3 = gpu_.matvec(m_.w3a[l], xbf);
                    vector<float> hh;
                    hh.reserve(2 * m_.ffn_chunk);
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> ha = swiglu(hh, m_.ffn_chunk);
                    h1 = gpu_.matvec(m_.w1b[l], xbf);
                    h3 = gpu_.matvec(m_.w3b[l], xbf);
                    hh.clear(); hh.reserve(2 * (m_.hidden - m_.ffn_chunk));
                    hh.insert(hh.end(), h1.begin(), h1.end());
                    hh.insert(hh.end(), h3.begin(), h3.end());
                    vector<float> hb = swiglu(hh, m_.hidden - m_.ffn_chunk);
                    o2 = add_q16(gpu_.matvec(m_.w2a[l], ha), gpu_.matvec(m_.w2b[l], hb));
                } else {
                    if (!m_.w1.empty()) {
                        vector<float> h1 = gpu_.matvec(m_.w1[l], xbf);
                        vector<float> h3 = gpu_.matvec(m_.w3[l], xbf);
                        h.reserve(2 * m_.hidden);
                        h.insert(h.end(), h1.begin(), h1.end());
                        h.insert(h.end(), h3.begin(), h3.end());
                    } else {
                        h = gpu_.matvec(m_.w13[l], xbf);
                    }
                    o2 = gpu_.matvec(m_.w2[l], swiglu(h));
                }
            } else {
                GLuint xtex = gpu_.upload_input(xbf);
                GLuint htex = 0;
                if (m_.ffn_chunked) {
                    GLuint ha, hb;
                    gpu_.ffn_split_chunked(m_.w1a[l], m_.w3a[l], m_.w1b[l], m_.w3b[l], xtex,
                                           m_.ffn_chunk, m_.hidden - m_.ffn_chunk, ha, hb);
                    o2 = add_q16(gpu_.matvec_from(m_.w2a[l], ha), gpu_.matvec_from(m_.w2b[l], hb));
                } else if (!m_.w1.empty()) htex = gpu_.ffn_split(m_.w1[l], m_.w3[l], xtex, m_.hidden);
                else htex = gpu_.ffn_fused(m_.w13[l], xtex, m_.hidden);
                if (!m_.ffn_chunked) o2 = gpu_.matvec_from(m_.w2[l], htex);
            }
            x = add_q16(x1, o2);
        }
        return x;
    }

    GLuint forward_gpu_hidden(int token, int pos, bool want_logits) {
        bool prof = getenv("LAYER_PROFILE") != nullptr;
        auto pt = Clock::now();
        auto mark = [&](const char* name, int l) {
            if (!prof) return;
            gpu_.sync();
            auto now = Clock::now();
            cerr << "PROF l=" << l << " " << name << " " << elapsed_ms(pt, now) << "\n";
            pt = now;
        };
        GLuint h = gpu_.upload_hidden(embedding(token));
        mark("embed", -1);
        for (int l = 0; l < m_.layers; ++l) {
            if (g_stop) return h;
            GLuint xb = gpu_.rmsnorm(h, gpu_.rms_att_tex_[l], gpu_.xb_);
            mark("rms_att", l);
            GLuint att_tex = gpu_.attention_block(m_.qkv[l], xb, l, pos);
            mark("attn", l);
            gpu_.matvec_add(m_.wo[l], att_tex, h, gpu_.hidden_b_);
            mark("wo", l);
            GLuint x1 = gpu_.hidden_b_.tex;
            GLuint xbf = gpu_.rmsnorm(x1, gpu_.rms_ffn_tex_[l], gpu_.xbf_);
            mark("rms_ffn", l);
            if (m_.ffn_chunked) {
                GLuint ha, hb;
                gpu_.ffn_split_chunked(m_.w1a[l], m_.w3a[l], m_.w1b[l], m_.w3b[l], xbf,
                                       m_.ffn_chunk, m_.hidden - m_.ffn_chunk, ha, hb);
                mark("ffn", l);
                gpu_.matvec_add(m_.w2a[l], ha, x1, gpu_.hidden_a_);
                gpu_.matvec_add(m_.w2b[l], hb, gpu_.hidden_a_.tex, gpu_.hidden_b_);
                mark("w2", l);
                h = gpu_.hidden_b_.tex;
            } else {
                GLuint hffn;
                if (!m_.w1.empty()) hffn = gpu_.ffn_split(m_.w1[l], m_.w3[l], xbf, m_.hidden);
                else hffn = gpu_.ffn_fused(m_.w13[l], xbf, m_.hidden);
                mark("ffn", l);
                gpu_.matvec_add(m_.w2[l], hffn, x1, gpu_.hidden_a_);
                mark("w2", l);
                h = gpu_.hidden_a_.tex;
            }
        }
        if (!want_logits) return h;
        GLuint xf = gpu_.rmsnorm(h, gpu_.rms_final_tex_, gpu_.xf_);
        mark("rms_final", -1);
        return xf;
    }

    vector<float> step(int token, int pos, bool want_logits) {
        if (gpu_layer_) {
            GLuint xf = forward_gpu_hidden(token, pos, want_logits);
            if (g_stop || !want_logits) return {};
            auto a = Clock::now();
            vector<float> out = gpu_.logits_from(m_.logits_tiles, xf);
            if (getenv("LAYER_PROFILE")) {
                gpu_.sync();
                cerr << "PROF logits " << elapsed_ms(a, Clock::now()) << "\n";
            }
            return out;
        }
        vector<float> x = forward(token, pos);
        if (g_stop || !want_logits) return {};
        vector<float> xb = rms_norm(x, m_.rms_final.data());
        return gpu_.logits(m_.logits_tiles, xb);
    }

    CandidateResult step_candidates(int token, int pos, int topk, float temp, float top_p) {
        if (gpu_layer_) {
            GLuint xf = forward_gpu_hidden(token, pos, true);
            if (g_stop) return {};
            return gpu_.logits_candidates_from(m_.logits_tiles, xf, topk, temp, top_p);
        }
        vector<float> x = forward(token, pos);
        if (g_stop) return {};
        vector<float> xb = rms_norm(x, m_.rms_final.data());
        return gpu_.logits_candidates(m_.logits_tiles, xb, topk, temp, top_p);
    }

private:
    Model& m_;
    GPU& gpu_;
    bool gpu_attn_ = false;
    bool gpu_layer_ = false;
    vector<vector<float>> kv_cache_k_, kv_cache_v_;
    vector<float> rope_cos_, rope_sin_;

    vector<float> rms_norm(const vector<float>& x, const float* w) {
        float ss = 0;
        for (float v : x) ss += v * v;
        ss /= x.size();
        float r = 1.0f / sqrtf(ss + 1e-5f);
        vector<float> o(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            o[i] = quantize_act_f(w[i] * (r * x[i]));
        return o;
    }

    vector<float> head_rms_norm(const vector<float>& v, int n_heads, const float* w) {
        vector<float> o(v.size());
        const int hd = m_.head_dim;
        for (int h = 0; h < n_heads; ++h) {
            const int base = h * hd;
            float ss = 0;
            for (int i = 0; i < hd; ++i) { float x = v[base + i]; ss += x * x; }
            float r = 1.0f / sqrtf(ss / hd + 1e-6f);
            for (int i = 0; i < hd; ++i) o[base + i] = w[i] * (r * v[base + i]);
        }
        return o;
    }

    vector<float> rope(const vector<float>& v, int n_heads, int pos) {
        vector<float> o = v;
        int half = m_.head_dim / 2;
        for (int h = 0; h < n_heads; ++h) {
            int base = h * m_.head_dim;
            if (m_.rope_half) {
                for (int i = 0; i < half; ++i) {
                    float c = rope_cos_[(size_t)pos * half + i];
                    float s = rope_sin_[(size_t)pos * half + i];
                    float a = o[base + i], b = o[base + i + half];
                    o[base + i] = a * c - b * s;
                    o[base + i + half] = a * s + b * c;
                }
            } else {
                for (int i = 0; i < half; ++i) {
                    float c = rope_cos_[(size_t)pos * half + i];
                    float s = rope_sin_[(size_t)pos * half + i];
                    float a = o[base + 2 * i], b = o[base + 2 * i + 1];
                    o[base + 2 * i] = a * c - b * s;
                    o[base + 2 * i + 1] = a * s + b * c;
                }
            }
        }
        return o;
    }

    vector<float> attention(const vector<float>& q, int layer, int pos) {
        int group = m_.heads / m_.kv_heads;
        vector<float> out(m_.dim);
        vector<float> scores(pos + 1);
        for (int h = 0; h < m_.heads; ++h) {
            int kvh = h / group;
            const float* kb = kv_cache_k_[layer].data() + (size_t)kvh * m_.head_dim;
            const float* vb = kv_cache_v_[layer].data() + (size_t)kvh * m_.head_dim;
            float mx = -1e30f;
            for (int t = 0; t <= pos; ++t) {
                float s = 0;
                for (int i = 0; i < m_.head_dim; ++i)
                    s += q[h * m_.head_dim + i] * kb[(size_t)t * m_.kv_dim + i];
                s /= sqrtf((float)m_.head_dim);
                scores[t] = s;
                mx = max(mx, s);
            }
            float sum = 0;
            for (int t = 0; t <= pos; ++t) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
            for (int t = 0; t <= pos; ++t) scores[t] /= sum;
            for (int i = 0; i < m_.head_dim; ++i) {
                float s = 0;
                for (int t = 0; t <= pos; ++t)
                    s += scores[t] * vb[(size_t)t * m_.kv_dim + i];
                out[h * m_.head_dim + i] = s;
            }
        }
        return out;
    }

    static vector<float> add_q16(const vector<float>& a, const vector<float>& b) {
        vector<float> o(a.size());
        for (size_t i = 0; i < a.size(); ++i) o[i] = quantize_act_f(a[i] + b[i]);
        return o;
    }

    vector<float> swiglu(const vector<float>& h, int n) {
        vector<float> o(n);
        for (int i = 0; i < n; ++i) {
            float a = h[i], b = h[n + i];
            o[i] = quantize_act_f((a / (1.0f + expf(-a))) * b);
        }
        return o;
    }

    vector<float> swiglu(const vector<float>& h) {
        return swiglu(h, m_.hidden);
    }
};

// -----------------------------------------------------------------------------
// Sampling and CLI.
// -----------------------------------------------------------------------------
static int sample_token(const vector<float>& logits, float temp, float top_p, mt19937& rng) {
    vector<pair<float, int>> p(logits.size());
    float mx = -1e30f;
    for (size_t i = 0; i < logits.size(); ++i) mx = max(mx, logits[i] / temp);
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float e = expf(logits[i] / temp - mx);
        p[i] = {e, (int)i}; sum += e;
    }
    for (auto& x : p) x.first /= sum;
    sort(p.begin(), p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    float c = 0; int cut = 0;
    for (; cut < (int)p.size(); ++cut) { c += p[cut].first; if (c >= top_p) { ++cut; break; } }
    float r = uniform_real_distribution<float>(0, 1)(rng) * c;
    float acc = 0;
    for (int i = 0; i < cut; ++i) { acc += p[i].first; if (r <= acc) return p[i].second; }
    return p[cut - 1].second;
}

static int argmax_token(const vector<float>& logits) {
    return (int)(max_element(logits.begin(), logits.end()) - logits.begin());
}

static int sample_candidate_result(const CandidateResult& r, float temp, float top_p, mt19937& rng) {
    if (r.cand.empty()) return 2;
    vector<Candidate> cand = r.cand;
    if (r.strict) {
        for (auto& c : cand) c.value = expf(c.value / temp - r.logsumexp);
    } else {
        float mx = -1e30f;
        for (auto& c : cand) mx = max(mx, c.value / temp);
        float sum = 0;
        for (auto& c : cand) { c.value = expf(c.value / temp - mx); sum += c.value; }
        for (auto& c : cand) c.value /= sum;
    }
    sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
    float cdf = 0; int cut = 0;
    for (; cut < (int)cand.size(); ++cut) { cdf += cand[cut].value; if (cdf >= top_p) { ++cut; break; } }
    float x = uniform_real_distribution<float>(0, 1)(rng) * cdf;
    float acc = 0;
    for (int i = 0; i < cut; ++i) { acc += cand[i].value; if (x <= acc) return cand[i].token; }
    return cand[cut - 1].token;
}

// -----------------------------------------------------------------------------
// Tool calls: minimal JSON extraction, built-in handlers, and a safe arithmetic
// parser.  The model only emits <tool_call>{...}</tool_call>; execution and the
// <tool_response> round-trip are done here.
// -----------------------------------------------------------------------------
static string json_escape(const string& s) {
    string o;
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8]; snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c); o += b;
                } else o += c;
                break;
        }
    }
    return o;
}

static void json_skip_ws(const string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p;
}

static bool json_key_pos(const string& obj, const string& key, size_t& p) {
    string pat = "\"" + key + "\"";
    size_t k = obj.find(pat);
    if (k == string::npos) return false;
    p = k + pat.size();
    json_skip_ws(obj, p);
    if (p >= obj.size() || obj[p] != ':') return false;
    ++p;
    json_skip_ws(obj, p);
    return true;
}

static bool json_get_string(const string& obj, const string& key, string& out) {
    size_t p = 0;
    if (!json_key_pos(obj, key, p)) return false;
    if (p >= obj.size() || obj[p] != '"') return false;
    ++p;
    out.clear();
    while (p < obj.size()) {
        char c = obj[p++];
        if (c == '"') return true;
        if (c != '\\') { out += c; continue; }
        if (p >= obj.size()) return false;
        char e = obj[p++];
        switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                if (p + 4 > obj.size()) return false;
                unsigned v = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = obj[p++]; v <<= 4;
                    if (h >= '0' && h <= '9') v |= h - '0';
                    else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
                    else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
                    else return false;
                }
                if (v < 0x80) out += (char)v;
                else if (v < 0x800) {
                    out += (char)(0xC0 | (v >> 6));
                    out += (char)(0x80 | (v & 0x3F));
                } else {
                    out += (char)(0xE0 | (v >> 12));
                    out += (char)(0x80 | ((v >> 6) & 0x3F));
                    out += (char)(0x80 | (v & 0x3F));
                }
                break;
            }
            default: return false;
        }
    }
    return false;
}

static bool json_get_object(const string& obj, const string& key, string& out) {
    size_t p = 0;
    if (!json_key_pos(obj, key, p) || p >= obj.size() || obj[p] != '{') return false;
    size_t start = p;
    int depth = 0; bool in_str = false, esc = false;
    for (; p < obj.size(); ++p) {
        char c = obj[p];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) { out = obj.substr(start, p - start + 1); return true; }
        }
    }
    return false;
}

struct ToolCall { string name, args; };

static vector<ToolCall> parse_tool_calls(const string& text) {
    vector<ToolCall> out;
    size_t pos = 0;
    const string open = "<tool_call>", close = "</tool_call>";
    while ((pos = text.find(open, pos)) != string::npos) {
        size_t a = pos + open.size();
        size_t b = text.find(close, a);
        if (b == string::npos) break;
        string body = text.substr(a, b - a);
        ToolCall c;
        if (json_get_string(body, "name", c.name)) {
            if (!json_get_object(body, "arguments", c.args)) c.args = "{}";
            out.push_back(std::move(c));
        }
        pos = b + close.size();
    }
    return out;
}

static bool read_text_file(const string& path, string& out) {
    ifstream f(path, ios::binary);
    if (!f) return false;
    stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static string base64_decode(const string& in) {
    static int tbl[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) tbl[i] = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tbl[(unsigned char)a[i]] = i;
        init = true;
    }
    string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (tbl[c] < 0) continue;
        val = (val << 6) + tbl[c];
        bits += 6;
        if (bits >= 0) {
            out += (char)((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return out;
}

static size_t utf8_complete_prefix(const string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int need = 0;
        if (c < 0x80) need = 0;
        else if ((c & 0xE0) == 0xC0) need = 1;
        else if ((c & 0xF0) == 0xE0) need = 2;
        else if ((c & 0xF8) == 0xF0) need = 3;
        else { ++i; continue; }
        if (i + (size_t)need >= s.size()) break;
        bool ok = true;
        for (int j = 1; j <= need; ++j)
            if (((unsigned char)s[i + j] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { ++i; continue; }
        i += (size_t)need + 1;
    }
    return i;
}

static string utf8_take(string& pending) {
    size_t n = utf8_complete_prefix(pending);
    if (n == 0) return {};
    string out = pending.substr(0, n);
    pending.erase(0, n);
    return out;
}

struct ExprParser {
    const char* p;
    bool ok = true;
    explicit ExprParser(const string& s) : p(s.c_str()) {}
    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }
    double expr() {
        double v = term();
        for (;;) {
            ws();
            if (*p == '+') { ++p; v += term(); }
            else if (*p == '-') { ++p; v -= term(); }
            else return v;
        }
    }
    double term() {
        double v = power();
        for (;;) {
            ws();
            if (*p == '*') {
                if (p[1] == '*') { ++p; ++p; v = pow(v, power()); }
                else { ++p; v *= power(); }
            } else if (*p == '/') { ++p; v /= power(); }
            else if (*p == '%') { ++p; v = fmod(v, power()); }
            else return v;
        }
    }
    double power() {
        double v = factor();
        ws();
        if (*p == '^') { ++p; v = pow(v, power()); }
        return v;
    }
    double factor() {
        ws();
        if (*p == '+') { ++p; return factor(); }
        if (*p == '-') { ++p; return -factor(); }
        if (*p == '(') {
            ++p; double v = expr(); ws();
            if (*p != ')') { ok = false; return 0; }
            ++p; return v;
        }
        char* end = nullptr;
        double v = strtod(p, &end);
        if (end == p) { ok = false; return 0; }
        p = end;
        return v;
    }
};

static string run_tool(const string& name, const string& args) {
    if (name == "get_current_time") {
        string tz;
        json_get_string(args, "timezone", tz);
        string old_tz;
        const char* old = getenv("TZ");
        if (old) old_tz = old;
        if (!tz.empty()) { setenv("TZ", tz.c_str(), 1); tzset(); }
        time_t now = time(nullptr);
        struct tm tmv {};
        localtime_r(&now, &tmv);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        if (!tz.empty()) {
            if (old) setenv("TZ", old_tz.c_str(), 1);
            else unsetenv("TZ");
            tzset();
        }
        return "{\"datetime\":\"" + json_escape(buf) + "\",\"timezone\":\"" +
               json_escape(tz.empty() ? string("local") : tz) + "\"}";
    }
    if (name == "calculate_math") {
        string expr;
        if (!json_get_string(args, "expression", expr)) return "{\"error\":\"missing expression\"}";
        ExprParser ep(expr);
        double v = ep.expr();
        ep.ws();
        if (!ep.ok || *ep.p != '\0') return "{\"error\":\"invalid expression\"}";
        char buf[64];
        if (fabs(v - round(v)) < 1e-9) snprintf(buf, sizeof(buf), "%.0f", v);
        else snprintf(buf, sizeof(buf), "%.6g", v);
        return "{\"result\":\"" + string(buf) + "\"}";
    }
    if (name == "system_status" || name.find("status") != string::npos) {
        double mem_free = 0, mem_avail = 0;
        {
            ifstream in("/proc/meminfo");
            string key;
            while (in >> key) {
                long value = 0;
                string unit;
                if (!(in >> value)) break;
                getline(in, unit);
                if (key == "MemFree:") mem_free = value / 1024.0;
                else if (key == "MemAvailable:") mem_avail = value / 1024.0;
            }
        }
        double temp_c = 0;
        {
            ifstream in("/sys/class/thermal/thermal_zone0/temp");
            long milli = 0;
            if (in >> milli) temp_c = milli / 1000.0;
        }
        double uptime_s = 0, load1 = 0;
        {
            ifstream in("/proc/uptime");
            in >> uptime_s;
        }
        {
            ifstream in("/proc/loadavg");
            in >> load1;
        }
        double disk_free_mb = 0;
        struct statvfs st {};
        if (statvfs("/", &st) == 0) disk_free_mb = (double)st.f_bavail * st.f_frsize / 1048576.0;
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "{\"temp_c\":%.1f,\"mem_free_mb\":%.1f,\"mem_avail_mb\":%.1f,"
                 "\"disk_free_mb\":%.1f,\"uptime_s\":%.0f,\"load1\":%.2f}",
                 temp_c, mem_free, mem_avail, disk_free_mb, uptime_s, load1);
        return string(buf);
    }
    return "{\"error\":\"unknown tool\"}";
}

int main(int argc, char** argv) {
    install_stop_handlers();
    string model_path = getenv("MODEL") ? getenv("MODEL") : "../models/llama2c/stories15M.bin";
    string tokenizer_path = getenv("TOKENIZER") ? getenv("TOKENIZER") : "../models/llama2c/tokenizer.bin";
    string prompt = "One day, Lily";
    string system_prompt;
    int gen = 8;
    bool prompt_set = false, gen_set = false;
    bool chat = false, qwen_template = false, minimind3_template = false, greedy = false, worker = false;
    bool open_thinking = false;
    string tools_json;
    string build_cache_prompt, build_cache_out, prefix_cache_path;
    uint32_t seed = 1234;
    float temp = 1.0f, top_p = 0.9f;
    int topk = getenv("TOPK") ? atoi(getenv("TOPK")) : 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) { prompt = argv[++i]; prompt_set = true; }
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); gen_set = true; }
        else if (!strcmp(argv[i], "--chat")) chat = true;
        else if (!strcmp(argv[i], "--worker")) worker = true;
        else if (!strcmp(argv[i], "--build-prefix-cache") && i + 2 < argc) {
            build_cache_prompt = argv[++i]; build_cache_out = argv[++i];
        }
        else if (!strcmp(argv[i], "--prefix-cache") && i + 1 < argc) prefix_cache_path = argv[++i];
        else if (!strcmp(argv[i], "--greedy")) greedy = true;
        else if (!strcmp(argv[i], "--thinking") || !strcmp(argv[i], "--open-thinking")) open_thinking = true;
        else if (!strcmp(argv[i], "--tools") && i + 1 < argc) tools_json = argv[++i];
        else if (!strcmp(argv[i], "--tools-file") && i + 1 < argc) {
            ifstream tf(argv[++i]);
            if (!tf) { cerr << "cannot open tools file\n"; return 1; }
            stringstream ss; ss << tf.rdbuf(); tools_json = ss.str();
        }
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) system_prompt = argv[++i];
        else if (!strcmp(argv[i], "--template") && i + 1 < argc) {
            const char* t = argv[++i];
            qwen_template = !strcmp(t, "qwen");
            minimind3_template = !strcmp(t, "minimind3");
            if (minimind3_template) qwen_template = true;
        }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--topp") && i + 1 < argc) top_p = atof(argv[++i]);
    }
    if (chat && !gen_set) gen = 128;

    Model model;
    if (!model.load(model_path)) return 1;
    GPU gpu;
    if (getenv("Q8_PARTS")) model.apply_precision(getenv("Q8_LAYERS") ? getenv("Q8_LAYERS") : "", getenv("Q8_PARTS"));
    model.split_ffn();
    if (getenv("Q4_WEIGHTS")) model.quantize_q4_weights(getenv("Q4_SCOPE") ? getenv("Q4_SCOPE") : "all");
    if (!tools_json.empty() && model.seq < 768) model.seq = 768;
    if ((worker || !build_cache_prompt.empty()) && model.seq < 1024 && !getenv("SEQ")) model.seq = 1024;
    Tokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path, model.vocab)) return 1;
    tokenizer.set_qwen(qwen_template || minimind3_template);
    tokenizer.set_bytelevel(minimind3_template);

    for (auto& m : model.qkv) gpu.upload(m);
    for (auto& m : model.wo) gpu.upload(m);
    for (auto& m : model.w13) gpu.upload(m);
    for (auto& m : model.w1) gpu.upload(m);
    for (auto& m : model.w3) gpu.upload(m);
    for (auto& m : model.w2) gpu.upload(m);
    for (auto& m : model.w1a) gpu.upload(m);
    for (auto& m : model.w1b) gpu.upload(m);
    for (auto& m : model.w3a) gpu.upload(m);
    for (auto& m : model.w3b) gpu.upload(m);
    for (auto& m : model.w2a) gpu.upload(m);
    for (auto& m : model.w2b) gpu.upload(m);
    for (auto& m : model.logits_tiles) gpu.upload(m);

    Session session(model, gpu);

    if (!build_cache_prompt.empty()) {
        string text;
        if (!read_text_file(build_cache_prompt, text)) { cerr << "cannot read cache prompt\n"; return 1; }
        auto tokens = tokenizer.encode(text, false, false);
        if (tokens.empty() || (int)tokens.size() > model.seq) {
            cerr << "cache prompt empty or too long (" << tokens.size() << ")\n"; return 1;
        }
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (g_stop) return 1;
            session.step(tokens[i], (int)i, false);
        }
        if (!session.save_prefix_cache(build_cache_out, tokens)) {
            cerr << "cannot save prefix cache (GPU attention required)\n"; return 1;
        }
        cout << "CACHE_SAVED " << build_cache_out << " tokens=" << tokens.size() << "\n";
        return 0;
    }

    if (worker) {
        auto emit = [&](const string& s) { cout << s << "\n" << flush; };
        vector<int> cache_tokens;
        if (!prefix_cache_path.empty()) {
            if (session.load_prefix_cache(prefix_cache_path, cache_tokens))
                cerr << "PREFIX_CACHE loaded " << cache_tokens.size() << " tokens\n";
            else
                cerr << "PREFIX_CACHE ignored " << prefix_cache_path << "\n";
        }
        emit("{\"event\":\"ready\",\"prefix_cache_tokens\":" + to_string(cache_tokens.size()) + "}");
        string line;
        while (getline(cin, line)) {
            if (g_stop) break;
            if (line.empty()) continue;
            vector<string> f;
            size_t start = 0;
            while (start <= line.size()) {
                size_t tab = line.find('\t', start);
                f.push_back(line.substr(start, tab == string::npos ? string::npos : tab - start));
                if (tab == string::npos) break;
                start = tab + 1;
            }
            if (f.size() < 5) { emit("{\"event\":\"error\",\"message\":\"bad request line\"}"); continue; }
            string prompt = base64_decode(f[0]);
            int max_new = max(0, atoi(f[1].c_str()));
            float wt = atof(f[2].c_str());
            float wp = atof(f[3].c_str());
            uint32_t seed = (uint32_t)strtoul(f[4].c_str(), nullptr, 10);
            if (wp <= 0.0f || wp > 1.0f) wp = 0.9f;
            auto tokens = tokenizer.encode(prompt, false, false);
            if (tokens.empty()) {
                emit("{\"event\":\"error\",\"message\":\"empty prompt\"}");
                continue;
            }
            if ((int)tokens.size() >= model.seq) {
                emit("{\"event\":\"error\",\"message\":\"context full\"}");
                continue;
            }
            int common = 0;
            while (common < (int)cache_tokens.size() && common < (int)tokens.size() &&
                   cache_tokens[common] == tokens[common]) ++common;
            int prefill_from = common;
            if (prefill_from >= (int)tokens.size()) prefill_from = (int)tokens.size() - 1;
            int prefill_tokens = (int)tokens.size() - prefill_from;
            emit("{\"event\":\"start\",\"prompt_tokens\":" + to_string(tokens.size()) +
                 ",\"cached_tokens\":" + to_string(common) +
                 ",\"prefill_tokens\":" + to_string(prefill_tokens) + "}");
            vector<float> logits;
            for (int i = prefill_from; i < (int)tokens.size(); ++i) {
                if (g_stop) break;
                logits = session.step(tokens[i], i, i + 1 == (int)tokens.size());
            }
            if (g_stop) break;
            int pos = (int)tokens.size();
            max_new = min(max_new, model.seq - pos);
            bool greedy_req = wt <= 0.0f;
            mt19937 rng(seed);
            string pending, full, finish = "length";
            int generated = 0;
            int prev = tokens.back();
            vector<int> all_tokens = tokens;
            for (int i = 0; i < max_new; ++i) {
                if (g_stop) { finish = "stop"; break; }
                int next = greedy_req ? argmax_token(logits)
                                      : sample_token(logits, wt, wp, rng);
                if (next == 1 || next == 2) {
                    session.step(next, pos++, false);
                    all_tokens.push_back(next);
                    finish = "stop";
                    break;
                }
                string piece = tokenizer.decode({next}, prev);
                pending += piece;
                string text = utf8_take(pending);
                if (!text.empty()) {
                    full += text;
                    emit("{\"event\":\"token\",\"text\":\"" + json_escape(text) + "\"}");
                }
                prev = next;
                ++generated;
                all_tokens.push_back(next);
                logits = session.step(next, pos++, true);
            }
            if (!pending.empty()) {
                string text = "\xEF\xBF\xBD";
                full += text;
                emit("{\"event\":\"token\",\"text\":\"" + json_escape(text) + "\"}");
            }
            if (g_stop) finish = "stop";
            if (g_stop) cache_tokens.clear();
            else cache_tokens = all_tokens;
            emit("{\"event\":\"done\",\"text\":\"" + json_escape(full) +
                 "\",\"finish_reason\":\"" + finish +
                 "\",\"prompt_tokens\":" + to_string(tokens.size()) +
                 ",\"completion_tokens\":" + to_string(generated) +
                 ",\"cached_tokens\":" + to_string(common) +
                 ",\"prefill_tokens\":" + to_string(prefill_tokens) + "}");
        }
        return 0;
    }


    vector<int> cli_cache_tokens;
    if (!worker && !prefix_cache_path.empty()) {
        if (session.load_prefix_cache(prefix_cache_path, cli_cache_tokens))
            cerr << "PREFIX_CACHE loaded " << cli_cache_tokens.size() << " tokens\n";
        else
            cerr << "PREFIX_CACHE ignored " << prefix_cache_path << "\n";
    }
    auto cache_prefix_len = [&](const vector<int>& tokens) {
        size_t n = 0;
        while (n < cli_cache_tokens.size() && n < tokens.size() &&
               cli_cache_tokens[n] == tokens[n]) ++n;
        return (int)n;
    };

    mt19937 rng(seed);
    auto run_turn = [&](const string& user, bool first, int pos) {
        auto turn_start = Clock::now();
        string text;
        if (minimind3_template) {
            string sys = system_prompt.empty()
                ? string("你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。")
                : system_prompt;
            if (!tools_json.empty()) {
                sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                       "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                       tools_json +
                       "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                       "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
            }
            if (first)
                text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
            text += "<|im_start|>user\n" + user + "<|im_end|>\n"
                    "<|im_start|>assistant\n";
            text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        } else if (qwen_template) {
            if (first)
                text = "<|im_start|>system\n" + (system_prompt.empty() ? string("You are a helpful assistant") : system_prompt) +
                       "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            else
                text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            if (open_thinking) text += "<think>\n";
        } else {
            if (first && !system_prompt.empty())
                text = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user + " [/INST]";
            else
                text = "[INST] " + user + " [/INST]";
        }
        auto tokens = tokenizer.encode(text, !qwen_template, !qwen_template);
        int skip = (first && !cli_cache_tokens.empty()) ? cache_prefix_len(tokens) : 0;
        for (int i = 0; i < skip; ++i) pos = i + 1;
        vector<float> logits;
        CandidateResult cand;
        bool cand_mode = topk != 0 && !greedy;
        for (size_t i = skip; i < tokens.size(); ++i) {
            if (g_stop) { gpu.sync(); return; }
            if (pos >= model.seq) { cout << "[context full]\n"; return; }
            bool last = (i + 1 == tokens.size());
            if (last && cand_mode) {
                cand = session.step_candidates(tokens[i], pos++, topk, temp, top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else {
                logits = session.step(tokens[i], pos++, last);
            }
        }
        if (g_stop) { gpu.sync(); return; }
        auto dec_start = Clock::now();
        int generated = 0;
        if (!cand_mode && getenv("LOGITS_OUT")) {
            ofstream f(getenv("LOGITS_OUT"), ios::binary);
            f.write((char*)logits.data(), logits.size() * 4);
        }
        cout << "Assistant: "; cout.flush();
        int prev = tokens.empty() ? 1 : tokens.back();
        for (int i = 0; i < gen && pos < model.seq; ++i) {
            if (g_stop) { gpu.sync(); return; }
            int next = greedy ? argmax_token(logits)
                              : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                           : sample_token(logits, temp, top_p, rng));
            if (!getenv("IGNORE_EOS") && (next == 1 || next == 2)) {
                session.step(next, pos++, false);
                break;
            }
            cout << tokenizer.decode({next}, prev); cout.flush();
            tokens.push_back(next);
            ++generated;
            prev = next;
            if (cand_mode) {
                cand = session.step_candidates(next, pos++, topk, temp, top_p);
                if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
            } else {
                logits = session.step(next, pos++, true);
            }
        }
        cout << "\n";
        if (getenv("TIMING")) {
            double ms = elapsed_ms(dec_start, Clock::now());
            cerr << "TIMING prefill=" << elapsed_ms(turn_start, dec_start) / 1000.0
                 << "s decode_avg=" << (generated ? ms / generated : 0.0) << "ms/token gen=" << generated << "\n";
        }
        if (getenv("TOKENS_OUT")) {
            ofstream f(getenv("TOKENS_OUT"), ios::binary);
            for (int t : tokens) f.write((char*)&t, 4);
        }
        if (g_stop) gpu.sync();
    };

    auto run_tool_turn = [&](const string& user, bool first) {
        string sys = system_prompt.empty()
            ? string("你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。")
            : system_prompt;
        if (!tools_json.empty()) {
            sys += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
                   tools_json +
                   "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                   "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        }
        string text;
        if (first || !tools_json.empty())
            text = "<|im_start|>system\n" + sys + "<|im_end|>\n";
        text += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";

        for (int round = 0; round < 3; ++round) {
            auto tokens = tokenizer.encode(text, false, false);
            if ((int)tokens.size() >= model.seq) { cout << "[context full]\n"; return; }
            vector<float> logits;
            CandidateResult cand;
            bool cand_mode = topk != 0 && !greedy;
            int skip = (first && round == 0 && !cli_cache_tokens.empty()) ? cache_prefix_len(tokens) : 0;
            for (size_t i = skip; i < tokens.size(); ++i) {
                if (g_stop) { gpu.sync(); return; }
                bool last = (i + 1 == tokens.size());
                if (last && cand_mode) {
                    cand = session.step_candidates(tokens[i], (int)i, topk, temp, top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(tokens[i], (int)i, last);
                }
            }
            if (g_stop) { gpu.sync(); return; }
            cout << "Assistant: "; cout.flush();
            string out;
            int prev = tokens.empty() ? 1 : tokens.back();
            int pos = (int)tokens.size();
            for (int i = 0; i < gen && pos < model.seq; ++i) {
                if (g_stop) { gpu.sync(); return; }
                int next = greedy ? argmax_token(logits)
                                  : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                               : sample_token(logits, temp, top_p, rng));
                if (next == 1 || next == 2) {
                    session.step(next, pos++, false);
                    break;
                }
                string piece = tokenizer.decode({next}, prev);
                cout << piece; cout.flush();
                out += piece;
                prev = next;
                if (cand_mode) {
                    cand = session.step_candidates(next, pos++, topk, temp, top_p);
                    if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
                } else {
                    logits = session.step(next, pos++, true);
                }
            }
            cout << "\n";
            vector<ToolCall> calls = parse_tool_calls(out);
            if (calls.empty()) return;
            text += out;
            text += "<|im_end|>\n<|im_start|>user\n";
            for (const auto& c : calls) {
                string result = run_tool(c.name, c.args);
                cout << "[tool] " << c.name << " " << c.args << "\n";
                cout << "[tool result] " << result << "\n";
                text += "<tool_response>\n" + result + "\n</tool_response>\n";
            }
            text += "<|im_end|>\n<|im_start|>assistant\n";
            text += open_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        }
    };

    if (chat) {
        if (!prompt_set) prompt.clear();
        if (!prompt.empty()) {
            if (tools_json.empty()) run_turn(prompt, true, 0);
            else run_tool_turn(prompt, true);
        } else {
            string line; bool first = true;
            while (!g_stop) {
                cout << "User: "; cout.flush();
                struct pollfd pfd { STDIN_FILENO, POLLIN, 0 };
                int pr;
                do { pr = poll(&pfd, 1, 100); }
                while ((pr == 0 || (pr < 0 && errno == EINTR)) && !g_stop);
                if (g_stop || pr < 0) break;
                if (!getline(cin, line)) break;
                if (line.empty()) continue;
                if (tools_json.empty()) run_turn(line, first, 0);
                else run_tool_turn(line, first);
                first = false;
                if (g_stop) {
                    cout << "\n[interrupted]\n";
                    g_stop = 0;
                }
            }
            if (g_stop) cout << "\n";
        }
        return 0;
    }

    auto tokens = tokenizer.encode(prompt, true, true);
    cout << "tokens:";
    for (int t : tokens) cout << " " << t;
    cout << "\n";
    auto start = Clock::now();
    vector<float> logits;
    CandidateResult cand;
    bool cand_mode = topk != 0 && !greedy;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (g_stop) break;
        bool last = (i + 1 == tokens.size());
        if (last && cand_mode) {
            cand = session.step_candidates(tokens[i], (int)i, topk, temp, top_p);
            if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
        } else {
            logits = session.step(tokens[i], (int)i, last);
        }
    }
    if (g_stop) { gpu.sync(); cout << "\n[interrupted]\n"; return 0; }
    double prefill = chrono::duration<double>(Clock::now() - start).count();
    cout << "prefill " << prefill << " s\n";
    if (!cand_mode && getenv("LOGITS_OUT")) {
        ofstream f(getenv("LOGITS_OUT"), ios::binary);
        f.write((char*)logits.data(), logits.size() * 4);
    }
    int generated = 0;
    double decode_ms = 0;
    for (int i = 0; i < gen; ++i) {
        if (g_stop) break;
        int next = greedy ? argmax_token(logits)
                          : (cand_mode ? sample_candidate_result(cand, temp, top_p, rng)
                                       : sample_token(logits, temp, top_p, rng));
        if (!getenv("IGNORE_EOS") && (next == 1 || next == 2)) break;
        cout << tokenizer.decode({next}, tokens.empty() ? 1 : tokens.back()); cout.flush();
        tokens.push_back(next);
        auto a = Clock::now();
        if (cand_mode) {
            cand = session.step_candidates(next, (int)tokens.size() - 1, topk, temp, top_p);
            if (!cand.complete) { logits = std::move(cand.full); cand_mode = false; }
        } else {
            logits = session.step(next, (int)tokens.size() - 1, true);
        }
        decode_ms += elapsed_ms(a, Clock::now());
        ++generated;
    }
    cout << "\n";
    if (getenv("TOKENS_OUT")) {
        ofstream f(getenv("TOKENS_OUT"), ios::binary);
        for (int t : tokens) f.write((char*)&t, 4);
    }
    if (g_stop) { gpu.sync(); cout << "[interrupted]\n"; return 0; }
    if (generated) cout << "decode avg " << decode_ms / generated << " ms/token\n";
    return 0;
}
