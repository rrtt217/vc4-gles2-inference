#include "gpu.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cstdlib>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <string>
#include <unordered_map>
#include <vector>


GPU::GPU(){ init_egl(); init_gl(); }

GPU::~GPU(){
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

void GPU::upload(Mat& m){
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

vector<float> GPU::matvec(const Mat& m, const vector<float>& x){
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

GLuint GPU::upload_input(const vector<float>& x){ return upload_activations(x); }

void GPU::matvec_to(const Mat& m, GLuint x, Target& out){
        matvec_to_row(m, x, out, 0);
    }

void GPU::matvec_to_row(const Mat& m, GLuint x, Target& out, int row){
        if (out.tex == 0 || out.w != m.rows || out.h <= row) out = make_target(m.rows, row + 1);
        draw(m, x, out, row);
        layer_finish("NF_MAT");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
    }

vector<float> GPU::matvec_from(const Mat& m, GLuint x){
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

GLuint GPU::ffn_split(const Mat& w1, const Mat& w3, GLuint x, int hidden){
        // The old non-paired q8 path (q24 matvec_to_row + swiglu_2row) has been
        // observed to miscompute w1/w3 for hidden=1408; the paired q16 path is
        // correct and also faster.  FFN_PAIR=0 keeps the old path for A/B.
        static const bool pair_env = !(getenv("FFN_PAIR") && strcmp(getenv("FFN_PAIR"), "0") == 0);
        if (pair_env && w1.type == WeightType::INT8 && w3.type == WeightType::INT8)
            return ffn_pair_chunk(w1, w3, x, hidden, ffn_pair_a_, ffn_h_a_);
        if (ffn_ab_.tex == 0 || ffn_ab_.w != hidden || ffn_ab_.h != 2) ffn_ab_ = make_target(hidden, 2);
        matvec_to_row(w1, x, ffn_ab_, 0);
        matvec_to_row(w3, x, ffn_ab_, 1);
        return swiglu_2row(ffn_ab_, hidden);
    }

GLuint GPU::ffn_fused(const Mat& w13, GLuint x, int hidden){
        // q8 W13 uses the paired q16 path: the old q24 path miscomputes
        // fused-W13 FFN.  q4/i16 keep the q24 path.
        if (w13.type == WeightType::INT8) {
            const int aw = x_texels(hidden, ffn_fmt_);
            if (ffn_pair_a_.tex == 0 || ffn_pair_a_.w != hidden) ffn_pair_a_ = make_target(hidden, 1);
            if (ffn_h_a_.tex == 0 || ffn_h_a_.w != aw) ffn_h_a_ = make_target(aw, 1);
            GLuint p = x_q8_pair_fused(hidden, w13.cols);
            glBindFramebuffer(GL_FRAMEBUFFER, ffn_pair_a_.fbo);
            glViewport(0, 0, hidden, 1);
            glUseProgram(p);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, w13.tex);
            glUniform1i(glGetUniformLocation(p, "W"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, x);
            glUniform1i(glGetUniformLocation(p, "X"), 1);
            glUniform1f(glGetUniformLocation(p, "ws"), w13.scale);
            GLuint loc = glGetAttribLocation(p, "p");
            glEnableVertexAttribArray(loc);
            glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLuint ps = x_swiglu_pair(hidden);
            glBindFramebuffer(GL_FRAMEBUFFER, ffn_h_a_.fbo);
            glViewport(0, 0, aw, 1);
            glUseProgram(ps);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, ffn_pair_a_.tex);
            glUniform1i(glGetUniformLocation(ps, "A"), 0);
            loc = glGetAttribLocation(ps, "p");
            glEnableVertexAttribArray(loc);
            glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
            else layer_finish("NF_FFN");
            if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
            return ffn_h_a_.tex;
        }
        matvec_to(w13, x, ffn_a_);
        return swiglu_fused(ffn_a_, hidden);
    }

// Paired gate matvec + swiglu for one FFN chunk.
    GLuint GPU::ffn_pair_chunk(const Mat& w1, const Mat& w3, GLuint x, int hidden, Target& pair_buf, Target& out){
        if (pair_buf.tex == 0 || pair_buf.w != hidden) pair_buf = make_target(hidden, 1);
        const int aw = x_texels(hidden, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        const bool i4 = (w1.type == WeightType::INT4);
        GLuint pp = i4 ? x_i4_pair(hidden, w1.cols) : x_q8_pair(hidden, w1.cols);
        glBindFramebuffer(GL_FRAMEBUFFER, pair_buf.fbo);
        glViewport(0, 0, hidden, 1);
        glUseProgram(pp);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, w1.tex);
        glUniform1i(glGetUniformLocation(pp, "W1"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, w3.tex);
        glUniform1i(glGetUniformLocation(pp, "W2"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(pp, "X"), 2);
        if (i4) {
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, w1.q4_param_tex);
            glUniform1i(glGetUniformLocation(pp, "P1"), 3);
            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, w3.q4_param_tex);
            glUniform1i(glGetUniformLocation(pp, "P2"), 4);
            glUniform1f(glGetUniformLocation(pp, "ps1"), w1.q4_param_scale);
            glUniform1f(glGetUniformLocation(pp, "ps2"), w3.q4_param_scale);
        } else {
            glUniform1f(glGetUniformLocation(pp, "ws1"), w1.scale);
            glUniform1f(glGetUniformLocation(pp, "ws2"), w3.scale);
        }
        GLuint loc = glGetAttribLocation(pp, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();

        GLuint ps = x_swiglu_pair(hidden);
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, aw, 1);
        glUseProgram(ps);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, pair_buf.tex);
        glUniform1i(glGetUniformLocation(ps, "A"), 0);
        loc = glGetAttribLocation(ps, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return out.tex;
    }

void GPU::ffn_split_chunked(const Mat& w1a, const Mat& w3a, const Mat& w1b, const Mat& w3b,
                           GLuint x, int n1, int n2, GLuint& ha, GLuint& hb){
        static const bool pair_env = !(getenv("FFN_PAIR") && strcmp(getenv("FFN_PAIR"), "0") == 0);
        // The paired shader assumes int8 weights; q4/i16 keep the old path.
        const bool pair = pair_env &&
                          ((w1a.type == WeightType::INT8 && w3a.type == WeightType::INT8) ||
                           (w1a.type == WeightType::INT4 && w3a.type == WeightType::INT4));
        if (pair) {
            ha = ffn_pair_chunk(w1a, w3a, x, n1, ffn_pair_a_, ffn_h_a_);
            hb = ffn_pair_chunk(w1b, w3b, x, n2, ffn_pair_b_, ffn_h_b_);
            return;
        }
        if (ffn_ab_a_.tex == 0 || ffn_ab_a_.w != n1) ffn_ab_a_ = make_target(n1, 2);
        if (ffn_ab_b_.tex == 0 || ffn_ab_b_.w != n2) ffn_ab_b_ = make_target(n2, 2);
        matvec_to_row(w1a, x, ffn_ab_a_, 0);
        matvec_to_row(w3a, x, ffn_ab_a_, 1);
        ha = swiglu_2row(ffn_ab_a_, n1, ffn_h_a_);
        matvec_to_row(w1b, x, ffn_ab_b_, 0);
        matvec_to_row(w3b, x, ffn_ab_b_, 1);
        hb = swiglu_2row(ffn_ab_b_, n2, ffn_h_b_);
    }

GLuint GPU::swiglu_2row(Target& src, int hidden){
        return swiglu_2row(src, hidden, ffn_h_);
    }

GLuint GPU::swiglu_2row(Target& src, int hidden, Target& out){
        const int aw = x_texels(hidden, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        GLuint p = x_swiglu_2row(hidden);
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, aw, 1);
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

GLuint GPU::swiglu_split(Target& a, Target& b){
        const int aw = x_texels(a.w, ffn_fmt_);
        if (ffn_h_.tex == 0 || ffn_h_.w != aw) ffn_h_ = make_target(aw, 1);
        GLuint p = x_swiglu(a.w, false);
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

GLuint GPU::swiglu_fused(Target& src, int hidden){
        const int aw = (hidden + 1) / 2;
        if (ffn_h_.tex == 0 || ffn_h_.w != aw) ffn_h_ = make_target(aw, 1);
        GLuint p = x_swiglu(hidden, true);
        glBindFramebuffer(GL_FRAMEBUFFER, ffn_h_.fbo);
        glViewport(0, 0, aw, 1);
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

vector<float> GPU::logits(const vector<Mat>& tiles, const vector<float>& x){
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

CandidateResult GPU::logits_candidates(const vector<Mat>& tiles, const vector<float>& x,
                                      int topk, float temp, float top_p){
        return logits_candidates_from(tiles, upload_activations(x), topk, temp, top_p);
    }

CandidateResult GPU::logits_candidates_from(const vector<Mat>& tiles, GLuint xt,
                                           int topk, float temp, float top_p){
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
                GLuint p = x_top2_fast(tiles[i].rows, chunk);
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
            GLuint p = x_top2(tiles[i].rows, chunk);
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

bool GPU::gpu_attn_enabled() const{
        return getenv("GPU_ATTN") && atoi(getenv("GPU_ATTN")) != 0;
    }

bool GPU::gpu_layer_enabled() const{
        return getenv("GPU_LAYER") && atoi(getenv("GPU_LAYER")) != 0;
    }

void GPU::init_attention(int dim, int kv_dim, int heads, int kv_heads, int head_dim,
                        int seq, bool rope_half, bool qk_norm, int layers,
                        const vector<float>& qn, const vector<float>& kn,
                        const vector<float>& rcos, const vector<float>& rsin){
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
        att_scores_ = make_target(seq / 2, heads);
        att_stats_ = make_target(heads, 1);
        att_probs_ = make_target(seq, heads);
        att_out_ = make_target(x_texels(dim, act_fmt_), 1);
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

GLuint GPU::attention_block(const Mat& qkv, GLuint x, int layer, int pos){
        if (qkv_target_.tex == 0 || kv_cache_.empty()) return 0;
        // Per-pass attribution: the block is five dependent draws and the whole
        // thing used to be one number in the profile.
        const bool prof = getenv("LAYER_PROFILE") != nullptr;
        auto tp = Clock::now();
        auto lap = [&](const char* name) {
            if (!prof) return;
            glFinish();
            auto now = Clock::now();
            cerr << "PROFATT " << name << " " << elapsed_ms(tp, now) << "\n";
            tp = now;
        };
        matvec_to(qkv, x, qkv_target_);
        att_finish();
        lap("qkv");
        kv_rope_write(layer, pos);
        lap("rope");
        att_scores(layer, pos);
        lap("scores");
        att_stats(pos);
        lap("stats");
        // q16-packed scores/stats are decoded directly in att_v; the old
        // standalone probability pass is no longer compatible and is retired.
        lap("probs");
        att_v(layer, pos);
        lap("v");
        return att_out_.tex;
    }

void GPU::init_persistent(int dim, int layers, const vector<float>& rms_att,
                        const vector<float>& rms_ffn, const vector<float>& rms_final){
        layer_fast_finish_ = true;
        const int rw = x_texels(dim, res_fmt_);
        const int aw = x_texels(dim, act_fmt_);
        hidden_a_ = make_target(rw, 1);
        hidden_b_ = make_target(rw, 1);
        xb_ = make_target(aw, 1);
        xbf_ = make_target(aw, 1);
        xf_ = make_target(aw, 1);
        rms_stats_ = make_target(1, 1);
        hidden_in_ = make_texture(rw, 1, nullptr);
        rms_att_tex_.clear(); rms_ffn_tex_.clear();
        for (int l = 0; l < layers; ++l) {
            vector<float> a(rms_att.begin() + (size_t)l * dim, rms_att.begin() + (size_t)(l + 1) * dim);
            vector<float> f(rms_ffn.begin() + (size_t)l * dim, rms_ffn.begin() + (size_t)(l + 1) * dim);
            rms_att_tex_.push_back(make_q16_pair_texture(a));
            rms_ffn_tex_.push_back(make_q16_pair_texture(f));
        }
        rms_final_tex_ = make_q16_pair_texture(rms_final);
    }

GLuint GPU::upload_hidden(const vector<float>& x){
        const int tw = x_texels((int)x.size(), res_fmt_);
        vector<uint8_t> data = x_pack(x, res_fmt_);
        glBindTexture(GL_TEXTURE_2D, hidden_in_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        layer_finish("NF_UPLOAD");
        return hidden_in_;
    }

GLuint GPU::rmsnorm(GLuint x, GLuint w, Target& out){
        const int aw = x_texels(att_dim_, act_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        if (rms_stats_.tex == 0) rms_stats_ = make_target(1, 1);
        if (rms_partial_.tex == 0 || rms_partial_.w != RMS_PARTIALS)
            rms_partial_ = make_target(RMS_PARTIALS, 1);
        GLuint loc = 0;
        GLuint ps = x_rms_split();
        glBindFramebuffer(GL_FRAMEBUFFER, rms_partial_.fbo);
        glViewport(0, 0, RMS_PARTIALS, 1);
        glUseProgram(ps);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, x);
        glUniform1i(glGetUniformLocation(ps, "X"), 0);
        loc = glGetAttribLocation(ps, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        GLuint pr = x_rms_reduce();
        glBindFramebuffer(GL_FRAMEBUFFER, rms_stats_.fbo);
        glViewport(0, 0, 1, 1);
        glUseProgram(pr);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, rms_partial_.tex);
        glUniform1i(glGetUniformLocation(pr, "S"), 0);
        loc = glGetAttribLocation(pr, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        layer_finish("NF_RMS");

        GLuint pn = x_rms_norm();
        glBindFramebuffer(GL_FRAMEBUFFER, out.fbo);
        glViewport(0, 0, aw, 1);
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

void GPU::matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out, XFormat xf){
        const bool pair = res_pack_ && m.type == WeightType::INT8;
        const bool pair4 = res_pack_ && m.type == WeightType::INT4;
        const int ow = (pair || pair4) ? (m.rows + 1) / 2 : m.rows;
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        GLuint p = pair4 ? x_i4_add_pair(m.rows, m.cols, xf)
                  : pair ? x_q8_add_pair(m.rows, m.cols, xf)
                  : (m.type == WeightType::INT8) ? x_q8_add(m.rows, m.cols, xf)
                  : (m.type == WeightType::INT4) ? x_i4_add(m.rows, m.cols, xf)
                                                 : x_i16_add(m.rows, m.cols, xf);
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
        const int strip = pair ? ((m.rows == 1536) ? 384 : (m.rows + 1) / 2)
                               : ((m.rows == 1536) ? 768 : m.rows);
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
            int w = min(strip, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        layer_finish("NF_MAT");
    }

vector<float> GPU::logits_from(const vector<Mat>& tiles, GLuint x){
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

vector<uint8_t> GPU::read_bytes(Target& t, int w, int h){
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        return b;
    }

void GPU::sync(){ glFinish(); }

bool GPU::needs_finish() const{
        if (getenv("FORCE_GLFINISH")) return true;
        return renderer_.find("VC4") != string::npos || renderer_.find("VideoCore") != string::npos;
    }

AttnConfig GPU::att_cfg() const{
        AttnConfig c;
        c.dim = att_dim_; c.kv_dim = att_kv_dim_; c.heads = att_heads_;
        c.kv_heads = att_kv_heads_; c.head_dim = att_head_dim_; c.seq = att_seq_;
        c.rope_half = att_rope_half_; c.qk_norm = att_qk_norm_;
        c.kv_bits = att_kv_bits_; c.kv_scale = att_kv_scale_; c.kv_width = att_kv_width_;
        return c;
    }

GLuint GPU::x_prog(const string& key, const string& src){
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        GLuint p = link_program(src);
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_q8(int rows, int cols){
        return x_prog("xq8_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_q8(rows, cols, act_fmt_));
    }

GLuint GPU::x_q8_pair(int rows, int cols){
        return x_prog("xq8p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_q8_pair(rows, cols, act_fmt_));
    }

GLuint GPU::x_q8_pair_fused(int rows, int cols){
        return x_prog("xq8pf_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_q8_pair_fused(rows, cols, act_fmt_));
    }

GLuint GPU::x_i4_pair(int rows, int cols){
        const string key = "xi4p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        GLuint p = link_program(matvec_i4_pair(rows, cols, act_fmt_));
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_q8_add(int rows, int cols, XFormat xf){
        return x_prog("xq8a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      matvec_q8_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_q8_add_pair(int rows, int cols, XFormat xf){
        return x_prog("xq8ap_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf),
                      matvec_q8_add_pair(rows, cols, xf));
    }

GLuint GPU::x_i4_add_pair(int rows, int cols, XFormat xf){
        const string key = "xi4ap_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        GLuint p = link_program(matvec_i4_add_pair(rows, cols, xf));
        programs_[key] = p;
        return p;
    }


GLuint GPU::x_i16(int rows, int cols){
        return x_prog("xi16_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_i16(rows, cols, act_fmt_));
    }

GLuint GPU::x_i16_add(int rows, int cols, XFormat xf){
        return x_prog("xi16a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      matvec_i16_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_i4(int rows, int cols){
        return x_prog("xi4_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_i4(rows, cols, act_fmt_));
    }

GLuint GPU::x_i4_add(int rows, int cols, XFormat xf){
        return x_prog("xi4a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      matvec_i4_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_swiglu_pair(int hidden){
        return x_prog("xswp_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_),
                      swiglu_pair(hidden, ffn_fmt_));
    }

GLuint GPU::x_swiglu_2row(int hidden){
        return x_prog("xsw2_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_),
                      ::swiglu_2row(hidden, ffn_fmt_));
    }

GLuint GPU::x_swiglu(int hidden, bool fused){
        return x_prog(string("xsw_") + (fused ? "f_" : "s_") + to_string(hidden) + "_" + x_format_name(ffn_fmt_),
                      swiglu(hidden, fused, ffn_fmt_));
    }

GLuint GPU::x_rms_split(){
        return x_prog("xrmss_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_),
                      rms_stats_split(att_dim_, res_fmt_));
    }

GLuint GPU::x_rms_reduce(){ return x_prog("xrmsr", rms_stats_reduce()); }

GLuint GPU::x_rms_norm(){
        return x_prog("xrmsn_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) + "_" + x_format_name(act_fmt_),
                      rms_norm(att_dim_, res_fmt_, act_fmt_));
    }

GLuint GPU::x_v(){
        return x_prog("xv_" + to_string(att_dim_) + "_" + x_format_name(act_fmt_),
                      ::att_v(att_cfg(), act_fmt_));
    }

GLuint GPU::x_kv_rope(){ return x_prog("xkv", att_kv_rope(att_cfg())); }

GLuint GPU::x_qnorm(){ return x_prog("xqn", att_qnorm_scale(att_cfg())); }

GLuint GPU::x_qrot(){ return x_prog("xqr", att_qrot(att_cfg())); }

GLuint GPU::x_score(bool hoist){ return x_prog(string("xsc_") + (hoist ? "1" : "0"), att_score(att_cfg(), hoist)); }

GLuint GPU::x_stats(){ return x_prog("xst", ::att_stats(att_cfg())); }

GLuint GPU::x_prob(){ return x_prog("xpr", att_prob(att_cfg())); }

GLuint GPU::x_sum(){ return x_prog("xsum", sum_logits()); }

GLuint GPU::x_top2_fast(int rows, int chunk){
        return x_prog("xt2f_" + to_string(rows) + "_" + to_string(chunk), top2_fast(rows, chunk));
    }

GLuint GPU::x_top2(int rows, int chunk){
        return x_prog("xt2_" + to_string(rows) + "_" + to_string(chunk), top2(rows, chunk));
    }

void GPU::init_egl(){
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

void GPU::init_gl(){
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    }

GLuint GPU::compile_shader(GLenum type, const string& src){
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

GLuint GPU::link_program(const string& fs){
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

GLuint GPU::make_texture(int w, int h, const void* data){
        GLuint t = 0; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFlush(); glFinish();
        return t;
    }

GLuint GPU::make_q24_texture(const vector<float>& x, float scale){
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

// RMS weight vectors packed two q16 per texel: the norm pass writes two (or
// four) outputs per fragment, so a pair read halves its W fetches.
GLuint GPU::make_q16_pair_texture(const vector<float>& x){
        vector<uint8_t> data = x_pack(x, XFormat::I16_2);
        return make_texture(x_texels((int)x.size(), XFormat::I16_2), 1, data.data());
    }

GLuint GPU::make_q16_texture(const vector<float>& x){
        vector<uint8_t> data((size_t)x.size() * 4);
        for (size_t i = 0; i < x.size(); ++i) {
            uint16_t u = (uint16_t)quantize_act(x[i]);
            data[i * 4 + 0] = u >> 8; data[i * 4 + 1] = u & 255;
            data[i * 4 + 2] = 0; data[i * 4 + 3] = 255;
        }
        return make_texture((int)x.size(), 1, data.data());
    }

Target GPU::make_target(int w, int h){
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

Target& GPU::target_for(int rows){
        auto it = targets_.find(rows);
        if (it != targets_.end()) return it->second;
        Target t = make_target(rows, 1);
        targets_[rows] = t;
        return targets_[rows];
    }

GLuint GPU::upload_activations(const vector<float>& x){
        const int n = (int)x.size();
        const int texels = x_texels(n, act_fmt_);
        vector<uint8_t> data = x_pack(x, act_fmt_);
        auto it = input_textures_.find(n);
        if (it == input_textures_.end()) {
            GLuint t = make_texture(texels, 1, data.data());
            input_textures_[n] = t;
            return t;
        }
        glBindTexture(GL_TEXTURE_2D, it->second);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texels, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return it->second;
    }



    void GPU::layer_finish(const char* skip_env){
        bool skip = skip_env && getenv(skip_env);
        if (skip_env && layer_fast_finish_ && !getenv("LAYER_SAFE_FINISH")) {
            string s(skip_env);
            if (s == "NF_RMS" || s == "NF_MAT" || s == "NF_ATT") skip = true;
        }
        if (skip) return;
        if (needs_finish()) { glFlush(); glFinish(); }
    }

void GPU::att_finish() {
    // VC4 needs an explicit finish between FBO -> texture passes.
    layer_finish("NF_ATT");
}

    void GPU::kv_rope_write(int layer, int pos){
        GLuint p = x_kv_rope();
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

void GPU::att_scores(int layer, int pos){
        // The q16 score-pair layout requires the hoisted q path; ATTN_HOIST=0
        // is intentionally ignored after the q16 attention change.
        static const bool hoist = true;
        if (!hoist) {
            GLuint pi = x_score(false);
            glBindFramebuffer(GL_FRAMEBUFFER, att_scores_.fbo);
            glViewport(0, 0, att_seq_ / 2, att_heads_);
            glUseProgram(pi);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qkv_target_.tex);
            glUniform1i(glGetUniformLocation(pi, "R"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, kv_cache_[layer].tex);
            glUniform1i(glGetUniformLocation(pi, "KV"), 1);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, rope_tex_);
            glUniform1i(glGetUniformLocation(pi, "RT"), 2);
            if (att_qk_norm_) {
                glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, q_norm_tex_[layer]);
                glUniform1i(glGetUniformLocation(pi, "QN"), 3);
            }
            glUniform1f(glGetUniformLocation(pi, "pos"), (float)pos);
            glUniform1f(glGetUniformLocation(pi, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
            GLuint li = glGetAttribLocation(pi, "p");
            glEnableVertexAttribArray(li);
            glVertexAttribPointer(li, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            att_finish();
            return;
        }
        // Position-only work first: qk-norm factor per head, then the rotated q.
        if (att_qk_norm_) {
            if (qscale_target_.tex == 0 || qscale_target_.w != att_heads_)
                qscale_target_ = make_target(att_heads_, 1);
            GLuint pq = x_qnorm();
            glBindFramebuffer(GL_FRAMEBUFFER, qscale_target_.fbo);
            glViewport(0, 0, att_heads_, 1);
            glUseProgram(pq);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qkv_target_.tex);
            glUniform1i(glGetUniformLocation(pq, "R"), 0);
            GLuint lq = glGetAttribLocation(pq, "p");
            glEnableVertexAttribArray(lq);
            glVertexAttribPointer(lq, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            att_finish();
        }
        if (qrot_target_.tex == 0 || qrot_target_.w != att_dim_ / 2)
            qrot_target_ = make_target(att_dim_ / 2, 1);
        GLuint pr = x_qrot();
        glBindFramebuffer(GL_FRAMEBUFFER, qrot_target_.fbo);
        glViewport(0, 0, att_dim_ / 2, 1);
        glUseProgram(pr);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qkv_target_.tex);
        glUniform1i(glGetUniformLocation(pr, "R"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, rope_tex_);
        glUniform1i(glGetUniformLocation(pr, "RT"), 1);
        if (att_qk_norm_) {
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, q_norm_tex_[layer]);
            glUniform1i(glGetUniformLocation(pr, "QN"), 2);
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, qscale_target_.tex);
            glUniform1i(glGetUniformLocation(pr, "QS"), 3);
        }
        glUniform1f(glGetUniformLocation(pr, "pos"), (float)pos);
        GLuint lr = glGetAttribLocation(pr, "p");
        glEnableVertexAttribArray(lr);
        glVertexAttribPointer(lr, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();

        GLuint p = x_score(true);
        glBindFramebuffer(GL_FRAMEBUFFER, att_scores_.fbo);
        glViewport(0, 0, att_seq_ / 2, att_heads_);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, qrot_target_.tex);
        glUniform1i(glGetUniformLocation(p, "Q"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, kv_cache_[layer].tex);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        glUniform1f(glGetUniformLocation(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::att_stats(int pos){
        GLuint p = x_stats();
        glBindFramebuffer(GL_FRAMEBUFFER, att_stats_.fbo);
        glViewport(0, 0, att_heads_, 1);
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

void GPU::att_probs(int pos){
        GLuint p = x_prob();
        glBindFramebuffer(GL_FRAMEBUFFER, att_probs_.fbo);
        glViewport(0, 0, att_seq_ / 2, att_heads_);
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

void GPU::att_v(int layer, int pos){
        // q16 score/stats layout has one packed texel per head; fused decode is mandatory.
        static const bool fuse_prob = true;
        GLuint p = x_v();
        glBindFramebuffer(GL_FRAMEBUFFER, att_out_.fbo);
        glViewport(0, 0, att_out_.w, 1);
        glUseProgram(p);
        if (fuse_prob) {
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, att_scores_.tex);
            glUniform1i(glGetUniformLocation(p, "S"), 0);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, att_stats_.tex);
            glUniform1i(glGetUniformLocation(p, "ST"), 2);
        } else {
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, att_probs_.tex);
            glUniform1i(glGetUniformLocation(p, "P"), 0);
        }
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, kv_cache_[layer].tex);
        glUniform1i(glGetUniformLocation(p, "KV"), 1);
        glUniform1f(glGetUniformLocation(p, "pos"), (float)pos);
        GLuint loc = glGetAttribLocation(p, "p");
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::draw(const Mat& m, GLuint x, Target& t, int row ){
        GLuint p = (m.type == WeightType::INT8) ? x_q8(m.rows, m.cols)
                  : (m.type == WeightType::INT4) ? x_i4(m.rows, m.cols)
                                                 : x_i16(m.rows, m.cols);
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

vector<int32_t> GPU::read_i32(Target& t, int w, int h){
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
