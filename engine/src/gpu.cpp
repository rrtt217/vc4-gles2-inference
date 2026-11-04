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

// Defined next to the bind cache helpers further down; declared here because the
// debug dump and the prefix cache invalidate the cache from earlier in the file.
static void gl_state_forget_bindings();

// Default-off probe (ATTN_DUP=qkv,rope,qknorm,qrot,score,stats,v or all):
// replay one attention pass per layer.  Every shader involved is deterministic
// and writes the same FBO from unchanged inputs, so the second draw reproduces
// the same texels and the output stays byte-identical.  The end-to-end delta is
// the true marginal cost of that pass, unlike the LAYER_PROFILE laps, whose glFinish
// also retires the draw issued before the lap started.
static bool attn_dup(const char* name){
    static const std::string env = getenv("ATTN_DUP") ? getenv("ATTN_DUP") : "";
    if (env.empty()) return false;
    const std::string want(name);
    size_t p = 0;
    while (p <= env.size()) {
        const size_t q = env.find(",", p);
        const std::string tok = env.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok == want || tok == "all") return true;
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return false;
}


// Q4_INLINE selects the INT4 activation-decode form per kernel group, so the
// generic and the inline build can be A/B-ed from one binary (the program
// cache key carries the choice).  Unset = all inline; Q4_INLINE=0 = all
// generic; otherwise a comma list of gate,w2,qkv enables only those.
static bool q4_inline(const char* name){
    static const char* env = getenv("Q4_INLINE");
    if (!env) return true;
    static const std::string s = env;
    if (s.empty() || s == "0") return false;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find(",", p);
        const std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (tok == name || tok == "all") return true;
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return false;
}

GPU::GPU(){
        init_egl();
        init_gl();
        GLint max_tex = 0, max_rb = 0, max_vp[2] = {0, 0};
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
        glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_rb);
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_vp);
        max_target_dim_ = max_tex;
        if (max_rb > 0 && max_rb < max_target_dim_) max_target_dim_ = max_rb;
        if (max_vp[0] > 0 && max_vp[0] < max_target_dim_) max_target_dim_ = max_vp[0];
        if (max_vp[1] > 0 && max_vp[1] < max_target_dim_) max_target_dim_ = max_vp[1];
        if (max_target_dim_ <= 0) max_target_dim_ = 2048;
    }

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
        // Weights may be borrowed straight from the checkpoint mmap, so every
        // reader goes through the w8()/w16()/w4() accessors; the +128 bias of INT8
        // cannot be applied in place to a read-only map.
        if (m.type == WeightType::INT8) {
            const int8_t* src = m.w8();
            const size_t n = (size_t)m.rows * m.cols;
            vector<uint8_t> data(n);
            for (size_t i = 0; i < n; ++i) data[i] = (uint8_t)((int)src[i] + 128);
            m.tex = make_texture(m.cols / 4, m.rows, data.data());
        } else if (m.type == WeightType::INT16) {
            const int16_t* src = m.w16();
            const int tex_w = (m.cols + 1) / 2;
            vector<uint8_t> data((size_t)m.rows * tex_w * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int c = 0; c < m.cols; ++c) {
                    uint16_t u = (uint16_t)src[(size_t)r * m.cols + c];
                    int o = ((size_t)r * tex_w + c / 2) * 4;
                    if ((c & 1) == 0) { data[o] = u & 255; data[o + 1] = u >> 8; }
                    else               { data[o + 2] = u & 255; data[o + 3] = u >> 8; }
                }
            }
            m.tex = make_texture(tex_w, m.rows, data.data());
        } else {
            const uint8_t* src4 = m.w4();
            const int16_t* s4 = m.s4();
            const int16_t* mn4 = m.m4();
            m.tex = make_texture((m.cols + 7) / 8, m.rows, src4);
            vector<uint8_t> pd((size_t)m.rows * m.groups * 4);
            for (int r = 0; r < m.rows; ++r) {
                for (int g = 0; g < m.groups; ++g) {
                    uint16_t sc = (uint16_t)s4[(size_t)r * m.groups + g];
                    uint16_t mn = (uint16_t)mn4[(size_t)r * m.groups + g];
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
            bind_fbo(ffn_pair_a_.fbo);
            glViewport(0, 0, hidden, 1);
            use_program(p);
            bind_tex(0, w13.tex);
            set_sampler(p, "W", 0);
            bind_tex(1, x);
            set_sampler(p, "X", 1);
            glUniform1f(uloc(p, "ws"), w13.scale);
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            GLuint ps = x_swiglu_pair(hidden);
            bind_fbo(ffn_h_a_.fbo);
            glViewport(0, 0, aw, 1);
            use_program(ps);
            bind_tex(0, ffn_pair_a_.tex);
            set_sampler(ps, "A", 0);
            setup_attrib(ps);
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
        bind_fbo(pair_buf.fbo);
        glViewport(0, 0, hidden, 1);
        use_program(pp);
        bind_tex(0, w1.tex);
        set_sampler(pp, "W1", 0);
        bind_tex(1, w3.tex);
        set_sampler(pp, "W2", 1);
        bind_tex(2, x);
        set_sampler(pp, "X", 2);
        if (i4) {
            bind_tex(3, w1.q4_param_tex);
            set_sampler(pp, "P1", 3);
            bind_tex(4, w3.q4_param_tex);
            set_sampler(pp, "P2", 4);
            glUniform1f(uloc(pp, "ps1"), w1.q4_param_scale);
            glUniform1f(uloc(pp, "ps2"), w3.q4_param_scale);
        } else {
            glUniform1f(uloc(pp, "ws1"), w1.scale);
            glUniform1f(uloc(pp, "ws2"), w3.scale);
        }
        setup_attrib(pp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();

        GLuint ps = x_swiglu_pair(hidden);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(ps);
        bind_tex(0, pair_buf.tex);
        set_sampler(ps, "A", 0);
        setup_attrib(ps);
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
            // Fused gate+swiglu: one draw per chunk instead of two.  Only the
            // q8 path has the fused kernel, and only I16_2 output is wired to w2.
            // Default ON (measured -20 ms/token).  The fused kernel differs from the two
            // pass sequence in the last bits of the swiglu output (PPL 312.1366 vs
            // 311.1636, 0.3% - far inside the 254-token corpus noise, see WORKLOG), but
            // the greedy gates are byte-identical.  FFN_FUSE=0 restores the two-pass
            // path for A/B and for any future bit-exactness work.
            static const bool fuse_env = !(getenv("FFN_FUSE") && strcmp(getenv("FFN_FUSE"), "0") == 0);
            if (fuse_env && w1a.type == WeightType::INT8 && ffn_fmt_ == XFormat::I16_2 &&
                (n1 & 1) == 0 && (n2 & 1) == 0) {
                ha = ffn_pair_swiglu_chunk(w1a, w3a, x, n1, ffn_h_a_);
                hb = ffn_pair_swiglu_chunk(w1b, w3b, x, n2, ffn_h_b_);
                return;
            }
            // q4 has no fused gate: matvec_i4_pair writes one row per fragment
            // and a separate swiglu draw packs it.  The fused form computes the
            // row pair in one fragment, swiglus both rows in-kernel and writes
            // the packed q16 pair, so it is one draw per chunk and bit-identical
            // (same group order, same qz/sl rounding).
            if (fuse_env && w1a.type == WeightType::INT4 && ffn_fmt_ == XFormat::I16_2 &&
                (n1 & 1) == 0 && (n2 & 1) == 0) {
                ha = ffn_pair_i4_swiglu_chunk(w1a, w3a, x, n1, ffn_h_a_);
                hb = ffn_pair_i4_swiglu_chunk(w1b, w3b, x, n2, ffn_h_b_);
                return;
            }

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
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, src.tex); set_sampler(p, "A", 0);
        setup_attrib(p);
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
        bind_fbo(ffn_h_.fbo);
        glViewport(0, 0, ffn_h_.w, 1);
        use_program(p);
        bind_tex(0, a.tex); set_sampler(p, "A", 0);
        bind_tex(1, b.tex); set_sampler(p, "B", 1);
        setup_attrib(p);
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
        bind_fbo(ffn_h_.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, src.tex); set_sampler(p, "A", 0);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return ffn_h_.tex;
    }

bool GPU::logits_pair() const{
        if (getenv("LOGITS_SPLIT")) return false;
        return act_fmt_ != XFormat::I8_4;
    }

// The paired kernel is q8-only; a q4 checkpoint keeps q4 vocabulary tiles and
// therefore the unpacked path (byte-identical to before this change).
bool GPU::logits_pair_ok(const vector<Mat>& tiles) const{
        if (!logits_pair()) return false;
        for (size_t i = 0; i < tiles.size(); ++i)
            if (tiles[i].type != WeightType::INT8) return false;
        return true;
    }

// Logits pass.  The paired path renders two vocabulary rows per fragment and
// packs both q16 logits into one texel, so the target is half as wide (768).
void GPU::logits_pass(const vector<Mat>& tiles, GLuint x){
        const int rows = (int)tiles.size();
        if (logits_pair_ok(tiles)) {
            if (logits_target_.tex == 0 || logits_target_.h != rows || logits_target_.w != 768)
                logits_target_ = make_target(768, rows);
            logits_packed_ = true;
            for (int i = 0; i < rows; ++i) {
                const Mat& m = tiles[i];
                GLuint p = x_q8_pair_logits(m.rows, m.cols);
                bind_fbo(logits_target_.fbo);
                use_program(p);
                bind_tex(0, m.tex); set_sampler(p, "W", 0);
                bind_tex(1, x); set_sampler(p, "X", 1);
                glUniform1f(uloc(p, "ws"), m.scale);
                setup_attrib(p);
                glViewport(0, i, (m.rows + 1) / 2, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
            return;
        }
        if (logits_target_.tex == 0 || logits_target_.h != rows || logits_target_.w != 1536)
            logits_target_ = make_target(1536, rows);
        logits_packed_ = false;
        for (int i = 0; i < rows; ++i) draw(tiles[i], x, logits_target_, i);
    }

// Read the logits target back as floats, unpacking the texel pairs when the
// paired pass ran.
vector<float> GPU::read_logits_full(const vector<Mat>& tiles){
        const int rows = (int)tiles.size();
        vector<float> out;
        out.reserve((size_t)rows * 1536);
        if (logits_packed_) {
            const int w = logits_target_.w;
            bind_fbo(logits_target_.fbo);
            vector<uint8_t> b((size_t)w * rows * 4);
            glReadPixels(0, 0, w, rows, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
            for (int i = 0; i < rows; ++i) {
                const int n = tiles[i].rows;
                for (int j = 0; j < n; ++j) {
                    const int t = j / 2, half = j - t * 2;
                    const size_t o = ((size_t)i * w + t) * 4 + (half ? 2 : 0);
                    int v = (int)b[o] * 256 + b[o + 1];
                    if (v >= 32768) v -= 65536;
                    out.push_back((float)v / ACT_SCALE);
                }
            }
            return out;
        }
        vector<int32_t> raw = read_i32(logits_target_, 1536, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < tiles[i].rows; ++j) {
                int v = raw[(size_t)i * 1536 + j];
                if (v >= 32768) v -= 65536;
                out.push_back((float)v / ACT_SCALE);
            }
        }
        return out;
    }

vector<float> GPU::logits(const vector<Mat>& tiles, const vector<float>& x){
        GLuint xt = upload_activations(x);
        logits_pass(tiles, xt);
        if (needs_finish()) { glFlush(); glFinish(); }
        return read_logits_full(tiles);

    }
CandidateResult GPU::logits_candidates(const vector<Mat>& tiles, const vector<float>& x,
                                      int topk, float temp, float top_p){
        return logits_candidates_from(tiles, upload_activations(x), topk, temp, top_p);
    }

CandidateResult GPU::logits_candidates_from(const vector<Mat>& tiles, GLuint xt,
                                           int topk, float temp, float top_p){
        const int rows = (int)tiles.size();
        logits_pass(tiles, xt);
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
                GLuint p = logits_packed_ ? x_top2_fast_packed(tiles[i].rows, chunk)
                          : x_top2_fast(tiles[i].rows, chunk);
                bind_fbo(candidate_target_.fbo);
                glViewport(0, i, chunks * 2, 1);
                use_program(p);
                bind_tex(0, logits_target_.tex);
                set_sampler(p, "L", 0);
                glUniform1f(uloc(p, "tr"), (float)i);
                glUniform1f(uloc(p, "nr"), (float)rows);
                setup_attrib(p);
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
            GLuint p = logits_packed_ ? x_top2_packed(tiles[i].rows, chunk)
                          : x_top2(tiles[i].rows, chunk);
            bind_fbo(candidate_target_.fbo);
            glViewport(0, i, chunks * 3, 1);
            use_program(p);
            bind_tex(0, logits_target_.tex);
            set_sampler(p, "L", 0);
            glUniform1f(uloc(p, "tr"), (float)i);
            glUniform1f(uloc(p, "nr"), (float)rows);
            glUniform1f(uloc(p, "invt"), 1.0f / max(0.0001f, temp));
            setup_attrib(p);
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
        res.full = read_logits_full(tiles);
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
                        const vector<float>& rcos, const vector<float>& rsin, bool qkv_pair){
        att_dim_ = dim; att_kv_dim_ = kv_dim; att_heads_ = heads;
        att_kv_heads_ = kv_heads; att_head_dim_ = head_dim; att_seq_ = seq;
        att_rope_half_ = rope_half; att_qk_norm_ = qk_norm;
        att_qkv_pair_ = qkv_pair && ((dim + 2 * kv_dim) % 2 == 0);
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
        qkv_target_ = make_target(att_qkv_pair_ ? (qkv_rows + 1) / 2 : qkv_rows, 1);
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
        // Two qkv rows per fragment: one activation decode feeds two dot
        // products, exactly like the paired logits pass.  The kernel writes the
        // same per-row q16 integers the 24-bit writer produced (x_st2q applies
        // the identical round/clamp), so the rope/qk-norm/qrot readers see the
        // same values.
        auto draw_qkv = [&]{
            if (att_qkv_pair_ && qkv.type == WeightType::INT8) {
                const int pairs = (qkv.rows + 1) / 2;
                GLuint p = x_prog("xqkvp_" + to_string(qkv.rows) + "_" + to_string(qkv.cols) +
                                  "_" + x_format_name(act_fmt_),
                                  matvec_q8_pair_logits(qkv.rows, qkv.cols, act_fmt_));
                bind_fbo(qkv_target_.fbo);
                glViewport(0, 0, pairs, 1);
                use_program(p);
                bind_tex(0, qkv.tex); set_sampler(p, "W", 0);
                bind_tex(1, x); set_sampler(p, "X", 1);
                glUniform1f(uloc(p, "ws"), qkv.scale);
                setup_attrib(p);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            } else {
                matvec_to(qkv, x, qkv_target_);
            }
        };
        draw_qkv();
        if (attn_dup("qkv")) draw_qkv();
        att_finish();
        lap("qkv");
        kv_rope_write(layer, pos);
        if (attn_dup("rope")) kv_rope_write(layer, pos);
        lap("rope");
        att_scores(layer, pos);
        lap("scores");
        att_stats(pos);
        if (attn_dup("stats")) att_stats(pos);
        lap("stats");
        // q16-packed scores/stats are decoded directly in att_v; the old
        // standalone probability pass is no longer compatible and is retired.
        lap("probs");
        att_v(layer, pos);
        if (attn_dup("v")) att_v(layer, pos);
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
        bind_tex(0, hidden_in_);
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
        bind_fbo(rms_partial_.fbo);
        glViewport(0, 0, RMS_PARTIALS, 1);
        use_program(ps);
        bind_tex(0, x);
        set_sampler(ps, "X", 0);
        setup_attrib(ps);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // RMS_FUSED folds the 32-way reduce into every normalising fragment:
        // same summation order, so bit-identical, one draw less per rmsnorm.
        // VC4 defaults to the fused reduce (one draw less per rmsnorm, same
        // summation order so bit-identical); other backends keep the split pass,
        // where a 32-fetch fragment per output texel may cost more than a draw.
        // ACT8=1 (int8 activations) does not accept the fused form.
        const bool fused_reduce = rms_fused() && act_fmt_ != XFormat::I8_4;
        if (!fused_reduce) {
            GLuint pr = x_rms_reduce();
            bind_fbo(rms_stats_.fbo);
            glViewport(0, 0, 1, 1);
            use_program(pr);
            bind_tex(0, rms_partial_.tex);
            set_sampler(pr, "S", 0);
            setup_attrib(pr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        layer_finish("NF_RMS");

        GLuint pn = x_rms_norm();
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(pn);
        bind_tex(0, x);
        set_sampler(pn, "X", 0);
        bind_tex(1, w);
        set_sampler(pn, "W", 1);
        bind_tex(2, fused_reduce ? rms_partial_.tex : rms_stats_.tex);
        set_sampler(pn, "S", 2);
        setup_attrib(pn);
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
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex);
        set_sampler(p, "W", 0);
        bind_tex(1, x);
        set_sampler(p, "X", 1);
        if (m.type == WeightType::INT4) {
            // P takes unit 2 on the q4 kernels, so R moves to 3.
            bind_tex(2, m.q4_param_tex);
            set_sampler(p, "P", 2);
            bind_tex(3, residual);
            set_sampler(p, "R", 3);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        } else {
            bind_tex(2, residual);
            set_sampler(p, "R", 2);
        }
        glUniform1f(uloc(p, "ws"), m.scale);
        setup_attrib(p);
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
        logits_pass(tiles, x);
        if (needs_finish()) { glFlush(); glFinish(); }
        return read_logits_full(tiles);
    }

void GPU::logits_pass_only(const vector<Mat>& tiles, GLuint x){
        logits_pass(tiles, x);
    }

vector<uint8_t> GPU::read_bytes(Target& t, int w, int h){
        bind_fbo(t.fbo);
        glViewport(0, 0, t.w, t.h);
        vector<uint8_t> b((size_t)w * h * 4);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, b.data());
        return b;
    }

void GPU::sync(){ glFinish(); }

void GPU::forget_state(){ gl_state_forget_bindings(); }

void GPU::dump_target(const Target& t, const char* name, int n){
        if (t.tex == 0) return;
        if (n > t.w) n = t.w;
        vector<uint8_t> b((size_t)t.w * 4);
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, t.w, 1, GL_RGBA, GL_UNSIGNED_BYTE, &b[0]);
        gl_state_forget_bindings();
        fprintf(stderr, "DUMP %s w=%d", name, t.w);
        for (int i = 0; i < n; i++) fprintf(stderr, " %02x%02x%02x%02x", b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]);
        fprintf(stderr, "\n");
}

bool GPU::rms_fused() const{
        // RMS_FUSED forces it on, RMS_SPLIT forces the old two-pass form.
        if (getenv("RMS_FUSED")) return true;
        if (getenv("RMS_SPLIT")) return false;
        return needs_finish();   // VC4 / VideoCore
    }

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
        c.qkv_pair = att_qkv_pair_;
        return c;
    }

// Default-off probe (SHADER_TIME=1): time every fragment-program link and
// report when it happened relative to process start.  First-token latency is
// a compile-count question first: x_prog also merges keys that build
// byte-identical program text, so a repeat compile cannot hide here.
bool GPU::shader_time_on(){
        static const bool on = getenv("SHADER_TIME") != nullptr;
        return on;
    }

double GPU::shader_time_ms(){
        return run_time_ms();
    }

static const char* g_prog_key = "?";
static Clock::time_point g_run_t0;

void gpu_set_run_t0(Clock::time_point t){ g_run_t0 = t; }

double run_time_ms(){
        return elapsed_ms(g_run_t0, Clock::now());
    }

// A program can pass link and still fail: vc4 generates the QPU code at the
// first draw, and when its register allocator gives up it skips the draw
// silently (vc4_draw.c: "shader compile failed, skipping draw call").  GL raises
// no error, so the kernel keeps returning whatever the target already held and
// every later number is wrong with no diagnostic.  Draw each new program once
// into a sentinel-filled 1x1 target and stop if nothing was written.
void GPU::check_program_runs(GLuint p, const string& key){
        if (getenv("NO_SHADER_SELFTEST")) return;
        if (probe_.tex == 0) probe_ = make_target(1, 1);
        bind_fbo(probe_.fbo);
        glViewport(0, 0, 1, 1);
        glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        use_program(p);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        vector<uint8_t> b = read_bytes(probe_, 1, 1);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        if (b.size() == 4 && b[0] == 51 && b[1] == 102 && b[2] == 153) {
                cerr << "program '" << key << "' did not draw: the driver skipped the\n"
                        "  draw call, most likely because vc4's register allocator failed\n"
                        "  (run with MESA_DEBUG=1 to see it).  Refusing to continue with a\n"
                        "  kernel that would silently return stale data.\n";
                exit(1);
        }
    }

GLuint GPU::x_prog(const string& key, const string& src){
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        // Two keys can build byte-identical text (xq8pl and xqkvp both do): a
        // program is a pure function of its shaders, so share the object and
        // skip the second compile.  64-bit hash; a collision would need two
        // different texts and is not a practical risk here.
        const size_t h = std::hash<string>{}(src);
        auto t = text_programs_.find(h);
        if (t != text_programs_.end()) {
            programs_[key] = t->second;
            if (shader_time_on())
                fprintf(stderr, "SHADER_TIME reuse key %s (identical text)\n", key.c_str());
            return t->second;
        }
        g_prog_key = key.c_str();
        GLuint p = link_program(src);
        check_program_runs(p, key);
        programs_[key] = p;
        text_programs_[h] = p;
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

GLuint GPU::x_q8_pair_swiglu(int rows, int cols){
        // Inline I16_2 decode is the default; FFN_SW_INLINE=0 restores the
        // generic X4Q form in the same binary, so both can be A/B-ed
        // interleaved on one binary.
        const char* v = getenv("FFN_SW_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        return x_prog("xq8ps_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(act_fmt_) + (inl ? "" : "_gen"),
                      matvec_q8_pair_swiglu(rows, cols, act_fmt_, inl));
    }

// Gate pair matvec for two rows per fragment plus swiglu, straight into a q16
// pair texture that the w2 matvec can read: one draw instead of two.
GLuint GPU::ffn_pair_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out){
        const int aw = x_texels(rows, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        GLuint p = x_q8_pair_swiglu(rows, w1.cols);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x); set_sampler(p, "X", 2);
        glUniform1f(uloc(p, "ws1"), w1.scale);
        glUniform1f(uloc(p, "ws2"), w3.scale);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return out.tex;
    }

GLuint GPU::x_q8_gate2(int rows, int cols){
        return x_prog("xq8g2_" + to_string(rows) + "_" + to_string(cols),
                      matvec_q8_gate2(rows, cols));
    }

GLuint GPU::x_q8_p2(int rows, int cols){
        return x_prog("xq8p2_" + to_string(rows) + "_" + to_string(cols),
                      matvec_q8_p2(rows, cols));
    }

// Position A then position B in one texture, one row wide: the 2-position
// kernels read A at texel [0,axw) and B at [axw,2*axw).  Keyed by length like
// upload_input, but kept in its own map: a single-position upload of the same
// length would otherwise overwrite one half.
GLuint GPU::upload_input2(const vector<float>& a, const vector<float>& b){
        const int n = (int)a.size();
        const int tw = x_texels(n, act_fmt_);
        vector<uint8_t> data = x_pack(a, act_fmt_);
        vector<uint8_t> db = x_pack(b, act_fmt_);
        if ((int)data.size() != tw * 4 || (int)db.size() != tw * 4) {
            cerr << "upload_input2: activation size mismatch\n";
            exit(1);
        }
        data.insert(data.end(), db.begin(), db.end());
        auto it = input_textures2_.find(n);
        if (it == input_textures2_.end()) {
            input_textures2_[n] = make_texture(2 * tw, 1, data.data());
            return input_textures2_[n];
        }
        bind_tex(0, it->second);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2 * tw, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        if (needs_finish()) { glFlush(); glFinish(); }
        return it->second;
    }

// Two-position FFN gate: one draw for both positions (see matvec_q8_gate2).
GLuint GPU::ffn_pair_swiglu_chunk2(const Mat& w1, const Mat& w3, GLuint x2, int rows, Target& out){
        if (out.tex == 0 || out.w != rows) out = make_target(rows, 1);
        GLuint p = x_q8_gate2(rows, w1.cols);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x2); set_sampler(p, "X", 2);
        glUniform1f(uloc(p, "ws1"), w1.scale);
        glUniform1f(uloc(p, "ws2"), w3.scale);
        setup_attrib(p);
        for (int x0 = 0; x0 < rows; x0 += 1024) {
            const int w = min(1024, rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return out.tex;
    }

// Two-position w2: reads the P2 gate output, writes one P2 q16 texel per row.
void GPU::matvec_p2(const Mat& m, GLuint xp2, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        GLuint p = x_q8_p2(m.rows, m.cols);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        glUniform1f(uloc(p, "ws"), m.scale);
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += 1024) {
            const int w = min(1024, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
    }

// q4 gate chunk: row pair per fragment, w1/w3 dots and both swiglu results in
// one draw (see matvec_i4_pair_swiglu).
GLuint GPU::ffn_pair_i4_swiglu_chunk(const Mat& w1, const Mat& w3, GLuint x, int rows, Target& out){
        const int aw = x_texels(rows, ffn_fmt_);
        if (out.tex == 0 || out.w != aw) out = make_target(aw, 1);
        GLuint p = x_i4_pair_swiglu(rows, w1.cols);
        bind_fbo(out.fbo);
        glViewport(0, 0, aw, 1);
        use_program(p);
        bind_tex(0, w1.tex); set_sampler(p, "W1", 0);
        bind_tex(1, w3.tex); set_sampler(p, "W2", 1);
        bind_tex(2, x); set_sampler(p, "X", 2);
        bind_tex(3, w1.q4_param_tex); set_sampler(p, "P1", 3);
        bind_tex(4, w3.q4_param_tex); set_sampler(p, "P2", 4);
        glUniform1f(uloc(p, "ps1"), w1.q4_param_scale);
        glUniform1f(uloc(p, "ps2"), w3.q4_param_scale);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (layer_fast_finish_ || getenv("FFN_FLUSH")) glFlush();
        else layer_finish("NF_FFN");
        if (getenv("FFN_SYNC")) { glFlush(); glFinish(); }
        return out.tex;
}

GLuint GPU::x_q8_pair_fused(int rows, int cols){
        return x_prog("xq8pf_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_q8_pair_fused(rows, cols, act_fmt_));
    }

GLuint GPU::x_i4_pair(int rows, int cols){
        const string key = "xi4p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key.c_str();
        GLuint p = link_program(matvec_i4_pair(rows, cols, act_fmt_));
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_i4_pair_swiglu(int rows, int cols){
        const bool inl = q4_inline("gate");
        return x_prog("xi4ps_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + (inl ? "" : "_gen"),
                      matvec_i4_pair_swiglu(rows, cols, act_fmt_, inl));
    }

// GL binding state is context-wide (GLES2 has no VAOs and the engine owns the
// context), so these live at file scope: the static texture helpers below bind
// on their own and can forget the cache without an instance.
static GLuint g_bound_tex[8] = {0, 0, 0, 0, 0, 0, 0, 0};
static GLuint g_cur_fbo = 0xFFFFFFFFu;
static GLuint g_cur_prog = 0xFFFFFFFFu;

static void gl_state_forget_bindings() {
    for (int i = 0; i < 8; i++) g_bound_tex[i] = 0xFFFFFFFFu;
    g_cur_fbo = 0xFFFFFFFFu;
}

GLint GPU::aloc(GLuint prog, const char* name) {
        if (prog != aloc_prog_ || !aloc_tab_) {
            aloc_tab_ = &aloc_cache_[prog];
            aloc_prog_ = prog;
        }
        for (const auto& e : *aloc_tab_) if (e.name == name) return e.loc;
        GLint loc = glGetAttribLocation(prog, name);
        aloc_tab_->push_back({name, loc});
        return loc;
    }

void GPU::bind_tex(int unit, GLuint tex) {
        if (unit < 0 || unit > 7 || g_bound_tex[unit] == tex) return;
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, tex);
        g_bound_tex[unit] = tex;
    }

void GPU::bind_fbo(GLuint fbo) {
        if (g_cur_fbo == fbo) return;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        g_cur_fbo = fbo;
    }

void GPU::use_program(GLuint prog) {
        if (g_cur_prog == prog) return;
        glUseProgram(prog);
        g_cur_prog = prog;
    }

void GPU::set_sampler(GLuint prog, const char* name, int unit) {
        // Usually a sampler keeps one unit for the life of its program, so the
        // uniform is written once; the q4 kernels re-point R from unit 2 to 3, so
        // the cached unit is verified rather than assumed.
        auto& v = sampler_set_[prog];
        for (auto& e : v) {
            if (e.name != name) continue;
            if (e.unit == unit) return;
            e.unit = unit;
            glUniform1i(uloc(prog, name), unit);
            return;
        }
        glUniform1i(uloc(prog, name), unit);
        v.push_back({name, unit});
    }

void GPU::setup_attrib(GLuint prog) {
        // GLES2 has no VAOs, so the single "p" attribute is context state: set it
        // once per (distinct) location instead of three calls per draw.
        GLint loc = aloc(prog, "p");
        if (loc < 0) return;
        if (attrib_ready_ && loc == attrib_p_) return;
        glEnableVertexAttribArray((GLuint)loc);
        glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        attrib_p_ = loc;
        attrib_ready_ = true;
    }

GLint GPU::uloc(GLuint prog, const char* name) {
        // Uniform names are string literals, so the pointer is a stable identity:
        // a per-program linear scan over a handful of entries beats hashing.
        if (prog != uloc_prog_ || !uloc_tab_) {
            uloc_tab_ = &uloc_cache_[prog];
            uloc_prog_ = prog;
        }
        for (const auto& e : *uloc_tab_) if (e.name == name) return e.loc;
        GLint loc = glGetUniformLocation(prog, name);
        uloc_tab_->push_back({name, loc});
        return loc;
    }

GLuint GPU::x_q8_add(int rows, int cols, XFormat xf){
        return x_prog("xq8a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      matvec_q8_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_q8_add_pair(int rows, int cols, XFormat xf){
        // The inline I16_2 decode is the default; Q8AP_INLINE=0 restores the
        // generic X4Q form in the same binary, so the two can be A/B-ed
        // interleaved without a second build.
        const char* v = getenv("Q8AP_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        return x_prog("xq8ap_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(xf) + (inl ? "" : "_gen"),
                      matvec_q8_add_pair(rows, cols, xf, inl));
    }

GLuint GPU::x_i4_add_pair(int rows, int cols, XFormat xf){
        const bool inl = q4_inline("w2");
        const string key = "xi4ap_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + (inl ? "" : "_gen");
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key.c_str();
        GLuint p = link_program(matvec_i4_add_pair(rows, cols, xf, inl));
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
        const bool inl = q4_inline("qkv");
        return x_prog("xi4_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + (inl ? "" : "_gen"),
                      matvec_i4(rows, cols, act_fmt_, inl));
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
        const bool fused = rms_fused() && act_fmt_ != XFormat::I8_4;
        return x_prog("xrmsn_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) + "_" +
                          x_format_name(act_fmt_) + (fused ? "_f" : ""),
                      rms_norm(att_dim_, res_fmt_, act_fmt_, fused));
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

GLuint GPU::x_q8_pair_logits(int rows, int cols){
        return x_prog("xq8pl_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_),
                      matvec_q8_pair_logits(rows, cols, act_fmt_));
    }

GLuint GPU::x_top2_fast_packed(int rows, int chunk){
        return x_prog("xt2fp_" + to_string(rows) + "_" + to_string(chunk), top2_fast_packed(rows, chunk));
    }

GLuint GPU::x_top2_packed(int rows, int chunk){
        return x_prog("xt2p_" + to_string(rows) + "_" + to_string(chunk), top2_packed(rows, chunk));
    }

// The q4 decode wants ESSL 3.00 for its bitwise nibble split (b>>4 / b&15;
// ESSL 1.00 forbids them).  It is on by default and resolved in two steps: the
// environment says whether it is wanted (Q4_SHIFT=0 turns it off), and
// init_egl records whether the context really offers ESSL 3.00.  A driver that
// cannot gets the float decode back with a warning instead of a fatal error.
static bool g_q4_shift_ok = false;

static bool q4_shift_wanted(){
        const char* e = getenv("Q4_SHIFT");
        return !(e && strcmp(e, "0") == 0);
    }

bool gpu_q4_shift(){ return g_q4_shift_ok; }

void GPU::init_egl(){
        // Q4_SHIFT emits ESSL 3.00 kernels, and Mesa only accepts those when the
        // front end is told the context is ES 3.0.  The overrides change the
        // version the *compiler* is offered, not the driver's capabilities (vc4
        // still has no MRT, no integer textures, and no fround_even), so they
        // are set here, before eglInitialize, and never overwrite a value the
        // caller picked.  Q4_SHIFT_MESA=0 skips this for an A/B.
        if (q4_shift_wanted() && !getenv("Q4_SHIFT_NO_MESA")) {
                setenv("MESA_GLES_VERSION_OVERRIDE", "3.0", 0);
                setenv("MESA_GLSL_VERSION_OVERRIDE", "300", 0);
        }
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
        if (q4_shift_wanted()) {
                const char* glsl = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
                g_q4_shift_ok = glsl && strncmp(glsl, "OpenGL ES GLSL ES 3", 19) == 0;
                if (g_q4_shift_ok) {
                        if (getenv("SHADER_TIME") || getenv("Q4_SHIFT_VERBOSE"))
                                fprintf(stderr, "Q4_SHIFT: integer nibble split on %s (ESSL %s)\n",
                                        renderer_.c_str(), glsl + 18);
                } else
                        fprintf(stderr, "Q4_SHIFT: %s offers %s; using the float q4 decode\n",
                                renderer_.c_str(), glsl ? glsl : "(unknown)");
        }
    }

void GPU::init_gl(){
        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        float verts[] = { -1, -1, 3, -1, -1, 3 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    }

static string es3ify(const string& src, bool vertex){
        if (src.find("#version") != string::npos) return src;
        if (vertex) return "#version 300 es\n#define attribute in\n" + src;
        // A duplicate precision statement is legal, so the original text can
        // stay untouched: only the version line, the out declaration and the
        // gl_FragColor alias are added.
        return "#version 300 es\nprecision highp float;\nout vec4 frag_color;\n"
               "#define gl_FragColor frag_color\n" + src;
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

GLuint GPU::link_program(const string& fs_in){
        const bool needs_es3 = g_q4_shift_ok && fs_in.find("&15") != string::npos;
        const string fs = needs_es3 ? es3ify(fs_in, false) : fs_in;
        string vs = "attribute vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }";
        // The vertex shader must match the fragment stage's language when that
        // stage was rewritten above; every other program stays ESSL 1.00.
        if (needs_es3) vs = es3ify(vs, true);
        const bool tm = shader_time_on();
        const double t0 = tm ? shader_time_ms() : 0.0;
        GLuint p = glCreateProgram();
        glAttachShader(p, compile_shader(GL_VERTEX_SHADER, vs));
        glAttachShader(p, compile_shader(GL_FRAGMENT_SHADER, fs));
        glLinkProgram(p);
        GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[8192]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
            cerr << "program link failed:\n" << log << "\n"; exit(1);
        }
        if (tm) {
            static int count = 0;
            static double total = 0.0;
            const double ms = shader_time_ms() - t0;
            ++count; total += ms;
            fprintf(stderr, "SHADER_TIME #%02d at %8.1f ms  link %6.1f ms  src %5zu B  h=%08zx  key=%s  cum %7.1f ms\n",
                    count, shader_time_ms(), ms, fs.size(),
                    (size_t)std::hash<string>{}(fs), g_prog_key, total);
        }
        return p;
    }

GLuint GPU::make_texture(int w, int h, const void* data){
        GLuint t = 0; glGenTextures(1, &t); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t); gl_state_forget_bindings();
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
        if (w <= 0 || h <= 0 || w > max_target_dim_ || h > max_target_dim_) {
            cerr << "target " << w << "x" << h
                 << " exceeds GL target limit " << max_target_dim_ << "\n";
            exit(1);
        }
        Target t; t.w = w; t.h = h;
        t.tex = make_texture(w, h, nullptr);
        glGenFramebuffers(1, &t.fbo);
        bind_fbo(t.fbo);
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
        bind_tex(0, it->second);
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
        bind_fbo(kv_cache_[layer].fbo);
        glViewport(0, pos, att_kv_width_, 1);
        use_program(p);
        bind_tex(0, qkv_target_.tex);
        set_sampler(p, "QKV", 0);
        bind_tex(1, rope_tex_);
        set_sampler(p, "RT", 1);
        if (att_qk_norm_) {
            bind_tex(2, k_norm_tex_[layer]);
            set_sampler(p, "KN", 2);
        }
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::att_scores(int layer, int pos){
        // The q16 score-pair layout requires the hoisted q path; ATTN_HOIST=0
        // is intentionally ignored after the q16 attention change.
        static const bool hoist = true;
        const bool sprof = getenv("LAYER_PROFILE") != nullptr;
        auto stp = Clock::now();
        auto slap = [&](const char* n) {
            if (!sprof) return;
            glFinish();
            auto nw = Clock::now();
            cerr << "PROFATT2 " << n << " " << elapsed_ms(stp, nw) << "\n";
            stp = nw;
        };
        if (!hoist) {
            GLuint pi = x_score(false);
            bind_fbo(att_scores_.fbo);
            glViewport(0, 0, att_seq_ / 2, att_heads_);
            use_program(pi);
            bind_tex(0, qkv_target_.tex);
            set_sampler(pi, "R", 0);
            bind_tex(1, kv_cache_[layer].tex);
            set_sampler(pi, "KV", 1);
            bind_tex(2, rope_tex_);
            set_sampler(pi, "RT", 2);
            if (att_qk_norm_) {
                bind_tex(3, q_norm_tex_[layer]);
                set_sampler(pi, "QN", 3);
            }
            glUniform1f(uloc(pi, "pos"), (float)pos);
            glUniform1f(uloc(pi, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
            setup_attrib(pi);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            att_finish();
            slap("qknorm");
            return;
        }
        // Position-only work first: qk-norm factor per head, then the rotated q.
        if (att_qk_norm_) {
            if (qscale_target_.tex == 0 || qscale_target_.w != att_heads_)
                qscale_target_ = make_target(att_heads_, 1);
            GLuint pq = x_qnorm();
            auto go = [&]{
                bind_fbo(qscale_target_.fbo);
                glViewport(0, 0, att_heads_, 1);
                use_program(pq);
                bind_tex(0, qkv_target_.tex);
                set_sampler(pq, "R", 0);
                setup_attrib(pq);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                att_finish();
            };
            go();
            if (attn_dup("qknorm")) go();
            slap("qknorm");
        }
        if (qrot_target_.tex == 0 || qrot_target_.w != att_dim_ / 2)
            qrot_target_ = make_target(att_dim_ / 2, 1);
        GLuint pr = x_qrot();
        auto go_qrot = [&]{
            bind_fbo(qrot_target_.fbo);
            glViewport(0, 0, att_dim_ / 2, 1);
            use_program(pr);
            bind_tex(0, qkv_target_.tex);
            set_sampler(pr, "R", 0);
            bind_tex(1, rope_tex_);
            set_sampler(pr, "RT", 1);
            if (att_qk_norm_) {
                bind_tex(2, q_norm_tex_[layer]);
                set_sampler(pr, "QN", 2);
                bind_tex(3, qscale_target_.tex);
                set_sampler(pr, "QS", 3);
            }
            glUniform1f(uloc(pr, "pos"), (float)pos);
            setup_attrib(pr);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            att_finish();
        };
        go_qrot();
        if (attn_dup("qrot")) go_qrot();
        slap("qrot");

        GLuint p = x_score(true);
        auto go_score = [&]{
            bind_fbo(att_scores_.fbo);
            glViewport(0, 0, att_seq_ / 2, att_heads_);
            use_program(p);
            bind_tex(0, qrot_target_.tex);
            set_sampler(p, "Q", 0);
            bind_tex(1, kv_cache_[layer].tex);
            set_sampler(p, "KV", 1);
            glUniform1f(uloc(p, "pos"), (float)pos);
            glUniform1f(uloc(p, "invsqrt"), 1.0f / sqrtf((float)att_head_dim_));
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            att_finish();
        };
        go_score();
        if (attn_dup("score")) go_score();
        slap("score");
    }

void GPU::att_stats(int pos){
        GLuint p = x_stats();
        bind_fbo(att_stats_.fbo);
        glViewport(0, 0, att_heads_, 1);
        use_program(p);
        bind_tex(0, att_scores_.tex);
        set_sampler(p, "S", 0);
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::att_probs(int pos){
        GLuint p = x_prob();
        bind_fbo(att_probs_.fbo);
        glViewport(0, 0, att_seq_ / 2, att_heads_);
        use_program(p);
        bind_tex(0, att_scores_.tex);
        set_sampler(p, "S", 0);
        bind_tex(1, att_stats_.tex);
        set_sampler(p, "ST", 1);
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::att_v(int layer, int pos){
        // q16 score/stats layout has one packed texel per head; fused decode is mandatory.
        static const bool fuse_prob = true;
        GLuint p = x_v();
        bind_fbo(att_out_.fbo);
        glViewport(0, 0, att_out_.w, 1);
        use_program(p);
        if (fuse_prob) {
            bind_tex(0, att_scores_.tex);
            set_sampler(p, "S", 0);
            bind_tex(2, att_stats_.tex);
            set_sampler(p, "ST", 2);
        } else {
            bind_tex(0, att_probs_.tex);
            set_sampler(p, "P", 0);
        }
        bind_tex(1, kv_cache_[layer].tex);
        set_sampler(p, "KV", 1);
        glUniform1f(uloc(p, "pos"), (float)pos);
        setup_attrib(p);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        att_finish();
    }

void GPU::draw(const Mat& m, GLuint x, Target& t, int row ){
        GLuint p = (m.type == WeightType::INT8) ? x_q8(m.rows, m.cols)
                  : (m.type == WeightType::INT4) ? x_i4(m.rows, m.cols)
                                                 : x_i16(m.rows, m.cols);
        bind_fbo(t.fbo);
        use_program(p);
        bind_tex(0, m.tex);
        set_sampler(p, "W", 0);
        bind_tex(1, x);
        set_sampler(p, "X", 1);
        if (m.type == WeightType::INT4) {
            bind_tex(2, m.q4_param_tex);
            set_sampler(p, "P", 2);
            glUniform1f(uloc(p, "ps"), m.q4_param_scale);
        }
        glUniform1f(uloc(p, "ws"), m.scale);
        setup_attrib(p);
        // VC4 is unreliable for a single 1536-column pass (the logits tile
        // and fused-W13 width).  Split only that width; other widths are fine
        // and keeping them in one draw avoids per-draw overhead.
        // A single draw must stay below GL_MAX_TEXTURE_SIZE: at exactly 2048
        // fragments wide vc4 silently produces wrong bytes (2040 is fine,
        // 2048 differs in 5029/8192 bytes and finishes 3x too fast), while the
        // same matrix drawn as two viewport-offset halves is exact.  Splitting
        // costs one draw (~10 us), so cap every strip at 1024 and keep the
        // gl_FragCoord-absolute convention the shader relies on.
        // MATVEC_STRIP overrides the cap for experiments.
        const char* sv = getenv("MATVEC_STRIP");
        const int env_strip = sv ? atoi(sv) : 0;
        const int cap = (env_strip > 0) ? env_strip : 1024;
        const int strip = std::min(cap, m.rows);
        for (int x0 = 0; x0 < m.rows; x0 += strip) {
            int w = min(strip, m.rows - x0);
            glViewport(x0, row, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
    }

vector<int32_t> GPU::read_i32(Target& t, int w, int h){
        bind_fbo(t.fbo);
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
