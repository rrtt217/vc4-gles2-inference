// The matvec family: the x_* program selectors and the passes that drive them.
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  Every `x_*`
// function here picks a shader builder for a (format, shape) pair and asks
// `x_prog` for the program; the passes that follow draw the weight texture in
// `kDrawCap` strips and read the result back.  The GLSL is in shaders.cpp, the
// program cache in gpu_prog.cpp, the draw limit in gpu_internal.h.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "logits_grid.h"   // kLogitsGrid: the vocabulary tiling
#include "xformat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void GPU::matvec_to(const Mat& m, GLuint x, Target& out){
        matvec_to_row(m, x, out, 0);
    }

void GPU::matvec_to_row(const Mat& m, GLuint x, Target& out, int row){
        if (out.tex == 0 || out.w != m.rows || out.h <= row) out = make_target(m.rows, row + 1);
        draw(m, x, out, row);
        layer_finish("NF_MAT");
        ffn_sync();
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

bool GPU::logits_pair() const{
        if (getenv("LOGITS_SPLIT")) return false;
        return act_fmt_ != XFormat::I8_4;
    }

bool GPU::logits_pair_ok(const vector<Mat>& tiles) const{
        if (!logits_pair()) return false;
        for (size_t i = 0; i < tiles.size(); ++i)
            if (tiles[i].type != WeightType::INT8) return false;
        return true;
    }

void GPU::logits_pass(const vector<Mat>& tiles, GLuint x){
        const int rows = (int)tiles.size();
        if (logits_pair_ok(tiles)) {
            if (logits_target_.tex == 0 || logits_target_.h != rows || logits_target_.w != kLogitsGridPacked)
                logits_target_ = make_target(kLogitsGridPacked, rows);
            logits_packed_ = true;
            for (int i = 0; i < rows; ++i) {
                const Mat& m = tiles[i];
                const bool wr = m.has_row_scale();
                GLuint p = x_q8_pair_logits(m.rows, m.cols, wr);
                bind_fbo(logits_target_.fbo);
                use_program(p);
                bind_tex(0, m.tex); set_sampler(p, "W", 0);
                bind_tex(1, x); set_sampler(p, "X", 1);
                glUniform1f(uloc(p, "ws"), m.scale);
                if (wr) { bind_tex(2, m.row_scale_tex); set_sampler(p, "WS", 2); }
                setup_attrib(p);
                glViewport(0, i, (m.rows + 1) / 2, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
            }
            return;
        }
        if (logits_target_.tex == 0 || logits_target_.h != rows || logits_target_.w != kLogitsGrid)
            logits_target_ = make_target(kLogitsGrid, rows);
        logits_packed_ = false;
        for (int i = 0; i < rows; ++i) draw(tiles[i], x, logits_target_, i);
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
                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
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
                    if (idx < M) cand.push_back({(float)value / ACT_SCALE, r * kLogitsGrid + idx});
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
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
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
                            cand.push_back({v, r * kLogitsGrid + idx});
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

void GPU::matvec_add(const Mat& m, GLuint x, GLuint residual, Target& out, XFormat xf){
        const bool pair = res_pack_ && (m.type == WeightType::INT8 ||
                                        m.type == WeightType::T3);
        const bool pair4 = res_pack_ && m.type == WeightType::INT4;
        const int ow = (pair || pair4) ? (m.rows + 1) / 2 : m.rows;
        if (out.tex == 0 || out.w != ow) out = make_target(ow, 1);
        // The packed residual layout is a property of the model, not of the
        // weight format: a row-scale q8 matrix must still write the packed
        // texel the rmsnorm reads, so the paired kernel grows a wrow variant
        // rather than the writer switching layouts.
        const bool wr = (m.type == WeightType::INT8 || m.type == WeightType::T3) &&
                        m.has_row_scale();
        // A t3 matrix must write the layout `res_pack_` says the reader expects:
        // the packed pair when the residual is two values a texel (VC4's I16_2),
        // the single value otherwise.
        // `xf` is the *activation* format (the X texture); the residual is
        // always res_fmt_.  Passing xf here decodes R with the wrong layout --
        // the q8 selectors read res_fmt_ themselves, which is why they never had
        // this bug.  Measured: every odd row gained exactly 255/256 and the reply
        // collapsed to one token.
        GLuint p = m.type == WeightType::T3
                       ? (res_pack_ ? x_t3_add_pair(m, xf, res_fmt_, wr)
                                    : x_t3_add(m, xf, res_fmt_, wr))
                  : pair4 ? x_i4_add_pair(m.rows, m.cols, xf, m.fmt4)
                  : pair ? x_q8_add_pair(m.rows, m.cols, xf, wr)
                  : (m.type == WeightType::INT8) ? x_q8_add(m.rows, m.cols, xf, wr)
                  : (m.type == WeightType::INT4) ? x_i4_add(m.rows, m.cols, xf, m.fmt4)
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
        // The q4 kernels take their scale as "ps"; "ws" does not exist there.
        if (m.type != WeightType::INT4) glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        setup_attrib(p);
        // Same rule as draw(): a single draw never exceeds 1024 fragments.  A
        // full-target-width draw is what gets dropped (docs/engine-internals.md
        // section 2), and a 1536-row matrix is exactly that for its own target.
        // The old special case split 1536 into 768+768 and left every other
        // width in one pass; the cap covers all of them, and `ow` is the
        // fragment count (rows/2 for the paired kernels), not the row count.
        // Same experimental override as draw(): MATVEC_STRIP caps the fragment
        // count per draw so a suspected full-width draw can be A/B-ed.
        const char* msv = getenv("MATVEC_STRIP");
        const int cap = (msv && atoi(msv) > 0) ? atoi(msv) : kDrawCap;
        const int strip = min(cap, ow);
        for (int x0 = 0; x0 < ow; x0 += strip) {
            int w = min(strip, ow - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        layer_finish("NF_MAT");
    }

vector<float> GPU::logits_from(const vector<Mat>& tiles, GLuint x){
        // LOGITS_PERFCNT=1 reports the QPU cycles the logits pass really runs.
        // Its wall time (the RBT finish line) cannot separate work from
        // waiting, and a per-tile wall lap cannot either: every lap needs its
        // own finish, which costs more than the tile it measures.  The monitor
        // injects no finish -- its results appear once the draws retire.
        unsigned long long li = 0, lf = 0, le = 0, lq = 0;
        const bool lc = pc_available() && getenv("LOGITS_PERFCNT");
        if (lc) pc_stage_begin();
        logits_pass(tiles, x);
        if (lc) {
            pc_stage_end(li, lf, le, lq);
            const double to_ms = 1.0 / (pc_qpus() * (double)pc_mhz() * 1000.0);
            fprintf(stderr, "LOGITS gpu=%.3f exec=%.3f idle=%.3f quads=%llu\n",
                    (double)lf * to_ms, (double)le * to_ms, (double)li * to_ms, lq);
        }
        const bool rt = getenv("READBACK_TIME");
        auto t0 = Clock::now();
        if (needs_finish()) { glFlush(); glFinish(); }
        auto t1 = Clock::now();
        vector<float> out = read_logits_full(tiles);
        if (rt) fprintf(stderr, "RBT finish %.3f ms readback %.3f ms (%zu floats)\n",
                        elapsed_ms(t0, t1), elapsed_ms(t1, Clock::now()), out.size());
        return out;
    }

bool GPU::logits_argmax_ok(const vector<Mat>& tiles) const{
        // Opt-in (GPU_ARGMAX=1); the full read-back is the default because
        // the reduction is a measured net LOSS on VC4.  READBACK_TIME (Pi
        // Zero, 2026-11-15) puts the hybrid's read-back at ~6 ms a token
        // (258 KiB), not the ~57 ms the PASS_DUP replay suggested -- that
        // figure was the finish waiting on the replayed logits draw, which
        // is unavoidable GPU work, not transfer.  The chunk pass re-reads
        // the whole logits texture on-GPU (~the same fetches) and adds three
        // pipeline drains, ~14 ms in all, so it cannot beat the 6 ms
        // read-back; the interleaved A/B agrees (hybrid ~+8 ms/token, dense
        // ~+2).  Kept as a byte-identical measurement arm -- it may pay
        // where glReadPixels is dearer -- and for the VC4 rolled-loop
        // finding its bring-up produced (docs/vc4-glsl-costs.md §6).
        // Only the packed q16 layout is reduced on the GPU: there the stored
        // bytes are the exact integers the CPU compares, so the winner is
        // byte-identical.  The unpacked (i16-weight / ACT8) layout keeps the
        // read-back even when the flag is set.
        const char* v = getenv("GPU_ARGMAX");
        if (!(v && strcmp(v, "1") == 0)) return false;
        return logits_pair_ok(tiles);
    }

int GPU::logits_argmax_from(const vector<Mat>& tiles, GLuint x){
        const int rows = (int)tiles.size();
        const bool rt = getenv("READBACK_TIME");
        auto t0 = Clock::now();
        auto lap = [&](const char* n) { if (rt) { fprintf(stderr, "RBT %s %.3f ms\n", n, elapsed_ms(t0, Clock::now())); t0 = Clock::now(); } };
        unsigned long long lc_idle = 0, lc_frag = 0, lc_exec = 0, lc_quads = 0;
        const bool lc_on = pc_available() && getenv("LOGITS_PERFCNT");
        if (lc_on) pc_stage_begin();
        logits_pass(tiles, x);
        // Same VC4 rule as logits_candidates_from: the reduction samples the
        // logits target as a texture, and the tile pipeline does not order
        // that sample after the FBO renders on its own.
        if (needs_finish()) { glFlush(); glFinish(); }
        if (lc_on) {
            pc_stage_end(lc_idle, lc_frag, lc_exec, lc_quads);
            const double to_ms = 1.0 / (pc_qpus() * (double)pc_mhz() * 1000.0);
            fprintf(stderr, "LOGITS gpu=%.3f exec=%.3f idle=%.3f quads=%llu\n",
                    (double)lc_frag * to_ms, (double)lc_exec * to_ms,
                    (double)lc_idle * to_ms, lc_quads);
        }
        lap("logits+finish");

        const int chunk = 16;
        int max_chunks = 0;
        for (auto& t : tiles) max_chunks = max(max_chunks, (t.rows + chunk - 1) / chunk);
        // Stage 1 reuses the proven top2_fast_packed kernel verbatim (two
        // ranks per chunk; only rank 0 is read afterwards).  A dedicated
        // single-rank text lost the index updates on VC4 -- see the comment
        // in shaders_attn.cpp -- so the reduction stays on the kernel the
        // candidate path already verifies on this board.
        const int cw = max_chunks * 2;
        if (argmax_chunk_.tex == 0 || argmax_chunk_.w != cw || argmax_chunk_.h != rows)
            argmax_chunk_ = make_target(cw, rows);
        // Clear to the sentinel texel (-32768, 65535): the short last tile's
        // trailing chunks keep it, so the group reduce needs no break (a
        // runtime-bound break kept the loop rolled, and VC4 loses the index
        // updates of a rolled loop -- see top1_group).  The sentinel is the
        // smallest q16 value and can never win.  Stage 1 draws only each
        // tile's real chunks (the TOPK_FAST viewport), so nothing overwrites
        // the trailing sentinels; re-cleared every token because the target
        // is reused and a fresh one is zero-filled, not sentinel.
        bind_fbo(argmax_chunk_.fbo);
        glClearColor(128.0f / 255.0f, 0.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < rows; ++i) {
            GLuint p = x_top2_fast_packed(tiles[i].rows, chunk);
            bind_fbo(argmax_chunk_.fbo);
            glViewport(0, i, ((tiles[i].rows + chunk - 1) / chunk) * 2, 1);
            use_program(p);
            bind_tex(0, logits_target_.tex);
            set_sampler(p, "L", 0);
            glUniform1f(uloc(p, "tr"), (float)i);
            glUniform1f(uloc(p, "nr"), (float)rows);
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        lap("stage1");

        // ARGMAX_DUMP=1: print the reduction chain next to the CPU argmax of
        // the same logits, so a wrong winner can be located stage by stage.
        const bool dbg = getenv("ARGMAX_DUMP");
        if (dbg) {
            vector<uint8_t> c = read_bytes(argmax_chunk_, cw, rows);
            for (int r = 0; r < min(rows, 2); ++r) {
                fprintf(stderr, "ARGMAX chunk row %d:", r);
                for (int j = 0; j < min(max_chunks, 6); ++j) {
                    const size_t o = ((size_t)r * cw + j * 2) * 4;
                    int v = (int)c[o] * 256 + c[o + 1]; if (v >= 32768) v -= 65536;
                    int idx = (int)c[o + 2] * 256 + c[o + 3];
                    fprintf(stderr, " (%d,%d)", v, idx);
                }
                fprintf(stderr, "\n");
            }
        }

        // Stage 2a: one fragment per (tile, group of 16 chunk winners),
        // then stage 2b: one fragment per tile.  Both loops stay at the
        // VC4-proven length (see top1_group); a single 96-iteration reduce
        // lost the index updates on this board.
        const int groups = (max_chunks + 15) / 16;
        if (argmax_group_.tex == 0 || argmax_group_.w != groups || argmax_group_.h != rows)
            argmax_group_ = make_target(groups, rows);
        {
            GLuint p = x_top1_group(max_chunks);
            bind_fbo(argmax_group_.fbo);
            glViewport(0, 0, groups, rows);
            use_program(p);
            bind_tex(0, argmax_chunk_.tex);
            set_sampler(p, "L", 0);
            glUniform1f(uloc(p, "nr"), (float)rows);
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        lap("group");
        if (argmax_tile_.tex == 0 || argmax_tile_.w != 1 || argmax_tile_.h != rows)
            argmax_tile_ = make_target(1, rows);
        {
            GLuint p = x_top1_final(groups);
            bind_fbo(argmax_tile_.fbo);
            glViewport(0, 0, 1, rows);
            use_program(p);
            bind_tex(0, argmax_group_.tex);
            set_sampler(p, "L", 0);
            glUniform1f(uloc(p, "nr"), (float)rows);
            setup_attrib(p);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        lap("final");

        // Read back one texel per tile (rows * 4 bytes) and keep the first
        // maximum across tiles -- the same tie rule as max_element, applied
        // at every level of the reduction.
        vector<uint8_t> raw = read_bytes(argmax_tile_, 1, rows);
        float best = -1e30f;
        int token = -1;
        for (int r = 0; r < rows; ++r) {
            const size_t o = (size_t)r * 4;
            int value = (int)raw[o] * 256 + raw[o + 1];
            if (value >= 32768) value -= 65536;
            int idx = (int)raw[o + 2] * 256 + raw[o + 3];
            if (dbg) fprintf(stderr, "ARGMAX tile %d: (%d,%d)%s\n", r, value, idx,
                             idx >= tiles[r].rows ? " INVALID" : "");
            if (idx >= tiles[r].rows) continue;
            float v = (float)value / ACT_SCALE;
            if (v > best) { best = v; token = r * kLogitsGrid + idx; }
        }
        lap("readback");
        if (dbg) {
            vector<float> lg = read_logits_full(tiles);
            int cpu = -1; float cb = -1e30f;
            for (size_t i = 0; i < lg.size(); ++i)
                if (lg[i] > cb) { cb = lg[i]; cpu = (int)i; }
            fprintf(stderr, "ARGMAX gpu=%d cpu=%d %s\n", token, cpu,
                    token == cpu ? "MATCH" : "MISMATCH");
            // Per-chunk winners of tile 0, GPU against CPU, to localise a
            // wrong index to the stage that produced it.
            vector<uint8_t> c = read_bytes(argmax_chunk_, cw, rows);
            const int M = tiles[0].rows;
            for (int ch = 0; ch < max_chunks; ++ch) {
                const size_t o = (size_t)ch * 2 * 4;
                int gv = (int)c[o] * 256 + c[o + 1]; if (gv >= 32768) gv -= 65536;
                int gi = (int)c[o + 2] * 256 + c[o + 3];
                float bv = -1e30f; int bi = -1;
                for (int k = 0; k < chunk; ++k) {
                    int idx = ch * chunk + k; if (idx >= M) break;
                    float v = lg[(size_t)idx];
                    if (v > bv) { bv = v; bi = idx; }
                }
                int cv = (int)(bv * ACT_SCALE);
                if (gv != cv || gi != bi)
                    fprintf(stderr, "ARGMAX ch%02d gpu=(%d,%d) cpu=(%d,%d) DIFF\n",
                            ch, gv, gi, cv, bi);
            }
        }
        return token;
    }

void GPU::logits_pass_only(const vector<Mat>& tiles, GLuint x){
        logits_pass(tiles, x);
    }

GLuint GPU::x_q8(int rows, int cols, bool wrow){
        return x_prog("xq8_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_t3(const Mat& m, bool wrow){
        // The tile shape is part of the program key: a 4096-row body folds into
        // two tiles of 2048 and the same (rows, cols) drawn as one tile is a
        // different shader.
        // A Mat that has not been through the uploader has no tile shape yet;
        // one tile is the fallback and is what every fitting body uses.
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : m.rows;
        const int tiles = m.t3_tile_h > 0 ? m.t3_tiles() : 1;
        if (getenv("T3_GEOM"))
                fprintf(stderr, "T3GEOM mv rows=%d cols=%d tile_h=%d th=%d tiles=%d\n",
                        m.rows, m.cols, m.t3_tile_h, th, tiles);
        return x_prog("xt3_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(act_fmt_) + "_" + to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3(m.rows, m.cols, act_fmt_, wrow, th, tiles));
    }

GLuint GPU::x_t3_add(const Mat& m, XFormat xf, XFormat rf, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : m.rows;
        const int tiles = m.t3_tile_h > 0 ? m.t3_tiles() : 1;
        // `xf` is the format of the X texture, which is NOT always act_fmt_: a
        // w2 add reads the FFN hidden stream (FFN_HID), and the two axes differ
        // as soon as ACT=i8 leaves the hidden on q16.  The q8 selectors have
        // always taken xf for this reason; the t3 ones hardcoded act_fmt_ and
        // only worked while the two formats coincided (found 2026-11-21 with
        // ACT=i8 on the t3 body: layer 0's FFN was decoded as I8_4).
        return x_prog("xt3a_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(xf) + "_" + x_format_name(rf) + "_" +
                      to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_add(m.rows, m.cols, xf, rf, wrow, th, tiles));
    }

GLuint GPU::x_q6(int rows, int cols, bool wrow){
        return x_prog("xq6_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(act_fmt_) + (wrow ? "_wr" : ""),
                      matvec_i6(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_q8_pair(int rows, int cols, bool wrow){
        return x_prog("xq8p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_q8_pair_swiglu(int rows, int cols, bool wrow, int hs){
        // Inline I16_2 decode is the default; FFN_SW_INLINE=0 restores the
        // generic X4Q form in the same binary, so both can be A/B-ed
        // interleaved on one binary.
        const char* v = getenv("FFN_SW_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        return x_prog("xq8ps_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(act_fmt_) + (inl ? "" : "_gen") + (wrow ? "_wr" : "") +
                      (hs ? "_h" + to_string(hs) : ""),
                      matvec_q8_pair_swiglu(rows, cols, act_fmt_, inl, wrow, hs));
    }

GLuint GPU::x_q8_gate2(int rows, int cols){
        return x_prog("xq8g2_" + to_string(rows) + "_" + to_string(cols),
                      matvec_q8_gate2(rows, cols));
    }

GLuint GPU::x_q8_p2(int rows, int cols, bool wrow){
        return x_prog("xq8p2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      matvec_q8_p2(rows, cols, wrow));
    }

void GPU::matvec_p2(const Mat& m, GLuint xp2, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        GLuint p = x_q8_p2(m.rows, m.cols, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

GLuint GPU::x_rms_stats2(){
        return x_prog("xrms2s_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_),
                      ::rms_stats2(att_dim_, res_fmt_));
    }

GLuint GPU::x_rms_norm_pair2(bool use_eps_uniform){
        return x_prog("xrms2n_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) +
                          (use_eps_uniform ? "_e" : ""),
                      ::rms_norm_pair2(att_dim_, res_fmt_, use_eps_uniform));
    }

GLuint GPU::x_q8_gate2_p2(int rows, int cols, bool wrow){
        return x_prog("xq8g2p2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      ::matvec_q8_gate2_p2(rows, cols, wrow));
    }

GLuint GPU::x_q8_p2_add_p2(int rows, int cols, bool wrow){
        return x_prog("xq8p2a2_" + to_string(rows) + "_" + to_string(cols) + (wrow ? "_wr" : ""),
                      ::matvec_q8_p2_add_p2(rows, cols, wrow));
    }

GLuint GPU::x_q8_p2_add(int rows, int cols, bool wrow){
        return x_prog("xq8p2a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(res_fmt_) + (wrow ? "_wr" : ""),
                      ::matvec_q8_p2_add(rows, cols, res_fmt_, wrow));
    }

GLuint GPU::x_p2_split(int rows){
        return x_prog("xp2sp_" + to_string(rows) + "_" + x_format_name(res_fmt_),
                      ::p2_split(rows, res_fmt_));
    }

void GPU::matvec_p2_add_p2(const Mat& m, GLuint xp2, GLuint rp2, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        GLuint p = x_q8_p2_add_p2(m.rows, m.cols, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        bind_tex(2, rp2); set_sampler(p, "RA", 2);
        glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(4, m.row_scale_tex); set_sampler(p, "WS", 4); }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

void GPU::matvec_p2_add(const Mat& m, GLuint xp2, GLuint ra, GLuint rb, Target& out){
        if (out.tex == 0 || out.w != m.rows) out = make_target(m.rows, 1);
        const bool wr = m.has_row_scale();
        GLuint p = x_q8_p2_add(m.rows, m.cols, wr);
        bind_fbo(out.fbo);
        use_program(p);
        bind_tex(0, m.tex); set_sampler(p, "W", 0);
        bind_tex(1, xp2); set_sampler(p, "X", 1);
        bind_tex(2, ra); set_sampler(p, "RA", 2);
        bind_tex(3, rb); set_sampler(p, "RB", 3);
        glUniform1f(uloc(p, "ws"), m.scale);
        if (wr) { bind_tex(4, m.row_scale_tex); set_sampler(p, "WS", 4); }
        setup_attrib(p);
        for (int x0 = 0; x0 < m.rows; x0 += kDrawCap) {
            const int w = min(kDrawCap, m.rows - x0);
            glViewport(x0, 0, w, 1);
            glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
        }
        if (ffn_flush_wanted(layer_fast_finish_)) glFlush();
        else layer_finish("NF_FFN");
        ffn_sync();
    }

GLuint GPU::x_q8_pair_fused(int rows, int cols, bool wrow){
        return x_prog("xq8pf_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair_fused(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_i4_pair(int rows, int cols){
        const string key = "xi4p_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_);
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key;
        GLuint p = link_program(matvec_i4_pair(rows, cols, act_fmt_));
        programs_[key] = p;
        return p;
    }

GLuint GPU::x_i4_pair_swiglu(int rows, int cols, int hs, bool fmt4){
        const bool inl = q4_inline("gate");
        const string tail = string(inl ? "" : "_gen") + (hs ? "_h" + to_string(hs) : "");
        if (fmt4)
                return x_prog("xi4ps4_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + tail,
                              matvec_i4_pair_swiglu_4x4(rows, cols, act_fmt_, inl, hs));
        return x_prog("xi4ps_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) + tail,
                      matvec_i4_pair_swiglu(rows, cols, act_fmt_, inl, hs));
    }

GLuint GPU::x_t3_pair(const Mat& m, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : (m.rows + 1) / 2;
        const int tiles = m.t3_tiles();
        return x_prog("xt3p_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(act_fmt_) + "_" + to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_pair(m.rows, m.cols, act_fmt_, wrow, th, tiles));
    }

GLuint GPU::x_t3_add_pair(const Mat& m, XFormat xf, XFormat rf, bool wrow){
        const int th = m.t3_tile_h > 0 ? m.t3_tile_h : (m.rows + 1) / 2;
        const int tiles = m.t3_tiles();
        return x_prog("xt3ap_" + to_string(m.rows) + "_" + to_string(m.cols) + "_" +
                      x_format_name(xf) + "_" + x_format_name(rf) + "_" +
                      to_string(th) + (wrow ? "_wr" : ""),
                      matvec_t3_add_pair(m.rows, m.cols, xf, rf, wrow, th, tiles));
    }

GLuint GPU::x_q8_add(int rows, int cols, XFormat xf, bool wrow){
        return x_prog("xq8a_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(xf) + "_" +
                      x_format_name(res_fmt_) + (wrow ? "_wr" : ""),
                      matvec_q8_add(rows, cols, xf, res_fmt_, wrow));
    }

GLuint GPU::x_q8_add_pair(int rows, int cols, XFormat xf, bool wrow){
        // The inline I16_2 decode is the default; Q8AP_INLINE=0 restores the
        // generic X4Q form in the same binary, so the two can be A/B-ed
        // interleaved without a second build.
        const char* v = getenv("Q8AP_INLINE");
        const bool inl = !(v && strcmp(v, "0") == 0);
        // The residual's format belongs in the key as well as in the text: a
        // bf16 residual takes the generic path even when the fast I16_2 one is
        // available, and the two are different programs.
        return x_prog("xq8ap_" + to_string(rows) + "_" + to_string(cols) + "_" +
                      x_format_name(xf) + (inl ? "" : "_gen") + (wrow ? "_wr" : "") +
                      (res_fmt_ == XFormat::I16_2 ? std::string() : std::string("_") + x_format_name(res_fmt_)),
                      matvec_q8_add_pair(rows, cols, xf, res_fmt_, inl, wrow));
    }

GLuint GPU::x_i4_add_pair(int rows, int cols, XFormat xf, bool fmt4){
        const bool inl = q4_inline("w2");
        // The residual's format is part of the key: this kernel writes two rows
        // into one texel, so a bf16 residual and a q16 one are different texts.
        const string key = string(fmt4 ? "xi4ap4_" : "xi4ap_") + to_string(rows) + "_" +
                           to_string(cols) + "_" + x_format_name(xf) + "_" +
                           x_format_name(res_fmt_) + (inl ? "" : "_gen");
        auto it = programs_.find(key);
        if (it != programs_.end()) return it->second;
        g_prog_key = key;
        GLuint p = link_program(fmt4 ? matvec_i4_add_pair_4x4(rows, cols, xf, res_fmt_, inl)
                                     : matvec_i4_add_pair(rows, cols, xf, res_fmt_, inl));
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

GLuint GPU::x_i4(int rows, int cols, bool fmt4){
        const bool inl = q4_inline("qkv");
        return x_prog(string(fmt4 ? "xi44_" : "xi4_") + to_string(rows) + "_" + to_string(cols) + "_" +
                          x_format_name(act_fmt_) + (inl ? "" : "_gen"),
                      fmt4 ? matvec_i4_4x4(rows, cols, act_fmt_, inl)
                           : matvec_i4(rows, cols, act_fmt_, inl));
    }

GLuint GPU::x_i4_add(int rows, int cols, XFormat xf, bool fmt4){
        return x_prog(string(fmt4 ? "xi4a4_" : "xi4a_") + to_string(rows) + "_" + to_string(cols) + "_" +
                          x_format_name(xf) + "_" + x_format_name(res_fmt_),
                      fmt4 ? matvec_i4_add_4x4(rows, cols, xf, res_fmt_)
                           : matvec_i4_add(rows, cols, xf, res_fmt_));
    }

GLuint GPU::x_swiglu_pair(int hidden, int hs){
        return x_prog("xswp_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      swiglu_pair(hidden, ffn_fmt_, hs));
    }

GLuint GPU::x_swiglu_2row(int hidden, int hs){
        return x_prog("xsw2_" + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      ::swiglu_2row(hidden, ffn_fmt_, hs));
    }

GLuint GPU::x_swiglu(int hidden, bool fused, int hs){
        return x_prog(string("xsw_") + (fused ? "f_" : "s_") + to_string(hidden) + "_" + x_format_name(ffn_fmt_) +
                          (hs ? "_h" + to_string(hs) : ""),
                      swiglu(hidden, fused, ffn_fmt_, hs));
    }

GLuint GPU::x_rms_split(){
        return x_prog("xrmss_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_),
                      rms_stats_split(att_dim_, res_fmt_));
    }

GLuint GPU::x_rms_reduce(){ return x_prog("xrmsr", rms_stats_reduce()); }

GLuint GPU::x_rms_norm(bool use_eps_uniform){
        const bool fused = rms_fused() && act_fmt_ != XFormat::I8_4;
        return x_prog("xrmsn_" + to_string(att_dim_) + "_" + x_format_name(res_fmt_) + "_" +
                          x_format_name(act_fmt_) + (fused ? "_f" : "") +
                          (use_eps_uniform ? "_e" : ""),
                      rms_norm(att_dim_, res_fmt_, act_fmt_, fused, use_eps_uniform));
    }

// Long context: one program a segment.  The key carries `segk` because the
// program text does (`SEGK`), and the program cache validates the text hash on
// every hit -- a shared key with two texts is the swap bug this cache already
// paid for once (docs/swap-drift.md).  segk == 0 keeps the shipped key, so the
// digest table and the cache behaviour do not move for a one-segment session.
static string seg_suffix(int k) { return k > 0 ? "_s" + to_string(k) : string(); }

GLuint GPU::x_v(int segk, bool accumulate){
        return x_prog("xv_" + to_string(att_dim_) + "_" + x_format_name(act_fmt_) +
                          seg_suffix(segk) + (accumulate ? "_add" : ""),
                      ::att_v(att_cfg(segk, accumulate), act_fmt_));
    }

GLuint GPU::x_v_fold(){
        return x_prog("xvf_" + to_string(att_dim_) + "_" + x_format_name(act_fmt_) +
                          "_" + to_string(att_nseg_),
                      att_v_fold_shader(att_cfg(), act_fmt_, att_nseg_));
    }

GLuint GPU::x_kv_rope(int segk){ return x_prog("xkv" + seg_suffix(segk), att_kv_rope(att_cfg(segk))); }

GLuint GPU::x_vscale(){ return x_prog("xvscale", ::att_vscale(att_cfg())); }

GLuint GPU::x_qnorm(){ return x_prog("xqn", att_qnorm_scale(att_cfg())); }

GLuint GPU::x_qrot(int segk){ return x_prog("xqr" + seg_suffix(segk), att_qrot(att_cfg(segk))); }

GLuint GPU::x_score(int segk){ return x_prog("xsc" + seg_suffix(segk), att_score(att_cfg(segk))); }

GLuint GPU::x_stats(int segk){ return x_prog("xst" + seg_suffix(segk), ::att_stats(att_cfg(segk))); }

// The qbatch variants bake n into the shader text (QB) and take the position's
// row offset qrow0 as a uniform; the serial getters above stay qbatch = 1.
GLuint GPU::x_score_batch(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xscb_" + to_string(n), att_score(c));
}

GLuint GPU::x_stats_row(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xstr_" + to_string(n), ::att_stats(c));
}

GLuint GPU::x_v_row(int n){
        AttnConfig c = att_cfg(); c.qbatch = n;
        return x_prog("xvr_" + to_string(n) + "_" + x_format_name(act_fmt_),
                      ::att_v(c, act_fmt_));
}

GLuint GPU::x_prob(){ return x_prog("xpr", att_prob(att_cfg())); }

GLuint GPU::x_top2_fast(int rows, int chunk){
        return x_prog("xt2f_" + to_string(rows) + "_" + to_string(chunk), top2_fast(rows, chunk));
    }

GLuint GPU::x_top2(int rows, int chunk){
        return x_prog("xt2_" + to_string(rows) + "_" + to_string(chunk), top2(rows, chunk));
    }

GLuint GPU::x_q8_pair_logits(int rows, int cols, bool wrow){
        return x_prog("xq8pl_" + to_string(rows) + "_" + to_string(cols) + "_" + x_format_name(act_fmt_) +
                      (wrow ? "_wr" : ""),
                      matvec_q8_pair_logits(rows, cols, act_fmt_, wrow));
    }

GLuint GPU::x_top2_fast_packed(int rows, int chunk){
        return x_prog("xt2fp_" + to_string(rows) + "_" + to_string(chunk), top2_fast_packed(rows, chunk));
    }

GLuint GPU::x_top2_packed(int rows, int chunk){
        return x_prog("xt2p_" + to_string(rows) + "_" + to_string(chunk), top2_packed(rows, chunk));
    }

GLuint GPU::x_top1_group(int n){
        return x_prog("xt1g_" + to_string(n), top1_group(n));
    }

GLuint GPU::x_top1_final(int groups){
        return x_prog("xt1f_" + to_string(groups), top1_final(groups));
    }
