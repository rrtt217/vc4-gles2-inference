// The vocabulary projection: tiles -> logits, candidates and the greedy argmax.
//
// The weight kernel is q8's paired logits builder (gpu_select.cpp); this file is
// the pass, the read-back and the top-k reductions that follow it.
#include "gpu.h"
#include "gpu_internal.h"
#include "logits_grid.h"
#include "xformat.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

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


vector<float> GPU::read_logits_full(const vector<Mat>& tiles){
        const int rows = (int)tiles.size();
        vector<float> out;
        out.reserve((size_t)rows * kLogitsGrid);
        if (logits_packed_) {
            const int w = logits_target_.w;
            bind_fbo(logits_target_.fbo);
            vector<uint8_t> b((size_t)w * rows * 4);
            note_sync();
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
        vector<int32_t> raw = read_i32(logits_target_, kLogitsGrid, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < tiles[i].rows; ++j) {
                int v = raw[(size_t)i * kLogitsGrid + j];
                if (v >= 32768) v -= 65536;
                out.push_back((float)v / ACT_SCALE);
            }
        }
        return out;
    }

