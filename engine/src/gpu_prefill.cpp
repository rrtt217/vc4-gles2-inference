// The batched matvec: RWKV-7's chunked prefill (P1, docs/rwkv7-plan.zh-CN.md).
//
// A prefill is the same layer loop over N tokens, and the only thing that makes
// it 512 serial passes is that every projection is drawn and read back once per
// token.  The projections are independent across tokens (only the elementwise
// chain and the WKV7 recurrence are sequential), so N inputs can share one draw
// per weight -- matvec_q8_batch reads its input row from gl_FragCoord.y -- and
// the whole block comes back with one glReadPixels.
//
// Two things make the result bit-identical to the serial path rather than
// merely close:
//
//  * the kernel text is matvec_q8's, with the row substituted: same weight
//    fetch, same decode, same loop order, same q24 encode, same scale uniform;
//  * the serial path's per-input 2^k normalisation (Session::rwkv_mv) is kept
//    per input.  k differs between inputs of one batch, so the inputs are
//    grouped by k: one draw per distinct k, each with the matrix's scale
//    pre-multiplied by its own 2^k -- exactly the uniform the serial path binds.
//    (A single k for the whole batch would coarsen the input grid of the rows
//    that needed a smaller k, and the q24 output grid is one 256th of a unit,
//    so that difference is visible in the last bits and would compound through
//    12 layers.)
//
// What this does *not* change: the MAC count, the per-input host work (packing,
// the 2^k scale, the read-back decode) and every layer's arithmetic.  Only the
// number of draws and host syncs drops, by the batch size.

#include "gpu.h"
#include "gpu_internal.h"
#include "shaders.h"
#include "xformat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

GLuint GPU::x_q8_batch(int rows, int cols, bool wrow){
        return x_prog("xq8b_" + std::to_string(rows) + "_" + std::to_string(cols) + "_" +
                          x_format_name(act_fmt_) + (wrow ? "_wr" : ""),
                      matvec_q8_batch(rows, cols, act_fmt_, wrow));
}

// One input row's own 2^k, the rule Session::rwkv_mv uses: put the row's peak in
// 32..63 on the q16 grid.  k <= 0 means "no normalisation" (the serial path
// calls matvec() unscaled in that case, and so does the batch: k = 0 below).
static int rwkv_row_k(const std::vector<float>& x){
        float mx = 0.0f;
        for (float v : x) mx = std::max(mx, std::fabs(v));
        if (!(mx > 0.0f)) return 0;
        int e = 0;
        std::frexp(mx, &e);              // mx = f * 2^e, f in [0.5, 1)
        const int k = e - 6;
        return k > 0 ? k : 0;
}

vector<vector<vector<float>>> GPU::matvec_batch_group(const vector<MvBatch>& groups){
        const size_t G = groups.size();
        vector<vector<vector<float>>> out(G);
        if (G == 0) return out;
        size_t N = 0;
        int rows_max = 0, in_w = 0;
        bool all_int8 = true;
        for (const MvBatch& g : groups) {
                // No batched kernel for q4/i16.  prefill_batch_ok() refuses those
                // containers, so this is the belt-and-braces arm: answer from the
                // serial path per input rather than from a kernel that cannot
                // decode the weights.
                if (g.m->type != WeightType::INT8) all_int8 = false;
                N = std::max(N, g.xs->size());
                rows_max = std::max(rows_max, g.m->rows);
                for (const vector<float>* x : *g.xs)
                        in_w = std::max(in_w, x_texels((int)x->size(), act_fmt_));
        }
        if (N == 0) return out;
        out.assign(G, vector<vector<float>>(N));
        if (!all_int8) {
                for (size_t i = 0; i < G; ++i)
                        for (size_t b = 0; b < groups[i].xs->size(); ++b)
                                out[i][b] = matvec(*groups[i].m, *(*groups[i].xs)[b]);
                return out;
        }

        // One block of N rows per matrix, in the same order in the input texture
        // and in the read-back target, so a fragment's y is its input's row
        // without a second uniform: matrix i's input b sits at row i*N + b.
        const int bh = (int)G * (int)N;
        // Both textures grow monotonically and are never shrunk: a per-call
        // make_target would put a texture in the scene ledger per matrix per
        // batch (392 of them for a 445-token prompt at n=16) and free none until
        // the model is released.  The row stride is therefore the texture's own
        // width (tin.w), which is what the kernel's xw uniform carries.
        Target& tin = rwkv_batch_in_;
        if (tin.tex == 0 || tin.w < in_w || tin.h < bh)
                tin = make_target(std::max(in_w, tin.w), std::max(bh, tin.h));
        Target& tout = rwkv_batch_out_;
        if (tout.tex == 0 || tout.w < rows_max || tout.h < bh)
                tout = make_target(std::max(rows_max, tout.w), std::max(bh, tout.h));
        const int tstride = tin.w;
        const size_t stride = (size_t)tstride * 4;
        vector<uint8_t> in((size_t)tstride * (size_t)bh * 4, 0);
        // Per matrix: the order its rows are packed in (sorted by k, so equal k
        // runs are contiguous and one viewport covers each group), and the k of
        // each texture row.
        vector<vector<int>> order(G);
        vector<int> krow((size_t)bh, 0);
        for (size_t i = 0; i < G; ++i) {
                const vector<const vector<float>*>& xs = *groups[i].xs;
                order[i].resize(xs.size());
                vector<int> ks(xs.size());
                for (size_t b = 0; b < xs.size(); ++b) {
                        order[i][b] = (int)b;
                        ks[b] = rwkv_row_k(*xs[b]);
                }
                std::stable_sort(order[i].begin(), order[i].end(),
                                 [&](int a, int b) { return ks[a] > ks[b]; });
                for (size_t b = 0; b < xs.size(); ++b) {
                        const int src = order[i][b];
                        const int k = ks[src];
                        krow[i * (size_t)N + b] = k;
                        const vector<float>& x = *xs[src];
                        vector<float> scaled(x.size());
                        const float inv = k > 0 ? std::ldexp(1.0f, -k) : 1.0f;
                        for (size_t j = 0; j < x.size(); ++j) scaled[j] = x[j] * inv;
                        vector<uint8_t> bytes = x_pack(scaled, act_fmt_);
                        const size_t row = i * N + b;
                        std::memcpy(in.data() + row * stride, bytes.data(),
                                    std::min(stride, bytes.size()));
                }
        }

        // The staging upload must not run with the texture attached to the bound
        // FBO: make_target() leaves its own framebuffer bound, so name the
        // output's before writing the input's.
        bind_fbo(tout.fbo);
        bind_tex(0, tin.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tstride, bh, GL_RGBA, GL_UNSIGNED_BYTE, in.data());
        if (needs_finish()) { glFlush(); glFinish(); }

        // VC4's draw cap: a single draw may cover at most kDrawCap *fragments*
        // (gpu_internal.h), so a batch of B rows shrinks the strip width to keep
        // width*B inside it.  That keeps the board's draw count per token the
        // serial path's -- which is the safe reading of the cap while
        // docs/performance.md's own question ("width or fragments?") is open on
        // the board -- and RWKV_PREFILL_STRIP overrides it for the experiment.
        static const char* strip_env = getenv("RWKV_PREFILL_STRIP");
        const int strip_cap = (strip_env && atoi(strip_env) > 0) ? atoi(strip_env) : kDrawCap;
        for (size_t i = 0; i < G; ++i) {
                const Mat& m = *groups[i].m;
                const vector<const vector<float>*>& xs = *groups[i].xs;
                if (xs.empty()) continue;
                const bool wrow = m.type == WeightType::INT8 && m.has_row_scale();
                const GLuint p = x_q8_batch(m.rows, m.cols, wrow);
                bind_fbo(tout.fbo);
                use_program(p);
                bind_tex(0, m.tex);
                set_sampler(p, "W", 0);
                bind_tex(1, tin.tex);
                set_sampler(p, "X", 1);
                if (wrow) { bind_tex(3, m.row_scale_tex); set_sampler(p, "WS", 3); }
                glUniform1f(uloc(p, "xh"), (float)bh);
                glUniform1f(uloc(p, "xw"), (float)tstride);
                glUniform1f(uloc(p, "ws"), m.scale);
                setup_attrib(p);
                int b = 0;
                while (b < (int)xs.size()) {
                        const int k = krow[i * (size_t)N + b];
                        int b1 = b + 1;
                        while (b1 < (int)xs.size() && krow[i * (size_t)N + b1] == k) ++b1;
                        // The serial path's Mat view: the same matrix with its
                        // scale pre-multiplied by this group's 2^k.  A power of
                        // two, so the multiply the kernel does is exact.
                        glUniform1f(uloc(p, "ws"), k > 0 ? std::ldexp(m.scale, k) : m.scale);
                        const int yn = (int)i * (int)N + b;
                        int strip = std::min(strip_cap, m.rows);
                        if (needs_finish())
                                strip = std::max(1, std::min(strip, strip_cap / (b1 - b)));
                        for (int x0 = 0; x0 < m.rows; x0 += std::max(1, strip)) {
                                const int w = std::min(std::max(1, strip), m.rows - x0);
                                glViewport(x0, yn, w, b1 - b);
                                glDrawArrays(GL_TRIANGLES, 0, 3); ++draws_;
                        }
                        b = b1;
                }
        }
        if (needs_finish()) { glFlush(); glFinish(); }
        const vector<int32_t> raw = read_i32(tout, rows_max, bh);
        // RWKV_PREFILL_CHECK: the batched block against the single-input path,
        // in the same process and the same run (the shape of the RWKV_GROUP_CHECK
        // bisect that found the grouped read-back's VC4 quirk).  A difference
        // here is a kernel/plumbing defect, not a model drift, so it names the
        // member, the row and the magnitude instead of leaving a PPL move.
        const bool self_check = env_flag("RWKV_PREFILL_CHECK");
        for (size_t i = 0; i < G; ++i) {
                const int rows = groups[i].m->rows;
                const vector<const vector<float>*>& xs = *groups[i].xs;
                for (size_t b = 0; b < xs.size(); ++b) {
                        const int dst = order[i][b];        // texture row -> input
                        vector<float>& o = out[i][dst];
                        o.resize((size_t)rows);
                        const size_t base = ((size_t)i * N + b) * (size_t)rows_max;
                        for (int j = 0; j < rows; ++j) {
                                int v = raw[base + (size_t)j];
                                if (v >= 32768) v -= 65536;
                                // No 2^k correction: the draw's ws uniform already
                                // carries this row's 2^k (the serial path folds it
                                // into the Mat view the same way), so the kernel's
                                // integer *is* the final q24 value.
                                o[(size_t)j] = (float)v / ACT_SCALE;
                        }
                }
        }
        if (self_check) {
                static int nprinted = 0;
                for (size_t i = 0; i < G; ++i) {
                        const vector<const vector<float>*>& xs = *groups[i].xs;
                        for (size_t b = 0; b < xs.size(); ++b) {
                                const int k = rwkv_row_k(*xs[b]);
                                vector<float> scaled(xs[b]->size());
                                const float inv = k > 0 ? std::ldexp(1.0f, -k) : 1.0f;
                                for (size_t j = 0; j < scaled.size(); ++j) scaled[j] = (*xs[b])[j] * inv;
                                Mat view = *groups[i].m;   // a view: borrowed bytes are not copied
                                if (k > 0) {
                                        if (view.type == WeightType::INT4) view.q4_param_scale = std::ldexp(view.q4_param_scale, k);
                                        else view.scale = std::ldexp(view.scale, k);
                                }
                                const vector<float> one = matvec(view, scaled);
                                const vector<float>& got = out[i][b];
                                int ndiff = 0, first = -1;
                                float mx = 0.0f;
                                for (size_t j = 0; j < one.size() && j < got.size(); ++j) {
                                        const float d = std::fabs(one[j] - got[j]);
                                        if (d > 0.0f) { if (first < 0) first = (int)j; ++ndiff; mx = std::max(mx, d); }
                                }
                                if (ndiff && nprinted++ < 12) {
                                        std::fprintf(stderr, "RWKV_PREFILL_CHECK member=%zu rows=%d cols=%d ndiff=%d/%d maxdiff=%g first=%d\n",
                                                     i, groups[i].m->rows, groups[i].m->cols, ndiff,
                                                     (int)one.size(), (double)mx, first);
                                        std::fprintf(stderr, "  values texx=%u k=%d x0=%g x1=%g | batch:",
                                                     groups[i].m->tex, k, (double)(*xs[b])[0],
                                                     (double)(*xs[b])[1]);
                                        for (int j = 0; j < 6; ++j) std::fprintf(stderr, " %d", raw[(size_t)i * N * rows_max + (size_t)b * rows_max + j]);
                                        std::fprintf(stderr, " | single:");
                                        // The single path's own q24 integers, so an offset in the
                                        // read-back is visible as a shift, not only as a diff.
                                        for (int j = 0; j < 6; ++j) std::fprintf(stderr, " %d", (int)lrintf(one[j] * ACT_SCALE));
                                        std::fprintf(stderr, "\n");
                                }
                        }
                }
        }
        return out;
}
