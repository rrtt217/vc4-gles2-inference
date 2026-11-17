// The dump path: the engine's own numbers, written out so an offline oracle can
// check them (scripts/check_att.py, dump_stats.py, docs/test-hooks.md).
//
// Split out of gpu.cpp (2026-11-14, P2-1 of the 2026-11-13 audit).  It is leaf code:
// it reads targets back and hashes rows, and nothing calls it unless a dump switch
// is on, so it can live apart without another seam.

#include "gpu.h"
#include "gpu_internal.h"

#include <cmath>
#include <cstring>
#include "xformat.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static unsigned long long fnv_row(const uint8_t* p, size_t n){
        unsigned long long h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h;
}

static void dump_row(const uint8_t* p, int w, const char* name, int n){
        if (n > w) n = w;
        fprintf(stderr, "DUMP %s w=%d h=%016llx", name, w, fnv_row(p, (size_t)w * 4));
        for (int i = 0; i < n; i++) fprintf(stderr, " %02x%02x%02x%02x", p[i*4], p[i*4+1], p[i*4+2], p[i*4+3]);
        fprintf(stderr, "\n");
}

void GPU::dump_target(const Target& t, const char* name, int n){
        if (t.tex == 0) return;
        vector<uint8_t> b((size_t)t.w * 4);
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, t.w, 1, GL_RGBA, GL_UNSIGNED_BYTE, &b[0]);
        gl_state_forget_bindings();
        dump_row(&b[0], t.w, name, n);
}

// The CPU path has no target to read back: the activation is a plain float
// vector.  Pack it exactly as a GPU producer would (x_pack applies the same
// q16 quantisation and the same layout) and print it with the same printer.
void GPU::dump_cpu_row(const vector<float>& v, XFormat f, const char* name, int n){
        // DUMP_F32 prints the raw IEEE bits instead of the q16 texels the GPU
        // would hold.  The q16 form is the right thing to compare between two
        // engine paths, but 1/256 is coarser than the interesting differences
        // in this model's early layers (activations of order 0.01), so a
        // numpy-vs-engine bisect needs the f32 form.
        if (getenv("DUMP_F32")) {
                static const bool on = true; (void)on;
                fprintf(stderr, "DUMP %s w=%d h=0", name, (int)v.size());
                for (float x : v) {
                        uint32_t u; memcpy(&u, &x, 4);
                        fprintf(stderr, " %08x", u);
                }
                fprintf(stderr, "\n");
                return;
        }
        const int w = x_texels((int)v.size(), f);
        vector<uint8_t> b = x_pack(v, f);
        if ((int)b.size() < w * 4) b.resize((size_t)w * 4, 0);
        dump_row(&b[0], w, name, n);
}

vector<uint8_t> read_target_all(const Target& t){
        vector<uint8_t> b((size_t)t.w * t.h * 4);
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, t.w, t.h, GL_RGBA, GL_UNSIGNED_BYTE, &b[0]);
        gl_state_forget_bindings();
        return b;
}

vector<uint8_t> read_target_row(const Target& t, int row){
        vector<uint8_t> b((size_t)t.w * 4);
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, row, t.w, 1, GL_RGBA, GL_UNSIGNED_BYTE, &b[0]);
        gl_state_forget_bindings();
        return b;
}

int q16_at(const vector<uint8_t>& b, int texel, int comp){
        const size_t o = (size_t)texel * 4 + (size_t)(comp ? 2 : 0);
        int v = (int)b[o] * 256 + (int)b[o + 1];
        if (v >= 32768) v -= 65536;
        return v;
}

void GPU::dump_attention(int layer, int pos){
        if (qrot_target_.tex == 0 || att_scores_.tex == 0 || att_stats_.tex == 0) return;
        // One meta line pins the shape and the grids, so the checker needs no
        // assumptions: value = int/256 for q and att_out, int/KVSCALE for K and
        // V, int/1024 for scores and the stats max, int/32 for the stats sum.
        static bool meta = true;
        if (meta) {
                fprintf(stderr, "ATTD meta %d %d %d %d %d %g %d\n", att_heads_, att_kv_heads_,
                        att_head_dim_, att_dim_, att_kv_dim_, (double)att_kv_scale_, att_seq_);
                meta = false;
        }
        {
                const vector<uint8_t> b = read_target_row(qrot_target_, 0);
                fprintf(stderr, "ATTD qrot %d %d %d", layer, pos, att_dim_);
                for (int i = 0; i < att_dim_; ++i) fprintf(stderr, " %d", q16_at(b, i / 2, i & 1));
                fprintf(stderr, "\n");
        }
        // Every K/V row the score and V passes read: rows 0..pos.
        for (int t = 0; t <= pos && t < att_seq_; ++t) {
                if (layer >= (int)kv_cache_.size()) break;
                const bool f32kv = att_kv_fmt_ == 2;
                if (att_kv_bits_ != 16 && !f32kv) break;
                const vector<uint8_t> b = read_target_row(kv_cache_[layer], t);
                auto at = [&](int texel) {
                        if (!f32kv) return 0;
                        const uint8_t* p = b.data() + (size_t)texel * 4;
                        uint32_t u = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                                     ((uint32_t)p[2] << 8) | (uint32_t)p[3];
                        float v = 0.0f;
                        memcpy(&v, &u, 4);
                        return (int)lrintf(v * att_kv_scale_);
                };
                fprintf(stderr, "ATTD kv %d %d %d %d", layer, pos, t, att_kv_dim_);
                for (int i = 0; i < att_kv_dim_; ++i)
                        fprintf(stderr, " %d", f32kv ? at(i) : q16_at(b, i / 2, i & 1));
                fprintf(stderr, "\n");
                fprintf(stderr, "ATTD kvv %d %d %d %d", layer, pos, t, att_kv_dim_);
                for (int i = 0; i < att_kv_dim_; ++i)
                        fprintf(stderr, " %d", f32kv ? at(att_kv_dim_ + i) : q16_at(b, att_kv_dim_ / 2 + i / 2, i & 1));
                fprintf(stderr, "\n");
        }
        {
                const vector<uint8_t> b = read_target_all(att_scores_);
                const int w = att_scores_.w;
                for (int h = 0; h < att_heads_; ++h) {
                        fprintf(stderr, "ATTD score %d %d %d", layer, pos, h);
                        for (int t = 0; t <= pos; ++t) fprintf(stderr, " %d", q16_at(b, h * w + t / 2, t & 1));
                        fprintf(stderr, "\n");
                }
        }
        {
                const vector<uint8_t> b = read_target_all(att_stats_);
                fprintf(stderr, "ATTD statsmax %d %d %d", layer, pos, att_heads_);
                for (int h = 0; h < att_heads_; ++h) fprintf(stderr, " %d", q16_at(b, h, 0));
                fprintf(stderr, "\n");
                fprintf(stderr, "ATTD statssum %d %d %d", layer, pos, att_heads_);
                for (int h = 0; h < att_heads_; ++h) fprintf(stderr, " %d", q16_at(b, h, 1));
                fprintf(stderr, "\n");
        }
        if (att_out_.tex) {
                const vector<uint8_t> b = read_target_row(att_out_, 0);
                fprintf(stderr, "ATTD attout %d %d %d", layer, pos, att_dim_);
                if (act_fmt_ == XFormat::F32) {
                        // One value a texel: print it on the q16 grid the checker
                        // compares in, from the float the kernel actually stored
                        // (the 1/256 rounding is the checker's, not the engine's).
                        for (int i = 0; i < att_dim_; ++i) {
                                const uint8_t* p = b.data() + (size_t)i * 4;
                                uint32_t u = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                                             ((uint32_t)p[2] << 8) | (uint32_t)p[3];
                                float v = 0.0f;
                                memcpy(&v, &u, 4);
                                fprintf(stderr, " %d", (int)lrintf(v * 256.0f));
                        }
                } else {
                        for (int i = 0; i < att_dim_; ++i) fprintf(stderr, " %d", q16_at(b, i / 2, i & 1));
                }
                fprintf(stderr, "\n");
        }
}