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
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <GLES2/gl2ext.h>

// GL_AMD_performance_monitor is the only timing facility VC4 exposes: there is
// no EXT_disjoint_timer_query on this driver (`eglinfo -s` lists no timer
// extension at all).  The entry points come from eglGetProcAddress; the
// constants are in gl2ext.h under their AMD names.
#ifndef GL_PERFMON_RESULT_AVAILABLE_AMD
#define GL_PERFMON_RESULT_AVAILABLE_AMD 0x8BC4
#define GL_PERFMON_RESULT_SIZE_AMD      0x8BC5
#define GL_PERFMON_RESULT_AMD           0x8BC6
#endif

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

// ---- hardware counters ----------------------------------------------------
//
// The four counters a profile pass needs.  `pc_stage_end` fills them in the
// order below; a stage line then reads as "cycles the QPU ran / cycles it sat
// idle / cycles a TMU fetch held it / texture quads it issued".  Divider is
// pc_qpus() * pc_qhz (12 * 300e6 on the board, measured -- see
// probes/perfcnt.h; the 4-QPU assumption is wrong for this driver's totals and
// makes a stage report 300% of its own wall time).
//
// The record layout is four words a counter: {group, counter, value, 0}.  This
// walked three at a time once, which reads the group id as the counter id.
namespace {
enum PcSlot { PC_IDLE = 0, PC_FRAG = 1, PC_EXEC = 2, PC_QUADS = 3, PC_N = 4 };

struct PcFns {
        void (GL_APIENTRY *gen)(GLsizei, GLuint*);
        void (GL_APIENTRY *del)(GLsizei, const GLuint*);
        void (GL_APIENTRY *sel)(GLuint, GLboolean, GLuint, GLsizei, const GLuint*);
        void (GL_APIENTRY *beg)(GLuint);
        void (GL_APIENTRY *end)(GLuint);
        void (GL_APIENTRY *data)(GLuint, GLenum, GLsizei, GLuint*, GLint*);
        void (GL_APIENTRY *groups)(GLint*, GLsizei, GLuint*);
        void (GL_APIENTRY *counters)(GLuint, GLint*, GLint*, GLsizei, GLuint*);
        void (GL_APIENTRY *cstr)(GLuint, GLuint, GLsizei, GLsizei*, GLchar*);
};
PcFns g_pc;
GLuint g_pc_ids[PC_N];
bool g_pc_ready = false;
}

static void* pc_sym(const char* n){
        return (void*)eglGetProcAddress(n);
}

void GPU::pc_init(){
        if (pc_ok_ || g_pc_ready) return;
        // Off unless asked: the monitor is whole-GPU and would otherwise count
        // another client's draws, and a probe run must not change because the
        // engine could read counters.
        const char* want = getenv("PERFCNT");
        if (!want || !want[0] || !strcmp(want, "0")) return;
        g_pc.gen = (decltype(g_pc.gen))pc_sym("glGenPerfMonitorsAMD");
        g_pc.del = (decltype(g_pc.del))pc_sym("glDeletePerfMonitorsAMD");
        g_pc.sel = (decltype(g_pc.sel))pc_sym("glSelectPerfMonitorCountersAMD");
        g_pc.beg = (decltype(g_pc.beg))pc_sym("glBeginPerfMonitorAMD");
        g_pc.end = (decltype(g_pc.end))pc_sym("glEndPerfMonitorAMD");
        g_pc.data = (decltype(g_pc.data))pc_sym("glGetPerfMonitorCounterDataAMD");
        g_pc.groups = (decltype(g_pc.groups))pc_sym("glGetPerfMonitorGroupsAMD");
        g_pc.counters = (decltype(g_pc.counters))pc_sym("glGetPerfMonitorCountersAMD");
        g_pc.cstr = (decltype(g_pc.cstr))pc_sym("glGetPerfMonitorCounterStringAMD");
        if (!g_pc.gen || !g_pc.sel || !g_pc.beg || !g_pc.end || !g_pc.data ||
            !g_pc.groups || !g_pc.counters || !g_pc.cstr) {
                fprintf(stderr, "PERFCNT: no GL_AMD_performance_monitor on this driver "
                                "(the counters are a VC4/Mesa feature; no timer query either)\n");
                return;
        }
        GLint ng = 0;
        g_pc.groups(&ng, 0, nullptr);
        if (ng <= 0) { fprintf(stderr, "PERFCNT: the driver reports no counter groups\n"); return; }
        GLuint gs[8] = {0};
        g_pc.groups(&ng, 8, gs);
        pc_group_ = gs[0];
        GLint n = 0, mx = 0;
        g_pc.counters(pc_group_, &n, &mx, 0, nullptr);
        vector<GLuint> ids((size_t)n);
        g_pc.counters(pc_group_, &n, &mx, n, ids.data());
        // Substring match, the names the driver uses: "idle-clk-cycles",
        // "clk-cycles-fragment-shading", "executing-valid-instr",
        // "text-quads-processed".
        static const char* const want_name[PC_N] = {
                "idle-clk-cycles", "clk-cycles-fragment-shading",
                "executing-valid-instr", "text-quads-processed" };
        bool found[PC_N] = {false, false, false, false};
        for (GLint i = 0; i < n; ++i) {
                char nm[128] = {0};
                GLsizei len = 0;
                g_pc.cstr(pc_group_, ids[(size_t)i], sizeof(nm) - 1, &len, nm);
                for (int k = 0; k < PC_N; ++k)
                        if (!found[k] && strstr(nm, want_name[k])) {
                                g_pc_ids[k] = ids[(size_t)i];
                                found[k] = true;
                        }
        }
        for (int k = 0; k < PC_N; ++k)
                if (!found[k]) { fprintf(stderr, "PERFCNT: counter \"%s\" is not exposed\n", want_name[k]); return; }
        g_pc.gen(1, &pc_monitor_);
        g_pc.sel(pc_monitor_, GL_TRUE, pc_group_, PC_N, g_pc_ids);
        g_pc_ready = true;
        pc_ok_ = true;
}

bool GPU::pc_available(){
        if (!pc_ok_) pc_init();
        return pc_ok_;
}

void GPU::pc_stage_begin(){
        if (!pc_ok_) return;
        // The mark before this one already finished the pipeline, so the region
        // measured is this stage's draws only.
        pc_armed_ = true;
        g_pc.beg(pc_monitor_);
}

bool GPU::pc_stage_end(unsigned long long& idle, unsigned long long& frag,
                       unsigned long long& exec, unsigned long long& quads){
        idle = frag = exec = quads = 0;
        if (!pc_ok_ || !pc_armed_) return false;
        pc_armed_ = false;
        g_pc.end(pc_monitor_);
        GLuint avail = 0;
        GLint written = 0;
        for (int i = 0; i < 200000 && !avail; ++i)
                g_pc.data(pc_monitor_, GL_PERFMON_RESULT_AVAILABLE_AMD, 4, &avail, &written);
        GLuint size = 0;
        g_pc.data(pc_monitor_, GL_PERFMON_RESULT_SIZE_AMD, 4, &size, &written);
        if (!avail || size < 16) return false;
        // A fresh buffer every call: the size is not known before the read, and
        // shrinking a reused one is the shape that faulted a standalone probe.
        vector<GLuint> res(size / 4 + 4);
        g_pc.data(pc_monitor_, GL_PERFMON_RESULT_AMD, size, res.data(), &written);
        unsigned long long v[PC_N] = {0, 0, 0, 0};
        for (size_t i = 0; i + 3 < res.size(); i += 4)
                for (int k = 0; k < PC_N; ++k)
                        if (res[i + 1] == g_pc_ids[k]) v[k] += res[i + 2];
        idle = v[PC_IDLE]; frag = v[PC_FRAG]; exec = v[PC_EXEC]; quads = v[PC_QUADS];
        return true;
}
