// GL objects and the scene they belong to: the texture/target factories, the
// release/reset path, and the bind cache the factories invalidate.
//
// Split out of gpu.cpp (2026-11-14).  Every other area of that file -- matvec
// kernels, attention, conv, FFN -- calls into this, so it is the first cut: with
// the shared state declared in gpu_internal.h, the next cuts can move without
// touching it again.
//
// `release_scene()` deletes every texture and framebuffer the context created and
// `reset_targets()` zeroes the scratch targets; together they are what lets a
// model be replaced without rebuilding the context (AGENTS.md, "Switching models
// without restarting the engine").  scripts/check_target_reset.sh keeps that
// ledger complete.

#include "gpu.h"
#include "gpu_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Scene ledgers, private to this file: nothing outside the release path may walk
// them (a texture is owned by whichever factory made it).
static std::vector<GLuint> g_scene_tex;
static std::vector<GLuint> g_scene_fbo;

int g_up_tex = 0;
double g_up_tex_ms = 0.0;

GLuint g_bound_tex[8] = {0, 0, 0, 0, 0, 0, 0, 0};
GLuint g_cur_fbo = 0xFFFFFFFFu;
GLuint g_cur_prog = 0xFFFFFFFFu;

static bool ffn_flush_on() { static const bool on = env_set("FFN_FLUSH"); return on; }
static bool ffn_sync_on() { static const bool on = env_set("FFN_SYNC"); return on; }

bool ffn_flush_wanted(bool fast_finish) { return fast_finish || ffn_flush_on(); }

void ffn_sync() {
    if (ffn_sync_on()) { glFlush(); glFinish(); }
}

void gl_state_forget_bindings() {
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


void GPU::reset_format_plan(){
        res_pack_ = false;
        act_fmt_ = XFormat::I16_2;
        ffn_fmt_ = XFormat::I16_2;
        res_fmt_ = XFormat::I16_1;
    }

void GPU::reset_targets(){
        // Every scratch target this GPU owns, in one place: release_scene()
        // deletes their textures, and a Target that still carried a deleted
        // name would be *reused* by the guarded create sites in this file
        // ("if (t.tex == 0 || ...)") - silent garbage instead of a new one.
        // scripts/check_target_reset.sh fails if a Target member is added
        // without being listed here.
        conv_t_b_ = Target();
        conv_t_c_ = Target();
        conv_t_x_ = Target();
        conv_cv_ = Target();
        conv_y_ = Target();
        logits_target_ = Target();
        candidate_target_ = Target();
        sum_target_ = Target();
        ffn_a_ = Target();
        ffn_b_ = Target();
        ffn_h_ = Target();
        ffn_ab_ = Target();
        ffn_ab_a_ = Target();
        ffn_ab_b_ = Target();
        ffn_h_a_ = Target();
        ffn_h_b_ = Target();
        qkv_target_ = Target();
        att_scores_ = Target();
        att_stats_ = Target();
        att_probs_ = Target();
        att_out_ = Target();
        qrot_target_ = Target();
        qscale_target_ = Target();
        hidden_a_ = Target();
        hidden_b_ = Target();
        xb_ = Target();
        xbf_ = Target();
        xf_ = Target();
        rms_stats_ = Target();
        rms_partial_ = Target();
        rms_partial2_ = Target();
        ffn_pair_a_ = Target();
        ffn_pair_b_ = Target();
        pp_a_ = Target();
        pp_a2_ = Target();
        pp_a3_ = Target();
        pp_b_ = Target();
        pp_b2_ = Target();
        pp_b3_ = Target();
        pp_in2_ = Target();
        pp_g_ = Target();
        pp_w2_ = Target();
        pp_w2b_ = Target();
        xb2_ = Target();
        xbf2_ = Target();
        probe_ = Target();
        conv_state_.clear();
        vs_targets_.clear();
        kv_cache_.clear();
        // target_for(rows) caches by row count and is filled lazily *during*
        // generation, so it is the one Target store that only exists after a
        // turn has run: leaving it behind made a model swap reuse deleted
        // handles, and only when the old model had already generated.
        targets_.clear();
    }

void GPU::release_scene(){
        gpu_dmabuf_release_all();
        // Deleting a name that is already gone is a no-op in GL, so a texture
        // rebuilt in place (the conv kernel) can sit in the list twice.

        for (GLuint t : g_scene_tex) glDeleteTextures(1, &t);
        for (GLuint f : g_scene_fbo) glDeleteFramebuffers(1, &f);
        g_scene_tex.clear();
        g_scene_fbo.clear();
        // The per-layer handle vectors point at names that are gone.  Keeping
        // them would make a reload reuse or re-delete stale names; the textures
        // themselves are already in the ledger above.
        reset_targets();
        // The upload cache is keyed by slot number and still holds the deleted
        // names; without this the next model would draw into textures that no
        // longer exist (the first swap's answer came back empty exactly so).
        input_textures_.clear();
        input_textures2_.clear();
        conv_k_tex_.clear();
        rms_att_tex_.clear();
        rms_ffn_tex_.clear();
        q_norm_tex_.clear();
        k_norm_tex_.clear();
        gl_state_forget_bindings();
        glFinish();
        // A deletion that goes wrong shows up here and nowhere else: nothing in
        // the engine reads glGetError, so this is the one check on the ledger.
        if (GLenum err = glGetError())
            fprintf(stderr, "RELEASE: gl error 0x%04x after scene release\n", err);
    }

GPU::~GPU(){
        if (dpy_ == EGL_NO_DISPLAY) return;
        // Run the same path a model swap runs, so it is exercised by every run
        // of the gate rather than only when someone changes models.
        release_scene();
        eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (ctx_ != EGL_NO_CONTEXT) {
            eglDestroyContext(dpy_, ctx_);
            ctx_ = EGL_NO_CONTEXT;
        }
        eglTerminate(dpy_);
        dpy_ = EGL_NO_DISPLAY;
    }

// Every allocation failure on this box is a memory failure first; print the
// budget with it.  CmaFree is the Pi's binding constraint (docs/performance.md:
// the MoE model at q8 takes the CMA and the session's FBOs fail), and the
// desktop driver has no CMA at all, so the line adapts.
static void report_mem(const char* tag){
        FILE* f = fopen("/proc/meminfo", "r");
        if (!f) return;
        char line[256];
        long cma_free = -1, cma_total = -1, mem_free = -1;
        while (fgets(line, sizeof line, f)) {
                if (!strncmp(line, "CmaFree:", 8)) sscanf(line + 8, "%ld", &cma_free);
                else if (!strncmp(line, "CmaTotal:", 9)) sscanf(line + 9, "%ld", &cma_total);
                else if (!strncmp(line, "MemFree:", 8)) sscanf(line + 8, "%ld", &mem_free);
        }
        fclose(f);
        if (cma_total > 0)
                fprintf(stderr, "%s: CmaFree %ld kB of %ld kB, MemFree %ld kB\n",
                        tag, cma_free, cma_total, mem_free);
        else
                fprintf(stderr, "%s: MemFree %ld kB (no CMA)\n", tag, mem_free);
}

static size_t g_tex_bytes = 0;    // uploaded texture bytes, for the failure log

// Q4_TEX_STATS=1: print, per texture, which path it took and its logical size.
// It deliberately does *not* report a CmaFree delta any more: on this driver
// those deltas are page-cache migration, not the texture (the same shape read
// 2020, 2436, 4576 and 4544 kB in four consecutive uploads, and an RGBA8 upload
// once *raised* CmaFree).  The driver's BO ledger is
// /sys/kernel/debug/dri/0/bo_stats, read around the whole process.
static bool q4_tex_stats(){
        static const char* e = getenv("Q4_TEX_STATS");
        static const bool v = e && strcmp(e, "0") != 0;
        return v;
}


GLuint GPU::make_texture(int w, int h, const void* data){
        const double t0 = shader_time_ms();
        GLuint t = 0; glGenTextures(1, &t); g_scene_tex.push_back(t);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t); gl_state_forget_bindings();
        while (glGetError() != GL_NO_ERROR) {}
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        if (GLenum e = glGetError()) {
                cerr << "texture upload failed: RGBA8 " << w << "x" << h
                     << " (0x" << hex << e << dec << "), "
                     << g_scene_tex.size() << " textures / " << g_tex_bytes << " B so far\n";
                report_mem("texture upload");
                exit(1);
        }
        g_tex_bytes += (size_t)w * h * 4;
        if (q4_tex_stats())
                fprintf(stderr, "rgba8  %5dx%-5d logical %6zu kB  upload\n",
                        w, h, (size_t)w * h * 4 / 1024);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFlush(); glFinish();
        g_up_tex_ms += shader_time_ms() - t0; ++g_up_tex;
        return t;
    }

GLuint GPU::make_q4_texture(int w, int h, const void* data){
        const double t0 = shader_time_ms();
        // Reverse the four nibbles of every 16-bit texel on the way in.  The
        // sampler hands back the texel's four weights in reverse column order
        // (GL packs R into the most significant nibble), which cost 52 ms of
        // the gate at 1216 rows as a .wzyx operand per dot.  Reversing here is
        // the same permutation with no quality effect, and lets the kernel dot
        // the texel against the activations in natural order.
        const size_t n = (size_t)w * h;
        const uint16_t* src = (const uint16_t*)data;
        std::vector<uint16_t> rev(n);
        for (size_t i = 0; i < n; ++i) {
                const uint16_t v = src[i];
                rev[i] = (uint16_t)(((v & 0x000F) << 12) | ((v & 0x00F0) << 4) |
                                    ((v & 0x0F00) >> 4) | ((v & 0xF000) >> 12));
        }
        // Imported where the platform allows it: a two-byte surface, which the
        // upload path cannot produce (gpu_dmabuf.cpp explains why).
        // `rev`, not `src`: an imported surface reports the word's nibbles
        // reversed (a self-check on the board: word 0xa727 samples as
        // 10 7 2 7), the same order the upload path sees as .wzyx.
        if (GLuint imported = gpu_dmabuf_texture(w, h, rev.data())) {
                g_tex_bytes += (size_t)w * h * 2;
                if (q4_tex_stats())
                        fprintf(stderr, "q4tex  %5dx%-5d logical %6zu kB  dma-buf tiled import\n",
                                w, h, (size_t)w * h * 2 / 1024);
                return imported;
        }
        GLuint t = 0; glGenTextures(1, &t); g_scene_tex.push_back(t);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t); gl_state_forget_bindings();
        // A 4-bit texel is two bytes, so a row of w texels is 2w bytes; the
        // default alignment of 4 only holds when that is a multiple of four.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        while (glGetError() != GL_NO_ERROR) {}
        // Which internal format to ask for depends on what the driver does with
        // it, and the two cases are opposite:
        //
        //  * vc4 cannot render to B4G4R4A4, so st_ChooseTextureFormat's render
        //    target binding sends find_supported_format to RGBA8 and the nibbles
        //    are expanded: four bytes a texel, measured against bo_stats.  That
        //    is why the import above exists; this path is its fallback there.
        //  * everywhere else RGBA4 *is* renderable, the same table entry is
        //    chosen, and a sized GL_RGBA4 really is two bytes a texel -- asking
        //    for the unsized form there throws half the memory away.
        //
        // The sized spelling is only legal from ES 3.0 (ES2 requires
        // internalformat == format) and the engine knows which context it got:
        // gpu_q4_shift() is true exactly when the driver reports ESSL 3.
        const GLint internal = gpu_q4_shift() ? GL_RGBA4 : GL_RGBA;
        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,
                     rev.data());
        if (GLenum e = glGetError()) {
                cerr << "texture upload failed: RGBA4 " << w << "x" << h
                     << " (0x" << hex << e << dec << "), "
                     << g_scene_tex.size() << " textures / " << g_tex_bytes << " B so far\n";
                report_mem("texture upload");
                exit(1);
        }
        g_tex_bytes += (size_t)w * h * 2;
        if (q4_tex_stats())
                fprintf(stderr, "q4tex  %5dx%-5d logical %6zu kB  upload (RGBA8, 2x)\n",
                        w, h, (size_t)w * h * 2 / 1024);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFlush(); glFinish();
        g_up_tex_ms += shader_time_ms() - t0; ++g_up_tex;
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
        glGenFramebuffers(1, &t.fbo); g_scene_fbo.push_back(t.fbo);
        bind_fbo(t.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            const GLenum err = glGetError();
            GLint max_tex = 0, max_rb = 0, max_vp[2] = { 0, 0 };
            glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
            glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_rb);
            glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_vp);
            cerr << "FBO incomplete: " << w << "x" << h
                 << " status 0x" << hex << status << dec
                 << " glerr 0x" << hex << err << dec
                 << " max_tex " << max_tex << " max_rb " << max_rb
                 << " max_viewport " << max_vp[0] << "x" << max_vp[1]
                 << " tex " << t.tex << ", textures " << g_scene_tex.size()
                 << " / " << g_tex_bytes << " B\n";
            report_mem("FBO incomplete");
            exit(1);
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

