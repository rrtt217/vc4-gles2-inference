#include "model.h"
#include "logits_grid.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

// The checkpoint is mapped once and kept for the process lifetime.  The
// loaders then point each Mat straight at its bytes in the map instead of
// allocating a private copy: on this 426 MB board the 69 MB q8 copy is
// what pushed the kernel into swap and evicted the file (and the Mesa
// driver) between runs, so a warm start read 69 MB from the SD card at
// 11 MB/s in every single run.  Borrowed pages are page cache: reclaimable,
// never swapped, and free to re-read.
bool Model::load(const string& path){
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { cerr << "cannot open model: " << path << "\n"; return false; }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < 4) {
        close(fd); cerr << "cannot stat model: " << path << "\n"; return false;
    }
    borrow_ = !(getenv("MODEL_BORROW") && !strcmp(getenv("MODEL_BORROW"), "0"));
    void* base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base != MAP_FAILED && !borrow_) { close(fd); fd = -1; }
    if (base == MAP_FAILED) { close(fd); cerr << "mmap failed: " << path << "\n"; return false; }
    source_size = (uint64_t)st.st_size;
    source_mtime = (uint64_t)st.st_mtime;
    fbase_ = (const uint8_t*)base;
    fsize_ = (size_t)st.st_size;
    // Start the SD read now: it overlaps the EGL/Mesa init that follows.
    // Bounded, not the whole file: on a 200 MB container the full prefetch
    // fills the page cache with the entire model before the upload starts, and
    // on the Pi those pages come from the same movable pool the textures need
    // (measured 2026-11-17, models/pocket-0.3b: 150 MB of page cache inside the
    // CMA area before the first texture).  drop_pages() releases each tensor as
    // it lands in a texture, so what stays resident is one prefetch window.
    madvise(base, std::min<size_t>((size_t)st.st_size, (size_t)8 << 20), MADV_WILLNEED);
    fpos_ = 0;
    uint32_t magic = 0; memcpy(&magic, fbase_, 4);
    bool ok = false;
    if (magic != 0x46434D4DU) {   // "MMCF"
            cerr << "not an MMCF container (magic 0x" << std::hex << magic << std::dec
                 << "); the legacy Q8M1/Q4M1/I16M1/llama2c readers are gone, convert with "
                    "engine/tools/mmcf_pack.py\n";
            munmap(base, (size_t)st.st_size);
            if (fd >= 0) close(fd);
            return false;
    }
    ok = load_mmcf();
    if (ok && t3_mats) {
            // One line, not a debug switch: a container whose body was expanded
            // 4x on the host is a fact the operator needs (the file is small,
            // the RAM is not), and it is also the signal that the kernel this
            // format was designed for is not in this build yet.
            if (t3_ram_bytes)
                    cerr << "t3 body: " << t3_mats << " matrices expanded to q8 on the host ("
                         << (t3_disk_bytes >> 20) << " MB on disk -> " << (t3_ram_bytes >> 20)
                         << " MB in RAM) because T3_EXPAND is set\n";
            else
                    cerr << "t3 body: " << t3_mats << " matrices, 2-bit, borrowed from the "
                         << "container and uploaded as they lie (" << (t3_disk_bytes >> 20)
                         << " MB of texture)\n";
    }
    fbase_ = nullptr; fsize_ = fpos_ = 0;
    if (ok && !borrow_ && tok_blob && tok_blob_bytes) {
            // load_mmcf() borrows the tokenizer blob from the mapping that
            // this path unmaps below; copy it before the munmap or the
            // tokenizer (read after load()) decodes a released page.
            const uint8_t* p = (const uint8_t*)tok_blob;
            tok_owned.assign(p, p + tok_blob_bytes);
            tok_blob = tok_owned.data();
    }
    if (ok && getenv("MODEL_DUMP")) {
            auto head = [](const char* n, const Mat& m) {
                    if (m.rows == 0) { cerr << "DUMP " << n << " empty\n"; return; }
                    char b[64]; int k = 0;
                    for (int i = 0; i < 8 && i < m.rows * m.cols; ++i)
                            k += snprintf(b + k, sizeof(b) - k, "%d ", (int)m.w8()[i]);
                    cerr << "DUMP " << n << " " << m.rows << "x" << m.cols << " s=" << m.scale
                         << " pb=" << m.prebiased << " [" << b << "]\n";
            };
            head("qkv0", qkv.empty() ? Mat() : qkv[0]);
            head("w2_0", w2.empty() ? Mat() : w2[0]);
            if (has_q8_emb) cerr << "DUMP emb0 [" << (int)emb_q8[0] << " " << (int)emb_q8[1] << "] es0=" << emb_scale[0] << "\n";
            cerr << "DUMP rms_att0=" << (rms_att.empty() ? 0.0f : rms_att[0]) << " theta=" << rope_theta << " half=" << rope_half << " qk=" << qk_norm << "\n";
    }
    if (borrow_ && ok) {
            map_ = base; map_size_ = (size_t)st.st_size;
            fd_ = fd;   // drop_pages() invalidates what the upload has consumed
    } else {
            munmap(base, (size_t)st.st_size);
            if (fd >= 0) close(fd);
    }
    return ok;
}

bool Model::probe(const string& path){
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < 4) {
        close(fd);
        return false;
    }
    void* base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return false;
    source_size = (uint64_t)st.st_size;
    source_mtime = (uint64_t)st.st_mtime;
    fbase_ = (const uint8_t*)base;
    fsize_ = (size_t)st.st_size;
    fpos_ = 0;
    // No MADV_WILLNEED: a probe reads the metadata block, not the tensors, and a
    // container that is about to be refused must not cost an SD card read first.
    uint32_t magic = 0;
    memcpy(&magic, fbase_, 4);
    const bool ok = magic == 0x46434D4DU && parse_header() && parse_metadata();
    fbase_ = nullptr;
    fsize_ = fpos_ = 0;
    // parse_metadata() points tok_blob into this mapping.  It is about to go, and
    // a live Model holding a pointer into an unmapped range is the exact hazard
    // load() documents; nothing a probe is asked for reads the blob.
    tok_blob = nullptr;
    tok_blob_bytes = 0;
    munmap(base, (size_t)st.st_size);
    return ok;
}

Model::~Model(){
    if (map_) munmap(map_, map_size_);
    map_ = nullptr; map_size_ = 0;
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
}
// After the upload the whole checkpoint is on the GPU.  Unmapping it drops the
// resident file pages from this process - they stay in the page cache, so the
// next start is still warm - and clears the borrowed pointers so nothing can
// read a stale view of a mapping that no longer exists.
// A container this build cannot run.  The three sites below are *prepare-time*
// refusals (an FFN hidden that needs more than two chunks, per-part precision on a
// MoE container): by the time they fire the model is half-built, so returning an
// error would hand the caller a model that cannot be used or freed coherently.
// Everything a caller *can* recover from -- missing file, bad magic, short read, a
// version this build does not know -- is a `false` from `Model::load()`.
[[noreturn]] static void refuse(const std::string& what) {
        std::cerr << "mmcf: " << what << "\n";
        exit(1);
}

// One tensor: unmap this process's view of the bytes that are already in a
// texture, then drop the (now unmapped, clean) pages behind them.  Both halves
// are needed -- MADV_DONTNEED alone only zaps the PTEs and leaves the pages in
// the page cache, and POSIX_FADV_DONTNEED will not invalidate a page that is
// still mapped.  A page dropped here is re-read from the file if anything
// touches those bytes again, so this trades no correctness for memory.
void Model::drop_pages(const void* p, size_t bytes){
        if (!map_ || fd_ < 0 || !p || !bytes) return;
        const char* base = (const char*)map_;
        const char* q = (const char*)p;
        if (q < base || q >= base + map_size_) return;
        const uint64_t page = 4096;
        uint64_t off = (uint64_t)(q - base) & ~(page - 1);
        uint64_t end = ((uint64_t)(q - base) + (uint64_t)bytes + page - 1) & ~(page - 1);
        if (end > (uint64_t)map_size_) end = (uint64_t)map_size_;
        if (end <= off) return;
        madvise((void*)(base + off), (size_t)(end - off), MADV_DONTNEED);
        posix_fadvise(fd_, (off_t)off, (off_t)(end - off), POSIX_FADV_DONTNEED);
}

void Model::drop_pages(const Mat& m){
        const size_t cols = (size_t)m.cols, rows = (size_t)m.rows;
        if (m.ext8) drop_pages(m.ext8, rows * cols);
        if (m.ext16) drop_pages(m.ext16, rows * cols * 2);
        if (m.ext4) drop_pages(m.ext4, rows * (cols / 2));
        if (m.groups > 0) {
                const size_t plane = rows * (size_t)m.groups * sizeof(int16_t);
                if (m.ext_scale4) drop_pages(m.ext_scale4, plane);
                if (m.ext_min4) drop_pages(m.ext_min4, plane);
        }
}

void Model::release_map(){
    if (map_) munmap(map_, map_size_);
    map_ = nullptr; map_size_ = 0;
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
    auto clear = [](vector<Mat>& v){
        for (auto& m : v) {
            m.ext8 = nullptr; m.ext16 = nullptr; m.ext4 = nullptr;
            m.ext_scale4 = nullptr; m.ext_min4 = nullptr;
        }
    };
    clear(qkv); clear(wo); clear(w13); clear(w1); clear(w3); clear(w2); clear(logits_tiles);
    clear(w1a); clear(w1b); clear(w3a); clear(w3b); clear(w2a); clear(w2b);
    // The RWKV-7 low-rank pairs are the only other Mats that point into the
    // mapping; a swap that left them behind would keep a stale ext8/ext4.
    clear(rwkv_w1); clear(rwkv_w2); clear(rwkv_v1); clear(rwkv_v2);
    clear(rwkv_a1); clear(rwkv_a2); clear(rwkv_g1); clear(rwkv_g2);
    // The three [W_r; W_k; W_v] bands are views into the same mapping.
    clear(rwkv_r); clear(rwkv_k); clear(rwkv_v);
}

void Model::materialize(Mat& m){
    if (m.ext8) { m.q8.assign(m.ext8, m.ext8 + (size_t)m.rows * m.cols); m.ext8 = nullptr; }
    if (m.ext16) { m.q16.assign(m.ext16, m.ext16 + (size_t)m.rows * m.cols); m.ext16 = nullptr; }
    if (m.ext4) { m.q4.assign(m.ext4, m.ext4 + (size_t)m.rows * (m.cols / 2)); m.ext4 = nullptr; }
    if (m.ext_scale4) { m.q4_scale.assign(m.ext_scale4, m.ext_scale4 + (size_t)m.rows * m.groups); m.ext_scale4 = nullptr; }
    if (m.ext_min4) { m.q4_min.assign(m.ext_min4, m.ext_min4 + (size_t)m.rows * m.groups); m.ext_min4 = nullptr; }
}

void Model::split_ffn(int max_dim){
        // RWKV-7's channel mix is one (hidden x dim) key and one (dim x hidden)
        // value matrix, not a w1/w3 gate pair: splitting it needs a decision
        // from the layer loop that will read it (P2), and slicing here would
        // also index an empty w3.  A 3072-row matrix is over VC4's 2048-row
        // upload limit, so the 0.1B on the board is a P5 item, not a reader
        // one.
        if (is_rwkv()) return;
        if (is_moe()) {
                // The 2048-row texture/viewport limit applies to the expert FFN,
                // not to the (unused) dense width.  Each expert is chunked on
                // its own; the sliced Mats share the mapping, so this only adds
                // texture descriptors.
                if (moe_hidden <= 2048) return;
                if (moe_hidden > 4096) refuse("expert hidden too large for two FFN chunks");
                ffn_chunked = true;
                ffn_chunk = (moe_hidden + 1) / 2;
                for (auto& layer : moe) {
                        for (auto& e : layer) {
                                e.w1a = slice_rows(e.w1, 0, ffn_chunk);
                                e.w1b = slice_rows(e.w1, ffn_chunk, moe_hidden);
                                e.w3a = slice_rows(e.w3, 0, ffn_chunk);
                                e.w3b = slice_rows(e.w3, ffn_chunk, moe_hidden);
                                e.w2a = slice_cols(e.w2, 0, ffn_chunk);
                                e.w2b = slice_cols(e.w2, ffn_chunk, moe_hidden);
                                e.w1 = Mat(); e.w3 = Mat(); e.w2 = Mat();
                        }
                }
                return;
        }
        if (hidden <= 2048) return;
        // Two chunks are all the FFN kernels carry, and each chunk is written to
        // a texture/FBO of its own width, so the real ceiling is 2x the driver's
        // target limit: 4096 on VC4 (2048) and larger on desktop GLES2.  It used
        // to be the literal 4096, which refused LFM2-350M's 4608 hidden even
        // though a 2304-wide chunk is legal on Xe (2026-11-12).
        const int chunk_limit = max_dim > 0 ? max_dim : 2048;
        if (hidden > 2 * chunk_limit)
                refuse("hidden " + std::to_string(hidden) + " needs two FFN chunks wider than the "
                       "GL target limit " + std::to_string(chunk_limit));
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

// The three row bands of qkv[l] = [W_r; W_k; W_v], as Mats the matvec family
// can draw.  Runs before the upload (the bands are borrowed views into the
// checkpoint mapping, which release_map() drops after the upload), and after
// read_tables() has checked qkv[l] is 3*dim x dim.
//
// This is the plan's P2 item 2: the container reuses the dense qkv id for the
// three matrices because they are drawn by the same kernel, and the layer loop
// needs them one at a time -- each band takes its own token-shifted input
// (x_r / x_k / x_v), so one 3*dim-row draw cannot serve it.
void Model::split_rwkv(int max_dim){
        if (!is_rwkv()) return;
        rwkv_r.assign(layers, Mat());
        rwkv_k.assign(layers, Mat());
        rwkv_v.assign(layers, Mat());
        for (int l = 0; l < layers; ++l) {
                if (qkv[l].rows != 3 * dim) refuse("rwkv layer " + std::to_string(l) +
                                                   "'s rkv is not 3*dim rows");
                rwkv_r[l] = slice_rows(qkv[l], 0, dim);
                rwkv_k[l] = slice_rows(qkv[l], dim, 2 * dim);
                rwkv_v[l] = slice_rows(qkv[l], 2 * dim, 3 * dim);
        }
        // apply_q4_fmt() indexes w3[l] / w3a[l] / w3b[l] whenever w1 / w1a are
        // sized (it assumes the dense swiglu triple).  RWKV has no w3 gate, so
        // give it empty Mats of the right length: want() skips rows==0, but the
        // indexing itself must be in bounds or the chunked path segfaults.
        w3.assign(layers, Mat());
        w3a.assign(layers, Mat());
        w3b.assign(layers, Mat());
        // Channel-mix key is hidden x dim; on VC4 a 3072-row texture is illegal.
        // Chunk it (and the value's columns) like the dense FFN so each upload
        // stays <= max_dim.  Xe (16384) never chunks.
        const int F = w1.empty() ? 0 : w1[0].rows;
        if (F > max_dim) {
                // The cut is even by default.  RWKV_CM_SPLIT=<rows> moves it
                // (docs/test-hooks.md): the chunk widths decide how many
                // kDrawCap-sized strips the key matvec and its relu^2 take
                // (draw() and rwkv_draw1 both split at 1024 fragments), so a
                // 2048+1024 cut is 3+3 strips against 1536+1536's 4+4 on VC4.
                // It is a measurement arm: the split point moves the q16
                // residual grid (two adds, whose first input is the chunk sum),
                // so the PPL it lands on is the arm's own, not the pin's.
                // Both halves must still fit one texture, and the cut must
                // leave both halves non-empty.
                int c = (F + 1) / 2;
                const int cut = env_int("RWKV_CM_SPLIT", 0);
                if (cut > 0) {
                        if (cut >= F || cut > max_dim || F - cut > max_dim) {
                                cerr << "RWKV_CM_SPLIT=" << cut << " is not a usable cut for "
                                     << F << " rows (texture limit " << max_dim
                                     << "); using " << c << "\n";
                        } else {
                                c = cut;
                        }
                }
                ffn_chunked = true;
                ffn_chunk = c;
                w1a.resize(layers); w1b.resize(layers);
                w2a.resize(layers); w2b.resize(layers);
                for (int l = 0; l < layers; ++l) {
                        w1a[l] = slice_rows(w1[l], 0, c);
                        w1b[l] = slice_rows(w1[l], c, F);
                        w2a[l] = slice_cols(w2[l], 0, c);
                        w2b[l] = slice_cols(w2[l], c, F);
                }
        }
}

void Model::apply_q4_fmt(){
        // Adopted 2026-11-15: the 4-bit view is the default for the dense FFN
        // gate.  It measured -6% on a q4 token on the Pi (286.2 -> 270.6,
        // three interleaved pairs, load 0.12) and -8% on the gate's own
        // marginal cost (MATVEC_DUP=ffn: 134.5 -> 123.9).  `Q4_FMT=0` restores
        // the packed view inside the same binary, which is the A/B the
        // measurement rules ask for.
        // Default on.  The view used to cost twice the packed one in VRAM,
        // because glTexImage2D cannot produce a two-byte surface on this driver
        // (docs/performance.md 2026-11-15); gpu_dmabuf.cpp imports the surface
        // itself as a T-tiled dma-buf and the cost is now +1.2% instead of +61%
        // (51524 kB of V3D BOs against 50892) with -23% on a q4 token.
        // Q4_FMT=0 keeps the packed view for an A/B.
        const char* v = getenv("Q4_FMT");
        if (v && strcmp(v, "0") == 0) return;
        // The hybrid is not refused any more: the failure that looked like a
        // 4-bit-format problem was the board running out of CMA while its other
        // services held most of it, and it reproduced on the packed view as
        // well (docs/performance.md, 2026-11-15).  Verified on Xe: the 4-bit
        // view answers identically to the packed one on lfm2.5-230M q4
        // ("我是一个聊天机器人..." and "The capital of France is Paris." for both).
        // If a board with little CMA free cannot hold the model, the upload now
        // says so with the size, the texture budget and CmaFree.
        // Only the fused, chunked gate reads the 4-bit view.  The unchunked
        // path (ffn_split -> matvec_to_row + swiglu_2row) reads packed nibbles.
        if (!ffn_chunked) {
                if (v) cerr << "Q4_FMT: the unchunked FFN path is not supported; using the packed view\n";
                return;
        }
        // With the *sized* GL_RGBA4 internal format the surface is two bytes a
        // texel (PIPE_FORMAT_B4G4R4A4_UNORM), so the view holds the same bytes
        // as the packed one: twice the texels, half the bytes each.  The unsized
        // GL_RGBA spelling fell back to RGBA8 and doubled the memory -- that is
        // what made the hybrid fail, and it is fixed in make_q4_texture.
        // probes/rgba4444_probe section H keeps both spellings measurable.
        vector<Mat*> mark4;
        auto want = [&](Mat& m) { if (m.rows > 0 && m.type == WeightType::INT4) mark4.push_back(&m); };
        if (is_moe()) {
                for (auto& layer : moe) for (auto& e : layer) {
                        want(e.w1a); want(e.w1b); want(e.w3a); want(e.w3b);
                }
        } else {
                if ((int)w1a.size() >= layers)
                        for (int l = 0; l < layers; ++l) {
                                want(w1a[l]); want(w1b[l]); want(w3a[l]); want(w3b[l]);
                                want(w2a[l]); want(w2b[l]);
                        }
                if ((int)w1.size() >= layers)
                        for (int l = 0; l < layers; ++l) { want(w1[l]); want(w3[l]); want(w2[l]); }
                if ((int)qkv.size() >= layers)
                        for (int l = 0; l < layers; ++l) { want(qkv[l]); want(wo[l]); }
                // The hybrid's conv block: three 1024-row slices of in_proj plus
                // out_proj, all INT4 and all through matvec_to / matvec_add,
                // which already have a 4x4 kernel.  They used to keep the packed
                // view, so the block missed the zero-decode path.
                if ((int)conv_in_b.size() >= layers)
                        for (int l = 0; l < layers; ++l) {
                                want(conv_in_b[l]); want(conv_in_c[l]); want(conv_in_x[l]);
                                want(conv_out[l]);
                        }
        }
        for (Mat* m : mark4) m->fmt4 = true;
}

// Metadata key 15: the SwiGLU hidden is written *2^-k by the FFN kernels, so
// every w2 (and its chunks, after split_ffn) has to read back *2^k.  For a
// power of two this is exact in float32 and costs one multiply here, with no
// shader change on the read side: q8/i16 carry it in the scale field (the ws
// uniform), q4 in q4_param_scale (the ps uniform), and the q8 row-scale path
// multiplies the row plane by the same unit, so all three forms follow.
void Model::apply_hidden_scale(){
        if (hid_scale.empty()) return;
        auto scale_mat = [](Mat& m, float s) {
                if (m.rows <= 0) return;
                if (m.type == WeightType::INT4) m.q4_param_scale *= s;
                else m.scale *= s;
        };
        for (int l = 0; l < layers; ++l) {
                const int k = hid_shift(l);
                if (k <= 0) continue;
                const float s = (float)(1u << k);
                if (is_moe()) {
                        for (auto& e : moe[l]) {
                                scale_mat(e.w2, s);
                                scale_mat(e.w2a, s);
                                scale_mat(e.w2b, s);
                        }
                } else {
                        if (l < (int)w2.size()) scale_mat(w2[l], s);
                        if (l < (int)w2a.size()) { scale_mat(w2a[l], s); scale_mat(w2b[l], s); }
                }
        }
}

bool Model::quantize_q4(Mat& m){
        materialize(m);
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

bool Model::quantize_q4_weights(const string& scope){
        bool attn = scope.empty() || scope == "all" || scope == "attn";
        bool ffn = scope.empty() || scope == "all" || scope == "ffn";
        bool ok = true;
        if (ffn && is_moe()) {
                // Before split_ffn() the whole expert is here, after it the two
                // chunks are; quantize whichever exists.
                for (auto& layer : moe) {
                        for (auto& e : layer) {
                                if (e.w1.rows) {
                                        ok &= quantize_q4(e.w1); ok &= quantize_q4(e.w3);
                                        ok &= quantize_q4(e.w2);
                                }
                                if (e.w1a.rows) {
                                        ok &= quantize_q4(e.w1a); ok &= quantize_q4(e.w3a);
                                        ok &= quantize_q4(e.w2a);
                                        ok &= quantize_q4(e.w1b); ok &= quantize_q4(e.w3b);
                                        ok &= quantize_q4(e.w2b);
                                }
                        }
                }
        }
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

void Model::apply_precision(const string& layers_spec, const string& parts_spec){
        if (is_moe()) {
                refuse("Q8_PARTS is not implemented for a MoE container; the expert FFNs "
                       "carry their own precision from the packer (re-pack with --quant)");
        }
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
            materialize(m);
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


// MMCF container.  Layout and field list: docs/checkpoint-container.md.
namespace {
struct MmcfRec {
        uint16_t id = 0;
        uint8_t dtype = 0, flags = 0, group = 0;
        uint32_t rows = 0, cols = 0;
        uint64_t off = 0, bytes = 0;
        uint32_t crc = 0;
        float scale = 0.0f;
        uint16_t aux = 0;     // v3: the expert index; v2 records have none
};

// v2 records are 48 B; v3 (expert tensors) is 64 B with `aux` in the tail
// padding, so the v2 field offsets are unchanged.
constexpr int MMCF_REC_V2 = 48;
constexpr int MMCF_REC_V3 = 64;

static uint16_t rd16(const uint8_t* p){ uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd32(const uint8_t* p){ uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t* p){ uint64_t v; memcpy(&v, p, 8); return v; }
static float rdf(const uint8_t* p){ float v; memcpy(&v, p, 4); return v; }
}  // namespace

bool Model::parse_header() {
        if (fsize_ < 64) { cerr << "mmcf: truncated header\n"; return false; }
        const uint8_t* h = fbase_;
        version_ = rd16(h + 4);
        // 2 dense, 3 mixture of experts, 4 RWKV-7.  v4 keeps v3's 64-byte
        // record: everything it adds is model-wide (metadata key 16) or a new
        // tensor kind whose shape the record already carries, so no per-tensor
        // field had to grow (docs/checkpoint-container.md, "v4").
        if (version_ != 2 && version_ != 3 && version_ != 4) {
                cerr << "mmcf: unsupported version " << version_ << "\n";
                return false;
        }
        rec_size_ = version_ >= 3 ? MMCF_REC_V3 : MMCF_REC_V2;
        if (rd16(h + 6) != 64) { cerr << "mmcf: bad header size\n"; return false; }
        n_tensors_ = rd32(h + 8);
        caps_ = rd32(h + 12);
        meta_off_ = rd32(h + 16); meta_size_ = rd32(h + 20);
        data_off_ = rd64(h + 24); const uint64_t file_size = rd64(h + 32);
        if (file_size != (uint64_t)fsize_) { cerr << "mmcf: size field " << file_size << " != file " << fsize_ << "\n"; return false; }
        {
                // crc32 over the first 40 bytes, stored at 40..43
                static uint32_t table[256];
                static bool init = false;
                if (!init) {
                        for (uint32_t i = 0; i < 256; ++i) {
                                uint32_t c = i;
                                for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
                                table[i] = c;
                        }
                        init = true;
                }
                uint32_t crc = 0xFFFFFFFFu;
                for (int i = 0; i < 40; ++i) crc = table[(crc ^ h[i]) & 0xFF] ^ (crc >> 8);
                crc ^= 0xFFFFFFFFu;
                const uint32_t stored = rd32(h + 40);
                if (stored && stored != crc) { cerr << "mmcf: header crc mismatch\n"; return false; }
        }
        if ((uint64_t)meta_off_ + meta_size_ > fsize_ || data_off_ > fsize_ ||
            data_off_ < (uint64_t)n_tensors_ * rec_size_) { cerr << "mmcf: bad section offsets\n"; return false; }

        table_off_ = data_off_ - (uint64_t)n_tensors_ * rec_size_;
        if (table_off_ < (uint64_t)meta_off_ + meta_size_ || (table_off_ % 16) != 0) {
                cerr << "mmcf: tensor table does not follow the metadata block\n";
                return false;
        }
    return true;
}

bool Model::parse_metadata() {
    if (!parse_header()) return false;
        bool have_model = false, have_rope = false, have_quant = false;
    int group = 0, split = 0;   // published as quant_group_/quant_split_
    for (uint64_t p = meta_off_; p + 8 <= (uint64_t)meta_off_ + meta_size_; ) {
                const uint16_t key = rd16(fbase_ + p), type = rd16(fbase_ + p + 2);
                const uint32_t len = rd32(fbase_ + p + 4);
                const uint8_t* payload = fbase_ + p + 8;
                if (p + 8 + len > (uint64_t)meta_off_ + meta_size_) { cerr << "mmcf: TLV overruns the block\n"; return false; }
                if (key == 1 && type == 1 && len >= 28) {
                        dim = (int)rd32(payload); hidden = (int)rd32(payload + 4);
                        layers = (int)rd32(payload + 8); heads = (int)rd32(payload + 12);
                        kv_heads = (int)rd32(payload + 16); vocab = (int)rd32(payload + 20);
                        seq = (int)rd32(payload + 24);
                        have_model = true;
                } else if (key == 2 && type == 1 && len >= 8) {
                        rope_theta = rdf(payload);
                        rope_half = payload[4] != 0;
                        qk_norm = payload[5] != 0;
                        have_rope = true;
                } else if (key == 3 && type == 1 && len >= 4) {
                        group = payload[1]; split = payload[3];
                        have_quant = true;
                } else if (key == 4 && type == 1 && len >= 3) {
                        template_kind = (int)payload[0];
                        const char* c = (const char*)payload + 1;
                        const char* e = (const char*)payload + len;
                        const char* nul = (const char*)memchr(c, 0, (size_t)(e - c));
                        if (nul) {
                                default_system.assign(c, (size_t)(nul - c));
                                const char* c2 = nul + 1;
                                const char* nul2 = (const char*)memchr(c2, 0, (size_t)(e - c2));
                                if (nul2) template_name.assign(c2, (size_t)(nul2 - c2));
                        }
                } else if (key == 8 && type == 1 && len >= 24) {
                        id_bos = (int)rd32(payload);
                        id_eos = (int)rd32(payload + 4);
                        id_stop = (int)rd32(payload + 8);
                        id_pad = (int)rd32(payload + 12);
                        id_unk = (int)rd32(payload + 16);
                        id_im_start = (int)rd32(payload + 20);
                        have_specials = true;
                } else if (key == 5 && type == 1 && len >= 4) {
                        capabilities = (int)rd32(payload);
                        have_capabilities = true;
                } else if (key == 9 && type == 1 && len >= 1) {
                        // n variants of (name\0, u32 len, Jinja text)
                        const uint8_t* q = payload;
                        const uint8_t* qe = payload + len;
                        const uint8_t* name = q;
                        q += 1;   // the count byte; the records follow
                        while (q < qe) {
                                const uint8_t* nul = (const uint8_t*)memchr(name, 0, (size_t)(qe - name));
                                if (!nul) break;
                                string vname((const char*)name, (size_t)(nul - name));
                                q = nul + 1;
                                if (q + 4 > qe) break;
                                const uint32_t tlen = rd32(q);
                                q += 4;
                                if (q + tlen > qe) break;
                                string body((const char*)q, tlen);
                                q += tlen;
                                if (vname.empty()) vname = "default";
                                chat_templates.emplace_back(vname, body);
                                if (vname == "default") chat_template = body;
                                name = q;
                        }
                        if (chat_template.empty() && !chat_templates.empty())
                                chat_template = chat_templates.front().second;
                } else if (key == 12 && type == 1 && len >= 1) {
                        res_q24 = payload[0] == 1;
                } else if (key == 13 && type == 1 && len >= 16) {
                        experts = (int)rd32(payload);
                        top_k = (int)rd32(payload + 4);
                        moe_hidden = (int)rd32(payload + 8);
                        moe_norm_topk = (rd32(payload + 12) & 1) != 0;
                } else if (key == 11 && type == 1 && len >= 4) {
                        float e; memcpy(&e, payload, 4);
                        if (e > 0.0f) { rms_eps = e; rms_eps_stated = true; }
                } else if (key == 14 && type == 1 && len >= 1) {
                        layer_type.assign(payload, payload + len);
                } else if (key == 15 && type == 1 && len >= 1) {
                        hid_scale.assign(payload, payload + len);
                } else if (key == 10 && type == 1 && len >= 2) {
                        add_bos = payload[0] == 255 ? -1 : (int)payload[0];
                        add_eos = payload[1] == 255 ? -1 : (int)payload[1];
                        // Third byte (2026-11-18): the tokenizer's prefix policy.
                        // Two-byte payloads keep the historical behaviour.
                        sp_prefix = len >= 3 && payload[2] == 1;
                } else if (key == 18 && type == 1 && len >= 4 && (len % 4) == 0) {
                        // A longrope model's static per-dimension rope table.
                        rope_factors.resize(len / 4);
                        for (size_t i = 0; i < rope_factors.size(); ++i) {
                                float f; memcpy(&f, payload + 4 * i, 4);
                                if (!(f > 0.0f) || !std::isfinite(f)) {
                                        cerr << "mmcf: rope factor " << i << " is " << f << "\n";
                                        return false;
                                }
                                rope_factors[i] = f;
                        }
                } else if (key == 17 && type == 1 && len >= 1) {
                        // The head input's store exponent (see model.h).
                        logit_shift = (int)payload[0];
                        if (logit_shift > 5) {
                                cerr << "mmcf: logits input scale 2^-" << logit_shift
                                     << " is out of range (0..5)\n";
                                return false;
                        }
                } else if (key == 16 && type == 1 && len >= 16) {
                        // RWKV-7 (v4): arch, head width, flags, pad, eps, four
                        // ranks -- little endian <BBBBfHHHH>.
                        rwkv.arch = (int)payload[0];
                        rwkv.head_size = (int)payload[1];
                        rwkv.gate = (payload[2] & 1) != 0;
                        float e; memcpy(&e, payload + 4, 4);
                        rwkv.norm_eps = e;
                        rwkv.r_w = (int)rd16(payload + 8);
                        rwkv.r_a = (int)rd16(payload + 10);
                        rwkv.r_v = (int)rd16(payload + 12);
                        rwkv.r_g = (int)rd16(payload + 14);
                }
                p += 8 + len;
        }
        quant_group_ = group;
    quant_split_ = split != 0;
    if (!have_model) { cerr << "mmcf: no model metadata\n"; return false; }
        if (!have_rope) rope_theta = 10000.0f;
        if (!have_quant) { cerr << "mmcf: no quant metadata\n"; return false; }
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        head_dim = dim / heads;
        kv_dim = kv_heads * head_dim;
        // The rope table indexes rope pairs, so its length is head_dim/2 and
        // nothing else: a table of the wrong length would rotate with the wrong
        // angles silently, which is worse than refusing to load.
        if (!rope_factors.empty() && (int)rope_factors.size() * 2 != head_dim) {
                cerr << "mmcf: " << rope_factors.size() << " rope factors but a "
                     << head_dim << "-wide head needs " << head_dim / 2 << "\n";
                return false;
        }
        // RWKV-7: the version and the metadata have to agree, and the head
        // width the WKV state is indexed by is stated twice (key 1's
        // dim/heads and key 16's head_size), so the reader checks them
        // against each other instead of picking one.
        if (rwkv.arch != 0) {
                if (version_ != 4) {
                        cerr << "mmcf: RWKV metadata in a version " << version_
                             << " container (RWKV is version 4)\n";
                        return false;
                }
                if (rwkv.arch != 1) {
                        cerr << "mmcf: unknown rwkv arch " << rwkv.arch << "\n";
                        return false;
                }
                if (rwkv.head_size <= 0 || head_dim != rwkv.head_size) {
                        cerr << "mmcf: rwkv head size " << rwkv.head_size
                             << " but dim/heads is " << dim << "/" << heads << "\n";
                        return false;
                }
                if (!(rwkv.norm_eps > 0.0f)) {
                        cerr << "mmcf: rwkv LayerNorm epsilon " << rwkv.norm_eps
                             << " is not positive\n";
                        return false;
                }
                if (rwkv.r_w <= 0 || rwkv.r_a <= 0 || rwkv.r_v <= 0) {
                        cerr << "mmcf: rwkv low ranks " << rwkv.r_w << "/" << rwkv.r_a
                             << "/" << rwkv.r_v << " must be positive (decay, ICLR and "
                             "value-residual are not optional)\n";
                        return false;
                }
                if (experts > 0) {
                        cerr << "mmcf: a container cannot be RWKV-7 and mixture of "
                             "experts at once\n";
                        return false;
                }
                if (!layer_type.empty()) {
                        cerr << "mmcf: layer types (key 14) in an RWKV-7 container; "
                             "its layers are all one kind\n";
                        return false;
                }
        } else if (version_ == 4) {
                cerr << "mmcf: version 4 without the RWKV-7 metadata (key 16)\n";
                return false;
        }
        if (!layer_type.empty() && (int)layer_type.size() != layers) {
                cerr << "mmcf: layer types " << layer_type.size() << " for "
                     << layers << " layers\n";
                return false;
        }
        if (!hid_scale.empty()) {
                if ((int)hid_scale.size() != layers) {
                        cerr << "mmcf: hidden-scale metadata has " << hid_scale.size()
                             << " entries for " << layers << " layers\n";
                        return false;
                }
                for (int l = 0; l < layers; ++l) {
                        if (hid_scale[l] > 5) {
                                cerr << "mmcf: layer " << l << " hidden scale 2^-"
                                     << (int)hid_scale[l] << " is out of range (0..5)\n";
                                return false;
                        }
                }
        }
        for (uint8_t t : layer_type) {
                if (t > 1) { cerr << "mmcf: unknown layer type " << (int)t << "\n"; return false; }
                hybrid = hybrid || t != 0;
        }
        (void)caps_;

    return true;
}

bool Model::read_tables() {
        // One pass over the table, creating whatever the model needs.  Unknown
        // ids are ignored: that is what lets a future writer add tensors.
        vector<MmcfRec> recs;
        recs.reserve(n_tensors_);
        for (uint32_t i = 0; i < n_tensors_; ++i) {
                const uint8_t* r = fbase_ + table_off_ + (uint64_t)i * rec_size_;
                MmcfRec m;
                m.id = rd16(r); m.dtype = r[2]; m.flags = r[3]; m.group = r[4];
                m.rows = rd32(r + 6); m.cols = rd32(r + 10);
                m.off = rd64(r + 14); m.bytes = rd64(r + 22);
                m.crc = rd32(r + 30); m.scale = rdf(r + 34);
                m.aux = version_ >= 3 ? rd16(r + 38) : 0;
                if (data_off_ + m.off + m.bytes > fsize_) { cerr << "mmcf: tensor " << i << " overruns the file\n"; return false; }
                recs.push_back(m);
        }

        auto as_f32 = [&](const MmcfRec& r, vector<float>& v) {
                v.resize((size_t)r.rows * r.cols);
                memcpy(v.data(), fbase_ + data_off_ + r.off, v.size() * 4);
        };
        auto as_i16 = [&](const MmcfRec& r, Mat& m) {
                m.rows = (int)r.rows; m.cols = (int)r.cols;
                m.type = WeightType::INT16;
                m.scale = r.scale;
                const int16_t* p = (const int16_t*)(fbase_ + data_off_ + r.off);
                if (borrow_) m.ext16 = p;
                else m.q16.assign(p, p + (size_t)r.rows * r.cols);
        };
        auto as_q8 = [&](const MmcfRec& r, Mat& m) {
                m.rows = (int)r.rows; m.cols = (int)r.cols;
                m.type = WeightType::INT8;
                m.scale = r.scale;
                m.prebiased = (r.dtype == 2);
                const int8_t* p = (const int8_t*)(fbase_ + data_off_ + r.off);
                if (borrow_) m.ext8 = p;
                else { m.q8.assign(p, p + (size_t)r.rows * r.cols); }
        };

        // A 2-bit ternary body (dtype 7, `--quant t3`): 16 weights an RGBA8
        // texel, four a byte, bit 0 magnitude and bit 1 sign.  The body is used
        // as it lies in the file -- the kernel fetches exactly these bytes, so
        // the upload is a straight copy and the 167 MB container is 167 MB of
        // texture, which is why the format exists at all.
        //
        // The weight a 2-bit level stands for.  `t3mat()` writes the record's
        // scale as the q8 *step* (the largest row magnitude / 127) and the
        // part-3 plane as `s_row / unit`, so the pair reconstructs `s_row/127`
        // -- exactly what a q8 level of +-127 multiplies.  A 2-bit level is
        // +-1, so `ws` has to carry the missing 127: `+-1 * 127*unit *
        // s_row/(127*unit) = +-s_row`.  This is checkable without a GPU:
        // `plane[0] * unit` is 127x smaller than `s_row[0]` for every row.
        auto as_t3 = [&](const MmcfRec& r, Mat& m) {
                m.rows = (int)r.rows; m.cols = (int)r.cols;
                m.scale = r.scale * 127.0f;   // a +-1 level carries +-127 steps
                const uint8_t* p = fbase_ + data_off_ + r.off;
                if (!getenv("T3_EXPAND")) {
                        m.type = WeightType::T3;
                        const size_t bytes =
                                (size_t)((m.rows + 1) / 2) * ((((m.cols + 7) / 8)) * 4);
                        if (borrow_) {
                                m.ext3 = p;
                        } else {
                                // MODEL_BORROW=0 releases the mapping, so the
                                // body has to be ours (the q8 path copies too).
                                m.q3.assign(p, p + bytes);
                        }
                        t3_mats++;
                        t3_disk_bytes += r.bytes;
                        return;
                }
                m.type = WeightType::INT8;
                m.scale = r.scale;                 // the q8 levels are +-127
                // The host expansion decodes the same texel grid the kernel
                // reads: one texel is rows 2y, 2y+1 by 8 columns, two bytes a row
                // (mmcf_hf.t3mat).  This is the oracle the 2-bit kernels are
                // diffed against, so it must read the body, not a copy of the
                // old row-major reading.
                const size_t stride = (size_t)((m.cols + 7) / 8) * 4;   // a row pair's bytes
                m.q8.resize((size_t)m.rows * m.cols);
                for (int y = 0; y < (m.rows + 1) / 2; ++y) {
                        const uint8_t* band = p + (size_t)y * stride;
                        for (int half = 0; half < 2; ++half) {
                                const int row = 2 * y + half;
                                if (row >= m.rows) break;
                                int8_t* dst = m.q8.data() + (size_t)row * m.cols;
                                for (int x = 0; x < m.cols; ++x) {
                                        const int g = x >> 3;             // column group
                                        const int b = (x >> 2) & 1;       // byte in the row
                                        const uint8_t byte = band[(size_t)g * 4 + half * 2 + b];
                                        const unsigned code = (byte >> (2 * (x & 3))) & 3u;
                                        dst[x] = (code & 1u) ? ((code & 2u) ? -127 : 127) : 0;
                                }
                        }
                }
                t3_mats++;
                t3_disk_bytes += r.bytes;
                t3_ram_bytes += (uint64_t)((size_t)m.rows * m.cols);
        };

        // q4 matrices arrive as three tensors (nibbles, quant_group_ scale, quant_group_ min)
        // sharing one id with `part` in the top two bits.  `aux` (the expert) is
        // part of the key: without it four experts would resolve to one Mat.
        struct Q4Parts { const MmcfRec* body = nullptr; const MmcfRec* scl = nullptr; const MmcfRec* mn = nullptr; };
        std::map<uint32_t, Q4Parts> q4map;
        // A q8 matrix may carry one f32 scale per output row instead of a
        // single per-matrix scale (part 3).  void.1 needs it: one scale leaves
        // 58-66% of layer 0's FFN weights at zero.
        std::map<uint32_t, const MmcfRec*> rowmap;
        auto rec_key = [](const MmcfRec& r) -> uint32_t {
                return ((uint32_t)r.aux << 16) | (uint32_t)(r.id & 0x3FFF);
        };

        auto as_mat = [&](const MmcfRec& r, Mat& m) {
                if (r.dtype == 1) as_i16(r, m);
                else if (r.dtype == 7) as_t3(r, m);
                else as_q8(r, m);
        };

        // RWKV-7's per-channel vectors arrive as f32 rows: one record per kind,
        // `rows` layers wide (1 for the two model-wide norms) and
        // `groups * dim` columns, in the HF/GGUF order.  A wrong width here
        // would not fail at the shape check below but inside a kernel, so it
        // is checked against the metadata's own dim.
        auto as_rwkv_vec = [&](const MmcfRec& r, vector<float>& dst, int groups,
                               bool global, const char* name) {
                const uint32_t want_rows = global ? 1u : (uint32_t)layers;
                if (r.dtype != 0 || r.rows != want_rows ||
                    r.cols != (uint32_t)(groups * dim) ||
                    r.bytes != (uint64_t)r.rows * r.cols * 4) {
                        cerr << "mmcf: rwkv " << name << " is dtype " << (int)r.dtype
                             << " " << r.rows << "x" << r.cols << " (" << r.bytes
                             << " B); the block wants f32 " << want_rows << "x"
                             << groups * dim << "\n";
                        return false;
                }
                as_f32(r, dst);
                return true;
        };
        // The eight low-rank kinds, indexed by layer.
        rwkv_w1.resize(layers); rwkv_w2.resize(layers);
        rwkv_v1.resize(layers); rwkv_v2.resize(layers);
        rwkv_a1.resize(layers); rwkv_a2.resize(layers);
        rwkv_g1.resize(layers); rwkv_g2.resize(layers);
        auto rwkv_mat = [&](int base, int layer) -> Mat* {
                if (layer < 0 || layer >= layers) return nullptr;
                switch (base) {
                case 0x24: return &rwkv_w1[layer];
                case 0x25: return &rwkv_w2[layer];
                case 0x26: return &rwkv_v1[layer];
                case 0x27: return &rwkv_v2[layer];
                case 0x28: return &rwkv_a1[layer];
                case 0x29: return &rwkv_a2[layer];
                case 0x2A: return &rwkv_g1[layer];
                case 0x2B: return &rwkv_g2[layer];
                default: return nullptr;
                }
        };

        vector<Mat> layers_qkv, layers_wo, layers_w1, layers_w3, layers_w13, layers_w2;
        vector<Mat> layers_conv_in, layers_conv_out;
        vector<vector<float>> layers_conv_w;
        layers_qkv.resize(layers); layers_wo.resize(layers); layers_w2.resize(layers);
        layers_conv_in.resize(layers); layers_conv_out.resize(layers);
        layers_conv_w.resize(layers);
        if (quant_split_) { layers_w1.resize(layers); layers_w3.resize(layers); }
        else layers_w13.resize(layers);
        vector<Mat> layers_router;
        vector<vector<MoeExpert>> layers_moe;
        if (experts > 0) {
                layers_router.resize(layers);
                layers_moe.resize(layers);
                for (auto& v : layers_moe) v.resize(experts);
        }

        for (const MmcfRec& r : recs) {
                const int base = r.id & 0xFF, layer = (r.id >> 8) & 0x3F, part = r.id >> 14;
                const int expert = (int)r.aux;
                if (part == 3) {   // per-row q8 scale plane
                        rowmap[rec_key(r)] = &r;
                        continue;
                }
                if (part != 0) {   // q4 scale/min plane
                        Q4Parts& q = q4map[rec_key(r)];
                        if (part == 1) q.scl = &r; else q.mn = &r;
                        continue;
                }
                if (r.dtype != 0 && r.dtype != 1 && r.dtype != 2 && r.dtype != 3 &&
                    r.dtype != 4 && r.dtype != 5 && r.dtype != 6 && r.dtype != 7) {
                        cerr << "mmcf: tensor 0x" << std::hex << r.id << std::dec
                             << " has unknown dtype " << (int)r.dtype << "\n";
                        return false;
                }
                if (r.aux != 0 && !(experts > 0 && (base == 0x12 || base == 0x13 || base == 0x15))) {
                        cerr << "mmcf: tensor 0x" << std::hex << r.id << std::dec
                             << " carries expert index " << expert << " on a kind that "
                             << "cannot be per-expert\n";
                        return false;
                }
                if (r.dtype == 4) {   // q4 body: remember it, resolve after the loop
                        q4map[rec_key(r)].body = &r;
                        continue;
                }
                if (base == 0x16) {   // MoE router
                        if (layer >= layers || expert != 0) {
                                cerr << "mmcf: router tensor for layer " << layer << "\n";
                                return false;
                        }
                        as_mat(r, layers_router[layer]);
                        continue;
                }
                if (experts > 0 && (base == 0x12 || base == 0x13 || base == 0x15)) {
                        if (expert >= experts || layer >= layers) {
                                cerr << "mmcf: expert " << expert << " of layer " << layer
                                     << " is outside the block (" << experts << " experts, "
                                     << layers << " layers)\n";
                                return false;
                        }
                        MoeExpert& e = layers_moe[layer][expert];
                        Mat* dst = base == 0x12 ? &e.w1 : base == 0x13 ? &e.w3 : &e.w2;
                        as_mat(r, *dst);
                        continue;
                }
                switch (base) {
                case 1: {  // embedding (stays signed: the CPU reads it)
                        if (r.dtype == 6) {
                                const size_t nlo = (size_t)r.rows * (r.cols / 2);
                                const size_t nhi = (size_t)r.rows * (r.cols / 4);
                                if (r.bytes != nlo + nhi) {
                                        cerr << "mmcf: q6 embedding body is " << r.bytes
                                             << " bytes, expected " << nlo + nhi << "\n";
                                        return false;
                                }
                                if (getenv("MMCF_DEBUG"))
                                        cerr << "MMCF q6 emb rows " << r.rows << " cols " << r.cols
                                             << " bytes " << r.bytes << " nlo " << nlo << "\n";
                                emb_q6_lo.resize(nlo);
                                emb_q6_hi.resize(nhi);
                                memcpy(emb_q6_lo.data(), fbase_ + data_off_ + r.off, nlo);
                                memcpy(emb_q6_hi.data(), fbase_ + data_off_ + r.off + nlo, nhi);
                                has_q6_emb = true;
                                break;
                        }
                        if (r.dtype != 3 && r.dtype != 2) {
                                cerr << "mmcf: embedding dtype " << (int)r.dtype << "\n";
                                return false;
                        }
                        emb_q8.resize((size_t)r.rows * r.cols);
                        memcpy(emb_q8.data(), fbase_ + data_off_ + r.off, emb_q8.size());
                        has_q8_emb = true;
                        break;
                }
                case 2: as_f32(r, emb_scale); break;
                case 3: as_f32(r, rms_att); break;
                case 4: as_f32(r, rms_ffn); break;
                case 5: as_f32(r, rms_final); break;
                case 6: as_f32(r, q_norm); break;
                case 7: as_f32(r, k_norm); break;
                case 8:    // int16 embedding (the i16 containers)
                        emb_i16.resize((size_t)r.rows * r.cols);
                        memcpy(emb_i16.data(), fbase_ + data_off_ + r.off, emb_i16.size() * 2);
                        has_i16_emb = true;
                        break;
                case 0x10: if (layer < layers) as_mat(r, layers_qkv[layer]); break;
                case 0x11: if (layer < (int)layers_wo.size()) as_mat(r, layers_wo[layer]); break;
                case 0x12: if (layer < (int)layers_w1.size()) as_mat(r, layers_w1[layer]); break;
                case 0x13: if (layer < (int)layers_w3.size()) as_mat(r, layers_w3[layer]); break;
                case 0x14: if (layer < (int)layers_w13.size()) as_mat(r, layers_w13[layer]); break;
                case 0x15: if (layer < (int)layers_w2.size()) as_mat(r, layers_w2[layer]); break;
                case 0x17: if (layer < layers) as_mat(r, layers_conv_in[layer]); break;
                case 0x18:  // depthwise conv kernel, f32: dim x k, ~12 KiB a layer
                        if (layer < layers) {
                                if (r.dtype != 0 || (uint64_t)r.bytes != (uint64_t)r.rows * r.cols * 4) {
                                        cerr << "mmcf: conv kernel of layer " << layer
                                             << " is not f32\n";
                                        return false;
                                }
                                const float* p = (const float*)(fbase_ + data_off_ + r.off);
                                layers_conv_w[layer].assign(p, p + (size_t)r.rows * r.cols);
                                if (conv_k && conv_k != (int)r.cols) {
                                        cerr << "mmcf: conv kernel width changes at layer "
                                             << layer << "\n";
                                        return false;
                                }
                                conv_k = (int)r.cols;
                        }
                        break;
                case 0x19: if (layer < layers) as_mat(r, layers_conv_out[layer]); break;
                // RWKV-7: nine f32 vector groups and eight low-rank pairs.  The
                // two model-wide norms (0x22/0x23) carry layer bits 0 because
                // there is one of each for the whole model.
                case 0x1A: if (!as_rwkv_vec(r, rwkv_lerp, 6, false, "time_mix_lerp")) return false; break;
                case 0x1B: if (!as_rwkv_vec(r, rwkv_bias, 3, false, "time_mix_bias")) return false; break;
                case 0x1C: if (!as_rwkv_vec(r, rwkv_scale, 3, false, "time_mix_scale")) return false; break;
                case 0x1D: if (!as_rwkv_vec(r, rwkv_att_norm, 2, false, "att_norm")) return false; break;
                case 0x1E: if (!as_rwkv_vec(r, rwkv_att_norm_2, 2, false, "att_norm_2")) return false; break;
                case 0x1F: if (!as_rwkv_vec(r, rwkv_ln_x, 2, false, "time_mix_ln")) return false; break;
                case 0x21: if (!as_rwkv_vec(r, rwkv_cm_lerp, 1, false, "channel_mix_lerp")) return false; break;
                case 0x22: if (!as_rwkv_vec(r, rwkv_tok_norm, 2, true, "tok_norm")) return false; break;
                case 0x23: if (!as_rwkv_vec(r, rwkv_out_norm, 2, true, "output_norm")) return false; break;
                case 0x24: case 0x25: case 0x26: case 0x27:
                case 0x28: case 0x29: case 0x2A: case 0x2B: {
                        Mat* m = rwkv_mat(base, layer);
                        if (!m) {
                                cerr << "mmcf: rwkv tensor 0x" << std::hex << r.id
                                     << std::dec << " has no layer\n";
                                return false;
                        }
                        as_mat(r, *m);
                        break;
                }
                case 0x20: { Mat m; as_mat(r, m); logits_tiles.push_back(m); break; }
                case 0x30:   // tokenizer blob, borrowed from the mapping
                        tok_blob = fbase_ + data_off_ + r.off;
                        tok_blob_bytes = (size_t)r.bytes;
                        break;
                default: break;  // tokenizer blob, template, unknown ids
                }
        }
        for (auto& kv : rowmap) {
                const MmcfRec& r = *kv.second;
                const int kind = r.id & 0xFF, layer = (r.id >> 8) & 0x3F;
                const int expert = (int)(kv.first >> 16);
                // aux = 0 is the first expert, so "is this an expert tensor" is
                // decided by the container being MoE and the kind, never by aux.
                const bool is_expert = experts > 0 &&
                        (kind == 0x12 || kind == 0x13 || kind == 0x15);
                Mat* m = nullptr;
                if (is_expert) {
                        if (layer >= layers || expert >= experts) {
                                cerr << "mmcf: row scale plane for expert " << expert
                                     << " of layer " << layer << " is out of range\n";
                                return false;
                        }
                        MoeExpert& e = layers_moe[layer][expert];
                        if (kind == 0x12) m = &e.w1;
                        else if (kind == 0x13) m = &e.w3;
                        else if (kind == 0x15) m = &e.w2;
                } else if (kind == 0x10 && layer < (int)layers_qkv.size()) m = &layers_qkv[layer];
                else if (kind == 0x11 && layer < (int)layers_wo.size()) m = &layers_wo[layer];
                else if (kind == 0x12 && layer < (int)layers_w1.size()) m = &layers_w1[layer];
                else if (kind == 0x13 && layer < (int)layers_w3.size()) m = &layers_w3[layer];
                else if (kind == 0x14 && layer < (int)layers_w13.size()) m = &layers_w13[layer];
                else if (kind == 0x15 && layer < (int)layers_w2.size()) m = &layers_w2[layer];
                else if (kind == 0x17 && layer < (int)layers_conv_in.size()) m = &layers_conv_in[layer];
                else if (kind == 0x19 && layer < (int)layers_conv_out.size()) m = &layers_conv_out[layer];
                else if (kind == 0x20 && layer < (int)logits_tiles.size()) m = &logits_tiles[layer];
                else if (kind >= 0x24 && kind <= 0x2B) m = rwkv_mat(kind, layer);
                if (!m || m->rows != (int)r.rows || r.bytes != (size_t)r.rows * 4) {
                        cerr << "mmcf: row scale plane 0x" << std::hex << r.id << std::dec
                             << " expert " << expert << " does not match its matrix\n";
                        return false;
                }
                // A fused W13 (id 0x14, quant_split_=0) is the one matrix whose two
                // rows per fragment sit rows apart, so its plane is paired the
                // way the kernel reads it.
                m->row_scale_hi = (kind == 0x14 && !quant_split_);
                const float* p = (const float*)(fbase_ + data_off_ + r.off);
                m->row_scale.assign(p, p + r.rows);
        }
        row_scale = !rowmap.empty();
        for (auto& kv : q4map) {
                const uint32_t key = kv.first;
                const uint16_t id = (uint16_t)(key & 0xFFFF);
                const int expert = (int)(key >> 16);
                Q4Parts& q = kv.second;
                if (!q.body || !q.scl || !q.mn) {
                        cerr << "mmcf: incomplete q4 tensor 0x" << std::hex << id << std::dec
                             << " expert " << expert << "\n";
                        return false;
                }
                Mat m;
                m.rows = (int)q.body->rows; m.cols = (int)q.body->cols;
                m.type = WeightType::INT4;
                m.group = 32;
                m.groups = (int)q.scl->cols;   // groups per row, not the plane size
                m.q4_param_scale = q.body->scale;
                // A truncated or mis-shaped plane used to be read out of bounds
                // (the copy path also copied one row's worth); validate the
                // three records against each other before either path runs.
                if (q.scl->rows != q.body->rows || q.mn->rows != q.body->rows ||
                    q.scl->cols != q.mn->cols ||
                    q.scl->bytes != (uint64_t)q.body->rows * q.scl->cols * 2 ||
                    q.mn->bytes != (uint64_t)q.body->rows * q.mn->cols * 2 ||
                    q.body->bytes != (uint64_t)q.body->rows * (q.body->cols / 2)) {
                        cerr << "mmcf: q4 tensor 0x" << std::hex << id << std::dec
                             << " expert " << expert
                             << " has inconsistent body/scale/min plane sizes\n";
                        return false;
                }
                if (!borrow_) {
                        m.q4.assign((const uint8_t*)(fbase_ + data_off_ + q.body->off),
                                    (const uint8_t*)(fbase_ + data_off_ + q.body->off) + q.body->bytes);
                        m.q4_scale.assign((const int16_t*)(fbase_ + data_off_ + q.scl->off),
                                          (const int16_t*)(fbase_ + data_off_ + q.scl->off) +
                                          (size_t)m.rows * m.groups);
                        m.q4_min.assign((const int16_t*)(fbase_ + data_off_ + q.mn->off),
                                        (const int16_t*)(fbase_ + data_off_ + q.mn->off) +
                                        (size_t)m.rows * m.groups);
                } else {
                        m.ext4 = (const uint8_t*)(fbase_ + data_off_ + q.body->off);
                        m.ext_scale4 = (const int16_t*)(fbase_ + data_off_ + q.scl->off);
                        m.ext_min4 = (const int16_t*)(fbase_ + data_off_ + q.mn->off);
                }
                const int kind = id & 0xFF, layer = (id >> 8) & 0x3F;
                if (experts > 0 && (kind == 0x12 || kind == 0x13 || kind == 0x15)) {
                        if (layer >= layers || expert >= experts) {
                                cerr << "mmcf: q4 expert tensor 0x" << std::hex << id
                                     << std::dec << " expert " << expert << " has no home\n";
                                return false;
                        }
                        MoeExpert& e = layers_moe[layer][expert];
                        if (kind == 0x12) e.w1 = m;
                        else if (kind == 0x13) e.w3 = m;
                        else e.w2 = m;
                        continue;
                }
                // only the vector that exists for this file's `quant_split_` flag is
                // sized, so index by its own size (an empty vector would be UB)
                if (kind == 0x10 && layer < (int)layers_qkv.size()) layers_qkv[layer] = m;
                else if (kind == 0x11 && layer < (int)layers_wo.size()) layers_wo[layer] = m;
                else if (kind == 0x12 && layer < (int)layers_w1.size()) layers_w1[layer] = m;
                else if (kind == 0x13 && layer < (int)layers_w3.size()) layers_w3[layer] = m;
                else if (kind == 0x14 && layer < (int)layers_w13.size()) layers_w13[layer] = m;
                else if (kind == 0x15 && layer < (int)layers_w2.size()) layers_w2[layer] = m;
                else if (kind == 0x17 && layer < (int)layers_conv_in.size()) layers_conv_in[layer] = m;
                else if (kind == 0x19 && layer < (int)layers_conv_out.size()) layers_conv_out[layer] = m;
                else if (kind >= 0x24 && kind <= 0x2B) {
                        Mat* dst = rwkv_mat(kind, layer);
                        if (dst) *dst = m;
                }
        }
        conv_in = layers_conv_in;
        conv_out = layers_conv_out;
        conv_w = layers_conv_w;
        if (hybrid) {
                conv_in_b.assign(layers, Mat());
                conv_in_c.assign(layers, Mat());
                conv_in_x.assign(layers, Mat());
                for (int l = 0; l < layers; ++l) {
                        if (!layer_type.empty() && !layer_type[l]) continue;
                        conv_in_b[l] = slice_rows(conv_in[l], 0, dim);
                        conv_in_c[l] = slice_rows(conv_in[l], dim, 2 * dim);
                        conv_in_x[l] = slice_rows(conv_in[l], 2 * dim, 3 * dim);
                }
        }
        if (experts > 0) {
                // What the runtime has: one router per layer, top-1 with a
                // normalised gate (so the selected expert is used with weight
                // 1.0), every layer sparse, expert FFNs as w1/w3/w2 pairs.
                if (top_k != 1) {
                        cerr << "mmcf: top_k " << top_k << " is not implemented; the "
                             << "engine routes top-1\n";
                        return false;
                }
                if (!moe_norm_topk) {
                        cerr << "mmcf: the MoE gate is not normalised (norm_topk_prob); "
                             << "top-1 without it needs the expert scaled by its softmax "
                             << "weight, which the runtime does not apply\n";
                        return false;
                }
                if (!quant_split_) {
                        cerr << "mmcf: MoE container with a fused W13; expert FFNs are "
                             << "w1/w3/w2\n";
                        return false;
                }
                if (moe_hidden <= 0 || moe_hidden > 4096) {
                        cerr << "mmcf: expert FFN width " << moe_hidden << " is out of range\n";
                        return false;
                }
                for (int l = 0; l < layers; ++l) {
                        if (layers_router[l].rows != experts || layers_router[l].cols != dim) {
                                cerr << "mmcf: layer " << l << " router is "
                                     << layers_router[l].rows << "x" << layers_router[l].cols
                                     << ", the block wants " << experts << "x" << dim << "\n";
                                return false;
                        }
                        for (int e = 0; e < experts; ++e) {
                                const MoeExpert& x = layers_moe[l][e];
                                if (!x.w1.rows || !x.w3.rows || !x.w2.rows) {
                                        cerr << "mmcf: layer " << l << " expert " << e
                                             << " is incomplete (w1/w3/w2)\n";
                                        return false;
                                }
                                if (x.w1.rows != moe_hidden || x.w1.cols != dim ||
                                    x.w3.rows != moe_hidden || x.w3.cols != dim ||
                                    x.w2.rows != dim || x.w2.cols != moe_hidden) {
                                        cerr << "mmcf: layer " << l << " expert " << e
                                             << " has the wrong shape\n";
                                        return false;
                                }
                        }
                }
                for (int l = 0; l < layers; ++l) {
                        if (layers_qkv[l].rows == 0 || layers_wo[l].rows == 0) {
                                cerr << "mmcf: layer " << l << " is missing qkv/wo\n";
                                return false;
                        }
                }
        } else {
                for (const MmcfRec& r : recs) {
                        const int base = r.id & 0xFF;
                        if (base == 0x16) {
                                cerr << "mmcf: a router tensor in a container without the "
                                     << "MoE metadata (key 13)\n";
                                return false;
                        }
                }
        }
        for (int l = 0; l < layers; ++l) {
                qkv.push_back(layers_qkv[l]);
                wo.push_back(layers_wo[l]);
                if (quant_split_) { w1.push_back(layers_w1[l]); w3.push_back(layers_w3[l]); }
                else w13.push_back(layers_w13[l]);
                w2.push_back(layers_w2[l]);
        }
        if (is_rwkv()) {
                // The shapes key 16 makes checkable.  Without this the four
                // reused ids would load as a transformer block with an RWKV
                // head width, and the first kernel to run would read the
                // wrong rows -- the same failure the MoE reader refuses.
                if (!quant_split_) {
                        cerr << "mmcf: an RWKV-7 container must say split_w13: its "
                                "channel mix is the w1/w2 pair (0x12/0x15), there is no "
                                "fused W13\n";
                        return false;
                }
                struct VecWant { const vector<float>* v; size_t n; const char* name; };
                const VecWant wants[] = {
                        {&rwkv_lerp, (size_t)layers * 6 * dim, "time_mix_lerp"},
                        {&rwkv_bias, (size_t)layers * 3 * dim, "time_mix_bias"},
                        {&rwkv_scale, (size_t)layers * 3 * dim, "time_mix_scale"},
                        {&rwkv_att_norm, (size_t)layers * 2 * dim, "att_norm"},
                        {&rwkv_att_norm_2, (size_t)layers * 2 * dim, "att_norm_2"},
                        {&rwkv_ln_x, (size_t)layers * 2 * dim, "time_mix_ln"},
                        {&rwkv_cm_lerp, (size_t)layers * dim, "channel_mix_lerp"},
                        {&rwkv_out_norm, (size_t)2 * dim, "output_norm"},
                };
                for (const VecWant& w : wants) {
                        if (w.v->size() != w.n) {
                                cerr << "mmcf: rwkv " << w.name << " has " << w.v->size()
                                     << " values, the block wants " << w.n << "\n";
                                return false;
                        }
                }
                // The token norm is optional: a source without a pre_ln has
                // none, and llama.cpp treats the missing one as absent too.
                if (!rwkv_tok_norm.empty() && rwkv_tok_norm.size() != (size_t)2 * dim) {
                        cerr << "mmcf: rwkv tok_norm has " << rwkv_tok_norm.size()
                             << " values, the block wants " << 2 * dim << "\n";
                        return false;
                }
                auto bad_shape = [&](const char* what, int l, const Mat& m,
                                    int rows, int cols) {
                        cerr << "mmcf: rwkv " << what << " of layer " << l << " is "
                             << m.rows << "x" << m.cols << ", the block wants " << rows
                             << "x" << cols << "\n";
                        return false;
                };
                for (int l = 0; l < layers; ++l) {
                        if (qkv[l].rows != 3 * dim || qkv[l].cols != dim)
                                return bad_shape("rkv", l, qkv[l], 3 * dim, dim);
                        if (wo[l].rows != dim || wo[l].cols != dim)
                                return bad_shape("wo", l, wo[l], dim, dim);
                        if (w1[l].rows != hidden || w1[l].cols != dim)
                                return bad_shape("channel_mix_key", l, w1[l], hidden, dim);
                        if (w2[l].rows != dim || w2[l].cols != hidden)
                                return bad_shape("channel_mix_value", l, w2[l], dim, hidden);
                        struct PairWant {
                                const vector<Mat>* down; const vector<Mat>* up;
                                int rank; const char* name; bool skip_first; bool required;
                        };
                        const PairWant pairs[] = {
                                {&rwkv_w1, &rwkv_w2, rwkv.r_w, "decay", false, true},
                                {&rwkv_a1, &rwkv_a2, rwkv.r_a, "iclr", false, true},
                                {&rwkv_v1, &rwkv_v2, rwkv.r_v, "value", true, true},
                                {&rwkv_g1, &rwkv_g2, rwkv.r_g, "gate", false, rwkv.gate},
                        };
                        for (const PairWant& pw : pairs) {
                                const Mat& d = (*pw.down)[l];
                                const Mat& u = (*pw.up)[l];
                                if (!d.rows && !u.rows) {
                                        // layer 0 records v_first instead of
                                        // mixing it, so it has no value pair; a
                                        // gated model must have every gate.
                                        if (pw.skip_first && l == 0) continue;
                                        if (!pw.required) continue;
                                        cerr << "mmcf: rwkv " << pw.name << " low rank of "
                                                "layer " << l << " is missing\n";
                                        return false;
                                }
                                if (d.rows != pw.rank || d.cols != dim)
                                        return bad_shape(pw.name, l, d, pw.rank, dim);
                                if (u.rows != dim || u.cols != pw.rank)
                                        return bad_shape(pw.name, l, u, dim, pw.rank);
                        }
                }
                if (!rwkv.gate) {
                        for (int l = 0; l < layers; ++l) {
                                if (!rwkv_g1[l].rows && !rwkv_g2[l].rows) continue;
                                cerr << "mmcf: rwkv gate tensors on layer " << l
                                     << " but key 16 says the model has no gate\n";
                                return false;
                        }
                }
        }
        if (hybrid) {
                if (conv_k < 2) {
                        cerr << "mmcf: a typed hybrid container with no conv kernel\n";
                        return false;
                }
                for (int l = 0; l < layers; ++l) {
                        if (layer_type[l]) {
                                if (conv_in[l].rows != 3 * dim || conv_in[l].cols != dim ||
                                    conv_out[l].rows != dim || conv_out[l].cols != dim ||
                                    conv_w[l].size() != (size_t)dim * conv_k) {
                                        cerr << "mmcf: layer " << l << " conv block is "
                                             << "incomplete (in " << conv_in[l].rows << "x"
                                             << conv_in[l].cols << ", out " << conv_out[l].rows
                                             << "x" << conv_out[l].cols << ", kernel "
                                             << conv_w[l].size() << ")\n";
                                        return false;
                                }
                        } else if (!qkv[l].rows || !wo[l].rows) {
                                cerr << "mmcf: layer " << l << " is typed attention but "
                                     << "has no qkv/wo\n";
                                return false;
                        }
                }
        } else {
                for (int l = 0; l < layers; ++l) {
                        if (conv_in[l].rows || conv_out[l].rows) {
                                cerr << "mmcf: conv tensors without the layer-type "
                                     << "metadata (key 14)\n";
                                return false;
                        }
                }
        }
        if (experts > 0) {
                router = std::move(layers_router);
                moe = std::move(layers_moe);
                w1.clear(); w3.clear(); w2.clear(); w13.clear();
        }
        if (getenv("MMCF_DEBUG")) {
                cerr << "MMCF dim=" << dim << " layers=" << layers << " hidden=" << hidden
                     << " heads=" << heads << " kv=" << kv_heads << " vocab=" << vocab
                     << " seq=" << seq << " theta=" << rope_theta << " half=" << rope_half
                     << " ropef=" << rope_factors.size()
                     << " segprefix=" << (sp_prefix ? 1 : 0)
                     << " qknorm=" << qk_norm << " quant_split_=" << quant_split_
                     << " experts=" << experts << " top_k=" << top_k
                     << " moe_hidden=" << moe_hidden << "\n";
                cerr << "MMCF qkv[0]=" << qkv[0].rows << "x" << qkv[0].cols << " scale=" << qkv[0].scale
                     << " prebiased=" << qkv[0].prebiased << " w2[0]=" << w2[0].rows << "x" << w2[0].cols
                     << " logits=" << logits_tiles.size() << " " << logits_tiles[0].rows << "x" << logits_tiles[0].cols
                     << " emb=" << emb_q8.size() << " embs=" << emb_scale.size() << "\n";
                if (is_rwkv())
                        cerr << "MMCF rwkv arch=" << rwkv.arch << " head=" << rwkv.head_size
                             << " r_w/a/v/g=" << rwkv.r_w << "/" << rwkv.r_a << "/"
                             << rwkv.r_v << "/" << rwkv.r_g << " gate=" << rwkv.gate
                             << " eps=" << rwkv.norm_eps
                             << " tok_norm=" << rwkv_tok_norm.size()
                             << " lerp=" << rwkv_lerp.size()
                             << " qkv0=" << qkv[0].rows << "x" << qkv[0].cols
                             << " cm=" << w1[0].rows << "x" << w1[0].cols
                             << " logit_shift=" << logit_shift << "\n";
        }
        // A container written with --head-dedup has no T_LOGITS tiles: a tied
        // head *is* the embedding matrix, whose rows already carry one scale
        // each.  Build the tiles as views into the embedding bytes and keep the
        // per-row plane, which is exactly what the logits kernel's wrow variant
        // reads (matvec_q8_pair_logits with `WS`).
        if (logits_tiles.empty() && has_q6_emb && !emb_scale.empty()) {
                float unit = 0.0f;
                for (int v = 0; v < (int)emb_scale.size() && v < vocab; ++v)
                        if (emb_scale[v] > unit) unit = emb_scale[v];
                if (unit <= 0.0f) { cerr << "mmcf: a dedup head needs embedding scales\n"; return false; }
                for (int st = 0; st < vocab; st += kLogitsGrid) {
                        const int rows = (vocab - st < kLogitsGrid) ? vocab - st : kLogitsGrid;
                        Mat m;
                        m.rows = rows; m.cols = dim;
                        m.type = WeightType::INT6;
                        m.scale = unit;
                        m.ext6_lo = emb_q6_lo.data() + (size_t)st * (dim / 2);
                        m.ext6_hi = emb_q6_hi.data() + (size_t)st * (dim / 4);
                        m.row_scale.assign(emb_scale.begin() + st, emb_scale.begin() + st + rows);
                        for (float& f : m.row_scale) f /= unit;
                        logits_tiles.push_back(std::move(m));
                }
                if (getenv("MMCF_DEBUG"))
                        cerr << "MMCF q6 tiles " << logits_tiles.size() << " unit " << unit
                             << " lo " << emb_q6_lo.size() << " hi " << emb_q6_hi.size() << "\n";
        }
        if (logits_tiles.empty() && has_q8_emb && !emb_scale.empty()) {
                float unit = 0.0f;
                for (int v = 0; v < (int)emb_scale.size() && v < vocab; ++v)
                        if (emb_scale[v] > unit) unit = emb_scale[v];
                if (unit <= 0.0f) {
                        cerr << "mmcf: a dedup head needs embedding scales\n";
                        return false;
                }
                for (int st = 0; st < vocab; st += kLogitsGrid) {
                        const int rows = (vocab - st < kLogitsGrid) ? vocab - st : kLogitsGrid;
                        Mat m;
                        m.rows = rows; m.cols = dim;
                        m.type = WeightType::INT8;
                        m.scale = unit;
                        m.prebiased = false;   // the embedding body stays signed
                        m.ext8 = emb_q8.data() + (size_t)st * dim;
                        m.row_scale.assign(emb_scale.begin() + st,
                                           emb_scale.begin() + st + rows);
                        for (float& f : m.row_scale) f /= unit;
                        logits_tiles.push_back(std::move(m));
                }
        }
        if (!has_q8_emb && !has_i16_emb && !has_q6_emb) {
                cerr << "mmcf: missing required tensors (no embedding)\n";
                return false;
        }
        // The RMSNorm weights are a transformer tensor; an RWKV-7 container
        // carries LayerNorm w+b instead (ids 0x1D-0x1F, 0x22, 0x23) and is
        // checked above.
        if (!is_rwkv() && (int)rms_att.size() != layers * dim) {
                cerr << "mmcf: missing required tensors (rms_att)\n";
                return false;
        }
        if (logits_tiles.empty()) {
                cerr << "mmcf: missing required tensors (no logits tiles)\n";
                return false;
        }
        // The vocabulary tiles are the contract the shaders and the token
        // mapping are written against (src/logits_grid.h).  A file packed with
        // another stride would decode with shifted tokens and no error at all,
        // so refuse it here instead.
        {
                const int want = (vocab + kLogitsGrid - 1) / kLogitsGrid;
                bool fit = (int)logits_tiles.size() == want;
                // Every tile but the last must be exactly one grid wide.  A
                // shorter stride can still produce the right tile *count*
                // (vocab 6400 at 1300 rows is 5 tiles, same as at 1536) and
                // every row fits, but the token mapping would shift by 236 -
                // and a run with such a file printed the correct greedy text
                // and the correct top-k 8 candidate sample, because every chosen token
                // happened to be in the first tile.
                for (size_t i = 0; fit && i < logits_tiles.size(); ++i) {
                        const int rows = logits_tiles[i].rows;
                        fit = (i + 1 == logits_tiles.size()) ? (rows > 0 && rows <= kLogitsGrid)
                                                             : (rows == kLogitsGrid);
                }
                if (!fit) {
                        cerr << "mmcf: vocabulary tiles do not fit the engine's grid: "
                             << logits_tiles.size() << " tiles for vocab " << vocab
                             << ", engine expects " << want << " at " << kLogitsGrid
                             << " rows per tile (all but the last exactly that wide);"
                             << " repack with tools/mmcf_pack.py\n";
                        return false;
                }
        }
        // An RWKV-7 container is runnable since P2: Session::forward_rwkv() is
        // the layer loop for the block, and split_rwkv() has already turned its
        // [W_r; W_k; W_v] into the three Mats that loop draws.  The refusals
        // that remain are the shape checks above -- a file that loads is a file
        // that runs.
        (void)quant_group_;
        return true;
}

bool Model::load_mmcf(){
    if (!parse_header()) return false;
    if (!parse_metadata()) return false;
    return read_tables();
    }

Mat Model::slice_rows(const Mat& src, int r0, int r1){
        Mat m; m.rows = r1 - r0; m.cols = src.cols; m.type = src.type; m.scale = src.scale;
        m.prebiased = src.prebiased;
        // A per-row scale plane follows the rows it belongs to: the FFN gate is
        // drawn from these two halves, and a copy that dropped the plane would
        // silently fall back to one scale for the whole chunk.
        if (!src.row_scale.empty())
                m.row_scale.assign(src.row_scale.begin() + r0, src.row_scale.begin() + r1);
        if (src.type == WeightType::T3) {
                // A row slice is a pointer offset, but only on a *row pair*: one
                // texel carries rows 2t and 2t+1 (docs/checkpoint-container.md,
                // dtype 7), so the body's row t is the texel row and the stride
                // is the texels a row-pair.  Every shape in this model is a
                // multiple of 2, and the FFN chunk is 2048.
                if (r0 % 2) {
                        cerr << "mmcf: a 2-bit body slices on a 2-row boundary (asked "
                                << r0 << ")\n";
                        exit(2);
                }
                m.ext3 = src.w3() + (size_t)(r0 / 2) * (((src.cols + 7) / 8) * 4);
                m.t3_tile_w = m.t3_tile_h = 0;
                return m;
        }
        if (src.type == WeightType::INT8) {
            if (src.ext8) {
                m.ext8 = src.ext8 + (size_t)r0 * m.cols;
            } else {
                m.q8.resize((size_t)m.rows * m.cols);
                memcpy(m.q8.data(), src.q8.data() + (size_t)r0 * m.cols, m.q8.size());
            }
        } else if (src.type == WeightType::INT16) {
            if (src.ext16) {
                m.ext16 = src.ext16 + (size_t)r0 * m.cols;
            } else {
                m.q16.resize((size_t)m.rows * m.cols);
                memcpy(m.q16.data(), src.q16.data() + (size_t)r0 * m.cols, m.q16.size() * 2);
            }
        } else {
            m.group = src.group; m.groups = src.groups; m.q4_param_scale = src.q4_param_scale;
            if (src.ext4) {
                m.ext4 = src.ext4 + (size_t)r0 * (m.cols / 2);
                m.ext_scale4 = src.ext_scale4 + (size_t)r0 * m.groups;
                m.ext_min4 = src.ext_min4 + (size_t)r0 * m.groups;
            } else {
                m.q4.resize((size_t)m.rows * (m.cols / 2));
                memcpy(m.q4.data(), src.q4.data() + (size_t)r0 * (m.cols / 2), m.q4.size());
                m.q4_scale.resize((size_t)m.rows * m.groups);
                memcpy(m.q4_scale.data(), src.q4_scale.data() + (size_t)r0 * m.groups, m.q4_scale.size() * 2);
                m.q4_min.resize((size_t)m.rows * m.groups);
                memcpy(m.q4_min.data(), src.q4_min.data() + (size_t)r0 * m.groups, m.q4_min.size() * 2);
            }
        }
        return m;
    }

Mat Model::slice_cols(const Mat& src, int c0, int c1){
        Mat m; m.rows = src.rows; m.cols = c1 - c0; m.type = src.type; m.scale = src.scale;
        m.prebiased = src.prebiased;
        m.row_scale = src.row_scale;   // a column slice keeps every row
        if (src.type == WeightType::T3) {
                // A column group is one texel, so a slice on an 8-column
                // boundary is a contiguous run of texels *within* a row pair --
                // one memcpy a row pair, the same shape the q8 column slice has.
                if ((c0 % 8) || ((c1 - c0) % 8)) {
                        cerr << "mmcf: a 2-bit body can only be column-sliced on an 8-column "
                                "boundary (" << c0 << ".." << c1 << ")\n";
                        exit(2);
                }
                const size_t src_stride = ((src.cols + 7) / 8) * 4;
                const size_t dst_stride = ((m.cols + 7) / 8) * 4;
                m.ext3 = nullptr;
                m.t3_tile_w = m.t3_tile_h = 0;
                m.q3.resize((size_t)((m.rows + 1) / 2) * dst_stride);
                for (int t = 0; t < (m.rows + 1) / 2; ++t)
                        memcpy(m.q3.data() + (size_t)t * dst_stride,
                               src.w3() + (size_t)t * src_stride + (c0 / 8) * 4, dst_stride);
                return m;
        }
        if (src.type == WeightType::INT8) {
            m.q8.resize((size_t)m.rows * m.cols);
            for (int r = 0; r < m.rows; ++r)
                memcpy(m.q8.data() + (size_t)r * m.cols, src.w8() + (size_t)r * src.cols + c0, m.cols);
        } else if (src.type == WeightType::INT16) {
            m.q16.resize((size_t)m.rows * m.cols);
            for (int r = 0; r < m.rows; ++r)
                memcpy(m.q16.data() + (size_t)r * m.cols, src.w16() + (size_t)r * src.cols + c0, m.cols * 2);
        } else {
            if ((c0 % src.group) != 0 || (m.cols % src.group) != 0)
                throw runtime_error("q4 column slice must be group aligned");
            m.group = src.group; m.groups = m.cols / src.group; m.q4_param_scale = src.q4_param_scale;
            const int g0 = c0 / src.group;
            m.q4.resize((size_t)m.rows * (m.cols / 2), 0);
            for (int r = 0; r < m.rows; ++r) {
                for (int c = 0; c < m.cols; c += 2) {
                    int s0 = c0 + c, s1 = s0 + 1;
                    int b0 = src.w4()[(size_t)r * (src.cols / 2) + s0 / 2];
                    int b1 = src.w4()[(size_t)r * (src.cols / 2) + s1 / 2];
                    int q0 = (s0 & 1) ? (b0 >> 4) : (b0 & 15);
                    int q1 = (s1 & 1) ? (b1 >> 4) : (b1 & 15);
                    m.q4[(size_t)r * (m.cols / 2) + c / 2] = (uint8_t)(q0 | (q1 << 4));
                }
            }
            m.q4_scale.resize((size_t)m.rows * m.groups);
            m.q4_min.resize((size_t)m.rows * m.groups);
            for (int r = 0; r < m.rows; ++r) {
                memcpy(m.q4_scale.data() + (size_t)r * m.groups,
                       src.s4() + (size_t)r * src.groups + g0, m.groups * 2);
                memcpy(m.q4_min.data() + (size_t)r * m.groups,
                       src.m4() + (size_t)r * src.groups + g0, m.groups * 2);
            }
        }
        return m;
    }

Mat Model::quantize(const float* src, int rows, int cols, WeightType type){
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
