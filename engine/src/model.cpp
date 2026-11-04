#include "model.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

// first hidden chunk; second is hidden - ffn_chunk

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
        void* base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) { cerr << "mmap failed: " << path << "\n"; return false; }
        source_size = (uint64_t)st.st_size;
        source_mtime = (uint64_t)st.st_mtime;
        fbase_ = (const uint8_t*)base;
        fsize_ = (size_t)st.st_size;
        // Start the SD read now: it overlaps the EGL/Mesa init that follows, and
        // a warm start then finds every page already in the page cache.
        madvise(base, (size_t)st.st_size, MADV_WILLNEED);
        fpos_ = 0;
        borrow_ = !(getenv("MODEL_BORROW") && !strcmp(getenv("MODEL_BORROW"), "0"));
        uint32_t magic = 0; memcpy(&magic, fbase_, 4);
        bool ok;
        if (magic == MAGIC_Q8 || magic == MAGIC_Q8_V2) ok = load_q8(path);
        else if (magic == MAGIC_I16 || magic == MAGIC_I16_V2) ok = load_i16(path);
        else if (magic == MAGIC_Q4) ok = load_q4(path);
        else ok = load_legacy(path);
        fbase_ = nullptr; fsize_ = fpos_ = 0;
        if (borrow_ && ok) { map_ = base; map_size_ = (size_t)st.st_size; }
        else munmap(base, (size_t)st.st_size);
        return ok;
    }

Model::~Model(){
    if (map_) munmap(map_, map_size_);
    map_ = nullptr; map_size_ = 0;
}
// After the upload the whole checkpoint is on the GPU.  Unmapping it drops the
// resident file pages from this process - they stay in the page cache, so the
// next start is still warm - and clears the borrowed pointers so nothing can
// read a stale view of a mapping that no longer exists.
void Model::release_map(){
    if (map_) munmap(map_, map_size_);
    map_ = nullptr; map_size_ = 0;
    auto clear = [](vector<Mat>& v){
        for (auto& m : v) {
            m.ext8 = nullptr; m.ext16 = nullptr; m.ext4 = nullptr;
            m.ext_scale4 = nullptr; m.ext_min4 = nullptr;
        }
    };
    clear(qkv); clear(wo); clear(w13); clear(w1); clear(w3); clear(w2); clear(logits_tiles);
    clear(w1a); clear(w1b); clear(w3a); clear(w3b); clear(w2a); clear(w2b);
}

void Model::materialize(Mat& m){
    if (m.ext8) { m.q8.assign(m.ext8, m.ext8 + (size_t)m.rows * m.cols); m.ext8 = nullptr; }
    if (m.ext16) { m.q16.assign(m.ext16, m.ext16 + (size_t)m.rows * m.cols); m.ext16 = nullptr; }
    if (m.ext4) { m.q4.assign(m.ext4, m.ext4 + (size_t)m.rows * (m.cols / 2)); m.ext4 = nullptr; }
    if (m.ext_scale4) { m.q4_scale.assign(m.ext_scale4, m.ext_scale4 + (size_t)m.rows * m.groups); m.ext_scale4 = nullptr; }
    if (m.ext_min4) { m.q4_min.assign(m.ext_min4, m.ext_min4 + (size_t)m.rows * m.groups); m.ext_min4 = nullptr; }
}

void Model::split_ffn(){
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

void Model::read_exact(void* p, size_t n){
        if (n > fsize_ - fpos_) { cerr << "truncated checkpoint\n"; exit(1); }
        memcpy(p, fbase_ + fpos_, n);
        fpos_ += n;
    }

Mat Model::read_mat(bool int16){
        Mat m;
        int32_t rows = 0, cols = 0; float scale = 0;
        read_exact(&rows, 4); read_exact(&cols, 4); read_exact(&scale, 4);
        m.rows = rows; m.cols = cols; m.scale = scale;
        m.type = int16 ? WeightType::INT16 : WeightType::INT8;
        if (rows < 0 || cols <= 0) { cerr << "bad matrix header\n"; exit(1); }
        const size_t n = (size_t)rows * cols;
        if (int16) read_seq(m.q16, m.ext16, n);
        else read_seq(m.q8, m.ext8, n);
        return m;
    }
bool Model::load_legacy(const string& path){
        const uint8_t* base = fbase_;
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
        // Legacy fp32 files have no precision field.  Quantizing them to int8
        // destroys the tiny 260K/15M TinyStories models (empirically: coherent
        // from I16, gibberish from Q8).  Default to I16; LEGACY_Q8=1 keeps the
        // old low-memory path for large checkpoints.
        const bool legacy_q8 = getenv("LEGACY_Q8") && strcmp(getenv("LEGACY_Q8"), "0") != 0;
        const WeightType qt = legacy_q8 ? WeightType::INT8 : WeightType::INT16;

        for (int l = 0; l < layers; ++l) {
            vector<float> fused((size_t)(dim + 2 * kv_dim) * dim);
            memcpy(fused.data(), wq + (size_t)l * dim * dim, (size_t)dim * dim * 4);
            memcpy(fused.data() + (size_t)dim * dim, wk + (size_t)l * kv_dim * dim, (size_t)kv_dim * dim * 4);
            memcpy(fused.data() + (size_t)(dim + kv_dim) * dim, wv + (size_t)l * kv_dim * dim, (size_t)kv_dim * dim * 4);
            qkv.push_back(quantize(fused.data(), dim + 2 * kv_dim, dim, qt));
            wo.push_back(quantize(wo_p + (size_t)l * dim * dim, dim, dim, qt));
            if (2 * hidden <= 2048) {
                vector<float> fused13((size_t)2 * hidden * dim);
                memcpy(fused13.data(), w1_p + (size_t)l * hidden * dim, (size_t)hidden * dim * 4);
                memcpy(fused13.data() + (size_t)hidden * dim, w3_p + (size_t)l * hidden * dim, (size_t)hidden * dim * 4);
                w13.push_back(quantize(fused13.data(), 2 * hidden, dim, qt));
            } else {
                w1.push_back(quantize(w1_p + (size_t)l * hidden * dim, hidden, dim, qt));
                w3.push_back(quantize(w3_p + (size_t)l * hidden * dim, hidden, dim, qt));
            }
            w2.push_back(quantize(w2_p + (size_t)l * dim * hidden, dim, hidden, qt));
        }
        for (int st = 0; st < vocab; st += 1536)
            logits_tiles.push_back(quantize(wcls + (size_t)st * dim, min(1536, vocab - st), dim, qt));

        return true;
    }

bool Model::load_q4(const string& path){
        uint32_t magic = 0; int32_t cfg[7]; float theta = 0; int32_t half = 0, split = 0, qk = 0;
        read_exact(&magic, 4); read_exact(cfg, 28); read_exact(&theta, 4);
        read_exact(&half, 4); read_exact(&split, 4); read_exact(&qk, 4);
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        rope_theta = theta; rope_half = half != 0; qk_norm = qk != 0;
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        has_q8_emb = true;
        emb_q8.resize((size_t)vocab * dim); read_exact(emb_q8.data(), emb_q8.size());
        emb_scale.resize(vocab); read_exact(emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(k_norm.data(), k_norm.size() * 4);
        }
        auto read_q4_mat = [&]() {
            int32_t rows = 0, cols = 0, groups = 0; float ps = 1.0f;
            read_exact(&rows, 4); read_exact(&cols, 4);
            read_exact(&groups, 4); read_exact(&ps, 4);
            Mat m; m.rows = rows; m.cols = cols; m.type = WeightType::INT4;
            m.group = 32; m.groups = groups; m.q4_param_scale = ps;
            read_seq(m.q4, m.ext4, (size_t)rows * (cols / 2));
            read_seq(m.q4_scale, m.ext_scale4, (size_t)rows * groups);
            read_seq(m.q4_min, m.ext_min4, (size_t)rows * groups);
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
            read_exact(&rows, 4); read_exact(&cols, 4); read_exact(&sc, 4);
            Mat m; m.rows = rows; m.cols = cols; m.scale = sc; m.type = WeightType::INT8;
            read_seq(m.q8, m.ext8, (size_t)rows * cols);
            logits_tiles.push_back(m);
        }
        return true;
    }

bool Model::load_q8(const string& path){
        uint32_t magic = 0; int32_t cfg[7]; int32_t split = 0;
        read_exact(&magic, 4); read_exact(cfg, 28);
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        if (magic == MAGIC_Q8_V2) {
            int32_t half = 0, qk = 0;
            read_exact(&rope_theta, 4); read_exact(&half, 4);
            read_exact(&split, 4); read_exact(&qk, 4);
            rope_half = half != 0; qk_norm = qk != 0;
        } else {
            read_exact(&split, 4);
            if (getenv("ROPE_THETA")) rope_theta = atof(getenv("ROPE_THETA"));
            rope_half = getenv("ROPE_HALF") != nullptr;
        }
        has_q8_emb = true;
        emb_q8.resize((size_t)vocab * dim);
        read_exact(emb_q8.data(), emb_q8.size());
        emb_scale.resize(vocab); read_exact(emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(k_norm.data(), k_norm.size() * 4);
        }
        for (int l = 0; l < layers; ++l) {
            qkv.push_back(read_mat(false));
            wo.push_back(read_mat(false));
            if (split) { w1.push_back(read_mat(false)); w3.push_back(read_mat(false)); }
            else w13.push_back(read_mat(false));
            w2.push_back(read_mat(false));
        }
        int nt = (vocab + 1535) / 1536;
        for (int i = 0; i < nt; ++i) logits_tiles.push_back(read_mat(false));
        return true;
    }

bool Model::load_i16(const string& path){
        uint32_t magic = 0; int32_t cfg[7]; float theta = 0; int32_t half = 0, split = 0;
        read_exact(&magic, 4); read_exact(cfg, 28); read_exact(&theta, 4);
        read_exact(&half, 4); read_exact(&split, 4);
        if (magic == MAGIC_I16_V2) {
            int32_t qk = 0; read_exact(&qk, 4); qk_norm = qk != 0;
        }
        dim = cfg[0]; hidden = cfg[1]; layers = cfg[2]; heads = cfg[3];
        kv_heads = cfg[4]; vocab = cfg[5]; seq = cfg[6];
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        rope_theta = theta; rope_half = half != 0;
        head_dim = dim / heads; kv_dim = kv_heads * head_dim;
        has_i16_emb = true;
        emb_i16.resize((size_t)vocab * dim);
        read_exact(emb_i16.data(), emb_i16.size() * 2);
        emb_scale.resize(vocab); read_exact(emb_scale.data(), emb_scale.size() * 4);
        rms_att.resize((size_t)layers * dim); read_exact(rms_att.data(), rms_att.size() * 4);
        rms_ffn.resize((size_t)layers * dim); read_exact(rms_ffn.data(), rms_ffn.size() * 4);
        rms_final.resize(dim); read_exact(rms_final.data(), rms_final.size() * 4);
        if (qk_norm) {
            q_norm.resize((size_t)layers * head_dim); read_exact(q_norm.data(), q_norm.size() * 4);
            k_norm.resize((size_t)layers * head_dim); read_exact(k_norm.data(), k_norm.size() * 4);
        }
        for (int l = 0; l < layers; ++l) {
            qkv.push_back(read_mat(true));
            wo.push_back(read_mat(true));
            if (split) { w1.push_back(read_mat(true)); w3.push_back(read_mat(true)); }
            else w13.push_back(read_mat(true));
            w2.push_back(read_mat(true));
        }
        int nt = (vocab + 1535) / 1536;
        for (int i = 0; i < nt; ++i) logits_tiles.push_back(read_mat(true));
        return true;
    }

Mat Model::slice_rows(const Mat& src, int r0, int r1){
        Mat m; m.rows = r1 - r0; m.cols = src.cols; m.type = src.type; m.scale = src.scale;
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
