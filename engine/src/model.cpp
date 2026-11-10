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
        bool ok = false;
        if (magic != 0x46434D4DU) {   // "MMCF"
                cerr << "not an MMCF container (magic 0x" << std::hex << magic << std::dec
                     << "); the legacy Q8M1/Q4M1/I16M1/llama2c readers are gone, convert with "
                        "engine/tools/mmcf_pack.py\n";
                munmap(base, (size_t)st.st_size);
                return false;
        }
        ok = load_mmcf();
        fbase_ = nullptr; fsize_ = fpos_ = 0;
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



// MMCF v2 container.  Layout and field list: docs/checkpoint-container.md.
namespace {
struct MmcfRec {
        uint16_t id = 0;
        uint8_t dtype = 0, flags = 0, group = 0;
        uint32_t rows = 0, cols = 0;
        uint64_t off = 0, bytes = 0;
        uint32_t crc = 0;
        float scale = 0.0f;
};

constexpr int MMCF_REC_SIZE = 48;

static uint16_t rd16(const uint8_t* p){ uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd32(const uint8_t* p){ uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const uint8_t* p){ uint64_t v; memcpy(&v, p, 8); return v; }
static float rdf(const uint8_t* p){ float v; memcpy(&v, p, 4); return v; }
}  // namespace

bool Model::load_mmcf(){
        if (fsize_ < 64) { cerr << "mmcf: truncated header\n"; return false; }
        const uint8_t* h = fbase_;
        if (rd16(h + 4) != 2) { cerr << "mmcf: unsupported version " << rd16(h + 4) << "\n"; return false; }
        if (rd16(h + 6) != 64) { cerr << "mmcf: bad header size\n"; return false; }
        const uint32_t n_tensors = rd32(h + 8);
        const uint32_t caps = rd32(h + 12);
        const uint32_t meta_off = rd32(h + 16), meta_size = rd32(h + 20);
        const uint64_t data_off = rd64(h + 24), file_size = rd64(h + 32);
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
        if ((uint64_t)meta_off + meta_size > fsize_ || data_off > fsize_ ||
            data_off < (uint64_t)n_tensors * MMCF_REC_SIZE) { cerr << "mmcf: bad section offsets\n"; return false; }

        bool have_model = false, have_rope = false, have_quant = false;
        int scheme = 1, group = 0, split = 0;
        const uint64_t table_off = data_off - (uint64_t)n_tensors * MMCF_REC_SIZE;
        for (uint64_t p = meta_off; p + 8 <= (uint64_t)meta_off + meta_size; ) {
                const uint16_t key = rd16(fbase_ + p), type = rd16(fbase_ + p + 2);
                const uint32_t len = rd32(fbase_ + p + 4);
                const uint8_t* payload = fbase_ + p + 8;
                if (p + 8 + len > (uint64_t)meta_off + meta_size) { cerr << "mmcf: TLV overruns the block\n"; return false; }
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
                        scheme = payload[0]; group = payload[1]; split = payload[3];
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
                } else if (key == 11 && type == 1 && len >= 4) {
                        float e; memcpy(&e, payload, 4);
                        if (e > 0.0f) { rms_eps = e; rms_eps_stated = true; }
                } else if (key == 10 && type == 1 && len >= 2) {
                        add_bos = payload[0] == 255 ? -1 : (int)payload[0];
                        add_eos = payload[1] == 255 ? -1 : (int)payload[1];
                }
                p += 8 + len;
        }
        if (!have_model) { cerr << "mmcf: no model metadata\n"; return false; }
        if (!have_rope) rope_theta = 10000.0f;
        if (!have_quant) { cerr << "mmcf: no quant metadata\n"; return false; }
        if (getenv("SEQ")) seq = atoi(getenv("SEQ"));
        head_dim = dim / heads;
        kv_dim = kv_heads * head_dim;
        (void)caps;

        // One pass over the table, creating whatever the model needs.  Unknown
        // ids are ignored: that is what lets a future writer add tensors.
        vector<MmcfRec> recs;
        recs.reserve(n_tensors);
        for (uint32_t i = 0; i < n_tensors; ++i) {
                const uint8_t* r = fbase_ + table_off + (uint64_t)i * MMCF_REC_SIZE;
                MmcfRec m;
                m.id = rd16(r); m.dtype = r[2]; m.flags = r[3]; m.group = r[4];
                m.rows = rd32(r + 6); m.cols = rd32(r + 10);
                m.off = rd64(r + 14); m.bytes = rd64(r + 22);
                m.crc = rd32(r + 30); m.scale = rdf(r + 34);
                if (data_off + m.off + m.bytes > fsize_) { cerr << "mmcf: tensor " << i << " overruns the file\n"; return false; }
                recs.push_back(m);
        }

        auto as_f32 = [&](const MmcfRec& r, vector<float>& v) {
                v.resize((size_t)r.rows * r.cols);
                memcpy(v.data(), fbase_ + data_off + r.off, v.size() * 4);
        };
        auto as_i16 = [&](const MmcfRec& r, Mat& m) {
                m.rows = (int)r.rows; m.cols = (int)r.cols;
                m.type = WeightType::INT16;
                m.scale = r.scale;
                const int16_t* p = (const int16_t*)(fbase_ + data_off + r.off);
                if (borrow_) m.ext16 = p;
                else m.q16.assign(p, p + (size_t)r.rows * r.cols);
        };
        auto as_q8 = [&](const MmcfRec& r, Mat& m) {
                m.rows = (int)r.rows; m.cols = (int)r.cols;
                m.type = WeightType::INT8;
                m.scale = r.scale;
                m.prebiased = (r.dtype == 2);
                const int8_t* p = (const int8_t*)(fbase_ + data_off + r.off);
                if (borrow_) m.ext8 = p;
                else { m.q8.assign(p, p + (size_t)r.rows * r.cols); }
        };

        // q4 matrices arrive as three tensors (nibbles, group scale, group min)
        // sharing one id with `part` in the top two bits.
        struct Q4Parts { const MmcfRec* body = nullptr; const MmcfRec* scl = nullptr; const MmcfRec* mn = nullptr; };
        std::map<uint16_t, Q4Parts> q4map;
        // A q8 matrix may carry one f32 scale per output row instead of a
        // single per-matrix scale (part 3).  void.1 needs it: one scale leaves
        // 58-66% of layer 0's FFN weights at zero.
        std::map<uint16_t, const MmcfRec*> rowmap;

        auto as_mat = [&](const MmcfRec& r, Mat& m) { if (r.dtype == 1) as_i16(r, m); else as_q8(r, m); };

        vector<Mat> layers_qkv, layers_wo, layers_w1, layers_w3, layers_w13, layers_w2;
        layers_qkv.resize(layers); layers_wo.resize(layers); layers_w2.resize(layers);
        if (split) { layers_w1.resize(layers); layers_w3.resize(layers); }
        else layers_w13.resize(layers);

        for (const MmcfRec& r : recs) {
                const int base = r.id & 0xFF, layer = (r.id >> 8) & 0x3F, part = r.id >> 14;
                if (part == 3) {   // per-row q8 scale plane
                        rowmap[r.id & 0x3FFF] = &r;
                        continue;
                }
                if (part != 0) {   // q4 scale/min plane
                        Q4Parts& q = q4map[r.id & 0x3FFF];
                        if (part == 1) q.scl = &r; else q.mn = &r;
                        continue;
                }
                if (r.dtype != 0 && r.dtype != 1 && r.dtype != 2 && r.dtype != 3 &&
                    r.dtype != 4 && r.dtype != 5) {
                        cerr << "mmcf: tensor 0x" << std::hex << r.id << std::dec
                             << " has unknown dtype " << (int)r.dtype << "\n";
                        return false;
                }
                if (r.dtype == 4) {   // q4 body: remember it, resolve after the loop
                        q4map[r.id].body = &r;
                        continue;
                }
                switch (base) {
                case 1: {  // embedding (stays signed: the CPU reads it)
                        emb_q8.resize((size_t)r.rows * r.cols);
                        memcpy(emb_q8.data(), fbase_ + data_off + r.off, emb_q8.size());
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
                        memcpy(emb_i16.data(), fbase_ + data_off + r.off, emb_i16.size() * 2);
                        has_i16_emb = true;
                        break;
                case 0x10: if (layer < layers) as_mat(r, layers_qkv[layer]); break;
                case 0x11: if (layer < (int)layers_wo.size()) as_mat(r, layers_wo[layer]); break;
                case 0x12: if (layer < (int)layers_w1.size()) as_mat(r, layers_w1[layer]); break;
                case 0x13: if (layer < (int)layers_w3.size()) as_mat(r, layers_w3[layer]); break;
                case 0x14: if (layer < (int)layers_w13.size()) as_mat(r, layers_w13[layer]); break;
                case 0x15: if (layer < (int)layers_w2.size()) as_mat(r, layers_w2[layer]); break;
                case 0x20: { Mat m; as_mat(r, m); logits_tiles.push_back(m); break; }
                case 0x30:   // tokenizer blob, borrowed from the mapping
                        tok_blob = fbase_ + data_off + r.off;
                        tok_blob_bytes = (size_t)r.bytes;
                        break;
                default: break;  // tokenizer blob, template, unknown ids
                }
        }
        for (auto& kv : rowmap) {
                const MmcfRec& r = *kv.second;
                const int kind = r.id & 0xFF, layer = (r.id >> 8) & 0x3F;
                Mat* m = nullptr;
                if (kind == 0x10 && layer < (int)layers_qkv.size()) m = &layers_qkv[layer];
                else if (kind == 0x11 && layer < (int)layers_wo.size()) m = &layers_wo[layer];
                else if (kind == 0x12 && layer < (int)layers_w1.size()) m = &layers_w1[layer];
                else if (kind == 0x13 && layer < (int)layers_w3.size()) m = &layers_w3[layer];
                else if (kind == 0x14 && layer < (int)layers_w13.size()) m = &layers_w13[layer];
                else if (kind == 0x15 && layer < (int)layers_w2.size()) m = &layers_w2[layer];
                else if (kind == 0x20 && layer < (int)logits_tiles.size()) m = &logits_tiles[layer];
                if (!m || m->rows != (int)r.rows || r.bytes != (size_t)r.rows * 4) {
                        cerr << "mmcf: row scale plane 0x" << std::hex << r.id << std::dec
                             << " does not match its matrix\n";
                        return false;
                }
                // A fused W13 (id 0x14, split=0) is the one matrix whose two
                // rows per fragment sit rows apart, so its plane is paired the
                // way the kernel reads it.
                m->row_scale_hi = (kind == 0x14 && !split);
                const float* p = (const float*)(fbase_ + data_off + r.off);
                m->row_scale.assign(p, p + r.rows);
        }
        row_scale = !rowmap.empty();
        for (auto& kv : q4map) {
                const uint16_t id = kv.first;
                Q4Parts& q = kv.second;
                if (!q.body || !q.scl || !q.mn) {
                        cerr << "mmcf: incomplete q4 tensor 0x" << std::hex << id << std::dec << "\n";
                        return false;
                }
                Mat m;
                m.rows = (int)q.body->rows; m.cols = (int)q.body->cols;
                m.type = WeightType::INT4;
                m.group = 32;
                m.groups = (int)q.scl->cols;   // groups per row, not the plane size
                m.q4_param_scale = q.body->scale;
                if (!borrow_) {
                        m.q4.assign((const uint8_t*)(fbase_ + data_off + q.body->off),
                                    (const uint8_t*)(fbase_ + data_off + q.body->off) + q.body->bytes);
                        m.q4_scale.assign((const int16_t*)(fbase_ + data_off + q.scl->off),
                                          (const int16_t*)(fbase_ + data_off + q.scl->off) + m.groups);
                        m.q4_min.assign((const int16_t*)(fbase_ + data_off + q.mn->off),
                                        (const int16_t*)(fbase_ + data_off + q.mn->off) + m.groups);
                } else {
                        m.ext4 = (const uint8_t*)(fbase_ + data_off + q.body->off);
                        m.ext_scale4 = (const int16_t*)(fbase_ + data_off + q.scl->off);
                        m.ext_min4 = (const int16_t*)(fbase_ + data_off + q.mn->off);
                }
                const int kind = id & 0xFF, layer = (id >> 8) & 0x3F;
                // only the vector that exists for this file's `split` flag is
                // sized, so index by its own size (an empty vector would be UB)
                if (kind == 0x10 && layer < (int)layers_qkv.size()) layers_qkv[layer] = m;
                else if (kind == 0x11 && layer < (int)layers_wo.size()) layers_wo[layer] = m;
                else if (kind == 0x12 && layer < (int)layers_w1.size()) layers_w1[layer] = m;
                else if (kind == 0x13 && layer < (int)layers_w3.size()) layers_w3[layer] = m;
                else if (kind == 0x14 && layer < (int)layers_w13.size()) layers_w13[layer] = m;
                else if (kind == 0x15 && layer < (int)layers_w2.size()) layers_w2[layer] = m;
        }
        for (int l = 0; l < layers; ++l) {
                qkv.push_back(layers_qkv[l]);
                wo.push_back(layers_wo[l]);
                if (split) { w1.push_back(layers_w1[l]); w3.push_back(layers_w3[l]); }
                else w13.push_back(layers_w13[l]);
                w2.push_back(layers_w2[l]);
        }
        if (getenv("MMCF_DEBUG")) {
                cerr << "MMCF dim=" << dim << " layers=" << layers << " hidden=" << hidden
                     << " heads=" << heads << " kv=" << kv_heads << " vocab=" << vocab
                     << " seq=" << seq << " theta=" << rope_theta << " half=" << rope_half
                     << " qknorm=" << qk_norm << " split=" << split << "\n";
                cerr << "MMCF qkv[0]=" << qkv[0].rows << "x" << qkv[0].cols << " scale=" << qkv[0].scale
                     << " prebiased=" << qkv[0].prebiased << " w2[0]=" << w2[0].rows << "x" << w2[0].cols
                     << " logits=" << logits_tiles.size() << " " << logits_tiles[0].rows << "x" << logits_tiles[0].cols
                     << " emb=" << emb_q8.size() << " embs=" << emb_scale.size() << "\n";
        }
        if ((!has_q8_emb && !has_i16_emb) || (int)rms_att.size() != layers * dim || logits_tiles.empty()) {
                cerr << "mmcf: missing required tensors\n";
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
        (void)group;
        return true;
    }

Mat Model::slice_rows(const Mat& src, int r0, int r1){
        Mat m; m.rows = r1 - r0; m.cols = src.cols; m.type = src.type; m.scale = src.scale;
        m.prebiased = src.prebiased;
        // A per-row scale plane follows the rows it belongs to: the FFN gate is
        // drawn from these two halves, and a copy that dropped the plane would
        // silently fall back to one scale for the whole chunk.
        if (!src.row_scale.empty())
                m.row_scale.assign(src.row_scale.begin() + r0, src.row_scale.begin() + r1);
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
