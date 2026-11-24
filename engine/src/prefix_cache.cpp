// The disk prefix cache: one KV blob per chat header, plus the sidecar that says
// what it is.
//
// Split out of session.cpp (2026-11-14): it is a self-contained unit -- serialise
// the KV targets, validate a header, write the sidecar -- and session.cpp had
// grown past a thousand lines with the model forward in it.  These are still
// Session member functions in this translation unit, so no API and no header
// changed, and `shader_hashes.sh` proves the emitted GLSL did not move.
//
// The header carries the model identity (container size and mtime), the shape,
// the rope base, the KV packing and the scale, so a cache built for another
// container is refused instead of loaded.  The sidecar adds the hashes of the
// inputs (system prompt, tool list) that the engine cannot check at load time.

#include "session.h"
#include "jinja/json.h"   // the cache sidecar: JSON is built, never concatenated
#include "xformat.h"      // x_format_name() in the sidecar

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

bool Session::save_prefix_cache(const string& path, const vector<int>& toks,
                               uint64_t system_fnv, uint64_t tools_fnv){
        // A hybrid model's cache carries the conv history too (version 4, flag
        // bit 32); without it a reload would resume from a state that never
        // existed.  That needs the GPU conv path, which owns the state textures.
        if (m_.hybrid && (!gpu_layer_ || gpu_.conv_state_row_bytes() == 0)) return false;
        if (!gpu_attn_ || gpu_.kv_cache_.empty() || toks.empty() || (int)toks.size() > m_.seq) return false;
        ofstream f(path, ios::binary);
        if (!f) return false;
        const char magic[8] = {'M','M','K','V','1','\0','\0','\0'};
        f.write(magic, 8);
        auto u32 = [&](uint32_t v) { f.write((char*)&v, 4); };
        auto u64 = [&](uint64_t v) { f.write((char*)&v, 8); };
        // Version 3 adds an optional per-head V step plane (8-bit KV with
        // KV_VSTEP); without it the file is byte-for-byte a version 2 file
        // with a new version number, so the default path keeps writing 2.
        const int vsw = gpu_.kv_.vstep ? gpu_.att_kv_heads_ : 0;
        u32(m_.hybrid ? 4u : vsw > 0 ? 3u : 2u);
        u64(m_.source_size); u64(m_.source_mtime);
        u32(m_.dim); u32(m_.hidden); u32(m_.layers); u32(m_.heads);
        u32(m_.kv_heads); u32(m_.vocab); u32(m_.seq); u32(m_.head_dim);
        f.write((char*)&m_.rope_theta, 4);
        // Bits 64/128 name the KV *codec* (bf16, f32).  Without them bf16 and
        // q16 write the same flags, scale and width (both 16/2048/kv_dim), so a
        // cache built under one was accepted under the other and decoded on the
        // wrong grid.  Bit 128 also separates f32 (scale is only a dump label)
        // from i24 when KV_GRID happens to meet at 512.
        u32((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
            (gpu_.kv_.bits == 16 ? 4u : gpu_.kv_.bits == 8 ? 8u : 0u) |
            (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u) |
            (m_.hybrid ? 32u : 0u) |
            (gpu_.kv_.codec == KvCodec::Bf16 ? 64u : gpu_.kv_.codec == KvCodec::F32 ? 128u : 0u));
        u32((uint32_t)toks.size());
        u32((uint32_t)gpu_.kv_.texels);
        // The field exists in every version >= 3 file, so a version 4 file
        // (hybrid) writes it even when the V-step plane is absent -- otherwise
        // the load's read shifts and the whole header mismatches.
        if (m_.hybrid || vsw > 0) u32((uint32_t)vsw);
        f.write((char*)&gpu_.kv_.scale, 4);
        for (int t : toks) u32((uint32_t)t);
        gpu_.sync();
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        const int width = gpu_.kv_.texels, n = (int)toks.size();
        vector<uint8_t> data((size_t)width * n * 4);
        for (int l = 0; l < m_.layers; ++l) {
            glBindFramebuffer(GL_FRAMEBUFFER, gpu_.kv_cache_[l].fbo);
            glReadPixels(0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            f.write((char*)data.data(), data.size());
        }
        // The conv history: two q16 rows per conv layer (version 4 only).
        if (m_.hybrid) {
            vector<uint8_t> st;
            for (int l = 0; l < m_.layers; ++l) {
                if (!m_.is_conv(l)) continue;
                for (int which = 0; which < 2; ++which) {
                    if (!gpu_.read_conv_state(l, which, st)) return false;
                    f.write((char*)st.data(), st.size());
                }
            }
        }
        // 8-bit per-head V steps: without this plane a reloaded cache would
        // dequantise every V row on the wrong grid, so the file must carry it.
        for (int l = 0; l < m_.layers && vsw > 0; ++l) {
            glBindFramebuffer(GL_FRAMEBUFFER, gpu_.vs_targets_[l].fbo);
            glReadPixels(0, 0, vsw, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            f.write((char*)data.data(), (size_t)vsw * n * 4);
        }
        gpu_.forget_state();
        if (!f) return false;
        // The sidecar says what this file *is*, written by the only component
        // that knows: the engine holds the KV layout, the packing flags, the
        // model identity (size and mtime of the container) and the token list.
        // A rebuild trigger (the server, a run script) compares these fields
        // instead of re-deriving them, which is what the old Python builder did
        // -- a second renderer and a second set of hashes.
        {
            uint64_t fnv = 1469598103934665603ull;
            for (int tok : toks) {
                const uint32_t v = (uint32_t) tok;
                for (int b = 0; b < 4; ++b) {
                    fnv ^= (uint8_t) (v >> (8 * b));
                    fnv *= 1099511628211ull;
                }
            }
            char hex[32];
            snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) fnv);
            common_json meta = common_json::object();
            meta["format"] = string("mmkv");
            meta["version"] = (long long) (m_.hybrid ? 4 : vsw > 0 ? 3 : 2);
            meta["source_size"] = (long long) m_.source_size;
            meta["source_mtime"] = (long long) m_.source_mtime;
            meta["dim"] = m_.dim;       meta["hidden"] = m_.hidden;
            meta["layers"] = m_.layers; meta["heads"] = m_.heads;
            meta["kv_heads"] = m_.kv_heads; meta["vocab"] = m_.vocab;
            meta["seq"] = m_.seq;       meta["head_dim"] = m_.head_dim;
            meta["rope_theta"] = (double) m_.rope_theta;
            meta["kv_bits"] = gpu_.kv_.bits;
            meta["kv_fmt"] = (long long) gpu_.kv_.codec;
            meta["act_fmt"] = string(x_format_name(gpu_.act_fmt_));
            meta["hybrid"] = m_.hybrid;
            meta["kv_scale"] = (double) gpu_.kv_.scale;
            meta["kv_width"] = gpu_.kv_.texels;
            meta["vstep"] = vsw;
            meta["n_tokens"] = (long long) toks.size();
            meta["tokens_fnv64"] = string(hex);
            char h2[32];
            snprintf(h2, sizeof(h2), "%016llx", (unsigned long long) system_fnv);
            meta["system_fnv64"] = string(h2);
            snprintf(h2, sizeof(h2), "%016llx", (unsigned long long) tools_fnv);
            meta["tools_fnv64"] = string(h2);
            meta["template"] = m_.template_name;
            ofstream sm(path + ".json");
            if (sm) sm << meta.dump_safe() << "\n";
        }
        return true;
    }

bool Session::load_prefix_cache(const string& path, vector<int>& toks){
        if (m_.hybrid && (!gpu_layer_ || gpu_.conv_state_row_bytes() == 0)) return false;
        if (!gpu_attn_ || gpu_.kv_cache_.empty()) return false;
        ifstream f(path, ios::binary);
        if (!f) return false;
        char magic[8] {};
        f.read(magic, 8);
        if (!f || memcmp(magic, "MMKV1", 5) != 0) return false;
        auto u32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
        auto u64 = [&](uint64_t& v) { f.read((char*)&v, 8); };
        uint32_t version = 0, dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
        uint32_t vocab = 0, seq = 0, head_dim = 0, flags = 0, n = 0, width = 0, vsw = 0;
        uint64_t size = 0, mtime = 0; float theta = 0, kscale = 0;
        u32(version); u64(size); u64(mtime);
        u32(dim); u32(hidden); u32(layers); u32(heads); u32(kv_heads);
        u32(vocab); u32(seq); u32(head_dim);
        f.read((char*)&theta, 4);
        u32(flags); u32(n); u32(width);
        if (version >= 3) u32(vsw);   // per-head V step plane present
        f.read((char*)&kscale, 4);
        if (!f || (version != 2 && version != 3 && version != 4) || dim != (uint32_t)m_.dim ||
            hidden != (uint32_t)m_.hidden ||
            layers != (uint32_t)m_.layers || heads != (uint32_t)m_.heads ||
            kv_heads != (uint32_t)m_.kv_heads || vocab != (uint32_t)m_.vocab ||
            seq != (uint32_t)m_.seq || head_dim != (uint32_t)m_.head_dim ||
            size != m_.source_size || mtime != m_.source_mtime ||
            theta != m_.rope_theta || width != (uint32_t)gpu_.kv_.texels ||
            vsw != (uint32_t)(gpu_.kv_.vstep ? gpu_.att_kv_heads_ : 0) ||
            n == 0 || n > (uint32_t)m_.seq ||
            flags != ((m_.qk_norm ? 1u : 0u) | (m_.rope_half ? 2u : 0u) |
                      (gpu_.kv_.bits == 16 ? 4u : gpu_.kv_.bits == 8 ? 8u : 0u) |
                      (gpu_.act_fmt_ == XFormat::I8_4 ? 16u : 0u) |
                      (m_.hybrid ? 32u : 0u) |
                      (gpu_.kv_.codec == KvCodec::Bf16 ? 64u : gpu_.kv_.codec == KvCodec::F32 ? 128u : 0u))) return false;
        if (memcmp(&kscale, &gpu_.kv_.scale, sizeof(float)) != 0) {
            cerr << "PREFIX_CACHE KV_GRID mismatch: cache " << kscale
                 << ", process " << gpu_.kv_.scale << endl;
            return false;
        }
        vector<int> tmp(n);
        for (uint32_t i = 0; i < n; ++i) { uint32_t v = 0; u32(v); tmp[i] = (int)v; }
        if (!f) return false;
        vector<uint8_t> data((size_t)width * n * 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        for (int l = 0; l < m_.layers; ++l) {
            f.read((char*)data.data(), data.size());
            if (!f) return false;
            glBindTexture(GL_TEXTURE_2D, gpu_.kv_cache_[l].tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        }
        if (vsw > 0) {
            for (int l = 0; l < m_.layers; ++l) {
                f.read((char*)data.data(), (size_t)vsw * n * 4);
                if (!f) return false;
                glBindTexture(GL_TEXTURE_2D, gpu_.vs_targets_[l].tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vsw, n, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
            }
        }
        if (version == 4) {
            vector<uint8_t> st((size_t)gpu_.conv_state_row_bytes());
            for (int l = 0; l < m_.layers; ++l) {
                if (!m_.is_conv(l)) continue;
                for (int which = 0; which < 2; ++which) {
                    f.read((char*)st.data(), st.size());
                    if (!f) return false;
                    if (!gpu_.write_conv_state(l, which, st)) return false;
                }
            }
        }
        gpu_.forget_state();
        gpu_.sync();
        toks = std::move(tmp);
        return true;
    }
