#pragma once

#include "common.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class Model {
public:
    int dim = 0, hidden = 0, layers = 0, heads = 0, kv_heads = 0;
    int vocab = 0, seq = 0, kv_dim = 0, head_dim = 0;
    float rope_theta = 10000.0f;
    bool rope_half = false;
    bool shared_embeddings = true;
    bool qk_norm = false;
    uint64_t source_size = 0, source_mtime = 0;

    vector<float> emb_f32, rms_att, rms_ffn, rms_final;
    vector<float> q_norm, k_norm; // [layers * head_dim], only if qk_norm
    vector<int8_t> emb_q8;
    vector<int16_t> emb_i16;
    vector<float> emb_scale;
    bool has_q8_emb = false, has_i16_emb = false;

    vector<Mat> qkv, wo, w13, w1, w3, w2, logits_tiles;
    vector<Mat> w1a, w1b, w3a, w3b, w2a, w2b;
    bool ffn_chunked = false;
    int ffn_chunk = 0; // first hidden chunk; second is hidden - ffn_chunk

    // Filled by the MMCF reader: the tokenizer blob lives inside the file and
    // the chat template / default system prompt come from its metadata.
    const void* tok_blob = nullptr;
    size_t tok_blob_bytes = 0;
    string tok_kind, default_system, template_name;
    int template_kind = -1;   // 0 plain, 1 chatml/qwen, 2 minimind3, 3 tinybrainbot
    // Metadata key 9: the model's own chat template (HuggingFace Jinja text).
    // Empty means "this container predates the key": the reader falls back to
    // template_kind.  `chat_templates` keeps every variant; `chat_template` is
    // the default one.
    string chat_template;
    vector<pair<string, string>> chat_templates;
    // Metadata key 10: whether the template expects a BOS/EOS around the
    // prompt.  -1 = not stated (derive from template_kind, the historical rule).
    int add_bos = -1, add_eos = -1;
    // Metadata key 11: the model's own RMSNorm epsilon.  The shaders used to
    // hardcode 1e-5, which is 10x MiniMind3's (and feng30m's, and supra2's)
    // 1e-6; that was invisible until a model turned out to be sensitive to it.
    float rms_eps = 1e-5f;
    // Metadata key 12: the residual stream's width.  False (or absent) keeps
    // the q16 grid every existing container was packed with.
    bool res_q24 = false;
    bool rms_eps_stated = false;   // false = use the shader's literal 1e-5
    int capabilities = 0;
    bool have_capabilities = false;
    // Special token ids from the container (-1 = absent).  Legacy files have no
    // such metadata and keep the historical 1 = BOS, 2 = EOS guess.
    bool have_specials = false;
    int id_bos = 1, id_eos = 2, id_stop = -1, id_pad = -1, id_unk = -1, id_im_start = -1;
    // (these defaults reproduce the historical 1 = BOS, 2 = EOS behaviour)
    bool load(const string& path);
    // MMCF v2 container (docs/checkpoint-container.md).
    bool load_mmcf();
    ~Model();
    // Drop the mapping once every weight is on the GPU.  The pages stay in the
    // page cache - clean file pages, not process memory - so the next start is
    // still warm, while this process carries no 69 MB of RSS through decode.
    void release_map();

    // Copy any borrowed weights back into their vectors (only the opt-in
    // rewrite paths need that; upload reads the borrowed bytes directly).
    static void materialize(Mat& m);


    void split_ffn();

    static bool quantize_q4(Mat& m);

    bool quantize_q4_weights(const string& scope);

    void apply_precision(const string& layers_spec, const string& parts_spec);


private:
    void read_exact(void* p, size_t n);

    template <typename T> void read_seq(vector<T>& v, const T*& ext, size_t n){
        const size_t bytes = n * sizeof(T);
        if (fpos_ + bytes > fsize_) { cerr << "truncated checkpoint\n"; exit(1); }
        if (borrow_ && (fpos_ % sizeof(T)) == 0) {
            ext = (const T*)(fbase_ + fpos_);
        } else {
            v.resize(n);
            memcpy(v.data(), fbase_ + fpos_, bytes);
        }
        fpos_ += bytes;
    }

    // The checkpoint stays mapped for the process lifetime: borrowed weights
    // point into it.  Those pages are page cache - reclaimable, never swapped -
    // so the next run reads them back from RAM instead of the SD card.
    const uint8_t* fbase_ = nullptr;
    size_t fsize_ = 0, fpos_ = 0;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    bool borrow_ = true;


    static Mat slice_rows(const Mat& src, int r0, int r1);

    static Mat slice_cols(const Mat& src, int c0, int c1);

    static Mat quantize(const float* src, int rows, int cols, WeightType type);
};
