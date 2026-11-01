#pragma once

#include "common.h"
#include <fstream>
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

    bool load(const string& path);

    void split_ffn();

    static bool quantize_q4(Mat& m);

    bool quantize_q4_weights(const string& scope);

    void apply_precision(const string& layers_spec, const string& parts_spec);

private:
    static void read_exact(ifstream& f, void* p, size_t n);

    Mat read_mat(ifstream& f, bool int16);

    bool load_legacy(const string& path);

    bool load_q4(const string& path);

    bool load_q8(const string& path);

    bool load_i16(const string& path);

    static Mat slice_rows(const Mat& src, int r0, int r1);

    static Mat slice_cols(const Mat& src, int c0, int c1);

    static Mat quantize(const float* src, int rows, int cols, WeightType type);
};
