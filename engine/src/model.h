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
    // Metadata key 18: a `longrope` model's *static* per-dimension frequency
    // table (head_dim/2 entries, pair i divides pair i's frequency).  Empty
    // means plain theta, which is every container packed before 2026-11-18.
    vector<float> rope_factors;
    bool shared_embeddings = true;
    bool qk_norm = false;
    // The container carries per-row q8 scale planes (tensor id part 3).  The
    // two-position prefill kernels have no wrow variant yet, so the session
    // falls back to the single-position path when this is set.
    bool row_scale = false;
    uint64_t source_size = 0, source_mtime = 0;

    vector<float> emb_f32, rms_att, rms_ffn, rms_final;
    vector<float> q_norm, k_norm; // [layers * head_dim], only if qk_norm
    vector<int8_t> emb_q8;
    // A 6-bit tied head (dtype 6): low nibble plane and high 2-bit plane, both
    // row-major.  The logits are rendered from these rows (dedup).
    vector<uint8_t> emb_q6_lo, emb_q6_hi;
    bool has_q6_emb = false;
    vector<int16_t> emb_i16;
    vector<float> emb_scale;
    bool has_q8_emb = false, has_i16_emb = false;

    vector<Mat> qkv, wo, w13, w1, w3, w2, logits_tiles;
    vector<Mat> w1a, w1b, w3a, w3b, w2a, w2b;
    bool ffn_chunked = false;
    int ffn_chunk = 0; // first hidden chunk; second is hidden - ffn_chunk

    // Mixture of experts (MMCF v3, metadata key 13).  `hidden` stays the
    // config's dense FFN width; every layer's FFN is one of `experts` blocks of
    // width `moe_hidden`, selected per token by `router[l]` (rows = experts),
    // top-1.  A container is either all-dense or all-MoE: a mixed block has no
    // representation here.
    int experts = 0, top_k = 0, moe_hidden = 0;
    bool moe_norm_topk = false;
    struct MoeExpert {
        Mat w1, w3, w2;        // moe_hidden <= 2048: the whole expert FFN
        Mat w1a, w3a, w2a;     // chunk 0 of a chunked expert FFN
        Mat w1b, w3b, w2b;     // chunk 1
    };
    vector<Mat> router;                // [layers]
    vector<vector<MoeExpert>> moe;     // [layers][experts]

    // RWKV-7 (MMCF v4, metadata key 16, docs/checkpoint-container.md).  It is
    // not a transformer block: no qkv semantics, no RoPE, no KV cache, one
    // 64x64 recurrence per head per layer and two token-shift vectors per
    // layer.  The reader fills these; the layer loop that consumes them is
    // session.cpp's (docs/rwkv7.md).  The four big matrices
    // reuse the dense fields on purpose -- `qkv` is [W_r; W_k; W_v] (3*dim x
    // dim), `wo` is W_o, `w1` the channel mix's key matrix (hidden x dim) and
    // `w2` its value matrix (dim x hidden) -- because their ids are the dense
    // ones, so the q4 and per-row-scale machinery needs no second path.
    struct Rwkv {
        int arch = 0;             // 0 = not RWKV, 1 = rwkv7 (key 16's arch)
        int head_size = 0;        // the WKV head width (64)
        int r_w = 0, r_a = 0, r_v = 0, r_g = 0;   // the four low ranks
        bool gate = false;        // the time-mix gate branch is present
        // LayerNorm epsilon from the file.  The per-head group norm keeps the
        // architecture's own 64e-5: it is not a parameter of the model.
        float norm_eps = 1e-5f;
    };
    // Version 5: an encoder-decoder.  The encoder block is bidirectional and
    // has no cross-attention; the decoder's blocks are post-norm and carry one.
    // The bias vectors and the position table are f32 rows because a quantised
    // LayerNorm weight is a systematic per-channel gain, and because there are
    // only 60 KiB of them (docs/checkpoint-container.md, "v5").
    int enc_layers = 0, enc_ffn = 0, enc_seq = 0;
    bool enc_post_norm = false, enc_scale_embed = false, enc_nogate = false;
    // The uploaded views: the reader fills these after validating the shapes,
    // so the session never iterates a half-populated stack.
    vector<Mat> u_enc_qkv, u_enc_wo, u_enc_w1, u_enc_w2;   // [enc_layers]
    vector<Mat> u_cross_q, u_cross_k, u_cross_v, u_cross_o; // [layers]
    int bias_dim_rows = 0, bias_ffn_rows = 0;       // metadata key 21
    vector<float> bias_dim, bias_ffn;               // rows x dim, rows x enc_ffn
    vector<float> logits_bias;                      // [vocab], added on the host
    vector<float> pos_table;                        // [pos_rows * dim]
    int pos_rows = 0;
    // The dim-wide row a block role lives at.  The order is the packer's
    // `mmcf_format.BIAS_DIM_ORDER`: every decoder layer first (15 rows), then
    // every encoder layer (9), and the two counts in key 21 are checked against
    // these formulas at load, so a row index cannot run past the tensor.
    static constexpr int kDecRows = 15, kEncRows = 9;
    static constexpr int kRowQ = 0, kRowK = 1, kRowV = 2, kRowO = 3;
    static constexpr int kRowCQ = 4, kRowCK = 5, kRowCV = 6, kRowCO = 7;
    static constexpr int kRowLn1W = 8, kRowLn1B = 9, kRowLn2W = 10, kRowLn2B = 11;
    static constexpr int kRowLn3W = 12, kRowLn3B = 13, kRowFc2B = 14;
    int dec_bias_row(int l, int which) const { return l * kDecRows + which; }
    int enc_bias_row(int l, int which) const {
        return (int)layers * kDecRows + l * kEncRows + (which == kRowO ? 3 : which);
    }
    // The encoder layer's rows are q,k,v,o,ln1w,ln1b,ln2w,ln2b,fc2b: the first
    // four and the last are the decoder's indices, so `which` above maps o->3.
    int enc_row(int l, int off) const { return (int)layers * kDecRows + l * kEncRows + off; }
    const float* bias_dim_row(int r) const { return bias_dim.data() + (size_t)r * dim; }
    const float* bias_ffn_row(int r) const { return bias_ffn.data() + (size_t)r * enc_ffn; }

    Rwkv rwkv;
    bool is_encdec() const { return version_ == 5; }
    bool is_rwkv() const { return rwkv.arch != 0; }
    // One row per layer, cols = count*dim, groups in the HF/GGUF order (so a
    // row can be compared against llama.cpp's tensor by tensor name).  The
    // two model-wide norms are a single row and absent when the source has no
    // token norm at all (rwkv_vocab's own 0.1B has one, llama.cpp reads it as
    // token_embd_norm).
    vector<float> rwkv_lerp;       // 0x1A: r, w, k, v, a, g
    vector<float> rwkv_bias;       // 0x1B: w0, a0, v0
    vector<float> rwkv_scale;      // 0x1C: k_k, k_a, r_k
    vector<float> rwkv_att_norm;   // 0x1D: w, b
    vector<float> rwkv_att_norm_2; // 0x1E: w, b
    vector<float> rwkv_ln_x;       // 0x1F: w, b (the per-head group norm)
    vector<float> rwkv_cm_lerp;    // 0x21
    vector<float> rwkv_tok_norm;   // 0x22: w, b; empty when absent
    vector<float> rwkv_out_norm;   // 0x23: w, b
    // The four low-rank pairs, one Mat per layer: down is (rank x dim) and up
    // is (dim x rank).  v (0x26/0x27) is empty on layer 0, which records
    // v_first instead of mixing it; g (0x2A/0x2B) is empty when key 16 says
    // the model has no gate.
    vector<Mat> rwkv_w1, rwkv_w2, rwkv_v1, rwkv_v2;
    vector<Mat> rwkv_a1, rwkv_a2, rwkv_g1, rwkv_g2;
    // The three row bands of qkv[l] = [W_r; W_k; W_v].  The container carries
    // them as one 3*dim x dim matrix (the dense qkv id), and the layer loop
    // needs three Mats because each band is drawn with its own token-shifted
    // input: split_rwkv() fills these from slice_rows() (the bytes stay
    // borrowed, so this is three texture views of one mapping, not a copy).
    // qkv itself is not uploaded for an RWKV container -- these three are, so
    // the weight traffic and the texture memory are the same as one draw.
    vector<Mat> rwkv_r, rwkv_k, rwkv_v;
    // split_rwkv(max_dim) also chunks the channel-mix key/value when the key's
    // row count (the FFN width) exceeds the GL target limit (VC4 2048): it reuses
    // ffn_chunked/ffn_chunk and w1a/w1b (key row halves) + w2a/w2b (value col
    // halves), exactly like the dense FFN split, so the layer loop draws two
    // <=2048-row textures instead of one 3072-row upload (GL_INVALID_VALUE).
    void split_rwkv(int max_dim);
    // Metadata key 17: the head's input (the final LayerNorm's output) is
    // written *2^-k so it fits the q16 activation rail, and every logits tile's
    // scale already carries the *2^k, so the dot product comes out in the
    // model's own units.  0 (or an absent key) means no scaling, which is every
    // container packed before 2026-11-16; RWKV-7 0.1B needs 2 (its final LN
    // peaks at 315.6, 2.5x the rail).
    int logit_shift = 0;
    // Drop everything this model owns, so the same object can hold a different
    // container.  Assigning a fresh instance is deliberate: the field list is
    // long (19 groups of vectors plus the scalars), and a hand-written reset
    // would be one missed field away from a leak that only the board sees.
    void reset() { *this = Model(); }

    bool is_moe() const { return experts > 0; }
    int ffn_width() const { return is_moe() ? moe_hidden : hidden; }

    // Filled by the MMCF reader: the tokenizer blob lives inside the file and
    // the chat template / default system prompt come from its metadata.
    const void* tok_blob = nullptr;
    size_t tok_blob_bytes = 0;
    // MODEL_BORROW=0 unmaps the container at the end of load(), so the blob the
    // reader pointed into that mapping has to be copied first (tok_blob is read
    // later, when the tokenizer is built).
    vector<uint8_t> tok_owned;
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
    // Metadata key 10's third byte: the tokenizer runs its space marker on
    // every text run between special tokens, not once per prompt.  A
    // SentencePiece vocabulary needs this ("user" and "\u2581user" are
    // different tokens); a byte-level one does not notice either way.
    bool sp_prefix = false;
    // Metadata key 10's fourth byte: 1 = SentencePiece unigram, which is what
    // an encoder-decoder's tokenizer is; absent (every container before this
    // byte grew) is the BPE merge pass.  Distinct from `tok_kind` above, which
    // is key 4's tokenizer *name*.
    bool tok_unigram = false;
    // How much of the body arrived as a 2-bit ternary record (dtype 7) and had
    // to be expanded to q8 on the host.  Zero for every other container; a load
    // that leaves it non-zero says so, because until the GPU kernel exists
    // (docs/bitcpm.md, Phase B2) the file's memory win does not reach the GPU.
    int t3_mats = 0;
    bool has_t3() const { return t3_mats > 0; }
    uint64_t t3_disk_bytes = 0, t3_ram_bytes = 0;
    // Metadata key 11: the model's own RMSNorm epsilon.  The shaders used to
    // hardcode 1e-5, which is 10x MiniMind3's (and feng30m's, and supra2's)
    // 1e-6; that was invisible until a model turned out to be sensitive to it.
    float rms_eps = 1e-5f;
    // Metadata key 14: one byte per layer, 0 = attention, 1 = the LFM2 conv
    // block.  Empty means every layer is attention (every model before LFM2).
    vector<uint8_t> layer_type;
    bool hybrid = false;
    // The conv block's per-layer pieces.  conv_in/conv_out are ordinary
    // matrices (3*dim x dim, dim x dim); conv_w is the depthwise kernel,
    // dim x k floats, kept on the host because the CPU path reads it and the
    // GPU upload repacks it into a texture.  Empty on attention layers.
    vector<Mat> conv_in, conv_out;
    // conv_in split by output rows: B (0:dim), C (dim:2dim), x (2dim:3dim).
    // The GPU path draws each slice into its own 1024-wide target; a single
    // 3072-wide draw would exceed VC4's 2048.  The slices borrow the same
    // bytes (row slices of a q4 matrix are contiguous).
    vector<Mat> conv_in_b, conv_in_c, conv_in_x;
    vector<vector<float>> conv_w;
    int conv_k = 0;
    bool is_conv(int l) const { return l >= 0 && l < (int)layer_type.size() && layer_type[l] != 0; }
    int n_conv_layers() const { int n = 0; for (size_t i = 0; i < layer_type.size(); ++i) n += layer_type[i] != 0; return n; }
    // Metadata key 12: the residual stream's width.  False (or absent) keeps
    // the q16 grid every existing container was packed with.
    bool res_q24 = false;
    // Metadata key 15: one byte per layer, the FFN hidden store's scale
    // exponent.  Layer k > 0 means the SwiGLU hidden is written *2^-k (so it
    // fits the q16 activation rail instead of being clamped) and w2's read-back
    // multiplies by 2^k (folded into its scale by apply_hidden_scale()).  Empty
    // = every layer unscaled, which is what every pre-2026-11-12 container is.
    vector<uint8_t> hid_scale;
    int hid_shift(int l) const {
        return (l >= 0 && l < (int)hid_scale.size()) ? (int)hid_scale[l] : 0;
    }
    // Fold 2^k into every w2 (and its chunks) after split_ffn(): the write side
    // divides the hidden, so the read side has to undo it exactly.
    void apply_hidden_scale();
    bool rms_eps_stated = false;   // false = use the shader's literal 1e-5
    int capabilities = 0;
    bool have_capabilities = false;
    // Special token ids from the container (-1 = absent).  Legacy files have no
    // such metadata and keep the historical 1 = BOS, 2 = EOS guess.
    bool have_specials = false;
    int id_bos = 1, id_eos = 2, id_stop = -1, id_pad = -1, id_unk = -1, id_im_start = -1;
    // (these defaults reproduce the historical 1 = BOS, 2 = EOS behaviour)
    bool load(const string& path);
    // Header and metadata only, for a caller that has to decide whether a
    // container is usable *before* releasing the model it is already running
    // (the REPL's /model).  The tensors are never touched: no map is kept, the
    // borrowed tokenizer blob is cleared before the unmap, and the prepare-time
    // refusals in split_ffn() cannot fire.  A file that is not a container this
    // build can read comes back false, with the reader's own message on stderr.
    bool probe(const string& path);
    // MMCF v2 container (docs/checkpoint-container.md).
    // The MMCF reader in three steps (2026-11-14, P2-1 of the 2026-11-13 audit):
    // `parse_header` validates the 64-byte header and keeps the derived offsets in
    // the members below, `load_mmcf` then walks the metadata and the tensor table.
    // Pure host code: none of this needs a GL context.
    bool parse_header();
    bool parse_metadata();
    bool read_tables();
    bool load_mmcf();
    ~Model();
    // Drop the mapping once every weight is on the GPU.  The pages stay in the
    // page cache - clean file pages, not process memory - so the next start is
    // still warm, while this process carries no 69 MB of RSS through decode.
    void release_map();

    // Release the resident pages of one tensor as soon as its bytes are in a
    // texture.  The container is borrowed, so the upload reads the file through
    // this mapping: a 200 MB container would keep a 200 MB clean page cache in
    // the movable zone while the textures it is being copied into need the same
    // memory, and on the Pi that is what makes a large upload fail with the
    // whole container still resident (2026-11-17, models/pocket-0.3b).
    // Correctness is unaffected: a page dropped here is re-read from the file if
    // anything touches those bytes again.
    void drop_pages(const Mat& m);
    void drop_pages(const void* p, size_t bytes);

    // Start the read of the *next* matrix while the current one is being
    // uploaded: one `madvise(MADV_WILLNEED)` over the borrowed bytes, so the
    // SD card faults pages ahead instead of inside the next glTexImage2D.
    // The window stays one matrix wide, which is what keeps `drop_pages()`
    // safe on a board where the container's pages and its textures share the
    // same movable pool.
    void prefetch(const Mat& m);

    // Copy any borrowed weights back into their vectors (only the opt-in
    // rewrite paths need that; upload reads the borrowed bytes directly).
    static void materialize(Mat& m);


    void split_ffn(int max_dim);
        // Q4_FMT=rgba4 probe: bind the FFN gate matrices as 4-bit texels (see
        // matvec_i4_pair_swiglu_4x4).  Must run after split_ffn and before the
        // upload; leaves every other INT4 matrix on the packed view.
        void apply_q4_fmt();

    static bool quantize_q4(Mat& m);

    bool quantize_q4_weights(const string& scope);

    void apply_precision(const string& layers_spec, const string& parts_spec);


private:
    // `read_exact`/`read_seq` were declared here and never called (0 sites on
    // 2026-11-14); `read_seq` was also the audit's worst exit() example -- a header
    // that terminates the process.  The loader reads through mmcf_* instead, which
    // returns bool.

    // The checkpoint stays mapped for the process lifetime: borrowed weights
    // point into it.  Those pages are page cache - reclaimable, never swapped -
    // so the next run reads them back from RAM instead of the SD card.
    // set by parse_header(), used by the metadata and tensor phases
    uint16_t version_ = 0;
    int rec_size_ = 0;
    uint32_t n_tensors_ = 0, caps_ = 0, meta_off_ = 0, meta_size_ = 0;
    uint64_t data_off_ = 0, table_off_ = 0;
    int quant_group_ = 0;    // metadata key 3: q8 group size
    bool quant_split_ = false;  // metadata key 3: W13 stored fused

    const uint8_t* fbase_ = nullptr;
    size_t fsize_ = 0, fpos_ = 0;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    bool borrow_ = true;
    // Kept open for the upload only, so drop_pages() can ask the kernel to
    // invalidate the pages it has already consumed (POSIX_FADV_DONTNEED needs
    // the descriptor; unmapping alone leaves them in the cache).
    int fd_ = -1;


    static Mat slice_rows(const Mat& src, int r0, int r1);

    static Mat slice_cols(const Mat& src, int c0, int c1);

    static Mat quantize(const float* src, int rows, int cols, WeightType type);
};
