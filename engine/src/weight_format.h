#pragma once

// The weight-format contract.
//
// XFormat is the contract for the streams a kernel reads and writes (xformat.h);
// this is the contract for the weights it multiplies them by.  A weight is not
// "a body layout" either: a format is a *body at a stated scale*, plus the rules
// its storage imposes on slicing, and the set of pipeline stages a kernel
// exists for.  Each row below answers all of those, and a format that cannot
// answer a row says why in the row it does not.
//
// The table is **capability**, not text.  The GLSL stays in shaders*.cpp (the
// emitted text is the numerics), and this header neither includes shaders.h nor
// holds a builder pointer: builders have different signatures (t3 carries a
// tile geometry, i4 a view choice, q8 a per-row plane), so a function-pointer
// table would force one signature and become a second source of truth.  The
// seam is the GPU adapter, which is the only code that includes both and does
// an exhaustive `switch (m.type)`.  A stage bit says "this pipeline entry has a
// kernel for this format"; the adapter says which one.
//
// The failure direction is deliberate and must stay that way:
//   * the table says yes and the adapter has no case -> a loud refusal
//     (wf_no_kernel), never another format's kernel;
//   * the adapter has a case and the table says no -> the path is unused
//     (slow, correct), never wrong.
// A missing `else`/`default` that falls back to another format is exactly the
// failure mode this contract exists to remove: at the time of writing,
// GPU::draw, GPU::matvec_add_bias, GPU::upload and Model::slice_* all ended in
// a silent `x_i16`/q4 fallback.
//
// `wrow`, `fmt4` and the t3 tile geometry are per-matrix state and stay in Mat;
// only facts that are true of the format itself live here.  The stream axes
// (ACT/RES/FFN) are XFormat's business: a caller that needs "P2 kernels and a
// q16 pair residual" asks wf_can() for the weight half and checks the stream
// half itself.
#include "common.h"
#include "xformat.h"

// The pipeline entries that select a weight kernel.  These are GPU entry points,
// not mathematical operations: PairRows is split into the sites whose
// eligibility differs (qkv, logits, FFN), and one builder may serve more than
// one entry (q8's matvec_q8_pair_logits serves both qkv and logits).
enum class WStage : uint8_t {
    Serial = 0,     // GPU::draw: one output row a fragment (or the format's own)
    Add,            // GPU::matvec_add*
    QkvPair,        // the qkv projection's two-rows-a-fragment kernel
    LogitsPair,     // the logits tiles' two-rows-a-fragment kernel
    FfnPair,        // w1/w3 as two matrices (before swiglu)
    FfnFusedGate,   // w1/w3 with swiglu in the kernel
    FfnFusedW13,    // one W13 matrix, rows r and r+rows in one fragment
    P2,             // two-position prefill: gate and w2
    P2Add,          // two-position prefill: the w2 residual add
    Batch,          // one draw for a batch of inputs (RWKV chunked prefill)
};

constexpr uint16_t wstage_bit(WStage s) { return (uint16_t)(1u << (unsigned)s); }

// How a matrix's real weights are reconstructed from the body.  A format may
// support more than one: q8 is Uniform unless the container carries a part-3
// per-row plane, and a q4 body is always group scale + group min.
constexpr uint8_t kWScaleUniform = 1u << 0;
constexpr uint8_t kWScalePerRow = 1u << 1;
constexpr uint8_t kWScaleGroupMinMax = 1u << 2;

constexpr uint16_t xfmt_bit(XFormat f) { return (uint16_t)(1u << (unsigned)f); }
// Every XFormat that carries values.  S24_1 is a partial sum, never an input.
constexpr uint16_t kActAll =
    xfmt_bit(XFormat::I16_1) | xfmt_bit(XFormat::I16_2) | xfmt_bit(XFormat::I8_4) |
    xfmt_bit(XFormat::I24_1) | xfmt_bit(XFormat::F32) | xfmt_bit(XFormat::BF16);
// The same set without the 8-bit stream.  I8_4 is validated for q8 (the ACT8
// pin) and for t3 (correct since 5b66ebc, measured in docs/bitcpm.md); the
// other two families' generic X4Q form compiles but was never accepted, and a
// stream format that has no acceptance record must not become one by default.
// The contract's failure direction is "table says no, adapter has a case ->
// unused (slow, correct), never wrong", so the declaration says no until a
// measurement says otherwise.
constexpr uint16_t kActAll8 = kActAll & ~xfmt_bit(XFormat::I8_4);
// matvec_q6 refuses anything else by name (bitwise 6-bit decode).
constexpr uint16_t kActQ6 =
    xfmt_bit(XFormat::I16_2) | xfmt_bit(XFormat::F32) | xfmt_bit(XFormat::BF16);

struct WeightFormat {
    WeightType type;
    const char* name;
    uint8_t scale_modes;      // kWScale* bits
    uint8_t row_align;        // slice_rows boundary
    uint8_t col_align;        // slice_cols boundary
    bool wrow;                // can carry a part-3 per-row scale plane
    uint16_t act_mask;        // XFormat inputs its kernels accept
    uint16_t caps;            // WStage bits with a kernel
};

// One row per WeightType, in enum order (wf() indexes by it).  The compiled
// test checks the order and the wire mapping; gate/check_weight_formats.py
// checks the table against common.h and mmcf_format.py.
constexpr WeightFormat kWeightFormats[] = {
    // q8: one byte a weight, four a texel; one scale, or one per output row.
    // Column slicing steps by four because every q8 kernel's loop does.
    {WeightType::INT8, "q8", kWScaleUniform | kWScalePerRow, 1, 4, true, kActAll,
     wstage_bit(WStage::Serial) | wstage_bit(WStage::Add) |
         wstage_bit(WStage::QkvPair) | wstage_bit(WStage::LogitsPair) |
         wstage_bit(WStage::FfnPair) | wstage_bit(WStage::FfnFusedGate) |
         wstage_bit(WStage::FfnFusedW13) | wstage_bit(WStage::P2) |
         wstage_bit(WStage::P2Add) | wstage_bit(WStage::Batch)},
    // i16: two weights a texel, one scale.  Serial and add only; the paired
    // paths fall back to the serial kernel rather than pretending.
    {WeightType::INT16, "i16", kWScaleUniform, 1, 1, false, kActAll8,
     wstage_bit(WStage::Serial) | wstage_bit(WStage::Add)},
    // q4: eight packed nibbles a texel plus the group scale/min plane.  A
    // column slice must stay on a group boundary; rows are independent.
    // P2/P2Add are the 4-bit view's (Q4_FMT=rgba4, the chunked FFN's default):
    // the packed nibble view has no two-position kernel, so pair_ok() also
    // asks for fmt4 rather than pretending.
    {WeightType::INT4, "q4", kWScaleGroupMinMax, 1, 32, false, kActAll8,
     wstage_bit(WStage::Serial) | wstage_bit(WStage::Add) |
         wstage_bit(WStage::FfnPair) | wstage_bit(WStage::FfnFusedGate) |
         wstage_bit(WStage::P2) | wstage_bit(WStage::P2Add)},
    // q6: two planes (4-bit low, 2-bit high), tied head only.  It is the one
    // format that refuses the 8-bit activation formats.
    {WeightType::INT6, "q6", kWScaleUniform | kWScalePerRow, 1, 8, true, kActQ6,
     wstage_bit(WStage::Serial)},
    // t3: 16 ternary weights a texel, two bit each, plus a part-3 row plane.
    // One texel carries a row pair, so a row slice is two rows; a column group
    // is eight columns.  The FFN pair path is bespoke (matvec_pair_to_row +
    // swiglu_pair2row), which is why the capability is still FfnPair.
    {WeightType::T3, "t3", kWScaleUniform | kWScalePerRow, 2, 8, true, kActAll,
     wstage_bit(WStage::Serial) | wstage_bit(WStage::Add) |
         wstage_bit(WStage::QkvPair) | wstage_bit(WStage::FfnPair)},
};

constexpr size_t kWeightFormatCount = sizeof(kWeightFormats) / sizeof(kWeightFormats[0]);
static_assert(kWeightFormatCount == 5, "one row per WeightType");

constexpr const WeightFormat& wf(WeightType t) {
    return kWeightFormats[(unsigned)t];
}

constexpr bool wf_can(WeightType t, WStage s) {
    return (wf(t).caps & wstage_bit(s)) != 0;
}

constexpr int wf_row_align(WeightType t) { return wf(t).row_align; }
constexpr int wf_col_align(WeightType t) { return wf(t).col_align; }

constexpr bool wf_accepts(WeightType t, XFormat f) {
    return (wf(t).act_mask & xfmt_bit(f)) != 0;
}

// Container dtype -> the weight type the kernels run on.  The mapping is not
// one-to-one: q8u8 (2) and q8i8 (3) are the same kernels and differ only in
// whether the +128 bias is already in the bytes; f32 (0) and blob (5) are not
// matrix bodies at all.  mmcf_format.py owns the wire names.
inline bool wf_from_dtype(uint8_t dtype, WeightType* out, bool* prebiased) {
    switch (dtype) {
        case 0: return false;  // f32: a norm/embedding vector, not a Mat
        case 1: *out = WeightType::INT16; *prebiased = false; return true;
        case 2: *out = WeightType::INT8; *prebiased = true; return true;
        case 3: *out = WeightType::INT8; *prebiased = false; return true;
        case 4: *out = WeightType::INT4; *prebiased = false; return true;
        case 5: return false;  // blob: the tokenizer
        case 6: *out = WeightType::INT6; *prebiased = false; return true;
        case 7: *out = WeightType::T3; *prebiased = false; return true;
    }
    return false;
}

constexpr uint8_t wf_dtype(WeightType t, bool prebiased) {
    return t == WeightType::INT8 ? (prebiased ? 2 : 3)
         : t == WeightType::INT16 ? 1
         : t == WeightType::INT4 ? 4
         : t == WeightType::INT6 ? 6
         : 7;
}

// The loud half of the contract: a stage the adapter reached without a case.
// A format enum added without a dispatch case must stop the run here instead of
// decoding the bytes with another format's kernel.
[[noreturn]] inline void wf_no_kernel(const char* who, WeightType t) {
    const char* name = (unsigned)t < kWeightFormatCount ? kWeightFormats[(unsigned)t].name
                                                       : "?";
    std::fprintf(stderr, "%s: weight type %d (%s) has no kernel for this path\n",
                 who, (int)t, name);
    std::exit(2);
}

// The activation half of the same refusal: a format's kernels declare which
// XFormats they can decode (q6 is the one that refuses the 8-bit streams).
[[noreturn]] inline void wf_no_activation(const char* who, WeightType t, XFormat f) {
    std::fprintf(stderr, "%s: weight type %s cannot read %s activations\n",
                 who, wf(t).name, x_format_name(f));
    std::exit(2);
}