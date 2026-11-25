// The weight-format contract: the table against the enum, the wire dtypes and
// the storage arithmetic.  No GPU, no model.
//
// The table is what the GPU dispatch and the eligibility predicates ask, so a
// wrong row is a silently mis-selected kernel (the failure the contract exists
// to remove).  These checks pin the row order wf() indexes by, the dtype
// mapping the packer and the loader share, the slice boundaries, and the caps
// that were transcribed from the old per-file type lists.
#include <cstdio>

#include "checks.h"
#include "weight_format.h"

namespace {

constexpr WStage kStages[] = {
    WStage::Serial, WStage::Add, WStage::QkvPair, WStage::LogitsPair,
    WStage::FfnPair, WStage::FfnFusedGate, WStage::FfnFusedW13,
    WStage::P2, WStage::P2Add, WStage::Batch,
};

bool has(WeightType t, WStage s) { return wf_can(t, s); }

void test_rows() {
    CHECK_LL(kWeightFormatCount, 5);
    for (size_t i = 0; i < kWeightFormatCount; ++i) {
        const WeightFormat& w = kWeightFormats[i];
        CHECK_LL((int)w.type, (int)i);       // wf() indexes by the enum
        CHECK(w.name != nullptr && w.name[0] != '\0');
        // Every format can at least answer the serial path, and every format
        // declares how it stores a scale.
        CHECK(wf_can(w.type, WStage::Serial));
        CHECK(w.scale_modes != 0);
        // Slices are cut on a power-of-two boundary (a texel, a row pair, a
        // group).  Anything else is not a slice rule the storage can state.
        CHECK(w.row_align >= 1 && (w.row_align & (w.row_align - 1)) == 0);
        CHECK(w.col_align >= 1 && (w.col_align & (w.col_align - 1)) == 0);
        // S24_1 is a partial sum, never an activation input.
        CHECK((w.act_mask & xfmt_bit(XFormat::S24_1)) == 0);
        CHECK(w.act_mask != 0);
        // Every caps bit names a real stage.
        uint16_t known = 0;
        for (WStage s : kStages) known |= wstage_bit(s);
        CHECK((w.caps & (uint16_t)~known) == 0);
    }
    CHECK_STR(wf(WeightType::INT8).name, "q8");
    CHECK_STR(wf(WeightType::INT16).name, "i16");
    CHECK_STR(wf(WeightType::INT4).name, "q4");
    CHECK_STR(wf(WeightType::INT6).name, "q6");
    CHECK_STR(wf(WeightType::T3).name, "t3");
}

void test_caps() {
    // q8 is the only format that carries every path.
    for (WStage s : kStages) CHECK(has(WeightType::INT8, s));
    // i16/i6 are serial(+add); neither has a pair-family kernel.
    CHECK(has(WeightType::INT16, WStage::Add));
    CHECK(!has(WeightType::INT16, WStage::QkvPair));
    CHECK(!has(WeightType::INT16, WStage::FfnPair));
    CHECK(!has(WeightType::INT6, WStage::Add));
    CHECK(!has(WeightType::INT6, WStage::Batch));
    // q4: FFN pair/fused and the two-position family in the 4-bit view;
    // no qkv/logits pair kernels (they run serial).
    CHECK(has(WeightType::INT4, WStage::FfnPair));
    CHECK(has(WeightType::INT4, WStage::FfnFusedGate));
    CHECK(!has(WeightType::INT4, WStage::QkvPair));
    CHECK(!has(WeightType::INT4, WStage::LogitsPair));
    CHECK(has(WeightType::INT4, WStage::P2));
    CHECK(has(WeightType::INT4, WStage::P2Add));
    // t3: qkv pair, its bespoke FFN pair and the two-position family; no
    // logits pair.
    CHECK(has(WeightType::T3, WStage::QkvPair));
    CHECK(has(WeightType::T3, WStage::FfnPair));
    CHECK(!has(WeightType::T3, WStage::LogitsPair));
    CHECK(has(WeightType::T3, WStage::P2));
    CHECK(has(WeightType::T3, WStage::P2Add));
}

void test_scale_and_activations() {
    CHECK_LL(wf(WeightType::INT8).scale_modes, kWScaleUniform | kWScalePerRow);
    CHECK_LL(wf(WeightType::INT16).scale_modes, kWScaleUniform);
    CHECK_LL(wf(WeightType::INT4).scale_modes, kWScaleGroupMinMax);
    CHECK_LL(wf(WeightType::INT6).scale_modes, kWScaleUniform | kWScalePerRow);
    CHECK_LL(wf(WeightType::T3).scale_modes, kWScaleUniform | kWScalePerRow);

    // matvec_q6's refusal is the reason q6 is the one exception.
    CHECK_LL(wf(WeightType::INT6).act_mask, kActQ6);
    CHECK(wf_accepts(WeightType::INT6, XFormat::I16_2));
    CHECK(wf_accepts(WeightType::INT6, XFormat::F32));
    CHECK(wf_accepts(WeightType::INT6, XFormat::BF16));
    CHECK(!wf_accepts(WeightType::INT6, XFormat::I16_1));
    CHECK(!wf_accepts(WeightType::INT6, XFormat::I8_4));
    // I8_4 has an acceptance record for q8 (the ACT8 pin) and for t3 (correct
    // since 5b66ebc, docs/bitcpm.md); q4 and i16 never had one, so the table
    // says no for them (the contract's "unused, slow, correct" direction).
    CHECK_LL(wf(WeightType::INT8).act_mask, kActAll);
    CHECK(wf_accepts(WeightType::INT8, XFormat::I8_4));
    CHECK_LL(wf(WeightType::T3).act_mask, kActAll);
    CHECK(wf_accepts(WeightType::T3, XFormat::I8_4));
    for (WeightType t : {WeightType::INT16, WeightType::INT4}) {
        CHECK_LL(wf(t).act_mask, kActAll8);
        CHECK(!wf_accepts(t, XFormat::I8_4));
        CHECK(!wf_accepts(t, XFormat::S24_1));
    }
}

void test_alignments() {
    CHECK_LL(wf_row_align(WeightType::INT8), 1);   // whole bytes, rows independent
    CHECK_LL(wf_col_align(WeightType::INT8), 4);   // every q8 loop steps by four
    CHECK_LL(wf_col_align(WeightType::INT16), 1);
    CHECK_LL(wf_col_align(WeightType::INT4), 32);  // the group scale/min pair
    CHECK_LL(wf_col_align(WeightType::INT6), 8);   // lo/hi planes
    CHECK_LL(wf_row_align(WeightType::T3), 2);     // one texel is a row pair
    CHECK_LL(wf_col_align(WeightType::T3), 8);     // one texel is a column group
    CHECK_LL(wf_row_align(WeightType::T3) % 2, 0);
}

void test_wire() {
    WeightType t = WeightType::INT8;
    bool pb = true;
    CHECK(!wf_from_dtype(0, &t, &pb));   // f32: a vector, not a Mat
    CHECK(!wf_from_dtype(5, &t, &pb));   // blob: the tokenizer
    CHECK(!wf_from_dtype(8, &t, &pb));

    CHECK(wf_from_dtype(1, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::INT16); CHECK(!pb);
    CHECK(wf_from_dtype(2, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::INT8); CHECK(pb);    // q8u8: bias applied
    CHECK(wf_from_dtype(3, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::INT8); CHECK(!pb);   // q8i8
    CHECK(wf_from_dtype(4, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::INT4);
    CHECK(wf_from_dtype(6, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::INT6);
    CHECK(wf_from_dtype(7, &t, &pb));
    CHECK_LL((int)t, (int)WeightType::T3);

    CHECK_LL(wf_dtype(WeightType::INT16, false), 1);
    CHECK_LL(wf_dtype(WeightType::INT8, true), 2);
    CHECK_LL(wf_dtype(WeightType::INT8, false), 3);
    CHECK_LL(wf_dtype(WeightType::INT4, false), 4);
    CHECK_LL(wf_dtype(WeightType::INT6, false), 6);
    CHECK_LL(wf_dtype(WeightType::T3, false), 7);
}

void test_bytes() {
    // The packer's four-byte texel arithmetic, against mat_bytes' switch.
    Mat m;
    m.rows = 4; m.cols = 8; m.groups = 1;
    m.type = WeightType::INT8;  CHECK_LL(mat_bytes(m), 32);
    m.type = WeightType::INT16; CHECK_LL(mat_bytes(m), 64);
    m.type = WeightType::INT4;  CHECK_LL(mat_bytes(m), 32);  // n/2 + rows*groups*4
    m.type = WeightType::INT6;  CHECK_LL(mat_bytes(m), 24);  // n/2 + n/4
    m.type = WeightType::T3;    CHECK_LL(mat_bytes(m), 8);   // ceil(rows/2)*ceil(cols/8)*4
}

}  // namespace

int main() {
    test_rows();
    test_caps();
    test_scale_and_activations();
    test_alignments();
    test_wire();
    test_bytes();
    if (checks::g_fail) {
        std::fprintf(stderr, "test_weight_format: %d check(s) failed\n", checks::g_fail);
        return 1;
    }
    std::printf("test_weight_format: %zu formats, wire mapping and caps pass\n",
                kWeightFormatCount);
    return 0;
}