// The switch registry: see switches.h for why it exists and what the rules are.
// The table is ordered by axis, then by name; `--dump-switches` prints it in
// that order, and docs/test-hooks.md documents the same names in prose.
#include "switches.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;

static const SwitchDef kSwitches[] = {
    // ---- Dump: the engine's numbers, written out for an offline oracle ------
    {"T3_EXPAND", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Load,
     "expand a 2-bit body to the q8 values it stands for at load time (4x the RAM); the oracle the t3 kernels are diffed against"},
    {"T3_GEOM", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the 2-bit body's tile geometry as the uploader made it and as the shader reads it -- two machines comparing numbers need this to be the same lines"},
    {"STAGE_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "per-layer xb/att/x1/xbf/hid/hidA as q16 texels (the pair path adds x1b/p2in/p2part); diff_paths.sh reads it"},
    {"HIDDEN_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the same for the layer output on the GPU path (GPU_LAYER 0 vs 1)"},
    {"DUMP_F32", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "dump rows as raw IEEE bits instead of q16 texels"},
    {"CHAT_DUMP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Dump,
     "the rendered prompt on stderr, segment by segment"},
    {"CHAT_CACHE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the cache split of every model call (cached / prefill / prompt)"},
    {"QKV_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "hash every qkv matvec row; 2 also prints the first two"},
    {"ATT_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the score/softmax integers at one position, for check_att.py"},
    {"FFN_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the first 8 texels of layer 0's gate output"},
    {"LOGITS_OUT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "float32 logits after prefill, for cmp_logits.py"},
    {"TOKENS_OUT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "prompt and generated token ids (int32)"},
    {"PPL_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "one `position token nll` line per scored token"},
    {"PPL_STRIDE", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Dump,
     "window stride for --ppl (0 restores the cold pre-2026-11-12 windows)"},
    {"MODEL_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the parsed container metadata, right after load"},
    {"MMCF_DEBUG", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "per-tensor detail while walking the container table"},
    {"MOE_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the route of every MoE layer/position"},
    {"MOE_FIXED", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "pin every MoE layer to one expert and skip the router (a different model)"},
    {"RWKV_DUMP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Dump,
     "the RWKV block's per-layer peaks (state/residual/sqrelu rails)"},
    {"CONV_DEBUG", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the LFM2 conv block's stages for layer 0, next to the CPU reference"},
    {"CPU_RMS_GPU", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the CPU layer loop's RMSNorm runs the GPU shader (A/B for the PPL gap)"},
    {"CPU_ADD_GPU", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "the CPU layer loop folds the residual add in the kernel"},
    {"PASS_DUP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "replay one fixed per-token pass (embed, rms_att, rms_ffn, logits); dup_env()"},
    {"MATVEC_DUP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "replay one weight-matrix pass per layer (wo,ffn,w2 or all); dup_env()"},
    {"ARGMAX_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "with the greedy fast path, print the chunk winners"},
    {"SHADER_SRC", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "print the GLSL of every program whose key contains the substring"},
    {"GLSL_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "write every program this run builds to <dir>/<key>.glsl: the seed for a kernel experiment"},
    {"GLSL_OVERRIDE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "compile <file> for <key>, one or more <key>=<file>[,<key>=<file>...]; a key that never builds is named at exit"},

    // ---- Probe: what one pass costs ----------------------------------------
    {"TEX_LIMIT", "", SwKind::Int, "0..16384", SwStatus::Probe, SwAxis::Probe,
     "lower the GL target limit every seq-sized texture is checked against (0 = the driver's); the board's 2048, on a machine that has more"},
    {"SHADER_TIME", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "STARTUP phase marks and one line per program"},
    {"TIMING", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "prefill and decode timings to stderr"},
    {"UPLOAD_TIME", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "upload wall time and the per-texture breakdown"},
    {"UPLOAD_SYNC", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "one glFinish per texture upload (the pre-batch form), for an A/B"},
    {"UPLOAD_PREFETCH", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "read the next matrix ahead of its upload (on unless set to 0)"},
    {"DRAWCOUNT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "draws per decoded token"},
    {"SYNCCOUNT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "host <-> GPU synchronisations per decoded token"},
    {"LAYER_PROFILE", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "per-pass marks inside a layer"},
    {"PERFCNT", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "with LAYER_PROFILE or DUP_PROFILE, add the V3D hardware counters to each mark (gpu/exec/idle/quads)"},
    {"LOGITS_PERFCNT", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "with PERFCNT, the logits pass's own counters (its wall time cannot separate work from waiting)"},
    {"DUP_PROFILE", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "with a *_DUP replay, price the replay itself: wall against gpu cycles"},
    {"PAIR_PROFILE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "the same for the two-position prefill pair"},
    {"ATTN_DUP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "replay one attention pass; the end-to-end delta is its marginal cost"},
    {"READBACK_TIME", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "the per-token logits finish/read-back time"},
    {"RWKV_PROFILE", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "the RWKV paths' per-phase timing"},
    {"RWKV_CHUNK_TEST", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Probe,
     "sweep the RWKV channel-mix cut (probe: it moves the numbers)"},

    // ---- Placement: which path runs ----------------------------------------
    {"GPU_LAYER", "", SwKind::Flag, "", SwStatus::Permanent, SwAxis::Place,
     "hidden state, attention, RMSNorm and FFN all on the GPU"},
    {"GPU_ATTN", "", SwKind::Flag, "", SwStatus::Permanent, SwAxis::Place,
     "GPU attention with a CPU FFN"},
    {"CPU_FFN", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Place,
     "force the FFN onto the CPU (overrides GPU_LAYER's FFN)"},
    {"GPU_FFN", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Place,
     "force the FFN onto the GPU"},
    {"CONV_SKIP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Place,
     "the LFM2 conv layers become identity copies"},
    {"NO_SHADER_SELFTEST", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Place,
     "skip the startup shader self-test"},
    {"DRAW_CLEAR", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Place,
     "clear the target before every draw (a correctness oracle, not a setting)"},
    {"MATVEC_STRIP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Place,
     "override the fragment budget of the matvec draws"},

    // ---- Act: the activation streams' precision ----------------------------
    // The GPU streams (the uploaded h, the attention output, xb, xbf) are one
    // axis with one value, and it covers what used to be three switches:
    // ACT_FMT (format), ACT8=1 (width) and ACT8=2 (the FFN hidden's width).
    {"ACT", "ACT_FMT", SwKind::Str, "i16|i8|bf16|f32", SwStatus::Permanent, SwAxis::Act,
     "the GPU activation streams' format: i16 q16 grid (default), i8, bf16, or bit-cast f32"},
    {"FFN_HID", "", SwKind::Str, "i16|i8|bf16|f32", SwStatus::Permanent, SwAxis::Act,
     "the FFN hidden stream's format: q16 (default), i8, bf16, or f32 past the q16 rail"},
    {"ACT_CPU", "ACT_F32", SwKind::Str, "q16|f32", SwStatus::Permanent, SwAxis::Act,
     "the CPU layer loop's activation grid (ACT_F32 meant f32, whatever its value)"},
    {"Q16_DECODE", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Act,
     "the q16 decode: fast classifies the -32768 rail at 32767.5, exact is integer"},

    // ---- Att: the attention block's own fixed-point ------------------------
    {"ATT_SCORE_GRID", "ATT_SCALE", SwKind::Num, "", SwStatus::Permanent, SwAxis::Att,
     "score/(max) grid: round(score*N), so +-32768/N range and 1/N step (1024)"},
    {"RMS_DUMP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Dump,
     "prints the 32 rms partial sums as the shader wrote them (RMS_DUMP=<layer> for one layer)"},
    {"RMS_STORE", "", SwKind::Str, "i24|f32", SwStatus::Probe, SwAxis::Act,
     "the rms partial sums: 24-bit fixed point (default, 1/4096 step, pins at a lane sum of 2048) or a bit-cast f32; a residual above ~11 needs f32"},
    {"ENC_FFN_SPLIT", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "v5: keep the encoder FFN's fc2 bias as a separate add (default: folded into w2, one rounding)"},
    {"ENC_WO_SPLIT", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "v5: keep the encoder wo bias as a separate add (default: folded into the add)"},
    {"DEC_BIAS_SPLIT", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 decoder: keep the wo/cross-o/fc2 biases as separate adds (default: folded)"},
    {"EXP", "ATT_EXP", SwKind::Str, "drv|poly", SwStatus::Probe, SwAxis::Act,
     "every exponential the shaders emit: drv (default) or a portable polynomial; VC4's SFU exp is 20x Xe's error"},
    {"ATT_SUM", "", SwKind::Str, "linear|log2", SwStatus::Permanent, SwAxis::Att,
     "the sum's domain: linear grid (default) or log2(sum)*2048 — a trade, see the doc"},
    {"ATT_SUM_GRID", "ATT_SUM_SCALE", SwKind::Num, "", SwStatus::Permanent, SwAxis::Att,
     "the linear sum grid in the same (max,sum) texel (32 = step 1/32)"},

    // ---- Kv: the KV cache's encoding --------------------------------------
    // One axis, one value.  KV_PACK (bits) and KV_FMT (codec) were its two
    // halves and their interaction was undefined; kRenames maps both.
    {"KV", "", SwKind::Str, "i24|q8|q16|bf16|f32", SwStatus::Permanent, SwAxis::Kv,
     "the KV value encoding (i24 fixed 24-bit, q16/q8 packed, bf16, or f32)"},
    {"KV_GRID", "KV_SCALE", SwKind::Num, "", SwStatus::Permanent, SwAxis::Kv,
     "the fixed-point KV scale (16384 for i24, 2048 for q16, 16 for q8)"},
    {"KV_VSTEP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Kv,
     "q8 only: a per-head, per-position V grid (buys q4 6% PPL, costs q8 6%)"},
    {"KV8_SIM", "", SwKind::Int, "0..6", SwStatus::Probe, SwAxis::Kv,
     "price an 8-bit KV grid on the CPU attention path (1..6 = which grids)"},
    {"KV8_SIM_SCALE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Kv,
     "the global grid the KV8_SIM pricing arms use (default 16)"},
    {"KV8_STATS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Kv,
     "the per-head V statistics KV8_SIM needs, on the CPU path"},

    // ---- Res: the residual stream's format --------------------------------
    {"RES", "RES_FMT", SwKind::Str, "i16|i16_1|i24|bf16|f32", SwStatus::Permanent, SwAxis::Res,
     "residual format: i16 packed pairs (default), i16_1 one a texel, i24, bf16, or f32"},
    {"PROJ", "", SwKind::Str, "i24|f32", SwStatus::Permanent, SwAxis::Act,
     "a projection's output: i24 (default) or f32 (exact, same decode cost, needs ESSL 3.00)"},

    // ---- Ffn: the FFN hidden stream and the finish policy ------------------
    {"FFN_FLUSH", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Ffn,
     "flush policy after the FFN (FFN_FLUSH=0 keeps the default)"},
    {"FFN_SYNC", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Ffn,
     "sync after the FFN's draws"},
    {"FFN_FUSE", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Ffn,
     "the fused gate+swiglu (default on); FFN_FUSE=0 restores the two-pass path"},
    {"FFN_PAIR", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Ffn,
     "two output rows per FFN fragment (default on); FFN_PAIR=0 disables"},
    {"QKV_PAIR", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Act,
     "two qkv rows per fragment (int8 only, default on); QKV_PAIR=0 disables"},
    {"FFN_SW_INLINE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "FFN_SW_INLINE=0 restores the generic X4Q activation decode in the swiglu"},
    {"Q8AP_INLINE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "Q8AP_INLINE=0 restores the generic X4Q activation decode"},
    {"LAYER_SAFE_FINISH", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "restore the fully-finished GPU layer path"},
    {"FORCE_GLFINISH", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "force the VC4 finish path on other drivers"},

    // ---- Q4: the 4-bit weight kernels --------------------------------------
    // This axis is where the regime hurt most: 26 arms from the 2026-11-16
    // campaign, most of them a *stage* of one measurement (the pricing arms and
    // the flat-loop variants), all still compiled into the acceptance path.
    {"Q4_WEIGHTS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "quantize to q4 at load (prefer an offline container: q8 -> q4 loses twice)"},
    {"Q4_SCOPE", "", SwKind::Str, "all|attn|ffn", SwStatus::Probe, SwAxis::Q4,
     "which parts Q4_WEIGHTS converts"},
    {"Q4_FMT", "", SwKind::Str, "rgba4|0", SwStatus::Probe, SwAxis::Q4,
     "bind an INT4 matrix as GL_RGBA4 (dma-buf); Q4_FMT=0 keeps the packed view"},
    {"Q4_SHIFT", "", SwKind::Str, "0|1", SwStatus::Permanent, SwAxis::Q4,
     "Q4_SHIFT=0 forces the float nibble decode instead of the ESSL 3.00 shift"},
    {"Q4_SHIFT_NO_MESA", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "skip the MESA ES 3.00 version override that Q4_SHIFT needs"},
    {"Q4_SHIFT_VERBOSE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "report the q4 decode choice at startup"},
    {"Q4_IMPORT_CHECK", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "after the first imported q4 surface, verify texel (0,0) end to end"},
    {"Q4_TEX_STATS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "per texture: dma-buf import or upload, and its logical size"},
    {"Q4_INLINE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "inline the q4 group decode into the kernel body"},
    {"Q4_MV4", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "the four-output q4 matvec form"},
    {"Q4_NOUNPACK", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "pricing arm: constant scale/min instead of the group plane (wrong weights)"},
    {"Q4_NOMIN", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "pricing arm: drop the q4 min term"},
    {"Q4_BFAST", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "pricing arm: the fast block form of the q4 gate"},
    {"Q4_HFAST", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "pricing arm: the fast half form of the q4 gate"},
    {"Q4_GATE_NESTED", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "restore the nested group-loop 4-bit FFN gate (pre-2026-11-16 default)"},
    {"Q4_QKV_FLAT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "the flat-loop form of the q4 qkv matvec"},
    {"Q4_W2_FLAT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Q4,
     "the flat-loop form of the q4 w2/wo matvec-add"},

    // ---- Rwkv: the RWKV-7 layer loop --------------------------------------
    {"RWKV_GPU", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "RWKV's fused GPU layer loop (P6); the CPU serial loop is the reference"},
    {"RWKV_WKV_CPU", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "with RWKV_GPU: the elementwise chain on the GPU, the WKV state on the host"},
    {"RWKV_WKV_ONE", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "with RWKV_GPU: force the WKV update into one draw"},
    {"RWKV_SHIFT_PP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "with RWKV_GPU: ping-pong the two token-shift copies (removes 24 draws a token)"},
    {"RWKV_GROUP", "", SwKind::Int, "0..4", SwStatus::Probe, SwAxis::Rwkv,
     "the serial RWKV path's matvec grouping (P7)"},
    {"RWKV_GROUP_CHECK", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "with the grouping on, recompute every grouped matvec and compare"},
    {"RWKV_CM_SPLIT", "", SwKind::Int, "", SwStatus::Probe, SwAxis::Rwkv,
     "the RWKV channel-mix cut point (moves the numbers: the add is per chunk)"},
    {"RWKV_PREFILL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "RWKV chunked prefill (P1): one draw per weight over a batch of tokens"},
    {"RWKV_PREFILL_N", "", SwKind::Int, "", SwStatus::Probe, SwAxis::Rwkv,
     "batch size for RWKV_PREFILL"},
    {"RWKV_PREFILL_STRIP", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Rwkv,
     "fragment budget of one batched prefill draw (default kDrawCap)"},
    {"RWKV_PREFILL_CHECK", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Rwkv,
     "compare every batched matvec row against the serial kernel, in the run"},

    // ---- Prefill: the two-position pair -----------------------------------
    {"PREFILL_PAIR", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Prefill,
     "the two-position prefill pair (on by default); PREFILL_PAIR=0 disables it"},
    {"PAIR_WROW", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Prefill,
     "keep the single-position prefill for a model with per-row scale planes"},
    {"PREFILL_REPEAT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Prefill,
     "replay the prefill n times, for timing"},
    {"PREFILL_BATCH", "", SwKind::Int, "0..16", SwStatus::Probe, SwAxis::Prefill,
     "fuse N positions' attention score passes into one draw (0 = off); the residual chain stays serial"},

    // ---- Encoder-decoder (MMCF v5) ----------------------------------------
    {"ENCDEC_DUMP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "print the encoder/decoder stage magnitudes and the first step's argmax"},
    {"ENCDEC_REPEAT", "", SwKind::Int, "", SwStatus::Probe, SwAxis::Encdec,
     "repeat encode+greedy decode n times in one process and report the last (warms shaders and clocks)"},
    {"QROT_COPY", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5: keep the qrot copy draw that NOROPE skips, for the byte-identical A/B"},
    {"XKV_SERIAL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: keep the per-position cross-KV store (default: all rows in two stores)"},
    {"XPROJ_SERIAL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: keep the per-position cross-KV projection (default: one batch matvec a layer)"},
    {"STATS_SERIAL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: keep the per-position stats draw (default: one draw for every query)"},
    {"ATTB_SERIAL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: keep the per-position qkv/KV-write/qrot front (default: one projection plane)"},
    {"VB_BATCH", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: one V draw for every query (opt-in; measured ~0.4 s slower than the per-position pass on Pi)"},
    {"LN_SERIAL", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Encdec,
     "v5 encoder: keep the per-position LayerNorm passes (default: split/reduce over a batch plane)"},

    // ---- Load: the container, the memory, the upload ----------------------
    {"MODEL", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Load,
     "the container to run when --model is absent"},
    {"SEQ", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Load,
     "override the container's KV cache length (split over several textures past the GL limit)"},
    {"MAX_SEG", "", SwKind::Int, "1..64", SwStatus::Permanent, SwAxis::Load,
     "how many seq-sized textures the context may be split into (16 = 32768 on the board)"},
    {"SEG_DUMP", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Probe,
     "print the attention slice geometry at startup (seq, limit, slice length, count)"},
    {"SEG_CAP", "", SwKind::Int, "0..16384", SwStatus::Probe, SwAxis::Probe,
     "force a smaller attention segment than the driver needs, to A/B the split on a machine that does not need it"},
    {"TOKENIZER", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Load,
     "override the container's tokenizer blob; nothing warns, leave it unset"},
    {"MODEL_BORROW", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Load,
     "MODEL_BORROW=0 copies the CPU weights instead of borrowing the mapping"},
    {"KEEP_CPU_WEIGHTS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Load,
     "keep the CPU weights after the upload (debugging only)"},
    {"Q8_LAYERS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Load,
     "per-layer weight precision mix at load"},
    {"Q8_PARTS", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Load,
     "per-part weight precision mix at load"},
    {"UPLOAD_XOR", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Load,
     "UPLOAD_XOR=0 restores the byte-at-a-time int8 -> uint8 conversion"},

    // ---- Tok: the tokenizer ----------------------------------------------
    {"TOK_NO_PREFIX", "", SwKind::Flag, "", SwStatus::Permanent, SwAxis::Tok,
     "do not force a leading space/prefix before the first token"},
    {"TOK_SPACE_FOLD", "", SwKind::Flag, "", SwStatus::Permanent, SwAxis::Tok,
     "fold the space that SentencePiece prefixes onto the next piece"},

    // ---- Gen: sampling, penalties, stop policy ---------------------------
    {"IGNORE_EOS", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "keep generating past the stop token"},
    {"BLOCK_MARKERS", "", SwKind::Flag, "", SwStatus::Probe, SwAxis::Gen,
     "mask the <|...|> marker tokens while generating"},
    {"REPETITION_PENALTY", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "sampler: repetition penalty"},
    {"FREQUENCY_PENALTY", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "sampler: frequency penalty"},
    {"PRESENCE_PENALTY", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "sampler: presence penalty"},
    {"NO_REPEAT_NGRAM", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "sampler: block an n-gram from repeating"},
    {"MIN_P", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Gen,
     "sampler: min-p cut"},
    {"TOPK_FAST", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Gen,
     "the fast top-k selection path"},
    {"GPU_ARGMAX", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Gen,
     "greedy: reduce the packed q16 logits on the GPU"},
    {"LOGITS_SPLIT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Gen,
     "unpaired logits rows (a kernel-shape arm of the logits matvec)"},
    {"RMS_FUSED", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "force the fused rmsnorm reduce"},
    {"RMS_SPLIT", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Ffn,
     "force the split rmsnorm reduce"},

    // ---- Prefix: the disk cache ------------------------------------------
    {"PREFIX_CACHE_RESAVE", "", SwKind::Str, "", SwStatus::Probe, SwAxis::Prefix,
     "write the loaded KV rows back through the save path, to diff the two"},

    // ---- Env: read, but set by the environment ---------------------------
    {"TZ", "", SwKind::Str, "", SwStatus::Permanent, SwAxis::Env,
     "the timezone the tool's date functions render in"},

    // ---- Retired: documented as gone; the source must not read them ------
    // The 2026-11-16 q4 campaign's nine arms.  Each was a stage removal (ONEW,
    // ONEROW, NOOUT, NOP: wrong results on purpose) or a loop shape on the way
    // to the flat 8-wide gate (FLAT, FLAT8, FLAT8M, FLAT8G, FLAT8B), and the
    // flat 8-wide body IS the default now, so the road is the conclusion.  They
    // were deleted on 2026-11-17 (194 lines, shader_hashes.txt unchanged, which
    // is what makes the deletion safe to believe); docs/performance.md keeps
    // what they measured.
    {"Q4_4X4_FLAT", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the 4-wide flat gate arm; the flat 8-wide body is the default"},
    {"Q4_4X4_FLAT8", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the flat 8-wide arm; it became the default and the arm was removed"},
    {"Q4_4X4_FLAT8M", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: pricing arm, metadata applied per iteration"},
    {"Q4_4X4_FLAT8G", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: pricing arm, metadata decoded once per group"},
    {"Q4_4X4_FLAT8B", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the byte-identical candidate; it is the shipped gate now"},
    {"Q4_4X4_ONEW", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: stage-removal arm, one matrix of the 4-bit gate (8.18 -> 6.31 ms)"},
    {"Q4_4X4_ONEROW", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: stage-removal arm, one row of the 4-bit gate (-> 5.71 ms)"},
    {"Q4_4X4_NOOUT", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: stage-removal arm; the driver deletes the kernel, so it measured 0.8 ms"},
    {"Q4_4X4_NOP", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: pricing arm, constant scale/min (84 ms of the gate's 159 on the Pi)"},
    {"ATTN_HOIST", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the hoisted attention path is mandatory, there is no other score kernel"},
    {"ATTN_FUSE_PROB", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: there is no probability texture to read, att_v fuses the decode"},
    {"LEGACY_Q8", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: gone with the legacy q8 reader (the container states its precision)"},
    {"ROPE_THETA", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the container carries the rope configuration"},
    {"ROPE_HALF", "", SwKind::Str, "", SwStatus::Retired, SwAxis::Retired,
     "dead: the container carries the rope configuration"},
};

// The legacy spellings, and what they mean now.  Two kinds live here:
//
//   - a pure respelling keeps its value (RES_FMT=i24_1 -> RES=i24_1), and
//   - a value that the new axis spells differently (KV_PACK=16 -> KV=q16),
//     including the ones that used to set *one half* of an axis (ACT8=1 set the
//     GPU act streams and the FFN hidden; ACT8=2 set only the hidden).
//
// A rewrite only happens when the canonical name is unset: the new spelling
// wins, and the old one says so on stderr instead of quietly overriding it.
static const SwRename kRenames[] = {
    // activation axis
    {"ACT_FMT", "", "ACT", ""},
    {"ACT_F32", "", "ACT_CPU", "f32"},        // set-to-anything meant "on"
    {"ACT8", "1", "ACT", "i8"},
    {"ACT8", "1", "FFN_HID", "i8"},
    {"ACT8", "2", "FFN_HID", "i8"},
    {"ACT8", "0", "ACT", "i16"},
    {"ACT8", "0", "FFN_HID", "i16"},
    // residual axis: RES_PACK=0 was the unpacked q16 stream, one value a texel
    {"RES_FMT", "", "RES", ""},
    {"RES_PACK", "0", "RES", "i16_1"},
    // KV axis: the two halves of it
    {"KV_PACK", "0", "KV", "i24"},
    {"KV_PACK", "8", "KV", "q8"},
    {"KV_PACK", "16", "KV", "q16"},
    {"KV_FMT", "", "KV", ""},
    {"KV_SCALE", "", "KV_GRID", ""},
    // attention grids
    {"ATT_SCALE", "", "ATT_SCORE_GRID", ""},
    {"ATT_SUM_SCALE", "", "ATT_SUM_GRID", ""},
};

static const int kSwitchCount = (int) (sizeof(kSwitches) / sizeof(kSwitches[0]));
static const int kRenameCount = (int) (sizeof(kRenames) / sizeof(kRenames[0]));

int sw_count() { return kSwitchCount; }

const SwitchDef& sw_at(int i) {
    if (i < 0 || i >= kSwitchCount) i = 0;
    return kSwitches[i];
}

const char* sw_axis_name(SwAxis a) {
    switch (a) {
        case SwAxis::Dump: return "dump";
        case SwAxis::Probe: return "probe";
        case SwAxis::Place: return "place";
        case SwAxis::Act: return "act";
        case SwAxis::Att: return "att";
        case SwAxis::Kv: return "kv";
        case SwAxis::Res: return "res";
        case SwAxis::Ffn: return "ffn";
        case SwAxis::Q4: return "q4";
        case SwAxis::Rwkv: return "rwkv";
        case SwAxis::Prefill: return "prefill";
        case SwAxis::Encdec: return "encdec";
        case SwAxis::Load: return "load";
        case SwAxis::Tok: return "tok";
        case SwAxis::Gen: return "gen";
        case SwAxis::Prefix: return "prefix";
        case SwAxis::Env: return "env";
        case SwAxis::Retired: return "retired";
    }
    return "?";
}

const char* sw_kind_name(SwKind k) {
    switch (k) {
        case SwKind::Flag: return "flag";
        case SwKind::Int: return "int";
        case SwKind::Num: return "num";
        case SwKind::Str: return "str";
    }
    return "?";
}

const char* sw_status_name(SwStatus s) {
    switch (s) {
        case SwStatus::Permanent: return "permanent";
        case SwStatus::Probe: return "probe";
        case SwStatus::Retired: return "retired";
    }
    return "?";
}

const SwitchDef* sw_find(const char* name) {
    if (!name) return nullptr;
    for (int i = 0; i < kSwitchCount; ++i)
        if (!strcmp(kSwitches[i].name, name)) return &kSwitches[i];
    return nullptr;
}

// "a|b|c" or "lo..hi" -> is `v` one of them?  An empty set means free form,
// which is what most switches are: `MODEL` takes a path, `SEQ` a number.
static bool value_ok(const char* set, const char* v) {
    if (!set || !*set) return true;
    if (const char* dots = strstr(set, "..")) {
        char* end = nullptr;
        const long lo = strtol(set, &end, 10);
        const long hi = strtol(dots + 2, &end, 10);
        char* vEnd = nullptr;
        const long x = strtol(v, &vEnd, 10);
        return vEnd && *vEnd == '\0' && x >= lo && x <= hi;
    }
    const size_t n = strlen(v);
    for (const char* p = set; *p;) {
        const char* q = strchr(p, '|');
        const size_t len = q ? (size_t) (q - p) : strlen(p);
        if (len == n && !strncmp(p, v, n)) return true;
        if (!q) break;
        p = q + 1;
    }
    return false;
}

namespace {
struct Resolved {
    map<string, string> value;   // canonical name -> value (only when set)
    bool inited = false;
};

Resolved g_sw;

// One line per legacy spelling, however many times the engine reads it.
static void warn_rename(const SwRename& r, const char* v) {
    fprintf(stderr,
            "[switch] %s is now %s=%s (the old spelling still works; see "
            "docs/test-hooks.md)\n",
            r.legacy, r.name, (r.becomes && *r.becomes) ? r.becomes : v);
}
}  // namespace

void sw_init() {
    if (g_sw.inited) return;
    g_sw.inited = true;

    // 1. the canonical spellings.
    for (int i = 0; i < kSwitchCount; ++i) {
        const SwitchDef& d = kSwitches[i];
        if (d.status == SwStatus::Retired) continue;
        if (const char* v = getenv(d.name)) g_sw.value[d.name] = v;
        // A pure respelling of the same axis, same values.
        if (!g_sw.value.count(d.name) && d.aliases && *d.aliases) {
            string list(d.aliases);
            for (size_t p = 0; p < list.size();) {
                size_t q = list.find(',', p);
                if (q == string::npos) q = list.size();
                const string alias = list.substr(p, q - p);
                p = q + 1;
                if (alias.empty()) continue;
                if (const char* v = getenv(alias.c_str())) {
                    g_sw.value[d.name] = v;
                    fprintf(stderr, "[switch] %s is now %s (the old spelling "
                                    "still works; see docs/test-hooks.md)\n",
                            alias.c_str(), d.name);
                    break;
                }
            }
        }
    }

    // 2. the legacy value spellings, only where the canonical name is unset.
    for (int i = 0; i < kRenameCount; ++i) {
        const SwRename& r = kRenames[i];
        const char* v = getenv(r.legacy);
        if (!v) continue;
        if (r.value && *r.value && strcmp(v, r.value) != 0) continue;
        if (g_sw.value.count(r.name)) continue;  // the new spelling wins
        const string nv = (r.becomes && *r.becomes) ? string(r.becomes) : string(v);
        g_sw.value[r.name] = nv;
        warn_rename(r, v);
    }

    // 3. a closed set is checked once, for every set switch, before any kernel
    //    reads it: a typo must not fall back to a default (the TOKENIZER lesson).
    for (int i = 0; i < kSwitchCount; ++i) {
        const SwitchDef& d = kSwitches[i];
        auto it = g_sw.value.find(d.name);
        if (it == g_sw.value.end()) continue;
        if (!value_ok(d.values, it->second.c_str())) {
            fprintf(stderr, "switch %s=%s: expected one of %s\n", d.name,
                    it->second.c_str(), d.values);
            exit(2);
        }
    }
}

bool sw_isset(const char* name) {
    sw_init();
    const SwitchDef* d = sw_find(name);
    const char* n = d ? d->name : name;
    return g_sw.value.count(n) != 0;
}

const char* sw_str(const char* name) {
    sw_init();
    const SwitchDef* d = sw_find(name);
    const char* n = d ? d->name : name;
    auto it = g_sw.value.find(n);
    if (it != g_sw.value.end()) return it->second.c_str();
    // an unregistered name still works: the registry is the inventory, not a
    // gatekeeper (docs_check.sh is what keeps the inventory complete).
    if (!d) return getenv(name);
    return nullptr;
}

const char* sw_enum(const char* name) { return sw_str(name); }

bool sw_flag(const char* name) {
    const char* v = sw_str(name);
    return v && strcmp(v, "0") != 0;
}

int sw_int(const char* name, int def) {
    const char* v = sw_str(name);
    return v ? atoi(v) : def;
}

double sw_num(const char* name, double def) {
    const char* v = sw_str(name);
    return v ? atof(v) : def;
}

void sw_dump(FILE* out) {
    sw_init();
    fprintf(out, "# %d switches (%d renames).  `set` is what this environment "
                 "resolved to.\n", kSwitchCount, kRenameCount);
    const char* axis = "";
    for (int i = 0; i < kSwitchCount; ++i) {
        const SwitchDef& d = kSwitches[i];
        const char* a = sw_axis_name(d.axis);
        if (strcmp(a, axis)) {
            fprintf(out, "\n## %s\n", a);
            axis = a;
        }
        auto it = g_sw.value.find(d.name);
        fprintf(out, "%-18s %-9s %-10s %-22s %s\n", d.name, sw_kind_name(d.kind),
                sw_status_name(d.status),
                (d.values && *d.values) ? d.values : "-",
                it != g_sw.value.end() ? ("= " + it->second).c_str() : "");
        fprintf(out, "  %s\n", d.note);
        if (d.aliases && *d.aliases) fprintf(out, "  aliases: %s\n", d.aliases);
    }
}
