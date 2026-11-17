"""The MMCF container layout, in one place.

Three tools read and write this format (`mmcf_pack.py`, `mmcf_dump.py`,
`mmcf_requant_q8.py`) and until 2026-11-14 each carried its own copy of the magic,
the record size, the `struct` format strings, the TLV helper and the tables of
dtypes, kinds and template kinds.  They happened to agree; the dtype table already
did not (q6 was missing from one copy, which is how P1-3 found it).  This module is
now the source, and `engine/gate/test_mmcf_format.py` proves it against real
containers and against the C++ reader.

Layout (little-endian, no implicit padding -- every `struct` format is `<`):

    0    magic "MMCF"
    4    version u16, header size u16, n_tensors u32, caps u32,
         meta_off u32? no: meta_off u32, meta_size u32, data_off u64, file_size u64
    40   header crc32 (of bytes 0..39, 0 = absent)
    44   reserved to header size (64)
    then the metadata TLV block, then the tensor record table, then the data.

A tensor record is 48 bytes in v2 and 64 in v3 (v3 appends `aux` for MoE)
and v4 (RWKV-7) keeps v3's 64-byte record unchanged: everything v4 adds is
model-wide (metadata key 16) or a new tensor *kind* whose shape the record
already carries, so no per-tensor field had to grow.  `rec_unpack` returns the
same 11-tuple for every version by padding the v2 case.

Versions: 2 dense, 3 mixture of experts, 4 RWKV-7.  The number is what lets an
older engine refuse a file at the header instead of half-loading it (a v4 file
carries tensors a v3 reader would ignore, and a reader that ignored them would
run a transformer block over an RWKV checkpoint).
"""

import struct
import zlib

MAGIC = b"MMCF"
HEADER_SIZE = 64
# magic, version, header size, n_tensors, caps, meta_off, meta_size, data_off, size
HEADER_FMT = "<4sHHIIIIQQ"
HEADER_CRC_OFF = 40
HEADER_CRC_SPAN = 40

REC_SIZE_V2 = 48
REC_SIZE_V3 = 64

# Container versions.  A file is dense (2), mixture of experts (3) or RWKV-7
# (4) and the reader dispatches on the number, not on a magic.
VERSION_DENSE = 2
VERSION_MOE = 3
VERSION_RWKV7 = 4


def rec_size(version):
    """Bytes per tensor record for a container version.

    v4 (RWKV-7) reuses v3's 64-byte record.  Nothing v4 adds is per tensor: the
    architecture's hyper-parameters (head width, the four low ranks, the
    LayerNorm epsilon) are model-wide and live in metadata key 16, and the new
    tensor kinds carry their shape in the record's existing rows/cols fields.
    A longer record would therefore be 16 zero bytes per tensor, and keeping 64
    is what lets one reader parse v2/v3/v4 tables with the same field offsets.
    """
    return REC_SIZE_V3 if version >= 3 else REC_SIZE_V2


# --- dtypes -------------------------------------------------------------
# 6 is the 6-bit embedding body `mmcf_pack.py --head-q6` writes: two planes per row
# (dim/2 bytes of 4 bits, dim/4 bytes of 2) plus a per-row scale tensor.
DT_F32, DT_I16, DT_Q8U8, DT_Q8I8, DT_Q4, DT_BLOB, DT_Q6 = range(7)
DTYPES = {DT_F32: "f32", DT_I16: "i16", DT_Q8U8: "q8u8", DT_Q8I8: "q8i8",
          DT_Q4: "q4", DT_BLOB: "blob", DT_Q6: "q6"}

# --- tensor ids ---------------------------------------------------------
# The id is kind(8) | layer(6) | part(2); part 0 is the tensor, parts 1/2 are a
# q4 matrix's group scale/min planes and part 3 is a q8 per-row scale plane.
#
# RWKV-7 (v4) reuses four ids and opens 17 of its own: the time-mix block has
# no qkv semantics, but its three projections are three (dim x dim) matrices in
# one row block (0x10, exactly how a qkv is read), and W_o / the channel-mix
# key / value have the same shapes as wo / w1 / w2 (0x11/0x12/0x15).  The
# remaining per-layer tensors are the ones a transformer block never had: six
# token-shift lerps, three low-rank bias vectors, three per-channel scales,
# three LayerNorms (w and b inside one row, like rms_att), the channel-mix
# lerp, and the four low-rank pairs (decay, value-residual, ICLR, gate).  The
# names follow llama.cpp's so a reader can compare the two.
KINDS = {
    1: "embedding", 2: "embedding_scale", 3: "rms_att", 4: "rms_ffn",
    5: "rms_final", 6: "q_norm", 7: "k_norm", 8: "embedding_i16",
    0x10: "qkv", 0x11: "wo", 0x12: "w1", 0x13: "w3", 0x14: "w13", 0x15: "w2",
    0x16: "router",
    0x17: "conv_in", 0x18: "conv_w", 0x19: "conv_out",
    0x1A: "time_mix_lerp", 0x1B: "time_mix_bias", 0x1C: "time_mix_scale",
    0x1D: "att_norm", 0x1E: "att_norm_2", 0x1F: "time_mix_ln",
    0x20: "logits_tile", 0x21: "channel_mix_lerp", 0x22: "tok_norm",
    0x23: "output_norm",
    0x24: "time_mix_w1", 0x25: "time_mix_w2",
    0x26: "time_mix_v1", 0x27: "time_mix_v2",
    0x28: "time_mix_a1", 0x29: "time_mix_a2",
    0x2A: "time_mix_g1", 0x2B: "time_mix_g2",
    0x30: "tokenizer_blob",
}

# Kinds a per-layer record uses (the layer bits are 0 for the model-wide
# vector records and for the two global norms).
PER_LAYER_KINDS = frozenset([0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
                              0x17, 0x18, 0x19,
                              0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B])

# The RWKV-7 ids as named constants, so the packer, the dump tool and the
# reader-side tables name the same number.
K_RWKV_LERP, K_RWKV_BIAS, K_RWKV_SCALE = 0x1A, 0x1B, 0x1C
K_RWKV_ATT_NORM, K_RWKV_ATT_NORM_2, K_RWKV_LN_X = 0x1D, 0x1E, 0x1F
K_RWKV_CM_LERP = 0x21
K_RWKV_TOK_NORM, K_RWKV_OUT_NORM = 0x22, 0x23
K_RWKV_W1, K_RWKV_W2 = 0x24, 0x25
K_RWKV_V1, K_RWKV_V2 = 0x26, 0x27
K_RWKV_A1, K_RWKV_A2 = 0x28, 0x29
K_RWKV_G1, K_RWKV_G2 = 0x2A, 0x2B
# kind -> (branch, 0 = down projection rank x dim, 1 = up projection dim x rank)
RWKV_MATS = {
    K_RWKV_W1: ("decay", 0), K_RWKV_W2: ("decay", 1),
    K_RWKV_V1: ("value", 0), K_RWKV_V2: ("value", 1),
    K_RWKV_A1: ("iclr", 0), K_RWKV_A2: ("iclr", 1),
    K_RWKV_G1: ("gate", 0), K_RWKV_G2: ("gate", 1),
}

# The f32 per-channel vector groups of a v4 container: kind -> (name, group
# order).  The record is rows = layers (1 for the two globals) and
# cols = len(groups) * dim, with the groups in the order listed -- the HF/GGUF
# order, so a reader can compare against llama.cpp byte for byte.
RWKV_VECTORS = {
    K_RWKV_LERP:       ("time_mix_lerp", ("r", "w", "k", "v", "a", "g")),
    K_RWKV_BIAS:       ("time_mix_bias", ("w0", "a0", "v0")),
    K_RWKV_SCALE:      ("time_mix_scale", ("k_k", "k_a", "r_k")),
    K_RWKV_ATT_NORM:   ("att_norm", ("w", "b")),
    K_RWKV_ATT_NORM_2: ("att_norm_2", ("w", "b")),
    K_RWKV_LN_X:       ("time_mix_ln", ("w", "b")),
    K_RWKV_CM_LERP:    ("channel_mix_lerp", ("k",)),
    K_RWKV_TOK_NORM:   ("tok_norm", ("w", "b")),
    K_RWKV_OUT_NORM:   ("output_norm", ("w", "b")),
}
# The model-wide ones (rows = 1, layer bits 0).
RWKV_GLOBAL_VECTORS = (K_RWKV_TOK_NORM, K_RWKV_OUT_NORM)

# --- RWKV-7 metadata (key 16) -------------------------------------------
# One record per model, 16 bytes: the architecture, the WKV head width, a flag
# word, the LayerNorm epsilon and the four low ranks.  The ranks cannot be
# derived: llama.cpp's fallback formula (round(d^e * m / 32) * 32) gives 160
# for 0.4B's gate where the config states 128, so the file carries them.
#
# The per-head group norm's epsilon (64e-5) is deliberately *not* here: it is
# part of the architecture (llama.cpp and the official numpy forward both
# hardcode it), and a model that wanted another one would be a different
# block, not a parameter.
KEY_RWKV = 16

# Metadata key 17: the head input's store scale exponent, one byte `k`.
#
# The vector that feeds the logits matvec is the final LayerNorm's output, and
# the engine writes it through the q16 activation format (+-127.996).  RWKV-7's
# 0.1B peaks at 315.6 there -- 2.5x the rail -- so the packer stores it *2^-k
# and multiplies every logits tile's scale by 2^k, which keeps the dot product
# in the model's own units with no kernel change.  Absent (or 0) = the input is
# written unscaled, which is every container packed before 2026-11-16.
KEY_LOGIT_SHIFT = 17
RWKV_META_FMT = "<BBBBfHHHH"    # arch, head_size, flags, pad, norm_eps, r_w, r_a, r_v, r_g
RWKV_META_SIZE = struct.calcsize(RWKV_META_FMT)
RWKV_ARCH_RWKV7 = 1
RWKV_ARCH_NAMES = {RWKV_ARCH_RWKV7: "rwkv7"}
# flags bit 0: the time-mix has the optional gate branch (W_g1/W_g2 present)
RWKV_FLAG_GATE = 1


def pack_rwkv_meta(arch=RWKV_ARCH_RWKV7, head_size=64, flags=0, norm_eps=1e-5,
                   r_w=0, r_a=0, r_v=0, r_g=0):
    """Metadata key 16's payload (16 bytes; see RWKV_META_FMT)."""
    return struct.pack(RWKV_META_FMT, int(arch), int(head_size), int(flags), 0,
                       float(norm_eps), int(r_w), int(r_a), int(r_v), int(r_g))


def parse_rwkv_meta(payload):
    """The inverse: the dict the reader and the dump tool use."""
    if len(payload) < RWKV_META_SIZE:
        raise ValueError("rwkv metadata is %d B, the record is %d"
                         % (len(payload), RWKV_META_SIZE))
    arch, head, flags, _pad, eps, r_w, r_a, r_v, r_g = \
        struct.unpack(RWKV_META_FMT, payload[:RWKV_META_SIZE])
    return dict(arch=arch, head_size=head, flags=flags, norm_eps=eps,
                r_w=r_w, r_a=r_a, r_v=r_v, r_g=r_g,
                gate=bool(flags & RWKV_FLAG_GATE))

SCHEMES = {0: "i16", 1: "q8", 2: "q4"}

# kind byte -> the name `mmcf_dump.py` prints.  Kind 1 covers both chatml and
# qwen, which is why the display name carries both.
TEMPLATE_KIND_NAMES = {0: "plain", 1: "chatml/qwen", 2: "minimind3", 3: "tinybrainbot"}

# name -> kind byte in metadata key 4.  The packer validates against this table
# (an unknown name used to become ChatML silently) and the engine's fallback path
# reads the *name*, so the two have to agree.
TEMPLATE_KINDS = {"plain": 0, "qwen": 1, "chatml": 1, "minimind3": 2, "tinybrainbot": 3}

# metadata key 5 mirrors the header's caps word: bit 3 is reserved and the packer
# never sets it, and the engine reads the word only to discard it.
CAPS = [(1, "chat"), (2, "tools"), (4, "vision"), (8, "reserved")]


def crc32(data):
    """The CRC the header and every tensor record carry."""
    return zlib.crc32(data) & 0xFFFFFFFF


def tlv(key, typ, payload):
    """One metadata entry: key u16, type u16, length u32, payload."""
    return struct.pack("<HHI", key, typ, len(payload)) + payload


def rec_unpack(rec):
    """(id, dtype, flags, group, rows, cols, off, bytes, crc, scale, aux)."""
    tid, dt, flags, group, _pad = struct.unpack("<HBBBB", rec[:6])
    rows, cols = struct.unpack("<II", rec[6:14])
    off, blen = struct.unpack("<QQ", rec[14:30])
    crc, = struct.unpack("<I", rec[30:34])
    scale, = struct.unpack("<f", rec[34:38])
    aux, = struct.unpack("<H", rec[38:40]) if len(rec) >= 40 else (0,)
    return tid, dt, flags, group, rows, cols, off, blen, crc, scale, aux


def pack_rec(tid, dt, flags, group, rows, cols, off, blen, crc, scale, aux=0,
             size=REC_SIZE_V3):
    """The inverse of rec_unpack; `size` truncates to the v2 layout."""
    b = struct.pack("<HBBBB", tid, dt, flags, group, 0)
    b += struct.pack("<II", rows, cols)
    b += struct.pack("<QQ", off, blen)
    b += struct.pack("<I", crc)
    b += struct.pack("<f", float(scale))
    b += struct.pack("<H", aux)
    # v2's 48-byte record is the 40 bytes above plus 8 reserved (zero) bytes; the
    # packer has always written them, so the round trip has to reproduce them.
    return b[:size] + b"\0" * max(0, size - len(b))


class Header:
    """The 64-byte header, parsed.  `raw` keeps the bytes for a round trip."""

    __slots__ = ("magic", "version", "header_size", "n_tensors", "caps",
                 "meta_off", "meta_size", "data_off", "file_size", "header_crc",
                 "header_crc_ok", "rec_size")

    def __init__(self, raw):
        if raw[:4] != MAGIC:
            raise ValueError("not an MMCF container (magic %r)" % raw[:4])
        (magic, self.version, self.header_size, self.n_tensors, self.caps,
         self.meta_off, self.meta_size, self.data_off, self.file_size) = \
            struct.unpack_from(HEADER_FMT, raw, 0)
        self.magic = magic
        self.header_crc, = struct.unpack_from("<I", raw, HEADER_CRC_OFF)
        self.header_crc_ok = (self.header_crc == 0 or
                              self.header_crc == crc32(raw[:HEADER_CRC_SPAN]))
        self.rec_size = rec_size(self.version)

    def pack(self):
        """Rebuild the first 64 bytes; used by the round-trip test."""
        out = struct.pack(HEADER_FMT, self.magic, self.version, self.header_size,
                          self.n_tensors, self.caps, self.meta_off, self.meta_size,
                          self.data_off, self.file_size)
        out += struct.pack("<I", crc32(out[:HEADER_CRC_SPAN]) if self.header_crc else 0)
        return out + b"\0" * (self.header_size - len(out))


def parse_header(raw):
    return Header(raw)
