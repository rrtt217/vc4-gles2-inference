"""The MMCF container layout, in one place.

Three tools read and write this format (`mmcf_pack.py`, `mmcf_dump.py`,
`mmcf_requant_q8.py`) and until 2026-11-14 each carried its own copy of the magic,
the record size, the `struct` format strings, the TLV helper and the tables of
dtypes, kinds and template kinds.  They happened to agree; the dtype table already
did not (q6 was missing from one copy, which is how P1-3 found it).  This module is
now the source, and `engine/scripts/test_mmcf_format.py` proves it against real
containers and against the C++ reader.

Layout (little-endian, no implicit padding -- every `struct` format is `<`):

    0    magic "MMCF"
    4    version u16, header size u16, n_tensors u32, caps u32,
         meta_off u32? no: meta_off u32, meta_size u32, data_off u64, file_size u64
    40   header crc32 (of bytes 0..39, 0 = absent)
    44   reserved to header size (64)
    then the metadata TLV block, then the tensor record table, then the data.

A tensor record is 48 bytes in v2 and 64 in v3 (v3 appends `scale` and `aux` for
MoE); `rec_unpack` returns the same 11-tuple for either by padding the v2 case.
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


def rec_size(version):
    """Bytes per tensor record for a container version."""
    return REC_SIZE_V3 if version >= 3 else REC_SIZE_V2


# --- dtypes -------------------------------------------------------------
# 6 is the 6-bit embedding body `mmcf_pack.py --head-q6` writes: two planes per row
# (dim/2 bytes of 4 bits, dim/4 bytes of 2) plus a per-row scale tensor.
DT_F32, DT_I16, DT_Q8U8, DT_Q8I8, DT_Q4, DT_BLOB, DT_Q6 = range(7)
DTYPES = {DT_F32: "f32", DT_I16: "i16", DT_Q8U8: "q8u8", DT_Q8I8: "q8i8",
          DT_Q4: "q4", DT_BLOB: "blob", DT_Q6: "q6"}

# --- tensor ids ---------------------------------------------------------
KINDS = {
    1: "embedding", 2: "embedding_scale", 3: "rms_att", 4: "rms_ffn",
    5: "rms_final", 6: "q_norm", 7: "k_norm", 8: "embedding_i16",
    0x10: "qkv", 0x11: "wo", 0x12: "w1", 0x13: "w3", 0x14: "w13", 0x15: "w2",
    0x16: "router", 0x20: "logits_tile", 0x30: "tokenizer_blob",
}

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
