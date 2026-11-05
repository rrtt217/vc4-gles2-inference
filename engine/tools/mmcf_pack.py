#!/usr/bin/env python3
"""Pack an existing vc4 checkpoint into the MMCF v2 container.

    mmcf_pack.py --in minimind3.q8 --tok minimind3.tok --out minimind3.mmcf \
                 --name minimind3 --template chatml \
                 --system "You are a helpful assistant." --chat

Why: the legacy files carry a positional header and need a new magic for every
change, the tokenizer and chat template live outside the file, and q8 stores
signed bytes that the engine must bias by +128 at upload time (69 MB of work
plus a 69 MB temporary buffer).  MMCF v2 has one header, a key/value metadata
block, an explicit tensor table, and a `q8u8` dtype that is already biased so
the upload is a straight copy.

Layout

    0   header 64 B
    64  metadata: TLV records (u16 key, u16 type, u32 len, payload)
        tensor table: n * 40 B
        data: every tensor 16-byte aligned

This tool also verifies itself: `--verify` re-reads the file and compares every
tensor with the source bytes.  A conversion that changes one byte is a bug, not
a re-baseline.
"""

import argparse
import json
import struct

# tensor table record: id, dtype, flags, group, pad, rows, cols, offset, bytes,
# crc32, pad, scale, pad  -> exactly 48 bytes, no implicit padding with "<"
# tensor table record, built by explicit concatenation so its size is exact:
#   id u16 | dtype u8 | flags u8 | group u8 | pad u8 | rows u32 | cols u32
#   offset u64 | bytes u64 | crc32 u32 | scale f32 | pad to 48
REC_SIZE = 48


def pack_rec(tid, dt, flags, group, rows, cols, off, blen, crc, scale):
    b = struct.pack("<HBBBB", tid, dt, flags, group, 0)
    b += struct.pack("<II", rows, cols)
    b += struct.pack("<QQ", off, blen)
    b += struct.pack("<I", crc)
    b += struct.pack("<f", float(scale))
    return b.ljust(REC_SIZE, b"\0")


def unpack_rec(rec):
    tid, dt, flags, group, _pad = struct.unpack("<HBBBB", rec[:6])
    rows, cols = struct.unpack("<II", rec[6:14])
    off, blen = struct.unpack("<QQ", rec[14:30])
    crc, = struct.unpack("<I", rec[30:34])
    scale, = struct.unpack("<f", rec[34:38])
    return tid, dt, rows, cols, off, blen, crc, scale
import sys
from pathlib import Path

MAGIC = b"MMCF"
VERSION = 2
HEADER_SIZE = 64
ALIGN = 16

DT_F32 = 0
DT_I16 = 1
DT_Q8U8 = 2   # q8, already biased to 0..255 (the upload is a copy)
DT_Q8I8 = 3   # q8, signed (the upload flips the sign bit)
DT_Q4 = 4
DT_BLOB = 5

# Tensor ids.  kind in the low byte, layer in bits 8..13, part in bits 14..15
# (part 0 = the tensor itself, 1/2 = the q4 group scale and min planes).
T_EMB, T_EMB_SCALE, T_RMS_ATT, T_RMS_FFN, T_RMS_FINAL = 1, 2, 3, 4, 5
T_Q_NORM, T_K_NORM = 6, 7
T_QKV, T_WO, T_W1, T_W3, T_W13, T_W2 = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
T_LOGITS = 0x20
T_TOKENIZER = 0x30

# from engine/src/common.h
MAGIC_Q8 = 0x51384D31      # Q8M1
MAGIC_Q8_V2 = 0x51384D32   # Q8M2
MAGIC_I16 = 0x4931364D     # I16M1
MAGIC_I16_V2 = 0x4931364E  # I16N
MAGIC_Q4 = 0x51344D31      # Q4M1


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def i32(self):
        return struct.unpack("<i", self.f.read(4))[0]

    def f32(self):
        return struct.unpack("<f", self.f.read(4))[0]

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit("truncated input")
        return b


def parse_legacy(path):
    """Return (meta, [(id, dtype, rows, cols, group, bytes)]).

    dtypes here are the *source* ones; the packer maps q8 to q8u8 by flipping
    the sign bit when it writes the data.
    """
    r = Reader(path)
    magic = r.u32()
    cfg = [r.i32() for _ in range(7)]
    dim, hidden, layers, heads, kv_heads, vocab, seq = cfg
    head_dim = dim // heads
    tensors = []
    meta = dict(dim=dim, hidden=hidden, layers=layers, heads=heads,
                kv_heads=kv_heads, vocab=vocab, seq=seq, head_dim=head_dim)

    def mat(tid, layer=0, q8=True):
        rows, cols = r.i32(), r.i32()
        scale = r.f32()
        n = rows * cols
        data = r.take(n if q8 else n * 2)
        tensors.append((tid | (layer << 8), DT_Q8I8 if q8 else DT_I16,
                        rows, cols, 0, data, scale))

    if magic in (MAGIC_Q8, MAGIC_Q8_V2):
        split = 0
        if magic == MAGIC_Q8_V2:
            meta["theta"] = r.f32()
            meta["half"] = r.i32() != 0
            split = r.i32()
            meta["qk_norm"] = r.i32() != 0
        else:
            split = r.i32()
            meta["theta"] = 10000.0
            meta["half"] = False
            meta["qk_norm"] = False
        meta["scheme"], meta["group"], meta["split"] = "q8", 0, bool(split)
        tensors.append((T_EMB, DT_Q8I8, vocab, dim, 0, r.take(vocab * dim), 0.0))  # CPU side: stay signed
        scale_bytes = r.take(vocab * 4)
        tensors.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, scale_bytes, 0.0))
        tensors.append((T_RMS_ATT, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FFN, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FINAL, DT_F32, 1, dim, 0, r.take(dim * 4), 0.0))
        if meta["qk_norm"]:
            tensors.append((T_Q_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))
            tensors.append((T_K_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))
        for l in range(layers):
            mat(T_QKV, l)
            mat(T_WO, l)
            if split:
                mat(T_W1, l)
                mat(T_W3, l)
            else:
                mat(T_W13, l)
            mat(T_W2, l)
        for i in range((vocab + 1535) // 1536):
            mat(T_LOGITS, i)
    elif magic in (MAGIC_I16, MAGIC_I16_V2):
        raise SystemExit("i16 packing not implemented yet")
    elif magic == MAGIC_Q4:
        theta = r.f32(); half = r.i32(); split = r.i32(); qk = r.i32()
        meta["theta"] = theta
        meta["half"] = half != 0
        meta["qk_norm"] = qk != 0
        meta["scheme"], meta["group"], meta["split"] = "q4", 32, bool(split)
        tensors.append((T_EMB, DT_Q8I8, vocab, dim, 0, r.take(vocab * dim), 0.0))
        tensors.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, r.take(vocab * 4), 0.0))
        tensors.append((T_RMS_ATT, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FFN, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FINAL, DT_F32, 1, dim, 0, r.take(dim * 4), 0.0))
        if meta["qk_norm"]:
            tensors.append((T_Q_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))
            tensors.append((T_K_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))

        def q4mat(tid, layer, part4):
            rows, cols, groups = r.i32(), r.i32(), r.i32()
            ps = r.f32()
            nib = r.take(rows * cols // 2)
            scl = r.take(rows * groups * 2)
            mn = r.take(rows * groups * 2)
            base = tid | (layer << 8)
            tensors.append((base, DT_Q4, rows, cols, 0, nib, ps))
            tensors.append((base | (1 << 14), DT_I16, rows, groups, 0, scl, 0.0))
            tensors.append((base | (2 << 14), DT_I16, rows, groups, 0, mn, 0.0))
            _ = part4

        for l in range(layers):
            q4mat(T_QKV, l, True)
            q4mat(T_WO, l, True)
            if split:
                q4mat(T_W1, l, True)
                q4mat(T_W3, l, True)
            else:
                q4mat(T_W13, l, True)
            q4mat(T_W2, l, True)
        for i in range((vocab + 1535) // 1536):
            rows, cols = r.i32(), r.i32()
            sc = r.f32()
            tensors.append((T_LOGITS | (i << 8), DT_Q8I8, rows, cols, 0, r.take(rows * cols), sc))
    else:
        raise SystemExit("unknown magic 0x%08x" % magic)
    return meta, tensors


def tlv(key, typ, payload):
    return struct.pack("<HHI", key, typ, len(payload)) + payload


def build(meta, tensors, tok_blob, args):
    model = struct.pack("<7i", meta["dim"], meta["hidden"], meta["layers"],
                        meta["heads"], meta["kv_heads"], meta["vocab"], meta["seq"])
    rope = struct.pack("<fBBH", float(meta.get("theta", 10000.0)),
                       1 if meta.get("half") else 0,
                       1 if meta.get("qk_norm") else 0, 0)
    scheme = {"i16": 0, "q8": 1, "q4": 2}[meta["scheme"]]
    quant = struct.pack("<BBBB", scheme, meta.get("group", 0),
                        1,  # q8u8 pre-biased
                        1 if meta.get("split") else 0)
    template = (bytes([1]) + args.system.encode() + b"\0" + args.template.encode() + b"\0")
    caps_val = (1 if args.chat else 0) | (2 if args.tools else 0) | (4 if args.vision else 0)
    caps = struct.pack("<I", caps_val)
    md = (tlv(1, 1, model) + tlv(2, 1, rope) + tlv(3, 1, quant) +
          tlv(4, 1, template) + tlv(5, 1, caps) + tlv(6, 1, args.name.encode() + b"\0"))
    if tok_blob is not None:
        tensors = tensors + [(T_TOKENIZER, DT_BLOB, len(tok_blob), 1, 0, tok_blob, 0.0)]
        md += tlv(7, 1, struct.pack("<i", len(tok_blob)))
    # NOTE: every TLV must be appended before the offsets below are computed.

    meta_off = HEADER_SIZE
    table_off = meta_off + len(md)
    table_off = (table_off + ALIGN - 1) // ALIGN * ALIGN
    table = bytearray()
    data = bytearray()
    data_off = table_off + len(tensors) * REC_SIZE
    import zlib
    for tid, dt, rows, cols, group, payload, scale in tensors:
        # +128 is a bit flip.  Texture tensors (the per-layer matrices and the
        # logits tiles) are stored biased so the upload is a copy; the embedding
        # stays signed because the CPU reads it.
        if dt == DT_Q8I8 and (tid & 0xFF) != T_EMB:
            payload = bytes((b ^ 0x80) for b in payload)
            dt = DT_Q8U8
        while (data_off + len(data)) % ALIGN:
            data.append(0)
        off = len(data)
        data += payload
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        table += pack_rec(tid, dt, 0, group, rows, cols, off, len(payload), crc, scale)
    data_off_final = table_off + len(tensors) * REC_SIZE
    file_size = data_off_final + len(data)
    head = MAGIC + struct.pack("<HHIIIIQQI", VERSION, HEADER_SIZE, len(tensors),
                               caps_val, meta_off, len(md), data_off_final, file_size, 0)
    head = head.ljust(HEADER_SIZE, b"\0")
    out = bytearray(head)
    out += md
    out += b"\0" * (table_off - len(out))
    out += table
    out += data
    return bytes(out)


def verify(path, tensors):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise SystemExit("verify: bad magic")
    ver, hsz, n, caps, mo, ms, doff, fsz, _crc = struct.unpack("<HHIIIIQQI", raw[4:44])
    if ver != VERSION or hsz != HEADER_SIZE:
        raise SystemExit("verify: header mismatch")
    if fsz != len(raw):
        raise SystemExit("verify: file size field %d != %d" % (fsz, len(raw)))
    import zlib
    bad = 0
    for i in range(n):
        rec = raw[doff - n * REC_SIZE + i * REC_SIZE: doff - n * REC_SIZE + (i + 1) * REC_SIZE]
        tid, dt, rows, cols, off, blen, crc, _sc = unpack_rec(rec)
        body = raw[doff + off: doff + off + blen]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            bad += 1
    return n, caps, mo, ms, doff, len(raw), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--tok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--template", default="chatml")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    meta, tensors = parse_legacy(args.src)
    tok = Path(args.tok).read_bytes() if args.tok else None
    raw = build(meta, tensors, tok, args)
    Path(args.out).write_bytes(raw)
    print("wrote %s: %d tensors, %d bytes (source %s)" %
          (args.out, len(tensors) + (1 if tok else 0), len(raw), args.src))

    if args.verify:
        n, caps, mo, ms, doff, fsz, bad = verify(args.out, tensors)
        print("verify: %d tensors, caps=0x%x meta=%d+%d data@%d size=%d crc_bad=%d" %
              (n, caps, mo, ms, doff, fsz, bad))
        if bad:
            raise SystemExit("verify failed")
        # weights must match the source bit for bit (after the bias)
        for tid, dt, rows, cols, group, payload, scale in tensors:
            if dt == DT_Q8I8 and tid != T_EMB:
                payload = bytes((b ^ 0x80) for b in payload)
            if payload not in Path(args.out).read_bytes():
                raise SystemExit("verify: tensor %d not found verbatim" % tid)
        print("verify: every tensor present verbatim")


if __name__ == "__main__":
    main()
