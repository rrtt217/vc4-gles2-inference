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
        tensor table: n * 48 B (v2, dense) or n * 64 B (v3, MoE)
        data: every tensor 16-byte aligned

This tool also verifies itself: `--verify` re-reads the file and compares every
tensor with the source bytes.  A conversion that changes one byte is a bug, not
a re-baseline.
"""

import argparse
import json
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmcf_grid import LOGITS_GRID
from mmcf_hf import (RESIDUAL_RAIL, RESIDUAL_WARN, is_llama2c, load_llama2c,
                     tensors_from_hf, tok_entry_count)

# tensor table record: id, dtype, flags, group, pad, rows, cols, offset, bytes,
# crc32, scale, aux -> exactly 64 bytes in v3, no implicit padding with "<".
# v2 records are 48 bytes with the same fields in the same places; v3 only adds
# `aux` (the expert index) inside v2's tail padding, so a v2 file still parses
# with aux = 0.  A container with expert tensors is version 3; a dense one stays
# version 2 and keeps its 48-byte table.
REC_SIZE_V2 = 48
REC_SIZE_V3 = 64


def rec_size(version):
    return REC_SIZE_V3 if version >= 3 else REC_SIZE_V2


def rec_unpack(rec):
    """(id, dtype, flags, group, rows, cols, off, bytes, crc, scale, aux)."""
    tid, dt, flags, group, _pad = struct.unpack("<HBBBB", rec[:6])
    rows, cols = struct.unpack("<II", rec[6:14])
    off, blen = struct.unpack("<QQ", rec[14:30])
    crc, = struct.unpack("<I", rec[30:34])
    scale, = struct.unpack("<f", rec[34:38])
    aux, = struct.unpack("<H", rec[38:40]) if len(rec) >= 40 else (0,)
    return tid, dt, flags, group, rows, cols, off, blen, crc, scale, aux


def pack_rec(tid, dt, flags, group, rows, cols, off, blen, crc, scale, aux=0, size=REC_SIZE_V3):
    b = struct.pack("<HBBBB", tid, dt, flags, group, 0)
    b += struct.pack("<II", rows, cols)
    b += struct.pack("<QQ", off, blen)
    b += struct.pack("<I", crc)
    b += struct.pack("<f", float(scale))
    b += struct.pack("<H", aux)
    return b.ljust(size, b"\0")


def unpack_rec(rec):
    tid, dt, _flags, _group, rows, cols, off, blen, crc, scale, _aux = rec_unpack(rec)
    return tid, dt, rows, cols, off, blen, crc, scale
import sys
from pathlib import Path

MAGIC = b"MMCF"
VERSION = 3          # newest this writer emits; a dense container is still v2
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
T_ROUTER = 0x16   # MoE gate, rows = experts, cols = dim (v3 containers only)
T_LOGITS = 0x20
T_TOKENIZER = 0x30
T_EMB_I16 = 8

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
            meta["rope_inferred"] = True   # the v1 header carries no rope
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
        for i in range((vocab + LOGITS_GRID - 1) // LOGITS_GRID):
            mat(T_LOGITS, i)
    elif magic in (MAGIC_I16, MAGIC_I16_V2):
        theta = r.f32(); half = r.i32(); split = r.i32()
        qk = 0
        if magic == MAGIC_I16_V2:
            qk = r.i32()
        meta["theta"] = theta
        meta["half"] = half != 0
        meta["qk_norm"] = qk != 0
        meta["scheme"], meta["group"], meta["split"] = "i16", 0, bool(split)
        # the i16 container starts with an int16 embedding, not q8
        tensors.append((T_EMB_I16, DT_I16, vocab, dim, 0, r.take(vocab * dim * 2), 0.0))
        tensors.append((T_EMB_SCALE, DT_F32, vocab, 1, 0, r.take(vocab * 4), 0.0))
        tensors.append((T_RMS_ATT, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FFN, DT_F32, layers, dim, 0, r.take(layers * dim * 4), 0.0))
        tensors.append((T_RMS_FINAL, DT_F32, 1, dim, 0, r.take(dim * 4), 0.0))
        if meta["qk_norm"]:
            tensors.append((T_Q_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))
            tensors.append((T_K_NORM, DT_F32, layers, head_dim, 0, r.take(layers * head_dim * 4), 0.0))
        for l in range(layers):
            mat(T_QKV, l, q8=False)
            mat(T_WO, l, q8=False)
            if split:
                mat(T_W1, l, q8=False)
                mat(T_W3, l, q8=False)
            else:
                mat(T_W13, l, q8=False)
            mat(T_W2, l, q8=False)
        for i in range((vocab + LOGITS_GRID - 1) // LOGITS_GRID):
            mat(T_LOGITS, i, q8=False)
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
        for i in range((vocab + LOGITS_GRID - 1) // LOGITS_GRID):
            rows, cols = r.i32(), r.i32()
            sc = r.f32()
            tensors.append((T_LOGITS | (i << 8), DT_Q8I8, rows, cols, 0, r.take(rows * cols), sc))
    else:
        raise SystemExit("unknown magic 0x%08x (supported: Q8M1/Q8M2/I16M1/I16N/Q4M1)" % magic)
    return meta, tensors


BOS_NAMES = ["<s>", "<bos>"]
EOS_NAMES = ["<|endoftext|>", "</s>", "<eos>"]
STOP_NAMES = ["<|im_end|>", "<|end|>", "<end_of_turn>"]
PAD_NAMES = ["<pad>", "<|endoftext|>"]
UNK_NAMES = ["<unk>", "<|unk|>"]


def vocab_of(tok_blob, n):
    """The blob is the engine's format: u32 maxlen, then (f32 score, i32 len, bytes)."""
    import struct as _s
    out, p = [], 4
    for _ in range(n):
        if p + 8 > len(tok_blob):
            break
        _score, ln = _s.unpack_from("<fi", tok_blob, p)
        p += 8
        out.append(tok_blob[p:p + ln].decode("utf-8", "replace"))
        p += ln
    return out


def specials_from(tok_blob, vocab, args):
    """-1 means absent.  A chatml template already opens with <|im_start|>, so it
    must not also get a BOS token in front of it."""
    # llama2.c stores "<s>" as "\n<s>\n", so match on the trimmed form
    stripped = [t.strip() for t in vocab]

    def find(names):
        for nm in names:
            if nm in stripped:
                return stripped.index(nm)
        return -1
    bos = -1 if args.template in ("qwen", "chatml", "minimind3", "tinybrainbot") else find(BOS_NAMES)
    eos = find(EOS_NAMES)
    stop = find(STOP_NAMES)
    pad = find(PAD_NAMES)
    unk = find(UNK_NAMES)
    im_start = stripped.index("<|im_start|>") if "<|im_start|>" in stripped else -1
    # A <|...|>-style end marker is this model's terminator; </s> is the
    # SentencePiece default it does not use to end a turn.
    if stop >= 0 and args.template == "tinybrainbot":
        eos = stop
    if args.bos is not None: bos = args.bos
    if args.eos is not None: eos = args.eos
    if args.stop is not None: stop = args.stop
    return bos, eos, stop, pad, unk, im_start


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
    # kind: 0 plain (llama2c), 1 chatml/qwen, 2 minimind3, 3 tinybrainbot.  The
    # name is kept for humans and as a fallback for a kind an old engine does
    # not know.
    kind = {"plain": 0, "qwen": 1, "chatml": 1, "minimind3": 2, "tinybrainbot": 3}.get(args.template, 1)
    template = (bytes([kind]) + args.system.encode() + b"\0" + args.template.encode() + b"\0")
    # A plain container is a completion model: advertising chat lets the engine
    # wrap a story prompt in the built-in [INST] template and answer with noise.
    # That combination shipped once (stories42M.q8) and cost a debugging round.
    if args.chat and args.template == "plain":
        raise SystemExit(
            "--chat with --template plain: this is a completion model, it has no chat\n"
            "template to render.  Drop --chat (use --prompt), or give the model's own\n"
            "template with --chat-template FILE and a template kind that has one.")
    caps_val = (1 if args.chat else 0) | (2 if args.tools else 0) | (4 if args.vision else 0)
    caps = struct.pack("<I", caps_val)
    md = (tlv(1, 1, model) + tlv(2, 1, rope) + tlv(3, 1, quant) +
          tlv(4, 1, template) + tlv(5, 1, caps) + tlv(6, 1, args.name.encode() + b"\0"))
    # key 9: the model's own chat template(s), HuggingFace Jinja text, stored
    # verbatim.  The default variant is named "" on disk ("default" in memory);
    # extra name=file arguments become variants for other prompt shapes.
    variants = list(getattr(args, "_chat_templates", []))
    if variants:
        body = bytes([len(variants)])
        for vname, vtext in variants:
            vb = vtext.encode("utf-8") if isinstance(vtext, str) else vtext
            body += vname.encode() + b"\0" + struct.pack("<I", len(vb)) + vb
        md += tlv(9, 1, body)
    if args.res == "q24":
        md += tlv(12, 1, struct.pack("<B", 1))
    if args.rms_eps is not None:
        md += tlv(11, 1, struct.pack("<f", args.rms_eps))
    if args.add_bos is not None or args.add_eos is not None:
        md += tlv(10, 1, struct.pack("<BB", 255 if args.add_bos is None else args.add_bos,
                                     255 if args.add_eos is None else args.add_eos))
    # key 13: the mixture-of-experts block.  Absent means dense (every existing
    # container).  flags bit 0 says the top-k gate is normalised, which is what
    # lets top-1 use the selected expert with weight 1.0.
    if meta.get("experts"):
        md += tlv(13, 1, struct.pack("<4i", int(meta["experts"]), int(meta["top_k"]),
                                     int(meta["moe_hidden"]),
                                     1 if meta.get("norm_topk") else 0))
    specials = (-1, -1, -1, -1, -1, -1)
    if tok_blob is not None:
        vocab = vocab_of(tok_blob, len(tok_blob) and 200000 or 0)
        specials = specials_from(tok_blob, vocab, args)
        md += tlv(8, 1, struct.pack("<6i", *specials))
    if tok_blob is not None:
        tensors = tensors + [(T_TOKENIZER, DT_BLOB, len(tok_blob), 1, 0, tok_blob, 0.0)]
        md += tlv(7, 1, struct.pack("<i", len(tok_blob)))
    # NOTE: every TLV must be appended before the offsets below are computed.

    meta_off = HEADER_SIZE
    table_off = meta_off + len(md)
    table_off = (table_off + ALIGN - 1) // ALIGN * ALIGN
    version = 3 if meta.get("experts") else 2
    rsize = rec_size(version)
    table = bytearray()
    data = bytearray()
    data_off = table_off + len(tensors) * rsize
    for rec in tensors:
        # dense producers append 7-tuples; an expert tensor carries an 8th
        # element, the expert index, which lands in the v3 aux field.
        tid, dt, rows, cols, group, payload, scale = rec[:7]
        aux = rec[7] if len(rec) > 7 else 0
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
        table += pack_rec(tid, dt, 0, group, rows, cols, off, len(payload), crc, scale,
                          aux=aux, size=rsize)
    data_off_final = table_off + len(tensors) * rsize
    file_size = data_off_final + len(data)
    body = MAGIC + struct.pack("<HHIIIIQQ", version, HEADER_SIZE, len(tensors),
                               caps_val, meta_off, len(md), data_off_final, file_size)
    assert len(body) == 40
    head = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
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
    if ver not in (2, 3) or hsz != HEADER_SIZE:
        raise SystemExit("verify: header mismatch")
    hcrc, = struct.unpack("<I", raw[40:44])
    if zlib.crc32(raw[:40]) & 0xFFFFFFFF != hcrc:
        raise SystemExit("verify: header crc mismatch")
    if fsz != len(raw):
        raise SystemExit("verify: file size field %d != %d" % (fsz, len(raw)))
    rsize = rec_size(ver)
    table = doff - n * rsize
    bad = 0
    for i in range(n):
        rec = raw[table + i * rsize: table + (i + 1) * rsize]
        tid, dt, rows, cols, off, blen, crc, _sc = unpack_rec(rec)
        body = raw[doff + off: doff + off + blen]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            bad += 1
    return n, caps, mo, ms, doff, len(raw), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="a legacy checkpoint, or a HuggingFace directory (config.json + "
                         "model.safetensors) to quantize directly; see --quant")
    ap.add_argument("--tok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--template", default="chatml")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--theta", type=float, help="rope theta when the header has none (Q8M1/I16M1)")
    ap.add_argument("--rope-half", action="store_true", help="rope rotates half the head (GPT-NeoX style)")
    ap.add_argument("--rope-plain", action="store_true", help="llama2 rope: theta 10000, interleaved")
    ap.add_argument("--assume-llama2-rope", action="store_true", help="accept theta=10000 half=false when the source is silent")
    ap.add_argument("--chat-template", metavar="FILE",
                    help="the model's chat template (Jinja text, e.g. tokenizer_config.json's "
                         "chat_template).  Default: read it from tokenizer_config.json next to --in")
    ap.add_argument("--chat-template-variant", metavar="NAME=FILE", action="append", default=[],
                    help="additional named template variant (repeatable)")
    ap.add_argument("--res", choices=["q16", "q24"], default="q16",
                    help="residual stream width: q16 (default, ±128) or q24 "
                         "(same 1/256 step, ±32768).  q24 is what a model that "
                         "trips the residual-rail check needs")
    ap.add_argument("--allow-over-rail", action="store_true",
                    help="pack even when the residual stream exceeds the engine's q16 rail")
    ap.add_argument("--no-rail-check", action="store_true",
                    help="skip the residual calibration (it costs ~2 s per model)")
    ap.add_argument("--rms-eps", type=float, default=None,
                    help="RMSNorm epsilon to record (key 11).  The engine defaults to 1e-5; "
                         "a HF config usually says 1e-6 and the packer warns when they differ")
    ap.add_argument("--add-bos", type=int, choices=[0, 1], default=None,
                    help="whether the engine should put BOS in front of the prompt")
    ap.add_argument("--add-eos", type=int, choices=[0, 1], default=None)
    ap.add_argument("--qk-norm", action="store_true", help="the file has per-head q/k norms")
    ap.add_argument("--bos", type=int, help="BOS id (-1 for none); default: derived from the tokenizer")
    ap.add_argument("--eos", type=int, help="EOS id; default: derived")
    ap.add_argument("--stop", type=int, help="extra stop id (e.g. <|im_end|>); default: derived")
    ap.add_argument("--q8-row-scale", action="store_true",
                    help="q8 with one scale per output row (a plane per matrix) "
                         "instead of one per matrix; for a model whose weight "
                         "rows span more than ~100:1")
    ap.add_argument("--quant", choices=["i16", "q8", "q4"], default="i16",
                    help="precision to build when --in is a HuggingFace directory (default i16)")
    ap.add_argument("--seq", type=int, default=512,
                    help="KV length to record for a HuggingFace source (default 512)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.src):
        # Straight from HuggingFace: no intermediate checkpoint, and the rope
        # comes from config.json rather than a flag.
        print("source: %s (quant %s, seq %d)" % (args.src, args.quant, args.seq))
        import json as _json
        _cfg = _json.load(open(os.path.join(args.src, "config.json")))
        _want = float(_cfg.get("rms_norm_eps") or 0.0)
        if _want and args.rms_eps is None and abs(_want - 1e-5) > 1e-12:
            print("note: config says rms_norm_eps %g, the engine default is 1e-05; "
                  "pass --rms-eps %g to use the model's value" % (_want, _want))
        _ntok = tok_entry_count(args.tok) if args.tok else 0
        if args.q8_row_scale and args.quant != "q8":
            raise SystemExit("--q8-row-scale only applies to --quant q8")
        meta, tensors = tensors_from_hf(args.src, args.quant, args.seq,
                                        rail_check=not args.no_rail_check,
                                        vocab_limit=_ntok or None,
                                        q8_row=args.q8_row_scale)
        if _cfg.get("tie_word_embeddings") is False:
            # load_minimind refuses a false tie with no lm_head, so reaching here
            # means the head really is the checkpoint's own matrix.
            print("head: tie_word_embeddings is false, using lm_head.weight")
        _vs = meta['vocab']
        if _ntok and _ntok != _vs:
            print("note: tokenizer has %d entries, container uses %d (embedding padded)"
                  % (_ntok, _vs))
    elif is_llama2c(args.src):
        # llama2.c fp32, also straight to a container; its rope is the llama2
        # layout by construction, so nothing has to be declared.
        print("source: %s (llama2c, quant %s)" % (args.src, args.quant))
        _m = load_llama2c(args.src)
        meta, tensors = tensors_from_hf(args.src, args.quant, _m['seq'], m=_m,
                                        rail_check=not args.no_rail_check)
    else:
        meta, tensors = parse_legacy(args.src)
        if not args.no_rail_check:
            print("note: no residual calibration for a legacy source (no float weights to "
                  "run); a model that rails at 128 will pack and then answer from a "
                  "truncated residual")
    # The residual stream is q16 at 1/256, so it clamps at 127.996.  A model
    # whose residual is larger still runs, but from a truncated state: supra2
    # reaches 2300 and echoes the prompt, feng30m reaches 212 and answers.  This
    # is the only cheap place to notice, so the packer says so and refuses well
    # past the rail unless told otherwise.
    peaks = meta.get("res_peaks")
    if peaks:
        rail = RESIDUAL_RAIL
        worst = max(peaks)
        print("residual: peak %.1f at layer %d (rail %.1f), per layer %s"
              % (worst, peaks.index(worst), rail, " ".join("%.0f" % v for v in peaks)))
        if worst > 4 * rail and not args.allow_over_rail:
            raise SystemExit(
                "this model's residual reaches %.0f, %.0fx the engine's q16 rail (%.1f).\n"
                "The forward pass will saturate and answer from a truncated state; see\n"
                "models/README.md (supra2-100M).  Pack it anyway with --allow-over-rail."
                % (worst, worst / rail, rail))
        if worst > RESIDUAL_WARN:
            print("warning: residual peak %.1f is at or over the %.1f rail; expect degraded "
                  "answers" % (worst, rail))
    # Sources without rope fields cannot state their rope, and a wrong theta or
    # half flag silently degrades every position: refuse to guess unless the
    # caller says so.
    if meta.get("rope_inferred"):
        if args.theta is None and not args.rope_plain and not args.assume_llama2_rope:
            raise SystemExit("this source has no rope fields (Q8M1/I16M1): pass --theta and "
                             "--rope-half, or --rope-plain for a llama2 model, or "
                             "--assume-llama2-rope to accept theta=10000 half=false")
    if args.theta is not None:
        meta["theta"] = args.theta
    if args.rope_half:
        meta["half"] = True
    if args.rope_plain:
        meta["theta"], meta["half"] = 10000.0, False
    if args.qk_norm:
        meta["qk_norm"] = True
    print("rope: theta=%g half=%s qk_norm=%s" % (meta.get("theta", 10000.0),
          bool(meta.get("half")), bool(meta.get("qk_norm"))))
    # Chat templates: explicit flag, else the HF source next to the checkpoint.
    templates = []
    tpath = args.chat_template
    if not tpath:
        cand = Path(args.src).parent / "tokenizer_config.json"
        if cand.exists():
            try:
                cfg = json.loads(cand.read_text(encoding="utf-8"))
                if cfg.get("chat_template"):
                    tpath = str(cand)
            except Exception:
                tpath = None
    if tpath:
        if tpath.endswith(".json"):
            cfg = json.loads(Path(tpath).read_text(encoding="utf-8"))
            text = cfg.get("chat_template")
            if isinstance(text, list):      # named variants in one field
                for entry in text:
                    if entry.get("name") == "default":
                        text = entry.get("template")
                        break
                else:
                    text = text[0].get("template") if text else None
            if not text:
                raise SystemExit(f"{tpath}: no chat_template field")
        else:
            text = Path(tpath).read_text(encoding="utf-8")
        templates.append(("", text))
        print("chat template: %d B from %s" % (len(text), tpath))
    for spec in args.chat_template_variant:
        if "=" not in spec:
            raise SystemExit("--chat-template-variant wants NAME=FILE")
        vname, vfile = spec.split("=", 1)
        templates.append((vname, Path(vfile).read_text(encoding="utf-8")))
    args._chat_templates = templates

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
        for rec in tensors:
            tid, dt, payload = rec[0], rec[1], rec[5]
            if dt == DT_Q8I8 and (tid & 0xFF) != T_EMB:
                payload = bytes((b ^ 0x80) for b in payload)
            if payload not in Path(args.out).read_bytes():
                raise SystemExit("verify: tensor 0x%x not found verbatim" % tid)
        print("verify: every tensor present verbatim")


if __name__ == "__main__":
    main()
