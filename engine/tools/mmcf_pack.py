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
        tensor table: n * 48 B (v2, dense), n * 64 B (v3 MoE and v4 RWKV-7)
        data: every tensor 16-byte aligned

This tool also verifies itself: `--verify` re-reads the file and compares every
tensor with the source bytes.  A conversion that changes one byte is a bug, not
a re-baseline.
"""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmcf_grid import LOGITS_GRID
from mmcf_hf import (ACTIVATION_RAIL, RESIDUAL_RAIL, RESIDUAL_WARN, RMS_LANE_RAIL,
                     is_llama2c,
                     load_llama2c, tensors_from_hf, tok_entry_count)

# tensor table record: id, dtype, flags, group, pad, rows, cols, offset, bytes,
# crc32, scale, aux -> exactly 64 bytes in v3, no implicit padding with "<".
# v2 records are 48 bytes with the same fields in the same places; v3 only adds
# `aux` (the expert index) inside v2's tail padding, so a v2 file still parses
# with aux = 0.  A container with expert tensors is version 3; a dense one stays
# version 2 and keeps its 48-byte table.
# The record layout lives in mmcf_format.py (one copy for the packer, the dump tool
# and the requantiser); `header_magic` etc. below are legacy checkpoint magics,
# which only this tool reads.
from mmcf_format import (MAGIC, REC_SIZE_V2, REC_SIZE_V3, HEADER_SIZE, TEMPLATE_KINDS, rec_size,  # noqa: E402
                         rec_unpack, pack_rec, tlv, crc32,
                         DT_F32, DT_I16, DT_Q8U8, DT_Q8I8, DT_Q4, DT_BLOB,
                         VERSION_DENSE, VERSION_MOE, VERSION_RWKV7,
                         KEY_RWKV, KEY_LOGIT_SHIFT, KEY_ROPE_FACTORS,
                         RWKV_FLAG_GATE, pack_rwkv_meta, pack_rope_factors)
from mmcf_hf import T_EMB, T_EMB_I16, T_LOGITS  # noqa: E402

# Tensor ids the extract left behind (2ca97cf); the values are the format's.
T_TOKENIZER = 0x30


# The tensor table's alignment.  It moved into mmcf_format.py's extract and was
# left undefined here, so the packer died on any real pack (2ca97cf); the value
# is the one the format was built with.
ALIGN = 16

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


# name -> kind byte in metadata key 6.  One table: the packer validates against
# it, and the engine's fallback path reads the *name* when the kind is unknown,
# so the two have to agree.
TEMPLATE_KINDS = {"plain": 0, "qwen": 1, "chatml": 1, "minimind3": 2, "tinybrainbot": 3}


def tlv(key, typ, payload):
    return struct.pack("<HHI", key, typ, len(payload)) + payload


def build(meta, tensors, tok_blob, args):
    model = struct.pack("<7i", meta["dim"], meta["hidden"], meta["layers"],
                        meta["heads"], meta["kv_heads"], meta["vocab"], meta["seq"])
    rope = struct.pack("<fBBH", float(meta.get("theta", 10000.0)),
                       1 if meta.get("half") else 0,
                       1 if meta.get("qk_norm") else 0, 0)
    scheme = {"i16": 0, "q8": 1, "q4": 2, "t3": 3}[meta["scheme"]]
    quant = struct.pack("<BBBB", scheme, meta.get("group", 0),
                        1,  # q8u8 pre-biased
                        1 if meta.get("split") else 0)
    # kind: 0 plain (llama2c), 1 chatml/qwen, 2 minimind3, 3 tinybrainbot.  The
    # name is kept for humans and as a fallback for a kind an old engine does
    # not know.
    # An unknown name used to fall through to kind 1 (chatml) without a word to
    # anyone, so `--template llama2_chat` produced a container that renders ChatML
    # to a model trained on [INST]: a silent wrong answer, and the engine cannot
    # tell because the kind byte is all it sees.
    kind = TEMPLATE_KINDS.get(args.template)
    if kind is None:
        raise SystemExit("unknown --template %r; known: %s"
                         % (args.template, ", ".join(sorted(TEMPLATE_KINDS))))
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
    # key 10: the BOS/EOS policy, and (third byte, added 2026-11-18) whether the
    # tokenizer prefixes *every* text run with its space marker.  A reader that
    # only looks at two bytes sees the old policy, which is why the byte is
    # appended rather than put in a key of its own.
    if (args.add_bos is not None or args.add_eos is not None
            or meta.get("sp_prefix")):
        body = struct.pack("<BB", 255 if args.add_bos is None else args.add_bos,
                           255 if args.add_eos is None else args.add_eos)
        if meta.get("sp_prefix"):
            body += struct.pack("<B", 1)
        md += tlv(10, 1, body)
    # key 14: one byte per layer, 0 = attention, 1 = the LFM2 conv block.
    # Absent means "every layer is attention", which is every other model.
    if meta.get('layer_types'):
        md += tlv(14, 1, bytes(1 if t == 'conv' else 0 for t in meta['layer_types']))
    # key 15: the FFN hidden store's per-layer scale exponent.  Layer k means
    # the SwiGLU hidden is written *2^-k (so the q16 activation rail fits) and
    # w2's read-back multiplies by 2^k.  Absent = every layer 0, which is every
    # container packed before 2026-11-12.
    if meta.get('hid_scale'):
        md += tlv(15, 1, bytes(int(k) for k in meta['hid_scale']))
    # key 13: the mixture-of-experts block.  Absent means dense (every existing
    # container).  flags bit 0 says the top-k gate is normalised, which is what
    # lets top-1 use the selected expert with weight 1.0.
    if meta.get("experts"):
        md += tlv(13, 1, struct.pack("<4i", int(meta["experts"]), int(meta["top_k"]),
                                     int(meta["moe_hidden"]),
                                     1 if meta.get("norm_topk") else 0))
    # key 16: the RWKV-7 block (version 4).  Arch, WKV head width, a flag word,
    # the LayerNorm epsilon and the four low ranks -- the ranks cannot be
    # re-derived from the shapes (llama.cpp's formula is wrong for 0.4B's
    # gate), and the dimensions the runtime indexes by cannot be guessed.
    if meta.get("rwkv"):
        r = meta["rwkv"]
        md += tlv(KEY_RWKV, 1, pack_rwkv_meta(
            arch=r["arch"], head_size=r["head_size"],
            flags=RWKV_FLAG_GATE if r.get("gate") else 0,
            norm_eps=r["norm_eps"], r_w=r["r_w"], r_a=r["r_a"],
            r_v=r["r_v"], r_g=r["r_g"]))
    # key 17: the head input is stored *2^-k, and the logits tiles' scales
    # already carry the *2^k (mmcf_hf.py folds it in).  Absent = k = 0.
    if meta.get("logit_shift"):
        md += tlv(KEY_LOGIT_SHIFT, 1, struct.pack("<B", int(meta["logit_shift"])))
    # key 18: the rope table a `longrope` model needs (BitCPM-CANN-0.5B, i.e.
    # MiniCPM4-0.5B, is the first).  Absent = plain theta, every container
    # packed before 2026-11-18, and the factor list is validated against the
    # head width by the engine as well as by the packer.
    if meta.get("rope_factors"):
        md += tlv(KEY_ROPE_FACTORS, 1, pack_rope_factors(meta["rope_factors"]))
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
    version = (VERSION_RWKV7 if meta.get("rwkv") else
               VERSION_MOE if meta.get("experts") else VERSION_DENSE)
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


def verify(path, tensors, tok=None):
    """Re-read the file and check its records against the source tensor table.

    The body CRC alone cannot see a wrong id, dtype, shape, scale or expert
    index: a container with two experts' `aux` swapped passes a CRC-only check
    and the engine then runs the wrong expert.  The records are written in the
    source-tensor order, so this compares them field by field (the audit's
    2026-11-12 mutant test is why the fields are checked and not just the CRC).
    """
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise SystemExit("verify: bad magic")
    ver, hsz, n, caps, mo, ms, doff, fsz, _crc = struct.unpack("<HHIIIIQQI", raw[4:44])
    if ver not in (VERSION_DENSE, VERSION_MOE, VERSION_RWKV7) or hsz != HEADER_SIZE:
        raise SystemExit("verify: header mismatch")
    hcrc, = struct.unpack("<I", raw[40:44])
    if zlib.crc32(raw[:40]) & 0xFFFFFFFF != hcrc:
        raise SystemExit("verify: header crc mismatch")
    if fsz != len(raw):
        raise SystemExit("verify: file size field %d != %d" % (fsz, len(raw)))
    rsize = rec_size(ver)
    table = doff - n * rsize
    want = []
    for rec in tensors:
        tid, dt, rows, cols, group, payload, scale = rec[:7]
        aux = rec[7] if len(rec) > 7 else 0
        if dt == DT_Q8I8 and (tid & 0xFF) != T_EMB:
            dt = DT_Q8U8
        want.append((tid, dt, int(rows), int(cols), int(group), float(scale), int(aux)))
    if tok is not None:
        want.append((T_TOKENIZER, DT_BLOB, len(tok), 1, 0, 0.0, 0))
    if len(want) != n:
        raise SystemExit("verify: %d records but %d source tensors" % (n, len(want)))
    bad = 0
    field_bad = 0
    for i in range(n):
        rec = raw[table + i * rsize: table + (i + 1) * rsize]
        tid, dt, _flags, group, rows, cols, off, blen, crc, scale, aux = rec_unpack(rec)
        body = raw[doff + off: doff + off + blen]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            bad += 1
        wtid, wdt, wrows, wcols, wgroup, wscale, waux = want[i]
        mism = []
        if tid != wtid:
            mism.append("id 0x%x != 0x%x" % (tid, wtid))
        if dt != wdt:
            mism.append("dtype %d != %d" % (dt, wdt))
        if rows != wrows or cols != wcols:
            mism.append("shape %dx%d != %dx%d" % (rows, cols, wrows, wcols))
        if group != wgroup:
            mism.append("group %d != %d" % (group, wgroup))
        if aux != waux:
            mism.append("expert %d != %d" % (aux, waux))
        if abs(scale - wscale) > 1e-6 * max(1.0, abs(wscale)):
            mism.append("scale %.7g != %.7g" % (scale, wscale))
        if mism:
            field_bad += 1
            if field_bad <= 8:
                print("verify: record %d (id 0x%x) does not match the source tensor: %s"
                      % (i, tid, ", ".join(mism)))
    if field_bad:
        raise SystemExit("verify: %d record(s) do not match the source tensor table"
                         % field_bad)
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
    ap.add_argument("--head-q6", action="store_true",
                    help="tied embedding stored as signed 6-bit (two planes) and "
                         "used as the logits matrix: --head-dedup plus 6-bit rows")
    ap.add_argument("--head-dedup", action="store_true",
                    help="tied embedding: write no logits tiles and let the engine "
                         "render them from the embedding rows with their per-row "
                         "scales (saves one full copy of the vocabulary head)")
    ap.add_argument("--q8-row-scale", action="store_true",
                    help="q8 with one scale per output row (a plane per matrix) "
                         "instead of one per matrix; for a model whose weight "
                         "rows span more than ~100:1")
    ap.add_argument("--quant", choices=["i16", "q8", "q4", "t3"], default="i16",
                    help="precision to build when --in is a HuggingFace directory (default i16)")
    ap.add_argument("--seq", type=int, default=512,
                    help="KV length to record for a HuggingFace source (default 512)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    # Fail on a name this build does not know before touching any weights: an
    # unknown template used to become ChatML silently.
    if args.template not in TEMPLATE_KINDS:
        raise SystemExit("unknown --template %r; known: %s"
                         % (args.template, ", ".join(sorted(TEMPLATE_KINDS))))

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
        if args.quant == "t3" and args.head_q6:
            # The head/embedding stays q8 either way; q6 would be the sixth
            # combination and it is not written.
            raise SystemExit("--head-q6 with --quant t3 is not supported; the "
                             "embedding is q8 in a t3 container")
        meta, tensors = tensors_from_hf(args.src, args.quant, args.seq,
                                        rail_check=not args.no_rail_check,
                                        vocab_limit=_ntok or None,
                                        q8_row=args.q8_row_scale,
                                        head_dedup=args.head_dedup,
                                        head_q6=args.head_q6)
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
        # The rail the container will actually be *read* at.  --res q24 stores
        # one value a texel at the same 1/256 step but +-32768 (key 12), so a
        # model that only trips the q16 rail is representable; using the q16
        # rail here refused SmolLM2 (peak 26884) with --res q24 asked for,
        # which is the one case the flag exists for (2026-11-18).
        rail = RESIDUAL_RAIL if args.res == "q16" else 32768.0
        warn = RESIDUAL_WARN if args.res == "q16" else 0.75 * rail
        worst = max(peaks)
        print("residual: peak %.1f at layer %d (rail %.1f, res %s), per layer %s"
              % (worst, peaks.index(worst), rail, args.res,
                 " ".join("%.0f" % v for v in peaks)))
        if worst > 4 * rail and not args.allow_over_rail:
            raise SystemExit(
                "this model's residual reaches %.0f, %.0fx the engine's %s rail (%.1f).\n"
                "The forward pass will saturate and answer from a truncated state; see\n"
                "models/README.md (supra2-100M).  %s"
                % (worst, worst / rail, args.res, rail,
                   "Pack it anyway with --allow-over-rail." if args.res == "q24" else
                   "Pack it with --res q24 (same 1/256 step, +-32768) if it fits, "
                   "or with --allow-over-rail."))
        if worst > warn:
            print("warning: residual peak %.1f is %.0f%% of the %.1f rail; expect degraded "
                  "answers" % (worst, 100.0 * worst / rail, rail))
    # The residual's *format* is not the only thing a large residual breaks: the
    # rmsnorm accumulates sum(x^2) per lane in 24-bit fixed point, and a lane
    # sum grows with the square of the residual.  A model that q24 represents
    # perfectly can still pin that store (SmolLM2: lane bound 1.3e10 against
    # 2048), and then inv_rms is wrong for every layer after the peak -- the
    # container answers with its end token, /not/ because the residual lost
    # range (measured 2026-11-18: PPL 4.4e55 with the 24-bit store against
    # 7.7157 with RMS_STORE=f32, same container, one switch).
    if peaks:
        lanebound = worst * worst * (int(meta.get('dim') or 0) / 32.0)
        if lanebound > RMS_LANE_RAIL:
            print("rms store: a lane sum can reach %.1e against the 24-bit rail %.0f "
                  "(%.0e x over); run the engine with RMS_STORE=f32, or inv_rms is "
                  "wrong for every layer after the peak"
                  % (lanebound, RMS_LANE_RAIL, lanebound / RMS_LANE_RAIL))
    # The FFN SwiGLU hidden goes through the same q16 activation format and is
    # clamped at the same rail.  It is not the residual, so the check above
    # cannot see it: MiniMind3-MoE's expert hidden reaches 250 at the chat
    # template's <|im_start|> positions while the dense 64M model peaks at 88.8.
    hpeaks = meta.get("hid_peaks")
    if hpeaks:
        hrail = ACTIVATION_RAIL
        worst = max(hpeaks)
        hscale = meta.get("hid_scale") or [0] * len(hpeaks)
        scaled = max(peak / float(1 << int(hscale[l])) for l, peak in enumerate(hpeaks))
        print("ffn hidden: peak %.1f at layer %d (activation rail %.1f), per layer %s"
              % (worst, hpeaks.index(worst), hrail, " ".join("%.0f" % v for v in hpeaks)))
        if meta.get("hid_scale"):
            print("ffn hidden: store scaled, layer k = %s (hidden * 2^-k, w2 read *2^k, "
                  "metadata key 15); worst store peak %.1f"
                  % (" ".join(str(int(k)) for k in hscale), scaled))
        if scaled > 4 * hrail and not args.allow_over_rail:
            raise SystemExit(
                "this model's FFN hidden still reaches %.0f after the per-layer scale, "
                "%.0fx the engine's q16 activation rail (%.1f); the SwiGLU output is "
                "clamped at every position that peaks, which is a truncated FFN, not a "
                "precision loss.  Pack it anyway with --allow-over-rail."
                % (scaled, scaled / hrail, hrail))
        if scaled > hrail:
            loss = 100.0 * (1.0 - hrail / scaled)
            print("warning: the FFN SwiGLU hidden is still clamped by the q16 activation "
                  "rail at the positions that reach %.0f after scaling (%.0f%% of that "
                  "hidden is cut); the error is silent.  See docs/checkpoint-container.md."
                  % (scaled, loss))
    # RWKV-7's channel mix is relu(x)^2 and its *square* is what travels through
    # the q16 activation rail -- the block's only amplifier, and on 0.1B the
    # stream closest to the rail.  The block above already reports the peak and
    # decides key 15; this line states the margin the plan asks for.
    if meta.get("rwkv") and hpeaks:
        worst = max(hpeaks)
        print("sqrelu: peak %.2f at layer %d, %.2fx under the %.1f activation rail"
              % (worst, hpeaks.index(worst), ACTIVATION_RAIL / worst, ACTIVATION_RAIL))
    # The head's input is the final LayerNorm's output, 2.5x over the same rail
    # on 0.1B.  It cannot be scaled per layer (there is one of it), so key 17
    # stores it *2^-k and the logits tiles read back *2^k.
    fpeak = meta.get("final_ln_peak")
    if fpeak:
        k = int(meta.get("logit_shift") or 0)
        print("final ln: peak %.2f (%.2fx the %.1f activation rail); stored *2^-%d, "
              "logits tile scales *2^%d (metadata key %s)"
              % (fpeak, fpeak / ACTIVATION_RAIL, ACTIVATION_RAIL, k, k,
                 "17" if k else "17 absent"))
        bound = meta.get("final_ln_bound")
        if bound:
            print("final ln: analytic worst case %.1f (sqrt(dim)*max|w| + max|b|) "
                  "would need k=%d; a prompt that reached it would clip at %.1f"
                  % (bound, max(0, math.ceil(math.log2(bound / ACTIVATION_RAIL))),
                     (1 << k) * ACTIVATION_RAIL))
    # The WKV state is not a rail: it is a per-head 64x64 recurrence that is
    # never renormalised, so its magnitude is what decides whether the Pi can
    # hold it in fixed point at all (docs/engine-internals.md).  Nothing in
    # the container changes because of it; it is measured, not enforced.
    speaks = meta.get("state_peaks")
    if speaks:
        worst = max(speaks)
        print("wkv state: |max| %.1f at layer %d, per layer %s"
              % (worst, speaks.index(worst), " ".join("%.1f" % v for v in speaks)))
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
    rf = meta.get("rope_factors")
    print("rope: theta=%g half=%s qk_norm=%s%s"
          % (meta.get("theta", 10000.0), bool(meta.get("half")), bool(meta.get("qk_norm")),
             "" if not rf else " table=%d factors (%.4f..%.4f, freq_i/=factor_i)"
             % (len(rf), rf[0], rf[-1])))
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
        n, caps, mo, ms, doff, fsz, bad = verify(args.out, tensors, tok)
        print("verify: %d tensors, caps=0x%x meta=%d+%d data@%d size=%d crc_bad=%d" %
              (n, caps, mo, ms, doff, fsz, bad))
        if bad:
            raise SystemExit("verify failed")
        # weights must match the source bit for bit (after the bias).  The file
        # is read once: this loop used to re-read it per tensor, which on a 0.4B
        # container (795 tensors) is 358 GB of memcpy for no extra check.
        blob = Path(args.out).read_bytes()
        for rec in tensors:
            tid, dt, payload = rec[0], rec[1], rec[5]
            if dt == DT_Q8I8 and (tid & 0xFF) != T_EMB:
                payload = bytes((b ^ 0x80) for b in payload)
            if payload not in blob:
                raise SystemExit("verify: tensor 0x%x not found verbatim" % tid)
        print("verify: every tensor present verbatim")


if __name__ == "__main__":
    main()
