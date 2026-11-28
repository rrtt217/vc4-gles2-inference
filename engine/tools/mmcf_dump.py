#!/usr/bin/env python3
"""Print an MMCF v2 container in human-readable form.

    mmcf_dump.py model.mmcf            # summary
    mmcf_dump.py model.mmcf --tensors  # one line per tensor
    mmcf_dump.py model.mmcf --check    # also verify every tensor CRC
    mmcf_dump.py model.mmcf --template # print the chat template text, raw

`--template` is the answer to "is the container's template the one the vendor
shipped": compare it byte for byte with the GGUF's tokenizer.chat_template.

The point is to make "the container is self-describing" a checkable fact rather
than a claim: everything the engine reads is printed here from the file alone.
"""

import argparse
import struct
import sys
import zlib

# The tables live in mmcf_format.py: one copy for all three tools.  `KINDS`,
# `DTYPES`, `SCHEMES` and `CAPS` are imported under these names because the rest
# of this file (and its output) uses them; `TEMPLATE_KINDS` here is kind -> name,
# the reverse of the packer's name -> kind.
from mmcf_format import (KINDS, DTYPES, SCHEMES, CAPS,  # noqa: E402
                         PER_LAYER_KINDS, RWKV_ARCH_NAMES, RWKV_MATS,
                         RWKV_VECTORS, parse_rwkv_meta, KEY_ROPE_FACTORS,
                         unpack_rope_factors,
                         TEMPLATE_KIND_NAMES as TEMPLATE_KINDS)


def rd(raw, off, fmt):
    return struct.unpack_from(fmt, raw, off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true")
    ap.add_argument("--check", action="store_true")
    # The template text is what a model actually sees; printing only its length
    # is how a container ends up "using the right template" for months without
    # anyone comparing the bytes.
    ap.add_argument("--template", metavar="NAME", nargs="?", const="default",
                    help="print the chat template text (default: the 'default' variant)")
    args = ap.parse_args()

    # With --template the container's Jinja text owns stdout, so the summary
    # goes to stderr: a diff of two templates must not have to strip a header.
    out = sys.stdout
    if args.template is not None:
        sys.stdout = sys.stderr

    raw = open(args.path, "rb").read()
    if raw[:4] != b"MMCF":
        raise SystemExit("not an MMCF container")
    ver, hsz, n, caps, mo, ms, doff, fsize = rd(raw, 4, "<HHIIIIQQ")
    hcrc, = rd(raw, 40, "<I")
    print(f"file        {args.path}  {len(raw)} B")
    print(f"version     {ver}   header {hsz} B   tensors {n}")
    print(f"sections    meta {mo}+{ms}   data {doff}   size field {fsize} "
          f"({'ok' if fsize == len(raw) else 'MISMATCH'})")
    if hcrc:
        ok = (zlib.crc32(raw[:40]) & 0xFFFFFFFF) == hcrc
        print(f"header crc  0x{hcrc:08x} {'ok' if ok else 'MISMATCH'}")
    else:
        print("header crc  absent")
    names = [n for b, n in CAPS if caps & b]
    print(f"caps        0x{caps:x} ({', '.join(names) if names else 'none'})")

    # metadata
    p = mo
    tok_bytes = 0
    has_moe = False
    have_eps = False
    have_rms_store = False
    has_rwkv = False
    tmpls = {}
    while p + 8 <= mo + ms:
        key, typ, ln = rd(raw, p, "<HHI")
        payload = raw[p + 8:p + 8 + ln]
        if key == 1 and ln >= 28:
            dim, hidden, layers, heads, kv, vocab, seq = struct.unpack_from("<7i", payload)
            print(f"model       dim {dim} hidden {hidden} layers {layers} heads {heads} "
                  f"kv_heads {kv} vocab {vocab} seq {seq}")
        elif key == 2 and ln >= 8:
            theta, half, qk, _ = struct.unpack_from("<fBBH", payload)
            print(f"rope        theta {theta:g} half {bool(half)} qk_norm {bool(qk)}")
        elif key == KEY_ROPE_FACTORS and ln >= 4 and (ln % 4) == 0:
            # A longrope model's static per-dimension table.
            rf = unpack_rope_factors(bytes(payload))
            print(f"rope table  {len(rf)} factors, pair 0..{len(rf)-1} = "
                  f"{rf[0]:.4f} .. {rf[-1]:.4f} (freq_i /= factor_i)")
        elif key == 3 and ln >= 4:
            scheme, group, preb, split = struct.unpack_from("<BBBB", payload)
            print(f"quant       {SCHEMES.get(scheme, scheme)} group {group} "
                  f"pre_biased {bool(preb)} split_w13 {bool(split)}")
        elif key == 4 and ln >= 3:
            kind = payload[0]
            parts = payload[1:].split(b"\0")
            system = parts[0].decode("utf-8", "replace") if parts else ""
            tname = parts[1].decode("utf-8", "replace") if len(parts) > 1 else ""
            print(f"template    kind {kind} ({TEMPLATE_KINDS.get(kind, 'unknown')}) name '{tname}'")
            print(f"system      {system!r}")
        elif key == 5 and ln >= 4:
            pass  # mirror of the header caps, already printed
        elif key == 6:
            print(f"name        {payload.rstrip(bytes([0])).decode('utf-8', 'replace')}")
        elif key == 7 and ln >= 4:
            tok_bytes, = struct.unpack_from("<i", payload)
            print(f"tokenizer   {tok_bytes} B embedded")
        elif key == 8 and ln >= 24:
            b, e, s, pd, u, im = struct.unpack_from("<6i", payload)
            f = lambda v: "-" if v < 0 else str(v)
            print(f"specials    bos {f(b)} eos {f(e)} stop {f(s)} "
                  f"pad {f(pd)} unk {f(u)} im_start {f(im)}")
        elif key == 9 and ln >= 1:
            # the model's own chat template(s), Jinja text
            q, nvar = 1, payload[0]
            for _ in range(nvar):
                nul = payload.index(0, q)
                vname = payload[q:nul].decode("utf-8", "replace") or "default"
                q = nul + 1
                vlen, = struct.unpack_from("<I", payload, q)
                q += 4
                body = payload[q:q + vlen]
                q += vlen
                tmpls[vname] = body
                first = body.split(b"\n", 1)[0].decode("utf-8", "replace")
                print(f"chat tmpl   '{vname}' {vlen} B  {first[:64]}")
        elif key == 20 and ln >= 6:
            # v5: the encoder block and the four block properties a
            # decoder-only container never had to state.
            n_enc, flags, enc_seq, enc_ffn = struct.unpack_from("<BBHH", payload)
            props = [n for n, bit in (("post-norm", 1), ("scale-embed", 2),
                                      ("nogate-ffn", 4), ("no-rope", 8)) if flags & bit]
            print(f"encoder     {n_enc} layers, seq {enc_seq}, ffn {enc_ffn}, "
                  + (" ".join(props) if props else "no flags"))
        elif key == 21 and ln >= 4:
            nd, nf = struct.unpack_from("<HH", payload)
            print(f"bias rows   {nd} of dim, {nf} of ffn (mmcf_format.BIAS_DIM_ORDER)")
        elif key == 12 and ln >= 1:
            print(f"residual    {payload[0] and 'q24 (1 value/texel, +-32768)' or 'q16 (2 per texel, +-128)'}")
        elif key == 11 and ln >= 4:
            import struct as _s
            have_eps = True
            print(f"rms eps     {_s.unpack('<f', bytes(payload[:4]))[0]:g}")
        elif key == 23 and ln >= 2:
            # A BiCrossGate FFN: the two activations, act1 on q1*x and act2 on
            # q2*x.  Printed because the block's tensor kinds (0x3D/0x3E) say
            # *which matrices* are there but not *what the FFN computes*, and a
            # reader that missed this key would run a SwiGLU over w1/w2.
            names = {1: "silu", 2: "gelu"}
            print("ffn block   bicross gate: act1 %s, act2 %s"
                  % (names.get(payload[0], "code %d" % payload[0]),
                     names.get(payload[1], "code %d" % payload[1])))
        elif key == 19 and ln >= 1:
            # The rms partial sums' store: absent = 24-bit fixed point, 1 = f32.
            # A model that saturates the 24-bit store needs the f32 bit, and the
            # reader has no other way to tell which container it is looking at.
            have_rms_store = True
            print(f"rms store   {'f32' if payload[0] == 1 else 'i24'}")
        elif key == 13 and ln >= 16:
            ne, tk, mh, mf = struct.unpack_from("<4i", payload)
            has_moe = True
            print(f"moe         experts {ne} top_k {tk} expert_hidden {mh} "
                  f"norm_topk_prob {bool(mf & 1)}")
        elif key == 15 and ln >= 1:
            print(f"ffn scale   hidden * 2^-k per layer: "
                  + " ".join(str(int(b)) for b in payload))
        elif key == 14 and ln >= 1:
            kinds = {0: "attn", 1: "conv"}
            print(f"layers      {ln} typed: "
                  + " ".join(kinds.get(b, "?%d" % b) for b in payload))
        elif key == 17 and ln >= 1:
            # v4: the head's input is stored *2^-k and the logits tiles read
            # back *2^k (their scales already carry it).
            print(f"logit scale  head input *2^-{payload[0]} "
                  f"(logits tile scales *2^{payload[0]})")
        elif key == 16 and ln >= 16:
            # v4: the RWKV-7 block.  The four low ranks are printed because
            # they cannot be re-derived from the shapes (llama.cpp's formula
            # gives 160 for 0.4B's gate where the config says 128).
            rk = parse_rwkv_meta(bytes(payload[:16]))
            has_rwkv = True
            print(f"rwkv        arch {RWKV_ARCH_NAMES.get(rk['arch'], rk['arch'])}"
                  f"  head {rk['head_size']}  eps {rk['norm_eps']:g}"
                  f"  ranks w {rk['r_w']} a {rk['r_a']} v {rk['r_v']} g {rk['r_g']}"
                  f"  gate {rk['gate']}")
        elif key == 10 and ln >= 2:
            ab, ae = payload[0], payload[1]
            fmt = lambda v: "unset" if v == 255 else str(v)
            extra = ""
            if ln >= 3:
                # Third byte (2026-11-18): the tokenizer's prefix policy.
                extra = f" seg_prefix {int(payload[2])}"
            if ln >= 4:
                # Fourth byte: 0 = BPE merge (the historical table), 1 =
                # SentencePiece unigram, which is what an encoder-decoder has.
                extra += " tok " + {0: "bpe", 1: "unigram"}.get(
                    int(payload[3]), "?%d" % int(payload[3]))
            print(f"bos/eos     add_bos {fmt(ab)} add_eos {fmt(ae)}{extra}")
        p += 8 + ln

    if not have_eps:
        # A file without key 11 uses the engine default, and printing nothing
        # here read as "this model has no eps" rather than "1e-05".
        print("rms eps     absent (engine default 1e-05)")

    if args.template is not None:
        body = tmpls.get(args.template)
        if body is None:
            raise SystemExit("no template variant '%s' (have: %s)"
                             % (args.template, ", ".join(sorted(tmpls)) or "none"))
        out.buffer.write(body)
        return

    # tensors.  The table starts at the first 16-byte boundary after the
    # metadata, so the record size follows from the two offsets; v2 is 48 and
    # v3 (expert tensors) is 64.
    table = (mo + ms + 15) // 16 * 16
    rsize, rem = divmod(doff - table, n) if n else (0, 0)
    if n and (rem or rsize not in (48, 64)):
        raise SystemExit(f"not an MMCF tensor table: {doff - table} B for {n} tensors")
    rows = []
    for i in range(n):
        r = raw[table + i * rsize: table + (i + 1) * rsize]
        tid, dt, flags, group, _pad = struct.unpack_from("<HBBBB", r, 0)
        rws, cls = struct.unpack_from("<II", r, 6)
        off, blen = struct.unpack_from("<QQ", r, 14)
        crc, = struct.unpack_from("<I", r, 30)
        scale, = struct.unpack_from("<f", r, 34)
        aux, = struct.unpack_from("<H", r, 38) if rsize >= 64 else (0,)
        rows.append((tid, dt, rws, cls, off, blen, crc, scale, aux))
    # Aliased records share one blob, so summing per-record lengths counts the
    # same bytes twice -- and on a container whose FFN is shared by layer pairs
    # that made the figure exceed the file and print as negative padding.  The
    # reported size is the bytes the file actually holds; the second line says
    # how many records point into another one's blob.
    shared = {}
    for t in rows:
        shared.setdefault((t[4], t[5]), []).append(t[0])
    total = sum(blen for _off, blen in shared)
    n_alias = sum(len(v) - 1 for v in shared.values() if len(v) > 1)
    # part 3 of a tensor id is a per-row q8 scale plane: one f32 per output row,
    # the row's scale divided by the matrix unit the record carries.
    rplanes = [t for t in rows if (t[0] >> 14) == 3]
    if rplanes:
        print(f"q8 rows     {len(rplanes)} per-row scale plane(s), {sum(t[5] for t in rplanes)} B "
              f"({sum(t[2] for t in rplanes)} rows total)")
    if has_moe:
        # v3 expert matrices: layers x experts x 3 bodies.  Counting `aux != 0`
        # missed expert 0 (24 of 96) and counted the q4/q8 plane records, which
        # also carry aux: the old line said 72 for a 96-tensor block.
        nex = sum(1 for t in rows
                  if (t[0] & 0xFF) in (0x12, 0x13, 0x15) and (t[0] >> 14) == 0)
        print(f"experts     {nex} expert matrix tensor(s)")
    else:
        nex = sum(1 for t in rows if t[8])
        if nex:
            print(f"experts     {nex} tensor(s) carry an expert index")
    if has_rwkv:
        # The nine f32 vector groups and the four low-rank pairs are what a
        # v4 file has instead of rms_att/qkv-rms norms and an FFN gate pair.
        vecs = [t for t in rows if (t[0] & 0xFF) in RWKV_VECTORS]
        print("rwkv vecs   %d f32 record(s): %s"
              % (len(vecs), ", ".join("%s %dx%d" % (KINDS[t[0] & 0xFF], t[2], t[3])
                                       for t in vecs)))
        pairs = {}
        for t in rows:
            k = t[0] & 0xFF
            if k in RWKV_MATS:
                name, which = RWKV_MATS[k]
                pairs.setdefault(name, [0, 0])[which] += 1
        print("rwkv lora   " + (" ".join("%s %d/%d" % (k, v[0], v[1])
                                          for k, v in sorted(pairs.items()))
                                or "none"))
    print(f"tensor data {total} B in {n} tensors, {len(raw) - total - table} B of headers/padding"
          + (f"; {n_alias} record(s) share another's blob" if n_alias else ""))
    if args.tensors:
        for tid, dt, rws, cls, off, blen, crc, scale, aux in rows:
            kind = tid & 0xFF
            layer = (tid >> 8) & 0x3F
            part = tid >> 14
            label = KINDS.get(kind, f"0x{kind:02x}")
            # A per-layer record prints its layer.  The model-wide f32
            # vector groups (0x1A-0x1F, 0x21) and the two global norms keep
            # layer bits 0 because there is one record for the whole model.
            if kind in PER_LAYER_KINDS or kind == 0x20:
                label += f".{layer}"
            if aux or (has_moe and kind in (0x12, 0x13, 0x14, 0x15)):
                label += f".e{aux}"
            if part:
                label += f".part{part}"
            print(f"  {label:22s} {DTYPES.get(dt, dt):5s} {rws:5d} x {cls:5d} "
                  f"off {off:10d} len {blen:9d} scale {scale:.6g} crc 0x{crc:08x}")
    if args.check:
        bad = 0
        for tid, dt, rws, cls, off, blen, crc, scale, aux in rows:
            body = raw[doff + off: doff + off + blen]
            if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
                bad += 1
                print(f"  CRC MISMATCH tensor id 0x{tid:04x} aux {aux}")
        print(f"crc check   {'all ' + str(n) + ' tensors ok' if not bad else str(bad) + ' bad'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
