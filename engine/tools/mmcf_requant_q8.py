#!/usr/bin/env python3
"""Rebuild the q8 matrices of an MMCF container from its i16 sibling.

    mmcf_requant_q8.py --in model.i16.mmcf --out model.q8.mmcf [--clip 99.9]

The i16 container is the exact source; every i16 matrix becomes q8u8 (the
pre-biased texture dtype) with one f32 scale per tensor.  `--clip P` sets that
scale from the P-th percentile of |w| instead of max|w|, so a handful of
outlier weights no longer crush the other 99.9% into two or three levels.
`--emb keep` leaves the embedding in i16, `--emb q8` quantizes it per row with
the container's own embedding_scale and writes the signed id-1 tensor.
"""
import argparse
import struct
import zlib

import numpy as np

ALIGN = 16
REC_SIZE = 48
MAT_BASES = (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x20)
EMB_BASES = (1, 8)


def parse(path):
    with open(path, 'rb') as f:
        head = f.read(64)
        if head[:4] != b'MMCF':
            raise SystemExit('%s: not an MMCF container' % path)
        n, caps = struct.unpack_from('<II', head, 8)
        meta_off, meta_size = struct.unpack_from('<II', head, 16)
        data_off, file_size = struct.unpack_from('<QQ', head, 24)
        f.seek(meta_off)
        meta = bytearray(f.read(meta_size))
        f.seek(data_off - n * REC_SIZE)
        recs = []
        for _ in range(n):
            b = f.read(REC_SIZE)
            tid, dt, flags, group, _pad = struct.unpack_from('<HBBBB', b, 0)
            rows, cols = struct.unpack_from('<II', b, 6)
            off, blen = struct.unpack_from('<QQ', b, 14)
            crc, = struct.unpack_from('<I', b, 30)
            scale, = struct.unpack_from('<f', b, 34)
            recs.append(dict(tid=tid, dt=dt, flags=flags, group=group,
                             rows=rows, cols=cols, off=off, blen=blen,
                             crc=crc, scale=scale))
    return dict(n=n, caps=caps, meta=meta, data_off=data_off,
                file_size=file_size, recs=recs)


def patch_scheme(meta, scheme, group, pre_biased, split=None):
    out = bytearray(meta)
    p = 0
    while p + 8 <= len(out):
        key, typ, ln = struct.unpack_from('<HHI', out, p)
        if key == 3 and typ == 1 and ln >= 4:
            o = p + 8
            out[o] = scheme
            out[o + 1] = group
            out[o + 2] = pre_biased
            out[o + 3] = out[o + 3] if split is None else split
            return out
        p += 8 + ln
    return out


def block_rows(buf, rec, off, r0, n, unit=False):
    C = rec['cols']
    a = np.frombuffer(buf, dtype='<i2', count=n * C,
                      offset=off + r0 * C * 2).reshape(n, C)
    return a.astype(np.float32) * (1.0 if unit else rec['scale'])


def clip_scale(buf, rec, off, pct):
    """P-th percentile of |w| over a deterministic stride sample."""
    total = rec['rows'] * rec['cols']
    step = max(1, total // 2000000)
    vals = []
    for r0 in range(0, rec['rows'], 128):
        n = min(128, rec['rows'] - r0)
        vals.append(np.abs(block_rows(buf, rec, off, r0, n)).ravel()[::step])
    v = np.concatenate(vals)
    q = 100.0 if pct is None else float(pct)
    return float(np.percentile(v, q)) / 127.0


def quantize_matrix(buf, rec, off, s):
    C = rec['cols']
    out = np.empty((rec['rows'], C), dtype=np.uint8)
    for r0 in range(0, rec['rows'], 128):
        n = min(128, rec['rows'] - r0)
        w = block_rows(buf, rec, off, r0, n)
        q = np.clip(np.rint(w / s), -127, 127).astype(np.int32) + 128
        out[r0:r0 + n] = q.astype(np.uint8)
    return out.tobytes()


def quantize_matrix_row(buf, rec, off):
    """q8 with one scale per output row: (biased body, unit, plane).

    Same arithmetic as the packer's --q8-row-scale, so rebuilding a container
    this way gives the bytes the packer would have written.  The plane travels
    unit-relative because the record's scale field is the unit.
    """
    C = rec['cols']
    out = np.empty((rec['rows'], C), dtype=np.uint8)
    sc = np.empty(rec['rows'], dtype=np.float32)
    for r0 in range(0, rec['rows'], 128):
        n = min(128, rec['rows'] - r0)
        w = block_rows(buf, rec, off, r0, n)
        rs = np.abs(w).max(axis=1) / 127.0
        rs[rs == 0] = 1e-6
        sc[r0:r0 + n] = rs
        out[r0:r0 + n] = (np.clip(np.rint(w / rs[:, None]), -127, 127)
                          .astype(np.int32) + 128).astype(np.uint8)
    unit = float(sc.max())
    return out.tobytes(), unit, (sc / unit).astype('<f4')


def quantize_emb(buf, rec, off, rowscale):
    """Signed int8 body (id 1) plus the per-row f32 scales it was built with."""
    rows, C = rec['rows'], rec['cols']
    q = np.empty((rows, C), dtype=np.int8)
    sc = np.empty(rows, dtype=np.float32)
    for r0 in range(0, rows, 128):
        n = min(128, rows - r0)
        w = block_rows(buf, rec, off, r0, n, unit=True) * rowscale[r0:r0 + n]
        rs = np.abs(w).max(axis=1) / 127.0
        rs[rs == 0] = 1e-6
        sc[r0:r0 + n] = rs
        q[r0:r0 + n] = np.clip(np.rint(w / rs[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    ap.add_argument('--clip', type=float, default=None)
    ap.add_argument('--emb', choices=('keep', 'q8'), default='keep')
    ap.add_argument('--row-scale', action='store_true',
                    help='one q8 scale per output row (a part-3 plane per matrix) '
                         'instead of one per matrix')
    args = ap.parse_args()

    src = parse(args.src)
    if args.src == args.dst:
        raise SystemExit('refusing to overwrite the source')
    buf = np.memmap(args.src, dtype=np.uint8, mode='r')
    off0 = src['data_off']

    emb_src_scale = None
    for r in src['recs']:
        if r['dt'] == 0 and (r['tid'] & 0xFF) == 2:
            emb_src_scale = np.frombuffer(buf, dtype='<f4', count=r['rows'] * r['cols'],
                                          offset=off0 + r['off']).reshape(r['rows'], 1)

    meta = patch_scheme(src['meta'], 1, 0, 1)
    table_off = (64 + len(meta) + ALIGN - 1) // ALIGN * ALIGN
    data_off = table_off + src['n'] * REC_SIZE   # patched below once n is known

    bodies = []
    emb_scale = None
    for r in src['recs']:
        base = r['tid'] & 0xFF
        tid, dt, scale = r['tid'], r['dt'], r['scale']
        body = bytes(buf[off0 + r['off']:off0 + r['off'] + r['blen']])
        if r['dt'] == 1 and base in EMB_BASES and args.emb == 'q8':
            body, emb_scale = quantize_emb(buf, r, off0 + r['off'], emb_src_scale)
            tid, dt, scale = 1, 3, 0.0
        elif r['dt'] == 1 and base in MAT_BASES:
            s = clip_scale(buf, r, off0 + r['off'], args.clip)
            if not (s > 0):
                s = 1e-6
            if args.row_scale:
                body, unit, plane = quantize_matrix_row(buf, r, off0 + r['off'])
                bodies.append((body, tid, 2, unit, r['rows'], r['cols'], r['flags'], r['group']))
                bodies.append((plane.tobytes(), tid | (3 << 14), 0, 0.0, r['rows'], 1, 0, 0))
                continue
            body = quantize_matrix(buf, r, off0 + r['off'], s)
            dt, scale = 2, s
        elif r['dt'] == 0 and base == 2 and emb_scale is not None:
            body = emb_scale.tobytes()
        bodies.append((body, tid, dt, scale, r['rows'], r['cols'], r['flags'], r['group']))

    n_out = len(bodies)
    data_off = table_off + n_out * REC_SIZE
    out = bytearray(b'MMCF')
    out += struct.pack('<HHII', 2, 64, n_out, src['caps'])
    out += struct.pack('<II', 64, len(meta))
    out += struct.pack('<QQ', data_off, 0)
    out += b'\0' * 24
    out = out[:64]
    out += meta
    while len(out) < table_off:
        out += b'\0'
    table = bytearray()
    data = bytearray()
    for body, tid, dt, scale, brows, bcols, bflags, bgroup in bodies:
        while (data_off + len(data)) % ALIGN:
            data.append(0)
        off = len(data)
        data += body
        crc = zlib.crc32(body) & 0xFFFFFFFF
        b = struct.pack('<HBBBB', tid, dt, bflags, bgroup, 0)
        b += struct.pack('<II', brows, bcols)
        b += struct.pack('<QQ', off, len(body))
        b += struct.pack('<I', crc)
        b += struct.pack('<f', float(scale))
        table += b.ljust(REC_SIZE, b'\0')
    file_size = data_off + len(data)
    struct.pack_into('<Q', out, 32, file_size)
    struct.pack_into('<I', out, 40, zlib.crc32(bytes(out[:40])) & 0xFFFFFFFF)
    out += table
    out += data
    with open(args.dst, 'wb') as f:
        f.write(out)
    print('wrote %s: %d tensors, %d bytes (source %d)' %
          (args.dst, n_out, file_size, src['file_size']))


main()
