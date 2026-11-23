"""Read a torch *legacy* checkpoint (`pytorch_model.bin`, torch < 1.6) without torch.

`pth_reader.py` reads the zip container that torch writes from 1.6 on.  A release
published before that (Helsinki-NLP's OPUS-MT family is the one in the store)
writes a bare pickle followed by the storage bodies, and `zipfile` refuses it with
"File is not a zip file".  This is the other reader; the API is the same
(`load(path) -> {name: numpy array}`) so a packer can try one and fall back.

Layout, from `torch/serialization.py` `_legacy_save`:

    LONG1 magic 0x1950a86a20f9469cfc6c          pickled
    PROTOCOL_VERSION 1001                       pickled
    sys_info dict                               pickled (little_endian, type_sizes)
    the state dict                              pickled; tensors are
                                                _rebuild_tensor[_v2](storage,
                                                offset, size, stride)
    the storage key list                        pickled, in *file* order
    per key: '<q' numel, then numel * element_size raw bytes

Three traps, all paid for once:

* The key list is its own pickle and it is the authority on the order the bodies
  follow.  Sorting the ids -- which is what the zip reader's table does -- puts
  every tensor at the wrong offset and still "loads": the shapes come out right
  and the numbers are garbage.
* The body header is numel **only**.  The zip format's `<qQ` (numel, nbytes) pair
  is the modern one; reading it here consumes a body's first eight bytes as a
  byte count and asks for a 1e18-byte read.
* `long` is 4 bytes in `sys_info` on this writer, so `type_sizes` cannot be used
  to size a numpy dtype.  The dtype comes from the storage class name.

`load()` asserts that the bodies consume the file exactly.  A misparse then fails
loudly at the read instead of producing plausible-looking weights, which is the
only failure mode worth guarding here: nothing downstream can tell a shifted
tensor from a badly trained one.

    python3 engine/tools/pth_legacy.py models/opus-mt-en-zh/pytorch_model.bin
"""
import collections
import pickle
import struct
import sys

import numpy as np

MAGIC = 0x1950a86a20f9469cfc6c
PROTOCOL = 1001

# torch's storage class name -> (numpy dtype, element size)
DTYPES = {"FloatStorage": ("<f4", 4), "DoubleStorage": ("<f8", 8),
          "HalfStorage": ("<f2", 2), "BFloat16Storage": ("<u2", 2),
          "LongStorage": ("<i8", 8), "IntStorage": ("<i4", 4),
          "ShortStorage": ("<i2", 2), "CharStorage": ("i1", 1),
          "ByteStorage": ("u1", 1), "BoolStorage": ("?", 1)}


def _unpickler(f, stype_of):
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
                return lambda st, off, size, stride, *a, **k: dict(
                    s=st, off=off, size=tuple(size), stride=tuple(stride))
            if mod == "torch" and name in ("Size", "Tensor"):
                return tuple
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod == "torch" and name.endswith("Storage"):
                return name
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            # ('storage', StorageClass, root_key, location, numel)
            stype_of[pid[2]] = pid[1]
            return ("storage", pid[1], pid[2], pid[4])

    return U(f)


def load(path):
    """name -> numpy array, in the order the checkpoint lists them."""
    stype_of = {}
    with open(path, "rb") as f:
        magic = pickle.load(f)
        if magic != MAGIC:
            raise ValueError("%s is not a legacy torch checkpoint (magic %r); "
                             "try pth_reader.load()" % (path, magic))
        if pickle.load(f) != PROTOCOL:
            raise ValueError("%s: unknown legacy protocol" % path)
        pickle.load(f)                       # sys_info
        obj = _unpickler(f, stype_of).load()
        keys = pickle.load(f)                # the body order, in file order
        bodies = {}
        for key in keys:
            numel = struct.unpack("<q", f.read(8))[0]
            if key not in stype_of:
                raise ValueError("%s: storage %r has no dtype" % (path, key))
            form, esz = DTYPES[stype_of[key]]
            buf = f.read(numel * esz)
            if len(buf) != numel * esz:
                raise ValueError("%s: storage %r truncated (%d of %d bytes)"
                                 % (path, key, len(buf), numel * esz))
            bodies[key] = buf
        end = f.tell()
        size = f.seek(0, 2)
        if end != size:
            raise ValueError("%s: bodies consumed %d of %d bytes; the layout is "
                             "not the one this reader knows" % (path, end, size))

    out = collections.OrderedDict()
    for name, t in obj.items():
        form, esz = DTYPES[t["s"][1]]
        buf = bodies[t["s"][2]]
        a = np.frombuffer(buf, dtype=form)
        views = 1
        for s in t["size"]:
            views *= s
        if views != len(a):
            raise ValueError("%s: %s is %s, storage holds %d values"
                             % (path, name, t["size"], len(a)))
        out[name] = np.lib.stride_tricks.as_strided(
            a[t["off"]:], shape=t["size"],
            strides=tuple(s * esz for s in t["stride"])).copy()
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    t = load(sys.argv[1])
    total = 0
    for name, a in t.items():
        total += a.size
        print("%-58s %-8s %s" % (name, a.dtype, tuple(a.shape)))
    print("%d tensors, %.1f M values" % (len(t), total / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
