"""Read a torch zip checkpoint (`pytorch_model.bin`) without torch.

A HuggingFace release that predates safetensors -- and the BitCPM-CANN family,
which ships only this -- publishes `pytorch_model.bin`: a zip holding `data.pkl`
and one member per storage.  `mmcf_hf` needs names, shapes and float32 arrays,
and that is all this returns, so a container can be packed from such a
checkpoint on a machine with no torch installed.

Layout (stable since torch 1.6, the zipfile format):

    data.pkl                a pickle of {name: _rebuild_tensor_v2(storage, ...)}
    data/<key>              the storage bytes, little endian
    version, byteorder      ignored

bfloat16 is the top half of an fp32, so widening is exact; `float16`, `float32`
and `float64` are read with the dtype the pickle names, not guessed from the
file size (`half` and `bfloat16` are both two bytes).

    from pth_reader import load
    tensors = load("pytorch_model.bin")     # name -> numpy array

The loader materialises every tensor (866 MB of bf16 here becomes 1.7 GB of
fp32), which is what the packer does anyway.
"""
import pickle
import zipfile

import numpy as np

# torch's storage class name (lower-cased, `Storage` stripped) -> (numpy dtype, bytes)
DTYPES = {"bfloat16": ("<u2", 2), "half": ("<f2", 2), "float": ("<f4", 4),
          "float32": ("<f4", 4), "double": ("<f8", 8), "float64": ("<f8", 8),
          "uint8": ("u1", 1), "int8": ("i1", 1), "int16": ("<i2", 2),
          "int32": ("<i4", 4), "int64": ("<i8", 8), "bool": ("?", 1)}


class _Unpickler(pickle.Unpickler):
    """Rebuild `_rebuild_tensor_v2` calls out of the zip's storages."""

    def __init__(self, f, zipf, prefix):
        super().__init__(f)
        self.zipf = zipf
        self.prefix = prefix

    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda st, off, size, stride, *a, **k: dict(
                storage=st, offset=off, size=size, stride=stride)
        if mod == "torch" and name in ("Size", "Tensor"):
            return tuple
        if mod == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if mod == "torch" and name.endswith("Storage"):
            return name                    # the dtype, as a string
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        kind, dtype, key = pid[0], pid[1], pid[2]
        numel = int(pid[4])
        kind = str(kind)
        dtype = getattr(dtype, "__name__", str(dtype)).replace("Storage", "").lower()
        if kind != "storage":
            raise SystemExit("pth_reader: unsupported persistent id %r" % (pid[:2],))
        name = None
        for cand in (self.prefix + str(key), self.prefix + "data/" + str(key)):
            if cand in self.zipf.namelist():
                name = cand
                break
        if name is None:                   # torch 2 writes `data/<key>`
            raise SystemExit("pth_reader: the zip has no member for storage %r" % (key,))
        nd, _ = DTYPES[dtype]
        arr = np.frombuffer(self.zipf.read(name), dtype=nd, count=numel)
        return ("storage", dtype, arr)


def load(path):
    """{tensor name: numpy array}, every one widened to float32."""
    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise SystemExit("pth_reader: %s has no data.pkl (not a torch zip)" % path)
        prefix = pkl[0][: -len("data.pkl")]
        obj = _Unpickler(z.open(pkl[0]), z, prefix).load()

    out = {}
    for name, rec in obj.items():
        if not isinstance(rec, dict) or "storage" not in rec:
            continue                        # a non-tensor entry (e.g. a str)
        _, dtype, flat = rec["storage"]
        size, stride = tuple(rec["size"]), tuple(rec["stride"])
        n = int(np.prod(size))
        off = int(rec["offset"])
        if len(size) <= 1 or all(s == 1 for s in stride[1:]):
            a = flat[off: off + n]
        else:                               # materialise through the stride
            idx = np.arange(n)
            take = np.zeros(n, dtype=np.int64)
            for d, (s, st) in enumerate(zip(size, stride)):
                take += ((idx // int(np.prod(size[d + 1:]))) % s) * st
            a = flat[off + take]
        if dtype == "bfloat16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        elif dtype != "float32":
            a = a.astype(np.float32)
        out[name] = a.reshape(size)
    return out