#!/usr/bin/env python3
"""Convert an I16M1/I16M2 checkpoint to Q8M1/Q8M2 (int8 weights)."""
import struct, sys
import numpy as np

Q8 = 0x51384D31
Q8_V2 = 0x51384D32
I16 = 0x4931364D
I16_V2 = 0x4931364E

def q8mat(w):
    mx=float(np.abs(w).max())
    if mx==0: mx=1
    s=mx/127.0
    q=np.clip(np.rint(w/s),-127,127).astype(np.int8)
    return q,float(s)

def main(src,dst):
    f=open(src,'rb'); magic,dim,hidden,L,H,KV,vocab,seq=struct.unpack('<I7i',f.read(32))
    if magic==I16_V2:
        theta,half,split,qk=struct.unpack('<fiii',f.read(16))
    else:
        theta,half,split=struct.unpack('<fii',f.read(12)); qk=0
    assert magic in (I16,I16_V2), hex(magic)
    emb16=np.frombuffer(f.read(vocab*dim*2),dtype='<i2').reshape(vocab,dim)
    emb_sc=np.frombuffer(f.read(vocab*4),dtype=np.float32)
    rms_att=np.frombuffer(f.read(L*dim*4),np.float32)
    rms_ffn=np.frombuffer(f.read(L*dim*4),np.float32)
    rms_final=np.frombuffer(f.read(dim*4),np.float32)
    q_norm=k_norm=None
    if qk:
        hd=dim//H
        q_norm=np.frombuffer(f.read(L*hd*4),np.float32)
        k_norm=np.frombuffer(f.read(L*hd*4),np.float32)
    def rmat():
        rows,cols=struct.unpack('<ii',f.read(8)); sc=struct.unpack('<f',f.read(4))[0]
        q=np.frombuffer(f.read(rows*cols*2),dtype='<i2').reshape(rows,cols)
        return q,sc
    # The fused q8 W13 GPU path is broken; always write split W1/W3 so the
    # paired q16 FFN path is used.  The header and tensor order are the Q8M1
    # split layout, independent of the source I16 split flag.
    out_split = 1
    with open(dst,'wb') as o:
        if qk:
            o.write(struct.pack('<I7ifiii',Q8_V2,dim,hidden,L,H,KV,vocab,seq,theta,half,out_split,qk))
        else:
            o.write(struct.pack('<I7ii',Q8,dim,hidden,L,H,KV,vocab,seq,out_split))
        er=np.empty((vocab,dim),np.int8); es=np.empty(vocab,np.float32)
        for i in range(vocab):
            q,s=q8mat(emb16[i].astype(np.float32)*emb_sc[i]); er[i]=q; es[i]=s
        o.write(er.tobytes()); o.write(es.tobytes())
        o.write(rms_att.tobytes()); o.write(rms_ffn.tobytes()); o.write(rms_final.tobytes())
        if qk:
            o.write(q_norm.tobytes()); o.write(k_norm.tobytes())
        def wm(q,sc):
            w=q.astype(np.float32)*sc
            q8,s=q8mat(w)
            o.write(struct.pack('<iif',q.shape[0],q.shape[1],s)); o.write(q8.tobytes())
        for l in range(L):
            wm(*rmat()); wm(*rmat())
            if split:
                wm(*rmat()); wm(*rmat())
            else:
                q, sc = rmat()
                wm(q[:hidden], sc); wm(q[hidden:], sc)
            wm(*rmat())
        nt=(vocab+1535)//1536
        for _ in range(nt): wm(*rmat())
    print(dst,'split',out_split,'qk_norm',qk)

main(sys.argv[1],sys.argv[2])
