"""Find the head permutation llama.cpp's qwen3next/qwen35moe converter applies to the Gated DeltaNet
tensors, by matching the GGUF (unsloth UD-Q4_K_M) against the owner's HF-named colibrì container."""
import json, struct, subprocess, sys, math, array
sys.path.insert(0, "../../06_Code/c")
import ggufinfo
S = "./work"
R = "https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64/resolve/main/"
G = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
import hashlib, os
os.makedirs(S+"/cache", exist_ok=True)
def fetch(url, s, e):
    key = S+"/cache/"+hashlib.md5(f"{url}:{s}:{e}".encode()).hexdigest()
    if os.path.exists(key): return open(key,"rb").read()
    b = subprocess.run(["curl","-SS","-L","--max-time","300","-r",f"{s}-{e}",url], capture_output=True, check=True).stdout
    open(key,"wb").write(b); return b
def f16(b):
    a = array.array("H"); a.frombytes(b); o=[]
    for h in a:
        s=-1.0 if h&0x8000 else 1.0; e=(h>>10)&0x1F; m=h&0x3FF
        o.append(s*(m/1024.0)*2.0**-14 if e==0 else (s*float('inf') if e==31 else s*(1+m/1024.0)*2.0**(e-15)))
    return o
def f32(b): a=array.array("f"); a.frombytes(b); return list(a)
def q8_0_rows(b, I):   # Q8_0: blocks of 32 -> f16 d + 32 int8; returns list of rows (floats)
    nb = I // 32; bs = 34; rows = []; rb = nb*bs
    for r in range(len(b)//rb):
        row=[]; base=r*rb
        for k in range(nb):
            d = f16(b[base+k*bs:base+k*bs+2])[0]
            q = array.array("b"); q.frombytes(b[base+k*bs+2:base+(k+1)*bs])
            row.extend(d*v for v in q)
        rows.append(row)
    return rows

# --- HF side: locate tensors
want = {"model.layers.0.linear_attn.A_log":None,"model.layers.0.linear_attn.dt_bias":None,
        "model.layers.0.linear_attn.in_proj_a.weight":None,"model.layers.0.linear_attn.in_proj_b.weight":None,
        "model.layers.0.linear_attn.conv1d.weight":None,"model.layers.0.linear_attn.in_proj_qkv.weight":None,
        "model.layers.0.linear_attn.in_proj_z.weight":None,"model.layers.0.linear_attn.out_proj.weight":None}
for i in range(41):
    url=f"{R}model-{i:05d}.safetensors"; hlen=struct.unpack("<Q",fetch(url,0,7))[0]; hdr=json.loads(fetch(url,8,8+hlen-1))
    for w in want:
        if w in hdr and want[w] is None:
            s,e=hdr[w]["data_offsets"]; want[w]=(url,8+hlen+s,8+hlen+e-1,hdr[w]["dtype"],hdr[w]["shape"])
    if all(v is not None for v in want.values()): break
hf={w:(f16(fetch(*v[:3])) if v[3]=="F16" else f32(fetch(*v[:3])), v[4]) for w,v in want.items()}
print("HF shapes:", {k.split('.')[-2]+'.'+k.split('.')[-1]: v[1] for k,v in hf.items()})

parts = ggufinfo.open_set(f"{S}/unsloth"); T={t.name:t for p in parts for t in p.tensors}
def gg_f32(n): t=T[n]; return f32(fetch(G,t.off,t.off+t.nbytes-1))
def gg_q8(n): t=T[n]; return q8_0_rows(fetch(G,t.off,t.off+t.nbytes-1), t.ne[0])

ssm_a=gg_f32("blk.0.ssm_a"); alog=hf["model.layers.0.linear_attn.A_log"][0]
tgt=[-math.exp(x) for x in alog]
perm=[2*i for i in range(16)]+[2*i+1 for i in range(16)]   # hypothesis: GGUF v-head i = HF v-head perm[i] (even heads, then odd)
err=max(abs(ssm_a[i]-tgt[perm[i]]) for i in range(32))
print("ssm_a[i] = -exp(A_log[perm[i]]), perm =", perm, "max err", f"{err:.2e}", "bijective:", sorted(perm)==list(range(32)))
dt=gg_f32("blk.0.ssm_dt.bias"); dtb=hf["model.layers.0.linear_attn.dt_bias"][0]
print("dt_bias follows same perm: max err", f"{max(abs(dt[i]-dtb[perm[i]]) for i in range(32)):.2e}")
# in_proj_a / b rows (32 x 2048) under the same head perm
for gname,hname in (("blk.0.ssm_alpha.weight","model.layers.0.linear_attn.in_proj_a.weight"),("blk.0.ssm_beta.weight","model.layers.0.linear_attn.in_proj_b.weight")):
    g=gg_f32(gname); h=hf[hname][0]; H=2048
    ok=max(abs(g[i*H+c]-h[perm[i]*H+c]) for i in range(32) for c in range(0,H,97))
    print(f"{gname} rows == HF rows[perm]: max err {ok:.2e}")
# conv1d: HF [8192,1,4] channels = [q (16 kheads x128) | k (16 x128) | v (32 vheads x128)]; GGUF ne {4, 8192}
conv_g=gg_f32("blk.0.ssm_conv1d.weight"); conv_h=hf["model.layers.0.linear_attn.conv1d.weight"][0]
def ch_match(g_ch):   # which HF channel has identical 4 taps
    pass
# derive candidate channel permutation from the head perm: v channels permuted by head perm; q/k: k-head perm = ?
kperm=[min(range(16), key=lambda j: abs(ssm_a[2*i]-tgt[2*j]) ) for i in range(16)]  # placeholder, test below
import itertools
def conv_err(chperm):
    return max(abs(conv_g[c*4+t]-conv_h[chperm[c]*4+t]) for c in range(0,8192,131) for t in range(4))
ident=list(range(8192)); print("conv identity err", f"{conv_err(ident):.2e}")
# hypothesis: v block (channels 4096..8191) permuted by head perm (128 per head); q,k blocks by a k-head perm derived as perm of v-head pairs
vperm=ident[:4096]+[4096+perm[h]*128+d for h in range(32) for d in range(128)]
print("conv v-block head-perm err", f"{conv_err(vperm):.2e}")
# find k-head permutation by matching q-block channels directly
def block_head_perm(off_g, off_h, nheads, hd):
    pm=[]
    for i in range(nheads):
        best=min(range(nheads), key=lambda j: sum(abs(conv_g[(off_g+i*hd+d)*4+t]-conv_h[(off_h+j*hd+d)*4+t]) for d in range(0,hd,7) for t in range(4)))
        pm.append(best)
    return pm
qperm=block_head_perm(0,0,16,128); kp=block_head_perm(2048,2048,16,128)
print("q-head perm", qperm, "k-head perm", kp)
full=[qperm[h]*128+d for h in range(16) for d in range(128)]+[2048+kp[h]*128+d for h in range(16) for d in range(128)]+[4096+perm[h]*128+d for h in range(32) for d in range(128)]
print("conv full-perm err", f"{conv_err(full):.2e}")
# in_proj_qkv rows (8192 x 2048, Q8_0 in GGUF, f16 in HF) under the same channel perm (dequant error expected ~1e-2 relative)
qkv_g=gg_q8("blk.0.attn_qkv.weight"); qkv_h=hf["model.layers.0.linear_attn.in_proj_qkv.weight"][0]; H=2048
def rel(rg, rh): num=sum((a-b)**2 for a,b in zip(rg,rh)); den=sum(b*b for b in rh)+1e-30; return math.sqrt(num/den)
rows=[0,1,500,2048,2100,4096,4097,4224,8000,8191]
print("attn_qkv rows vs HF rows[full perm]  rel-err:", [f"{rel(qkv_g[r], qkv_h[full[r]*H:(full[r]+1)*H]):.3f}" for r in rows])
print("attn_qkv rows vs HF same index        rel-err:", [f"{rel(qkv_g[r], qkv_h[r*H:(r+1)*H]):.3f}" for r in rows])
z_g=gg_q8("blk.0.attn_gate.weight"); z_h=hf["model.layers.0.linear_attn.in_proj_z.weight"][0]
zperm=[perm[h]*128+d for h in range(32) for d in range(128)]
print("attn_gate rows vs in_proj_z[head perm] rel-err:", [f"{rel(z_g[r], z_h[zperm[r]*H:(zperm[r]+1)*H]):.3f}" for r in (0,1,200,2048,4095)])
out_g=gg_q8("blk.0.ssm_out.weight"); out_h=hf["model.layers.0.linear_attn.out_proj.weight"][0]; V=4096
print("ssm_out row0 vs out_proj row0 (cols permuted by head perm) rel-err:", f"{rel(out_g[0], [out_h[0*V+zperm[c]] for c in range(V)]):.3f}", " same-index:", f"{rel(out_g[0], out_h[0:V]):.3f}")
json.dump({"vhead_perm":perm,"qhead_perm":qperm,"khead_perm":kp}, open(f"{S}/deltanet_perm.json","w"))
