#!/usr/bin/env python3
"""Per-category collection progress toward 500/object.
Counts ONLY the authoritative collector runs (one per size class), not smoke/retries."""
import re, os, glob, time
REPO="/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip"
T=500
# authoritative run logs: newest xcat_regrasp collector + newest xcat_small collector
def newest(pat_tag):
    best=None
    for lg in glob.glob(f"{REPO}/logs/slurm_logs/*.out"):
        h=open(lg,errors="ignore").read(600)
        if f"tag={pat_tag} " in h or f"TAG={pat_tag}" in h:
            if best is None or os.path.getmtime(lg)>os.path.getmtime(best): best=lg
    return best
def count(lg):
    cur=None; g={}; sv=[]
    if not lg or not os.path.exists(lg): return g,None,None
    lines=open(lg,errors="ignore").read().splitlines()
    for ln in lines:
        m=re.search(r'scene object -> (\w+)',ln); cur=m.group(1) if m else cur
        if re.search(r'ep \d+: env \d+\s+OK\b.*mode=',ln) and cur: g[cur]=g.get(cur,0)+1
        m=re.search(r'\[(\d+)/\d+ saved\]',ln)
        if m: sv.append((int(m.group(1)),))
    return g, (sv[-1][0] if sv else sum(g.values())), lg

rr=newest("xcat_regrasp"); rs=newest("xcat_small")
g_rr,tot_rr,_=count(rr); g_rs,tot_rs,_=count(rs)
have={"mushroom":0,"banana_lying":500,"kiwi":0,"egg_boiled":0,"grape":0,"cherry":0,"tomato":0,"raspberry":0}
for c,n in {**g_rr,**{k:g_rs.get(k,0) for k in g_rs}}.items():
    if c in have: have[c]=have[c]+n if c!="banana_lying" else 500+n
# regrasp collector STOPPED at 499 (merged run 26-08-30-rdz) — hardcode its split if log parse thin
FIXED={"mushroom":111,"banana_lying":626,"kiwi":139,"egg_boiled":123}
for c,v in FIXED.items(): have[c]=max(have[c],v)

order=["mushroom","banana_lying","kiwi","egg_boiled","grape","cherry","tomato","raspberry"]
print(f"{'category':13s} {'have':>4s}/{T}")
for c in order:
    n=have[c]; bar="#"*int(20*min(n,T)/T)
    print(f"  {c:13s} {min(n,T):3d}/{T}  [{bar:<20s}]" + (f"  (+{n-T} over)" if n>T else ""))
tot=sum(min(have[c],T) for c in order)
print(f"\n  TOTAL {tot}/{8*T} ({100*tot/(8*T):.0f}%) toward 8 x {T}")
# live small-collector speed
if rs:
    txt=open(rs,errors="ignore").read()
    mm=re.findall(r'\[(\d+)/\d+ saved\]',txt)
    if mm:
        cur=int(mm[-1]); start=os.path.getmtime(rs)  # proxy
        # find job start from first line timestamp is unreliable; use SLURM elapsed via env? just print current
        print(f"\n  small collector live: {cur} saved this run")
