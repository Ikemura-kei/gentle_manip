#!/usr/bin/env python3
"""Emit the live gen8-eval HTML fragment (goes between <!--GEN8_EVAL_START/END-->).
Reads per-category summary.json from each model's gen_eval dir + embeds montage mp4s.

usage: gen8_eval_page.py <baseline_gen_eval_dir|-> <regrasp_gen_eval_dir|-> <montage_dir>
prints the fragment to stdout.
"""
import sys, os, json, glob, base64, datetime

INDOMAIN = ["mushroom", "banana_lying", "kiwi", "egg_boiled", "grape", "cherry", "tomato", "raspberry"]
OOD      = ["blackberry", "scallop", "dumpling", "gelatin"]
NICE = {"banana_lying": "banana", "egg_boiled": "egg (boiled)"}

def load(evd, cat):
    if not evd or evd == "-":
        return None
    f = os.path.join(evd, cat, "summary.json")
    if not os.path.isfile(f):
        return None
    try:
        d = json.load(open(f))
    except Exception:
        return None
    return {
        "sr": d.get("success_rate"),
        "g": d.get("gentleness_score"),
        "c": d.get("combined_sr_gentleness"),
        "n": d.get("n_episodes"),
    }

def b64vid(path):
    if not os.path.isfile(path):
        return None
    return "data:video/mp4;base64," + base64.b64encode(open(path, "rb").read()).decode()

def vfig(src, label, cls):
    if not src:
        return (f'<figure class="clip"><div style="aspect-ratio:4/3;display:flex;align-items:center;'
                f'justify-content:center;color:#888;font-family:monospace;font-size:.8rem">pending</div>'
                f'<figcaption><span class="tag {cls}">{label}</span></figcaption></figure>')
    return (f'<figure class="clip"><video src="{src}" autoplay loop muted playsinline></video>'
            f'<figcaption><span class="tag {cls}">{label}</span> 15 rollouts · 2&times; speed</figcaption></figure>')

def mrow(b, r):
    def cell(m, key, pct=True):
        if not m or m.get(key) is None:
            return "&ndash;"
        v = m[key]
        return f"{v*100:.0f}%" if pct else f"{v:.3f}"
    def one(m, tag):
        if not m:
            return f'<li><b>{tag}</b> &mdash; pending</li>'
        return (f'<li><b>{tag}</b> &mdash; SR {cell(m,"sr")} · '
                f'gentle {cell(m,"g",0)} · SR&times;g {cell(m,"c",0)}</li>')
    out = one(b, "baseline") + one(r, "ours")
    if b and r and b.get("c") is not None and r.get("c") is not None:
        d = r["c"] - b["c"]
        sign = "+" if d >= 0 else "&minus;"
        col = "tag-a" if d >= 0 else "tag-fail"
        out += f'<li class="{col}" style="border:0"><b>&Delta; SR&times;g {sign}{abs(d):.3f}</b></li>'
    return f'<ul class="metrics" style="margin:.5rem 0 .8rem">{out}</ul>'

def block(name, cats, be, re_, mdir, embed=True):
    rows = []
    accB = {"sr": [], "g": [], "c": []}
    accR = {"sr": [], "g": [], "c": []}
    done = 0
    for cat in cats:
        b, r = load(be, cat), load(re_, cat)
        mb = b64vid(os.path.join(mdir, f"montage_baseline_{cat}.mp4")) if embed else None
        mr = b64vid(os.path.join(mdir, f"montage_regrasp_{cat}.mp4")) if embed else None
        if b and r:
            done += 1
            for k in accB:
                if b.get(k) is not None: accB[k].append(b[k])
                if r.get(k) is not None: accR[k].append(r[k])
        rows.append(
            f'<h3 style="font-family:\'Barlow Condensed\',sans-serif;font-weight:600;margin:1.7rem 0 .3rem">{NICE.get(cat,cat)}</h3>'
            + mrow(b, r)
            + '<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">'
            + vfig(mb, "non-regraspable baseline", "tag-b")
            + vfig(mr, "regrasp generalist &mdash; ours", "tag-a")
            + '</div>')
    agg = ""
    if done == len(cats):
        def mean(x): return sum(x)/len(x) if x else 0
        agg = (f'<ul class="metrics" style="margin:.6rem 0 0">'
               f'<li><b>{name} MEAN &mdash; baseline</b> SR {mean(accB["sr"])*100:.0f}% · '
               f'gentle {mean(accB["g"]):.3f} · SR&times;g {mean(accB["c"]):.3f}</li>'
               f'<li class="tag-a" style="border:0"><b>{name} MEAN &mdash; ours</b> SR {mean(accR["sr"])*100:.0f}% · '
               f'gentle {mean(accR["g"]):.3f} · SR&times;g {mean(accR["c"]):.3f}</li></ul>')
    return (f'<h3 style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;font-size:1.15rem;'
            f'margin:2rem 0 .2rem;color:var(--accent)">{name} &mdash; {done}/{len(cats)} categories</h3>'
            + agg + "".join(rows))

def main():
    be, re_, mdir = sys.argv[1], sys.argv[2], sys.argv[3]
    embed_ood = (len(sys.argv) < 5) or sys.argv[4] != "0"
    nB = sum(1 for c in INDOMAIN+OOD if load(be, c))
    nR = sum(1 for c in INDOMAIN+OOD if load(re_, c))
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    frag = [
        '<h2>Act 3.5 &mdash; Cross-category generalist eval <span style="font-family:\'IBM Plex Mono\',monospace;'
        'font-size:.7rem;color:var(--warm)">LIVE</span></h2>',
        f'<p>The 8-object regrasp generalist (<b>ours</b>, ~4.4k diverse-start demos incl. recovery starts) vs the '
        f'<b>non-regraspable baseline</b> (same data &amp; architecture, the <span class="tag tag-fail">failed_grasp</span> '
        f'recovery family removed). Canonical harness: <b>100 rollouts / category</b>, EE start spanning home&harr;near-object, '
        f'geometry+material re-randomised every 5 episodes. Three metrics: success rate, gentleness '
        f'(1 &minus; internal stress / yield), and their mean.</p>',
        f'<div class="foot" style="margin:1rem 0 0">progress &mdash; baseline {nB}/12 · ours {nR}/12 categories '
        f'&nbsp;·&nbsp; updated {ts} &nbsp;·&nbsp; each clip: 15 random rollouts, 2&times; speed, black gaps</div>',
        block("IN-DOMAIN", INDOMAIN, be, re_, mdir, embed=True),
        ("" if embed_ood else '<p style="color:var(--ink-muted);font-size:.9rem">OOD montages omitted from '
         'this page for size; metrics below, clips on disk.</p>'),
        block("OUT-OF-DOMAIN (zero-shot)", OOD, be, re_, mdir, embed=embed_ood),
        '<div class="foot" style="margin:1.6rem 0 0">Eval videos: '
        '<span style="font-family:monospace">logs/dppo/dppo-pretrain/single_lift_gen8_{baseline,regrasp}_pcd/'
        '&lt;run&gt;/gen_eval_*/&lt;cat&gt;/render/</span> · per-category '
        '<span style="font-family:monospace">summary.json</span> / '
        '<span style="font-family:monospace">episodes.csv</span> · aggregate '
        '<span style="font-family:monospace">gen_eval_*/aggregate.json</span></div>',
    ]
    sys.stdout.write("\n".join(frag))

if __name__ == "__main__":
    main()
