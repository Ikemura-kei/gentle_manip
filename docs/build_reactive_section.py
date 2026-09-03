#!/usr/bin/env python3
"""Insert / refresh the 'Act 3 - Reactive policy' block in regrasp_demos.html,
right AFTER the <!--GEN8_EVAL_END--> marker (outside the gen8-refresh-managed region).
Idempotent: replaces an existing <!--REACT_START-->..<!--REACT_END--> block.
"""
import base64, json, os, re

SP   = "/tmp/claude-4004623/-home-yifeid-git-gentle-manip/d288935c-983e-4b48-8b67-26b01f3d4989/scratchpad"
REPO = "/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip"
HTML = os.path.join(SP, "regrasp_demos.html")
M    = os.path.join(SP, "montages")

V3_RX   = os.path.join(REPO, "logs/dppo/dppo-pretrain/single_lift_gen8_reactive_pcd/pkoie/gen_eval_20260902_112303")
V3_CLN  = os.path.join(REPO, "logs/dppo/dppo-pretrain/single_lift_gen8_reactive_pcd/pkoie/gen_eval_clean_1058")
RX_FAIR = os.path.join(REPO, "logs/dppo/dppo-pretrain/single_lift_gen8_regrasp_pcd/lorap/gen_eval_20260903_082522")

CATS = [("mushroom", "mushroom"), ("banana", "banana_lying"),
        ("kiwi", "kiwi"), ("egg", "egg_boiled")]

# lorap zero-shot under the WEAKER (0.75) drag -- Phase B, reactive_eval_20260902_0312
RX_ZS = {"mushroom": (0.60, 0.674), "banana": (0.77, 0.645),
         "kiwi": (0.65, 0.754), "egg": (0.80, 0.593)}

def load(d, key):
    f = os.path.join(d, key, "summary.json")
    if not os.path.isfile(f):
        return None
    j = json.load(open(f))
    return (j["success_rate"], j["gentleness_score"])

def b64(p):
    return "data:video/mp4;base64," + base64.b64encode(open(p, "rb").read()).decode()

collect_vid = b64(os.path.join(M, "reactive_collect_sm.mp4"))
v3_vid      = b64(os.path.join(M, "reactive_v3_4cat_sm.mp4"))

def cell(v):
    if v is None:
        return '<td class="pend">&middot;</td><td class="pend">&middot;</td><td class="pend">&middot;</td>'
    sr, g = v
    return f'<td>{sr:.2f}</td><td>{g:.2f}</td><td class="c">{sr*g:.2f}</td>'

def mean(vs):
    # only a mean when every category is present -- a partial mean is misleading
    if not vs or any(v is None for v in vs):
        return None
    return (sum(x[0] for x in vs) / len(vs), sum(x[1] for x in vs) / len(vs))

rows, m_zs, m_fair, m_v3, m_cln = [], [], [], [], []
for disp, key in CATS:
    zs   = RX_ZS.get(disp)
    fair = load(RX_FAIR, key)
    v3   = load(V3_RX, key)
    cln  = load(V3_CLN, key)
    m_zs.append(zs); m_fair.append(fair); m_v3.append(v3); m_cln.append(cln)
    rows.append(f"<tr><th>{disp}</th>{cell(zs)}{cell(fair)}{cell(v3)}{cell(cln)}</tr>")
mzs, mfair, mv3, mcln = mean(m_zs), mean(m_fair), mean(m_v3), mean(m_cln)
rows.append(f'<tr class="mean"><th>mean</th>{cell(mzs)}{cell(mfair)}{cell(mv3)}{cell(mcln)}</tr>')
table_rows = "\n".join(rows)

fair_txt = (f"lorap under the <i>same</i> 0.90 drag lands "
            f"<b>{mfair[0]:.2f}</b> SR / <b>{mfair[1]:.2f}</b> gentleness"
            if mfair else
            "lorap under the <i>same</i> 0.90 drag is still evaluating")

# verdict line depends on whether the fair baseline is in
if mfair:
    if mfair[0] - mv3[0] > 0.08:
        headline = ("the reactive-recovery mix <b>cost success rate</b> under this "
                    "out-of-distribution drag while buying gentleness")
    elif abs(mfair[0] - mv3[0]) <= 0.08:
        headline = ("v3 and the non-reactive baseline are <b>even on success</b> under the "
                    "0.90 drag &mdash; and v3 is clearly gentler")
    else:
        headline = "v3 <b>recovers success rate</b> the drag takes from the baseline, and is gentler"
else:
    headline = ("<b>Gentleness recovered; success rate did not</b> &mdash; under a drag "
                "stronger than v3 was trained for")

block = f'''<!--REACT_START-->
<h2>Act 3 &mdash; Reactive policy <span style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--warm)">under disturbance</span></h2>

<p>A harder test of the same idea: <b>the object is yanked sideways while the arm is
descending.</b> Each episode, with 90&nbsp;% probability, a random lateral impulse
(0.40&ndash;0.95&nbsp;m/s, held four sim-frames, random direction) hits the object
mid-approach and slides it 6&ndash;18&nbsp;cm. The arm has to notice, re-approach the
object's <i>new</i> position, and still land a gentle grasp&ndash;lift.</p>

<p>No new architecture &mdash; the reactive policy (<b>v3</b>) is the same DP3 point-cloud
diffusion policy, made reactive by <b>(i)</b> mixing <b>~22&nbsp;% reactive-recovery
demonstrations</b> into the training set (1096 diverse-start + 309 object-dragged&rarr;re-target
episodes) and <b>(ii)</b> a shared sim/real
<span style="font-family:'IBM Plex Mono',monospace">object_at_gripper</span> cue (fraction of
the cloud near the gripper + its mean offset, obs&nbsp;dim 8&rarr;12). Recovery is learned
visual servoing: the cloud shifts, and the policy has seen enough
&ldquo;cloud&nbsp;shifts&nbsp;&rarr;&nbsp;redirect&rdquo; to follow it.</p>

<h3 style="font-family:'Barlow Condensed',sans-serif;font-weight:600;margin:1.9rem 0 .4rem">How the reactive demos are made</h3>
<p>The scripted CMA-ES demonstrator gets the same random drag, waits for the object to
settle, and &mdash; if it moved &mdash; raises the end-effector to a hover over the new
spot and descends again. Those trajectories are the
<span class="tag tag-fail">reactive_recover</span> demo family.</p>
<figure class="clip" style="max-width:520px">
  <video src="{collect_vid}" autoplay loop muted playsinline></video>
  <figcaption>scripted demonstrator &mdash; object dragged &rarr; re-target &rarr; grasp</figcaption>
</figure>

<h3 style="font-family:'Barlow Condensed',sans-serif;font-weight:600;margin:2.1rem 0 .4rem">Eval &mdash; 4 large categories, 60 rollouts each</h3>
<div style="overflow-x:auto">
<table class="react-tbl" style="border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:.78rem;margin:.4rem 0 1rem;min-width:640px">
<thead>
<tr><th rowspan="2" style="text-align:left;padding:.35rem .6rem">category</th>
    <th colspan="3">lorap zero-shot<br><span class="sub">non-reactive &middot; 0.75 drag</span></th>
    <th colspan="3">lorap<br><span class="sub">non-reactive &middot; 0.90 drag</span></th>
    <th colspan="3">v3 reactive<br><span class="sub">0.90 drag</span></th>
    <th colspan="3">v3 reactive<br><span class="sub">no drag</span></th></tr>
<tr>{"".join('<th class="sub2">%s</th>' % h for h in ["SR", "g", "&times;"] * 4)}</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
<style>
.react-tbl th{{padding:.3rem .55rem;font-weight:600;text-align:center;border-bottom:1px solid var(--line-strong)}}
.react-tbl .sub{{font-weight:400;color:var(--ink-muted);font-size:.9em}}
.react-tbl .sub2{{font-weight:400;color:var(--ink-muted);border-bottom:1px solid var(--line)}}
.react-tbl td{{padding:.28rem .55rem;text-align:right}}
.react-tbl td.c{{color:var(--accent);font-weight:600}}
.react-tbl td.pend{{color:var(--line-strong);text-align:center}}
.react-tbl tr.mean th,.react-tbl tr.mean td{{border-top:1px solid var(--line-strong);font-weight:700}}
</style>

<div style="border-left:3px solid var(--warm);padding:.6rem 0 .6rem 1rem;margin:.6rem 0 1.2rem;background:color-mix(in srgb, var(--warm) 6%, transparent)">
<b>Read.</b> {headline}. v3-reactive holds
gentleness under the disturbance (<b>{mv3[1]:.2f}</b>, matching its own no-drag
<b>{mcln[1]:.2f}</b> and above lorap zero-shot's <b>{mzs[1]:.2f}</b>) &mdash; it does not
panic-crush when the object jumps. Its success rate under the 0.90 drag is
<b>{mv3[0]:.2f}</b> (<b>{mv3[0]-mcln[0]:+.2f}</b> vs its own clean run), with mushroom the
worst case (0.23). The reactive-recovery demos were collected with gentle
0.12&ndash;0.38&nbsp;m/s drags; this eval slides the object 2&ndash;3&times; faster (a
deliberately-visible perturbation), so v3 is being asked to generalise outside its
recovery-training distribution. {fair_txt}. Next step: a <b>v4</b> with recovery demos
collected at the eval drag speed.</div>

<figure class="clip" style="max-width:560px">
  <video src="{v3_vid}" autoplay loop muted playsinline></video>
  <figcaption>v3 under perturbation &mdash; mushroom, banana, kiwi, egg &middot; 15 rollouts each, 2&times; speed</figcaption>
</figure>
<!--REACT_END-->'''

src = open(HTML).read()
if "<!--REACT_START-->" in src:
    src = re.sub(r"<!--REACT_START-->.*?<!--REACT_END-->", block, src, flags=re.S)
else:
    src = src.replace("<!--GEN8_EVAL_END-->", "<!--GEN8_EVAL_END-->\n\n" + block, 1)
open(HTML, "w").write(src)
print(f"reactive block {'refreshed' if '<!--REACT_START-->' in open(HTML).read() else 'inserted'}; "
      f"page -> {len(src.encode())/1e6:.2f} MB; fair-baseline={'IN' if mfair else 'pending'}")
