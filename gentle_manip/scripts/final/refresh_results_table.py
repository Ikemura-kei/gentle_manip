"""Regenerate the main table in docs/final/results.md from the eval dirs on disk.

Rebuilds the table wholesale rather than appending, so a duplicate or a missed entry is
structurally impossible: every eval with a summary.json appears exactly once. Prose sections are
left untouched. Run after any eval finishes:

    uv run --project envs/dppo python gentle_manip/scripts/final/refresh_results_table.py
"""
import glob, json, os, re, sys
import numpy as np

RUNS = {"wiayg": "baseline", "mmgyy": "G3", "bpfnl": "G4", "fsynt": "G5"}
DATES = ("2026-09-08", "2026-09-09")
DOC = "docs/final/results.md"


def _mm(a):                       # normalized [-1,1] -> gripper width mm (absolute mode)
    return ((np.asarray(a, float) + 1) / 2 * 0.088) * 1000


def _at_grasp(d, K):
    """Mean AT-GRASP closing width (mm): the width where the gripper stops closing, averaged over
    every attempt in an episode and then over episodes.

    An attempt is a local minimum of commanded width, >=6 mm below the preceding open level, with
    the EE down at the table in the steps BEFORE it. The "before" matters: the width minimum
    coincides with lift-off, so testing the EE at the minimum rejects every real grasp.

    This is more meaningful than `exec mm` (the episode MINIMUM), which is dominated by whichever
    single attempt closed furthest and hides a policy that closes correctly once in three tries.
    Successes close below the object, failures close at or above it.
    """
    per_ep = []
    for f in sorted(glob.glob(d + "signals/ep*.npz")):
        try:
            z = np.load(f)
        except Exception:
            continue
        w = _mm(z["action_chunks"][:, :, -1]).reshape(-1)
        ee = np.asarray(z["ee_pos"], float)
        if not len(ee):
            continue
        zz = np.repeat(ee[:, 2], K)[: len(w)] * 1000
        n = min(len(w), len(zz)); w, zz = w[:n], zz[:n]
        if n < 12:
            continue
        floor, out = zz.min(), []
        for i in range(8, n - 2):
            if w[i] > w[max(0, i-3):i+4].min() + 1e-9: continue
            if w[max(0, i-40):i].max() - w[i] < 6.0:   continue
            if zz[i-8:i].min() > floor + 15.0:         continue
            if out and i - out[-1][0] < 10:
                if w[i] < out[-1][1]: out[-1] = (i, w[i])
                continue
            out.append((i, w[i]))
        if out:
            per_ep.append(np.mean([x[1] for x in out]))
    return np.mean(per_ep) if per_ep else None


def _widths(d, K):
    """(mean tightest PLANNED width, mean tightest EXECUTED width) in mm, or None."""
    P, E = [], []
    for f in sorted(glob.glob(d + "signals/ep*.npz")):
        try:
            z = np.load(f)
        except Exception:
            continue
        ew = _mm(z["action_chunks"][:, :, -1]).reshape(-1)
        E.append(ew.min())
        fu = z["action_chunks_full"] if "action_chunks_full" in z.files else None
        if fu is not None and fu.size:
            T, H, _ = fu.shape
            P.append(min(_mm(fu[t, h, -1]) for t in range(T) for h in range(H) if t * K + h < ew.size))
    return (np.mean(P) if P else None), (np.mean(E) if E else None)


def rows():
    out = []
    for d in sorted(glob.glob("logs/dppo/dppo-pretrain/**/eval/*/", recursive=True)):
        if not os.path.exists(d + "summary.json"):
            continue
        s = json.load(open(d + "summary.json"))
        ck = str(s.get("checkpoint", ""))
        run = ck.split("/")[-3] if "/" in ck else "?"
        stamp = d.split("/")[-2]
        if run not in RUNS or not stamp.startswith(DATES):
            continue
        obj = str(s.get("experiment", "")).replace("single_lift_", "").split("_soft")[0]
        n = s.get("n_episodes", 0)
        su, ev = round(s["success_rate"] * n), round(s["ever_success_rate"] * n)
        ex = 8 if (s.get("max_policy_steps") or 99) < 50 else 4
        lab = open(d + "LABEL.txt").read().strip() if os.path.exists(d + "LABEL.txt") else ""
        pol = RUNS[run] + ("+ens" if "ens" in lab else "")
        if re.search(r"m0?\.?33", lab):
            pol += " m.33"
        pl, exm = _widths(d, ex)
        gr = _at_grasp(d, ex)
        out.append((obj, pol, os.path.basename(ck).replace(".pt", ""), ex, n, su, ev, ev - su,
                    s.get("stress_top20_ttop20_mean"), s.get("stress_max_tmax_mean"),
                    pl, exm, gr, stamp[-8:]))
    order = {"cherry_tomato": 0, "mushroom": 1, "tofu": 2, "banana_chunk": 3}
    out.sort(key=lambda r: (order.get(r[0], 9), r[1], r[3], r[2], r[13]))
    return out


def main():
    rs = rows()
    tbl = ["| object | policy | ckpt | exec | success | ever | hold | sust kPa | plan mm | exec mm | grasp mm | run |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o, p, c, e, n, su, ev, ho, ss, pk, pl, exm, gr, w in rs:
        tbl.append(f"| {o} | {p} | {c} | {e} | {su}/{n} | {ev} | {ho} | {ss/1000:.1f} | "
                   f"{f'{pl:.1f}' if pl else '—'} | {f'{exm:.1f}' if exm else '—'} | "
                   f"{f'{gr:.1f}' if gr else '—'} | {w} |")
    doc = open(DOC).read()
    i = doc.index("| object | policy |")
    j = doc.index("\n## The noise floor")
    open(DOC, "w").write(doc[:i] + "\n".join(tbl) + "\n" + doc[j:])
    stamps = [r[13] for r in rs]
    dup = {s for s in stamps if stamps.count(s) > 1}
    print(f"{DOC}: {len(rs)} evals written" + (f"  DUPLICATES: {dup}" if dup else "  (no duplicates)"))


if __name__ == "__main__":
    main()
