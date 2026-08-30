"""Trim each eval render clip to end shortly after its success moment (variable
length; no frozen carry-down tail). Failures are left untouched.
Args: <eval_dir>  [act_steps=4]  [tail_frames=16]
Reads <eval_dir>/episodes.csv (first_success_step in POLICY steps) and rewrites
<eval_dir>/render/batchNN_envM.mp4 in place."""
import sys, os, csv, glob, re
import numpy as np
try:
    import imageio.v2 as iio
except Exception:
    import imageio as iio

ed = sys.argv[1]
act = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tail = int(sys.argv[3]) if len(sys.argv) > 3 else 16
rows = {}
with open(os.path.join(ed, "episodes.csv")) as f:
    for r in csv.DictReader(f):
        rows[(int(r["batch"]), int(r["env"]))] = r
n_trim = 0
for p in glob.glob(os.path.join(ed, "render", "batch*_env*.mp4")):
    m = re.match(r"batch(\d+)_env(\d+)", os.path.basename(p))
    k = (int(m.group(1)), int(m.group(2)))
    r = rows.get(k)
    if not r or int(r.get("success", 0)) != 1:
        continue
    fss = int(r.get("first_success_step", -1))
    if fss < 0:
        continue
    cut = fss * act + tail
    rd = iio.get_reader(p)
    nf = rd.count_frames()
    if cut >= nf - 2:
        continue
    frames = [np.asarray(rd.get_data(i)) for i in range(min(cut, nf))]
    rd.close()
    iio.mimwrite(p, frames, fps=30, quality=8, macro_block_size=1)
    n_trim += 1
print(f"[trim] cut {n_trim} success clips to first_success_step+{tail} frames")
