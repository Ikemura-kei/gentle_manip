"""Print the state_N.pt whose epoch is closest to the lowest-val-loss epoch parsed
from a DPPO pretrain log. Falls back to the highest-N checkpoint. Args: <run_dir> <train_log>"""
import sys, glob, os, re
rd = sys.argv[1]
log = sys.argv[2] if len(sys.argv) > 2 else ""
cks = {int(re.findall(r'(\d+)', os.path.basename(p))[0]): p
       for p in glob.glob(os.path.join(rd, "checkpoint", "state_*.pt"))}
best_ep, best_v = None, 1e9
if log and os.path.exists(log):
    for m in re.finditer(r'-\s*(\d+):\s*train loss\s+[\d.]+\s*\|\s*val loss\s+([\d.]+)', open(log).read()):
        ep, v = int(m.group(1)), float(m.group(2))
        if v < best_v:
            best_v, best_ep = v, ep
if best_ep is None or not cks:
    print(cks[max(cks)] if cks else "")
    sys.exit()
pick = min(cks, key=lambda e: abs(e - best_ep))
sys.stderr.write(f"[pick] best val {best_v:.4f} @ ep{best_ep} -> state_{pick}.pt\n")
print(cks[pick])
