#! /bin/bash
# Sequential demo-collection campaign over a plan file, using the UNCHANGED recipe in
# collect_demo_template.sh (table-z 0.0138, --scene-dr-every 1, --record-video 25, soft sim).
# Per-object episode count and sub-env count come from logs/campaign/plan.json.
# Writes logs/campaign/status.json after every object so the dashboard can read live progress.
set -u; cd "$(dirname "$0")/../../.."
PLAN=${PLAN:-logs/campaign/plan.json}
STATUS=${STATUS:-logs/campaign/status.json}   # overridable so a follow-up campaign does not clobber the first
mkdir -p logs/campaign

python3 - "$PLAN" "$STATUS" <<'PY'
import json, sys, time
plan = json.load(open(sys.argv[1]))
json.dump({"started": time.time(), "total": len(plan), "total_demos": sum(j["n"] for j in plan),
           "done": [], "current": None, "failed": []}, open(sys.argv[2], "w"), indent=1)
PY

N=$(python3 -c "import json;print(len(json.load(open('$PLAN'))))")
for i in $(seq 0 $((N-1))); do
  read -r OBJ NEP NENV <<<"$(python3 -c "
import json; j=json.load(open('$PLAN'))[$i]; print(j['obj'], j['n'], j['envs'])")"
  python3 - "$STATUS" "$OBJ" "$NEP" "$NENV" "$i" <<'PY'
import json, sys, time
s = json.load(open(sys.argv[1]))
s["current"] = {"obj": sys.argv[2], "n": int(sys.argv[3]), "envs": int(sys.argv[4]),
                "idx": int(sys.argv[5]), "started": time.time()}
json.dump(s, open(sys.argv[1], "w"), indent=1)
PY
  echo "### [$((i+1))/$N] $OBJ  n=$NEP envs=$NENV  $(date '+%F %T')"
  t0=$(date +%s)
  OBJ="$OBJ" N_EPISODES="$NEP" N_ENVS="$NENV" SEED=0 EXTRA_ARGS="${EXTRA_ARGS:-}" \
    bash gentle_manip/scripts/final/collect_demo_template.sh > "logs/campaign/${OBJ}.log" 2>&1
  rc=$?; t1=$(date +%s)
  python3 - "$STATUS" "$OBJ" "$rc" "$((t1-t0))" <<'PY'
import json, sys, glob, os, yaml
s = json.load(open(sys.argv[1])); obj, rc, dt = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rec = {"obj": obj, "rc": rc, "sec": dt, "saved": None, "success_rate": None}
runs = sorted(glob.glob(f"dataset/demos/single_lift_{obj}_soft/*/stats.yaml"), key=os.path.getmtime)
if runs:
    try:
        st = yaml.safe_load(open(runs[-1]))
        rec["saved"] = st.get("episodes_saved"); rec["success_rate"] = st.get("success_rate")
        rec["sub_yield"] = st.get("sub_yield_frac")
    except Exception: pass
s["done"].append(rec)
if rc != 0: s["failed"].append(obj)
s["current"] = None
json.dump(s, open(sys.argv[1], "w"), indent=1)
PY
  echo "###   rc=$rc  $(( (t1-t0)/60 )) min"
done
echo "### CAMPAIGN COMPLETE $(date '+%F %T')"
