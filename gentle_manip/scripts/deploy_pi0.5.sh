#! /bin/bash
# pi0.5 VLA deploy (local). Runs in OPENPI's OWN 3.11 venv (Python 3.8 envs/dp3 cannot host
# openpi), with the repo on PYTHONPATH so gentle_manip stays out of openpi's resolver.
# One-time setup done 2026-09-03: openpi cloned at pin 215abfb + `uv sync --no-install-package
# evdev` (evdev is teleop-only and fails to build against this box's kernel headers), then
# `uv pip install pyrealsense2==2.54.2.5684 xArm-Python-SDK opencv-python scipy` into its venv.
#
# --repo-id gm/real7_ext is REQUIRED: the norm stats live at assets/gm/real7_ext/ (the training
# repo id), not the config default physical-intelligence/libero.
# Prompt must be one of the 6 trained phrasings; a "cube" was never in the REAL training set
# (mushroom, grape, tomato, padron_pepper, cherry_tomato, strawberry, tofu) — for anything
# unnamed use the generic "pick up the object from table gently".

cd "$(dirname "$0")/../.."
PYTHONPATH=$PWD third_party/openpi/.venv/bin/python \
  -m gentle_manip.scripts.deploy_real_pi05 \
  --checkpoint downloaded_runs/pi05_real7_ext/29999 \
  --repo-id gm/real7_ext \
  --prompt "pick up the object from table gently" \
  --max-pos-step-m 0.0065 \
  --record dataset/real_deploy/pi05_real7_e29999
