#!/usr/bin/env python3
"""Build a 15-rollout montage mp4 for one (model, category): up to 15 random eval
clips, 2x speed, small, each separated by a short black interval. Idempotent.

usage: gen8_montage.py <gen_eval_dir> <category> <model_label> <out_dir>
"""
import sys, os, glob, random, subprocess, hashlib
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
N     = 15
W     = 448          # native clips 640x480; keep 4:3. Higher res than the first pass (300).
H     = 336
FPS   = 14
CRF   = 28           # 448px keeps it well above the first pass; fits 16MB with OOD
GAP   = 0.3
CLIPCAP = 40.0       # no effective cap: _trim_eval_clips already bounds success clips to fss+16 (keeps full recovery); do NOT re-cut here


def main():
    gen_eval_dir, cat, model, out_dir = sys.argv[1:5]
    clips = sorted(glob.glob(os.path.join(gen_eval_dir, cat, "render", "*.mp4")))
    if len(clips) < 3:
        print(f"  [{model}/{cat}] {len(clips)} clips -> skip")
        return
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"montage_{model}_{cat}.mp4")

    newest_in = max(os.path.getmtime(c) for c in clips)
    if os.path.exists(out) and os.path.getmtime(out) >= newest_in:
        print(f"  [{model}/{cat}] up-to-date ({len(clips)} clips)")
        return

    seed = int(hashlib.md5(f"{model}:{cat}".encode()).hexdigest()[:8], 16)
    pick = random.Random(seed).sample(clips, min(N, len(clips)))
    n = len(pick)
    nblk = n - 1

    inputs = []
    for c in pick:
        inputs += ["-i", c]

    fp = []
    for i in range(n):
        fp.append(
            f"[{i}:v]trim=end={CLIPCAP},setpts=PTS-STARTPTS,setpts=0.5*PTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={FPS},setsar=1,format=yuv420p[v{i}]"
        )
    filt = ";".join(fp)
    if nblk:
        filt += f";color=c=black:s={W}x{H}:r={FPS}:d={GAP},format=yuv420p[blkbase]"
        filt += ";[blkbase]split=" + str(nblk) + "".join(f"[b{j}]" for j in range(nblk))
    seq = ""
    for i in range(n):
        seq += f"[v{i}]"
        if i != n - 1:
            seq += f"[b{i}]"
    filt += f";{seq}concat=n={n + nblk}:v=1:a=0[out]"

    cmd = [FFMPEG, "-y", *inputs,
           "-filter_complex", filt, "-map", "[out]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{model}/{cat}] FFMPEG FAIL\n{r.stderr[-1500:]}")
        sys.exit(1)
    print(f"  [{model}/{cat}] -> {os.path.getsize(out)/1024:.0f} KB "
          f"({n} rollouts, {len(clips)} clips avail)")


if __name__ == "__main__":
    main()
