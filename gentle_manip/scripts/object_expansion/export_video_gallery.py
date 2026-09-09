"""Build a compact JSON payload of ONE compressed, base64-embeddable grasp-demo clip per
fully-collected category, for a standalone video-gallery Artifact (a static page can't reach
the cluster filesystem or an assets store not granted on this account -- base64 data URIs are
the only path, so clips must be small: ~200-300KB each to keep ~25 categories under the
Artifact's 16MB page cap).

Usage (must run in envs/sim_arrhenius via sbatch -- imageio_ffmpeg/cv2 aren't on the login node):
    uv run --project envs/sim_arrhenius python -m gentle_manip.scripts.object_expansion.export_video_gallery \
        --out /tmp/video_gallery.json --max-kb 260
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def compress_clip(src: Path, dst: Path, *, width: int = 360, crf: int = 30, fps: int = 20) -> bool:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vf", f"scale={width}:-2,fps={fps}",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf),
        "-an", "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def find_best_clip(run_dir: Path, max_kb: int, tmp_dir: Path, name: str) -> dict | None:
    vdir = run_dir / "videos"
    if not vdir.exists():
        return None
    mp4s = sorted(vdir.glob("*_success.mp4"))
    if not mp4s:
        return None
    src = mp4s[0]
    # Try progressively more aggressive compression until under budget.
    for width, crf in ((360, 28), (300, 32), (240, 34), (200, 36)):
        dst = tmp_dir / f"{name}.mp4"
        try:
            ok = compress_clip(src, dst, width=width, crf=crf)
        except Exception:
            ok = False
        if not ok:
            continue
        size_kb = dst.stat().st_size / 1024
        if size_kb <= max_kb:
            data = base64.b64encode(dst.read_bytes()).decode("ascii")
            dst.unlink(missing_ok=True)
            return {"b64": data, "size_kb": round(size_kb, 1), "src": src.name}
    dst = tmp_dir / f"{name}.mp4"
    if dst.exists():
        dst.unlink(missing_ok=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-kb", type=int, default=260)
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/video_gallery_tmp"))
    args = ap.parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    demos_root = REPO_ROOT / "dataset" / "demos"
    entries = []
    for task_dir in sorted(demos_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("single_lift_") or not task_dir.name.endswith("_soft"):
            continue
        obj_name = task_dir.name[len("single_lift_"):-len("_soft")]
        run_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()], reverse=True)
        if not run_dirs:
            continue
        run_dir = run_dirs[0]
        stats_path = run_dir / "stats.yaml"
        if not stats_path.exists():
            continue
        import yaml
        s = yaml.safe_load(stats_path.read_text())
        clip = find_best_clip(run_dir, args.max_kb, args.tmp_dir, obj_name)
        print(f"{obj_name}: {'ok ' + str(clip['size_kb']) + 'KB' if clip else 'FAILED (no clip under budget)'}", flush=True)
        if clip is None:
            continue
        entries.append({
            "name": obj_name,
            "success_rate": s.get("success_rate"),
            "ever_success_rate": s.get("ever_success_rate"),
            "episodes_saved": s.get("episodes_saved"),
            "clip_b64": clip["b64"],
            "clip_size_kb": clip["size_kb"],
        })

    args.out.write_text(json.dumps({"clips": entries}, separators=(",", ":")))
    total_mb = args.out.stat().st_size / 1e6
    print(f"\n{len(entries)} clips exported -> {args.out} ({total_mb:.2f} MB)")


if __name__ == "__main__":
    main()
