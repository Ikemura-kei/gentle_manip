"""[tool] Prepped view(s) -> raw TripoSG mesh(es).

Used by: scripts/mesh_from_photos_generate.sbatch
Status: active

    python generate.py --object mushroom1 --view back --seed 0

Loads the model once and sweeps every (view, seed) pair, writing one raw.glb per
pair plus a run-level manifest. Doc section 4 asks for three seeds per object kept
side by side: generative output varies and a research pipeline should see the
spread rather than trust one sample.

NOTE ON VIEWS: TripoSG is a SINGLE-IMAGE model -- it cannot fuse views the way
the banned EU-excluded generator run_multi_image does. Passing several views here generates several
INDEPENDENT meshes to compare, it does not give one mesh conditioned on all of them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[2]
TRIPOSG = REPO / "third_party" / "TripoSG"
sys.path.insert(0, str(TRIPOSG))
sys.path.insert(0, str(TRIPOSG / "scripts"))
# APPEND (not insert) the diso stub: a real diso in site-packages must win. See
# scripts/mesh_from_photos/shims/diso/__init__.py for why the stub exists at all.
sys.path.append(str(Path(__file__).resolve().parent / "shims"))

from huggingface_hub import snapshot_download  # noqa: E402
from image_process import prepare_image  # noqa: E402  (upstream preprocessing)
from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--views", nargs="+", default=["back"],
                    help="prepped view stems to generate from (each is independent)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--flash-decoder", action="store_true",
                    help="use TripoSG's DiffDMC decoder. OFF by default: it needs `diso`, "
                         "an sdist-only CUDA extension with no aarch64 wheel. The default "
                         "path is hierarchical_extract_geometry + skimage marching cubes.")
    ap.add_argument("--root", default=str(REPO / "obj_meshes"))
    args = ap.parse_args()

    obj_dir = Path(args.root) / args.object
    prepped = obj_dir / "prepped"
    runs = obj_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    missing = [v for v in args.views if not (prepped / f"{v}.png").exists()]
    if missing:
        raise SystemExit(f"missing prepped views {missing} in {prepped}; run prep_images.py first")

    weights = Path(snapshot_download(repo_id="VAST-AI/TripoSG"))
    print(f"[weights] {weights}", flush=True)

    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(str(weights)).to("cuda", torch.float16)
    print("[pipeline] loaded", flush=True)

    manifest = {
        "object": args.object,
        "model": "VAST-AI/TripoSG",
        "model_licence": "MIT (code and weights)",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "decoder": "flash/DiffDMC" if args.flash_decoder else "hierarchical/skimage-marching-cubes",
        "note": "single-image model: one independent mesh per (view, seed)",
        "runs": [],
    }

    for view in args.views:
        # Upstream preprocessing: composite RGBA over white, crop to the alpha bbox,
        # pad to square. rmbg_net is never invoked because our PNG already has a
        # valid alpha channel (see TripoSG scripts/image_process.py::load_image).
        img = prepare_image(str(prepped / f"{view}.png"),
                            bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
        for seed in args.seeds:
            tag = f"{view}_seed{seed}"
            t0 = time.time()
            try:
                out = pipe(
                    image=img,
                    generator=torch.Generator(device=pipe.device).manual_seed(seed),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    use_flash_decoder=args.flash_decoder,
                ).samples[0]
            except Exception as e:  # noqa: BLE001
                # One failed extraction must not cost the whole sweep.
                print(f"[gen] {tag}: FAILED ({type(e).__name__}: {e})", flush=True)
                manifest["runs"].append({"tag": tag, "view": view, "seed": seed,
                                         "error": f"{type(e).__name__}: {e}"})
                continue
            dt = time.time() - t0

            if out[0] is None:
                print(f"[gen] {tag}: FAILED (extractor returned no geometry)", flush=True)
                manifest["runs"].append({"tag": tag, "view": view, "seed": seed,
                                         "error": "extractor returned no geometry"})
                continue

            verts, faces = out[0].astype(np.float32), np.ascontiguousarray(out[1])
            mesh = trimesh.Trimesh(verts, faces)
            d = runs / tag
            d.mkdir(parents=True, exist_ok=True)
            mesh.export(d / "raw.glb")

            rec = {
                "tag": tag, "view": view, "seed": seed,
                "seconds": round(dt, 1),
                "raw_vertices": int(len(mesh.vertices)),
                "raw_faces": int(len(mesh.faces)),
                "raw_extent": [round(float(x), 5) for x in mesh.extents],
                "raw_watertight": bool(mesh.is_watertight),
                "raw_components": int(len(mesh.split(only_watertight=False))),
            }
            manifest["runs"].append(rec)
            print(f"[gen] {tag}: {rec['raw_faces']} faces, {dt:.1f}s, "
                  f"watertight={rec['raw_watertight']}, comps={rec['raw_components']}", flush=True)

    (runs / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {len(manifest['runs'])} meshes -> {runs}")


if __name__ == "__main__":
    main()
