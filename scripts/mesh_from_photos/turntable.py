"""[tool] Rotating turntable render of a mesh, next to the reference photo.

Used by: scripts/mesh_from_photos_generate.sbatch (and standalone)
Status: active

    python turntable.py --mesh obj_meshes/mushroom1/runs/back_seed0/clean.obj \
                        --reference obj_meshes/mushroom1/prepped/back.png \
                        --out obj_meshes/mushroom1/turntable.mp4

Self-contained software rasteriser (numpy z-buffer, Phong-interpolated normals).
Deliberately no OpenGL/EGL/OSMesa: Arrhenius compute nodes are headless and
pyrender/open3d have no working aarch64 GPU-less path here. ~90 frames at 720px
takes well under a minute for a 12k-face mesh.

The reference panel matters: doc section 7 warns these models apply a symmetry
prior and invent the unphotographed underside, so the mesh must be eyeballed
against the actual photo rather than trusted.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

UP_AXIS = {"x": 0, "y": 1, "z": 2}


def rot(axis: int, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def background(h: int, w: int) -> np.ndarray:
    top, bot = np.array([0.10, 0.11, 0.14]), np.array([0.20, 0.21, 0.25])
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return (top + (bot - top) * ramp).astype(np.float32) * np.ones((1, w, 1), np.float32)


def render(verts: np.ndarray, faces: np.ndarray, vnorm: np.ndarray,
           size: int, dist: float, focal: float) -> np.ndarray:
    """Perspective z-buffer rasteriser with per-pixel normal interpolation."""
    h = w = size
    img = background(h, w)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    depth = dist - verts[:, 2]
    depth = np.maximum(depth, 1e-4)
    sx = w * 0.5 + focal * verts[:, 0] / depth
    sy = h * 0.5 - focal * verts[:, 1] / depth
    screen = np.stack([sx, sy], 1)

    tri = screen[faces]                       # (m,3,2)
    # Signed area in screen space; y is flipped, so front faces are negative.
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    area = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    keep = area < -1e-9
    idx = np.nonzero(keep)[0]

    td = depth[faces]                         # (m,3)
    tn = vnorm[faces]                         # (m,3,3)

    # Two-light rig: warm key from upper-left-front, cool fill from lower-right.
    key_dir = np.array([-0.5, 0.7, 0.9], np.float32); key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([0.8, -0.35, 0.5], np.float32); fill_dir /= np.linalg.norm(fill_dir)
    albedo = np.array([0.86, 0.78, 0.66], np.float32)   # mushroom-ish off-white
    key_col = np.array([1.00, 0.96, 0.90], np.float32)
    fill_col = np.array([0.45, 0.52, 0.68], np.float32)

    xs_all = np.arange(w, dtype=np.float32)
    ys_all = np.arange(h, dtype=np.float32)

    for f in idx:
        p = tri[f]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())) + 1, w)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue

        px = xs_all[x0:x1][None, :]
        py = ys_all[y0:y1][:, None]
        ax, ay = p[0]; bx, by = p[1]; cx, cy = p[2]
        inv = 1.0 / area[f]
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * td[f, 0] + w1 * td[f, 1] + w2 * td[f, 2]
        sub = zbuf[y0:y1, x0:x1]
        hit = inside & (z < sub)
        if not hit.any():
            continue
        sub[hit] = z[hit]

        n = (w0[..., None] * tn[f, 0] + w1[..., None] * tn[f, 1] + w2[..., None] * tn[f, 2])
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)

        lam_k = np.clip(n @ key_dir, 0, 1)[..., None]
        lam_f = np.clip(n @ fill_dir, 0, 1)[..., None]
        # Rim term: grazing angles to the viewer, keeps the silhouette readable.
        rim = np.clip(1.0 - np.clip(n[..., 2], 0, 1), 0, 1)[..., None] ** 3
        col = albedo * (0.09 + 1.00 * lam_k * key_col + 0.28 * lam_f * fill_col) + 0.26 * rim
        np.copyto(img[y0:y1, x0:x1], np.clip(col, 0, 1), where=hit[..., None])

    return img


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def ref_panel(path: Path, size: int) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = to_u8(background(size, size))
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    a = np.asarray(im).astype(np.float32) / 255.0
    region = canvas[y:y + im.height, x:x + im.width].astype(np.float32) / 255.0
    blend = a[..., :3] * a[..., 3:4] + region * (1 - a[..., 3:4])
    canvas[y:y + im.height, x:x + im.width] = (blend * 255).astype(np.uint8)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="output .mp4 (a .gif is written alongside)")
    ap.add_argument("--reference", default=None, help="photo to pin beside the render")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--up", choices=list(UP_AXIS), default="y")
    ap.add_argument("--elevation", type=float, default=14.0, help="camera tilt, degrees")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)

    v = v - v.mean(0)
    v = v / np.linalg.norm(v, axis=1).max()

    up = UP_AXIS[args.up]
    # Bring the chosen up-axis to +Y for the camera, so the object spins upright.
    if up == 0:
        pre = rot(2, np.pi / 2)
    elif up == 2:
        pre = rot(0, -np.pi / 2)
    else:
        pre = np.eye(3, dtype=np.float32)
    tilt = rot(0, np.deg2rad(args.elevation))

    dist, focal = 3.2, args.size * 1.28
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = ref_panel(Path(args.reference), args.size) if args.reference else None

    import imageio.v2 as imageio
    frames = []
    for i in range(args.frames):
        M = tilt @ rot(1, 2 * np.pi * i / args.frames) @ pre
        img = to_u8(render(v @ M.T, f, vn @ M.T, args.size, dist, focal))
        if ref is not None:
            sep = np.full((args.size, 2, 3), 40, dtype=np.uint8)
            img = np.concatenate([ref, sep, img], axis=1)
        frames.append(img)
        if (i + 1) % 15 == 0:
            print(f"  frame {i + 1}/{args.frames}", flush=True)

    imageio.mimsave(out, frames, fps=args.fps, quality=8, macro_block_size=1)
    gif = out.with_suffix(".gif")
    imageio.mimsave(gif, frames[::2], duration=1000 / (args.fps / 2), loop=0)
    imageio.imwrite(out.with_name(out.stem + "_still.png"), frames[0])
    print(f"[turntable] {out}  ({args.frames} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    print(f"[turntable] {gif}")


if __name__ == "__main__":
    main()
