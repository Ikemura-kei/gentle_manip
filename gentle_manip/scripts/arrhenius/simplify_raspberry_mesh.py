"""Option C: fatten drupelet necks / smooth lobes on the raspberry meshes, size-preserving.
Pipeline per mesh: voxelize (0.5mm) -> binary close+dilate (fills necks, fattens thin features)
-> fill holes -> marching cubes -> taubin smooth -> decimate ~4k faces -> rescale to the ORIGINAL
bbox (true-to-life size kept, per user: policy must learn to grasp genuinely small fruit).
Metric: 'thinness' = max erosion depth the solid survives (voxels) — higher = fatter necks.
"""
import numpy as np, trimesh, sys
from scipy import ndimage
from skimage import measure
import fast_simplification

VOX = 0.0005
for name in ["raspberry1","raspberry2","raspberry3","raspberry4","raspberry5"]:
    m = trimesh.load(f"gentle_manip/assets/objects/{name}.obj", force='mesh')
    ext0 = m.bounding_box.extents.copy(); c0 = m.bounding_box.centroid.copy()
    vg = m.voxelized(VOX).fill()
    M = np.pad(vg.matrix, 6)                          # headroom so dilation never touches the array edge
                                                      # (marching cubes leaves OPEN faces at clipped borders)
    def thin(mat):
        d, n = mat.copy(), 0
        while d.any(): d = ndimage.binary_erosion(d); n += 1
        return n
    t_before = thin(M)
    M2 = ndimage.binary_closing(M, iterations=2)          # fill neck gaps
    M2 = ndimage.binary_dilation(M2, iterations=2)        # fatten ~1.0mm/side
    M2 = ndimage.binary_fill_holes(M2)
    t_after = thin(M2)
    verts, faces, _, _ = measure.marching_cubes(M2.astype(np.float32), 0.5)
    ms = trimesh.Trimesh(verts * VOX, faces[:, ::-1], process=True)
    print(f"  {name}: post-MC watertight={ms.is_watertight} faces={len(ms.faces)}")
    trimesh.smoothing.filter_taubin(ms, lamb=0.5, nu=-0.53, iterations=15)
    print(f"  {name}: post-smooth watertight={ms.is_watertight}")
    if len(ms.faces) > 6000:
        for agg in (7, 5, 3, 1):                      # gentler collapses keep it manifold (planner's trick)
            v, f = fast_simplification.simplify(np.asarray(ms.vertices,np.float64),
                                                np.asarray(ms.faces,np.int64), target_count=6000, agg=agg)
            cand = trimesh.Trimesh(v, f, process=True)
            cand.fill_holes()
            if cand.is_watertight:
                ms = cand; break
        # none watertight -> keep the undecimated marching-cubes mesh (fine downstream)
    # rescale each axis back to the ORIGINAL bbox and recentre
    ext1 = ms.bounding_box.extents
    ms.apply_scale(ext0 / ext1)
    ms.apply_translation(c0 - ms.bounding_box.centroid)
    assert ms.is_watertight, f"{name}: not watertight after pipeline"
    out = f"gentle_manip/assets/objects/{name}_smooth.obj"
    ms.export(out)
    e = ms.bounding_box.extents
    print(f"{name}: faces {len(m.faces)}->{len(ms.faces)}  bbox {1e3*ext0.round(4)}->{1e3*e.round(4)}mm  "
          f"thinness {t_before}->{t_after} erosion-voxels ({VOX*1e3:.1f}mm each)  watertight={ms.is_watertight}")
