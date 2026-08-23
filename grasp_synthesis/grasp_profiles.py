"""The named grasp-synthesis objectives — ONE module, imported by BOTH the collector and the
benchmark.

Until this file existed, the profiles lived only in the benchmark: the collector had no
--grasp-execute-offset flag at all, so a "v5 collection" would have synthesized with the OLD
mis-scored objective while the benchmark measured the new one — the fourth instance this project
of a knob meaning different things in the two programs (shelf gate, shelf resolver, --traj v4
schedule, now the objective itself). A profile name must resolve to the same kwargs everywhere.
"""
from __future__ import annotations

GRASP_PROFILES = {
    # historical benchmark: strict flush-alignment metric, no diversity, pitch seeds at 0
    "strict": {},
    # what collect_demos_synth_v3.py ACTUALLY runs (its argparse defaults, :770-791)
    "collector_v3": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0,
                         jitter_pos=0.003, pitch_seed_deg=25.0),
    # v4: keep the collector's diversity, but re-enable the anti-pinch peak term and add the
    # geometry priors. Weights are tuned in Iteration 3; these are the starting points.
    "v4": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
               pitch_seed_deg=25.0, w_peak=None, w_com=0.0, w_tilt=0.0, w_occ=0.0, area_min=0.0),
    # v4fix: the OPERATING-POINT correction. The executor closes 4.5mm tighter than the width the
    # objective scores (2.5mm base squeeze + 2mm firm), which is ~10x more stress — measured 54.8kPa
    # executed vs 5.4kPa scored, i.e. 1.37x the mushroom's yield on 100% of episodes.
    # execute_offset scores each candidate at the width actually commanded.
    # area_min is NOT optional here: correcting the operating point alone makes the optimizer grasp
    # something too thin to compress (contact area 66 -> 12 mm2, align 0.98 -> 0.19), and w_peak
    # alone does not stop it. Offline, all three together give 21.0kPa executed (0.53x yield) with
    # 41mm2 of flush contact — a 2.6x reduction. THIS PROFILE IS THE ONE TO BENCHMARK NEXT.
    # area_min 4e-5 chosen by an offline sweep over 4 poses, not guessed. It DOMINATES 3e-5 (more
    # contact area, 59 vs 54 mm2, at slightly LOWER executed stress) and sits at the start of a
    # plateau -- 5e-5 gives an identical result. Every value 0..5e-5 stayed holdable 4/4, so the
    # floor is not bought with grip. Sweep: 0 -> 9mm2 pinch @5.1kPa; 1-2e-5 -> 21mm2 @10.9kPa;
    # 3e-5 -> 54mm2 @15.5kPa; 4-5e-5 -> 59mm2 @15.0kPa. Historical: 66mm2 @54.8kPa.
    "v4fix": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
                  pitch_seed_deg=25.0, w_peak=0.3, area_min=4e-5,
                  execute_offset=0.0045),
    # v5 = v4fix + the camera-azimuth bound. Occlusion was one of the THREE original v4 defects and
    # the only one still standing: 38% of v4fix episodes hide >50% of the object, 24% hide >80%,
    # and the soft w_occ penalty is provably inert (identical output for weights 0..20000 — the
    # occlusion-reducing candidates sit at a flat infeasibility floor where a weight has no
    # gradient). The bound is structural instead: closing-axis azimuth to the camera ray capped
    # (roll_max pattern), applied as a shaped penalty at EVERY ladder rung + seed fan centred on
    # the perpendicular direction. 45 deg is the starting value — the 45-vs-60 sweep on the
    # *_grasp_eval_pcd experiment (ground-truth occ_pcd_lift) picks the final one.
    "v5": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
               pitch_seed_deg=25.0, w_peak=0.3, area_min=4e-5,
               execute_offset=0.0045, cam_azimuth_max_deg=45.0),
    # v5c — the COLLECTION profile (2026-08-22). execute_offset is RETIRED here: it removes the
    # historical 4.5mm blind over-squeeze, and the FEM's holdable margin does not survive MPM at
    # honest widths (collector bisect on true-size meshes: offset alone 8/8 -> 1/8; +4g hold
    # margin still 3/8; v5-minus-offset 6/8). w_peak/area_min are dropped with it — they were
    # tuned AT the offset operating point. What remains measured-safe on true-size meshes:
    # the collector_v3 diversity defaults + the azimuth occlusion bound (8/8 on the hardest
    # scene-DR batch). Re-admitting the offset requires calibrating the FEM's mu/margin against
    # MPM first — the '_current_mesh scale' fix (43b388a) explains why the benchmark never saw
    # this: it planned on nominal-size meshes for every scaled scene.
    "v5c": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
                pitch_seed_deg=25.0, cam_azimuth_max_deg=45.0),
}
