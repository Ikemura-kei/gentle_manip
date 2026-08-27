# Introduction drafts — archive

Dated snapshots of `intro_icra.tex`. Restore with:
`cp archive/<file> ../intro_icra.tex && bash ../build.sh`

| version | date | what it was |
|---|---|---|
| `intro_icra_v1_2026-08-26.tex` | 2026-08-26 | First ICRA-style draft (938 w). Three-strand related work (compliant / tactile / stress-aware), generous non-hostile framing, four contributions. **Claimed "coverage over count" as our own finding** — superseded once the literature sweep found Lin et al. ICLR 2025 had already established it. |
| `intro_icra_v2_2026-08-26.tex` | 2026-08-26 | v1 + literature-sweep citations (Lin et al. data scaling, MimicGen, DefGraspSim, sim-real co-training); "coverage over count" demoted from our claim to a cited result. 1126 w, ~2 columns. Superseded by v3 (DextrAH-G register + application-paper contribution ordering). |
| `intro_icra_v3_2026-08-26.tex` | 2026-08-26 | DextrAH-G register (verbatim-calibrated), inline numbered contributions, application-paper ordering. 647 w. **Built ¶1 on stiffness as the central difficulty — withdrawn in v4**: the policy observes only a point cloud so cannot be stiffness-adaptive, and our own failure data implicates pose/size, not stiffness. Also lost the Fig. 1 reference. |
