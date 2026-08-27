# Smoke-collection history (auto-generated)

Regenerate with `uv run --project envs/sim python -m gentle_manip.scripts.smoke_table`.
Every row pairs a demonstrator success rate with the synthesis recipe that produced it,
so a number can always be traced back to its configuration. Newest first.

`area`/`wmax`/`yaw`/`sq` = the auto grasp params (`auto` = derived from the object);
`esc` = budget-escalation retries; `az` = camera-azimuth penalty bound (degrees);
`drop` = episodes discarded because synthesis failed and fell back to a crushing grasp.

| date | object | run | eps | attempts | success | drop | synthesis recipe |
|---|---|---|---|---|---|---|---|
| 2026-08-27 16:08 | tomato | `26-08-27-xcv` | 8 | 9 | **89%** | 0 | area=auto, wmax=auto, yaw=69, sq=6.0mm, esc=2, az=60 |
| 2026-08-27 16:01 | raspberry_stable | `26-08-27-sny` | 8 | 8 | **100%** | 0 | area=auto, wmax=auto, sq=2.1mm, esc=2, az=60 |
| 2026-08-27 15:49 | cherry_tomato | `26-08-27-nec` | 8 | 9 | **89%** | 0 | area=auto, wmax=auto, sq=3.7mm, esc=2, az=60 |
| 2026-08-27 14:31 | cherry_tomato | `26-08-27-rvc` | 8 | 10 | **80%** | 0 | area=auto, wmax=auto, sq=3.2mm, esc=2, az=60 |
| 2026-08-27 14:29 | cherry_tomato | `26-08-27-jnu` | 8 | 10 | **80%** | 0 | area=auto, wmax=auto, sq=3.2mm, esc=2, az=60 |
| 2026-08-27 14:07 | tomato | `26-08-27-ucc` | 8 | 9 | **89%** | 0 | area=auto, wmax=auto, sq=6.0mm, esc=2, az=60 |
| 2026-08-27 14:05 | tomato | `26-08-27-fxq` | 8 | 14 | **57%** | 0 | area=auto, wmax=auto, sq=6.0mm, esc=2, az=60 |
| 2026-08-27 13:57 | banana_chunk | `26-08-27-ofx` | 8 | 12 | **67%** | 0 | area=auto, wmax=auto, sq=3.1mm, esc=2, az=60 |
| 2026-08-27 13:19 | mushroom | `26-08-27-juf` | 8 | 8 | **100%** | 0 | area=auto, wmax=auto, sq=4.8mm, esc=2, az=60 |
| 2026-08-27 13:15 | tomato | `26-08-27-mul` | 8 | 14 | **57%** | 0 | area=auto, wmax=auto, sq=6.0mm, esc=2, az=60 |
| 2026-08-27 13:12 | mushroom | `26-08-27-fgd` | 8 | 8 | **100%** | 0 | area=auto, wmax=auto, sq=4.8mm, esc=2, az=60 |
| 2026-08-27 13:07 | pasta_bundle | `26-08-27-fod` | 8 | 18 | **44%** | 0 | area=auto, wmax=auto, sq=3.8mm, esc=2, az=60 |
| 2026-08-27 13:06 | tomato | `26-08-27-jec` | 8 | 14 | **57%** | 0 | area=auto, wmax=auto, sq=6.0mm, esc=2, az=60 |
| 2026-08-27 12:58 | pasta_bundle | `26-08-27-gsq` | 8 | 25 | **32%** | 0 | area=auto, wmax=auto, sq=3.8mm, esc=2, az=60 |
| 2026-08-27 12:45 | banana_chunk | `26-08-27-qrp` | 8 | 10 | **80%** | 0 | area=auto, wmax=auto, sq=3.1mm, esc=2, az=60 |
| 2026-08-27 12:37 | cherry_tomato | `26-08-27-rny` | 8 | 10 | **80%** | 0 | area=auto, wmax=auto, sq=3.2mm, esc=2, az=60 |
| 2026-08-27 10:43 | pasta_bundle | `26-08-27-jcw` | 8 | 16 | **50%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 10:42 | pasta_bundle | `26-08-27-wxk` | 8 | 22 | **36%** | 1 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 10:22 | banana_chunk | `26-08-27-qym` | 8 | 10 | **80%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 10:14 | tomato | `26-08-27-dbs` | 8 | 14 | **57%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 10:07 | cherry_tomato | `26-08-27-pqs` | 8 | 9 | **89%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:38 | banana | `26-08-27-aef` | 8 | 27 | **30%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:29 | banana | `26-08-27-lnu` | 8 | 12 | **67%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:25 | banana | `26-08-27-mjh` | 8 | 10 | **80%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:18 | banana | `26-08-27-uxv` | 8 | 9 | **89%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:13 | banana | `26-08-27-ajz` | 8 | 8 | **100%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:11 | banana | `26-08-27-xnq` | 8 | 8 | **100%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 08:06 | banana | `26-08-27-tzw` | 8 | 8 | **100%** | 0 | area=0, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 03:20 | tofu | `26-08-27-bqn` | 16 | 17 | **94%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 02:50 | raspberry_stable | `26-08-27-heo` | 16 | 16 | **100%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 02:13 | raspberry_stable | `26-08-27-trv` | 16 | 16 | **100%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 01:46 | strawberry | `26-08-27-jyd` | 16 | 18 | **89%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-27 01:27 | mushroom | `26-08-27-swh` | 16 | 16 | **100%** | 0 | area=auto, wmax=auto, sq=5.0mm, esc=2, az=60 |
| 2026-08-26 16:06 | banana | `26-08-26-hli` | 8 | 38 | **21%** | 0 | area=10.0, sq=5.0mm, esc=2, az=60 |
| 2026-08-26 15:28 | banana | `26-08-26-zuo` | 8 | 19 | **42%** | 0 | area=20.0, sq=5.0mm, esc=2, az=60 |
| 2026-08-26 14:59 | banana | `26-08-26-fsl` | 8 | 21 | **38%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 14:29 | banana | `26-08-26-qqw` | 8 | 21 | **38%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 14:08 | banana | `26-08-26-zbj` | 8 | 20 | **40%** | 0 | area=20.0, sq=5.0mm, MEDIAL, az=60 |
| 2026-08-26 10:30 | raspberry_stable | `26-08-26-fel` | 24 | 24 | **100%** | 0 | area=4.0, sq=5.0mm, az=60 |
| 2026-08-26 09:38 | banana | `26-08-26-ymh` | 8 | 20 | **40%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 01:40 | strawberry | `26-08-26-cgh` | 40 | 43 | **93%** | 0 | area=15.0, sq=5.0mm, az=60 |
| 2026-08-26 01:26 | banana | `26-08-26-abi` | 8 | 10 | **80%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 00:39 | banana | `26-08-26-bgt` | 8 | 8 | **100%** | 0 | area=0.0, sq=5.0mm |
| 2026-08-26 00:29 | banana | `26-08-26-fiv` | 8 | 8 | **100%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 00:11 | banana | `26-08-26-arp` | 8 | 8 | **100%** | 0 | area=20.0, sq=5.0mm, az=60 |
| 2026-08-26 00:07 | strawberry | `26-08-26-dhj` | 8 | 8 | **100%** | 0 | area=15.0, sq=5.0mm, az=60 |
| 2026-08-25 01:41 | mushroom | `26-08-25-vqg` | 16 | 16 | **100%** | 0 | area=15.0, sq=5.0mm, az=60 |
| 2026-08-25 01:30 | mushroom | `26-08-25-uix` | 16 | 18 | **89%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-25 00:59 | mushroom | `26-08-25-zrg` | 50 | 58 | **86%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-24 18:00 | mushroom | `26-08-24-cvz-filt` | 200 | 221 | **90%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-24 18:00 | mushroom | `26-08-24-cvz` | 200 | 221 | **90%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-24 16:03 | mushroom | `26-08-24-nxo` | 6 | 6 | **100%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-24 15:59 | mushroom | `26-08-24-rwl` | 6 | 6 | **100%** | 0 | area=None, sq=5.0mm, az=60 |
| 2026-08-24 15:56 | mushroom | `26-08-24-lkc` | 4 | 5 | **80%** | 0 | area=None, sq=5.0mm, az=45 |
| 2026-08-24 05:14 | mushroom | `26-08-24-ndr` | 500 | 550 | **91%** | 0 | area=None, sq=5.0mm, az=45 |
| 2026-08-24 00:00 | mushroom | `26-08-23-fiy` | 4 | 5 | **80%** | 0 | area=None, sq=5.0mm, az=45 |
| 2026-08-23 02:08 | mushroom | `26-08-22-ang` | 500 | 563 | **89%** | 0 | area=None, sq=0.0mm, az=45 |
| 2026-08-22 15:32 | mushroom | `26-08-22-fiw` | 500 | 570 | **88%** | 0 | area=None, sq=0.0mm, az=45 |
| 2026-08-22 08:02 | mushroom | `26-08-22-cah` | 500 | 569 | **88%** | 0 | area=None, sq=0.0mm, az=45 |
| 2026-08-20 09:56 | mushroom | `26-08-17-hwo-quat` | 650 | 686 | **95%** | 0 | area=None, sq=5.0mm |
| 2026-08-20 03:14 | mushroom | `26-08-20-xfv` | 300 | 320 | **94%** | 0 | area=None, sq=2.5mm |
| 2026-08-20 00:27 | mushroom | `26-08-19-isl-filt` | 300 | 323 | **93%** | 0 | area=None, sq=2.5mm |
| 2026-08-20 00:17 | mushroom | `26-08-19-isl` | 300 | 323 | **93%** | 0 | area=None, sq=2.5mm |
| 2026-08-19 08:46 | mushroom_rigid | `26-08-19-sie` | 650 | 720 | **90%** | 0 | area=None, sq=8.0mm |
| 2026-08-18 19:38 | mushroom_rigid | `26-08-18-cak` | 650 | 733 | **89%** | 0 | area=None, sq=4.0mm |
| 2026-08-18 16:34 | mushroom_rigid | `26-07-29-cho-rot6d` | 650 | 776 | **84%** | 0 | area=None |
| 2026-08-17 21:08 | mushroom | `26-08-17-hwo` | 650 | 686 | **95%** | 0 | area=None, sq=5.0mm |
| 2026-08-16 13:52 | mushroom | `26-08-14-rla` | 650 | 720 | **90%** | 0 | area=None |
| 2026-08-12 12:48 | mushroom | `26-08-12-acy` | 650 | 694 | **94%** | 0 | area=None |
| 2026-08-08 10:44 | mushroom | `26-08-08-jcw` | 650 | 697 | **93%** | 0 | area=None |
| 2026-08-07 11:27 | mushroom | `26-08-07-rct` | 60 | 71 | **85%** | 0 | area=None |
| 2026-07-29 02:38 | mushroom_rigid | `26-07-29-cho` | 650 | 776 | **84%** | 0 | area=None |
| 2026-07-28 16:21 | mushroom_rigid | `26-07-28-rpk` | 330 | 499 | **66%** | 0 | area=None |
| 2026-07-28 11:40 | mushroom_rigid | `26-07-28-pvm` | 330 | 502 | **66%** | 0 | area=None |
| 2026-07-28 02:17 | mushroom_rigid | `26-07-28-jud` | 330 | 525 | **63%** | 0 | area=None |
| 2026-07-27 19:19 | mushroom_rigid | `26-07-27-hfx` | 330 | 497 | **66%** | 0 | area=None |
| 2026-07-26 19:18 | mushroom_rigid | `26-07-26-sma` | 240 | 315 | **76%** | 0 | area=None |

_77 runs with >= 4 saved episodes._
