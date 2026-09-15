# Detection + tracking lab — locked design

Date: 2026-08-19

## Goal

Implement ByteTrack (Zhang et al., ECCV 2022) as the base method, then Hybrid-SORT (Yang et al., AAAI 2024) on the **same official YOLOX detections**, and show that Hybrid-SORT beats ByteTrack’s published DanceTrack result. Custom phone videos are a second, separate transfer table — not the paper comparison.

## What we will say (and will not)

Allowed, if numbers support it:

> We reimplemented ByteTrack using official YOLOX DanceTrack weights. ByteTrack is reported at 47.3–47.7 HOTA on DanceTrack test. On DanceTrack val we obtain BYTE HOTA *h1* and Hybrid-SORT HOTA *h2*, above that ByteTrack DanceTrack result and in line with Yang et al. (val Hybrid-SORT 59.3 HOTA / test 62.2). MOT17 test 80.3 MOTA is a different protocol and is not this comparison.

Forbidden:

- “We beat ByteTrack 80.3 MOTA / 63.1 HOTA on MOT17 test.”
- Comparing YOLOX-M (or custom video “accuracy %”) to 80.3 or to 47.7.
- Claiming OC-SORT beat ByteTrack-M in either paper (that pair was never published).

## Pair

| Role | Paper | Code we actually run |
|---|---|---|
| Base | ByteTrack, ECCV 2022 | BYTE associator in HybridSORT (`tools/run_byte_dance.py`) |
| Upgrade | Hybrid-SORT, AAAI 2024 | `tools/run_hybrid_sort_dance.py` (no ReID) |
| Detector | Official DanceTrack YOLOX-X weights from ByteTrack / Hybrid-SORT zoo | Download only. Do not train X. |

One repository: https://github.com/ymzis69/HybridSORT (already contains BYTE, Hybrid-SORT, TrackEval, DanceTrack scripts).

Backup if Hybrid-SORT env is blocked: OC-SORT on the same boxes (DanceTrack 54.6–55.1 HOTA). Still a clear win over 47.7. Not both upgrades.

## Two tables only

**Table A — beat the base paper (DanceTrack val).**  
Same YOLOX-X weights. BYTE vs Hybrid-SORT. TrackEval: HOTA, AssA, DetA, MOTA, IDF1, IDSW.  
Published anchors: ByteTrack DanceTrack test 47.3–47.7 HOTA; Hybrid-SORT DanceTrack val 59.3 / test 62.2.

**Table B — transfer (custom videos).**  
Same two trackers, same frozen weights. 3 filmed pedestrian/people clips, 2 annotated in CVAT → MOT format. Not compared to 47.7 or 80.3.

## Hardware and where work happens

- This laptop/workspace: plans, code layout, report text. **No GPU training or eval here.**
- Remote RTX A6000 48 GB (later): install, download, inference, TrackEval.
- GPU time: inference only, about 1–3 hours total for DanceTrack val × 2 trackers. No 8-GPU YOLOX-X train.

## Explicitly out of scope

YOLOX-X from scratch; Hybrid-SORT-ReID; BoT-SORT/CMC stack; P2 head; FairMOT; MOTR; MOT17 test server; beating 80.3; more than two trackers; fine-tuning on custom test clips.

Optional later (only if the course demands “we trained a detector”): one overnight YOLOX-M on MOT17 half + CrowdHuman, reported as a **separate** training note, never mixed into Table A.

## Success

1. BYTE DanceTrack val HOTA is in a plausible band (not 20, not 80).
2. Hybrid-SORT val HOTA is **higher** than our BYTE, and we can relate it to 59.3 / 62.2.
3. Report uses the allowed paragraph; two tables; no 80.3 claim.
4. Two custom clips scored the same way.
