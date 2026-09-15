# Verification record

Mode: degraded (one-shot specialists; no persistent agent teams; no Codex).

| Claim | Citation | Verdict | Action |
|---|---|---|---|
| ByteTrack MOT17 test 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA; ablation 76.6 / 79.3 / 159 IDs; train 8×V100 batch 48, YOLOX-X, 1440×800, 80 ep, MOT17+CrowdHuman+Cityperson+ETHZ | arXiv:2110.06864 + FoundationVision/ByteTrack | SUPPORTS (fact-checker + orchestrator primary read) | Use as stated |
| DanceTrack: uniform appearance + diverse motion; oracle IoU HOTA 98.1 MOT17 vs 72.8 DanceTrack; appearance can hurt; ByteTrack DanceTrack 47.7 HOTA / 89.6 MOTA vs MOT17 63.1 / 80.3 | arXiv:2111.14690 | SUPPORTS | Use as stated; note MOTA higher on DanceTrack |
| MOTA = 1-(FN+FP+IDSW)/GT; detector-heavy; HOTA ≈ √(DetA·AssA); MOTA R² 0.96 DetA / 0.46 AssA; TrackEval recommends HOTA | arXiv:2009.07736 + TrackEval | SUPPORTS (R² not Pearson r) | Qualify as authors' R² |
| Post-2022 MOT17/20 TBD tables share ByteTrack YOLOX-X; online HOTA 63.1→66.6 via association | arXiv:2506.13457 | SUPPORTS | Use as stated |
| OC-SORT MOT17 63.2 HOTA / 78.0 MOTA vs ByteTrack 63.1 / 80.3 same dets; DanceTrack 54.6–55.1 HOTA vs ByteTrack 47.3; 700+ FPS assoc | arXiv:2203.14360 orchestrator read | SUPPORTS (primary) | Use as stated |
| BoT-SORT MOT17 val: ByteTrack* 77.66/79.77/67.88; +CMC 78.31/81.51/69.06; +ReID 78.46/82.07/69.17; test 80.5/80.2/65.0 | arXiv:2206.14651 orchestrator read | SUPPORTS (primary) | CMC is the load-bearing delta |
| Hybrid-SORT TCM +4 HOTA DanceTrack | specialist-arch, not independently re-read | not independently verified | Mention only as specialist-supported, not load-bearing |
| FairMOT MOT17-only 69.8 MOTA; 2×2080Ti batch 12 | specialist-papers | not independently re-read this pass | Use as supported, not well-supported |

Codex: unavailable.
