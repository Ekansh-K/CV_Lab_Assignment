# Evidence trail — specialist-papers

## Sources read (full or near-full)

1. Adžemović 2025 survey arXiv HTML 2506.13457 — full fetch, Tables 3–5 extracted.
2. ByteTrack official README (FoundationVision/ByteTrack) — train cmds, metrics, extra data.
3. OC-SORT official README (noahcao/OC_SORT) — metrics, YOLOX stack, 700 FPS association.
4. FairMOT official README + arXiv 2004.01888 HTML — architecture, 2x2080Ti 30h, MOT17-only 69.8.
5. MOTR official README — 8x 2080Ti, 2.5–4 days, HOTA 57.8 MOT17 / 54.2 Dance.
6. MOTRv2 official README — 8 GPU train, DanceTrack 69.9 reported in README (73.4 in paper/survey), 5 commits.
7. BoT-SORT official README (NirAharon/BoT-SORT) — YOLOX/YOLOv7, 65.0 HOTA MOT17.
8. CenterTrack official README — 67.8 MOTA MOT17 private, 2-GPU custom train, COCO 80-class.
9. MeMOTR official README — 8 GPU >=32GB, checkpoint ~10GB, Dance 68.5 HOTA, MOT17 58.8, demo notebook.
10. BoostTrack official README — 66.6 HOTA MOT17, reused Deep OC-SORT/ByteTrack dets, interpolation.

## Not fully read
- Guan 2025 Springer survey: search snippet only (TBD vs E2E, ByteTrack as detector-centric landmark).
- Ultralytics track docs: fetch returned shell only; default BoT-SORT is unverified here.
- BoxMOT: GitHub web_search summary only (~8.3k stars).

## Candidate metrics (from sources above)

### MOT17 test (private dets unless noted)
- ByteTrack: 80.3 MOTA, 77.3 IDF1, 63.1 HOTA (README; survey Table 3)
- BoT-SORT-ReID: 80.5 MOTA, 80.2 IDF1, 65.0 HOTA (README)
- OC-SORT private: 78.0 MOTA, 77.5 IDF1, 63.2 HOTA (README)
- FairMOT: 73.7 MOTA, 72.3 IDF1 (README); MOT17-only train 69.8 MOTA
- CenterTrack private: 67.8 MOTA (README)
- MOTR: 73.4 MOTA, 68.6 IDF1, 57.8 HOTA (README)
- MOTRv2: 78.6 MOTA, 75.0 IDF1, 62.0 HOTA (survey Table 3)
- MeMOTR: 58.8 HOTA (README)
- BoostTrack++: 66.6 HOTA, 80.7 MOTA, 82.2 IDF1 (README; survey)

### DanceTrack test
- ByteTrack: 53.6 HOTA, 92.3 MOTA, 55.3 IDF1 (survey Table 5, produced by survey authors)
- OC-SORT: 55.1 HOTA, 89.4 MOTA (README + survey)
- MOTR: 54.2 HOTA (README)
- MeMOTR: 68.5 HOTA (README)
- MOTRv2: 73.4 HOTA paper / 69.9 README table

### Train GPU
- ByteTrack: `python3 tools/train.py ... -d 8 -b 48`
- FairMOT: batch 12, ~30h on 2x RTX 2080 Ti (paper §5.2)
- MOTR: 8x 2080Ti, 2.5d V100 / 4d 2080Ti
- MOTRv2: "training MOTR on 8 GPUs"
- MeMOTR: 8 GPUs, recommended >=32GB; `--use-checkpoint` ~10GB
- CenterTrack: example `--gpus 0,1`
- OC-SORT/BoostTrack/BoT-SORT association: no detector training in their SOTA numbers (reuse YOLOX)

## Rejected as base
- YOLO+OpenCV SORT blogs: not architecture papers
- BoostTrack++: 2024 SOTA but heuristic plug-ins + interpolation on reused dets
- MOTRv2: 5 commits, 8 GPU, YOLOX+MOTR hybrid
- Offline GNN (SUSHI, CoNo-Link)
- UniAD: planning stack, not a lab MOT paper
