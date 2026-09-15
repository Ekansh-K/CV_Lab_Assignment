# Audit — specialist-papers

## Independence axes

Concentrated:
- **Paradigm**: most 2022–2024 MOT17 numbers share ByteTrack's YOLOX-X private detections (Adžemović Table 3 caption). Agreement on TBD > E2E on MOT17 is partly detector-controlled.
- **Authorship**: ByteTrack and FairMOT share Yifu Zhang / Xinggang Wang line; MOTR and MOTRv2 are Megvii.
- **Upstream dets**: ByteTrack, OC-SORT, BoT-SORT, BoostTrack, Deep OC-SORT all ride the same YOLOX-X.

Independent on those axes:
- FairMOT / CenterTrack: own CenterNet detectors, pre-YOLOX era.
- MOTR / MeMOTR: DETR-family, own detections.
- Adžemović survey (Belgrade) vs Guan Springer vs official READMEs: three production processes.

## Adversarial checks

1. **Is YOLO+ByteTrack just a tutorial echo chamber?**
   - Tested via survey Tables 3–5 (research), GitHub official repos (primary), BoostTrack/BoT-SORT reusing YOLOX (primary).
   - Outcome: NOT an echo chamber for *practice/research baselines*. It IS an echo chamber if the assignment is "implement a network architecture" — BYTE is a heuristic.

2. **Are 2024 SOTA papers better lab bases?**
   - BoostTrack++ / ImprAsso / CoNo-Link: higher HOTA, but association heuristics or offline GNN, reused dets. Worse for "implement architecture" and "significant change."
   - Outcome: opposition found; does not displace ByteTrack/FairMOT as lab bases.

3. **Is E2E actually unusable on 1x A6000?**
   - MOTR official: 8x 2080Ti. MeMOTR: 8x >=32GB, but checkpoint path ~10GB/GPU so 1x 48GB can *fit* a tiny batch, not match paper schedule.
   - Outcome: inference-ready yes; paper-repro training no.

4. **Can students approach paper numbers?**
   - ByteTrack README: 80.3 needs extra data + interpolation + per-seq tune; ablation 76.6 on half-val is the honest target.
   - FairMOT: MOT17-only 69.8 vs 73.7 with MIX+CH — approachable if they use the MOT17-only row.

## Falsifiability

- Dominance of TBD+YOLOX on MOT17: falsifiable via leaderboard; currently holds for online methods through BoostTrack++ (still TBD+YOLOX).
- E2E wins DanceTrack: falsifiable; holds in survey Table 5.
- 1x A6000 can train FairMOT: paper used 2x 11GB; 48GB is larger — practically confirmed by published batch/GPU, not re-run here.
- Student-video domain match: not testable without the videos. Hypothesis: closer to DanceTrack/SportsMOT than MOT17.

## Cross-tier corroboration

- TBD still dominant on MOT17: peer-review-adjacent survey + official READMEs + 2024 BoostTrack paper.
- YOLOX as shared detector: survey caption + ByteTrack/OC-SORT/BoT-SORT/BoostTrack READMEs.
- FairMOT trains on 2 GPUs: paper text + README scripts.
