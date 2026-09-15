# Working notes — specialist-arch

Date: 2026-08-18
Question: lab-scale architecture interventions after a published det+track baseline, A6000 48GB, custom student videos.

## Framing (from orchestrator 00-framing.md)
Lab student: pick paper → implement → train public set → test custom videos → significant architecture change (layers). Hardware 1× RTX A6000 48GB.

## Open survey (not name-first)
Firecrawl research search + GitHub web search surfaced:
- Low-score box recovery (ByteTrack BYTE) and adaptive-threshold variants
- Weak-cue association (Hybrid-SORT TCM / HMIoU)
- Observation-centric KF (OC-SORT ORU/OCM)
- Camera-motion compensation + KF width/height (BoT-SORT)
- JDE ReID heads (FairMOT, CSTrack, YOLO11-JDE, FeatureSORT extra attributes)
- Learned association / E2E (TDLP, MOTR, MOTRv2, TrackFormer)
- P2 extra detection scale for small objects (Ultralytics YAML + UAV/small-object papers)
- Adaptive confidence / LQTTrack / MR2-ByteTrack (rescoring, not new necks)

Dominant pattern: association cost + motion model + optional appearance, NOT novel necks.

## Primary reads (WebFetch = summarized ceiling; Firecrawl scrape = full-fidelity)
Scraped (Firecrawl):
- https://docs.ultralytics.com/modes/train/ (batch, freeze, AMP, autobatch)
- https://github.com/ifzhang/ByteTrack (8 GPU, batch 48, mix datasets)

WebFetch long-read:
- ByteTrack ar5iv 2110.06864
- YOLOX arxiv 2107.08430
- Hybrid-SORT 2308.00783
- OC-SORT 2203.14360
- BoT-SORT 2206.14651
- DanceTrack 2111.14690
- FairMOT ar5iv 2004.01888
- YOLO11-JDE 2501.13710

Context7: /ultralytics/ultralytics freeze + two-stage finetune + autobatch.

## Key numbers captured

### ByteTrack
- BYTE: keep almost every box; high-score first match; unmatched tracks ↔ low-score via IoU (not ReID on low-score).
- Applied to 9 trackers: IDF1 +1 to +10 (CenterTrack +9.8 IDF1, IDs 528→144).
- SORT 74.6 MOTA / 76.9 IDF1 / 291 IDs → BYTE 76.6 / 79.3 / 159 on MOT17 val.
- Paper ByteTrack: YOLOX-X, 1440×800, 80 epochs, mix MOT17+CrowdHuman+Cityperson+ETHZ, 8× V100, batch 48, ~12h.
- GitHub: `python3 tools/train.py -f ... -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth`
- Second association must use IoU; ReID on low-score boxes is unreliable.

### YOLOX
- Default train: 300 epochs, batch 128 on typical 8-GPU, multi-scale 448–832.
- Decoupled head: 38.5 → 39.6 AP (+1.1); essential for end-to-end (coupled −4.2 AP, decoupled −0.8).
- Anchor-free +0.9; multi-positives +2.1; SimOTA +2.3 → 47.3 AP.
- Lite decoupled: 1×1 then two 3×3 branches; +1.1 ms latency.

### OC-SORT
- Fixes SORT dummy-update error accumulation during occlusion.
- ORU + OCM; 700+ FPS association on CPU given detections.
- DanceTrack: large gain vs SORT; designed for non-linear motion.
- Same YOLOX dets as ByteTrack for fair MOT tables.

### Hybrid-SORT
- Weak cues: confidence (TCM), height (HMIoU), velocity (ROCM).
- DanceTrack-val component ablation (HOTA): baseline 53.1 → +ROCM 53.7 → +TCM 57.7 → +HMIoU 59.3 → +ReID 63.0; ReID halves FPS 27.8→15.5.
- TCM +4.0 HOTA, −0.7 FPS; HMIoU +1.6; ROCM +0.6.
- TCM on other trackers DanceTrack: DeepSORT +4.9 HOTA; ByteTrack +? ; MOT17 gains tiny (+0.2 to +0.9).
- Training-free, plug-and-play.
- DanceTrack paper numbers much more movable than MOT17 (saturated linear motion).

### BoT-SORT MOT17 val (same YOLOX as ByteTrack)
- ByteTrack* 77.66 / 79.77 / 67.88
- +KF width/height: 77.67 / 79.89 / 68.12 (tiny)
- +CMC: 78.31 / 81.51 / 69.06 (main jump)
- +Pred: 78.39 / 81.53 / 69.11
- +ReID: 78.46 / 82.07 / 69.17 (small on MOT17)
- IoU alone 78.4/81.5/69.1; ReID alone 73.7/70.0/62.4 (worse); min(IoU, masked cosine) 78.5/82.1/69.2
- CMC targets camera motion (MOT17-13 rotation example via cMOTA)

### DanceTrack (adversarial for ReID)
- Uniform clothes + diverse motion; 10× MOT17 images.
- Oracle GT boxes: IoU-only HOTA 72.8; appearance+IoU+motion 59.7 (appearance hurts).
- DeepSORT worse than SORT on DanceTrack val (45.8 vs 47.8 HOTA).
- BYTE 47.1; OC-SORT 52.1 HOTA on val with YOLOX dets.
- Fine-grained pose/mask help more than ReID; KITTI depth transfer hurts (domain shift).

### FairMOT
- Detection vs ReID compete (anchors, shared features, high-dim ReID).
- Homogeneous center-based ReID branch 128-d; DLA-34 multi-layer fusion.
- Train: batch 12, 1088×608, 30 epochs, ~30h on 2× 2080 Ti.
- Center sampling beats ROI-Align / POS-Anchor for IDF1 and IDs.
- Bigger backbone ≠ better MOT; FPN/DLA fusion matters more.
- Uncertainty / GradNorm to balance losses.

### YOLO11-JDE
- Add ReID branch to YOLO11s decoupled head: two 3×3 + 1×1, 128-d.
- Self-supervised via Mosaic + triplet (hard pos, semi-hard neg).
- Ablations CrowdHuman+MOT17 det, 30 ep, batch 32, 640: 128-d best; identity labels not needed (self-sup ≥ labeled).
- Final: 100 ep, batch 64, 1280.
- MOT17 test HOTA 56.6 / MOTA 65.8 vs CountingMOT 63.6/81.3 — detection-limited, not ReID-limited; <10M params vs YOLOX-X ~100M.
- Custom tracker integrating motion+appearance+location: HOTA 56.89 FairMOT default → 60.06 custom.

### Training memory
- Ultralytics: batch=-1 ≈ 60% VRAM; batch=0.70 fraction; freeze=N first layers; two-stage freeze-10 then unfreeze.
- ByteTrack official 8×GPU batch 48 → 6/GPU YOLOX-X 1440×800 FP16.
- A6000 48GB comfortably exceeds one V100 (16/32GB): can run official per-GPU batch or larger.
- YOLOX paper: batch 128 / 8 GPU = 16/GPU at 640.
- MOTR: 8×2080Ti ~96h (DecoderTracker related work). MOTRv2: 8 GPU train.sh + YOLOX prior 8×V100 16 ep.
- FastReID SBS-50 / appearance head: easy on 48GB.

### P2 extra scale
- Official YOLOv8-p2 YAML (Detect P2/4–P5/32) in Ultralytics community thread.
- Small-object / UAV papers claim P2 (stride 4) keeps tiny objects ≥8×8.
- No MOT17 ablation in sources I fully read — thinner evidence than BYTE/OC/CMC.
- Memory: extra high-res maps; still fine for s/m/l @640 on 48GB.

## Independence notes
- ByteTrack / OC-SORT / Hybrid-SORT / BoT-SORT share SORT paradigm and often the same YOLOX-X detections (upstream detector dependence). Association claims should be attributed as “given ByteTrack YOLOX dets”.
- DanceTrack is independent dataset paper (Sun et al.).
- YOLOX is independent detection paper.
- FairMOT and YOLO11-JDE are independent JDE implementations (CenterNet vs YOLO11; CE vs triplet).
- Ultralytics = vendor docs (incentive: ease of use).

## Adversarial checks run
- “ReID always helps” — FALSIFIED by DanceTrack oracle + DeepSORT < SORT + BoT-SORT ReID-only collapse.
- “Need new neck/attention to move metrics” — NOT supported; Hybrid-SORT training-free +4 HOTA TCM; BoT-SORT CMC is the big jump.
- “E2E MOTR is the lab upgrade” — training scale 8 GPU, not A6000-from-scratch.
- “Swap activation / tiny KF Q,R is architecture” — BoT-SORT KF width change is +0.24 HOTA; cosmetic for a lab report.
- Did not find a high-quality source claiming CBAM-in-neck is the standard MOT upgrade.

## Failure modes for custom student video (hypotheses to diagnose first)
- Handheld / walking camera → CMC
- Sports / dance uniforms → do NOT add ReID; add OC-SORT/Hybrid-SORT
- Campus pedestrians with distinct clothes + occlusion → JDE or SDE ReID + BYTE
- Distant small people/vehicles → P2 or higher imgsz
- Class mismatch (cars, bikes, balls) → detector fine-tune + class head, not tracker layers
- Paper-number gap → mix data + YOLOX-X + private-det protocol, not a missing layer
