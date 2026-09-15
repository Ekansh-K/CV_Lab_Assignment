# What to implement for detection and tracking

## Takeaways

Implement **ByteTrack** (YOLOX detector + BYTE association) as the base paper. Train the detector on **MOT17 half-train + CrowdHuman** (the paper’s own ablation recipe). Compare your numbers to the paper’s **MOT17 half-val row (76.6 MOTA / 79.3 IDF1)**, not the 80.3 MOTA test-set headline. Film **the same class you trained on** (pedestrians, unless you deliberately switch the train set). Annotate three custom sequences in MOT format and score them with TrackEval. Then make one published association change plus one real network-layer change, chosen after you see how your videos fail.

That pairing satisfies the assignment without chasing an 8-GPU SOTA paper. The 2022–2026 workhorse in this field is tracking-by-detection: a YOLOX-class detector plus a SORT-family associator. Later papers (OC-SORT, BoT-SORT, BoostTrack++) mostly refine association on the *same* YOLOX-X detections. End-to-end query trackers (MOTR, MOTRv2) win on dance/sports motion and are the wrong training size for one A6000 if you start from scratch.

On a 48 GB RTX A6000 you can train YOLOX-S/M/L comfortably, YOLOX-X at the official per-GPU batch, a 128-d ReID head, and any of the association modules. You should not train MOTR/MOTRv2 from scratch to reproduce those papers.

## What this project actually is

The work is a reproduce-then-improve study, not “run YOLO on a phone video.” The expected loop is:

1. Pick a published detection + multi-object-tracking architecture with code and reported numbers.
2. Implement that architecture (detector and associator, not a blog wrapper).
3. Train on a standard public dataset that the paper actually used.
4. Reproduce the paper on the paper’s own split and protocol.
5. Test on videos you film, as a *separate* domain-shift experiment.
6. Diagnose why the numbers differ (they will).
7. Change a real module — association, motion model, or a network head — and show before/after on both the public val split and the custom set.

Two numbers live in two tables. A single sentence of the form “we got 95% on our video vs the paper’s 76 MOTA” is the comparison this literature was built to stop.

## The paper to start from

ByteTrack (Zhang et al., ECCV 2022) is the right default. The method is YOLOX detections plus BYTE: match high-score boxes first, then recover occluded objects by matching unmatched tracks to *low-score* boxes with IoU. The official numbers on MOT17 test are 80.3 MOTA, 77.3 IDF1, 63.1 HOTA at about 30 FPS on a V100.[^1][^2] The official GitHub ablation model, trained on CrowdHuman + MOT17 half-train and evaluated on MOT17 half-val, is 76.6 MOTA / 79.3 IDF1 / 159 ID switches.[^2] That ablation row is the number you can honestly stand next to.

The headline 80.3 is not a student target. That model is YOLOX-X at 1440×800, 80 epochs, batch 48 on 8 V100s, trained on MOT17 + CrowdHuman + CityPersons + ETHZ, then submitted to the hidden test server, with interpolation and per-sequence threshold tuning in the official description.[^1][^2]

Why this paper and not “whatever is SOTA”:

- It names both halves of the system (YOLOX, BYTE). You can implement BYTE from the paper in a week and train YOLOX on one 48 GB card.
- It is the actual 2022 inflection for tracking-by-detection. A 2025 survey states that MOT17/MOT20 tracking-by-detection tables since then reuse ByteTrack’s private YOLOX-X detections; online MOT17 HOTA only moved from 63.1 to BoostTrack++’s 66.6 by association heuristics on that same detector.[^3]
- Official code runs on a raw video (`demo_track.py`) and publishes smaller YOLOX-S/M/L weights, so you are not forced to train the X model.
- The published follow-ups *are* your phase-2 change: OC-SORT and BoT-SORT.

Do not start from BoostTrack++ (results paper on reused weights), MOTRv2 (8-GPU, five-commit repo), or a YOLO+OpenCV-SORT tutorial. Those fail the “implement their architecture, then change it” clause.

Two alternatives only if the course reads “architecture” as “a joint network”:

- **FairMOT** (IJCV 2021): CenterNet/DLA-34 with a 128-d ReID head. MOT17-only is reported at 69.8 MOTA; training is batch 12 for ~30 h on two 2080 Tis, which fits one A6000. DCNv2 compilation is the operational risk.
- **CenterTrack** if the custom videos are multi-class (cars + people). It ships an official COCO 80-class tracking model. Weaker MOT17 numbers; older code.

If the custom videos are dance or team sports with near-identical clothes, OC-SORT is a better *base* than ByteTrack, because that paper’s contribution is exactly Kalman failure under non-linear motion.[^4] In that case ByteTrack becomes the weaker baseline you beat.

## Train set, custom videos, and an honest comparison

Pick the paper first, then the public set. MOT17 is the pairing for ByteTrack. CrowdHuman is extra *detection* data, not a tracking test set. COCO is only pretraining. If you film cars, you should have trained on KITTI, BDD100K, or UA-DETRAC — not MOT17.

**Stage A — the only column that may face a paper table.** Train on MOT17 half-train (+ CrowdHuman if you want the ablation row). Evaluate with TrackEval on MOT17 half-val. Report HOTA, DetA, AssA, MOTA, IDF1, ID switches, FP, FN. Tag every number with dataset, split, private vs public detections, and extra data. If you cannot submit to the hidden test server, do not compare val to test.

MOTA is detector-heavy: it is \(1-\sum(\mathrm{FN}+\mathrm{FP}+\mathrm{IDSW})/\sum\mathrm{GT}\), and on MOT17 the HOTA authors report that detection errors outweigh ID switches by roughly 40–180×. Their MOT17 scatter has MOTA \(R^2 = 0.96\) with DetA and \(0.46\) with AssA. HOTA is the geometric mean of DetA and AssA (then averaged over localisation thresholds) and is the metric TrackEval marks as recommended.[^5][^6] Lead with HOTA; still print MOTA and IDF1. They diagnose different errors.

**Stage B — a different table.** Film five pedestrian clips that actually stress a tracker: static sidewalk; same place handheld; two people in similar clothes crossing; leave-and-reenter; indoor or backlight. Annotate three of them in CVAT as MOT tracks (stable IDs through occlusion, box on every evaluated frame). Export `gt.txt` + `seqinfo.ini` + seqmaps. Run TrackEval with `--DO_PREPROC False` unless you implemented MOT16 distractor rules. The other two clips are a failure gallery only.

Without identity ground truth, MOTA / HOTA / IDF1 are undefined. “95% of frames looked correct” is not a tracking metric.

DanceTrack is the published warning that MOT17 skill does not transfer even to other *people* videos. Oracle IoU matching is 98.1 HOTA on MOT17 val and 72.8 on DanceTrack val; adding appearance can *hurt*. Published ByteTrack is 63.1 HOTA / 80.3 MOTA on MOT17 test and 47.7 HOTA / 89.6 MOTA on DanceTrack test — MOTA stays high while association collapses.[^7]

After any change, re-run *both* stages. Improving custom HOTA while wrecking MOT17 val is transfer, not a better method.

## The architecture change

Follow-up papers after ByteTrack do not treat “a new neck” as the standard upgrade. What repeatedly moves association metrics is: keep low-score boxes (BYTE), repair Kalman error during occlusion (OC-SORT), compensate camera motion (BoT-SORT), and add appearance only when people look different. The layer changes that papers actually ablate are a decoupled head (already present in YOLOX/YOLOv8/YOLO11), a ~128-d ReID branch, and an extra high-resolution detection scale (P2) for small objects.

On MOT17, OC-SORT with the *same* YOLOX detections as ByteTrack is 63.2 HOTA / 78.0 MOTA versus ByteTrack’s 63.1 / 80.3 — essentially a tie on that benchmark, and MOTA is slightly worse.[^4] On DanceTrack the same swap is 54.6–55.1 HOTA versus ByteTrack’s 47.3. BoT-SORT’s MOT17 val ablation, starting from a ByteTrack reimplementation (77.66 MOTA / 79.77 IDF1 / 67.88 HOTA), shows Kalman width/height as almost cosmetic; affine camera-motion compensation is the jump (78.31 / 81.51 / 69.06); extra ReID is +0.06 HOTA.[^8] ReID-only loses to IoU in that paper.

So the change has to be chosen from the custom-video failure mode, not from a MOT17 leaderboard delta.

**Default package (campus pedestrians, phone camera):**

1. **Association (required, training-free):** keep BYTE, add BoT-SORT camera-motion compensation, and replace the linear Kalman update with OC-SORT’s observation-centric re-update and momentum. Same frozen detections, two trackers, report HOTA / IDF1 / ID switches.
2. **Layers (required, so the report is not “we swapped a Kalman filter”):** add a 128-d appearance head on the detector (FairMOT / YOLO11-JDE style) *only if* people in the videos wear distinct clothes. If they do not — sports kits, similar jackets — skip ReID; add a P2 (stride-4) detection scale instead if the failure is missed small/distant people.

Do not: swap activations; retune Kalman Q/R and call it architecture; fine-tune on the custom *test* clips; add ReID because “DeepSORT did”; train MOTR from scratch.

## What the A6000 can actually run

| Work | Official recipe | On 1× A6000 48 GB |
|---|---|---|
| BYTE / OC-SORT association | CPU, no weights | Instant |
| BoT-SORT CMC | OpenCV GMC / ECC | Instant |
| YOLOX-S/M/L train | smaller than the paper X | Comfortable at 640–800, AMP |
| YOLOX-X as in ByteTrack | 8×V100, total batch 48, 1440×800, ~12 h | Fits the official *per-GPU* batch (6); wall-clock is longer unless you accumulate to 48 |
| 128-d JDE head | FairMOT batch 12 @ 1088×608; YOLO11-JDE batch 32 @ 640 | Easy |
| FastReID appearance tower | one GPU | Easy |
| MOTR / MOTRv2 from scratch | 8 GPUs, days | Do not use as the plan |

Practical detector recipe: start from COCO-pretrained YOLOX-M or YOLOX-L (or the official ByteTrack-M/L weights), train the ablation mix at 800 or 1280 with AMP and autobatch (~60% of 48 GB). Use official YOLOX-X weights for an upper-bound inference run if you want a detector-matched comparison to OC-SORT/BoT-SORT tables. Do not spend the project matching 80.3 MOTA.

## Conflicts

**TBD vs end-to-end.** Tracking-by-detection still wins on crowded linear MOT17/20; query trackers win on DanceTrack.[^3] That is a domain split, not a source error. Student-filmed campus or sports video is closer to DanceTrack than to MOT17, which is why OC-SORT belongs in phase 2 even if MOT17 barely moves.

**Appearance.** ReID is a default upgrade in MOT17-style papers and a demonstrated *harm* on DanceTrack’s oracle table.[^7] Treat it as conditional on visual diversity.

**Private detections.** Both public-detector and private-detector MOT17 protocols are valid and must be labelled. Mixing them is not.

**Which scalar ranks the work.** MOT Challenge historically ranked on MOTA. TrackEval, HOTA, and DanceTrack treat HOTA as the balanced score. Report all three families.

## Gaps

- A6000 batch sizes were inferred from published multi-GPU recipes, not profiled on this card.
- Hybrid-SORT’s DanceTrack TCM gains were not independently re-read in the orchestrator pass; they are not load-bearing for the default package.
- FairMOT’s MOT17-only 69.8 MOTA was not re-fetched this pass; use it only if you choose Pair B.
- The MOT Challenge website itself was not readable; protocol details come from TrackEval, MOT16, and the papers.
- No custom video exists yet. If you film vehicles or a sport with identical kits, the default package changes (different train set, no ReID).
- Persistent research-agent teams and a Codex cross-model review were unavailable in this environment. Verification is the Claude-family fact-checkers plus orchestrator reads of ByteTrack, OC-SORT, BoT-SORT, and the ByteTrack README.

## Confidence assessment

**High** on: ByteTrack as the pedagogical base; 80.3 vs 76.6 as the honest comparison row; two-table evaluation; MOTA vs HOTA roles; DanceTrack as the transfer warning; OC-SORT / BoT-SORT CMC as the documented phase-2 modules; A6000 being enough for YOLOX-scale training and not enough to reproduce MOTR schedules.

**Medium** on: exact A6000 batch/imgsz; whether your still-unfilmed videos will look more like MOT17 or DanceTrack; P2 as a tracking (not just detection) win.

**Low** on: any claim that a layer change will beat the paper’s MOT17 *test* number. That number is a different protocol.

## Sources

[^1]: Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. 2022. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV / arXiv:2110.06864. https://arxiv.org/html/2110.06864 — Type: peer-reviewed paper (orchestrator full read)

[^2]: FoundationVision / Yifu Zhang. 2022. ByteTrack official repository. GitHub. https://github.com/FoundationVision/ByteTrack — Type: official code / README

[^3]: Momir Adžemović. 2025. Deep Learning-Based Multi-Object Tracking: A Comprehensive Survey from Foundations to State-of-the-Art. arXiv:2506.13457. https://arxiv.org/html/2506.13457 — Type: preprint survey

[^4]: Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. 2023. Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking. CVPR / arXiv:2203.14360. https://arxiv.org/html/2203.14360 — Type: peer-reviewed paper (orchestrator full read)

[^5]: Jonathon Luiten, Aljoša Ošep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. 2020. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. IJCV / arXiv:2009.07736. https://arxiv.org/html/2009.07736 — Type: peer-reviewed

[^6]: Jonathon Luiten and Arne Hoffhues. 2020. TrackEval. GitHub. https://github.com/JonathonLuiten/TrackEval — Type: official evaluation kit

[^7]: Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. 2022. DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion. CVPR / arXiv:2111.14690. https://arxiv.org/html/2111.14690 — Type: peer-reviewed benchmark paper

[^8]: Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. 2022. BoT-SORT: Robust Associations Multi-Pedestrian Tracking. arXiv:2206.14651. https://arxiv.org/html/2206.14651 — Type: preprint (orchestrator full read of method + ablation)

[^9]: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. 2021. YOLOX: Exceeding YOLO Series in 2021. arXiv:2107.08430. https://arxiv.org/html/2107.08430 — Type: preprint (detector used by ByteTrack)

[^10]: Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. 2021. FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking. IJCV / arXiv:2004.01888. https://arxiv.org/html/2004.01888 — Type: peer-reviewed (alternative joint-network paper)

[^11]: Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. 2016. MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831. https://ar5iv.labs.arxiv.org/html/1603.00831 — Type: official benchmark paper

[^12]: CVAT.ai. MOT format. https://docs.cvat.ai/docs/dataset_management/formats/format-mot/ — Type: official annotation-tool docs
