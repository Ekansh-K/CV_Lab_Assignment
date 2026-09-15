# Research Report: Base papers for a detection + multi-object-tracking lab assignment

## Takeaways

The 2024–2026 workhorse is not “YOLO the detector” in the abstract and not MOTR-style end-to-end tracking. It is **tracking-by-detection with a YOLOX-class detector plus a SORT-family associator**. Almost every online MOT17/MOT20 number published since ByteTrack uses *the same private YOLOX-X detections*; later papers (OC-SORT, BoT-SORT, Deep OC-SORT, BoostTrack++) are association refinements on that stack, not new detectors. End-to-end query trackers (MOTR, MeMOTR, MOTRv2, MOTIP) win on DanceTrack-style non-linear motion and lose on crowded linear MOTChallenge scenes.

For *this* assignment the ranking is pedagogical, not SOTA:

1. **ByteTrack (ECCV 2022)** — best default base if students implement YOLOX (or a smaller YOLO) + BYTE themselves, train the detector, and treat association as the later modification surface.
2. **OC-SORT (CVPR 2023)** — better default if student-filmed video is sports/campus/non-linear; the paper *is* a motion architecture.
3. **FairMOT (IJCV 2021)** — best paper if “implement their architecture” is read as “implement a joint network,” and the only candidate whose published MOT17-only number a student can reasonably approach on one A6000.
4. **CenterTrack (ECCV 2020)** — best older joint paper for multi-class custom video (official COCO 80-class tracking model).
5. **MeMOTR (ICCV 2023)** — only E2E paper that is maintained enough and memory-checkpointed enough to even consider; still an 8-GPU paper.
6. **BoT-SORT (2022)** — do not assign as the *first* paper; it is the natural *architectural change* applied to ByteTrack (CMC + Kalman state + ReID).

Do not assign BoostTrack++, MOTRv2, offline GNNs, or YOLO+OpenCV-SORT blogs as the base paper. Headline MOTChallenge test numbers (ByteTrack 80.3 MOTA, BoT-SORT 65.0 HOTA) are not honest student targets.

Most load-bearing claims below are well-supported from official READMEs plus the 2025 Adžemović survey tables. GPU-fit for a *single* A6000 is inferred from published multi-GPU recipes, not re-run. Cross-model self-review was unavailable (no Codex subagent in this environment).

## Findings

### What is actually dominant in 2024–2026

Tracking-by-detection remains the dominant MOT paradigm, and 2022 is a real inflection: ByteTrack for TBD association and MOTR for query-based E2E `[CITED][WELL-SUPPORTED]`.[^1] Heuristic TBD still wins on dense, approximately linear pedestrian scenes (MOT17/MOT20); learned association and E2E win when motion is complex (DanceTrack, SportsMOT) `[CITED][WELL-SUPPORTED]`.[^1] A second 2025 survey independently organizes the same TBD split (detector-centric vs association-centric) and treats ByteTrack as the detector-centric landmark `[CITED][SUPPORTED]`.[^2]

The part the assignment text under-states is the **detector lock-in**. On MOT17 and MOT20, “all tracking-by-detection methods use the same private detector — ByteTrack’s YOLOX-X” `[CITED][WELL-SUPPORTED]`.[^1] From 2022 to 2024, online HOTA on MOT17 moved only from ByteTrack’s 63.1 to BoostTrack++’s 66.6 — a 3.4-point gain achieved by association heuristics on that same detector, not by a new detection architecture `[CITED][WELL-SUPPORTED]`.[^1] Re-ID adds ~0.4 HOTA on MOT17 in BoT-SORT’s own ablation; camera-motion compensation adds ~1.0–1.5 HOTA on MOT17; a rich heuristic set is what actually separates BoostTrack++ from ByteTrack `[CITED][SUPPORTED]`.[^1]

On DanceTrack the ranking inverts. ByteTrack sits at 53.6 HOTA; MOTRv2 at 73.4; even the weakest modern E2E in the survey table (MeMOTR, 68.5) beats the best TBD/offline method (CoNo-Link, 63.8) `[CITED][WELL-SUPPORTED]`.[^1] MOTA is already ~92 for TBD there, so the gap is association, not detection. That domain split matters for student-filmed video: a handheld campus or sports clip is closer to DanceTrack/SportsMOT than to MOT17’s linear pedestrians `[CONCLUSION][SUPPORTED]`.

Practitioner ecosystems follow the TBD+YOLOX/YOLO line, not MOTR. ByteTrack, FairMOT, BoT-SORT, and OC-SORT are the official paper repos that still define the stack; BoxMOT and Ultralytics are wrappers around that stack, not papers `[CITED][SUPPORTED]`.[^3][^4][^5][^8] `[HYPOTHESIS]` Ultralytics’ `track` mode is widely described as defaulting to BoT-SORT; that specific default was not confirmed from a full docs read in this pass.

### What “implement their architecture” actually means

The field’s workhorses split into three implementability classes.

**Association-only papers** (ByteTrack, OC-SORT, BoT-SORT, Deep OC-SORT, BoostTrack) pair a *named detector* (almost always YOLOX) with a *named associator*. The architecture a student can write from the paper in a week is the associator: BYTE’s two-stage high/low-score matching; OC-SORT’s observation-centric re-update / momentum / recovery; BoT-SORT’s CMC + Kalman state. The detector is a separate, larger project (YOLOX training). Official ByteTrack training is `python3 tools/train.py -f … -d 8 -b 48` — eight GPUs, batch 48 — on CrowdHuman + MOT17 + Cityperson + ETHZ `[CITED][WELL-SUPPORTED]`.[^3] The association itself has no learned weights.

**Joint detection–embedding papers** (FairMOT, JDE, CenterTrack) *are* network architectures. FairMOT is CenterNet/DLA-34 with homogeneous detection and ReID heads, 128-D identity embeddings at object centers, and a hierarchical Kalman+ReID+IoU associator `[CITED][WELL-SUPPORTED]`.[^6][^7] CenterTrack is a pair-of-frames CenterNet that predicts a detection heatmap plus an offset to the previous center `[CITED][WELL-SUPPORTED]`.[^9] These are the papers for which “implement the architecture” is not a euphemism for “call `BYTETracker.update`.”

**End-to-end query papers** (MOTR, MOTRv2, MeMOTR, MOTIP) are real architectures (track queries, temporal aggregation, memory attention) and are the wrong size for a weeks-long lab unless the course accepts inference-plus-a-small-module-change. MOTR trains on 8× RTX 2080 Ti (about 2.5 days on V100, 4 days on 2080 Ti) `[CITED][WELL-SUPPORTED]`.[^10] MOTRv2’s official recipe is “training MOTR on 8 GPUs,” the repo has five commits, and the method is explicitly YOLOX proposals into MOTR — the authors themselves say it is not a pure E2E detector `[CITED][WELL-SUPPORTED]`.[^11][^1] MeMOTR is the only E2E repo that documents a memory-optimized path (~10 GB/GPU with gradient checkpoint) and ships a custom-video notebook `[CITED][WELL-SUPPORTED]`.[^12]

### Numbers a student can actually approach

ByteTrack’s advertised MOT17 test result is 80.3 MOTA, 77.3 IDF1, 63.1 HOTA at 30 FPS on one V100 `[CITED][WELL-SUPPORTED]`.[^3] The same README says the comparable *reproducible* number is 76.6 MOTA on MOT17 half-val with the ablation model (CrowdHuman + MOT17 half-train), and that 80+ MOTA on the test server needs interpolation plus “carefully tune the test image size and high score detection threshold of each sequence” `[CITED][WELL-SUPPORTED]`.[^3] `[CONCLUSION][WELL-SUPPORTED]` A lab that grades against 80.3 is grading against extra data, interpolation, and per-sequence test-time search.

FairMOT publishes the data-ablation a lab needs: MOT17-only 69.8 MOTA / 69.9 IDF1; MIX 72.9 / 73.2; CrowdHuman+MIX 73.7 / 72.3 `[CITED][WELL-SUPPORTED]`.[^7] The half-val checkpoint they release scores 69.1 MOTA / 72.8 IDF1 `[CITED][WELL-SUPPORTED]`.[^7] Training is batch 12 for 30 epochs, “about 30 hours on two RTX 2080 Ti GPUs” `[CITED][WELL-SUPPORTED]`.[^6] One 48 GB A6000 is more memory than two 11 GB 2080 Tis combined; FairMOT is the only joint paper whose *published training recipe* clearly fits the hardware `[SYNTHESIS][SUPPORTED]` from the FairMOT GPU statement and the A6000’s 48 GB.

OC-SORT’s MOT17 private numbers (78.0 MOTA, 77.5 IDF1, 63.2 HOTA) and DanceTrack 55.1 HOTA / 89.4 MOTA come from *reusing previous methods’ detections* `[CITED][WELL-SUPPORTED]`.[^4] Association alone is ~700 FPS on CPU; the GPU cost is whatever detector they plug in `[CITED][WELL-SUPPORTED]`.[^4] Students who train a smaller YOLOX (S/M/L) on MOT17 half will not hit 78 MOTA and should compare on the same detections the paper used, or report their own detector’s DetA separately `[CONCLUSION][SUPPORTED]`.

### Ranked candidates

**1. ByteTrack — Zhang et al., ECCV 2022.** Detector YOLOX (X/L/M/S/tiny/nano in the official zoo); tracker BYTE (associate every box: high-score IoU first, then low-score boxes to unmatched tracks). Train data for the headline model: CrowdHuman + MOT17 + Cityperson + ETHZ. Reported: MOT17 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA; MOT20 77.8 / 75.2 / 61.3; DanceTrack 53.6 HOTA in the survey’s re-run `[CITED][WELL-SUPPORTED]`.[^3][^1] Code: https://github.com/FoundationVision/ByteTrack (official train, track, demo, ONNX/TensorRT). GPU: official train 8 GPUs batch 48; inference ~30 FPS V100; A6000 can train YOLOX-S/M/L comfortably and YOLOX-X at reduced batch `[CITED][WELL-SUPPORTED]`.[^3] `[HYPOTHESIS]` YOLOX-X at batch 4–8 on 48 GB is plausible but was not profiled here.

Why it works as the base: it is the actual 2022–2026 reference; the detector and associator are named and separable; official `demo_track.py` runs on a raw video; the modification surface is huge (swap BYTE for OC-SORT / add ReID / add CMC / change the detector family). Why it fails if misread: implementing BYTE on top of Ultralytics `model.track()` is a blog post, not a paper implementation; 80.3 MOTA is the wrong comparison number.

**2. OC-SORT — Cao et al., CVPR 2023.** Detector: YOLOX (same family; official repo vendors ByteTrack’s YOLOX). Tracker: observation-centric SORT (ORU / OCM / OCR) — a pure motion model, no ReID required. Reported: MOT17 private 63.2 HOTA / 78.0 MOTA / 77.5 IDF1; MOT20 private 62.4 / 75.9 / 76.4; DanceTrack 55.1 / 89.4 / 54.2; KITTI-cars 76.5 HOTA `[CITED][WELL-SUPPORTED]`.[^4] Code: https://github.com/noahcao/OC_SORT. GPU: association is CPU-side; detector training is the ByteTrack problem.

Why it works: student-filmed video will violate Kalman linear-motion assumptions; this paper’s contribution *is* that failure mode; Deep OC-SORT (ICIP 2023) is a documented next step (add adaptive appearance) `[CITED][SUPPORTED]`.[^4] Why it would not: if the course insists the student train a novel *network*, OC-SORT has none.

**3. FairMOT — Zhang et al., IJCV 2021.** Detector: anchor-free CenterNet on DLA-34 (also ResNet, HRNet, and a released YOLOv5s light model). Tracker: center ReID embedding + Kalman + hierarchical IoU fallback (MOTDT-style). Train: CrowdHuman pretrain then MIX, or MOT17 only. Reported: MOT17 73.7 MOTA / 72.3 IDF1 at ~26–30 FPS; MOT17-only 69.8 MOTA; MOT20 61.8 MOTA; half-val 69.1 MOTA `[CITED][WELL-SUPPORTED]`.[^7] Code: https://github.com/ifzhang/FairMOT, including `demo.py` and a custom-dataset recipe. GPU: 2× 2080 Ti, batch 12, ~30 h `[CITED][WELL-SUPPORTED]`.[^6] — **fits one A6000**.

Why it works: this is the cleanest match to “implement their architecture, train, compare, then change it.” The paper’s own ablations *are* the modification menu (backbone, ReID dimension, loss balancing, sampling at center vs ROI-Align). Why it would not: last meaningful release 2021; DLA-34 depends on DCNv2 compilation, a known student-time sink; the method is pedestrian-centric; 2024 SOTA TBD papers have left it behind on MOT17 by ~7 MOTA / ~7 HOTA `[SYNTHESIS][SUPPORTED]` from FairMOT 73.7 MOTA vs ByteTrack 80.3 / BoostTrack++ 66.6 HOTA.

**4. CenterTrack — Zhou, Koltun, Krähenbühl, ECCV 2020.** Detector+tracker: CenterNet on (current frame, previous frame, rendered previous heatmap) predicting heatmap + offset. Reported: MOT17 private 67.8 MOTA at 22 FPS; public 61.5; KITTI-cars 89.44 MOTA; nuScenes monocular 3D AMOTA@0.2 27.8 `[CITED][WELL-SUPPORTED]`.[^9] Code: https://github.com/xingyizhou/CenterTrack. Official custom-dataset training uses `--gpus 0,1` `[CITED][WELL-SUPPORTED]`.[^9] A COCO-trained 80-class tracking model is released for mixed-category video `[CITED][WELL-SUPPORTED]`.[^9]

Why it works: student videos are rarely MOT17 pedestrians; CenterTrack is the only listed paper that officially ships a multi-class tracker and a monocular-3D extension. Why it would not: 20 commits, DCNv2, no long-range ReID, 67.8 MOTA is easy to “beat” by swapping in a 2022 detector rather than by a real idea.

**5. MeMOTR — Gao & Wang, ICCV 2023.** Detector: DAB-Deformable-DETR (ResNet-50, COCO-pretrained). Tracker: track queries plus a long-term memory-attention layer (true E2E, no Hungarian). Reported: DanceTrack 68.5 HOTA / 80.5 DetA / 58.4 AssA; SportsMOT 70.0 HOTA (no extra data); MOT17 58.8 HOTA; BDD100K val mTETA 53.6 `[CITED][WELL-SUPPORTED]`.[^12] Code: https://github.com/MCG-NJU/MeMOTR (82 commits, DanceTrack/SportsMOT/BDD scripts, `tools/demo.ipynb`). GPU: official train is 8 processes, “GPUs with >= 32 GB”; `--use-checkpoint` “will only take about 10 GB” `[CITED][WELL-SUPPORTED]`.[^12]

Why it works: it is the only E2E paper that is both maintained and honest about memory; DanceTrack/SportsMOT match student-filmed motion; the memory layer is a real module a student can change. Why it would not: still an 8-GPU schedule; MOT17 58.8 HOTA is *worse* than 2022 ByteTrack; Deformable Attention must compile; one A6000 can *fit* a checkpointed batch-1 run, not reproduce the paper `[CONCLUSION][SUPPORTED]`.

**6. BoT-SORT — Aharon, Orfaig, Bobrovsky, arXiv 2022.** Detector: ByteTrack YOLOX (also YOLOv7 in the repo). Tracker: BYTE + camera-motion compensation + a corrected Kalman state + optional FastReID. Reported: MOT17 BoT-SORT-ReID 80.5 MOTA / 80.2 IDF1 / 65.0 HOTA; MOT20-ReID 77.8 / 77.5 / 63.3 `[CITED][WELL-SUPPORTED]`.[^8] Code: https://github.com/NirAharon/BoT-SORT. ReID trains on one GPU via FastReID; the detector is not retrained for the paper numbers `[CITED][SUPPORTED]`.[^8]

Assign this as the *second-phase change* to ByteTrack, not as the base. It is what production stacks actually ship, but “implement BoT-SORT” is “implement three tricks,” which is a weak architecture paper and a strong engineering delta.

### Explicitly unsuitable as the base paper

BoostTrack / BoostTrack++ (2024) is the current online MOT17/MOT20 HOTA leader in the survey (66.6 / 66.4) `[CITED][WELL-SUPPORTED]`.[^1][^13] The official repo runs on *reused* Deep OC-SORT / ByteTrack weights, then linear and gradient-boosting interpolation; it documents a shape-similarity bug that the authors left in because the tuned λs fit the bug `[CITED][WELL-SUPPORTED]`.[^13] That is a results paper, not a student architecture.

MOTRv2 (CVPR 2023) is the DanceTrack headline (73.4 HOTA in the paper/survey; 69.9 in the README table) `[CITED][CONTESTED]`.[^11][^1] Five commits, 8-GPU train, and a YOLOX proposal front-end make it a poor lab base even though it is the E2E method everyone cites. MOTR itself is cleaner conceptually but 8×2080 Ti and 57.8 MOT17 HOTA `[CITED][WELL-SUPPORTED]`.[^10]

JDE, DeepSORT, and YOLO+OpenCV SORT are either superseded (JDE → FairMOT) or not architecture papers. Offline GNN trackers (SUSHI, CoNo-Link) fail the “test on a student video” requirement (they need the whole sequence). UniAD is a planning stack.

### Suggested pairing for the assignment’s two phases

`[CONCLUSION][SUPPORTED]` The assignment’s five clauses are jointly satisfied by one of two pairings, not by a single SOTA paper:

- **Pair A (default, linear / pedestrian / “compare to MOT17”).** Phase 1: ByteTrack — implement YOLOX-S/M (or run official YOLOX-X weights) + BYTE from the paper; train on MOT17 half + CrowdHuman if time; compare to the *ablation* 76.6 MOTA, not 80.3. Phase 2: replace BYTE with OC-SORT or BoT-SORT (CMC + ReID). That is a significant, published architectural change with known deltas (~0–2 HOTA on MOT17, larger on DanceTrack).
- **Pair B (joint network, one GPU, honest numbers).** Phase 1: FairMOT DLA-34 or the official YOLOv5s light model; train MOT17-only; compare to 69.8 MOTA. Phase 2: change the backbone, ReID dimension, or replace the hierarchical associator with BYTE/OC-SORT. DCNv2 risk is the main operational caveat; the YOLOv5s variant exists specifically to avoid some of that.
- **Pair C (only if student video is dance/sports and the course wants E2E).** Phase 1: MeMOTR on DanceTrack with official checkpoint + val evaluation (HOTA 68.5 is the paper number). Phase 2: memory-layer or query-interaction change. Training from scratch on one A6000 is a stretch goal, not the plan.

### Hardware fit, compressed

| Paper | Official train | Fits 1× A6000 48 GB? | What the student actually trains |
| --- | --- | --- | --- |
| ByteTrack | 8 GPU, b=48, YOLOX-X | Detector yes at smaller size/batch; BYTE has no weights | YOLOX |
| OC-SORT | Reuses dets | Yes (association is CPU) | Optional YOLOX |
| FairMOT | 2× 2080 Ti, b=12, ~30 h | Yes | Joint det+ReID net |
| CenterTrack | Example `--gpus 0,1` | Yes | Joint pair-frame net |
| MeMOTR | 8× ≥32 GB; ckpt ~10 GB | Fits batch-1 checkpointed; not the paper schedule | Full DETR tracker |
| MOTR / MOTRv2 | 8× 2080 Ti | Uncomfortable / not the recipe | Full DETR tracker |
| BoT-SORT / BoostTrack | Reuse YOLOX; optional ReID | Yes | Optional ReID |

## Audit

**Search path.** Open survey first (2024–2025 MOT surveys, MOT17/DanceTrack leaderboards, GitHub topic/star surface), then primary READMEs and one full survey HTML, then adversarial 2024 SOTA and E2E repos. Training-memory names (ByteTrack, MOTR, FairMOT, OC-SORT) were treated as leads, not the candidate set; BoostTrack++, MeMOTR, MOTIP, CoNo-Link, UCMCTrack, Hybrid-SORT, SparseTrack surfaced from the survey tables rather than from prior.

**Independence.** MOT17 TBD numbers are **not independent of ByteTrack’s YOLOX-X detections** — Adžemović states this as the table protocol.[^1] Agreement that “TBD beats E2E on MOT17” could hold even if YOLOX-X, not the associator, is doing the work. Independent axes that *do* exist: FairMOT/CenterTrack use CenterNet, not YOLOX; MOTR/MeMOTR use DETR detections; Adžemović (Belgrade, 2025 survey) vs official author READMEs vs Guan (Springer 2025) vs BoostTrack (different authors, 2024 journal). Guan was only snippet-read, so it is not used as a load-bearing second full source.

**Adversarial checks (per major claim).**

- *TBD+YOLOX is the 2024–2026 workhorse.* Tested against 2024 SOTA (BoostTrack++, ImprAsso, CoNo-Link) and E2E (MOTRv2, MOTIP, MeMOTR). Opposition found on DanceTrack (E2E wins) and on raw MOT17 HOTA (BoostTrack++ +3.4). Neither opponent is a better *lab* paper. Label: `[SUPPORTED]` for “workhorse,” not “SOTA.”
- *Students cannot approach headline MOT17 test numbers.* Tested against FairMOT’s MOT17-only row and ByteTrack’s ablation table — both papers already publish the lower numbers. Survived.
- *E2E cannot train on one 48 GB GPU.* MeMOTR’s 10 GB checkpoint path is opposing evidence that *memory* fits; the 8-process recipe is opposing evidence that *the paper run* does not. Claim narrowed to “cannot reproduce the paper schedule.”
- *YOLO+ByteTrack is a tutorial echo chamber.* Falsifiability check: if wrong, surveys and MOTChallenge tables would show a different shared detector or a different 2022 inflection. They do not. The echo chamber is real only for *blog implementations*, not for the research baseline.

**Falsifiability.** Dominance on MOT17 is empirically falsifiable (leaderboard). Suitability for a lab is only partly falsifiable (code exists, GPU recipes exist; “weeks not months” is a judgment). Student-video domain match is not testable here.

**Cross-tier corroboration.** TBD+YOLOX lock-in: survey (research) + ByteTrack/OC-SORT/BoT-SORT/BoostTrack READMEs (primary author docs). FairMOT 2-GPU train: paper text + README scripts. E2E 8-GPU: MOTR and MeMOTR READMEs independently.

**Retrieval depth.** Adžemović HTML, FairMOT HTML, and the listed GitHub READMEs were fetched and read. Guan 2025 and Ultralytics docs were not full-fidelity reads. Star-count claims from GitHub’s AI search were discarded.

**Cross-model self-review.** Unavailable — no `codex:codex-rescue` subagent in this environment.

## Premise Check

Several premises in the question do not survive.

1. **“Implement a paper’s architecture” is ambiguous, and for the field’s workhorses it is the wrong phrase.** ByteTrack/OC-SORT/BoT-SORT/BoostTrack are association *procedures* on a frozen detector. The architecture, if any, is YOLOX — which those papers did not invent. A course that wants a network will be happier with FairMOT, CenterTrack, or MeMOTR, which are not the 2024 workhorses.

2. **“Compare numbers to the paper” is systematically unfair on MOT17 test.** Headline numbers use extra detection data (CrowdHuman, Cityperson, ETHZ, MIX), linear interpolation, and per-sequence test-time thresholds. The honest comparison set is MOT17 half-val / DanceTrack val / the paper’s own ablation row.

3. **YOLO is not “only an example.”** YOLOX is the actual shared private detector of the post-2022 MOTChallenge leaderboard. Ultralytics YOLO is the practitioner default. Pretending the course is detector-agnostic will still produce a YOLO-family detector unless the instructor forces CenterNet or DETR.

4. **Student-filmed custom video is a different task from MOT17.** MOT17/20 are crowded, mostly linear pedestrians. If students film a hallway, a pitch, or a dance, DanceTrack/SportsMOT (and OC-SORT / MeMOTR) are the relevant papers; ByteTrack’s MOT17 MOTA will look fine and its IDF1/HOTA will not.

5. **“Significant architectural change” is easy in TBD and hard in E2E.** TBD’s modularity is why it is pedagogically superior: the published next papers *are* the change. E2E changes (memory layer, query interaction) are real but sit inside an 8-GPU training loop.

Better-posed question: *Which paper gives a student a named detector, a named associator, official training code that fits one 48 GB GPU, a val protocol that does not require the MOTChallenge test server, and a published follow-up that counts as a significant change?* That question’s answer is ByteTrack → OC-SORT/BoT-SORT, or FairMOT → BYTE/backbone, not “whatever is SOTA.”

## Conflicts

**MOTRv2 DanceTrack HOTA is 73.4 in the survey/paper abstract and 69.9 in the official README results table** `[CONTESTED]`.[^1][^11] The README table is unlabeled as val vs test; the survey attributes 73.4 to the published paper. Prefer the paper/survey 73.4 for “what the authors claim on test,” and do not make students match either number.

**CenterTrack MOT17 MOTA is 67.8 in the official README and 67.3 in the paper abstract** `[CONTESTED]`.[^9] Difference is small; use 67.8 as the repo-reproduced private-detection number.

**BoT-SORT MOT17 HOTA is 65.0 in the official README and 65.0 in the survey; MOTA is 80.5 (README ReID) vs 80.5 (survey) — aligned.** No conflict.

**Does E2E “win” MOT?** Quality sources disagree by *dataset*, not by error. MOT17/20: TBD wins `[CITED][WELL-SUPPORTED]`.[^1] DanceTrack: E2E wins `[CITED][WELL-SUPPORTED]`.[^1] SportsMOT: TBD wins in the survey, but the comparison is contaminated because TBD trains YOLOX on train+val while E2E trains on train only `[CITED][SUPPORTED]`.[^1] Resolve: report by domain; do not declare a global winner.

**Is FairMOT “SOTA”?** Its own README still says it “rank[s] first” on MOT17 `[CITED][WELL-SUPPORTED]` as a historical claim;[^7] that has been false since ByteTrack (2022). Treat the rank claim as outdated; treat the metrics as valid.

## Gaps

- Single-A6000 training was **inferred** from published multi-GPU recipes, not profiled. YOLOX-X batch size on 48 GB is unverified.
- Ultralytics’ default tracker and BoxMOT’s exact paper coverage were not full-page reads.
- Guan 2025 was snippet-only; a full read could add or contradict the detector-centric vs association-centric taxonomy.
- No MOTChallenge live leaderboard page was successfully fetched (search hit ResearchGate tables and the survey instead).
- VisDrone / BDD / KITTI were not surveyed at the same depth as MOT17/DanceTrack; CenterTrack and MOTR report some of those numbers, but “best paper for driving video” is under-covered.
- Hybrid-SORT’s official repo was not fetched after a wrong-URL miss; it is in the survey table (64.0 HOTA MOT17) but not ranked.
- Whether DCNv2 still builds on current PyTorch (FairMOT/CenterTrack) is a known operational risk, not re-tested.

## Sources

[^1]: Momir Adžemović. 2025. *Deep Learning-Based Multi-Object Tracking: A Comprehensive Survey from Foundations to State-of-the-Art*. arXiv:2506.13457. Retrieved from https://arxiv.org/html/2506.13457 — Type: preprint / institutional (survey)

[^2]: (Guan et al.). 2025. *Multi-object tracking review: retrospective and emerging trend*. Artificial Intelligence Review / Springer. Retrieved from https://link.springer.com/article/10.1007/s10462-025-11212-y — Type: peer-reviewed (snippet-level retrieval only)

[^3]: Yifu Zhang et al. / FoundationVision. 2022. *ByteTrack* official repository README. GitHub. Retrieved from https://github.com/FoundationVision/ByteTrack — Type: editorial / author-official

[^4]: Jinkun Cao et al. / noahcao. 2023. *OC-SORT* official repository README. GitHub. Retrieved from https://github.com/noahcao/OC_SORT — Type: editorial / author-official

[^5]: GitHub. 2026. *Topic: multi-object-tracking*. Retrieved from https://github.com/topics/multi-object-tracking — Type: community-vetted

[^6]: Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu. 2021. *FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking*. IJCV / arXiv:2004.01888. Retrieved from https://arxiv.org/html/2004.01888 — Type: peer-reviewed

[^7]: Yifu Zhang et al. / ifzhang. 2021. *FairMOT* official repository README. GitHub. Retrieved from https://github.com/ifzhang/FairMOT — Type: editorial / author-official

[^8]: Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky. 2022. *BoT-SORT* official repository README. GitHub. Retrieved from https://github.com/NirAharon/BoT-SORT — Type: editorial / author-official

[^9]: Xingyi Zhou, Vladlen Koltun, Philipp Krähenbühl. 2020. *CenterTrack* official repository README. GitHub. Retrieved from https://github.com/xingyizhou/CenterTrack — Type: editorial / author-official

[^10]: Fangao Zeng et al. / megvii-research. 2022. *MOTR* official repository README. GitHub. Retrieved from https://github.com/megvii-research/MOTR — Type: editorial / author-official

[^11]: Yuang Zhang et al. / megvii-research. 2023. *MOTRv2* official repository README. GitHub. Retrieved from https://github.com/megvii-research/MOTRv2 — Type: editorial / author-official

[^12]: Ruopeng Gao, Limin Wang / MCG-NJU. 2023. *MeMOTR* official repository README. GitHub. Retrieved from https://github.com/MCG-NJU/MeMOTR — Type: editorial / author-official

[^13]: Vukasin Stanojevic, Branimir Todorovic. 2024. *BoostTrack* official repository README. GitHub. Retrieved from https://github.com/vukasin-stanojevic/BoostTrack — Type: editorial / author-official

## Label Definitions

- **[CITED]** — Factual claim taken from a retrieved named source; footnote points at that source.
- **[SYNTHESIS]** — Claim produced by combining two or more cited facts; the inputs are named in the sentence.
- **[CONCLUSION]** — Analyst judgment over the cited evidence, not a sentence that appears in any source.
- **[HYPOTHESIS]** — Provisional, not verified in this pass; do not rely on it without checking.
- **[TRAINING DATA]** — From model memory, not a retrieved source. Not used for load-bearing claims in this report.
- **[WELL-SUPPORTED]** — Falsifiable, and either independently corroborated across sources/tiers or a close paraphrase of a primary authoritative source that was fetched and read (official README, paper HTML).
- **[SUPPORTED]** — Falsifiable and backed by quality sources, but corroboration is thin, within-tier, or one source was only snippet-read.
- **[CONTESTED]** — Quality sources disagree and the disagreement is not fully resolved.
- **[UNFALSIFIABLE]** — Not the kind of claim this research can empirically settle. Not used above.
- **[WEAKLY-SUPPORTED]** — Only anonymous/unverified sources, or all sources share a decisive dependence. Not used for major claims above.
