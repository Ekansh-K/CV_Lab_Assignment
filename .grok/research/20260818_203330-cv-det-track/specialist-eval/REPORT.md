# Research Report: Honest train-public / test-custom evaluation for student detection+tracking

## Takeaways

A student cannot put a number computed on phone videos next to a paper’s MOT17/DanceTrack/VisDrone table and call that a comparison. Serious MOT papers treat official-split metrics and extra videos as different experiments. The honest protocol is two-stage: **(A) reproduce the paper on the paper’s split, detector protocol, and training-data recipe**, then **(B) report custom videos separately** as a domain-shift / stress test, with TrackEval numbers only on a small identity-annotated subset and qualitative evidence on the rest. Most published “80 MOTA” figures are *private-detector* results trained with extra static-person images, not a tracker running on MOT Challenge’s public DPM/FRCNN/SDP boxes. Custom footage without per-frame identity ground truth cannot produce MOTA, IDF1, HOTA, or ID-switch counts; those metrics are undefined, not “approximately 95%.” The custom set is “strong enough for a lab report” if it stresses the *same object class the model was trained on*, includes crossings/occlusion/camera motion/re-entry, and has at least a few fully annotated sequences in MOT format — not if it is one easy hallway clip scored by eyeball.

Most load-bearing claims below are well-supported by official kits and primary papers. How large a custom set must be to be “enough” is a lab-standard judgment, not a published threshold.

## Findings

### What official metrics actually measure, and what they need

Multi-object tracking is jointly a detection problem, a localisation problem, and an identity-association problem. The CLEAR MOTA score subtracts false negatives, false positives, and identity switches from one, normalised by the number of ground-truth objects: \(\mathrm{MOTA}=1-\sum_t(\mathrm{FN}_t+\mathrm{FP}_t+\mathrm{IDSW}_t)/\sum_t\mathrm{GT}_t\) `[CITED][WELL-SUPPORTED]`.[^1][^2] Because FN and FP usually dwarf ID-switch counts, MOTA is dominated by the detector `[CITED][WELL-SUPPORTED]`.[^2][^3] IDF1 instead does a global one-to-one match of *trajectories* and scores identity precision/recall; it is association-heavy and can behave non-monotonically in detection `[CITED][WELL-SUPPORTED]`.[^2] HOTA was designed to sit between those poles: it is (approximately) the geometric mean of a detection Jaccard (DetA) and an association Jaccard (AssA), averaged over localisation thresholds, and is the metric TrackEval marks as recommended `[CITED][WELL-SUPPORTED]`.[^2][^4] DanceTrack adopted HOTA as its primary score for exactly this reason `[CITED][WELL-SUPPORTED]`.[^5]

All three families require the same kind of ground truth: every evaluated frame has boxes, each box has a stable identity, and a similarity threshold (typically IoU ≥ 0.5 for MOTA matching) decides true positives `[CITED][WELL-SUPPORTED]`.[^1][^2][^4] TrackEval’s MOT Challenge format is a 10-column CSV per sequence (`frame, id, left, top, width, height, conf, x, y, z`), 1-based, with `x,y,z = -1` for 2D `[CITED][WELL-SUPPORTED]`.[^6] MOT16/17 ground truth additionally stores an ignore flag, a class id, and a visibility ratio; evaluation is *only* on the pedestrian class, and distractors (sitting people, reflections, mannequins, people behind glass) are neither rewarded nor penalised `[CITED][WELL-SUPPORTED]`.[^1] Without that GT file, TrackEval cannot emit MOTA/HOTA/IDF1. A student who never annotates identities therefore has no tracking metric — only a video.

Detection-only mAP (COCO-style) is a different task. It is the right number for “did YOLOX fire on this frame?” and the wrong number for “did the tracker keep ID 3 through the occlusion.” VisDrone itself splits these: Task 1/2 are detection, Task 3 is single-object tracking, Task 4 is MOT `[CITED][WELL-SUPPORTED]`.[^7] Training a detector on COCO and then quoting COCO mAP as if it were MOTA is a category error `[CONCLUSION][WELL-SUPPORTED]`.

### Public datasets that pair with papers — and why the pairing is not transferable to phone video

The clean pairings are paper-and-split pairings, not paper-and-arbitrary-video pairings.

**MOT16/17/20** are the pedestrian MOT Challenge. MOT16 introduced consistently re-annotated sequences, public DPM detections, ignore/distractor classes, and a hidden test set so methods cannot tune on the leaderboard `[CITED][WELL-SUPPORTED]`.[^1] MOT17 reuses that video set and ships three public detectors (DPM, Faster R-CNN, SDP); a large literature still reports a separate *private detection* protocol in which the method’s own detector is allowed `[CITED][WELL-SUPPORTED]`.[^3] MOT20 is the crowded, mostly-static-camera sibling `[CITED][SUPPORTED]`.[^3][^5] These are the only numbers that may sit next to a MOT Challenge table.

**CrowdHuman, CityPersons, and ETHZ** are not tracking test sets. ByteTrack (following CenterTrack / FairMOT / JDE) trains the detector on CrowdHuman plus MOT17 — and, for the MOT17 *test* submission, also CityPersons and ETHZ `[CITED][WELL-SUPPORTED]`.[^3] A student who trains YOLOX only on MOT17 train and then compares to ByteTrack’s 80.3 MOTA is not reproducing ByteTrack.

**DanceTrack** (100 videos: 40/25/35 train/val/test, ~10× MOT17’s frame count) holds appearance nearly constant and makes motion nonlinear. Oracle detections plus IoU matching score ~98 HOTA on MOT17 val and only ~73 HOTA on DanceTrack val; adding a Re-ID model *hurts* DanceTrack `[CITED][WELL-SUPPORTED]`.[^5] Published ByteTrack is 63.1 HOTA / 80.3 MOTA on MOT17 test and 47.7 HOTA / 89.6 MOTA on DanceTrack test — MOTA stays high while HOTA and AssA collapse `[CITED][WELL-SUPPORTED]`.[^5] That is the cleanest published demonstration that “we still have high MOTA” can hide association failure, and that MOT17 competence does not transfer even to other *people* videos.

**BDD100K** is multi-class driving MOT (8 classes, 1400/200/400 train/val/test). ByteTrack reports mMOTA / mIDF1 there and notes that Kalman motion fails under large camera motion and low annotation frame-rate, so they drop Kalman and add Re-ID `[CITED][WELL-SUPPORTED]`.[^3] A phone video shot from a moving car is closer to BDD than to MOT17; a MOT17-trained Kalman tracker is expected to break.

**KITTI tracking** is another driving benchmark; TrackEval is its official 2D-box eval kit `[CITED][WELL-SUPPORTED]`.[^4] **VisDrone** is aerial: 288 clips, 10k stills, 2.6M boxes, separate DET / VID / SOT / MOT / crowd-counting tasks and its *own* MATLAB toolkits `[CITED][WELL-SUPPORTED]`.[^7] **UA-DETRAC** (vehicles, surveillance viewpoint) is historically scored with PR-MOTA, not MOT Challenge MOTA `[CITED][SUPPORTED]`.[^2] **COCO** is still-image detection and appears in MOT papers only as detector pretraining (ByteTrack initialises YOLOX from COCO weights) `[CITED][WELL-SUPPORTED]`.[^3]

A paper trained on X cannot be numerically compared on custom videos without caveats because the published scalar is a function of (dataset, split, class set, detector protocol, extra training images, input resolution, interpolation post-process, and official eval script). Changing any one of those changes the number. Domain shift from MOT17 pedestrians to “whatever the student filmed” is not a small covariate shift: DanceTrack already shows a large AssA drop *inside* the person class `[SYNTHESIS][WELL-SUPPORTED]` from cited DanceTrack oracle/benchmark tables and cited MOT16 class protocol.

### How serious papers handle “own videos”

They mostly do not use own videos as the comparison. ByteTrack’s SOTA claims are MOT17/MOT20/HiEve/BDD100K *test-server* numbers under a named protocol; qualitative figures illustrate the method; ablation uses the CenterTrack-style MOT17 half-train / half-val split, explicitly *not* the hidden test set `[CITED][WELL-SUPPORTED]`.[^3] DanceTrack’s contribution *is* a new public benchmark with a private test annotation, not a private lab tape `[CITED][WELL-SUPPORTED]`.[^5] MOT16’s reason for existing is that pre-benchmark papers were incomparable: different subsets, different detections, different eval scripts, overfit sequences `[CITED][WELL-SUPPORTED]`.[^1] TrackEval’s first-class “Evaluate on your own custom benchmark” path is: convert to an existing format (they recommend MOT Challenge), build `gt/<Seq>/gt/gt.txt` plus `seqinfo.ini` and `seqmaps`, and run it as a *named challenge*, with `--DO_PREPROC False` unless distractor preprocessing is wanted `[CITED][WELL-SUPPORTED]`.[^4][^6]

So the honest comparison is exactly the two-column design the question hypothesised: **(A) reproduce the paper metric on the paper’s test or published val split; (B) separately report custom-video qualitative results plus TrackEval on a small annotated subset.** Mixing A and B in one sentence (“76 vs 95”) is the failure mode MOT Challenge was built to stop `[CONCLUSION][WELL-SUPPORTED]`.

No primary paper retrieved here treats an unlabeled student video as a substitute for the official test set. A GitHub-search summary that “the official server only accepts public detections” is **false as a description of current practice**: ByteTrack’s main tables are private-detector results “directly obtained from the official MOT Challenge evaluation server” `[CITED][WELL-SUPPORTED]`.[^3] MOT16 already allowed other detections if they are “noted as part of the tracker’s description and also displayed in the ratings table” `[CITED][WELL-SUPPORTED]`.[^1] The constraint is labelling, not a ban.

### Minimum viable annotation for custom videos

If the student wants any MOTA/HOTA/IDF1 on custom data, they must produce MOT Challenge-style GT.

**Tooling.** CVAT exports “Bounding Box tracks” to MOT: `gt/gt.txt` with `frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility` plus `img1/` frames `[CITED][WELL-SUPPORTED]`.[^8] That is the same family of CSV TrackEval consumes; the student still needs `seqinfo.ini` (imWidth, imHeight, frameRate, seqLength, imDir, name) and a seqmap listing sequence names `[CITED][WELL-SUPPORTED]`.[^6] Label Studio and Roboflow can also emit MOT-like tracks; they are not required if CVAT is used. After export, run TrackEval’s `run_mot_challenge.py` with `BENCHMARK=<YourChallenge>` and `--DO_PREPROC False` unless ignore-class preprocessing was actually implemented `[CITED][WELL-SUPPORTED]`.[^4][^6]

**What to annotate (protocol, not vibe).** MOT16’s rules are the right lab standard: mark the *target class* the tracker is supposed to output (for a MOT17-trained pedestrian system: upright walking/standing people, including cyclists if the paper did); keep a tight box that contains all of the person; start as early and end as late as the location is unambiguous; **keep the same ID through occlusions if the path can be determined**; assign a new ID after a long ambiguous disappearance or a leave-and-reenter of the field of view; use an ignore/distractor class for sitting people, reflections, posters, so the eval neither rewards nor punishes them `[CITED][WELL-SUPPORTED]`.[^1] DanceTrack’s rule is the same identity persistence through full occlusion `[CITED][WELL-SUPPORTED]`.[^5] Interpolating every *N*th keyframe in CVAT is acceptable *only if the exported GT has a box on every evaluated frame*. Evaluating MOTA on keyframes and pretending it is video MOTA is not.

**Budget that is lab-viable and not a fake benchmark.** MOT16 train is 7 sequences, 5,316 frames, 512 identities `[CITED][WELL-SUPPORTED]`.[^1] A student will not match that. A defensible lab minimum is: **five filmed sequences, three of them fully annotated**, each 15–30 s at 10–15 fps (roughly 200–450 boxes-per-sequence if 4–8 people are visible), identity-consistent, exported to TrackEval. That is enough to compute HOTA/MOTA/IDF1/IDs *and* to show per-sequence variance. It is not enough to claim a new SOTA. Two further sequences can stay qualitative (failure gallery). Annotating one 20-second easy clip of a single walking person is *not* a test set `[CONCLUSION][SUPPORTED]`.

**What to film so the tracker is actually stressed.** Copy the axes MOT16 and DanceTrack already treat as the point of a benchmark: static vs moving camera, viewpoint (high / eye-level / low), lighting (day, indoor, backlight), density, occlusion, and — the DanceTrack lesson — similar appearance plus crossing trajectories `[CITED][WELL-SUPPORTED]`.[^1][^5] For a pedestrian system, a useful five-clip slate is: (1) sidewalk, static phone, moderate crowd; (2) same location, handheld camera motion; (3) two people in similar clothes crossing and occluding; (4) person leaves the frame and re-enters; (5) indoor low light or strong backlight. If the student instead wants cars, they should have trained on KITTI, BDD100K, UA-DETRAC, or VisDrone MOT — not MOT17 `[CONCLUSION][WELL-SUPPORTED]`.

### Which metrics to report, and which are meaningless without GT

| Quantity | Needs identity GT? | Use on official split | Use on custom videos |
|---|---|---|---|
| HOTA, DetA, AssA, LocA | Yes | Primary tracking number | Only on annotated subset; label as custom |
| MOTA, MOTP, FP, FN, IDSW, FM, MT/ML | Yes | Report beside HOTA; do not rank by MOTA alone | Same; MOTA-without-AssA is how DanceTrack looks “easy” |
| IDF1 / IDP / IDR | Yes | Association view | Same |
| Detection mAP / AP50 on sampled frames | Boxes only, no IDs | Detector ablation | Allowed as *detection* transfer, never as tracking |
| FPS / latency | No | Fine | Fine |
| “Accuracy %” of frames that look good | No | Never | Never as a MOTA proxy |
| Manual ID-switch tally on a 10 s clip | Informal | Not a paper number | Allowed if labelled “manual, not TrackEval” |

HOTA’s authors show MOTA correlates 0.96 with DetA and only 0.46 with AssA on MOT Challenge, while IDF1 does the reverse (0.97 with AssA, 0.58 with DetA); HOTA sits in between `[CITED][SUPPORTED]`.[^2] DanceTrack’s table is the worked example: MOTA ~87–92 with HOTA ~42–55 `[CITED][WELL-SUPPORTED]`.[^5] A lab report that quotes only MOTA on an easy custom clip is performing that failure on purpose.

### Failure modes of “we got 95% on our video vs the paper’s 76 MOTA”

These are the apples-to-oranges moves the literature already warns about, rewritten as student mistakes.

1. **Different test distribution.** Paper number is MOT17/20 test (crowded, mixed static/moving cameras, night, 14–30 fps). Student number is a short easy clip. MOT16 exists because even PETS S2L1 was already >90% with good detections and was overfit `[CITED][WELL-SUPPORTED]`.[^1]
2. **Public vs private detections.** A 2016-era public-detection MOTA in the 30s and ByteTrack’s private-detection 80.3 MOTA are not the same leaderboard `[CITED][WELL-SUPPORTED]`.[^1][^3]
3. **Hidden extra training data.** ByteTrack’s MOT17 test model sees CrowdHuman + CityPersons + ETHZ, 1440×800, Mosaic/Mixup, 80 epochs `[CITED][WELL-SUPPORTED]`.[^3] Omitting that and quoting 80.3 as the “paper baseline you should hit on a laptop” is dishonest.
4. **Train/test leakage.** Ablations belong on MOT17 *half val* (CenterTrack split), not on the training sequences used to fit the detector, and not on the hidden test set `[CITED][WELL-SUPPORTED]`.[^3][^5]
5. **MOTA as a detection score.** High custom MOTA + unreported AssA/IDF1/IDs is the DanceTrack pattern `[CITED][WELL-SUPPORTED]`.[^2][^5]
6. **Class mismatch.** MOT Challenge scores pedestrians only and ignores distractors `[CITED][WELL-SUPPORTED]`.[^1] Counting cars, backpacks, or sitting people on a custom video, or training on COCO’s 80 classes and testing as if it were MOT17, changes the task.
7. **Metric invented at test time.** “95% of frames looked correct,” “the box stayed on the person,” or detection mAP renamed as MOTA. None of these is MOTA `[CONCLUSION][WELL-SUPPORTED]`.
8. **Eval-script drift.** MOT16’s reason for a central server was conflicting IoU thresholds (25% vs 50%), different ID-switch definitions, and private eval scripts `[CITED][WELL-SUPPORTED]`.[^1] Use TrackEval; do not reimplement MOTA in a notebook.
9. **Post-process mismatch.** Many MOT17 submissions interpolate lost tracks. Comparing a raw tracker to an interpolated paper number is another silent protocol change `[HYPOTHESIS]` (common in reproductions; confirm against the specific paper’s official description before asserting it).
10. **Improvement claimed on the wrong split.** Fine-tuning on the custom videos and then quoting the new custom score against the *paper’s MOT17 number* attributes domain adaptation to architecture. The A-set must be re-measured after any change, so the student can say whether they improved transfer, wrecked the reproduction, or both `[CONCLUSION][SUPPORTED]`.

### Recommended evaluation protocol for this assignment

**0. Choose the paper first, then the public train set.** Do not pick MOT17 and a VisDrone paper, or COCO and a MOT17 table. Suggested default for a pedestrian lab: implement a tracking-by-detection method whose paper reports MOT17 private-detection HOTA/MOTA/IDF1 (ByteTrack is the documented example). Train the detector the way the paper did, or *explicitly* train a reduced recipe and treat the paper’s 80.3 as an *upper bound you are not claiming to match* `[CONCLUSION][WELL-SUPPORTED]`.

**1. Stage A — reproduction (the only column that may face a paper table).**

- Data: MOT17 train. For development, use the first-half-train / last-half-val split ByteTrack and DanceTrack both attribute to CenterTrack `[CITED][WELL-SUPPORTED]`.[^3][^5]
- Extra data: either include CrowdHuman (and disclose it) or omit it (and do not compare to the extra-data row).
- Protocol tag on every number: `{dataset} / {split} / {public|private detections} / {extra data list} / {TrackEval HOTA+CLEAR+Identity}`.
- Metrics: HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN. One TrackEval command, not a hand formula `[CITED][WELL-SUPPORTED]`.[^4][^6]
- If the course cannot submit to the hidden test server, compare to the paper’s *validation* numbers when the paper publishes them, or state that test-set comparison is out of scope. Do not compare val to test.

**2. Stage B — custom transfer (a different table).**

- Film the five-clip slate above, same class as Stage A.
- Annotate three clips in CVAT as MOT tracks; export; add `seqinfo.ini` + seqmaps; evaluate with TrackEval `--DO_PREPROC False` unless distractors were annotated to MOT16 rules `[CITED][WELL-SUPPORTED]`.[^4][^6][^8]
- Report per-sequence and pooled HOTA/MOTA/IDF1/IDs. Put a still of each failure mode (ID swap at a cross, lost track in blur, false positive on a poster) in a figure.
- Unannotated clips: qualitative only.

**3. If Stage B is worse than Stage A (it will be).** That is the assignment’s improvement loop, and it is valid only if the student changes one thing at a time and re-runs *both* stages:

- Data: fine-tune the detector on a **held-out** custom train clip that is not in the custom test three; or add CrowdHuman if Stage A omitted it.
- Association: IoU+Kalman is the MOT17-friendly baseline and the DanceTrack-weak one `[CITED][WELL-SUPPORTED]`.[^5] BYTE’s low-score second match, or a stronger motion model, is a legitimate architecture change.
- Do not fine-tune on the custom test clips.

**4. Comparison paragraph that does not overclaim.** Use this shape, filling in the student’s actual numbers:

> On the MOT17 validation half-split, under the private-detection protocol and training on {data recipe}, our reimplementation reaches HOTA *h* / MOTA *m* / IDF1 *i*. The paper reports HOTA *H* / MOTA *M* / IDF1 *I* on the MOT17 **test** set with {their data recipe}; these splits are not interchangeable, so we do not claim to have matched or beaten the paper. On a separately annotated custom set of *N* pedestrian sequences (*F* frames, filmed at {conditions}), the same weights obtain HOTA *h2* / MOTA *m2* / IDF1 *i2*. The drop is concentrated in AssA / ID switches under {crossing / camera motion / lighting}, which is consistent with known MOT17→hard-motion transfer (DanceTrack). After {fine-tune / BYTE / …} the custom HOTA is *h3*; the MOT17 val HOTA is *h4*. We interpret this as {better transfer / worse reproduction / both}.

That paragraph is the product. A single “we got 95% vs 76 MOTA” sentence is not.

## Audit

**Search path.** Open surveys for MOT evaluation protocol, TrackEval, DanceTrack, ByteTrack training, VisDrone, and CVAT MOT export. Firecrawl search after the first batch returned HTTP 429; remaining discovery used known official URLs. Full or near-full reads: TrackEval official MOTChallenge kit and root README, ByteTrack ar5iv, HOTA arXiv HTML (through the metric definition and MOTA/IDF1 critique), DanceTrack arXiv HTML (full), MOT16 ar5iv (through evaluation and annotation protocol), CVAT MOT format page, VisDrone-Dataset README. Failed: `motchallenge.net/data/MOT17/` (HTTP error), Codabench MOT17 (JS shell, no protocol text). Cross-model Codex review was unavailable in this environment (no `codex:codex-rescue` subagent).

**Independence.** HOTA, TrackEval, and the MOT Challenge official kit share authors (Luiten; Leal-Taixé also spans MOT16 and HOTA). ByteTrack and DanceTrack share ByteDance/HKU authors. That cluster could in principle agree on HOTA-as-primary because they built it. Two independence breaks still hold: (1) CVAT’s MOT export is a tooling document from a different organisation and matches the MOT16 CSV independently; (2) DanceTrack is written as a *critique* of MOT17-style success, so its “MOTA stays high while association dies” result is adversarial to the usual SOTA narrative, not an echo. VisDrone is a third lab and a third toolkit. The MOTA *formula* is one 2008 definition reused everywhere — agreement on the equation is not multi-study replication.

**Adversarial checks (per major claim).**

- “You may not numerically compare custom video to a paper MOTA.” Searched papers for a combined table. Not found. TrackEval models custom data as a separate challenge. **Tested, no opposition found** among quality sources; this is consensus in the MOT establishment, not an independent replication outside it.
- “Official MOT forbids private detections.” A secondary web-search summary claimed this. ByteTrack’s server-reported private-detector tables and MOT16 §III-B contradict it. **Tested, opposition found; primary sources win.** The live claim is: both protocols exist and must be labelled.
- “MOTA is a sufficient single score.” HOTA and DanceTrack oppose. **Tested, opposition found** — report the HOTA family as well.
- “Minimum three annotated sequences is enough.” No paper states a student-lab minimum. **Not testable as a universal**; labelled as a lab recommendation.

**Falsifiability.** Metric-requirement claims are definitional and practically checkable (TrackEval without `gt.txt` fails). ByteTrack’s 80.3/77.3/63.1 and extra-data recipe are checkable against Table 4 and §4.1. DanceTrack’s MOTA-vs-HOTA split is checkable against Tables 2–3. “Custom cars ≠ MOT17 pedestrians” is a domain claim already illustrated by an in-class shift (DanceTrack) plus MOT16’s class protocol. “Strong enough for a lab report” is a value standard; we do not assign it a support star as an empirical law.

**Cross-tier corroboration.** Peer-reviewed / arXiv benchmarks (MOT16, HOTA, DanceTrack, ByteTrack) + official eval kit (TrackEval) + independent annotation software (CVAT) all encode the same rule: numbers come from identity GT in a stated format on a stated split.

## Premise Check

The question’s useful premise is right: a valid experiment and an honest paper comparison are different jobs, and custom phone video is a hostile test distribution. Three premises do not hold as stated.

1. **“Compare with the paper” is not one number.** The assignment language invites a single delta. The literature’s unit of comparison is a *protocol* (split × detector × extra data × eval kit). A better question is: “What two tables should the report contain, and which sentences are forbidden?”
2. **“Detection+tracking” is not one task.** MOT, MOTS, SOT (VisDrone Task 3), and COCO detection are different output spaces and metrics `[^7]`. A student who trains YOLO on COCO and runs a Kalman associator has not implemented “the MOT17 paper” unless that is actually the paper’s recipe.
3. **“Strong enough for a lab report” has no published threshold.** Treating MOT16’s 7-sequence train set as the bar would make the assignment impossible; treating one unannotated clip as enough would make it vacuous. The operational standard used here is: same class, stressed recording plan, TrackEval on a few fully annotated sequences, qualitative gallery on the rest.

A better-posed question, which this report answers: *How do you separate reproduction from transfer so that neither number is a lie?*

## Conflicts

**Private detections: unofficial vs first-class.** A secondary web-search summary described private-detector MOT17 results as non-standard and unranked. ByteTrack’s main SOTA tables are private-detector results taken from the official server, and MOT16 already required only that non-provided detections be disclosed `[^1][^3]`. **Better supported: both protocols are valid; mixing them is not.**

**Which scalar ranks trackers.** MOT Challenge historically ranked on MOTA `[^1][^2]`; HOTA’s authors, TrackEval, DanceTrack, and KITTI/BDD practice treat HOTA as the balanced ranking metric `[^2][^4][^5]`. **Better supported for a 2020s lab report: lead with HOTA, still print MOTA and IDF1.** They diagnose different errors; they are not interchangeable.

**Does appearance help?** MOT17-style papers treat Re-ID as a default upgrade. DanceTrack’s oracle and YOLOX-association tables show Re-ID can *lower* HOTA when clothes are uniform `[^5]`. **Not a contradiction if scoped:** appearance helps when identities are visually distinct (MOT17, BDD); it is not a universal.

No source retrieved here defends comparing an unlabeled custom-video “accuracy %” to a paper MOTA.

## Gaps

- MOT Challenge’s live website and Codabench text were not readable (HTTP / JS). Protocol details therefore come from the official TrackEval kit, MOT16 paper, and papers that cite the server — not from a 2026 screenshot of the leaderboard UI.
- UA-DETRAC, KITTI, and BDD format documents were not independently scraped; pairing claims for those sets rest on TrackEval’s benchmark list, HOTA’s related-work survey, and ByteTrack’s BDD section. Fine for “do not compare MOT17 to KITTI,” thinner for writing a KITTI-specific protocol.
- No primary source states a student annotation budget. The 3-of-5 sequence recommendation is a lab-standard conclusion, not a cited norm.
- Interpolation, GSI, and other test-time track-completion tricks are widely used in MOT17 submissions; this pass did not read a systematic survey of who interpolates. Flagged as a hypothesis to check against the chosen paper.
- Cross-model self-review (Codex) was unavailable.

## Sources

[^1]: Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. 2016. *MOT16: A Benchmark for Multi-Object Tracking*. arXiv:1603.00831. Retrieved from https://ar5iv.labs.arxiv.org/html/1603.00831 — Type: peer-reviewed / preprint of the official MOT16 benchmark paper

[^2]: Jonathon Luiten, Aljoša Ošep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. 2020. *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking*. International Journal of Computer Vision / arXiv:2009.07736. Retrieved from https://arxiv.org/html/2009.07736 — Type: peer-reviewed

[^3]: Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. 2021. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. arXiv:2110.06864. Retrieved from https://ar5iv.labs.arxiv.org/html/2110.06864 — Type: peer-reviewed / preprint (ECCV-line MOT paper)

[^4]: Jonathon Luiten and Arne Hoffhues. 2020. *TrackEval*. GitHub. Retrieved from https://github.com/JonathonLuiten/TrackEval — Type: official evaluation code / community-vetted primary kit

[^5]: Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. 2022. *DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion*. arXiv:2111.14690. Retrieved from https://arxiv.org/html/2111.14690 — Type: peer-reviewed / preprint of the official DanceTrack paper

[^6]: Jonathon Luiten and MOTChallenge maintainers. 2020. *MOTChallenge Official Evaluation Kit — Multi-Object Tracking — MOT15, MOT16, MOT17, MOT20*. TrackEval docs. Retrieved from https://raw.githubusercontent.com/JonathonLuiten/TrackEval/master/docs/MOTChallenge-Official/Readme.md — Type: official evaluation documentation

[^7]: VisDrone / AISKYEYE team. 2019–2021. *VisDrone-Dataset* (README; cites Zhu et al., IEEE TPAMI 2021, “Detection and tracking meet drones challenge”). GitHub. Retrieved from https://github.com/VisDrone/VisDrone-Dataset — Type: official dataset documentation

[^8]: CVAT.ai. 2024–2026. *MOT format*. CVAT documentation. Retrieved from https://docs.cvat.ai/docs/dataset_management/formats/format-mot/ — Type: official product documentation / editorial review

## Label Definitions

- **[CITED]** — The sentence restates something in a source that was retrieved and read for this report, and a footnote points at that source.
- **[SYNTHESIS]** — Combines two or more cited facts into a claim no single source states verbatim (inputs named nearby).
- **[CONCLUSION]** — The investigator’s judgment from the cited evidence; not a quote.
- **[HYPOTHESIS]** — Plausible, not verified against a primary source in this pass; do not rely on it without checking the chosen paper.
- **[WELL-SUPPORTED]** — Falsifiable, and either a close paraphrase of a primary source that *defines or exhibits* the fact (official kit, official format, the paper’s own table/protocol) that was read, or independent sources that agree across labs or document types.
- **[SUPPORTED]** — Falsifiable and backed by quality sources, but corroboration is thinner (single paper’s related-work claim, or a secondary description of a dataset not fully re-read).
- **[UNFALSIFIABLE]** — Not the kind of claim this research setting can empirically settle (not used as a load-bearing finding above; the “lab-report enough” bar is treated as a stated standard instead).
