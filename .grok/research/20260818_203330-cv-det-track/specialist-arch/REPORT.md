# Research Report: Lab-scale architecture changes after a published detection+tracking baseline

## Takeaways

The follow-up literature after SORT / DeepSORT / FairMOT / ByteTrack / MOTR does **not** treat “a new neck” as the dominant upgrade. What repeatedly moves association metrics is (i) using low-score detections instead of throwing them away, (ii) repairing Kalman-filter error during occlusion and non-linear motion, (iii) compensating camera motion on handheld or vehicle-mounted video, and (iv) adding appearance only when identities are visually distinct. True *layer* changes that papers actually ablate are narrower: a decoupled classification/regression head, a joint-detection-and-embedding (JDE) ReID branch of about 128 dimensions, multi-scale fusion (FPN/DLA/P2) rather than a bigger backbone, and extra high-resolution detection scales for small objects.

For this lab, the assignment’s “significant change in the architecture (layers)” and the metric-moving literature are slightly misaligned. A defensible report therefore pairs **one real layer intervention** with **one association-module change** that targets a failure mode visible on the student’s own video. Dataset-only fine-tuning and hyperparameter-only work do not satisfy the assignment; ReLU→SiLU, a Kalman Q/R tweak, or swapping SiLU for GELU do not either. Training MOTR / MOTRv2 from scratch on one A6000 is the other failure mode — too large, not because the idea is bad.

On a single RTX A6000 48 GB the comfortable recipes are: Ultralytics YOLO11s/m or YOLOX-s/m/l at 640 with AMP and `batch=-1` (≈60% VRAM) or `freeze=10` two-stage fine-tune; YOLOX-X at ByteTrack’s 1440×800 with per-GPU batch 6–12 (the official recipe is total batch 48 on 8 GPUs); a 128-d JDE head or a FastReID SBS-50 appearance tower; P2 four-scale YOLO at 640 for s/m/l. Uncomfortable or out of scope: 8-GPU MOTR-scale video transformers from scratch, and matching published MOT17 *test* numbers without the paper’s mix of CrowdHuman + MOT + CityPersons + ETHZ and the private-detection protocol.

Most claims below are supported by primary papers with ablations plus official training docs. The main contested point is appearance: ReID helps MOT17-style pedestrians and hurts DanceTrack-style uniforms. Cross-model self-review (Codex) was unavailable in this environment.

## Findings

### What follow-up papers actually change

An open survey of MOT follow-ups (association, ReID, neck, low-score recovery, temporal fusion) did not surface a wave of novel necks as the standard improvement on a known baseline. The cluster that appears independently across ByteTrack, OC-SORT, Hybrid-SORT, BoT-SORT, FairMOT, and YOLO11-JDE is: **keep more detections, fix the motion model, optionally add appearance, optionally add a small homogeneous head.** `[SYNTHESIS][WELL-SUPPORTED]` from the six primary papers below, which share a SORT-family *paradigm* but differ in institution (HUST/ByteDance, CMU, Dalian, Tel Aviv, Microsoft/HUST, Barcelona/Aalborg) and in whether the change is trained or training-free.

ByteTrack’s BYTE association is the cleanest documented “use every box” change. High-score detections are matched first; unmatched tracklets are then matched to *low-score* boxes with IoU (not ReID), so occluded true positives are recovered and unmatched low-score boxes are discarded as background. `[CITED][SUPPORTED]` Zhang et al. report that applying BYTE to nine existing trackers raises IDF1 by about 1–10 points; on MOT17 validation, CenterTrack goes from 64.2 to 74.0 IDF1 and 528 to 144 ID switches, and SORT goes from 74.6/76.9/291 (MOTA/IDF1/IDs) to 76.6/79.3/159.[^1] They explicitly warn that ReID features on low-score boxes are unreliable because those boxes are occluded or blurred.[^1] That is a module change, not a layer change, but it is the single most replicated “architecture of the tracker” idea in the 2022–2024 SORT line.

OC-SORT attacks a different failure of the same Kalman pipeline: during occlusion SORT does a “dummy update” that trusts the linear prediction, so velocity noise accumulates as \(O(T^2)\) in position. Observation-centric re-update (ORU) backfills a virtual trajectory between the last and the re-associated observation; observation-centric momentum (OCM) adds a direction-consistency term from *observations*, not filter estimates. `[CITED][SUPPORTED]` Cao et al. present this as still simple, online, and real-time (they quote 700+ FPS for association on CPU given off-the-shelf detections) and show the largest relative gains on DanceTrack, where motion is non-linear.[^2]

Hybrid-SORT then argues that even IoU + appearance (“strong cues”) become ambiguous under clustering, and adds *weak* cues: tracklet confidence (TCM), height-modulated IoU (HMIoU), and a more robust corner-based velocity term (ROCM). `[CITED][SUPPORTED]` On DanceTrack-val their component table is unusually clean: 53.1 HOTA baseline → 53.7 with ROCM → 57.7 with TCM → 59.3 with HMIoU → 63.0 with an extra ReID model, while TCM costs 0.7 FPS and ReID drops FPS from 27.8 to 15.5.[^3] TCM plugged into SORT, DeepSORT, MOTDT, and ByteTrack consistently helps, with the largest jumps on DanceTrack (e.g. DeepSORT +4.9 HOTA) and only tenths of a HOTA point on MOT17.[^3] That pattern is important for the lab: **the same module can look like a leaderboard win on dance/sports and a rounding error on MOT17 pedestrians.**

BoT-SORT is the camera-motion and box-state paper. `[CITED][SUPPORTED]` Starting from a reimplementation of ByteTrack on MOT17 val (77.66 MOTA / 79.77 IDF1 / 67.88 HOTA), changing the Kalman state from aspect-ratio to explicit width/height is almost cosmetic (+0.24 HOTA); adding affine camera-motion compensation (CMC) via sparse optical flow + RANSAC is the load-bearing jump (78.31 / 81.51 / 69.06); adding a separate FastReID appearance model is a further +0.06 HOTA on that split.[^4] Their ReID-only matcher *loses* badly to IoU (73.7 MOTA vs 78.4). The fusion they keep is not a learned λ-sum but a masked min(IoU, cosine).[^4] For a handheld student camera, CMC is the intervention with the best documented before/after on a moving-camera MOT17 sequence (their cMOTA plot on MOT17-13).

On the *layer* side, two independent JDE papers agree on the same small head. FairMOT adds a homogeneous 128-d ReID map on an anchor-free CenterNet/DLA-34 and treats detection and identity as equal tasks, because anchor-based one-shot trackers bias the network toward boxes, share features that detection and ReID do not want, and over-parameterize identity (512–1024-d). `[CITED][SUPPORTED]` Center-sampled 128-d features beat ROI-Align and positive-anchor sampling on IDF1 and ID switches; multi-layer fusion (FPN/DLA/HRNet) beats simply using ResNet-50; they train batch 12 at 1088×608 for 30 epochs in about 30 hours on two RTX 2080 Ti GPUs.[^5] YOLO11-JDE, five years later on Ultralytics YOLO11s, adds almost the same thing: two 3×3 convs + a 1×1 embedding, 128-d, trained with triplet loss and Mosaic so identity labels are optional. `[CITED][SUPPORTED]` Their sequential ablations pick 128-d over 64/256, unit loss weight, and *self-supervised* identity (CrowdHuman + MOT17 boxes without IDs) over extra labeled ReID sets; a 100-epoch run uses batch 64 at 1280.[^6] On MOT17-test they are detection-limited (HOTA 56.6, MOTA 65.8) compared with YOLOX-X JDE systems, which is the honest lab expectation if the student starts from YOLO11s rather than YOLOX-X.[^6]

YOLOX’s own roadmap is the canonical “this is a layer change and here is the AP” table. `[CITED][SUPPORTED]` Replacing the coupled YOLOv3 head with a lite decoupled head (1×1 to 256, then parallel 3×3 stacks for cls and reg) is +1.1 COCO AP (38.5 → 39.6) and is required for their optional end-to-end variant (coupled end-to-end drops 4.2 AP; decoupled drops 0.8).[^7] Anchor-free, center sampling, and SimOTA are larger, but they are training/assignment changes, not new modules a student can draw as a block diagram. If the baseline is already YOLOX, YOLOv8, or YOLO11, **the head is already decoupled** — repeating YOLOX’s trick is not a change.

Extra high-resolution detection scale (P2 / stride-4) is the other genuine layer intervention. Ultralytics documents a YOLOv8-p2 YAML whose Detect head consumes P2–P5, and community plus small-object papers describe a fourth head on the P2 map so that a tiny target is not collapsed below ~8×8.[^8][^9] I did **not** find a MOT17 ablation of P2 of the same quality as BYTE or TCM; treat P2 as well-motivated for *small-object detection* and only indirectly for tracking (fewer FNs → higher MOTA, not necessarily fewer ID switches).

What is *not* dominant: swapping activations; adding a generic CBAM/SE block “because attention”; training a video transformer matcher from scratch. MOTRv2 still bootstraps from a YOLOX detector trained on 8× V100 and then trains MOTR on 8 GPUs; a related report quotes MOTR at ~96 hours on 8× 2080 Ti.[^10][^11] That is a thesis-scale recipe, not a lab one.

### The independence problem on MOT17 numbers

ByteTrack, OC-SORT, Hybrid-SORT, and BoT-SORT often **share the same YOLOX-X detections**. `[SYNTHESIS][WELL-SUPPORTED]` from the four papers’ experiment sections, which say so explicitly.[^1][^2][^3][^4] Agreement that “Hybrid-SORT beats OC-SORT by 0.4 HOTA on MOT17-test” is therefore agreement about an *association cost*, not an independent replication of a detector. DanceTrack is the better stress test for association (easy detection, hard IDs). MOT17 is partly saturated: Hybrid-SORT’s authors and the DanceTrack paper both say MOT17/20 are small and mostly linear, so association gains shrink to a few tenths of HOTA.[^3][^12]

Falsifiability check on “everyone agrees BYTE/OC-SORT/CMC work”: if they were wrong, the sources would still agree because they share SORT code ancestry and often the same detector weights. That is a real dependence. What saves the claims is (a) BYTE’s *cross-tracker* table (nine different association stacks, same idea), (b) Hybrid-SORT’s *cross-tracker* TCM/HMIoU tables, and (c) DanceTrack’s oracle experiment, which does not use those trackers at all and still shows linear IoU succeeding on MOT17 and failing on dance.[^1][^3][^12] I treat the directional claims (low-score recovery helps occlusion; observation-centric motion helps non-linear motion; CMC helps camera motion; appearance is conditional) as well-supported, and the exact MOT17-test deltas as only supported.

### Domain gap: what will actually move *custom student video*

The gap between a paper’s MOT17-test MOTA and a student’s campus/sports clip is usually not a missing layer. `[CONCLUSION][SUPPORTED]` It is some mix of: COCO or MOT-pedestrian weights on the wrong classes; no CrowdHuman-scale crowd pretraining; a different detector size than YOLOX-X; a different input resolution; a different score threshold; camera ego-motion the Kalman filter was never told about; and an eval protocol that is not MOTChallenge’s private-detection server. ByteTrack’s own test recipe is 80 epochs on MOT17+CrowdHuman+CityPersons+ETHZ at 1440×800, not “COCO YOLOX-X + SORT.”[^1][^13]

Map failure mode → intervention, rather than intervention → hope:

- **Occlusion / fragmented tracks / disappearing IDs on pedestrians.** BYTE’s second association. Measure IDF1, ID switches, mostly-tracked, and a count of recovered low-score TPs (Zhang et al. plot this).[^1]
- **Non-linear motion (sports, dance, sudden turns).** OC-SORT ORU+OCM or Hybrid-SORT TCM+HMIoU. Hybrid-SORT’s +4 HOTA from TCM is on DanceTrack-val, not MOT17.[^2][^3]
- **Handheld shake, walking cameraman, panning.** BoT-SORT CMC. Their MOT17-13 cMOTA drop during a right turn is the teaching figure.[^4]
- **ID switches among people who look different** (street clothes, not a team). JDE 128-d head (FairMOT / YOLO11-JDE) or a separate FastReID tower. Measure AssA/IDF1 *and* detection AP, because FairMOT’s whole point is that the two tasks fight.[^5][^6]
- **ID switches among people who look the same** (jerseys, dance costumes, lab coats). Do **not** add ReID. DanceTrack’s oracle with ground-truth boxes finds that adding appearance *lowers* HOTA from 72.8 (IoU only) to 59.7 (appearance+IoU+motion); DeepSORT loses to SORT on the same YOLOX boxes (45.8 vs 47.8 HOTA).[^12] Hybrid-SORT still gains here from *confidence and height*, not from clothes embeddings.[^3]
- **Missed small / distant objects.** P2 head or simply train at 1280–1440. Measure AP\(_S\) and FN, then MOTA. Tracking-only metrics will move only if the misses were the bottleneck.
- **Wrong classes** (cars, bikes, balls, a lecturer + slides). This is a detector class/head and fine-tune problem. A ReID branch trained on pedestrians will not invent a bicycle identity space.

Adversarial sourcing on appearance was successful: the strongest opposing view is not a blog post, it is DanceTrack’s design thesis and its oracle table.[^12] A claim that “adding ReID is the lab architecture change” that ignores clothing similarity is not defensible.

### A6000 48 GB: what is actually trainable

Official recipes, not folklore:

- **ByteTrack / YOLOX-X MOT recipe.** Paper and GitHub agree: 8 devices, total batch 48, FP16, COCO-pretrained YOLOX-X, 1440×800, mix datasets. That is **6 images per GPU**.[^1][^13] An A6000 48 GB is larger than a 16 GB V100 and comparable to or larger than a 32 GB V100, so the *per-GPU* recipe fits; the student cannot match wall-clock of 8 GPUs but can train the same model. Expect roughly 8× longer than the paper’s ~12 hours if they keep batch 6, or use gradient accumulation to fake total batch 48.
- **YOLOX COCO recipe.** Default batch 128 on “typical 8-GPU devices” (16/GPU) at multi-scale 448–832, 300 epochs, from scratch after Mosaic/MixUp.[^7] On one A6000, YOLOX-L/X at 640 with AMP and batch 16–32 is the realistic COCO-style fine-tune; 300-epoch from-scratch X is a long weekend, not an afternoon.
- **FairMOT.** Batch 12 at 1088×608, two 2080 Ti (11 GB), 30 hours.[^5] Trivial on 48 GB; a student can raise batch and add the ReID branch without freezing.
- **YOLO11-JDE.** Ablations at batch 32 / 640 / 30 epochs; final model batch 64 / 1280 / 100 epochs on YOLO11s.[^6] That final batch is the most memory-hungry JDE recipe in this set and is still in 48 GB territory for a ~9–10 M parameter model (they emphasize <10 M params).
- **Ultralytics first-party knobs.** `batch=-1` targets ~60% CUDA memory; `batch=0.70` sets the fraction; single-GPU training retries OOM by halving batch; `freeze=10` (or a list of layer indices) freezes the first N layers; the fine-tune guide’s two-stage pattern is freeze backbone ~20 epochs then unfreeze at lower `lr0`.[^14][^15] This is the correct A6000 workflow for “I added a head, I should not wreck COCO features.”
- **Separate ReID (BoT-SORT / FastReID SBS-50).** Appearance is a crop-level ResNet; 48 GB is not the constraint. The constraint is annotation (or using a pedestrian ReID checkpoint that will be wrong on vehicles/uniforms).
- **MOTR-class.** 8-GPU training is the published unit.[^10][^11] A6000 can *fine-tune a small frozen-backbone decoder* only if the student already has a working MOTR checkpoint; it cannot honestly reproduce MOTRv2.

P2 costs activation memory (stride-4 maps). On 48 GB, YOLO11s/m-p2 at 640 is comfortable; YOLO11x-p2 at 1280 is the first recipe I would call tight. I did not measure this card; those statements are scaled from official per-GPU batches and Ultralytics autobatch, not from an A6000 profiler. `[CONCLUSION][SUPPORTED]`

### Menu of five interventions (pick one layer + one association)

Each item is tied to a failure mode, a published ablation, an A6000 recipe, and a metric that can move on a *custom* split. None is a novel architecture invented here.

**1. JDE appearance head (true layers).** Add a 128-d embedding branch on the existing decoupled head, as in YOLO11-JDE (two 3×3 + 1×1, no BN on the last conv) or FairMOT’s center map.[^5][^6] Train with triplet + Mosaic so CrowdHuman and unlabeled custom stills are enough; do not start from 512-d classification ReID. **Use when** student video has distinct clothing and ID switches after occlusion. **A6000:** YOLO11s or YOLOX-s/m, `imgsz=640`, batch 16–32 or `batch=-1`, optional `freeze=10` for 20 epochs then full train 50–100 epochs; keep ReID loss weight ≤ 1 so detection AP does not collapse (both papers see this).[^5][^6] **Measure:** IDF1, IDs, HOTA-AssA on a held-out custom clip; detection mAP on the same frames so a detection regression is visible. **Do not use** on team sports or dance.

**2. Extra P2 detection scale (true layers).** Copy the official YOLOv8-p2 / four-head Detect pattern so the neck upsamples to stride 4 and a fourth head sees small objects.[^8] **Use when** the baseline misses distant people, balls, or cars (high FN, not high IDs). **A6000:** YOLO11s/m, 640, AMP, autobatch; if VRAM spikes, freeze backbone and drop to batch 8. **Measure:** COCO-style AP\(_S\) or a size-stratified miss rate, then MOTA/FN. Expect little IDF1 change if association was already fine.

**3. Decoupled head, only if the baseline is still coupled (true layers).** If the implemented paper is YOLOv3/JDE-YOLOv3-style, YOLOX’s lite decoupled head is the textbook layer change (+1.1 AP, faster convergence).[^7] If the baseline is ByteTrack/YOLOX or Ultralytics v8/v11, this is already present — pick (1) or (2) instead.

**4. BYTE + observation-centric motion (association architecture, training-free).** Implement BYTE’s two-stage match and, if motion is non-linear, OC-SORT’s ORU/OCM or Hybrid-SORT’s TCM (add confidence and its velocity to the Kalman state; cost = |ĉ_trk − c_det|).[^1][^2][^3] **Use when** tracks break under occlusion or crossings. **A6000:** no training. **Measure:** same detections, two trackers; report IDF1/IDs/HOTA. This is the highest-leverage *metric* change and the weakest *layer* change — pair it with (1) or (2) so the report has a drawn module *and* a table that moves.

**5. Camera-motion compensation (module, training-free).** BoT-SORT GMC: sparse features, optical flow, RANSAC affine, apply to the Kalman mean and covariance before IoU matching.[^4] **Use when** the phone or robot camera moves. **A6000:** CPU OpenCV, negligible vs detector. **Measure:** IDF1/IDs on the shaky clip only (do not average away the effect on a static clip). Optional next step is their width/height Kalman state, but that is too small to be the “significant” change by itself (+0.24 HOTA).[^4]

Recommended default package for a pedestrian campus video with a handheld camera: **(1) or (2) as the layer story**, plus **(4) BYTE** and **(5) CMC** as the metric story. Recommended default for sports/dance: **(2) if small players**, **never (1)**, plus **(4) OC-SORT/Hybrid-SORT**.

### What is too cosmetic and what is too large

Too cosmetic for “significant architecture (layers)”: activation swap; changing Kalman process-noise scalars; turning Mosaic off; a 0.05 confidence-threshold sweep; adding SE/CBAM without a small-object diagnosis and an ablation. BoT-SORT’s own KF-width change is the cautionary number: real paper, real table, +0.12 MOTA.[^4]

Too large: MOTR/MOTRv2/MeMOTR from scratch; training YOLOX-X 80 epochs on the full ByteTrack mix *only* to chase the paper’s 80.3 MOTA (that is a reproduction, not an architecture change); a 2048-d ReID tower plus a learned graph matcher. MOTRv2 is explicitly “bootstrapping E2E by a pretrained detector” — the interesting idea is the query decoder on top of YOLOX, and that decoder’s published training unit is 8 GPUs.[^10]

### How to measure so the before/after is honest

Hold the evaluation protocol fixed. For an association-only change, hold the detector weights and input size fixed (OC-SORT, Hybrid-SORT, and BoT-SORT all do this).[^2][^3][^4] For a layer change, report **detection mAP and tracking metrics** on the same custom split (FairMOT’s AP vs TPR vs IDF1 split is the model).[^5] Do not compare custom-video MOTA to MOT17-test 80.3. Use HOTA (or at least IDF1 + IDs + MOTA) because MOTA is detection-dominated — ByteTrack and DanceTrack both say this.[^1][^12] A ten-clip custom set with a pre-registered failure tag (occlusion / shake / small / uniform) is enough for a lab; a single highlight reel is not.

## Premise Check

The question assumes there exists a class of “significant architectural (layer) changes” that are simultaneously (a) what follow-up papers do, (b) likely to beat a paper baseline on *custom* video, and (c) trainable on one A6000. Premise (a) is only half true: the papers that actually move MOT metrics after 2022 are mostly association-cost and motion-model papers; layer papers exist (decoupled head, JDE branch, P2) but they are not the leaderboard engine. Premise (b) is often false if “paper baseline” means MOT17-test numbers — that gap is domain, mix-data, and protocol. Premise (c) holds for YOLO-scale detectors, JDE heads, and ReID towers, and fails for MOTR-scale E2E.

A better question, which this report answers: *given a TBD baseline (YOLOX/YOLO11 + SORT/ByteTrack), which one layer addition plus which one training-free association change should be chosen after looking at the custom video’s failure mode?*

A second suspect premise: that appearance is a safe default upgrade. DanceTrack was built to falsify that, and it does.[^12]

## Conflicts

**Appearance helps vs appearance hurts.** BoT-SORT-ReID and Hybrid-SORT-ReID report extra IDF1/HOTA on MOT17/DanceTrack when a strong separate ReID is added on top of an already good motion tracker.[^3][^4] DanceTrack’s oracle and DeepSORT-vs-SORT table say appearance matching is net harmful when clothes are uniform.[^12] **Resolution:** both can be true. Appearance is conditional on inter-identity visual diversity. For the lab, diagnose with a cosine-distance histogram on a few custom frames (DanceTrack’s own analysis method) before spending a week on a JDE head.[^12] Better-supported default when unsure: motion/BYTE/CMC, not ReID.

**“Detection is the MOT bottleneck” vs “association is the bottleneck.”** ByteTrack’s 80.3 MOTA on MOT17 is widely read as “just get a better YOLOX.” DanceTrack shows DetA/MOTA already high and AssA in the 20–40s.[^12] **Resolution:** dataset-dependent. Custom student video can be either. Measure detection mAP on sampled custom frames *before* choosing P2 vs OC-SORT.

**JDE vs separate ReID.** FairMOT/YOLO11-JDE argue one-shot is enough and cheaper.[^5][^6] BoT-SORT/Hybrid-SORT use a separate FastReID/BoT model and treat JDE as out of scope.[^3][^4] YOLO11-JDE’s MOT17 numbers are well below YOLOX-X + BYTE, which they attribute to the *detector*, not the embeddings.[^6] **Resolution:** for a layer-change lab, JDE is the right artifact (you can draw it). For raw IDF1 on pedestrians, SDE ReID on YOLOX-X detections is stronger and still A6000-feasible, but it is a second network, not a layer in the paper’s detector.

**Training-free association vs learned/E2E matchers.** MOTR/MOTRv2 and graph matchers can beat SORT-like methods on DanceTrack at much higher compute.[^3][^10] Hybrid-SORT’s claim is that weak cues close most of that gap while staying real-time.[^3] **Resolution:** for one 48 GB card and a lab calendar, the training-free side is better supported as *feasible*. E2E is better supported as *SOTA-seeking*, not as a week-long assignment.

## Gaps

- No actual student video was available, so the recommended default package is conditional. The first lab step is still a failure-mode watch-through, not a neck.
- P2’s benefit is well documented for small-object *detection*; a MOT-specific P2 ablation of BYTE-quality was not in the pages I fully read. Support for “P2 will raise IDF1” is weak.
- Exact A6000-48GB batch/imgsz pairs were inferred from official 8-GPU recipes and Ultralytics autobatch, not profiled on this SKU.
- FeatureSORT-style extra attribute heads (color, clothing) appeared in search but were not deep-read; they are a plausible layer story if the student can label attributes, and they are not load-bearing here.
- CSTrack’s cross-correlation decoupling is cited by FairMOT as a concurrent feature-conflict paper; I did not re-read CSTrack’s tables.
- Persistent specialist-papers / specialist-eval reports were not yet in the workspace, so paper-choice and eval-protocol details may later constrain which of (1)–(5) is legal.
- Cross-model Codex self-review was unavailable (no `codex:codex-rescue` subagent in this Grok environment).

## Sources

[^1]: Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. 2022. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. arXiv:2110.06864. Retrieved from https://ar5iv.labs.arxiv.org/html/2110.06864 — Type: peer-reviewed (ECCV 2022); retrieval: WebFetch long-read.

[^2]: Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. 2023. *Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*. arXiv:2203.14360. Retrieved from https://arxiv.org/html/2203.14360 — Type: peer-reviewed (CVPR 2023); retrieval: WebFetch long-read.

[^3]: Mingzhan Yang, Guangxin Han, Bin Yan, Wenhua Zhang, Jinqing Qi, Huchuan Lu, and Dong Wang. 2024. *Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking*. arXiv:2308.00783. Retrieved from https://arxiv.org/html/2308.00783v2 — Type: peer-reviewed (AAAI 2024); retrieval: WebFetch long-read.

[^4]: Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. 2022. *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv:2206.14651. Retrieved from https://arxiv.org/html/2206.14651 — Type: preprint / identified expert; retrieval: WebFetch long-read.

[^5]: Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. 2021. *FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking*. *IJCV*. arXiv:2004.01888. Retrieved from https://ar5iv.labs.arxiv.org/html/2004.01888 — Type: peer-reviewed; retrieval: WebFetch long-read.

[^6]: Iñaki Erregue, Kamal Nasrollahi, and Sergio Escalera. 2025. *YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID*. arXiv:2501.13710. Retrieved from https://arxiv.org/html/2501.13710v1 — Type: preprint / identified expert; retrieval: WebFetch long-read.

[^7]: Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. 2021. *YOLOX: Exceeding YOLO Series in 2021*. arXiv:2107.08430. Retrieved from https://arxiv.org/html/2107.08430 — Type: preprint / identified expert (widely used detector); retrieval: WebFetch long-read.

[^8]: Ultralytics community (BurhanQ / Venkat). 2025. *Adding a new head to the YOLO11n model to detect very small objects* (includes official-style YOLOv8-p2 YAML). Ultralytics Community. Retrieved from https://community.ultralytics.com/t/adding-a-new-head-to-the-yolo11n-model-to-detect-very-small-objects/876 — Type: community-vetted; retrieval: search snippet + YAML excerpt.

[^9]: YOLO11-4K authors. 2025. *YOLO11-4K: An Efficient Architecture for Real-Time Small Object Detection*. arXiv:2512.16493. Retrieved from https://arxiv.org/html/2512.16493v1 — Type: preprint; retrieval: snippet-only (P2 motivation only; not load-bearing).

[^10]: Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. 2023. *MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors*. CVPR 2023. Retrieved from https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MOTRv2_Bootstrapping_End-to-End_Multi-Object_Tracking_by_Pretrained_Object_Detectors_CVPR_2023_paper.pdf and https://github.com/megvii-research/MOTRv2 — Type: peer-reviewed + official repo; retrieval: snippet / README (8-GPU train; YOLOX 8×V100, 16 epochs).

[^11]: DecoderTracker authors. 2024. *DecoderTracker: Decoder-Only End-To-End method for Multiple-Object Tracking*. arXiv:2310.17170. Retrieved from https://arxiv.org/html/2310.17170v6 — Type: preprint; retrieval: snippet (MOTR 8×2080 Ti ~96 h).

[^12]: Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. 2022. *DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion*. arXiv:2111.14690. Retrieved from https://arxiv.org/html/2111.14690 — Type: peer-reviewed (CVPR 2022); retrieval: WebFetch long-read.

[^13]: FoundationVision / Yifu Zhang. 2021–2022. *ByteTrack* (official repository README, training commands). GitHub. Retrieved from https://github.com/ifzhang/ByteTrack — Type: community-vetted / official code; retrieval: Firecrawl scrape (full-fidelity).

[^14]: Ultralytics. 2026. *Model Training with Ultralytics YOLO* (Train mode; `batch`, `imgsz`, `freeze`, AMP/autobatch). Ultralytics Docs. Retrieved from https://docs.ultralytics.com/modes/train/ — Type: editorial / vendor engineering docs; retrieval: Firecrawl scrape (full-fidelity).

[^15]: Ultralytics. 2026. *Fine-tuning guide* (`freeze=10`, two-stage freeze-then-unfreeze). Ultralytics Docs via Context7 `/ultralytics/ultralytics`. Retrieved from https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/finetuning-guide.md — Type: editorial / vendor docs; retrieval: Context7 query-docs.

## Label Definitions

- **[CITED]** — A specific factual claim taken from a named source that was retrieved in this run (paper HTML, official docs, or official repo). Footnote required.
- **[SYNTHESIS]** — A claim that is not stated by any one source but follows from combining two or more cited facts (named in the sentence).
- **[CONCLUSION]** — The analyst’s judgment after weighing the cited evidence; not a quotation.
- **[HYPOTHESIS]** — Provisional; not used as a load-bearing recommendation in this report.
- **[TRAINING DATA]** — From model prior only; not used for load-bearing claims here.
- **[WELL-SUPPORTED]** — Falsifiable; either independent sources that cross quality tiers or independence axes agree, or a close paraphrase of a primary source that *defines or exhibits* the fact (official train flag, official README command) and was read at full fidelity.
- **[SUPPORTED]** — Falsifiable; quality sources agree but corroboration is limited, within-family (shared YOLOX detections / SORT paradigm), or the page was read via a lossy extractor (WebFetch). Single empirical studies stay here even when fully read.
- **[WEAKLY-SUPPORTED]** — Only anonymous sources or snippet-only retrieval. P2 MOT benefit is close to this; it is not used as a load-bearing tracking claim.
- **[CONTESTED]** — Quality sources disagree and the disagreement is not resolved. Appearance-on-MOT is resolved *conditionally* rather than left contested.
- **[UNFALSIFIABLE]** — Not empirically decidable in this context (not used above).

*Methodology (short).* Open survey via Firecrawl research search and GitHub web search; long-read of eight primary papers via page fetch; full-fidelity scrape of Ultralytics Train docs and the ByteTrack README (2 of the 3-scrape budget); Context7 for freeze/autobatch APIs; adversarial pass aimed at “ReID always helps” and “the upgrade is a new neck.” Stopped when further queries were confirming the same SORT-family cluster rather than adding a new intervention class.
