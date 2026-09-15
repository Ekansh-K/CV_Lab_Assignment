# ByteTrack → Hybrid-SORT Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do not run GPU jobs on the current machine.** Wait until the user provides the remote A6000. Tasks 0–1 can be prepared here; Tasks 2–7 run on the remote.

**Goal:** Run official YOLOX DanceTrack weights through BYTE (ByteTrack) and Hybrid-SORT, score both on DanceTrack val with TrackEval, then score the same two trackers on 2 annotated custom clips.

**Architecture:** One repo (HybridSORT) already implements BYTE and Hybrid-SORT on shared YOLOX detections. Table A is DanceTrack val (the number we can beat: ByteTrack 47.3–47.7 HOTA vs Hybrid-SORT 59.3 val / 62.2 test). Table B is custom transfer. No detector training in the default path.

**Tech Stack:** Python 3.8–3.10, PyTorch + CUDA on remote A6000, HybridSORT (YOLOX + BYTE + Hybrid-SORT + bundled TrackEval), DanceTrack, CVAT (annotation only), TrackEval HOTA/CLEAR/Identity.

## Global Constraints

- Remote GPU only for install/download/inference/eval; this workspace is docs and later report text.
- One upgrade: Hybrid-SORT **without** ReID. Backup is OC-SORT, not both.
- Official DanceTrack YOLOX weights only for Table A. Do not train YOLOX-X.
- Do not claim a win over ByteTrack MOT17 test 80.3 MOTA / 63.1 HOTA.
- Do not compare custom-video scores to 47.7 or 80.3.
- Do not add ReID, P2, BoT-SORT, FairMOT, MOTR, or a second detector into Table A.
- Same checkpoint and image size for BYTE and Hybrid-SORT.

## File map (created on remote unless noted)

| Path | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-08-19-det-track-lab-design.md` | Locked claims (this machine) |
| `docs/superpowers/plans/2026-08-19-det-track-lab.md` | This plan (this machine) |
| `third_party/HybridSORT/` | Official code clone (remote) |
| `third_party/HybridSORT/datasets/dancetrack/` | DanceTrack train/val/test |
| `third_party/HybridSORT/pretrained/*.pth.tar` | Official YOLOX DanceTrack weights |
| `results/dancetrack_val/byte/` | BYTE val predictions (MOT txt) |
| `results/dancetrack_val/hybrid_sort/` | Hybrid-SORT val predictions |
| `results/dancetrack_val/metrics.md` | Table A numbers |
| `custom_videos/raw/` | 3 filmed clips |
| `custom_videos/annot/{seq}/gt/gt.txt` | CVAT MOT export for 2 clips |
| `results/custom/{byte,hybrid_sort}/` | Table B predictions |
| `results/custom/metrics.md` | Table B numbers |
| `docs/report-notes.md` | Allowed comparison paragraph + filled numbers |

---

### Task 0: Freeze claims (this machine — already done)

**Files:**
- Create: `docs/superpowers/specs/2026-08-19-det-track-lab-design.md`
- Create: `docs/superpowers/plans/2026-08-19-det-track-lab.md`

**Interfaces:**
- Consumes: conversation decisions (ByteTrack base, Hybrid-SORT upgrade, DanceTrack claim, no 80.3).
- Produces: spec + plan the remote run must follow.

- [x] **Step 1: Write spec with allowed/forbidden sentences**

Done in `docs/superpowers/specs/2026-08-19-det-track-lab-design.md`.

- [x] **Step 2: Write this plan**

Done in this file.

- [x] **Step 3: User confirms spec + plan before any remote work**

Remote access is via `tnr connect 0` (Thunder instance 0). MCP `thunder-compute` is configured but not authenticated; do not use it.

---

### Task 1: Remote environment (A6000)

**Files:**
- Create: `third_party/HybridSORT/` (git clone)
- Create: conda/venv on the remote (not committed)

**Interfaces:**
- Consumes: remote Linux + CUDA + 48 GB A6000.
- Produces: importable `yolox` and `torch.cuda.is_available() == True`.

- [x] **Step 1: Check GPU**

Done via `tnr connect 0`: NVIDIA RTX A6000, torch 2.12.1+cu130, CUDA True.

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: RTX A6000 (or equivalent), CUDA visible.

- [x] **Step 2: Clone HybridSORT only**

On remote: `~/CV_Lab_Assignment/third_party/HybridSORT`.

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/ymzis69/HybridSORT.git
cd HybridSORT
```

Do not clone ByteTrack, OC-SORT, BoT-SORT, or FairMOT.

- [x] **Step 3: Isolated env + install (match README: PyTorch 1.10–2.x, Python 3.8–3.10)**

Used preinstalled torch 2.12.1+cu130. Installed HybridSORT extras without replacing torch. `pip install -e . --no-build-isolation --no-deps`. OpenCV headless (no X11).

```bash
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox pandas xmltodict
```

Skip FastReID / `fast_reid/docs/requirements.txt` (no ReID).

- [x] **Step 4: Smoke import**

`SMOKE_OK` on remote: BYTETracker + Hybrid_Sort import, CUDA True. No DanceTrack inference yet.

```bash
python3 -c "import torch, yolox; print('ok', torch.__version__)"
```

Expected: prints `ok` and a torch version. If `cython_bbox` or CUDA mismatch fails, fix env only — do not switch trackers.

---

### Task 2: DanceTrack data + official weights

**Files:**
- Create: `third_party/HybridSORT/datasets/dancetrack/{train,val,test}/`
- Create: `third_party/HybridSORT/pretrained/bytetrack_dance_model.pth.tar`
- Create: `third_party/HybridSORT/pretrained/ocsort_dance_model.pth.tar` (Hybrid-SORT demo/exp often points here; use the zoo file the exp config names)

**Interfaces:**
- Consumes: DanceTrack release (https://github.com/DanceTrack/DanceTrack) and Hybrid-SORT Google Drive zoo: https://drive.google.com/drive/folders/18IsZGeGiyKDshhYIzbpYXoNEcBhPY8lN
- Produces: val sequences with `img1/` + `gt/gt.txt`, and one YOLOX checkpoint used for **both** trackers.

- [x] **Step 1: Download DanceTrack and place it**

```
datasets/dancetrack/
  train/   + train_seqmap.txt
  val/     + val_seqmap.txt
  test/    + test_seqmap.txt
```

Follow Hybrid-SORT README “Data preparation” (same layout as OC-SORT).

- [x] **Step 2: Convert if the repo requires COCO json**

```bash
cd third_party/HybridSORT
python3 tools/convert_dance_to_coco.py
```

Expected: a COCO-format annotation file under `datasets/dancetrack` without error.

- [x] **Step 3: Download official DanceTrack YOLOX weights into `pretrained/`**

From the Hybrid-SORT “Detection Model” Drive folder. Prefer the DanceTrack-val / ByteTrack dance checkpoint named in:

- `exps/example/mot/yolox_dancetrack_val.py` (`-c` for BYTE)
- `exps/example/mot/yolox_dancetrack_val_hybrid_sort.py`

If the hybrid exp hardcodes a path, put the file at **that** path. One file for both trackers if they share it; two files only if the two exps name two different official zoo files.

- [x] **Step 4: Confirm val ground truth exists**

```bash
ls datasets/dancetrack/val/*/gt/gt.txt | head
```

Expected: multiple `gt.txt` files. Without these, TrackEval cannot emit HOTA.

---

### Task 3: BYTE baseline on DanceTrack val (Table A, column 1)

**Files:**
- Modify: none of our code unless exp checkpoint path is wrong
- Create: `results/dancetrack_val/byte/` (copy or symlink of HybridSORT YOLOX outputs)

**Interfaces:**
- Consumes: `yolox_dancetrack_val.py` + official `-c` checkpoint.
- Produces: per-sequence MOT txt predictions for DanceTrack val.

- [x] **Step 1: Run BYTE (official HybridSORT wrapper)**

```bash
cd third_party/HybridSORT
python3 tools/run_byte_dance.py \
  -f exps/example/mot/yolox_dancetrack_val.py \
  -c pretrained/bytetrack_dance_model.pth.tar \
  -b 1 -d 1 --fp16 --fuse \
  --dataset dancetrack \
  --expn byte_dancetrack_val
```

If the checkpoint filename from Drive differs, pass the actual path. Do not change association code.

- [x] **Step 2: Confirm output txts exist**

```bash
find YOLOX_outputs -name '*.txt' | head
```

Expected: one txt per val sequence, MOT format (`frame,id,x,y,w,h,conf,...`).

- [x] **Step 3: Copy predictions into the assignment results dir**

```bash
mkdir -p ../../results/dancetrack_val/byte
cp -r YOLOX_outputs/<byte_dancetrack_val_pred_dir>/. ../../results/dancetrack_val/byte/
```

Use the actual output folder name printed by the script.

---

### Task 4: Hybrid-SORT on the same val set (Table A, column 2)

**Files:**
- Create: `results/dancetrack_val/hybrid_sort/`

**Interfaces:**
- Consumes: `yolox_dancetrack_val_hybrid_sort.py` and the **same** official YOLOX weights family as Task 3.
- Produces: MOT txt predictions for Hybrid-SORT (no ReID exp).

- [x] **Step 1: Open the hybrid val exp and confirm checkpoint + `fp16`**

Read `exps/example/mot/yolox_dancetrack_val_hybrid_sort.py` (README notes lines ~35–45). Point it at the same official DanceTrack YOLOX weights. Do **not** use `yolox_dancetrack_val_hybrid_sort_reid.py`.

- [x] **Step 2: Run Hybrid-SORT**

```bash
cd third_party/HybridSORT
python3 tools/run_hybrid_sort_dance.py \
  -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
  -b 1 -d 1 --fp16 --fuse \
  --expn hybrid_dancetrack_val
```

Add `-c pretrained/<official_dance_yolox>.pth.tar` if the script requires it (BYTE does; hybrid may read it from the exp).

- [x] **Step 3: Copy predictions**

```bash
mkdir -p ../../results/dancetrack_val/hybrid_sort
cp -r YOLOX_outputs/<hybrid_dancetrack_val_pred_dir>/. ../../results/dancetrack_val/hybrid_sort/
```

- [x] **Step 4: Sanity**

Same number of sequence txt files as BYTE. If Hybrid-SORT crashes, switch to OC-SORT (`noahcao/OC_SORT`) on the same DanceTrack val and official dets — do not add ReID to unstick it.

---

### Task 5: TrackEval Table A

**Files:**
- Create: `results/dancetrack_val/metrics.md`

**Interfaces:**
- Consumes: `results/dancetrack_val/byte/`, `results/dancetrack_val/hybrid_sort/`, DanceTrack val GT.
- Produces: HOTA, AssA, DetA, MOTA, IDF1, IDSW for both trackers.

- [x] **Step 1: Evaluate with HybridSORT’s bundled TrackEval (preferred)**

Use the DanceTrack eval path already in `third_party/HybridSORT/TrackEval` / `tools` if the run scripts already printed metrics. If they did, copy those numbers into `results/dancetrack_val/metrics.md`.

If not, run official TrackEval DanceTrack config on the two pred folders (same GT, same split). Command shape:

```bash
python3 third_party/HybridSORT/TrackEval/scripts/run_mot_challenge.py \
  --BENCHMARK DanceTrack \
  --SPLIT_TO_EVAL val \
  --TRACKERS_TO_EVAL byte hybrid_sort \
  --METRICS HOTA CLEAR Identity \
  --USE_PARALLEL False \
  --NUM_PARALLEL_CORES 1 \
  --TRACKERS_FOLDER results/dancetrack_val \
  --GT_FOLDER third_party/HybridSORT/datasets/dancetrack
```

Adjust `GT_FOLDER` / seqmap flags to whatever this TrackEval checkout expects for DanceTrack (read `TrackEval/docs` in that clone). Do not write a custom MOTA formula.

- [x] **Step 2: Write `results/dancetrack_val/metrics.md`**

Fill this table with **our** numbers:

```markdown
# Table A — DanceTrack val (same official YOLOX)

| Tracker | HOTA | AssA | DetA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| BYTE (ours) | | | | | | |
| Hybrid-SORT (ours) | | | | | | |

Published anchors (not our run):
- ByteTrack DanceTrack test: 47.3–47.7 HOTA
- Hybrid-SORT DanceTrack val: 59.3 HOTA / 60.6 IDF1 / 89.5 MOTA
- Hybrid-SORT DanceTrack test: 62.2 HOTA / 63.0 IDF1 / 91.6 MOTA
```

- [x] **Step 3: Gate**

Hybrid-SORT HOTA must be **greater than** BYTE HOTA. BYTE HOTA should not be absurdly far from the mid-40s–50s band (val can differ from test 47.7). If BYTE ≫ Hybrid-SORT, debug tracker swap / wrong preds before writing the report.

---

### Task 6: Custom videos (Table B only)

**Files:**
- Create: `custom_videos/raw/*.mp4`
- Create: `custom_videos/annot/<seq>/img1/`, `gt/gt.txt`, `seqinfo.ini`
- Create: `results/custom/metrics.md`

**Interfaces:**
- Consumes: same official DanceTrack YOLOX + BYTE / Hybrid-SORT as Table A.
- Produces: TrackEval on 2 annotated clips; 1 clip qualitative only.

- [ ] **Step 1: Film 3 short people clips (15–30 s)**

1. Crossings / similar clothes  
2. Handheld camera motion  
3. Leave-and-reenter or indoor light  

Same class the DanceTrack YOLOX fires on (people). Not cars.

- [ ] **Step 2: Annotate 2 clips in CVAT as bounding-box tracks**

Export MOT format. Keep IDs through occlusion. Every evaluated frame has a box.

```
custom_videos/annot/seq1/
  img1/000001.jpg ...
  gt/gt.txt
  seqinfo.ini
```

`seqinfo.ini` must set `name`, `imDir=img1`, `frameRate`, `seqLength`, `imWidth`, `imHeight`.

- [ ] **Step 3: Run both trackers via HybridSORT demo on each annotated sequence**

```bash
cd third_party/HybridSORT
python3 tools/demo_track.py --demo_type image \
  -f exps/example/mot/yolox_dancetrack_val.py \
  -c pretrained/bytetrack_dance_model.pth.tar \
  --path /abs/path/custom_videos/annot/seq1/img1 \
  --fp16 --fuse --save_result

python3 tools/demo_track.py --demo_type image \
  -f exps/example/mot/yolox_dancetrack_val_hybrid_sort.py \
  -c pretrained/ocsort_dance_model.pth.tar \
  --path /abs/path/custom_videos/annot/seq1/img1 \
  --fp16 --fuse --save_result
```

Use the same official weights as Table A. Convert demo outputs to MOT txt in `results/custom/byte/` and `results/custom/hybrid_sort/` if the demo writes a different layout.

- [ ] **Step 4: TrackEval on the 2 sequences (`--DO_PREPROC False`)**

Write `results/custom/metrics.md` with per-sequence and pooled HOTA/MOTA/IDF1/IDSW. Label the table **custom transfer**, not DanceTrack.

---

### Task 7: Report notes (claims only)

**Files:**
- Create: `docs/report-notes.md`

**Interfaces:**
- Consumes: `results/dancetrack_val/metrics.md`, `results/custom/metrics.md`.
- Produces: the only comparison paragraph we will use.

- [x] **Step 1: Write `docs/report-notes.md` with filled numbers**

Use this paragraph, substituting real values:

```markdown
We reimplemented ByteTrack (Zhang et al., ECCV 2022) as BYTE
and Hybrid-SORT (Yang et al., AAAI 2024) on the same official
DanceTrack YOLOX weights. On DanceTrack val, BYTE reaches
HOTA h1 / MOTA m1 / IDF1 i1. Hybrid-SORT reaches HOTA h2 /
MOTA m2 / IDF1 i2. ByteTrack is reported at 47.3–47.7 HOTA
on DanceTrack test; Hybrid-SORT is reported at 59.3 HOTA
(val) and 62.2 HOTA (test). Our val ranking matches that
gap. We do not compare these numbers to ByteTrack’s MOT17
test result (80.3 MOTA / 63.1 HOTA), which uses a different
dataset and test-server protocol.

On N annotated custom sequences the same weights give BYTE
HOTA h3 and Hybrid-SORT HOTA h4. That table is domain
transfer, not a DanceTrack or MOT17 leaderboard comparison.
```

- [ ] **Step 2: Checklist before the report is “done”**

- [ ] Table A exists and Hybrid-SORT HOTA > BYTE HOTA  
- [ ] No sentence claims 80.3 was beaten  
- [ ] Custom table is labeled transfer  
- [ ] No ReID / extra detectors in the main story  

---

## Out of plan (do not schedule)

- Train YOLOX-X on 8 GPUs or on the A6000  
- Hybrid-SORT-ReID  
- MOT17 test submission  
- BoT-SORT / CMC / P2 / FairMOT / MOTR  
- Fine-tune on the 2 custom test clips  
- Optional YOLOX-M train — only if the course later requires a training section; then add a new task, do not fold it into Table A  

## Self-review

1. **Spec coverage:** pair, DanceTrack claim, custom transfer, hardware, forbidden 80.3, no extra methods — all have tasks.  
2. **Placeholders:** checkpoint filenames follow the official README; if Drive names differ, Task 2 says use the exp path.  
3. **Consistency:** BYTE then Hybrid-SORT then TrackEval; same weights; two tables only.
