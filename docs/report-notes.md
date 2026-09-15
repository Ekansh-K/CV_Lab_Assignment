# Report notes (allowed claims only)

Filled from `results/dancetrack_val/metrics.md`. Table B is not filled yet — needs the three filmed clips (Task 6).

## Table A paragraph (use this)

We reimplemented ByteTrack (Zhang et al., ECCV 2022) as BYTE
and Hybrid-SORT (Yang et al., AAAI 2024) on the same official
DanceTrack YOLOX weights. On DanceTrack val, BYTE reaches
HOTA 47.1 / MOTA 88.3 / IDF1 52.0. Hybrid-SORT reaches HOTA 59.6 /
MOTA 89.5 / IDF1 60.9. ByteTrack is reported at 47.3–47.7 HOTA
on DanceTrack test; Hybrid-SORT is reported at 59.3 HOTA
(val) and 62.2 HOTA (test). Our val ranking matches that
gap. We do not compare these numbers to ByteTrack’s MOT17
test result (80.3 MOTA / 63.1 HOTA), which uses a different
dataset and test-server protocol.

On three public Mixkit people clips the same frozen weights
run as a qualitative transfer demo (no MOT GT, so no HOTA).
Clip 3 (two people in a park) keeps 2 IDs for both BYTE and
Hybrid-SORT. That demo is domain transfer, not a DanceTrack
or MOT17 leaderboard comparison.

## Checklist

- [x] Table A exists and Hybrid-SORT HOTA > BYTE HOTA (59.570 > 47.124)
- [x] No sentence claims 80.3 was beaten
- [ ] Custom table is labeled transfer (pending footage)
- [x] No ReID / extra detectors in the main story

## Exact TrackEval COMBINED (do not round in the table file)

| Tracker | HOTA | AssA | DetA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| BYTE | 47.124 | 31.561 | 70.648 | 88.301 | 51.972 | 1947 |
| Hybrid-SORT no-ReID | 59.570 | 45.137 | 78.979 | 89.507 | 60.873 | 1701 |

Setup: HybridSORT (`ymzis69/HybridSORT`) on Thunder instance 0 (1× RTX A6000). Checkpoint `bytetrack_dance_model.pth.tar`. BYTE via `tools/run_byte_dance.py`. Hybrid-SORT via `tools/run_hybrid_sort_dance.py` with `hybrid_sort_with_reid=False`, TCM + Height_Modulated_IoU.
