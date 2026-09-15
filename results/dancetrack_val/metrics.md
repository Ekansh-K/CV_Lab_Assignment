# Table A — DanceTrack val (same official YOLOX)

Same checkpoint: `pretrained/bytetrack_dance_model.pth.tar` (YOLOX-X, 800×1440).
Same 25 val sequences. TrackEval HOTA / CLEAR / Identity, `DO_PREPROC=False`.

| Tracker | HOTA | AssA | DetA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| BYTE (ours) | 47.124 | 31.561 | 70.648 | 88.301 | 51.972 | 1947 |
| Hybrid-SORT (ours, no ReID) | 59.570 | 45.137 | 78.979 | 89.507 | 60.873 | 1701 |

Gate: Hybrid-SORT HOTA (59.570) > BYTE HOTA (47.124).

Published anchors (not our run):
- ByteTrack DanceTrack test: 47.3–47.7 HOTA
- ByteTrack DanceTrack val (official toolkit table): 47.1 HOTA / 70.5 DetA / 31.5 AssA / 88.2 MOTA / 51.9 IDF1
- Hybrid-SORT DanceTrack val: 59.3 HOTA / 60.6 IDF1 / 89.5 MOTA
- Hybrid-SORT DanceTrack test: 62.2 HOTA / 63.0 IDF1 / 91.6 MOTA

Notes:
- BYTE val is a near-exact reproduction of the official val table (47.1 / 31.5 / 70.5 / 88.2 / 51.9).
- Hybrid-SORT no-ReID val is slightly above the paper’s 59.3 HOTA (59.57) on the same official DanceTrack YOLOX weights.
- Do not compare these numbers to ByteTrack MOT17 test 80.3 MOTA / 63.1 HOTA.
