# Custom / internet clips — qualitative transfer (not Table A)

Same frozen DanceTrack YOLOX-X + BYTE / Hybrid-SORT as Table A.
Three Mixkit people clips (free license). **No MOT ground truth**, so **no HOTA**.
Do not compare these rows to DanceTrack 47.7 or MOT17 80.3.

| Clip | Source | Tracker | Frames | Unique IDs | Boxes / frame |
|---|---|---|---:|---:|---:|
| clip1_crosswalk | Mixkit 4401 Tokyo junction | BYTE | 450 | 433 | 13.57 |
| clip1_crosswalk | Mixkit 4401 | Hybrid-SORT | 450 | 587 | 9.03 |
| clip2_street_crowd | Mixkit 4437 Japan street | BYTE | 450 | 52 | 8.42 |
| clip2_street_crowd | Mixkit 4437 | Hybrid-SORT | 450 | 59 | 8.62 |
| clip3_park_walk | Mixkit 35399 park | BYTE | 318 | 2 | 2.00 |
| clip3_park_walk | Mixkit 35399 | Hybrid-SORT | 318 | 2 | 2.00 |

Notes:
- Clip 3 is the sanity check: two people, both trackers keep exactly two IDs.
- Clip 1 is distant / time-lapse; unique-ID counts are not a quality score.
- Quantitative paper comparison remains **Table A** (`results/dancetrack_val/metrics.md`): BYTE 47.124 HOTA vs Hybrid-SORT 59.570 HOTA.
