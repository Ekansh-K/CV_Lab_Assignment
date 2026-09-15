# Audit — specialist-eval

## Independence axes
- **Authorship concentration:** Luiten appears on HOTA, TrackEval, and MOTChallenge official kit. MOT16 authors (Milan, Leal-Taixé, Reid, Roth, Schindler) overlap HOTA (Leal-Taixé) and MOTChallenge journal. ByteTrack and DanceTrack share ByteDance / HKU authors (Zhang, Sun, Jiang, Luo). This is the MOT establishment, not independent journalism.
- **Independence that still holds:** CVAT docs (Intel/CVAT.org) are independent of MOT authors. VisDrone (Tianjin / AISKYEYE) is a different lab and uses a different toolkit. DanceTrack *adversarially* tests MOT17-style numbers rather than echoing them.
- **Upstream evidence:** MOTA formula is one definition (CLEAR 2008) reused everywhere — agreement on the formula is definitional, not multi-study corroboration.
- **Incentive:** Benchmark papers want their protocol used; tracker papers want SOTA tables. Neither has an incentive to bless “95% on my phone video vs 76 MOTA.”

## Adversarial checks
| Claim | Search | Outcome |
|---|---|---|
| Paper MOTA can be compared to custom-video accuracy | Looked for papers that put private-video MOTA in the same table as MOT17 test | Not found in ByteTrack/DanceTrack/MOT16/HOTA. TrackEval treats custom data as a *separate challenge*. **No opposition that this comparison is valid.** |
| Official MOT only allows public detections | GitHub-search summary said private is unofficial; ByteTrack tables 4–6 and MOT16 §III-B say private is allowed if labeled | **CONTESTED secondary vs primary.** Primary wins: both protocols exist; they must not be mixed. |
| MOTA is the right single number | HOTA paper and DanceTrack argue MOTA is detection-dominated | **Tested, opposition found** — report MOTA *and* HOTA/IDF1/AssA. |
| Qualitative-only custom test is enough for a lab | Assignment asks comparison + improvement; metrics need GT | Not a literature claim. **Judgment:** qualitative is honest; numbers without GT are not. |

## Falsifiability
- “MOTA requires GT IDs” — definitional / practically testable (run TrackEval without gt.txt → fails).
- “ByteTrack 80.3 MOTA is on MOT17 *test* under *private* detections with extra data” — falsifiable by reading Table 4 + §4.1.
- “Custom phone video of cars is the same task as MOT17” — falsifiable as a domain claim; DanceTrack already shows even *person→person* shift collapses AssA.
- “Minimum 3 sequences is enough for a publishable benchmark” — **not testable as a universal**; we treat it as assignment-grade, not SOTA-grade.

## Cross-tier
- Peer-reviewed / arXiv benchmark papers + official eval kit + vendor annotation docs (CVAT) converge on: same format, same metrics, *same split* for numerical comparison.
