# Evidence log — specialist-eval

## Retrieval method
- Firecrawl search: MOT metrics, TrackEval, DanceTrack, ByteTrack, MOT Challenge (one 429 after first batch).
- Full-fidelity / near-full reads via web_fetch of raw GitHub + ar5iv HTML:
  - TrackEval MOTChallenge official Readme (raw) — READ
  - TrackEval README — READ
  - ByteTrack ar5iv 2110.06864 — READ (truncated tail)
  - HOTA arxiv html 2009.07736 — READ (truncated after §7)
  - DanceTrack arxiv html 2111.14690 — READ (full)
  - MOT16 ar5iv 1603.00831 — READ (near-full)
  - CVAT MOT format docs — READ
  - VisDrone-Dataset README — READ
- motchallenge.net/data/MOT17 — FAIL (HTTP)
- Codabench MOT17 page — JS shell, no protocol text
- GitHub web_search used only as orientation; not cited as primary

## Sources kept
1. Luiten/Hoffhues TrackEval + MOTChallenge official kit
2. Zhang et al. ByteTrack (ECCV-era arXiv 2110.06864)
3. Luiten et al. HOTA IJCV 2020 / arXiv 2009.07736
4. Sun et al. DanceTrack arXiv 2111.14690
5. Milan et al. MOT16 arXiv 1603.00831
6. CVAT MOT export docs
7. VisDrone-Dataset README + Zhu et al. 2021 citation on that page

## Claims to load-bear
- MOTA/HOTA/IDF1 require identity-consistent per-frame GT
- Public vs private detection protocols are different comparison classes
- Extra static-person data (CrowdHuman, CityPersons, ETHZ) is part of many SOTA recipes
- TrackEval officially supports custom MOT-format challenges
- Papers report official-split numbers; custom video is not how SOTA is claimed
- DanceTrack shows MOT17-trained association can look strong while failing on similar-appearance / nonlinear motion
- MOT16 defines pedestrian-only eval + ignore/distractor classes
- CVAT exports MOT gt.txt compatible with TrackEval after seqinfo/seqmaps
