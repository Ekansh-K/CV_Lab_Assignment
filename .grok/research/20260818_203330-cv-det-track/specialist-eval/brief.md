# Brief: Honest train-public / test-custom MOT evaluation

## Question
How should a student train a detection+tracking system on a public dataset and then test on custom-made videos so that (1) the experiment is valid, (2) comparison to a paper's published numbers is honest (not apples-to-oranges), and (3) the custom test set is strong enough for a lab report?

## Presuppositions to test
- Students can numerically compare custom-video results to a paper's published MOTA/HOTA. **Likely false** unless they also reproduce the paper's official split.
- "Detection+tracking" is a single comparable task. **Contested**: MOT, MOTS, SOT, and generic class-agnostic tracking are different protocols.
- Custom phone videos can be a valid test set. **Partially true** as a domain-shift / qualitative stress test; **false** as a substitute for MOT Challenge numbers.
- Papers commonly train on public data and report official numbers on own videos as the main result. **To verify.**
- There exist clean train-set / paper pairings. **Mostly true for MOT papers (MOT17, CrowdHuman, DanceTrack), weaker for arbitrary student objects.**

## Ambiguous terms
- Detection+tracking: MOT vs detector+association vs SOT
- Compare with the paper: same split? same detector? private detections vs public detections?
- Valid: internal validity of the lab experiment vs external validity vs reproducibility
- Strong enough: assignment-grade vs publishable

## Lines of inquiry
1. Official MOT / DanceTrack / VisDrone / TrackEval / HOTA definitions and what they require (GT format, public vs private detections).
2. How papers handle extra/private test videos (if at all).
3. Minimum viable annotation (tools, frames, sequences, TrackEval input).
4. Which metrics are meaningless without identity GT.
5. Common student apples-to-oranges mistakes.
6. Concrete assignment protocol.

## Named leads (hypotheses only — expand via open search)
MOT Challenge, DanceTrack, VisDrone, CrowdHuman, COCO, KITTI, BDD100K, UA-DETRAC, TrackEval, HOTA (Luiten), CLEAR MOT, IDF1, CVAT, Label Studio, Roboflow, ByteTrack private-detection protocol.
