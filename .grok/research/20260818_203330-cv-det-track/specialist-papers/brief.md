# Brief: Base papers for a detection+tracking lab assignment

## Question
Which published object-detection + multi-object-tracking architectures are actually suitable as the base paper for a CV lab assignment that requires: implement architecture, train on public dataset, test on student-filmed custom videos, compare numbers to paper, then make a significant architectural change.

## Presuppositions to test
- "Implement a paper's architecture" is feasible in weeks. May be true for TBD (detector + association) and false for E2E MOTR-style from scratch.
- Students can approach paper numbers on a public benchmark with 1x A6000 48GB.
- Custom student videos are a valid test domain (domain shift vs MOT17 pedestrians).
- There exist papers that cleanly pair a named detector with a named tracker.
- YOLO is only an example; the field may or may not actually be YOLO-dominated.
- Recon's TBD vs E2E split (ByteTrack / MOTR 2022 inflection) is accurate.

## Lines of inquiry
1. Open survey: what methods appear as 2024–2026 MOT workhorses (surveys, leaderboards, popular repos) — not a named list.
2. Which papers have official/maintained code that trains (not only infers) on 1x 48GB GPU.
3. Which report numbers students can reasonably approach (no 8xA100 extra-data SOTA).
4. Which pair named detector + named tracker so "implement architecture" is clean.
5. What is too old / toy / application-only vs real architecture papers.
6. Adversarial: is YOLO+ByteTrack actually dominant, or is that a tutorial echo chamber? What E2E methods are actually used as lab/course bases? What fails on custom video?

## Training-data leads (hypotheses, not the candidate set)
ByteTrack, BoT-SORT, OC-SORT, FairMOT, MOTR/MOTRv2, TrackFormer, QDTrack, CenterTrack, Deep OC-SORT, Hybrid-SORT, BoostTrack, BoxMOT, Ultralytics built-in trackers.

## Better-posed questions
- Which papers have a modular architecture a student can modify in weeks?
- Which benchmarks allow private-val comparison without MOTChallenge test-server lock-in?
- Is "implement from paper" vs "run official code + modify" the real assignment interpretation?
