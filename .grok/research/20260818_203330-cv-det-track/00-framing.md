# Framing (orchestrator)

## Audience
CV lab student who must pick a paper, implement it, train on a public set, test on custom videos, then change the architecture.

## Intent
Decide: which paper/architecture, which train set, which custom-video protocol, which architecture change, given 1× RTX A6000 48GB.

## Core questions
1. What published detection+tracking architecture is implementable for a lab and has comparable numbers?
2. What architecture change is significant, feasible on 48GB, and targets custom-video failure modes?
3. How to evaluate custom videos without dishonest comparison to the paper?

## Recon structure
- MOT splits TBD vs E2E (Guan 2025; Adžemović 2025).
- TBD: detector-centric vs association-centric.
- 2022 inflection: ByteTrack (TBD) and MOTR (E2E).
- Heuristic association strong on linear dense motion; learned/E2E on complex motion.
- Metrics: MOTA, IDF1, ID-switch, HOTA; detection mAP separately.

## Specialists
- specialist-papers (investigator): candidate papers
- specialist-arch (analyst): architecture interventions on A6000
- specialist-eval (investigator): datasets + honest comparison protocol
