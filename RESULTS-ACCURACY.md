# Local accuracy work — results (2026-11-07)

Spec: `SPEC-ACCURACY.md`. Detailed numbers: `logs/` (local only), `oyj/complexity/CHECK.md`.

## Before / after

| Measure | Before | After |
|---|---|---|
| Local CLM, tune cases (31), compare.py rule | 16/31 (spike, v1 text) | 16/31 v1 text, 20/31 v2 text |
| Local CLM, hold-out (36), compare.py rule | — | 24/36 v1 text, 21/36 v2 text |
| TypeSafe, hold-out (36) | — | 34/36 v1 text, 36/36 v2 text |
| Local CLM + gate + calibration, hold-out: kept-accuracy / coverage | — | v1 64.7 % / 47.2 %, v2 100 % / 22.2 % |
| Complexity question, test (100): level agreement with TypeSafe | CLM score cases on hold-out: 5/7 raw | ownyourjev 95 % |
| Complexity question, test: decision agreement (hook rule) | — | ownyourjev 71 % (85 % without dev calibrator) |
| Complexity question, CPU latency p90 | — | 216 ms |

The hold-out v2 run was repeated with the final code (`logs/final-holdout-v2.json`) and gives the same numbers. `logs/p2-eval.json` is a single run; no measured code changed after it.

Hand check limit: each of the 220 seed/dev briefs has one blind reviewer label plus the TypeSafe label. The controller read only the 56 disagreements (`oyj/complexity/CHECK.md`).

## Gate verdicts

1. Gate P1 (kept-accuracy ≥ 95 %, coverage ≥ 50 %, risky bash p ≥ 0.6): **FAIL** for both text versions.
2. Gate P2 (decision agreement ≥ 95 %, ECE ≤ 0.05, p90 ≤ 100 ms): **FAIL** on all three (71 %, 0.181, 216 ms).

## What is live

1. `jev-guard.json` has `jev.local_gate` (v1 values) and `jev.capture` (`enabled: false`). The backend stays `"typesafe"`, so hooks behave as before.
2. No question set is routed to the ownyourjev model (the routing step runs only after Gate P2 passes).

## What the numbers say

1. The local CLM is not a TypeSafe replacement. The confidence gate makes it safe only by sending most sets back to TypeSafe.
2. The v2 question wording (`spike/PHRASING.md`) helps TypeSafe (34 → 36 of 36). It is a candidate hook text change, not applied.
3. The ownyourjev model learns the complexity question well (95 % same level as TypeSafe) but misses on confidence and on CPU speed. The hand check found that TypeSafe over-rates mechanical renames (53 of 220 labels changed), so copying TypeSafe exactly may be the wrong goal.

## Next options (owner decision)

1. Calibrate the ownyourjev model on TypeSafe labels, and try a smaller backbone or max-len 128 for latency.
2. Apply the v2 wording to the hook questions.
3. Turn on capture (`jev.capture.enabled: true`) to collect real hook data for later training.
4. The CLM head fine-tune fallback (plan §6) has not started.
