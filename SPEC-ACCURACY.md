# local-clm accuracy — Implementation Specification

Status: APPROVED FOR IMPLEMENTATION after the independent review (`.agent/reports/G-SPEC-review.md`, all Critical/Important fixed). Commit 1 on `feat/local-accuracy`.
This file is authoritative for the accuracy work. `SPEC.md` stays authoritative for the service build. Only the controller (main session) changes this file.
Plan source: `.agent/HANDOFF-accuracy.md` (owner order 2026-11-06). Owner decisions 2026-11-07 are in §2.

## 0. Contents
1. Goal and end state
2. Owner decisions (2026-11-07)
3. Facts (verified 2026-11-07)
4. Repository layout (new and changed files)
5. Coordination protocol
6. Batches and steps
7. Acceptance gates
8. Guardrails

---

## 1. Goal and end state

1. Phase 1 (P1): the three cheap fixes on the CLM path are measured on new hold-out cases: confidence gate, better question text, calibration. `.agent/reports/P1.md` holds the numbers and a pass/fail against gate P1.
2. The P1 confidence gate is live in `jev_client` for CLM answers: a gated-out answer set goes to TypeSafe.
3. An opt-in capture of TypeSafe calls exists in `jev_client`. Default off.
4. Phase 2 (P2): one ownyourjev model answers the agent re-tier question set (`decisions.complexity_tier`) on CPU. It is trained on synthetic task briefs labeled by TypeSafe, with 220 hand-checked labels. `.agent/reports/P2.md` holds the numbers and a pass/fail against gate P2.
5. If the P2 model passes gate P2, `jev_client` routes that question set to it when the backend is `"local"`. Every error falls back to TypeSafe.
6. The default backend stays `"typesafe"`. Nothing switches without the owner.

## 2. Owner decisions (2026-11-07)

| Topic | Decision |
|---|---|
| Scope | P1 + P2 on ONE fixed question set with synthetic states. Real captured data comes later. |
| Gates | As proposed in HANDOFF-accuracy.md §4 and §5 (copied in §7). |
| Labels | TypeSafe is the only labeler. 220 labels are hand-checked (20 seeds + 200 dev). Claude checks; the owner reviews `P2.md`. |
| Capture | Add the code, default off. The owner switches it on. |
| Backend | Stays `"typesafe"` in `jev-guard.json`. bash-guard stays `local=False`. |

## 3. Facts (verified 2026-11-07)

| Fact | Value |
|---|---|
| P1 baseline | `logs/spike-q6.json`: local 16/31, TypeSafe 31/31. Per type local: noul 7/10, choice 7/13, score 2/8. Risky bash p (local) 0.75, 0.60, 0.77; safe bash p 0.58, 0.61. |
| CLM answers | `clm/schema.py:140,144`: choice `{choice, confidence, probabilities}`, score `{score, confidence, legend, probabilities}`. `confidence` = top probability minus the mean of the others (`schema.py:122`). noul `{noul}`. |
| CLM yes/no text | No `criteria` → options "Yes. This is true: X" / "No. This is false: X" (`schema.py:93-101`). Explicit `criteria: {"true","false"}` replaces them. |
| `spike/compare.py` | Needs `--a-url` and `--b-url`. Reads only `spike/cases.json`. `p_*` is set only for noul. `bg_*` summary keys assume ids `bg1..bg5`. |
| Re-tier question set | `jevlib/decisions.py:33-79` `complexity_tier(goals)`: state `{"note": "independent task complexity"}`, one question per goal `task_<i>` = `score({"question": "How complex is this task for an AI coding agent?", "task": g}, _COMPLEXITY_LEVELS)` with the 3 levels at `decisions.py:23-27`. Result uses `score` → `config.tier_for_score`, and `confidence < 0.5` → not confident. Caller `agent_state_hooks.py:_jev_agent` passes one goal `f"{description}\n{prompt[:3000]}"`. It routes only when `confident` is true (`agent_state_hooks.py:210-213`). `jev-guard.jsonl` has 32 `route` events (confident routes only) from 2026-10-18 to 2026-11-06; the call count is not logged. |
| TypeSafe score answer | Live call 2026-11-07: `{type, score, confidence, legend, probabilities: {"0","1","2"}}`. `score` = expected level (0.47/0.53 → 1.52). `confidence` = top minus mean of others (0.53 → 0.30; 0.81 → 0.72; 0.88 → 0.82). Clear tasks give one-hot probabilities. |
| Tier bounds | `config.tier_for_score`: `score >= routing.thresholds.deep_min` → deep, `>= standard_min` → standard, else quick (`jevlib/config.py:98-117`). |
| ownyourjev | `github.com/erfaneshrati/ownyourjev` commit `845123fb834fed17dcd84531af6afe468b540e6c` (2026-10-19), MIT. Study: `.agent/reports/P2-study.md`. |
| ownyourjev install | Trial 2026-11-07: `uv venv -p 3.13` + `uv pip install --no-sources <clone>` works on this PC: torch 2.15.0+cpu, transformers 5.22.2, onnxruntime 1.31.1. |
| ownyourjev model | Encoder + CLS pooling + linear head (`models/decision.py`). The encoder sees only `render_state(state)` (`spec.py:141-160`): dict → `key: value` blocks. The question text is NOT encoded. Model identity = `spec_hash` of the question, else the model directory name = qid (`serve.py:38-42`). |
| ownyourjev score answer | `score` = expected level `Σ i·p_i`; `confidence` = max probability (`decision.py:98-99`). This differs from TypeSafe/CLM confidence (top minus mean of others). |
| ownyourjev serve | `ownyourjev serve <registry> --host 127.0.0.1 --port N`. Default host 127.0.0.1. Serve runs the PyTorch model only (no ONNX mode). No network use at run time found. Same `/v1/systemone` request/response shape. |
| ownyourjev ONNX | `export.OnnxPredictor(model_dir, onnx_path, threads)` has `.answer(state)` (applies `calibration.json`) and `.probs(states)` (`export.py:86-110`). `calibrate` uses `model.int8.onnx` when it exists (`cli.py:214-219`), so calibrate after export = calibrated for the int8 runtime. |
| CPU latency (reviewer probe) | gte-modernbert torch fp32, batch 1, random weights: p90 379 ms at 64 tokens, 1447 ms at 256, 2957 ms at 512. The torch serve cannot meet 100 ms. The int8 ONNX path is the only candidate. |
| ownyourjev CLI details | `train SPEC DATA` takes one data file (`cli.py:116`). spec.json = `{"questions": {qid: question}}` (`cli.py:22-24`). dev for `calibrate` = `{"state", "gold": {qid: answer}}` (`cli.py:197,212`). With `--soft`, `meta.soft[qid]` overrides `labels` (`train.py:105-113`). `cli.py:15` loads a `.env` from cwd upward. `_read_states` reads files with the locale encoding (cp1252 on this PC). |
| ownyourjev Windows | `gpu.py:11` imports `fcntl` at top level. It is imported only when `device == "auto"` and CUDA exists. CPU path does not import it. |
| Backbone | `Alibaba-NLP/gte-modernbert-base` revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, Apache-2.0, `model.safetensors` 298 041 568 bytes sha256 `3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba`. ownyourjev spec: CLS pooling, lr 3e-5. An unknown backbone path gets CLS pooling and lr 2e-5 (`backbones.py:65-72`), so a local path plus `--lr 3e-5` equals the named spec. |
| jev_client local path | `call_jev(..., local=True)`: backend `"local"` and payload ≤ 4000 chars → `localclm.is_up()` → POST CLM → return answers. Down → `localclm.start(probe=False)` → TypeSafe. Any exception → `local_clm_error` → TypeSafe. Tests: `tests/test_jev_local.py` (10), `tests/test_localclm.py` (14). |

## 4. Repository layout (new and changed files)

```
spike/compare.py              changed: --cases, optional --a-url, conf/probs for all types
spike/cases-holdout.json      new: >= 30 hold-out cases (P1)
spike/cases-v2.json           new: cases.json with rewritten question text
spike/cases-holdout-v2.json   new: hold-out with rewritten question text
spike/PHRASING.md             new: the phrasing rules (P1.3)
spike/gate.py                 new: fit gate + temperature, report metrics
tests/test_gate.py            new
oyj/requirements-oyj.in/.txt  new: hashed deps for .venv-oyj (ownyourjev installed --no-deps)
oyj/complexity/pool.jsonl     new: synthetic task briefs (committed: synthetic, not captured)
oyj/complexity/labeled.jsonl  new: TypeSafe labels for the pool
oyj/complexity/{seeds,dev,train,test}.jsonl  new: splits
oyj/complexity/CHECK.md       new: hand-check record
oyj/label.py                  new: TypeSafe labeling + splits
oyj/eval.py                   new: agreement, ECE, latency on test split
oyj/serve_onnx.py             new: minimal int8 ONNX server (ownyourjev serve has no ONNX mode)
oyj/complexity/spec.json      new: the fixed question
tests/test_oyj.py             new
localclm.py                   changed: optional third component "oyj" (ownyourjev serve)
setup.ps1                     changed: -WithOyj switch
.gitignore                    changed: .venv-oyj/, data/
integration/jev_client_local.md  changed: records the new jev_client edits
tests/test_jev_local.py       changed: gate, capture, oyj route tests
logs/p1-*.json, logs/p2-*.json   results (logs/ is gitignored; numbers go into the reports)
models/oyj/complexity/        trained model (models/ is gitignored)
```

Out of repo (no git, back up first as `*.bak-20261107-accuracy`): `C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py`, `C:/Users/molna/.claude/agent-hooks/jev-guard.json`.

## 5. Coordination protocol

Same as `SPEC.md` §5: workers follow the brief, write `.agent/reports/<batch>.md` (Steps / Files changed / Deviations / Open questions), do not commit, return ≤ 10 lines. The controller re-runs every Verify, reviews the diff, and commits. Haiku workers get mechanical briefs with exact files and checks. The controller checks their data output (counts, duplicates, label sanity) before use.

## 6. Batches and steps

### Batch A1 — P1 tooling (files: `spike/compare.py`, `spike/gate.py`, `tests/test_gate.py`)

S1.1 `compare.py`:
1. Add `--cases PATH` (default `spike/cases.json`).
2. Make `--a-url` optional. Without it: skip engine A; set `agree`, `pdiff`, `cos` to `None`; summary keys `agree`, `pdiff_*`, `cos_*`, `acc_a` are `None`.
3. Each row adds `conf_b`, `probs_b`, `conf_typesafe`, `probs_typesafe` (choice/score: the answer's `confidence` and `probabilities`; noul: `None`). Keep `p_*` for noul.
4. TypeSafe calls use `local=False` and set `JEV_CAPTURE_OFF=1` (see S4.3).
5. Replace the fixed `bg1..bg5` summary with: `bg_risky_*` = rows whose qid starts with `bg` and `expect` is true; `bg_safe_*` = the same with `expect` false. Lists may be empty.
6. Verify: `python -m py_compile spike/compare.py`; run with the service up: `.venv/Scripts/python.exe spike/compare.py --b-url http://127.0.0.1:8090/v1/embeddings --typesafe --out logs/p1-baseline.json` gives `acc_b` and `acc_typesafe` (expected near 16 and 31).

S1.2 `gate.py` (stdlib only, Python 3.13):
1. Input: `--tune PATH --holdout PATH` (compare.py outputs), `--out PATH`.
2. Calibration: one temperature T per type. noul: `p' = sigmoid(logit(p)/T)` (clip p to [1e-6, 1-1e-6]). choice/score: `p'_i ∝ p_i^(1/T)`. Confidence after calibration = top minus mean of others. T grid: 0.25..4.0 in steps of 0.05. `fit_T(cases)` = grid argmin of log-loss. Final T = `fit_T(all tune cases)`. Report LOO log-loss on the tune file (each case scored with `fit_T(others)`) for T and for T = 1. `use_T[type]` = LOO log-loss with T < LOO log-loss with T = 1 (decided on tune only). Ground truth = case `expect`.
3. Gate: noul answer with `lo ≤ p' ≤ hi` → TypeSafe. choice/score answer with `confidence' < c` → TypeSafe. Grid: lo, hi in 0.05 steps with lo ≤ 0.5 ≤ hi; c in 0.02 steps over [0, 1]. Pick per type the setting with the highest tune coverage whose tune kept-accuracy ≥ 95 %; tie → the wider band / higher c. 0 kept cases → kept-accuracy 1.0, coverage 0.
4. Report for tune and hold-out, per type and overall: n, accuracy, log-loss and ECE (10 equal-width bins, top-label confidence) before and after T, coverage, kept-accuracy, gated count; and every `bg` case p before/after T.
5. Correctness uses the caller's decision rule. noul: `(p' ≥ threshold) == expect` (case field `threshold`, default 0.5). choice: argmax == expect. score: case field `tiers: [t1, t2]` (default `[0.5, 1.5]`) maps `Σ i·p'_i` to a level (`< t1` → 0, `< t2` → 1, else 2) == expect. ECE uses top-label confidence and argmax == expect for every type.
6. The hold-out writer sets `threshold` from the hook: bash-guard 0.6, done-check 0.3, bulk 0.7, others 0.5; complexity cases set `tiers` = `[standard_min, deep_min]` from `jev-guard.json` / config defaults. `cases.json` keeps the defaults.
7. Verify: `python -m unittest tests.test_gate -v` passes. Tests (pure functions, no files): temperature on noul and choice, `fit_T(overconfident set) > 1`, gate fit returns 100 % coverage when all answers are right, the 0-kept rule, ECE of a perfect set is 0, threshold and tiers decide correctness.

### Batch A2 — P1 cases (files: `spike/cases-holdout.json`, `spike/PHRASING.md`, `spike/cases-v2.json`, `spike/cases-holdout-v2.json`)

S2.1 `cases-holdout.json` (writer must not see local results):
1. ≥ 30 cases, ids `h01..`. Same shape as `cases.json`: `{id, state, questions: {qid: question}, expect}`, one question per case.
2. Built from the real hook question sets, with the exact hook question text: bash-guard (qid `bg`, ≥ 6 cases, ≥ 3 risky and ≥ 3 safe), done-check (≥ 4), bulk brief (≥ 3), complexity score (≥ 6, the 3 levels of `decisions.py:23-27`, expect = level index), skill route choice (≥ 6, a roster of ≥ 5 skills from the real skill list), plus ≥ 5 other choice/noul cases in the style of `cases.json`.
3. `expect` is the ground truth a careful senior engineer would give. The controller checks every expect.
4. Verify: `python3 -c` script: ≥ 30 cases, unique ids, every type is noul/choice/score, choice expect is a criteria key, score expect is a valid index, ≥ 3 risky and ≥ 3 safe `bg`.

S2.2 `PHRASING.md`: the rules of HANDOFF-accuracy.md §4 1.3, written as numbered rules, with one example per question type.

S2.3 `cases-v2.json`, `cases-holdout-v2.json`: apply PHRASING.md to every case. Keep `id`, `state`, `expect`, qid, question type and choice keys unchanged. Score levels keep their order and count.
Verify: a `python3 -c` script shows the four invariants per file (ids, expect, types, choice keys / level count) are equal to the source file.

### Batch A3 — P1 runs and report (controller)

1. Run compare.py (TypeSafe on) on: `cases.json`, `cases-holdout.json`, `cases-v2.json`, `cases-holdout-v2.json` → `logs/p1-{base,holdout,v2,holdout-v2}.json`.
2. Run gate.py on (base → holdout) and (v2 → holdout-v2) → `logs/p1-gate-{v1,v2}.json`.
3. Write `.agent/reports/P1.md`: table per text version (v1/v2): hold-out accuracy raw, coverage, kept-accuracy, log-loss/ECE before/after T, risky bash p; gate P1 verdict for BOTH versions (no post-hoc pick); the version for the live gate = the better tune result (kept-accuracy, then coverage); proposed hook text changes (not applied).
4. The gate goes live in A4 even if gate P1 fails. It acts only when backend = `"local"`, so it is inert by default.

### Batch A4 — jev_client gate + capture (controller; out of repo)

S4.1 Back up `jev_client.py` and `jev-guard.json` as `*.bak-20261107-accuracy`.
S4.2 Gate: after a CLM answer set, apply the P1 temperature and gate (values from the chosen version in P1.md, stored in `jev-guard.json` `jev.local_gate = {"noul_band": [lo, hi], "min_confidence": {"choice": c, "score": c}, "temperature": {"noul": T, "choice": T, "score": T}}`). If any answer is gated out → whole set to TypeSafe and log `local_clm_gated`. Missing key → no gate, T = 1. T for a type is 1 unless `use_T[type]` (tune LOO, S1.2) is true.
Temperature rules: noul rewrites `noul`. choice rewrites `probabilities` and recomputes `confidence` (top minus mean of others); `choice` is unchanged. score rewrites `probabilities` (ordered by int key) and recomputes `score` = Σ i·p'_i and `confidence`. The gate uses the calibrated values. The gate code runs inside the existing `try` (error → `local_clm_error` → TypeSafe).
S4.3 Capture: `jev.capture = {"enabled": false, "path": "C:/Users/molna/VSCodeRepos/local-clm/data/capture.jsonl", "max_mb": 50}`. On a TypeSafe success with enabled true: append one JSON line `{"ts", "set", "state", "questions", "answers"}` where `set` = `"|".join(f"{qid}:{q['type']}" for qid, q in sorted(questions.items()))`. Skip when the file is ≥ max_mb. Never raises (log `capture_error`). Creates the parent directory. Env `JEV_CAPTURE_OFF=1` disables capture (label.py, compare.py and eval.py set it, so synthetic runs never enter captured data).
S4.4 Tests in `tests/test_jev_local.py`: gate passes a confident set; gate sends a flat noul to TypeSafe; temperature changes a returned p; capture off writes nothing; capture on writes one line; capture over cap writes nothing.
Verify: `.venv/Scripts/python.exe -m unittest discover -s tests -v` all pass; the agent-hooks tests `python3 C:/Users/molna/.claude/agent-hooks/test_done_bash_guard.py` pass.

### Batch B1 — ownyourjev environment (files: `oyj/requirements-oyj.in`, `oyj/requirements-oyj.txt`, `setup.ps1`, `.gitignore`)

1. `requirements-oyj.in`: the ownyourjev `pyproject.toml` dependencies at the pinned commit; torch from PyPI (CPU wheel).
2. `requirements-oyj.txt`: `uv pip compile --generate-hashes --python-version 3.13 oyj/requirements-oyj.in`.
3. `setup.ps1 -WithOyj`: create `.venv-oyj` (Python 3.13), install `requirements-oyj.txt` with `--require-hashes`, install `git+https://github.com/erfaneshrati/ownyourjev@845123fb834fed17dcd84531af6afe468b540e6c` with `--no-deps --no-build-isolation` as a separate command (VCS URLs cannot be hashed; `hatchling` is pinned in requirements-oyj), download the backbone files at the pinned revision (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`) into `models/gte-modernbert-base/` with `Get-Verified` (sha256 of the small files computed once at download and written into setup.ps1).
4. `.gitignore`: add `.venv-oyj/` and `data/`.
5. Verify: `pwsh setup.ps1 -WithOyj` prints `setup: ok`; a second run downloads nothing; `.venv-oyj/Scripts/python.exe -c "import ownyourjev, onnxruntime"` exits 0; `git check-ignore data/x .venv-oyj/x` lists both.

### Batch B2 — synthetic pool (files: `oyj/complexity/pool.jsonl`; haiku workers in parallel, one part file each)

1. 4 workers, each writes `oyj/complexity/pool-<k>.jsonl` with 175 lines `{"uid": "p<k>-<nnn>", "state": {"task": "<description>\n<prompt>"}}`.
2. Each brief looks like a real Claude Code subagent brief: a 3–8 word description line, then the prompt. Mix per worker: about 1/3 trivial/mechanical, 1/3 scoped implementation, 1/3 design/debug/multi-file. Lengths: about 1/3 under 300 chars, 1/3 300–1200, 1/3 1200–3000. Varied languages, repos and tools. No real secrets, no real personal data.
3. The controller merges the parts into `pool.jsonl`, drops exact and near duplicates (same first 80 chars, or word-set Jaccard ≥ 0.8), and checks the count (≥ 600) and the length mix.
4. Verify: `python3` script prints count, duplicate count 0, length buckets.

### Batch B3 — labels and splits (files: `oyj/label.py`, `oyj/complexity/{labeled,seeds,dev,train,test}.jsonl`, `oyj/complexity/CHECK.md`, `tests/test_oyj.py`)

S3.1 `label.py label` (sets `JEV_CAPTURE_OFF=1`; writes JSON with `ensure_ascii=True`): for each pool line, call `jev_client.call_jev(state, questions, local=False)` with exactly the `complexity_tier` shape for one goal (`{"note": "independent task complexity"}`, `{"task_0": score({"question": ..., "task": task}, _COMPLEXITY_LEVELS)}`), imported from `jevlib.decisions`. Write ownyourjev Example lines: `{"uid", "state": {"task": task}, "labels": {"complexity": <argmax level int>}, "meta": {"labeler": "typesafe", "soft": {"complexity": [p0, p1, p2]}, "ts_score": s, "ts_confidence": c}}`. Resume-safe: skip uids already in the output. Errors: log and skip the line.
S3.2 `label.py split --seed 0`: shuffle; seeds 20, dev 200, test 100, train = rest. Stratify by label.
S3.3 Hand check: the controller reads every seed and dev line and writes the gold label. `seeds.jsonl` and `dev.jsonl` use the gold label in `labels` and in `gold`. `CHECK.md` records: n checked, n changed, the changed uids with a one-line reason. For a changed uid, `meta.soft.complexity` = one-hot(gold) (else `--soft` training ignores the correction).
S3.4 Tests: `tests/test_oyj.py` — split sizes and disjointness; label line shape from a fake `call_jev`.
Verify: `python -m unittest tests.test_oyj -v`; counts of each split file; no uid in two splits.

### Batch B4 — train, export, calibrate, eval (controller)

All `.venv-oyj` commands run as `env -u TYPESAFE_API_KEY PYTHONUTF8=1 ...` from the repo root. Check once that no `.env` exists on the path up to `C:/`.
1. `oyj/complexity/spec.json` = `{"questions": {"complexity": {"type": "score", "instructions": "How complex is this task for an AI coding agent?", "criteria": [the 3 levels of decisions.py:23-27 verbatim]}}}`. `train-all.jsonl` = seeds + train concatenated.
2. Train: `ownyourjev train oyj/complexity/spec.json oyj/complexity/train-all.jsonl --qid complexity --backbone models/gte-modernbert-base --lr 3e-5 --method full --soft --max-len 256 --out models/oyj/complexity`. CPU only. Record wall time. If one epoch takes > 60 min, stop and use `--method frozen` instead; record it.
3. Export ONNX int8, then calibrate on `dev.jsonl` (calibrate then fits the int8 runtime, which is the served one).
4. `oyj/serve_onnx.py` (the served runtime, ≤ 40 lines): FastAPI app, `POST /v1/systemone` and `GET /v1/models`, one `OnnxPredictor(dir, dir/"model.int8.onnx", threads=4)` per model dir under the registry, keyed by qid = dir name; unknown qid → 404. uvicorn on `--host 127.0.0.1 --port N`.
5. `oyj/eval.py`: spawns `serve_onnx.py` on port 8711 with `localclm.child_env()`, POSTs each `test.jsonl` state one at a time, then stops it. Metrics:
   - decision agreement (primary): `tier if confident else None` for TypeSafe and for oyj; tier = `config.tier_for_score(score)`; confident = confidence ≥ 0.5, with oyj confidence recomputed from probabilities as top minus mean of others;
   - tier agreement and level agreement (secondary);
   - ECE: oyj top-label probability vs TypeSafe argmax level, 10 equal-width bins;
   - latency p50/p90 at batch size 1, overall and per length bucket (< 300, 300–1200, ≥ 1200 chars);
   - TypeSafe self-agreement: re-label 50 test items with TypeSafe; report decision agreement as the ceiling;
   - gold accuracy on dev (sanity).
   Write `logs/p2-eval.json`.
6. Write `.agent/reports/P2.md` with the numbers and the gate P2 verdict. If the ceiling is < 95 %, say so next to the verdict.

### Batch B5 — serve and route (only if gate P2 passes; files: `localclm.py`, `tests/test_localclm.py`; out of repo `jev_client.py`)

1. `localclm.py`: new `OYJ_PORT = 8701`, `oyj_cmd()` = `.venv-oyj/Scripts/python.exe oyj/serve_onnx.py models/oyj --host 127.0.0.1 --port 8701`, `oyj_up(timeout=0.3)` probes `GET /v1/models`, `start_oyj()` spawns it lazily on its own: own pid file `run/oyj.pid`, own lock `run/oyj.lock` with the same staleness rule, same `_down_active()` check, same `_spawn` (key dropped). It does not depend on CLM files or CLM health. `start()` and the CLM tests stay unchanged. `stop()` also kills `oyj`. `up` also calls `start_oyj()` when `models/oyj/complexity/model.int8.onnx` and `.venv-oyj` exist.
2. `jev_client.call_jev`, after `lc = _local_clm(cfg) if local else None` and BEFORE the 4000-char CLM block: when `lc` is set and the question set has the complexity shape (state == `{"note": "independent task complexity"}`, every question type `score` with `criteria == _COMPLEXITY_LEVELS`, `instructions` a dict whose `question` is the fixed text and which has `task`): `lc.oyj_up()` → POST each task to 8701 as state `{"task": task}` with question `complexity` = the spec.json question; map answers back to `task_<i>`; recompute `confidence` (top minus mean of others) from `probabilities`; return. Down → `lc.start_oyj()` → TypeSafe. Any error → log `local_oyj_error` → TypeSafe. The P1 CLM gate does not apply to oyj answers.
3. Tests: `start_oyj` spawns with CLM up and with CLM files missing; `stop` kills oyj; jev_client complexity shape → oyj; a score question with the same note but other criteria → unchanged path; oyj error → TypeSafe; oyj down → `start_oyj` + TypeSafe.
Verify: full unit suites pass; a live `decisions.complexity_tier(["Fix a typo in README"])` with backend `"local"` in a temp config returns a tier from oyj (the real backend file stays `"typesafe"`).

### Batch C — re-evaluate, review, PR (controller)

1. Re-run A3 step 1 for hold-out v2 and B4 step 3 with the final code. Record in `.agent/reports/FINAL.md` a before/after table.
2. Adversarial independent review (`reviewer` agent) of the whole branch against this SPEC. Fix every Critical/Important.
3. Open the PR to `develop`. Iterate until shippable. Squash-merge into `develop`.

## 7. Acceptance gates

| Gate | Pass when |
|---|---|
| G-SPEC | Independent review done, all Critical/Important fixed, SPEC committed alone. |
| G-A1 | S1.1 and S1.2 Verify pass. |
| G-A2 | S2.1 and S2.3 Verify pass; controller checked every expect. |
| Gate P1 | On hold-out (best text version), gate + calibration: kept-accuracy ≥ 95 %, coverage ≥ 50 %, every risky bash case p ≥ 0.6. Pass or fail is recorded; a fail does not stop P2. |
| G-A4 | All unit tests and agent-hooks tests pass; backups exist; `integration/jev_client_local.md` updated. |
| G-B1 | B1 Verify pass. |
| G-B3 | Splits disjoint; 220 checked; CHECK.md written. |
| Gate P2 | On `test.jsonl`: decision agreement with TypeSafe ≥ 95 % (B4.5), ECE ≤ 0.05, CPU latency p90 ≤ 100 ms (batch size 1). Fail → the set stays on TypeSafe, B5 is skipped, recorded in P2.md. |
| G-C | FINAL.md written; reviewer verdict has no open Critical/Important; PR merged into develop. |

## 8. Guardrails

1. `jev-guard.json` `backend` stays `"typesafe"`. bash-guard keeps `local=False`.
2. Every server binds to `127.0.0.1`. No autostart. Lazy start and manual stop only.
3. Pin every download (commit sha / version + sha256). Never `weights_only=False`, never `trust_remote_code`.
4. `TYPESAFE_API_KEY` never goes to a local process or a child environment (`child_env()` drops it; the oyj spawn uses the same `_spawn`; controller runs of `.venv-oyj` use `env -u TYPESAFE_API_KEY`). No `.env` file on the repo path (ownyourjev loads one).
5. Captured hook data stays under gitignored `data/`. Never commit or upload it. Synthetic pool data may be committed.
6. Tune on `cases*.json` and train/dev; measure on hold-out/test only.
7. Out-of-repo edits: backup first, record in `integration/jev_client_local.md`.
8. Do not change the hook question text in `agent_state_hooks.py` or `decisions.py`.
9. Owner-facing text: plain terms, no task codes.
