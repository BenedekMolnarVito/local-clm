"""Weight update for the ownyourjev complexity model from captured TypeSafe answers.

Manual:  python3 oyj/update.py           (any amount of new captured data)
Auto:    python3 oyj/update.py --auto    (started by localclm.note_capture; needs UPDATE_MIN_NEW new tasks)
Steps: capture -> examples (TypeSafe answers are the labels) -> retrain on synthetic + captured -> export int8
-> fit T on dev -> eval current and candidate on synthetic test + captured hold-out -> promote only if not worse.
No TypeSafe call and no API key: every child gets localclm.child_env().
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "oyj"))
import label  # noqa: E402
import localclm  # noqa: E402

QUESTION = "How complex is this task for an AI coding agent?"
WORK = localclm.DATA / "oyj"
NEXT = ROOT / "models" / "oyj-next"
PREV = ROOT / "models" / "oyj-prev"
OYJ = ROOT / ".venv-oyj" / "Scripts" / "ownyourjev.exe"


def captured_examples(lines, levels):
    """Capture lines of the complexity_tier shape -> ownyourjev Example rows, one per task, deduped by task text."""
    out = {}
    for c in lines:
        if c.get("state") != {"note": "independent task complexity"}:
            continue
        for qid, q in (c.get("questions") or {}).items():
            ins, a = q.get("instructions"), (c.get("answers") or {}).get(qid)
            if (q.get("type") != "score" or q.get("criteria") != levels or not isinstance(ins, dict)
                    or ins.get("question") != QUESTION or not isinstance(ins.get("task"), str) or not a):
                continue
            soft = [float(a["probabilities"][str(i)]) for i in range(len(levels))]
            uid = "c-" + hashlib.sha1(ins["task"].encode("utf-8")).hexdigest()[:12]
            out[uid] = {"uid": uid, "state": {"task": ins["task"]}, "labels": {label.QID: soft.index(max(soft))},
                        "meta": {"labeler": "typesafe", "soft": {label.QID: soft},
                                 "ts_score": a["score"], "ts_confidence": a["confidence"]}}
    return list(out.values())


def buckets(rows):
    """Split captured rows by task hash: 1/5 test, 1/5 calibration, 3/5 train. A task never moves between buckets."""
    b = {"train": [], "cal": [], "test": []}
    for r in rows:
        b[{0: "test", 1: "cal"}.get(int(r["uid"][2:], 16) % 5, "train")].append(r)
    return b


def agreement(ev, captured):
    rows = [x for x in ev["per_case"] if x["uid"].startswith("c-") == captured]
    return sum(x["ts_dec"] == x["oyj_dec"] for x in rows) / len(rows) if rows else None


def better(new, cur, synth_tol=0.02):
    """Promote rule (same test set for both): no loss on decision and level agreement overall, and at most
    synth_tol loss on the synthetic part, so a model that only fits recent hook traffic is not kept."""
    return (new["decision_agreement"] >= cur["decision_agreement"] and new["level_agreement"] >= cur["level_agreement"]
            and agreement(new, False) >= agreement(cur, False) - synth_tol)


def run(cmd, log):
    with open(log, "ab") as f:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={**localclm.child_env(), "PYTHONUTF8": "1"},
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def evaluate(registry, test, out, log):
    run([sys.executable, ROOT / "oyj" / "eval.py", "--registry", registry, "--test", test, "--self-n", "0", "--out", out], log)
    return json.loads(Path(out).read_text())


def save_state(**kw):
    try:
        st = json.loads(localclm.UPDATE_STATE.read_text())
    except (OSError, ValueError):
        st = {}
    st.update(kw)
    localclm.UPDATE_STATE.write_text(json.dumps(st, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, label.HOOKS)
    from jevlib import config
    from jevlib.decisions import _COMPLEXITY_LEVELS
    localclm.DATA.mkdir(parents=True, exist_ok=True)
    if not localclm._take_lock(localclm.UPDATE_LOCK, stale=localclm.UPDATE_STALE_S):
        print("update: another update is running")
        return 0
    try:
        save_state(last_run=time.time(), pending=0)  # captures during the run count toward the next window
        cap = Path(config.load_config()["jev"].get("capture", {}).get("path", localclm.DATA / "capture.jsonl"))
        rows = captured_examples(label.read(cap) if cap.exists() else [], _COMPLEXITY_LEVELS)
        need = localclm.UPDATE_MIN_NEW if a.auto else 1
        if len(rows) < need:
            save_state(last_result={"ts": time.time(), "status": "not_enough_data", "n_captured": len(rows)})
            print(f"update: {len(rows)} captured tasks, need {need}")
            return 0
        WORK.mkdir(parents=True, exist_ok=True)
        log = localclm.LOGS / "oyj-update-run.log"
        b = buckets(rows)
        # always from the base backbone on all data (no fine-tune on top of the last model): no drift into recent traffic
        label.write(WORK / "train-all.jsonl", label.read(label.DIR / "train-all.jsonl") + b["train"])
        label.write(WORK / "dev.jsonl", label.read(label.DIR / "dev-ts.jsonl") + b["cal"])
        label.write(WORK / "test.jsonl", label.read(label.DIR / "test.jsonl") + b["test"])
        shutil.rmtree(NEXT, ignore_errors=True)
        run([OYJ, "train", label.DIR / "spec.json", WORK / "train-all.jsonl", "--qid", label.QID,
             "--backbone", ROOT / "models" / "gte-modernbert-base", "--lr", "3e-5", "--method", "full", "--soft",
             "--max-len", "256", "--out", NEXT / label.QID], log)
        run([OYJ, "export", NEXT / label.QID], log)
        run([sys.executable, ROOT / "oyj" / "eval.py", "--registry", NEXT, "--fit-T", "--self-n", "0",
             "--dev", WORK / "dev.jsonl"], log)
        new = evaluate(NEXT, WORK / "test.jsonl", WORK / "eval-next.json", log)
        cur = evaluate(localclm.OYJ_REGISTRY, WORK / "test.jsonl", WORK / "eval-current.json", log)
        keys = ("decision_agreement", "level_agreement", "ece_vs_typesafe", "lat_p90_ms")
        result = {"ts": time.time(), "n_captured": len(rows), **{f"n_cap_{k}": len(v) for k, v in b.items()},
                  "current": {**{k: cur[k] for k in keys}, "synthetic": agreement(cur, False), "captured": agreement(cur, True)},
                  "candidate": {**{k: new[k] for k in keys}, "synthetic": agreement(new, False), "captured": agreement(new, True)},
                  "promoted": better(new, cur)}
        if result["promoted"]:
            localclm.stop_oyj()  # Windows keeps the model files open while it runs; the next hook call restarts it
            shutil.rmtree(PREV, ignore_errors=True)
            localclm.OYJ_REGISTRY.rename(PREV)
            NEXT.rename(localclm.OYJ_REGISTRY)
        save_state(last_result=result)
        print(json.dumps(result, indent=1))
        return 0
    except Exception as e:  # noqa: BLE001 — record and leave the current model in place
        save_state(last_result={"ts": time.time(), "status": "error", "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        localclm.UPDATE_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
