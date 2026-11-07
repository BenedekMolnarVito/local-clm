"""Gate P2 evaluation: the int8 ONNX complexity model against TypeSafe labels on the test split.

Run with the system Python: python3 oyj/eval.py [--out logs/p2-eval.json] [--self-n 50]
Starts oyj/serve_onnx.py on 127.0.0.1:8711 (no API key in its env), queries it one state at a time, stops it.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "oyj"))
sys.path.insert(0, str(ROOT / "spike"))
import label  # noqa: E402
import localclm  # noqa: E402
from gate import confidence, ece  # noqa: E402

PORT = 8711
URL = f"http://127.0.0.1:{PORT}"
QUESTION = {"type": "score", "instructions": "How complex is this task for an AI coding agent?"}


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 1) if xs else None


def ask(state, criteria):
    body = json.dumps({"state": state, "questions": {"complexity": {**QUESTION, "criteria": criteria}}}).encode()
    req = urllib.request.Request(URL + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        a = json.loads(r.read())["answers"]["complexity"]
    return a, (time.perf_counter() - t0) * 1000


def decision(score, conf, cfg, tier_for_score):
    return tier_for_score(score, cfg) if conf >= 0.5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs" / "p2-eval.json"))
    ap.add_argument("--self-n", type=int, default=50)
    a = ap.parse_args()
    os.environ["JEV_CAPTURE_OFF"] = "1"
    sys.path.insert(0, label.HOOKS)
    from jevlib import config, envload, jev_client
    from jevlib.decisions import _COMPLEXITY_LEVELS
    envload.ensure()
    cfg = config.load_config()
    env = {**localclm.child_env(), "PYTHONUTF8": "1"}
    srv = subprocess.Popen([str(ROOT / ".venv-oyj" / "Scripts" / "python.exe"), str(ROOT / "oyj" / "serve_onnx.py"),
                            str(ROOT / "models" / "oyj"), "--port", str(PORT)], cwd=ROOT, env=env)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        test = label.read(label.DIR / "test.jsonl")
        for r in test[:3]:
            ask(r["state"], _COMPLEXITY_LEVELS)  # warm-up
        rows = []
        for r in test:
            ans, ms = ask(r["state"], _COMPLEXITY_LEVELS)
            probs = [ans["probabilities"][str(i)] for i in range(3)]
            m = r["meta"]
            rows.append({"uid": r["uid"], "chars": len(r["state"]["task"]), "ms": ms,
                         "ts_level": r["labels"]["complexity"], "oyj_level": probs.index(max(probs)), "oyj_top": max(probs),
                         "ts_dec": decision(m["ts_score"], m["ts_confidence"], cfg, config.tier_for_score),
                         "oyj_dec": decision(ans["score"], confidence(dict(enumerate(probs))), cfg, config.tier_for_score),
                         "ts_tier": config.tier_for_score(m["ts_score"], cfg), "oyj_tier": config.tier_for_score(ans["score"], cfg)})
        dev = label.read(label.DIR / "dev.jsonl")
        dev_ok = 0
        for r in dev:
            ans, _ = ask(r["state"], _COMPLEXITY_LEVELS)
            p = [ans["probabilities"][str(i)] for i in range(3)]
            dev_ok += p.index(max(p)) == r["gold"]["complexity"]
    finally:
        srv.terminate()
        srv.wait(10)
    self_rows = []
    for r in test[:a.self_n]:  # TypeSafe test-retest: the ceiling for agreement
        again = label.label_row(r, jev_client.call_jev, jev_client.score, _COMPLEXITY_LEVELS)["meta"]
        self_rows.append(decision(again["ts_score"], again["ts_confidence"], cfg, config.tier_for_score)
                         == decision(r["meta"]["ts_score"], r["meta"]["ts_confidence"], cfg, config.tier_for_score))
    n = len(rows)
    buckets = {"<300": [x for x in rows if x["chars"] < 300], "300-1200": [x for x in rows if 300 <= x["chars"] < 1200],
               ">=1200": [x for x in rows if x["chars"] >= 1200]}
    s = {"n_test": n,
         "decision_agreement": sum(x["ts_dec"] == x["oyj_dec"] for x in rows) / n,
         "tier_agreement": sum(x["ts_tier"] == x["oyj_tier"] for x in rows) / n,
         "level_agreement": sum(x["ts_level"] == x["oyj_level"] for x in rows) / n,
         "ece_vs_typesafe": ece([(x["oyj_top"], x["oyj_level"] == x["ts_level"]) for x in rows]),
         "lat_p50_ms": pct([x["ms"] for x in rows], 50), "lat_p90_ms": pct([x["ms"] for x in rows], 90),
         "lat_p90_by_len": {k: pct([x["ms"] for x in v], 90) for k, v in buckets.items()},
         "n_by_len": {k: len(v) for k, v in buckets.items()},
         "typesafe_self_agreement": sum(self_rows) / len(self_rows) if self_rows else None, "n_self": len(self_rows),
         "dev_gold_accuracy": dev_ok / len(dev),
         "confident_ts": sum(x["ts_dec"] is not None for x in rows), "confident_oyj": sum(x["oyj_dec"] is not None for x in rows),
         "per_case": rows}
    out = json.dumps(s, indent=1)
    Path(a.out).write_text(out, encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "per_case"}, indent=1))


if __name__ == "__main__":
    main()
