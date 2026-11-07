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


def scaled(probs, T):
    """Temperature on probabilities (p_i ∝ p_i^(1/T), same as logits / T) -> (probs, score, confidence)."""
    w = [max(p, 1e-12) ** (1 / T) for p in probs]
    ps = [x / sum(w) for x in w]
    return ps, sum(i * p for i, p in enumerate(ps)), confidence(dict(enumerate(ps)))


def fit_decision_T(dev, cfg, tier_for_score):
    """T on the grid 0.25..4.0 that maximizes hook-decision agreement with TypeSafe on dev; tie -> T closest to 1."""
    def agree(T):
        return sum(decision(*scaled(p, T)[1:], cfg, tier_for_score)
                   == decision(m["ts_score"], m["ts_confidence"], cfg, tier_for_score) for p, m in dev)
    grid = [round(0.25 + 0.05 * i, 2) for i in range(76)]
    return max(grid, key=lambda T: (agree(T), -abs(T - 1))), agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs" / "p2-eval.json"))
    ap.add_argument("--self-n", type=int, default=50)
    ap.add_argument("--registry", default=str(ROOT / "models" / "oyj"))
    ap.add_argument("--test", default=str(label.DIR / "test.jsonl"))
    ap.add_argument("--fit-T", action="store_true", help="serve uncalibrated, fit T on dev-ts, write calibration.json")
    a = ap.parse_args()
    cal = Path(a.registry) / "complexity" / "calibration.json"
    if a.fit_T:
        cal.unlink(missing_ok=True)
    os.environ["JEV_CAPTURE_OFF"] = "1"
    sys.path.insert(0, label.HOOKS)
    from jevlib import config, envload, jev_client
    from jevlib.decisions import _COMPLEXITY_LEVELS
    if a.self_n:
        envload.ensure()  # TypeSafe key only for the test-retest calls; update.py runs with --self-n 0 and no key
    cfg = config.load_config()
    env = {**localclm.child_env(), "PYTHONUTF8": "1"}
    srv = subprocess.Popen([str(ROOT / ".venv-oyj" / "Scripts" / "python.exe"), str(ROOT / "oyj" / "serve_onnx.py"),
                            a.registry, "--port", str(PORT)], cwd=ROOT, env=env)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        test = label.read(a.test)
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
        dev_ok, dev_probs = 0, []
        for r in dev:
            ans, _ = ask(r["state"], _COMPLEXITY_LEVELS)
            p = [ans["probabilities"][str(i)] for i in range(3)]
            dev_ok += p.index(max(p)) == r["gold"]["complexity"]
            dev_probs.append((p, r["meta"]))  # meta keeps the TypeSafe score/confidence also for gold-changed rows
    finally:
        srv.terminate()
        srv.wait(10)
    if a.fit_T:
        T, agree = fit_decision_T(dev_probs, cfg, config.tier_for_score)
        cal.write_text(json.dumps({"method": "temperature", "params": {"T": T}, "n_fit": len(dev_probs),
                                   "source": "oyj/eval.py --fit-T: hook-decision agreement with TypeSafe on dev"}, indent=1))
        print(f"fit T={T} dev decision agreement {agree(T)}/{len(dev_probs)} (T=1: {agree(1.0)})")
        return
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
