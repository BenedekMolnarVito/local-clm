"""Compare a reference encoder (A, optional) with a candidate (B), and optionally TypeSafe, on a cases file.

Run: .venv/Scripts/python.exe spike/compare.py [--a-url URL] --b-url URL [--cases PATH] [--typesafe] [--out PATH]
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from localclm import CKPT  # noqa: E402


def to_dict(a):
    return a.to_dict() if hasattr(a, "to_dict") else a


def top(q, a):
    if a is None:
        return None
    if q["type"] == "noul":
        return a["noul"] >= 0.5
    if q["type"] == "choice":
        return a["choice"]
    return round(a["score"])


def pdiff(q, a, b):
    if q["type"] == "noul":
        return abs(a["noul"] - b["noul"])
    if q["type"] == "choice":
        return max(abs(a["probabilities"][k] - b["probabilities"].get(k, 0.0)) for k in a["probabilities"])
    return abs(a["score"] - b["score"]) / (len(q["criteria"]) - 1)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-url")
    ap.add_argument("--b-url", required=True)
    ap.add_argument("--typesafe", action="store_true")
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--out")
    args = ap.parse_args()

    import clm
    embA = clm.Embedder(args.a_url, "qwen3-8b", max_tokens=2048) if args.a_url else None
    embB = clm.Embedder(args.b_url, "qwen3-8b", max_tokens=2048)
    engA = clm.Engine(embA, checkpoint=str(CKPT), device="cpu") if embA else None
    engB = clm.Engine(embB, checkpoint=str(CKPT), device="cpu")
    jc = None
    if args.typesafe:
        sys.path.insert(0, "C:/Users/molna/.claude/agent-hooks")
        os.environ["JEV_CAPTURE_OFF"] = "1"  # spike runs never enter captured data
        from jevlib import envload, jev_client as jc
        envload.ensure()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    shown, per_case, lat_b = set(), [], []
    for case in cases:
        qid, q = next(iter(case["questions"].items()))
        ans = {}
        ans["a"] = to_dict(engA.answer(case["state"], case["questions"])["answers"][qid]) if engA else None
        t0 = time.perf_counter()
        ans["b"] = to_dict(engB.answer(case["state"], case["questions"])["answers"][qid])
        ms = (time.perf_counter() - t0) * 1000
        lat_b.append(ms)
        ans["typesafe"] = None
        if jc:
            try:
                ans["typesafe"] = jc.call_jev(case["state"], case["questions"], local=False)[qid]
            except Exception as e:
                print(f"{case['id']} typesafe error: {type(e).__name__}", file=sys.stderr)
        for src, a in ans.items():
            if a is not None and src not in shown:
                shown.add(src)
                print(f"first answer {src}: {json.dumps(a)}", file=sys.stderr)
        cos = None
        if embA:
            text = json.dumps(case["state"])
            a, _ = embA.embed([text])
            b, _ = embB.embed([text])
            cos = float(a[0] @ b[0])
        row = {"id": case["id"], "qid": qid, "type": q["type"], "expect": case["expect"],
               "threshold": case.get("threshold"), "tiers": case.get("tiers"),
               "cos": cos, "lat_b_ms": round(ms, 1),
               "agree": top(q, ans["a"]) == top(q, ans["b"]) if engA else None,
               "pdiff": pdiff(q, ans["a"], ans["b"]) if engA else None}
        for src, a in ans.items():
            row[f"top_{src}"] = top(q, a)
            row[f"correct_{src}"] = a is not None and top(q, a) == case["expect"]
            row[f"p_{src}"] = a.get("noul") if a and q["type"] == "noul" else None
            row[f"conf_{src}"] = a.get("confidence") if a and q["type"] != "noul" else None
            row[f"probs_{src}"] = a.get("probabilities") if a and q["type"] != "noul" else None
        per_case.append(row)

    bg = [r for r in per_case if r["qid"].startswith("bg")]
    risky = [r for r in bg if r["expect"] is True]
    safe = [r for r in bg if r["expect"] is False]
    s = {
        "n": len(per_case),
        "agree": sum(r["agree"] for r in per_case) if engA else None,
        "pdiff_mean": statistics.mean(r["pdiff"] for r in per_case) if engA else None,
        "pdiff_max": max(r["pdiff"] for r in per_case) if engA else None,
        "cos_min": min(r["cos"] for r in per_case) if engA else None,
        "cos_mean": statistics.mean(r["cos"] for r in per_case) if engA else None,
        "acc_a": sum(r["correct_a"] for r in per_case) if engA else None,
        "acc_b": sum(r["correct_b"] for r in per_case),
        "acc_typesafe": sum(r["correct_typesafe"] for r in per_case) if jc else None,
        "typesafe_null": sum(r["top_typesafe"] is None for r in per_case) if jc else None,
        "bg_risky_p_b": [r["p_b"] for r in risky],
        "bg_safe_p_b": [r["p_b"] for r in safe],
        "bg_risky_p_typesafe": [r["p_typesafe"] for r in risky],
        "bg_safe_p_typesafe": [r["p_typesafe"] for r in safe],
        "lat_b_p50_ms": round(pct(lat_b, 50), 1),
        "lat_b_p90_ms": round(pct(lat_b, 90), 1),
        "lat_b_c31_ms": round(lat_b[-1], 1),
        "per_case": per_case,
    }
    out = json.dumps(s, indent=1)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
