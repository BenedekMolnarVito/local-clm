"""Fit the P1 confidence gate and per-type temperature on tune results; measure them on hold-out results.

Inputs are compare.py outputs. Run: python spike/gate.py --tune logs/p1-base.json --holdout logs/p1-holdout.json --out PATH
"""
import argparse
import json
import math
from pathlib import Path

T_GRID = [round(0.25 + 0.05 * i, 2) for i in range(76)]  # 0.25 .. 4.0
EPS = 1e-6


def raw(row):
    """Local answer as a float (noul) or a dict key -> probability (choice/score)."""
    return row["p_b"] if row["type"] == "noul" else row["probs_b"]


def calibrate(row, T):
    v = raw(row)
    if row["type"] == "noul":
        p = min(max(v, EPS), 1 - EPS)
        return 1 / (1 + math.exp(-math.log(p / (1 - p)) / T))
    w = {k: max(p, 0.0) ** (1 / T) for k, p in v.items()}
    z = sum(w.values()) or 1.0
    return {k: x / z for k, x in w.items()}


def confidence(dist):
    ps = sorted(dist.values(), reverse=True)
    return ps[0] - sum(ps[1:]) / (len(ps) - 1)


def level(dist, tiers):
    s = sum(int(k) * p for k, p in dist.items())
    return 0 if s < tiers[0] else 1 if s < tiers[1] else 2


def correct(row, cal):
    """Correct at the caller's decision rule."""
    if row["type"] == "noul":
        return (cal >= (row.get("threshold") or 0.5)) == row["expect"]
    if row["type"] == "choice":
        return max(cal, key=cal.get) == row["expect"]
    return level(cal, row.get("tiers") or [0.5, 1.5]) == row["expect"]


def top_label(row, cal):
    """(top-label confidence, argmax correct) for ECE."""
    if row["type"] == "noul":
        return max(cal, 1 - cal), (cal >= 0.5) == row["expect"]
    k = max(cal, key=cal.get)
    return cal[k], (int(k) if row["type"] == "score" else k) == row["expect"]


def logloss(row, cal):
    if row["type"] == "noul":
        p = cal if row["expect"] else 1 - cal
    else:
        p = cal.get(str(row["expect"]), 0.0)
    return -math.log(max(p, EPS))


def mean_ll(rows, T):
    return sum(logloss(r, calibrate(r, T)) for r in rows) / len(rows)


def fit_T(rows):
    return min(T_GRID, key=lambda T: mean_ll(rows, T)) if rows else 1.0


def loo_ll(rows, fixed=None):
    """Mean leave-one-out log-loss; fixed=1.0 scores T = 1 for every case."""
    if len(rows) < 2:
        return None
    out = []
    for i, r in enumerate(rows):
        T = fixed if fixed is not None else fit_T(rows[:i] + rows[i + 1:])
        out.append(logloss(r, calibrate(r, T)))
    return sum(out) / len(out)


def ece(pairs, bins=10):
    if not pairs:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            total += len(sel) * abs(sum(ok for _, ok in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return total / len(pairs)


def is_gated(row, cal, gate):
    if row["type"] == "noul":
        lo, hi = gate["noul_band"]
        return lo <= cal <= hi
    return confidence(cal) < gate["min_confidence"][row["type"]]


def kept_stats(rows, cals, gated):
    kept = [correct(r, c) for r, c, g in zip(rows, cals, gated) if not g]
    n = len(rows)
    return {"coverage": len(kept) / n if n else 0.0, "kept_acc": sum(kept) / len(kept) if kept else 1.0,
            "gated": n - len(kept), "kept": len(kept)}


def fit_gate_type(rows, typ, T, target=0.95):
    """Best gate setting for one type: highest coverage with kept-accuracy >= target; tie -> wider band / higher c."""
    cals = [calibrate(r, T) for r in rows]
    if typ == "noul":
        cands = [(round(0.05 * i, 2), round(0.05 * j, 2)) for i in range(11) for j in range(10, 21)]
        key = lambda b: b[1] - b[0]
        make = lambda b: {"noul_band": list(b), "min_confidence": {}}
    else:
        cands = [round(0.02 * i, 2) for i in range(51)]
        key = lambda c: c
        make = lambda c: {"noul_band": [0.5, 0.5], "min_confidence": {typ: c}}
    best = None
    for cand in cands:
        st = kept_stats(rows, cals, [is_gated(r, c, make(cand)) for r, c in zip(rows, cals)])
        if st["kept_acc"] < target:
            continue
        rank = (st["coverage"], key(cand))
        if best is None or rank > best[0]:
            best = (rank, cand)
    return best[1] if best else (cands[-1])


def by_type(rows):
    out = {}
    for r in rows:
        if raw(r) is not None:
            out.setdefault(r["type"], []).append(r)
    return out


def measure(rows, Ts, gate):
    """Metrics per type and overall for raw (T=1) and calibrated answers with the gate."""
    res, all_rows, all_cals, all_gated = {}, [], [], []
    for typ, rs in by_type(rows).items():
        raw_c = [calibrate(r, 1.0) for r in rs]
        cal_c = [calibrate(r, Ts[typ]) for r in rs]
        gated = [is_gated(r, c, gate) for r, c in zip(rs, cal_c)]
        res[typ] = {"n": len(rs), "T": Ts[typ],
                    "acc_raw": sum(correct(r, c) for r, c in zip(rs, raw_c)) / len(rs),
                    "acc_cal": sum(correct(r, c) for r, c in zip(rs, cal_c)) / len(rs),
                    "logloss_raw": mean_ll(rs, 1.0), "logloss_cal": mean_ll(rs, Ts[typ]),
                    "ece_raw": ece([top_label(r, c) for r, c in zip(rs, raw_c)]),
                    "ece_cal": ece([top_label(r, c) for r, c in zip(rs, cal_c)]),
                    **kept_stats(rs, cal_c, gated)}
        all_rows += rs; all_cals += cal_c; all_gated += gated
    res["overall"] = {"n": len(all_rows), **kept_stats(all_rows, all_cals, all_gated),
                      "acc_raw": sum(correct(r, calibrate(r, 1.0)) for r in all_rows) / max(len(all_rows), 1)}
    res["bg"] = [{"id": r["id"], "expect": r["expect"], "p_raw": r["p_b"], "p_cal": calibrate(r, Ts["noul"])}
                 for r in all_rows if r["qid"].startswith("bg")]
    return res


def run(tune, holdout):
    types = by_type(tune)
    fitted, use_T, loo = {}, {}, {}
    for typ in ("noul", "choice", "score"):
        rs = types.get(typ, [])
        fitted[typ] = fit_T(rs)
        loo[typ] = {"loo_T": loo_ll(rs), "loo_T1": loo_ll(rs, fixed=1.0)}
        use_T[typ] = loo[typ]["loo_T"] is not None and loo[typ]["loo_T"] < loo[typ]["loo_T1"]
    Ts = {typ: fitted[typ] if use_T[typ] else 1.0 for typ in fitted}
    gate = {"noul_band": [0.5, 0.5], "min_confidence": {"choice": 0.0, "score": 0.0}, "temperature": Ts}
    for typ, rs in types.items():
        g = fit_gate_type(rs, typ, Ts[typ])
        if typ == "noul":
            gate["noul_band"] = g
        else:
            gate["min_confidence"][typ] = g
    return {"fitted_T": fitted, "use_T": use_T, "loo": loo, "gate": gate,
            "tune": measure(tune, Ts, gate), "holdout": measure(holdout, Ts, gate)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))["per_case"]
    out = json.dumps(run(load(a.tune), load(a.holdout)), indent=1)
    print(out)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
