"""Synthetic complexity data for ownyourjev: merge pool parts, label with TypeSafe, split.

Run with the system Python (it imports jevlib from ~/.claude/agent-hooks):
  python3 oyj/label.py merge            pool-*.jsonl -> pool.jsonl (drops near duplicates)
  python3 oyj/label.py label            pool.jsonl -> labeled.jsonl (TypeSafe, resume-safe)
  python3 oyj/label.py split [--changes gold-changes.json]   labeled.jsonl -> seeds/dev/test/train
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DIR = Path(__file__).resolve().parent / "complexity"
HOOKS = "C:/Users/molna/.claude/agent-hooks"
QID = "complexity"
SIZES = {"seeds": 20, "dev": 200, "test": 100}  # train = the rest


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)  # ascii: ownyourjev reads with the locale codec


def words(s):
    return set(re.findall(r"\w+", s.lower()))


def merge(parts):
    """Concatenate pool parts; drop exact/near duplicates (same first 80 chars or word Jaccard >= 0.8)."""
    kept, seen, sets = [], set(), []
    for row in parts:
        t = row["state"]["task"]
        w = words(t)
        if t[:80] in seen or any(len(w & s) / max(len(w | s), 1) >= 0.8 for s in sets):
            continue
        seen.add(t[:80])
        sets.append(w)
        kept.append(row)
    return kept


def label_row(row, call_jev, score, levels):
    """One pool row -> ownyourjev Example line labeled by TypeSafe, in the exact complexity_tier shape."""
    task = row["state"]["task"]
    q = {"task_0": score({"question": "How complex is this task for an AI coding agent?", "task": task}, levels)}
    a = call_jev({"note": "independent task complexity"}, q, local=False)["task_0"]
    soft = [float(a["probabilities"][str(i)]) for i in range(len(levels))]
    return {"uid": row["uid"], "state": {"task": task}, "labels": {QID: max(range(len(soft)), key=soft.__getitem__)},
            "meta": {"labeler": "typesafe", "soft": {QID: soft}, "ts_score": a["score"], "ts_confidence": a["confidence"]}}


def split(rows, seed=0, sizes=SIZES):
    """Stratified by label: exact split sizes, largest-remainder allocation per label."""
    rng = random.Random(seed)
    groups = {}
    for r in sorted(rows, key=lambda r: r["uid"]):
        groups.setdefault(r["labels"][QID], []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    n = len(rows)
    out = {k: [] for k in (*sizes, "train")}
    taken = {lab: 0 for lab in groups}
    for name, size in sizes.items():
        share = {lab: size * len(g) / n for lab, g in groups.items()}
        alloc = {lab: int(s) for lab, s in share.items()}
        for lab in sorted(share, key=lambda x: share[x] - alloc[x], reverse=True)[:size - sum(alloc.values())]:
            alloc[lab] += 1
        for lab, k in alloc.items():
            out[name] += groups[lab][taken[lab]:taken[lab] + k]
            taken[lab] += k
    for lab, g in groups.items():
        out["train"] += g[taken[lab]:]
    return out


def apply_gold(rows, changes):
    """Checked rows: gold = TypeSafe label unless changed; a changed row trains on one-hot(gold)."""
    for r in rows:
        g = changes.get(r["uid"], r["labels"][QID])
        if r["uid"] in changes:
            r["labels"][QID] = g
            r["meta"]["soft"][QID] = [1.0 if i == g else 0.0 for i in range(len(r["meta"]["soft"][QID]))]
        r["gold"] = {QID: g}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "label", "split"])
    ap.add_argument("--changes")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "merge":
        parts = [r for p in sorted(DIR.glob("pool-*.jsonl")) for r in read(p)]
        kept = merge(parts)
        write(DIR / "pool.jsonl", kept)
        print(f"merged {len(parts)} -> {len(kept)}")
    elif a.cmd == "label":
        os.environ["JEV_CAPTURE_OFF"] = "1"  # synthetic runs never enter captured data
        sys.path.insert(0, HOOKS)
        from jevlib import envload, jev_client
        from jevlib.decisions import _COMPLEXITY_LEVELS
        envload.ensure()
        out = DIR / "labeled.jsonl"
        done = {r["uid"] for r in read(out)} if out.exists() else set()
        todo = [r for r in read(DIR / "pool.jsonl") if r["uid"] not in done]

        def one(row):
            try:
                return label_row(row, jev_client.call_jev, jev_client.score, _COMPLEXITY_LEVELS)
            except Exception as e:  # noqa: BLE001 — skip the line, a re-run retries it
                print(f"{row['uid']} error {type(e).__name__}: {e}", file=sys.stderr)

        with ThreadPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
            for res in ex.map(one, todo):
                if res:
                    f.write(json.dumps(res, ensure_ascii=True) + "\n")
                    f.flush()
        print(f"labeled {len(read(out))} (this run {len(todo)} attempted)")
    else:
        parts = split(read(DIR / "labeled.jsonl"))
        changes = {k: int(v) for k, v in json.loads(Path(a.changes).read_text()).items()} if a.changes else {}
        for name in ("seeds", "dev"):
            apply_gold(parts[name], changes)
        for name, rows in parts.items():
            write(DIR / f"{name}.jsonl", rows)
        write(DIR / "train-all.jsonl", parts["seeds"] + parts["train"])
        print({k: len(v) for k, v in parts.items()})


if __name__ == "__main__":
    main()
