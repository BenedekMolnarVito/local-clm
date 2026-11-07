import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oyj"))
import label  # noqa: E402

LEVELS = ["a", "b", "c"]


def row(i, lab):
    return {"uid": f"u{i:03d}", "state": {"task": f"t{i}"}, "labels": {"complexity": lab},
            "meta": {"soft": {"complexity": [1.0 if j == lab else 0.0 for j in range(3)]}}}


class OyjTest(unittest.TestCase):
    def test_split_sizes_disjoint_stratified(self):
        rows = [row(i, i % 3 if i % 5 else 2) for i in range(400)]
        out = label.split(rows, sizes={"seeds": 20, "dev": 200, "test": 100})
        self.assertEqual({k: len(v) for k, v in out.items()}, {"seeds": 20, "dev": 200, "test": 100, "train": 80})
        uids = [r["uid"] for v in out.values() for r in v]
        self.assertEqual(len(uids), len(set(uids)))
        self.assertEqual(len(uids), 400)
        share = lambda rs: sum(r["labels"]["complexity"] == 2 for r in rs) / len(rs)
        self.assertAlmostEqual(share(out["dev"]), share(rows), delta=0.02)

    def test_split_deterministic(self):
        rows = [row(i, i % 3) for i in range(400)]
        self.assertEqual(label.split(rows), label.split(list(reversed(rows))))

    def test_label_row_shape(self):
        seen = {}

        def fake_call(state, questions, local=True):
            seen.update(state=state, questions=questions, local=local)
            return {"task_0": {"score": 1.2, "confidence": 0.4, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}}

        out = label.label_row({"uid": "p1-001", "state": {"task": "Fix\nfix it"}}, fake_call,
                              lambda ins, crit: {"type": "score", "instructions": ins, "criteria": crit}, LEVELS)
        self.assertEqual(seen["state"], {"note": "independent task complexity"})
        self.assertFalse(seen["local"])
        self.assertEqual(seen["questions"]["task_0"]["instructions"]["task"], "Fix\nfix it")
        self.assertEqual(out["labels"], {"complexity": 1})
        self.assertEqual(out["meta"]["soft"]["complexity"], [0.1, 0.6, 0.3])
        self.assertEqual(out["state"], {"task": "Fix\nfix it"})

    def test_apply_gold_changes_soft(self):
        rows = label.apply_gold([row(1, 0), row(2, 1)], {"u001": 2})
        self.assertEqual(rows[0]["gold"], {"complexity": 2})
        self.assertEqual(rows[0]["meta"]["soft"]["complexity"], [0.0, 0.0, 1.0])
        self.assertEqual(rows[1]["gold"], {"complexity": 1})

    def test_merge_drops_near_duplicates(self):
        a = {"uid": "1", "state": {"task": "Rename foo to bar in utils.py and update the three callers"}}
        b = {"uid": "2", "state": {"task": "Rename foo to bar in utils.py and update the three callers now"}}
        c = {"uid": "3", "state": {"task": "Design a cache layer for the search service"}}
        self.assertEqual([r["uid"] for r in label.merge([a, b, c])], ["1", "3"])


if __name__ == "__main__":
    unittest.main()
