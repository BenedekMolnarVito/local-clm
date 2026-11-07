import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spike"))
import gate  # noqa: E402


def noul(p, expect, threshold=None, qid="q"):
    return {"id": qid, "qid": qid, "type": "noul", "p_b": p, "probs_b": None, "expect": expect, "threshold": threshold}


def choice(probs, expect):
    return {"id": "c", "qid": "q", "type": "choice", "p_b": None, "probs_b": probs, "expect": expect}


def score(probs, expect, tiers=None):
    return {"id": "s", "qid": "q", "type": "score", "p_b": None, "probs_b": probs, "expect": expect, "tiers": tiers}


class GateTest(unittest.TestCase):
    def test_temperature_noul(self):
        r = noul(0.8, True)
        self.assertAlmostEqual(gate.calibrate(r, 1.0), 0.8)
        self.assertLess(gate.calibrate(r, 2.0), 0.8)
        self.assertGreater(gate.calibrate(r, 0.5), 0.8)

    def test_temperature_choice(self):
        r = choice({"a": 0.6, "b": 0.3, "c": 0.1}, "a")
        c = gate.calibrate(r, 2.0)
        self.assertAlmostEqual(sum(c.values()), 1.0)
        self.assertLess(c["a"], 0.6)
        self.assertEqual(max(c, key=c.get), "a")

    def test_fit_T_overconfident(self):
        rows = [noul(0.95, i % 3 != 0) for i in range(30)]  # 95 % sure, 67 % right
        self.assertGreater(gate.fit_T(rows), 1.0)

    def test_full_coverage_when_all_right(self):
        rows = [noul(0.9, True), noul(0.1, False), noul(0.7, True)]
        band = gate.fit_gate_type(rows, "noul", 1.0)
        st = gate.kept_stats(rows, [r["p_b"] for r in rows],
                             [gate.is_gated(r, r["p_b"], {"noul_band": band}) for r in rows])
        self.assertEqual(st["coverage"], 1.0)

    def test_gate_drops_flat_wrong(self):
        rows = [noul(0.9, True), noul(0.1, False), noul(0.55, False), noul(0.45, True)]
        lo, hi = gate.fit_gate_type(rows, "noul", 1.0)
        self.assertTrue(lo <= 0.45 and hi >= 0.55)

    def test_zero_kept(self):
        self.assertEqual(gate.kept_stats([noul(0.5, True)], [0.5], [True]),
                         {"coverage": 0.0, "kept_acc": 1.0, "gated": 1, "kept": 0})

    def test_ece_perfect(self):
        self.assertEqual(gate.ece([(1.0, True), (1.0, True), (0.0, False)]), 0.0)

    def test_threshold_and_tiers(self):
        self.assertTrue(gate.correct(noul(0.4, True, threshold=0.3), 0.4))
        self.assertFalse(gate.correct(noul(0.4, True), 0.4))
        r = score({"0": 0.5, "1": 0.5, "2": 0.0}, 0, tiers=[0.6, 1.5])  # expected level 0.5 < 0.6
        self.assertTrue(gate.correct(r, r["probs_b"]))
        self.assertFalse(gate.correct(score(r["probs_b"], 0), r["probs_b"]))  # default tiers: 0.5 -> level 1

    def test_run_shapes(self):
        tune = [noul(0.9, True), noul(0.2, False), choice({"a": 0.7, "b": 0.3}, "a"),
                score({"0": 0.1, "1": 0.8, "2": 0.1}, 1)]
        out = gate.run(tune, tune)
        self.assertEqual(set(out["gate"]["temperature"]), {"noul", "choice", "score"})
        self.assertEqual(out["holdout"]["overall"]["n"], 4)


if __name__ == "__main__":
    unittest.main()
