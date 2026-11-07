import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, r"C:/Users/molna/.claude/agent-hooks")
from jevlib import jev_client  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
LOCAL_URL = "http://127.0.0.1:8700/v1/systemone"
TS_URL = "https://api.typesafe.ai/v1/systemone"
Q = {"q": {"type": "noul", "instructions": "Is it?"}}
LOCAL_ANS = {"q": {"type": "noul", "noul": 0.9}}
TS_ANS = {"q": {"type": "noul", "noul": 0.1}}


class FakeResp(io.BytesIO):
    def __init__(self, answers):
        super().__init__(json.dumps({"answers": answers}).encode())


class JevLocalTest(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_BASE_URL"}
        env["TYPESAFE_API_KEY"] = "test"
        for p in (mock.patch.dict(os.environ, env, clear=True), mock.patch.object(jev_client, "log_event")):
            p.start()
            self.addCleanup(p.stop)
        self.backend("local")
        p = mock.patch("urllib.request.urlopen", side_effect=self.fake_urlopen)
        self.urlopen = p.start()
        self.addCleanup(p.stop)
        self.local_fails = False

    def tearDown(self):
        jev_client._LOCALCLM = None

    def backend(self, name, home=ROOT):
        cfg = {"jev": {"backend": name, "local_clm_home": home, "model": "jev-latest", "timeout_seconds": 3}}
        p = mock.patch.object(jev_client.config, "load_config", return_value=cfg)
        p.start()
        self.addCleanup(p.stop)

    def fake_urlopen(self, req, timeout=None):
        if req.full_url == LOCAL_URL:
            if self.local_fails:
                raise urllib.error.URLError("refused")
            return FakeResp(LOCAL_ANS)
        return FakeResp(TS_ANS)

    def local(self, up, started=True):
        jev_client._LOCALCLM = mock.Mock(BASE_URL="http://127.0.0.1:8700",
                                         is_up=mock.Mock(return_value=up), start=mock.Mock(return_value=started))
        return jev_client._LOCALCLM

    def urls(self):
        return [c.args[0].full_url for c in self.urlopen.call_args_list]

    def test_backend_typesafe_never_touches_local(self):
        self.backend("typesafe")
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        self.assertIsNone(jev_client._LOCALCLM)
        self.assertEqual(self.urls(), [TS_URL])

    def test_local_up_routes_local(self):
        self.local(up=True)
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), LOCAL_ANS)
        req = self.urlopen.call_args.args[0]
        self.assertEqual(req.full_url, LOCAL_URL)
        self.assertEqual(json.loads(req.data)["model"], "clm-latest")
        self.assertIsNone(req.get_header("Authorization"))

    def test_local_down_starts_and_falls_back(self):
        lc = self.local(up=False)
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        lc.start.assert_called_once_with(probe=False)
        self.assertEqual(self.urls(), [TS_URL])

    def test_local_error_falls_back(self):
        self.local(up=True)
        self.local_fails = True
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        self.assertEqual(self.urls(), [LOCAL_URL, TS_URL])

    def test_local_skips_long_payload(self):
        lc = self.local(up=True)
        self.assertEqual(jev_client.call_jev({"s": "x" * 10_000}, Q), TS_ANS)
        lc.is_up.assert_not_called()
        self.assertEqual(self.urls(), [TS_URL])

    def test_missing_home_falls_back(self):
        self.backend("local", home=str(Path(ROOT) / "no-such-dir"))
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        self.assertEqual(self.urls(), [TS_URL])

    def test_local_timeout_bounded(self):
        self.local(up=True)
        jev_client.call_jev({"s": 1}, Q, timeout=10)
        self.assertLessEqual(self.urlopen.call_args.kwargs["timeout"], 1.5)

    def test_local_false_stays_on_typesafe(self):
        lc = self.local(up=True)
        self.assertEqual(jev_client.safe_decide({"s": 1}, Q, local=False), TS_ANS)
        lc.is_up.assert_not_called()
        self.assertEqual(self.urls(), [TS_URL])

    def test_is_up_raises_falls_back(self):
        lc = self.local(up=True)
        lc.is_up.side_effect = OSError("boom")
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        self.assertEqual(self.urls(), [TS_URL])

    def test_failed_load_not_retried(self):
        self.backend("local", home=str(Path(ROOT) / "no-such-dir"))
        with mock.patch("os.path.isfile", return_value=False) as isfile:
            jev_client.call_jev({"s": 1}, Q)
            jev_client.call_jev({"s": 1}, Q)
        self.assertEqual(isfile.call_count, 1)

    def gate(self, **jev):
        cfg = jev_client.config.load_config()
        cfg["jev"].update(jev)

    def test_gate_keeps_confident(self):
        self.local(up=True)
        self.gate(local_gate={"noul_band": [0.4, 0.6], "min_confidence": {}, "temperature": {}})
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), LOCAL_ANS)
        self.assertEqual(self.urls(), [LOCAL_URL])

    def test_gate_sends_flat_noul_to_typesafe(self):
        self.local(up=True)
        self.gate(local_gate={"noul_band": [0.4, 0.95], "min_confidence": {}, "temperature": {}})
        self.assertEqual(jev_client.call_jev({"s": 1}, Q), TS_ANS)
        self.assertEqual(self.urls(), [LOCAL_URL, TS_URL])
        jev_client._LOCALCLM.start.assert_not_called()

    def test_gate_temperature_rewrites_answers(self):
        a = {"type": "noul", "noul": 0.9}
        jev_client._calibrate(a, 2.0)
        self.assertLess(a["noul"], 0.9)
        s = {"type": "score", "score": 1.0, "confidence": 0.8, "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}
        jev_client._calibrate(s, 2.0)
        self.assertAlmostEqual(sum(s["probabilities"].values()), 1.0)
        self.assertAlmostEqual(s["score"], 1.0)
        self.assertLess(s["confidence"], 0.8)
        c = {"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.25, "b": 0.75}}
        jev_client._calibrate(c, 0.5)
        self.assertEqual(c["choice"], "b")
        self.assertGreater(c["confidence"], 0.5)

    def test_gate_low_choice_confidence(self):
        self.assertTrue(jev_client._gated({"q": {"type": "choice", "confidence": 0.3}},
                                          {"min_confidence": {"choice": 0.5}}))
        self.assertFalse(jev_client._gated({"q": {"type": "choice", "confidence": 0.7}},
                                           {"min_confidence": {"choice": 0.5}}))

    def capture_cfg(self, tmp, enabled=True, max_mb=50):
        path = os.path.join(tmp, "data", "capture.jsonl")
        self.backend("typesafe")
        self.gate(capture={"enabled": enabled, "path": path, "max_mb": max_mb})
        return path

    def test_capture_off_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.capture_cfg(tmp, enabled=False)
            jev_client.call_jev({"s": 1}, Q)
            self.assertFalse(os.path.exists(path))

    def test_capture_on_writes_one_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.capture_cfg(tmp)
            jev_client.call_jev({"s": 1}, Q)
            with open(path, encoding="utf-8") as fh:
                rows = [json.loads(x) for x in fh]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["set"], "q:noul")
            self.assertEqual(rows[0]["answers"], TS_ANS)

    def test_capture_env_off(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"JEV_CAPTURE_OFF": "1"}):
            path = self.capture_cfg(tmp)
            jev_client.call_jev({"s": 1}, Q)
            self.assertFalse(os.path.exists(path))

    def test_capture_over_cap_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.capture_cfg(tmp, max_mb=0)
            os.makedirs(os.path.dirname(path))
            open(path, "w").close()
            jev_client.call_jev({"s": 1}, Q)
            self.assertEqual(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
