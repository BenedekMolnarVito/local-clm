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


if __name__ == "__main__":
    unittest.main()
