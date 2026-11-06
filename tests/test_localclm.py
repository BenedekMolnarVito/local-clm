import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import localclm  # noqa: E402

REAL_IS_UP = localclm.is_up
REAL_DOWN_ACTIVE = localclm._down_active


class FakeResp(io.BytesIO):
    status = 200


class LocalClmTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        run = self.tmp / "run"
        self.patch("RUN", run)
        self.patch("LOGS", self.tmp / "logs")
        self.patch("LOCK", run / "start.lock")
        self.patch("DOWN_FLAG", run / "down.flag")
        self.patch("_down_active", return_value=False)
        self.is_up = self.patch("is_up", return_value=False)
        self.alive = self.patch("_alive", return_value=False)
        self.popen = self.patch("subprocess.Popen", side_effect=[mock.Mock(pid=p) for p in (111, 222, 333)])
        self.run = self.patch("subprocess.run")
        self.patch("time.sleep")
        self.patch("_boot_id", return_value=223)

    def patch(self, name, *new, **kw):
        p = mock.patch(f"localclm.{name}", *new, **kw)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def exe_ok(self):
        for name in ("LLAMA_EXE", "PY_EXE", "MODEL", "CKPT"):
            f = self.tmp / name
            f.touch()
            self.patch(name, f)

    def start_log(self):
        return (localclm.LOGS / "start.log").read_text()

    def test_is_up_false_when_unreachable(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")):
            self.assertFalse(REAL_IS_UP())

    def test_is_up_requires_embedder(self):
        body = lambda e: FakeResp(json.dumps({"ok": True, "embedder": e}).encode())
        with mock.patch("urllib.request.urlopen", return_value=body(False)):
            self.assertFalse(REAL_IS_UP())
        with mock.patch("urllib.request.urlopen", return_value=body(True)):
            self.assertTrue(REAL_IS_UP())

    def test_start_noop_when_up(self):
        self.is_up.return_value = True
        self.assertFalse(localclm.start())
        self.popen.assert_not_called()

    def test_start_spawns_two_and_writes_pids(self):
        self.exe_ok()
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}):
            self.assertTrue(localclm.start())
        self.assertEqual(self.popen.call_count, 2)
        self.assertEqual(self.popen.call_args_list[0].args[0][0], str(localclm.LLAMA_EXE))
        for c in self.popen.call_args_list:
            self.assertNotIn("TYPESAFE_API_KEY", c.kwargs["env"])
        self.assertEqual((localclm.RUN / "llama-server.pid").read_text(), "111")
        self.assertEqual((localclm.RUN / "clm-serve.pid").read_text(), "222")
        self.assertTrue(localclm.LOCK.exists())

    def test_start_respects_fresh_lock(self):
        self.exe_ok()
        localclm.RUN.mkdir(parents=True)
        localclm.LOCK.write_text("x")
        self.assertFalse(localclm.start())
        self.popen.assert_not_called()

    def test_start_replaces_stale_lock(self):
        self.exe_ok()
        localclm.RUN.mkdir(parents=True)
        localclm.LOCK.write_text("x")
        old = time.time() - 400
        os.utime(localclm.LOCK, (old, old))
        self.assertTrue(localclm.start())

    def test_start_missing_binary(self):
        self.exe_ok()
        missing = self.tmp / "nope" / "llama-server.exe"
        self.patch("LLAMA_EXE", missing)
        self.assertFalse(localclm.start())
        self.popen.assert_not_called()
        self.assertIn(str(missing), self.start_log())
        self.assertFalse(localclm.LOCK.exists())

    def test_start_retries_without_breakaway(self):
        self.exe_ok()
        self.popen.side_effect = [OSError("breakaway denied"), mock.Mock(pid=111), mock.Mock(pid=222)]
        self.assertTrue(localclm.start())
        self.assertEqual(self.popen.call_args_list[1].kwargs["creationflags"] & 0x01000000, 0)
        self.assertIn("breakaway=False", self.start_log())

    def test_wait_up_clears_lock(self):
        localclm.RUN.mkdir(parents=True)
        localclm.LOCK.write_text("x")
        self.is_up.side_effect = [False, True]
        self.assertTrue(localclm.wait_up(5))
        self.assertFalse(localclm.LOCK.exists())

    def test_stop_skips_dead_or_foreign_pid(self):
        localclm.RUN.mkdir(parents=True)
        (localclm.RUN / "clm-serve.pid").write_text("111")
        (localclm.RUN / "llama-server.pid").write_text("222")
        localclm.LOCK.write_text("x")
        self.alive.side_effect = lambda pid, image, since: pid == 222
        self.assertEqual(localclm.stop(), [222])
        self.run.assert_called_once()
        argv = self.run.call_args.args[0]
        self.assertIn("/T", argv)
        self.assertIn("/F", argv)
        self.assertFalse((localclm.RUN / "clm-serve.pid").exists())
        self.assertFalse((localclm.RUN / "llama-server.pid").exists())
        self.assertFalse(localclm.LOCK.exists())
        self.assertTrue(localclm.DOWN_FLAG.exists())

    def test_start_blocked_after_down(self):
        self.patch("_down_active", REAL_DOWN_ACTIVE)
        localclm.stop()
        self.assertEqual(localclm.DOWN_FLAG.read_text(), "223")
        self.assertFalse(localclm.start())
        self.popen.assert_not_called()
        localclm.DOWN_FLAG.write_text("222")  # earlier boot
        self.exe_ok()
        self.assertTrue(localclm.start())
        self.assertFalse(localclm.DOWN_FLAG.exists())

    def test_start_skips_live_component(self):
        self.exe_ok()
        localclm.RUN.mkdir(parents=True)
        (localclm.RUN / "llama-server.pid").write_text("222")
        self.alive.side_effect = lambda pid, image, since: pid == 222
        self.assertTrue(localclm.start())
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.popen.call_args.args[0][0], str(localclm.PY_EXE))

    def test_corrupt_pid_file(self):
        self.exe_ok()
        localclm.RUN.mkdir(parents=True)
        (localclm.RUN / "clm-serve.pid").write_text("")
        (localclm.RUN / "llama-server.pid").write_text("222")
        self.alive.side_effect = lambda pid, image, since: pid == 222
        self.assertEqual(localclm.stop(), [222])  # empty file does not abort the llama kill
        localclm.DOWN_FLAG.unlink()
        (localclm.RUN / "clm-serve.pid").write_text("garbage")
        self.assertTrue(localclm.start())


class AliveTest(unittest.TestCase):
    """Real Win32 calls against a real child process (no mocks)."""

    def test_alive_checks_image_and_start_time(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(p.kill)
        since = time.time()  # like the pid file mtime, written right after Popen
        self.assertTrue(localclm._alive(p.pid, Path(sys.executable).name, since))
        self.assertFalse(localclm._alive(p.pid, "llama-server.exe", since))
        self.assertFalse(localclm._alive(p.pid, Path(sys.executable).name, since + 100))  # reused PID
        self.assertFalse(localclm._alive(10**20, "python.exe", since))  # corrupt pid file
        p.kill()
        p.wait()
        self.assertFalse(localclm._alive(p.pid, Path(sys.executable).name, since))


if __name__ == "__main__":
    unittest.main()
