"""Lazy launcher for the local CLM service (llama-server encoder + clm.server heads).

CLI: python localclm.py <up|down|status> [--wait SECONDS]
Imported by jev_client.py; must stay stdlib-only and silent on import.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLAMA_EXE = ROOT / "bin" / "llama" / "llama-server.exe"
PY_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = ROOT / "models" / os.environ.get("LOCAL_CLM_GGUF", "Qwen3-8B-Q6_K.gguf")
CKPT = Path(os.environ.get("CLM_CKPT") or Path.home() / ".cache" / "clm" / "CLM_v0.1-8B.pt")
RUN = ROOT / "run"
LOGS = ROOT / "logs"
LOCK = RUN / "start.lock"
DOWN_FLAG = RUN / "down.flag"
EMB_PORT = 8090
CLM_PORT = 8700
BASE_URL = f"http://127.0.0.1:{CLM_PORT}"
LOCK_STALE_S = 300

BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
FLAGS = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | BREAKAWAY


def is_up(timeout: float = 0.3) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as r:
            if r.status != 200:
                return False
            d = json.loads(r.read())
            return d.get("ok") is True and d.get("embedder") is True
    except Exception:
        return False


def llama_cmd() -> list[str]:
    return [str(LLAMA_EXE), "-m", str(MODEL), "--embeddings", "--pooling", "last",
            "-c", "2048", "-b", "2048", "-ub", "2048", "-np", "1", "-ngl", "99",
            "--host", "127.0.0.1", "--port", str(EMB_PORT), "--alias", "qwen3-8b"]


def clm_cmd() -> list[str]:
    return [str(PY_EXE), "-m", "clm.server", "--host", "127.0.0.1", "--port", str(CLM_PORT),
            "--emb-url", f"http://127.0.0.1:{EMB_PORT}/v1/embeddings", "--emb-model", "qwen3-8b",
            "--max-tokens", "2048", "--device", "cpu", "--no-download", "--no-ui", "--ckpt", str(CKPT)]


def child_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in ("TYPESAFE_API_KEY", "CLM_API_KEY")}


def _boot_id() -> int:
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters") as key:
        return winreg.QueryValueEx(key, "BootId")[0]


def _write_down_flag():
    RUN.mkdir(parents=True, exist_ok=True)
    tmp = DOWN_FLAG.with_suffix(".tmp")
    tmp.write_text(str(_boot_id()))
    os.replace(tmp, DOWN_FLAG)


def _down_active() -> bool:
    if not DOWN_FLAG.exists():
        return False
    try:
        if int(DOWN_FLAG.read_text()) == _boot_id():
            return True
    except ValueError:
        pass
    DOWN_FLAG.unlink(missing_ok=True)  # flag from an earlier boot, or unreadable
    return False


def _created(pid: int) -> float | None:
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        t = [ctypes.c_ulonglong() for _ in range(4)]
        if not k.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        return t[0].value / 1e7 - 11644473600  # FILETIME (100 ns since 1601) -> Unix
    finally:
        k.CloseHandle(h)


def _alive(pid: int, image: str, since: float) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=2).stdout  # hook path: never hang
        if image.lower() not in out.lower():
            return False
        c = _created(pid)
        return c is not None and since - 30 <= c <= since + 1  # rejects a reused PID
    except Exception:
        return False


def _read_pid(pid_file: Path) -> int:
    try:
        return int(pid_file.read_text())
    except ValueError:
        return 0  # empty/corrupt file (crash mid-write); PID 0 never matches our image


def _log(line: str):
    with open(LOGS / "start.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def _take_lock() -> bool:
    for _ in range(2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - LOCK.stat().st_mtime < LOCK_STALE_S:
                return False  # another start is in progress
            # shortcut: two starts racing on one stale lock can both spawn; upgrade to a named mutex if double spawns show up in start.log
            LOCK.unlink(missing_ok=True)
            continue
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    return False


def _spawn(cmd: list[str], log_path: Path) -> tuple[int, bool]:
    with open(log_path, "ab") as log:
        kw = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                  env=child_env(), close_fds=True)
        try:
            return subprocess.Popen(cmd, creationflags=FLAGS, **kw).pid, True
        except OSError:
            return subprocess.Popen(cmd, creationflags=FLAGS & ~BREAKAWAY, **kw).pid, False


def start(probe: bool = True) -> bool:
    locked = False
    try:
        if _down_active():
            return False
        if probe and is_up():
            return False
        RUN.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        if not _take_lock():
            return False
        locked = True
        for p in (LLAMA_EXE, PY_EXE, MODEL, CKPT):
            if not Path(p).exists():
                LOCK.unlink(missing_ok=True)
                _log(f"missing {p}")
                return False
        for name, cmd, image in (("llama-server", llama_cmd(), "llama-server.exe"),
                                 ("clm-serve", clm_cmd(), "python.exe")):
            pid_file = RUN / f"{name}.pid"
            if pid_file.exists() and _alive(_read_pid(pid_file), image, pid_file.stat().st_mtime):
                continue
            if _down_active():  # a `down` ran meanwhile
                LOCK.unlink(missing_ok=True)
                return False
            pid, breakaway = _spawn(cmd, LOGS / f"{name}.log")
            pid_file.write_text(str(pid))
            _log(f"spawn {name} pid={pid} breakaway={breakaway}")
        return True  # lock stays; wait_up()/stop() or staleness clears it
    except Exception as e:
        try:
            if locked:
                LOCK.unlink(missing_ok=True)
            _log(f"error {type(e).__name__}: {e}")
        except Exception:
            pass
        return False


def wait_up(timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if is_up():
            LOCK.unlink(missing_ok=True)
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def stop() -> list[int]:
    _write_down_flag()  # first, so a concurrent start() stops before its next spawn
    killed = []
    for name, image in (("clm-serve", "python.exe"), ("llama-server", "llama-server.exe")):
        pid_file = RUN / f"{name}.pid"
        if not pid_file.exists():
            continue
        pid = _read_pid(pid_file)
        if _alive(pid, image, pid_file.stat().st_mtime):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            killed.append(pid)
        pid_file.unlink(missing_ok=True)
    LOCK.unlink(missing_ok=True)
    return killed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="localclm.py")
    ap.add_argument("cmd", choices=["up", "down", "status"])
    ap.add_argument("--wait", type=float, metavar="SECONDS")
    a = ap.parse_args(argv)
    if a.cmd == "up":
        DOWN_FLAG.unlink(missing_ok=True)
        start()
        if a.wait is not None:
            wait_up(a.wait)
        if is_up():
            print("up")
            return 0
        print("failed" if a.wait is not None else "starting")
        return 1 if a.wait is not None else 0
    if a.cmd == "down":
        print(f"stopped {stop()}")
        return 0
    up = is_up()
    print("up" if up else "down")
    return 0 if up else 3


if __name__ == "__main__":
    sys.exit(main())
