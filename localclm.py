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
OYJ_PY = ROOT / ".venv-oyj" / "Scripts" / "python.exe"
OYJ_REGISTRY = ROOT / "models" / "oyj"
OYJ_MODEL = OYJ_REGISTRY / "complexity" / "model.int8.onnx"
OYJ_PORT = 8701
OYJ_URL = f"http://127.0.0.1:{OYJ_PORT}"
DATA = ROOT / "data"
UPDATE_STATE = DATA / "oyj-update.json"
UPDATE_LOCK = DATA / "oyj-update.lock"
UPDATE_MIN_NEW = 200  # new captured complexity tasks per auto weight update: +50 % on the 399 synthetic train rows
UPDATE_MIN_DAYS = 2  # spacing between auto updates (each is ~20+ min of idle-priority CPU training)
UPDATE_STALE_S = 6 * 3600

BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
IDLE = 0x00000040  # IDLE_PRIORITY_CLASS
# CREATE_NO_WINDOW, not DETACHED_PROCESS: a venv python.exe respawns the base python, and without a console that child opens a visible one
FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | BREAKAWAY


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


def oyj_cmd() -> list[str]:
    return [str(OYJ_PY), str(ROOT / "oyj" / "serve_onnx.py"), str(OYJ_REGISTRY),
            "--host", "127.0.0.1", "--port", str(OYJ_PORT)]


def oyj_up(timeout: float = 0.3) -> bool:
    try:
        with urllib.request.urlopen(OYJ_URL + "/v1/models", timeout=timeout) as r:
            return r.status == 200 and any(m.get("id") == "complexity" for m in json.loads(r.read()).get("data", []))
    except Exception:
        return False


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


def _alive(pid: int, image: str, since: float) -> bool:
    # Win32 calls, not tasklist: no subprocess on the hook path, so no hang and no timeout guess
    k = ctypes.windll.kernel32
    try:
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    except ctypes.ArgumentError:
        return False  # pid too large for a DWORD (corrupt pid file)
    if not h:
        return False
    try:
        buf, n = ctypes.create_unicode_buffer(32768), ctypes.c_ulong(32768)
        if not k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return False
        t = [ctypes.c_ulonglong() for _ in range(4)]
        if not k.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return False
        created = t[0].value / 1e7 - 11644473600  # FILETIME (100 ns since 1601) -> Unix
        return Path(buf.value).name.lower() == image.lower() and since - 30 <= created <= since + 1  # rejects a reused PID
    except Exception:
        return False
    finally:
        k.CloseHandle(h)


def _read_pid(pid_file: Path) -> int:
    try:
        return int(pid_file.read_text())
    except ValueError:
        return 0  # empty/corrupt file (crash mid-write); PID 0 never matches our image


def _log(line: str):
    with open(LOGS / "start.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def _take_lock(lock: Path | None = None, stale: float = LOCK_STALE_S) -> bool:
    lock = lock or LOCK
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - lock.stat().st_mtime < stale:
                return False  # another start is in progress
            # shortcut: two starts racing on one stale lock can both spawn; upgrade to a named mutex if double spawns show up in start.log
            lock.unlink(missing_ok=True)
            continue
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    return False


def _spawn(cmd: list[str], log_path: Path, flags: int = FLAGS) -> tuple[int, bool]:
    with open(log_path, "ab") as log:
        kw = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                  env=child_env(), close_fds=True)
        try:
            return subprocess.Popen(cmd, creationflags=flags, **kw).pid, True
        except OSError:
            return subprocess.Popen(cmd, creationflags=flags & ~BREAKAWAY, **kw).pid, False


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


def start_oyj() -> bool:
    """Lazy start of the ownyourjev ONNX server; independent of the CLM files and health."""
    lock = RUN / "oyj.lock"
    try:
        if _down_active() or not (OYJ_PY.exists() and OYJ_MODEL.exists()):
            return False
        RUN.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        pid_file = RUN / "oyj.pid"
        if pid_file.exists() and _alive(_read_pid(pid_file), "python.exe", pid_file.stat().st_mtime):
            return False  # running, still loading
        if not _take_lock(lock):
            return False
        pid, breakaway = _spawn(oyj_cmd(), LOGS / "oyj.log")
        pid_file.write_text(str(pid))
        _log(f"spawn oyj pid={pid} breakaway={breakaway}")
        return True  # lock stays; staleness or stop() clears it
    except Exception as e:
        try:
            lock.unlink(missing_ok=True)
            _log(f"error oyj {type(e).__name__}: {e}")
        except Exception:
            pass
        return False


def stop_oyj() -> list[int]:
    return _kill(("oyj", "python.exe"))


def note_capture(n_tasks: int) -> bool:
    """Count newly captured complexity tasks; start `oyj/update.py --auto` when the window is full. Never raises."""
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        try:
            st = json.loads(UPDATE_STATE.read_text())
        except (OSError, ValueError):
            st = {}
        now = time.time()
        st["pending"] = st.get("pending", 0) + n_tasks  # shortcut: parallel hooks can lose a count; the trigger is approximate, update.py counts real data
        due = (st["pending"] >= UPDATE_MIN_NEW and now - st.get("last_run", 0) >= UPDATE_MIN_DAYS * 86400
               and now - st.get("spawned", 0) >= UPDATE_STALE_S and not UPDATE_LOCK.exists())
        if due:
            pid, _ = _spawn([sys.executable, str(ROOT / "oyj" / "update.py"), "--auto"], LOGS / "oyj-update.log", FLAGS | IDLE)
            st["spawned"] = now
            _log(f"spawn oyj-update pid={pid} pending={st['pending']}")
        tmp = UPDATE_STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        os.replace(tmp, UPDATE_STATE)
        return due
    except Exception:
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


def _kill(*procs) -> list[int]:
    killed = []
    for name, image in procs:
        pid_file = RUN / f"{name}.pid"
        if not pid_file.exists():
            continue
        pid = _read_pid(pid_file)
        if _alive(pid, image, pid_file.stat().st_mtime):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            killed.append(pid)
        pid_file.unlink(missing_ok=True)
        (RUN / f"{name}.lock").unlink(missing_ok=True)
    return killed


def stop() -> list[int]:
    _write_down_flag()  # first, so a concurrent start() stops before its next spawn
    killed = _kill(("clm-serve", "python.exe"), ("llama-server", "llama-server.exe"), ("oyj", "python.exe"))
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
        start_oyj()
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
