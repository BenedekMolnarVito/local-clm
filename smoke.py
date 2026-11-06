"""End-to-end smoke test: lazy start of the local CLM service through the real JEV client."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOOKS = "C:/Users/molna/.claude/agent-hooks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, HOOKS)

import localclm  # noqa: E402
from jevlib import envload, jev_client  # noqa: E402

STATE = {"task": "say hi to the user"}
QUESTIONS = {"q": jev_client.noul("The task is about greeting.")}


def fail(msg, code=1):
    print(f"FAIL {msg}")
    sys.exit(code)


def tail(path, n=20):
    try:
        print(f"--- {path.name}")
        print("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]))
    except OSError as e:
        print(f"--- {path.name}: {e}")


def main():
    envload.ensure()
    cfg = json.loads(Path(HOOKS, "jev-guard.json").read_text(encoding="utf-8"))
    if cfg["jev"].get("backend") != "local":
        fail("backend not local", 2)

    localclm.stop()
    localclm.DOWN_FLAG.unlink(missing_ok=True)  # a cold start after `down` needs it gone
    deadline = time.monotonic() + 15
    while localclm.is_up() and time.monotonic() < deadline:
        time.sleep(0.5)
    if localclm.is_up():
        fail("still up after stop")
    print("cold: down")

    t = time.perf_counter()
    ans = jev_client.safe_decide(STATE, QUESTIONS)
    ms = (time.perf_counter() - t) * 1000
    print(f"call1: {ans} ({ms:.0f} ms)")
    if ms > 4500:
        fail("call1 too slow")
    if not (localclm.LOCK.exists() or (localclm.RUN / "llama-server.pid").exists()):
        fail("boot not triggered")
    print("boot: triggered")

    t = time.monotonic()
    if not localclm.wait_up(240):
        print("FAIL not up after 240 s")
        tail(localclm.LOGS / "llama-server.log")
        tail(localclm.LOGS / "clm-serve.log")
        sys.exit(1)
    print(f"up after {time.monotonic() - t:.0f} s")

    os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:9"  # any answer now must come from the local service
    times = []
    for n in range(2, 7):
        t = time.perf_counter()
        try:
            ans = jev_client.call_jev(STATE, QUESTIONS)
        except Exception as e:
            fail(f"call{n} {type(e).__name__}")
        ms = (time.perf_counter() - t) * 1000
        if not isinstance(ans, dict) or "q" not in ans:
            fail(f"call{n} {ans}")
        times.append(ms)
        print(f"call{n}: {ans} ({ms:.0f} ms)")

    print(f"SMOKE PASS p50={statistics.median(times):.0f} ms")


if __name__ == "__main__":
    main()
