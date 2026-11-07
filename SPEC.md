# local-clm — Implementation Specification

Status: APPROVED FOR IMPLEMENTATION after the independent review (commit 1 on `feat/local-clm-service`).
This file is authoritative. If code and this file disagree, this file wins. Only the controller (main session) can change this file.

## 0. Contents
1. Goal and end state
2. Facts this spec depends on (verified 2026-11-06)
3. Architecture
4. Repository layout
5. Coordination protocol (controller ↔ worker)
6. Batches and steps
7. Acceptance gates
8. Guardrails (global)
9. Git and review flow

---

## 1. Goal and end state

A local, TypeSafe-compatible decision service ("CLM") runs on this PC. Claude agents call it through the existing JEV client. Requirements:

1. The service does NOT start at Windows login. Nothing goes into Task Scheduler, the Startup folder, the Run registry keys, or Windows services.
2. The service starts in the background the first time an agent calls the JEV client after a reboot (lazy start).
3. The call that triggers the start does not wait for it. That call falls back to TypeSafe (the current behaviour).
4. When the service is up, JEV client calls go to `http://127.0.0.1:8700/v1/systemone`.
5. The owner stops the service by hand: `python C:/Users/molna/VSCodeRepos/local-clm/localclm.py down`. `down` only stops the service. The next JEV call starts it again in the background, the same as after a reboot.
5a. The local path never makes a hook exceed its budget: worst case added latency is ≤ 1.8 s (0.3 s probe + 1.5 s local POST).
6. No Docker. The encoder runs as a native Windows process on the GPU (Vulkan).

End state is proven by the smoke test in gate G5 (§7), with its output shown in chat.

## 2. Facts this spec depends on (verified 2026-11-06)

| Fact | Value |
|---|---|
| GPU | AMD Radeon RX 5700, 8 GB VRAM, RDNA1. No CUDA, no ROCm. Vulkan works. |
| CPU / RAM | Ryzen 7 3700X (AVX2), 32 GB |
| Python | 3.13.5 (`python`). `uv` 0.9.8 is installed. |
| llama.cpp pin | release `b11829` (2026-10-22), asset `llama-b11829-bin-win-vulkan-x64.zip`, sha256 `839ff55ce9cbee6a099b403b640ca6b694ee5d74b815e39f77fc5d2b20962f0d` |
| Encoder model | HF repo `Qwen/Qwen3-8B-GGUF`. `Qwen3-8B-Q6_K.gguf` 6.73 GB, sha256 `cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6`. `Qwen3-8B-Q5_K_M.gguf` 5.85 GB, sha256 `068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738`. `Qwen3-8B-Q8_0.gguf` 8.71 GB, sha256 `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6` (reference only). |
| CLM package | PyPI `contrastive-lm==0.1.0` (uploaded 2026-09-24). It declares `vllm` (does not install on Windows) and `pyarrow` (not needed) as dependencies → install with `--no-deps`. |
| CLM runtime deps | `numpy`, `requests`, `torch` (PyPI Windows wheels are CPU builds), `fastapi`, `uvicorn`. (`huggingface_hub` is not needed: setup downloads the head itself.) |
| CLM head | HF `Contrastive-LM/CLM-v0.1-8B` at commit `e939398d4556fcd9400c76fa8c5a513202f42b0a`, file `CLM_v0.1-8B.pt`, 75 557 149 bytes, sha256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`. Path: `%USERPROFILE%/.cache/clm/CLM_v0.1-8B.pt` (`clm.heads.download()` returns an existing file without a download). It is loaded with `torch.load(path, map_location="cpu")`. torch ≥ 2.6 defaults to `weights_only=True` (safe loading). |
| CLM server entry | `python -m clm.server` works (`server.py` ends with `if __name__ == "__main__": main()`). Use it, not the `clm-serve.exe` trampoline: the trampoline PID is not the server PID. |
| uvicorn logging | `clm-serve` runs uvicorn with `log_level="warning"` → no access log. A log file cannot prove that a request was served locally. |
| `clm-serve` flags | `--host` (default `0.0.0.0`, MUST pass `127.0.0.1`), `--port` (8700), `--emb-url`, `--emb-model` (`qwen3-8b`), `--max-tokens` (2048), `--ckpt`, `--device`, `--no-download`, `--no-ui` |
| `clm-serve` routes | `GET /health` → `{"ok": true, "embedder": <bool>, "models": [...], ...}`. `embedder` is true when `GET <emb-base>/v1/models` returns 200. `POST /v1/systemone` body `{state, model, questions}` → `{"model", "answers", "usage"}`. Unknown model → error. Model name: `clm-latest`. |
| Embedder request | CLM sends `POST /v1/embeddings` with `{"model": "qwen3-8b", "input": [..], "encoding_format": "base64", "truncate_prompt_tokens": 2048}`, then L2-normalises the result itself. It expects last-token pooling. |
| JEV client | `C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py`, function `call_jev(state, questions, *, timeout=None)`. It reads `TYPESAFE_BASE_URL` and `TYPESAFE_API_KEY`, and the config `jev.model` (`jev-latest`) and `jev.timeout_seconds` (5) from `C:/Users/molna/.claude/agent-hooks/jev-guard.json`. It returns `data["answers"]`. All 11 call sites (`agent_state_hooks.py` 5, `jevlib/decisions.py` 6) go through `call_jev` / `safe_decide`. Every real caller passes a **dict** state. `~/.claude` is NOT a git repository. |
| Hook budgets | `bash-guard` and `done-check` call `safe_decide(..., timeout=3)` and run under 20 s hook timeouts (`settings.json`). A killed `bash-guard` skips its `ask` → a destructive command runs. Callers threshold raw probabilities tuned on TypeSafe: `RISKY_MIN=0.6` (bash-guard), `DONE_MIN=0.3`, `0.7`, `0.6` (`agent_state_hooks.py`). |
| Windows connect | A connect to a closed `127.0.0.1` port does not fail fast on Windows. Each `is_up()` probe costs its full timeout while the service is down. |

## 3. Architecture

```
agent / hook ─► jev_client.call_jev()
                  │ backend == "local"?
                  ├─ payload > 4000 chars → TypeSafe
                  ├─ localclm.is_up()  (GET :8700/health, 0.3 s timeout)
                  │     up   → POST http://127.0.0.1:8700/v1/systemone (model "clm-latest", timeout ≤ 1.5 s)
                  │            error → TypeSafe
                  │     down → localclm.start(probe=False) (detached, returns < 1 s) → TypeSafe
                  └─ TypeSafe (unchanged code path)

localclm.start() spawns two detached processes:
  1. bin/llama/llama-server.exe               (Vulkan GPU, :8090, /v1/embeddings, last-token pooling)
  2. .venv/Scripts/python.exe -m clm.server   (CPU heads, :8700, /v1/systemone)
```

Both processes bind to `127.0.0.1` only. Logs go to `logs/`. PIDs go to `run/`.

## 4. Repository layout (final)

| Path | Tracked | Made in step |
|---|---|---|
| `SPEC.md` | yes | controller |
| `README.md` | yes | S5.1 |
| `.gitignore` | yes | S1.1 |
| `requirements.in` | yes | S1.2 |
| `requirements.txt` | yes (generated lock) | S1.2 |
| `setup.ps1` | yes | S1.3–S1.6 |
| `localclm.py` | yes | S2.1–S2.7 |
| `tests/test_localclm.py` | yes | S2.8 |
| `spike/cases.json` | yes | S3.1 |
| `spike/compare.py` | yes | S3.2 |
| `integration/jev_client_local.md` | yes | S4.1 |
| `tests/test_jev_local.py` | yes | S4.4 |
| `smoke.py` | yes | S5.2 |
| `bin/`, `models/`, `.venv/`, `logs/`, `run/`, `.agent/` | NO (ignored) | — |

No other files. A worker that wants another file reports `BLOCKED`.

## 5. Coordination protocol (controller ↔ worker)

### 5.1 Roles
- **Controller** = the main session (opus). It dispatches one batch at a time, verifies, commits, and writes pushback.
- **Worker** = one `implementer` subagent with `model: "haiku"`, one per batch. A worker does every step of its batch in order, in one session.
- Batches run in serial. Batches never run in parallel (they share `.gitignore`, `README.md` and the venv).

### 5.2 Worker rules
0. Run every Verify command exactly as written, with the **PowerShell tool** (pwsh 7). `$LASTEXITCODE` exists only in PowerShell. Do not run Verify commands in Git Bash.
1. Read this file first. Read only §2, §5, §8 and your batch in §6.
2. Do the steps of your batch in order. Do not start a step from another batch.
3. After each step, run its **Verify** command. Do not continue until it passes.
4. If a Verify fails 3 times, or a fact in §2 is wrong, stop. Report `BLOCKED` with the step ID, the command, and the last 20 lines of its output.
5. Do not commit. Do not push. Do not switch branches. The controller commits.
6. Do not edit `SPEC.md`. Do not edit files outside the batch's file list, except `.agent/reports/`.
7. Do not run `localclm.py down` or kill processes that you did not start in this batch.
8. Write the report to `.agent/reports/<batch-id>.md` (format §5.3). Reply with at most 10 lines: status + report path.

### 5.3 Report format (`.agent/reports/<batch-id>.md`)
```
# <batch-id> report — attempt <n>
Status: DONE | BLOCKED
## Steps
| Step | Status | Verify command | Result (exit code + key line) |
## Files changed
<path> — <one line>
## Deviations from SPEC
<none | each deviation + why>
## Open questions
<none | list>
```

### 5.4 Controller verification (per batch)
1. Read the report.
2. Run `git status --porcelain`. Every changed path must be in the batch file list.
3. Re-run every Verify command of the batch. Only the controller's run counts.
4. Read the diff against the step requirements.
5. All pass → commit (§9) and mark the batch `done` in `.agent/TASKS.md`.
6. Any fail → write pushback (§5.5). Re-dispatch the same worker (SendMessage) or a new worker with the pushback file.
7. After 3 failed attempts for one batch, the controller fixes the batch itself and records why in `.agent/PROGRESS.md`.

### 5.5 Pushback format (`.agent/pushback/<batch-id>-<n>.md`)
```
# Pushback <batch-id> attempt <n>
Verdict: REJECTED
## Failures
### F<k>: <step id> — <one-line title>
- Requirement: <quote the SPEC line>
- Observed: <command + exact output, or file:line + code>
- Expected: <exact expected output or behaviour>
- Fix: <concrete instruction: file, function, what to change>
## Keep as is
<steps that passed — do not touch them>
## Re-verify
<the exact commands the worker must run before reporting again>
```
A new worker that gets a pushback reads: this SPEC, then the pushback file, then the previous report.

## 6. Batches and steps

Every path is relative to the repository root `C:/Users/molna/VSCodeRepos/local-clm` unless it is absolute. Run commands from the repo root with the PowerShell tool (pwsh 7), see §5.2 rule 0.

---

### Batch B1 — setup (files: `.gitignore`, `requirements.in`, `requirements.txt`, `setup.ps1`)

**S1.1 `.gitignore`.** Write exactly these lines:
```
bin/
models/
.venv/
logs/
run/
.agent/
__pycache__/
*.pyc
```
Verify: `python -c "import subprocess as s;o=s.run(['git','check-ignore','bin/x','models/x','.venv/x','logs/x','run/x','.agent/x'],capture_output=True,text=True).stdout.split();assert len(o)==6,o;print('ok')"` → `ok`.

**S1.2 Python lock.** Write `requirements.in` with exactly these lines (one per line, no versions):
`numpy`, `requests`, `torch>=2.6`, `fastapi`, `uvicorn`.
Then run:
`uv pip compile requirements.in --python-version 3.13 --python-platform windows --exclude-newer 2026-10-23 --generate-hashes -o requirements.txt`
Verify: `python -c "import re;t=open('requirements.txt').read();m=re.search(r'^torch==2\.(\d+)',t,re.M);assert m and int(m.group(1))>=6;assert 'vllm' not in t and '--hash=sha256:' in t;print('ok')"` → `ok`.

**S1.3 `setup.ps1` skeleton.** Create `setup.ps1` with:
- First lines: `#Requires -Version 7`, `$ErrorActionPreference = 'Stop'`, `$ProgressPreference = 'SilentlyContinue'`, `Set-Location $PSScriptRoot`.
- A `param([switch]$WithReference, [switch]$WithQ5)` block (directly after `#Requires`, before the other lines, as PowerShell requires).
- A helper function `Get-Verified($Url, $OutFile, $Sha256)`. It does nothing when `$OutFile` exists and its SHA-256 matches. Otherwise it writes `download: $OutFile` to the host, creates the parent folder, downloads with `Invoke-WebRequest -Uri $Url -OutFile "$OutFile.part" -Resume` (a re-run continues a partial `.part` file), checks the SHA-256 (`Get-FileHash -Algorithm SHA256`, compare case-insensitive), then moves `.part` to `$OutFile` (`Move-Item -Force`). On a mismatch it deletes the `.part` file and throws `"sha256 mismatch: $OutFile"`.
- A final line `Write-Host 'setup: ok'`.
Verify: `pwsh -NoProfile -Command '[void][scriptblock]::Create((Get-Content -Raw setup.ps1)); "parse ok"'` → `parse ok`.

**S1.4 llama.cpp.** In `setup.ps1`, add a section that:
1. Calls `Get-Verified` with URL `https://github.com/ggml-org/llama.cpp/releases/download/b11829/llama-b11829-bin-win-vulkan-x64.zip`, OutFile `bin/llama-b11829-bin-win-vulkan-x64.zip`, and the sha256 from §2.
2. Extracts the zip into `bin/llama/` with `Expand-Archive -Force` when `bin/llama/llama-server.exe` does not exist.
3. If the exe is not at `bin/llama/llama-server.exe` after the extraction but is in one subfolder of `bin/llama/`, moves that subfolder's content up one level. Then throws when `bin/llama/llama-server.exe` is still missing.
Verify: `pwsh -NoProfile -File setup.ps1` exits 0, then `bin/llama/llama-server.exe --version` prints a line that contains `11829`.

**S1.5 Models.** In `setup.ps1`, add a section that calls `Get-Verified` for:
- always: `Qwen3-8B-Q6_K.gguf`
- when `-WithQ5`: `Qwen3-8B-Q5_K_M.gguf`
- when `-WithReference`: `Qwen3-8B-Q8_0.gguf`
URL form: `https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/<file>`. OutFile: `models/<file>`. Hashes: §2.
Verify: `pwsh -NoProfile -File setup.ps1 -WithReference` exits 0 (this downloads ~15.5 GB; expect 10–40 min; run it in the foreground with a 600 s timeout and re-run until it exits 0 — finished files are skipped), then `Get-ChildItem models/*.gguf | Select-Object Name,Length` lists Q6_K and Q8_0.

**S1.6 venv, CLM and head.** In `setup.ps1`, add a section that:
1. Runs `uv venv .venv --python 3.13` when `.venv/Scripts/python.exe` does not exist.
2. Runs `uv pip install --require-hashes -r requirements.txt --python .venv/Scripts/python.exe`.
3. Runs `uv pip install --no-deps contrastive-lm==0.1.0 --python .venv/Scripts/python.exe`.
4. Calls `Get-Verified` with URL `https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/e939398d4556fcd9400c76fa8c5a513202f42b0a/CLM_v0.1-8B.pt`, OutFile `"$env:USERPROFILE/.cache/clm/CLM_v0.1-8B.pt"`, sha256 from §2.
5. Checks `$LASTEXITCODE` after every native command (`uv`) and throws on non-zero.
Verify (all must pass):
- `pwsh -NoProfile -File setup.ps1` exits 0 and prints `setup: ok`. A second run also exits 0 and prints no `download:` line.
- `.venv/Scripts/python.exe -m clm.server --help` output contains `--emb-url`, `--device`, `--host`, `--no-download` and `--no-ui`.
- `.venv/Scripts/python.exe -c "import os,torch; p=os.path.expanduser('~/.cache/clm/CLM_v0.1-8B.pt'); torch.load(p,map_location='cpu'); print('head ok', torch.__version__)"` prints `head ok`. If this raises a `weights_only` error: report BLOCKED. Do NOT pass `weights_only=False`.

---

### Batch B2 — launcher (files: `localclm.py`, `tests/test_localclm.py`)

`localclm.py` uses only the Python standard library. It is imported by `jev_client.py` (Python 3.13, hook context, must not print to stdout on import) and also runs as a CLI. Derive every path below `RUN`/`LOGS` inside the functions (tests patch `RUN`, `LOGS`, `LOCK`).

**S2.1 Constants.** At module top:
```python
ROOT = Path(__file__).resolve().parent
LLAMA_EXE = ROOT / "bin" / "llama" / "llama-server.exe"
PY_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = ROOT / "models" / os.environ.get("LOCAL_CLM_GGUF", "Qwen3-8B-Q6_K.gguf")
CKPT = Path(os.environ.get("CLM_CKPT") or Path.home() / ".cache" / "clm" / "CLM_v0.1-8B.pt")
RUN = ROOT / "run"
LOGS = ROOT / "logs"
LOCK = RUN / "start.lock"
EMB_PORT = 8090
CLM_PORT = 8700
BASE_URL = f"http://127.0.0.1:{CLM_PORT}"
LOCK_STALE_S = 300
```
Verify: `python -c "import localclm as m; print(m.BASE_URL)"` → `http://127.0.0.1:8700`, and nothing else is printed.

**S2.2 `is_up(timeout: float = 0.3) -> bool`.** GET `BASE_URL + "/health"` with `urllib.request`. Returns True only when the HTTP status is 200 AND the JSON has `ok is True` AND `embedder is True`. Returns False on any exception. Never raises.
Verify: `python -c "import localclm as m; print(m.is_up())"` → `False` (service not running).

**S2.3 Commands.** Two functions return argument lists (no shell strings):
- `llama_cmd() -> list[str]`: `[str(LLAMA_EXE), "-m", str(MODEL), "--embeddings", "--pooling", "last", "-c", "2048", "-b", "2048", "-ub", "2048", "-np", "1", "-ngl", "99", "--host", "127.0.0.1", "--port", str(EMB_PORT), "--alias", "qwen3-8b"]`
- `clm_cmd() -> list[str]`: `[str(PY_EXE), "-m", "clm.server", "--host", "127.0.0.1", "--port", str(CLM_PORT), "--emb-url", f"http://127.0.0.1:{EMB_PORT}/v1/embeddings", "--emb-model", "qwen3-8b", "--max-tokens", "2048", "--device", "cpu", "--no-download", "--no-ui", "--ckpt", str(CKPT)]`
- `child_env() -> dict`: a copy of `os.environ` without the keys `TYPESAFE_API_KEY` and `CLM_API_KEY`.
Verify: `python -c "import localclm as m; c=m.llama_cmd(); assert '--pooling' in c and c[c.index('--host')+1]=='127.0.0.1'; d=m.clm_cmd(); assert d[d.index('--host')+1]=='127.0.0.1' and '--no-download' in d and d[1:3]==['-m','clm.server']; import os; os.environ['CLM_API_KEY']='x'; assert 'CLM_API_KEY' not in m.child_env(); print('ok')"` → `ok`.

**S2.4 Stop has no flag.** `stop()` only stops the processes. The launcher has no down flag, no boot id and no `winreg` read. Do not add one.
Verify: covered by S2.8 test 11.

**S2.5 `_created(pid: int) -> float | None` and `_alive(pid: int, image: str, since: float) -> bool`.**
- `_created`: return the process creation time as a Unix timestamp, or None. Code:
```python
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
```
- `_alive`: run `subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True)`. Return True only when `image.lower()` is in `stdout.lower()` AND `c = _created(pid)` is not None AND `since - 30 <= c <= since + 1`. `since` is the mtime of the pid file (it is written right after `Popen`). This rejects a reused PID (a foreign process created at another time). Return False on any exception. Image names: `llama-server.exe` for llama, `python.exe` for clm.
Verify: `python -c "import os,time,localclm as m; p=os.getpid(); c=m._created(p); assert c and abs(time.time()-c)<60; assert m._alive(p,'python.exe',c); assert not m._alive(p,'python.exe',c+3600); assert not m._alive(p,'llama-server.exe',c); print('ok')"` → `ok`.

**S2.6 `start(probe: bool = True) -> bool`.** It must return in less than 1 second. Behaviour, in order:
1. If `probe` and `is_up()` → return False.
2. Create `RUN` and `LOGS`. Take the lock: `os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` and write the current time. If the lock exists and its mtime is newer than `LOCK_STALE_S` seconds → return False (another start is in progress). If it is older → delete it and try once more.
3. If `LLAMA_EXE`, `PY_EXE`, `MODEL` or `CKPT` is missing → delete the lock, append one line to `LOGS/start.log` that names the missing path, and return False.
4. For each component in order — (`llama-server`, `llama_cmd()`, image `llama-server.exe`), then (`clm-serve`, `clm_cmd()`, image `python.exe`):
   - If `RUN/<name>.pid` exists and `_alive(<pid>, <image>, <pid file mtime>)` → skip this component (it already runs).
   - Else spawn it with `subprocess.Popen(cmd, stdin=DEVNULL, stdout=<log>, stderr=STDOUT, cwd=ROOT, env=child_env(), creationflags=FLAGS, close_fds=True)`, then close the log file handle in the parent. Log: `LOGS/<name>.log`, opened in append-binary mode. Write the PID to `RUN/<name>.pid` (plain integer text).
   - `FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB` (constants from `subprocess`; `CREATE_BREAKAWAY_FROM_JOB` = `0x01000000`). If Popen raises `OSError` with the breakaway flag, retry once without `CREATE_BREAKAWAY_FROM_JOB`.
   - Append `spawn <name> pid=<pid> breakaway=<True|False>` to `LOGS/start.log`.
5. Do NOT delete the lock. `wait_up()` or `stop()` deletes it; it also goes stale after `LOCK_STALE_S`.
6. Return True.
`start()` never raises. On an unexpected exception, it deletes the lock only if this call created it, appends the exception type and message to `LOGS/start.log`, and returns False.
Verify: covered by tests in S2.8.

**S2.7 `wait_up`, `stop` and CLI.**
- `wait_up(timeout: float = 180.0) -> bool`: poll `is_up()` every 1 second until it is True or the timeout passes. When True, delete `LOCK` if it exists. Return the result.
- `stop() -> list[int]`: for each pid file in `RUN` (`clm-serve.pid` first, then `llama-server.pid`): read the PID; if `_alive(pid, image, <pid file mtime>)`, run `subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)` and add the PID to the result; delete the pid file. Delete `LOCK`. Return the list of killed PIDs.
- CLI `python localclm.py <up|down|status> [--wait SECONDS]` with `argparse`:
  - `up`: call `start()`. With `--wait N` call `wait_up(N)`. Print `up` when `is_up()` is True at the end, else `starting` (no wait) or `failed` (wait timed out). Exit code 0 for `up` and `starting`, 1 for `failed`.
  - `down`: call `stop()`, print `stopped <pids>`, exit 0.
  - `status`: print `up` or `down`. Exit 0 for `up`, 3 for `down`.
Verify: `python localclm.py status; "exit=$LASTEXITCODE"` → `down` then `exit=3`.

**S2.8 Tests `tests/test_localclm.py`.** Use `unittest` and `unittest.mock` only. Put the repo root on `sys.path` so that `import localclm` works when run from the repo root. In `setUp`, patch `localclm.RUN`, `localclm.LOGS`, `localclm.LOCK` into a `tempfile.TemporaryDirectory()`. Also patch these defaults in `setUp` (a test overrides one when it needs another value; no test may reach a real process, port or sleep): `localclm.is_up` → False, `localclm._alive` → False, `localclm.subprocess.Popen` → a mock whose `.pid` is 111, then 222, then 333, `localclm.subprocess.run` → a mock, `localclm.time.sleep` → a no-op. Fixture "exe ok" = `LLAMA_EXE`, `PY_EXE`, `MODEL`, `CKPT` patched to existing empty temp files. Required tests (names are fixed):
1. `test_is_up_false_when_unreachable` — patch `urllib.request.urlopen` to raise `URLError` → False.
2. `test_is_up_requires_embedder` — fake 200 response with `{"ok": true, "embedder": false}` → False; with `embedder: true` → True.
3. `test_start_noop_when_up` — `is_up` patched True → `start()` is False and `Popen` is not called.
4. `test_start_spawns_two_and_writes_pids` — "exe ok", `is_up` False, `_alive` False, `Popen` mocked with `.pid` 111 then 222 → True, `Popen` called twice (llama first, its argv[0] is `LLAMA_EXE`), `env` has no `TYPESAFE_API_KEY`, pid files contain `111` and `222`, the lock exists.
5. `test_start_respects_fresh_lock` — a fresh lock file exists → False, `Popen` not called.
6. `test_start_replaces_stale_lock` — "exe ok", a lock file with mtime set 400 s in the past (`os.utime`) → True.
7. `test_start_missing_binary` — `LLAMA_EXE` points to a missing path → False, no `Popen`, `start.log` names the path, no lock left.
8. `test_start_retries_without_breakaway` — "exe ok", the first `Popen` call raises `OSError`, later calls succeed → True; the retry's `creationflags & 0x01000000 == 0`; `start.log` contains `breakaway=False`.
9. `test_wait_up_clears_lock` — `is_up` returns False, then True; `time.sleep` patched → `wait_up(5)` is True and the lock is deleted.
10. `test_stop_skips_dead_or_foreign_pid` — pid files 111 (clm) and 222 (llama); `_alive` returns True for 222 only; `subprocess.run` mocked → `stop()` returns `[222]`, `taskkill` called once with `/T` and `/F`, both pid files and the lock deleted.
11. `test_start_after_down` — "exe ok", `is_up` False; `stop()` then `start()` → True, and `Popen` is called twice (llama, then clm). A `down` never blocks the next start.
12. `test_start_skips_live_component` — "exe ok", `RUN/llama-server.pid` contains `222`, `_alive` True for 222 → True, `Popen` called once (clm only).
Verify: `python -m unittest discover -s tests -v` → all 12 tests pass, `OK`.

---

### Batch B3 — accuracy spike (files: `spike/cases.json`, `spike/compare.py`)

This batch needs the GPU, ~9 GB of free RAM, and network access to TypeSafe. If Q6_K does not fit in VRAM, use the Q5 fallback in S3.3 step 6. Report BLOCKED only if Q5_K_M also does not fit.

**S3.1 `spike/cases.json`.** Write a JSON list of exactly 31 cases. Each case is `{"id": "c01".."c31", "state": <dict>, "questions": {<qid>: <question>}, "expect": <value>}` with 1 question per case. `state` is a dict, as the real callers send (for example `{"task": "..."}`, `{"command": "...", "cwd": "..."}`). Question wire format (same as TypeSafe):
- 10 cases `{"type": "noul", "instructions": "<yes/no statement>"}`, `expect` = `true` or `false`.
- 12 cases `{"type": "choice", "instructions": "...", "criteria": {"<opt>": "<description>", ...}}` with 2–5 options, `expect` = the correct option key.
- 8 cases `{"type": "score", "instructions": "...", "criteria": ["<level0>", "<level1>", ...]}` with 3–5 levels, `expect` = the correct level index (int).
- 1 case `c31`: a choice with exactly 20 options (a skill roster: 20 short skill names with one-line descriptions); `expect` = the correct key.
- Five of the 10 noul cases are bash-guard cases with ids `bg1`..`bg5` as their question id. They copy the question exactly from `agent_state_hooks.py` (find the `bash-guard` question text with Grep in `C:/Users/molna/.claude/agent-hooks/agent_state_hooks.py`, and the state shape it sends). Three are risky (`expect: true`): `git reset --hard` with a dirty tree, `rm -rf src`, `git push --force origin main`. Two are safe (`expect: false`): `rm -rf node_modules`, `rm -rf build`.
Topics for the others: software engineering agent work (routing a task to a team, picking a tool, judging if a test failure is flaky, is a diff risky, which file to edit, is a claim supported by a log). Each case must be unambiguous to a human. 3 states have a total JSON length of 1500–3000 characters.
Verify: `python -c "import json,collections as c;d=json.load(open('spike/cases.json'));assert len(d)==31;t=c.Counter(list(x['questions'].values())[0]['type'] for x in d);assert t=={'noul':10,'choice':13,'score':8},t;assert all(isinstance(x['state'],dict) and 'expect' in x for x in d);assert sum(1500<=len(json.dumps(x['state']))<=3000 for x in d)>=3;q=[k for x in d for k in x['questions']];assert all(f'bg{i}' in q for i in range(1,6));assert len(d[30]['questions'][next(iter(d[30]['questions']))]['criteria'])==20;print('ok')"` → `ok`.

**S3.2 `spike/compare.py`.** Run it with `.venv/Scripts/python.exe spike/compare.py --a-url URL --b-url URL [--typesafe] [--out PATH]`.
1. Builds two engines: `clm.Engine(clm.Embedder(url, "qwen3-8b", max_tokens=2048), checkpoint=<CKPT path as in S2.1>, device="cpu")` for A (reference) and B (candidate). If `clm.Engine`/`clm.Embedder` do not exist with this signature in 0.1.0, read `.venv/Lib/site-packages/clm/engine.py` and `__init__.py` and adapt the construction only; record the deviation in the report.
2. For each case: `answers = engine.answer(case["state"], case["questions"])` on A and B. Measure the wall time per call.
3. With `--typesafe`: also call `jevlib.jev_client.call_jev(case["state"], case["questions"])` (import from `C:/Users/molna/.claude/agent-hooks`; `jev-guard.json` backend must still be `typesafe`). Call `jevlib.envload.ensure()` once before the first call (it loads `TYPESAFE_API_KEY` as the hooks do). On error, record `null` for that case and print the exception type to stderr.
4. Embedding check: `a, _ = embA.embed([text]); b, _ = embB.embed([text])` with `text = json.dumps(case["state"])`; vectors are already L2-normalised → `cos = float(a[0] @ b[0])`.
5. Per case, for each source (a, b, typesafe):
   - `top`: noul → `p >= 0.5` (bool, p = answer `noul`); choice → `choice`; score → `round(score)`. Print the first answer dict of each source once to stderr so that the field names are visible.
   - `correct = top == case["expect"]`.
   Also per case: `agree = top_a == top_b`; `pdiff`: noul → `|p_a - p_b|`; choice → max abs difference over `probabilities`; score → `|s_a - s_b| / (levels - 1)`.
6. Prints a JSON summary to stdout: `{"n", "agree", "pdiff_mean", "pdiff_max", "cos_min", "cos_mean", "acc_a", "acc_b", "acc_typesafe", "typesafe_null", "bg_risky_p_b": [p for bg1-3], "bg_safe_p_b": [p for bg4-5], "bg_risky_p_typesafe", "bg_safe_p_typesafe", "lat_b_p50_ms", "lat_b_p90_ms", "lat_b_c31_ms", "per_case": [...]}` (`acc_*` are counts of correct cases; `typesafe_null` is the count of cases with a `null` TypeSafe answer; both are null without `--typesafe`). With `--out`, it also writes the summary to that file.
Verify: `.venv/Scripts/python.exe spike/compare.py --help` exits 0. The real run is S3.3.

**S3.3 Run the spike.** Steps:
1. Start the reference encoder on the CPU: `$ref = Start-Process -PassThru -WindowStyle Hidden -FilePath bin/llama/llama-server.exe -ArgumentList '-m','models/Qwen3-8B-Q8_0.gguf','--embeddings','--pooling','last','-c','2048','-b','2048','-ub','2048','-np','1','-ngl','0','--host','127.0.0.1','--port','8091','--alias','qwen3-8b' -RedirectStandardOutput logs/ref.out.log -RedirectStandardError logs/ref.log`. Record `$ref.Id`.
2. Start the candidate encoder on the GPU the same way with `models/Qwen3-8B-Q6_K.gguf`, `-ngl 99`, port `8090`, logs `logs/cand.out.log` / `logs/cand.log`. Record its Id.
3. Wait until `GET http://127.0.0.1:8090/v1/models` and `:8091/v1/models` return 200 (max 300 s).
4. POST one 3000-word input to `:8090/v1/embeddings` and record the status code and the first 200 characters of the response (over-length behaviour).
5. Run `.venv/Scripts/python.exe spike/compare.py --a-url http://127.0.0.1:8091/v1/embeddings --b-url http://127.0.0.1:8090/v1/embeddings --typesafe --out logs/spike-q6.json`.
6. If Q6_K did not load fully on the GPU (out-of-memory in `logs/cand.log`, or the "offloaded N/M layers" line shows N < M): stop the candidate, run `pwsh -NoProfile -File setup.ps1 -WithQ5`, and repeat steps 2–5 with `Qwen3-8B-Q5_K_M.gguf` and `--out logs/spike-q5.json`.
7. Stop both servers: `Stop-Process -Id <recorded ids>` (only the PIDs you started).
8. Put in the report: the summary JSON(s) without `per_case`, the "offloaded N/M layers" line from `logs/cand.log`, the result of step 4, and the list of cases where `correct` differs between b and typesafe.
Verify: `logs/spike-q6.json` (or `-q5`) exists, its `n` is 31, `acc_typesafe` is not null, and `typesafe_null <= 2`.

---

### Batch B4 — JEV client integration (files: `integration/jev_client_local.md`, `tests/test_jev_local.py`, and OUTSIDE the repo: `C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py`, `C:/Users/molna/.claude/agent-hooks/jev-guard.json`)

WARNING: `~/.claude` is not under git and the hooks run in every session. Back up before any edit. A syntax error here breaks every hook.

**S4.1 Backups.** Copy `jev_client.py` to `jev_client.py.bak-20261106-localclm` and `jev-guard.json` to `jev-guard.json.bak-20261106-localclm`, in the same folders.
Verify: `python -c "import hashlib as h;b=r'C:/Users/molna/.claude/agent-hooks/';f=lambda p:h.sha256(open(p,'rb').read()).hexdigest();assert f(b+'jevlib/jev_client.py')==f(b+'jevlib/jev_client.py.bak-20261106-localclm');assert f(b+'jev-guard.json')==f(b+'jev-guard.json.bak-20261106-localclm');print('ok')"` → `ok`.

**S4.2 Config.** In `jev-guard.json`, add two keys inside the existing `"jev"` object: `"backend": "typesafe"` and `"local_clm_home": "C:/Users/molna/VSCodeRepos/local-clm"`. Change nothing else. (The controller switches `backend` to `"local"` in gate G5.)
Verify: `python -c "import json;j=json.load(open(r'C:/Users/molna/.claude/agent-hooks/jev-guard.json'))['jev'];assert j['backend']=='typesafe' and j['model']=='jev-latest';print('ok')"` → `ok`.

**S4.3 `call_jev` local branch.** In `jev_client.py`:
1. Add a module-level variable `_LOCALCLM = None` and a helper `_local_clm(cfg)`. Order: if `cfg["jev"].get("backend") != "local"` → return None. If `_LOCALCLM` is not None → return it. Else load `<local_clm_home>/localclm.py` with `importlib.util.spec_from_file_location("localclm", path)` + `module_from_spec` + `spec.loader.exec_module`, store it in `_LOCALCLM`, return it. Return None when the file is missing or on any import exception.
2. At the start of `call_jev`, before the `TYPESAFE_API_KEY` check (move the existing `cfg = config.load_config()` and timeout lines above it; do not load the config twice), add:
   - `lc = _local_clm(cfg)`.
   - If `lc` is not None and `len(json.dumps([state, questions], ensure_ascii=False)) <= 4000` (comment: `# shortcut: char cap, llama-server rejects >2048 tokens; upgrade: tokenize-aware truncation`):
     - If `lc.is_up()`: POST `{"state": state, "model": "clm-latest", "questions": questions}` to `lc.BASE_URL + "/v1/systemone"` with header `Content-Type: application/json` only (no Authorization header) and timeout `min(timeout, 1.5)`. On success, parse the JSON and return `data["answers"]` after the same `malformed_response` check as the TypeSafe path. On any exception, call `log_event("local_clm_error", detail=type(exc).__name__)` and continue to the TypeSafe path.
     - Else: call `lc.start(probe=False)` inside `try/except Exception: pass`; call `log_event("local_clm_starting")` only when it returned True; continue to the TypeSafe path.
3. Do not change any other function. Do not change the TypeSafe code path. With `backend: "typesafe"` the behaviour must be identical to the backup.
Verify (all must pass):
- `python -m py_compile C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py; "exit=$LASTEXITCODE"` → `exit=0` (the hook below swallows a SyntaxError, so this is the real syntax check).
- `python -c "import sys;sys.path.insert(0,r'C:/Users/molna/.claude/agent-hooks');import jevlib.jev_client as j;assert j._LOCALCLM is None;print('import ok')"` → `import ok`.
- `'{"tool_input":{"command":"rm -rf build"},"cwd":"."}' | python C:/Users/molna/.claude/agent-hooks/agent_state_hooks.py bash-guard; "exit=$LASTEXITCODE"` → `exit=0`, and no `Traceback` in the output.

**S4.4 Tests `tests/test_jev_local.py`.** Import `jevlib.jev_client` from `C:/Users/molna/.claude/agent-hooks` (insert that path in `sys.path`). In every test:
- `mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"})` and remove `TYPESAFE_BASE_URL`.
- Patch `jev_client.log_event` (no writes to the real log).
- Patch `jev_client.config.load_config` to return `{"jev": {"backend": ..., "local_clm_home": <repo root>, "model": "jev-latest", "timeout_seconds": 3}}`.
- Patch `urllib.request.urlopen` as used by `jev_client` (never a real network call). A fake response is a context manager whose `.read()` returns JSON bytes.
- Set `jev_client._LOCALCLM` directly: `None` for tests 1 and 6, and `mock.Mock(BASE_URL="http://127.0.0.1:8700", is_up=..., start=...)` for tests 2–5. Reset it to `None` in `tearDown`.
Required tests (names are fixed):
1. `test_backend_typesafe_never_touches_local` — backend `typesafe` → `_LOCALCLM` stays None, and the only URL called is the TypeSafe URL.
2. `test_local_up_routes_local` — backend `local`, `is_up` True → request URL is `http://127.0.0.1:8700/v1/systemone`, body `model` is `clm-latest`, no `Authorization` header, returns `answers`.
3. `test_local_down_starts_and_falls_back` — `is_up` False, `start` returns True → `start` called once with `probe=False`, then the TypeSafe URL is called.
4. `test_local_error_falls_back` — `is_up` True, the local `urlopen` call raises `URLError`, the second call returns a TypeSafe answer → the TypeSafe answer is returned.
5. `test_local_skips_long_payload` — a dict state with a 10 000-char value → `is_up` and the local URL are not called; the TypeSafe URL is called.
6. `test_missing_home_falls_back` — backend `local`, `local_clm_home` points to a missing folder → the TypeSafe path is used, no exception.
7. `test_local_timeout_bounded` — `is_up` True → the local `urlopen` call gets `timeout <= 1.5`.
Verify: `python -m unittest discover -s tests -v` → all 19 tests (12 + 7) pass, `OK`.

**S4.5 `integration/jev_client_local.md`.** Write: what changed in `jev_client.py` and `jev-guard.json` (function names, keys), the backup file names, how to switch backends (`"backend": "local"` / `"typesafe"`), and how to roll back (copy the `.bak-20261106-localclm` files back). Include the full unified diff of `jev_client.py` against its backup (`git diff --no-index <bak> <file>` output, pasted in a ```diff block).
Verify: `python -c "t=open('integration/jev_client_local.md',encoding='utf-8').read();assert '```diff' in t and '_local_clm' in t;print('ok')"` → `ok`.

---

### Batch B5 — docs and smoke script (files: `README.md`, `smoke.py`)

**S5.1 `README.md`.** Keep the existing two lines. Add these sections, in order: `## What it is` (3 lines + the §3 diagram), `## Setup` (`pwsh setup.ps1`, disk ~7 GB, `-WithReference`/`-WithQ5`), `## Use` (`python localclm.py up --wait 180 | status | down`; lazy start through `jev-guard.json` `backend: local`; `down` stops the service; the next JEV call starts it again), `## Memory` (VRAM ~7 GB, RAM ~1.5 GB; stop with `down`), `## Not done` (no autostart, by design; no Docker, the AMD GPU cannot be reached from Docker on Windows).
Verify: `python -c "t=open('README.md',encoding='utf-8').read();assert all(h in t for h in ['## What it is','## Setup','## Use','## Memory','## Not done']);print('ok')"` → `ok`.

**S5.2 `smoke.py`.** Standard library only. Run it with `python smoke.py`. It proves the lazy start end to end through the real JEV client:
1. Import `localclm` (repo root) and `jevlib.jev_client` (from `C:/Users/molna/.claude/agent-hooks`).
2. Check that `jev-guard.json` `jev.backend == "local"`. If not: print `FAIL backend not local` and exit 2.
3. Call `localclm.stop()`. Wait until `localclm.is_up()` is False (max 15 s). Print `cold: down`.
4. Call `jev_client.safe_decide(STATE, QUESTIONS)` once (STATE a small dict, QUESTIONS one fixed noul question). Print `call1: <answers or UNAVAILABLE> (<ms> ms)`. If `<ms>` > 4500 → print `FAIL call1 too slow` and exit 1. Then check that `localclm.LOCK` exists or `RUN/llama-server.pid` exists → print `boot: triggered`, else print `FAIL boot not triggered` and exit 1.
5. Call `localclm.wait_up(240)`. If False: print `FAIL not up after 240 s`, print the last 20 lines of `logs/llama-server.log` and `logs/clm-serve.log`, exit 1. Else print `up after <s> s`.
6. Make TypeSafe unreachable for this process: `os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:9"`. Any answer now must come from the local service.
7. Call `jev_client.call_jev(STATE, QUESTIONS)` 5 times and time each call. Each call must return a dict that has the question id; otherwise (or when it raises, e.g. `JevError` because the local call failed and TypeSafe is unreachable) print `FAIL call<n> <exception type or answer>` and exit 1. Print `call<n>: <answer> (<ms> ms)`.
8. Print `SMOKE PASS p50=<ms> ms`, and exit 0.
Do not call `stop()` at the end (the service stays up, as in real use).
Verify: `python -c "import ast;ast.parse(open('smoke.py').read());print('ok')"` → `ok`. The real run is gate G5 (controller).

## 7. Acceptance gates (controller runs these; the output is shown in chat)

| Gate | After | Pass condition |
|---|---|---|
| G0 | SPEC | The independent reviewer verdict is APPROVE, or every Critical/Important finding is fixed in SPEC.md. Then commit 1 = SPEC.md only. |
| G1 | B1 | All S1.x Verify commands pass on the controller's run. |
| G2 | B2 | `python -m unittest discover -s tests -v` → 12 tests OK. |
| G3 | B3 | For the chosen quant: `agree >= 29` of 31, `pdiff_mean <= 0.05`, `cos_mean >= 0.99`, `acc_b >= 26` of 31 and `typesafe_null <= 2`, `acc_b >= acc_typesafe - 3`, every `bg_risky_p_b >= 0.6`, `lat_b_p50_ms <= 1000`, `lat_b_p90_ms <= 1200` (the local POST timeout is 1.5 s). If Q6_K and Q5_K_M both fail: STOP and ask the owner (do not continue to B4). If only the bash-guard condition fails, ask the owner whether bash-guard may stay on TypeSafe. The chosen quant becomes the default `MODEL` in `localclm.py` (controller edit, only if it is not Q6_K). |
| G4 | B4 | 19 tests OK. With `backend: typesafe`, one real `safe_decide` call returns answers (not UNAVAILABLE), and the S4.3 bash-guard Verify passes. |
| G5 | B5 | (a) Controller sets `backend: local` and runs `python smoke.py` → `SMOKE PASS`. (b) Real hook path: `python localclm.py down`, then from a NEW pwsh process run the bash-guard hook command from S4.3 Verify; within 240 s, `python localclm.py status` from another new process → `up`; `logs/start.log` shows the `breakaway=` value (if `False`, report it). (c) `python localclm.py down` → `status` gives `down`; a bash-guard hook call then starts it again: `status` gives `up` within 240 s (no flag blocks the start). (d) No autostart: `Get-ScheduledTask \| ? { $_.Actions.Execute -match 'llama\|clm' }`, `Get-Service \| ? { $_.Name -match 'llama\|clm' }`, `HKCU:\Software\Microsoft\Windows\CurrentVersion\Run`, `HKLM:\Software\Microsoft\Windows\CurrentVersion\Run`, and `$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup` name neither `llama` nor `clm`. |
| G6 | PR | Adversarial review of the branch → findings fixed. PR opened. Ponytail review of the PR → findings fixed, G2/G4 tests and the G5 smoke test re-run and pass. Squash merge into `develop`. |

## 8. Guardrails (global)

1. Bind every server to `127.0.0.1`. Never to `0.0.0.0`.
2. Never send `TYPESAFE_API_KEY` to the local server, and never pass it into the child processes' environment.
3. Never use `weights_only=False`, `--trust-remote-code`, or unpinned downloads. Every download has a pinned version and a sha256: llama.cpp zip, GGUF files, the head (pinned HF commit + sha256), and Python packages (`--generate-hashes` + `--require-hashes`; `contrastive-lm==0.1.0` is pinned by version).
4. No autostart of any kind. No Windows service, task, Run key or Startup shortcut.
5. No Docker.
6. Do not change TypeSafe behaviour when `backend` is `"typesafe"`.
7. Fail open: any local error falls back to TypeSafe. A hook never waits for the boot. The local path adds at most 1.8 s to a call.
8. Do not read or `grep` inside `.venv/`, `bin/` or `models/`, except the two CLM files named in S3.2 step 1.
9. Do not add dependencies beyond `requirements.in`.

## 9. Git and review flow

1. Branches: `main` → `develop` (pushed) → `feat/local-clm-service`.
2. Commit 1 = `SPEC.md` only, after G0.
3. One commit per batch after its gate passes: `feat(setup): ...`, `feat(launcher): ...`, `feat(spike): ...`, `feat(jev): ...`, `docs: ...`. The controller writes the commit messages.
4. Files outside the repo (S4) are not committed. Their diff lives in `integration/jev_client_local.md`.
5. G6: an independent adversarial `reviewer` over the full branch diff and this SPEC → fix wave → push → PR to `develop` → `ponytail:ponytail-review` of the PR → fix wave → re-run G2/G4/G5 → squash merge into `develop`.
