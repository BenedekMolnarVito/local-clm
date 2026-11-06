# local-clm
Locally running Contrastive LM model for enabling agents with quick decisions 

## What it is
A local stand-in for TypeSafe JEV typed decisions: Qwen3-8B embeddings on the GPU, CLM heads on the CPU.
The JEV client uses it only when `jev-guard.json` sets `"backend": "local"`; every error or long payload falls back to TypeSafe.
Both processes start lazily on the first call and bind to `127.0.0.1` only.

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

## Setup
Run `pwsh setup.ps1`. It needs about 7 GB of disk.
Add `-WithReference` for the Q8 CPU reference model, or `-WithQ5` for the Q5_K_M model.

## Use
- `python localclm.py up --wait 180` starts both processes and waits until they answer.
- `python localclm.py status` prints `up` or `down`.
- `python localclm.py down` stops both processes.

Lazy start: set `"backend": "local"` in `C:/Users/molna/.claude/agent-hooks/jev-guard.json`. The first JEV call starts the service and goes to TypeSafe.
After `down`, lazy start is blocked until the next reboot or the next `up`.

WARNING: The local model is less accurate. bash-guard calls `safe_decide(..., local=False)`, so the destructive-command check always uses TypeSafe.

## Memory
VRAM about 7 GB, RAM about 1.5 GB. Stop it with `python localclm.py down` to free both.

## Not done
- No autostart, by design.
- No Docker: the AMD GPU cannot be reached from Docker on Windows.
- Accuracy is below TypeSafe: the spike gave 16/31 local against 31/31 TypeSafe (`logs/spike-q6.json`). Keep `"backend": "typesafe"` unless you accept that.
