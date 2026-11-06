# JEV client: local CLM backend

These files are outside the repo (`~/.claude` is not under git). This note is their record.

## What changed

`C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py`:
- New module variable `_LOCALCLM = None` (the loaded `localclm` module, cached).
- New helper `_local_clm(cfg)`. It returns None unless `jev.backend` is `"local"`. Else it loads `<local_clm_home>/localclm.py` once. It returns None when the file is missing or the import fails.
- `call_jev()` loads the config first. With the local backend and a payload of at most 4000 characters:
  - Service up (`localclm.is_up()`, 0.3 s): POST to `http://127.0.0.1:8700/v1/systemone`, model `clm-latest`, no `Authorization` header, timeout `min(timeout, 1.5)`. Any error → log `local_clm_error` → TypeSafe.
  - Service down: `localclm.start(probe=False)` (detached, does not wait) → log `local_clm_starting` → TypeSafe.
- The TypeSafe code path is unchanged.

`C:/Users/molna/.claude/agent-hooks/jev-guard.json`, inside `"jev"`: new keys `"backend": "typesafe"` and `"local_clm_home": "C:/Users/molna/VSCodeRepos/local-clm"`.

## Backups

- `C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py.bak-20261106-localclm`
- `C:/Users/molna/.claude/agent-hooks/jev-guard.json.bak-20261106-localclm`

## Switch backends

1. Open `C:/Users/molna/.claude/agent-hooks/jev-guard.json`.
2. Set `"backend": "local"` to use the local service, or `"backend": "typesafe"` to use TypeSafe only.

The change takes effect on the next hook call. The default is `"typesafe"`: the spike measured 16/31 correct for the local model against 31/31 for TypeSafe (`logs/spike-q6.json`).

## Roll back

1. Copy `jev_client.py.bak-20261106-localclm` over `jev_client.py`.
2. Copy `jev-guard.json.bak-20261106-localclm` over `jev-guard.json`.

## Diff of `jev_client.py` against its backup

`diff
diff --git a/C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py.bak-20261106-localclm b/C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py
index 4fabf44..5198772 100644
--- a/C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py.bak-20261106-localclm
+++ b/C:/Users/molna/.claude/agent-hooks/jevlib/jev_client.py
@@ -29,6 +29,7 @@ from . import config
 # ---------------------------------------------------------------------------
 
 UNAVAILABLE = object()
+_LOCALCLM = None
 
 
 class JevError(Exception):
@@ -43,6 +44,27 @@ def _base_url() -> str:
     return os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
 
 
+def _local_clm(cfg):
+    """The local-clm launcher module when backend is "local", else None. Never raises."""
+    global _LOCALCLM
+    if cfg["jev"].get("backend") != "local":
+        return None
+    if _LOCALCLM is not None:
+        return _LOCALCLM
+    try:
+        import importlib.util
+        path = os.path.join(cfg["jev"]["local_clm_home"], "localclm.py")
+        if not os.path.isfile(path):
+            return None
+        spec = importlib.util.spec_from_file_location("localclm", path)
+        mod = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(mod)
+        _LOCALCLM = mod
+        return mod
+    except Exception:  # noqa: BLE001 — fail open to TypeSafe
+        return None
+
+
 # ---------------------------------------------------------------------------
 # Question builders
 # ---------------------------------------------------------------------------
@@ -94,14 +116,38 @@ def call_jev(state, questions, *, timeout=None) -> dict:
     Raises JevError for all error conditions.
     Returns the answers dict on success.
     """
+    cfg = config.load_config()
+    if timeout is None:
+        timeout = cfg["jev"]["timeout_seconds"]
+
+    lc = _local_clm(cfg)
+    # shortcut: char cap, llama-server rejects >2048 tokens; upgrade: tokenize-aware truncation
+    if lc is not None and len(json.dumps([state, questions], ensure_ascii=False)) <= 4000:
+        if lc.is_up():
+            try:
+                body = json.dumps({"state": state, "model": "clm-latest", "questions": questions}).encode("utf-8")
+                req = urllib.request.Request(lc.BASE_URL + "/v1/systemone", data=body,
+                                             headers={"Content-Type": "application/json"}, method="POST")
+                with urllib.request.urlopen(req, timeout=min(timeout, 1.5)) as resp:
+                    data = json.loads(resp.read())
+                answers = data.get("answers")
+                if not isinstance(answers, dict):
+                    raise JevError("malformed_response")
+                return answers
+            except Exception as exc:  # noqa: BLE001 — fail open to TypeSafe
+                log_event("local_clm_error", detail=type(exc).__name__)
+        else:
+            try:
+                if lc.start(probe=False):
+                    log_event("local_clm_starting")
+            except Exception:  # noqa: BLE001
+                pass
+
     key = os.environ.get("TYPESAFE_API_KEY", "").strip()
     if not key:
         raise JevError("no_api_key")
 
-    cfg = config.load_config()
     model = cfg["jev"]["model"]
-    if timeout is None:
-        timeout = cfg["jev"]["timeout_seconds"]
 
     payload = json.dumps({"state": state, "model": model, "questions": questions}).encode("utf-8")
     url = _base_url() + "/v1/systemone"
`
