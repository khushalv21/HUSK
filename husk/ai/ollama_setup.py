import json
import urllib.request
import urllib.error
from typing import Callable, Dict, List, Optional
from husk.ai.adapters import get_ssl_context

# Small, permissively-licensed defaults used for the zero-API-key "local" provider so
# `husk init` can offer a path that works without any account or paid key at all.
DEFAULT_CHAT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

LogFn = Optional[Callable[[str], None]]


def is_ollama_running(host: str) -> bool:
    """
    Returns True if an Ollama server responds at `host`.
    """
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5, context=get_ssl_context()) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_installed_models(host: str) -> List[str]:
    """
    Returns the list of model names already pulled on the given Ollama host.
    """
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5, context=get_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def _model_present(installed: List[str], model: str) -> bool:
    # Ollama tags are sometimes reported with a ":latest"/variant suffix, so also
    # match on the base model name before the colon.
    base = model.split(":")[0]
    return any(m == model or m.split(":")[0] == base for m in installed)


def pull_model(host: str, model: str, log_fn: LogFn = None) -> bool:
    """
    Streams `ollama pull <model>` progress via Ollama's HTTP API. Returns True on success.

    Ollama emits one JSON progress line per network chunk received — for a ~1GB layer
    that's thousands of lines, nearly all identical ("pulling <digest>") except for the
    completed/total byte counts. Logging every line as-is would flood the terminal, so
    progress is throttled to one line per 10% per layer; only genuinely distinct status
    changes (e.g. "verifying sha256 digest", "success") are logged as they happen.
    """
    url = f"{host}/api/pull"
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    last_logged_bucket: Dict[str, int] = {}
    last_plain_status: Optional[str] = None
    try:
        with urllib.request.urlopen(req, timeout=600, context=get_ssl_context()) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    status = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if status.get("error"):
                    if log_fn:
                        log_fn(f"  [ERROR] {model}: {status['error']}")
                    return False

                status_text = status.get("status", "")
                total = status.get("total")
                completed = status.get("completed")

                if total and completed is not None:
                    pct = int(completed * 100 / total)
                    bucket = pct - (pct % 10)
                    key = status.get("digest") or status_text
                    if log_fn and last_logged_bucket.get(key) != bucket:
                        last_logged_bucket[key] = bucket
                        log_fn(f"  [{model}] {status_text}: {pct}%")
                elif status_text and status_text != last_plain_status:
                    last_plain_status = status_text
                    if log_fn:
                        log_fn(f"  [{model}] {status_text}")
        return True
    except Exception as e:
        if log_fn:
            log_fn(f"  [ERROR] Failed to pull {model}: {e}")
        return False


def ensure_models(host: str, models: List[str], log_fn: LogFn = None, announce_present: bool = True) -> bool:
    """
    Ensures every model in `models` is present on the given Ollama host, pulling any
    that are missing. Returns False if Ollama itself isn't reachable, or if any pull fails.

    `announce_present` controls whether already-installed models get a log line too;
    callers doing a routine pre-flight check on every command invocation (rather than an
    explicit `husk init`) usually want this off to avoid repeating the same line forever.
    """
    if not is_ollama_running(host):
        if log_fn:
            log_fn(
                f"Could not reach Ollama at {host}.\n"
                "  * Install it from https://ollama.com/download, make sure it's running, "
                "then re-run this command."
            )
        return False

    installed = list_installed_models(host)
    all_ok = True
    for model in models:
        if _model_present(installed, model):
            if log_fn and announce_present:
                log_fn(f"  * {model}: already installed.")
            continue
        if log_fn:
            log_fn(f"  * {model}: not found locally, pulling now (this may take a few minutes)...")
        if not pull_model(host, model, log_fn=log_fn):
            all_ok = False
    return all_ok
