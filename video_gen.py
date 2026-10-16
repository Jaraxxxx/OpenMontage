#!/usr/bin/env python
"""Simple curl-style video generation via Portkey -> Vertex/Gemini.

Usage:
  python video_gen.py --model veo-3.1 --prompt "..." --seconds 8 --out clip.mp4
  python video_gen.py --model omni-flash --prompt "..." --seconds 8 --out clip.mp4
  python video_gen.py --status <task_id>

Config (first match wins):
  1. --base-url / --key CLI args
  2. PORTKEY_BASE_URL / PORTKEY_API_KEY env vars
  3. Values below
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# ── defaults (override via CLI or env) ─────────────────────────────
DEFAULT_BASE_URL = "https://api.portkey.ai/v1"
# Never hardcode keys: set PORTKEY_API_KEY in .env (see .env.example).
DEFAULT_KEY = None

POLL_INTERVAL = 10          # seconds between status checks
MAX_WAIT = 900              # give up after this long
TERMINAL = {"completed", "succeeded", "failed", "cancelled", "canceled", "error"}


def http(method: str, url: str, key: str, body: dict | None = None, timeout: int = 120) -> tuple[int, dict | bytes, str]:
    """Minimal curl-equivalent. Returns (status, parsed_json_or_bytes, content_type)."""
    headers = {
        "x-portkey-api-key": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            raw = resp.read()
            return resp.status, (json.loads(raw) if "json" in ct else raw), ct
    except HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), "json"
        except Exception:
            return e.code, raw, "text"


def create_task(base: str, key: str, model: str, prompt: str, seconds: str | None, size: str | None) -> tuple[int, dict, str]:
    body: dict = {"model": model, "prompt": prompt}
    if seconds:
        body["seconds"] = str(seconds)
    if size:
        body["size"] = size
    return http("POST", f"{base}/videos", key, body)


def get_task(base: str, key: str, task_id: str) -> tuple[int, dict, str]:
    return http("GET", f"{base}/videos/{task_id}", key)


def download(base: str, key: str, task_id: str, out: Path) -> None:
    status, payload, _ = http("GET", f"{base}/videos/{task_id}/content", key, timeout=600)
    if status != 200 or isinstance(payload, dict):
        raise RuntimeError(f"download failed: {status} {str(payload)[:200]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    print(f"saved: {out} ({len(payload) / 1024:.0f} KB)")


def poll_until_done(base: str, key: str, task_id: str) -> dict:
    waited = 0
    while waited < MAX_WAIT:
        status, task, _ = get_task(base, key, task_id)
        if status != 200:
            raise RuntimeError(f"poll failed: {status} {str(task)[:300]}")
        state = str(task.get("status", "")).lower()
        print(f"  [{waited:4d}s] status={state}")
        if state in TERMINAL:
            return task
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    raise RuntimeError(f"timed out after {MAX_WAIT}s — rerun with --status {task_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Portkey video generation (curl-style)")
    ap.add_argument("--model", default="veo-3.1", help="veo-3.0 | veo-3.1 | omni-flash | @slug/model")
    ap.add_argument("--prompt", help="what to generate")
    ap.add_argument("--seconds", help="clip length e.g. 4, 8")
    ap.add_argument("--size", help="e.g. 1280x720 or 720x1280")
    ap.add_argument("--out", default="clip.mp4", help="output mp4 path")
    ap.add_argument("--status", metavar="TASK_ID", help="check/retry an existing task")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()

    import os
    base = (args.base_url or os.environ.get("PORTKEY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = args.key or os.environ.get("PORTKEY_API_KEY") or DEFAULT_KEY

    try:
        if args.status:
            task_id = args.status
            task = poll_until_done(base, key, task_id)
        else:
            if not args.prompt:
                ap.error("--prompt is required (or use --status TASK_ID)")
            print(f"creating task: model={args.model}")
            status, resp, _ = create_task(base, key, args.model, args.prompt, args.seconds, args.size)
            if status not in (200, 201):
                print(f"create failed: {status}\n{json.dumps(resp, indent=1)[:800] if isinstance(resp, dict) else resp[:800]}")
                sys.exit(1)
            task_id = resp.get("id")
            if not task_id:
                # Some shapes return the video inline (e.g. omni/interactions translation)
                print(json.dumps(resp, indent=1)[:2000])
                sys.exit(0)
            print(f"task_id: {task_id}")
            task = poll_until_done(base, key, task_id)

        state = str(task.get("status", "")).lower()
        if state not in ("completed", "succeeded"):
            print(f"task ended: {state}\n{json.dumps(task, indent=1)[:800]}")
            sys.exit(2)

        download(base, key, task_id, Path(args.out))
        print("DONE")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
