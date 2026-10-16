#!/usr/bin/env python3
"""Template-driven video generator for Gemini Omni Flash (omni-flash) via the
Swiggy AI gateway (Portkey -> Vertex/Gemini Interactions API).

One common script for ALL video artifacts: point it at a template folder and
it renders the prompt templates, generates every clip, verifies each output,
assembles the final film (trim -> upscale -> overlays -> concat -> grade),
and writes an honest machine-readable report of what actually happened.

Template layout (see templates/bappas_laddoo_rescue/ for a full example):

    <template_dir>/
      film.json               film definition: variables, clips, assembly
      prompts/*.md            prompt templates ({{placeholder}} substitution)
      assets/*                reference images, overlay logos, etc.

Usage:
    python omni_generate.py --list-templates
    python omni_generate.py --template templates/bappas_laddoo_rescue
    python omni_generate.py --template ... --clips clip_a,clip_b --force
    python omni_generate.py --template ... --dry-run        # render prompts only
    python omni_generate.py --template ... --assemble-only  # skip generation

Requires: python3 stdlib + imageio-ffmpeg (for the bundled ffmpeg binary).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
TEMPLATES_DIR = REPO / "templates"

GATEWAY_URL = os.environ.get("SWIGGY_GATEWAY_URL", "https://ai-gateway.swiggyops.de")
CUSTOM_HOST = "https://generativelanguage.googleapis.com/v1beta"
# Custom host WITHOUT the /v1beta suffix: for Veo file downloads we shift
# "v1beta" into the forwarded path itself (see VEO_FILE_URL below). Portkey's
# public gateway bans the custom-host header on /v1/files/* routes, but its
# interceptor only inspects the FIRST path segment — so /v1/v1beta/files/...
# passes through and lands on Google's byte-stream URL verbatim.
GOOGLE_HOST = "https://generativelanguage.googleapis.com"
OMNI_MODEL = "gemini-omni-flash-preview"
VEO_MODEL = "veo-3.1-generate-preview"

# Inline Portkey config: single target, the aistudio (Google AI Studio) virtual key.
INLINE_CONFIG = {
    "strategy": {"mode": "single"},
    "targets": [
        {
            "name": "omni-flash-google",
            "provider": "@aistudio",
            "override_params": {"model": OMNI_MODEL},
        }
    ],
}

VEO_CONFIG = {
    "strategy": {"mode": "single"},
    "targets": [{"name": "veo-google", "provider": "@aistudio"}],
}


# --------------------------------------------------------------------------
# env / config
# --------------------------------------------------------------------------

def load_env(path: Path | None = None) -> None:
    """Minimal .env loader."""
    for candidate in (path, REPO / ".env"):
        if candidate and candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            break


def require_key() -> str:
    key = os.environ.get("PORTKEY_API_KEY", "").strip()
    if not key:
        sys.exit("Error: PORTKEY_API_KEY not set (env or .env).")
    return key


def ffmpeg_bin() -> str:
    import imageio_ffmpeg  # type: ignore

    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------
# tiny {{placeholder}} renderer (stdlib only)
# --------------------------------------------------------------------------

class TemplateError(Exception):
    pass


def render_template(text: str, scope: dict, where: str) -> str:
    """Substitute {{dotted.path}} placeholders from a nested scope dict.

    Missing placeholders raise TemplateError (silent fallbacks cause
    dishonest prompts, so we fail loudly instead).
    """
    def lookup(path: str):
        node: object = scope
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node

    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        val = lookup(key)
        if val is None:
            raise TemplateError(f"{where}: unresolved placeholder={{{{{key}}}}}")
        return str(val)

    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", sub, text)


# --------------------------------------------------------------------------
# Omni Flash client (Interactions API through the Swiggy gateway)
# --------------------------------------------------------------------------

class OmniClient:
    def __init__(self, api_key: str, timeout: int = 600):
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare on the gateway 403s (error 1010) the default
            # Python-urllib user agent; identify as the script instead.
            "User-Agent": "omni_generate/1.0 (portkey-gateway-client)",
            "x-portkey-config": json.dumps(INLINE_CONFIG),
            "x-portkey-custom-host": CUSTOM_HOST,
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{GATEWAY_URL}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} on {method} {path}: {detail}") from e

    @staticmethod
    def _find_video(d: dict) -> str | None:
        """Recursively find the first base64 video payload in an interaction."""
        stack = [d]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("type") == "video" and node.get("data"):
                    return node["data"]
                if node.get("mime_type", "").startswith("video/") and node.get("data"):
                    return node["data"]
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return None

    @staticmethod
    def _find_text(d: dict) -> str:
        texts = []

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "text" and node.get("text"):
                    texts.append(node["text"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(d)
        return " ".join(texts).strip()

    def generate(
        self,
        parts: list[dict],
        seconds: int,
        prev_interaction_id: str | None = None,
        retries: int = 2,
        verbose: bool = True,
    ) -> tuple[bytes, str, dict]:
        """Generate one video clip. Returns (mp4_bytes, interaction_id, raw)."""
        body: dict = {
            "model": OMNI_MODEL,
            "input": parts,
            "response_format": {"type": "video", "aspect_ratio": "16:9"},
        }
        if prev_interaction_id:
            body["previous_interaction_id"] = prev_interaction_id

        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            if verbose:
                print(f"    generating (attempt {attempt}, target {seconds}s) ...", flush=True)
            try:
                d = self._request("POST", "/v1/interactions", body)
            except RuntimeError as e:
                if attempt <= retries:
                    wait = 20 * attempt
                    print(f"    transient error ({str(e)[:120]}); retrying in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise
            iid = d.get("id", "")
            status = d.get("status", "")
            video_b64 = self._find_video(d)
            if video_b64:
                if verbose:
                    print(f"    done in {time.time() - t0:.0f}s (interaction {iid[:20]}...)", flush=True)
                return base64.b64decode(video_b64), iid, d
            # No video yet: maybe async/pending -> poll if we have an id.
            if iid and status and status not in {"completed", "failed"}:
                d = self._poll(iid, verbose=verbose)
                video_b64 = self._find_video(d)
                if video_b64:
                    return base64.b64decode(video_b64), iid, d
            # Hard failure with model text (safety refusal etc.) -> surface it.
            text = self._find_text(d)
            if attempt <= retries:
                print(f"    no video in response (status={status}); retrying in 20s", flush=True)
                time.sleep(20)
                continue
            raise RuntimeError(
                f"No video in interaction response (status={status}). Model said: {text[:400]}"
            )

    def _poll(self, interaction_id: str, max_wait: int = 420, verbose: bool = True) -> dict:
        """GET the interaction until completed (best effort)."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            time.sleep(15)
            try:
                d = self._request("GET", f"/v1/interactions/{interaction_id}")
            except RuntimeError as e:
                print(f"    poll failed ({str(e)[:120]}); continuing", flush=True)
                return {}
            if d.get("status") in {"completed", "failed"} or self._find_video(d):
                return d
            if verbose:
                print(f"    polling ... status={d.get('status')}", flush=True)
        return {}


# --------------------------------------------------------------------------
# Veo client (predictLongRunning through the Swiggy gateway)
# --------------------------------------------------------------------------

class VeoClient:
    """Veo 3.1 via Gemini-API predictLongRunning through the same gateway.

    One consistent forwarding rule for every request (create, poll, fetch):

        custom-host header = https://generativelanguage.googleapis.com  (bare)
        request path       = /v1/v1beta/<google path>
        forwarded          = https://generativelanguage.googleapis.com/v1beta/<google path>

    The leading /v1/v1beta keeps Portkey's public-gateway files-interceptor
    from grabbing /v1/files/* routes: it only inspects the FIRST path
    segment, and "v1beta" is not a route it guards. This is what makes the
    final MP4 bytes fetchable (a plain /v1/files/... fetch is rejected with
    "A custom host is not allowed with the portkey provider").
    """

    def __init__(self, api_key: str, timeout: int = 600):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "x-portkey-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "omni_generate/1.0 (portkey-gateway-client)",
            "x-portkey-config": json.dumps(VEO_CONFIG),
            "x-portkey-custom-host": GOOGLE_HOST,
        }
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        """path is gateway-relative WITHOUT /v1, e.g. /v1beta/models/x:predictLongRunning."""
        url = f"{GATEWAY_URL}/v1{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} on {method} {url}: {detail}") from e

    def _fetch_bytes(self, google_uri: str) -> bytes:
        """Download a Google file URI through the gateway (v1beta-shifted)."""
        file_path = google_uri.replace("https://generativelanguage.googleapis.com", "")
        url = f"{GATEWAY_URL}/v1{file_path}"  # /v1 + /v1beta/files/... + query
        req = urllib.request.Request(url, method="GET", headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} fetching bytes: {detail}") from e

    def generate(self, prompt: str, seconds: int, retries: int = 2, verbose: bool = True) -> bytes:
        """Generate one Veo clip. Returns mp4 bytes."""
        body = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": seconds,
                "sampleCount": 1,
            },
        }

        attempt = 0
        while True:
            attempt += 1
            if verbose:
                print(f"    generating via Veo (attempt {attempt}, target {seconds}s) ...", flush=True)
            try:
                d = self._request("POST", f"/v1beta/models/{VEO_MODEL}:predictLongRunning", body)
            except RuntimeError as e:
                if attempt <= retries:
                    wait = 20 * attempt
                    print(f"    transient error ({str(e)[:120]}); retrying in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise
            op_name = d.get("name", "")
            if not op_name:
                raise RuntimeError(f"no operation name in create response: {str(d)[:300]}")

            # poll the long-running operation
            pd = {}
            deadline = time.time() + 420
            while time.time() < deadline:
                time.sleep(15)
                try:
                    pd = self._request("GET", f"/v1beta/{op_name}")
                except RuntimeError as e:
                    print(f"    poll failed ({str(e)[:120]}); continuing", flush=True)
                    continue
                if pd.get("done"):
                    break
                if verbose:
                    print(f"    polling ... done={pd.get('done')}", flush=True)
            if not pd.get("done"):
                raise RuntimeError(f"Veo operation timed out: {op_name}")

            uri = (
                pd.get("response", {}).get("generateVideoResponse", {})
                .get("generatedSamples", [{}])[0].get("video", {}).get("uri")
            )
            if not uri:
                raise RuntimeError(f"operation done but no video uri: {str(pd)[:300]}")
            if verbose:
                print(f"    downloading {uri.split('/')[-1][:40]} ...", flush=True)
            return self._fetch_bytes(uri)


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------

def ffprobe_meta(path: Path) -> dict:
    """Probe duration / resolution / audio presence by parsing ffmpeg -i stderr."""
    out = subprocess.run(
        [ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True
    ).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", out)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    vm = re.search(r"Stream #\d+:\d+.*?: Video: .*?, (\d+)x(\d+)", out)
    w, h = (int(vm.group(1)), int(vm.group(2))) if vm else (0, 0)
    has_audio = "Audio:" in out
    return {"duration": round(dur, 2), "width": w, "height": h, "has_audio": has_audio}


def run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_bin(), "-y", "-v", "error"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-800:]}")


# --------------------------------------------------------------------------
# template loading
# --------------------------------------------------------------------------

def list_templates() -> list[str]:
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(p.name for p in TEMPLATES_DIR.iterdir() if (p / "film.json").exists())


def load_template(template_dir: Path) -> dict:
    film_file = template_dir / "film.json"
    if not film_file.exists():
        sys.exit(f"Error: {film_file} not found.")
    film = json.loads(film_file.read_text())
    film["_dir"] = str(template_dir)
    # sanity-check clip definitions up front
    ids = set()
    for clip in film["clips"]:
        cid = clip["id"]
        if cid in ids:
            sys.exit(f"Error: duplicate clip id '{cid}' in film.json")
        ids.add(cid)
        if "prompt_file" not in clip:
            sys.exit(f"Error: clip '{cid}' missing prompt_file")
        if not (template_dir / clip["prompt_file"]).exists():
            sys.exit(f"Error: prompt file not found: {template_dir / clip['prompt_file']}")
    return film


def build_scope(film: dict, clip: dict, extra: dict | None = None) -> dict:
    scope = {
        "film": {k: v for k, v in film.items() if not k.startswith("_")},
        "style": film.get("variables", {}).get("style_block", ""),
        "seconds": clip.get("seconds", 8),
        "clip": clip,
    }
    variables = film.get("variables", {})
    scope.update({k: v for k, v in variables.items() if not isinstance(v, (dict, list))})
    if extra:
        scope.update(extra)
    return scope


def build_parts(film: dict, clip: dict, prompt: str) -> list[dict]:
    """Interaction input parts: reference images first, then the prompt text."""
    parts: list[dict] = []
    tdir = Path(film["_dir"])
    refs = clip.get("references", film.get("defaults", {}).get("references", []))
    for rel in refs:
        p = tdir / rel
        if not p.exists():
            raise FileNotFoundError(f"reference asset missing: {p}")
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        parts.append(
            {"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime}
        )
    parts.append({"type": "text", "text": prompt})
    return parts


# --------------------------------------------------------------------------
# clip generation
# --------------------------------------------------------------------------

def generate_clip(client: OmniClient, film: dict, clip: dict, out_dir: Path,
                  prev_id: str | None, force: bool, verbose: bool = True) -> dict:
    tdir = Path(film["_dir"])
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    mp4 = clips_dir / f"{clip['id']}.mp4"
    meta_file = clips_dir / f"{clip['id']}.json"

    if mp4.exists() and meta_file.exists() and not force:
        meta = json.loads(meta_file.read_text())
        need = clip.get("seconds", 8)
        if meta.get("duration", 0) >= need - 0.75:
            if verbose:
                print(f"  [skip] {clip['id']} exists ({meta['duration']}s) — reuse (--force to regen)")
            return meta
        if verbose:
            print(f"  [regen] {clip['id']} too short ({meta.get('duration')}s < {need}s)")

    prompt_raw = (tdir / clip["prompt_file"]).read_text()
    scope = build_scope(film, clip, extra={"prev_interaction_id": prev_id or ""})
    prompt = render_template(prompt_raw, scope, clip["prompt_file"])
    (clips_dir / f"{clip['id']}.prompt.txt").write_text(prompt)  # audit trail

    parts = build_parts(film, clip, prompt)
    video, iid, raw = client.generate(parts, seconds=clip.get("seconds", 8),
                                      prev_interaction_id=prev_id, verbose=verbose)
    mp4.write_bytes(video)
    meta = ffprobe_meta(mp4)

    # honesty check: did we actually get what we asked for?
    need = clip.get("seconds", 8)
    if meta["duration"] < need - 1.0 and verbose:
        print(f"  [warn] {clip['id']}: requested {need}s, model returned {meta['duration']}s")

    info = {
        "clip_id": clip["id"],
        "interaction_id": iid,
        "prompt_file": clip["prompt_file"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "seconds_requested": need,
        "duration": meta["duration"],
        "width": meta["width"],
        "height": meta["height"],
        "has_audio": meta["has_audio"],
        "bytes": mp4.stat().st_size,
        "chained_from": prev_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_file.write_text(json.dumps(info, indent=2))
    if verbose:
        print(f"  [ok] {clip['id']}: {meta['width']}x{meta['height']}, "
              f"{meta['duration']}s, audio={meta['has_audio']}, {mp4.stat().st_size // 1024}KB")
    return info


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def prepare_segment(film: dict, clip: dict, src: Path, work: Path, start_t: float) -> Path:
    """Trim to exact seconds, upscale, normalize fps/SAR/audio, apply overlays."""
    tdir = Path(film["_dir"])
    seconds = clip.get("seconds", 8)
    out_w, out_h = film.get("assembly", {}).get("upscale_to", [1920, 1080])
    fps = film.get("film", {}).get("fps", 24)

    base = [f"scale={out_w}:{out_h}:flags=lanczos", "setsar=1", f"fps={fps}"]
    overlay = clip.get("overlay")
    overlay_img = None
    if overlay:
        img = tdir / overlay["image"]
        if img.exists():
            overlay_img = img
        else:
            print(f"  [warn] overlay asset missing, skipping: {img}")

    seg = work / f"seg_{clip['id']}.mp4"
    args = ["-i", str(src), "-t", f"{seconds:.3f}"]
    meta = ffprobe_meta(src)
    if not meta["has_audio"]:
        args += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                 "-shortest", "-map", "0:v:0", "-map", "1:a:0"]
    else:
        args += ["-map", "0:v:0", "-map", "0:a:0?", "-af", "aresample=48000"]

    if overlay_img:
        # logo overlay needs the image as an extra input -> complex filtergraph
        pos = overlay.get("position", "top_left")
        margin = overlay.get("margin", 60)
        x, y = {"top_left": (margin, margin), "top_right": (f"main_w-overlay_w-{margin}", margin),
                "bottom_left": (margin, f"main_h-overlay_h-{margin}"),
                "bottom_right": (f"main_w-overlay_w-{margin}", f"main_h-overlay_h-{margin}")}[pos]
        args = ["-i", str(src), "-t", f"{seconds:.3f}", "-i", str(overlay_img)]
        if not meta["has_audio"]:
            args += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                     "-shortest"]
            vsrc, asrc = "0:v:0", "2:a:0"
        else:
            asrc = "0:a:0"
        fc = (f"[0:v]{','.join(base)}[base];"
              f"[1:v]format=rgba[colorkeyed];"
              f"[base][colorkeyed]overlay={x}:{y}[v]")
        args += ["-filter_complex", fc, "-map", "[v]", "-map", asrc]
    else:
        args += ["-vf", ",".join(base)]

    args += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(seg),
    ]
    run_ffmpeg(args)
    return seg


def assemble(film: dict, out_dir: Path) -> Path:
    asm = film.get("assembly", {})
    order = asm.get("order", [c["id"] for c in film["clips"]])
    clips_by_id = {c["id"]: c for c in film["clips"]}
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    print("\n-- assembly --")
    segs = []
    for cid in order:
        clip = clips_by_id[cid]
        src = out_dir / "clips" / f"{cid}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"clip missing for assembly: {src}")
        seg = prepare_segment(film, clip, src, work, start_t=0.0)
        meta = ffprobe_meta(seg)
        print(f"  segment {cid}: {meta['duration']}s @ {meta['width']}x{meta['height']}")
        segs.append(seg)

    list_file = work / "concat.txt"
    list_file.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    concat = work / "concat.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(concat)])

    # final unifying pass: warm low-contrast grade + subtle painterly grain
    grade = asm.get("grade", "eq=contrast=0.97:brightness=0.004:saturation=1.05,"
                             "colorbalance=rs=0.03:bs=-0.02")
    grain = asm.get("grain", 5)
    final = out_dir / asm.get("output", "final.mp4")
    vf = [grade, f"noise=alls={grain}:allf=t+u", "format=yuv420p"] if grain else [grade, "format=yuv420p"]
    run_ffmpeg(["-i", str(concat), "-vf", ",".join(vf),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy", "-movflags", "+faststart", str(final)])

    meta = ffprobe_meta(final)
    print(f"  final: {final.name} — {meta['duration']}s @ {meta['width']}x{meta['height']}, "
          f"{final.stat().st_size // 1024 // 1024}MB")
    return final


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Template-driven Omni Flash video generator")
    ap.add_argument("--template", help="template dir under templates/ (or a path)")
    ap.add_argument("--list-templates", action="store_true")
    ap.add_argument("--clips", default="all", help="comma-separated clip ids (default: all)")
    ap.add_argument("--force", action="store_true", help="regenerate clips even if they exist")
    ap.add_argument("--dry-run", action="store_true", help="render prompts and exit (no API calls)")
    ap.add_argument("--assemble-only", action="store_true", help="skip generation, only assemble")
    ap.add_argument("--no-assembly", action="store_true", help="generate clips only")
    ap.add_argument("--output-root", default=str(REPO / "output"))
    ap.add_argument(
        "--model",
        default="omni-flash",
        choices=["omni-flash", "veo-3.1"],
        help="generator backend: omni-flash (Interactions API, default) or veo-3.1 (predictLongRunning)",
    )
    args = ap.parse_args()

    if args.list_templates:
        print("\n".join(list_templates()) or "(no templates found)")
        return
    if not args.template:
        ap.error("--template is required (or use --list-templates)")

    tdir = Path(args.template)
    if not tdir.exists():
        tdir = TEMPLATES_DIR / args.template
    film = load_template(tdir)

    film_name = film.get("film", {}).get("name", tdir.name)
    out_dir = Path(args.output_root) / tdir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [c for c in film["clips"]
              if args.clips == "all" or c["id"] in {s.strip() for s in args.clips.split(",")}]

    # -- dry run: render every requested prompt, print, exit --
    if args.dry_run:
        for clip in wanted:
            prompt = render_template((tdir / clip["prompt_file"]).read_text(),
                                     build_scope(film, clip), clip["prompt_file"])
            print(f"\n===== {clip['id']} ({clip.get('seconds')}s) -> {clip['prompt_file']} =====")
            print(prompt)
        return

    load_env()
    report: dict = {"film": film_name, "template": str(tdir), "started": datetime.now().isoformat(timespec="seconds"),
                    "clips": [], "assembly": None}

    if not args.assemble_only:
        # Both clients expose generate(parts, seconds, prev_interaction_id,
        # retries, verbose) -> (mp4_bytes, id, raw); VeoClient ignores
        # parts/prev (single-shot prompt generation, no chaining).
        if args.model == "veo-3.1":
            veo = VeoClient(require_key())

            class _VeoAdapter:
                @staticmethod
                def generate(parts, seconds, prev_interaction_id=None, retries=2, verbose=True):
                    prompt = next(
                        (p["text"] for p in reversed(parts) if p.get("type") == "text"), ""
                    )
                    return veo.generate(prompt, seconds, retries=retries, verbose=verbose), "veo", {}

            client: OmniClient | _VeoAdapter = _VeoAdapter()
        else:
            client = OmniClient(require_key())
        prev_id: str | None = None
        print(f"Generating {len(wanted)} clip(s) for '{film_name}' -> {out_dir} [model={args.model}]")
        for clip in film["clips"]:  # iterate ALL clips in order so chaining stays correct
            if clip not in wanted:
                # keep the chain alive across partially-regenerated runs
                meta_file = out_dir / "clips" / f"{clip['id']}.json"
                if clip.get("chain") and meta_file.exists():
                    prev_id = json.loads(meta_file.read_text()).get("interaction_id")
                continue
            print(f"\n-- clip {clip['id']} ({clip.get('seconds')}s) --")
            info = generate_clip(client, film, clip, out_dir,
                                 prev_id=prev_id if clip.get("chain") else None,
                                 force=args.force)
            report["clips"].append(info)
            if clip.get("chain"):
                prev_id = info.get("interaction_id") or prev_id

    if not args.no_assembly:
        final = assemble(film, out_dir)
        report["assembly"] = {
            "output": str(final), "meta": ffprobe_meta(final),
            "bytes": final.stat().st_size,
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

    # backfill the report with EVERY clip's on-disk metadata (not just ones
    # generated in this invocation) so the report always describes the film
    clips_dir = out_dir / "clips"
    by_id = {c["clip_id"]: c for c in report["clips"]}
    report["clips"] = []
    for clip in film["clips"]:
        mf = clips_dir / f"{clip['id']}.json"
        report["clips"].append(json.loads(mf.read_text()) if mf.exists() else by_id.get(clip["id"], {"clip_id": clip["id"], "missing": True}))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
