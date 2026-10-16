"""Portkey AI gateway adapter for cloud video generation.

Routes video generation through Portkey (https://portkey.ai) so every
supported video model — Gemini/Veo, Seedance, Sora, and more — is callable
through a single gateway key instead of one provider key per vendor.

Uses Portkey's OpenAI-compatible Videos API:
  POST /v1/videos                create a generation task
  GET  /v1/videos/{id}           poll task status
  GET  /v1/videos/{id}/content   download the finished MP4

Models are addressed as ``@provider-slug/model`` (e.g.
``@google/veo-3.1``, ``@openai/sora-2``); the slug is what routes the
request to the right vendor inside the gateway. Provider credentials live
in the Portkey dashboard (provider integrations / virtual keys), so no
per-vendor env vars are needed for models reached through this tool.

Set ``PORTKEY_API_KEY`` to enable. Optional: ``PORTKEY_BASE_URL``
(enterprise/self-hosted gateway), ``PORTKEY_VIRTUAL_KEY`` and
``PORTKEY_CONFIG`` (gateway routing config id).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


# Rough per-second USD rates for cost estimation. These are planning
# figures, not quotes — Portkey billing shows the actual charge.
_DEFAULT_COST_PER_SECOND = {
    "veo": 0.40,
    "sora-2-pro": 0.50,
    "sora-2": 0.10,
    "seedance": 0.05,
    "kling": 0.07,
    "minimax": 0.05,
}
_FALLBACK_COST_PER_SECOND = 0.30

_POLL_INTERVAL_SECONDS = 10.0
_TERMINAL_STATUSES = {"completed", "succeeded", "failed", "cancelled", "canceled"}


class PortkeyVideo(BaseTool):
    """Generate video through the Portkey AI gateway (single-key routing)."""

    name = "portkey_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "portkey"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:PORTKEY_API_KEY"]
    install_instructions = (
        "Set PORTKEY_API_KEY in .env (https://app.portkey.ai/ → API Keys).\n"
        "Add your video-model credentials (Google/Veo, fal.ai/Seedance, "
        "OpenAI/Sora, ...) as provider integrations or virtual keys in the "
        "Portkey dashboard — OpenMontage only needs the one gateway key."
    )
    agent_skills = ["portkey-video", "ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "native_audio": True,
        "provider_routing": True,
        "virtual_keys": True,
        "gateway_fallbacks": True,
    }
    best_for = [
        "one API key for every Portkey-routed video model",
        "gateway-level routing, fallbacks, and budget governance",
        "switching vendors by changing the model slug, not the tool",
    ]
    not_good_for = ["offline generation", "local GPU generation"]
    fallback_tools = ["veo_video", "seedance_ark", "kling_video", "minimax_video", "wan_video"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "model"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "description": (
                    "Gateway model id. Either a plain alias ('veo-3.1', "
                    "'omni-flash') when the API key carries a routing "
                    "config, or '@provider-slug/model' (e.g. "
                    "'@google/veo-3.1') for explicit gateway routing."
                ),
            },
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "seconds": {
                "type": "string",
                "description": "Clip length, e.g. '4', '8', '12'",
            },
            "size": {
                "type": "string",
                "description": "Frame size, e.g. '1280x720' or '720x1280'",
            },
            "input_reference_path": {
                "type": "string",
                "description": "Local image path for image_to_video",
            },
            "output_path": {"type": "string"},
            "timeout_seconds": {
                "type": "integer",
                "default": 900,
                "description": "Max seconds to poll before giving up",
            },
            "cost_per_second_usd": {
                "type": "number",
                "description": "Override the built-in cost-per-second estimate",
            },
            "task_action": {
                "type": "string",
                "enum": ["generate", "status", "download"],
                "default": "generate",
            },
            "task_id": {
                "type": "string",
                "description": "Gateway video id (for status/download actions)",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "seconds", "size"]
    side_effects = ["writes video file to output_path", "calls the Portkey gateway"]
    user_visible_verification = [
        "Watch generated clip for visual quality and motion",
        "Confirm the gateway log in the Portkey dashboard matches the model used",
    ]

    # ---- Status ----

    def get_status(self) -> ToolStatus:
        if os.environ.get("PORTKEY_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    # ---- Cost ----

    def _cost_per_second(self, model: str, override: Any) -> float:
        if isinstance(override, (int, float)) and override > 0:
            return float(override)
        lowered = model.lower()
        for key, rate in _DEFAULT_COST_PER_SECOND.items():
            if key in lowered:
                return rate
        return _FALLBACK_COST_PER_SECOND

    def _parse_seconds(self, raw: Any) -> int:
        text = str(raw or "").strip().lower().rstrip("s")
        try:
            return int(text)
        except ValueError:
            return 8

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = str(inputs.get("model", ""))
        per_second = self._cost_per_second(model, inputs.get("cost_per_second_usd"))
        return round(per_second * self._parse_seconds(inputs.get("seconds")), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 180.0

    # ---- HTTP helpers ----

    def _base_url(self) -> str:
        return os.environ.get("PORTKEY_BASE_URL", "https://api.portkey.ai/v1").rstrip("/")

    def _headers(self, json_body: bool) -> dict[str, str]:
        headers = {"x-portkey-api-key": os.environ.get("PORTKEY_API_KEY", "")}
        if json_body:
            headers["Content-Type"] = "application/json"
        virtual_key = os.environ.get("PORTKEY_VIRTUAL_KEY")
        if virtual_key:
            headers["x-portkey-virtual-key"] = virtual_key
        config = os.environ.get("PORTKEY_CONFIG") or os.environ.get("PORTKEY_CONFIG_ID")
        if config:
            headers["x-portkey-config"] = config
        return headers

    def _safe_error(self, exc: BaseException | str) -> str:
        text = str(exc)
        api_key = os.environ.get("PORTKEY_API_KEY", "")
        if api_key and api_key in text:
            text = text.replace(api_key, "[redacted]")
        return text[:500]

    def _create_task(self, inputs: dict[str, Any]) -> str:
        """POST /videos; returns the gateway video id."""
        model = str(inputs["model"]).strip()
        body: dict[str, Any] = {"prompt": inputs["prompt"], "model": model}
        if inputs.get("seconds"):
            body["seconds"] = str(inputs["seconds"])
        if inputs.get("size"):
            body["size"] = str(inputs["size"])

        reference_path = inputs.get("input_reference_path")
        if inputs.get("operation", "text_to_video") == "image_to_video":
            if not reference_path:
                raise ValueError("image_to_video requires input_reference_path")
            # Multipart form: the reference image rides alongside the fields.
            with open(reference_path, "rb") as fh:
                response = requests.post(
                    f"{self._base_url()}/videos",
                    headers=self._headers(json_body=False),
                    data={k: str(v) for k, v in body.items()},
                    files={"input_reference": (Path(reference_path).name, fh)},
                    timeout=120,
                )
        else:
            response = requests.post(
                f"{self._base_url()}/videos",
                headers=self._headers(json_body=True),
                json=body,
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        video_id = payload.get("id")
        if not video_id:
            raise RuntimeError(f"Portkey create returned no id: {payload}")
        return str(video_id)

    def _get_task(self, task_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self._base_url()}/videos/{task_id}",
            headers=self._headers(json_body=True),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _download(self, task_id: str, output_path: Path) -> None:
        response = requests.get(
            f"{self._base_url()}/videos/{task_id}/content",
            headers=self._headers(json_body=True),
            timeout=600,
            stream=True,
        )
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)

    # ---- Execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        started = time.time()
        action = str(inputs.get("task_action", "generate") or "generate")
        try:
            if action == "status":
                task = self._get_task(str(inputs["task_id"]))
                return ToolResult(
                    success=True,
                    data={"task": task, "status": task.get("status")},
                    model=task.get("model"),
                    duration_seconds=round(time.time() - started, 2),
                )

            estimated_cost_usd = self.estimate_cost(inputs)
            if action == "download":
                task_id = str(inputs["task_id"])
                task = self._get_task(task_id)
            else:
                if not inputs.get("prompt"):
                    return ToolResult(success=False, error="prompt is required")
                if not inputs.get("model"):
                    return ToolResult(success=False, error="model is required")
                task_id = self._create_task(inputs)
                task = self._get_task(task_id)

            timeout_seconds = float(inputs.get("timeout_seconds") or 900)
            while str(task.get("status", "")).lower() not in _TERMINAL_STATUSES:
                if time.time() - started > timeout_seconds:
                    return ToolResult(
                        success=False,
                        data={"task_id": task_id, "status": task.get("status")},
                        error=(
                            f"Timed out after {int(timeout_seconds)}s polling "
                            f"task {task_id}; retry with task_action='status'"
                        ),
                        duration_seconds=round(time.time() - started, 2),
                    )
                time.sleep(_POLL_INTERVAL_SECONDS)
                task = self._get_task(task_id)

            status = str(task.get("status", "")).lower()
            if status not in {"completed", "succeeded"}:
                error_detail = (
                    task.get("error") or task.get("last_error") or {}
                )
                message = (
                    error_detail.get("message", "")
                    if isinstance(error_detail, dict)
                    else str(error_detail)
                )
                return ToolResult(
                    success=False,
                    data={"task_id": task_id, "status": status},
                    error=f"Portkey video task {status}"
                    + (f": {message}" if message else ""),
                    duration_seconds=round(time.time() - started, 2),
                    model=task.get("model"),
                )

            output_path = Path(
                inputs.get("output_path") or f"portkey_{task_id}.mp4"
            )
            self._download(task_id, output_path)

            from tools.video._shared import probe_output

            return ToolResult(
                success=True,
                data={
                    "provider": self.provider,
                    "task_id": task_id,
                    "status": status,
                    "model": task.get("model") or inputs.get("model"),
                    "prompt": inputs.get("prompt"),
                    "operation": inputs.get("operation", "text_to_video"),
                    "output": str(output_path),
                    "output_path": str(output_path),
                    "format": "mp4",
                    "size": inputs.get("size"),
                    "seconds": inputs.get("seconds"),
                    "estimated_cost_usd": estimated_cost_usd,
                    **probe_output(output_path),
                },
                artifacts=[str(output_path)],
                cost_usd=estimated_cost_usd,
                duration_seconds=round(time.time() - started, 2),
                model=str(task.get("model") or inputs.get("model") or ""),
            )
        except requests.HTTPError as exc:
            body = ""
            response = getattr(exc, "response", None)
            if response is not None:
                body = (response.text or "")[:300]
            return ToolResult(
                success=False,
                error=f"Portkey API error: {self._safe_error(exc)}{': ' + body if body else ''}",
                duration_seconds=round(time.time() - started, 2),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Portkey video request failed: {self._safe_error(exc)}",
                duration_seconds=round(time.time() - started, 2),
            )
