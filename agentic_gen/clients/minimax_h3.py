from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from ..media.validation import validate_deliverable


class MiniMaxH3Client:
    """Generation-only client. Deliberately exposes no lifecycle operations."""

    def __init__(self, service_url: str) -> None:
        self.service_url = service_url.rstrip("/")
        self._health: dict[str, Any] | None = None

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.service_url}/health")
            response.raise_for_status()
            payload = response.json()
        if payload.get("status") != "ready":
            raise RuntimeError(f"H3 service is not ready: {payload}")
        self._health = payload
        return payload

    def generate(self, *, image_path: Path, motion_prompt: str, output_path: Path) -> dict[str, Any]:
        contract = self._health or self.health()
        requested_frames = int(contract["requested_frames"])
        delivered_frames = int(contract["delivered_frames"])
        fps = int(contract["fps"])
        request = {
            "image_path": str(image_path.resolve()),
            "motion_prompt": motion_prompt,
            "output_path": str(output_path.resolve()),
            "seed": 42,
            "requested_frames": requested_frames,
            "delivered_frames": delivered_frames,
            "fps": fps,
        }
        timeout = httpx.Timeout(connect=10.0, read=7200.0, write=30.0, pool=7200.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{self.service_url}/generate", json=request)
            response.raise_for_status()
            payload = response.json()
        if payload.get("status") != "complete":
            raise RuntimeError(f"Unexpected H3 response: {payload}")
        payload["validated_media"] = validate_deliverable(
            output_path, frames=delivered_frames, fps=fps
        )
        return payload
