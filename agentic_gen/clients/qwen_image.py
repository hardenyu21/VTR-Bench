from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from ..media.video_io import image_data_url, verify_image
from ..state.artifact_store import sha256_file


class QwenImageClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        width: int = 1344,
        height: int = 768,
    ) -> None:
        self.api_key = api_key
        compatible_suffix = "/compatible-mode/v1"
        if not base_url.rstrip("/").endswith(compatible_suffix):
            raise ValueError(
                "QwenImageClient requires a Bailian OpenAI-compatible base URL ending in "
                f"{compatible_suffix!r}"
            )
        workspace_root = base_url.rstrip("/")[: -len(compatible_suffix)]
        self.endpoint = (
            f"{workspace_root}/api/v1/services/aigc/multimodal-generation/generation"
        )
        self.model = model
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive")

    @staticmethod
    def _save_url(url: str, target: Path) -> dict[str, Any]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if url.startswith("data:"):
            _, encoded = url.split(",", 1)
            target.write_bytes(base64.b64decode(encoded))
            transport = "base64"
        else:
            with httpx.Client(follow_redirects=True, timeout=180.0) as client:
                response = client.get(url)
                response.raise_for_status()
                target.write_bytes(response.content)
            transport = "temporary_url"
        width, height = verify_image(target)
        return {"width": width, "height": height, "sha256": sha256_file(target), "transport": transport}

    @staticmethod
    def _extract_image_urls(payload: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for choice in payload.get("output", {}).get("choices", []):
            for item in choice.get("message", {}).get("content", []):
                image = item.get("image") if isinstance(item, dict) else None
                if image:
                    urls.append(str(image))
        return urls

    def generate(
        self, *, prompt: str, negative_prompt: str, count: int, targets: list[Path], reference: Path | None = None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = []
        if reference is not None:
            content.append({"image": image_data_url(reference)})
        content.append({"text": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "negative_prompt": negative_prompt,
                "seed": 42,
                "prompt_extend": False,
                "watermark": False,
                "n": count,
                "size": f"{self.width}*{self.height}",
            },
        }
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Qwen Image request failed with HTTP {response.status_code}: {response.text[:1000]}"
            ) from exc
        payload = response.json()
        urls = self._extract_image_urls(payload)
        if len(urls) != len(targets):
            raise RuntimeError(f"Requested {len(targets)} images, received {len(urls)}")
        request_id = payload.get("request_id")
        metadata = [self._save_url(url, target) for url, target in zip(urls, targets, strict=True)]
        for details in metadata:
            details["request_id"] = request_id
        return metadata
