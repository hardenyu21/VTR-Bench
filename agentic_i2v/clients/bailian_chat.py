from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..media.video_io import image_data_url


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        key: int(getattr(usage, key))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if usage is not None and getattr(usage, key, None) is not None
    }


class BailianChatClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=600.0, max_retries=2)
        self.model = model

    def tool_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments or "{}"),
                }
            )
        return {"content": message.content or "", "tool_calls": calls, "usage": usage_dict(response)}

    def inspect_json(self, *, system: str, text: str, images: list[Path]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.2,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        payload["_api"] = {"model": self.model, "usage": usage_dict(response)}
        return payload

