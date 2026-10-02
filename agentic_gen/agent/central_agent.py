from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from ..schemas import TOOL_INPUT_MODELS
from .context_builder import compact_context
from .prompts import CENTRAL_AGENT_PROMPT


class ToolCallingClient(Protocol):
    def tool_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


TOOL_DESCRIPTIONS = {
    "generate_keyframes": "Generate one or more new opening-frame candidates, optionally branching from an image.",
    "inspect_keyframes": "Inspect visible evidence in one or more image candidates.",
    "edit_keyframe": "Create a targeted edited child image without overwriting its parent.",
    "generate_video": (
        "Generate one serialized MiniMax-H3 video from an image candidate. Supply only additive, "
        "delta-only motion_refinement; the runtime preserves and prepends original_prompt verbatim."
    ),
    "inspect_video": "Inspect chronological samples from a video candidate.",
    "extract_frames": "Extract deterministic frames at requested positions for closer evidence.",
    "compare_candidates": "Compare two or more candidate images or videos and identify regressions.",
    "finalize_candidate": "Explicitly select one video candidate and terminate the workflow.",
}


def tool_definitions(models: dict[str, type[BaseModel]] | None = None) -> list[dict[str, Any]]:
    selected = models or TOOL_INPUT_MODELS
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in selected.items()
    ]


class CentralAgent:
    def __init__(self, client: ToolCallingClient) -> None:
        self.client = client

    def decide(self, state) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": CENTRAL_AGENT_PROMPT},
            *state.messages,
            {"role": "user", "content": compact_context(state)},
        ]
        return self.client.tool_turn(messages, tool_definitions())
