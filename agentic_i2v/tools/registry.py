from __future__ import annotations

from typing import Any

from ..schemas import TOOL_INPUT_MODELS, ToolResult
from .base import ToolContext, WorkflowTool
from .compare_candidates import CompareCandidatesTool
from .edit_keyframe import EditKeyframeTool
from .extract_frames import ExtractFramesTool
from .finalize_candidate import FinalizeCandidateTool
from .generate_keyframes import GenerateKeyframesTool
from .generate_video import GenerateVideoTool
from .inspect_keyframes import InspectKeyframesTool
from .inspect_video import InspectVideoTool


class ToolRegistry:
    def __init__(self) -> None:
        tools = [
            GenerateKeyframesTool(), InspectKeyframesTool(), EditKeyframeTool(), GenerateVideoTool(),
            InspectVideoTool(), ExtractFramesTool(), CompareCandidatesTool(), FinalizeCandidateTool(),
        ]
        self.tools: dict[str, WorkflowTool] = {tool.name: tool for tool in tools}

    def execute(self, context: ToolContext, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        validated = TOOL_INPUT_MODELS[name].model_validate(arguments)
        return self.tools[name].run(context, validated)

