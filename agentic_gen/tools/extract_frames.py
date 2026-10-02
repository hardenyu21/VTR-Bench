from __future__ import annotations

from ..media import extract_positions
from ..schemas import CandidateKind, ExtractFramesInput, ToolResult
from .base import ToolContext


class ExtractFramesTool:
    name = "extract_frames"

    def run(self, context: ToolContext, arguments: ExtractFramesInput) -> ToolResult:
        candidate = context.graph.require(arguments.candidate_id, CandidateKind.VIDEO)
        directory = context.artifacts.frames / arguments.candidate_id / "positions"
        paths = extract_positions(
            context.artifacts.resolve(candidate.artifact_path), directory, arguments.positions_seconds
        )
        return ToolResult(
            tool_name=self.name,
            summary=f"Extracted {len(paths)} deterministic frame(s) from {arguments.candidate_id}",
            candidate_ids=[arguments.candidate_id],
            artifacts=[context.artifacts.relative(path) for path in paths],
            payload={"positions_seconds": arguments.positions_seconds},
        )

