from __future__ import annotations

import json

from ..media import sample_uniform
from ..schemas import CandidateKind, InspectionReport, InspectVideoInput, ToolResult
from .base import ToolContext


class InspectVideoTool:
    name = "inspect_video"

    def run(self, context: ToolContext, arguments: InspectVideoInput) -> ToolResult:
        candidate = context.graph.require(arguments.candidate_id, CandidateKind.VIDEO)
        frame_dir = context.artifacts.frames / arguments.candidate_id / f"fps_{arguments.sample_fps:g}"
        frames = sample_uniform(
            context.artifacts.resolve(candidate.artifact_path), frame_dir, arguments.sample_fps
        )
        payload = context.chat.inspect_json(
            system=context.prompts["video_inspector"],
            text=json.dumps(
                {
                    "original_prompt": context.state.prompt_en,
                    "candidate_id": arguments.candidate_id,
                    "motion_refinement": candidate.inputs.get("motion_refinement", ""),
                    "composed_h3_prompt": candidate.inputs.get("composed_h3_prompt", ""),
                    "sample_fps": arguments.sample_fps,
                },
                ensure_ascii=False,
            ),
            images=frames,
        )
        api = payload.pop("_api", {})
        report_model = InspectionReport.model_validate(payload)
        observations = report_model.observations
        context.graph.observe(arguments.candidate_id, observations)
        report = context.artifacts.reports / f"inspect_{arguments.candidate_id}_{len(context.state.tool_calls):04d}.json"
        context.artifacts.atomic_json(report, {**report_model.model_dump(mode="json"), "_api": api})
        return ToolResult(
            tool_name=self.name,
            summary=report_model.summary,
            candidate_ids=[arguments.candidate_id],
            artifacts=[context.artifacts.relative(report)],
            observations=observations,
        )
