from __future__ import annotations

import json

from ..media import sample_uniform
from ..schemas import CandidateKind, CompareCandidatesInput, ComparisonReport, ToolResult
from .base import ToolContext


class CompareCandidatesTool:
    name = "compare_candidates"

    def run(self, context: ToolContext, arguments: CompareCandidatesInput) -> ToolResult:
        paths = []
        attachment_map = []
        for candidate_id in arguments.candidate_ids:
            candidate = context.graph.require(candidate_id)
            if candidate.kind == CandidateKind.IMAGE:
                paths.append(context.artifacts.resolve(candidate.artifact_path))
                attachment_map.append({"candidate_id": candidate_id, "type": "image", "count": 1})
            else:
                frames = sample_uniform(
                    context.artifacts.resolve(candidate.artifact_path),
                    context.artifacts.frames / candidate_id / "compare_fps_0.5",
                    0.5,
                )
                paths.extend(frames)
                attachment_map.append({"candidate_id": candidate_id, "type": "video_samples", "count": len(frames)})
        payload = context.chat.inspect_json(
            system=context.prompts["compare"],
            text=json.dumps(
                {
                    "original_prompt": context.state.prompt_en,
                    "focus": arguments.focus,
                    "attachment_map": attachment_map,
                },
                ensure_ascii=False,
            ),
            images=paths,
        )
        api = payload.pop("_api", {})
        report_model = ComparisonReport.model_validate(payload)
        observations = report_model.observations
        report = context.artifacts.reports / f"compare_{len(context.state.tool_calls):04d}.json"
        context.artifacts.atomic_json(report, {**report_model.model_dump(mode="json"), "_api": api})
        return ToolResult(
            tool_name=self.name,
            summary=report_model.summary,
            candidate_ids=arguments.candidate_ids,
            artifacts=[context.artifacts.relative(report)],
            observations=observations,
            payload={
                "preferred_candidate_id": report_model.preferred_candidate_id,
                "ranking": report_model.ranking,
            },
        )
