from __future__ import annotations

import json

from ..schemas import CandidateKind, InspectionReport, InspectKeyframesInput, ToolResult
from .base import ToolContext


class InspectKeyframesTool:
    name = "inspect_keyframes"

    def run(self, context: ToolContext, arguments: InspectKeyframesInput) -> ToolResult:
        candidates = [context.graph.require(item, CandidateKind.IMAGE) for item in arguments.candidate_ids]
        paths = [context.artifacts.resolve(item.artifact_path) for item in candidates]
        payload = context.chat.inspect_json(
            system=context.prompts["keyframe_inspector"],
            text=json.dumps(
                {
                    "original_prompt": context.state.prompt_en,
                    "images_in_attachment_order": arguments.candidate_ids,
                },
                ensure_ascii=False,
            ),
            images=paths,
        )
        api = payload.pop("_api", {})
        report_model = InspectionReport.model_validate(payload)
        observations = report_model.observations
        for candidate_id in arguments.candidate_ids:
            context.graph.observe(candidate_id, observations)
        report = context.artifacts.reports / f"inspect_images_{len(context.state.tool_calls):04d}.json"
        context.artifacts.atomic_json(report, {**report_model.model_dump(mode="json"), "_api": api})
        return ToolResult(
            tool_name=self.name,
            summary=report_model.summary,
            candidate_ids=arguments.candidate_ids,
            artifacts=[context.artifacts.relative(report)],
            observations=observations,
            payload={"ranking": report_model.ranking},
        )
