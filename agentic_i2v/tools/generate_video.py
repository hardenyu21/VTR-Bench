from __future__ import annotations

from ..prompt_composer import compose_h3_prompt, text_sha256
from ..schemas import Candidate, CandidateKind, GenerateVideoInput, ToolResult
from ..state import sha256_file
from .base import ToolContext, validate_rollbacks


class GenerateVideoTool:
    name = "generate_video"

    def run(self, context: ToolContext, arguments: GenerateVideoInput) -> ToolResult:
        validate_rollbacks(context, arguments.rollback_candidate_ids)
        keyframe = context.graph.require(arguments.keyframe_id, CandidateKind.IMAGE)
        parent_id = arguments.parent_video_id or arguments.keyframe_id
        if arguments.parent_video_id:
            context.graph.require(arguments.parent_video_id, CandidateKind.VIDEO)
        candidate_id = context.graph.next_id(CandidateKind.VIDEO)
        output = context.artifacts.candidate_path("video", candidate_id, ".mp4")
        composed_h3_prompt, motion_refinement = compose_h3_prompt(
            context.state.prompt_en, arguments.motion_refinement
        )
        result = context.h3.generate(
            image_path=context.artifacts.resolve(keyframe.artifact_path),
            motion_prompt=composed_h3_prompt,
            output_path=output,
        )
        recorded_inputs = arguments.model_dump(mode="json")
        recorded_inputs["motion_refinement"] = motion_refinement
        recorded_inputs.update(
            {
                "original_prompt_sha256": context.state.prompt_sha256,
                "composed_h3_prompt": composed_h3_prompt,
                "composed_h3_prompt_sha256": text_sha256(composed_h3_prompt),
                "prompt_composition": "verbatim original + delta-only motion refinement",
            }
        )
        context.graph.add(
            Candidate(
                candidate_id=candidate_id,
                kind=CandidateKind.VIDEO,
                parent_id=parent_id,
                created_by=self.name,
                artifact_path=context.artifacts.relative(output),
                sha256=sha256_file(output),
                inputs={**recorded_inputs, "service_result": result},
                planner_diagnosis={"summary": arguments.diagnosis},
            )
        )
        for rolled_back_id in arguments.rollback_candidate_ids:
            context.graph.rollback(rolled_back_id)
        media = result.get("validated_media", result.get("media", {}))
        return ToolResult(
            tool_name=self.name,
            summary=(
                f"Generated validated {media.get('frames', 'unknown')}-frame video {candidate_id}"
            ),
            candidate_ids=[candidate_id],
            artifacts=[context.artifacts.relative(output)],
            payload={"media": media},
        )
