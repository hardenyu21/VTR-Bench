from __future__ import annotations

from ..schemas import Candidate, CandidateKind, EditKeyframeInput, ToolResult
from ..state import sha256_file
from .base import ToolContext, validate_rollbacks


class EditKeyframeTool:
    name = "edit_keyframe"

    def run(self, context: ToolContext, arguments: EditKeyframeInput) -> ToolResult:
        validate_rollbacks(context, arguments.rollback_candidate_ids)
        parent = context.graph.require(arguments.candidate_id, CandidateKind.IMAGE)
        source = context.artifacts.resolve(parent.artifact_path)
        candidate_id = context.graph.next_id(CandidateKind.IMAGE)
        target = context.artifacts.candidate_path("image", candidate_id, ".png")
        prompt = arguments.edit_instruction
        if arguments.region_description:
            prompt += f"\nTarget region: {arguments.region_description}"
        details = context.images.generate(
            prompt=prompt,
            negative_prompt="blurred text, distorted letters, illegible print, unintended changes",
            count=1,
            targets=[target],
            reference=source,
        )[0]
        context.graph.add(
            Candidate(
                candidate_id=candidate_id,
                kind=CandidateKind.IMAGE,
                parent_id=arguments.candidate_id,
                created_by=self.name,
                artifact_path=context.artifacts.relative(target),
                sha256=sha256_file(target),
                inputs={**arguments.model_dump(mode="json"), "provider": details},
                planner_diagnosis={"summary": arguments.diagnosis},
            )
        )
        for rolled_back_id in arguments.rollback_candidate_ids:
            context.graph.rollback(rolled_back_id)
        return ToolResult(
            tool_name=self.name,
            summary=f"Created targeted edit {candidate_id} from {arguments.candidate_id}",
            candidate_ids=[candidate_id],
            artifacts=[context.artifacts.relative(target)],
        )
