from __future__ import annotations

from ..schemas import CandidateKind, CandidateStatus, FinalizeCandidateInput, ToolResult
from ..media import validate_deliverable
from .base import ToolContext, validate_rollbacks


class FinalizeCandidateTool:
    name = "finalize_candidate"

    def run(self, context: ToolContext, arguments: FinalizeCandidateInput) -> ToolResult:
        validate_rollbacks(context, arguments.rollback_candidate_ids)
        if arguments.candidate_id in arguments.rollback_candidate_ids:
            raise ValueError("The finalized candidate cannot also be rolled back")
        candidate = context.graph.require(arguments.candidate_id, CandidateKind.VIDEO)
        if candidate.status in {CandidateStatus.REJECTED, CandidateStatus.ROLLED_BACK}:
            raise ValueError(f"Cannot finalize {candidate.status.value} candidate {candidate.candidate_id}")
        candidate_path = context.artifacts.resolve(candidate.artifact_path)
        recorded_media = candidate.inputs.get("service_result", {}).get("validated_media", {})
        media = validate_deliverable(
            candidate_path,
            frames=int(recorded_media.get("frames", 240)),
            fps=int(recorded_media.get("fps", 24)),
        )
        deliverable = context.artifacts.publish_final_video(candidate_path, context.state.case_id)
        context.graph.finalize(
            arguments.candidate_id,
            arguments.rationale,
            str(deliverable),
        )
        for rolled_back_id in arguments.rollback_candidate_ids:
            context.graph.rollback(rolled_back_id)
        return ToolResult(
            tool_name=self.name,
            summary=f"Finalized {arguments.candidate_id}",
            candidate_ids=[arguments.candidate_id],
            artifacts=[candidate.artifact_path, str(deliverable)],
            payload={
                "rationale": arguments.rationale,
                "media": media,
                "deliverable": str(deliverable),
            },
        )
