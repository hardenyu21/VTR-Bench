from __future__ import annotations

from ..schemas import Candidate, CandidateKind, GenerateKeyframesInput, ToolResult
from ..state import sha256_file
from .base import ToolContext, validate_rollbacks


class GenerateKeyframesTool:
    name = "generate_keyframes"

    def run(self, context: ToolContext, arguments: GenerateKeyframesInput) -> ToolResult:
        validate_rollbacks(context, arguments.rollback_candidate_ids)
        reference = None
        if arguments.parent_id:
            parent = context.graph.require(arguments.parent_id, CandidateKind.IMAGE)
            reference = context.artifacts.resolve(parent.artifact_path)
        ids = [context.graph.next_id(CandidateKind.IMAGE)]
        # Reserve unique IDs in memory without adding partially generated candidates.
        first = int(ids[0].split("-")[1])
        ids = [f"img-{first + offset:04d}" for offset in range(arguments.count)]
        targets = [context.artifacts.candidate_path("image", item, ".png") for item in ids]
        metadata = context.images.generate(
            prompt=arguments.generation_prompt,
            negative_prompt=arguments.negative_prompt,
            count=arguments.count,
            targets=targets,
            reference=reference,
        )
        for candidate_id, target, details in zip(ids, targets, metadata, strict=True):
            context.graph.add(
                Candidate(
                    candidate_id=candidate_id,
                    kind=CandidateKind.IMAGE,
                    parent_id=arguments.parent_id,
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
            summary=f"Generated {len(ids)} immutable image candidate(s)",
            candidate_ids=ids,
            artifacts=[context.artifacts.relative(path) for path in targets],
        )
