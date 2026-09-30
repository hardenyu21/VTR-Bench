from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..clients import BailianChatClient, MiniMaxH3Client, QwenImageClient
from ..schemas import StrictModel, ToolResult, WorkflowState
from ..state import ArtifactStore, CandidateGraph, CheckpointStore


@dataclass
class ToolContext:
    state: WorkflowState
    graph: CandidateGraph
    artifacts: ArtifactStore
    checkpoints: CheckpointStore
    chat: BailianChatClient
    images: QwenImageClient
    h3: MiniMaxH3Client
    prompts: dict[str, str]


class WorkflowTool(Protocol):
    name: str

    def run(self, context: ToolContext, arguments: StrictModel) -> ToolResult: ...


def observations_from(payload: dict[str, Any]):
    from ..schemas import Observation

    observations = []
    for item in payload.get("observations", []):
        if not isinstance(item, dict):
            observations.append(Observation(category="critic", summary=str(item)))
            continue
        observations.append(
            Observation(
                category=str(item.get("category", "critic")),
                summary=str(item.get("summary", item.get("observation", "unspecified observation"))),
                severity=max(0, min(3, int(item.get("severity", 1)))),
                evidence_locations=[str(value) for value in item.get("evidence_locations", [])],
            )
        )
    return observations


def validate_rollbacks(context: ToolContext, candidate_ids: list[str]) -> None:
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("rollback_candidate_ids contains duplicates")
    for candidate_id in candidate_ids:
        context.graph.require(candidate_id)
