from __future__ import annotations

import re

from ..schemas import Candidate, CandidateKind, CandidateStatus, FinalSelection, Observation, WorkflowState


class CandidateGraph:
    def __init__(self, state: WorkflowState) -> None:
        self.state = state

    def next_id(self, kind: CandidateKind) -> str:
        prefix = "img" if kind == CandidateKind.IMAGE else "vid"
        pattern = re.compile(rf"^{prefix}-(\d+)$")
        numbers = [int(match.group(1)) for key in self.state.candidates if (match := pattern.match(key))]
        return f"{prefix}-{max(numbers, default=0) + 1:04d}"

    def add(self, candidate: Candidate) -> None:
        if candidate.candidate_id in self.state.candidates:
            raise ValueError(f"Candidate already exists: {candidate.candidate_id}")
        if candidate.parent_id and candidate.parent_id not in self.state.candidates:
            raise KeyError(f"Unknown parent candidate: {candidate.parent_id}")
        self.state.candidates[candidate.candidate_id] = candidate

    def require(self, candidate_id: str, kind: CandidateKind | None = None) -> Candidate:
        candidate = self.state.candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        if kind is not None and candidate.kind != kind:
            raise ValueError(f"{candidate_id} is {candidate.kind.value}, expected {kind.value}")
        return candidate

    def observe(self, candidate_id: str, observations: list[Observation]) -> None:
        self.require(candidate_id).observations.extend(observations)

    def reject(self, candidate_id: str) -> None:
        self.require(candidate_id).status = CandidateStatus.REJECTED

    def rollback(self, candidate_id: str) -> None:
        self.require(candidate_id).status = CandidateStatus.ROLLED_BACK

    def finalize(
        self, candidate_id: str, rationale: str, deliverable_path: str | None = None
    ) -> FinalSelection:
        candidate = self.require(candidate_id, CandidateKind.VIDEO)
        if candidate.status in {CandidateStatus.REJECTED, CandidateStatus.ROLLED_BACK}:
            raise ValueError(f"Cannot finalize {candidate.status.value} candidate {candidate_id}")
        for item in self.state.candidates.values():
            if item.status == CandidateStatus.FINALIZED:
                item.status = CandidateStatus.ACTIVE
        candidate.status = CandidateStatus.FINALIZED
        selection = FinalSelection(
            candidate_id=candidate_id,
            rationale=rationale,
            deliverable_path=deliverable_path,
        )
        self.state.final_selection = selection
        self.state.completed = True
        return selection
