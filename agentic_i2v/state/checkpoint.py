from __future__ import annotations

import hashlib

from ..schemas import BudgetState, WorkflowState, timestamp
from .artifact_store import ArtifactStore, sha256_file


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class CheckpointStore:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def create_or_load(self, case_id: str, prompt_en: str, budget: BudgetState) -> WorkflowState:
        if self.artifacts.state_path.is_file():
            state = WorkflowState.model_validate_json(self.artifacts.state_path.read_text(encoding="utf-8"))
            expected = prompt_digest(prompt_en)
            if state.case_id != case_id or state.prompt_sha256 != expected or state.prompt_en != prompt_en:
                raise RuntimeError("Checkpoint identity or immutable prompt does not match this request")
            self.validate_artifacts(state)
            return state
        state = WorkflowState(
            case_id=case_id,
            prompt_en=prompt_en,
            prompt_sha256=prompt_digest(prompt_en),
            budget=budget.model_copy(deep=True),
        )
        self.save(state)
        return state

    def save(self, state: WorkflowState) -> None:
        state.updated_at = timestamp()
        self.artifacts.atomic_json(self.artifacts.state_path, state.model_dump(mode="json"))

    def validate_artifacts(self, state: WorkflowState) -> None:
        for candidate in state.candidates.values():
            path = self.artifacts.resolve(candidate.artifact_path)
            if not path.is_file():
                raise RuntimeError(f"Missing checkpoint artifact: {candidate.artifact_path}")
            if sha256_file(path) != candidate.sha256:
                raise RuntimeError(f"Artifact checksum mismatch: {candidate.artifact_path}")

    def write_event(self, name: str, payload: dict) -> None:
        target = self.artifacts.reports / f"{name}.json"
        self.artifacts.atomic_json(target, payload)
