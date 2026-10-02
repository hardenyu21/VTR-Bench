from __future__ import annotations

from .media import validate_deliverable
from .schemas import Candidate, CandidateKind, CandidateStatus, WorkflowState
from .state import ArtifactStore


class RescueSelector:
    """Deterministic selection used only when agentic exploration cannot terminate."""

    def __init__(self, state: WorkflowState, artifacts: ArtifactStore) -> None:
        self.state = state
        self.artifacts = artifacts

    @staticmethod
    def _observation_score(candidate: Candidate) -> tuple[float, int, int, str]:
        severities = [item.severity for item in candidate.observations]
        if not severities:
            return (10.0, 3, 0, candidate.candidate_id)
        return (
            sum(severities) / len(severities),
            max(severities),
            -len(severities),
            candidate.candidate_id,
        )

    def _latest_preference(self, kind: CandidateKind, eligible: set[str]) -> str | None:
        for record in reversed(self.state.tool_calls):
            if record.status != "complete" or not record.result:
                continue
            payload = record.result.get("payload", {})
            preferred = payload.get("preferred_candidate_id")
            if preferred in eligible and self.state.candidates[preferred].kind == kind:
                return preferred
            for candidate_id in payload.get("ranking", []):
                if candidate_id in eligible and self.state.candidates[candidate_id].kind == kind:
                    return candidate_id
        return None

    def best_video(self) -> str | None:
        eligible: list[Candidate] = []
        for candidate in self.state.candidates.values():
            if candidate.kind != CandidateKind.VIDEO or candidate.status != CandidateStatus.ACTIVE:
                continue
            path = self.artifacts.resolve(candidate.artifact_path)
            try:
                recorded_media = candidate.inputs.get("service_result", {}).get(
                    "validated_media", {}
                )
                validate_deliverable(
                    path,
                    frames=int(recorded_media.get("frames", 240)),
                    fps=int(recorded_media.get("fps", 24)),
                )
            except Exception:
                continue
            eligible.append(candidate)
        ids = {candidate.candidate_id for candidate in eligible}
        preferred = self._latest_preference(CandidateKind.VIDEO, ids)
        if preferred:
            return preferred
        if not eligible:
            return None
        return min(eligible, key=self._observation_score).candidate_id

    def best_image(self) -> str | None:
        eligible = [
            candidate
            for candidate in self.state.candidates.values()
            if candidate.kind == CandidateKind.IMAGE
            and candidate.status == CandidateStatus.ACTIVE
            and self.artifacts.resolve(candidate.artifact_path).is_file()
        ]
        ids = {candidate.candidate_id for candidate in eligible}
        preferred = self._latest_preference(CandidateKind.IMAGE, ids)
        if preferred:
            return preferred
        if not eligible:
            return None
        return min(eligible, key=self._observation_score).candidate_id


def conservative_motion_refinement() -> str:
    return (
        "Animate the depicted scene with restrained, physically coherent motion. Preserve every "
        "visible text string, typography, text-bearing surface, object identity, layout, and camera "
        "framing from the conditioning image. Keep printed surfaces readable and stable; avoid "
        "warping, occlusion, cropping, rotation away from camera, or newly invented text."
    )
