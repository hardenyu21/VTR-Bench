from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class CandidateStatus(str, Enum):
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FINALIZED = "finalized"


class Observation(StrictModel):
    category: str
    summary: str
    severity: int = Field(default=1, ge=0, le=3)
    evidence_locations: list[str] = Field(default_factory=list)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> int:
        if isinstance(value, str):
            normalized = value.strip().lower()
            labels = {
                "none": 0,
                "info": 0,
                "low": 1,
                "minor": 1,
                "medium": 2,
                "moderate": 2,
                "high": 3,
                "major": 3,
                "critical": 3,
            }
            if normalized in labels:
                return labels[normalized]
            try:
                return int(normalized)
            except ValueError as exc:
                raise ValueError(f"Unknown severity label: {value}") from exc
        return int(value)

    @field_validator("evidence_locations", mode="before")
    @classmethod
    def normalize_evidence_locations(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]


class Candidate(StrictModel):
    candidate_id: str
    kind: CandidateKind
    parent_id: str | None = None
    created_by: str
    artifact_path: str
    sha256: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    planner_diagnosis: dict[str, Any] = Field(default_factory=dict)
    status: CandidateStatus = CandidateStatus.ACTIVE
    created_at: str = Field(default_factory=timestamp)


class BudgetState(StrictModel):
    max_actions: int = Field(default=20, ge=1)
    max_api_calls: int = Field(default=30, ge=1)
    max_video_generations: int = Field(default=3, ge=1)
    actions_used: int = Field(default=0, ge=0)
    api_calls_used: int = Field(default=0, ge=0)
    video_generations_used: int = Field(default=0, ge=0)

    def remaining(self) -> dict[str, int]:
        return {
            "actions": max(0, self.max_actions - self.actions_used),
            "api_calls": max(0, self.max_api_calls - self.api_calls_used),
            "video_generations": max(0, self.max_video_generations - self.video_generations_used),
        }


class ToolCallRecord(StrictModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["running", "complete", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str = Field(default_factory=timestamp)
    finished_at: str | None = None


class FinalSelection(StrictModel):
    candidate_id: str
    rationale: str
    deliverable_path: str | None = None
    selected_at: str = Field(default_factory=timestamp)


class WorkflowState(StrictModel):
    schema_version: Literal[3] = 3
    case_id: str
    prompt_en: str
    prompt_sha256: str
    candidates: dict[str, Candidate] = Field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    final_selection: FinalSelection | None = None
    completed: bool = False
    rescue_used: bool = False
    rescue_reason: str | None = None
    created_at: str = Field(default_factory=timestamp)
    updated_at: str = Field(default_factory=timestamp)


class ToolResult(StrictModel):
    tool_name: str
    summary: str
    candidate_ids: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class InspectionReport(StrictModel):
    summary: str
    observations: list[Observation] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)


class ComparisonReport(InspectionReport):
    preferred_candidate_id: str | None = None


class GenerateKeyframesInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    generation_prompt: str = Field(min_length=1)
    negative_prompt: str = "blurred text, distorted letters, illegible print"
    count: int = Field(default=1, ge=1, le=4)
    vary: list[Literal["composition", "typography", "carrier_placement", "visual_treatment"]] = Field(
        default_factory=list
    )
    parent_id: str | None = None
    rollback_candidate_ids: list[str] = Field(default_factory=list)


class InspectKeyframesInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1, max_length=4)


class EditKeyframeInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_id: str
    edit_instruction: str = Field(min_length=1)
    region_description: str | None = None
    rollback_candidate_ids: list[str] = Field(default_factory=list)


class GenerateVideoInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    keyframe_id: str
    motion_refinement: str = Field(
        min_length=1,
        description=(
            "Delta-only camera, motion, timing, and temporal-stability guidance. Do not copy or "
            "rewrite the original request, scene description, actions, or required visible text. "
            "The runtime preserves the original prompt verbatim and appends this refinement."
        ),
    )
    parent_video_id: str | None = None
    rollback_candidate_ids: list[str] = Field(default_factory=list)


class InspectVideoInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_id: str
    sample_fps: float = Field(default=1.0, gt=0.0, le=2.0)


class ExtractFramesInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_id: str
    positions_seconds: list[float] = Field(min_length=1, max_length=12)

    @field_validator("positions_seconds")
    @classmethod
    def validate_positions(cls, values: list[float]) -> list[float]:
        if any(value < 0.0 or value > 10.0 for value in values):
            raise ValueError("positions must be within the 0-10 second deliverable")
        return values


class CompareCandidatesInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=2, max_length=6)
    focus: str = "visible text fidelity, carriers, motion, integrity, and regressions"


class FinalizeCandidateInput(StrictModel):
    diagnosis: str = Field(min_length=1)
    candidate_id: str
    rationale: str = Field(min_length=1)
    rollback_candidate_ids: list[str] = Field(default_factory=list)


TOOL_INPUT_MODELS: dict[str, type[StrictModel]] = {
    "generate_keyframes": GenerateKeyframesInput,
    "inspect_keyframes": InspectKeyframesInput,
    "edit_keyframe": EditKeyframeInput,
    "generate_video": GenerateVideoInput,
    "inspect_video": InspectVideoInput,
    "extract_frames": ExtractFramesInput,
    "compare_candidates": CompareCandidatesInput,
    "finalize_candidate": FinalizeCandidateInput,
}
