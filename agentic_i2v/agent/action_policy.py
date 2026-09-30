from __future__ import annotations

from ..schemas import BudgetState


API_COST = {
    "central_agent": 1,
    "generate_keyframes": 1,
    "inspect_keyframes": 1,
    "edit_keyframe": 1,
    "generate_video": 1,
    "inspect_video": 1,
    "extract_frames": 0,
    "compare_candidates": 1,
    "finalize_candidate": 0,
}


class BudgetExceeded(RuntimeError):
    pass


class ActionPolicy:
    def __init__(self, budget: BudgetState) -> None:
        self.budget = budget

    def charge_central_turn(self) -> None:
        self._ensure(api_calls=1)
        self.budget.api_calls_used += 1

    def charge_tool(self, tool_name: str) -> None:
        api_calls = API_COST[tool_name]
        videos = 1 if tool_name == "generate_video" else 0
        self._ensure(actions=1, api_calls=api_calls, videos=videos)
        self.budget.actions_used += 1
        self.budget.api_calls_used += api_calls
        self.budget.video_generations_used += videos

    def _ensure(self, *, actions: int = 0, api_calls: int = 0, videos: int = 0) -> None:
        if self.budget.actions_used + actions > self.budget.max_actions:
            raise BudgetExceeded("action budget exhausted")
        if self.budget.api_calls_used + api_calls > self.budget.max_api_calls:
            raise BudgetExceeded("API-call budget exhausted")
        if self.budget.video_generations_used + videos > self.budget.max_video_generations:
            raise BudgetExceeded("video-generation budget exhausted")

