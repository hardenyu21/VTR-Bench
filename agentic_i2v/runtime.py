from __future__ import annotations

import json
import uuid
from typing import Any

from .agent import ActionPolicy, BudgetExceeded, CentralAgent
from .agent.context_builder import compact_context
from .schemas import ToolCallRecord, timestamp
from .rescue import RescueSelector, conservative_motion_refinement
from .state import CheckpointStore
from .tools import ToolContext, ToolRegistry


class WorkflowRuntime:
    def __init__(
        self,
        *,
        context: ToolContext,
        checkpoints: CheckpointStore,
        agent: CentralAgent,
        registry: ToolRegistry,
    ) -> None:
        self.context = context
        self.checkpoints = checkpoints
        self.agent = agent
        self.registry = registry
        self.policy = ActionPolicy(context.state.budget)

    def recover_interrupted_calls(self) -> None:
        changed = False
        for record in self.context.state.tool_calls:
            if record.status == "running":
                record.status = "failed"
                record.error = "Interrupted before a complete tool result was checkpointed"
                record.finished_at = timestamp()
                changed = True
        if changed:
            self.checkpoints.save(self.context.state)

    def run(self) -> dict[str, Any]:
        self.recover_interrupted_calls()
        while not self.context.state.completed:
            try:
                self.policy.charge_central_turn()
            except BudgetExceeded as exc:
                return self.rescue(str(exc))
            self.checkpoints.save(self.context.state)
            decision = self.agent.decide(self.context.state)
            calls = decision.get("tool_calls", [])
            if not calls:
                return self.rescue("Central agent returned no tool call")
            self.context.state.messages.append(
                {"role": "user", "content": compact_context(self.context.state)}
            )
            self.context.state.messages.append(
                {
                    "role": "assistant",
                    "content": decision.get("content", ""),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for call in calls
                    ],
                }
            )
            for call in calls:
                name = str(call["name"])
                arguments = dict(call.get("arguments", {}))
                try:
                    self.policy.charge_tool(name)
                except BudgetExceeded as exc:
                    return self.rescue(str(exc))
                record = ToolCallRecord(
                    call_id=str(call.get("id") or uuid.uuid4().hex),
                    tool_name=name,
                    arguments=arguments,
                    status="running",
                )
                self.context.state.tool_calls.append(record)
                self.checkpoints.save(self.context.state)
                try:
                    result = self.registry.execute(self.context, name, arguments)
                    record.status = "complete"
                    record.result = result.model_dump(mode="json")
                    self.context.state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": record.call_id,
                            "name": name,
                            "content": json.dumps(record.result, ensure_ascii=False),
                        }
                    )
                except Exception as exc:
                    record.status = "failed"
                    record.error = f"{type(exc).__name__}: {exc}"
                    self.context.state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": record.call_id,
                            "name": name,
                            "content": json.dumps({"error": record.error}, ensure_ascii=False),
                        }
                    )
                finally:
                    record.finished_at = timestamp()
                    self.checkpoints.save(self.context.state)
                if self.context.state.completed:
                    break
        return self._result()

    def _execute_rescue_tool(self, name: str, arguments: dict[str, Any]):
        record = ToolCallRecord(
            call_id=f"rescue-{uuid.uuid4().hex}",
            tool_name=name,
            arguments=arguments,
            status="running",
        )
        self.context.state.tool_calls.append(record)
        self.checkpoints.save(self.context.state)
        try:
            result = self.registry.execute(self.context, name, arguments)
            record.status = "complete"
            record.result = result.model_dump(mode="json")
            return result
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.finished_at = timestamp()
            self.checkpoints.save(self.context.state)

    def rescue(self, reason: str) -> dict[str, Any]:
        if self.context.state.completed:
            return self._result()
        self.context.state.rescue_used = True
        self.context.state.rescue_reason = reason
        self.checkpoints.save(self.context.state)
        selector = RescueSelector(self.context.state, self.context.artifacts)
        video_id = selector.best_video()
        if video_id is None:
            image_id = selector.best_image()
            if image_id is None:
                generated = self._execute_rescue_tool(
                    "generate_keyframes",
                    {
                        "diagnosis": f"Guaranteed-delivery rescue: {reason}",
                        "generation_prompt": self.context.state.prompt_en,
                        "negative_prompt": "blurred text, distorted letters, illegible print",
                        "count": 1,
                        "vary": [],
                        "parent_id": None,
                        "rollback_candidate_ids": [],
                    },
                )
                image_id = generated.candidate_ids[0]
            generated_video = self._execute_rescue_tool(
                "generate_video",
                {
                    "diagnosis": f"Guaranteed-delivery rescue: {reason}",
                    "keyframe_id": image_id,
                    "motion_refinement": conservative_motion_refinement(),
                    "parent_video_id": None,
                    "rollback_candidate_ids": [],
                },
            )
            video_id = generated_video.candidate_ids[0]
        rollback_ids = [
            candidate.candidate_id
            for candidate in self.context.state.candidates.values()
            if candidate.kind.value == "video"
            and candidate.status.value == "active"
            and candidate.candidate_id != video_id
        ]
        self._execute_rescue_tool(
            "finalize_candidate",
            {
                "diagnosis": f"Deterministic guaranteed-delivery finalization: {reason}",
                "candidate_id": video_id,
                "rationale": (
                    "Agentic exploration could not terminate within its guardrail. Selected the "
                    "best valid active historical video using explicit comparison preference when "
                    "available, otherwise the lowest observed-severity score."
                ),
                "rollback_candidate_ids": rollback_ids,
            },
        )
        return self._result()

    def _result(self) -> dict[str, Any]:
        selection = self.context.state.final_selection
        assert selection is not None
        candidate = self.context.graph.require(selection.candidate_id)
        return {
            "case_id": self.context.state.case_id,
            "candidate_id": selection.candidate_id,
            "artifact_path": selection.deliverable_path
            or str(self.context.artifacts.resolve(candidate.artifact_path)),
            "rationale": selection.rationale,
            "budget": self.context.state.budget.model_dump(mode="json"),
            "rescue_used": self.context.state.rescue_used,
            "rescue_reason": self.context.state.rescue_reason,
        }
