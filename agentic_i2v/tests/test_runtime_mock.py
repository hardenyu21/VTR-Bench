from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from agentic_i2v.runtime import WorkflowRuntime
from agentic_i2v.schemas import BudgetState, Candidate, ToolResult
from agentic_i2v.state import ArtifactStore, CandidateGraph, CheckpointStore, sha256_file


class FakeAgent:
    def decide(self, state):
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-final",
                    "name": "finalize_candidate",
                    "arguments": {
                        "diagnosis": "Existing video has sufficient evidence",
                        "candidate_id": "vid-0001",
                        "rationale": "Best historical candidate",
                        "rollback_candidate_ids": [],
                    },
                }
            ],
        }


class NoToolAgent:
    def decide(self, state):
        return {"content": "I am done", "tool_calls": []}


class FakeRegistry:
    def execute(self, context, name, arguments):
        context.graph.finalize(arguments["candidate_id"], arguments["rationale"])
        return ToolResult(tool_name=name, summary="finalized", candidate_ids=[arguments["candidate_id"]])


class RuntimeMockTests(unittest.TestCase):
    def test_bounded_loop_is_provider_mockable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory), "case")
            checkpoints = CheckpointStore(artifacts)
            state = checkpoints.create_or_load(
                "case", "original prompt", BudgetState(max_actions=2, max_api_calls=2, max_video_generations=1)
            )
            video = artifacts.videos / "vid-0001.mp4"
            video.write_bytes(b"test-only")
            graph = CandidateGraph(state)
            graph.add(
                Candidate(
                    candidate_id="vid-0001", kind="video", created_by="fixture",
                    artifact_path=artifacts.relative(video), sha256=sha256_file(video),
                )
            )
            checkpoints.save(state)
            context = SimpleNamespace(state=state, graph=graph, artifacts=artifacts)
            runtime = WorkflowRuntime(
                context=context, checkpoints=checkpoints, agent=FakeAgent(), registry=FakeRegistry()
            )
            result = runtime.run()
            self.assertEqual(result["candidate_id"], "vid-0001")
            self.assertTrue(state.completed)
            self.assertEqual(state.budget.actions_used, 1)
            self.assertEqual(state.budget.api_calls_used, 1)

    def test_no_tool_response_uses_deterministic_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory), "case")
            checkpoints = CheckpointStore(artifacts)
            state = checkpoints.create_or_load(
                "case", "original prompt", BudgetState(max_actions=2, max_api_calls=2, max_video_generations=1)
            )
            video = artifacts.videos / "vid-0001.mp4"
            video.write_bytes(b"test-only")
            graph = CandidateGraph(state)
            graph.add(
                Candidate(
                    candidate_id="vid-0001", kind="video", created_by="fixture",
                    artifact_path=artifacts.relative(video), sha256=sha256_file(video),
                )
            )
            checkpoints.save(state)
            context = SimpleNamespace(state=state, graph=graph, artifacts=artifacts)
            runtime = WorkflowRuntime(
                context=context, checkpoints=checkpoints, agent=NoToolAgent(), registry=FakeRegistry()
            )
            with patch("agentic_i2v.rescue.validate_deliverable", return_value={"frames": 240}):
                result = runtime.run()
            self.assertEqual(result["candidate_id"], "vid-0001")
            self.assertTrue(result["rescue_used"])
            self.assertEqual(state.final_selection.candidate_id, "vid-0001")


if __name__ == "__main__":
    unittest.main()
