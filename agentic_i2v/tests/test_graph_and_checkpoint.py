from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_i2v.schemas import BudgetState, Candidate, CandidateKind, WorkflowState
from agentic_i2v.state import ArtifactStore, CandidateGraph, CheckpointStore, sha256_file


class GraphAndCheckpointTests(unittest.TestCase):
    def test_graph_branches_and_finalizes_only_video(self) -> None:
        state = WorkflowState(case_id="x", prompt_en="p", prompt_sha256="h")
        graph = CandidateGraph(state)
        graph.add(Candidate(candidate_id="img-0001", kind="image", created_by="test", artifact_path="a", sha256="x"))
        graph.add(Candidate(candidate_id="vid-0001", kind="video", parent_id="img-0001", created_by="test", artifact_path="b", sha256="y"))
        self.assertEqual(graph.next_id(CandidateKind.IMAGE), "img-0002")
        with self.assertRaises(ValueError):
            graph.finalize("img-0001", "not a video")
        graph.finalize("vid-0001", "best evidence")
        self.assertTrue(state.completed)
        self.assertEqual(state.final_selection.candidate_id, "vid-0001")
        with self.assertRaises(ValueError):
            graph.add(state.candidates["img-0001"])

    def test_checkpoint_rejects_prompt_change_and_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory), "case-1")
            checkpoints = CheckpointStore(artifacts)
            state = checkpoints.create_or_load("case-1", "original", BudgetState())
            artifact = artifacts.images / "img-0001.png"
            artifact.write_bytes(b"first")
            state.candidates["img-0001"] = Candidate(
                candidate_id="img-0001", kind="image", created_by="test",
                artifact_path=artifacts.relative(artifact), sha256=sha256_file(artifact),
            )
            checkpoints.save(state)
            loaded = checkpoints.create_or_load("case-1", "original", BudgetState())
            self.assertEqual(loaded.prompt_en, "original")
            with self.assertRaises(RuntimeError):
                checkpoints.create_or_load("case-1", "changed", BudgetState())
            artifact.write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                checkpoints.create_or_load("case-1", "original", BudgetState())

    def test_final_video_uses_global_id_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory), "AD-0001")
            source = artifacts.videos / "vid-0001.mp4"
            source.write_bytes(b"video-one")
            target = artifacts.publish_final_video(source, "AD-0001")
            self.assertEqual(target.name, "AD-0001.mp4")
            self.assertEqual(target.read_bytes(), b"video-one")
            self.assertEqual(artifacts.publish_final_video(source, "AD-0001"), target)
            other = artifacts.videos / "vid-0002.mp4"
            other.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                artifacts.publish_final_video(other, "AD-0001")


if __name__ == "__main__":
    unittest.main()
