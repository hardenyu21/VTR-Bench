"""Test the two public entrypoints without model inference or paid APIs."""

import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from vtr_bench import agentic
from vtr_bench import cli
from vtr_bench import evaluate
from vtr_bench import io_utils


class EntryPointTests(unittest.TestCase):
    """Verify simple inputs and preserve the underlying workflow contracts."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        (self.videos / "AD-0001.mp4").write_bytes(b"mock video")

    def test_only_two_public_entrypoints(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            cli.main(["--help"])
        self.assertIn("{evaluate,agentic}", output.getvalue())
        with mock.patch.object(cli.evaluate, "main", return_value=0) as run:
            self.assertEqual(cli.main(["evaluate", "videos"]), 0)
        run.assert_called_once_with(["videos"])

    def test_video_folder_subset_and_single_file(self):
        self.assertEqual(evaluate.video_ids(self.videos), ["AD-0001"])
        self.assertEqual(
            evaluate.video_ids(self.videos / "AD-0001.mp4"), ["AD-0001"]
        )

    def test_unknown_and_duplicate_video_ids_rejected(self):
        original = self.videos / "AD-0001.mp4"
        variant = self.videos / "AD-0001.MP4"
        variant.write_bytes(b"duplicate")
        with (
            mock.patch.object(
                pathlib.Path, "iterdir", return_value=iter([original, variant])
            ),
            self.assertRaisesRegex(ValueError, "Duplicate"),
        ):
            evaluate.video_ids(self.videos)
        if not variant.samefile(original):
            variant.unlink()
        (self.videos / "unknown.mp4").write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            evaluate.video_ids(self.videos)

    def test_dry_run_requires_only_video_path(self):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(evaluate.main([str(self.videos), "--dry-run"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["videos"], 1)
        self.assertEqual(report["metrics"], ["checklist", "wer"])

    def test_both_metrics_run_and_matching_results_resume(self):
        output = self.root / "results"
        model = self.root / "model"
        model.mkdir()
        commands = []

        def fake_engine(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertIn("stdout", kwargs)
            task = command[command.index("--task") + 1]
            target = pathlib.Path(command[command.index("--output-root") + 1])
            commands.append(command)
            io_utils.write_json(target / "run_manifest.json", {})
            io_utils.write_json(
                target / "reports/metrics.json",
                {"task": task, "overall": {"mock": 1}},
            )

        with mock.patch.object(
            evaluate.subprocess, "run", side_effect=fake_engine
        ):
            evaluate.run_evaluation(self.videos, model, output)
            evaluate.run_evaluation(self.videos, model, output)
        self.assertEqual(
            [c[c.index("--task") + 1] for c in commands],
            ["checklist", "wer", "checklist", "wer"],
        )
        self.assertNotIn("--resume", commands[0])
        self.assertIn("--resume", commands[2])
        report = json.loads((output / "metrics.json").read_text())
        self.assertEqual(set(report), {"checklist", "wer"})

    def test_evaluation_stops_if_first_engine_fails(self):
        with (
            mock.patch.object(
                evaluate.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "mock engine"),
            ) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            evaluate.run_evaluation(self.videos, self.root, self.root / "out")
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / "out/metrics.json").exists())

    def test_single_case_agentic_uses_preserved_runner(self):
        with mock.patch.object(
            agentic.subprocess, "call", return_value=0
        ) as run:
            agentic.main(["--case-id", "AD-0001"])
        self.assertEqual(
            run.call_args.args[0][1:],
            ["-m", "agentic_i2v", "--case-id", "AD-0001"],
        )

    def test_batch_is_serial_single_worker(self):
        with mock.patch.object(
            agentic.subprocess, "call", return_value=0
        ) as run:
            agentic.main([])
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--worker-count") + 1], "1")

    def test_service_is_a_mode_of_agentic_not_an_extra_entrypoint(self):
        with mock.patch.object(
            agentic.subprocess, "call", return_value=0
        ) as run:
            agentic.main(["--serve"])
        self.assertEqual(
            run.call_args.args[0][1:], ["-m", "agentic_i2v.service.h3_server"]
        )


if __name__ == "__main__":
    unittest.main()
