"""Run Checklist and WER together from one video path."""

import argparse
import json
import os
import pathlib
import subprocess
import sys

from vtr_bench import data
from vtr_bench import io_utils


def video_ids(video_path: pathlib.Path) -> list[str]:
    """Check public-ID filenames before either evaluation process starts."""
    cases = data.load_cases()
    if video_path.is_file() and video_path.suffix.lower() == ".mp4":
        files = [video_path]
    elif video_path.is_dir():
        files = sorted(
            path
            for path in video_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        )
    else:
        raise FileNotFoundError("Supply an MP4 or a directory containing MP4s")
    identities = []
    for path in files:
        if path.stem not in cases:
            raise ValueError(f"Unknown benchmark video name: {path.name}")
        if path.stem in identities:
            raise ValueError(f"Duplicate video ID: {path.stem}")
        if video_path.is_dir():
            path.resolve().relative_to(video_path.resolve())
        identities.append(path.stem)
    if not identities:
        raise ValueError("No benchmark videos found")
    return identities


def run_evaluation(
    video_path: pathlib.Path, model: pathlib.Path, output: pathlib.Path
) -> dict:
    """Use isolated sequential processes and retain the frozen protocols."""
    adapter = pathlib.Path(__file__).parent / "evaluation/evaluate.py"
    reports = {}
    for task in ("checklist", "wer"):
        task_root = output / "details" / task
        command = [
            sys.executable,
            str(adapter),
            "--task",
            task,
            "--video-root",
            str(video_path),
            "--model-path",
            str(model),
            "--output-root",
            str(task_root),
            "--allow-missing",
        ]
        if (task_root / "run_manifest.json").is_file():
            command.append("--resume")
        task_root.parent.mkdir(parents=True, exist_ok=True)
        with (task_root.parent / f"{task}.log").open(
            "a", encoding="utf-8"
        ) as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT
            )
        reports[task] = json.loads(
            (task_root / "reports/metrics.json").read_text(encoding="utf-8")
        )
    io_utils.write_json(output / "metrics.json", reports)
    return reports


def main(argv: list[str] | None = None) -> int:
    """Evaluate available benchmark videos with one configured evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos", type=pathlib.Path, help="Video directory or MP4"
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Check inputs without inference"
    )
    args = parser.parse_args(argv)
    source = args.videos.expanduser().resolve(strict=True)
    identities = video_ids(source)
    model_value = os.environ.get("VTR_EVALUATOR_MODEL", "").strip()
    output = (args.output or pathlib.Path("results") / source.stem).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "videos": len(identities),
                    "benchmark_cases": 300,
                    "metrics": ["checklist", "wer"],
                    "evaluator_configured": bool(model_value),
                    "output": str(output / "metrics.json"),
                },
                indent=2,
            )
        )
        return 0
    if not model_value:
        parser.error(
            "Set VTR_EVALUATOR_MODEL to a local evaluator checkpoint once"
        )
    model = pathlib.Path(model_value).expanduser().resolve(strict=True)
    if not model.is_dir():
        parser.error("VTR_EVALUATOR_MODEL must be a checkpoint directory")
    reports = run_evaluation(source, model, output)
    print(
        json.dumps(
            {
                "checklist": reports["checklist"]["overall"],
                "wer": reports["wer"]["overall"],
                "output": str(output / "metrics.json"),
            },
            indent=2,
        )
    )
    return 0
