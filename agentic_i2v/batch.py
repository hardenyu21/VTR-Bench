from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from vtr_bench import paths

from .media import validate_deliverable
from .state.artifact_store import ArtifactStore


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_case_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    case_ids = [str(row.get("global_id") or row.get("id") or "").strip() for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every prompt row must have global_id or id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Prompt IDs are not unique")
    return case_ids


def shard(items: list[str], index: int, count: int) -> list[str]:
    start = len(items) * index // count
    end = len(items) * (index + 1) // count
    return items[start:end]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one resumable Agentic I2V batch shard")
    parser.add_argument("--prompt-file", type=Path, default=paths.data_file("prompts.json"))
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be within worker-count")
    project = paths.project_root()
    runs_root = paths.agentic_runs_root(project)
    prompt_file = args.prompt_file.expanduser().resolve(strict=True)
    all_ids = load_case_ids(prompt_file)
    assigned = shard(all_ids, args.worker_index, args.worker_count)
    batch_root = runs_root / "batch"
    log_root = batch_root / f"worker_{args.worker_index}"
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = batch_root / f"worker_{args.worker_index}.json"
    completed: list[str] = []
    failed: dict[str, str] = {}
    env = os.environ.copy()
    env["VTEXTBENCH_PROMPT_FILE"] = str(prompt_file)
    env["VTEXTBENCH_H3_SERVICE_URL"] = args.endpoint.rstrip("/")

    def save(current: str | None, state: str) -> None:
        ArtifactStore.atomic_json(
            status_path,
            {
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "endpoint": args.endpoint,
                "assigned_count": len(assigned),
                "first_id": assigned[0] if assigned else None,
                "last_id": assigned[-1] if assigned else None,
                "completed_count": len(completed),
                "completed_ids": completed,
                "failed": failed,
                "current_id": current,
                "state": state,
                "updated_at": timestamp(),
            },
        )

    save(None, "starting")
    for case_id in assigned:
        deliverable = runs_root / "final_videos" / f"{case_id}.mp4"
        if deliverable.is_file():
            try:
                validate_deliverable(deliverable)
                completed.append(case_id)
                save(None, "running")
                continue
            except Exception:
                pass
        log_path = log_root / f"{case_id}.log"
        success = False
        for attempt in range(1, args.max_attempts + 1):
            save(case_id, f"attempt_{attempt}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{timestamp()}] attempt {attempt}\n")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m", "agentic_i2v",
                        "--case-id", case_id,
                        "--prompt-file", str(prompt_file),
                    ],
                    cwd=project,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if deliverable.is_file():
                try:
                    validate_deliverable(deliverable)
                    completed.append(case_id)
                    failed.pop(case_id, None)
                    success = True
                    break
                except Exception as exc:
                    failed[case_id] = f"invalid deliverable after attempt {attempt}: {exc}"
            else:
                failed[case_id] = f"attempt {attempt} exited {result.returncode}"
            save(case_id, "retry_wait")
            time.sleep(min(60, 10 * attempt))
        if not success:
            save(None, "running_with_failures")
        else:
            save(None, "running")
    save(None, "complete" if not failed else "complete_with_failures")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
