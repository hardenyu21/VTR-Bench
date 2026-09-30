from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vtr_bench import paths

from .agent import CentralAgent
from .agent.prompts import TOOL_PROMPTS
from .clients import BailianChatClient, MiniMaxH3Client, QwenImageClient
from .config import RuntimeConfig
from .runtime import WorkflowRuntime
from .schemas import BudgetState
from .state import ArtifactStore, CandidateGraph, CheckpointStore
from .tools import ToolContext, ToolRegistry


def load_case(path: Path, case_id: str) -> tuple[str, str]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    for row in rows:
        identity = str(row.get("global_id") or row.get("id"))
        if identity == case_id:
            return identity, str(row["prompt_en"])
    raise KeyError(f"Unknown case ID {case_id!r} in {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded Agentic I2V v3 prompt-preserving workflow"
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--max-api-calls", type=int, default=30)
    parser.add_argument("--max-video-generations", type=int, default=3)
    parser.add_argument("--check", action="store_true", help="Validate configuration and H3 health only")
    parser.add_argument(
        "--rescue",
        action="store_true",
        help="Deterministically finalize or generate a valid deliverable outside the exploration budget",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = paths.project_root()
    budget = BudgetState(
        max_actions=args.max_actions,
        max_api_calls=args.max_api_calls,
        max_video_generations=args.max_video_generations,
    )
    config = RuntimeConfig.from_env(root, prompt_file=args.prompt_file, budget=budget)
    case_id, prompt = load_case(config.prompt_file, args.case_id)
    h3 = MiniMaxH3Client(config.h3_service_url)
    health = h3.health()
    if args.check:
        print(json.dumps({"case_id": case_id, "prompt_sha256_checked": True, "h3": health}, ensure_ascii=False, indent=2))
        return 0
    artifacts = ArtifactStore(config.runs_root, case_id)
    checkpoints = CheckpointStore(artifacts)
    state = checkpoints.create_or_load(case_id, prompt, config.budget)
    graph = CandidateGraph(state)
    chat = BailianChatClient(config.bailian_api_key, config.bailian_base_url, config.qwen_chat_model)
    images = QwenImageClient(
        config.bailian_api_key,
        config.bailian_base_url,
        config.qwen_image_model,
        width=config.image_width,
        height=config.image_height,
    )
    context = ToolContext(
        state=state,
        graph=graph,
        artifacts=artifacts,
        checkpoints=checkpoints,
        chat=chat,
        images=images,
        h3=h3,
        prompts=TOOL_PROMPTS,
    )
    runtime = WorkflowRuntime(
        context=context,
        checkpoints=checkpoints,
        agent=CentralAgent(chat),
        registry=ToolRegistry(),
    )
    result = runtime.rescue("Explicit operator rescue") if args.rescue else runtime.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
