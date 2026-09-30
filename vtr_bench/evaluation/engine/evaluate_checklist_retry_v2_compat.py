#!/usr/bin/env python3
"""Run the frozen checklist evaluator with approved compatibility overrides."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).with_name("evaluate_checklist_retry_v2.py")
SPEC = importlib.util.spec_from_file_location("evaluate_checklist_retry_v2_frozen", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ORIGINAL_VALIDATE = MODULE.validate_and_normalize


def validate_and_normalize_with_padded_ids(
    parsed: Any,
    expected_ids: list[str],
) -> tuple[list[str], list[dict[str, str]] | None]:
    if not isinstance(parsed, list):
        return ORIGINAL_VALIDATE(parsed, expected_ids)
    remapped: list[Any] = []
    for index, item in enumerate(parsed):
        mapped = item
        if index < len(expected_ids) and isinstance(item, dict):
            item_id = item.get("id")
            expected_id = expected_ids[index]
            if (
                isinstance(item_id, str)
                and re.fullmatch(r"C0(?:1[0-9]|20)", item_id)
                and item_id[2:] == expected_id[1:]
            ):
                mapped = {**item, "id": expected_id}
        remapped.append(mapped)
    return ORIGINAL_VALIDATE(remapped, expected_ids)


MODULE.validate_and_normalize = validate_and_normalize_with_padded_ids


def argument_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def prior_truncation_streak(case_root: Path) -> int:
    responses = sorted(
        (case_root / "attempts").glob("round_*/response.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    streak = 0
    for path in reversed(responses):
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        if response.get("finish_reason") != "length":
            break
        streak += 1
    return streak


def video_key(messages: Any) -> str | None:
    try:
        content = messages[-1]["content"]
        return next(
            item["video_url"]["url"]
            for item in content
            if item.get("type") == "video_url"
        )
    except (KeyError, StopIteration, TypeError):
        return None


def case_root_for_messages(messages: Any, output_root: Path) -> Path | None:
    """Resolve direct case-layout videos; in-memory streaks cover symlink targets."""

    try:
        video = Path(str(video_key(messages)).removeprefix("file://"))
        candidate = output_root / video.parent.parent.name / video.parent.name
    except (AttributeError, TypeError):
        return None
    return candidate if candidate.parent.parent == output_root else None


def adaptive_max_tokens(base: int, truncation_streak: int) -> int:
    # The approved checklist policy is 1024 -> 2048 -> 4096. A follow-up
    # repair may start at 2048, but must still never exceed 4096.
    return min(base * (2**truncation_streak), 4096)


def install_adaptive_chat() -> None:
    output_root_value = argument_value("--output-root")
    if output_root_value is None:
        raise RuntimeError("--output-root is required for adaptive token budgeting")
    output_root = Path(output_root_value).expanduser().resolve()
    os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    from vllm import LLM

    original_chat = LLM.chat
    in_memory_streaks: dict[str, int] = {}

    def adaptive_chat(self: Any, messages: Any, *args: Any, **kwargs: Any) -> Any:
        sampling = kwargs.get("sampling_params")
        if sampling is None and args:
            sampling = args[0]
        key = video_key(messages)
        base = int(argument_value("--max-tokens") or 1024)
        persisted = 0
        case_root = case_root_for_messages(messages, output_root)
        if case_root is not None:
            persisted = prior_truncation_streak(case_root)
        streak = max(persisted, in_memory_streaks.get(key or "", 0))
        if sampling is not None:
            sampling.max_tokens = adaptive_max_tokens(base, streak)

        result = original_chat(self, messages, *args, **kwargs)
        try:
            finish_reason = result[0].outputs[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            finish_reason = None
        if key:
            in_memory_streaks[key] = streak + 1 if finish_reason == "length" else 0
        return result

    LLM.chat = adaptive_chat


if __name__ == "__main__":
    install_adaptive_chat()
    raise SystemExit(MODULE.main())
