#!/usr/bin/env python3
"""Evaluate video checklists with tagged JSON and bounded per-case retries."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "vtextbench_checklist_tagged_retry_v1_20260908"

SYSTEM_PROMPT = """TASK

You are a strict video checklist evaluator. Inspect the complete video and evaluate every checklist item independently using only directly visible evidence.

Answer yes only when the requested condition is clearly supported by the video; otherwise answer no.

OUTPUT

Return exactly one valid JSON array enclosed in <answer> and </answer>. Do not output anything else.

The array must contain exactly one entry for every checklist item, preserving the original order. Each entry must contain only id and answer. The answer must be either yes or no.

The user message includes an output example. Follow only its structure; never copy its yes/no values as answers for the current video."""


def output_example(count: int = 20) -> str:
    values = [
        {"id": f"C{index:02d}", "answer": "yes" if index % 2 else "no"}
        for index in range(1, count + 1)
    ]
    return f"<answer>{json.dumps(values, ensure_ascii=False, separators=(',', ':'))}</answer>"


RETRY_FEEDBACK = {
    "incomplete_answer_block": (
        "Your previous attempt failed because the required <answer>...</answer> block "
        "was incomplete, likely due to truncation or repetition. Please correct this "
        "in the current attempt."
    ),
    "invalid_json": (
        "Your previous attempt failed because the content inside "
        "<answer>...</answer> was not valid JSON. Please correct this in the current "
        "attempt."
    ),
    "invalid_answer_label": (
        "Your previous attempt failed because one or more answers were not yes or no. "
        "Please correct this in the current attempt."
    ),
    "other": "Your previous attempt failed. Please correct the output in the current attempt.",
}


def user_prompt(
    checklist: list[dict[str, Any]],
    previous_failure_feedback: str | None = None,
) -> str:
    checklist_text = "\n".join(
        f'{item["id"]}: {item["question_en"]}' for item in checklist
    )
    prompt = f"""CHECKLIST

{checklist_text}

OUTPUT EXAMPLE

{output_example(len(checklist))}

Based on the video and the checklist listed above, please output exactly one valid JSON array enclosed in <answer> and </answer>, following the format shown above. Do not output anything else."""
    if previous_failure_feedback:
        prompt += f"""

PREVIOUS ATTEMPT FEEDBACK

{previous_failure_feedback}"""
    return prompt


def parse_first_answer_block(
    raw: str,
) -> tuple[Any | None, list[str], dict[str, Any]]:
    matches = list(
        re.finditer(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL | re.IGNORECASE)
    )
    if not matches:
        return None, ["missing complete <answer>...</answer> block"], {
            "complete_answer_block_count": 0,
            "selected_answer_block_index": None,
            "nonempty_prefix_before_first_block": bool(raw.strip()),
            "nonempty_suffix_after_first_block": False,
            "additional_complete_answer_blocks": 0,
            "repair_attempted": False,
            "repair_applied": None,
        }

    first = matches[0]
    content = first.group(1).strip()
    diagnostics = {
        "complete_answer_block_count": len(matches),
        "selected_answer_block_index": 1,
        "nonempty_prefix_before_first_block": bool(raw[: first.start()].strip()),
        "nonempty_suffix_after_first_block": bool(raw[first.end() :].strip()),
        "additional_complete_answer_blocks": len(matches) - 1,
        "repair_attempted": False,
        "repair_applied": None,
    }
    try:
        return json.loads(content), [], diagnostics
    except json.JSONDecodeError as original_error:
        if content.startswith("[") and not content.endswith("]"):
            diagnostics["repair_attempted"] = True
            try:
                parsed = json.loads(content + "]")
            except json.JSONDecodeError as repair_error:
                return None, [
                    f"invalid JSON in first answer block after appending ]: {repair_error}"
                ], diagnostics
            diagnostics["repair_applied"] = "append_closing_square_bracket"
            return parsed, [], diagnostics
        return None, [f"invalid JSON in first answer block: {original_error}"], diagnostics


def validate_and_normalize(
    parsed: Any,
    expected_ids: list[str],
) -> tuple[list[str], list[dict[str, str]] | None]:
    if not isinstance(parsed, list):
        return ["answer JSON must be an array"], None
    errors: list[str] = []
    normalized: list[dict[str, str]] = []
    if len(parsed) != len(expected_ids):
        errors.append(
            f"answer item count mismatch: expected {len(expected_ids)}, got {len(parsed)}"
        )
    for index, expected_id in enumerate(expected_ids):
        if index >= len(parsed):
            errors.append(f"missing checklist item {expected_id}")
            continue
        item = parsed[index]
        if not isinstance(item, dict):
            errors.append(f"item {index + 1} must be an object")
            continue
        if set(item) != {"id", "answer"}:
            errors.append(f"item {index + 1} must contain only id and answer")
        item_id = item.get("id")
        if item_id != expected_id:
            errors.append(
                f"checklist ID/order mismatch at item {index + 1}: "
                f"expected {expected_id}, got {item_id!r}"
            )
        answer = item.get("answer")
        if not isinstance(answer, str):
            errors.append(f"{expected_id}: answer must be a string containing yes or no")
            continue
        normalized_answer = answer.strip().lower()
        if normalized_answer not in {"yes", "no"}:
            errors.append(f"{expected_id}: answer must normalize to yes or no")
            continue
        normalized.append({"id": expected_id, "answer": normalized_answer})
    if len(parsed) > len(expected_ids):
        errors.append(f"unexpected extra items after {expected_ids[-1]}")
    return errors, normalized if not errors else None


def classify_retry_reason(
    errors: list[str],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    if diagnostics.get("complete_answer_block_count") == 0:
        code = "incomplete_answer_block"
    elif any(error.startswith("invalid JSON") for error in errors):
        code = "invalid_json"
    elif any(
        "answer must normalize to yes or no" in error
        or "answer must be a string containing yes or no" in error
        for error in errors
    ):
        code = "invalid_answer_label"
    else:
        code = "other"
    return code, RETRY_FEEDBACK[code]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "unavailable"}
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {
        "status": "available",
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": frames / fps if fps else None,
    }
    capture.release()
    return result


def load_checklists(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list):
        raise ValueError("checklist suite must contain a cases array")
    by_id: dict[str, dict[str, Any]] = {}
    expected_ids = [f"C{index:02d}" for index in range(1, 21)]
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ValueError(f"invalid or duplicate case_id: {case_id!r}")
        checklist = case.get("checklist")
        if [item.get("id") for item in checklist or []] != expected_ids:
            raise ValueError(f"{case_id}: checklist must contain C01 through C20")
        by_id[case_id] = case
    if suite.get("case_count") != len(by_id):
        raise ValueError("top-level case_count does not match cases array")
    return suite, by_id


def parse_profile_video_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected PROFILE=VIDEO_ROOT")
    profile, raw_path = value.split("=", 1)
    if not profile or not raw_path:
        raise argparse.ArgumentTypeError("expected PROFILE=VIDEO_ROOT")
    return profile, Path(raw_path).expanduser().resolve()


def build_jobs(
    checklists: dict[str, dict[str, Any]],
    profile_roots: list[tuple[str, Path]],
    selected_jobs: list[str] | None,
    expected_jobs: int,
) -> list[dict[str, Any]]:
    selected: set[tuple[str, str]] | None = None
    if selected_jobs:
        selected = set()
        for value in selected_jobs:
            if ":" not in value:
                raise ValueError(f"invalid --job {value!r}; expected PROFILE:CASE_ID")
            selected.add(tuple(value.split(":", 1)))
    jobs: list[dict[str, Any]] = []
    for profile, root in profile_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for video in sorted(root.glob("*/seed42.mp4")):
            case_id = video.parent.name
            if selected is not None and (profile, case_id) not in selected:
                continue
            case = checklists.get(case_id)
            if case is None:
                raise RuntimeError(f"{profile}:{case_id}: no matching checklist")
            jobs.append({
                "profile": profile,
                "case_id": case_id,
                "video": video.resolve(),
                "case": case,
            })
    identities = [(job["profile"], job["case_id"]) for job in jobs]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate profile/case job")
    if selected is not None and set(identities) != selected:
        missing = sorted(selected - set(identities))
        raise RuntimeError(f"selected jobs not found: {missing}")
    if expected_jobs and len(jobs) != expected_jobs:
        raise RuntimeError(f"expected {expected_jobs} jobs, got {len(jobs)}")
    return jobs


def result_from_decisions(job: dict[str, Any], decisions: list[dict[str, str]]) -> dict[str, Any]:
    checklist_by_id = {item["id"]: item for item in job["case"]["checklist"]}
    items = []
    for decision in decisions:
        source = checklist_by_id[decision["id"]]
        items.append({
            "id": decision["id"],
            "category": source["category"],
            "question_en": source["question_en"],
            "answer": decision["answer"],
            "expected_answer": source["expected_answer"],
        })
    yes_count = sum(item["answer"] == "yes" for item in items)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "profile": job["profile"],
        "case_id": job["case_id"],
        "family": job["case"]["family"],
        "items": items,
        "summary": {
            "checklist_count": len(items),
            "yes_count": yes_count,
            "no_count": len(items) - yes_count,
            "yes_rate": yes_count / len(items),
        },
    }


def aggregate(output_root: Path, expected_jobs: int) -> dict[str, Any]:
    statuses = Counter()
    selected_attempts = Counter()
    retry_reasons = Counter()
    invalid_cases = []
    for path in output_root.glob("*/*/case_status.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value.get("status", "unknown")
        statuses[status] += 1
        if value.get("selected_attempt") is not None:
            selected_attempts[str(value["selected_attempt"])] += 1
        for attempt in value.get("attempts", []):
            reason = attempt.get("retry_reason_code")
            if reason:
                retry_reasons[reason] += 1
        if status != "pass":
            invalid_cases.append({
                "profile": path.parent.parent.name,
                "case_id": path.parent.name,
                "errors": value.get("errors", []),
            })
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_root.glob("*/*/result.json")
    ]
    by_profile: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    repaired_cases = 0
    for result in results:
        summary = result["summary"]
        for key in ("checklist_count", "yes_count", "no_count"):
            totals[key] += summary[key]
            by_profile[result["profile"]][key] += summary[key]
            by_family[result["family"]][key] += summary[key]
        by_profile[result["profile"]]["case_count"] += 1
        by_family[result["family"]]["case_count"] += 1
        repaired_cases += bool(result.get("repair_applied"))

    def finalize(groups: dict[str, Counter]) -> dict[str, Any]:
        return {
            key: {
                **dict(value),
                "yes_rate": value["yes_count"] / value["checklist_count"],
            }
            for key, value in sorted(groups.items())
        }

    attempted = sum(statuses.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "execution_status": "complete" if attempted == expected_jobs else "partial",
        "expected_case_count": expected_jobs,
        "attempted_case_count": attempted,
        "valid_case_count": len(results),
        "invalid_case_count": attempted - len(results),
        "case_status_counts": dict(statuses),
        "selected_attempt_counts": dict(selected_attempts),
        "retry_reason_counts": dict(retry_reasons),
        "repaired_json_case_count": repaired_cases,
        "checklist_count": totals["checklist_count"],
        "yes_count": totals["yes_count"],
        "no_count": totals["no_count"],
        "yes_rate": (
            totals["yes_count"] / totals["checklist_count"]
            if totals["checklist_count"]
            else None
        ),
        "by_profile": finalize(by_profile),
        "by_family": finalize(by_family),
        "invalid_cases": invalid_cases,
        "penalty_score_computed": False,
        "created_at": datetime.now().astimezone().isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checklists-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile-video-root",
        action="append",
        required=True,
        type=parse_profile_video_root,
    )
    parser.add_argument("--job", action="append", dest="selected_jobs")
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rounds != 3:
        raise ValueError("this protocol requires --max-rounds 3")
    if args.tensor_parallel_size != 1:
        raise ValueError("this protocol requires tensor_parallel_size=1")
    project_root = args.project_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    checklists_path = args.checklists_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    suite, checklists = load_checklists(checklists_path)
    jobs = build_jobs(
        checklists,
        args.profile_video_root,
        args.selected_jobs,
        args.expected_jobs,
    )
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    code_sha256 = sha256_file(Path(__file__).resolve())
    run_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "model": {
            "name": args.model_name,
            "path": str(model_path),
        },
        "checklists": {
            "path": str(checklists_path),
            "sha256": sha256_file(checklists_path),
            "suite": suite.get("suite"),
            "suite_case_count": suite.get("case_count"),
            "checklist_count_per_case": suite.get("checklist_count_per_case"),
        },
        "job_identities": [
            {"profile": job["profile"], "case_id": job["case_id"]} for job in jobs
        ],
        "prompt_protocol": {
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "user_prompt_template_sha256": hashlib.sha256(
                inspect.getsource(user_prompt).encode()
            ).hexdigest(),
            "categories_sent_to_model": False,
            "first_complete_answer_block_only": True,
            "json_repair": "append one closing ] only when array starts with [ and lacks ]",
            "retry_feedback_position": "exact end of user prompt",
            "retry_feedback_by_reason": RETRY_FEEDBACK,
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "frequency_penalty": args.frequency_penalty,
            "max_tokens": args.max_tokens,
            "seed": None,
            "stop": None,
        },
        "execution": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "batch_size": 1,
            "enable_thinking": False,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_rounds": args.max_rounds,
            "video_fps": args.video_fps,
            "penalty_score_computed": False,
        },
        "code_sha256": code_sha256,
    }
    fingerprint = sha256_json(run_identity)
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_fingerprint") != fingerprint:
            raise RuntimeError("run manifest fingerprint mismatch; use a new output root")
        if not args.resume:
            raise RuntimeError("output root already exists; pass --resume")
    elif output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("nonempty output root has no matching run manifest")
    else:
        write_json(manifest_path, {
            **run_identity,
            "run_fingerprint": fingerprint,
            "command": [sys.executable, *sys.argv],
            "created_at": datetime.now().astimezone().isoformat(),
        })
        write_text(output_root / "frozen_prompts" / "system_prompt.txt", SYSTEM_PROMPT)
        write_text(
            output_root / "frozen_prompts" / "user_prompt_example.txt",
            user_prompt(jobs[0]["case"]["checklist"]),
        )

    os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    import torch
    from importlib.metadata import version
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        allowed_local_media_path=str(project_root),
        limit_mm_per_prompt={"video": 1},
        media_io_kwargs={"video": {"fps": args.video_fps}},
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=args.max_tokens,
    )
    runtime = {
        "python": sys.version,
        "vllm": version("vllm"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }

    for index, job in enumerate(jobs, start=1):
        case_root = output_root / job["profile"] / job["case_id"]
        status_path = case_root / "case_status.json"
        if args.resume and status_path.is_file():
            saved = json.loads(status_path.read_text(encoding="utf-8"))
            if saved.get("status") in {"pass", "invalid"}:
                print(json.dumps({
                    "index": index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "skipped_terminal",
                }), flush=True)
                continue
        elif case_root.exists() and any(case_root.iterdir()):
            raise RuntimeError(
                f"{job['profile']}:{job['case_id']}: partial data requires --resume"
            )

        checklist = job["case"]["checklist"]
        expected_ids = [item["id"] for item in checklist]
        base_user = user_prompt(checklist)
        input_record = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": args.run_id,
            "run_fingerprint": fingerprint,
            "profile": job["profile"],
            "case_id": job["case_id"],
            "family": job["case"]["family"],
            "video_path": str(job["video"]),
            "video_sha256": sha256_file(job["video"]),
            "video_metadata": video_metadata(job["video"]),
            "checklists_json": str(checklists_path),
            "checklists_json_sha256": sha256_file(checklists_path),
            "checklist_case_sha256": sha256_json(job["case"]),
            "system_prompt": SYSTEM_PROMPT,
            "initial_user_prompt": base_user,
            "model_visible_checklist": [
                {"id": item["id"], "question_en": item["question_en"]}
                for item in checklist
            ],
            "categories_visible_to_model": False,
            "runtime": runtime,
        }
        write_json(case_root / "input.json", input_record)

        completed_attempts: list[dict[str, Any]] = []
        previous_feedback: str | None = None
        for prior_number in range(1, args.max_rounds + 1):
            prior_path = case_root / "attempts" / f"round_{prior_number}" / "response.json"
            if not prior_path.is_file():
                break
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if prior.get("status") == "pass":
                raise RuntimeError(
                    f"{job['profile']}:{job['case_id']}: passed attempt lacks terminal status"
                )
            previous_feedback = prior.get("retry_feedback")
            completed_attempts.append({
                "round": prior_number,
                "status": prior.get("status"),
                "retry_reason_code": prior.get("retry_reason_code"),
            })
        start_round = len(completed_attempts) + 1
        if start_round > args.max_rounds:
            raise RuntimeError(
                f"{job['profile']}:{job['case_id']}: exhausted rounds lack terminal status"
            )

        for round_number in range(start_round, args.max_rounds + 1):
            user = user_prompt(checklist, previous_feedback)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": job["video"].as_uri()}},
                    {"type": "text", "text": user},
                ]},
            ]
            started = time.time()
            request_output = llm.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )[0]
            completion = request_output.outputs[0]
            raw = completion.text
            parsed, parse_errors, diagnostics = parse_first_answer_block(raw)
            validation_errors, normalized = validate_and_normalize(parsed, expected_ids)
            errors = [*parse_errors, *validation_errors]
            retry_reason_code = retry_feedback = None
            if errors:
                retry_reason_code, retry_feedback = classify_retry_reason(
                    errors, diagnostics
                )
            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            attempt_input = {
                **input_record,
                "round": round_number,
                "previous_failure_feedback": previous_feedback,
                "user_prompt": user,
                "input_token_count": (
                    len(prompt_token_ids) if prompt_token_ids is not None else None
                ),
                "sampling_params_repr": repr(sampling),
            }
            response_record = {
                "round": round_number,
                "status": "pass" if not errors else "invalid_response",
                "parsed_answer": parsed,
                "normalized_answer": normalized,
                "validation_errors": errors,
                "answer_block_diagnostics": diagnostics,
                "retry_reason_code": retry_reason_code,
                "retry_feedback": retry_feedback,
                "output_token_count": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": getattr(completion, "stop_reason", None),
                "wall_seconds": round(time.time() - started, 3),
                "vllm_version": runtime["vllm"],
            }
            attempt_root = case_root / "attempts" / f"round_{round_number}"
            write_json(attempt_root / "input.json", attempt_input)
            write_text(attempt_root / "response_raw.txt", raw)
            write_json(attempt_root / "response.json", response_record)
            write_text(case_root / "response_raw.txt", raw)
            write_json(case_root / "response.json", response_record)
            completed_attempts.append({
                "round": round_number,
                "status": response_record["status"],
                "retry_reason_code": retry_reason_code,
            })

            if not errors and normalized is not None:
                result = result_from_decisions(job, normalized)
                result["selected_round"] = round_number
                result["repair_applied"] = diagnostics.get("repair_applied")
                write_json(case_root / "result.json", result)
                write_json(status_path, {
                    "status": "pass",
                    "rounds_completed": round_number,
                    "selected_round": round_number,
                    "repair_applied": diagnostics.get("repair_applied"),
                    "attempts": completed_attempts,
                })
                print(json.dumps({
                    "index": index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "pass",
                    "round": round_number,
                    "yes_count": result["summary"]["yes_count"],
                    "repair_applied": diagnostics.get("repair_applied"),
                }, ensure_ascii=False), flush=True)
                break

            if round_number < args.max_rounds:
                previous_feedback = retry_feedback
                write_json(status_path, {
                    "status": "retrying",
                    "rounds_completed": round_number,
                    "next_round": round_number + 1,
                    "last_errors": errors,
                    "last_retry_reason_code": retry_reason_code,
                    "attempts": completed_attempts,
                })
                print(json.dumps({
                    "index": index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "retrying",
                    "round": round_number,
                    "retry_reason_code": retry_reason_code,
                }, ensure_ascii=False), flush=True)
                continue

            write_json(status_path, {
                "status": "invalid",
                "rounds_completed": round_number,
                "errors": errors,
                "last_retry_reason_code": retry_reason_code,
                "attempts": completed_attempts,
            })
            print(json.dumps({
                "index": index,
                "profile": job["profile"],
                "case_id": job["case_id"],
                "status": "invalid",
                "round": round_number,
                "retry_reason_code": retry_reason_code,
                "errors": errors,
            }, ensure_ascii=False), flush=True)

        write_json(output_root / "progress.json", aggregate(output_root, len(jobs)))

    summary = aggregate(output_root, len(jobs))
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
