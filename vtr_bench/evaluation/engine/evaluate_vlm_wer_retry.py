#!/usr/bin/env python3
"""Tagged-JSON VLM transcription with bounded per-case retries and WER scoring."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate_vlm_wer as common


PROTOCOL_VERSION = "vtextbench_vlm_wer_tagged_retry_v5_20260907"

RETRY_FEEDBACK = {
    "unclosed_answer_repetition": (
        "Your previous attempt failed because the <answer>...</answer> block was not "
        "closed and the output entered a repetition loop. Please correct this in the "
        "current attempt."
    ),
    "transcription_too_long": (
        "Your previous attempt failed because a single transcription text value was "
        "too long and likely hallucinated. Please correct this in the current attempt."
    ),
    "field_conflict": (
        "Your previous attempt failed because the output fields conflicted with the "
        "required constraints. Please correct this in the current attempt."
    ),
    "other": "Your previous attempt failed. Please correct the output in the current attempt.",
}

SYSTEM_PROMPT = """TASK

You are a strict visual-text transcription evaluator. Inspect the video and transcribe only the text on the specified target carriers.

RULES

1. Inspect every target independently across the video.
2. First locate the specified carrier, then inspect only the text on that carrier. Do not substitute another carrier or use background text.
3. Transcribe only characters that can be reliably identified from the visible glyphs. Preserve clearly visible text even if it is misspelled, ungrammatical, or meaningless. Do not correct, translate, paraphrase, infer, or complete the content.
4. Omit portions that cannot be reliably transcribed. Do not guess missing characters or insert <UNK>, ellipses, brackets, or other placeholders for unreadable content. Preserve punctuation that is actually visible.
5. Keep readable portions in their original order and preserve visible line breaks where possible. When omitting an unreadable portion, do not concatenate the surrounding fragments into a new word; separate them with whitespace.
6. If the same carrier appears multiple times, use the clearest occurrence that is directly supported by the video.
7. Each target contains an id, carrier, and language. Return the id exactly as given. Use carrier only to locate the text region. Use language only to identify the writing system, never to infer missing content.

OUTPUT

Return exactly one valid JSON object enclosed in <answer> and </answer>. Do not output anything before <answer> or after </answer>.

The JSON object must be keyed by target ID, with exactly one entry for every target. Each value must contain only status, text, and readability_note. Keep each text value at or below 640 characters.

- Use carrier_not_found when the specified carrier cannot be found in the video. Set text and readability_note to empty strings.
- Use text_unreadable when the carrier exists but none of its target text can be reliably transcribed. Set text to an empty string and provide a readability_note.
- Use transcribed when at least some characters can be reliably transcribed. Set text to the readable content only. If any portions were omitted because they were unreadable, provide a readability_note; otherwise set readability_note to an empty string.
- For each nonempty readability_note: Briefly describe the observable reading difficulties for the target as a whole in one or two sentences. Do not enumerate individual words or unreadable spans, and do not speculate about underlying causes. If the difficulty cannot be determined, state that explicitly.
- The user message includes an output example. Follow only its structure; never copy its statuses, text, or readability_note as answers for the current video."""

OUTPUT_EXAMPLE = """<answer>
{
  "T01": {
    "status": "transcribed",
    "text": "VISIBLE TEXT",
    "readability_note": ""
  },
  "T02": {
    "status": "transcribed",
    "text": "READABLE PORTION",
    "readability_note": "Only part of the text can be reliably transcribed; the remaining characters cannot be distinguished."
  },
  "T03": {
    "status": "text_unreadable",
    "text": "",
    "readability_note": "The text is visible, but individual characters cannot be distinguished."
  },
  "T04": {
    "status": "carrier_not_found",
    "text": "",
    "readability_note": ""
  }
}
</answer>"""


def user_prompt(
    masked_scene_description: str,
    public_targets: list[dict[str, str]],
    previous_failure_feedback: str | None = None,
) -> str:
    prompt = f"""SCENE DESCRIPTION

{masked_scene_description}

TARGET CARRIERS

{json.dumps(public_targets, ensure_ascii=False, indent=2)}

OUTPUT EXAMPLE

{OUTPUT_EXAMPLE}

Based on the video and the information listed above, please output exactly one valid JSON object enclosed in <answer> and </answer>, following the format shown above. Do not output anything before <answer> or after </answer>."""
    if previous_failure_feedback:
        prompt += f"""

PREVIOUS ATTEMPT FEEDBACK

{previous_failure_feedback}"""
    return prompt


def parse_first_answer_block(
    raw: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    matches = list(
        re.finditer(
            r"<answer>\s*(.*?)\s*</answer>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if not matches:
        return None, ["missing complete <answer>...</answer> block"], {
            "complete_answer_block_count": 0,
            "selected_answer_block_index": None,
            "nonempty_prefix_before_first_block": bool(raw.strip()),
            "nonempty_suffix_after_first_block": False,
            "additional_complete_answer_blocks": 0,
        }

    first = matches[0]
    diagnostics = {
        "complete_answer_block_count": len(matches),
        "selected_answer_block_index": 1,
        "nonempty_prefix_before_first_block": bool(raw[: first.start()].strip()),
        "nonempty_suffix_after_first_block": bool(raw[first.end() :].strip()),
        "additional_complete_answer_blocks": len(matches) - 1,
    }
    try:
        parsed = json.loads(first.group(1))
    except json.JSONDecodeError as error:
        return None, [f"invalid JSON in first answer block: {error}"], diagnostics
    if not isinstance(parsed, dict):
        return None, ["first answer block must contain a JSON object"], diagnostics
    return parsed, [], diagnostics


def output_schema(target_ids: list[str]) -> dict[str, Any]:
    def value_schema(status: str) -> dict[str, Any]:
        transcribed = status == "transcribed"
        unreadable = status == "text_unreadable"
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": [status]},
                "text": {
                    "type": "string",
                    "minLength": 1 if transcribed else 0,
                    "maxLength": common.MAX_TRANSCRIPTION_CHARS if transcribed else 0,
                },
                "readability_note": {
                    "type": "string",
                    "minLength": 1 if unreadable else 0,
                    **({"maxLength": 0} if status == "carrier_not_found" else {}),
                },
            },
            "required": ["status", "text", "readability_note"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            text_id: {
                "oneOf": [value_schema(status) for status in sorted(common.STATUSES)]
            }
            for text_id in target_ids
        },
        "required": target_ids,
        "additionalProperties": False,
    }


def validate_answer(
    parsed: dict[str, Any] | None,
    target_ids: list[str],
) -> list[str]:
    if not isinstance(parsed, dict):
        return ["parsed answer missing"]
    errors: list[str] = []
    if list(parsed.keys()) != target_ids:
        errors.append(f"target IDs/order mismatch: {list(parsed.keys())}")
    expected_fields = {"status", "text", "readability_note"}
    for text_id in target_ids:
        value = parsed.get(text_id)
        if not isinstance(value, dict):
            errors.append(f"{text_id}: value must be object")
            continue
        if set(value) != expected_fields:
            errors.append(f"{text_id}: fields must be {sorted(expected_fields)} only")
            continue
        status = value.get("status")
        text = value.get("text")
        note = value.get("readability_note")
        if status not in common.STATUSES:
            errors.append(f"{text_id}: invalid status {status!r}")
        if not isinstance(text, str):
            errors.append(f"{text_id}: text must be string")
        elif len(text) > common.MAX_TRANSCRIPTION_CHARS:
            errors.append(
                f"{text_id}: text exceeds {common.MAX_TRANSCRIPTION_CHARS} characters"
            )
        elif "<UNK>" in text:
            errors.append(f"{text_id}: text must not contain <UNK>")
        elif status == "transcribed" and not text.strip():
            errors.append(f"{text_id}: transcribed text must be nonempty")
        elif status in {"carrier_not_found", "text_unreadable"} and text != "":
            errors.append(f"{text_id}: non-transcribed text must be empty")
        if not isinstance(note, str):
            errors.append(f"{text_id}: readability_note must be a string")
        elif status == "text_unreadable" and not note.strip():
            errors.append(f"{text_id}: text_unreadable requires readability_note")
        elif status == "carrier_not_found" and note != "":
            errors.append(f"{text_id}: carrier_not_found requires empty readability_note")
    return errors


def answer_for_terminal_scoring(
    parsed: dict[str, Any],
    target_ids: list[str],
) -> dict[str, dict[str, str]]:
    """Build a conservative scoreable answer from a parsed terminal JSON object.

    Strict validation still controls retries.  After the final allowed attempt, a
    successfully parsed JSON object is scoreable even when it violates the output
    schema.  Missing or unusable target values become empty transcriptions; any
    string text that is present is retained for WER scoring.
    """

    answer: dict[str, dict[str, str]] = {}
    for text_id in target_ids:
        value = parsed.get(text_id)
        if not isinstance(value, dict):
            answer[text_id] = {
                "status": "text_unreadable",
                "text": "",
                "readability_note": "",
            }
            continue
        text = value.get("text")
        text = text if isinstance(text, str) else ""
        status = value.get("status")
        if text:
            status = "transcribed"
        elif status not in common.STATUSES:
            status = "text_unreadable"
        note = value.get("readability_note")
        answer[text_id] = {
            "status": status,
            "text": text if status == "transcribed" else "",
            "readability_note": note if isinstance(note, str) else "",
        }
    return answer


def classify_retry_reason(
    raw: str,
    errors: list[str],
    answer_block_diagnostics: dict[str, Any],
    repetition_diagnostics: dict[str, Any],
    finish_reason: str | None,
) -> tuple[str, str]:
    """Map a failed attempt to one of the four user-approved machine-readable reasons."""

    lowered = raw.lower()
    unclosed_answer = (
        answer_block_diagnostics.get("complete_answer_block_count") == 0
        and "<answer>" in lowered
        and "</answer>" not in lowered
    )
    raw_repetition = (
        repetition_diagnostics.get("unparsed_raw_output") or {}
    ).get("suspected_repetition", False)
    if unclosed_answer and (finish_reason == "length" or raw_repetition):
        code = "unclosed_answer_repetition"
        return code, RETRY_FEEDBACK[code]

    if any("text exceeds" in error for error in errors):
        code = "transcription_too_long"
        return code, RETRY_FEEDBACK[code]

    field_error_fragments = (
        "target IDs/order mismatch",
        "value must be object",
        "fields must be",
        "invalid status",
        "text must be string",
        "transcribed text must be nonempty",
        "non-transcribed text must be empty",
        "readability_note must be a string",
        "requires readability_note",
        "requires empty readability_note",
        "text must not contain <UNK>",
    )
    if any(fragment in error for error in errors for fragment in field_error_fragments):
        code = "field_conflict"
        return code, RETRY_FEEDBACK[code]

    code = "other"
    return code, RETRY_FEEDBACK[code]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=common.DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe-name", required=True)
    parser.add_argument("--attempt-id", default="single_pass")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--job", action="append", dest="job_specs", metavar="PROFILE:CASE_ID")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--max-attempts-per-case", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def aggregate_single(output_root: Path, expected_jobs: int) -> dict[str, Any]:
    statuses = Counter()
    scores: list[dict[str, Any]] = []
    invalid_cases: list[dict[str, Any]] = []
    for status_path in output_root.glob("*/*/case_status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status_name = status.get("status", "unknown")
        statuses[status_name] += 1
        if status_name not in {"pass", "pass_after_max_attempts"}:
            invalid_cases.append({
                "profile": status_path.parent.parent.name,
                "case_id": status_path.parent.name,
                "status": status_name,
                "errors": status.get("errors", []),
            })
    for score_path in output_root.glob("*/*/score.json"):
        scores.append(json.loads(score_path.read_text(encoding="utf-8")))

    case_wers = [score["summary"]["wer"] for score in scores]
    totals = Counter()
    for score in scores:
        for key in ("distance", "reference_token_count"):
            totals[key] += score["summary"].get(key, 0)
    attempted = sum(statuses.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "execution_status": "complete" if attempted == expected_jobs else "partial",
        "expected_case_count": expected_jobs,
        "attempted_case_count": attempted,
        "strict_valid_output_case_count": statuses.get("pass", 0),
        "accepted_after_max_attempts_case_count": statuses.get(
            "pass_after_max_attempts", 0
        ),
        "scored_output_case_count": len(scores),
        "unscored_output_case_count": attempted - len(scores),
        "valid_output_case_count": len(scores),
        "invalid_output_case_count": attempted - len(scores),
        "case_status_counts": dict(statuses),
        "case_balanced_wer": sum(case_wers) / len(case_wers) if case_wers else None,
        "corpus_wer": (
            totals["distance"] / totals["reference_token_count"]
            if totals["reference_token_count"] else None
        ),
        "invalid_cases": invalid_cases,
        "created_at": datetime.now().astimezone().isoformat(),
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_attempts_per_case <= 3:
        raise ValueError("--max-attempts-per-case must be between 1 and 3")
    os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    model = args.model_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    jobs = common.build_jobs(args.mode, args.job_specs, args.sample_manifest)

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    import torch

    sample_manifest_path = (
        args.sample_manifest.expanduser().resolve() if args.sample_manifest else None
    )
    run_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "recipe_name": args.recipe_name,
        "mode": args.mode,
        "model": common.model_metadata(model),
        "code_sha256": common.sha256_file(Path(__file__).resolve()),
        "sample_manifest_path": str(sample_manifest_path) if sample_manifest_path else None,
        "sample_manifest_sha256": (
            common.sha256_file(sample_manifest_path) if sample_manifest_path else None
        ),
        "job_identities": [
            {"profile": job["profile"], "case_id": job["case_id"], "layer": job["layer"]}
            for job in jobs
        ],
        "prompt_protocol": {
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": common.sha256_text(SYSTEM_PROMPT),
            "user_prompt_template_source_sha256": common.sha256_text(inspect.getsource(user_prompt)),
            "output_format": "exactly one <answer>JSON object</answer> block",
            "retry_feedback_position": "appended to the end of the user prompt",
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
        },
        "model_execution": {
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "batch_size": 1,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_thinking": False,
            "model_calls_per_case_maximum": args.max_attempts_per_case,
        },
        "video_preprocessing": {
            "loader_backend": os.environ["VLLM_VIDEO_LOADER_BACKEND"],
            "requested_fps": args.video_fps,
            "limit_mm_per_prompt": {"video": 1},
        },
        "runtime": {
            "python": sys.version,
            "vllm": vllm_version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    run_fingerprint = common.sha256_json(run_identity)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError("output root already has a run manifest; use --resume or a new root")
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("resume fingerprint mismatch; use a new output root")
    elif any(output_root.iterdir()):
        raise RuntimeError("nonempty output root has no run manifest; use a new output root")
    else:
        common.write_json(manifest_path, {
            **run_identity,
            "run_fingerprint": run_fingerprint,
            "attempt_id": args.attempt_id,
            "command": [sys.executable, *sys.argv],
            "created_at": datetime.now().astimezone().isoformat(),
        })
        frozen = output_root / "frozen_prompts"
        common.write_text(frozen / "system_prompt.txt", SYSTEM_PROMPT)
        common.write_text(frozen / "user_prompt_template.py.txt", inspect.getsource(user_prompt))
        common.write_json(
            frozen / "output_schema_example.json",
            output_schema(["T01", "T02"]),
        )

    llm = LLM(
        model=str(model),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        allowed_local_media_path=str(common.PROJECT),
        limit_mm_per_prompt={"video": 1},
        media_io_kwargs={"video": {"fps": args.video_fps}},
        enforce_eager=True,
    )

    for job_index, job in enumerate(jobs, start=1):
        result_root = output_root / job["profile"] / job["case_id"]
        status_path = result_root / "case_status.json"
        if args.resume and status_path.is_file():
            saved_status = json.loads(status_path.read_text(encoding="utf-8"))
            if saved_status.get("status") in {
                "pass",
                "pass_after_max_attempts",
                "invalid",
            }:
                print(json.dumps({
                    "index": job_index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "skipped_terminal",
                }), flush=True)
                continue
        elif result_root.exists() and any(result_root.iterdir()):
            raise RuntimeError(
                f"{job['profile']}:{job['case_id']}: existing partial data requires --resume"
            )

        case = json.loads(job["prompt"].read_text(encoding="utf-8"))
        masked, public, private = common.mask_scene_prompt(case)
        target_ids = [item["id"] for item in public]
        base_user = user_prompt(masked, public)
        leaked = [
            item["id"] for item in private
            if item["verbatim"] in SYSTEM_PROMPT or item["verbatim"] in base_user
        ]
        if leaked:
            raise RuntimeError(
                f"{job['profile']}:{job['case_id']}: private reference leaked in request: {leaked}"
            )

        schema = output_schema(target_ids)
        input_record = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": args.run_id,
            "run_fingerprint": run_fingerprint,
            "attempt_id": args.attempt_id,
            "case_id": job["case_id"],
            "profile": job["profile"],
            "sample_layer": job["layer"],
            "video_path": str(job["video"]),
            "video_sha256": common.sha256_file(job["video"]),
            "video_metadata": common.video_metadata(job["video"]),
            "prompt_path": str(job["prompt"]),
            "prompt_sha256": common.sha256_file(job["prompt"]),
            "masked_scene_prompt": masked,
            "public_targets": public,
            "private_reference_sha256": common.sha256_json(private),
            "private_reference_visible_in_request": False,
            "system_prompt": SYSTEM_PROMPT,
            "initial_user_prompt": base_user,
            "request": {
                "schema_used_for_post_validation_only": schema,
                "max_attempts_per_case": args.max_attempts_per_case,
            },
        }
        common.write_json(result_root / "input.json", input_record)

        completed_attempts: list[dict[str, Any]] = []
        previous_failure_feedback: str | None = None
        for prior_number in range(1, args.max_attempts_per_case + 1):
            prior_response_path = result_root / "attempts" / f"round_{prior_number}" / "response.json"
            if not prior_response_path.is_file():
                break
            prior_response = json.loads(prior_response_path.read_text(encoding="utf-8"))
            if prior_response.get("status") == "pass":
                raise RuntimeError(
                    f"{job['profile']}:{job['case_id']}: passed attempt lacks terminal case status"
                )
            previous_failure_feedback = prior_response.get("retry_feedback")
            completed_attempts.append({
                "attempt": prior_number,
                "status": prior_response.get("status"),
                "retry_reason_code": prior_response.get("retry_reason_code"),
            })

        start_attempt = len(completed_attempts) + 1
        if start_attempt > args.max_attempts_per_case:
            raise RuntimeError(
                f"{job['profile']}:{job['case_id']}: exhausted attempts lack terminal case status"
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

        for attempt_number in range(start_attempt, args.max_attempts_per_case + 1):
            user = user_prompt(masked, public, previous_failure_feedback)
            leaked = [item["id"] for item in private if item["verbatim"] in user]
            if leaked:
                raise RuntimeError(
                    f"{job['profile']}:{job['case_id']}: private reference leaked in retry request: {leaked}"
                )
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
            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            input_token_count = len(prompt_token_ids) if prompt_token_ids is not None else None
            wall_seconds = time.time() - started

            parsed, errors, answer_block_diagnostics = parse_first_answer_block(raw)
            errors.extend(validate_answer(parsed, target_ids))
            repetition = common.response_diagnostics(
                raw,
                parsed,
                True,
                len(completion.token_ids),
                completion.finish_reason,
                args.max_tokens,
            )
            retry_reason_code = None
            retry_feedback = None
            if errors:
                retry_reason_code, retry_feedback = classify_retry_reason(
                    raw,
                    errors,
                    answer_block_diagnostics,
                    repetition,
                    completion.finish_reason,
                )

            attempt_root = result_root / "attempts" / f"round_{attempt_number}"
            attempt_input = {
                **input_record,
                "attempt_number": attempt_number,
                "previous_failure_feedback": previous_failure_feedback,
                "user_prompt": user,
                "request": {
                    **input_record["request"],
                    "input_token_count": input_token_count,
                    "sampling_params_repr": repr(sampling),
                },
            }
            response_record = {
                "attempt_number": attempt_number,
                "status": "pass" if not errors else "invalid_response",
                "answer": parsed,
                "validation_errors": errors,
                "answer_block_diagnostics": answer_block_diagnostics,
                "retry_reason_code": retry_reason_code,
                "retry_feedback": retry_feedback,
                "input_token_count": input_token_count,
                "output_token_count": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": getattr(completion, "stop_reason", None),
                "wall_seconds": round(wall_seconds, 3),
                "vllm_version": vllm_version,
            }
            common.write_json(attempt_root / "input.json", attempt_input)
            common.write_text(attempt_root / "response_raw.txt", raw)
            common.write_json(attempt_root / "response.json", response_record)
            common.write_json(attempt_root / "repetition_diagnostics.json", {"single": repetition})
            common.write_text(result_root / "response_raw.txt", raw)
            common.write_json(result_root / "response.json", response_record)
            common.write_json(result_root / "repetition_diagnostics.json", {"single": repetition})
            completed_attempts.append({
                "attempt": attempt_number,
                "status": response_record["status"],
                "retry_reason_code": retry_reason_code,
            })

            if not errors:
                score = common.score_case(
                    job["case_id"], job["profile"], job["layer"], private, parsed
                )
                score["protocol_version"] = PROTOCOL_VERSION
                score["selected_attempt"] = attempt_number
                common.write_json(result_root / "score.json", score)
                common.write_json(status_path, {
                    "status": "pass",
                    "attempts_completed": attempt_number,
                    "selected_attempt": attempt_number,
                    "attempts": completed_attempts,
                })
                print(json.dumps({
                    "index": job_index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "pass",
                    "attempt": attempt_number,
                    "output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "wer": score["summary"]["wer"],
                }, ensure_ascii=False), flush=True)
                break

            if attempt_number < args.max_attempts_per_case:
                previous_failure_feedback = retry_feedback
                common.write_json(status_path, {
                    "status": "retrying",
                    "attempts_completed": attempt_number,
                    "next_attempt": attempt_number + 1,
                    "last_errors": errors,
                    "last_retry_reason_code": retry_reason_code,
                    "attempts": completed_attempts,
                })
                print(json.dumps({
                    "index": job_index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "retrying",
                    "attempt": attempt_number,
                    "retry_reason_code": retry_reason_code,
                }, ensure_ascii=False), flush=True)
                continue

            if isinstance(parsed, dict):
                scoring_answer = answer_for_terminal_scoring(parsed, target_ids)
                score = common.score_case(
                    job["case_id"],
                    job["profile"],
                    job["layer"],
                    private,
                    scoring_answer,
                )
                score["protocol_version"] = PROTOCOL_VERSION
                score["selected_attempt"] = attempt_number
                score["terminal_acceptance_policy"] = (
                    "score_first_parsed_json_object_after_max_attempts"
                )
                score["strict_validation_errors"] = errors
                common.write_json(result_root / "score.json", score)
                common.write_json(status_path, {
                    "status": "pass_after_max_attempts",
                    "attempts_completed": attempt_number,
                    "selected_attempt": attempt_number,
                    "strict_validation_errors": errors,
                    "terminal_acceptance_policy": (
                        "score_first_parsed_json_object_after_max_attempts"
                    ),
                    "attempts": completed_attempts,
                })
                print(json.dumps({
                    "index": job_index,
                    "profile": job["profile"],
                    "case_id": job["case_id"],
                    "status": "pass_after_max_attempts",
                    "attempt": attempt_number,
                    "strict_validation_errors": errors,
                    "wer": score["summary"]["wer"],
                }, ensure_ascii=False), flush=True)
                break

            common.write_json(status_path, {
                "status": "invalid",
                "attempts_completed": attempt_number,
                "errors": errors,
                "last_retry_reason_code": retry_reason_code,
                "attempts": completed_attempts,
            })
            print(json.dumps({
                "index": job_index,
                "profile": job["profile"],
                "case_id": job["case_id"],
                "status": "invalid",
                "attempt": attempt_number,
                "retry_reason_code": retry_reason_code,
                "errors": errors,
            }, ensure_ascii=False), flush=True)

        common.write_json(output_root / "progress.json", aggregate_single(output_root, len(jobs)))

    summary = aggregate_single(output_root, len(jobs))
    common.write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
