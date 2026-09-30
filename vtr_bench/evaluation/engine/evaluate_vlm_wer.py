"""Current retry evaluator support; historical online score kept for audit only."""

from __future__ import annotations

import argparse

import hashlib

import inspect

import json

import os

import re

import subprocess

import sys

import tempfile

import time

import unicodedata

from collections import Counter

from datetime import datetime

from importlib.metadata import PackageNotFoundError, version as package_version

from pathlib import Path

from typing import Any

STATUSES = {"transcribed", "carrier_not_found", "text_unreadable"}

SCHEMA_VERSION = "1.7"

REPETITION_RULE_VERSION = "vtextbench_repetition_warning_v1_20260907"

TOKENIZER_VERSION = "vtextbench_wer_v1_nfc_20260905"

SCORING_VERSION = "vtextbench_wer_v2_hypothesis_token_cap_20260906"

MAX_TRANSCRIPTION_CHARS = 640

CONNECTORS = {".", ",", "/", "-", "‐", "‑", "–", "'", "’", "°"}

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)

def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
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

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
    except Exception as error:
        return {
            "status": "unavailable",
            "reason": f"ffprobe failed: {type(error).__name__}: {error}",
        }
    streams = parsed.get("streams") or []
    stream = streams[0] if streams else {}
    return {
        "status": "available",
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "duration_seconds": stream.get("duration") or parsed.get("format", {}).get("duration"),
    }

def model_metadata(path: Path) -> dict[str, Any]:
    known_repositories = {
        "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
        "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    }
    configuration_files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        candidate = path / name
        if candidate.is_file():
            configuration_files[name] = {
                "sha256": sha256_file(candidate),
                "size": candidate.stat().st_size,
            }
    weights = [
        {"name": candidate.name, "size": candidate.stat().st_size}
        for candidate in sorted(path.glob("*.safetensors"))
    ]
    return {
        "path": str(path),
        "source_repository": known_repositories.get(path.name),
        "download_transport": "unspecified; locally supplied checkpoint",
        "revision": {
            "status": "unavailable",
            "reason": "the downloaded directory does not retain a Hub commit reference",
        },
        "configuration_files": configuration_files,
        "weight_inventory": weights,
        "weight_inventory_sha256": sha256_json(weights),
        "weight_content_hash": {
            "status": "unavailable",
            "reason": "full model-weight hashing is outside this pilot; inventory is recorded instead",
        },
    }

def longest_character_run(value: str) -> dict[str, Any] | None:
    if not value:
        return None
    best_character = value[0]
    best_start = 0
    best_length = 1
    start = 0
    for index in range(1, len(value) + 1):
        if index < len(value) and value[index] == value[start]:
            continue
        length = index - start
        if length > best_length:
            best_character = value[start]
            best_start = start
            best_length = length
        start = index
    return {
        "character": best_character,
        "start": best_start,
        "end": best_start + best_length,
        "length": best_length,
    }

def longest_repeated_substring_run(value: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for unit_length in range(2, 17):
        limit = len(value) - unit_length * 16
        for start in range(max(0, limit + 1)):
            unit = value[start : start + unit_length]
            count = 1
            cursor = start + unit_length
            while value[cursor : cursor + unit_length] == unit:
                count += 1
                cursor += unit_length
            covered = count * unit_length
            if count < 16 or covered < 128:
                continue
            candidate = {
                "unit": unit,
                "unit_length": unit_length,
                "repeat_count": count,
                "covered_length": covered,
                "start": start,
                "end": cursor,
            }
            if best is None or (
                candidate["covered_length"],
                -candidate["unit_length"],
                -candidate["start"],
            ) > (
                best["covered_length"],
                -best["unit_length"],
                -best["start"],
            ):
                best = candidate
    return best

def repetition_diagnostic(value: str) -> dict[str, Any]:
    character_run = longest_character_run(value)
    substring_run = longest_repeated_substring_run(value)
    whitespace_runs = [
        {"start": match.start(), "end": match.end(), "length": len(match.group())}
        for match in re.finditer(r"\s{2,}", value)
    ]
    longest_whitespace = max(whitespace_runs, key=lambda item: item["length"], default=None)
    warning_reasons = []
    if character_run and character_run["length"] >= 64:
        warning_reasons.append("same_character_run_ge_64")
    if substring_run is not None:
        warning_reasons.append("substring_2_to_16_repeated_ge_16_covering_ge_128")
    if longest_whitespace and longest_whitespace["length"] >= 64:
        warning_reasons.append("whitespace_run_ge_64")
    return {
        "length": len(value),
        "suspected_repetition": bool(warning_reasons),
        "warning_reasons": warning_reasons,
        "longest_character_run": character_run,
        "longest_repeated_substring_run": substring_run,
        "longest_whitespace_run": longest_whitespace,
        "reached_text_character_limit": len(value) == MAX_TRANSCRIPTION_CHARS,
    }

def is_core_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category[0] in {"L", "N", "M"} or character == "_"

def is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )

def wer_tokens(text: str) -> list[str]:
    text = " ".join(unicodedata.normalize("NFC", text).split())
    tokens: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace():
            index += 1
            continue
        if text.startswith("<UNK>", index):
            tokens.append("<UNK>")
            index += 5
            continue
        if is_han(character):
            tokens.append(character)
            index += 1
            continue
        if character in {"-", "−", "+"} and index + 1 < len(text) and text[index + 1].isdigit():
            start = index
            index += 1
        elif is_core_character(character):
            start = index
            index += 1
        else:
            tokens.append(character)
            index += 1
            continue
        while index < len(text):
            character = text[index]
            if is_han(character):
                break
            if is_core_character(character):
                index += 1
                continue
            if (
                character in CONNECTORS
                and index + 1 < len(text)
                and is_core_character(text[index + 1])
                and not is_han(text[index + 1])
            ):
                index += 1
                continue
            if character == "%" and any(char.isdigit() for char in text[start:index]):
                index += 1
            break
        tokens.append(text[start:index])
    return tokens

def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            diagonal = table[row - 1][column - 1]
            deletion = table[row - 1][column]
            insertion = table[row][column - 1]
            candidates = [
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            # Keep the alignment deterministic and prefer a conventional
            # substitution over an equally costly delete/insert alignment.
            table[row][column] = min(
                candidates,
                key=lambda value: (value[0], value[2] + value[3], value[1], value[2], value[3]),
            )
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {
        "distance": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }

def tokenizer_self_test() -> dict[str, Any]:
    cases = {
        "λ = 632.8 nm": ["λ", "=", "632.8", "nm"],
        "mass = 5 kg": ["mass", "=", "5", "kg"],
        "H₂O + CO₂ → H₂CO₃": ["H₂O", "+", "CO₂", "→", "H₂CO₃"],
        "双缝干涉实验": ["双", "缝", "干", "涉", "实", "验"],
        "ENERGY <UNK> REPORT": ["ENERGY", "<UNK>", "REPORT"],
        "λ = 632.B nm": ["λ", "=", "632.B", "nm"],
        "−0.19 kJ/mol": ["−0.19", "kJ/mol"],
    }
    observed = {text: wer_tokens(text) for text in cases}
    failures = {text: {"expected": cases[text], "observed": observed[text]} for text in cases if observed[text] != cases[text]}
    edit_test = edit_counts(["λ", "=", "632.8", "nm"], ["λ", "=", "632.B", "nm"])
    if edit_test != {"distance": 1, "substitutions": 1, "deletions": 0, "insertions": 0}:
        failures["edit_distance"] = {"observed": edit_test}
    truncation_reference = ["A", "B"]
    truncation_raw_hypothesis = ["X", "Y", "Z", "Q"]
    truncation_hypothesis = truncation_raw_hypothesis[: len(truncation_reference)]
    truncation_counts = edit_counts(truncation_reference, truncation_hypothesis)
    if truncation_hypothesis != ["X", "Y"] or truncation_counts["distance"] > len(
        truncation_reference
    ):
        failures["hypothesis_token_truncation"] = {
            "reference": truncation_reference,
            "raw_hypothesis": truncation_raw_hypothesis,
            "scored_hypothesis": truncation_hypothesis,
            "counts": truncation_counts,
        }
    return {"version": TOKENIZER_VERSION, "status": "pass" if not failures else "fail", "cases": observed, "failures": failures}

def mask_scene_prompt(case: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    prompt = case["prompt_en"]
    required = case.get("required_text")
    if not isinstance(required, list) or not required:
        raise ValueError(f"{case.get('id')}: missing required_text")
    public_targets = []
    private_reference = []
    for item in required:
        text_id = item["id"]
        verbatim = item["verbatim"]
        if case["prompt_en"].count(verbatim) != 1:
            raise ValueError(f"{case['id']}:{text_id}: verbatim occurrence count is not one")
        if prompt.count(verbatim) != 1:
            raise ValueError(f"{case['id']}:{text_id}: replacement order collision")
        prompt = prompt.replace(verbatim, f"[TARGET {text_id}]", 1)
        public_targets.append({"id": text_id, "carrier": item["carrier"], "language": item["language"]})
        private_reference.append({"id": text_id, "verbatim": verbatim})
    for item in private_reference:
        if item["verbatim"] in prompt:
            raise ValueError(f"{case['id']}:{item['id']}: reference remains after masking")
    return prompt, public_targets, private_reference

def response_diagnostics(
    raw: str,
    parsed: dict[str, Any] | None,
    include_readability_note: bool,
    output_token_count: int,
    finish_reason: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    if isinstance(parsed, dict):
        for text_id, value in parsed.items():
            if not isinstance(value, dict):
                continue
            fields = {"text": repetition_diagnostic(value.get("text", ""))}
            if include_readability_note:
                fields["readability_note"] = repetition_diagnostic(
                    value.get("readability_note", "")
                )
            targets[text_id] = fields
    parsed_warnings = [
        {"target_id": text_id, "field": field_name, **diagnostic}
        for text_id, fields in targets.items()
        for field_name, diagnostic in fields.items()
        if diagnostic["suspected_repetition"]
    ]
    return {
        "rule_version": REPETITION_RULE_VERSION,
        "raw_json_parseable": isinstance(parsed, dict),
        "output_token_count": output_token_count,
        "configured_max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "max_tokens_exhausted": finish_reason == "length"
        and output_token_count >= max_tokens,
        "targets": targets,
        "parsed_field_warning_count": len(parsed_warnings),
        "parsed_field_warnings": parsed_warnings,
        "unparsed_raw_output": (
            None if isinstance(parsed, dict) else repetition_diagnostic(raw)
        ),
    }

def score_case(
    case_id: str,
    profile: str,
    sample_layer: str,
    private: list[dict[str, str]],
    answer: dict[str, Any],
) -> dict[str, Any]:
    items = []
    totals = Counter()
    for reference_item in private:
        text_id = reference_item["id"]
        reference = reference_item["verbatim"]
        status = answer[text_id]["status"]
        hypothesis = answer[text_id]["text"] if status == "transcribed" else ""
        reference_tokens = wer_tokens(reference)
        hypothesis_tokens_raw = wer_tokens(hypothesis)
        hypothesis_tokens = hypothesis_tokens_raw[: len(reference_tokens)]
        truncated_token_count = len(hypothesis_tokens_raw) - len(hypothesis_tokens)
        counts = edit_counts(reference_tokens, hypothesis_tokens)
        if counts["distance"] > len(reference_tokens):
            raise RuntimeError(f"{case_id}:{text_id}: truncated WER distance exceeds reference length")
        for key in ("substitutions", "deletions", "insertions", "distance"):
            totals[key] += counts[key]
        totals["reference_token_count"] += len(reference_tokens)
        items.append({
            "text_id": text_id,
            "status": status,
            "readability_note": answer[text_id]["readability_note"],
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_tokens": reference_tokens,
            "hypothesis_tokens_raw": hypothesis_tokens_raw,
            "hypothesis_tokens": hypothesis_tokens,
            "hypothesis_truncated": truncated_token_count > 0,
            "truncated_token_count": truncated_token_count,
            "hypothesis_token_count_before_truncation": len(hypothesis_tokens_raw),
            "hypothesis_token_count_after_truncation": len(hypothesis_tokens),
            **counts,
            "reference_token_count": len(reference_tokens),
            "wer": counts["distance"] / len(reference_tokens),
        })
    reference_total = totals["reference_token_count"]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "profile": profile,
        "sample_layer": sample_layer,
        "tokenizer_version": TOKENIZER_VERSION,
        "scoring_version": SCORING_VERSION,
        "items": items,
        "summary": {
            **dict(totals),
            "wer": totals["distance"] / reference_total,
            "text_count": len(items),
            "exact_text_count": sum(item["distance"] == 0 for item in items),
        },
    }

PROJECT = Path.cwd()
DEFAULT_MODEL = Path("models/evaluator")

def build_jobs(*args, **kwargs):
    raise RuntimeError("Use the portable evaluate.py entry point")

