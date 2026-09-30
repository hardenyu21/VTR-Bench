#!/usr/bin/env python3
"""Static checks for the bounded-retry tagged-JSON VLM-WER evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_vlm_wer_retry.py")
SPEC = importlib.util.spec_from_file_location("evaluate_vlm_wer_retry", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    targets = [{"id": "T01", "carrier": "sign", "language": "English"}]
    user = MODULE.user_prompt("A masked scene.", targets)
    assert user.startswith("SCENE DESCRIPTION\n\nA masked scene.")
    assert "TARGET CARRIERS" in user
    assert "OUTPUT EXAMPLE\n\n<answer>" in user
    assert user.endswith(
        "Based on the video and the information listed above, please output exactly one valid JSON object enclosed in <answer> and </answer>, following the format shown above. Do not output anything before <answer> or after </answer>."
    )
    assert "Return exactly one valid JSON object enclosed in <answer> and </answer>." in MODULE.SYSTEM_PROMPT
    assert "Omit portions that cannot be reliably transcribed." in MODULE.SYSTEM_PROMPT
    assert "READABLE PORTION" in user
    assert "failure_modes" not in MODULE.SYSTEM_PROMPT
    assert "failure_modes" not in user
    assert "VISIBLE <UNK>" not in user
    assert "PREVIOUS ATTEMPT FEEDBACK" not in user
    retry_user = MODULE.user_prompt(
        "A masked scene.",
        targets,
        MODULE.RETRY_FEEDBACK["field_conflict"],
    )
    assert retry_user.endswith(MODULE.RETRY_FEEDBACK["field_conflict"])
    assert retry_user.count("PREVIOUS ATTEMPT FEEDBACK") == 1

    raw = '<answer>{"T01":{"status":"transcribed","text":"ABC","readability_note":""}}</answer>'
    parsed, errors, diagnostics = MODULE.parse_first_answer_block(raw)
    assert not errors
    assert diagnostics["complete_answer_block_count"] == 1
    assert diagnostics["selected_answer_block_index"] == 1
    assert parsed == {"T01": {"status": "transcribed", "text": "ABC", "readability_note": ""}}
    assert not MODULE.validate_answer(parsed, ["T01"])

    partial_without_placeholder = {
        "T01": {
            "status": "transcribed",
            "text": "READABLE PORTION",
            "readability_note": "Only part of the text can be reliably transcribed.",
        }
    }
    assert not MODULE.validate_answer(partial_without_placeholder, ["T01"])
    with_placeholder = {
        "T01": {
            "status": "transcribed",
            "text": "READABLE <UNK>",
            "readability_note": "Only part is readable.",
        }
    }
    assert any("must not contain <UNK>" in error for error in MODULE.validate_answer(with_placeholder, ["T01"]))
    unreadable_without_note = {
        "T01": {"status": "text_unreadable", "text": "", "readability_note": ""}
    }
    assert any("requires readability_note" in error for error in MODULE.validate_answer(unreadable_without_note, ["T01"]))
    absent_with_note = {
        "T01": {
            "status": "carrier_not_found",
            "text": "",
            "readability_note": "Not visible.",
        }
    }
    assert any("requires empty readability_note" in error for error in MODULE.validate_answer(absent_with_note, ["T01"]))

    terminal_answer = MODULE.answer_for_terminal_scoring(
        {
            "T01": {
                "status": "transcribed",
                "text": "A" * 700,
                "readability_note": "",
            },
            "T02": {"status": "bad", "text": "VISIBLE"},
        },
        ["T01", "T02", "T03"],
    )
    assert len(terminal_answer["T01"]["text"]) == 700
    assert terminal_answer["T02"] == {
        "status": "transcribed",
        "text": "VISIBLE",
        "readability_note": "",
    }
    assert terminal_answer["T03"] == {
        "status": "text_unreadable",
        "text": "",
        "readability_note": "",
    }

    second = '<answer>{"T01":{"status":"text_unreadable","text":"","readability_note":"Unreadable."}}</answer>'
    parsed, errors, diagnostics = MODULE.parse_first_answer_block(
        "prefix" + raw + "suffix" + second
    )
    assert not errors
    assert parsed == {"T01": {"status": "transcribed", "text": "ABC", "readability_note": ""}}
    assert diagnostics == {
        "complete_answer_block_count": 2,
        "selected_answer_block_index": 1,
        "nonempty_prefix_before_first_block": True,
        "nonempty_suffix_after_first_block": True,
        "additional_complete_answer_blocks": 1,
    }

    bad_first = '<answer>{invalid json}</answer>' + raw
    parsed, errors, diagnostics = MODULE.parse_first_answer_block(bad_first)
    assert parsed is None
    assert any("invalid JSON in first answer block" in error for error in errors)
    assert diagnostics["complete_answer_block_count"] == 2

    parsed, errors, diagnostics = MODULE.parse_first_answer_block("no answer block")
    assert parsed is None
    assert errors == ["missing complete <answer>...</answer> block"]
    assert diagnostics["complete_answer_block_count"] == 0

    repetition = {
        "unparsed_raw_output": {"suspected_repetition": True},
        "parsed_field_warnings": [],
    }
    code, feedback = MODULE.classify_retry_reason(
        "<answer>" + "AB" * 200,
        ["missing complete <answer>...</answer> block"],
        {"complete_answer_block_count": 0},
        repetition,
        "length",
    )
    assert code == "unclosed_answer_repetition"
    assert feedback == MODULE.RETRY_FEEDBACK[code]

    code, _ = MODULE.classify_retry_reason(
        raw,
        ["T01: text exceeds 640 characters"],
        {"complete_answer_block_count": 1},
        {},
        "stop",
    )
    assert code == "transcription_too_long"

    code, _ = MODULE.classify_retry_reason(
        raw,
        ["T01: carrier_not_found requires empty readability_note"],
        {"complete_answer_block_count": 1},
        {},
        "stop",
    )
    assert code == "field_conflict"

    code, feedback = MODULE.classify_retry_reason(
        "<answer>{bad}</answer>",
        ["invalid JSON in first answer block"],
        {"complete_answer_block_count": 1},
        {},
        "stop",
    )
    assert code == "other"
    assert feedback == "Your previous attempt failed. Please correct the output in the current attempt."

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert source.count("llm.chat(") == 1
    assert '"model_calls_per_case_maximum": args.max_attempts_per_case' in source
    assert 'default=3' in source
    assert 'default=1024' in source
    assert 'stop=["</answer>"]' not in source
    assert "include_stop_str_in_output" not in source
    assert 'chat_template_kwargs={"enable_thinking": False}' in source
    assert "--base-seed" not in source
    assert "request_seed" not in source
    assert "seed=args" not in source
    assert "score_first_parsed_json_object_after_max_attempts" in source
    print("all static bounded-retry VLM-WER checks passed")


if __name__ == "__main__":
    main()
