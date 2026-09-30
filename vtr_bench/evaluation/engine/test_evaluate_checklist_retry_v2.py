#!/usr/bin/env python3
"""Static and parser tests for the 20-item checklist retry evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_checklist_retry_v2.py")
SPEC = importlib.util.spec_from_file_location("evaluate_checklist_retry_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

COMPAT_PATH = Path(__file__).with_name("evaluate_checklist_retry_v2_compat.py")
COMPAT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_checklist_retry_v2_compat", COMPAT_PATH
)
assert COMPAT_SPEC and COMPAT_SPEC.loader
COMPAT = importlib.util.module_from_spec(COMPAT_SPEC)
COMPAT_SPEC.loader.exec_module(COMPAT)
# Exercise the production compatibility validator.
MODULE.validate_and_normalize = COMPAT.validate_and_normalize_with_padded_ids


def checklist() -> list[dict[str, str]]:
    return [
        {
            "id": f"C{index:02d}",
            "category": "must_not_be_sent",
            "question_en": f"Question {index}?",
        }
        for index in range(1, 21)
    ]


def main() -> None:
    user = MODULE.user_prompt(checklist())
    assert user.startswith("CHECKLIST\n\nC01: Question 1?")
    assert "[must_not_be_sent]" not in user
    assert "category" not in user
    assert "C20: Question 20?" in user
    assert user.count('"id":"C') == 20
    assert "PREVIOUS ATTEMPT FEEDBACK" not in user
    retry_user = MODULE.user_prompt(
        checklist(), MODULE.RETRY_FEEDBACK["invalid_answer_label"]
    )
    assert retry_user.endswith(MODULE.RETRY_FEEDBACK["invalid_answer_label"])
    assert retry_user.count("PREVIOUS ATTEMPT FEEDBACK") == 1
    assert "RULES" not in MODULE.SYSTEM_PROMPT
    assert "Do not output anything else." in MODULE.SYSTEM_PROMPT

    raw = '<answer>[{"id":"C01","answer":"Yes"}]</answer>'
    parsed, errors, diagnostics = MODULE.parse_first_answer_block(raw)
    assert not errors
    assert diagnostics["repair_applied"] is None
    validation_errors, normalized = MODULE.validate_and_normalize(parsed, ["C01"])
    assert not validation_errors
    assert normalized == [{"id": "C01", "answer": "yes"}]

    padded_ids = [
        {"id": f"C0{index}", "answer": "YES" if index % 2 else "No"}
        for index in range(10, 21)
    ]
    validation_errors, normalized = MODULE.validate_and_normalize(
        padded_ids, [f"C{index}" for index in range(10, 21)]
    )
    assert not validation_errors
    assert normalized == [
        {"id": f"C{index}", "answer": "yes" if index % 2 else "no"}
        for index in range(10, 21)
    ]
    assert COMPAT.adaptive_max_tokens(1024, 0) == 1024
    assert COMPAT.adaptive_max_tokens(1024, 1) == 2048
    assert COMPAT.adaptive_max_tokens(1024, 2) == 4096
    assert COMPAT.adaptive_max_tokens(1024, 3) == 4096

    validation_errors, normalized = MODULE.validate_and_normalize(
        [{"id": "C001", "answer": "yes"}], ["C01"]
    )
    assert normalized is None
    assert validation_errors == [
        "checklist ID/order mismatch at item 1: expected C01, got 'C001'"
    ]

    validation_errors, normalized = COMPAT.validate_and_normalize_with_padded_ids(
        padded_ids, [f"C{index}" for index in range(10, 21)]
    )
    assert not validation_errors
    assert normalized == [
        {"id": f"C{index}", "answer": "yes" if index % 2 else "no"}
        for index in range(10, 21)
    ]

    repaired_raw = '<answer>[{"id":"C01","answer":" NO "}</answer>'
    parsed, errors, diagnostics = MODULE.parse_first_answer_block(repaired_raw)
    assert not errors
    assert diagnostics["repair_applied"] == "append_closing_square_bracket"
    validation_errors, normalized = MODULE.validate_and_normalize(parsed, ["C01"])
    assert not validation_errors
    assert normalized == [{"id": "C01", "answer": "no"}]

    parsed, errors, diagnostics = MODULE.parse_first_answer_block(
        '<answer>[{"id":"C01","answer":]</answer>'
    )
    assert parsed is None
    assert errors[0].startswith("invalid JSON")
    code, _ = MODULE.classify_retry_reason(errors, diagnostics)
    assert code == "invalid_json"

    parsed, errors, diagnostics = MODULE.parse_first_answer_block("<answer>[{}]")
    assert parsed is None
    code, _ = MODULE.classify_retry_reason(errors, diagnostics)
    assert code == "incomplete_answer_block"

    parsed, errors, diagnostics = MODULE.parse_first_answer_block(
        '<answer>[{"id":"C01","answer":"uncertain"}]</answer>'
    )
    validation_errors, normalized = MODULE.validate_and_normalize(parsed, ["C01"])
    assert normalized is None
    code, feedback = MODULE.classify_retry_reason(validation_errors, diagnostics)
    assert code == "invalid_answer_label"
    assert "not yes or no" in feedback

    parsed, errors, diagnostics = MODULE.parse_first_answer_block(
        'prefix<answer>[{"id":"C01","answer":"yes"}]</answer>suffix'
        '<answer>[]</answer>'
    )
    assert not errors
    assert diagnostics["complete_answer_block_count"] == 2
    assert diagnostics["selected_answer_block_index"] == 1
    assert diagnostics["nonempty_prefix_before_first_block"]
    assert diagnostics["nonempty_suffix_after_first_block"]

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'default=1024' in source
    assert 'default=3' in source
    assert 'chat_template_kwargs={"enable_thinking": False}' in source
    assert 'stop=["</answer>"]' not in source
    assert "SamplingParams(" in source
    assert '"seed": None' in source
    assert '"penalty_score_computed": False' in source
    print("all checklist retry evaluator checks passed")


if __name__ == "__main__":
    main()
