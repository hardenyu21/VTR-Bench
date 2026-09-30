#!/usr/bin/env python3
"""Versioned VTextBench scoring using contextual tokenizer v2."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from vtextbench_tokenizer_v2 import TOKENIZER_VERSION, wer_tokens


SCORING_VERSION = "vtextbench_wer_v3_rplusn_bounded_20260913"


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return deterministic unit-cost Levenshtein S/D/I counts."""

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
            table[row][column] = min(
                candidates,
                key=lambda value: (
                    value[0],
                    value[2] + value[3],
                    value[1],
                    value[2],
                    value[3],
                ),
            )
    distance, substitutions, deletions, insertions = table[-1][-1]
    return {
        "distance": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def _validate_extra_tokens(extra_tokens: int) -> None:
    if isinstance(extra_tokens, bool) or not isinstance(extra_tokens, int) or extra_tokens < 0:
        raise ValueError("extra_tokens (N) must be a nonnegative integer")


def score_text(reference: str, hypothesis: str, *, extra_tokens: int) -> dict[str, Any]:
    """Score one target using an R+N hypothesis cap and bounded final WER."""

    _validate_extra_tokens(extra_tokens)
    reference_tokens = wer_tokens(reference)
    hypothesis_tokens_raw = wer_tokens(hypothesis)
    if not reference_tokens:
        raise ValueError("reference must contain at least one scored token")

    hypothesis_cap = len(reference_tokens) + extra_tokens
    hypothesis_tokens = hypothesis_tokens_raw[:hypothesis_cap]
    counts = edit_counts(reference_tokens, hypothesis_tokens)
    raw_wer = counts["distance"] / len(reference_tokens)
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "scoring_version": SCORING_VERSION,
        "extra_tokens": extra_tokens,
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens_raw": hypothesis_tokens_raw,
        "hypothesis_tokens": hypothesis_tokens,
        "hypothesis_token_cap": hypothesis_cap,
        "hypothesis_truncated": len(hypothesis_tokens_raw) > hypothesis_cap,
        "truncated_token_count": max(0, len(hypothesis_tokens_raw) - hypothesis_cap),
        **counts,
        "reference_token_count": len(reference_tokens),
        "raw_wer": raw_wer,
        "wer": min(1.0, raw_wer),
    }


def score_case(
    items: Iterable[Mapping[str, Any]],
    *,
    extra_tokens: int,
) -> dict[str, Any]:
    """Aggregate S/D/I across targets, then apply min(1, raw case WER)."""

    _validate_extra_tokens(extra_tokens)
    scored_items = []
    totals = {
        "distance": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_token_count": 0,
    }
    for item in items:
        result = score_text(
            str(item.get("reference", "")),
            str(item.get("hypothesis", "")),
            extra_tokens=extra_tokens,
        )
        result["text_id"] = item.get("text_id") or item.get("id")
        scored_items.append(result)
        for key in totals:
            totals[key] += int(result[key])
    if not scored_items:
        raise ValueError("case must contain at least one target")
    raw_wer = totals["distance"] / totals["reference_token_count"]
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "scoring_version": SCORING_VERSION,
        "extra_tokens": extra_tokens,
        "items": scored_items,
        "summary": {
            **totals,
            "raw_wer": raw_wer,
            "wer": min(1.0, raw_wer),
            "text_count": len(scored_items),
            "truncated_text_count": sum(
                int(item["hypothesis_truncated"]) for item in scored_items
            ),
        },
    }


__all__ = [
    "SCORING_VERSION",
    "edit_counts",
    "score_case",
    "score_text",
]
