#!/usr/bin/env python3
from __future__ import annotations

import unittest

from vtextbench_wer_v2 import edit_counts, score_case, score_text


class VTextBenchWerV2Tests(unittest.TestCase):
    def test_edit_counts(self) -> None:
        self.assertEqual(
            edit_counts(["A", "B", "C"], ["A", "X", "C", "D"]),
            {"distance": 2, "substitutions": 1, "deletions": 0, "insertions": 1},
        )

    def test_r_plus_n_cap(self) -> None:
        result = score_text("A B", "A B C D E", extra_tokens=1)
        self.assertEqual(result["hypothesis_tokens"], ["A", "B", "C"])
        self.assertEqual(result["hypothesis_token_cap"], 3)
        self.assertEqual(result["truncated_token_count"], 2)
        self.assertEqual(result["raw_wer"], 0.5)
        self.assertEqual(result["wer"], 0.5)

    def test_bounded_wer_is_applied_after_distance(self) -> None:
        result = score_text("A", "X Y Z", extra_tokens=2)
        self.assertEqual(result["distance"], 3)
        self.assertEqual(result["raw_wer"], 3.0)
        self.assertEqual(result["wer"], 1.0)

    def test_case_aggregates_before_bounding(self) -> None:
        result = score_case(
            [
                {"text_id": "T01", "reference": "A", "hypothesis": "X Y Z"},
                {"text_id": "T02", "reference": "A B C", "hypothesis": "A B C"},
            ],
            extra_tokens=2,
        )
        self.assertEqual(result["summary"]["distance"], 3)
        self.assertEqual(result["summary"]["reference_token_count"], 4)
        self.assertEqual(result["summary"]["raw_wer"], 0.75)
        self.assertEqual(result["summary"]["wer"], 0.75)

    def test_n_is_required_to_be_nonnegative_integer(self) -> None:
        for bad in [-1, 1.5, True]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_text("A", "A", extra_tokens=bad)  # type: ignore[arg-type]

if __name__ == "__main__":
    unittest.main()
