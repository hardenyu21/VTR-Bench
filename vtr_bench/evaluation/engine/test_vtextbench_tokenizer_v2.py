#!/usr/bin/env python3
from __future__ import annotations

import unittest

from vtextbench_tokenizer_v2 import tokenize, wer_tokens


class VTextBenchTokenizerV2Tests(unittest.TestCase):
    def assert_tokens(self, text: str, expected: list[str]) -> None:
        self.assertEqual(wer_tokens(text), expected)

    def test_ordinary_punctuation_is_ignored(self) -> None:
        expected = ["hello"]
        for text in ["hello", "hello,", "hello;", "hello!", '"hello"', "hello—"]:
            with self.subTest(text=text):
                self.assert_tokens(text, expected)

    def test_ordinary_spacing_is_collapsed(self) -> None:
        expected = ["hello", "world"]
        for text in ["hello world", "hello   world", "hello\nworld", " hello\tworld "]:
            with self.subTest(text=text):
                self.assert_tokens(text, expected)

    def test_apostrophe_variants_are_low_impact(self) -> None:
        self.assert_tokens("don't", ["dont"])
        self.assert_tokens("don’t", ["dont"])
        self.assert_tokens("dont", ["dont"])

    def test_hyphenated_prose_matches_space_separated_prose(self) -> None:
        self.assert_tokens("well-being", ["well", "being"])
        self.assert_tokens("well being", ["well", "being"])

    def test_identifiers_keep_semantic_hyphens_and_normalize_space(self) -> None:
        expected = ["SC-07"]
        self.assert_tokens("SC-07", expected)
        self.assert_tokens("SC - 07", expected)
        self.assert_tokens("SC–07", expected)
        self.assert_tokens("TR-B17", ["TR-B17"])
        self.assert_tokens("2026-09-13", ["2026-09-13"])
        self.assert_tokens("L-5", ["L-5"])

    def test_dotted_abbreviation_ignores_dots(self) -> None:
        self.assert_tokens("U.S.A.", ["USA"])

    def test_version_and_ip_are_stable(self) -> None:
        self.assert_tokens("v1.2.3", ["v1.2.3"])
        self.assert_tokens("192.168.1.1", ["192.168.1.1"])

    def test_number_and_unit_are_separate_with_or_without_space(self) -> None:
        expected = ["95", "°C"]
        self.assert_tokens("95°C", expected)
        self.assert_tokens("95 °C", expected)
        self.assert_tokens("95 ° C", expected)
        self.assert_tokens("48V", ["48", "V"])
        self.assert_tokens("48 V", ["48", "V"])
        self.assert_tokens("360° Visibility", ["360", "°", "Visibility"])
        self.assert_tokens("40 ° N", ["40", "°N"])

    def test_percent_and_currency_are_semantic(self) -> None:
        self.assert_tokens("95%", ["95", "%"])
        self.assert_tokens("95 %", ["95", "%"])
        self.assert_tokens("$12.50", ["$", "12.50"])

    def test_scientific_number_is_atomic(self) -> None:
        self.assert_tokens("−1.25e-3 V", ["−1.25e-3", "V"])
        self.assert_tokens("12:30", ["12:30"])

    def test_slash_compounds_normalize_space(self) -> None:
        expected = ["kJ/mol"]
        self.assert_tokens("kJ/mol", expected)
        self.assert_tokens("kJ / mol", expected)
        self.assert_tokens("A/B", ["A/B"])
        self.assert_tokens("A / B", ["A/B"])
        self.assert_tokens("3 / 4", ["3/4"])
        self.assert_tokens("kg · m / s²", ["kg·m/s²"])
        self.assert_tokens("dR / dt", ["dR/dt"])
        self.assert_tokens("µg/m³ · +8 Pa", ["µg/m³", "+8", "Pa"])
        self.assert_tokens("kJ/(mol·K)", ["kJ", "/", "(", "mol·K", ")"])

    def test_prose_slashes_and_middle_dots_are_ignored_separators(self) -> None:
        self.assert_tokens("save/load", ["save", "load"])
        self.assert_tokens("red / yellow", ["red", "yellow"])
        self.assert_tokens("Plan·Keep", ["Plan", "Keep"])
        self.assert_tokens("95°C · 30 sec", ["95", "°C", "30", "sec"])
        self.assert_tokens("Remaining: 3.2 km · ETA: 24 min", ["Remaining", "3.2", "km", "ETA", "24", "min"])
        self.assert_tokens("Power: OD -2.50 / OS -2.75", ["Power", "OD", "-2.50", "OS", "-2.75"])
        self.assert_tokens("Search/Match Results: FOM=98.2", ["Search", "Match", "Results", "FOM", "=", "98.2"])
        self.assert_tokens("'tight' /taɪt/ (Modern)", ["tight", "taɪt", "Modern"])

    def test_equation_ignores_operator_spacing(self) -> None:
        expected = ["λ", "=", "632.8", "nm"]
        self.assert_tokens("λ=632.8nm", expected)
        self.assert_tokens("λ = 632.8 nm", expected)

    def test_formula_operators_and_grouping_are_preserved(self) -> None:
        self.assert_tokens(
            "H₂O + CO₂ → H₂CO₃",
            ["H₂O", "+", "CO₂", "→", "H₂CO₃"],
        )
        self.assert_tokens("f(x, y)=x²+y²", ["f", "(", "x", ",", "y", ")", "=", "x²", "+", "y²"])
        self.assert_tokens("x-y", ["x", "-", "y"])
        self.assert_tokens("1 - N/K", ["1", "-", "N/K"])
        self.assert_tokens("ln N = (ln 2 / T½) · t", ["ln", "N", "=", "(", "ln", "2/T½", ")", "·", "t"])
        self.assert_tokens("[1, 2]", ["[", "1", ",", "2", "]"])
        self.assert_tokens("(x, y)", ["(", "x", ",", "y", ")"])
        self.assert_tokens("|x|", ["|", "x", "|"])
        self.assert_tokens("Rf = distance / dye front", ["Rf", "=", "distance", "/", "dye", "front"])

    def test_prose_hyphens_remain_ignored_inside_math_heavy_text(self) -> None:
        self.assert_tokens(
            "q = kA; cold-pressed and auto-off",
            ["q", "=", "kA", "cold", "pressed", "and", "auto", "off"],
        )

    def test_prose_parentheses_are_ignored(self) -> None:
        self.assert_tokens("hello (world)", ["hello", "world"])

    def test_vertical_bar_is_formula_only(self) -> None:
        self.assert_tokens("Mode: WOK | Power: 80%", ["Mode", "WOK", "Power", "80", "%"])
        self.assert_tokens("Mode | Power | Temp", ["Mode", "Power", "Temp"])
        self.assert_tokens("P(A | B)", ["P", "(", "A", "|", "B", ")"])
        self.assert_tokens("|x|", ["|", "x", "|"])
        self.assert_tokens("A=1 | B=2", ["A", "=", "1", "B", "=", "2"])
        self.assert_tokens("Sparrow |||| Tit || Blackbird |||", ["Sparrow", "Tit", "Blackbird"])
        self.assert_tokens(
            "Zn(s) | Zn²⁺(aq) || Cu²⁺(aq) | Cu(s)",
            ["Zn", "(", "s", ")", "|", "Zn²", "⁺", "(", "aq", ")", "|", "|", "Cu²", "⁺", "(", "aq", ")", "|", "Cu", "(", "s", ")"],
        )

    def test_code_quote_marks_are_ordinary_punctuation(self) -> None:
        self.assert_tokens("`hello`", ["hello"])

    def test_han_and_kana_are_character_granular(self) -> None:
        self.assert_tokens("双缝干涉实验", ["双", "缝", "干", "涉", "实", "验"])
        self.assert_tokens("欠けた記憶", ["欠", "け", "た", "記", "憶"])

    def test_emoji_zwj_sequence_is_one_token(self) -> None:
        self.assert_tokens("👩🏽‍🔬 test", ["👩🏽‍🔬", "test"])

    def test_url_email_hashtag_and_mention_are_atomic(self) -> None:
        self.assert_tokens("https://example.com/a/b.", ["https://example.com/a/b"])
        self.assert_tokens("a.b@example.com", ["a.b@example.com"])
        self.assert_tokens("#SceneText @tester", ["#SceneText", "@tester"])

    def test_token_kinds_expose_auditable_decisions(self) -> None:
        detailed = tokenize("95°C, hello; A / B")
        self.assertEqual(
            [(token.value, token.kind) for token in detailed],
            [("95", "number"), ("°C", "unit"), ("hello", "word"), ("A/B", "compound")],
        )


if __name__ == "__main__":
    unittest.main()
