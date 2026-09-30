from __future__ import annotations

import unittest

from agentic_i2v.prompt_composer import (
    MOTION_HEADER,
    ORIGINAL_HEADER,
    PRECEDENCE_RULE,
    compose_h3_prompt,
)
from agentic_i2v.schemas import GenerateVideoInput


class PromptComposerTests(unittest.TestCase):
    def test_original_prompt_is_preserved_verbatim_once(self) -> None:
        original = 'Scene description. Text reads exactly: "PR #1482".'
        refinement = "Keep the camera locked and use subtle hand motion."
        composed, normalized = compose_h3_prompt(original, refinement)
        self.assertEqual(normalized, refinement)
        self.assertEqual(composed.count(original), 1)
        self.assertIn(f"{ORIGINAL_HEADER}\n{original}", composed)
        self.assertIn(f"{MOTION_HEADER}\n{refinement}", composed)
        self.assertTrue(composed.endswith(PRECEDENCE_RULE))

    def test_verbatim_original_prefix_is_deduplicated_without_rewriting(self) -> None:
        original = "Original request with required text ABC-123."
        refinement = original + "\n\nAdditional motion: keep the camera fixed."
        composed, normalized = compose_h3_prompt(original, refinement)
        self.assertEqual(normalized, "Additional motion: keep the camera fixed.")
        self.assertEqual(composed.count(original), 1)
        self.assertIn("ABC-123", composed)

    def test_refinement_must_add_information(self) -> None:
        with self.assertRaises(ValueError):
            compose_h3_prompt("Original", "Original")

    def test_generate_video_contract_uses_motion_refinement(self) -> None:
        fields = GenerateVideoInput.model_fields
        self.assertIn("motion_refinement", fields)
        self.assertNotIn("motion_prompt", fields)


if __name__ == "__main__":
    unittest.main()
