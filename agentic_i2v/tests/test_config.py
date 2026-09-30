from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_i2v.cli import build_parser
from agentic_i2v.config import RuntimeConfig
from agentic_i2v.schemas import BudgetState


class RuntimeConfigTests(unittest.TestCase):
    def test_chat_model_defaults_to_undated_alias(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BAILIAN_API_KEY": "test-key",
                "BAILIAN_BASE_URL": "https://example.invalid/v1",
            },
            clear=True,
        ):
            config = RuntimeConfig.from_env(Path("/tmp/vtextbench-config-test"))

        self.assertEqual(config.qwen_chat_model, "qwen3.7-plus")

    def test_video_generation_budget_defaults_to_three(self) -> None:
        self.assertEqual(BudgetState().max_video_generations, 3)
        args = build_parser().parse_args(["--case-id", "TEST-0001"])
        self.assertEqual(args.max_video_generations, 3)


if __name__ == "__main__":
    unittest.main()
