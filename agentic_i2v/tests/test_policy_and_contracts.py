from __future__ import annotations

import unittest

from agentic_i2v.agent import ActionPolicy, BudgetExceeded, tool_definitions
from agentic_i2v.schemas import BudgetState, Observation, TOOL_INPUT_MODELS


class PolicyAndContractTests(unittest.TestCase):
    def test_all_eight_tools_have_strict_schemas(self) -> None:
        definitions = tool_definitions()
        self.assertEqual(len(definitions), 8)
        self.assertEqual({item["function"]["name"] for item in definitions}, set(TOOL_INPUT_MODELS))
        for definition in definitions:
            schema = definition["function"]["parameters"]
            self.assertFalse(schema.get("additionalProperties", True))
            self.assertIn("diagnosis", schema.get("required", []))

    def test_budget_counts_agent_and_video(self) -> None:
        budget = BudgetState(max_actions=2, max_api_calls=3, max_video_generations=1)
        policy = ActionPolicy(budget)
        policy.charge_central_turn()
        policy.charge_tool("generate_video")
        self.assertEqual(budget.video_generations_used, 1)
        with self.assertRaises(BudgetExceeded):
            policy.charge_tool("generate_video")

    def test_provider_observation_labels_are_normalized(self) -> None:
        observation = Observation(
            category="text",
            summary="illegible",
            severity="Critical",
            evidence_locations="upper-left sign",
        )
        self.assertEqual(observation.severity, 3)
        self.assertEqual(observation.evidence_locations, ["upper-left sign"])


if __name__ == "__main__":
    unittest.main()
