from __future__ import annotations

import json

from ..schemas import WorkflowState


def compact_context(state: WorkflowState) -> str:
    candidates = []
    for candidate in state.candidates.values():
        candidates.append(
            {
                "id": candidate.candidate_id,
                "kind": candidate.kind.value,
                "parent": candidate.parent_id,
                "status": candidate.status.value,
                "created_by": candidate.created_by,
                "inputs": candidate.inputs,
                "observations": [item.model_dump(mode="json") for item in candidate.observations[-8:]],
            }
        )
    payload = {
        "original_prompt": state.prompt_en,
        "candidate_graph": candidates,
        "remaining_budget": state.budget.remaining(),
        "completed": state.completed,
    }
    return json.dumps(payload, ensure_ascii=False)

