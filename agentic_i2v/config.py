from __future__ import annotations

import csv
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vtr_bench import paths

from .schemas import BudgetState


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_root: Path
    runs_root: Path
    prompt_file: Path
    qwen_chat_model: str = "qwen3.7-plus"
    qwen_image_model: str = "qwen-image-3.0"
    image_width: int = Field(default=1344, gt=0)
    image_height: int = Field(default=768, gt=0)
    h3_service_url: str = "http://127.0.0.1:18123"
    bailian_api_key: str = Field(repr=False, exclude=True)
    bailian_base_url: str
    budget: BudgetState = Field(default_factory=BudgetState)

    @classmethod
    def from_env(
        cls,
        project_root: Path,
        *,
        prompt_file: Path | None = None,
        budget: BudgetState | None = None,
    ) -> "RuntimeConfig":
        root = project_root.resolve()
        key = os.environ.get("BAILIAN_API_KEY", "").strip()
        base_url = os.environ.get("BAILIAN_BASE_URL", "").strip().rstrip("/")
        credential_file = os.environ.get("BAILIAN_CREDENTIAL_FILE", "").strip()
        if (not key or not base_url) and credential_file:
            with Path(credential_file).expanduser().open(encoding="utf-8-sig", newline="") as handle:
                values = {
                    row[0].strip(): row[1].strip()
                    for row in csv.reader(handle)
                    if len(row) >= 2 and row[0].strip()
                }
            key = key or values.get("apiKey", "").strip()
            base_url = base_url or values.get("openAiCompatible", "").strip().rstrip("/")
        if not key or not base_url:
            raise RuntimeError(
                "Set BAILIAN_API_KEY and BAILIAN_BASE_URL, or BAILIAN_CREDENTIAL_FILE"
            )
        selected_prompt = prompt_file or Path(
            os.environ.get(
                "VTEXTBENCH_PROMPT_FILE",
                paths.data_file("prompts.json"),
            )
        )
        return cls(
            project_root=root,
            runs_root=paths.agentic_runs_root(root),
            prompt_file=selected_prompt.expanduser().resolve(),
            qwen_chat_model=os.environ.get("VTEXTBENCH_QWEN_CHAT_MODEL", "qwen3.7-plus"),
            qwen_image_model=os.environ.get("VTEXTBENCH_QWEN_IMAGE_MODEL", "qwen-image-3.0"),
            image_width=int(os.environ.get("VTEXTBENCH_IMAGE_WIDTH", "1344")),
            image_height=int(os.environ.get("VTEXTBENCH_IMAGE_HEIGHT", "768")),
            h3_service_url=os.environ.get(
                "VTEXTBENCH_H3_SERVICE_URL", "http://127.0.0.1:18123"
            ).rstrip("/"),
            bailian_api_key=key,
            bailian_base_url=base_url,
            budget=budget or BudgetState(),
        )
