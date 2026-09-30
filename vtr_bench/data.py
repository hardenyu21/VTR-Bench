"""Read the bundled benchmark's original prompts and public case IDs."""

import json
import pathlib
import re

from vtr_bench import paths


def load_cases(prompt_file: pathlib.Path | None = None) -> dict[str, dict]:
    """Load canonical records and reject duplicate or unsafe identifiers."""
    source = prompt_file or paths.data_file("prompts.json")
    records = json.loads(source.read_text(encoding="utf-8"))["cases"]
    cases = {}
    for record in records:
        identity = record["id"]
        if not re.fullmatch(r"(?:AD|SCI|UI|CULT|LIFE)-\d{4}", identity):
            raise ValueError(f"Invalid case ID: {identity!r}")
        if identity in cases or not isinstance(record.get("prompt_en"), str):
            raise ValueError(f"Duplicate ID or invalid prompt: {identity}")
        cases[identity] = record
    if not cases:
        raise ValueError("The prompt set is empty")
    return cases
