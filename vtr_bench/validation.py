"""Validate aligned inputs, frozen evaluation engines, and release hygiene."""

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys

from vtr_bench import data
from vtr_bench import io_utils
from vtr_bench import paths


def validate_sources() -> dict:
    """Require exact original prompts and unchanged evaluation engine bytes."""
    cases = data.load_cases()
    provenance = json.loads(paths.data_file("provenance.json").read_text())
    for name, expected in provenance["data_sha256"].items():
        if io_utils.sha256_file(paths.data_file(name)) != expected:
            raise ValueError(f"Bundled data changed: {name}")
    original = paths.data_file("original_prompts.csv")
    if io_utils.sha256_file(original) != provenance["original_csv_sha256"]:
        raise ValueError("The original prompt CSV changed")
    with original.open(encoding="utf-8-sig", newline="") as handle:
        csv_prompts = {
            row["id"]: row["prompt_en"] for row in csv.DictReader(handle)
        }
    if csv_prompts != {key: row["prompt_en"] for key, row in cases.items()}:
        raise ValueError("Generation and evaluation prompts differ")
    engine = pathlib.Path(__file__).parent / "evaluation/engine"
    for name, expected in provenance["frozen_evaluation_engine_sha256"].items():
        if io_utils.sha256_file(engine / name) != expected:
            raise ValueError(f"Frozen evaluation code changed: {name}")
    return {
        "cases": len(cases),
        "prompt_en_matches_original_csv": True,
        "frozen_evaluation_engines": "unchanged",
    }


def scan_release(root: pathlib.Path) -> list[str]:
    """Find credentials and machine identifiers, allowing reviewed fixtures."""
    violations = []
    ignored = {
        ".git",
        ".venv",
        "cache",
        "generated_videos",
        "videos",
        "runs",
        "__pycache__",
        "build",
        "dist",
        ".ruff_cache",
    }
    private_path = re.compile(r"/(?:Users|home)/[^/\s]+/")
    address = re.compile(r"(?<![\d.=])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
    fixtures = {
        "vtr_bench/evaluation/engine/test_vtextbench_tokenizer_v2.py": {
            "192.168.1.1"
        },
        "vtr_bench/validation.py": {"192.168.1.1"},
    }
    credential = re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    )
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(
            part in ignored
            or part.startswith(".venv")
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.name == ".env" or path.suffix in {".pem", ".key"}:
            violations.append(str(relative))
            continue
        if (
            path.suffix
            not in {
                ".py",
                ".md",
                ".txt",
                ".json",
                ".toml",
                ".yml",
                ".service",
            }
            and path.name != ".env.example"
        ):
            continue
        text = path.read_text(encoding="utf-8")
        remote_addresses = set(address.findall(text)) - {"127.0.0.1", "0.0.0.0"}
        remote_addresses -= fixtures.get(relative.as_posix(), set())
        if (
            credential.search(text)
            or private_path.search(text)
            or remote_addresses
        ):
            violations.append(str(relative))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run all input checks; optional source-tree scanning is offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=pathlib.Path)
    args = parser.parse_args(argv)
    report = validate_sources()
    evaluation = pathlib.Path(__file__).parent / "evaluation/evaluate.py"
    subprocess.run(
        [sys.executable, str(evaluation), "--validate-only"], check=True
    )
    if args.release_root:
        violations = scan_release(args.release_root)
        if violations:
            raise ValueError(f"Release privacy scan failed: {violations}")
        report["privacy_scan"] = "pass"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
