"""Two public entrypoints: video evaluation and Agentic I2V."""

import argparse
import sys

from vtr_bench import agentic
from vtr_bench import evaluate


def main(argv: list[str] | None = None) -> int:
    """Expose workflows without importing model runtimes."""
    values = list(sys.argv[1:] if argv is None else argv)
    workflows = {"evaluate": evaluate.main, "agentic": agentic.main}
    if values and values[0] in workflows:
        return workflows[values[0]](values[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate", help="Evaluate a video folder or MP4")
    commands.add_parser(
        "agentic", help="Generate videos with the Agentic workflow"
    )
    parser.parse_args(values)
    return 0
