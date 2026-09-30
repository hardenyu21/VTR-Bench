"""One entrypoint for single-case, batch, and resident Agentic H3 service."""

import argparse
import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    """Forward to the preserved Agentic implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", help="Run one case; omit to run all 300")
    parser.add_argument(
        "--serve", action="store_true", help="Start the resident H3 service"
    )
    args = parser.parse_args(argv)
    if args.serve and args.case_id:
        parser.error("--serve and --case-id are mutually exclusive")
    if args.serve:
        arguments = ["-m", "agentic_i2v.service.h3_server"]
    elif args.case_id:
        arguments = ["-m", "agentic_i2v", "--case-id", args.case_id]
    else:
        endpoint = os.environ.get(
            "VTEXTBENCH_H3_SERVICE_URL", "http://127.0.0.1:18123"
        )
        arguments = [
            "-m",
            "agentic_i2v.batch",
            "--worker-index",
            "0",
            "--worker-count",
            "1",
            "--endpoint",
            endpoint,
        ]
    return subprocess.call([sys.executable, *arguments])
