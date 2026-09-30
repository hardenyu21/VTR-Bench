"""Run the supplied offline suites and the integration checks."""

import pathlib
import subprocess
import sys

import agentic_i2v


def main(argv: list[str] | None = None) -> int:
    """Run CPU tests without loading model weights or calling providers."""
    if argv:
        raise ValueError("The offline test runner takes no arguments")
    package = pathlib.Path(__file__).resolve().parent
    agentic = pathlib.Path(agentic_i2v.__file__).resolve().parent
    commands = (
        [sys.executable, str(package / "evaluation/run_tests.py")],
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(agentic / "tests"),
            "-v",
        ],
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(package / "tests"),
            "-v",
        ],
    )
    for command in commands:
        subprocess.run(command, check=True)
    print("All offline suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
