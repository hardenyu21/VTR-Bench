"""Run the retained offline regression tests from one entrypoint."""

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
    evaluation = package / "evaluation"
    commands = (
        (
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "engine",
                "-p",
                "test_vtextbench*.py",
                "-v",
            ],
            evaluation,
        ),
        (
            [sys.executable, "-m", "unittest", "test_portable", "-v"],
            evaluation,
        ),
        (
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(agentic / "tests"),
                "-v",
            ],
            None,
        ),
        (
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(package / "tests"),
                "-v",
            ],
            None,
        ),
    )
    for command, directory in commands:
        subprocess.run(command, cwd=directory, check=True)
    print("All offline suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
