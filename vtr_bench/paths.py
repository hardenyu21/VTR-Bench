"""Locate bundled inputs and user-owned output directories."""

import os
import pathlib


def data_file(name: str) -> pathlib.Path:
    """Return a path inside the installed benchmark data directory."""
    return pathlib.Path(__file__).resolve().parent / "data" / name


def project_root() -> pathlib.Path:
    """Resolve the workspace from the environment or current directory."""
    value = os.environ.get("VTR_BENCH_PROJECT_ROOT")
    value = value or os.environ.get("VTEXTBENCH_PROJECT_ROOT")
    return pathlib.Path(value or pathlib.Path.cwd()).expanduser().resolve()


def agentic_runs_root(root: pathlib.Path) -> pathlib.Path:
    """Resolve the run directory shared by the single-case and batch clients."""
    value = os.environ.get("VTR_BENCH_RUNS_ROOT")
    path = (
        pathlib.Path(value)
        if value
        else root / "generated_videos/experiments/agentic"
    )
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()
