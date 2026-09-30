"""Internal source-only packager; not a public workflow entrypoint."""

import argparse
import pathlib
import tempfile
import zipfile

from vtr_bench import io_utils
from vtr_bench import validation


def source_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Collect source files, rejecting symlinks and excluding runtime state."""
    ignored = {
        ".git",
        "__pycache__",
        ".ruff_cache",
        "build",
        "dist",
        "cache",
        "models",
        "videos",
        "generated_videos",
        "runs",
        "results",
    }
    selected = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if any(
            part in ignored
            or part.startswith(".venv")
            or part.endswith(".egg-info")
            for part in parts
        ):
            continue
        if path.name in {".DS_Store", "release_manifest.json"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Do not package symbolic links: {path.name}")
        if path.is_file():
            selected.append(path)
    return selected


def main(argv: list[str] | None = None) -> int:
    """Write a SHA256 manifest, then a portable source-only archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    output = args.output or root.parent / "VTR-Bench.zip"
    if output.resolve().is_relative_to(root):
        parser.error("the archive must be outside the source tree")
    violations = validation.scan_release(root)
    if violations:
        raise ValueError(f"Release privacy scan failed: {violations}")
    files = source_files(root)
    manifest = root / "release_manifest.json"
    io_utils.write_json(
        manifest,
        {
            "algorithm": "sha256",
            "files": {
                path.relative_to(root).as_posix(): io_utils.sha256_file(path)
                for path in files
            },
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip") as temp:
        with zipfile.ZipFile(temp.name, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in [*files, manifest]:
                info = zipfile.ZipInfo(
                    "VTR-Bench/" + path.relative_to(root).as_posix(),
                    date_time=(2026, 9, 30, 0, 0, 0),
                )
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        temp_path = pathlib.Path(temp.name)
        output.write_bytes(temp_path.read_bytes())
    print(f"Packaged {len(files) + 1} files: {output.name}")
    print(f"SHA256: {io_utils.sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
