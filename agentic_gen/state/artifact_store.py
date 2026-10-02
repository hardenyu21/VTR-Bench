from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    """Owns one case directory and performs recoverable atomic writes."""

    def __init__(self, runs_root: Path, case_id: str) -> None:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in case_id)
        if not safe or safe in {".", ".."}:
            raise ValueError(f"Unsafe case ID: {case_id!r}")
        self.root = (runs_root / safe).resolve()
        self.runs_root = runs_root.resolve()
        self.deliverables = self.runs_root / "final_videos"
        self.images = self.root / "artifacts" / "images"
        self.videos = self.root / "artifacts" / "videos"
        self.frames = self.root / "artifacts" / "frames"
        self.reports = self.root / "artifacts" / "reports"
        self.state_path = self.root / "state.json"
        for directory in (self.images, self.videos, self.frames, self.reports, self.deliverables):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def resolve(self, relative_path: str) -> Path:
        resolved = (self.root / relative_path).resolve()
        resolved.relative_to(self.root)
        return resolved

    def candidate_path(self, kind: str, candidate_id: str, suffix: str) -> Path:
        directory = self.images if kind == "image" else self.videos
        return directory / f"{candidate_id}{suffix}"

    def publish_final_video(self, source: Path, global_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in global_id)
        if not safe or safe in {".", ".."}:
            raise ValueError(f"Unsafe global ID: {global_id!r}")
        source = source.resolve(strict=True)
        target = self.deliverables / f"{safe}.mp4"
        source_hash = sha256_file(source)
        if target.exists():
            if sha256_file(target) != source_hash:
                raise RuntimeError(f"Conflicting deliverable already exists: {target}")
            return target
        temporary = self.deliverables / f".{safe}.{os.getpid()}.partial.mp4"
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if sha256_file(target) != source_hash:
                    raise RuntimeError(f"Conflicting deliverable appeared concurrently: {target}")
            return target
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def atomic_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
