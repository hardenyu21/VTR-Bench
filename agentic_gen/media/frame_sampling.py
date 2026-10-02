from __future__ import annotations

import subprocess
from pathlib import Path


def sample_uniform(video: Path, output_dir: Path, sample_fps: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"fps={sample_fps:g}", "-q:v", "2", str(output_dir / "frame_%03d.jpg"),
        ],
        check=True,
    )
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video}")
    return frames


def extract_positions(video: Path, output_dir: Path, positions: list[float]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, position in enumerate(positions, start=1):
        output = output_dir / f"frame_{index:03d}_{position:.3f}s.jpg"
        if not output.is_file():
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{position:.6f}", "-i", str(video), "-frames:v", "1",
                    "-q:v", "2", str(output),
                ],
                check=True,
            )
        outputs.append(output)
    return outputs

