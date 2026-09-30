from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def validate_deliverable(path: Path, *, frames: int = 240, fps: int = 24) -> dict[str, Any]:
    payload = probe_video(path)
    video = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {path}")
    counted = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    rate = video.get("avg_frame_rate", "0/1")
    numerator, denominator = (int(value) for value in rate.split("/"))
    actual_fps = numerator / denominator if denominator else 0.0
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    if counted != frames:
        raise RuntimeError(f"Expected {frames} frames, found {counted}")
    if abs(actual_fps - fps) > 0.01:
        raise RuntimeError(f"Expected {fps} FPS, found {actual_fps}")
    if abs(duration - frames / fps) > 0.05:
        raise RuntimeError(f"Expected {frames / fps:.3f}s, found {duration:.3f}s")
    return {"frames": counted, "fps": actual_fps, "duration_seconds": duration}

