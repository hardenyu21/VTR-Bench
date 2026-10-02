from __future__ import annotations

import importlib.metadata
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from PIL import Image

from agentic_gen.service.vllm_backend import (
    VideoGenerator,
    _audio_to_numpy,
    _extract_media,
    _peak_memory_mb,
    _profiles,
    _verify_video,
    _video_to_numpy,
    _write_media,
)

from ..media.first_frame_trim import trim_conditioning_frame, trim_leading_audio_frame
from ..state.artifact_store import ArtifactStore, sha256_file


FPS = int(os.environ.get("VTEXTBENCH_H3_FPS", "24"))
REQUESTED_FRAMES = int(os.environ.get("VTEXTBENCH_H3_REQUESTED_FRAMES", "241"))
DELIVERED_FRAMES = REQUESTED_FRAMES - 1
SEED = 42


def agentic_h3_profile():
    base = _profiles()["minimax-h3"]
    width = int(os.environ.get("VTEXTBENCH_H3_WIDTH", str(base.width)))
    height = int(os.environ.get("VTEXTBENCH_H3_HEIGHT", str(base.height)))
    cpu_offload = os.environ.get("VTEXTBENCH_H3_CPU_OFFLOAD", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    return replace(
        base,
        key="minimax-h3-fl2va-agentic-v3-prompt-preserving",
        width=width,
        height=height,
        extra_args={"task": "fl2va", "audio_flow_shift": 3.0, "aspect_ratio": "16:9"},
        engine_args={**base.engine_args, "enable_cpu_offload": cpu_offload},
        notes=(
            "Persistent Agentic I2V H3 service; four GPUs; "
            f"CPU offload={'enabled' if cpu_offload else 'disabled'}."
        ),
    )


def enforce_exact_duration(path: Path, duration_seconds: float) -> None:
    temporary = path.with_name(path.stem + ".exact.tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0",
                "-map", "0:a:0?", "-t", f"{duration_seconds:.6f}", "-c:v", "copy", "-c:a", "aac",
                "-movflags", "+faststart", str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class H3Backend:
    def __init__(self, project_root: Path, models_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.models_root = models_root.resolve()
        self.profile = agentic_h3_profile()
        print(f"Loading persistent MiniMax-H3 engine from {self.profile.resolved_model(self.models_root)}", flush=True)
        self.generator = VideoGenerator(self.profile, self.models_root)
        print("Persistent MiniMax-H3 engine is ready", flush=True)

    def generate(self, *, image_path: Path, motion_prompt: str, output: Path) -> dict[str, Any]:
        metadata_path = output.with_suffix(".metadata.json")
        if output.is_file() and metadata_path.is_file():
            _verify_video(output, DELIVERED_FRAMES)
            import json

            return json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        payload = {"prompt": motion_prompt, "multi_modal_data": {"image": image}}
        sampling = self.profile.build_sampling(seed=SEED, requested_frames=REQUESTED_FRAMES)
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        raw_output = self.generator.omni.generate(payload, sampling)
        elapsed = time.perf_counter() - started
        raw_video, raw_audio, audio_rate = _extract_media(raw_output)
        all_frames = _video_to_numpy(raw_video)
        internal_frames = int(all_frames.shape[0])
        frames = trim_conditioning_frame(
            all_frames, requested=REQUESTED_FRAMES, delivered=DELIVERED_FRAMES
        )
        if int(frames.shape[0]) < DELIVERED_FRAMES:
            raise RuntimeError(
                f"Model returned {frames.shape[0]} post-conditioning frames, "
                f"fewer than required {DELIVERED_FRAMES}"
            )
        frames = frames[:DELIVERED_FRAMES]
        audio = _audio_to_numpy(
            raw_audio, max_samples=round(REQUESTED_FRAMES / FPS * audio_rate)
        )
        audio = trim_leading_audio_frame(
            audio, audio_rate=audio_rate, fps=FPS, frames=DELIVERED_FRAMES
        )
        _write_media(output, frames, audio, FPS, audio_rate)
        if audio is not None:
            enforce_exact_duration(output, DELIVERED_FRAMES / FPS)
        media = _verify_video(output, DELIVERED_FRAMES)
        metadata = {
            "model": str(self.profile.resolved_model(self.models_root)),
            "task": "fl2va",
            "input_image": str(image_path.relative_to(self.project_root)),
            "input_image_sha256": sha256_file(image_path),
            "motion_prompt": motion_prompt,
            "seed": SEED,
            "parameters": {
                "width": self.profile.width,
                "height": self.profile.height,
                "fps": FPS,
                "requested_frames": REQUESTED_FRAMES,
                "dropped_initial_frames": 1,
                "delivered_frames": DELIVERED_FRAMES,
                "duration_seconds": DELIVERED_FRAMES / FPS,
                "internal_generated_frames": internal_frames,
                "steps": self.profile.steps,
                "flow_shift": self.profile.flow_shift,
                "audio_flow_shift": self.profile.extra_args["audio_flow_shift"],
            },
            "parallel": asdict(self.profile.parallel),
            "engine_args": self.profile.engine_args,
            "elapsed_seconds": elapsed,
            "peak_memory_mb": _peak_memory_mb(raw_output),
            "media": media,
            "output": str(output.relative_to(self.project_root)),
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "runtime": {
                "vllm": importlib.metadata.version("vllm"),
                "vllm_omni": importlib.metadata.version("vllm-omni"),
            },
        }
        ArtifactStore.atomic_json(metadata_path, metadata)
        return metadata
