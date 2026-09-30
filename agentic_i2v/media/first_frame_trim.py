from __future__ import annotations

from typing import Any


def trim_leading_audio_frame(audio: Any | None, *, audio_rate: int, fps: int = 24, frames: int = 240) -> Any | None:
    if audio is None:
        return None
    start = round(audio_rate / fps)
    length = round(frames / fps * audio_rate)
    if audio.ndim == 1:
        return audio[start : start + length]
    if audio.ndim == 2 and audio.shape[0] <= 8:
        return audio[:, start : start + length]
    if audio.ndim == 2:
        return audio[start : start + length]
    raise RuntimeError(f"Unexpected normalized audio shape {audio.shape}")


def trim_conditioning_frame(frames: Any, *, requested: int = 241, delivered: int = 240) -> Any:
    if int(frames.shape[0]) < requested:
        raise RuntimeError(f"Expected at least {requested} generated frames, found {frames.shape[0]}")
    result = frames[1:requested]
    if int(result.shape[0]) != delivered:
        raise RuntimeError(f"Expected {delivered} delivered frames, found {result.shape[0]}")
    return result

