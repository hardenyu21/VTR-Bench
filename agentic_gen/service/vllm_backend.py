#!/usr/bin/env python3
"""Installed-package-only runtime helpers for the resident H3 service.

The service imports ``vllm_omni`` from
the active environment and does not add the local vllm-omni checkout to
``sys.path`` or import examples from that checkout.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import vllm_omni
from diffusers.utils import export_to_video
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput



WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def _install_minimax_h3_low_memory_decode() -> None:
    """Avoid full-video CUDA copies in H3's final pixel denormalization.

    The upstream remote VAE flattens all frames, applies an out-of-place
    Normalize, then applies an out-of-place clamp. At 1280x704x243 this makes
    multiple ~2.5 GiB CUDA copies after diffusion has already consumed nearly
    all of an H20. The operations are elementwise, so applying the exact same
    inverse normalization and clamp in-place on the decoded BCHTW tensor is
    numerically equivalent. Postprocessing then transfers directly to FP32 on
    CPU instead of first materializing another full FP32 CUDA tensor.

    This is installed at module import time, including in multiprocessing
    spawn workers, while continuing to import H3 entirely from the installed
    ``vllm_omni`` package.
    """
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
    from vllm_omni.diffusion.models.minimax_h3 import vae as minimax_h3_vae

    if getattr(minimax_h3_vae.MiniMaxH3VideoVAE, "_vtextbench_low_memory_decode", False):
        return

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        decoded = self.model.decode_base(latent * std + mean)
        processor = self.model.processor
        frames = decoded.unsqueeze(2) if decoded.ndim == 4 and processor.use_3d_conv else decoded
        if frames.ndim not in (4, 5) or frames.shape[1] != 3:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")

        inverse_mean = tuple(float(value) for value in processor.transform_rev.mean)
        inverse_std = tuple(float(value) for value in processor.transform_rev.std)
        if len(inverse_mean) != 3 or len(inverse_std) != 3:
            raise ValueError("MiniMax H3 pixel denormalizer must contain three RGB channels")
        for channel, (channel_mean, channel_std) in enumerate(zip(inverse_mean, inverse_std)):
            frames[:, channel].sub_(channel_mean).div_(channel_std)
        frames.clamp_(0, 1)

        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames

    def post_process(output: Any, output_type: str = "np") -> Any:
        if not isinstance(output, tuple) or len(output) != 2:
            return output
        video, audio = output
        if output_type == "latent":
            return output
        if output_type == "np":
            video = video.detach().to(device="cpu", dtype=torch.float32)
            video = video.permute(0, 2, 3, 4, 1).clamp_(0, 1).numpy()
            audio = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
            video = [sample for sample in video]
        return {
            "video": video,
            "audio": audio,
            "audio_sample_rate": pipeline_minimax_h3.MINIMAX_H3_AUDIO_SAMPLE_RATE,
            "fps": pipeline_minimax_h3.MINIMAX_H3_FPS,
        }

    # DiffusionOutput sends this callable through a multiprocessing queue.
    # Give the closure a stable module-level pickle identity in every spawn
    # process before assigning it to the installed pipeline module.
    post_process.__name__ = "_vtextbench_minimax_h3_post_process"
    post_process.__qualname__ = "_vtextbench_minimax_h3_post_process"
    post_process.__module__ = __name__
    globals()[post_process.__name__] = post_process
    decode_latent._vtextbench_low_memory_decode = True
    minimax_h3_vae.MiniMaxH3VideoVAE.decode_latent = decode_latent
    minimax_h3_vae.MiniMaxH3VideoVAE._vtextbench_low_memory_decode = True
    pipeline_minimax_h3._minimax_h3_post_process = post_process


_install_minimax_h3_low_memory_decode()


def _install_high_resolution_async_output_timeout() -> None:
    """Allow high-resolution VAE output transfer to finish.

    vLLM-Omni 0.26.0 hard-codes a 30-second wait for asynchronous worker
    output.  LTX-2.3 at 1280x704x241 completes denoising successfully but its
    VAE/D2H result transfer can exceed that limit.  Raising the client-side
    wait changes neither sampling nor model output.
    """
    from vllm_omni.diffusion import diffusion_engine

    diffusion_engine._ASYNC_OUTPUT_TIMEOUT = max(
        float(diffusion_engine._ASYNC_OUTPUT_TIMEOUT),
        300.0,
    )


_install_high_resolution_async_output_timeout()


@dataclass(frozen=True)
class ParallelPlan:
    gpus: int = 1
    tensor_parallel_size: int = 1
    cfg_parallel_size: int = 1
    ulysses_degree: int = 1
    ulysses_mode: str | None = None
    ring_degree: int = 1
    vae_patch_parallel_size: int = 1
    text_encoder_tp_size: int = 1
    vae_parallel_mode: str = "tile"
    use_hsdp: bool = False
    hsdp_shard_size: int = 1
    hsdp_replicate_size: int = 1

    def build(self) -> DiffusionParallelConfig:
        kwargs: dict[str, Any] = dict(
            tensor_parallel_size=self.tensor_parallel_size,
            cfg_parallel_size=self.cfg_parallel_size,
            ulysses_degree=self.ulysses_degree,
            ring_degree=self.ring_degree,
            vae_patch_parallel_size=self.vae_patch_parallel_size,
            text_encoder_tp_size=self.text_encoder_tp_size,
            vae_parallel_mode=self.vae_parallel_mode,
            use_hsdp=self.use_hsdp,
            hsdp_shard_size=self.hsdp_shard_size,
            hsdp_replicate_size=self.hsdp_replicate_size,
        )
        if self.ulysses_mode is not None:
            kwargs["ulysses_mode"] = self.ulysses_mode
        return DiffusionParallelConfig(**kwargs)


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_dir: str
    width: int
    height: int
    steps: int
    fps: int = 24
    guidance_scale: float | None = None
    guidance_scale_2: float | None = None
    flow_shift: float | None = None
    boundary_ratio: float | None = None
    negative_prompt: str | None = None
    model_class_name: str | None = None
    parallel: ParallelPlan = field(default_factory=ParallelPlan)
    extra_args: dict[str, Any] = field(default_factory=dict)
    engine_args: dict[str, Any] = field(default_factory=dict)
    internal_num_frames: int | None = None
    output_type: str | None = None
    enforce_eager: bool = True
    vae_use_tiling: bool = True
    notes: str = ""

    def resolved_model(self, models_root: Path) -> Path:
        return models_root / self.model_dir

    def build_prompt(self, prompt_suite: dict[str, Any]) -> dict[str, str]:
        if self.model_class_name == "LingBotVideoPipeline":
            caption = prompt_suite["lingbot_caption"]
            # LingBot is trained with a structured caption.  Match the official
            # loader exactly: preserve the complete object and serialize it as
            # compact JSON instead of reducing it to one nested field or using
            # Python's single-quoted ``str(dict)`` representation.
            if isinstance(caption, (dict, list)):
                prompt = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
            else:
                prompt = str(caption)
        else:
            # Final VTextBench prompt records expose the canonical generation
            # text as ``prompt_en``.  Older benchmark records use
            # ``plain_prompt``; accept both without requiring a disposable
            # adapter file for an otherwise immutable prompt record.
            plain_prompt = prompt_suite.get("plain_prompt", prompt_suite.get("prompt_en"))
            if plain_prompt is None:
                raise KeyError("prompt suite must contain plain_prompt or prompt_en")
            prompt = str(plain_prompt)
        payload = {"prompt": prompt}
        if self.negative_prompt is not None:
            payload["negative_prompt"] = self.negative_prompt
        return payload

    def build_sampling(self, *, seed: int, requested_frames: int) -> OmniDiffusionSamplingParams:
        generated_frames = self.generated_frame_count(requested_frames)
        kwargs: dict[str, Any] = {
            "height": self.height,
            "width": self.width,
            "num_frames": generated_frames,
            "num_inference_steps": self.steps,
            "seed": seed,
            "fps": self.fps,
            "extra_args": dict(self.extra_args),
        }
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        if self.guidance_scale_2 is not None:
            kwargs["guidance_scale_2"] = self.guidance_scale_2
        if self.output_type is not None:
            kwargs["output_type"] = self.output_type
        return OmniDiffusionSamplingParams(**kwargs)

    def generated_frame_count(self, requested_frames: int) -> int:
        """Return the native frame count needed to deliver the requested clip.

        MiniMax H3 accepts only 17n+5 frame counts.  The original profile used
        a fixed 124-frame value for every request, which silently limited H3 to
        five-second clips even when the CLI requested a longer video.  Preserve
        124 as the five-second minimum, then align longer requests upward.
        """
        if self.model_class_name == "MiniMaxH3Pipeline":
            generated_frames = max(requested_frames, self.internal_num_frames or 0)
            while generated_frames % 17 != 5:
                generated_frames += 1
            return generated_frames
        return self.internal_num_frames or requested_frames

    def build_engine(self, models_root: Path) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": str(self.resolved_model(models_root)),
            # Some single-stage deploy templates default to one device even when
            # a non-trivial parallel_config is supplied.  Keep the requested
            # worker count explicit so the topology is actually instantiated.
            "num_gpus": self.parallel.gpus,
            "parallel_config": self.parallel.build(),
            "enforce_eager": self.enforce_eager,
            "vae_use_tiling": self.vae_use_tiling,
        }
        if self.model_class_name:
            kwargs["model_class_name"] = self.model_class_name
        if self.flow_shift is not None:
            kwargs["flow_shift"] = self.flow_shift
        if self.boundary_ratio is not None:
            kwargs["boundary_ratio"] = self.boundary_ratio
        kwargs.update(self.engine_args)
        return kwargs


def _profiles() -> dict[str, ModelProfile]:
    # Parallelize both the tensor computation and the two CFG paths.
    four_gpu_hunyuan = ParallelPlan(gpus=4, tensor_parallel_size=2, cfg_parallel_size=2, vae_patch_parallel_size=4)
    return {
        "hunyuan-480p": ModelProfile(
            key="hunyuan-480p",
            model_dir="HunyuanVideo-1.5-Diffusers-480p_t2v",
            width=832,
            height=480,
            steps=50,
            guidance_scale=6.0,
            flow_shift=5.0,
            parallel=four_gpu_hunyuan,
            # Cold-loading the 50 GB checkpoint from NAS can exceed Omni's
            # 600-second orchestrator deadline even while shard loading is
            # still progressing.  This changes startup tolerance only; model
            # parameters and measured generation time remain unchanged.
            engine_args={"init_timeout": 7200, "stage_init_timeout": 7200},
            notes="Official 480p size, 50 steps, CFG 6, flow shift 5.",
        ),
        "hunyuan-720p": ModelProfile(
            key="hunyuan-720p",
            model_dir="HunyuanVideo-1.5-Diffusers-720p_t2v",
            width=1280,
            height=720,
            steps=50,
            guidance_scale=6.0,
            flow_shift=9.0,
            parallel=four_gpu_hunyuan,
            notes="Official 720p size, 50 steps, CFG 6, flow shift 9.",
        ),
        "ltx-2.3": ModelProfile(
            key="ltx-2.3",
            model_dir="LTX-2.3-Diffusers",
            width=768,
            height=512,
            steps=30,
            guidance_scale=3.0,
            # Keep the postprocessed video as a tensor until vLLM-Omni's
            # explicit SHM transport packs it. The default ``np`` path turns
            # a 241-frame 720p-class result into a ~2.6 GiB ndarray that the
            # 0.26.0 SHM packer does not recognize and intermittently wedges
            # MessageQueue serialization. ``pt`` applies the same pixel
            # denormalization and changes only the IPC representation.
            output_type="pt",
            parallel=ParallelPlan(gpus=4, tensor_parallel_size=2, cfg_parallel_size=2, vae_patch_parallel_size=4),
            notes="Official T2VA profile: 768x512, 30 steps, CFG 3.",
        ),
        "ltx-2.3-720p": ModelProfile(
            key="ltx-2.3-720p",
            model_dir="LTX-2.3-Diffusers",
            # LTX-2.3 requires both dimensions to be divisible by 32.  Use the
            # project's accepted 720p-class canvas rather than invalid 1280x720.
            width=1280,
            height=704,
            steps=30,
            guidance_scale=3.0,
            output_type="pt",
            parallel=ParallelPlan(gpus=4, tensor_parallel_size=2, cfg_parallel_size=2, vae_patch_parallel_size=4),
            notes=(
                "LTX-2.3 native 720p-class T2VA profile: 1280x704 (32-aligned), "
                "30 steps, CFG 3."
            ),
        ),
        "wan2.2-a14b": ModelProfile(
            key="wan2.2-a14b",
            model_dir="Wan2.2-T2V-A14B-Diffusers",
            width=1280,
            height=720,
            steps=40,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            flow_shift=5.0,
            boundary_ratio=0.875,
            negative_prompt=WAN_NEGATIVE_PROMPT,
            parallel=ParallelPlan(
                gpus=4,
                ulysses_degree=2,
                cfg_parallel_size=2,
                vae_patch_parallel_size=4,
                use_hsdp=True,
                hsdp_shard_size=4,
            ),
            # Loading both A14B transformer stages from NAS can legitimately
            # exceed Omni's 600-second orchestration defaults on a cold cache.
            # These limits affect startup tolerance only, not sampling.
            engine_args={"init_timeout": 1800, "stage_init_timeout": 1800},
            notes=(
                "Official 720p dual-DiT parameters; frame count is supplied "
                "per request."
            ),
        ),
        "wan2.2-5b": ModelProfile(
            key="wan2.2-5b",
            model_dir="Wan2.2-TI2V-5B-Diffusers",
            width=1280,
            height=704,
            steps=50,
            guidance_scale=5.0,
            flow_shift=5.0,
            boundary_ratio=0.875,
            negative_prompt=WAN_NEGATIVE_PROMPT,
            parallel=ParallelPlan(gpus=4, ulysses_degree=2, cfg_parallel_size=2, vae_patch_parallel_size=4),
            notes=(
                "Official 1280x704 T2V profile: 50 steps, CFG 5, boundary ratio 0.875; frame count "
                "is supplied per request."
            ),
        ),
        "lingbot-dense": ModelProfile(
            key="lingbot-dense",
            model_dir="lingbot-video-dense-1.3b",
            width=832,
            height=480,
            steps=40,
            guidance_scale=3.0,
            flow_shift=3.0,
            model_class_name="LingBotVideoPipeline",
            parallel=ParallelPlan(gpus=4, ulysses_degree=4, ulysses_mode="advanced_uaa"),
            vae_use_tiling=False,
            engine_args={"init_timeout": 1800, "stage_init_timeout": 1800},
            notes="Official structured-caption base profile without refiner.",
        ),
        "lingbot-moe": ModelProfile(
            key="lingbot-moe",
            model_dir="lingbot-video-moe-30b-a3b",
            width=832,
            height=480,
            steps=40,
            guidance_scale=3.0,
            flow_shift=3.0,
            model_class_name="LingBotVideoPipeline",
            parallel=ParallelPlan(gpus=4, ulysses_degree=4, ulysses_mode="advanced_uaa"),
            vae_use_tiling=False,
            # The 30B MoE checkpoint contains 713 weight entries per worker.
            # Four workers cold-loading these shards from NAS exceeded the
            # former 30-minute limit at roughly 53%, even though loading was
            # still making progress.  This only extends startup tolerance; it
            # does not change sampling parameters or benchmark timing.
            engine_args={"init_timeout": 7200, "stage_init_timeout": 7200},
            notes="vLLM-Omni base MoE path; refiner is not supported by the installed runtime.",
        ),
        "minimax-h3": ModelProfile(
            key="minimax-h3",
            model_dir="MiniMax-H3/FL2VA",
            width=1344,
            height=768,
            steps=50,
            flow_shift=12.0,
            model_class_name="MiniMaxH3Pipeline",
            parallel=ParallelPlan(
                gpus=4,
                ulysses_degree=4,
                vae_patch_parallel_size=4,
                text_encoder_tp_size=4,
                vae_parallel_mode="tile",
            ),
            extra_args={"task": "t2va", "audio_flow_shift": 3.0, "aspect_ratio": "16:9"},
            # Four workers cold-loading the H3 text encoder and DiT shards from
            # NAS can exceed Omni's default 600-second startup window.  These
            # values affect startup tolerance only, not sampling or timing.
            engine_args={
                "init_timeout": 1800,
                "stage_init_timeout": 1800,
                # Keep the 63 GB Qwen3-VL encoder and 62 GB DiT mutually
                # exclusive on GPU. Without this, 1344x768x241 leaves less
                # than 1 GiB for subsequent-request activations.
                "enable_cpu_offload": True,
            },
            internal_num_frames=124,
            enforce_eager=False,
            notes=(
                "Official four-GPU Ulysses4 + text-encoder TP4 + VAE tile4 profile. "
                "H3 requests are aligned upward to 17n+5 frames, then video/audio are trimmed "
                "to the requested output length."
            ),
        ),
        "minimax-h3-long": ModelProfile(
            key="minimax-h3-long",
            model_dir="MiniMax-H3/FL2VA",
            # H3 requires spatial dimensions aligned to a multiple of 32.
            width=1280,
            height=704,
            steps=50,
            flow_shift=12.0,
            model_class_name="MiniMaxH3Pipeline",
            parallel=ParallelPlan(
                gpus=4,
                ulysses_degree=4,
                vae_patch_parallel_size=4,
                text_encoder_tp_size=4,
                vae_parallel_mode="tile",
            ),
            extra_args={"task": "t2va", "audio_flow_shift": 3.0, "aspect_ratio": "16:9"},
            engine_args={
                "init_timeout": 1800,
                "stage_init_timeout": 1800,
                "enable_cpu_offload": True,
            },
            internal_num_frames=124,
            enforce_eager=False,
            notes=(
                "Long-form H3 profile: official 50-step/flow-shift-12 four-GPU topology "
                "with a native 1280x704 (32-aligned) canvas. Frame requests are aligned "
                "dynamically to 17n+5; the native canvas is delivered without resizing."
            ),
        ),
    }


def _peak_memory_mb(output: Any) -> float:
    value = output[0] if isinstance(output, list) and output else output
    peak = getattr(value, "peak_memory_mb", 0.0) if value is not None else 0.0
    if not peak and value is not None:
        inner = getattr(value, "request_output", None)
        if isinstance(inner, list):
            inner = inner[0] if inner else None
        peak = getattr(inner, "peak_memory_mb", 0.0) if inner is not None else 0.0
    return float(peak or 0.0)


def _extract_media(output: Any) -> tuple[Any, Any | None, int]:
    audio = None
    audio_rate = 24000
    value = output[0] if isinstance(output, list) and len(output) == 1 else output
    if isinstance(value, OmniRequestOutput):
        mm = value.multimodal_output or {}
        audio = mm.get("audio")
        audio_rate = int(mm.get("audio_sample_rate", audio_rate))
        if value.is_pipeline_output and value.request_output is not None:
            value = value.request_output
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
        if isinstance(value, OmniRequestOutput):
            mm = value.multimodal_output or {}
            audio = mm.get("audio", audio)
            audio_rate = int(mm.get("audio_sample_rate", audio_rate))
            if not value.images:
                raise RuntimeError("No video frames found in OmniRequestOutput")
            value = value.images[0] if len(value.images) == 1 else value.images
    if isinstance(value, tuple) and len(value) == 2:
        value, tuple_audio = value
        audio = tuple_audio if audio is None else audio
    if isinstance(value, dict):
        audio = value.get("audio", audio)
        audio_rate = int(value.get("audio_sample_rate", audio_rate))
        value = value.get("frames", value.get("video"))
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (np.ndarray, torch.Tensor)):
        if value[0].ndim in (4, 5):
            value = value[0]
    return value, audio, audio_rate


def _frame_to_numpy(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (3, 4):
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / 255.0
    array = array.astype(np.float32, copy=False)
    if float(array.min()) < 0.0:
        array = array.clip(-1.0, 1.0) * 0.5 + 0.5
    return array.clip(0.0, 1.0)


def _video_to_numpy(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        # Diffusers' native ``output_type="np"`` path casts decoded frames to
        # FP32 before NumPy conversion.  LTX uses ``output_type="pt"`` so its
        # large result can travel through vLLM-Omni's tensor-aware SHM path;
        # mirror the same FP32 cast here because NumPy cannot represent BF16.
        video = video.detach().to(device="cpu", dtype=torch.float32).numpy()
    if isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], (list, np.ndarray, torch.Tensor)):
            candidate = video[0]
            candidate_ndim = getattr(candidate, "ndim", None)
            if isinstance(candidate, list) or candidate_ndim in (4, 5):
                video = candidate
        if isinstance(video, list):
            return np.stack([_frame_to_numpy(frame) for frame in video], axis=0)
    array = np.asarray(video)
    if array.ndim == 5:
        array = array[0]
    if array.ndim != 4:
        raise RuntimeError(f"Unexpected generated video shape {array.shape}")
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = array.transpose(1, 2, 3, 0)
    return np.stack([_frame_to_numpy(frame) for frame in array], axis=0)


def _audio_to_numpy(audio: Any | None, *, max_samples: int) -> np.ndarray | None:
    if audio is None:
        return None
    if isinstance(audio, list):
        audio = audio[0] if audio else None
    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    array = np.asarray(audio, dtype=np.float32).squeeze()
    if array.ndim == 1:
        return array[:max_samples]
    if array.ndim == 2 and array.shape[0] <= 8:
        return array[:, :max_samples]
    if array.ndim == 2:
        return array[:max_samples]
    raise RuntimeError(f"Unexpected generated audio shape {array.shape}")


def _write_media(path: Path, frames: np.ndarray, audio: np.ndarray | None, fps: int, audio_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio is None:
        export_to_video(list(frames), str(path), fps=fps)
        return
    frames_u8 = (frames.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.write_bytes(mux_video_audio_bytes(frames_u8, audio, fps=float(fps), audio_sample_rate=audio_rate))


def _verify_video(path: Path, expected_frames: int) -> dict[str, Any]:
    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        decoded_frames = sum(1 for _ in container.decode(video=0))
        audio_streams = len(container.streams.audio)
        metadata = {
            "codec": video_stream.codec_context.name,
            "width": video_stream.width,
            "height": video_stream.height,
            "fps": float(video_stream.average_rate),
            "decoded_frames": decoded_frames,
            "audio_streams": audio_streams,
        }
    if decoded_frames != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} decoded frames, got {decoded_frames}: {path}")
    return metadata


class VideoGenerator:
    def __init__(self, profile: ModelProfile, models_root: Path):
        self.profile = profile
        self.models_root = models_root
        self._stage_config_dir: tempfile.TemporaryDirectory[str] | None = None
        model_path = profile.resolved_model(models_root)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        if torch.cuda.device_count() < profile.parallel.gpus:
            raise RuntimeError(
                f"Profile {profile.key} needs {profile.parallel.gpus} visible GPUs; "
                f"found {torch.cuda.device_count()}"
            )
        engine_kwargs = profile.build_engine(models_root)
        needs_explicit_stage = profile.key.startswith(("hunyuan-", "wan2.2-"))
        if profile.parallel.gpus > 1 and needs_explicit_stage:
            # A few built-in deploy templates (notably HunyuanVideo and Wan)
            # otherwise retain their single-GPU `devices` entry even when an
            # overriding parallel_config is supplied to Omni.  Materialize an
            # ephemeral stage config so worker placement and topology cannot
            # silently diverge from the profile metadata.
            self._stage_config_dir = tempfile.TemporaryDirectory(prefix="vtextbench_t2v_")
            stage_config_path = Path(self._stage_config_dir.name) / "stage_config.json"
            parallel = asdict(profile.parallel)
            parallel.pop("gpus")
            parallel = {key: value for key, value in parallel.items() if value is not None}
            stage_config = {
                "async_chunk": False,
                "trust_remote_code": True,
                "distributed_executor_backend": "mp",
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": ",".join(str(index) for index in range(profile.parallel.gpus)),
                        "max_num_seqs": 1,
                        "parallel_config": parallel,
                        "enforce_eager": profile.enforce_eager,
                        "vae_use_tiling": profile.vae_use_tiling,
                    }
                ],
            }
            stage_config_path.write_text(json.dumps(stage_config), encoding="utf-8")
            engine_kwargs["stage_configs_path"] = str(stage_config_path)
        try:
            self.omni = Omni(**engine_kwargs)
        except Exception:
            if self._stage_config_dir is not None:
                self._stage_config_dir.cleanup()
            raise

    def close(self) -> None:
        try:
            self.omni.close()
        finally:
            if self._stage_config_dir is not None:
                self._stage_config_dir.cleanup()

    def generate(
        self,
        *,
        prompt_suite: dict[str, Any],
        output: Path,
        seed: int,
        requested_frames: int,
    ) -> dict[str, Any]:
        prompt = self.profile.build_prompt(prompt_suite)
        sampling = self.profile.build_sampling(seed=seed, requested_frames=requested_frames)
        start = time.perf_counter()
        raw_output = self.omni.generate(prompt, sampling)
        elapsed = time.perf_counter() - start
        peak_mb = _peak_memory_mb(raw_output)
        raw_video, raw_audio, audio_rate = _extract_media(raw_output)
        frames = _video_to_numpy(raw_video)
        internal_frames = int(frames.shape[0])
        if internal_frames < requested_frames:
            raise RuntimeError(f"Model returned {internal_frames} frames, fewer than requested {requested_frames}")
        frames = frames[:requested_frames]
        audio = _audio_to_numpy(raw_audio, max_samples=round(requested_frames / self.profile.fps * audio_rate))
        _write_media(output, frames, audio, self.profile.fps, audio_rate)
        media = _verify_video(output, requested_frames)
        return {
            "profile": self.profile.key,
            "prompt_id": prompt_suite.get("id"),
            "prompt_sha256": hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest(),
            "model": str(self.profile.resolved_model(self.models_root)),
            "requested_frames": requested_frames,
            "internal_generated_frames": internal_frames,
            "trimmed_frames": internal_frames - requested_frames,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "peak_memory_mb": peak_mb,
            "parameters": {
                "width": self.profile.width,
                "height": self.profile.height,
                "fps": self.profile.fps,
                "steps": self.profile.steps,
                "guidance_scale": self.profile.guidance_scale,
                "guidance_scale_2": self.profile.guidance_scale_2,
                "flow_shift": self.profile.flow_shift,
                "boundary_ratio": self.profile.boundary_ratio,
                "extra_args": self.profile.extra_args,
                "output_type": self.profile.output_type,
            },
            "parallel": asdict(self.profile.parallel),
            "memory_management": {
                "enable_cpu_offload": bool(self.profile.engine_args.get("enable_cpu_offload", False)),
                "enable_layerwise_offload": bool(self.profile.engine_args.get("enable_layerwise_offload", False)),
                "enable_distributed_layerwise_offload": bool(
                    self.profile.engine_args.get("enable_distributed_layerwise_offload", False)
                ),
                "pin_cpu_memory": bool(self.profile.engine_args.get("pin_cpu_memory", True)),
            },
            "runtime": {
                "vllm_omni_version": importlib.metadata.version("vllm-omni"),
                "vllm_omni_import": str(Path(vllm_omni.__file__).resolve()),
                "torch_version": torch.__version__,
            },
            "media": media,
            "output": str(output.resolve()),
            "notes": self.profile.notes,
        }
