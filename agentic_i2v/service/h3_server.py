from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from vtr_bench import paths

from .h3_backend import DELIVERED_FRAMES, FPS, REQUESTED_FRAMES, SEED, H3Backend


HOST = os.environ.get("VTEXTBENCH_H3_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VTEXTBENCH_H3_SERVICE_PORT", "18123"))
PROJECT_ROOT = paths.project_root()
MODELS_ROOT = Path(os.environ.get("VTEXTBENCH_MODELS_ROOT", PROJECT_ROOT / "models")).expanduser().resolve()
OUTPUT_ROOT = (PROJECT_ROOT / "generated_videos").resolve()
RUNS_ROOT = paths.agentic_runs_root(PROJECT_ROOT)


class ServiceState:
    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.generation_lock = threading.Lock()
        self.busy = False
        self.pending = 0
        self.active_request: str | None = None
        self.backend = H3Backend(PROJECT_ROOT, MODELS_ROOT)

    def health(self) -> dict[str, Any]:
        with self.guard:
            return {
                "status": "ready",
                "pid": os.getpid(),
                "busy": self.busy,
                "pending_requests": self.pending,
                "active_request": self.active_request,
                "model": str(self.backend.profile.resolved_model(MODELS_ROOT)),
                "width": self.backend.profile.width,
                "height": self.backend.profile.height,
                "requested_frames": REQUESTED_FRAMES,
                "delivered_frames": DELIVERED_FRAMES,
                "fps": FPS,
                "duration_seconds": DELIVERED_FRAMES / FPS,
                "serialization": "single shared engine; one generation at a time",
                "cpu_offload": bool(
                    self.backend.profile.engine_args.get("enable_cpu_offload", False)
                ),
            }


STATE: ServiceState


def checked_source(value: Any) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Input image is not a file: {path}")
    if not any(path.is_relative_to(root) for root in (PROJECT_ROOT, RUNS_ROOT)):
        raise ValueError("Input image must be within the project or configured runs directory")
    return path


def checked_output(value: Any) -> Path:
    path = Path(str(value)).expanduser().resolve(strict=False)
    if not any(path.is_relative_to(root) for root in (OUTPUT_ROOT, RUNS_ROOT)):
        raise ValueError("Output must be within generated_videos or the configured runs directory")
    if path.suffix.lower() != ".mp4":
        raise ValueError("Output must be an .mp4 file")
    return path


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "VTextBenchH3/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.client_address[0]} {fmt % args}", flush=True)

    def write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.write_json(200, STATE.health()) if self.path == "/health" else self.write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.write_json(404, {"error": "not found"})
            return
        request_id = uuid.uuid4().hex
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            expected = {"seed": SEED, "requested_frames": REQUESTED_FRAMES, "delivered_frames": DELIVERED_FRAMES, "fps": FPS}
            for key, value in expected.items():
                if int(request.get(key, -1)) != value:
                    raise ValueError(f"{key} must be {value}")
            motion_prompt = str(request.get("motion_prompt", "")).strip()
            if not motion_prompt:
                raise ValueError("motion_prompt is required")
            image_path = checked_source(request.get("image_path"))
            output_path = checked_output(request.get("output_path"))
            with STATE.guard:
                STATE.pending += 1
            try:
                with STATE.generation_lock:
                    with STATE.guard:
                        STATE.pending -= 1
                        STATE.busy = True
                        STATE.active_request = request_id
                    metadata = STATE.backend.generate(
                        image_path=image_path, motion_prompt=motion_prompt, output=output_path
                    )
            finally:
                with STATE.guard:
                    if STATE.active_request == request_id:
                        STATE.busy = False
                        STATE.active_request = None
            self.write_json(200, {"status": "complete", "request_id": request_id, "output": metadata["output"], "sha256": metadata["sha256"], "media": metadata["media"]})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.write_json(400, {"status": "error", "type": type(exc).__name__, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.write_json(500, {"status": "error", "type": type(exc).__name__, "error": str(exc)})


def main() -> int:
    global STATE
    STATE = ServiceState()
    server = Server((HOST, PORT), Handler)
    print(f"H3 service listening at http://{HOST}:{PORT}", flush=True)
    server.serve_forever(poll_interval=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
