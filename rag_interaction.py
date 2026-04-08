from __future__ import annotations

import argparse
import re
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import requests

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.components.processors.classifier_options import ClassifierOptions
except ImportError:
    mp = None
    np = None
    python = None
    vision = None
    ClassifierOptions = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import zmq
except ImportError:
    zmq = None

VIDEO_CLIENT_IMPORT_ERROR: Optional[BaseException] = None
try:
    from rag_interaction_video_client import (
        DEFAULT_CAMERA_ID as DEFAULT_VIDEO_CAMERA_ID,
        DEFAULT_FPS as DEFAULT_VIDEO_FPS,
        DEFAULT_HEIGHT as DEFAULT_VIDEO_HEIGHT,
        DEFAULT_JPEG_QUALITY as DEFAULT_VIDEO_JPEG_QUALITY,
        DEFAULT_PORT as DEFAULT_VIDEO_PORT,
        DEFAULT_SERVER_HOST as DEFAULT_VIDEO_SERVER_HOST,
        DEFAULT_SOURCE as DEFAULT_VIDEO_SOURCE,
        DEFAULT_TOPIC as DEFAULT_VIDEO_TOPIC,
        DEFAULT_WIDTH as DEFAULT_VIDEO_WIDTH,
        ZmqFramePublisher,
        stream_realsense as stream_video_realsense,
        stream_webcam as stream_video_webcam,
    )
except Exception as exc:
    VIDEO_CLIENT_IMPORT_ERROR = exc
    DEFAULT_VIDEO_SERVER_HOST = "127.0.0.1"
    DEFAULT_VIDEO_PORT = 5550
    DEFAULT_VIDEO_TOPIC = "rgb"
    DEFAULT_VIDEO_SOURCE = "webcam"
    DEFAULT_VIDEO_CAMERA_ID = 0
    DEFAULT_VIDEO_WIDTH = 640
    DEFAULT_VIDEO_HEIGHT = 480
    DEFAULT_VIDEO_FPS = 30
    DEFAULT_VIDEO_JPEG_QUALITY = 80
    ZmqFramePublisher = None
    stream_video_realsense = None
    stream_video_webcam = None

GESTURE_SCROLLING_IMPORT_ERROR: Optional[BaseException] = None
try:
    from gesture_scrolling import (
        DEFAULT_FPS,
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        DEFAULT_GESTURE_HOLD_SEC,
        DEFAULT_INITIAL_MEAN_WINDOW,
        DEFAULT_MIN_GESTURE_SCORE,
        DEFAULT_ZONE_WIDTH,
        STATE_NONE,
        compute_zone_bounds,
        default_model_path as default_gesture_model_path,
        draw_overlay,
        get_thumb_tip_x_px,
        get_thumb_up_hand_index,
        mean_int,
        resolve_state,
    )
except Exception as exc:
    GESTURE_SCROLLING_IMPORT_ERROR = exc
    DEFAULT_FPS = 30
    DEFAULT_FRAME_HEIGHT = 480
    DEFAULT_FRAME_WIDTH = 640
    DEFAULT_GESTURE_HOLD_SEC = 0.35
    DEFAULT_INITIAL_MEAN_WINDOW = 5
    DEFAULT_MIN_GESTURE_SCORE = 0.5
    DEFAULT_ZONE_WIDTH = 120
    STATE_NONE = "none"

    def default_gesture_model_path() -> str:
        return "gesture_recognizer.task"

    def _missing_gesture_dependency(*_args, **_kwargs):
        raise RuntimeError(
            "Gesture client dependencies are unavailable. "
            f"Original import error: {GESTURE_SCROLLING_IMPORT_ERROR}"
        )

    compute_zone_bounds = _missing_gesture_dependency
    draw_overlay = _missing_gesture_dependency
    get_thumb_tip_x_px = _missing_gesture_dependency
    get_thumb_up_hand_index = _missing_gesture_dependency
    mean_int = _missing_gesture_dependency
    resolve_state = _missing_gesture_dependency


REALTIME_ASR_IMPORT_ERROR: Optional[BaseException] = None
try:
    from realtime_asr_vad_pyaudio import (
        DEFAULT_CHUNK,
        DEFAULT_ENERGY_THRESHOLD_RMS,
        DEFAULT_INPUT_RATE,
        MODEL_AUDIO_RATE,
        DEFAULT_VAD_THRESHOLD,
        DEFAULT_WHISPER_MODEL,
        RealTimeMicAsr,
        list_input_devices,
        open_microphone_stream,
    )
except Exception as exc:
    REALTIME_ASR_IMPORT_ERROR = exc
    DEFAULT_CHUNK = 1024
    DEFAULT_ENERGY_THRESHOLD_RMS = 250.0
    DEFAULT_INPUT_RATE = 48000
    MODEL_AUDIO_RATE = 16000
    DEFAULT_VAD_THRESHOLD = 0.35
    DEFAULT_WHISPER_MODEL = "openai/whisper-small"
    RealTimeMicAsr = None

    def list_input_devices() -> None:
        raise RuntimeError(
            "Speech client dependencies are unavailable. "
            f"Original import error: {REALTIME_ASR_IMPORT_ERROR}"
        )

    def open_microphone_stream(*_args, **_kwargs):
        raise RuntimeError(
            "Speech client dependencies are unavailable. "
            f"Original import error: {REALTIME_ASR_IMPORT_ERROR}"
        )


DEFAULT_SERVER_URL = "http://192.168.50.4:5550"
DEFAULT_HTTP_TIMEOUT_SEC = 10
DEFAULT_COOLDOWN_SEC = 5.0
DEFAULT_MODE = "client"
DEFAULT_SERVER_HOST = "192.168.50.4"
DEFAULT_SERVER_PORT = 5550
DEFAULT_MAX_STORED_MESSAGES = 200
DEFAULT_GESTURE_VIDEO_SOURCE = "auto"
DEFAULT_VIDEO_BIND_HOST = "0.0.0.0"
DEFAULT_VIDEO_STREAM_TIMEOUT_MS = 1000
VALID_DIRECTIONS = {"left", "right", "close", "open"}
ROUTE_COMMAND = "command"
ROUTE_QUERY = "query"


def normalize_simple_command(command: Any) -> Optional[str]:
    if not isinstance(command, str):
        return None

    normalized = " ".join(command.strip().lower().split())
    if normalized in VALID_DIRECTIONS:
        return normalized
    return None


class InteractionCoordinator:
    def __init__(
        self,
        server_url: str,
        cooldown_sec: float,
        http_timeout_sec: int,
    ):
        self._server_base_url = build_server_base_url(server_url)
        self._command_url = build_endpoint_url(server_url, ROUTE_COMMAND)
        self._query_url = build_endpoint_url(server_url, ROUTE_QUERY)
        self._cooldown_sec = cooldown_sec
        self._http_timeout_sec = http_timeout_sec

        self._lock = threading.Lock()
        self._latest_direction = STATE_NONE
        self._retry_not_before_by_route = {
            ROUTE_COMMAND: 0.0,
            ROUTE_QUERY: 0.0,
        }

    @property
    def server_base_url(self) -> str:
        return self._server_base_url

    def time_until_ready(self, route: str) -> float:
        with self._lock:
            return max(0.0, self._retry_not_before_by_route[route] - time.monotonic())

    def update_direction(self, direction: str) -> None:
        if not isinstance(direction, str):
            return

        direction = " ".join(direction.strip().lower().split())
        if direction not in VALID_DIRECTIONS and direction != STATE_NONE:
            return

        with self._lock:
            if direction == self._latest_direction:
                return
            self._latest_direction = direction

        normalized_direction = normalize_simple_command(direction)
        if normalized_direction is None:
            return

        self._post_route(
            route=ROUTE_COMMAND,
            url=self._command_url,
            payload={"command": normalized_direction},
        )

    def update_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return

        command = normalize_simple_command(prompt)
        if command is not None:
            print(
                f"[RAG_CLIENT][RECOGNIZED_COMMAND] {command}",
                flush=True,
            )
            self._post_route(
                route=ROUTE_COMMAND,
                url=self._command_url,
                payload={"command": command},
            )
            return

        print(f"[RAG_CLIENT][RECOGNIZED_QUERY] {prompt}", flush=True)
        self._post_route(
            route=ROUTE_QUERY,
            url=self._query_url,
            payload={"query": prompt},
        )

    def _post_route(
        self,
        route: str,
        url: str,
        payload: dict[str, str],
    ) -> bool:
        retry_remaining = self.time_until_ready(route)
        if retry_remaining > 0:
            print(
                f"[RAG_CLIENT][SKIP][{route.upper()}] cooldown={retry_remaining:.1f}s payload={payload}",
                flush=True,
            )
            return False

        try:
            print(f"[RAG_CLIENT][SENDING][{route.upper()}] url={url} payload={payload}", flush=True)
            response = requests.post(
                url,
                json=payload,
                timeout=self._http_timeout_sec,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(
                f"[RAG_CLIENT][ERROR][{route.upper()}] POST failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            with self._lock:
                self._retry_not_before_by_route[route] = time.monotonic() + self._cooldown_sec
            return False

        with self._lock:
            self._retry_not_before_by_route[route] = 0.0

        print(
            f"[RAG_CLIENT][POST][{route.upper()}] status={response.status_code} body={truncate_text(response.text)}",
            flush=True,
        )
        return True


def build_server_base_url(server_url: str) -> str:
    clean_url = server_url.rstrip("/")
    for suffix in ("/command", "/query", "/interact", "/health"):
        if clean_url.endswith(suffix):
            return clean_url[: -len(suffix)]
    return clean_url


def build_endpoint_url(server_url: str, endpoint: str) -> str:
    return build_server_base_url(server_url) + f"/{endpoint}"


def build_health_url(server_url: str) -> str:
    return build_server_base_url(server_url) + "/health"


def check_server_health(server_url: str, http_timeout_sec: int) -> None:
    health_url = build_health_url(server_url)
    try:
        response = requests.get(health_url, timeout=http_timeout_sec)
        response.raise_for_status()
        preview = response.text.strip()
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(
            f"[RAG_CLIENT][SERVER_HEALTH] OK url={health_url} status={response.status_code} body={preview}",
            flush=True,
        )
    except requests.RequestException as exc:
        print(
            f"[RAG_CLIENT][SERVER_HEALTH][ERROR] url={health_url} error={exc}",
            file=sys.stderr,
            flush=True,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_text(text: str, limit: int = 200) -> str:
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def build_missing_fastapi_message() -> str:
    return (
        "FastAPI server mode requires 'fastapi' and 'uvicorn'. Install them with:\n"
        "  /home/dzmitry/gesture_real_time_control/.venv/bin/pip install fastapi uvicorn"
    )


def require_fastapi_dependencies() -> tuple[Any, Any]:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError(build_missing_fastapi_message()) from exc
    return FastAPI, HTTPException


def require_uvicorn() -> Any:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(build_missing_fastapi_message()) from exc
    return uvicorn


def build_missing_video_client_message() -> str:
    return (
        "Video client mode requires rag_interaction_video_client.py and its dependencies. "
        f"Original import error: {VIDEO_CLIENT_IMPORT_ERROR}"
    )


class InteractionRouterService:
    def __init__(
        self,
        max_stored_messages: int,
    ):
        self._lock = threading.Lock()
        max_messages = max(1, max_stored_messages)
        self._commands: deque[dict[str, Any]] = deque(maxlen=max_messages)
        self._queries: deque[dict[str, Any]] = deque(maxlen=max_messages)

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                "commands": len(self._commands),
                "queries": len(self._queries),
            }
        return {
            "ok": True,
            "status": "healthy",
            "routes": ["/command", "/query", "/health"],
            "counts": counts,
        }

    def record_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        command = normalize_simple_command(payload.get("command"))
        if command is None:
            raise ValueError(
                f"Field 'command' must be one of {sorted(VALID_DIRECTIONS)}"
            )

        record = {
            "command": command,
            "received_at_utc": utc_now_iso(),
        }

        with self._lock:
            self._commands.appendleft(record)

        print(f"[RAG_SERVER][COMMAND] {record}", flush=True)
        return record

    def record_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")

        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Field 'query' must be a non-empty string")

        record = {
            "query": query.strip(),
            "received_at_utc": utc_now_iso(),
        }

        with self._lock:
            self._queries.appendleft(record)

        print(f"[RAG_SERVER][QUERY] {record}", flush=True)
        return record


def create_fastapi_app(args: argparse.Namespace) -> Any:
    FastAPI, HTTPException = require_fastapi_dependencies()
    service = InteractionRouterService(
        max_stored_messages=args.max_stored_messages,
    )
    app = FastAPI(
        title="RAG Interaction Server",
        version="0.3.0",
        description="Minimal FastAPI server with /command and /query endpoints.",
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "ok": True,
            "routes": ["/command", "/query", "/health"],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health_payload()

    @app.post("/command")
    def command(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = service.record_command(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "route": ROUTE_COMMAND, "data": record}

    @app.post("/query")
    def query(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = service.record_query(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "route": ROUTE_QUERY, "data": record}

    return app


def run_server(args: argparse.Namespace) -> int:
    uvicorn = require_uvicorn()
    app = create_fastapi_app(args)
    print(
        "[RAG_SERVER] Starting FastAPI server "
        f"host={args.host} port={args.port} routes=/command,/query",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_video_client_mode(args: argparse.Namespace) -> int:
    if (
        VIDEO_CLIENT_IMPORT_ERROR is not None
        or ZmqFramePublisher is None
        or stream_video_webcam is None
        or stream_video_realsense is None
    ):
        raise RuntimeError(build_missing_video_client_message())

    video_args = argparse.Namespace(
        server_host=args.video_server_host,
        port=args.video_port,
        topic=args.video_topic,
        source=args.video_source,
        camera_id=args.video_camera_id,
        width=args.video_width,
        height=args.video_height,
        fps=args.video_fps,
        jpeg_quality=args.video_jpeg_quality,
        mirror=args.video_mirror,
        preview=args.video_preview,
    )

    publisher = ZmqFramePublisher(
        server_host=video_args.server_host,
        port=video_args.port,
        topic=video_args.topic,
    )
    print(
        "[RAG_VIDEO_CLIENT] Streaming video "
        f"source={video_args.source} endpoint={publisher.endpoint} "
        f"topic={video_args.topic} size={video_args.width}x{video_args.height} "
        f"fps={video_args.fps} jpeg_quality={video_args.jpeg_quality}",
        flush=True,
    )

    try:
        if video_args.source == "realsense":
            return stream_video_realsense(video_args, publisher)
        return stream_video_webcam(video_args, publisher)
    except KeyboardInterrupt:
        print("[RAG_VIDEO_CLIENT] Stopping...", flush=True)
        return 0
    finally:
        publisher.close()


class SpeechRecognitionRunner:
    def __init__(self, args: argparse.Namespace, coordinator: InteractionCoordinator):
        self._args = args
        self._coordinator = coordinator
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="SpeechRecognitionRunner",
        )
        self._asr: Optional[RealTimeMicAsr] = None
        self._pa = None
        self._stream = None
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_audio_stream()

        if self._asr is not None:
            try:
                self._asr.close()
            except Exception:
                pass
            self._asr = None

        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            if REALTIME_ASR_IMPORT_ERROR is not None or RealTimeMicAsr is None:
                raise RuntimeError(
                    "Speech recognition dependencies are unavailable. "
                    f"Original import error: {REALTIME_ASR_IMPORT_ERROR}"
                )

            self._asr = RealTimeMicAsr(
                input_rate=self._args.input_rate,
                frames_per_buffer=self._args.frames_per_buffer,
                whisper_model_name=self._args.whisper_model,
                language=self._args.language or None,
                vad_threshold=self._args.vad_threshold,
                phrase_end_delay_sec=self._args.phrase_end_delay_sec,
                enable_drone_commands=False,
                on_transcript=self._coordinator.update_prompt,
                audio_debug=self._args.audio_debug,
                energy_threshold_rms=self._args.energy_threshold_rms,
            )
            self._open_audio_stream()
            print(
                "[RAG_CLIENT][ASR] Listening "
                f"input_rate={self._args.input_rate}Hz -> model_rate={MODEL_AUDIO_RATE}Hz "
                f"whisper={self._args.whisper_model} language={self._args.language or 'auto'}",
                flush=True,
            )

            while not self._stop_event.is_set():
                try:
                    raw = self._stream.read(
                        self._args.frames_per_buffer,
                        exception_on_overflow=False,
                    )
                except OSError as exc:
                    if self._stop_event.is_set():
                        break
                    print(
                        f"[RAG_CLIENT][ASR][WARN] Audio read failed: {exc}. Reopening microphone...",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._close_audio_stream()
                    time.sleep(1.0)
                    self._open_audio_stream()
                    print("[RAG_CLIENT][ASR] Microphone reopened.", flush=True)
                    continue
                self._asr.process_raw_audio(raw)
        except Exception as exc:
            self.error = exc
            self._stop_event.set()
            print(f"[RAG_CLIENT][ERROR] Speech runner failed: {exc}", file=sys.stderr, flush=True)

    def _open_audio_stream(self) -> None:
        self._pa, self._stream = open_microphone_stream(
            microphone_device_id=self._args.mic_id,
            input_rate=self._args.input_rate,
            frames_per_buffer=self._args.frames_per_buffer,
        )

    def _close_audio_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
class GestureDirectionRunner:
    def __init__(self, args: argparse.Namespace, coordinator: InteractionCoordinator):
        self._args = args
        self._coordinator = coordinator
        self._last_state = STATE_NONE

    def _ensure_dependencies(self) -> None:
        if (
            cv2 is None
            or mp is None
            or np is None
            or python is None
            or vision is None
            or ClassifierOptions is None
            or GESTURE_SCROLLING_IMPORT_ERROR is not None
        ):
            raise RuntimeError(
                "Gesture direction dependencies are unavailable. "
                f"Original import error: {GESTURE_SCROLLING_IMPORT_ERROR}"
            )

    def _create_recognizer(self):
        base_options = python.BaseOptions(model_asset_path=self._args.gesture_model)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=ClassifierOptions(
                score_threshold=self._args.min_score,
            ),
            custom_gesture_classifier_options=ClassifierOptions(
                score_threshold=self._args.min_score,
            ),
        )
        return vision.GestureRecognizer.create_from_options(options)

    def _iter_realsense_frames(self):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install the RealSense SDK Python package first."
            )

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            DEFAULT_FRAME_WIDTH,
            DEFAULT_FRAME_HEIGHT,
            rs.format.bgr8,
            DEFAULT_FPS,
        )

        pipeline_started = False
        try:
            pipeline.start(config)
            pipeline_started = True
            print("[RAG_CLIENT][VIDEO] Using local RealSense color stream.", flush=True)

            for _ in range(10):
                try:
                    pipeline.wait_for_frames(5000)
                except RuntimeError as exc:
                    if "Frame didn't arrive within 5000" not in str(exc):
                        raise
                    print("[RAG_CLIENT] Warning: RealSense warmup timeout", flush=True)

            while True:
                try:
                    frames = pipeline.wait_for_frames(5000)
                except RuntimeError as exc:
                    if "Frame didn't arrive within 5000" not in str(exc):
                        raise
                    print("[RAG_CLIENT] Warning: RealSense frame timeout", flush=True)
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_bgr = np.asanyarray(color_frame.get_data())
                yield cv2.flip(frame_bgr, 1)
        finally:
            if pipeline_started:
                pipeline.stop()

    def _iter_stream_frames(self):
        if zmq is None:
            raise RuntimeError(
                "pyzmq is not installed. Install it with: /home/dzmitry/gesture_real_time_control/.venv/bin/pip install pyzmq"
            )

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 10)
        sock.setsockopt(zmq.RCVTIMEO, self._args.video_stream_timeout_ms)
        sock.bind(f"tcp://{self._args.video_bind_host}:{self._args.video_port}")
        sock.setsockopt(zmq.SUBSCRIBE, self._args.video_topic.encode("utf-8"))

        print(
            "[RAG_CLIENT][VIDEO] Waiting for streamed RGB video "
            f"bind={self._args.video_bind_host}:{self._args.video_port} topic={self._args.video_topic}",
            flush=True,
        )

        try:
            while True:
                try:
                    _, ts_bytes, payload = sock.recv_multipart()
                    _ = struct.unpack("d", ts_bytes)[0]
                except zmq.Again:
                    continue

                frame_buffer = np.frombuffer(payload, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    print("[RAG_CLIENT][VIDEO][WARN] Failed to decode JPEG frame.", flush=True)
                    continue

                yield frame_bgr
        finally:
            sock.close()

    def _iter_frames(self):
        source = self._args.gesture_video_source
        if source == "realsense":
            yield from self._iter_realsense_frames()
            return

        if source == "stream":
            yield from self._iter_stream_frames()
            return

        try:
            yield from self._iter_realsense_frames()
        except RuntimeError as exc:
            print(
                f"[RAG_CLIENT][VIDEO][WARN] RealSense unavailable ({exc}). Falling back to streamed RGB input.",
                flush=True,
            )
            yield from self._iter_stream_frames()

    def run(self) -> None:
        self._ensure_dependencies()
        recognizer = self._create_recognizer()

        zone_left_px: Optional[int] = None
        zone_right_px: Optional[int] = None
        last_timestamp_ms = 0
        initial_pose_samples: deque[int] = deque(maxlen=max(1, self._args.initial_mean_window))
        initial_pose_locked = False
        last_seen_monotonic: Optional[float] = None
        last_thumb_x_px: Optional[int] = None

        try:
            for frame_bgr in self._iter_frames():
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                result = recognizer.recognize_for_video(mp_image, timestamp_ms)
                frame_width = frame_bgr.shape[1]

                state = STATE_NONE
                thumb_x_px: Optional[int] = None
                now = time.monotonic()

                thumb_up_index = get_thumb_up_hand_index(result, self._args.min_score)
                if thumb_up_index is None:
                    if (
                        last_seen_monotonic is not None
                        and now - last_seen_monotonic <= self._args.gesture_hold_sec
                        and zone_left_px is not None
                        and zone_right_px is not None
                        and last_thumb_x_px is not None
                    ):
                        thumb_x_px = last_thumb_x_px
                        state = resolve_state(thumb_x_px, zone_left_px, zone_right_px)
                    else:
                        initial_pose_samples.clear()
                        initial_pose_locked = False
                        last_seen_monotonic = None
                        last_thumb_x_px = None
                        zone_left_px = None
                        zone_right_px = None
                else:
                    thumb_x_px = get_thumb_tip_x_px(result, thumb_up_index, frame_width)
                    last_thumb_x_px = thumb_x_px
                    last_seen_monotonic = now

                    if not initial_pose_locked:
                        initial_pose_samples.append(thumb_x_px)
                        center_x_px = mean_int(initial_pose_samples)
                        zone_left_px, zone_right_px = compute_zone_bounds(
                            center_x_px=center_x_px,
                            zone_width_px=self._args.zone_width,
                            frame_width=frame_width,
                        )
                        if len(initial_pose_samples) >= initial_pose_samples.maxlen:
                            initial_pose_locked = True
                    elif zone_left_px is None or zone_right_px is None:
                        zone_left_px, zone_right_px = compute_zone_bounds(
                            center_x_px=thumb_x_px,
                            zone_width_px=self._args.zone_width,
                            frame_width=frame_width,
                        )

                    state = resolve_state(thumb_x_px, zone_left_px, zone_right_px)

                if state != self._last_state:
                    self._last_state = state
                    print(f"[RAG_CLIENT][DIRECTION] {state}", flush=True)
                    self._coordinator.update_direction(state)

                draw_overlay(frame_bgr, zone_left_px, zone_right_px, thumb_x_px, state)
                cv2.imshow("RAG Interaction Direction", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            cv2.destroyAllWindows()
            recognizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated speech + gesture client, minimal FastAPI server, "
            "or ZMQ video streaming client for rag_interaction."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("client", "server", "video-client"),
        default=DEFAULT_MODE,
        help=f"Run as the speech/gesture client, HTTP server, or video streaming client. Default: {DEFAULT_MODE}",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server base URL used by the client. Default: {DEFAULT_SERVER_URL}",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=DEFAULT_COOLDOWN_SEC,
        help=f"Retry delay after a failed POST request in seconds. Default: {DEFAULT_COOLDOWN_SEC}",
    )
    parser.add_argument(
        "--http-timeout",
        type=int,
        default=DEFAULT_HTTP_TIMEOUT_SEC,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_HTTP_TIMEOUT_SEC}",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help=f"FastAPI bind host in server mode. Default: {DEFAULT_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"FastAPI bind port in server mode. Default: {DEFAULT_SERVER_PORT}",
    )
    parser.add_argument(
        "--max-stored-messages",
        type=int,
        default=DEFAULT_MAX_STORED_MESSAGES,
        help=f"How many recent routed messages to keep in memory in server mode. Default: {DEFAULT_MAX_STORED_MESSAGES}",
    )
    parser.add_argument(
        "--gesture-video-source",
        choices=("auto", "realsense", "stream"),
        default=DEFAULT_GESTURE_VIDEO_SOURCE,
        help=(
            "Where gesture frames come from in client mode. "
            f"Default: {DEFAULT_GESTURE_VIDEO_SOURCE}"
        ),
    )
    parser.add_argument(
        "--video-bind-host",
        default=DEFAULT_VIDEO_BIND_HOST,
        help=f"Bind host for receiving streamed RGB video in client mode. Default: {DEFAULT_VIDEO_BIND_HOST}",
    )
    parser.add_argument(
        "--video-server-host",
        default=DEFAULT_VIDEO_SERVER_HOST,
        help=f"ZMQ receiver host in video-client mode. Default: {DEFAULT_VIDEO_SERVER_HOST}",
    )
    parser.add_argument(
        "--video-port",
        type=int,
        default=DEFAULT_VIDEO_PORT,
        help=f"ZMQ receiver RGB port in video-client mode. Default: {DEFAULT_VIDEO_PORT}",
    )
    parser.add_argument(
        "--video-topic",
        default=DEFAULT_VIDEO_TOPIC,
        help=f"ZMQ topic in video-client mode. Default: {DEFAULT_VIDEO_TOPIC}",
    )
    parser.add_argument(
        "--video-source",
        choices=("webcam", "realsense"),
        default=DEFAULT_VIDEO_SOURCE,
        help=f"Video source in video-client mode. Default: {DEFAULT_VIDEO_SOURCE}",
    )
    parser.add_argument(
        "--video-camera-id",
        type=int,
        default=DEFAULT_VIDEO_CAMERA_ID,
        help=f"OpenCV webcam id in video-client mode. Default: {DEFAULT_VIDEO_CAMERA_ID}",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help=f"Requested video frame width in video-client mode. Default: {DEFAULT_VIDEO_WIDTH}",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help=f"Requested video frame height in video-client mode. Default: {DEFAULT_VIDEO_HEIGHT}",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help=f"Requested video FPS in video-client mode. Default: {DEFAULT_VIDEO_FPS}",
    )
    parser.add_argument(
        "--video-jpeg-quality",
        type=int,
        default=DEFAULT_VIDEO_JPEG_QUALITY,
        help=f"JPEG quality 0-100 in video-client mode. Default: {DEFAULT_VIDEO_JPEG_QUALITY}",
    )
    parser.add_argument(
        "--video-mirror",
        action="store_true",
        help="Mirror video frames horizontally in video-client mode.",
    )
    parser.add_argument(
        "--video-preview",
        action="store_true",
        help="Show a local preview window in video-client mode. Press q to stop.",
    )
    parser.add_argument(
        "--video-stream-timeout-ms",
        type=int,
        default=DEFAULT_VIDEO_STREAM_TIMEOUT_MS,
        help=(
            "ZMQ receive timeout in milliseconds when using streamed RGB input in client mode. "
            f"Default: {DEFAULT_VIDEO_STREAM_TIMEOUT_MS}"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input microphones and exit",
    )
    parser.add_argument(
        "--mic-id",
        type=int,
        default=None,
        help="PyAudio microphone device index. Default: first available input.",
    )
    parser.add_argument(
        "--input-rate",
        type=int,
        default=DEFAULT_INPUT_RATE,
        help="Microphone capture rate in Hz",
    )
    parser.add_argument(
        "--frames-per-buffer",
        type=int,
        default=DEFAULT_CHUNK,
        help="PyAudio frames_per_buffer",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="Hugging Face Whisper model id or short name like 'medium'",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Optional Whisper language code. Empty means auto-detect.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help="Silero VAD threshold",
    )
    parser.add_argument(
        "--energy-threshold-rms",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD_RMS,
        help="Fallback RMS threshold that can trigger phrase capture even if Silero VAD misses speech.",
    )
    parser.add_argument(
        "--audio-debug",
        action="store_true",
        help="Print periodic microphone RMS/peak debug info to help diagnose VAD issues.",
    )
    parser.add_argument(
        "--phrase-end-delay-sec",
        type=float,
        default=0.5,
        help="Seconds of silence before a phrase is finalized",
    )
    parser.add_argument(
        "--gesture-model",
        "--model",
        type=str,
        dest="gesture_model",
        default=default_gesture_model_path(),
        help="Path to the MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Unused for RealSense input; kept for compatibility with gesture_scrolling.py",
    )
    parser.add_argument(
        "--zone-width",
        type=int,
        default=DEFAULT_ZONE_WIDTH,
        help="Anchor zone width in pixels",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_GESTURE_SCORE,
        help="Minimum Thumb_Up confidence",
    )
    parser.add_argument(
        "--initial-mean-window",
        type=int,
        default=DEFAULT_INITIAL_MEAN_WINDOW,
        help="Number of first Thumb_Up frames to average for the initial zone anchor",
    )
    parser.add_argument(
        "--gesture-hold-sec",
        type=float,
        default=DEFAULT_GESTURE_HOLD_SEC,
        help="How long to keep the last direction after Thumb_Up briefly disappears",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "server":
        try:
            return run_server(args)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 1
    if args.mode == "video-client":
        try:
            return run_video_client_mode(args)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 1

    if args.list_devices:
        list_input_devices()
        return 0

    coordinator = InteractionCoordinator(
        server_url=args.server_url,
        cooldown_sec=args.cooldown_sec,
        http_timeout_sec=args.http_timeout,
    )
    speech_runner = SpeechRecognitionRunner(args, coordinator)
    direction_runner = GestureDirectionRunner(args, coordinator)

    check_server_health(coordinator.server_base_url, args.http_timeout)
    speech_runner.start()

    print(
        f"[RAG_CLIENT] Server URL: {coordinator.server_base_url} "
        f"(query={build_endpoint_url(coordinator.server_base_url, ROUTE_QUERY)}, "
        f"command={build_endpoint_url(coordinator.server_base_url, ROUTE_COMMAND)})",
        flush=True,
    )
    print(
        "[RAG_CLIENT] Gesture video source: "
        f"{args.gesture_video_source} "
        f"(stream bind={args.video_bind_host}:{args.video_port} topic={args.video_topic})",
        flush=True,
    )
    print("[RAG_CLIENT] Speech recognition and direction detection started.", flush=True)
    print("[RAG_CLIENT] Press 'q' in the direction window or Ctrl+C to stop.", flush=True)

    try:
        direction_runner.run()
    except KeyboardInterrupt:
        print("[RAG_CLIENT] Stopping...", flush=True)
    finally:
        speech_runner.stop()

    if speech_runner.error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
