from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.processors.classifier_options import ClassifierOptions

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

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
from realtime_asr_vad_pyaudio import (
    DEFAULT_CHUNK,
    DEFAULT_ENERGY_THRESHOLD_RMS,
    DEFAULT_INPUT_RATE,
    MODEL_AUDIO_RATE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT_SEC,
    DEFAULT_VAD_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    RealTimeMicAsr,
    list_input_devices,
    open_microphone_stream,
)


DEFAULT_SERVER_URL = "http://127.0.0.1:8000/interact"
DEFAULT_HTTP_TIMEOUT_SEC = 10
DEFAULT_COOLDOWN_SEC = 5.0
VALID_DIRECTIONS = {"left", "right", "none"}


@dataclass
class PendingInteraction:
    event_id: int
    prompt: str
    direction: str
    trigger: str
    created_monotonic: float


class InteractionCoordinator:
    def __init__(
        self,
        server_url: str,
        cooldown_sec: float,
        http_timeout_sec: int,
    ):
        self._server_url = server_url
        self._cooldown_sec = cooldown_sec
        self._http_timeout_sec = http_timeout_sec

        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._latest_direction = STATE_NONE
        self._latest_prompt: Optional[str] = None
        self._pending: deque[PendingInteraction] = deque()
        self._retry_not_before = 0.0
        self._next_event_id = 1

    def time_until_ready(self) -> float:
        with self._lock:
            return max(0.0, self._retry_not_before - time.monotonic())

    def _enqueue_pending_locked(
        self,
        prompt: str,
        direction: str,
        trigger: str,
    ) -> PendingInteraction:
        pending = PendingInteraction(
            event_id=self._next_event_id,
            prompt=prompt,
            direction=direction,
            trigger=trigger,
            created_monotonic=time.monotonic(),
        )
        self._next_event_id += 1
        self._pending.append(pending)
        return pending

    def update_direction(self, direction: str) -> None:
        direction = direction.strip().lower()
        if direction not in VALID_DIRECTIONS:
            return

        with self._lock:
            if direction == self._latest_direction:
                return
            self._latest_direction = direction
            latest_prompt = self._latest_prompt
            if latest_prompt:
                self._enqueue_pending_locked(
                    prompt=latest_prompt,
                    direction=direction,
                    trigger="direction",
                )

        if latest_prompt is None:
            return

        retry_remaining = self.time_until_ready()
        if retry_remaining > 0:
            print(
                f"[RAG_CLIENT][QUEUE] Direction change queued for {retry_remaining:.1f}s retry delay",
                flush=True,
            )
        self.flush_pending()

    def update_prompt(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return

        retry_remaining = self.time_until_ready()
        with self._lock:
            self._latest_prompt = prompt
            self._enqueue_pending_locked(
                prompt=prompt,
                direction=self._latest_direction,
                trigger="phrase",
            )

        print(f"[RAG_CLIENT][RECOGNIZED_PHRASE] {prompt}", flush=True)
        if retry_remaining > 0:
            print(
                f"[RAG_CLIENT][QUEUE] Phrase queued for {retry_remaining:.1f}s retry delay",
                flush=True,
            )
        self.flush_pending()

    def flush_pending(self) -> bool:
        if not self._flush_lock.acquire(blocking=False):
            return False

        sent_any = False
        try:
            while True:
                with self._lock:
                    now = time.monotonic()
                    if not self._pending or now < self._retry_not_before:
                        return sent_any

                    pending = self._pending[0]
                    payload = {
                        "prompt": pending.prompt,
                        "direction": pending.direction,
                    }

                try:
                    print(f"[RAG_CLIENT][SENDING] {payload}", flush=True)
                    response = requests.post(
                        self._server_url,
                        json=payload,
                        timeout=self._http_timeout_sec,
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:
                    print(f"[RAG_CLIENT][ERROR] POST failed: {exc}", file=sys.stderr, flush=True)
                    with self._lock:
                        self._retry_not_before = time.monotonic() + self._cooldown_sec
                    return sent_any

                with self._lock:
                    if self._pending and self._pending[0].event_id == pending.event_id:
                        self._pending.popleft()
                    self._retry_not_before = 0.0

                response_preview = response.text.strip()
                if len(response_preview) > 200:
                    response_preview = response_preview[:200] + "..."
                print(f"[RAG_CLIENT][POST] {payload}", flush=True)
                print(
                    f"[RAG_CLIENT][SERVER_RESPONSE] status={response.status_code} body={response_preview}",
                    flush=True,
                )
                sent_any = True
        finally:
            self._flush_lock.release()


def build_health_url(server_url: str) -> str:
    if server_url.endswith("/interact"):
        return server_url[: -len("/interact")] + "/health"
    return server_url.rstrip("/") + "/health"


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
            self._asr = RealTimeMicAsr(
                input_rate=self._args.input_rate,
                frames_per_buffer=self._args.frames_per_buffer,
                whisper_model_name=self._args.whisper_model,
                language=self._args.language or None,
                vad_threshold=self._args.vad_threshold,
                phrase_end_delay_sec=self._args.phrase_end_delay_sec,
                ollama_host=self._args.ollama_host,
                ollama_model=self._args.ollama_model,
                ollama_timeout_sec=self._args.ollama_timeout,
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


class CooldownFlushWorker:
    def __init__(self, coordinator: InteractionCoordinator):
        self._coordinator = coordinator
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="CooldownFlushWorker",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._coordinator.flush_pending()
            time.sleep(0.1)


class GestureDirectionRunner:
    def __init__(self, args: argparse.Namespace, coordinator: InteractionCoordinator):
        self._args = args
        self._coordinator = coordinator
        self._last_state = STATE_NONE

    def run(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed. Install the RealSense SDK Python package first.")

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
        recognizer = vision.GestureRecognizer.create_from_options(options)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            DEFAULT_FRAME_WIDTH,
            DEFAULT_FRAME_HEIGHT,
            rs.format.bgr8,
            DEFAULT_FPS,
        )

        zone_left_px: Optional[int] = None
        zone_right_px: Optional[int] = None
        last_timestamp_ms = 0
        pipeline_started = False
        initial_pose_samples: deque[int] = deque(maxlen=max(1, self._args.initial_mean_window))
        initial_pose_locked = False
        last_seen_monotonic: Optional[float] = None
        last_thumb_x_px: Optional[int] = None

        try:
            pipeline.start(config)
            pipeline_started = True

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
                frame_bgr = cv2.flip(frame_bgr, 1)
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
            if pipeline_started:
                pipeline.stop()
            cv2.destroyAllWindows()
            recognizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated speech + direction client that POSTs prompt and direction to a REST server.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"REST endpoint that receives POST requests. Default: {DEFAULT_SERVER_URL}",
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
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server URL used by the shared ASR module. Default: {DEFAULT_OLLAMA_HOST}",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name used by the shared ASR module. Default: {DEFAULT_OLLAMA_MODEL}",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT_SEC,
        help=f"Ollama request timeout in seconds. Default: {DEFAULT_OLLAMA_TIMEOUT_SEC}",
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
    if args.list_devices:
        list_input_devices()
        return 0

    coordinator = InteractionCoordinator(
        server_url=args.server_url,
        cooldown_sec=args.cooldown_sec,
        http_timeout_sec=args.http_timeout,
    )
    speech_runner = SpeechRecognitionRunner(args, coordinator)
    cooldown_worker = CooldownFlushWorker(coordinator)
    direction_runner = GestureDirectionRunner(args, coordinator)

    check_server_health(args.server_url, args.http_timeout)
    speech_runner.start()
    cooldown_worker.start()

    print(f"[RAG_CLIENT] Server URL: {args.server_url}", flush=True)
    print("[RAG_CLIENT] Speech recognition and direction detection started.", flush=True)
    print("[RAG_CLIENT] Press 'q' in the direction window or Ctrl+C to stop.", flush=True)

    try:
        direction_runner.run()
    except KeyboardInterrupt:
        print("[RAG_CLIENT] Stopping...", flush=True)
    finally:
        cooldown_worker.stop()
        speech_runner.stop()

    if speech_runner.error is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
