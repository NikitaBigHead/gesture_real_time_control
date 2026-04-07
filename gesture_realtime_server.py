"""
ZMQ gesture server with ROS 2 drone control.

Receives RGB, depth, and audio from the network like `server.py`, but the
vision path reuses the same gesture-to-command pipeline as `deep_control`.

Audio transcription is optional. When the required packages are installed,
audio is segmented with Silero VAD and transcribed with Whisper medium after
2 seconds of trailing silence.

Suggested extra packages for audio:
    pip install SpeechRecognition openai-whisper silero-vad
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from textwrap import wrap
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import zmq
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent
DEEP_CONTROL_DIR = ROOT_DIR / "deep_control"
if str(DEEP_CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_CONTROL_DIR))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from control_commands import (
    collect_command_events,
    create_command_tracker_state,
    log_command_events,
    reset_command_tracker_state,
)
from control_config import COMMAND_MISSING_SLOT_HOLD_FRAMES
from control_overlay import draw_control_overlay, draw_hand_skeleton, draw_person_bbox
from control_state import (
    FrameControlState,
    compute_frame_control_state,
    create_tracking_state,
    reset_tracking_state,
)
from gesture_drone_node import GestureDroneController

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import torch
except ImportError:
    torch = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    from silero_vad import load_silero_vad
except ImportError:
    load_silero_vad = None


BIND_HOST = "0.0.0.0"
PORT_RGB = 5550
PORT_DEPTH = 5551
PORT_AUDIO = 5552

RS_WIDTH = 640
RS_HEIGHT = 480
DEPTH_SCALE_METERS = 0.001  # client sends uint16 depth in millimeters

MODEL_AUDIO_RATE = 16000
DEFAULT_AUDIO_INPUT_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 1024
SILERO_WINDOW_SAMPLES = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GESTURE_SERVER] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def normalize_voice_command(text: str) -> Optional[str]:
    normalized = " ".join(text.strip().lower().replace("_", " ").split())
    if not normalized:
        return None

    tokens = set(re.findall(r"[a-z]+", normalized))
    if "land" in tokens:
        return "land"
    if "explore" in tokens and "face" in tokens:
        return "explore_face"
    if "follow" in tokens:
        return "follow"
    if {"stay", "stop", "hover"} & tokens:
        return "stay"
    return None


class VoiceCommandPublisher(Node):
    def __init__(self, topic: str):
        super().__init__("gesture_voice_command_bridge")
        self._topic = topic
        self._pub = self.create_publisher(String, topic, 10)
        self.get_logger().info(f"Voice command publisher ready on {topic}")

    def publish_command(self, command: str, transcript: str) -> None:
        msg = String()
        msg.data = command
        self._pub.publish(msg)
        self.get_logger().info(
            f"Published voice command '{command}' from transcript='{transcript}'"
        )


def list_input_microphones() -> None:
    if pyaudio is None:
        raise RuntimeError("PyAudio is not installed.")

    pa = pyaudio.PyAudio()
    try:
        found = False
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                found = True
                print(
                    f"[{index}] {info.get('name', 'Unknown')} | "
                    f"inputs={info.get('maxInputChannels')} | "
                    f"default_rate={int(info.get('defaultSampleRate', 0))}"
                )
        if not found:
            print("No input microphones found.")
    finally:
        pa.terminate()


def open_microphone_stream(microphone_device_id: Optional[int], input_rate: int, frames_per_buffer: int):
    if pyaudio is None:
        raise RuntimeError("PyAudio is not installed.")

    pa = pyaudio.PyAudio()
    if microphone_device_id is not None:
        info = pa.get_device_info_by_index(microphone_device_id)
        if info.get("maxInputChannels", 0) <= 0:
            pa.terminate()
            raise RuntimeError(f"Device {microphone_device_id} has no input channels.")
        device_index = microphone_device_id
        log.info("Using local microphone [%d] %s", device_index, info.get("name", "Unknown"))
    else:
        device_index = None
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                device_index = index
                log.info(
                    "Using first available local microphone [%d] %s",
                    index,
                    info.get("name", "Unknown"),
                )
                break
        if device_index is None:
            pa.terminate()
            raise RuntimeError("No input microphone was found.")

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=input_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=frames_per_buffer,
    )
    return pa, stream


def first_existing_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def default_gesture_model_path() -> str:
    return first_existing_path(
        ROOT_DIR / "gesture_recognizer.task",
        DEEP_CONTROL_DIR / "gesture_recognizer.task",
        ROOT_DIR / "gesture_recognizer_1.0.task",
    )


def default_pose_model_path() -> str:
    return first_existing_path(
        DEEP_CONTROL_DIR / "pose_landmarker_heavy.task",
        ROOT_DIR / "pose_landmarker_full.task",
        DEEP_CONTROL_DIR / "pose_landmarker_full.task",
    )


def build_base_options(model_path: str, use_gpu: bool) -> python.BaseOptions:
    kwargs = {"model_asset_path": model_path}
    if use_gpu:
        kwargs["delegate"] = python.BaseOptions.Delegate.GPU
    return python.BaseOptions(**kwargs)


class StreamStats:
    def __init__(self, name: str, window: int = 100):
        self.name = name
        self._lock = threading.Lock()
        self._count = 0
        self._delays_ms: deque[float] = deque(maxlen=window)

    def record(self, send_ts: float) -> None:
        delay_ms = (time.time() - send_ts) * 1000.0
        with self._lock:
            self._count += 1
            self._delays_ms.append(delay_ms)
            if self._count % 100 == 0:
                avg_delay = float(np.mean(self._delays_ms))
                log.info("[%s] packets=%d avg_delay=%.1f ms", self.name, self._count, avg_delay)


class AudioRateConverter:
    def __init__(self, input_rate: int, output_rate: int):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("Audio sample rates must be positive.")

        self._input_rate = input_rate
        self._output_rate = output_rate
        self._step = input_rate / output_rate
        self._buffer = np.empty(0, dtype=np.float32)
        self._position = 0.0

    def convert(self, samples: np.ndarray) -> np.ndarray:
        if self._input_rate == self._output_rate:
            return samples

        if samples.size == 0:
            return np.empty(0, dtype=np.int16)

        self._buffer = np.concatenate((self._buffer, samples.astype(np.float32)))
        if self._buffer.size < 2:
            return np.empty(0, dtype=np.int16)

        max_position = self._buffer.size - 1
        positions = np.arange(self._position, max_position, self._step, dtype=np.float64)
        if positions.size == 0:
            return np.empty(0, dtype=np.int16)

        indices = positions.astype(np.int64)
        fractions = positions - indices
        converted = self._buffer[indices] * (1.0 - fractions) + self._buffer[indices + 1] * fractions

        consumed = int(indices[-1])
        self._buffer = self._buffer[consumed:]
        self._position = positions[-1] + self._step - consumed

        return np.clip(np.round(converted), -32768, 32767).astype(np.int16)


class AudioSpeechPipeline:
    def __init__(
        self,
        enabled: bool,
        input_sample_rate: int,
        whisper_model_name: str,
        language: Optional[str],
        vad_threshold: float,
        end_delay_sec: float,
        on_phrase: Optional[Callable[[str], None]] = None,
        pre_roll_sec: float = 0.35,
        min_phrase_sec: float = 0.30,
    ):
        self.enabled = enabled
        self._input_sample_rate = input_sample_rate
        self._model_sample_rate = MODEL_AUDIO_RATE
        self._rate_converter = AudioRateConverter(input_sample_rate, self._model_sample_rate)
        self._whisper_model_name = whisper_model_name
        self._language = language
        self._vad_threshold = vad_threshold
        self._end_delay_sec = end_delay_sec
        self._pre_roll_chunks = max(1, int(pre_roll_sec * input_sample_rate / AUDIO_CHUNK))
        self._min_phrase_samples = int(min_phrase_sec * self._model_sample_rate)

        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._pre_roll_chunks)
        self._phrase_chunks: list[np.ndarray] = []
        self._phrase_active = False
        self._last_speech_monotonic: Optional[float] = None
        self._vad_tail = np.empty(0, dtype=np.int16)

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._vad_model = None
        self._recognizer = sr.Recognizer() if sr is not None else None
        self._whisper_model = None
        self._phrase_lock = threading.Lock()
        self._latest_phrase = ""
        self._latest_phrase_monotonic: Optional[float] = None
        self._on_phrase = on_phrase
        self._use_speech_recognition_backend = (
            sr is not None
            and whisper is not None
            and hasattr(sr.Recognizer, "recognize_whisper")
        )
        self._fp16 = bool(torch is not None and torch.cuda.is_available())

        if not enabled:
            log.info("Audio transcription disabled by flag.")
            return

        missing = []
        if torch is None:
            missing.append("torch")
        if load_silero_vad is None:
            missing.append("silero-vad")
        if whisper is None:
            missing.append("openai-whisper")

        if missing:
            self.enabled = False
            log.warning(
                "Audio transcription disabled; missing packages: %s",
                ", ".join(missing),
            )
            return

        self._vad_model = load_silero_vad()
        if hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()

        self._worker = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="AudioTranscriptionWorker",
        )
        self._worker.start()
        log.info(
            "Audio transcription enabled: input_rate=%dHz -> model_rate=%dHz, Whisper %s (phrase_end_delay=%.1fs)",
            input_sample_rate,
            self._model_sample_rate,
            whisper_model_name,
            end_delay_sec,
        )

    def _chunk_contains_speech(self, samples: np.ndarray) -> bool:
        if self._vad_model is None or torch is None:
            return False

        merged = np.concatenate((self._vad_tail, samples))
        speech_detected = False
        offset = 0

        while offset + SILERO_WINDOW_SAMPLES <= len(merged):
            window = merged[offset:offset + SILERO_WINDOW_SAMPLES]
            offset += SILERO_WINDOW_SAMPLES

            audio_tensor = torch.from_numpy(window.astype(np.float32) / 32768.0)
            speech_prob = float(self._vad_model(audio_tensor, self._model_sample_rate).item())
            if speech_prob >= self._vad_threshold:
                speech_detected = True

        self._vad_tail = merged[offset:]
        return speech_detected

    def process_chunk(self, samples: np.ndarray, send_ts: float) -> None:
        if not self.enabled:
            return

        chunk = self._rate_converter.convert(np.asarray(samples, dtype=np.int16).copy())
        if chunk.size == 0:
            return

        self._pre_roll.append(chunk)
        speech_detected = self._chunk_contains_speech(chunk)
        now = time.monotonic()

        if speech_detected:
            if not self._phrase_active:
                self._phrase_chunks = [pre.copy() for pre in self._pre_roll]
                self._phrase_active = True
            else:
                self._phrase_chunks.append(chunk)
            self._last_speech_monotonic = now
            return

        if not self._phrase_active:
            return

        self._phrase_chunks.append(chunk)
        if self._last_speech_monotonic is None:
            return

        if now - self._last_speech_monotonic >= self._end_delay_sec:
            self._finalize_phrase()

    def _finalize_phrase(self) -> None:
        if not self._phrase_chunks:
            self._reset_phrase_state()
            return

        utterance = np.concatenate(self._phrase_chunks)
        self._reset_phrase_state(reset_vad=True)

        if len(utterance) < self._min_phrase_samples:
            return

        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            log.warning("Dropping utterance because transcription queue is full.")

    def _reset_phrase_state(self, reset_vad: bool = False) -> None:
        self._phrase_chunks = []
        self._phrase_active = False
        self._last_speech_monotonic = None
        self._vad_tail = np.empty(0, dtype=np.int16)
        if reset_vad and self._vad_model is not None and hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()

    def _transcription_worker(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if utterance is None:
                break

            try:
                text = self._transcribe(utterance)
                if text:
                    print("detected speech",text)
                    with self._phrase_lock:
                        self._latest_phrase = text
                        self._latest_phrase_monotonic = time.monotonic()
                    log.info("[Speech] %s", text)
                    if self._on_phrase is not None:
                        try:
                            self._on_phrase(text)
                        except Exception as exc:
                            log.error("Voice command callback failed: %s", exc)
            except Exception as exc:
                log.error("Audio transcription failed: %s", exc)

    def _transcribe(self, utterance: np.ndarray) -> str:
        if self._use_speech_recognition_backend and self._recognizer is not None:
            audio_data = sr.AudioData(utterance.tobytes(), self._model_sample_rate, 2)
            kwargs = {
                "model": self._whisper_model_name,
                "fp16": self._fp16,
            }
            if self._language:
                kwargs["language"] = self._language
            text = self._recognizer.recognize_whisper(audio_data, **kwargs)
            return text.strip()

        if self._whisper_model is None:
            log.info("Loading Whisper model '%s'...", self._whisper_model_name)
            self._whisper_model = whisper.load_model(self._whisper_model_name)

        audio = utterance.astype(np.float32) / 32768.0
        kwargs = {"fp16": self._fp16}
        if self._language:
            kwargs["language"] = self._language
        result = self._whisper_model.transcribe(audio, **kwargs)
        return result.get("text", "").strip()

    def close(self) -> None:
        if not self.enabled:
            return

        if self._phrase_active and self._phrase_chunks:
            self._finalize_phrase()

        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    def get_latest_phrase(self, max_age_sec: float = 8.0) -> str:
        if not self.enabled:
            return ""

        with self._phrase_lock:
            if not self._latest_phrase or self._latest_phrase_monotonic is None:
                return ""
            if time.monotonic() - self._latest_phrase_monotonic > max_age_sec:
                return ""
            return self._latest_phrase


class GestureServerRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_ts: Optional[float] = None
        self._stop_requested = threading.Event()

        self._last_timestamp_ms = 0
        self._no_hand_frames = 0
        self._tracking_state = create_tracking_state()
        self._command_state = create_command_tracker_state()

        self._drone_controller = None
        self._drone_spin_thread = None
        self._voice_command_publisher = None
        self._owns_rclpy = False
        self._last_voice_command = ""
        self._last_voice_command_monotonic = 0.0

        self._recognizer = self._create_gesture_recognizer(args.model, args.use_gpu)
        self._pose_landmarker = self._create_pose_landmarker(args.pose_model, args.use_gpu)
        if not args.disable_voice_command_pub:
            self._start_voice_command_publisher()
        self._audio = AudioSpeechPipeline(
            enabled=not args.disable_audio_recognition,
            input_sample_rate=args.audio_input_rate,
            whisper_model_name=args.whisper_model,
            language=args.audio_language or None,
            vad_threshold=args.vad_threshold,
            end_delay_sec=args.vad_end_delay_sec,
            on_phrase=self._handle_detected_phrase,
        )

        if not args.no_drone_control:
            self._start_drone_controller()

    def _create_gesture_recognizer(self, model_path: str, use_gpu: bool):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gesture model not found: {model_path}")

        options = vision.GestureRecognizerOptions(
            base_options=build_base_options(model_path, use_gpu),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.GestureRecognizer.create_from_options(options)

    def _create_pose_landmarker(self, model_path: str, use_gpu: bool):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model not found: {model_path}")

        options = vision.PoseLandmarkerOptions(
            base_options=build_base_options(model_path, use_gpu),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.4,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _ensure_rclpy(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

    def _start_drone_controller(self) -> None:
        self._ensure_rclpy()

        self._drone_controller = GestureDroneController(
            mavros_prefix=self.args.mavros_prefix,
            pose_topic=self.args.drone_pose_topic,
            move_duration_sec=self.args.drone_move_duration,
            move_speed_xy=self.args.drone_move_speed_xy,
            move_speed_z=self.args.drone_move_speed_z,
            yaw_kp=self.args.drone_yaw_kp,
            max_yaw_rate=self.args.drone_max_yaw_rate,
            yaw_tolerance_deg=self.args.drone_yaw_tolerance_deg,
            yaw_sign=self.args.drone_yaw_sign,
        )
        self._drone_spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self._drone_controller,),
            daemon=True,
            name="GestureDroneSpin",
        )
        self._drone_spin_thread.start()
        log.info(
            "ROS control enabled: cmd_vel=%s pose=%s",
            self._drone_controller.cmd_vel_topic,
            self._drone_controller.pose_topic,
        )

    def _start_voice_command_publisher(self) -> None:
        self._ensure_rclpy()
        self._voice_command_publisher = VoiceCommandPublisher(self.args.voice_command_topic)

    def _handle_detected_phrase(self, phrase: str) -> None:
        if self._voice_command_publisher is None:
            return

        command = normalize_voice_command(phrase)
        if command is None:
            return

        now = time.monotonic()
        if (
            command == self._last_voice_command
            and now - self._last_voice_command_monotonic < self.args.voice_command_min_repeat_sec
        ):
            log.info(
                "Skipping duplicate voice command '%s' (cooldown %.1fs)",
                command,
                self.args.voice_command_min_repeat_sec,
            )
            return

        self._voice_command_publisher.publish_command(command, phrase)
        self._last_voice_command = command
        self._last_voice_command_monotonic = now

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def process_depth(self, depth: np.ndarray, send_ts: float) -> None:
        if self.args.flip_horizontal:
            depth = cv2.flip(depth, 1)

        with self._depth_lock:
            self._latest_depth = depth.copy()
            self._latest_depth_ts = send_ts

    def process_audio(self, samples: np.ndarray, send_ts: float) -> None:
        self._audio.process_chunk(samples, send_ts)

    def _draw_detected_phrase(self, frame: np.ndarray) -> None:
        phrase = self._audio.get_latest_phrase()
        if not phrase:
            return

        lines = wrap(phrase, width=44)[:3]
        line_height = 28
        box_top = 12
        box_left = 12
        box_width = min(frame.shape[1] - 24, 620)
        box_height = 18 + line_height * (len(lines) + 1)

        cv2.rectangle(
            frame,
            (box_left, box_top),
            (box_left + box_width, box_top + box_height),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            frame,
            (box_left, box_top),
            (box_left + box_width, box_top + box_height),
            (0, 180, 255),
            2,
        )
        cv2.putText(
            frame,
            "Speech:",
            (box_left + 12, box_top + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        for index, line in enumerate(lines, start=1):
            cv2.putText(
                frame,
                line,
                (box_left + 12, box_top + 28 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def process_rgb(self, frame: np.ndarray, send_ts: float) -> None:
        with self._depth_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            depth_ts = self._latest_depth_ts

        if depth is None:
            return

        if self.args.max_depth_age_sec > 0 and depth_ts is not None:
            if abs(send_ts - depth_ts) > self.args.max_depth_age_sec:
                return

        if self.args.flip_horizontal:
            frame = cv2.flip(frame, 1)

        frame_depth_colormap = None
        if self.args.show_depth and not self.args.no_display:
            frame_depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03),
                cv2.COLORMAP_JET,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._next_timestamp_ms()

        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        draw_person_bbox(frame, pose_result)

        if result.hand_landmarks:
            self._no_hand_frames = 0
            frame_state = compute_frame_control_state(
                result,
                pose_result,
                depth,
                DEPTH_SCALE_METERS,
                None,
                frame.shape[1],
                frame.shape[0],
                self._tracking_state,
            )
            command_events = collect_command_events(
                frame_state,
                self._command_state,
                timestamp_ms,
            )
            log_command_events(command_events)
            if self._drone_controller is not None and command_events:
                self._drone_controller.handle_events(command_events)
            draw_hand_skeleton(frame, result.hand_landmarks)
            draw_control_overlay(frame, frame_state)
        else:
            self._no_hand_frames += 1
            empty_frame_state = FrameControlState(
                active_hand_slot=None,
                active_hand_index=None,
                hands=[],
            )
            command_events = collect_command_events(
                empty_frame_state,
                self._command_state,
                timestamp_ms,
            )
            log_command_events(command_events)
            if self._drone_controller is not None and command_events:
                self._drone_controller.handle_events(command_events)
            if self._no_hand_frames >= COMMAND_MISSING_SLOT_HOLD_FRAMES:
                reset_tracking_state(self._tracking_state)
                reset_command_tracker_state(self._command_state)

        if self.args.no_display:
            return

        cv2.imshow("Gesture Realtime Server", frame)
        if frame_depth_colormap is not None:
            cv2.imshow("Depth", frame_depth_colormap)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self._stop_requested.set()

    def should_stop(self) -> bool:
        return self._stop_requested.is_set()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def close(self) -> None:
        self._audio.close()

        if self._drone_controller is not None:
            try:
                self._drone_controller.send_velocity_command(0.0, 0.0, 0.0, 0.0)
            finally:
                self._drone_controller.destroy_node()
        if self._voice_command_publisher is not None:
            self._voice_command_publisher.destroy_node()
            self._voice_command_publisher = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        if self._drone_spin_thread is not None:
            self._drone_spin_thread.join(timeout=1.0)

        self._recognizer.close()
        self._pose_landmarker.close()
        cv2.destroyAllWindows()


class RGBReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context, runtime: GestureServerRuntime, bind_host: str, port: int):
        super().__init__(daemon=True, name="RGBReceiver")
        self._runtime = runtime
        self._bind_host = bind_host
        self._port = port
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 10)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.bind(f"tcp://{bind_host}:{port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"rgb")
        self._stop = threading.Event()
        self._stats = StreamStats("RGB")

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("RGBReceiver listening on %s:%s", self._bind_host, self._port)
        while not self._stop.is_set() and not self._runtime.should_stop():
            try:
                _, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]

                buf = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    log.warning("RGB decode failed.")
                    continue

                self._stats.record(send_ts)
                self._runtime.process_rgb(frame, send_ts)
            except zmq.Again:
                pass
            except Exception as exc:
                log.error("RGBReceiver error: %s", exc)

        self._sock.close()


class DepthReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context, runtime: GestureServerRuntime, bind_host: str, port: int):
        super().__init__(daemon=True, name="DepthReceiver")
        self._runtime = runtime
        self._bind_host = bind_host
        self._port = port
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 10)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.bind(f"tcp://{bind_host}:{port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"depth")
        self._stop = threading.Event()
        self._stats = StreamStats("Depth")
        self._expected_bytes = RS_WIDTH * RS_HEIGHT * 2

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("DepthReceiver listening on %s:%s", self._bind_host, self._port)
        while not self._stop.is_set() and not self._runtime.should_stop():
            try:
                _, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]

                if len(payload) != self._expected_bytes:
                    log.warning("Depth payload has unexpected size: %d", len(payload))
                    continue

                depth = np.frombuffer(payload, dtype=np.uint16).reshape(RS_HEIGHT, RS_WIDTH)
                self._stats.record(send_ts)
                self._runtime.process_depth(depth, send_ts)
            except zmq.Again:
                pass
            except Exception as exc:
                log.error("DepthReceiver error: %s", exc)

        self._sock.close()


class AudioReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context, runtime: GestureServerRuntime, bind_host: str, port: int):
        super().__init__(daemon=True, name="AudioReceiver")
        self._runtime = runtime
        self._bind_host = bind_host
        self._port = port
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 50)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.bind(f"tcp://{bind_host}:{port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"audio")
        self._stop = threading.Event()
        self._stats = StreamStats("Audio")

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("AudioReceiver listening on %s:%s", self._bind_host, self._port)
        while not self._stop.is_set() and not self._runtime.should_stop():
            try:
                _, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]
                samples = np.frombuffer(payload, dtype=np.int16)
                self._stats.record(send_ts)
                self._runtime.process_audio(samples, send_ts)
            except zmq.Again:
                pass
            except Exception as exc:
                log.error("AudioReceiver error: %s", exc)

        self._sock.close()


class LocalAudioReceiver(threading.Thread):
    def __init__(
        self,
        runtime: GestureServerRuntime,
        sample_rate: int,
        frames_per_buffer: int,
        microphone_device_id: Optional[int],
    ):
        super().__init__(daemon=True, name="LocalAudioReceiver")
        self._runtime = runtime
        self._sample_rate = sample_rate
        self._frames_per_buffer = frames_per_buffer
        self._microphone_device_id = microphone_device_id
        self._stop = threading.Event()
        self._stream = None
        self._pa = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info(
            "LocalAudioReceiver using ground-station microphone at %d Hz (buffer=%d)",
            self._sample_rate,
            self._frames_per_buffer,
        )
        try:
            self._pa, self._stream = open_microphone_stream(
                self._microphone_device_id,
                self._sample_rate,
                self._frames_per_buffer,
            )
            while not self._stop.is_set() and not self._runtime.should_stop():
                raw = self._stream.read(
                    self._frames_per_buffer,
                    exception_on_overflow=False,
                )
                samples = np.frombuffer(raw, dtype=np.int16)
                self._runtime.process_audio(samples, time.time())
        except Exception as exc:
            log.error("LocalAudioReceiver error: %s", exc)
            self._runtime.request_stop()
        finally:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa is not None:
                self._pa.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZMQ gesture server with ROS control")
    parser.add_argument("--bind-host", type=str, default=BIND_HOST, help="Bind host")
    parser.add_argument("--port-rgb", type=int, default=PORT_RGB, help="RGB port")
    parser.add_argument("--port-depth", type=int, default=PORT_DEPTH, help="Depth port")
    parser.add_argument("--port-audio", type=int, default=PORT_AUDIO, help="Audio port")
    parser.add_argument(
        "--model",
        type=str,
        default=default_gesture_model_path(),
        help="Path to MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default=default_pose_model_path(),
        help="Path to MediaPipe pose landmarker task model",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use MediaPipe GPU delegate for gesture and pose models",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show depth colormap window",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Mirror RGB and depth frames horizontally before processing",
    )
    parser.add_argument(
        "--max-depth-age-sec",
        type=float,
        default=0.35,
        help="Maximum accepted age difference between RGB and latest depth frame",
    )
    parser.add_argument(
        "--no-drone-control",
        action="store_true",
        help="Disable ROS drone control and only run gesture recognition",
    )
    parser.add_argument(
        "--mavros-prefix",
        type=str,
        default="/gesture",
        help="Velocity topic prefix. Use /gesture for the bridge node, or a MAVROS namespace for direct publishing.",
    )
    parser.add_argument(
        "--drone-pose-topic",
        type=str,
        default="/mavros/local_position/pose",
        help="Pose topic used for yaw feedback",
    )
    parser.add_argument(
        "--drone-move-duration",
        type=float,
        default=0.8,
        help="Duration for translation commands generated from fist_release",
    )
    parser.add_argument(
        "--drone-move-speed-xy",
        type=float,
        default=0.35,
        help="XY speed for translation commands",
    )
    parser.add_argument(
        "--drone-move-speed-z",
        type=float,
        default=0.25,
        help="Z speed for translation commands",
    )
    parser.add_argument(
        "--drone-yaw-kp",
        type=float,
        default=0.8,
        help="P gain for yaw control",
    )
    parser.add_argument(
        "--drone-max-yaw-rate",
        type=float,
        default=0.8,
        help="Max yaw rate",
    )
    parser.add_argument(
        "--drone-yaw-tolerance-deg",
        type=float,
        default=4.0,
        help="Yaw error tolerance in degrees",
    )
    parser.add_argument(
        "--drone-yaw-sign",
        type=float,
        default=1.0,
        help="Sign multiplier for palm yaw delta",
    )
    parser.add_argument(
        "--audio-input-rate",
        type=int,
        default=DEFAULT_AUDIO_INPUT_RATE,
        help="Incoming PCM sample rate from the client. Audio is resampled to 16000 Hz for VAD/Whisper.",
    )
    parser.add_argument(
        "--use-local-mic",
        action="store_true",
        help="Use a microphone attached to the ground station instead of the Orange Pi audio stream",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available local input microphones and exit",
    )
    parser.add_argument(
        "--mic-id",
        type=int,
        default=None,
        help="Local microphone device index for --use-local-mic. Default: first available input.",
    )
    parser.add_argument(
        "--mic-frames-per-buffer",
        type=int,
        default=AUDIO_CHUNK,
        help="PyAudio frames_per_buffer when using --use-local-mic",
    )
    parser.add_argument(
        "--disable-audio-recognition",
        action="store_true",
        help="Disable audio VAD/transcription",
    )
    parser.add_argument(
        "--disable-voice-command-pub",
        action="store_true",
        help="Do not publish recognized voice commands to ROS",
    )
    parser.add_argument(
        "--voice-command-topic",
        type=str,
        default="/voice_cmd",
        help="ROS topic for canonical voice commands such as stay/follow/explore_face/land",
    )
    parser.add_argument(
        "--voice-command-min-repeat-sec",
        type=float,
        default=1.5,
        help="Minimum time before publishing the same canonical voice command again",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium",
        help="Whisper model size for audio transcription",
    )
    parser.add_argument(
        "--audio-language",
        type=str,
        default="",
        help="Optional Whisper language code. Empty means auto-detect.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech threshold",
    )
    parser.add_argument(
        "--vad-end-delay-sec",
        type=float,
        default=2.0,
        help="Seconds of silence to wait before finalizing a phrase",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_mics:
        list_input_microphones()
        return

    log.info("=== ZMQ Gesture Realtime Server ===")
    log.info(
        "Ports: RGB=%d Depth=%d Audio=%d | MAVROS prefix=%s",
        args.port_rgb,
        args.port_depth,
        args.port_audio,
        args.mavros_prefix,
    )
    if args.no_drone_control:
        log.info("Drone control publishing is disabled (--no-drone-control). No /.../setpoint_velocity/cmd_vel topic will be created by this server.")
    else:
        log.info(
            "Remote velocity commands will be published on %s/setpoint_velocity/cmd_vel",
            args.mavros_prefix,
        )

    ctx = zmq.Context()
    runtime = GestureServerRuntime(args)
    receivers = [
        RGBReceiver(ctx, runtime, args.bind_host, args.port_rgb),
        DepthReceiver(ctx, runtime, args.bind_host, args.port_depth),
    ]
    if args.use_local_mic:
        receivers.append(
            LocalAudioReceiver(
                runtime,
                sample_rate=args.audio_input_rate,
                frames_per_buffer=args.mic_frames_per_buffer,
                microphone_device_id=args.mic_id,
            )
        )
    else:
        receivers.append(AudioReceiver(ctx, runtime, args.bind_host, args.port_audio))

    for receiver in receivers:
        receiver.start()

    try:
        while not runtime.should_stop():
            time.sleep(0.2)
    except KeyboardInterrupt:
        log.info("Stopping gesture server...")
    finally:
        for receiver in receivers:
            receiver.stop()
        for receiver in receivers:
            receiver.join(timeout=3.0)
        runtime.close()
        ctx.term()
        log.info("Gesture server stopped.")


if __name__ == "__main__":
    main()
