"""
Real-time microphone ASR with PyAudio, Silero VAD, Hugging Face Whisper,
and Ollama-based drone command extraction.

Reads audio from a local microphone, segments phrases with Silero VAD,
transcribes them with Whisper from Hugging Face, and optionally sends the
recognized text to Ollama so it can be converted into structured drone
commands.

Suggested packages:
    pip install pyaudio silero-vad transformers torch requests
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    Node = object
    String = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

try:
    from silero_vad import load_silero_vad
except ImportError:
    load_silero_vad = None

try:
    from olama_client import (
        DEFAULT_BASE_URL as DEFAULT_OLLAMA_HOST,
        DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL,
        DEFAULT_TIMEOUT_SEC as DEFAULT_OLLAMA_TIMEOUT_SEC,
        send_prompt as send_ollama_prompt,
    )
except ImportError:
    # DEFAULT_OLLAMA_HOST = "http://192.168.50.26:11434"
    DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
    DEFAULT_OLLAMA_MODEL = "qwen3.5:2b"
    DEFAULT_OLLAMA_TIMEOUT_SEC = 120
    send_ollama_prompt = None


MODEL_AUDIO_RATE = 16000
DEFAULT_INPUT_RATE = 48000
DEFAULT_CHUNK = 1024
SILERO_WINDOW_SAMPLES = 512
DEFAULT_WHISPER_MODEL = "openai/whisper-medium"
DEFAULT_VAD_THRESHOLD = 0.35
DEFAULT_ENERGY_THRESHOLD_RMS = 250.0
ALLOWED_DRONE_COMMANDS = {
    "forward",
    "backward",
    "up",
    "down",
    "left",
    "right",
    "turn right",
    "turn left",
}
ROTATION_COMMANDS = {"turn right", "turn left"}
_NULL_STRINGS = {"", "none", "null"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REALTIME_ASR] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def build_missing_dependency_message(missing: list[str]) -> str:
    package_map = {
        "pyaudio": "pyaudio",
        "torch": "torch",
        "transformers": "transformers",
        "silero-vad": "silero-vad",
    }
    install_packages = " ".join(package_map[name] for name in missing)
    install_cmd = (
        "/home/dzmitry/gesture_real_time_control/.venv/bin/pip install "
        + install_packages
    )

    lines = [
        "Missing packages in the active .venv: " + ", ".join(missing),
        "Install them with:",
        f"  {install_cmd}",
    ]
    if "pyaudio" in missing:
        lines.extend(
            [
                "If PyAudio build fails, install PortAudio first:",
                "  sudo apt install portaudio19-dev python3-dev",
            ]
        )
    return "\n".join(lines)


def normalize_whisper_model_name(model_name: str) -> str:
    if not model_name:
        return DEFAULT_WHISPER_MODEL

    if "/" in model_name or os.path.exists(model_name):
        return model_name

    return f"openai/whisper-{model_name}"


def normalize_optional_number(value: Any) -> Optional[int | float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _NULL_STRINGS:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group(0))
    else:
        return None

    return int(number) if number.is_integer() else number


def normalize_drone_command(command: Any) -> Optional[str]:
    if command is None:
        return None
    if not isinstance(command, str):
        return None

    normalized = " ".join(command.strip().lower().split())
    if normalized in _NULL_STRINGS:
        return None

    alias_map = {
        "move forward": "forward",
        "go forward": "forward",
        "move backward": "backward",
        "go backward": "backward",
        "move up": "up",
        "go up": "up",
        "move down": "down",
        "go down": "down",
        "move left": "left",
        "go left": "left",
        "move right": "right",
        "go right": "right",
        "rotate right": "turn right",
        "rotate left": "turn left",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in ALLOWED_DRONE_COMMANDS:
        return None
    return normalized


class DroneCommandExtractor:
    def __init__(self, ollama_host: str, ollama_model: str, timeout_sec: int):
        if send_ollama_prompt is None:
            raise RuntimeError("Ollama client is unavailable. Make sure requests is installed.")

        self._ollama_host = ollama_host
        self._ollama_model = ollama_model
        self._timeout_sec = timeout_sec

    def extract(self, transcript: str) -> dict[str, Optional[str | int | float]]:
        raw_response = send_ollama_prompt(
            prompt=self._build_prompt(transcript),
            base_url=self._ollama_host,
            model=self._ollama_model,
            timeout_sec=self._timeout_sec,
        )
        return self._normalize_payload(self._parse_response(raw_response))

    def _build_prompt(self, transcript: str) -> str:
        return (
            "You convert recognized speech into a structured drone command.\n"
            "Allowed commands: forward, backward, up, down, left, right, turn right, turn left.\n"
            "Return JSON only with exactly these keys: command, distance, angle.\n"
            "Rules:\n"
            '- "command" must be one of the allowed commands or null if no drone command is present.\n'
            '- "distance" must be a number only when the user explicitly gives a movement distance; otherwise null.\n'
            '- "angle" must be a number in degrees only when the user explicitly gives a turn angle; otherwise null.\n'
            "- Do not invent a distance or angle if the phrase does not define one.\n"
            "- For forward/backward/up/down/left/right commands, angle must be null.\n"
            "- For turn right/turn left commands, distance must be null.\n"
            "- Convert written numbers to numerals.\n"
            "- Output JSON only, with no markdown and no explanation.\n\n"
            f"Recognized phrase: {json.dumps(transcript, ensure_ascii=True)}"
        )

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        candidates = [text]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match and match.group(0) not in candidates:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

            try:
                payload = ast.literal_eval(candidate)
                if isinstance(payload, dict):
                    return payload
            except (SyntaxError, ValueError):
                pass

        raise RuntimeError(f"Could not parse Ollama response as a dict: {response_text}")

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Optional[str | int | float]]:
        command = normalize_drone_command(payload.get("command"))
        distance = normalize_optional_number(payload.get("distance"))
        angle = normalize_optional_number(payload.get("angle"))

        if distance is not None:
            distance = min(distance, 2)

        if command in ROTATION_COMMANDS:
            distance = None
        elif command in ALLOWED_DRONE_COMMANDS:
            angle = None
        else:
            distance = None
            angle = None

        return {
            "command": command,
            "distance": distance,
            "angle": angle,
        }


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


def list_input_devices() -> None:
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


def open_microphone_stream(
    microphone_device_id: Optional[int],
    input_rate: int,
    frames_per_buffer: int,
):
    if pyaudio is None:
        raise RuntimeError("PyAudio is not installed.")

    pa = pyaudio.PyAudio()
    if microphone_device_id is not None:
        info = pa.get_device_info_by_index(microphone_device_id)
        if info.get("maxInputChannels", 0) <= 0:
            pa.terminate()
            raise RuntimeError(f"Device {microphone_device_id} has no input channels.")
        device_index = microphone_device_id
        log.info("Using microphone [%d] %s", device_index, info.get("name", "Unknown"))
    else:
        device_index = None
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                device_index = index
                log.info("Using first available microphone [%d] %s", index, info.get("name", "Unknown"))
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


class RealTimeMicAsr:
    def __init__(
        self,
        input_rate: int,
        frames_per_buffer: int,
        whisper_model_name: str,
        language: Optional[str],
        vad_threshold: float,
        phrase_end_delay_sec: float,
        ollama_host: str,
        ollama_model: str,
        ollama_timeout_sec: int,
        enable_drone_commands: bool = True,
        pre_roll_sec: float = 0.35,
        min_phrase_sec: float = 0.30,
        on_drone_command=None,
        on_transcript=None,
        audio_debug: bool = False,
        energy_threshold_rms: float = DEFAULT_ENERGY_THRESHOLD_RMS,
    ):
        missing = []
        if pyaudio is None:
            missing.append("pyaudio")
        if torch is None:
            missing.append("torch")
        if hf_pipeline is None:
            missing.append("transformers")
        if load_silero_vad is None:
            missing.append("silero-vad")
        if missing:
            raise RuntimeError(build_missing_dependency_message(missing))

        self._input_rate = input_rate
        self._frames_per_buffer = frames_per_buffer
        self._language = language
        self._vad_threshold = vad_threshold
        self._phrase_end_delay_sec = phrase_end_delay_sec
        self._min_phrase_samples = int(min_phrase_sec * MODEL_AUDIO_RATE)
        self._pre_roll_chunks = max(1, int(pre_roll_sec * input_rate / frames_per_buffer))
        self._on_drone_command = on_drone_command
        self._on_transcript = on_transcript
        self._audio_debug = audio_debug
        self._energy_threshold_rms = max(0.0, float(energy_threshold_rms))
        self._last_audio_debug_monotonic = 0.0

        self._converter = AudioRateConverter(input_rate, MODEL_AUDIO_RATE)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._pre_roll_chunks)
        self._phrase_chunks: list[np.ndarray] = []
        self._phrase_active = False
        self._last_speech_monotonic: Optional[float] = None
        self._vad_tail = np.empty(0, dtype=np.int16)
        self._speech_started_logged = False

        self._vad_model = load_silero_vad()
        if hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()

        self._whisper_model_name = normalize_whisper_model_name(whisper_model_name)
        self._whisper_pipeline = None
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pipeline_device = 0 if torch.cuda.is_available() else -1
        self._drone_command_extractor: Optional[DroneCommandExtractor] = None

        self._transcription_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._command_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._transcription_worker_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="HuggingFaceWhisperWorker",
        )
        self._command_worker_thread: Optional[threading.Thread] = None

        if enable_drone_commands:
            self._drone_command_extractor = DroneCommandExtractor(
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                timeout_sec=ollama_timeout_sec,
            )
            self._command_worker_thread = threading.Thread(
                target=self._command_worker,
                daemon=True,
                name="OllamaDroneCommandWorker",
            )
            log.info(
                "Drone command extraction enabled: host=%s model=%s",
                ollama_host,
                ollama_model,
            )
        else:
            log.info("Drone command extraction disabled.")

        self._transcription_worker_thread.start()
        if self._command_worker_thread is not None:
            self._command_worker_thread.start()

        log.info(
            "ASR ready: input_rate=%dHz -> %dHz, chunk=%d, HuggingFace Whisper=%s",
            input_rate,
            MODEL_AUDIO_RATE,
            frames_per_buffer,
            self._whisper_model_name,
        )

    def _chunk_contains_speech(self, samples: np.ndarray) -> bool:
        merged = np.concatenate((self._vad_tail, samples))
        speech_detected = False
        offset = 0

        while offset + SILERO_WINDOW_SAMPLES <= len(merged):
            window = merged[offset:offset + SILERO_WINDOW_SAMPLES]
            offset += SILERO_WINDOW_SAMPLES

            audio_tensor = torch.from_numpy(window.astype(np.float32) / 32768.0)
            speech_prob = float(self._vad_model(audio_tensor, MODEL_AUDIO_RATE).item())
            if speech_prob >= self._vad_threshold:
                speech_detected = True

        self._vad_tail = merged[offset:]
        return speech_detected

    def process_raw_audio(self, raw_bytes: bytes) -> None:
        input_samples = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        chunk = self._converter.convert(input_samples)
        if chunk.size == 0:
            return

        self._pre_roll.append(chunk)
        vad_detected = self._chunk_contains_speech(chunk)
        energy_detected = self._chunk_has_energy(chunk)
        speech_detected = vad_detected or energy_detected
        self._maybe_log_audio_debug(chunk, speech_detected, vad_detected, energy_detected)
        now = time.monotonic()

        if speech_detected:
            if not self._phrase_active:
                self._phrase_chunks = [pre.copy() for pre in self._pre_roll]
                self._phrase_active = True
                if not self._speech_started_logged:
                    log.info("Speech detected. Listening for end of phrase...")
                    self._speech_started_logged = True
            else:
                self._phrase_chunks.append(chunk)
            self._last_speech_monotonic = now
            return

        if not self._phrase_active:
            return

        self._phrase_chunks.append(chunk)
        if self._last_speech_monotonic is None:
            return

        if now - self._last_speech_monotonic >= self._phrase_end_delay_sec:
            self._finalize_phrase()

    def _finalize_phrase(self) -> None:
        if not self._phrase_chunks:
            self._reset_phrase_state()
            return

        utterance = np.concatenate(self._phrase_chunks)
        self._reset_phrase_state(reset_vad=True)
        if utterance.size < self._min_phrase_samples:
            return

        log.info(
            "Phrase captured: %.2f sec. Transcribing...",
            utterance.size / MODEL_AUDIO_RATE,
        )
        self._transcription_queue.put_nowait(utterance)

    def _reset_phrase_state(self, reset_vad: bool = False) -> None:
        self._phrase_chunks = []
        self._phrase_active = False
        self._last_speech_monotonic = None
        self._vad_tail = np.empty(0, dtype=np.int16)
        self._speech_started_logged = False
        if reset_vad and hasattr(self._vad_model, "reset_states"):
            self._vad_model.reset_states()

    def _chunk_has_energy(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        chunk_f32 = chunk.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(chunk_f32))))
        return rms >= self._energy_threshold_rms

    def _maybe_log_audio_debug(
        self,
        chunk: np.ndarray,
        speech_detected: bool,
        vad_detected: bool,
        energy_detected: bool,
    ) -> None:
        if not self._audio_debug:
            return

        now = time.monotonic()
        if now - self._last_audio_debug_monotonic < 1.0:
            return

        chunk_f32 = chunk.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(chunk_f32)))) if chunk_f32.size else 0.0
        peak = int(np.max(np.abs(chunk))) if chunk.size else 0
        log.info(
            "Audio debug: rms=%.1f peak=%d speech=%s vad=%s energy=%s vad_threshold=%.2f energy_threshold_rms=%.1f",
            rms,
            peak,
            "yes" if speech_detected else "no",
            "yes" if vad_detected else "no",
            "yes" if energy_detected else "no",
            self._vad_threshold,
            self._energy_threshold_rms,
        )
        self._last_audio_debug_monotonic = now

    def _transcription_worker(self) -> None:
        while True:
            utterance = self._transcription_queue.get()
            if utterance is None:
                break

            try:
                text = self._transcribe(utterance)
            except Exception as exc:
                log.error("Hugging Face Whisper transcription failed: %s", exc)
                continue

            if not text:
                continue

            print(f"[ASR] {text}", flush=True)

            try:
                if self._on_transcript is not None:
                    self._on_transcript(text)
            except Exception as exc:
                log.error("Transcript callback failed: %s", exc)
                continue

            try:
                self._enqueue_drone_command(text)
            except Exception as exc:
                log.error("Drone command enqueue failed: %s", exc)

    def _transcribe(self, utterance: np.ndarray) -> str:
        if self._whisper_pipeline is None:
            log.info("Loading Hugging Face Whisper model '%s'...", self._whisper_model_name)
            self._whisper_pipeline = hf_pipeline(
                task="automatic-speech-recognition",
                model=self._whisper_model_name,
                torch_dtype=self._torch_dtype,
                device=self._pipeline_device,
            )

        audio = utterance.astype(np.float32) / 32768.0
        generate_kwargs = {"task": "transcribe"}
        if self._language:
            generate_kwargs["language"] = self._language
        result = self._whisper_pipeline(
            {"sampling_rate": MODEL_AUDIO_RATE, "raw": audio},
            generate_kwargs=generate_kwargs,
        )
        return result.get("text", "").strip()

    def _enqueue_drone_command(self, transcript: str) -> None:
        if self._command_worker_thread is None:
            return
        self._command_queue.put_nowait(transcript)

    def _command_worker(self) -> None:
        while True:
            transcript = self._command_queue.get()
            if transcript is None:
                break

            self._emit_drone_command(transcript)

    # def _emit_drone_command(self, transcript: str) -> None:
    #     if self._drone_command_extractor is None:
    #         return

    #     try:
    #         command_dict = self._drone_command_extractor.extract(transcript)
    #         print(f"[DRONE_CMD] {command_dict}", flush=True)
    #     except Exception as exc:
    #         log.error("Ollama command extraction failed: %s", exc)

    def _emit_drone_command(self, transcript: str) -> None:
        if self._drone_command_extractor is not None:
            try:
                command_dict = self._drone_command_extractor.extract(transcript)
                print(f"[DRONE_CMD] {command_dict}", flush=True)
                if command_dict.get("command") is not None and self._on_drone_command is not None:
                    self._on_drone_command(command_dict)
                return
            except Exception as exc:
                log.error("Ollama command extraction failed: %s", exc)

        text = transcript.lower()

        fallback = None
        if "forward" in text:
            fallback = {"command": "forward", "distance": 1, "angle": None}
        elif "backward" in text or "backwards" in text:
            fallback = {"command": "backward", "distance": 1, "angle": None}
        elif "left" in text and "turn" in text:
            fallback = {"command": "turn left", "distance": None, "angle": 45}
        elif "right" in text and "turn" in text:
            fallback = {"command": "turn right", "distance": None, "angle": 45}

        if fallback is not None:
            print(f"[DRONE_CMD_FALLBACK] {fallback}", flush=True)
            if self._on_drone_command is not None:
                self._on_drone_command(fallback)
                


    def close(self) -> None:
        if self._phrase_active and self._phrase_chunks:
            self._finalize_phrase()

        self._transcription_queue.put_nowait(None)
        self._transcription_worker_thread.join(timeout=5.0)

        if self._command_worker_thread is not None:
            self._command_queue.put_nowait(None)
            self._command_worker_thread.join(timeout=5.0)

class VoiceCommandPublisher(Node):
    def __init__(self):
        super().__init__('voice_command_publisher')
        self.pub = self.create_publisher(String, '/voice/drone_command', 10)
        self._queue = queue.Queue()
        self.create_timer(0.05, self._flush_queue)
        self.get_logger().info('Voice command publisher started')

    def enqueue_command(self, command_dict):
        self._queue.put(command_dict)

    def _flush_queue(self):
        while not self._queue.empty():
            command_dict = self._queue.get()
            if command_dict.get('command') is None:
                continue
            msg = String()
            msg.data = json.dumps(command_dict)
            self.pub.publish(msg)
            self.get_logger().info(f'Published voice command: {msg.data}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time microphone ASR with VAD")
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
        "--disable-ollama",
        action="store_true",
        help="Disable Ollama command extraction and only print ASR text.",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama server URL. Default: {DEFAULT_OLLAMA_HOST}",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_OLLAMA_MODEL}",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT_SEC,
        help=f"Ollama request timeout in seconds. Default: {DEFAULT_OLLAMA_TIMEOUT_SEC}",
    )
    return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     if args.list_devices:
#         list_input_devices()
#         return

#     asr = RealTimeMicAsr(
#         input_rate=args.input_rate,
#         frames_per_buffer=args.frames_per_buffer,
#         whisper_model_name=args.whisper_model,
#         language=args.language or None,
#         vad_threshold=args.vad_threshold,
#         phrase_end_delay_sec=args.phrase_end_delay_sec,
#         ollama_host=args.ollama_host,
#         ollama_model=args.ollama_model,
#         ollama_timeout_sec=args.ollama_timeout,
#         enable_drone_commands=not args.disable_ollama,
#     )
#     pa = None
#     stream = None

#     try:
#         pa, stream = open_microphone_stream(
#             microphone_device_id=args.mic_id,
#             input_rate=args.input_rate,
#             frames_per_buffer=args.frames_per_buffer,
#         )
#         log.info("Listening... Press Ctrl+C to stop.")
#         while True:
#             raw = stream.read(args.frames_per_buffer, exception_on_overflow=False)
#             asr.process_raw_audio(raw)
#     except KeyboardInterrupt:
#         log.info("Stopping...")
#     finally:
#         asr.close()
#         if stream is not None:
#             stream.stop_stream()
#             stream.close()
#         if pa is not None:
#             pa.terminate()

def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    rclpy.init()
    ros_node = VoiceCommandPublisher()
    ros_spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_spin_thread.start()

    asr = RealTimeMicAsr(
        input_rate=args.input_rate,
        frames_per_buffer=args.frames_per_buffer,
        whisper_model_name=args.whisper_model,
        language=args.language or None,
        vad_threshold=args.vad_threshold,
        phrase_end_delay_sec=args.phrase_end_delay_sec,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        ollama_timeout_sec=args.ollama_timeout,
        enable_drone_commands=not args.disable_ollama,
        on_drone_command=ros_node.enqueue_command,
        audio_debug=args.audio_debug,
        energy_threshold_rms=args.energy_threshold_rms,
    )

    pa = None
    stream = None

    try:
        pa, stream = open_microphone_stream(
            microphone_device_id=args.mic_id,
            input_rate=args.input_rate,
            frames_per_buffer=args.frames_per_buffer,
        )
        log.info("Listening... Press Ctrl+C to stop.")
        while True:
            raw = stream.read(args.frames_per_buffer, exception_on_overflow=False)
            asr.process_raw_audio(raw)
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        asr.close()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if pa is not None:
            pa.terminate()
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
