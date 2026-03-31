"""
Real-time microphone ASR with PyAudio, Silero VAD, and Hugging Face Whisper.

Reads audio from a local microphone, segments phrases with Silero VAD,
transcribes them with Whisper from Hugging Face, and prints recognized
speech to the console.

Suggested packages:
    pip install pyaudio silero-vad transformers torch
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

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


MODEL_AUDIO_RATE = 16000
DEFAULT_INPUT_RATE = 48000
DEFAULT_CHUNK = 1024
SILERO_WINDOW_SAMPLES = 512
DEFAULT_WHISPER_MODEL = "openai/whisper-medium"

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
        pre_roll_sec: float = 0.35,
        min_phrase_sec: float = 0.30,
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

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="HuggingFaceWhisperWorker",
        )
        self._worker.start()

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
        speech_detected = self._chunk_contains_speech(chunk)
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
        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            log.warning("Dropping phrase because transcription queue is full.")

    def _reset_phrase_state(self, reset_vad: bool = False) -> None:
        self._phrase_chunks = []
        self._phrase_active = False
        self._last_speech_monotonic = None
        self._vad_tail = np.empty(0, dtype=np.int16)
        self._speech_started_logged = False
        if reset_vad and hasattr(self._vad_model, "reset_states"):
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
                    print(f"[ASR] {text}", flush=True)
            except Exception as exc:
                log.error("Hugging Face Whisper transcription failed: %s", exc)

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

    def close(self) -> None:
        if self._phrase_active and self._phrase_chunks:
            self._finalize_phrase()

        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=5.0)


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
        default=0.5,
        help="Silero VAD threshold",
    )
    parser.add_argument(
        "--phrase-end-delay-sec",
        type=float,
        default=1.2,
        help="Seconds of silence before a phrase is finalized",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    asr = RealTimeMicAsr(
        input_rate=args.input_rate,
        frames_per_buffer=args.frames_per_buffer,
        whisper_model_name=args.whisper_model,
        language=args.language or None,
        vad_threshold=args.vad_threshold,
        phrase_end_delay_sec=args.phrase_end_delay_sec,
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


if __name__ == "__main__":
    main()
