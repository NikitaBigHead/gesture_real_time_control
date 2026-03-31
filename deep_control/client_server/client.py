"""
ZMQ Client — Orange Pi
Sends RGB frames, depth data, and microphone audio to the server (PC).

Dependencies:
    pip install pyzmq numpy pyrealsense2 pyaudio opencv-python
"""

import time
import threading
import struct
import logging
import argparse
from typing import Optional

import zmq
import numpy as np
import cv2

# --- Optional imports (they may be missing in the dev environment) ---
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logging.warning("pyrealsense2 не найден — используется заглушка-генератор кадров.")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("pyaudio не найден — аудиопоток будет эмулирован.")


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
SERVER_IP   = "192.168.1.100"   # IP of your PC
PORT_RGB    = 5550
PORT_DEPTH  = 5551
PORT_AUDIO  = 5552

# RealSense
RS_WIDTH    = 640
RS_HEIGHT   = 480
RS_FPS      = 30

# Audio
AUDIO_RATE       = 16000   # Hz
AUDIO_CHANNELS   = 1
AUDIO_CHUNK      = 1024    # samples per read
AUDIO_FORMAT_PA  = 8       # pyaudio.paInt16
MICROPHONE_DEVICE_ID = None  # Use None for the first available microphone

# JPEG compression for RGB (0–100)
JPEG_QUALITY = 80

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  ZMQ Publisher — wrapper
# ─────────────────────────────────────────────
class Publisher:
    """Simple ZMQ PUB socket with reconnect support."""

    def __init__(self, port: int, topic: bytes):
        self._ctx   = zmq.Context.instance()
        self._sock  = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 10)        # do not build up lag
        self._sock.connect(f"tcp://{SERVER_IP}:{port}")
        self._topic = topic
        time.sleep(0.3)  # give ZMQ time to connect

    def send(self, payload: bytes) -> None:
        """Send a message: [topic][timestamp_8b][payload]."""
        ts = struct.pack("d", time.time())
        self._sock.send_multipart([self._topic, ts, payload], zmq.NOBLOCK)

    def close(self) -> None:
        self._sock.close()


# ─────────────────────────────────────────────
#  RGB thread
# ─────────────────────────────────────────────
class RGBStreamer(threading.Thread):
    def __init__(self, pipeline_ref, pub: Publisher):
        super().__init__(daemon=True, name="RGBStreamer")
        self._pipe   = pipeline_ref
        self._pub    = pub
        self._stop   = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("RGB-поток запущен.")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        while not self._stop.is_set():
            try:
                if REALSENSE_AVAILABLE and self._pipe:
                    frames      = self._pipe.wait_for_frames(timeout_ms=200)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    img = np.asanyarray(color_frame.get_data())
                else:
                    # Fallback: colored noise
                    img = np.random.randint(0, 255,
                                           (RS_HEIGHT, RS_WIDTH, 3),
                                           dtype=np.uint8)

                _, buf = cv2.imencode(".jpg", img, encode_param)
                self._pub.send(buf.tobytes())

            except Exception as exc:
                log.error("RGB ошибка: %s", exc)
                time.sleep(0.1)

        log.info("RGB-поток остановлен.")


# ─────────────────────────────────────────────
#  Depth thread
# ─────────────────────────────────────────────
class DepthStreamer(threading.Thread):
    def __init__(self, pipeline_ref, pub: Publisher):
        super().__init__(daemon=True, name="DepthStreamer")
        self._pipe = pipeline_ref
        self._pub  = pub
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("Depth-поток запущен.")

        while not self._stop.is_set():
            try:
                if REALSENSE_AVAILABLE and self._pipe:
                    frames       = self._pipe.wait_for_frames(timeout_ms=200)
                    depth_frame  = frames.get_depth_frame()
                    if not depth_frame:
                        continue
                    depth_arr = np.asanyarray(depth_frame.get_data())   # uint16, mm
                else:
                    # Fallback: random depth from 0 to 4000 mm
                    depth_arr = np.random.randint(0, 4000,
                                                  (RS_HEIGHT, RS_WIDTH),
                                                  dtype=np.uint16)

                # Send raw bytes (uint16 little-endian).
                # The size is fixed, so the server knows it from the config.
                self._pub.send(depth_arr.tobytes())

            except Exception as exc:
                log.error("Depth ошибка: %s", exc)
                time.sleep(0.1)

        log.info("Depth-поток остановлен.")


# ─────────────────────────────────────────────
#  Audio thread
# ─────────────────────────────────────────────
class AudioStreamer(threading.Thread):
    def __init__(self, pub: Publisher, microphone_device_id: Optional[int] = None):
        super().__init__(daemon=True, name="AudioStreamer")
        self._pub = pub
        self._stop = threading.Event()
        self._microphone_device_id = microphone_device_id

    def stop(self):
        self._stop.set()

    def _open_microphone(self):
        """Open the selected microphone and return (pa_instance, stream)."""
        pa = pyaudio.PyAudio()

        if self._microphone_device_id is not None:
            info = pa.get_device_info_by_index(self._microphone_device_id)
            if info["maxInputChannels"] <= 0:
                raise ValueError(
                    f"Device {self._microphone_device_id} has no input channels."
                )
            device_index = self._microphone_device_id
            log.info("Using microphone: [%d] %s", device_index, info["name"])
        else:
            # Find the first device with audio input
            device_index = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    device_index = i
                    log.info("Using first available microphone: [%d] %s", i, info["name"])
                    break

            if device_index is None:
                raise RuntimeError("No input microphone was found.")

        stream = pa.open(
            format=AUDIO_FORMAT_PA,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=AUDIO_CHUNK,
        )
        return pa, stream

    def run(self):
        log.info("Аудио-поток запущен.")

        if PYAUDIO_AVAILABLE:
            try:
                pa, stream = self._open_microphone()
            except Exception as exc:
                log.error("Не удалось открыть микрофон: %s — переключаюсь на заглушку.", exc)
                pa, stream = None, None
        else:
            pa, stream = None, None

        try:
            while not self._stop.is_set():
                try:
                    if stream:
                        raw = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                    else:
                        # Fallback: silence with light noise
                        silence = np.zeros(AUDIO_CHUNK, dtype=np.int16)
                        silence += np.random.randint(-50, 50,
                                                     AUDIO_CHUNK,
                                                     dtype=np.int16)
                        raw = silence.tobytes()
                        time.sleep(AUDIO_CHUNK / AUDIO_RATE)

                    self._pub.send(raw)

                except Exception as exc:
                    log.error("Аудио ошибка: %s", exc)
                    time.sleep(0.05)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if pa:
                pa.terminate()

        log.info("Аудио-поток остановлен.")


# ─────────────────────────────────────────────
#  Main client function
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ RealSense client")
    parser.add_argument(
        "--mic-id",
        type=int,
        default=MICROPHONE_DEVICE_ID,
        help="Microphone device id from PyAudio. Default: first available input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log.info("=== ZMQ RealSense Client ===")
    log.info("Сервер: %s  |  RGB:%d  Depth:%d  Audio:%d",
             SERVER_IP, PORT_RGB, PORT_DEPTH, PORT_AUDIO)
    log.info("Microphone device id: %s",
             args.mic_id if args.mic_id is not None else "auto")

    # Initialize RealSense
    pipeline = None
    if REALSENSE_AVAILABLE:
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT,
                              rs.format.bgr8, RS_FPS)
            cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT,
                              rs.format.z16,  RS_FPS)
            pipeline.start(cfg)
            log.info("RealSense запущен (%dx%d @ %d fps).",
                     RS_WIDTH, RS_HEIGHT, RS_FPS)
        except Exception as exc:
            log.error("RealSense не найден: %s — работаю с заглушками.", exc)
            pipeline = None

    # Create publishers
    pub_rgb   = Publisher(PORT_RGB,   b"rgb")
    pub_depth = Publisher(PORT_DEPTH, b"depth")
    pub_audio = Publisher(PORT_AUDIO, b"audio")

    # Start threads
    threads = [
        RGBStreamer(pipeline, pub_rgb),
        DepthStreamer(pipeline, pub_depth),
        AudioStreamer(pub_audio, microphone_device_id=args.mic_id),
    ]
    for t in threads:
        t.start()

    log.info("Все потоки запущены. Ctrl+C для остановки.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Остановка...")
    finally:
        for t in threads:
            t.stop()
        for t in threads:
            t.join(timeout=3)
        if pipeline:
            pipeline.stop()
        pub_rgb.close()
        pub_depth.close()
        pub_audio.close()
        log.info("Клиент завершён.")


if __name__ == "__main__":
    main()
