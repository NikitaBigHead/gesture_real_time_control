"""
ZMQ Server — ПК
Receives RGB frames, depth data, and audio from the client (Orange Pi).
Contains placeholder handlers for each stream.

Dependencies:
    pip install pyzmq numpy opencv-python
"""

import time
import struct
import threading
import logging
from typing import Optional
from collections import deque

import zmq
import numpy as np
import cv2


# ─────────────────────────────────────────────
#  Configuration (must match client.py)
# ─────────────────────────────────────────────
BIND_HOST  = "0.0.0.0"   # listen on all interfaces
PORT_RGB   = 5550
PORT_DEPTH = 5551
PORT_AUDIO = 5552

# RealSense — depth frame size
RS_WIDTH  = 640
RS_HEIGHT = 480

# Audio
AUDIO_RATE     = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK    = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Stream statistics
# ─────────────────────────────────────────────
class StreamStats:
    """Frame and delay counter for one stream."""

    def __init__(self, name: str, window: int = 100):
        self.name    = name
        self._lock   = threading.Lock()
        self._count  = 0
        self._delays: deque = deque(maxlen=window)
        self._t_prev: Optional[float] = None

    def record(self, send_ts: float):
        now  = time.time()
        delay = (now - send_ts) * 1000  # ms
        with self._lock:
            self._count += 1
            self._delays.append(delay)

            # Log once every 100 frames
            if self._count % 100 == 0:
                avg = np.mean(self._delays)
                log.info("[%s] кадров: %d | задержка avg=%.1f мс",
                         self.name, self._count, avg)


# ─────────────────────────────────────────────
#  Placeholder handlers
#  Replace the body of these functions with your own logic.
# ─────────────────────────────────────────────

def handle_rgb(frame: np.ndarray, send_ts: float) -> None:
    """
    PLACEHOLDER: process an RGB frame.

    Parameters
    ----------
    frame   : np.ndarray  — BGR image, shape (H, W, 3), dtype uint8
    send_ts : float       — UNIX send time on the client side (seconds)

    Ideas for implementation:
        - object detection (YOLO, MediaPipe ...)
        - face or pose tracking
        - video recording (cv2.VideoWriter)
        - display in a cv2.imshow window
    """
    # Example: show the frame (comment it out if there is no display)
    # cv2.imshow("RGB", frame)
    # cv2.waitKey(1)
    pass


def handle_depth(depth: np.ndarray, send_ts: float) -> None:
    """
    PLACEHOLDER: process a depth map.

    Parameters
    ----------
    depth   : np.ndarray  — depth map, shape (H, W), dtype uint16, units are mm
    send_ts : float       — UNIX send time on the client side

    Ideas for implementation:
        - point cloud generation
        - obstacle detection
        - map building (SLAM)
        - colormap visualization
    """
    # Example: colormap visualization
    # depth_viz = cv2.applyColorMap(
    #     cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
    # )
    # cv2.imshow("Depth", depth_viz)
    # cv2.waitKey(1)
    pass


def handle_audio(samples: np.ndarray, send_ts: float) -> None:
    """
    PLACEHOLDER: process an audio buffer.

    Parameters
    ----------
    samples : np.ndarray  — int16, shape (AUDIO_CHUNK * AUDIO_CHANNELS,)
    send_ts : float       — UNIX send time on the client side

    Ideas for implementation:
        - VAD (Voice Activity Detection)
        - speech recognition (Whisper, Vosk ...)
        - save to a WAV file
        - stream to an external system
    """
    # Example: estimate the signal level
    # rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    # if rms > 500:
    #     log.debug("Audio: RMS=%.0f (activity)", rms)
    pass


# ─────────────────────────────────────────────
#  Receivers (subscriber threads)
# ─────────────────────────────────────────────

class RGBReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context):
        super().__init__(daemon=True, name="RGBReceiver")
        self._sock  = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 10)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)        # ms
        self._sock.bind(f"tcp://{BIND_HOST}:{PORT_RGB}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"rgb")
        self._stop  = threading.Event()
        self._stats = StreamStats("RGB")

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("RGBReceiver слушает на порту %d.", PORT_RGB)
        while not self._stop.is_set():
            try:
                topic, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]

                # Decode JPEG into BGR
                buf   = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    log.warning("RGB: не удалось декодировать кадр.")
                    continue

                self._stats.record(send_ts)
                handle_rgb(frame, send_ts)

            except zmq.Again:
                pass   # timeout, keep going
            except Exception as exc:
                log.error("RGBReceiver: %s", exc)

        self._sock.close()
        log.info("RGBReceiver остановлен.")


class DepthReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context):
        super().__init__(daemon=True, name="DepthReceiver")
        self._sock  = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 10)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.bind(f"tcp://{BIND_HOST}:{PORT_DEPTH}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"depth")
        self._stop  = threading.Event()
        self._stats = StreamStats("Depth")
        self._frame_bytes = RS_WIDTH * RS_HEIGHT * 2  # uint16

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("DepthReceiver слушает на порту %d.", PORT_DEPTH)
        while not self._stop.is_set():
            try:
                topic, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]

                if len(payload) != self._frame_bytes:
                    log.warning("Depth: неожиданный размер %d байт.", len(payload))
                    continue

                depth = np.frombuffer(payload, dtype=np.uint16).reshape(RS_HEIGHT, RS_WIDTH)
                self._stats.record(send_ts)
                handle_depth(depth, send_ts)

            except zmq.Again:
                pass
            except Exception as exc:
                log.error("DepthReceiver: %s", exc)

        self._sock.close()
        log.info("DepthReceiver остановлен.")


class AudioReceiver(threading.Thread):
    def __init__(self, ctx: zmq.Context):
        super().__init__(daemon=True, name="AudioReceiver")
        self._sock  = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 50)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.bind(f"tcp://{BIND_HOST}:{PORT_AUDIO}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"audio")
        self._stop  = threading.Event()
        self._stats = StreamStats("Audio")

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("AudioReceiver слушает на порту %d.", PORT_AUDIO)
        while not self._stop.is_set():
            try:
                topic, ts_bytes, payload = self._sock.recv_multipart()
                send_ts = struct.unpack("d", ts_bytes)[0]

                samples = np.frombuffer(payload, dtype=np.int16)
                self._stats.record(send_ts)
                handle_audio(samples, send_ts)

            except zmq.Again:
                pass
            except Exception as exc:
                log.error("AudioReceiver: %s", exc)

        self._sock.close()
        log.info("AudioReceiver остановлен.")


# ─────────────────────────────────────────────
#  Main server function
# ─────────────────────────────────────────────
def main():
    log.info("=== ZMQ RealSense Server ===")
    log.info("Ожидаю подключение клиента (Orange Pi)...")
    log.info("Порты:  RGB=%d  Depth=%d  Audio=%d",
             PORT_RGB, PORT_DEPTH, PORT_AUDIO)

    ctx = zmq.Context()

    receivers = [
        RGBReceiver(ctx),
        DepthReceiver(ctx),
        AudioReceiver(ctx),
    ]
    for r in receivers:
        r.start()

    log.info("Сервер запущен. Ctrl+C для остановки.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Остановка...")
    finally:
        for r in receivers:
            r.stop()
        for r in receivers:
            r.join(timeout=3)
        ctx.term()
        log.info("Сервер завершён.")


if __name__ == "__main__":
    main()
