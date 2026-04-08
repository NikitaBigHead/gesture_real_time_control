from __future__ import annotations

import argparse
import logging
import struct
import time

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import zmq
except ImportError:
    zmq = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 5550
DEFAULT_TOPIC = "rgb"
DEFAULT_SOURCE = "webcam"
DEFAULT_CAMERA_ID = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_JPEG_QUALITY = 80
DEFAULT_CONNECT_DELAY_SEC = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RAG_VIDEO_CLIENT] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


class ZmqFramePublisher:
    def __init__(self, server_host: str, port: int, topic: str):
        if zmq is None:
            raise RuntimeError("pyzmq is not installed. Install it with: pip install pyzmq")

        self._endpoint = f"tcp://{server_host}:{port}"
        self._topic = topic.encode("utf-8")
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 4)
        self._sock.connect(self._endpoint)
        time.sleep(DEFAULT_CONNECT_DELAY_SEC)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def send_jpeg(self, jpeg_bytes: bytes) -> bool:
        timestamp = struct.pack("d", time.time())
        try:
            self._sock.send_multipart([self._topic, timestamp, jpeg_bytes], zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        self._sock.close()


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Install it with: pip install opencv-python")


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is not installed. Install it with: pip install numpy")


def create_jpeg(frame, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return encoded.tobytes()


def open_webcam(camera_id: int, width: int, height: int, fps: int):
    require_opencv()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with id={camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_realsense(width: int, height: int, fps: int):
    require_numpy()
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is not installed. Install the Intel RealSense Python package first."
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def maybe_show_preview(frame, enabled: bool) -> bool:
    if not enabled:
        return False

    cv2.imshow("RAG Video Client", frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def log_stats(sent_frames: int, dropped_frames: int, last_report: float) -> float:
    now = time.monotonic()
    if now - last_report >= 5.0:
        log.info("Frames sent=%d dropped=%d", sent_frames, dropped_frames)
        return now
    return last_report


def stream_webcam(args: argparse.Namespace, publisher: ZmqFramePublisher) -> int:
    cap = open_webcam(args.camera_id, args.width, args.height, args.fps)
    log.info(
        "Streaming webcam id=%d to %s topic=%s",
        args.camera_id,
        publisher.endpoint,
        args.topic,
    )

    sent_frames = 0
    dropped_frames = 0
    last_report = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Webcam read failed.")
                time.sleep(0.05)
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            jpeg_bytes = create_jpeg(frame, args.jpeg_quality)
            if publisher.send_jpeg(jpeg_bytes):
                sent_frames += 1
            else:
                dropped_frames += 1

            if maybe_show_preview(frame, args.preview):
                break

            last_report = log_stats(sent_frames, dropped_frames, last_report)
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    return 0


def stream_realsense(args: argparse.Namespace, publisher: ZmqFramePublisher) -> int:
    require_opencv()
    require_numpy()
    pipeline = open_realsense(args.width, args.height, args.fps)
    log.info(
        "Streaming RealSense color video to %s topic=%s",
        publisher.endpoint,
        args.topic,
    )

    sent_frames = 0
    dropped_frames = 0
    last_report = time.monotonic()

    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            if args.mirror:
                frame = cv2.flip(frame, 1)

            jpeg_bytes = create_jpeg(frame, args.jpeg_quality)
            if publisher.send_jpeg(jpeg_bytes):
                sent_frames += 1
            else:
                dropped_frames += 1

            if maybe_show_preview(frame, args.preview):
                break

            last_report = log_stats(sent_frames, dropped_frames, last_report)
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone video streaming client for rag_interaction-compatible ZMQ RGB receivers.",
    )
    parser.add_argument(
        "--server-host",
        default=DEFAULT_SERVER_HOST,
        help=f"Receiver host. Default: {DEFAULT_SERVER_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Receiver RGB port. Default: {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"ZMQ topic name. Default: {DEFAULT_TOPIC}",
    )
    parser.add_argument(
        "--source",
        choices=("webcam", "realsense"),
        default=DEFAULT_SOURCE,
        help=f"Video source. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=DEFAULT_CAMERA_ID,
        help=f"OpenCV webcam id when --source webcam. Default: {DEFAULT_CAMERA_ID}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Requested frame width. Default: {DEFAULT_WIDTH}",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Requested frame height. Default: {DEFAULT_HEIGHT}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Requested frame rate. Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality 0-100. Default: {DEFAULT_JPEG_QUALITY}",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally before sending.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a local preview window. Press q to stop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_opencv()

    publisher = ZmqFramePublisher(
        server_host=args.server_host,
        port=args.port,
        topic=args.topic,
    )
    log.info(
        "Video source=%s endpoint=%s topic=%s size=%dx%d fps=%d jpeg_quality=%d",
        args.source,
        publisher.endpoint,
        args.topic,
        args.width,
        args.height,
        args.fps,
        args.jpeg_quality,
    )

    try:
        if args.source == "realsense":
            return stream_realsense(args, publisher)
        return stream_webcam(args, publisher)
    except KeyboardInterrupt:
        log.info("Stopping video client...")
        return 0
    finally:
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
