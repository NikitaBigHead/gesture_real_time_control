from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.processors.classifier_options import ClassifierOptions

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


THUMB_TIP_IDX = 4
DEFAULT_ZONE_WIDTH = 120
DEFAULT_MIN_GESTURE_SCORE = 0.5
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_INITIAL_MEAN_WINDOW = 5
DEFAULT_GESTURE_HOLD_SEC = 0.35
STATE_LEFT = "left"
STATE_RIGHT = "right"
STATE_NONE = "none"


def first_existing_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def default_model_path() -> str:
    root_dir = Path(__file__).resolve().parent
    return first_existing_path(
        root_dir / "gesture_recognizer.task",
        root_dir / "gesture_recognizer_1.0.task",
        root_dir / "deep_control" / "gesture_recognizer.task",
    )


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def mean_int(values: deque[int]) -> int:
    return int(round(sum(values) / len(values)))


def get_thumb_up_hand_index(
    result: vision.GestureRecognizerResult,
    min_score: float,
) -> Optional[int]:
    for hand_index, gestures in enumerate(result.gestures):
        if not gestures:
            continue
        top_gesture = gestures[0]
        if top_gesture.category_name == "Thumb_Up" and top_gesture.score >= min_score:
            return hand_index
    return None


def get_thumb_tip_x_px(
    result: vision.GestureRecognizerResult,
    hand_index: int,
    frame_width: int,
) -> int:
    thumb_tip = result.hand_landmarks[hand_index][THUMB_TIP_IDX]
    return int(thumb_tip.x * frame_width)


def compute_zone_bounds(center_x_px: int, zone_width_px: int, frame_width: int) -> tuple[int, int]:
    half_width = max(1, zone_width_px // 2)
    left = clamp(center_x_px - half_width, 0, frame_width - 1)
    right = clamp(center_x_px + half_width, 0, frame_width - 1)
    if right <= left:
        right = min(frame_width - 1, left + 1)
    return left, right


def resolve_state(current_x_px: int, zone_left_px: int, zone_right_px: int) -> str:
    if zone_left_px <= current_x_px <= zone_right_px:
        return STATE_NONE
    if current_x_px < zone_left_px:
        return STATE_LEFT
    return STATE_RIGHT


def draw_overlay(
    frame_bgr,
    zone_left_px: Optional[int],
    zone_right_px: Optional[int],
    thumb_x_px: Optional[int],
    state: str,
) -> None:
    frame_height, frame_width = frame_bgr.shape[:2]

    if zone_left_px is not None and zone_right_px is not None:
        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (zone_left_px, 0),
            (zone_right_px, frame_height - 1),
            (0, 120, 255),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0, frame_bgr)
        cv2.rectangle(
            frame_bgr,
            (zone_left_px, 0),
            (zone_right_px, frame_height - 1),
            (0, 180, 255),
            thickness=2,
        )

    if thumb_x_px is not None:
        cv2.line(
            frame_bgr,
            (thumb_x_px, 0),
            (thumb_x_px, frame_height - 1),
            (0, 255, 0),
            thickness=2,
        )

    cv2.putText(
        frame_bgr,
        f"state: {state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        "Gesture: Thumb_Up",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 255, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        "q / esc to quit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def print_state_if_changed(state: str, last_state: Optional[str]) -> str:
    if state != last_state:
        print(state, flush=True)
    return state


def run_gesture_scrolling(
    model_path: str,
    camera_id: int,
    zone_width_px: int,
    min_gesture_score: float,
    initial_mean_window: int,
    gesture_hold_sec: float,
) -> None:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed. Install the RealSense SDK Python package first.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=ClassifierOptions(
            score_threshold=min_gesture_score,
        ),
        custom_gesture_classifier_options=ClassifierOptions(
            score_threshold=min_gesture_score,
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
    last_state: Optional[str] = STATE_NONE
    last_timestamp_ms = 0
    pipeline_started = False
    initial_pose_samples: deque[int] = deque(maxlen=max(1, initial_mean_window))
    initial_pose_locked = False
    last_seen_monotonic: Optional[float] = None
    last_thumb_x_px: Optional[int] = None

    try:
        pipeline.start(config)
        pipeline_started = True

        for _ in range(10):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
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

            thumb_up_index = get_thumb_up_hand_index(result, min_gesture_score)
            if thumb_up_index is None:
                if (
                    last_seen_monotonic is not None
                    and now - last_seen_monotonic <= gesture_hold_sec
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
                        zone_width_px=zone_width_px,
                        frame_width=frame_width,
                    )
                    if len(initial_pose_samples) >= initial_pose_samples.maxlen:
                        initial_pose_locked = True
                elif zone_left_px is None or zone_right_px is None:
                    center_x_px = thumb_x_px
                    zone_left_px, zone_right_px = compute_zone_bounds(
                        center_x_px=center_x_px,
                        zone_width_px=zone_width_px,
                        frame_width=frame_width,
                    )

                state = resolve_state(thumb_x_px, zone_left_px, zone_right_px)

            last_state = print_state_if_changed(state, last_state)
            draw_overlay(frame_bgr, zone_left_px, zone_right_px, thumb_x_px, state)

            cv2.imshow("Gesture Scrolling", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        recognizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thumb-up gesture scrolling state detector")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model_path(),
        help="Path to the MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Unused for RealSense input; kept for compatibility",
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
        help="How long to keep the last state after Thumb_Up briefly disappears",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gesture_scrolling(
        model_path=args.model,
        camera_id=args.camera_id,
        zone_width_px=args.zone_width,
        min_gesture_score=args.min_score,
        initial_mean_window=args.initial_mean_window,
        gesture_hold_sec=args.gesture_hold_sec,
    )


if __name__ == "__main__":
    main()
