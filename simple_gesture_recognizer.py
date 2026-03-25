import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.processors.classifier_options import ClassifierOptions 

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw_hand_skeleton(image_bgr, hand_landmarks_list):
    h, w, _ = image_bgr.shape
    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(image_bgr, (x, y), 3, (0, 255, 255), -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                image_bgr,
                points[start_idx],
                points[end_idx],
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )


def draw_gesture_labels(image_bgr, recognition_result):
    h, w, _ = image_bgr.shape
    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        text_x = int(min(xs) * w)
        text_y = max(int(min(ys) * h) - 10, 25)

        handedness = ""
        if i < len(recognition_result.handedness) and recognition_result.handedness[i]:
            handedness = recognition_result.handedness[i][0].category_name

        gesture_lines = ["Unknown"]
        if i < len(recognition_result.gestures) and recognition_result.gestures[i]:
            gesture_lines = [
                f"{gesture.category_name}: {gesture.score:.2f}"
                for gesture in recognition_result.gestures[i]
            ]

        title = handedness if handedness else f"Hand {i + 1}"
        cv2.putText(
            image_bgr,
            title,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        for line_index, gesture_line in enumerate(gesture_lines, start=1):
            cv2.putText(
                image_bgr,
                gesture_line,
                (text_x, text_y + 26 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 255, 120),
                2,
                cv2.LINE_AA,
            )


def run_realtime_gesture_detection(model_path, camera_id):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,

        canned_gesture_classifier_options=ClassifierOptions(
        score_threshold=0.1,
    ),
    custom_gesture_classifier_options=ClassifierOptions(
        score_threshold=0.1,
    ),
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    last_timestamp_ms = 0
    try:
        pipeline.start(config)

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

            if result.hand_landmarks:
                draw_hand_skeleton(frame_bgr, result.hand_landmarks)
                draw_gesture_labels(frame_bgr, result)

            cv2.imshow("Simple Gesture Recognizer", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        recognizer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand gesture recognizer")
    parser.add_argument(
        "--model",
        default="gesture_recognizer.task",
        help="Path to MediaPipe gesture recognizer .task model",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Unused for RealSense input; kept for compatibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_gesture_detection(args.model, args.camera_id)

