import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from build_config import P0_MOVING_AVG_WINDOW
from drawing import draw_gesture_labels, draw_hand_skeleton


def run_realtime_gesture_detection(model_path, camera_id, show_depth):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gesture model not found: {model_path}")

    gesture_base_options = python.BaseOptions(model_asset_path=model_path)
    gesture_options = vision.GestureRecognizerOptions(
        base_options=gesture_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    last_timestamp_ms = 0
    p0_histories = [deque(maxlen=P0_MOVING_AVG_WINDOW) for _ in range(2)]
    control_start_points = [None for _ in range(2)]
    yaw_reference_vectors = [None for _ in range(2)]
    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")

        for _ in range(10):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            frame_bgr = cv2.flip(frame_bgr, 1)
            depth_image = cv2.flip(depth_image, 1)
            frame_depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_hand_skeleton(frame_bgr, result.hand_landmarks)
                draw_gesture_labels(
                    frame_bgr,
                    result,
                    depth_image,
                    depth_scale,
                    intrinsics,
                    p0_histories,
                    control_start_points,
                    yaw_reference_vectors,
                )

            cv2.imshow("Real-time Gesture Detection", frame_bgr)
            if show_depth:
                cv2.imshow("Depth", frame_depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        recognizer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detector")
    parser.add_argument(
        "--model",
        type=str,
        default="gesture_recognizer.task",
        help="Path to MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device id (default: 0)",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show the depth colormap window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_realtime_gesture_detection(args.model, args.camera_id, args.show_depth)
