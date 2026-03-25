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

from control_config import GESTURE_MEDIAN_WINDOW, P0_MOVING_AVG_WINDOW
from control_overlay import draw_gesture_labels, draw_hand_skeleton


def run_realtime_gesture_detection(model_path, pose_model_path, camera_id, show_depth):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gesture model not found: {model_path}")
    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(
            f"Pose model not found: {pose_model_path}. "
            "Provide a MediaPipe PoseLandmarker .task file via --pose-model."
        )

    gesture_base_options = python.BaseOptions(model_asset_path=model_path)
    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    gesture_options = vision.GestureRecognizerOptions(
        base_options=gesture_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    recognizer = vision.GestureRecognizer.create_from_options(gesture_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    last_timestamp_ms = 0
    hand_slots = ("left", "right")
    gesture_histories = {
        hand_slot: deque(maxlen=GESTURE_MEDIAN_WINDOW) for hand_slot in hand_slots
    }
    p0_histories = {
        hand_slot: deque(maxlen=P0_MOVING_AVG_WINDOW) for hand_slot in hand_slots
    }
    control_start_points = {hand_slot: None for hand_slot in hand_slots}
    yaw_states = {
        hand_slot: {
            "forearm_vec": None,
            "palm_vec": None,
            "yaw_deg": None,
            "invalid_count": 0,
            "mode": "hold",
        }
        for hand_slot in hand_slots
    }
    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")

        for _ in range(10):
            try:
                pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                if "Frame didn't arrive within 5000" not in str(exc):
                    raise
                print("Warning: timed out while warming up RealSense frames")

        while True:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                if "Frame didn't arrive within 5000" not in str(exc):
                    raise
                print("Warning: RealSense frame timeout, retrying")
                continue

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

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
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_hand_skeleton(frame_bgr, result.hand_landmarks)
                draw_gesture_labels(
                    frame_bgr,
                    result,
                    pose_result,
                    depth_image,
                    depth_scale,
                    intrinsics,
                    gesture_histories,
                    p0_histories,
                    control_start_points,
                    yaw_states,
                )
            else:
                for history in gesture_histories.values():
                    history.clear()
                for history in p0_histories.values():
                    history.clear()
                for hand_slot in control_start_points:
                    control_start_points[hand_slot] = None
                    yaw_states[hand_slot] = {
                        "forearm_vec": None,
                        "palm_vec": None,
                        "yaw_deg": None,
                        "invalid_count": 0,
                        "mode": "hold",
                    }

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
        pose_landmarker.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detector")
    parser.add_argument(
        "--model",
        type=str,
        default="gesture_recognizer.task",
        help="Path to MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="pose_landmarker.task",
        help="Path to MediaPipe pose landmarker task model",
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
    run_realtime_gesture_detection(args.model, args.pose_model, args.camera_id, args.show_depth)
