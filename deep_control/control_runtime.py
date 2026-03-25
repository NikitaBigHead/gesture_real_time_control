import argparse
import os
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import rclpy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from control_commands import (
    collect_command_events,
    create_command_tracker_state,
    log_command_events,
    reset_command_tracker_state,
)
from control_overlay import draw_control_overlay, draw_hand_skeleton
from control_state import compute_frame_control_state, create_tracking_state, reset_tracking_state
from gesture_drone_node import GestureDroneController


def start_gesture_drone_controller(args):
    if not rclpy.ok():
        rclpy.init(args=None)
    controller = GestureDroneController(
        mavros_prefix=args.mavros_prefix,
        pose_topic=args.drone_pose_topic,
        move_duration_sec=args.drone_move_duration,
        move_speed_xy=args.drone_move_speed_xy,
        move_speed_z=args.drone_move_speed_z,
        yaw_kp=args.drone_yaw_kp,
        max_yaw_rate=args.drone_max_yaw_rate,
        yaw_tolerance_deg=args.drone_yaw_tolerance_deg,
        yaw_sign=args.drone_yaw_sign,
    )

    if args.drone_auto_takeoff:
        controller.set_mode("GUIDED")
        time.sleep(1.0)
        controller.arm(True)
        time.sleep(2.0)
        controller.takeoff(args.drone_takeoff_altitude)
        time.sleep(3.0)

    spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
    spin_thread.start()
    return controller, spin_thread


def stop_gesture_drone_controller(controller, spin_thread, auto_land):
    if controller is None:
        return

    try:
        controller.send_velocity_command(0.0, 0.0, 0.0, 0.0)
        if auto_land:
            controller.land(wait=False)
            time.sleep(1.0)
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


def run_realtime_gesture_detection(model_path, pose_model_path, camera_id, show_depth, args):
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
    tracking_state = create_tracking_state()
    command_state = create_command_tracker_state()
    drone_controller = None
    drone_spin_thread = None
    pipeline_started = False
    try:
        if not args.no_drone_control:
            drone_controller, drone_spin_thread = start_gesture_drone_controller(args)

        profile = pipeline.start(config)
        pipeline_started = True
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
                frame_state = compute_frame_control_state(
                    result,
                    pose_result,
                    depth_image,
                    depth_scale,
                    intrinsics,
                    frame_bgr.shape[1],
                    frame_bgr.shape[0],
                    tracking_state,
                )
                command_events = collect_command_events(frame_state, command_state, timestamp_ms)
                log_command_events(command_events)
                if drone_controller is not None and command_events:
                    drone_controller.handle_events(command_events)
                draw_hand_skeleton(frame_bgr, result.hand_landmarks)
                draw_control_overlay(frame_bgr, frame_state)
            else:
                reset_tracking_state(tracking_state)
                reset_command_tracker_state(command_state)

            cv2.imshow("Real-time Gesture Detection", frame_bgr)
            if show_depth:
                cv2.imshow("Depth", frame_depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        recognizer.close()
        pose_landmarker.close()
        stop_gesture_drone_controller(
            drone_controller,
            drone_spin_thread,
            auto_land=args.drone_auto_land,
        )
        if drone_controller is None and rclpy.ok():
            rclpy.shutdown()


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
        default="pose_landmarker_heavy.task",
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
    parser.add_argument(
        "--no-drone-control",
        action="store_true",
        help="Disable ROS drone control and only show gesture recognition windows",
    )
    parser.add_argument(
        "--mavros-prefix",
        type=str,
        default="/drone2/mavros",
        help="MAVROS namespace prefix",
    )
    parser.add_argument(
        "--drone-pose-topic",
        type=str,
        default="/drone2/mavros/vision_pose/pose",
        help="Pose topic used to estimate current drone yaw",
    )
    parser.add_argument(
        "--drone-move-duration",
        type=float,
        default=0.8,
        help="How long a fist_release translation command is applied",
    )
    parser.add_argument(
        "--drone-move-speed-xy",
        type=float,
        default=0.35,
        help="XY speed for fist_release translation commands",
    )
    parser.add_argument(
        "--drone-move-speed-z",
        type=float,
        default=0.25,
        help="Z speed for fist_release translation commands",
    )
    parser.add_argument(
        "--drone-yaw-kp",
        type=float,
        default=0.8,
        help="Proportional gain for palm_release yaw control",
    )
    parser.add_argument(
        "--drone-max-yaw-rate",
        type=float,
        default=0.8,
        help="Max angular velocity for palm_release yaw control",
    )
    parser.add_argument(
        "--drone-yaw-tolerance-deg",
        type=float,
        default=4.0,
        help="Yaw error tolerance in degrees before rotation stops",
    )
    parser.add_argument(
        "--drone-yaw-sign",
        type=float,
        default=1.0,
        help="Sign multiplier for palm yaw delta, useful if the hand-to-drone direction is inverted",
    )
    parser.add_argument(
        "--drone-auto-takeoff",
        action="store_true",
        help="Send GUIDED, arm and takeoff commands before starting gesture control",
    )
    parser.add_argument(
        "--drone-takeoff-altitude",
        type=float,
        default=0.7,
        help="Target takeoff altitude when --drone-auto-takeoff is used",
    )
    parser.add_argument(
        "--drone-auto-land",
        action="store_true",
        help="Send a land command on shutdown",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_realtime_gesture_detection(
        args.model,
        args.pose_model,
        args.camera_id,
        args.show_depth,
        args,
    )
