import argparse
import math
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
P0_MOVING_AVG_WINDOW = 10
XY_DEAD_ZONE = 0.03
Z_DEAD_ZONE = 0.02
XY_MAX_DELTA = 0.30
Z_MAX_DELTA = 0.12
TURN_DEAD_ZONE_DEG = 10


def clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))


def normalize_axis(delta, dead_zone, max_delta):
    """Map displacement to [-1, 1] with a central dead zone."""
    if max_delta <= dead_zone:
        raise ValueError("max_delta must be greater than dead_zone")

    if abs(delta) <= dead_zone:
        return 0.0

    if delta > 0:
        normalized = (delta - dead_zone) / (max_delta - dead_zone)
    else:
        normalized = (delta + dead_zone) / (max_delta - dead_zone)
    return clamp(normalized)


def get_control_vector(current_xyz, origin_xyz):
    """Return normalized control vector relative to the origin point."""
    cur_x, cur_y, cur_z = current_xyz
    org_x, org_y, org_z = origin_xyz
    return (
        normalize_axis(cur_x - org_x, XY_DEAD_ZONE, XY_MAX_DELTA),
        normalize_axis(cur_y - org_y, XY_DEAD_ZONE, XY_MAX_DELTA),
        normalize_axis(cur_z - org_z, Z_DEAD_ZONE, Z_MAX_DELTA),
    )


def get_movement_directions(current_xyz, origin_xyz):
    """Return movement directions relative to the origin point."""
    cur_x, cur_y, cur_z = current_xyz
    org_x, org_y, org_z = origin_xyz
    directions = []

    if cur_y < org_y - XY_DEAD_ZONE:
        directions.append("up")
    elif cur_y > org_y + XY_DEAD_ZONE:
        directions.append("down")

    if cur_x < org_x - XY_DEAD_ZONE:
        directions.append("left")
    elif cur_x > org_x + XY_DEAD_ZONE:
        directions.append("right")

    if cur_z < org_z - Z_DEAD_ZONE:
        directions.append("forward")
    elif cur_z > org_z + Z_DEAD_ZONE:
        directions.append("backward")

    return directions


def draw_hand_skeleton(image_bgr, hand_landmarks_list):
    """Draw hand skeleton for all detected hands."""
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


def draw_gesture_labels(image_bgr, recognition_result, p0_histories, control_start_points):
    """Draw top gesture label near each detected hand."""
    h, w, _ = image_bgr.shape

    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        if i >= len(p0_histories):
            p0_histories.append(deque(maxlen=P0_MOVING_AVG_WINDOW))
        if i >= len(control_start_points):
            control_start_points.append(None)

        x0 = hand_landmarks[0].x
        y0 = hand_landmarks[0].y
        z0_raw = hand_landmarks[0].z
        p0_histories[i].append((x0, y0, z0_raw))
        avg_x0 = sum(v[0] for v in p0_histories[i]) / len(p0_histories[i])
        avg_y0 = sum(v[1] for v in p0_histories[i]) / len(p0_histories[i])
        avg_z0_raw = sum(v[2] for v in p0_histories[i]) / len(p0_histories[i])
        avg_z0 = abs(avg_z0_raw)

        gesture_name = "Unknown"
        gesture_score = 0.0
        if i < len(recognition_result.gestures) and recognition_result.gestures[i]:
            top_gesture = recognition_result.gestures[i][0]
            gesture_name = top_gesture.category_name
            gesture_score = top_gesture.score

        control_detected = gesture_name == "Closed_Fist"
        if control_detected:
            if control_start_points[i] is None:
                control_start_points[i] = (avg_x0, avg_y0, avg_z0_raw)
        else:
            control_start_points[i] = None

        handedness = ""
        if i < len(recognition_result.handedness) and recognition_result.handedness[i]:
            handedness = recognition_result.handedness[i][0].category_name

        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        # print(f"Gesture: {gesture_name}, {hand_landmarks}")
        text_x = int(min(xs) * w)
        text_y = int(min(ys) * h) - 10
        text_y = max(text_y, 25)

        label = f"{gesture_name} ({gesture_score:.2f})"
        if handedness:
            label = f"{label} [{handedness}]"

        cv2.putText(
            image_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            f"p0(avg10) x: {avg_x0:.3f}",
            (text_x, text_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            f"p0(avg10) y: {avg_y0:.3f}",
            (text_x, text_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            f"p0(avg10) z0: {avg_z0:.3f}",
            (text_x, text_y + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            "Control gesture: Closed_Fist",
            (text_x, text_y + 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

        if control_start_points[i] is not None:
            start_x, start_y, start_z = control_start_points[i]
            start_px = int(start_x * w)
            start_py = int(start_y * h)
            cv2.circle(image_bgr, (start_px, start_py), 6, (0, 0, 255), -1)
            control_vector = get_control_vector(
                (avg_x0, avg_y0, avg_z0_raw), (start_x, start_y, start_z)
            )

            directions = get_movement_directions(
                (avg_x0, avg_y0, avg_z0_raw), (start_x, start_y, start_z)
            )
            if directions:
                cv2.putText(
                    image_bgr,
                    " / ".join(directions),
                    (text_x, text_y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image_bgr,
                "ctrl xyz: "
                f"{control_vector[0]:+.2f}, {control_vector[1]:+.2f}, {control_vector[2]:+.2f}",
                (text_x, text_y + 156),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 80, 255),
                2,
                cv2.LINE_AA,
            )

        if gesture_name == "Open_Palm" and len(hand_landmarks) > 12:
            x12 = hand_landmarks[12].x
            y12 = hand_landmarks[12].y

            p0 = (int(x0 * w), int(y0 * h))
            p0_top = (p0[0], 0)
            p12 = (int(x12 * w), int(y12 * h))
            up_len = max(p0[1], 1)
            zone_dx = int(up_len * math.tan(math.radians(TURN_DEAD_ZONE_DEG)))
            left_zone_top = (max(p0[0] - zone_dx, 0), 0)
            right_zone_top = (min(p0[0] + zone_dx, w - 1), 0)

            cv2.line(image_bgr, p0, p0_top, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image_bgr, p0, left_zone_top, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(image_bgr, p0, right_zone_top, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(image_bgr, p0, p12, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(image_bgr, p12, 5, (255, 255, 0), -1)

            dx = x12 - x0
            dy = y12 - y0
            angle_from_center_deg = math.degrees(math.atan2(dx, -dy))
            if abs(angle_from_center_deg) <= TURN_DEAD_ZONE_DEG:
                turn_text = "No Turn"
            elif dx < 0:
                turn_text = "Turn Right"
            else:
                turn_text = "Turn Left"
            cv2.putText(
                image_bgr,
                turn_text,
                (text_x, text_y + 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    for i in range(len(recognition_result.hand_landmarks), len(control_start_points)):
        control_start_points[i] = None


def run_realtime_gesture_detection(model_path, camera_id):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with id={camera_id}")

    last_timestamp_ms = 0
    p0_histories = [deque(maxlen=P0_MOVING_AVG_WINDOW) for _ in range(2)]
    control_start_points = [None for _ in range(2)]
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

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
                draw_gesture_labels(frame_bgr, result, p0_histories, control_start_points)

            cv2.imshow("Real-time Gesture Detection", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_gesture_detection(args.model, args.camera_id)
