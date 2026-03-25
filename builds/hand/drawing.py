from collections import deque

import cv2

from build_config import HAND_CONNECTIONS, P0_MOVING_AVG_WINDOW, ROCK_GESTURE_NAMES
from depth_utils import get_3d_point_at_landmark, get_depth_at_pixel
from gesture_math import (
    blend_direction_vectors,
    compute_palm_azimuth_deg,
    get_control_vector,
    get_movement_directions,
    is_yaw_control_active,
    normalize_gesture_name,
    vec_normalize,
    vec_sub,
)


def get_palm_forward_vector_3d(hand_landmarks, depth_image, depth_scale, intrinsics, width, height):
    if len(hand_landmarks) <= 12:
        return None

    p0 = get_3d_point_at_landmark(depth_image, depth_scale, intrinsics, hand_landmarks[0], width, height)
    p9 = get_3d_point_at_landmark(depth_image, depth_scale, intrinsics, hand_landmarks[9], width, height)
    p12 = get_3d_point_at_landmark(depth_image, depth_scale, intrinsics, hand_landmarks[12], width, height)
    if p0 is None or p9 is None or p12 is None:
        return None

    wrist_to_base = vec_sub(p9, p0)
    wrist_to_tip = vec_sub(p12, p0)
    return blend_direction_vectors(
        vec_normalize(wrist_to_tip),
        vec_normalize(wrist_to_base),
        primary_weight=0.7,
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
            cv2.line(image_bgr, points[start_idx], points[end_idx], (0, 200, 0), 2, cv2.LINE_AA)


def draw_gesture_labels(
    image_bgr,
    recognition_result,
    depth_image,
    depth_scale,
    intrinsics,
    p0_histories,
    control_start_points,
    yaw_reference_vectors,
):
    h, w, _ = image_bgr.shape

    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        if i >= len(p0_histories):
            p0_histories.append(deque(maxlen=P0_MOVING_AVG_WINDOW))
        if i >= len(control_start_points):
            control_start_points.append(None)

        x0 = hand_landmarks[0].x
        y0 = hand_landmarks[0].y
        px0 = int(x0 * w)
        py0 = int(y0 * h)
        z0_m = get_depth_at_pixel(depth_image, depth_scale, px0, py0)
        if z0_m is None:
            continue

        p0_histories[i].append((x0, y0, z0_m))
        avg_x0 = sum(v[0] for v in p0_histories[i]) / len(p0_histories[i])
        avg_y0 = sum(v[1] for v in p0_histories[i]) / len(p0_histories[i])
        avg_z0 = sum(v[2] for v in p0_histories[i]) / len(p0_histories[i])

        gesture_name = "Unknown"
        gesture_score = 0.0
        gesture_lines = ["Unknown"]
        if i < len(recognition_result.gestures) and recognition_result.gestures[i]:
            top_gesture = recognition_result.gestures[i][0]
            gesture_lines = [
                f"{gesture.category_name}: {gesture.score:.2f}"
                for gesture in recognition_result.gestures[i]
            ]
            gesture_name = top_gesture.category_name
            gesture_score = top_gesture.score

        normalized_gesture_name = normalize_gesture_name(gesture_name)
        control_detected = normalized_gesture_name in ROCK_GESTURE_NAMES
        if control_detected:
            if control_start_points[i] is None:
                control_start_points[i] = (avg_x0, avg_y0, avg_z0)
        else:
            control_start_points[i] = None

        handedness = ""
        if i < len(recognition_result.handedness) and recognition_result.handedness[i]:
            handedness = recognition_result.handedness[i][0].category_name

        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]
        text_x = int(min(xs) * w)
        text_y = max(int(min(ys) * h) - 10, 25)

        label = f"{gesture_name} ({gesture_score:.2f})"
        if handedness:
            label = f"{label} [{handedness}]"

        cv2.putText(image_bgr, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        for line_index, gesture_line in enumerate(gesture_lines, start=1):
            cv2.putText(
                image_bgr,
                gesture_line,
                (text_x, text_y + 26 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (120, 255, 120),
                2,
                cv2.LINE_AA,
            )
            print(f"  {gesture_line}")

        info_y = text_y + 26 * (len(gesture_lines) + 1)
        cv2.putText(
            image_bgr,
            f"p0(avg10) xyz: {avg_x0:.2f}, {avg_y0:.2f}, {avg_z0:.2f} m",
            (text_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if control_start_points[i] is not None:
            start_x, start_y, start_z = control_start_points[i]
            start_px = int(start_x * w)
            start_py = int(start_y * h)
            cv2.circle(image_bgr, (start_px, start_py), 6, (0, 0, 255), -1)
            control_vector = get_control_vector((avg_x0, avg_y0, avg_z0), (start_x, start_y, start_z))
            directions = get_movement_directions((avg_x0, avg_y0, avg_z0), (start_x, start_y, start_z))
            if directions:
                cv2.putText(
                    image_bgr,
                    " / ".join(directions),
                    (text_x, info_y + 26),
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
                (text_x, info_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 80, 255),
                2,
                cv2.LINE_AA,
            )

        palm_forward_vector = get_palm_forward_vector_3d(hand_landmarks, depth_image, depth_scale, intrinsics, w, h)
        palm_status = "ok" if palm_forward_vector is not None else "missing"
        cv2.putText(
            image_bgr,
            f"palm yaw: {palm_status}",
            (text_x, info_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )

        yaw_abs_deg = None
        if palm_forward_vector is not None:
            yaw_abs_deg = compute_palm_azimuth_deg(palm_forward_vector)
        if yaw_abs_deg is not None:
            cv2.putText(
                image_bgr,
                f"yaw abs: {yaw_abs_deg:+.1f} deg",
                (text_x, info_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )

        yaw_active = is_yaw_control_active(gesture_name, palm_forward_vector)
        if yaw_active and palm_forward_vector is not None:
            if yaw_reference_vectors[i] is None:
                yaw_reference_vectors[i] = palm_forward_vector
            cv2.putText(
                image_bgr,
                "yaw ref: locked",
                (text_x, info_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            yaw_reference_vectors[i] = None
            cv2.putText(
                image_bgr,
                "yaw ref: waiting",
                (text_x, info_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )

    for i in range(len(recognition_result.hand_landmarks), len(control_start_points)):
        control_start_points[i] = None
    for i in range(len(recognition_result.hand_landmarks), len(yaw_reference_vectors)):
        yaw_reference_vectors[i] = None
