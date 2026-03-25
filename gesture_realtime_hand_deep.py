import argparse
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
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
DEPTH_MEDIAN_WINDOW_RADIUS = 2
VECTOR_EMA_ALPHA = 0.55
ANGLE_EMA_BETA = 0.6
YAW_SUSPECT_DELTA_PER_FRAME_DEG = 20.0
MAX_YAW_DELTA_PER_FRAME_DEG = 30.0
INVALID_HOLD_FRAMES = 2
PALM_GESTURE_NAMES = {"palm", "open_palm"}
ROCK_GESTURE_NAMES = {"rock", "closed_fist"}
NONE_GESTURE_NAMES = {"none", "unknown"}


def get_depth_at_pixel(depth_image, depth_scale, px, py):
    """Return distance in meters using median depth around a pixel."""
    height, width = depth_image.shape
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    x0 = max(0, px - DEPTH_MEDIAN_WINDOW_RADIUS)
    x1 = min(width, px + DEPTH_MEDIAN_WINDOW_RADIUS + 1)
    y0 = max(0, py - DEPTH_MEDIAN_WINDOW_RADIUS)
    y1 = min(height, py + DEPTH_MEDIAN_WINDOW_RADIUS + 1)
    depth_patch = depth_image[y0:y1, x0:x1]
    valid_depths = depth_patch[depth_patch > 0]
    if valid_depths.size == 0:
        return None
    depth_raw = float(np.median(valid_depths))
    return depth_raw * depth_scale


def get_3d_point_at_pixel(depth_image, depth_scale, intrinsics, px, py):
    """Return a 3D point in camera coordinates for a pixel, or None."""
    depth_m = get_depth_at_pixel(depth_image, depth_scale, px, py)
    if depth_m is None:
        return None
    original_px = intrinsics.width - 1 - px
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(original_px), float(py)], depth_m)
    return (point[0], point[1], point[2])


def get_3d_point_at_landmark(depth_image, depth_scale, intrinsics, landmark, width, height):
    px = int(landmark.x * width)
    py = int(landmark.y * height)
    return get_3d_point_at_pixel(depth_image, depth_scale, intrinsics, px, py)


def get_mediapipe_point(landmark):
    """Return landmark coordinates directly in MediaPipe normalized space."""
    return (float(landmark.x), float(landmark.y), float(landmark.z))


def clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))


def normalize_gesture_name(gesture_name):
    if not gesture_name:
        return "unknown"
    return gesture_name.strip().lower()


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(v):
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v, eps=1e-6):
    norm = vec_norm(v)
    if norm < eps:
        return None
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def project_to_horizontal(v):
    """Project a 3D vector onto the x-z plane and normalize it."""
    return vec_normalize((v[0], 0.0, v[2]))


def project_onto_plane(v, plane_normal):
    """Project vector onto the plane perpendicular to plane_normal."""
    normal = vec_normalize(plane_normal)
    if normal is None:
        return None
    projection = (
        v[0] - vec_dot(v, normal) * normal[0],
        v[1] - vec_dot(v, normal) * normal[1],
        v[2] - vec_dot(v, normal) * normal[2],
    )
    return vec_normalize(projection)


def signed_angle_on_xz_plane(v_ref, v_now):
    """Return signed azimuth delta in radians on the horizontal plane."""
    ref_xz = project_to_horizontal(v_ref)
    now_xz = project_to_horizontal(v_now)
    if ref_xz is None or now_xz is None:
        return None

    cross_y = ref_xz[2] * now_xz[0] - ref_xz[0] * now_xz[2]
    dot_xz = ref_xz[0] * now_xz[0] + ref_xz[2] * now_xz[2]
    return math.atan2(cross_y, dot_xz)


def blend_direction_vectors(primary_vec, secondary_vec, primary_weight=0.7):
    """Blend two normalized 3D directions into a single normalized vector."""
    if primary_vec is None:
        return secondary_vec
    if secondary_vec is None:
        return primary_vec

    secondary_weight = 1.0 - primary_weight
    blended = (
        primary_weight * primary_vec[0] + secondary_weight * secondary_vec[0],
        primary_weight * primary_vec[1] + secondary_weight * secondary_vec[1],
        primary_weight * primary_vec[2] + secondary_weight * secondary_vec[2],
    )
    return vec_normalize(blended)


def orient_like_reference(vector, reference_vector):
    """Flip vector sign to match the reference direction."""
    if vector is None or reference_vector is None:
        return vector
    if vec_dot(vector, reference_vector) < 0:
        return (-vector[0], -vector[1], -vector[2])
    return vector


def ema_vector(previous_vector, current_vector, alpha):
    """Smooth a direction vector with EMA and renormalize."""
    if current_vector is None:
        return previous_vector
    if previous_vector is None:
        return current_vector

    current_vector = orient_like_reference(current_vector, previous_vector)
    blended = (
        alpha * previous_vector[0] + (1.0 - alpha) * current_vector[0],
        alpha * previous_vector[1] + (1.0 - alpha) * current_vector[1],
        alpha * previous_vector[2] + (1.0 - alpha) * current_vector[2],
    )
    return vec_normalize(blended)


def clamp_angle_step(previous_angle_deg, current_angle_deg, max_step_deg):
    if previous_angle_deg is None or current_angle_deg is None:
        return current_angle_deg
    delta = current_angle_deg - previous_angle_deg
    delta = max(-max_step_deg, min(max_step_deg, delta))
    return previous_angle_deg + delta


def ema_angle(previous_angle_deg, current_angle_deg, beta):
    if current_angle_deg is None:
        return previous_angle_deg
    if previous_angle_deg is None:
        return current_angle_deg
    return beta * previous_angle_deg + (1.0 - beta) * current_angle_deg


def is_suspect_angle_jump(previous_angle_deg, current_angle_deg, suspect_delta_deg):
    if previous_angle_deg is None or current_angle_deg is None:
        return False
    return abs(current_angle_deg - previous_angle_deg) > suspect_delta_deg


def compute_azimuth_deg(direction_xyz, reference_direction_xyz=None):
    """Estimate azimuth on the horizontal plane for a 3D direction vector."""
    direction_vec = vec_normalize(direction_xyz)
    if direction_vec is None:
        return None

    if reference_direction_xyz is not None:
        delta_rad = signed_angle_on_xz_plane(reference_direction_xyz, direction_vec)
        if delta_rad is None:
            return None
        return math.degrees(delta_rad)

    fused_xz = project_to_horizontal(direction_vec)
    if fused_xz is None:
        return None
    return math.degrees(math.atan2(fused_xz[0], -fused_xz[2]))


def signed_angle_2d(v_ref, v_now):
    """Return signed angle between 2D vectors in radians."""
    ref_norm = math.hypot(v_ref[0], v_ref[1])
    now_norm = math.hypot(v_now[0], v_now[1])
    if ref_norm < 1e-6 or now_norm < 1e-6:
        return None

    ref_unit = (v_ref[0] / ref_norm, v_ref[1] / ref_norm)
    now_unit = (v_now[0] / now_norm, v_now[1] / now_norm)
    cross = ref_unit[0] * now_unit[1] - ref_unit[1] * now_unit[0]
    dot = clamp(ref_unit[0] * now_unit[0] + ref_unit[1] * now_unit[1], -1.0, 1.0)
    return math.atan2(cross, dot)


def is_yaw_control_active(gesture_name, yaw_direction_vector):
    """Enable yaw control for open-palm style gestures when geometry is valid."""
    if yaw_direction_vector is None:
        return False
    return normalize_gesture_name(gesture_name) in PALM_GESTURE_NAMES


def get_pose_arm_landmarks(pose_result, hand_wrist_landmark):
    """Return shoulder, elbow, wrist pose landmarks for the nearest pose arm."""
    if not pose_result or not pose_result.pose_landmarks:
        return None
    if not pose_result.pose_landmarks[0]:
        return None

    pose_landmarks = pose_result.pose_landmarks[0]
    arm_candidates = ((11, 13, 15), (12, 14, 16))
    best_arm = None
    best_distance = None
    for shoulder_idx, elbow_idx, wrist_idx in arm_candidates:
        wrist_lm = pose_landmarks[wrist_idx]
        elbow_lm = pose_landmarks[elbow_idx]
        visibility = min(
            getattr(wrist_lm, "visibility", 1.0),
            getattr(elbow_lm, "visibility", 1.0),
        )
        if visibility < 0.2:
            continue

        dx = wrist_lm.x - hand_wrist_landmark.x
        dy = wrist_lm.y - hand_wrist_landmark.y
        distance = dx * dx + dy * dy
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_arm = (
                pose_landmarks[shoulder_idx],
                pose_landmarks[elbow_idx],
                pose_landmarks[wrist_idx],
            )
    return best_arm


def get_palm_forward_vector_3d(hand_landmarks, depth_image, depth_scale, intrinsics, width, height):
    """Estimate palm-forward direction using wrist -> MCP-center axis in MediaPipe space."""
    if len(hand_landmarks) <= 17:
        return None

    p0 = get_mediapipe_point(hand_landmarks[0])

    mcp_indices = (5, 9, 13, 17)
    mcp_points = [get_mediapipe_point(hand_landmarks[idx]) for idx in mcp_indices]

    palm_center = (
        sum(point[0] for point in mcp_points) / len(mcp_points),
        sum(point[1] for point in mcp_points) / len(mcp_points),
        sum(point[2] for point in mcp_points) / len(mcp_points),
    )
    return vec_normalize(vec_sub(palm_center, p0))


def get_forearm_and_palm_forward_vectors(
    hand_landmarks,
    pose_result,
    depth_image,
    depth_scale,
    intrinsics,
    width,
    height,
):
    """Return forearm axis and palm-forward vector in MediaPipe normalized space."""
    pose_arm = get_pose_arm_landmarks(pose_result, hand_landmarks[0])
    if pose_arm is None:
        return None, None, None, None, None

    _, elbow_lm, wrist_lm = pose_arm
    elbow_xyz = get_mediapipe_point(elbow_lm)
    wrist_xyz = get_mediapipe_point(wrist_lm)

    forearm_vec = vec_normalize(vec_sub(wrist_xyz, elbow_xyz))
    palm_forward_vec = get_palm_forward_vector_3d(
        hand_landmarks, depth_image, depth_scale, intrinsics, width, height
    )
    if forearm_vec is None or palm_forward_vec is None:
        return forearm_vec, None, elbow_xyz, wrist_xyz, pose_arm

    if vec_dot(palm_forward_vec, forearm_vec) < 0:
        palm_forward_vec = (-palm_forward_vec[0], -palm_forward_vec[1], -palm_forward_vec[2])
    return forearm_vec, palm_forward_vec, elbow_xyz, wrist_xyz, pose_arm


def get_forearm_and_palm_forward_vectors_2d(hand_landmarks, pose_result):
    """Return 2D forearm and palm-forward vectors in image coordinates."""
    pose_arm = get_pose_arm_landmarks(pose_result, hand_landmarks[0])
    if pose_arm is None:
        return None, None

    _, elbow_lm, wrist_lm = pose_arm
    forearm_vec_2d = (wrist_lm.x - elbow_lm.x, wrist_lm.y - elbow_lm.y)

    mcp_indices = (5, 9, 13, 17)
    mcp_points = [(hand_landmarks[idx].x, hand_landmarks[idx].y) for idx in mcp_indices]
    palm_center_2d = (
        sum(point[0] for point in mcp_points) / len(mcp_points),
        sum(point[1] for point in mcp_points) / len(mcp_points),
    )
    palm_forward_vec_2d = (
        palm_center_2d[0] - hand_landmarks[0].x,
        palm_center_2d[1] - hand_landmarks[0].y,
    )
    return forearm_vec_2d, palm_forward_vec_2d


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


def draw_gesture_labels(
    image_bgr,
    recognition_result,
    pose_result,
    depth_image,
    depth_scale,
    intrinsics,
    p0_histories,
    control_start_points,
    yaw_states,
):
    """Draw gesture labels near each detected hand."""
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
            control_vector = get_control_vector(
                (avg_x0, avg_y0, avg_z0), (start_x, start_y, start_z)
            )

            directions = get_movement_directions(
                (avg_x0, avg_y0, avg_z0), (start_x, start_y, start_z)
            )
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

        forearm_axis_vector, palm_forward_vector, elbow_xyz, wrist_xyz, pose_arm = get_forearm_and_palm_forward_vectors(
            hand_landmarks,
            pose_result,
            depth_image,
            depth_scale,
            intrinsics,
            w,
            h,
        )
        forearm_status = "missing"
        if pose_arm is not None:
            _, elbow_lm, wrist_lm = pose_arm
            elbow_px = int(elbow_lm.x * w)
            elbow_py = int(elbow_lm.y * h)
            wrist_px = int(wrist_lm.x * w)
            wrist_py = int(wrist_lm.y * h)
            cv2.line(
                image_bgr,
                (elbow_px, elbow_py),
                (wrist_px, wrist_py),
                (255, 120, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.circle(image_bgr, (elbow_px, elbow_py), 5, (255, 120, 0), -1)
            cv2.circle(image_bgr, (wrist_px, wrist_py), 5, (255, 120, 0), -1)
            forearm_status = "ok"
        cv2.putText(
            image_bgr,
            f"forearm: {forearm_status}",
            (text_x, info_y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )
        palm_forward_status = "ok" if palm_forward_vector is not None else "missing"
        cv2.putText(
            image_bgr,
            f"palm forward: {palm_forward_status}",
            (text_x, info_y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )
        if i >= len(yaw_states):
            yaw_states.append(
                {
                    "forearm_vec": None,
                    "palm_vec": None,
                    "yaw_deg": None,
                    "invalid_count": 0,
                    "mode": "hold",
                }
            )

        yaw_state = yaw_states[i]
        if forearm_axis_vector is not None and palm_forward_vector is not None:
            smoothed_forearm = ema_vector(yaw_state["forearm_vec"], forearm_axis_vector, VECTOR_EMA_ALPHA)
            smoothed_palm = ema_vector(yaw_state["palm_vec"], palm_forward_vector, VECTOR_EMA_ALPHA)
            yaw_state["forearm_vec"] = smoothed_forearm
            yaw_state["palm_vec"] = smoothed_palm
            raw_yaw_delta_deg_3d = None
            if smoothed_forearm is not None and smoothed_palm is not None:
                delta_rad = signed_angle_on_xz_plane(smoothed_forearm, smoothed_palm)
                raw_yaw_delta_deg_3d = None if delta_rad is None else math.degrees(delta_rad)

            forearm_vec_2d, palm_vec_2d = get_forearm_and_palm_forward_vectors_2d(hand_landmarks, pose_result)
            raw_yaw_delta_deg_2d = None
            if forearm_vec_2d is not None and palm_vec_2d is not None:
                delta_rad_2d = signed_angle_2d(forearm_vec_2d, palm_vec_2d)
                raw_yaw_delta_deg_2d = None if delta_rad_2d is None else math.degrees(delta_rad_2d)

            raw_yaw_delta_deg = raw_yaw_delta_deg_3d
            yaw_state["mode"] = "3d"
            if is_suspect_angle_jump(yaw_state["yaw_deg"], raw_yaw_delta_deg_3d, YAW_SUSPECT_DELTA_PER_FRAME_DEG):
                raw_yaw_delta_deg = raw_yaw_delta_deg_2d
                yaw_state["mode"] = "2d"

            clamped_yaw_delta_deg = clamp_angle_step(
                yaw_state["yaw_deg"], raw_yaw_delta_deg, MAX_YAW_DELTA_PER_FRAME_DEG
            )
            yaw_state["yaw_deg"] = ema_angle(yaw_state["yaw_deg"], clamped_yaw_delta_deg, ANGLE_EMA_BETA)
            yaw_state["invalid_count"] = 0
        else:
            yaw_state["invalid_count"] += 1
            yaw_state["mode"] = "hold"
            if yaw_state["invalid_count"] > INVALID_HOLD_FRAMES:
                yaw_state["forearm_vec"] = None
                yaw_state["palm_vec"] = None
                yaw_state["yaw_deg"] = None

        yaw_delta_deg = yaw_state["yaw_deg"]
        if yaw_delta_deg is not None:
            cv2.putText(
                image_bgr,
                f"yaw delta: {yaw_delta_deg:+.1f} deg",
                (text_x, info_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                f"yaw mode: {yaw_state['mode']}",
                (text_x, info_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )

    for i in range(len(recognition_result.hand_landmarks), len(control_start_points)):
        control_start_points[i] = None
    for i in range(len(recognition_result.hand_landmarks), len(yaw_states)):
        yaw_states[i]["invalid_count"] += 1
        if yaw_states[i]["invalid_count"] > INVALID_HOLD_FRAMES:
            yaw_states[i]["forearm_vec"] = None
            yaw_states[i]["palm_vec"] = None
            yaw_states[i]["yaw_deg"] = None


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
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.99,
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
    p0_histories = [deque(maxlen=P0_MOVING_AVG_WINDOW) for _ in range(2)]
    control_start_points = [None for _ in range(2)]
    yaw_states = []
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
                    p0_histories,
                    control_start_points,
                    yaw_states,
                )

            cv2.imshow("Real-time Gesture Detection", frame_bgr)
            if show_depth:
                cv2.imshow("Depth", frame_depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except Exception as e:
        print(f"Error: {e}")
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
        "--camera-id",
        type=int,
        default=0,
        help="Camera device id (default: 0)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="pose_landmarker.task",
        help="Path to MediaPipe pose landmarker task model",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show the depth colormap window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_gesture_detection(args.model, args.pose_model, args.camera_id, args.show_depth)
