import math
from collections import deque
from statistics import median

import cv2

from control_config import (
    ANGLE_EMA_BETA,
    GESTURE_MEDIAN_WINDOW,
    HAND_CONNECTIONS,
    INVALID_HOLD_FRAMES,
    MAX_YAW_DELTA_PER_FRAME_DEG,
    P0_MOVING_AVG_WINDOW,
    ROCK_GESTURE_NAMES,
    VECTOR_EMA_ALPHA,
)
from control_depth import get_depth_at_pixel
from control_geometry import get_forearm_and_palm_forward_vectors, get_pose_arm_match
from control_math import (
    clamp_angle_step,
    ema_angle,
    ema_vector,
    get_control_vector,
    get_movement_directions,
    normalize_gesture_name,
    signed_angle_on_xz_plane,
)

HAND_PRIORITY = ("right", "left")


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


def create_yaw_state():
    return {
        "forearm_vec": None,
        "palm_vec": None,
        "yaw_deg": None,
        "invalid_count": 0,
        "mode": "hold",
    }


def create_p0_history():
    return deque(maxlen=P0_MOVING_AVG_WINDOW)


def create_gesture_history():
    return deque(maxlen=GESTURE_MEDIAN_WINDOW)


def get_smoothed_gesture(history, default_name, default_score):
    if not history:
        return default_name, default_score

    gesture_names = {item["name"] for item in history}
    if not gesture_names:
        return default_name, default_score

    best_name = None
    best_key = None
    best_score = default_score
    for gesture_name in gesture_names:
        presence = [1 if item["name"] == gesture_name else 0 for item in history]
        scores = [item["score"] for item in history if item["name"] == gesture_name]
        last_seen_index = max(i for i, item in enumerate(history) if item["name"] == gesture_name)
        gesture_score = median(scores)
        sort_key = (
            median(presence),
            sum(presence),
            last_seen_index,
            gesture_score,
        )
        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_name = gesture_name
            best_score = gesture_score

    return best_name, best_score


def get_history_average_xyz(history):
    if not history:
        return None

    return (
        sum(v[0] for v in history) / len(history),
        sum(v[1] for v in history) / len(history),
        sum(v[2] for v in history) / len(history),
    )


def reset_hand_tracking_state(hand_slot, p0_histories, control_start_points, yaw_states):
    if hand_slot not in p0_histories:
        p0_histories[hand_slot] = create_p0_history()
    p0_histories[hand_slot].clear()
    control_start_points[hand_slot] = None
    yaw_states[hand_slot] = create_yaw_state()


def get_hand_metadata(recognition_result, pose_result, hand_index):
    gesture_name = "Unknown"
    gesture_score = 0.0
    gesture_lines = ["Unknown"]
    if hand_index < len(recognition_result.gestures) and recognition_result.gestures[hand_index]:
        top_gesture = recognition_result.gestures[hand_index][0]
        gesture_lines = [
            f"{gesture.category_name}: {gesture.score:.2f}"
            for gesture in recognition_result.gestures[hand_index]
        ]
        gesture_name = top_gesture.category_name
        gesture_score = top_gesture.score

    hand_landmarks = recognition_result.hand_landmarks[hand_index]
    hand_slot, pose_arm, pose_distance_sq = get_pose_arm_match(pose_result, hand_landmarks[0])
    handedness_label = hand_slot.capitalize() if hand_slot else ""

    return {
        "gesture_name": gesture_name,
        "gesture_score": gesture_score,
        "gesture_lines": gesture_lines,
        "handedness_label": handedness_label,
        "hand_slot": hand_slot,
        "pose_arm": pose_arm,
        "pose_distance_sq": pose_distance_sq,
    }


def select_active_hand(recognition_result, pose_result):
    best_indices_by_slot = {}
    best_distances_by_slot = {}

    for hand_index in range(len(recognition_result.hand_landmarks)):
        metadata = get_hand_metadata(recognition_result, pose_result, hand_index)
        hand_slot = metadata["hand_slot"]
        if hand_slot is None:
            continue

        pose_distance_sq = metadata["pose_distance_sq"]
        if (
            hand_slot not in best_indices_by_slot
            or pose_distance_sq < best_distances_by_slot[hand_slot]
        ):
            best_indices_by_slot[hand_slot] = hand_index
            best_distances_by_slot[hand_slot] = pose_distance_sq

    for hand_slot in HAND_PRIORITY:
        if hand_slot in best_indices_by_slot:
            return hand_slot, best_indices_by_slot[hand_slot]
    return None, None


def draw_gesture_labels(
    image_bgr,
    recognition_result,
    pose_result,
    depth_image,
    depth_scale,
    intrinsics,
    gesture_histories,
    p0_histories,
    control_start_points,
    yaw_states,
):
    """Draw all detected hands and compute control only for the active hand."""
    h, w, _ = image_bgr.shape

    active_hand_slot, active_hand_index = select_active_hand(recognition_result, pose_result)
    for hand_slot in HAND_PRIORITY:
        if hand_slot != active_hand_slot:
            reset_hand_tracking_state(
                hand_slot,
                p0_histories,
                control_start_points,
                yaw_states,
            )

    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        metadata = get_hand_metadata(recognition_result, pose_result, i)
        gesture_name = metadata["gesture_name"]
        gesture_score = metadata["gesture_score"]
        gesture_lines = metadata["gesture_lines"]
        handedness = metadata["handedness_label"]
        hand_slot = metadata["hand_slot"]
        is_active_hand = i == active_hand_index and hand_slot == active_hand_slot

        if hand_slot is not None:
            if hand_slot not in gesture_histories:
                gesture_histories[hand_slot] = create_gesture_history()
            gesture_histories[hand_slot].append(
                {
                    "name": gesture_name,
                    "score": gesture_score,
                }
            )
            gesture_name, gesture_score = get_smoothed_gesture(
                gesture_histories[hand_slot],
                gesture_name,
                gesture_score,
            )

        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]
        text_x = int(min(xs) * w)
        text_y = max(int(min(ys) * h) - 10, 25)

        label = f"{gesture_name} ({gesture_score:.2f})"
        if handedness:
            label = f"{label} [{handedness}]"
        if is_active_hand:
            label = f"{label} <ACTIVE>"

        cv2.putText(
            image_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255) if is_active_hand else (0, 255, 0),
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


        if not is_active_hand:
            cv2.putText(
                image_bgr,
                "tracked only",
                (text_x, text_y + 26 * (len(gesture_lines) + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            continue

        x0 = hand_landmarks[0].x
        y0 = hand_landmarks[0].y

        if hand_slot not in p0_histories:
            p0_histories[hand_slot] = create_p0_history()
        history = p0_histories[hand_slot]

        px0 = int(x0 * w)
        py0 = int(y0 * h)
        z0_m = get_depth_at_pixel(depth_image, depth_scale, px0, py0)
        if z0_m is None:
            history_avg = get_history_average_xyz(history)
            if history_avg is None:
                print(history_avg)
                continue
            z0_m = history_avg[2]

        history.append((x0, y0, z0_m))
        avg_x0, avg_y0, avg_z0 = get_history_average_xyz(history)

        normalized_gesture_name = normalize_gesture_name(gesture_name)
        control_detected = normalized_gesture_name in ROCK_GESTURE_NAMES
        show_angle_text = not control_detected
        if control_detected:
            if control_start_points[hand_slot] is None:
                control_start_points[hand_slot] = (avg_x0, avg_y0, avg_z0)
        else:
            print("No control detected")
            control_start_points[hand_slot] = None

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

        if control_start_points[hand_slot] is not None:
            start_x, start_y, start_z = control_start_points[hand_slot]
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
        if show_angle_text:
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
        if show_angle_text:
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

        yaw_state = yaw_states.setdefault(hand_slot, create_yaw_state())
        if forearm_axis_vector is not None and palm_forward_vector is not None:
            smoothed_forearm = ema_vector(yaw_state["forearm_vec"], forearm_axis_vector, VECTOR_EMA_ALPHA)
            smoothed_palm = ema_vector(yaw_state["palm_vec"], palm_forward_vector, VECTOR_EMA_ALPHA)
            yaw_state["forearm_vec"] = smoothed_forearm
            yaw_state["palm_vec"] = smoothed_palm

            raw_yaw_delta_deg_3d = None
            if smoothed_forearm is not None and smoothed_palm is not None:
                delta_rad = signed_angle_on_xz_plane(smoothed_forearm, smoothed_palm)
                raw_yaw_delta_deg_3d = None if delta_rad is None else math.degrees(delta_rad)

            raw_yaw_delta_deg = raw_yaw_delta_deg_3d
            yaw_state["mode"] = "3d"

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
        if show_angle_text and yaw_delta_deg is not None:
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
