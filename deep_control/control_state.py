import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median

from control_config import (
    ANGLE_EMA_BETA,
    GESTURE_MEDIAN_WINDOW,
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


@dataclass
class TrackingState:
    gesture_histories: dict
    p0_histories: dict
    control_start_points: dict
    yaw_states: dict


@dataclass
class HandControlState:
    hand_index: int
    hand_landmarks: list
    gesture_name: str
    gesture_score: float
    gesture_lines: list[str]
    handedness_label: str
    hand_slot: str | None
    is_active: bool
    avg_wrist_xyz: tuple[float, float, float] | None = None
    control_origin_xyz: tuple[float, float, float] | None = None
    control_vector: tuple[float, float, float] | None = None
    movement_directions: list[str] = field(default_factory=list)
    pose_arm: object = None
    forearm_status: str = "missing"
    palm_forward_status: str = "missing"
    yaw_deg: float | None = None
    yaw_mode: str = "hold"
    show_angle_text: bool = True


@dataclass
class FrameControlState:
    active_hand_slot: str | None
    active_hand_index: int | None
    hands: list[HandControlState]


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


def create_tracking_state(hand_slots=HAND_PRIORITY):
    return TrackingState(
        gesture_histories={hand_slot: create_gesture_history() for hand_slot in hand_slots},
        p0_histories={hand_slot: create_p0_history() for hand_slot in hand_slots},
        control_start_points={hand_slot: None for hand_slot in hand_slots},
        yaw_states={hand_slot: create_yaw_state() for hand_slot in hand_slots},
    )


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


def reset_hand_tracking_state(hand_slot, tracking_state):
    if hand_slot not in tracking_state.p0_histories:
        tracking_state.p0_histories[hand_slot] = create_p0_history()
    tracking_state.p0_histories[hand_slot].clear()
    tracking_state.control_start_points[hand_slot] = None
    tracking_state.yaw_states[hand_slot] = create_yaw_state()


def reset_tracking_state(tracking_state):
    for history in tracking_state.gesture_histories.values():
        history.clear()
    for history in tracking_state.p0_histories.values():
        history.clear()
    for hand_slot in tracking_state.control_start_points:
        tracking_state.control_start_points[hand_slot] = None
        tracking_state.yaw_states[hand_slot] = create_yaw_state()


def get_recognition_handedness_slot(recognition_result, hand_index):
    if hand_index >= len(recognition_result.handedness):
        return None
    if not recognition_result.handedness[hand_index]:
        return None

    handedness_label = recognition_result.handedness[hand_index][0].category_name
    if not handedness_label:
        return None

    normalized_label = handedness_label.strip().lower()
    if normalized_label in ("left", "right"):
        return normalized_label
    return None


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
    recognition_handedness_slot = get_recognition_handedness_slot(recognition_result, hand_index)
    if hand_slot is None and len(recognition_result.hand_landmarks) == 1:
        hand_slot = recognition_handedness_slot

    handedness_label = hand_slot.capitalize() if hand_slot else ""
    if not handedness_label and recognition_handedness_slot is not None:
        handedness_label = recognition_handedness_slot.capitalize()

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
        if pose_distance_sq is None:
            pose_distance_sq = float("inf")
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


def update_wrist_position(
    hand_landmarks,
    depth_image,
    depth_scale,
    frame_width,
    frame_height,
    tracking_state,
    hand_slot,
):
    x0 = hand_landmarks[0].x
    y0 = hand_landmarks[0].y

    if hand_slot not in tracking_state.p0_histories:
        tracking_state.p0_histories[hand_slot] = create_p0_history()
    history = tracking_state.p0_histories[hand_slot]

    px0 = int(x0 * frame_width)
    py0 = int(y0 * frame_height)
    z0_m = get_depth_at_pixel(depth_image, depth_scale, px0, py0)
    if z0_m is None:
        history_avg = get_history_average_xyz(history)
        if history_avg is None:
            return None
        z0_m = history_avg[2]

    history.append((x0, y0, z0_m))
    return get_history_average_xyz(history)


def update_yaw_state(yaw_state, forearm_axis_vector, palm_forward_vector):
    if forearm_axis_vector is not None and palm_forward_vector is not None:
        smoothed_forearm = ema_vector(
            yaw_state["forearm_vec"], forearm_axis_vector, VECTOR_EMA_ALPHA
        )
        smoothed_palm = ema_vector(yaw_state["palm_vec"], palm_forward_vector, VECTOR_EMA_ALPHA)
        yaw_state["forearm_vec"] = smoothed_forearm
        yaw_state["palm_vec"] = smoothed_palm

        raw_yaw_delta_deg = None
        if smoothed_forearm is not None and smoothed_palm is not None:
            delta_rad = signed_angle_on_xz_plane(smoothed_forearm, smoothed_palm)
            raw_yaw_delta_deg = None if delta_rad is None else math.degrees(delta_rad)

        yaw_state["mode"] = "3d"
        clamped_yaw_delta_deg = clamp_angle_step(
            yaw_state["yaw_deg"], raw_yaw_delta_deg, MAX_YAW_DELTA_PER_FRAME_DEG
        )
        yaw_state["yaw_deg"] = ema_angle(
            yaw_state["yaw_deg"], clamped_yaw_delta_deg, ANGLE_EMA_BETA
        )
        yaw_state["invalid_count"] = 0
    else:
        yaw_state["invalid_count"] += 1
        yaw_state["mode"] = "hold"
        if yaw_state["invalid_count"] > INVALID_HOLD_FRAMES:
            yaw_state["forearm_vec"] = None
            yaw_state["palm_vec"] = None
            yaw_state["yaw_deg"] = None

    return yaw_state["yaw_deg"], yaw_state["mode"]


def compute_frame_control_state(
    recognition_result,
    pose_result,
    depth_image,
    depth_scale,
    intrinsics,
    frame_width,
    frame_height,
    tracking_state,
):
    active_hand_slot, active_hand_index = select_active_hand(recognition_result, pose_result)
    for hand_slot in HAND_PRIORITY:
        if hand_slot != active_hand_slot:
            reset_hand_tracking_state(hand_slot, tracking_state)

    hands = []
    for hand_index, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        metadata = get_hand_metadata(recognition_result, pose_result, hand_index)
        gesture_name = metadata["gesture_name"]
        gesture_score = metadata["gesture_score"]
        gesture_lines = metadata["gesture_lines"]
        hand_slot = metadata["hand_slot"]
        is_active_hand = hand_index == active_hand_index and hand_slot == active_hand_slot

        if hand_slot is not None:
            if hand_slot not in tracking_state.gesture_histories:
                tracking_state.gesture_histories[hand_slot] = create_gesture_history()
            tracking_state.gesture_histories[hand_slot].append(
                {
                    "name": gesture_name,
                    "score": gesture_score,
                }
            )
            gesture_name, gesture_score = get_smoothed_gesture(
                tracking_state.gesture_histories[hand_slot],
                gesture_name,
                gesture_score,
            )

        hand_state = HandControlState(
            hand_index=hand_index,
            hand_landmarks=hand_landmarks,
            gesture_name=gesture_name,
            gesture_score=gesture_score,
            gesture_lines=gesture_lines,
            handedness_label=metadata["handedness_label"],
            hand_slot=hand_slot,
            is_active=is_active_hand,
        )

        if is_active_hand and hand_slot is not None:
            avg_xyz = update_wrist_position(
                hand_landmarks,
                depth_image,
                depth_scale,
                frame_width,
                frame_height,
                tracking_state,
                hand_slot,
            )
            hand_state.avg_wrist_xyz = avg_xyz

            normalized_gesture_name = normalize_gesture_name(gesture_name)
            control_detected = normalized_gesture_name in ROCK_GESTURE_NAMES
            hand_state.show_angle_text = not control_detected
            if control_detected:
                if tracking_state.control_start_points[hand_slot] is None and avg_xyz is not None:
                    tracking_state.control_start_points[hand_slot] = avg_xyz
            else:
                tracking_state.control_start_points[hand_slot] = None

            hand_state.control_origin_xyz = tracking_state.control_start_points[hand_slot]
            if avg_xyz is not None and hand_state.control_origin_xyz is not None:
                hand_state.control_vector = get_control_vector(avg_xyz, hand_state.control_origin_xyz)
                hand_state.movement_directions = get_movement_directions(
                    avg_xyz, hand_state.control_origin_xyz
                )

            forearm_axis_vector, palm_forward_vector, _, _, pose_arm = (
                get_forearm_and_palm_forward_vectors(
                    hand_landmarks,
                    pose_result,
                    depth_image,
                    depth_scale,
                    intrinsics,
                    frame_width,
                    frame_height,
                )
            )
            hand_state.pose_arm = pose_arm
            hand_state.forearm_status = "ok" if pose_arm is not None else "missing"
            hand_state.palm_forward_status = "ok" if palm_forward_vector is not None else "missing"

            yaw_state = tracking_state.yaw_states.setdefault(hand_slot, create_yaw_state())
            hand_state.yaw_deg, hand_state.yaw_mode = update_yaw_state(
                yaw_state, forearm_axis_vector, palm_forward_vector
            )

        hands.append(hand_state)

    return FrameControlState(
        active_hand_slot=active_hand_slot,
        active_hand_index=active_hand_index,
        hands=hands,
    )
