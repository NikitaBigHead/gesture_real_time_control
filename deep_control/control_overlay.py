import cv2

from control_config import HAND_CONNECTIONS


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


def _get_text_anchor(hand_landmarks, width, height):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    text_x = int(min(xs) * width)
    text_y = max(int(min(ys) * height) - 10, 25)
    return text_x, text_y


def _draw_pose_arm(image_bgr, pose_arm, width, height):
    if pose_arm is None:
        return

    _, elbow_lm, wrist_lm = pose_arm
    elbow_px = int(elbow_lm.x * width)
    elbow_py = int(elbow_lm.y * height)
    wrist_px = int(wrist_lm.x * width)
    wrist_py = int(wrist_lm.y * height)
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


def draw_control_overlay(image_bgr, frame_state):
    """Draw the precomputed hand control state."""
    h, w, _ = image_bgr.shape

    for hand_state in frame_state.hands:
        text_x, text_y = _get_text_anchor(hand_state.hand_landmarks, w, h)

        label = f"{hand_state.gesture_name} ({hand_state.gesture_score:.2f})"
        if hand_state.handedness_label:
            label = f"{label} [{hand_state.handedness_label}]"
        if hand_state.is_active:
            label = f"{label} <ACTIVE>"

        cv2.putText(
            image_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255) if hand_state.is_active else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        for line_index, gesture_line in enumerate(hand_state.gesture_lines, start=1):
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

        info_y = text_y + 26 * (len(hand_state.gesture_lines) + 1)
        if not hand_state.is_active:
            cv2.putText(
                image_bgr,
                "tracked only",
                (text_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            continue

        if hand_state.avg_wrist_xyz is not None:
            avg_x0, avg_y0, avg_z0 = hand_state.avg_wrist_xyz
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

        if hand_state.control_origin_xyz is not None:
            start_x, start_y, _ = hand_state.control_origin_xyz
            start_px = int(start_x * w)
            start_py = int(start_y * h)
            cv2.circle(image_bgr, (start_px, start_py), 6, (0, 0, 255), -1)
            if hand_state.movement_directions:
                cv2.putText(
                    image_bgr,
                    " / ".join(hand_state.movement_directions),
                    (text_x, info_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if hand_state.control_vector is not None:
                control_vector = hand_state.control_vector
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

        _draw_pose_arm(image_bgr, hand_state.pose_arm, w, h)
        if hand_state.show_angle_text:
            cv2.putText(
                image_bgr,
                f"forearm: {hand_state.forearm_status}",
                (text_x, info_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 120, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                f"palm forward: {hand_state.palm_forward_status}",
                (text_x, info_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 120, 0),
                2,
                cv2.LINE_AA,
            )

        if hand_state.show_angle_text and hand_state.yaw_deg is not None:
            cv2.putText(
                image_bgr,
                f"yaw delta: {hand_state.yaw_deg:+.1f} deg",
                (text_x, info_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                f"yaw mode: {hand_state.yaw_mode}",
                (text_x, info_y + 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 160, 255),
                2,
                cv2.LINE_AA,
            )


def draw_gesture_labels(image_bgr, frame_state):
    draw_control_overlay(image_bgr, frame_state)
