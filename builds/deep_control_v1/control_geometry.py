from control_depth import get_mediapipe_point
from control_math import vec_dot, vec_normalize, vec_sub


def get_pose_arm_match(pose_result, hand_wrist_landmark):
    """Return the nearest pose arm as (hand_slot, pose_arm, distance_sq)."""
    if not pose_result or not pose_result.pose_landmarks:
        return None, None, None
    if not pose_result.pose_landmarks[0]:
        return None, None, None

    pose_landmarks = pose_result.pose_landmarks[0]
    arm_candidates = (
        ("left", (11, 13, 15)),
        ("right", (12, 14, 16)),
    )
    best_hand_slot = None
    best_arm = None
    best_distance = None
    for hand_slot, (shoulder_idx, elbow_idx, wrist_idx) in arm_candidates:
        wrist_lm = pose_landmarks[wrist_idx]

        dx = wrist_lm.x - hand_wrist_landmark.x
        dy = wrist_lm.y - hand_wrist_landmark.y
        distance = dx * dx + dy * dy
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_hand_slot = hand_slot
            best_arm = (
                pose_landmarks[shoulder_idx],
                pose_landmarks[elbow_idx],
                pose_landmarks[wrist_idx],
            )
    return best_hand_slot, best_arm, best_distance


def get_pose_arm_landmarks(pose_result, hand_wrist_landmark):
    """Return shoulder, elbow, wrist pose landmarks for the nearest pose arm."""
    _, pose_arm, _ = get_pose_arm_match(pose_result, hand_wrist_landmark)
    return pose_arm


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
