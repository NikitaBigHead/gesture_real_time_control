import math

from build_config import PALM_GESTURE_NAMES, XY_DEAD_ZONE, XY_MAX_DELTA, Z_DEAD_ZONE, Z_MAX_DELTA


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


def vec_norm(v):
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v, eps=1e-6):
    norm = vec_norm(v)
    if norm < eps:
        return None
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def project_to_horizontal(v):
    return vec_normalize((v[0], 0.0, v[2]))


def signed_angle_on_xz_plane(v_ref, v_now):
    ref_xz = project_to_horizontal(v_ref)
    now_xz = project_to_horizontal(v_now)
    if ref_xz is None or now_xz is None:
        return None

    cross_y = ref_xz[2] * now_xz[0] - ref_xz[0] * now_xz[2]
    dot_xz = ref_xz[0] * now_xz[0] + ref_xz[2] * now_xz[2]
    return math.atan2(cross_y, dot_xz)


def blend_direction_vectors(primary_vec, secondary_vec, primary_weight=0.7):
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


def compute_palm_azimuth_deg(palm_forward_xyz, reference_forward_xyz=None):
    palm_vec = vec_normalize(palm_forward_xyz)
    if palm_vec is None:
        return None

    if reference_forward_xyz is not None:
        delta_rad = signed_angle_on_xz_plane(reference_forward_xyz, palm_vec)
        if delta_rad is None:
            return None
        return math.degrees(delta_rad)

    palm_xz = project_to_horizontal(palm_vec)
    if palm_xz is None:
        return None
    return math.degrees(math.atan2(palm_xz[0], -palm_xz[2]))


def is_yaw_control_active(gesture_name, palm_forward_vector):
    if palm_forward_vector is None:
        return False
    return normalize_gesture_name(gesture_name) in PALM_GESTURE_NAMES


def normalize_axis(delta, dead_zone, max_delta):
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
    cur_x, cur_y, cur_z = current_xyz
    org_x, org_y, org_z = origin_xyz
    return (
        normalize_axis(cur_x - org_x, XY_DEAD_ZONE, XY_MAX_DELTA),
        normalize_axis(cur_y - org_y, XY_DEAD_ZONE, XY_MAX_DELTA),
        normalize_axis(cur_z - org_z, Z_DEAD_ZONE, Z_MAX_DELTA),
    )


def get_movement_directions(current_xyz, origin_xyz):
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
