import math

from control_config import XY_DEAD_ZONE, XY_MAX_DELTA, Z_DEAD_ZONE, Z_MAX_DELTA


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
    """Project a 3D vector onto the x-z plane and normalize it."""
    return vec_normalize((v[0], 0.0, v[2]))

def signed_angle_on_xz_plane(v_ref, v_now):
    """Return signed azimuth delta in radians on the horizontal plane."""
    ref_xz = project_to_horizontal(v_ref)
    now_xz = project_to_horizontal(v_now)
    if ref_xz is None or now_xz is None:
        return None

    cross_y = ref_xz[2] * now_xz[0] - ref_xz[0] * now_xz[2]
    dot_xz = ref_xz[0] * now_xz[0] + ref_xz[2] * now_xz[2]
    return math.atan2(cross_y, dot_xz)

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

    vx = normalize_axis(cur_x - org_x, XY_DEAD_ZONE, XY_MAX_DELTA)
    vy = normalize_axis(cur_y - org_y, XY_DEAD_ZONE, XY_MAX_DELTA)
    vz = normalize_axis(cur_z - org_z, Z_DEAD_ZONE, Z_MAX_DELTA)

    vec = [vx, vy, vz]
    max_idx = max(range(3), key=lambda i: abs(vec[i]))

    result = [0.0, 0.0, 0.0]
    result[max_idx] = vec[max_idx]

    return tuple(result)



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
