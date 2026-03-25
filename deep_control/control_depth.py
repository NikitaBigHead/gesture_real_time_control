import numpy as np

from control_config import DEPTH_MEDIAN_WINDOW_RADIUS


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

def get_mediapipe_point(landmark):
    """Return landmark coordinates directly in MediaPipe normalized space."""
    return (float(landmark.x), float(landmark.y), float(landmark.z))
