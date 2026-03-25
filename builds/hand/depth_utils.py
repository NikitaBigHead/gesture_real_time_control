import pyrealsense2 as rs


def get_depth_at_pixel(depth_image, depth_scale, px, py):
    height, width = depth_image.shape
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    depth_raw = int(depth_image[py, px])
    if depth_raw <= 0:
        return None
    return depth_raw * depth_scale


def get_3d_point_at_pixel(depth_image, depth_scale, intrinsics, px, py):
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
