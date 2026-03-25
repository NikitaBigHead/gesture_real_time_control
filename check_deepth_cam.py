import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# Viewer usually selects a valid stream profile automatically.
# In code it is safer to request a concrete depth stream before start().
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print(f"Device: {device}")
print(f"Product Line: {device_product_line}")

try:
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")
    cv2.namedWindow("RealSense RGB", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("RealSense Depth", cv2.WINDOW_AUTOSIZE)

    # Skip several frames so auto-exposure / sensor warmup can settle.
    for _ in range(10):
        frames = pipeline.wait_for_frames(5000)

        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        print(depth_image.min(), depth_image.max())


    while True:
        frames = pipeline.wait_for_frames(5000)


        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("RealSense Depth", depth_colormap)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            cv2.destroyAllWindows()
            break

        width, height = depth_frame.get_width(), depth_frame.get_height()
        dist = depth_frame.get_distance(width // 2, height // 2)
        print(f"The camera is facing an object {dist:.3f} meters away", end="\r")

except RuntimeError as err:
    print(f"RealSense runtime error: {err}")
    print("If RealSense Viewer is open, close it first so Python can own the device.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
