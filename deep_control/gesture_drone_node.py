# import math
# import threading
# import time

# import rclpy
# from geometry_msgs.msg import PoseStamped, TwistStamped
# from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
# from rclpy.node import Node


# def wrap_angle_rad(angle_rad):
#     return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# class GestureDroneController(Node):
#     def __init__(
#         self,
#         mavros_prefix="/mavros",
#         pose_topic="/mavros/local_position/pose",
#         move_duration_sec=0.8,
#         move_speed_xy=0.8,
#         move_speed_z=0.25,
#         yaw_kp=0.8,
#         max_yaw_rate=0.8,
#         yaw_tolerance_deg=4.0,
#         yaw_sign=1.0,
#     ):
#         super().__init__("gesture_drone_controller")
#         self.pose_topic = pose_topic
#         self.cmd_vel_topic = f"{mavros_prefix}/setpoint_velocity/cmd_vel"

#         self.velocity_pub = self.create_publisher(
#             TwistStamped, self.cmd_vel_topic, 10
#         )
#         self.pose_sub = self.create_subscription(
#             PoseStamped, pose_topic, self.drone_pose_callback, 10
#         )

#         # self.set_mode_client = self.create_client(SetMode, f"{mavros_prefix}/set_mode")
#         # self.arm_client = self.create_client(CommandBool, f"{mavros_prefix}/cmd/arming")
#         # self.takeoff_client = self.create_client(CommandTOL, f"{mavros_prefix}/cmd/takeoff")
#         # self.land_client = self.create_client(CommandTOL, f"{mavros_prefix}/cmd/land")

#         self.move_duration_sec = move_duration_sec
#         self.move_speed_xy = move_speed_xy
#         self.move_speed_z = move_speed_z
#         self.yaw_kp = yaw_kp
#         self.max_yaw_rate = max_yaw_rate
#         self.yaw_tolerance_rad = math.radians(yaw_tolerance_deg)
#         self.yaw_sign = yaw_sign

#         self.drone_pose = None
#         self.current_yaw = None

#         self._lock = threading.Lock()
#         self._command_until = 0.0
#         self._linear_cmd = (0.0, 0.0, 0.0)
#         self._yaw_rate_cmd = 0.0
#         self._rotation_target_yaw = None
#         self._rotation_deadline = 0.0
#         self._sent_stop = False

#         self.control_timer = self.create_timer(0.05, self._control_loop)
#         self.get_logger().info(
#             "Gesture drone controller started "
#             f"(cmd_vel={self.cmd_vel_topic}, pose={self.pose_topic})"
#         )

#     def drone_pose_callback(self, msg):
#         self.drone_pose = msg
#         q = msg.pose.orientation
#         self.current_yaw = math.atan2(
#             2.0 * (q.w * q.z + q.x * q.y),
#             1.0 - 2.0 * (q.y * q.y + q.z * q.z),
#         )

#     def handle_events(self, events):
#         for event in events:
#             self.handle_event(event)

#     def handle_event(self, event):
#         if event.event_type == "fist_release":
#             self._queue_translation(event)
#             return

#         if event.event_type == "palm_release":
#             self._queue_rotation(event)

#     def _queue_translation(self, event):
#         control_vector = event.control_vector
#         if control_vector is None:
#             self.get_logger().warn("Ignoring fist_release without control vector")
#             return

#         vx = -self.move_speed_xy * control_vector[2]
#         vy = -self.move_speed_xy * control_vector[0]
#         vz = -self.move_speed_z * control_vector[1]

#         with self._lock:
#             self._rotation_target_yaw = None
#             self._rotation_deadline = 0.0
#             self._linear_cmd = (vx, vy, vz)
#             self._yaw_rate_cmd = 0.0
#             self._command_until = time.monotonic() + self.move_duration_sec
#             self._sent_stop = False

#         self.get_logger().info(
#             "Queued move command "
#             f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
#             f"from directions={event.movement_directions or ['none']}"
#         )

#     def _queue_rotation(self, event):
#         if event.yaw_deg is None:
#             self.get_logger().warn("Ignoring palm_release without yaw_deg")
#             return

#         yaw_delta_rad = math.radians(self.yaw_sign * event.yaw_deg)
#         duration_hint = max(
#             1.0,
#             abs(yaw_delta_rad) / max(self.max_yaw_rate, 1e-3) * 2.0,
#         )

#         if self.current_yaw is None:
#             # Fallback: open-loop yaw-rate command when pose yaw is unavailable.
#             yaw_rate = max(
#                 -self.max_yaw_rate,
#                 min(self.max_yaw_rate, self.yaw_kp * yaw_delta_rad),
#             )
#             if abs(yaw_rate) < 0.05:
#                 yaw_rate = math.copysign(min(0.2, self.max_yaw_rate), yaw_delta_rad)

#             with self._lock:
#                 self._rotation_target_yaw = None
#                 self._rotation_deadline = 0.0
#                 self._linear_cmd = (0.0, 0.0, 0.0)
#                 self._yaw_rate_cmd = yaw_rate
#                 self._command_until = time.monotonic() + duration_hint
#                 self._sent_stop = False

#             self.get_logger().warn(
#                 "Pose yaw unavailable, using open-loop rotation "
#                 f"rate={yaw_rate:+.2f} rad/s duration={duration_hint:.2f}s"
#             )
#             return

#         target_yaw = wrap_angle_rad(self.current_yaw + yaw_delta_rad)

#         with self._lock:
#             self._linear_cmd = (0.0, 0.0, 0.0)
#             self._yaw_rate_cmd = 0.0
#             self._command_until = 0.0
#             self._rotation_target_yaw = target_yaw
#             self._rotation_deadline = time.monotonic() + duration_hint
#             self._sent_stop = False

#         self.get_logger().info(
#             f"Queued rotate command delta={event.yaw_deg:+.1f} deg target_yaw={target_yaw:+.2f} rad"
#         )

#     def _control_loop(self):
#         now = time.monotonic()

#         with self._lock:
#             rotation_target_yaw = self._rotation_target_yaw
#             rotation_deadline = self._rotation_deadline
#             linear_cmd = self._linear_cmd
#             yaw_rate_cmd = self._yaw_rate_cmd
#             command_until = self._command_until
#             sent_stop = self._sent_stop

#         if rotation_target_yaw is not None:
#             if self.current_yaw is None:
#                 return

#             yaw_err = wrap_angle_rad(rotation_target_yaw - self.current_yaw)
#             if abs(yaw_err) <= self.yaw_tolerance_rad or now >= rotation_deadline:
#                 with self._lock:
#                     self._rotation_target_yaw = None
#                     self._rotation_deadline = 0.0
#                     self._linear_cmd = (0.0, 0.0, 0.0)
#                     self._yaw_rate_cmd = 0.0
#                     self._command_until = 0.0
#                     self._sent_stop = False
#                 self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
#                 return

#             yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.yaw_kp * yaw_err))
#             self.send_velocity_command(0.0, 0.0, 0.0, yaw_rate)
#             return

#         if now < command_until:
#             self.send_velocity_command(linear_cmd[0], linear_cmd[1], linear_cmd[2], yaw_rate_cmd)
#             return

#         if not sent_stop:
#             self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
#             with self._lock:
#                 self._sent_stop = True

#     def send_velocity_command(self, vx, vy, vz, yaw_rate):
#         msg = TwistStamped()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.header.frame_id = "vicon"
#         msg.twist.linear.x = vx
#         msg.twist.linear.y = vy
#         msg.twist.linear.z = vz
#         msg.twist.angular.z = yaw_rate
#         self.velocity_pub.publish(msg)

#     # def set_mode(self, mode):
#     #     req = SetMode.Request()
#     #     req.custom_mode = mode
#     #     self.set_mode_client.call_async(req)
#     #     self.get_logger().info(f"Sent mode change to {mode}")

#     # def arm(self, value):
#     #     req = CommandBool.Request()
#     #     req.value = value
#     #     self.arm_client.call_async(req)
#     #     self.get_logger().info(f"Sent arming command: {value}")

#     # def takeoff(self, altitude=0.8):
#     #     while self.drone_pose is None:
#     #         rclpy.spin_once(self, timeout_sec=0.1)
#     #         time.sleep(0.1)

#     #     req = CommandTOL.Request()
#     #     req.altitude = altitude
#     #     req.latitude = 0.0
#     #     req.longitude = 0.0
#     #     req.min_pitch = 0.0
#     #     req.yaw = 0.0
#     #     future = self.takeoff_client.call_async(req)
#     #     rclpy.spin_until_future_complete(self, future)
#     #     result = future.result()
#     #     success = bool(result and result.success)
#     #     self.get_logger().info(f"Takeoff success: {success}")
#     #     return success

#     # def land(self, wait=True):
#     #     req = CommandTOL.Request()
#     #     req.altitude = 0.0
#     #     req.latitude = 0.0
#     #     req.longitude = 0.0
#     #     req.min_pitch = 0.0
#     #     req.yaw = 0.0
#     #     future = self.land_client.call_async(req)
#     #     if not wait:
#     #         self.get_logger().info("Sent land command")
#     #         return future
#     #     rclpy.spin_until_future_complete(self, future)
#     #     result = future.result()
#     #     success = bool(result and result.success)
#     #     self.get_logger().info(f"Land success: {success}")
#     #     return success



















import math
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node


def wrap_angle_rad(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class GestureDroneController(Node):
    def __init__(
        self,
        mavros_prefix="/gesture",
        pose_topic="/drone2/mavros/local_position/pose",
        move_duration_sec=0.8,
        move_speed_xy=0.8,
        move_speed_z=0.25,
        yaw_kp=0.8,
        max_yaw_rate=0.8,
        yaw_tolerance_deg=4.0,
        yaw_sign=1.0,
    ):
        super().__init__("gesture_drone_controller")
        self.pose_topic = pose_topic
        self.cmd_vel_topic = f"{mavros_prefix}/setpoint_velocity/cmd_vel"

        self.velocity_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.drone_pose_callback, 10
        )

        self.move_duration_sec = move_duration_sec
        self.move_speed_xy = move_speed_xy
        self.move_speed_z = move_speed_z
        self.yaw_kp = yaw_kp
        self.max_yaw_rate = max_yaw_rate
        self.yaw_tolerance_rad = math.radians(yaw_tolerance_deg)
        self.yaw_sign = yaw_sign

        self.drone_pose = None
        self.current_yaw = None

        self._lock = threading.Lock()
        self._command_until = 0.0
        self._linear_cmd = (0.0, 0.0, 0.0)
        self._yaw_rate_cmd = 0.0
        self._rotation_target_yaw = None
        self._rotation_deadline = 0.0
        self._sent_stop = False

        self.control_timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info(
            "Gesture drone controller started "
            f"(cmd_vel={self.cmd_vel_topic}, pose={self.pose_topic})"
        )

    def drone_pose_callback(self, msg):
        self.drone_pose = msg
        q = msg.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def handle_events(self, events):
        for event in events:
            self.handle_event(event)

    def handle_event(self, event):
        if event.event_type == "fist_release":
            self._queue_translation(event)
            return

        if event.event_type == "palm_release":
            self._queue_rotation(event)

    def _queue_translation(self, event):
        control_vector = event.control_vector
        if control_vector is None:
            self.get_logger().warn("Ignoring fist_release without control vector")
            return

        vx = -self.move_speed_xy * control_vector[2]
        vy = -self.move_speed_xy * control_vector[0]
        vz = -self.move_speed_z * control_vector[1]

        with self._lock:
            self._rotation_target_yaw = None
            self._rotation_deadline = 0.0
            self._linear_cmd = (vx, vy, vz)
            self._yaw_rate_cmd = 0.0
            self._command_until = time.monotonic() + self.move_duration_sec
            self._sent_stop = False

        self.get_logger().info(
            "Queued move command "
            f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
            f"from directions={event.movement_directions or ['none']}"
        )

    def _queue_rotation(self, event):
        if event.yaw_deg is None:
            self.get_logger().warn("Ignoring palm_release without yaw_deg")
            return

        yaw_delta_rad = math.radians(self.yaw_sign * event.yaw_deg)
        duration_hint = max(
            1.0,
            abs(yaw_delta_rad) / max(self.max_yaw_rate, 1e-3) * 2.0,
        )

        if self.current_yaw is None:
            yaw_rate = max(
                -self.max_yaw_rate,
                min(self.max_yaw_rate, self.yaw_kp * yaw_delta_rad),
            )
            if abs(yaw_rate) < 0.05:
                yaw_rate = math.copysign(min(0.2, self.max_yaw_rate), yaw_delta_rad)

            with self._lock:
                self._rotation_target_yaw = None
                self._rotation_deadline = 0.0
                self._linear_cmd = (0.0, 0.0, 0.0)
                self._yaw_rate_cmd = yaw_rate
                self._command_until = time.monotonic() + duration_hint
                self._sent_stop = False

            self.get_logger().warn(
                "Pose yaw unavailable, using open-loop rotation "
                f"rate={yaw_rate:+.2f} rad/s duration={duration_hint:.2f}s"
            )
            return

        target_yaw = wrap_angle_rad(self.current_yaw + yaw_delta_rad)

        with self._lock:
            self._linear_cmd = (0.0, 0.0, 0.0)
            self._yaw_rate_cmd = 0.0
            self._command_until = 0.0
            self._rotation_target_yaw = target_yaw
            self._rotation_deadline = time.monotonic() + duration_hint
            self._sent_stop = False

        self.get_logger().info(
            f"Queued rotate command delta={event.yaw_deg:+.1f} deg target_yaw={target_yaw:+.2f} rad"
        )

    def _control_loop(self):
        now = time.monotonic()

        with self._lock:
            rotation_target_yaw = self._rotation_target_yaw
            rotation_deadline = self._rotation_deadline
            linear_cmd = self._linear_cmd
            yaw_rate_cmd = self._yaw_rate_cmd
            command_until = self._command_until
            sent_stop = self._sent_stop

        if rotation_target_yaw is not None:
            if self.current_yaw is None:
                return

            yaw_err = wrap_angle_rad(rotation_target_yaw - self.current_yaw)
            if abs(yaw_err) <= self.yaw_tolerance_rad or now >= rotation_deadline:
                with self._lock:
                    self._rotation_target_yaw = None
                    self._rotation_deadline = 0.0
                    self._linear_cmd = (0.0, 0.0, 0.0)
                    self._yaw_rate_cmd = 0.0
                    self._command_until = 0.0
                    self._sent_stop = False
                self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
                return

            yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.yaw_kp * yaw_err))
            self.send_velocity_command(0.0, 0.0, 0.0, yaw_rate)
            return

        if now < command_until:
            self.send_velocity_command(linear_cmd[0], linear_cmd[1], linear_cmd[2], yaw_rate_cmd)
            return

        if not sent_stop:
            self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
            with self._lock:
                self._sent_stop = True

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = yaw_rate
        self.velocity_pub.publish(msg)