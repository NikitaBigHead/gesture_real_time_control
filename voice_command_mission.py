#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import String


def wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class VoiceCommandMission(Node):
    def __init__(self):
        super().__init__('voice_command_mission')

        # ----------------------------
        # Parameters
        # ----------------------------
        self.takeoff_alt = 2.0
        self.bound_xy = 10.0
        self.max_abs_z = 15
        self.pub_rate = 20.0
        self.stabilize_time = 5.0
        self.hold_time = 5.0

        self.max_vx = 0.6
        self.max_vy = 0.6
        self.max_vz = 0.35
        self.max_yaw_rate = 0.8

        self.pos_kp_xy = 0.9
        self.pos_kp_z = 0.9
        self.yaw_kp = 1.2

        self.xy_tol = 0.08
        self.z_tol = 0.08
        self.yaw_tol = math.radians(4.0)

        # ----------------------------
        # State
        # ----------------------------
        self.current_pose = None
        self.current_yaw = None
        self.phase = 'idle'
        self.stabilize_start_time = None
        self.active_job = None

        # ----------------------------
        # Publishers / Subscribers
        # ----------------------------
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # self.pose_sub = self.create_subscription(
        #     PoseStamped,
        #     '/mavros/local_position/pose',
        #     self.pose_callback,
        #     10
        # )
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            pose_qos
        )

        self.voice_cmd_sub = self.create_subscription(
            String,
            '/voice/drone_command',
            self.voice_cmd_callback,
            10
        )

        # ----------------------------
        # Service clients
        # ----------------------------
        self.set_mode_client = self.create_client(
            SetMode, '/mavros/set_mode'
        )
        self.arm_client = self.create_client(
            CommandBool, '/mavros/cmd/arming'
        )
        self.takeoff_client = self.create_client(
            CommandTOL, '/mavros/cmd/takeoff'
        )
        self.land_client = self.create_client(
            CommandTOL, '/mavros/cmd/land'
        )

        self.timer = self.create_timer(1.0 / self.pub_rate, self.control_loop)

        self.get_logger().info('VoiceCommandMission node started')

    # ----------------------------
    # ROS callbacks
    # ----------------------------
    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg
        q = msg.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def voice_cmd_callback(self, msg: String):
        if self.phase != 'voice_control':
            self.get_logger().warn(f'Ignoring command while phase={self.phase}')
            return

        if self.current_pose is None:
            self.get_logger().warn('Ignoring voice command: pose unavailable')
            return

        if self.active_job is not None:
            self.get_logger().warn('Ignoring voice command: already executing another command')
            return

        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Invalid JSON command: {e}')
            return

        command = payload.get('command')
        distance = payload.get('distance')
        angle = payload.get('angle')

        if command is None:
            self.get_logger().warn('Ignoring empty command')
            return

        job = self.build_job(command, distance, angle)
        if job is None:
            return

        self.active_job = job
        self.get_logger().info(f'Accepted voice command: {payload}')

    # ----------------------------
    # Helpers
    # ----------------------------
    def wait_for_service(self, client, name):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {name} service...')

    def set_mode(self, mode='GUIDED'):
        self.wait_for_service(self.set_mode_client, 'set_mode')
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None and result.mode_sent:
            self.get_logger().info(f'Mode changed to {mode}')
            return True
        self.get_logger().error(f'Failed to change mode to {mode}')
        return False

    def arm(self):
        self.wait_for_service(self.arm_client, 'arming')
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None and result.success:
            self.get_logger().info('Drone armed successfully')
            return True
        self.get_logger().error('Failed to arm drone')
        return False

    def takeoff(self, altitude):
        self.wait_for_service(self.takeoff_client, 'takeoff')
        req = CommandTOL.Request()
        req.altitude = altitude
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None and result.success:
            self.get_logger().info(f'Takeoff successful to {altitude:.2f} m')
            return True
        self.get_logger().error('Takeoff failed')
        return False

    def land(self):
        self.wait_for_service(self.land_client, 'land')
        req = CommandTOL.Request()
        req.altitude = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        future = self.land_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None and result.success:
            self.get_logger().info('Landing successful')
            return True
        self.get_logger().error('Landing failed')
        return False

    def publish_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def current_xyz(self):
        p = self.current_pose.pose.position
        return p.x, p.y, p.z

    def build_job(self, command, distance, angle):
        x0, y0, z0 = self.current_xyz()
        yaw0 = self.current_yaw if self.current_yaw is not None else 0.0

        # translation commands
        if command in {'forward', 'backward', 'left', 'right', 'up', 'down'}:
            if distance is None:
                self.get_logger().warn('Ignoring move command without explicit distance')
                return None

            distance = float(distance)
            if distance <= 0.0:
                self.get_logger().warn('Ignoring move command with non-positive distance')
                return None

            body_dx = 0.0
            body_dy = 0.0
            dz = 0.0

            if command == 'forward':
                body_dx = distance
            elif command == 'backward':
                body_dx = -distance
            elif command == 'left':
                body_dy = distance
            elif command == 'right':
                body_dy = -distance
            elif command == 'up':
                dz = distance
            elif command == 'down':
                dz = -distance

            # body -> world (ENU, x forward, y left)
            dx = math.cos(yaw0) * body_dx - math.sin(yaw0) * body_dy
            dy = math.sin(yaw0) * body_dx + math.cos(yaw0) * body_dy

            x1 = x0 + dx
            y1 = y0 + dy
            z1 = z0 + dz

            if abs(x1) > self.bound_xy or abs(y1) > self.bound_xy:
                self.get_logger().warn(
                    f'Rejecting command: target out of XY bounds ({x1:.2f}, {y1:.2f})'
                )
                return None

            if z1 < 0.2 or z1 > self.max_abs_z:
                self.get_logger().warn(
                    f'Rejecting command: target z out of bounds ({z1:.2f})'
                )
                return None

            return {
                'type': 'translate',
                'stage': 'out',
                'hold_until': None,
                'start': {'x': x0, 'y': y0, 'z': z0, 'yaw': yaw0},
                'target_out': {'x': x1, 'y': y1, 'z': z1},
                'target_back': {'x': x0, 'y': y0, 'z': z0},
            }

        # rotation commands
        if command in {'turn right', 'turn left'}:
            if angle is None:
                self.get_logger().warn('Ignoring turn command without explicit angle')
                return None

            angle_rad = math.radians(float(angle))
            if angle_rad <= 0.0:
                self.get_logger().warn('Ignoring turn command with non-positive angle')
                return None

            sign = -1.0 if command == 'turn right' else 1.0
            yaw1 = wrap_angle_rad(yaw0 + sign * angle_rad)

            return {
                'type': 'rotate',
                'stage': 'out',
                'hold_until': None,
                'start': {'x': x0, 'y': y0, 'z': z0, 'yaw': yaw0},
                'target_out_yaw': yaw1,
                'target_back_yaw': yaw0,
            }

        self.get_logger().warn(f'Ignoring unsupported command: {command}')
        return None

    def execute_translate_job(self, now, job, x, y, z):
        if job['stage'] == 'hold':
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            if now >= job['hold_until']:
                job['stage'] = 'back'
                self.get_logger().info('Hold complete, returning to start position')
            return

        target = job['target_out'] if job['stage'] == 'out' else job['target_back']

        ex = target['x'] - x
        ey = target['y'] - y
        ez = target['z'] - z

        xy_err = math.hypot(ex, ey)

        if xy_err <= self.xy_tol and abs(ez) <= self.z_tol:
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)

            if job['stage'] == 'out':
                job['stage'] = 'hold'
                job['hold_until'] = now + self.hold_time
                self.get_logger().info('Reached target position, holding')
            else:
                self.get_logger().info('Returned to start position')
                self.active_job = None
            return

        vx = self.clamp(self.pos_kp_xy * ex, -self.max_vx, self.max_vx)
        vy = self.clamp(self.pos_kp_xy * ey, -self.max_vy, self.max_vy)
        vz = self.clamp(self.pos_kp_z * ez, -self.max_vz, self.max_vz)

        self.publish_velocity(vx, vy, vz, 0.0)

    def execute_rotate_job(self, now, job, yaw):
        if job['stage'] == 'hold':
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            if now >= job['hold_until']:
                job['stage'] = 'back'
                self.get_logger().info('Hold complete, rotating back to start yaw')
            return

        target_yaw = job['target_out_yaw'] if job['stage'] == 'out' else job['target_back_yaw']
        yaw_err = wrap_angle_rad(target_yaw - yaw)

        if abs(yaw_err) <= self.yaw_tol:
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)

            if job['stage'] == 'out':
                job['stage'] = 'hold'
                job['hold_until'] = now + self.hold_time
                self.get_logger().info('Reached target yaw, holding')
            else:
                self.get_logger().info('Returned to start yaw')
                self.active_job = None
            return

        yaw_rate = self.clamp(self.yaw_kp * yaw_err, -self.max_yaw_rate, self.max_yaw_rate)
        self.publish_velocity(0.0, 0.0, 0.0, yaw_rate)

    # ----------------------------
    # Main control loop
    # ----------------------------
    def control_loop(self):
        if self.phase == 'idle':
            return

        if self.current_pose is None:
            self.get_logger().warn('Waiting for pose...')
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            return

        x, y, z = self.current_xyz()
        yaw = self.current_yaw if self.current_yaw is not None else 0.0
        now = time.monotonic()

        if self.phase == 'stabilize':
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            if now - self.stabilize_start_time >= self.stabilize_time:
                self.get_logger().info('Switching to voice control mode')
                self.phase = 'voice_control'
            return

        if self.phase == 'voice_control':
            if abs(x) >= self.bound_xy or abs(y) >= self.bound_xy:
                self.get_logger().warn(
                    f'Boundary exceeded at x={x:.2f}, y={y:.2f}. Landing...'
                )
                self.publish_velocity(0.0, 0.0, 0.0, 0.0)
                success = self.land()
                if success:
                    self.phase = 'landing_called'
                return

            if self.active_job is None:
                self.publish_velocity(0.0, 0.0, 0.0, 0.0)
                return

            if self.active_job['type'] == 'translate':
                self.execute_translate_job(now, self.active_job, x, y, z)
            elif self.active_job['type'] == 'rotate':
                self.execute_rotate_job(now, self.active_job, yaw)
            return

        if self.phase == 'landing_called':
            if z <= 0.15:
                self.get_logger().info('Landed. Mission complete.')
                self.phase = 'done'
            return

    def run_mission(self):
        self.get_logger().info('Waiting for pose estimate...')
        start_wait = time.time()
        while rclpy.ok() and self.current_pose is None and (time.time() - start_wait < 10.0):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_pose is None:
            self.get_logger().warn('No pose received yet, continuing anyway.')

        if not self.set_mode('GUIDED'):
            return
        time.sleep(0.5)

        if not self.arm():
            return
        time.sleep(2.0)

        if not self.takeoff(self.takeoff_alt):
            return

        self.get_logger().info('Climbing...')
        time.sleep(8.0)

        self.get_logger().info('Takeoff complete. Stabilizing before voice control...')
        self.stabilize_start_time = time.monotonic()
        self.phase = 'stabilize'


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandMission()

    try:
        node.run_mission()
        while rclpy.ok() and node.phase != 'done':
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user, sending zero velocity...')
        node.publish_velocity(0.0, 0.0, 0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()