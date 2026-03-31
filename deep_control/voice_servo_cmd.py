#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceServoCommandPublisher(Node):
    def __init__(self):
        super().__init__('voice_servo_command_publisher')
        self.pub = self.create_publisher(String, '/gesture/servo_cmd', 10)

        # Give publisher a moment to initialize, then send startup command
        self.create_timer(0.5, self._startup_once)
        self._startup_sent = False

        self.get_logger().info('Voice servo command publisher started')

    def _startup_once(self):
        if self._startup_sent:
            return
        self.send_cmd('s1 2000')
        self._startup_sent = True

    def send_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.pub.publish(msg)
        self.get_logger().info(f'Sent servo command: {cmd}')


def input_loop(node: VoiceServoCommandPublisher):
    print("\nType commands:")
    print("  wall   -> sends: s2 1400")
    print("  screen -> sends: s2 1600")
    print("  stop   -> sends: stop")
    print("  q      -> quit\n")

    while rclpy.ok():
        try:
            text = input("voice> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if text in ('q', 'quit', 'exit'):
            break
        elif text == 'wall':
            node.send_cmd('s2 1400')
        elif text == 'screen':
            node.send_cmd('s2 1600')
        elif text == 'stop':
            node.send_cmd('stop')
        elif text == '':
            continue
        else:
            print("Unknown command. Use: wall, screen, stop, q")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceServoCommandPublisher()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        # small delay so startup s1 2000 gets out first
        time.sleep(1.0)
        input_loop(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()